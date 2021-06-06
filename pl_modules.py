import os
from pytorch_lightning import profiler

from torch.nn.modules.dropout import Dropout
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import torch
import wandb
import pickle
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset
from torch import nn, distributions
from torch.nn import functional as F
from torch import distributions as D
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
import pytorch_lightning as pl
from torch.distributions import Normal, Laplace, kl_divergence, kl, Categorical
import math
from data_modules import MNISTDataModule, BasicDataModule
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from torch import autograd
from torch.utils.data import ConcatDataset
from autoencoder import SimpleAutoencoder, VaDE
from utils import best_of_n_gmm_ray
from callbacks import ClusteringEvaluationCallback, cluster_acc, PretrainingCallback, LoadPretrained
from pytorch_lightning.callbacks import Callback
from scipy.optimize import linear_sum_assignment as linear_assignment
import ray

'''
A Pytorch Lightning Module for VaDE. 
wraps the VaDE model and adds fonctionallity for data and training steps.
'''
class PLVaDE(pl.LightningModule):
    def __init__(self, n_neurons=[784, 512, 256, 10], batch_norm=False, dropout=0., activation='relu', k=10, 
                 lr=1e-3, lr_gmm=1e-3, pretrain_lr=2e-3, do_pretrain=True, warmup_epochs=10, latent_logvar_bias_init=0,
                 device='cuda', pretrain_epochs=50, batch_size=1024, pretrained_model_file=None, init_gmm_file=None,
                 covariance_type='diag', data_size=None, data_random_seed=42, multivariate_latent=False, rank=3, dataset='mnist'):
        super(PLVaDE, self).__init__()
        self.save_hyperparameters()
        self.bs = batch_size
        self.dataset, self.data_size = dataset, data_size
        pretrain_model, init_gmm = self.init_params(n_neurons, batch_norm, k, pretrain_epochs)
        self.pretrained_model, self.init_gmm = [pretrain_model], init_gmm
        self.model = VaDE(n_neurons=n_neurons, k=k, device=device, activation=activation, dropout=dropout,
                          pretrain_model=pretrain_model, init_gmm=init_gmm, logger=self.log, latent_logvar_bias_init=latent_logvar_bias_init,
                          covariance_type=covariance_type, multivariate_latent=multivariate_latent, rank=rank)
        
    def prepare_data(self):
        if self.hparams['dataset'] == 'mnist':
            train_ds = MNIST("data", download=True)
            valid_ds = MNIST("data", download=True, train=False)
        elif self.hparams['dataset'] == 'fmnist':
            train_ds = FashionMNIST("data", download=True)
            valid_ds = FashionMNIST("data", download=True, train=False)
        else:
            raise Exception(f"dataset must be either 'mnist' or 'fmnist', instead is {self.hparams['dataset']}")
        to_tensor_dataset = lambda ds: TensorDataset(ds.data.view(-1, 28**2).float()/255., ds.targets)
        self.train_ds, self.valid_ds = map(to_tensor_dataset, [train_ds, valid_ds])
        if self.hparams['data_size'] is not None:
            n_sample = self.hparams['data_size']
            to_subset = lambda ds: torch.utils.data.random_split(ds, 
                                                                 [n_sample, len(ds) - n_sample],
                                                                 torch.Generator().manual_seed(self.hparams['data_random_seed']))[0]
            self.train_ds = to_subset(self.train_ds)
        self.all_ds = ConcatDataset([self.train_ds, self.valid_ds])

    def init_params(self, n_neurons, batch_norm, k, pretrain_epochs):
        if not self.hparams['do_pretrain']:
            return None, None
        self.prepare_data()
        pretrain_model = SimpleAutoencoder(n_neurons, lr=self.hparams['pretrain_lr'], 
                            activation=self.hparams['activation'], dropout=self.hparams['dropout'])
        pretrain_model.val_dataloader = self.val_dataloader
        pretrain_model.train_dataloader = self.train_dataloader
        if self.hparams['pretrained_model_file'] is None:
            trainer = pl.Trainer(gpus=1, max_epochs=pretrain_epochs, progress_bar_refresh_rate=30)
            trainer.fit(pretrain_model)
        else:
            pretrain_model.load_state_dict(torch.load(self.hparams['pretrained_model_file'])['state_dict'])
        if self.hparams['init_gmm_file'] is None:
            X_encoded = pretrain_model.encode_ds(self.all_ds)
            ray.init(ignore_reinit_error=True)
            init_gmm = best_of_n_gmm_ray(X_encoded, k, covariance_type=self.hparams['covariance_type'], n=10, n_init=3)
        else:
            with open(self.hparams['init_gmm_file'], 'rb') as file:
                init_gmm = pickle.load(file)
        return pretrain_model, init_gmm
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=1, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.hparams['batch_size']*4, shuffle=False, num_workers=1, pin_memory=True) 

    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params' : self.model.model_params},
                                 {'params': self.model.gmm_params, 'lr': self.hparams['lr_gmm']}],
                                  self.hparams['lr'], weight_decay=0.00)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch:  (epoch+1)/(self.hparams['warmup_epochs'] + 1) if epoch < self.hparams['warmup_epochs'] else 0.9**(epoch//10))
        return [opt], [sched]
    
    def training_step(self, batch, batch_idx):
        bx, by = batch
        result = self.model.shared_step(bx)
        for k, v in result.items():
            self.log('train/' + k, v, logger=True)
        return result

    def validation_step(self, batch, batch_idx):
        bx, by = batch
        result = self.model.shared_step(bx)
        for k, v in result.items():
            self.log('valid/' + k, v, logger=True)
        return result

    def cluster_data(self, dl=None, ds_type='all'):
        if not dl:   
            if ds_type=='all':
                dl = DataLoader(self.all_ds, batch_size=4096, shuffle=False, num_workers=0)
            elif ds_type=='train':
                dl = DataLoader(self.train_ds, batch_size=4096, shuffle=False, num_workers=0, pin_memory=False)
            elif ds_type=='valid':
                dl = DataLoader(self.valid_ds, batch_size=4096, shuffle=False, num_workers=0, pin_memory=False)
            else:
                raise Exception("Incorrect ds_type (can be one of 'train', 'valid', 'all')")
        return self.model.cluster_data(dl)


if __name__ == '__main__':
    model = PLVaDE(n_neurons=[784, 512, 512, 2048, 10], k=10, lr=2e-3, pretrain_lr=3e-4, covariance_type='full', 
                   batch_size=2**8, pretrain_epochs=20, do_pretrain=False,
                   multivariate_latent=False, rank=5, device='cuda', dataset='mnist')
    base_dm = MNISTDataModule(dataset='mnist', bs=256, data_size=10000)
    logger = pl.loggers.WandbLogger(project='VADE')
    SEED = 45
    torch.manual_seed(SEED)
    np.random.seed(SEED) 
    callbacks = [ClusteringEvaluationCallback(ds_type='all'), 
                 PretrainingCallback(200, 3e-4, save_dir='saved_models1', seed=SEED, early_stop=True)]
    trainer = pl.Trainer(gpus=1, logger=logger, progress_bar_refresh_rate=50, max_epochs=30,
                        callbacks=callbacks, 
                        log_every_n_steps=5)

    trainer.fit(model, datamodule=base_dm)
