import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import torch
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
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from torch import autograd
from torch.utils.data import ConcatDataset
from autoencoder import SimpleAutoencoder, VaDE, ClusteringEvaluationCallback, cluster_acc
from pytorch_lightning.callbacks import Callback
from scipy.optimize import linear_sum_assignment as linear_assignment


class PLVaDE(pl.LightningModule):
    def __init__(self, n_neurons=[784, 512, 256, 10], batch_norm=False, k=10, lr=1e-3, pretrain_lr=2e-3,
                 device='cuda', pretrain_epochs=50, batch_size=1024, pretrained_model_file=None, init_gmm_file=None,
                 covariance_type='diag', data_size=None, multivariate_latent=False, rank=3, dataset='mnist'):
        super(PLVaDE, self).__init__()
        self.save_hyperparameters()
        self.bs = batch_size
        # self.pretrained_params_file = pretrained_params_file
        pretrain_model, init_gmm = self.init_params(n_neurons, batch_norm, k, pretrain_epochs)
        self.pretrained_model, self.init_gmm = [pretrain_model], init_gmm
        self.model = VaDE(n_neurons=n_neurons, batch_norm=batch_norm, k=k, device=device, 
                          pretrain_model=pretrain_model, init_gmm=init_gmm, logger=self.log,
                          covariance_type=covariance_type, multivariate_latent=multivariate_latent, rank=rank)
        
    def prepare_data(self):
        if self.hparams['dataset'] == 'mnist':
            self.train_ds = MNIST("data", download=True)
            self.valid_ds = MNIST("data", download=True, train=False)
        elif self.hparams['dataset'] == 'fmnist':
            self.train_ds = FashionMNIST("data", download=True)
            self.valid_ds = FashionMNIST("data", download=True, train=False)
        to_tensor_dataset = lambda ds: TensorDataset(ds.data.view(-1, 28**2).float()/255., ds.targets)
        self.train_ds, self.valid_ds = map(to_tensor_dataset, [train_ds, valid_ds])
        if self.hparams['data_size'] is not None:
            n_sample = self.hparams['data_size']
            to_subset = lambda ds: torch.utils.data.random_split(ds, 
                                                                 [n_sample, len(ds) - n_sample],
                                                                 torch.Generator().manual_seed(42))[0]
            self.train_ds, self.valid_ds = map(to_subset, [self.train_ds, self.valid_ds])
        self.all_ds = ConcatDataset([self.train_ds, self.valid_ds])


    def init_params(self, n_neurons, batch_norm, k, pretrain_epochs):
        self.prepare_data()
        pretrain_model = SimpleAutoencoder(n_neurons, batch_norm=batch_norm, lr=self.hparams['pretrain_lr'])
        pretrain_model.val_dataloader = self.val_dataloader
        pretrain_model.train_dataloader = self.train_dataloader
        if self.hparams['pretrained_model_file'] is None:
            logger = pl.loggers.WandbLogger(project='AE clustering')
            # trainer = pl.Trainer(gpus=1, logger=logger, callbacks=[ClusteringEvaluationCallback()],
            #                      max_epochs=pretrain_epochs, progress_bar_refresh_rate=10)
            trainer = pl.Trainer(gpus=1, max_epochs=pretrain_epochs, progress_bar_refresh_rate=10)
            trainer.fit(pretrain_model)
        else:
            pretrain_model.load_state_dict(torch.load(self.hparams['pretrained_model_file'])['state_dict'])
        # transform = transforms.Compose([transforms.ToTensor(), 
        #                         transforms.Lambda(lambda x: torch.flatten(x))])
        # dataset = MNIST("data", download=True, transform=transform)
        if self.hparams['init_gmm_file'] is None:
            X_encoded = pretrain_model.encode_ds(self.all_ds)
            # X_encoded = pretrain_model.encode_ds(dataset)
            init_gmm = GaussianMixture(k, covariance_type=self.hparams['covariance_type'], n_init=3)
            init_gmm.fit(X_encoded)
        else:
            with open(self.hparams['init_gmm_file'], 'rb') as file:
                init_gmm = pickle.load(file)
        return pretrain_model, init_gmm
        

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=1)
                        #   num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=8)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=1) 
                        #   num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=8)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), self.hparams['lr'], weight_decay=0.00)
        # opt = torch.optim.AdamW(list(self.parameters())[3:], self.hparams['lr'], weight_decay=0.00)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch:  (epoch+1)/10 if epoch < 0 else 0.9**(epoch//10))
        return [opt], [sched]
    
    def training_step(self, batch, batch_idx):
        bx, by = batch
        result = self.model.shared_step(bx)
        for k, v in result.items():
            self.log(k, v, logger=True)
        return result

    def validation_step(self, batch, batch_idx):
        bx, by = batch
        result = self.model.shared_step(bx)
        for k, v in result.items():
            self.log(k, v, logger=True)
        return result

    def cluster_data(self, ds_type='all'):
        if ds_type=='all':
            dl = DataLoader(self.all_ds, batch_size=4096, shuffle=False, num_workers=1)
        elif ds_type=='train':
            dl = DataLoader(self.train_ds, batch_size=4096, shuffle=False, num_workers=1, pin_memory=True)
        elif ds_type=='valid':
            dl = DataLoader(self.valid_ds, batch_size=4096, shuffle=False, num_workers=1, pin_memory=True)
        else:
            raise Exception("Incorrect ds_type (can be one of 'train', 'valid', 'all')")
        return self.model.cluster_data(dl)


def test_gmm():
    model = PLVaDE(n_neurons=[784, 512, 512, 2048, 10], k=10, lr=1e-3, covariance_type='full', batch_size=256, pretrain_epochs=10,
                pretrained_model_file="AE clustering/5wn5ybl3/checkpoints/epoch=69-step=16449.ckpt", 
                init_gmm_file='saved_gmm_init/5wn5ybl3/gmm-full-0.pkl',
                multivariate_latent=True, rank=5, device='cuda:0')
    pretrained_model = model.pretrained_model[0]
    y_true = np.stack([model.all_ds[i][1] for i in range(len(model.all_ds))])
    X_encoded = pretrained_model.encode_ds(model.all_ds)
    for i in range(10):
        init_gmm = GaussianMixture(10, covariance_type='diag', n_init=3)
        y_pred = init_gmm.fit_predict(X_encoded)
        acc = cluster_acc(y_true, y_pred)
        print('log likelihood:', init_gmm.score(X_encoded))
        print('Accuracy: ', acc)
        if acc > 0.9:
            import pickle
            with open(f'saved_gmm_init/5wn5ybl3/gmm-diag-{i}.pkl', 'wb') as file:
                pickle.dump(init_gmm, file)



if __name__ == '__main__':
    # test_gmm()
    # exit()
    # train_ds = MNIST("data", download=True)
    # valid_ds = MNIST("data", download=True, train=False)
    # to_tensor_dataset = lambda ds: TensorDataset(ds.data.cuda().view(-1, 28**2).float()/255., ds.targets.cuda())
    # train_ds, valid_ds = map(to_tensor_dataset, [train_ds, valid_ds])
    # print(train_ds[3])
    # exit()
    model = PLVaDE(n_neurons=[784, 512, 512, 2048, 10], k=10, lr=2e-3, covariance_type='diag', batch_size=256, pretrain_epochs=10,
                   pretrained_model_file="AE clustering/5wn5ybl3/checkpoints/epoch=69-step=16449.ckpt", 
                #    init_gmm_file='saved_gmm_init/5wn5ybl3/gmm-full-0.pkl',
                   init_gmm_file='saved_gmm_init/5wn5ybl3/gmm-diag-0.pkl',
                   multivariate_latent=False, rank=5, device='cuda:0')

    logger = pl.loggers.WandbLogger(project='VADE')
    trainer = pl.Trainer(gpus=1, logger=logger, progress_bar_refresh_rate=10, max_epochs=50, 
                        callbacks=[ClusteringEvaluationCallback()], log_every_n_steps=1, profiler='advanced')

    trainer.fit(model)