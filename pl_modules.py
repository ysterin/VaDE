import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import torch
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
from autoencoder import SimpleAutoencoder, VaDE
from pytorch_lightning.callbacks import Callback
from scipy.optimize import linear_sum_assignment as linear_assignment


class PLVaDE(pl.LightningModule):
    def __init__(self, n_neurons=[784, 512, 256, 10], batch_norm=False, k=10, lr=1e-3, 
                 device='cuda', pretrain_epochs=50, batch_size=1024, pretrained_params_file=None):
        super(PLVaDE, self).__init__()
        self.bs = batch_size
        self.pretrained_params_file = pretrained_params_file
        pretrain_model, init_gmm = self.init_params(n_neurons, batch_norm, k, pretrain_epochs)
        self.model = VaDE(n_neurons=n_neurons, batch_norm=batch_norm, k=k, device=device, 
                          pretrain_model=pretrain_model, init_gmm=init_gmm, logger=self.log)
        self.hparams = {'lr': lr}
        
    def prepare_data(self):
        train_ds = MNIST("data", download=True)
        valid_ds = MNIST("data", download=True, train=False)
        to_tensor_dataset = lambda ds: TensorDataset(ds.data.view(-1, 28**2).float()/255., ds.targets)
        self.train_ds, self.valid_ds = map(to_tensor_dataset, [train_ds, valid_ds])
        self.all_ds = ConcatDataset([self.train_ds, self.valid_ds])


    def init_params(self, n_neurons, batch_norm, k, pretrain_epochs):
        self.prepare_data()
        pretrain_model = SimpleAutoencoder(n_neurons, batch_norm=batch_norm, lr=2e-3)
        pretrain_model.val_dataloader = self.val_dataloader
        pretrain_model.train_dataloader = self.train_dataloader
        trainer = pl.Trainer(gpus=1, max_epochs=pretrain_epochs, progress_bar_refresh_rate=10)
        trainer.fit(pretrain_model)
        dataset = MNIST("data", download=True, transform=transform)
        X_encoded = pretrain_model.encode_ds(dataset)
        init_gmm = GaussianMixture(k, covariance_type='diag')
        init_gmm.fit(X_encoded)
        return pretrain_model, init_gmm
        

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.bs, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.bs, shuffle=False, num_workers=8)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), self.hparams['lr'], weight_decay=0.00)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: 0.9**(epoch//10))
        return [opt], [sched]
    
    def training_step(self, batch, batch_idx):
        bx, by = batch
        loss, rec_loss, kl_loss, bce_loss = self.model.shared_step(bx)
        result = {'loss': loss, 
                  'rec_loss': rec_loss.detach(),
                  'kl_loss': kl_loss.detach(),
                  'bce_loss': bce_loss.detach()}
        for k, v in result.items():
            self.log(k, v, logger=True)
        return result

    def validation_step(self, batch, batch_idx):
        bx, by = batch
        loss, rec_loss, kl_loss, bce_loss = self.model.shared_step(bx)
        result = {'loss': loss.detach(), 
                  'rec_loss': rec_loss.detach(),
                  'kl_loss': kl_loss.detach(),
                  'bce_loss': bce_loss.detach()}
        for k, v in result.items():
            self.log(k, v, logger=True)
        return result

    def cluster_data(self, dl=None):
        if not dl:
            dl = DataLoader(self.all_ds, batch_size=2**12, shuffle=False, num_workers=8)
        return self.model.cluster_data(dl)
