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

# ray.init()

# def best_of_n_gmm(x, n_clusters=10, n=10, covariance_type='full', n_init=1):
#     scores_dict = {}
#     for i in range(n):
#         gmm = GaussianMixture(n_clusters, covariance_type=covariance_type, n_init=n_init)
#         gmm.fit(x)
#         log_likelihood = gmm.score(x)
#         scores_dict[gmm] = log_likelihood
#     best_gmm, best_score = max(scores_dict.items(), key=lambda o: o[1])
#     return best_gmm

# @ray.remote
# def fit_gmm(x, n_clusters=10, covariance_type='full', n_init=1, random_state=None):
#     gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, n_init=n_init, random_state=random_state)
#     gmm.fit(x)
#     log_likelihood = gmm.score(x)
#     return gmm, log_likelihood

# def best_of_n_gmm_ray(x, n_clusters=10, n=10, covariance_type='full', n_init=1):
#     # ray.init(ignore_reinit_error=True)
#     ss = np.random.SeedSequence(42)
#     random_states = [np.random.RandomState(np.random.PCG64(c)) for c in ss.spawn(n)]
#     x_id =  ray.put(x)
#     scores = ray.get([fit_gmm.remote(x_id, n_clusters, covariance_type, n_init, st) for st in random_states])
#     best_gmm, best_score = max(scores, key=lambda o: o[1])
# #     ray.shutdown()
#     return best_gmm


class PLVaDE(pl.LightningModule):
    def __init__(self, n_neurons=[784, 512, 256, 10], batch_norm=False, dropout=0., activation='relu', k=10, 
                 lr=1e-3, pretrain_lr=2e-3, do_pretrain=True,
                 device='cuda', pretrain_epochs=50, batch_size=1024, pretrained_model_file=None, init_gmm_file=None,
                 covariance_type='diag', data_size=None, data_random_seed=42, multivariate_latent=False, rank=3, dataset='mnist'):
        super(PLVaDE, self).__init__()
        self.save_hyperparameters()
        self.bs = batch_size
        pretrain_model, init_gmm = self.init_params(n_neurons, batch_norm, k, pretrain_epochs)
        self.pretrained_model, self.init_gmm = [pretrain_model], init_gmm
        self.model = VaDE(n_neurons=n_neurons, k=k, device=device, activation=activation, dropout=dropout,
                          pretrain_model=pretrain_model, init_gmm=init_gmm, logger=self.log,
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
            # self.train_ds, self.valid_ds = map(to_subset, [self.train_ds, self.valid_ds])
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
            # wandb.init()
            # logger = pl.loggers.WandbLogger(project='AE clustering', offline=True, log_model=True)
            # trainer = pl.Trainer(gpus=1, logger=logger, callbacks=[ClusteringEvaluationCallback()],
            #                      max_epochs=pretrain_epochs, progress_bar_refresh_rate=10)
            trainer = pl.Trainer(gpus=1, max_epochs=pretrain_epochs, progress_bar_refresh_rate=30)
            trainer.fit(pretrain_model)
            # wandb.finish()
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
                        #   num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=8)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.hparams['batch_size']*4, shuffle=False, num_workers=1, pin_memory=True) 
                        #   num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=8)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), self.hparams['lr'], weight_decay=0.00)
        # opt = torch.optim.AdamW(list(self.parameters())[3:], self.hparams['lr'], weight_decay=0.00)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch:  (epoch+1)/10 if epoch < 10 else 0.9**(epoch//10))
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


def test_gmm():
    model = PLVaDE(n_neurons=[784, 512, 512, 2048, 10], k=10, lr=1e-3, covariance_type='full', batch_size=256, pretrain_epochs=10,
                # pretrained_model_file="AE clustering/5wn5ybl3/checkpoints/epoch=69-step=16449.ckpt", 
                pretrained_model_file="AE clustering/dla63r4s/checkpoints/epoch=49-step=11749.ckpt",
                init_gmm_file='saved_gmm_init/5wn5ybl3/gmm-full-0.pkl',
                multivariate_latent=True, rank=5, device='cuda:0', dataset='fmnist')
    pretrained_model = model.pretrained_model[0]
    y_true = np.stack([model.all_ds[i][1] for i in range(len(model.all_ds))])
    X_encoded = pretrained_model.encode_ds(model.all_ds)
    for i in range(10):
        init_gmm = GaussianMixture(10, covariance_type='full', n_init=3)
        y_pred = init_gmm.fit_predict(X_encoded)
        acc = cluster_acc(y_true, y_pred)
        nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
        ari = metrics.adjusted_rand_score(y_true, y_pred)
        print('log likelihood:', init_gmm.score(X_encoded))
        print('Accuracy: ', acc)
        print('NMI: ', nmi)
        print('ARI: ', ari)
        if True:
            import pickle
            with open(f'saved_gmm_init/dla63r4s/gmm-full-acc={acc:.2f}.pkl', 'wb') as file:
                pickle.dump(init_gmm, file)


# from data_modules import MNISTDataModule, BasicDataModule
# from pathlib import Path
# from pytorch_lightning.callbacks import EarlyStopping

# class LoadPretrained(pl.Callback):
#     def __init__(self, save_dir, seed):
#         super(LoadPretrained, self).__init__()
#         self.save_dir, self.seed = save_dir, seed
#         self.path = Path(save_dir) / f"seed={seed}.h5"

#     def on_fit_start(self, trainer, pl_module):
#         pl_module.model.load_state_dict(torch.load(self.path))


# class PretrainingCallback(pl.Callback):
#     def __init__(self, epochs=None, lr=None, n_clusters=10, n_gmm_restarts=10, seed=None, save_dir=None, early_stop=False):
#         super(PretrainingCallback, self).__init__()
#         if not seed:
#             seed = torch.seed() % 2**32
#         self.seed = seed
#         torch.manual_seed(seed)
#         np.random.seed(seed)
#         self.save_dir = save_dir
#         self.epochs, self.lr, self.n_clusters = epochs, lr, n_clusters
#         self.n_gmm_restarts = n_gmm_restarts
#         self.datamodule = None
#         self.callbacks = []
#         if early_stop:
#             self.callbacks.append(EarlyStopping(monitor='valid/loss', patience=10))

#     def save_pretrained(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
#         if not self.save_dir:
#             return
#         save_dir = Path(self.save_dir) 
#         os.makedirs(save_dir, exist_ok=True)
#         torch.save(pl_module.model.state_dict(), save_dir / f"seed={self.seed}.h5")

#     def on_fit_start(self, trainer, pl_module):
#         pretrain_model = SimpleAutoencoder(pl_module.hparams['n_neurons'], 
#                                            lr=self.lr if self.lr else pl_module.hparams['pretrain_lr'], 
#                                            activation=pl_module.hparams['activation'], 
#                                            dropout=pl_module.hparams['dropout'])
#         pretrain_trainer = pl.Trainer(gpus=1, 
#                                       max_epochs=self.epochs if self.epochs else pl_module.hparams['pretrain_epochs'], 
#                                       progress_bar_refresh_rate=50, 
#                                       callbacks=self.callbacks)
#         datamodule = None
#         if trainer.datamodule:
#             if isinstance(trainer.datamodule, MNISTDataModule):
#                 datamodule = trainer.datamodule
#             elif trainer.datamodule.hasattr('base_datamodule') and isinstance(trainer.datamodule.base_datamodule, MNISTDataModule):
#                 datamodule = trainer.datamodule.base_datamodule
#         else: 
#             datamodule = BasicDataModule(pl_module.train_ds, pl_module.valid_ds, batch_size=pl_module.hparams['batch_size'])
        
#         pretrain_trainer.fit(pretrain_model, datamodule=datamodule)
#         X_encoded = pretrain_model.encode_ds(datamodule.all_ds)
#         ray.init(ignore_reinit_error=True)
#         init_gmm = best_of_n_gmm_ray(X_encoded, self.n_clusters, covariance_type=pl_module.hparams['covariance_type'], 
#                                      n=self.n_gmm_restarts, n_init=3)
#         targets = np.array([x[1] for x in datamodule.all_ds])
#         accuracy = cluster_acc(init_gmm.predict(X_encoded), targets)
#         pl_module.model.load_parameters(pretrain_model=pretrain_model, init_gmm=init_gmm)
        
#         self.save_pretrained(trainer, pl_module)



if __name__ == '__main__':
    model = PLVaDE(n_neurons=[784, 512, 512, 2048, 10], k=10, lr=2e-3, pretrain_lr=3e-4, covariance_type='full', 
                   batch_size=2**8, pretrain_epochs=20, do_pretrain=False,
                #    pretrained_model_file="AE clustering/5wn5ybl3/checkpoints/epoch=69-step=16449.ckpt", 
                #    init_gmm_file='saved_gmm_init/5wn5ybl3/gmm-full-acc=0.95.pkl',
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
