from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset
from torch import nn, distributions
from torch.nn import functional as F
from torch import distributions as D
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
import pytorch_lightning as pl
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import numpy as np 
from autoencoder import SimpleAutoencoder, VaDE
from utils import best_of_n_gmm_ray
from scipy.optimize import linear_sum_assignment as linear_assignment
from data_modules import MNISTDataModule, BasicDataModule

import ray

class PretrainingCallback(pl.Callback):
    def __init__(self, epochs=None, lr=None, n_clusters=10, n_gmm_restarts=10):
        super(PretrainingCallback, self).__init__()
        self.epochs, self.lr, self.n_clusters = epochs, lr, n_clusters
        self.n_gmm_restarts = n_gmm_restarts
        self.datamodule = None

    # def on_init_end(self, trainer):
    #     self.datamodule = trainer.datamodule

    def on_fit_start(self, trainer, pl_module):
        pretrain_model = SimpleAutoencoder(pl_module.hparams['n_neurons'], 
                                          lr=self.lr if self.lr else pl_module.hparams['pretrain_lr'], 
                                          activation=pl_module.hparams['activation'], 
                                          dropout=pl_module.hparams['dropout'])
        pretrain_trainer = pl.Trainer(gpus=1, 
                                      max_epochs=self.epochs if self.epochs else pl_module.hparams['pretrain_epochs'], 
                                      progress_bar_refresh_rate=50)
        datamodule = None
        if trainer.datamodule:
            if isinstance(trainer.datamodule, MNISTDataModule):
                datamodule = trainer.datamodule
            elif trainer.datamodule.hasattr('base_datamodule') and isinstance(trainer.datamodule.base_datamodule, MNISTDataModule):
                datamodule = trainer.datamodule.base_datamodule
            else: 
                datamodule = BasicDataModule(pl_module.train_ds, pl_module.valid_ds, batch_size=pl_module.batch_size)
                # pretrain_model.train_dataloader = lambda: DataLoader(pl_module.valid_ds, batch_size=256, shuffle=False, num_workers=8)
                # pretrain_model.val_dataloader = pl_module.val_dataloader
        
        pretrain_trainer.fit(pretrain_model, datamodule=datamodule)

        X_encoded = pretrain_model.encode_ds(datamodule.all_ds)
        ray.init(ignore_reinit_error=True)
        init_gmm = best_of_n_gmm_ray(X_encoded, self.n_clusters, covariance_type=pl_module.hparams['covariance_type'], 
                                     n=self.n_gmm_restarts, n_init=3)
        pl_module.model.load_parameters(pretrain_model=pretrain_model, init_gmm=init_gmm)


def cluster_acc(Y_pred, Y):
    assert Y_pred.shape == Y.shape
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in zip(*ind)])*1.0 / Y_pred.size

def clustering_accuracy(gt, cluster_assignments):
    mat = metrics.confusion_matrix(cluster_assignments, gt, labels=np.arange(max(max(gt), max(cluster_assignments)) + 1))
    cluster_assignments = mat.argmax(axis=1)[cluster_assignments]
    return metrics.accuracy_score(gt, cluster_assignments)


class ClusteringEvaluationCallback(pl.callbacks.Callback):
    def __init__(self, on_start=True, postfix='', **kwargs):
        super(ClusteringEvaluationCallback, self).__init__()
        self.on_start = on_start
        self.postfix = postfix
        self.idx = -1
        self.kwargs = kwargs
        if 'ds_type' not in self.kwargs:
            self.kwargs['ds_type'] = 'all'

    def evaluate_clustering(self, trainer, pl_module):
        if trainer.datamodule is not None:
            if isinstance(trainer.datamodule, MNISTDataModule):
                datamodule = trainer.datamodule
            elif trainer.datamodule.hasattr('base_datamodule') and isinstance(trainer.datamodule.base_datamodule, MNISTDataModule):
                datamodule = trainer.datamodule.base_datamodule
            else: 
                datamodule = BasicDataModule(pl_module.train_ds, pl_module.valid_ds, batch_size=pl_module.batch_size)
            if self.kwargs['ds_type'] == 'train':
                ds = datamodule.train_ds
            elif self.kwargs['ds_type'] == 'valid':
                ds = datamodule.valid_ds
            elif self.kwargs['ds_type'] == 'all':
                ds = datamodule.all_ds 
            else:
                raise Exception(f"ds_type can be only 'train', 'valid', 'all'")
            dl = DataLoader(ds, batch_size=1024, shuffle=False)
            gt, labels, _ = pl_module.cluster_data(dl=dl, **self.kwargs)
        else:
            gt, labels, _ = pl_module.cluster_data(**self.kwargs)
        nmi, acc2 = metrics.normalized_mutual_info_score(labels, gt), clustering_accuracy(gt, labels)
        acc = cluster_acc(labels, gt)
        ari = metrics.adjusted_rand_score(labels, gt)
        prefix = self.kwargs['ds_type'] + '_'
        if self.kwargs['ds_type'] == 'all':
            prefix = ''
        pl_module.log(prefix + 'NMI' + self.postfix, nmi, on_epoch=True)
        pl_module.log(prefix + 'ACC' + self.postfix, acc, on_epoch=True)
        pl_module.log(prefix + 'ACC2' + self.postfix, acc2, on_epoch=True)
        pl_module.log(prefix + 'ARI' + self.postfix, ari, on_epoch=True)


    def on_epoch_start(self, trainer, pl_module):
        if self.on_start:
            self.evaluate_clustering(trainer, pl_module)

    def on_epoch_end(self, trainer, pl_module):
        if not self.on_start:
            self.evaluate_clustering(trainer, pl_module)
