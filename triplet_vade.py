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
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from torch import autograd
from torch.utils.data import ConcatDataset
from autoencoder import SimpleAutoencoder, VaDE, normal_to_multivariate, cross_entropy, kl_distance
from data_modules import TripletDataset, CombinedDataset
from callbacks import ClusteringEvaluationCallback, cluster_acc
from utils import best_of_n_gmm_ray
from pytorch_lightning.callbacks import Callback
from scipy.optimize import linear_sum_assignment as linear_assignment
import ray

# kl divergence between two tensors representing discrete distributions.
def kl_div(p, q):
    log_diff = p.log() - q.log()
    return (p * log_diff).sum(dim=1)

# Jensen Shannon distance between two tensors representing discrete distributions.
def js_dist(p, q, eps=1e-8):
    p, q = (p + eps), (q + eps)
    p, q = p / p.sum(dim=1, keepdims=True), q / q.sum(dim=1, keepdims=True)
    m = (p + q) / 2
    return (kl_div(p, m) + kl_div(q, m)) / 2

# similarity measure between logits of two discrete distributions.
def similarity(logits1, logits2):
    p = logits1.softmax(dim=1)
    q = logits2.softmax(dim=1)
    return torch.einsum('bi,bi->b', p, q)

def split_dist(dist, n=3):
    bs = dist.mean.shape[0] // n
    if isinstance(dist, D.Independent):
        if isinstance(dist.base_dist, D.Normal):
            return [D.Independent(D.Normal(loc, scale), 1) for loc, scale in \
                        list(zip(dist.base_dist.loc.split(bs), dist.base_dist.scale.split(bs)))]

def ifnone(x, val):
    if x is None:
        return val
    return x


'''
A Variational Autoencoder model that combines a VaDE with triplet loss. Used for clustering.
triplet_loss_margin: margin used in the regular triplet loss.
triplet_loss_alpha: weight of regular triplet loss.
triplet_loss_margin_kl: margin of triplet loss with distance between the latent distributions.
triplet_loss_alpha_kl: weight of triplet loss with distance between the latent distributions.
triplet_loss_margin_cls: margin of triplet loss with distance between class distiobutions.
triplet_loss_alpha_kl: weight of triplet loss with distance between the class distiobutions.

'''
class TripletVaDE(pl.LightningModule):
    def __init__(self, n_neurons=[784, 512, 256, 10],
                 batch_norm=False,
                 k=10, 
                 dataset='mnist',
                 lr=1e-3, 
                 pretrain_lr=3e-4,
                 lr_gmm = None,
                 activation='relu',
                 dropout=0.0,
                 batch_size=256, 
                 device='cuda', 
                 do_pretrain=True,
                 pretrain_epochs=50, 
                 pretrained_model_file=None, 
                 init_gmm_file=None,
                 covariance_type='diag',
                 data_size=None,
                 weight_decay=0.0,
                 latent_logvar_bias_init=0.,
                 autoencoder_loss_alpha=1.0,
                 triplet_loss_margin=0.5,
                 triplet_loss_alpha=1.,
                 triplet_loss_margin_kl=20, 
                 triplet_loss_alpha_kl=0.,
                 triplet_loss_alpha_cls=0., 
                 triplet_loss_margin_cls=0.2,
                 triplet_loss_alpha_sim=0.,
                 warmup_epochs=10,
                 n_samples_for_triplets=1000,
                 n_triplets=None):
        super(TripletVaDE, self).__init__()
        self.save_hyperparameters()
        if lr_gmm is None:
            self.hparams['lr_gmm'] = lr
        self.batch_size = batch_size
        self.n_neurons, self.pretrain_epochs, self.batch_norm = n_neurons, pretrain_epochs, batch_norm
        pretrain_model, init_gmm = self.init_params()
        self.model = VaDE(n_neurons=n_neurons, k=k, device=device, covariance_type=covariance_type,
                          latent_logvar_bias_init=latent_logvar_bias_init,
                          pretrain_model=pretrain_model, init_gmm=init_gmm, logger=self.log, activation=activation, dropout=dropout)
        lr_gmm = ifnone(lr_gmm, lr)

    def prepare_data(self):
        if self.hparams['dataset'] == 'mnist':
            self.train_ds = MNIST("data", download=True)
            self.valid_ds = MNIST("data", download=True, train=False)
        elif self.hparams['dataset'] == 'fmnist':
            self.train_ds = FashionMNIST("data", download=True)
            self.valid_ds = FashionMNIST("data", download=True, train=False)
        to_tensor_dataset = lambda ds: TensorDataset(ds.data.view(-1, 28**2).float()/255., ds.targets)
        self.train_ds, self.valid_ds = map(to_tensor_dataset, [self.train_ds, self.valid_ds])
        if self.hparams['data_size'] is not None:
            n_sample = self.hparams['data_size']
            to_subset = lambda ds: torch.utils.data.random_split(ds, 
                                                                 [n_sample, len(ds) - n_sample],
                                                                 torch.Generator().manual_seed(42))[0]
            self.train_ds = to_subset(self.train_ds)
        self.all_ds = ConcatDataset([self.train_ds, self.valid_ds])
        self.train_triplet_ds = CombinedDataset(self.train_ds, data_size=self.hparams['n_samples_for_triplets'], max_triplets=self.hparams['n_triplets'])
        self.valid_triplet_ds = CombinedDataset(self.valid_ds, data_size=self.hparams['n_samples_for_triplets']) 
            
    def pretrain_model(self):
        n_neurons, pretrain_epochs, batch_norm = self.n_neurons, self.pretrain_epochs, self.batch_norm
        self.prepare_data()
        pretrained_model = SimpleAutoencoder(n_neurons, lr=self.hparams['pretrain_lr'])
        pretrained_model.val_dataloader = lambda: DataLoader(self.valid_ds, batch_size=256, shuffle=False, num_workers=8)
        pretrained_model.train_dataloader = lambda: DataLoader(self.train_ds, batch_size=256, shuffle=True, num_workers=8)
        gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(gpus=gpus, max_epochs=pretrain_epochs, progress_bar_refresh_rate=20)
        trainer.fit(pretrained_model)
        return pretrained_model
    
    def init_params(self):
        if not self.hparams['pretrained_model_file']:
            if not self.hparams['do_pretrain']:
                return None, None
            pretrained_model = self.pretrain_model()
        else:
            pretrained_model = SimpleAutoencoder(self.n_neurons)
            state_dict = torch.load(self.hparams['pretrained_model_file'])
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            pretrained_model.load_state_dict(state_dict)
            self.prepare_data()
        if not self.hparams['init_gmm_file']:
            X_encoded = pretrained_model.encode_ds(self.all_ds)
            ray.init(ignore_reinit_error=True)
            init_gmm = best_of_n_gmm_ray(X_encoded, self.hparams['k'], covariance_type=self.hparams['covariance_type'], n=10, n_init=3)
        else:
            with open(self.hparams['init_gmm_file'], 'rb') as file:
                init_gmm = pickle.load(file)
        
        return pretrained_model, init_gmm
        
    def train_dataloader(self):
        return DataLoader(self.train_triplet_ds, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.valid_triplet_ds, batch_size=self.batch_size, num_workers=0)

    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.model.model_params},
                                 {'params': self.model.gmm_params, 'lr': self.hparams['lr_gmm']}], 
                                self.hparams['lr'], weight_decay=self.hparams['weight_decay'])
        lr_rate_function = lambda epoch: max(min((epoch+1)/self.hparams['warmup_epochs'], 0.9**(epoch//10)), 3e-2)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_rate_function)
        return [opt], [sched]

    def triplet_loss(self, anchor_z_dist, pos_z_dist, neg_z_dist):
        anchor_z, pos_z, neg_z = anchor_z_dist.mean.squeeze(1), pos_z_dist.mean.squeeze(1), neg_z_dist.mean.squeeze(1)
   
        d1, d2 = torch.linalg.norm(anchor_z - pos_z, dim=1), torch.linalg.norm(anchor_z - neg_z, dim=1)
        assert len(anchor_z.shape) == 2
        results = {}
        results['anchor_pos_distance_no_norm'] = d1.mean()
        results['anchor_neg_distance_no_norm'] = d2.mean()
        results['correct_triplet_pct_no_norm'] = (d1 < d2).float().mean()*100
        anchor_z, pos_z, neg_z = map(lambda t: t / t.norm(dim=1, keepdim=True), [anchor_z, pos_z, neg_z])
        d1, d2 = torch.linalg.norm(anchor_z - pos_z, dim=1), torch.linalg.norm(anchor_z - neg_z, dim=1)
        assert len(anchor_z.shape) == 2
        results['anchor_pos_distance'] = d1.mean()
        results['anchor_neg_distance'] = d2.mean()
        results['correct_triplet_pct'] = (d1 < d2).float().mean()*100
        loss = torch.relu(d1 - d2 + self.hparams['triplet_loss_margin']).mean()
        results['triplet_loss'] = loss
        return results

    def triplet_loss_kl(self, anchor_z_dist, pos_z_dist, neg_z_dist):
        d1, d2 = kl_distance(anchor_z_dist, pos_z_dist), kl_distance(anchor_z_dist, neg_z_dist)
        results = {}
        results['anchor_pos_distance_kl'] = d1.mean()
        results['anchor_neg_distance_kl'] = d2.mean()
        results['correct_triplet_pct_kl'] = (d1 < d2).float().mean() * 100
        loss = torch.relu(d1 - d2 + self.hparams['triplet_loss_margin_kl']).mean()
        results['triplet_loss_kl'] = loss
        return results

    def triplet_loss_cls(self, anchor_logits, pos_logits, neg_logits):
        d1 = js_dist(anchor_logits.softmax(1), pos_logits.softmax(1))
        d2 = js_dist(anchor_logits.softmax(1), neg_logits.softmax(1))
        assert len(d1.shape) == 1
        results = {}
        results['anchor_pos_distance_cls'] = d1.mean()
        results['anchor_neg_distance_cls'] = d2.mean()
        results['correct_triplet_pct_cls'] = (d1 < d2).float().mean() * 100
        loss = torch.relu(d1 - d2 + self.hparams['triplet_loss_margin_cls']).mean()
        results['triplet_loss_cls'] = loss
        return results

    def triplet_loss_sim(self, anchor_logits, pos_logits, neg_logits):
        sim1 = similarity(anchor_logits, pos_logits)
        sim2 = similarity(anchor_logits, neg_logits)
        assert len(sim1.shape) == 1
        results = {}
        results['anchor_pos_similarity'] = sim1.mean()
        results['anchor_neg_similarity'] = sim2.mean()
        results['correct_triplet_pct_sim'] = (sim1 > sim2).float().mean() * 100
        loss = - (sim1 / (sim1 + sim2)).log().mean()
        results['triplet_loss_sim'] = loss
        return results
    
    def shared_step(self, batch, batch_idx):
        bx, triplet_dict = batch
        bs, *_ = bx.shape
        result = self.model.shared_step(bx)
        triplet_batch = torch.cat([triplet_dict[s] for s in ['anchor', 'positive', 'negative']])
        triplet_dists, logits = self.model.z_dist_and_classification_logits(triplet_batch)
        triplet_dists = split_dist(triplet_dists, n=3)
        logits = torch.split(logits, bs)
        result['main_loss'] = result['loss'].detach().clone()
        result['loss'] *= self.hparams['autoencoder_loss_alpha']
        if self.hparams['triplet_loss_alpha'] > 0:
            result.update(self.triplet_loss(*triplet_dists))
            result['loss'] += self.hparams['triplet_loss_alpha'] * result['triplet_loss']
        if self.hparams['triplet_loss_alpha_kl'] > 0:
            result.update(self.triplet_loss_kl(*triplet_dists))
            result['loss'] += self.hparams['triplet_loss_alpha_kl'] * result['triplet_loss_kl']
        if self.hparams['triplet_loss_alpha_cls'] > 0:
            result.update(self.triplet_loss_cls(*logits))
            result['loss'] += self.hparams['triplet_loss_alpha_cls'] * result['triplet_loss_cls']
        if self.hparams['triplet_loss_alpha_sim'] > 0:
            result.update(self.triplet_loss_sim(*logits))
            result['loss'] += self.hparams['triplet_loss_alpha_sim'] * result['triplet_loss_sim']
        return result

    def validation_step(self, batch, batch_idx):
        self.model.training = False
        result = self.shared_step(batch, batch_idx)
        for k, v in result.items():
            self.log('valid/' + k, v, logger=True, on_epoch=True)
        return result

    def training_step(self, batch, batch_idx):
        self.model.training = True
        result = self.shared_step(batch, batch_idx)
        for k, v in result.items():
            self.log('train/' + k, v, logger=True)
        return result

    def cluster_data(self, dl=None, ds_type='all'):
        if not dl:
            if ds_type=='all':
                dl = DataLoader(self.all_ds, batch_size=1024, shuffle=False, num_workers=8)
            elif ds_type=='train':
                dl = DataLoader(self.train_ds, batch_size=1024, shuffle=False, num_workers=8)
            elif ds_type=='valid':
                dl = DataLoader(self.valid_ds, batch_size=1024, shuffle=False, num_workers=8)
            else:
                raise Exception("Incorrect ds_type (can be one of 'train', 'valid', 'all')")
        return self.model.cluster_data(dl)


pretrained_model_file = "AE clustering/5wn5ybl3/checkpoints/epoch=69-step=16449.ckpt"
init_gmm_file = "saved_gmm_init/5wn5ybl3/gmm-full-acc=0.85.pkl"
if __name__=='__main__':
    model = TripletVaDE(n_neurons=[784, 512, 512, 2048, 10], pretrain_epochs=20, lr=2e-3, triplet_loss_alpha=0.01, autoencoder_loss_alpha=0., 
    triplet_loss_alpha_kl=0.0, triplet_loss_margin_kl=300, triplet_loss_alpha_cls=100, triplet_loss_margin_cls=0.3,
     pretrained_model_file=pretrained_model_file, init_gmm_file=init_gmm_file, covariance_type='full')
    logger = pl.loggers.WandbLogger(project='TripletVaDE')
    trainer = pl.Trainer(gpus=1, max_epochs=20, logger=logger, progress_bar_refresh_rate=10, callbacks=[ClusteringEvaluationCallback(ds_type='valid')])
    trainer.fit(model)
