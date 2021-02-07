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


def cross_entropy(P, Q):
    return kl_divergence(P, Q) + P.entropy()

def kl_distance(P, Q):
    return 0.5 * (kl_divergence(P, Q) + kl_divergence(Q, P))


class TripletDataset(IterableDataset):
    def __init__(self, data, labels, transform=None, data_size=None, max_samples=None, seed=None):
        super(TripletDataset, self).__init__()
        assert len(data) == len(labels)
        self.seed = seed
        # if transform is not None:
        #     self.transform = transform
        # else:
        #     self.transform = transforms.Lambda(lambda x: x)
        self.rng = np.random.Generator(np.random.PCG64(seed=self.seed))
        if data_size and data_size < len(data):
            idxs = self.rng.choice(len(data), size=data_size, replace=False)
            data, labels = data[idxs], labels[idxs]
        self.data_size = len(data)
        self.data = data
        self.labels = labels

        self.label_set = list(set(labels.numpy()))
        self.data_dict = {lbl: [self.data[i] for i in range(self.data_size) if self.labels[i]==lbl] \
                            for lbl in self.label_set}
        self.n_classes = len(self.label_set)
        self.class_sizes = {lbl: len(self.data_dict[lbl]) for lbl in self.label_set}
        if not max_samples:
            max_samples = sum([n*(n-1)//2 * (self.data_size-n) for n in self.class_sizes.values()])
        self.max_samples = max_samples


    def __len__(self):
        return self.max_samples        

    def __iter__(self):
        self.rng = np.random.Generator(np.random.PCG64(seed=self.seed))
        for i in range(self.max_samples):
            anchor_label, neg_label = self.rng.choice(self.label_set, size=2, replace=False)
            try:
                anchor_idx, positive_idx = self.rng.choice(self.class_sizes[anchor_label], size=2, replace=False)
            except ValueError as e:
                continue
            negative_idx = self.rng.choice(self.class_sizes[neg_label], size=1)[0]
            yield {'anchor': self.data_dict[anchor_label][anchor_idx], 
                   'positive': self.data_dict[anchor_label][positive_idx], 
                   'negative': self.data_dict[neg_label][negative_idx]}


class CombinedDataset(Dataset):
    def __init__(self, dataset, transform=None, data_size=None, max_samples=None, seed=None):
        super(CombinedDataset, self).__init__()
        self.data = dataset.data.view(-1, 28**2).float() / 255.0
        self.triplet_dataset = TripletDataset(self.data, dataset.targets, transform, data_size, max_samples=len(dataset), seed=seed)
        self.triplets_iterator = iter(self.triplet_dataset)
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i, (sample, triplet) in enumerate(zip(self.data ,self.triplet_dataset)):
            yield (sample, triplet)

    def __getitem__(self, idx):
        sample = self.data[idx]
        triplet = next(self.triplets_iterator)
        return (sample, triplet)


class TripletVaDE(pl.LightningModule):
    def __init__(self, n_neurons=[784, 512, 256, 10], batch_norm=True, k=10, lr=1e-3, batch_size=256, device='cuda', 
                 pretrain_epochs=50, pretrained_model=None, triplet_loss_alpha=1.):
        super(TripletVaDE, self).__init__()
        self.batch_size = batch_size
        self.n_neurons, self.pretrain_epochs, self.batch_norm = n_neurons, pretrain_epochs, batch_norm
        pretrain_model, init_gmm = self.init_params(k, pretrained_model)
        self.model = VaDE(n_neurons=n_neurons, batch_norm=batch_norm, k=k, device=device, 
                          pretrain_model=pretrain_model, init_gmm=init_gmm, logger=self.log)
        self.hparams = {'lr': lr}
        self.triplet_loss_margin = 0.5
        self.triplet_loss_margin_kl = 25
        self.triplet_loss_alpha = triplet_loss_alpha

    def prepare_data(self):
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x).float()/255.)])
        self.train_ds = MNIST("data", download=True)
        self.valid_ds = MNIST("data", download=True, train=False)
        to_tensor_dataset = lambda ds: TensorDataset(ds.data.view(-1, 28**2).float()/255., ds.targets)
        self.train_ds, self.valid_ds = map(to_tensor_dataset, [self.train_ds, self.valid_ds])
        self.all_ds = ConcatDataset([self.train_ds, self.valid_ds])
        self.train_triplet_ds = CombinedDataset(MNIST("data", download=True))
                                                # transform=transforms.Lambda(lambda x: torch.flatten(x)/256))
        self.valid_triplet_ds = CombinedDataset(MNIST("data", download=True, train=False)) 
                                                # transform=transforms.Lambda(lambda x: torch.flatten(x)/256), seed=42)

    def pretrain_model(self):
        n_neurons, pretrain_epochs, batch_norm = self.n_neurons, self.pretrain_epochs, self.batch_norm
        self.prepare_data()
        pretrained_model = SimpleAutoencoder(n_neurons, batch_norm=batch_norm, lr=3e-4)
        pretrained_model.val_dataloader = lambda: DataLoader(self.valid_ds, batch_size=256, shuffle=False, num_workers=8)
        pretrained_model.train_dataloader = lambda: DataLoader(self.train_ds, batch_size=256, shuffle=True, num_workers=8)
        gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(gpus=gpus, max_epochs=pretrain_epochs, progress_bar_refresh_rate=20)
        trainer.fit(pretrained_model)
        return pretrained_model
    
    def init_params(self, k, pretrained_model=None):
        if not pretrained_model:
            pretrained_model = self.pretrain_model()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x).float()/255.)])
        X_encoded = pretrained_model.encode_ds(self.all_ds)
        init_gmm = GaussianMixture(k, covariance_type='diag', n_init=5)
        init_gmm.fit(X_encoded)
        return pretrained_model, init_gmm
        
    def train_dataloader(self):
        return DataLoader(self.train_triplet_ds, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid_triplet_ds, batch_size=self.batch_size, num_workers=8)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), self.hparams['lr'], weight_decay=0.00)
        # sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: (epoch+1)/10 if epoch < 10 else 0.95**(epoch//10))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: 0.9**(epoch//10))
        return [opt], [sched]

    def triplet_loss(self, triplets_batch):
        anchor_z, pos_z, neg_z = map(lambda s: self.model.encode(triplets_batch[s]).sample(), 
                                     ['anchor', 'positive', 'negative'])
        anchor_z, pos_z, neg_z = map(lambda t: t / t.norm(dim=1, keepdim=True), [anchor_z, pos_z, neg_z])
        d1, d2 = torch.linalg.norm(anchor_z - pos_z, dim=1), torch.linalg.norm(anchor_z - neg_z, dim=1)
        self.log('anchor_pos_distance', d1.mean(), logger=True)
        self.log('anchor_neg_distance', d2.mean(), logger=True)
        self.log('correct_triplet_pct', (d1 < d2).float().mean()*100)
        loss = torch.relu(d1 - d2 + self.triplet_loss_margin).mean()
        return loss

    def triplet_loss_kl(self, triplets_batch):
        anchor_z_dist, pos_z_dist, neg_z_dist = map(lambda s: self.model.encode(triplets_batch[s]), 
                                                    ['anchor', 'positive', 'negative'])
        d1, d2 = kl_distance(anchor_z_dist, pos_z_dist), kl_distance(anchor_z_dist, neg_z_dist)
        self.log('anchor_pos_distance_kl', d1.mean(), logger=True)
        self.log('anchor_neg_distance_kl', d2.mean(), logger=True)
        self.log('correct_triplet_pct_kl', (d1 < d2).float().mean()*100)
        loss = torch.relu(d1 - d2 + self.triplet_loss_margin_kl).mean()
        return loss
    
    def shared_step(self, batch, batch_idx):
        bx, triplets_batch = batch
        result = self.model.shared_step(bx)
        result['triplet_loss'] = self.triplet_loss(triplets_batch)
        result['triplet_loss_kl'] = self.triplet_loss_kl(triplets_batch)
        # result['loss'] += self.triplet_loss_alpha * result['triplet_loss']
        for k, v in result.items():
            self.log(k, v, logger=True)
        return result

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def cluster_data(self, dl=None):
        if not dl:
            dl = DataLoader(self.all_ds, batch_size=1024, shuffle=False, num_workers=4)
        return self.model.cluster_data(dl)