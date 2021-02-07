import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, ConcatDataset, TensorDataset
from torch import nn, distributions
from torch.nn import functional as F
from torch import distributions as D
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torch.distributions import Normal, Laplace, kl_divergence, kl, Categorical
import math
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from torch import autograd
from pytorch_lightning.callbacks import Callback
from scipy.optimize import linear_sum_assignment as linear_assignment



transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Lambda(lambda x: torch.flatten(x))])


def cluster_acc(Y_pred, Y):
    # from sklearn.utils. import linear_assignment
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
    def __init__(self, on_start=True, **kwargs):
        super(ClusteringEvaluationCallback, self).__init__()
        self.on_start = on_start
        self.kwargs = kwargs

    def on_epoch_start(self, trainer, pl_module):
        if self.on_start:
            gt, labels, _ = pl_module.cluster_data(**self.kwargs)
            nmi, acc1, acc2 = metrics.normalized_mutual_info_score(labels, gt), clustering_accuracy(labels, gt), clustering_accuracy(gt, labels)
            acc3 = cluster_acc(labels, gt)
            pl_module.log('NMI', nmi, on_epoch=True)
            pl_module.log('ACC1', acc1, on_epoch=True)
            pl_module.log('ACC2', acc2, on_epoch=True)

    def on_epoch_end(self, trainer, pl_module):
        if not self.on_start:
            gt, labels, _ = pl_module.cluster_data(**self.kwargs)
            nmi, acc1, acc2 = metrics.normalized_mutual_info_score(labels, gt), clustering_accuracy(labels, gt), clustering_accuracy(gt, labels)
            acc3 = cluster_acc(labels, gt)
            pl_module.log('NMI', nmi, on_epoch=True)
            pl_module.log('ACC1', acc1, on_epoch=True)
            pl_module.log('ACC2', acc2, on_epoch=True)


def get_autoencoder(n_neurons, batch_norm=True):
    enc_layers = len(n_neurons)
    layer_dims = n_neurons + n_neurons[-2::-1]
    n_layers = len(layer_dims)
    return nn.Sequential(*[nn.Sequential(nn.Linear(layer_dims[i], layer_dims[i+1]),
                                         nn.Identity() \
                                            if i+2 == enc_layers or i+2 == n_layers or not batch_norm \
                                            else nn.BatchNorm1d(layer_dims[i+1]),
                                         nn.Identity() \
                                            if i+2 == enc_layers or i+2 == n_layers \
                                            else nn.ELU()) \
                           for i in range(n_layers - 1)])

class LatentDistribution(nn.Module):
    prior = Normal(0, 1)
    def __init__(self, in_features, out_features, sigma=None, same_sigma=False):
        super(LatentDistribution, self).__init__()
        self.mu_fc = nn.Linear(in_features, out_features)
        if sigma:
            self.sigma = sigma
        else:
            if same_sigma:
                self.logvar_fc = nn.Linear(in_features, 1)
                self.logvar_fc.weight.data.zero_()
                self.logvar_fc.bias.data.zero_()
            else:
                self.logvar_fc = nn.Linear(in_features, out_features)
        
    
    def forward(self, x):
        mu = self.mu_fc(x)
        if hasattr(self, 'sigma'):
            sigma = self.sigma
        else:
            logvar = self.logvar_fc(x)
            sigma = torch.exp(logvar / 2)
        self.dist = Normal(mu, sigma)
        return self.dist
    
    def sample(self, l=1):
        return self.dist.rsample()

    def kl_loss(self, prior=None):
        if not prior:
            prior = self.prior
        return kl_divergence(self.dist, prior).sum(dim=-1)


class MultivariateNormalDistribution(nn.Module):
    def __init__(self, in_features, out_features):
        super(MultivariateNormalDistribution, self).__init__()

class BernoulliDistribution(nn.Module):
    def __init__(self, in_features, out_features):
        super(BernoulliDistribution, self).__init__()
        self.probs = nn.Sequential(nn.Linear(in_features, out_features), nn.Sigmoid())
    
    def forward(self, x):
        self.dist = D.Bernoulli(probs=self.probs(x))
        return self.dist


def cross_entropy(P, Q):
    return kl_divergence(P, Q) + P.entropy()

def kl_distance(P, Q):
    return 0.5 * (kl_divergence(P, Q) + kl_divergence(Q, P))

def xlogx(x, eps=1e-12):
    xlog = x * (x + eps).log()
    return xlog

def get_encoder_decoder(n_neurons, batch_norm=True):
    n_layers = len(n_neurons) - 1
    encoder_layers = [nn.Sequential(nn.Linear(n_neurons[i], n_neurons[i+1]),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(n_neurons[i+1]) if batch_norm else nn.Identity()) for i in range(n_layers - 1)]
    encoder_layers.append(nn.Linear(n_neurons[-2], n_neurons[-1]))
    n_neurons = n_neurons[::-1]
    decoder_layers = [nn.Sequential(nn.Linear(n_neurons[i], n_neurons[i+1]),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(n_neurons[i+1]) if batch_norm else nn.Identity()) for i in range(n_layers - 1)]
    decoder_layers.append(nn.Sequential(nn.Linear(n_neurons[-2], n_neurons[-1]), nn.Sigmoid()))
    return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)


class SimpleAutoencoder(pl.LightningModule):
    def __init__(self, n_neurons=[784, 512, 256, 10], lr=1e-3, batch_norm=False, batch_size=512):
        super(SimpleAutoencoder, self).__init__()
        self.hparams = {'lr': lr}
        self.batch_size = batch_size
        self.k = n_neurons[-1]
        self.encoder, self.decoder = get_encoder_decoder(n_neurons, batch_norm)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), self.hparams['lr'])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.2 ,patience=20, verbose=True, min_lr=1e-6)
        scheduler = {'optimizer': opt, 'scheduler': sched, 'interval': 'epoch', 'monitor': 'val_checkpoint_on', 'reduce_on_plateau': True}
        return scheduler

    def shared_step(self, batch, batch_idx):
        bx, by = batch
        z = self.encoder(bx)
        out = self.decoder(z)
        loss = F.mse_loss(out, bx)
        # bce_loss = F.binary_cross_entropy(torch.clamp(out, 1e-6, 1 - 1e-6), bx, reduction='mean')
        bce_loss = F.binary_cross_entropy(out, bx, reduction='mean')
        self.log('loss', loss)
        self.log('bce_loss', bce_loss)
        lmbda = 0.00
        reg = lmbda * (z**2).mean()
        self.log('regularization', reg)
        return bce_loss
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def train_dataloader(self):
        dataset = MNIST("data", download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def val_dataloader(self):
        dataset = MNIST("data", download=True, transform=transform, train=False)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def encode(self, batch):
        bx, by = batch
        return self.encoder(bx)

    def encode_ds(self, ds):
        dl = DataLoader(ds, batch_size=2**11, num_workers=8, shuffle=False)
        self.eval()
        encoded = []
        with torch.no_grad():
            for batch in dl:
                encoded.append(self.encode(batch).detach().cpu().numpy())
        return np.concatenate(encoded, axis=0)

    def encode_dl(self, dl):
        self.eval()
        encoded = []
        labels = []
        with torch.no_grad():
            for bx, by in dl:
                encoded.append(self.encode((bx.cuda(), by)).detach().cpu().numpy())
                labels.append(by.detach().cpu().numpy())

        return np.concatenate(encoded, axis=0), np.concatenate(labels, axis=0)


    def cluster(self, ds, k=10, method='kmeans'):
        X = self.encode_ds(ds)
        kmeans = KMeans(k)
        return kmeans.fit_predict(X)

    def cluster_data(self, dl=None, method='gmm-diag', n_init=3):
        self.eval()
        if not dl:
            dl = self.val_dataloader()
        if method == 'kmeans':
            clustering_algo = KMeans(n_clusters=self.k)
        elif method == 'gmm-full':
            clustering_algo = GaussianMixture(n_components=self.k, covariance_type='full', n_init=n_init)
        elif method == 'gmm-diag':
            clustering_algo = GaussianMixture(n_components=self.k, covariance_type='diag', n_init=n_init)
        X_encoded, true_labels = self.encode_dl(dl)
        # import pdb; pdb.set_trace()
        predicted_labels = clustering_algo.fit_predict(X_encoded)
        return true_labels, predicted_labels, X_encoded


class VaDE(nn.Module):
    def __init__(self, n_neurons=[784, 512, 256, 10], batch_norm=True, k=10, lr=1e-3, device='cuda',
                 pretrain_model=None, init_gmm=None, logger=None):
        super(VaDE, self).__init__()
        self.k = k
        self.log = logger
        self.n_neurons, self.batch_norm = n_neurons, batch_norm
        self.hparams = {'lr': lr}
        self.latent_dim = n_neurons[-1]
        self.mixture_logits = nn.Parameter(torch.zeros(k, device=device))
        self.mu_c = nn.Parameter(torch.zeros(self.latent_dim, k, device=device))
        self.sigma_c = nn.Parameter(torch.ones(self.latent_dim, k, device=device))
        n_layers = len(n_neurons) - 1
        layers = list()
        for i in range(n_layers-1):
            layer = nn.Sequential(nn.Linear(n_neurons[i], n_neurons[i+1]),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(n_neurons[i+1]) if batch_norm else nn.Identity())
            layer.register_forward_hook(self.register_stats(f"encoder_{i}"))
            layers.append(layer)
        self.encoder = nn.Sequential(*layers)
        self.latent_dist = LatentDistribution(n_neurons[-2], n_neurons[-1])
        
        self.latent_dist.mu_fc.register_forward_hook(self.register_stats(f"latent mu"))
        self.latent_dist.logvar_fc.register_forward_hook(self.register_stats(f"latent logvar"))
        layers = list()
        n_neurons = n_neurons[::-1]
        for i in range(n_layers-1):
            layers.append(nn.Sequential(nn.Linear(n_neurons[i], n_neurons[i+1]),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(n_neurons[i+1]) if batch_norm else nn.Identity()))
        self.decoder = nn.Sequential(*layers)
        self.sigma = nn.Parameter(torch.ones(1)*np.sqrt(0.1), requires_grad=True)
        # self.sigma = 1
        # self.out_dist = LatentDistribution(n_neurons[-2], n_neurons[-1], same_sigma=True)
        self.out_dist = BernoulliDistribution(n_neurons[-2], n_neurons[-1])
        if pretrain_model is not None and init_gmm is not None:
            self.mixture_logits.data = torch.Tensor(np.log(init_gmm.weights_))
            self.mu_c.data = torch.Tensor(init_gmm.means_.T)
            self.sigma_c.data = torch.Tensor(init_gmm.covariances_.T).sqrt()
            self.encoder.load_state_dict(pretrain_model.encoder[:-1].state_dict())
            self.decoder.load_state_dict(pretrain_model.decoder[:-1].state_dict())
            self.latent_dist.mu_fc.load_state_dict(pretrain_model.encoder[-1].state_dict())
            # self.out_dist.mu_fc.load_state_dict(pretrain_model.decoder[-1].state_dict())
            self.out_dist.probs[0].load_state_dict(pretrain_model.decoder[-1][0].state_dict())
            
    def forward(self, bx):
        x = self.encoder(bx)
        z_dist = self.latent_dist(x)
        z = z_dist.rsample()
        x_dist = self.out_dist(self.decoder(z))
        return x_dist

    def register_stats(self, layer_name):
        def hook_fn(module, inputs, outputs):
            self.log(f"{layer_name} mean", inputs[0].mean())
            self.log(f"{layer_name} std", inputs[0].std(dim=0).mean())
        return hook_fn

    def encode(self, bx):
        x = self.encoder(bx)
        return self.latent_dist(x)

    def shared_step(self, bx):
        x = self.encoder(bx)
        z_dist = self.latent_dist(x)
        if z_dist.scale.max() > 100:
            import pdb; pdb.set_trace()
        self.log("latent dist std", z_dist.scale.mean().detach())
        z = z_dist.rsample()
        x_dist = self.out_dist(self.decoder(z))
        x_recon_loss = - x_dist.log_prob(bx).sum(dim=-1)
        # bce_loss = F.binary_cross_entropy(torch.clamp(x_dist.mean, 1e-6, 1.0-1e-6), bx)
        bce_loss = F.binary_cross_entropy(x_dist.mean, bx, reduction='mean')
        ###################################
        
        comp_dists = [D.Normal(self.mu_c[:,i], self.sigma_c[:,i]) for i in range(self.k)]
        gmm = D.MixtureSameFamily(D.Categorical(logits=self.mixture_logits), D.Normal(self.mu_c, self.sigma_c))
        log_p_z_c = gmm.component_distribution.log_prob(z.unsqueeze(-1)).sum(dim=1)
        log_q_c_z = torch.log_softmax(log_p_z_c + self.mixture_logits, dim=-1)  # dims: (bs, k)
        q_c_z = log_q_c_z.exp()
        cross_entropies = torch.stack([cross_entropy(z_dist, comp_dists[i]).sum(dim=1) for i in range(self.k)], dim=-1)
        crosent_loss1 = - (cross_entropies * q_c_z).sum(dim=-1)
        crosent_loss2 = (q_c_z * (gmm.mixture_distribution.probs[None] + 1e-9).log()).sum(dim=-1)
        ent_loss1 = z_dist.entropy().sum(dim=-1)
        ent_loss2 = - (q_c_z * log_q_c_z).sum(dim=-1)

        self.log('ent_loss1', - ent_loss1.mean(), logger=True)
        self.log('crosent_loss1', - crosent_loss1.mean(), logger=True)
        self.log('ent_loss2', - ent_loss2.mean(), logger=True)
        self.log('crosent_loss2', - crosent_loss2.mean(), logger=True)
        self.log('gmm likelihood', - gmm.log_prob(z).mean())

        ############################################################################
        kl_loss = - crosent_loss1 - crosent_loss2 - ent_loss1 - ent_loss2
        loss = x_recon_loss + kl_loss

        if loss.isnan().any():
            import pdb; pdb.set_trace()

        result = {'loss': loss.mean(),
                  'x_recon_loss': x_recon_loss.mean(),
                  'kl_loss': kl_loss.mean(),
                  'bce_loss': bce_loss.mean()}
                #   'z_dist': z_dist, 'z': z}
        return result

    def cluster_data(self, dl=None):
        self.eval()
        vade_gmm = D.MixtureSameFamily(D.Categorical(logits=self.mixture_logits), D.Normal(self.mu_c, self.sigma_c))
        true_labels = []
        predicted_labels = []
        X_encoded = []
        with torch.no_grad():
            for bx, by in dl:
                if torch.cuda.is_available():
                    bx = bx.cuda()
                x_encoded = self.latent_dist(self.encoder(bx)).loc
                X_encoded.append(x_encoded)
                true_labels.append(by)
                log_p_z_given_c = vade_gmm.component_distribution.log_prob(x_encoded.unsqueeze(2)).sum(dim=1)
                predicted_labels.append((log_p_z_given_c + vade_gmm.mixture_distribution.logits).softmax(dim=-1).argmax(dim=-1))
                
        true_labels = torch.cat(true_labels).cpu().numpy()
        predicted_labels = torch.cat(predicted_labels).cpu().numpy()
        X_encoded = torch.cat(X_encoded).cpu().numpy()
        return true_labels, predicted_labels, X_encoded
