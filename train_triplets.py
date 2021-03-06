
import torch
import pytorch_lightning as pl
import importlib
import numpy as np
import wandb
from triplet_vade import TripletVaDE
from autoencoder import SimpleAutoencoder, VaDE
from callbacks import ClusteringEvaluationCallback, cluster_acc, PretrainingCallback
from data_modules import MNISTDataModule, CombinedDataModule

defaults = {'layer1': 512, 'layer2': 512, 'layer3': 2048, 'hid_dim': 10,
           'lr': 2e-3, 
           'lr_gmm': 2e-3, 
           'batch_size': 256, 
           'batch_norm': False,
           'weight_decay': 0.0,
           'device': 'cuda',
           'activation': 'relu', 
           'dropout': 0.0,
           'do_pretrain': False,
           'pretrain_epochs': 200,
           'latent_logvar_bias_init': 0.,
           'autoencoder_loss_alpha': 1.0,
           'triplet_loss_margin': 0.5, 
           'triplet_loss_alpha': 0.0, 
           'warmup_epochs':10, 
           'triplet_loss_margin_kl': 20,
           'triplet_loss_alpha_kl': 0., 
           'triplet_loss_margin_cls': 0.5,
           'triplet_loss_alpha_cls': 0, 
           'n_samples_for_triplets': None, 
           'data_size': None, 
           'dataset': 'mnist',
           'n_samples_for_triplets': None,
           'pretrained_model_file': None, 
           'init_gmm_file': None,
           'covariance_type': 'full', 
           'epochs':50}

wandb.init(config=defaults, project='VaDE Triplets')
config = wandb.config

def main():
    triplets_model = TripletVaDE(n_neurons=[784, config.layer1, config.layer2, config.layer3, config.hid_dim], 
                                 batch_norm=config.batch_norm,
                                 data_size=config.data_size,
                                 n_triplets=config.n_triplets,
                                 n_samples_for_triplets=config.n_samples_for_triplets,
                                 lr=config.lr,
                                 lr_gmm=config.lr_gmm,
                                 weight_decay=config.weight_decay,
                                 dataset=config.dataset,
                                 batch_size=config.batch_size,
                                 latent_logvar_bias_init=config.latent_logvar_bias_init,
                                 do_pretrain=config.do_pretrain, 
                                 pretrain_epochs=config.pretrain_epochs, 
                                 pretrained_model_file=config.pretrained_model_file,
                                 init_gmm_file=config.init_gmm_file,
                                 device=config.device,
                                 autoencoder_loss_alpha=config.autoencoder_loss_alpha,
                                 triplet_loss_margin=config.triplet_loss_margin,
                                 triplet_loss_alpha=config.triplet_loss_alpha,
                                 triplet_loss_margin_kl=config.triplet_loss_margin_kl,
                                 triplet_loss_alpha_kl=config.triplet_loss_alpha_kl,
                                 triplet_loss_margin_cls=config.triplet_loss_margin_cls,
                                 triplet_loss_alpha_cls=config.triplet_loss_alpha_cls,
                                 warmup_epochs=config.warmup_epochs,
                                 covariance_type=config.covariance_type)

    logger = pl.loggers.WandbLogger()
    base_datamodule = MNISTDataModule(dataset=config.dataset, data_size=config.data_size, bs=config.batch_size)
    datamodule = CombinedDataModule(base_datamodule, n_samples_for_triplets=config.n_samples_for_triplets, 
                                    n_triplets=config.n_triplets, batch_size=config.batch_size)
    callbacks = [PretrainingCallback(epochs=config.pretrain_epochs, lr=config.pretrain_lr, early_stop=False, save_dir='saved_models'),
                ClusteringEvaluationCallback(), 
                ClusteringEvaluationCallback(ds_type='train'), 
                ClusteringEvaluationCallback(ds_type='valid')]
    trainer = pl.Trainer(gpus=1, logger=logger, progress_bar_refresh_rate=10, log_every_n_steps=20, 
                         callbacks=callbacks, max_epochs=config.epochs)

    trainer.fit(triplets_model, datamodule)


if __name__ == '__main__':
    main()
