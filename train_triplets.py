
import torch
import pytorch_lightning as pl
import importlib
import numpy as np
import wandb
# from triplet_vade import TripletVaDE
from triplet_vade import TripletVaDE
from autoencoder import SimpleAutoencoder, VaDE, ClusteringEvaluationCallback, cluster_acc

pretriained_model = 'pretrained_models/radiant-surf-28/autoencoder-epoch=55-loss=0.011.ckpt'

defaults = {'layer1': 512, 'layer2': 512, 'layer3': 2048,
            'lr': 2e-3, 
            'lr_gmm': 2e-3, 
            'batch_size': 256, 
            'batch_norm': False,
            'device': 'cuda',
            'pretrain_epochs': 50, 
            'triplet_loss_margin': 0.5, 
            'triplet_loss_alpha': 0.0, 
            'warmup_epochs':10, 
            'triplet_loss_margin_kl': 20,
            'triplet_loss_alpha_kl': 1., 
            'n_samples_for_triplets': None, 
            'data_size': None, 
            'pretrained_model_file': pretriained_model, 
            'covariance_type': 'full', 
            'epochs':50}

wandb.init(config=defaults, project='VaDE Triplets')
config = wandb.config

def main():
    triplets_model = TripletVaDE(n_neurons=[784, config.layer1, config.layer2, config.layer3, 10], 
                                 batch_norm=config.batch_norm,
                                 lr=config.lr,
                                 lr_gmm=config.lr_gmm,
                                 data_size=config.data_size,
                                 batch_size=config.batch_size,
                                 pretrain_epochs=config.pretrain_epochs, 
                                 pretrained_model_file=config.pretrained_model_file,
                                 device=config.device,
                                 triplet_loss_margin=config.triplet_loss_margin,
                                 triplet_loss_alpha=config.triplet_loss_alpha,
                                 triplet_loss_margin_kl=config.triplet_loss_margin_kl,
                                 triplet_loss_alpha_kl=config.triplet_loss_alpha_kl,
                                 warmup_epochs=config.warmup_epochs,
                                 covariance_type=config.covariance_type)

    logger = pl.loggers.WandbLogger()
    trainer = pl.Trainer(gpus=1, logger=logger, progress_bar_refresh_rate=10, 
                         callbacks=[ClusteringEvaluationCallback()], max_epochs=config.epochs)

    trainer.fit(triplets_model)


if __name__ == '__main__':
    main()