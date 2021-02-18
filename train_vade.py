
import torch
import pytorch_lightning as pl
import importlib
import numpy as np
import wandb
# from triplet_vade import TripletVaDE
#from triplet_vade import TripletVaDE
from pl_modules import PLVaDE
from autoencoder import SimpleAutoencoder, VaDE, ClusteringEvaluationCallback, cluster_acc

pretriained_model = 'pretrained_models/radiant-surf-28/autoencoder-epoch=55-loss=0.011.ckpt'

defaults = {'layer1': 512, 'layer2': 512, 'layer3': 2048, 'hid_dim': 10,
            'lr': 2e-3, 
            'batch_size': 256, 
            'batch_norm': False,
            'device': 'cuda',
            'pretrain_epochs': 50, 
            'data_size': None, 
            'dataset': 'mnist',
            'init_gmm_file': None,
            'pretrained_model_file': pretriained_model, 
            'multivariate_latent': True,
            'rank': 5,
            'covariance_type': 'full', 
            'epochs':50}

wandb.init(config=defaults, project='VADE')
config = wandb.config

def main():
    model = PLVaDE(n_neurons=[784, config.layer1, config.layer2, config.layer3, config.hid_dim], 
                                 lr=config.lr,
                                 data_size=config.data_size,
                                 dataset=config.dataset,
                                 batch_size=config.batch_size,
                                 pretrain_epochs=config.pretrain_epochs, 
                                 pretrained_model_file=config.pretrained_model_file,
                                 device=config.device,
                                 covariance_type=config.covariance_type,
                                 init_gmm_file=config.init_gmm_file,
                                 multivariate_latent=config.multivariate_latent,
                                 rank=config.rank)

    logger = pl.loggers.WandbLogger()
    trainer = pl.Trainer(gpus=1, logger=logger, progress_bar_refresh_rate=10, 
                         callbacks=[ClusteringEvaluationCallback()], max_epochs=config.epochs)

    trainer.fit(model)


if __name__ == '__main__':
    main()
