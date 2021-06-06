import torch
import pytorch_lightning as pl
import importlib
import numpy as np
import wandb
from autoencoder import SimpleAutoencoder, VaDE
from callbacks import  ClusteringEvaluationCallback, cluster_acc
from data_modules import MNISTDataModule

defaults = {'layer1': 500, 'layer2': 500, 'layer3': 2000, 'hid_dim': 10,
            'lr': 3e-4, 
            'batch_size': 256, 
            'device': 'cuda',
            # 'batch_norm': False,
            'clustering_method': 'gmm-full',
            'epochs':100,
            'dataset': 'mnist',
            'data_size': 10000,
            'data_random_state': 42}

wandb.init(config=defaults, project='AE clustering')
config = wandb.config

def main():
    model = SimpleAutoencoder(n_neurons=[784, config.layer1, config.layer2, config.layer3, config.hid_dim], 
                              dataset=config.dataset, data_size=config.data_size, data_random_state=config.data_random_state,
                              lr=config.lr, batch_size=config.batch_size)
    datamodule = MNISTDataModule(dataset=config.dataset, data_size=config.data_size, bs=config.batch_size, seed=config.data_random_state)
    logger = pl.loggers.WandbLogger()
    trainer = pl.Trainer(gpus=1, logger=logger, progress_bar_refresh_rate=10, 
                         callbacks=[ClusteringEvaluationCallback(ds_type='all', on_start=False, method=config.clustering_method),
                          ClusteringEvaluationCallback(ds_type='all', on_start=False, method='gmm-full', postfix='_single')],
                         max_epochs=config.epochs)

    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()

