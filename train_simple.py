import torch
import pytorch_lightning as pl
import importlib
import numpy as np
import wandb
# from triplet_vade import TripletVaDE
from triplet_vade import TripletVaDE
from autoencoder import SimpleAutoencoder, VaDE, ClusteringEvaluationCallback, cluster_acc


defaults = {'layer1': 512, 'layer2': 512, 'layer3': 2048, 'hid_dim': 10,
            'lr': 2e-3, 
            'batch_size': 256, 
            'device': 'cuda',
            'clustering_method': 'gmm-diag',
            'epochs':5,
            'dataset': 'fmnist',
            'data_size': 5000,
            'data_random_state': 42}

wandb.init(config=defaults, project='AE clustering')
config = wandb.config

def main():
    model = SimpleAutoencoder(n_neurons=[784, config.layer1, config.layer2, config.layer3, config.hid_dim], 
                              dataset=config.dataset, data_size=config.data_size, data_random_state=config.data_random_state,
                              lr=config.lr, batch_size=config.batch_size)
    logger = pl.loggers.WandbLogger()
    trainer = pl.Trainer(gpus=1, logger=logger, progress_bar_refresh_rate=10, 
                         callbacks=[ClusteringEvaluationCallback(on_start=False, method=config.clustering_method)],
#                         ClusteringEvaluationCallback(ds_type='train', on_start=False, method=config.clustering_method)],
                         max_epochs=config.epochs)

    trainer.fit(model)


if __name__ == '__main__':
    main()

