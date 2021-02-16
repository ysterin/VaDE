import torch
import pytorch_lightning as pl
import importlib
import numpy as np
import wandb
# from triplet_vade import TripletVaDE
from triplet_vade import TripletVaDE
from autoencoder import SimpleAutoencoder, VaDE, ClusteringEvaluationCallback, cluster_acc


defaults = {'layer1': 512, 'layer2': 512, 'layer3': 2048,
            'lr': 2e-3, 
            'batch_size': 256, 
            'batch_norm': False,
            'device': 'cuda',
            'data_size': None, 
            'epochs':50}

wandb.init(config=defaults, project='AE clustering')
config = wandb.config

def main():
    model = SimpleAutoencoder(n_neurons=[784, config.layer1, config.layer2, config.layer3, 10], 
                              lr=config.lr, batch_norm=config.batch_norm, batch_size=config.batch_size)
    logger = pl.loggers.WandbLogger()
    trainer = pl.Trainer(gpus=1, logger=logger, progress_bar_refresh_rate=10, 
                         callbacks=[ClusteringEvaluationCallback()], max_epochs=config.epochs)

    trainer.fit(model)


if __name__ == '__main__':
    main()

