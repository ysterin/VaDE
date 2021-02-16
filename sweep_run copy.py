import torch
import pytorch_lightning as pl
import importlib
import numpy as np
import wandb
from pl_modules import PLVaDE
from autoencoder import SimpleAutoencoder, VaDE, ClusteringEvaluationCallback, cluster_acc


def sweep_iteration():
    wandb.init()
    logger = pl.loggers.WandbLogger()
    vade = PLVaDE(n_neurons=[784, 512, 512, wandb.config.n_last_layer, 10], 
                  k=10, 
                  lr=wandb.config.learning_rate, 
                  batch_norm=False, 
                  pretrain_epochs=wandb.config.pretrain_epochs, 
                  covariance_type=wandb.config.covariance_type, 
                  data_size=wandb.config.data_size, 
                  batch_size=wandb.config.batch_size,
                  debug=False)
    
    trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=wandb.config.training_epochs, 
                        callbacks=[ClusteringEvaluationCallback()])
    
    trainer.fit(vade)

