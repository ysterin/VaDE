import torch
import pytorch_lightning as pl
import importlib
import numpy as np
from torch._C import default_generator
import wandb
import ray
import argparse
# from triplet_vade import TripletVaDE
#from triplet_vade import TripletVaDE
from pl_modules import PLVaDE
from autoencoder import SimpleAutoencoder, VaDE
from callbacks import ClusteringEvaluationCallback, cluster_acc, PretrainingCallback, LoadPretrained
from data_modules import MNISTDataModule, BasicDataModule
#pretriained_model = 'pretrained_models/radiant-surf-28/autoencoder-epoch=55-loss=0.011.ckpt'

defaults = {'layer1': 500, 'layer2': 500, 'layer3': 2000, 'hid_dim': 10,
            'dropout': 0., 
            'activation': 'relu',
            'lr': 2e-3, 
            'pretrain_lr': 3e-4,
            'do_pretrain': False,
            'batch_size': 256, 
            'batch_norm': False,
            'device': 'cuda',
            'pretrain_epochs': 100, 
            'data_size': None, 
            'dataset': 'mnist',
            'init_gmm_file': None,
            'pretrained_model_file': None, 
            'multivariate_latent': False,
            'rank': 5,
            'covariance_type': 'full', 
            'epochs':300,
            'seed': 42,
            'data_random_state': 42}

def run_with_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    wandb.init(project='VADE', config=defaults)
    config = wandb.config
    wandb.config.update({'seed': seed}, allow_val_change=True)
    model = PLVaDE(n_neurons=[784, config.layer1, config.layer2, config.layer3, config.hid_dim], 
                    dropout=config.dropout,
                    activation=config.activation,
                    lr=config.lr,
                    pretrain_lr=config.pretrain_lr,
                    # data_size=config.data_size,
                    # dataset=config.dataset,
                    # data_random_seed=config.data_random_state,
                    batch_size=config.batch_size,
                    pretrain_epochs=config.pretrain_epochs, 
                    pretrained_model_file=config.pretrained_model_file,
                    device=config.device,
                    covariance_type=config.covariance_type,
                    init_gmm_file=config.init_gmm_file,
                    multivariate_latent=config.multivariate_latent,
                    do_pretrain=config.do_pretrain,
                    rank=config.rank)

    logger = pl.loggers.WandbLogger(project='VADE')
    datamodule = MNISTDataModule(dataset=config.dataset, data_size=config.data_size, bs=config.batch_size, seed=config.data_random_state)
    callbacks = [ClusteringEvaluationCallback(), 
                #  LoadPretrained(seed=seed, save_dir='saved_models')]
                 PretrainingCallback(epochs=config.pretrain_epochs, lr=config.pretrain_lr, seed=seed, 
                                     log=False, early_stop=False, save_dir='saved_models')]
    trainer = pl.Trainer(gpus=1, logger=logger, progress_bar_refresh_rate=50, log_every_n_steps=10,
                         callbacks=callbacks, max_epochs=config.epochs)

    trainer.fit(model, datamodule=datamodule)
    wandb.join()


SEED = 42 
N_RUNS = 10

def main():
    ray.init(ignore_reinit_error=True)
    seed_sequence = np.random.SeedSequence(SEED)
    streams = [np.random.default_rng(ss) for ss in seed_sequence.spawn(N_RUNS)]
    for i in range(N_RUNS):
        seed = int.from_bytes(streams[i].bytes(4), 'big')
        run_with_seed(seed)
    ray.shutdown()


if __name__ == '__main__':
    main()
