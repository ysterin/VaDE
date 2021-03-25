import torch
import pytorch_lightning as pl
import importlib
import numpy as np
import wandb
# from triplet_vade import TripletVaDE
import triplet_vade
import pl_modules
# importlib.reload(triplet_vade)
from autoencoder import SimpleAutoencoder, VaDE
from callbacks import ClusteringEvaluationCallback, cluster_acc, LoadPretrained
from pl_modules import PLVaDE
from data_modules import MNISTDataModule, BasicDataModule
SEED = 1456625023
# SEED = 2537561875

torch.manual_seed(SEED)
np.random.seed(SEED)


model = PLVaDE(n_neurons=[784, 500, 500, 2000, 10], 
                lr=2e-3,
                lr_gmm=2e-3,
                # pretrain_lr=3e-4,
                # data_size=config.data_size,
                # dataset=config.dataset,
                # data_random_seed=config.data_random_state,
                # batch_size=256,
                # pretrain_epochs=config.pretrain_epochs, 
                # pretrained_model_file=config.pretrained_model_file,
                # device=config.device,
                warmup_epochs=0,
                latent_logvar_bias_init=-5,
                covariance_type='diag',
                # init_gmm_file=config.init_gmm_file,
                multivariate_latent=False,
                do_pretrain=False,
                rank=1)

logger = pl.loggers.WandbLogger(project='VADE', group='test_diag_different_gmm_lr')
datamodule = MNISTDataModule(dataset='mnist', data_size=None, bs=256, seed=42)
callbacks = [ClusteringEvaluationCallback(), 
                LoadPretrained(seed=SEED, save_dir='saved_models')]

trainer = pl.Trainer(gpus=1, logger=logger, progress_bar_refresh_rate=10, log_every_n_steps=1, 
                         callbacks=callbacks, max_epochs=20)

trainer.fit(model, datamodule)