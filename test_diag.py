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


class KLAnnealing(pl.Callback):
    def __init__(self, schedule, max_alpha=1.):
        super(KLAnnealing, self).__init__()
        # self.init, self.lmbda, self.max = init, lmbda, max_alpha
        self.max = max_alpha
        self.schedule = schedule
        self.idx = 0
    
    def on_train_epoch_start(self, trainer, pl_module):
        assert isinstance(pl_module, (PLVaDE, ))
        # alpha = self.schedule(self.idx)
        alpha = min(self.max, self.schedule(self.idx))
        pl_module.model.alpha_kl = alpha
        pl_module.log('alpha_kl', alpha)
        self.idx += 1
        # return super().on_t   rain_epoch_start(trainer, pl_module)


# class LinearKLAnnealing(pl.Callback):
#     def __init__(self, init=0, m=5e-2, max_alpha=1.):
#         super(LinearKLAnnealing, self).__init__()
#         self.init, self.m, self.max = init, m, max_alpha
#         self.idx = 0
SEED = 1456625023
# SEED = 2537561875

torch.manual_seed(SEED)
np.random.seed(SEED)

cov_type = 'diag'

model = PLVaDE(n_neurons=[784, 500, 500, 2000, 10], 
                lr=3e-4 / 784,
                # lr_gmm=2e-2 / 10,
                # pretrain_lr=3e-4,
                # data_size=config.data_size,
                # dataset=config.dataset,
                # data_random_seed=config.data_random_state,
                # batch_size=256,
                # pretrain_epochs=config.pretrain_epochs, 
                # pretrained_model_file=config.pretrained_model_file,
                # device=config.device,
                alpha_kl = 1.,
                warmup_epochs=0,
                latent_logvar_bias_init=-5,
                covariance_type=cov_type,
                # init_gmm_file=config.init_gmm_file,
                multivariate_latent=False,
                do_pretrain=False,
                rank=1)

logger = pl.loggers.WandbLogger(project='VADE', group='test_diag_different_gmm_lr')

datamodule = MNISTDataModule(dataset='mnist', data_size=None, bs=256, seed=42)
callbacks = [ClusteringEvaluationCallback(),
                LoadPretrained(seed=SEED, save_dir='saved_models'),
                KLAnnealing(schedule=lambda i: 1)]

trainer = pl.Trainer(gpus=1, logger=logger, progress_bar_refresh_rate=10, log_every_n_steps=1, 
                         callbacks=callbacks, max_epochs=5000)

trainer.fit(model, datamodule)