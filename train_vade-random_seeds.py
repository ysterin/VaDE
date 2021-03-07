import torch
import pytorch_lightning as pl
import importlib
import numpy as np
from torch._C import default_generator
import wandb
# from triplet_vade import TripletVaDE
#from triplet_vade import TripletVaDE
from pl_modules import PLVaDE
from autoencoder import SimpleAutoencoder, VaDE, ClusteringEvaluationCallback, cluster_acc

#pretriained_model = 'pretrained_models/radiant-surf-28/autoencoder-epoch=55-loss=0.011.ckpt'

defaults = {'layer1': 500, 'layer2': 500, 'layer3': 2000, 'hid_dim': 10,
            'dropout': 0.3, 
            'activation': 'relu',
            'lr': 2e-3, 
            'pretrain_lr': 3e-4,
            'batch_size': 64, 
            'batch_norm': False,
            'device': 'cuda',
            'pretrain_epochs': 200, 
            'data_size': 2000, 
            'dataset': 'mnist',
            'init_gmm_file': None,
            'pretrained_model_file': None, 
            'multivariate_latent': False,
            'rank': 5,
            'covariance_type': 'full', 
            'epochs':100,
            'seed': 42}

# wandb.init(config=defaults, project='VADE')
# config = wandb.config
SEED = 42
N_RUNS = 1
# torch.manual_seed(SEED)
# np.random.seed(SEED)

def main():
    # torch.manual_seed(SEED)
    # seeds = torch.randint(10000000, size=(N_RUNS,))
    seed_sequence = np.random.SeedSequence(SEED)
    streams = [np.random.default_rng(ss) for ss in seed_sequence.spawn(N_RUNS)]
    for i in range(N_RUNS):
        seed = int.from_bytes(streams[i].bytes(4), 'big')
        torch.manual_seed(seed)
        np.random.seed(seed)
        wandb.init(config=defaults, project='VADE', group='seeds-mnist-2k-1')
        config = wandb.config
        wandb.config.update({'seed': seed}, allow_val_change=True)
        model = PLVaDE(n_neurons=[784, config.layer1, config.layer2, config.layer3, config.hid_dim], 
                        dropout=config.dropout,
                        activation=config.activation,
                        lr=config.lr,
                        pretrain_lr=config.pretrain_lr,
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

        logger = pl.loggers.WandbLogger(project='VADE')
        trainer = pl.Trainer(gpus=1, logger=logger, progress_bar_refresh_rate=50, log_every_n_steps=1,
                             callbacks=[ClusteringEvaluationCallback()], max_epochs=config.epochs)

        trainer.fit(model)
        wandb.finish()
        import ray; ray.shutdown()


if __name__ == '__main__':
    main()
