
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import argparse
from pl_modules import PLVaDE
import numpy as np 
from autoencoder import cluster_acc, SimpleAutoencoder
import pytorch_lightning as pl
import torch
import sys
import wandb
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('run', type=str)


def test_gmm(pretrained_model):
    # model = PLVaDE(n_neurons=[784, 512, 512, 2048, 10], k=10, lr=1e-3, covariance_type='full', batch_size=256, pretrain_epochs=10,
    #             pretrained_model_file="AE clustering/5wn5ybl3/checkpoints/epoch=69-step=16449.ckpt", 
    #             # init_gmm_file='saved_gmm_init/5wn5ybl3/gmm-full-0.pkl',
    #             multivariate_latent=True, rank=5, device='cuda:0')
    # pretrained_model = model.pretrained_model[0]
    y_true = np.stack([pretrained_model.all_ds[i][1] for i in range(len(pretrained_model.all_ds))])
    X_encoded = pretrained_model.encode_ds(pretrained_model.all_ds)
    for i in range(5):
        init_gmm = GaussianMixture(10, covariance_type='full', n_init=3)
        y_pred = init_gmm.fit_predict(X_encoded)
        acc = cluster_acc(y_true, y_pred)
        nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
        ari = metrics.adjusted_rand_score(y_true, y_pred)
        print('log likelihood:', init_gmm.score(X_encoded))
        print('Accuracy: ', acc)
        print('NMI: ', nmi)
        print('ARI: ', ari)
        # if acc > 0.9:
        #     import pickle
        #     with open(f'saved_gmm_init/5wn5ybl3/gmm-diag-{i}.pkl', 'wb') as file:
        #         pickle.dump(init_gmm, file)

import os
if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    print(args.run)
    wandb.init(project='AE clustering', resume='must', id=args.run)
    autoencoder = SimpleAutoencoder(n_neurons=wandb.config.n_neurons)
    autoencoder.prepare_data()
    run_path = Path(f'AE clustering/{args.run}')
    checkpoint_files = os.listdir(run_path /'checkpoints')
    print(checkpoint_files)
    autoencoder.load_from_checkpoint(run_path /'checkpoints' / checkpoint_files[-1])
    test_gmm(autoencoder)
    # api = wandb.Api()
    # api.run('shukistern/AE clustering/' + args.run)
    # run = api.run('shukistern/AE clustering/dla63r4s')
    # wandb.restore('checkpoints/')
    # print(wandb.run)

