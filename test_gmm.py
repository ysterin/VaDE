
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import argparse, pickle, os
from pl_modules import PLVaDE
import numpy as np 
from autoencoder import cluster_acc, SimpleAutoencoder
import pytorch_lightning as pl
import torch
import sys
import wandb
from pathlib import Path

def test_gmm(run_id, ds_type='train', save=False):

    wandb.init(project='AE clustering', resume='must', id=run_id, config=defaults)
    autoencoder = SimpleAutoencoder(n_neurons=wandb.config.n_neurons, dataset=wandb.config.dataset, 
                                    data_size=wandb.config.data_size, data_random_state=wandb.config.data_random_state)
    autoencoder.prepare_data()
    run_path = Path(f'AE clustering/{run_id}')
    checkpoint_files = os.listdir(run_path /'checkpoints')
    checkpoint_file = checkpoint_files[-1]
    autoencoder.load_state_dict(torch.load(run_path /'checkpoints' / checkpoint_file)['state_dict'])

    y_true = np.stack([autoencoder.all_ds[i][1] for i in range(len(autoencoder.all_ds))])
    X_encoded = autoencoder.encode_ds(autoencoder.all_ds)
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
        os.makedirs(f'saved_gmm_init/{run_id}', exist_ok=True)
        if save:
            with open(f'saved_gmm_init/{run_id}/gmm-full-acc={acc:.2f}.pkl', 'wb') as file:
                pickle.dump(init_gmm, file)

parser = argparse.ArgumentParser()
parser.add_argument('run_id', type=str)
parser.add_argument('--checkppint_file', type=str, required=False, nargs=1)
parser.add_argument('--ds_type', type=str, choices=['train', 'valid', 'all'], default='all')
parser.add_argument('--save', action='store_true')

defaults = {'dataset': 'mnist', 'data_size': None, 'data_random_state': 42}

if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    # run_id = args.run_id
    print(args)
    test_gmm(args.run_id, args.ds_type, args.save)
    # wandb.init(project='AE clustering', resume='must', id=run_id, config=defaults)
    # autoencoder = SimpleAutoencoder(n_neurons=wandb.config.n_neurons, dataset=wandb.config.dataset, 
    #                                 data_size=wandb.config.data_size, data_random_state=wandb.config.data_random_state)
    # autoencoder.prepare_data()
    # run_path = Path(f'AE clustering/{run_id}')
    # checkpoint_files = os.listdir(run_path /'checkpoints')
    # print(checkpoint_files)
    # autoencoder.load_state_dict(torch.load(run_path /'checkpoints' / checkpoint_files[-1])['state_dict'])
    # test_gmm(autoencoder)

    # model = PLVaDE(n_neurons=[784, 512, 512, 2048, 10], k=10, lr=1e-3, covariance_type='full', batch_size=256, pretrain_epochs=10,
    #         # pretrained_model_file="AE clustering/5wn5ybl3/checkpoints/epoch=69-step=16449.ckpt", 
    #         pretrained_model_file=run_path /'checkpoints' / checkpoint_files[-1],
    #         init_gmm_file='saved_gmm_init/5wn5ybl3/gmm-full-0.pkl',
    #         multivariate_latent=True, rank=5, device='cuda:0', dataset='mnist')
    # pretrained_model = model.pretrained_model[0]
    # pretrained_model.prepare_data()
    # test_gmm(pretrained_model)
    # api = wandb.Api()
    # api.run('shukistern/AE clustering/' + args.run)
    # run = api.run('shukistern/AE clustering/dla63r4s')
    # wandb.restore('checkpoints/')
    # print(wandb.run)

