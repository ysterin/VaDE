
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import argparse, pickle, os
from pl_modules import PLVaDE
import numpy as np 
from autoencoder import SimpleAutoencoder
from callbacks import cluster_acc
import pytorch_lightning as pl
import torch
import sys
import wandb
from pathlib import Path
import ray


def kl_mixture_score(gmm: GaussianMixture, X_encoded: np.array, eps=1e-8):
    q_c_z = gmm.predict_proba(X_encoded)
    log_q_c_z = np.log(q_c_z + eps)
    mixture_logits = np.log(gmm.weights_)
    kl_div = np.einsum('ij,ij->i', q_c_z, log_q_c_z - mixture_logits)
    return kl_div.mean()

#fit a GMM model on the data X.
@ray.remote
def fit_gmm(x, n_clusters=10, covariance_type='full', n_init=1, random_state=None):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, n_init=n_init, random_state=random_state)
    gmm.fit(x)
    log_likelihood = gmm.score(x)
    kl_score = kl_mixture_score(gmm, x)
    return gmm, log_likelihood, kl_score

# trains n GMM models and selects the best according to kl score or likelihood.
def best_of_n_gmm_ray(x, n_clusters=10, n=10, covariance_type='full', n_init=1):
    ray.init(ignore_reinit_error=True)
    ss = np.random.SeedSequence(42)
    random_states = [np.random.RandomState(np.random.PCG64(c)) for c in ss.spawn(n)]
    x_id =  ray.put(x)
    scores = ray.get([fit_gmm.remote(x_id, n_clusters, covariance_type, n_init, st) for st in random_states])
    best_gmm_ll, best_likelihood, _ = max(scores, key=lambda o: o[1])
    best_gmm_kl, _, best_kl_score = max(scores, key=lambda o: o[2])
    return best_gmm_ll, best_gmm_kl


# trains gmm model on the data encoded by the autoencder according to run_id.
def test_gmm(run_id, ds_type='train', n_runs=5, save=False, cov_type='full'):
    project = 'AE clustering'
    try:
        wandb.init(project=project, resume='must', id=run_id)
    except wandb.errors.error.UsageError:
        project = 'AE-clustering'
        wandb.init(project=project, resume='must', id=run_id)
    autoencoder = SimpleAutoencoder(n_neurons=wandb.config.n_neurons, dataset=wandb.config.dataset, 
                                    data_size=wandb.config.data_size, data_random_state=wandb.config.data_random_state)
    print(wandb.config.dataset)
    autoencoder.prepare_data()
    run_path = Path(f'{project}/{run_id}')
    if not os.path.exists(run_path): 
        for path in Path('wandb').glob(f"run-*-{run_id}"):
            if os.path.exists(path / 'files' / project):
                run_path = path / 'files' / project / run_id 
                break

    checkpoint_files = os.listdir(run_path / 'checkpoints')
    checkpoint_file = checkpoint_files[-1]
    autoencoder.load_state_dict(torch.load(run_path /'checkpoints' / checkpoint_file)['state_dict'])

    y_true = np.stack([autoencoder.all_ds[i][1] for i in range(len(autoencoder.all_ds))])
    X_encoded = autoencoder.encode_ds(autoencoder.all_ds)
    best_gmm_ll, best_gmm_kl = best_of_n_gmm_ray(X_encoded, n_clusters=10, covariance_type=cov_type, n_init=3, n=n_runs)
    y_pred_ll = best_gmm_ll.predict(X_encoded)
    acc_ll = cluster_acc(y_true, y_pred_ll)
    y_pred_kl = best_gmm_kl.predict(X_encoded)
    acc_kl = cluster_acc(y_true, y_pred_kl)
    if save:
        with open(f'saved_gmm_init/{run_id}/gmm-{cov_type}-acc={acc_kl:.2f}.pkl', 'wb') as file:
            pickle.dump(best_gmm_kl, file)
        with open(f'saved_gmm_init/{run_id}/gmm-{cov_type}-acc={acc_ll:.2f}.pkl', 'wb') as file:
            pickle.dump(best_gmm_ll, file)


parser = argparse.ArgumentParser()
parser.add_argument('run_id', type=str)
parser.add_argument('--num', type=int, default=5)
parser.add_argument('--checkpoint_file', type=str, required=False, nargs=1)
parser.add_argument('--ds_type', type=str, choices=['train', 'valid', 'all'], default='all')
parser.add_argument('--save', action='store_true')
parser.add_argument('--covariance_type', type=str, choices=['full', 'diag'], default='full')


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    print(args) 
    test_gmm(args.run_id, args.ds_type, args.num, args.save, args.covariance_type)

