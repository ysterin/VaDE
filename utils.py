import ray
from sklearn.mixture import GaussianMixture
import numpy as np


def kl_mixture_score(gmm: GaussianMixture, X_encoded: np.array, eps=1e-8):
    q_c_z = gmm.predict_proba(X_encoded)
    log_q_c_z = np.log(q_c_z + eps)
    mixture_logits = np.log(gmm.weights_)
    kl_div = np.einsum('ij,ij->i', q_c_z, log_q_c_z - mixture_logits)
    return kl_div.mean()

@ray.remote
def fit_gmm(x, n_clusters=10, covariance_type='full', n_init=3, random_state=None):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, n_init=n_init, random_state=random_state)
    gmm.fit(x)
    # score = gmm.score(x)
    score = kl_mixture_score(gmm, x)
    return gmm, score


# @ray.remote
# def fit_gmm(x, n_clusters=10, covariance_type='full', n_init=1, random_state=None):
#     gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, n_init=n_init, random_state=random_state)
#     gmm.fit(x)
#     log_likelihood = gmm.score(x)
#     return gmm, log_likelihood

def best_of_n_gmm_ray(x, n_clusters=10, n=10, covariance_type='full', n_init=1):
    ray.init(ignore_reinit_error=True)
    ss = np.random.SeedSequence(42)
    random_states = [np.random.RandomState(np.random.PCG64(c)) for c in ss.spawn(n)]
    x_id =  ray.put(x)
    scores = ray.get([fit_gmm.remote(x_id, n_clusters, covariance_type, n_init, st) for st in random_states])
    best_gmm, best_score = max(scores, key=lambda o: o[1])
#     ray.shutdown()
    return best_gmm

