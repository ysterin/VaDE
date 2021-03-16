import ray
from sklearn.mixture import GaussianMixture
import numpy as np

@ray.remote
def fit_gmm(x, n_clusters=10, covariance_type='full', n_init=1, random_state=None):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, n_init=n_init, random_state=random_state)
    gmm.fit(x)
    log_likelihood = gmm.score(x)
    return gmm, log_likelihood

def best_of_n_gmm_ray(x, n_clusters=10, n=10, covariance_type='full', n_init=1):
    ray.init(ignore_reinit_error=True)
    ss = np.random.SeedSequence(42)
    random_states = [np.random.RandomState(np.random.PCG64(c)) for c in ss.spawn(n)]
    x_id =  ray.put(x)
    scores = ray.get([fit_gmm.remote(x_id, n_clusters, covariance_type, n_init, st) for st in random_states])
    best_gmm, best_score = max(scores, key=lambda o: o[1])
#     ray.shutdown()
    return best_gmm

