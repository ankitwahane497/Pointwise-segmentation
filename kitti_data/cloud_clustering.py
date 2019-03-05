import numpy as np
from sklearn.mixture import *
from utils import *

def get_Gaussian_labels(pcl, n_components_in, n_points_in):
    #BayesianGaussianMixture
    BGM = BayesianGaussianMixture(
        n_components= n_components_in, covariance_type='full', weight_concentration_prior=1e-2,
        weight_concentration_prior_type='dirichlet_process',
        mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(4),
        init_params="random", max_iter=1000, random_state=2).fit(pcl[:,:4])
    bgm_labels = BGM.predict(pcl[:,:4])
    pcl_out = np.zeros((n_points_in,6))
    pcl_out[:,:5] = pcl[:,:5]
    pcl_out[:,-1] = bgm_labels
    return pcl_out


def get_cluster(pcl,n_cluster, num_cluster_samples):
    #given last is the instance id
    uniq = np.unique(pcl[:,-1])
    frames = np.zeros((n_cluster,num_cluster_samples,5))
    for i in range(len(uniq)):
        label_idx = int(uniq[i])
        pcl_label = pcl[pcl[:,-1] == label_idx][:,:-1]
        frames[i] = fix_samples(pcl_label, num_cluster_samples)
    return frames
