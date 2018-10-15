
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import kmeans as kmeans
import pca as pca

def get_dist(y,i,j):
    return np.sum((y[:,i]-y[:,j])**2)

def get_weight(y,i,j,sigma):
    d = get_dist(y,i,j)
    return np.exp(- d / (2 * (sigma ** 2)))

def get_k_neighbour(y,k,index):
    sample_size = y.shape[1]
    dist = [0]*sample_size
    for i in range(sample_size):
        dist[i] = get_dist(y,i,index)
    result = np.argsort(dist)
    return result[1:k+1]


def get_w_matrix(y,k,sigma):
    sample_size = y.shape[1]
    w = np.zeros((sample_size,sample_size))
    for i in range(sample_size):
        indices = get_k_neighbour(y, k, i)
        for idx in indices:
            w[i, idx] = get_weight(y, i, idx, sigma)
        w[i, i] = 0
    return w

def get_laplacian(W):
    N = W.shape[1]
    L = np.zeros((N,N))
    for i in range(N):
        L[i,i] = np.sum(W[i,:])
    L = L - W
    return L

def spectral_cluster(W,k):
    L = get_laplacian(W)
    Lambda, v = np.linalg.eig(L)
    length = len(Lambda)
    eigen_pairs = [(Lambda[i], v[:, i]) for i in range(length)]
    eigen_pairs = sorted(eigen_pairs, reverse=False, key=lambda k: k[0])
    temp = np.column_stack((eigen_pairs[i][1] for i in range(k)))
    result= kmeans.k_means(temp.T,2,5)

    return result



