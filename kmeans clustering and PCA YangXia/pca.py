
import numpy as np


def get_kernel(x):
    length = x.shape[1]
    kernel = np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            kernel[i,j] = np.sum((x[:,i]-x[:,j])**2)/-20
    return np.matrix(np.exp(kernel))

def pca(y,d):
    y_mean = y.mean(1)
    y_bar = (y.T-y_mean.T).T
    u,s,v = np.linalg.svd(y_bar)
    out_u = np.matrix(u[:,0:d])
    x = np.matrix(out_u.T * y)
    return out_u,x

def kernel_pca(K,d):
    N = K.shape[0]
    one_N = np.ones((N, N))/N
    K = K - one_N.dot(K) - K.dot(one_N) + one_N.dot(K).dot(one_N)

    Lambda, w = np.linalg.eig(K)
    length = len(Lambda)
    eigen_pairs = [(Lambda[i], w[:, i]) for i in range(length)]
    eigen_pairs = sorted(eigen_pairs, reverse=True, key=lambda k: k[0])

    return np.column_stack((eigen_pairs[i][1] for i in range(d)))