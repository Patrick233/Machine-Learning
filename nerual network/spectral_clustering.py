import scipy.io as sio
import numpy as np
import math as math
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import kmeans as kmeans
from numpy import linalg as la

def compute_distance(x, y):
    return np.sum((x - y) * (x - y))


def cal_weight(y, i, j, sigma):
    d = compute_distance(y[:, i], y[:, j])
    d = 0 - d
    return np.exp(d / (2 * (sigma ** 2)))


def get_neighbors(y, k, idx):
    n = y.shape[1]
    d = np.zeros(n)
    for i in range(n):
        d[i] = compute_distance(y[:, i], y[:, idx])
    indices = np.argsort(d)
    indices = indices[1: k + 1]
    return indices


def cal_weight_matrix(y, k, sigma):
    n = y.shape[1]
    w = np.zeros((n, n))
    for i in range(n):
        indices = get_neighbors(y, k, i)
        for idx in indices:
            w[i, idx] = cal_weight(y, i, idx, sigma)
            w[idx, i] = w[i, idx]
        w[i, i] = 0
    return w


def cal_laplacian_matrix(w):
    n = w.shape[0]
    laplacian = np.zeros((n, n))
    for i in range(n):
        laplacian[i, i] = np.sum(w[:, i])
    laplacian -= w
    return laplacian


def spectral_clustering(y, k, sigma, d):
    w = cal_weight_matrix(y, k, sigma)
    print("w")
    laplacian = cal_laplacian_matrix(w)
    print("l")
    values, vectors = la.eig(laplacian)
    # print((laplacian * np.matrix(vectors[0]).T) - values[0] * np.matrix(vectors[0]).T)
    length = len(values)
    pairs = [(values[i], vectors[:, i]) for i in range(length)]
    pairs = sorted(pairs, reverse=False, key=lambda k: k[0])
    vectors = np.column_stack((pairs[i][1] for i in range(d)))
    return vectors