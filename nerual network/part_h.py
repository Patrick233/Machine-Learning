
import scipy.io as sio
import numpy as np
import math as math
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import kmeans as km
import spectral_clustering as spectral
from numpy import linalg as la

def class_error(result):
    sum_error = 0
    for j in range(500):
        for k in range(500):
            class_j = j/50
            class_k = k/50
            if(class_j == class_k & result[j]!=result[k]):
                sum_error = sum_error+1
            if(class_j != class_k & result[j]==result[k]):
                sum_error = sum_error+1
    return sum_error


if __name__ == "__main__":
    data = sio.loadmat('ExtYaleB10.mat')
    train_sample = data['train']
    test_sample = data['test']
    x_train_full = np.column_stack(train_sample[0, i][:, :, j].reshape(192 * 168, 1) for i in range(10) for j in range(50))
    x_test_full = np.column_stack(test_sample[0, i][:, :, j].reshape(192 * 168, 1) for i in range(10) for j in range(14))
    I = np.identity(10)
    y_train = np.column_stack(I[:, i] for i in range(10) for j in range(50))
    y_test = np.column_stack(I[:, i] for i in range(10) for j in range(14))

    print("spectral_clustering computing")
    v = spectral.spectral_clustering(x_train_full, 10, 1, 10)
    for i in range(v.shape[1]):
        v[:, i] = v[:, i] / (la.norm(v[:, i]))
    final_c, final_z = km.k_means(np.asarray(v.T), 10, 10)
    y_train = np.zeros(500, )
    cost = 0
    for i in range(10):
        for j in range(50):
            y_train[i * 50 + j] = i
    for i in range(500):
        for j in range(10):
            if final_z[i, j] == 1:
                if y_train[i] != j:
                    cost += 1
    print("Errors of spectral clustering", cost)

