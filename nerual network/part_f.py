
import scipy.io as sio
import numpy as np
import math as math
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import kmeans as kmeans
import pca as pca


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

    #apply pca to reduce d=2
    print ("Dimension reduction calculating")
    u, x_reduced_2 = pca.pca(x_train_full, 2)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']
    markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')
    for i in range(10):
        start = i * 50
        end = (i + 1) * 50
        x = x_reduced_2[0, start:end]
        y = x_reduced_2[1, start:end]
        plt.scatter(np.asarray(x), np.asarray(y), marker=markers[i], color=colors[i])
    plt.show()



