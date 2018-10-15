import scipy.io as sio
import numpy as np
import pandas as pd
import random as rd
import math
import matplotlib.pyplot as plt
from scipy.special import expit


def compute_cost(X, y, w, my_lambda):
    m = len(X)
    return -1.0 / m * (y.T * np.log(expit(X * w)) + (1 - y.T) * np.log(1 - expit(X * w))) + my_lambda * w.T * w

def gradient_descent(X, y, alpha, my_lambda, iters):
    w = np.matrix(np.zeros((X.shape[1], 1)))
    m = len(X)
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X.T * (expit(X * w) - y) + my_lambda * w) / m
        w = w - alpha * error
        cost[i] = compute_cost(X, y, w, my_lambda)

    #         if i%50==0:
    #             print("Loss iter",i,": ",cost[i])
    return w, cost

def get_cv(X, y, k, ith):
    folder_size = X.shape[0] // k
    trn_size = X.shape[0] - folder_size
    X_train = np.matrix(np.zeros((trn_size, X.shape[1])))
    y_train = np.matrix(np.zeros((trn_size, y.shape[1])))
    test_start = ith * folder_size
    test_end = (ith + 1) * folder_size
    index = 0
    for j in range(X.shape[0]):
        if (j < test_start or j > test_end):
            X_train[index] = X[j]
            y_train[index] = y[j]
            index += 1

    X_test = X[test_start: test_end]
    y_test = y[test_start: test_end]

    return X_train, y_train, X_test, y_test

def find_lambda(X, y, alpha, K, iters):
    my_set = [0.00001, 0.0001, 0.0002, 0.001, 0.01, 0.1]
    min_error = 9999999;
    lambda_opt = 0
    for my_lambda in my_set:
        new_error = 0
        for i in range(0, K):
            X_train, y_train, X_test, y_test = get_cv(X, y, K, i)
            theta = np.matrix(np.zeros((X.shape[1], 1)))
            theta, c = gradient_descent(X_train, y_train,
                                        alpha, my_lambda, iters)
            new_error += compute_cost(X_test, y_test, theta, my_lambda)

        if (new_error < min_error):
            min_error = new_error
            lambda_opt = my_lambda

    return lambda_opt

def plot_result(X, y, w):
    xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]

    probs = expit(grid * w).reshape(xx.shape)
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                          vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(X[:, 0], X[:, 1], c=y[:], s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(-5, 5), ylim=(-5, 5),
           xlabel="$X_1$", ylabel="$X_2$")

    plt.show()

def mis_classify(x,y,w):
    count = 0
    for i in range(x.shape[0]):
        result = expit(x[i,:]*w)
        if (y[i, 0] == 1) and (result < 0.5):
            count = count+ 1
        if (y[i, 0] == 0) and (result > 0.5):
            count = count + 1
    return count


def main():
    # Set the parameters
    iters = 1000
    alpha = 0.01
    # Loading the data
    data = sio.loadmat('HW2_Data/data1.mat')
    x_train = data['X_trn']
    # Adding the intersection
    train_ones = np.full((len(x_train), 1), 1)
    x_train = np.concatenate((x_train, train_ones), 1)
    y_train = data['Y_trn']
    x_test = data['X_tst']
    test_ones = np.full((len(x_test), 1), 1)
    x_test = np.concatenate((x_test, test_ones), 1)
    y_test = data['Y_tst']

    # Cross-validation to get lambda
    my_lambda = find_lambda(x_train, y_train, alpha, 10, iters)

    # train the model using data from training set
    w, cost = gradient_descent(x_train, y_train, alpha, my_lambda, iters)
    print "w is ", w
    print "Cost on the traing set is: ", mis_classify(x_train, y_train,w)
    print "Cost on the test set is: ", mis_classify(x_test, y_test,w)

    # Set the parameters
    iters = 1000
    alpha = 0.01
    # Loading the data
    data = sio.loadmat('HW2_Data/data2.mat')
    x_train = data['X_trn']
    # Adding the intersection
    train_ones = np.full((len(x_train), 1), 1)
    x_train = np.concatenate((x_train, train_ones), 1)
    y_train = data['Y_trn']
    x_test = data['X_tst']
    test_ones = np.full((len(x_test), 1), 1)
    x_test = np.concatenate((x_test, test_ones), 1)
    y_test = data['Y_tst']

    # Cross-validation to get lambda
    my_lambda = find_lambda(x_train, y_train, alpha, 10, iters)

    # train the model using data from training set
    w, cost = gradient_descent(x_train, y_train, alpha, my_lambda, iters)
    print "w is ", w
    print "Cost on the traing set is: ", mis_classify(x_train, y_train,w)
    print "Cost on the test set is: ", mis_classify(x_test, y_test,w)


if __name__ == "__main__":
    main()
