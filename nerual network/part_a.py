import scipy.io as sio
import numpy as np
import math as math
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

def initialize_weights_bias(no_input_units, no_hidden_units,no_output_units):
    mu = 0
    sigma = 0.01
    b_1 = np.random.normal(mu, sigma, no_hidden_units)
    w_1 = sigma * np.random.randn(no_input_units,no_hidden_units) + mu
    b_2 = np.random.normal(mu, sigma, no_output_units)
    w_2 = sigma * np.random.randn(no_hidden_units,no_output_units) + mu
    return np.matrix(w_1), np.matrix(b_1), np.matrix(w_2), np.matrix(b_2)


def feed_forward(input_x, w_1, b_1, w_2, b_2, act_fnct):
    a_1 = input_x
    z_2 = np.matrix(a_1) * np.matrix(w_1)  + b_1
    a_2 = act_fnct(z_2)
    
    z_3 = np.matrix(a_2) * np.matrix(w_2)  + b_2
    a_3 = soft_max(z_3)
    return z_2,a_2,z_3,a_3


def mis_error(xi,y,w1,b1,w2,b2,act_fn):
    z2,a2,z3,a3 = feed_forward(xi,w1,b1,w2,b2,act_fn)
    max_idx = 0
    max_val = a3[0,0]
    for i in range(10):
        if(a3[i,0]>max_val):
            max_idx = i
            max_val = a3[i,0]
    
    return abs(1-y[max_idx,0])


def back_propagation(a_1,y,z_2,a_2,a_3,w_2,act_diff):
    S_3 = a_3 - y
    delta_w_2 = a_2.T * S_3
    delta_b_2 = S_3
    S_2 = matr_product(S_3 * w_2.T,act_diff(z_2))
    delta_w_1 = np.matrix(a_1).T * S_2 
    delta_b_1 = S_2
    
    return delta_w_1, delta_b_1, delta_w_2, delta_b_2


def matr_product(x,y):
    size = x.shape[1]
    result = np.zeros((1,size))
    for i in range(size):
        result[0,i] = x[0,i]*y[0,i]
    return result


def soft_max(X):
    E = np.exp(X)
    sum = np.sum(E)
    return E / sum

def grad_actication(z,grad_act_fn):
    m = z.shape[0]
    a = np.zeros((m,1))
    for i in range(m):
        a[i,0] = grad_act_fn(z[i,0])
    return a

def sigmoid(X):
    return 1.0/(1+np.exp(-X))

def sigmoid_grad(X):
    return matr_product(sigmoid(X),(1-sigmoid(X)))


def hyperbolic(X):
    return np.tanh(X)

def hyperbolic_grad(X):
    return 1.0 / matr_product(np.tanh(X), np.tanh(X))


def rectifier(X):
    return np.maximum(X,0)


def relu_grad(X):
    result = X
    for i in range(len(X)):
        if (X[i,0]>0):
            result[i,0]=1
        else:
            result[i,0]=0
    return result


def mis_error(xi,y,w1,b1,w2,b2,act_fn):
    z2,a2,z3,a3 = feed_forward(xi,w1,b1,w2,b2,act_fn)
    max_idx = 0
    max_val = a3[0,0]
    for i in range(10):
        if(a3[0,i]>max_val):
            max_idx = i
            max_val = a3[0,i]
    
    return abs(1-y[max_idx]) 


def total_error(w1, w2, b1, b2, x, y, active):
    total_error = 0
    for i_re in range(140):
        xi = x[:, i_re]
        y = y[:, i_re]
        total_error = total_error + mis_error(xi, y, w1, b1, w2, b2, active)
    return total_error


def train_neural_network(x, act_fnctn, act_diff, activation_dif, no_hidden_units):
    batch_size = 200
    no_input_units = 32256  # number of input units of should be 192 * 168 = 32256(S_1 = 32256)
    no_output_units = 10  # number of output units should be 10 (S_3 = 10)
    training_size_each_class = 50  # number of training data in each classification
    N = no_output_units * training_size_each_class  # whole data size
    w_1, b_1, w_2, b_2 = initialize_weights_bias(no_input_units, no_hidden_units, no_output_units)

    # convert input matrix into shape(32256, 500) matrix, where 32256 is number of input unit, 500 is number of training datq
    x_train = x.T
    I = np.identity(10)
    y_train = np.column_stack(I[:, i] for i in range(10) for j in range(50))  # Y is classification label
    learning_rate = 0.05
    T = 1000  # using iteration instead of convergence
    for it in range(T):
        print(it)
        w_1_iter = 0
        b_1_iter = 0
        w_2_iter = 0
        b_2_iter = 0
        for k in range(batch_size):  # go through all the training set
            i = rd.randint(0, 499)
            z_2, a_2, z_3, a_3 = feed_forward(x_train[i, :], w_1, b_1, w_2, b_2, act_fnctn)
            delta_w_1, delta_b_1, delta_w_2, delta_b_2 = back_propagation(x_train[i, :], y_train[:, i], z_2, a_2, a_3,
                                                                          w_2, act_diff)
            w_1_iter = w_1_iter + delta_w_1
            b_1_iter = b_1_iter + delta_b_1
            w_2_iter = w_2_iter + delta_w_2
            b_2_iter = b_2_iter + delta_b_2
        w_1 = w_1 - learning_rate * w_1_iter / batch_size
        b_1 = b_1 - learning_rate * b_1_iter / batch_size
        w_2 = w_2 - learning_rate * w_2_iter / batch_size
        b_2 = b_2 - learning_rate * b_2_iter / batch_size

        if it % 10 == 0:
            total_error = 0
            for i_re in range(500):
                xi = x_train[i_re, :]
                y = y_train[:, i_re]
                total_error = total_error + mis_error(xi, y, w_1, b_1, w_2, b_2, act_fnctn)
            print("total_error", total_error)
            if total_error < 20:
                break
    return w_1, b_1, w_2, b_2

if __name__ == "__main__":
    data = sio.loadmat('ExtYaleB10.mat')
    train_sample = data['train']
    test_sample = data['test']
    x_train = np.column_stack(train_sample[0, i][:, :, j].reshape(192 * 168, 1) for i in range(10) for j in range(50))
    x_test = np.column_stack(test_sample[0, i][:, :, j].reshape(192 * 168, 1) for i in range(10) for j in range(14))
    I = np.identity(10)
    y_train = np.column_stack(I[:, i] for i in range(10) for j in range(50))
    y_test = np.column_stack(I[:, i] for i in range(10) for j in range(14))

    #Train the nerual_net with sigmoid as activation function, hidden layer:32,64,128
    w1, b1, w2, b2 = train_neural_network(x_train, sigmoid, sigmoid_grad, sigmoid_grad, 32)
    print("Test error in for sigmoid,32 hidden neuron", total_error(w1, b1, w2, b2,x_test,y_test,sigmoid))

    w1, b1, w2, b2 = train_neural_network(x_train, sigmoid, sigmoid_grad, sigmoid_grad, 64)
    print("Test error in for sigmoid,64 hidden neuron", total_error(w1, b1, w2, b2, x_test, y_test, sigmoid))

    w1, b1, w2, b2 = train_neural_network(x_train, sigmoid, sigmoid_grad, sigmoid_grad, 128)
    print("Test error in for sigmoid,128 hidden neuron", total_error(w1, b1, w2, b2, x_test, y_test, sigmoid))

    # Train the nerual_net with hyperbolic as activation function, hidden layer:32,64,128
    w1, b1, w2, b2 = train_neural_network(x_train, hyperbolic, hyperbolic_grad, hyperbolic_grad, 32)
    print("Test error in for hyperbolic,32 hidden neuron", total_error(w1, b1, w2, b2, x_test, y_test, hyperbolic))

    w1, b1, w2, b2 = train_neural_network(x_train, hyperbolic, hyperbolic_grad, hyperbolic_grad, 64)
    print("Test error in for hyperbolic,64 hidden neuron", total_error(w1, b1, w2, b2, x_test, y_test, hyperbolic))

    w1, b1, w2, b2 = train_neural_network(x_train, hyperbolic, hyperbolic_grad, hyperbolic_grad, 128)
    print("Test error in for hyperbolic,128 hidden neuron", total_error(w1, b1, w2, b2, x_test, y_test, hyperbolic))

    # Train the nerual_net with rectify as activation function, hidden layer:32,64,128
    w1, b1, w2, b2 = train_neural_network(x_train, rectifier, relu_grad, relu_grad, 32)
    print("Test error in for recitify,32 hidden neuron", total_error(w1, b1, w2, b2, x_test, y_test, rectifier))

    w1, b1, w2, b2 = train_neural_network(x_train, rectifier, relu_grad, relu_grad, 64)
    print("Test error in for recitify,64 hidden neuron", total_error(w1, b1, w2, b2, x_test, y_test, rectifier))

    w1, b1, w2, b2 = train_neural_network(x_train, rectifier, relu_grad, relu_grad, 128)
    print("Test error in for recitify,128 hidden neuron", total_error(w1, b1, w2, b2, x_test, y_test, rectifier))
