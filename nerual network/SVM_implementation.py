
# coding: utf-8

# In[343]:

import scipy.io as sio
import numpy as np
import pandas as pd
import random as rd
import math
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn import svm


def get_kernel(x):
    return np.dot(x,x.T)

def pre_processing(y):
    m = len(y)
    for i in range(m):
        if y[i,0]==0:
            y[i,0]=-1
    return y

def f(alpha,y,i,kernel,b):
    temp = alpha*y
    return np.sum(temp*kernel[i])+b


def get_w(alpha,x,y):
    w = alpha * y
    return np.sum(w*x,axis=0)


def predict(X,y,alpha,b, x_k):
    m = len(X)
    sum = 0
    for i in range(m):
        sum += alpha[i]*y[i][0]*np.mat(X[i,:])*x_k.T
    return sum + b


def compute_L_H(alpha_i,alpha_j,y_i,y_j,C):
    if(y_i!=y_j):
        L = max(0,alpha_j-alpha_i)
        H = min(C,C+alpha_j-alpha_i)
    else:
        L = max(0,alpha_i+alpha_j-C)
        H = min(C,alpha_i+alpha_j)
    return L,H


def compute_yita(kernel,i,j):
    return 2*kernel[i,j]-kernel[i,i]-kernel[j,j]


def SMO(X,y,C,tol,max_passes):
    m = len(X)
    alpha = np.ones((X.shape[0],1))
    b =0
    passes = 0
    kernel = get_kernel(X)
    while(passes<max_passes):
        num_changed_alphas = 0
        for i in range(m):
            Ei = f(alpha,y,i,kernel,b)-y[i]
            if((y[i][0]*Ei< -tol and alpha[i]<C) or (y[i][0]*Ei> tol and alpha[i]>0)):
                j = rd.randint(0,m-1)
                while(j==i):
                    j = rd.randint(0,m-1)
                Ej = f(alpha,y,j,kernel,b)-y[j]
                old_alpha_i = alpha[i]
                old_alpha_j = alpha[j]
                L,H = compute_L_H(alpha[i],alpha[j],y[i,:],y[j,:],C)
                if(L==H):
                    continue
                yita = compute_yita(kernel,i,j)
                if(yita>=0):
                    continue
                alpha[j] = alpha[j] - y[j]*(Ei-Ej)/yita
                if(alpha[j]>H):
                    alpha[j] = H
                if(alpha[j]<L):
                    alpha[j] = L
                if(abs(alpha[j]-old_alpha_j)<1e-5):
                    continue
                alpha[i] = alpha[i]+y[i]*y[j]*(old_alpha_j-alpha[j])
                b1 = b - Ei - y[i]*(alpha[i]-old_alpha_i)*kernel[i,i]-y[j][0]*(alpha[j]-old_alpha_j)*kernel[i,j]
                b2 = b - Ej - y[i]*(alpha[i]-old_alpha_i)*kernel[i,j]-y[j][0]*(alpha[j]-old_alpha_j)*kernel[j,j]
                if(alpha[i]>0 and alpha[j]<C):
                    b = b1
                elif(alpha[j]>0 and alpha[j]<C):
                    b = b2
                else:
                    b = (b1+b2)/2
                num_changed_alphas = num_changed_alphas +1
        if(num_changed_alphas==0):
            passes = passes+1
        else:
            passes = 0
    return alpha,b


def compute_cost(w, b, x, y):
    count = 0
    for i in range(x.shape[0]):
        if (y[i, 0] == 1) and (np.dot(w, x[i]) + b < 0):
            count = count+ 1
        if (y[i, 0] == -1) and (np.dot(w, x[i]) + b >= 0):
            count = count + 1
    return count


def main():
    #Set the parameters
    C = 10
    tol = 0.01
    max_passes = 500
    #for dataset1
    data = sio.loadmat('HW2_Data/data1.mat')
    x_train = data['X_trn']
    y_train = pre_processing(np.array(data['Y_trn'], dtype=int))
    x_test = data['X_tst']
    y_test = pre_processing(np.array(data['Y_tst'], dtype=int))

    #train the model using data from training set
    alpha, b = SMO(x_train, y_train, C, tol, max_passes)
    w = get_w(alpha, x_train, y_train)

    #compute the error on training set and test set
    print "Error on the training set is ",compute_cost(w, b, x_train, y_train)
    print "Error on the testing set is ", compute_cost(w, b, x_test, y_test)

    clf = svm.SVC()
    clf.fit(x_train, y_train.ravel())
    print(np.asarray(clf.predict(x_test)))
    #
    # for dataset2
    data = sio.loadmat('HW2_Data/data2.mat')
    x_train = data['X_trn']
    y_train = pre_processing(np.array(data['Y_trn'], dtype=int))
    x_test = data['X_tst']
    y_test = pre_processing(np.array(data['Y_tst'], dtype=int))

    # Change parameter here to so that
    C=60

    # train the model using data from training set
    alpha, b = SMO(x_train, y_train, C, tol, max_passes)
    w = get_w(alpha, x_train, y_train)

    # compute the error on training set and test set
    print "Error on the training set is ", compute_cost(w, b, x_train, y_train)
    print "Error on the testing set is ", compute_cost(w, b, x_test, y_test)

    clf = svm.SVC()
    clf.fit(x_train, y_train.ravel())
    print(np.asarray(clf.predict(x_test)))



if __name__ =="__main__":
    main()



