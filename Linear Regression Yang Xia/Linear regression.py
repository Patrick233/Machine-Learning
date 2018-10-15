
# coding: utf-8

# In[1]:

import scipy.io as sio
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt


# In[2]:

data = sio.loadmat('dataset1.mat')
x_train = data['X_trn']
y_train = data['Y_trn']
x_test = data['X_tst']
y_test = data['Y_tst']
X = np.matrix(x_train)
y_tr = np.matrix(y_train)
y_ts = np.matrix(y_test)


def get_fi(X,n):
    rows = X.shape[0]
    result = np.zeros((rows,n+1))
    for i in range(0,rows):
        result[i,0] = 1
        for j in range(1,n+1):
            result[i,j] = X[i] **(j)
    return np.matrix(result)


# In[3]:

def computeCost(X, y, theta):
    inner = np.power(((X * theta) - y), 2)
    return np.sum(inner) / len(X)


# In[4]:

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = theta.ravel().shape[1]
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[j,0] = theta[j,0] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
        if i % 1000 == 0:
            print("Loss iter",i,": ",cost[i])
        
    return theta, cost


# In[5]:

def closedForm(X,y):
    
    return (X.T*X).I*X.T*y


# In[6]:

def SGD(X,y,theta,alpha,iters,batch_size):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = theta.ravel().shape[1]
    cost = np.zeros(iters)
    
    for i in range(iters):
        rand = np.arange(X.shape[0])
        rd.shuffle(rand)
        batch = rand[:batch_size]
        
        x_batch = np.matrix(np.zeros((batch_size,X.shape[1])))
        y_batch = np.matrix(np.zeros((batch_size,1)))
        
        count = 0
        for k in batch:
            x_batch[count] = X[k,:]
            y_batch[count] = y[k]
            count = count+1
        
        error = (x_batch * theta) - y_batch
        
        theta = theta - alpha*(x_batch.T*(error)/(batch_size))
        
        '''
        for j in range(parameters):
            term = np.multiply(error, x_batch[:,j])
            temp[j,0] = theta[j,0] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp'''
        cost[i] = computeCost(X, y, theta)
        
        if i % 1000 == 0:
            print("Loss iter",i,": ",cost[i])
        
    return theta, cost


# In[7]:
def main():
    for i in (10,20,30,40,50,60,70,80,90,100):
        X = get_fi(x_train,3)
        theta = np.matrix(np.zeros((X.shape[1],1)))
        g,cost = SGD(X,y_tr,theta,0.001,5000,i)
        print(g.T)
        X_test = get_fi(x_test,3)
        c = computeCost(X_test,y_ts,g)
        print("For size:",i, "training error:", computeCost(X,y_tr,g))
        print("test error: ",c)


    # In[10]:

    alpha = 0.0001
    iters = 100000
    batch_size = 50
    for i in [3,5]:
        X = get_fi(x_train,i)
        theta = np.matrix(np.zeros((X.shape[1],1)))
        if(i==2):
            alpha = 0.001
        if(i==3):
            alpha = 0.001
        if(i==5):
            alpha = 0.00001
            iters = iters*10
        g,cost = SGD(X,y_tr,theta,alpha,iters,50)
        print(g)
        X_test = get_fi(x_test,i)
        c = computeCost(X_test,y_ts,g)
        print("test error: ",c)

if __name__ == "__main__":
        main()
# In[ ]:



