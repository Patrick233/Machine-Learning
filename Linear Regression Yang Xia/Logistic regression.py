
# coding: utf-8

# In[184]:

import scipy.io as sio
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt


# In[203]:

data = sio.loadmat('dataset2.mat')
x_train = data['X_trn']
y_train = np.matrix(data['Y_trn'])
x_test = np.matrix(data['X_tst'])
y_test = np.matrix(data['Y_tst'])
X = np.matrix(x_train)
y = np.matrix(y_train)


# In[13]:

def get_fi(X,n):
    rows = X.shape[0]
    result = np.zeros((rows,n+1))
    for i in range(0,rows):
        result[i,0] = 1
        for j in range(1,n+1):
            result[i,j] = X[i] **(j)
    return np.matrix(result)


# In[39]:

def computeCost(X, y, theta,my_lambda):
    ridge = my_lambda*np.sum(np.power(theta,2))
    inner = np.power(((X * theta) - y), 2)
    return np.sum(inner) / len(X) + ridge


# In[49]:

def closeForm(X, y, my_lambda):

    a = X.T*X+my_lambda*np.matlib.identity(X.shape[1])
    
    return a.I*X.T*y


# In[193]:

def SGD(X,y,theta,alpha,iters,batch_size, my_lambda):
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
        
        error = 2*((x_batch * theta) - y_batch)

        #reg = 2*my_lambda*theta
        '''
        for j in range(parameters):
            term = np.multiply(error, x_batch[:,j])
            temp[j,0] = theta[j,0] - ((alpha / len(X)) * np.sum(term)+reg[j])
        '''
        
        
        theta = theta - alpha*(x_batch.T*(error)/(batch_size)+my_lambda*theta/(batch_size))
        cost[i] = computeCost(X, y, theta,my_lambda)
        
        #if i % 1000 == 0:
           # print("Loss iter",i,": ",cost[i])
        
    return theta, cost

#computeCost(X_train, y, theta,0.02)


# In[172]:

def get_cv(X, y, k, ith):
    folder_size = X.shape[0] // k
    trn_size = (k - 1) * folder_size
    X_train = np.matrix(np.zeros((trn_size, X.shape[1])))
    y_train = np.matrix(np.zeros((trn_size, y.shape[1])))
    test_start = ith * folder_size
    test_end = (ith + 1) * folder_size
    index = 0
    for j in range(X.shape[0]):
        if(j < test_start or j >= test_end):
            X_train[index] = X[j]
            y_train[index] = y[j]
            index += 1
    
    X_test = X[test_start: test_end]
    y_test = y[test_start: test_end]
    
    return X_train, y_train, X_test, y_test


# In[173]:

def find_lambda(X,y,K,alpha,iters,batch_size):
    my_set = [0.0001,0.001,0.01,0.1]
    min_error = 9999999;
    lambda_opt = 0
    for my_lambda in my_set:
        new_error = 0
        for i in range(0,K):
            
            X_train, y_train,X_test, y_test = get_cv(X, y, K, i)
            theta = np.matrix(np.zeros((X.shape[1],1)))
            theta, c = SGD(X_train, y_train, theta,
                           alpha, iters, batch_size, my_lambda)
            new_error += computeCost(X_test, y_test, theta, my_lambda)
            
        if(new_error<min_error):
            min_error = new_error
            lambda_opt = my_lambda
            
    return lambda_opt
        


# In[ ]:
def main():
    for n in [2,3,5]:
        for k in [2]:
            X_train =get_fi(X,n)
            theta = np.matrix(np.zeros((X_train.shape[1],1)))
            my_lambda = find_lambda(X_train,y_train,k,0.00001,10000,50)
            print(my_lambda)
            g,cost = SGD(X_train,y_train,theta,0.00001,5000,50, my_lambda)
            print(g.T)
            X_test = get_fi(x_test,n)
            c = computeCost(X_test,y_test,g,my_lambda)
            print("training error:", computeCost(X_train,y_train,g,my_lambda))
            print("test error: ",c)

# In[ ]:

if __name__ == "__main__":
        main()

