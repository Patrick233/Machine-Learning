import scipy.io as sio
import numpy as np
import pandas as pd
import random as rd
import math
import matplotlib.pyplot as plt
from scipy.special import expit

def k_means(Y,k,r):
    N = Y.shape[1]
    min_cost = float('inf')
    min_centroid = np.zeros((k,Y.shape[0]))
    for i in range(r):
        centroids = random_initial(k,Y)
        changes = N
        result = [0]*N
        iters = 0
        while(iters<100):
            for j in range(N):
                result[j] = assign_cluster(centroids,np.matrix(Y[:,j]))
            centroid = new_centroid(Y,result,k)
            iters = iters+1
        cost = kmeans_cost(Y,centroids,result)
        if(cost<min_cost):
            min_cost = cost
            min_centroid = centroid
    result = [0] * N

    for j in range(N):
        result[j] = assign_cluster(min_centroid, Y[:, j])
    return result

def random_initial(k,Y):
    sample_size = Y.shape[1]
    feature_size = Y.shape[0]
    centroids = np.matrix(np.zeros((feature_size,k)))
    count = 0
    while(count<k):
        index = rd.randint(0,sample_size-1)
        centroids[:,count] = Y[:,index]
        count = count + 1
    return centroids

def assign_cluster(centroid,point):
    label = centroid.shape[1]
    min_dist  = float('inf')
    min_index = -1
    for i in range(label):
        dist = np.linalg.norm(np.matrix(point) - centroid[:,i])
        if(dist<min_dist):
            min_dist = dist
            min_index = i
    return min_index

def new_centroid(Y,result,k):
    sample_size = Y.shape[1]
    feature_size = Y.shape[0]
    centroid = np.matrix(np.zeros((feature_size,k)))
    count = [0]*k
    for i in range(sample_size):
        index = result[i]
        count[index] = count[index]+1
        centroid[:,index] = centroid[:,index]+Y[:,i]
    for i in range(k):
        centroid[:,i] = centroid[:,i]/count[i]
    return centroid

def kmeans_cost(Y,centroids,result):
    total_cost = 0
    sample_size = Y.shape[1]
    for i in range(sample_size):
        total_cost = total_cost + np.linalg.norm(np.matrix(Y[:,i]-centroids[:,result[i]]))
    return total_cost