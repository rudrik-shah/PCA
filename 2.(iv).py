#!/usr/bin/env python
# coding: utf-8

# # 2.(iv)

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import random
import math


# ##### Reading Data from the csv File

# In[2]:


df = pd.read_csv("C:/Users/rudri/OneDrive/Desktop/PRML/Assignments/PRML Assignment 1/Dataset.csv",header = None, names = ['x','y'])
df


# #### Original Dataset

# In[3]:


x_coordinates = np.array(df.x)
y_coordinates = np.array(df.y)
plt.scatter(x_coordinates,y_coordinates)
plt.title("Original Dataset")
plt.show()


# #### Function to find Euclidean Distance

# In[4]:


def distance(x1,x2,x3,x4,m1,m2,m3,m4):
    d1 = (x1 - m1) * (x1 - m1)
    d2 = (x2 - m2) * (x2 - m2)
    d3 = (x3 - m3) * (x3 - m3)
    d4 = (x4 - m4) * (x4 - m4)
    d = d1 + d2 + d3 +d4  
    return d


# #### Function to find Maximum among given Four Numbers

# In[5]:


def max_(a,b,c,d):
    if(a > b):
        if(a > c):
            if(a > d):
                return 1
            else:
                return 4
        else:
            if(c > d):
                return 3
            else:
                return 4
    else:
        if(b > c):
            if(b > d):
                return 2
            else:
                return 4
        else:
            if(c > d):
                return 3
            else:
                return 4 


# #### Function to run New Clustering Definition

# In[6]:


def new_clustering(h):
    z = np.zeros(1000)
    for i in range(0,1000):
        z[i] = max_(h[i][0],h[i][1],h[i][2],h[i][3])        # Assigning datapoint to cluster

    color = ["red", "indigo", "green", "blue"]
    plt.xlabel("X-Coordinates")
    plt.ylabel("Y-Coordinates")
    plt.title("Clustered Dataset")
    for i in range(len(z)):
        plt.scatter(x_coordinates[i], y_coordinates[i], c = color[(int)(z[i])-1])        # Plotting Datapoint according to their Clusters
    plt.show()


# #### Function to Computing H Matrix

# In[7]:


def h_matrix(kc):
    eigen_value,eigen_vector = np.linalg.eigh(kc)          # Finding Eigen Values & Eigen Vectors of Kernel Matrix

    # Sorting Eigen Values & Eigen Vectors
    top_eigen_value = eigen_value.argsort()[::-1]
    sorted_eigen_value = eigen_value[top_eigen_value]
    sorted_eigen_vector = eigen_vector[:,top_eigen_value]

    # Taking Top K Eigen Vectors
    x1 = sorted_eigen_vector[:,0]
    x2 = sorted_eigen_vector[:,1]
    x3 = sorted_eigen_vector[:,2]
    x4 = sorted_eigen_vector[:,3]

    # Computing Matrix H by top K Eigen Vectors
    h = np.column_stack((x1,x2,x3,x4))

    # Normalizing rows of H Matrix
    for i in range(0,1000):
        sum_ = 0.0
        for j in range(0,4):
            sum_ = sum_ + (h[i][j])**2
        h[i] = h[i] / sum_**(1/2)
    
    return h


# #### Quadratic Kernel

# In[8]:


k_temp = df.dot(df.T)
k_temp = k_temp + 1
k = k_temp**2
kc = np.identity(1000)-(1/1000)
kc = kc.dot(k).dot(kc)

h = h_matrix(kc)

new_clustering(h)


# #### Cubic Kernel

# In[9]:


k_temp = df.dot(df.T)
k_temp = k_temp + 1
k = k_temp**3
kc = np.identity(1000)-(1/1000)
kc = kc.dot(k).dot(kc)

h = h_matrix(kc)

new_clustering(h)


# #### Radial Basis Kernel

# In[10]:


data = np.array(df)
d = np.ndarray((1000,1000))
for i in range(0,1000):
    for j in range(0,1000):
        temp = data[i] - data[j]        # Computing (x - yT)(x - yT)
        d[i][j] = np.dot(temp.T, temp)
        
d = -d        # Taking Negation of the matrix

for sigma in range(1,11):
    t = d
    sigma = sigma / 10
    print("Plot for Sigma = ",sigma)
    temp = sigma**2                  # Computing sigma^2
    temp = 2 * temp                 # Multiplying result with 2
    t = t / temp                    # Dividing each element of matix by the result 
    
    t = np.exp(t)                 # Taking exponent, hence computing the Kernel Matrix K
    
    h = h_matrix(t)
    
    new_clustering(h)

