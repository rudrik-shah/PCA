#!/usr/bin/env python
# coding: utf-8

# # 2.(iii)

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


# #### Function of K-Means Algorithm

# In[5]:


def k_means(h):
    z = np.random.randint(1,5,1000)        # Assigning Clusters to each data point randomly
    c = 0

    while(True):
        c += 1        # Computing number of rounds the loop will run
        mean_ = np.zeros([4,4])
        sum_ = np.zeros([4,4])
        count_ = np.zeros(4)

        z_previous = np.copy(z)

        for i in range(0,len(z)):
            sum_[z[i]-1][0] += h[i][0]        # Computing sum of x1 of same cluster data points
            sum_[z[i]-1][1] += h[i][1]        # Computing sum of x2 of same cluster data points
            sum_[z[i]-1][0] += h[i][2]        # Computing sum of x3 of same cluster data points
            sum_[z[i]-1][1] += h[i][3]        # Computing sum of x4 of same cluster data points
            count_[z[i]-1] += 1             # Computing number of data points present in a cluster

        for i in range(0,len(count_)):
            mean_[i][0] = sum_[i][0] / count_[i]        # Computing mean of x1 of a cluster
            mean_[i][1] = sum_[i][1] / count_[i]        # Computing mean of x2 of a cluster
            mean_[i][2] = sum_[i][2] / count_[i]        # Computing mean of x3 of a cluster
            mean_[i][3] = sum_[i][3] / count_[i]        # Computing mean of x4 of a cluster

        diff = 0
        temp = np.copy(z)
        for i in range(0,len(z)):
                 # Finding distance of data point to mean of the cluster it is assigned 
            d = distance(h[i][0],h[i][1],h[i][2],h[i][3],mean_[z[i]-1][0],mean_[z[i]-1][1],mean_[z[i]-1][2],mean_[z[i]-1][3])
            diff += d
            for j in range(0,len(count_)):
                    # Finding distance of data point to mean of all the clusters
                m = distance(h[i][0],h[i][1],h[i][2],h[i][3],mean_[j][0],mean_[j][1],mean_[j][2],mean_[j][3])
                if(m < d):        # Checking if distance to any other mean is less than its owm mean
                    d = m
                    temp[i] = j + 1        # If yes, then making a jump to other cluster
        z = np.copy(temp)

        if(np.array_equal(z_previous, z)):        # If two simlutaneous indicator functions are same => No jumps are made => Break the loop
            break

    color = ["red", "indigo", "green", "blue"]
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Spectral Clustered Dataset")
    for i in range(len(z)):
        plt.scatter(x_coordinates[i], y_coordinates[i], c = color[z[i]-1])        # Plotting the data points with different colours as their clusters assigned
    plt.show()


# #### Function to Compute H Matrix

# In[6]:


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


# #### Quadratic Polynomail Kernel

# In[7]:


k_temp = df.dot(df.T)
k_temp = k_temp + 1
k = k_temp**2
kc = np.identity(1000)-(1/1000)
kc = kc.dot(k).dot(kc)

h = h_matrix(kc)

k_means(h)


# #### Cubic Polynomial Kernel

# In[10]:


k_temp = df.dot(df.T)
k_temp = k_temp + 1
k = k_temp**3
kc = np.identity(1000)-(1/1000)
kc = kc.dot(k).dot(kc)

h = h_matrix(kc)

k_means(h)


# #### Radial Basis Kernel 

# In[9]:


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
    
    k_means(h)

