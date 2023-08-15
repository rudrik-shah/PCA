#!/usr/bin/env python
# coding: utf-8

# # 2.(ii)

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import random
import math
import sys


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


# #### Function to find Euclidean Distance between two points

# In[4]:


def distance(x1,y1,x2,y2):       # Function to find Euclidean distance between two points
    d1 = (x2 - x1) * (x2 - x1)
    d2 = (y2 - y1) * (y2 - y1)
    d = d1 + d2
    return d


# ##### K-Means Algorithm for Different K

# In[5]:


for k in range(2,6):
    z = np.random.randint(1,k+1,len(x_coordinates))        # Assigning Clusters to each data point randomly
    c = 0
    x = []
    e = []

    while(True):

        mean_ = np.zeros([k,2])
        sum_ = np.zeros([k,2])
        count_ = np.zeros(k)
        c += 1        # Computing number of rounds the loop will run
        
        z_previous = np.copy(z)

        for i in range(0,len(z)):
            sum_[z[i]-1][0] += x_coordinates[i]      # Computing sum of x-coordinates of same cluster data points
            sum_[z[i]-1][1] += y_coordinates[i]      # Computing sum of y-coordinates of same cluster data points
            count_[z[i]-1] += 1                      # Computing number of data points present in a cluster

        for i in range(0,len(count_)):
            mean_[i][0] = sum_[i][0] / count_[i]        # Computing mean of x-coordinates of a cluster
            mean_[i][1] = sum_[i][1] / count_[i]        # Computing mean of y-coordinates of a cluster

        diff=0
        temp = np.copy(z)
        for i in range(0,len(z)):
            d = distance(x_coordinates[i],y_coordinates[i],mean_[z[i]-1][0],mean_[z[i]-1][1])        # Finding distance of data point to mean of the cluster it is assigned 
            diff += d
            for j in range(0,len(count_)):
                m = distance(x_coordinates[i],y_coordinates[i],mean_[j][0],mean_[j][1])        # Finding distance of data point to mean of all the clusters
                if(m < d):        # Checking if distance to any other mean is less than its owm mean
                    d = m
                    temp[i] = j + 1        # If yes, then making a jump to other cluster
        z = np.copy(temp)

        e.append(diff)
        x.append(c)

        if(np.array_equal(z_previous, z)):        # If two simlutaneous indicator functions are same => No jumps are made => Break the loop
            break

    color = ["red", "indigo", "green", "orange","yellow"]
    plt.xlabel("X-Coordinates")
    plt.ylabel("Y-Coordinates")
    plt.title(f"Clustered Dataset for k = {k}")
    for i in range(len(z)):
        plt.scatter(x_coordinates[i], y_coordinates[i], c = color[z[i]-1])        # Plotting the data points with different colours as their clusters assigned
    plt.show()
    
    # Plotting Vornoi Regions
    plt.xlabel("X-Coordinates")
    plt.ylabel("Y-Coordinates")
    plt.title(f"Vornoi Region for k = {k}")
    for i in np.arange(-10.0, 10.5, 0.5):
        for j in np.arange(-10.0, 10.5, 0.5):
            c = 0
            m = float(sys.maxsize)
            for n in range(k):
                d = distance(i, j, mean_[n][0], mean_[n][1])        # Finding distance of data point to all the mean
                if m > d:       # Checking for the minimum distance and updating 
                    m = d
                    c = n
            plt.scatter(i, j, c = color[c])        # Colouring the region according to the minimum distance
    for i in range(k):
        plt.scatter(mean_[i][0], mean_[i][1], marker="o", color="k")
    plt.show()

