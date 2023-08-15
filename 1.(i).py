#!/usr/bin/env python
# coding: utf-8

# # 1.(i)

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import math


# ##### Reading Data from the csv File

# In[2]:


df = pd.read_csv("C:/Users/rudri/OneDrive/Desktop/PRML/Assignments/PRML Assignment 1/Dataset.csv",header = None, names = ['x','y'])
df


# ##### Original Dataset

# In[3]:


x_coordinates = np.array(df.x)
y_coordinates = np.array(df.y)
plt.scatter(x_coordinates,y_coordinates)
plt.xlabel("X-Coordinates")
plt.ylabel("Y-Coordinates")
plt.title("Original Dataset")
plt.show()


# ##### Centering Dataset

# In[4]:


sumx = np.sum(x_coordinates)        # finding sum of all x-coordinates
sumy = np.sum(y_coordinates)        # finding sum of all y-coordinates
    
meanx = sumx / len(x_coordinates)        # finding mean of x-coordinates
meany = sumy / len(y_coordinates)        # finding mean of y-coordinates

x = x_coordinates - meanx        # centering the x-coordinates
y = y_coordinates - meany        # centering the y-coordinates

plt.scatter(x,y)        
plt.xlabel("X-Coordinates of Centered Dataset")
plt.ylabel("Y-Coordinates of Centered Dataset")
plt.title("Centered Dataset")
plt.show()


# ##### Computing Covariance Matrix

# In[5]:


c = np.cov(x,y)
c


# ##### Finding Eigen Values & Eigen Vectors of the Covariance Matrix

# In[6]:


eigen_value, eigen_vector = np.linalg.eigh(c)
print(eigen_value)
print(eigen_vector)


# #### Ploting top Principal Components

# In[7]:


plt.quiver([0,0],[0,0], eigen_vector[1][1], eigen_vector[1][0], color = 'r', scale = 2.6)
plt.quiver([0,0],[0,0], eigen_vector[0][1], eigen_vector[0][0], color = 'purple', scale = 3)
plt.xlabel("X-Coordinates")
plt.ylabel("Y-Coordinates")
plt.title("Top Principal Components")
plt.legend(["First Principal Component", "Second Principal Component"], loc = "upper right")
plt.show()


# In[8]:


plt.scatter(x,y)
plt.quiver([0,0],[0,0], eigen_vector[1][1], eigen_vector[1][0], color = 'r', scale=2.4)
plt.quiver([0,0],[0,0], eigen_vector[0][1], eigen_vector[0][0], color = 'purple', scale=3.3)
plt.xlabel("X-Coordinates")
plt.ylabel("Y-Coordinates")
plt.title("Top Principal Components in Dataset")
plt.legend(["Dataset","First Principal Component", "Second Principal Component"], loc = "upper right")
plt.show()

