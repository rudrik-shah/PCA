#!/usr/bin/env python
# coding: utf-8

# # 1.(ii)

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


# In[3]:


x_coordinates = np.array(df.x)
y_coordinates = np.array(df.y)
plt.scatter(x_coordinates,y_coordinates)
plt.xlabel("X-Axis of Dataset")
plt.ylabel("Y-Axis of Dataset")
plt.title("Original Dataset")
plt.show()


# ##### Computing Covariance Matrix

# In[4]:


c = np.cov(x_coordinates,y_coordinates)
c


# ##### Finding Eigen Values & Eigen Vectors of Covariance Matrix

# In[5]:


eigen_value,eigen_vector = np.linalg.eigh(c)
print(eigen_value)
print(eigen_vector)


# In[6]:


eigen_vector[1].dot(eigen_vector[0])


# ##### Plotting top Principal Components

# In[7]:


plt.quiver([0,0],[0,0], eigen_vector[1][1], eigen_vector[1][0], color = 'r', scale = 2.6)
plt.quiver([0,0],[0,0], eigen_vector[0][1], eigen_vector[0][0], color = 'purple', scale = 3)
plt.xlabel("X-Coordinates")
plt.ylabel("Y-Coordinates")
plt.title("Top Principal Components")
plt.legend(["First Principal Component", "Second Principal Component"], loc = "upper right")
plt.show()


# In[8]:


plt.scatter(x_coordinates,y_coordinates)
plt.quiver([0,0],[0,0], eigen_vector[1][1], eigen_vector[1][0], color = 'r', scale=2.4)
plt.quiver([0,0],[0,0], eigen_vector[0][1], eigen_vector[0][0], color = 'purple', scale=3.3)
plt.xlabel("X-Coordinates")
plt.ylabel("Y-Coordinates")
plt.title("Top Principal Components in Dataset")
plt.legend(["Dataset","First Principal Component", "Second Principal Component"], loc = "upper right")
plt.show()

