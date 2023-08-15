#!/usr/bin/env python
# coding: utf-8

# # 1.(iii)

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
plt.xlabel("X-Coordinates of Dataset")
plt.ylabel("Y-Coordinates of Dataset")
plt.title("Original Dataset")
plt.show()


# ## (A) Polynomial Kernel

# ### Quadratic Kernel

# #### Finding XXt

# In[4]:


k_temp = df.dot(df.T)


# #### Computing XXt+1

# In[5]:


k_temp = k_temp + 1


# #### Computing (XXt+1)^2 , i.e., Kernel Matrix K

# In[6]:


k = k_temp**2


# #### Centering Kernel Matrix Kc

# In[7]:


kc = np.identity(1000)-(1/1000)
kc = kc.dot(k).dot(kc)


# #### Computing Eigen Values & Eigen Vectors of Centered Kernel Matrix Kc

# In[8]:


eigen_value,eigen_vector = np.linalg.eigh(kc)


# #### Sorting Eigen Values & Eigen Vectors

# In[9]:


top_eigen_value = eigen_value.argsort()[::-1]
sorted_eigen_value = eigen_value[top_eigen_value]
sorted_eigen_vector = eigen_vector[:,top_eigen_value]


# #### Finding top Alpha(k) 's

# In[10]:


alpha1 = np.divide(sorted_eigen_vector[:,0],math.sqrt(sorted_eigen_value[0]))
alpha2 = np.divide(sorted_eigen_vector[:,1],math.sqrt(sorted_eigen_value[1]))


# #### Computing x(i)'s by Multiplying with Kernel Matrix

# In[11]:


x1 = np.matmul(alpha1,k)
x2 = np.matmul(alpha2,k)


# #### Plotting projection of datapoints onto the Top Principal Components 

# In[12]:


plt.scatter(x1,x2)
plt.xlabel("X-Coordinates")
plt.ylabel("Y-Coordinates")
plt.title("Quadratic Polynomial Kernel PCA Output")
plt.show


# ### Cubic Kernel

# #### Finding XXt

# In[13]:


k = df.dot(df.T)


# #### Computing XXt+1

# In[14]:


k = k + 1


# #### Computing (XXt+1)^3 , i.e., Kernel Matrix K

# In[15]:


k = k**3


# #### Centering Kernel Matrix

# In[16]:


kc = np.identity(1000)-(1/1000)
kc = kc.dot(k).dot(kc)


# #### Computing Eigen Values & Eigen Vectors of the Kernel Matrix

# In[17]:


eigen_value,eigen_vector = np.linalg.eigh(kc)


# #### Sorting Eigen Values & Eigen Vectors

# In[18]:


top_eigen_value = eigen_value.argsort()[::-1]
sorted_eigen_value = eigen_value[top_eigen_value]
sorted_eigen_vector = eigen_vector[:,top_eigen_value]


# #### Finding top Alpha(k)'s

# In[19]:


alpha1 = np.divide(sorted_eigen_vector[:,0],math.sqrt(sorted_eigen_value[0]))
alpha2 = np.divide(sorted_eigen_vector[:,1],math.sqrt(sorted_eigen_value[1]))


# #### Computing x(i)'s by Multiplying with Kernel Matrix

# In[20]:


x1 = np.matmul(alpha1,kc)
x2 = np.matmul(alpha2,kc)


# #### Plotting projections of Datapoints onto the Top Principal Components

# In[21]:


plt.scatter(x1,x2)
plt.xlabel("X-Coordinates")
plt.ylabel("Y-Coordinates")
plt.title("Cubic Polynomial Kernel PCA Output")
plt.show


# ## (B) Radial Basis Kernel

# In[22]:


data = np.array(df)


# In[23]:


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
    
    # Centering the Kernel Matrix K
    kc = np.identity(1000)-(1/1000)
    kc = kc.dot(t).dot(kc)
    
    # Finding Eigen Values & Eigen Vectors of Kernel Matrix
    eigen_value, eigen_vector = np.linalg.eigh(kc)

    # Sorting Eigen Values & Eigen Vectors
    top_eigen_value = eigen_value.argsort()[::-1]
    sorted_eigen_value = eigen_value[top_eigen_value]
    sorted_eigen_vector = eigen_vector[:,top_eigen_value]
    
    # Computing Alpha(K)'s
    alpha1 = np.divide(sorted_eigen_vector[:,0],math.sqrt(sorted_eigen_value[0]))
    alpha2 = np.divide(sorted_eigen_vector[:,1],math.sqrt(sorted_eigen_value[1]))
    
    # Computing top Principal Components 
    x1 = np.matmul(alpha1,kc)
    x2 = np.matmul(alpha2,kc)
    
    # Plotting projection of datapoints onto the Top Principal Components
    plt.scatter(x1,x2)
    plt.title("PCA for Radial Basis Kernel")
    plt.xlabel("X-Coordinates")
    plt.ylabel("Y-Coordinates")
    plt.show()

