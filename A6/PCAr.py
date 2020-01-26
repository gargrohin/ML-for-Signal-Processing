#!/usr/bin/env python
# coding: utf-8

# PCA implementation

# In[2]:


import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# k = 2

# In[23]:


np.random.seed(11)

set = pd.read_csv("mnist_train.csv")
labels = set['label']
cleanSet = set.drop("label", axis=1)

a=cleanSet.to_numpy()
mean=np.mean(a, axis=0)
std=np.std(a, axis=0)
a=a-mean
a=a/(std+1e-100)
normData=a
temp= normData
covarMat=np.matmul(temp.T, temp)
eigValues, eigVectors = eigh(covarMat, eigvals=(782,783))
eigVectors = eigVectors.T
pcaCord = np.matmul(eigVectors, temp.T)
pcaCordOrg=pcaCord
pcaCord = np.vstack((pcaCord, labels))


# transformed Data

# In[24]:


fig, ax = plt.subplots()
for g in np.unique(pcaCord[2]):
    ix = np.where(pcaCord[2] == g)
    ax.scatter(pcaCord[0][ix], pcaCord[1][ix], label = g, s=10)
ax.legend()
plt.show()


# Bases vectors

# In[25]:


fig, axs = plt.subplots(1, 2)
grid = eigVectors[0].reshape(28,28)
axs[0].imshow(grid)
axs[0].set_title("First basis vector")

grid = eigVectors[1].reshape(28,28)
axs[1].imshow(grid)
axs[1].set_title("Second basis vector")

plt.show()


# In[26]:


recX=(np.matmul(eigVectors.T, pcaCordOrg)).T
recX=recX*(std+1e-100)
recX=recX+mean


# Reconstruction Images

# In[27]:


grid = recX[1].reshape(28,28) # reshape from 1d to 2d pixel array
plotThese=recX[[1,3,5,7,9,0,13,15,17,4]]
fig=plt.figure(figsize=(10, 10))
    
for i in range(0,10):
    fig.add_subplot(1,10,i+1)
    img = plotThese[i].reshape(28,28) # reshape from 1d to 2d pixel array
    plt.imshow(img)
    plt.title(str(i))
    plt.axis('off')
plt.show()
    


# k = 3

# In[28]:


np.random.seed(15)

temp= normData
covarMat=np.matmul(temp.T, temp)
eigValues3, eigVectors3 = eigh(covarMat, eigvals=(781,783))
eigVectors3 = eigVectors3.T
pcaCord3 = np.matmul(eigVectors3, temp.T)
pcaCord3Org=pcaCord3
pcaCord3 = np.vstack((pcaCord3, labels))


# transformed data

# In[29]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
for g in np.unique(pcaCord3[3]):
    ix = np.where(pcaCord3[3] == g)
    ax.scatter(pcaCord3[0][ix], pcaCord3[1][ix], pcaCord3[2][ix], label = g, s=10)
ax.legend()
plt.show()


# bases vectors

# In[30]:


fig, axs = plt.subplots(1, 3)
grid = eigVectors3[0].reshape(28,28)
axs[0].imshow(grid)
axs[0].set_title("First basis vector")

grid = eigVectors3[1].reshape(28,28)
axs[1].imshow(grid)
axs[1].set_title("Second basis vector")

grid = eigVectors3[2].reshape(28,28)
axs[2].imshow(grid)
axs[2].set_title("Third basis vector")

plt.show()


# In[31]:


recX=(np.matmul(eigVectors3.T, pcaCord3Org)).T
recX=recX*(std+1e-100)
recX=recX+mean


# Reconstruction images

# In[32]:


plotThese=recX[[1,3,5,7,9,0,13,15,17,4]]
fig=plt.figure(figsize=(10, 10))
    
for i in range(0,10):
    fig.add_subplot(1,10,i+1)
    img = plotThese[i].reshape(28,28) # reshape from 1d to 2d pixel array
    plt.imshow(img)
    plt.title(str(i))
    plt.axis('off')
plt.show()
    

