import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt
from sklearn.decomposition import NMF
from sklearn.neighbors import KNeighborsClassifier

# ===============================================
# Program Parameters
# ===============================================

train_samples = 1000
nmf_components = 49

# ===============================================
# Functions
# ===============================================

np.random.seed(0)

# ===============================================
# Import Data & Clean
# ===============================================

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

inputs_train = X_train[0:train_samples].astype('float32') / 255.
inputs_train = inputs_train.reshape((len(inputs_train), np.prod(inputs_train.shape[1:])))

'''
inputs_test = X_test[0:train_samples].astype('float32') / 255.
inputs_test = inputs_test.reshape((len(inputs_test), np.prod(inputs_test.shape[1:])))
'''

# ===============================================
# Raw images
# ===============================================

fig=plt.figure(figsize=(8, 8))
rows = min(int(np.sqrt(train_samples)),10)
columns = min(int(train_samples / rows),10)
for i in range(0, columns*rows):
    img = inputs_train[i].reshape((28, 28))
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.imshow(img)
plt.show()

# ===============================================
# NMF Deconstruction
# ===============================================

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
nmf = NMF(n_components=nmf_components, random_state=0, solver='mu', init='random', max_iter=200, tol=1e-4)

W = nmf.fit_transform(inputs_train)
H = nmf.components_

# H = n_components x pixels (list of components)

fig=plt.figure(figsize=(8, 8))
rows = min(int(np.sqrt(nmf_components)),10)
columns = min(int(nmf_components / rows),10)
for i in range(0, columns*rows):
    img = H[i].reshape((28, 28))
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.imshow(img)
plt.show()

# W = samples x n_components (component mapping for each sample)

fig=plt.figure(figsize=(8, 8))
rows = min(int(np.sqrt(train_samples)),10)
columns = min(int(train_samples / rows),10)
for i in range(0, columns*rows):
    img = W[i].reshape((int(np.sqrt(nmf_components)), int(np.sqrt(nmf_components))))
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.imshow(img)
plt.show()

# ===============================================
# Reconstruction
# ===============================================

output = nmf.inverse_transform(W)

fig=plt.figure(figsize=(8, 8))
rows = min(int(np.sqrt(train_samples)),10)
columns = min(int(train_samples / rows),10)
for i in range(0, columns*rows):
    img = output[i].reshape((28, 28))
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.imshow(img)
plt.show()

# ===============================================
