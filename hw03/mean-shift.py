
# coding: utf-8

# In[1]:

'''
 * @Author: 
 * @Date: 2019-05-27
'''


# In[2]:

import numpy as np
import numpy.matlib as ml
import random
import matplotlib.pyplot as plt


# In[3]:

def calc_prob(k, pMiu, pSigma):
    Px = np.zeros([len(samples.T), k], dtype=float)
    for i in range(k):
        Xshift = np.mat(X - pMiu[i, :])
        inv_pSigma = np.mat(pSigma[:, :, i]).I
        coef = pow((2*np.pi), (len(X[0])/2)) *             np.sqrt(np.linalg.det(np.mat(pSigma[:, :, i])))
        for j in range(len(samples.T)):
            tmp = (Xshift[j, :] * inv_pSigma * Xshift[j, :].T)
            Px[j, i] = 1.0 / coef * np.exp(-0.5*tmp)
    return Px


# In[4]:

def distmat(X, Y):
    n = len(X)
    m = len(Y)
    xx = ml.sum(X*X, axis=1)
    yy = ml.sum(Y*Y, axis=1)
    xy = ml.dot(X, Y.T)
    return np.tile(xx, (m, 1)).T + np.tile(yy, (n, 1)) - 2*xy


# In[5]:

def init_params(centers, k):
    pMiu = centers
    pPi = np.zeros([1, k], dtype=float)
    pSigma = np.zeros([len(X[0]), len(X[0]), k], dtype=float)
    dist = distmat(X, centers)
    labels = dist.argmin(axis=1)
    for j in range(k):
        idx_j = (labels == j).nonzero()
        pMiu[j] = X[idx_j].mean(axis=0)
        pPi[0, j] = 1.0 * len(X[idx_j]) / len(samples.T)
        pSigma[:, :, j] = np.cov(np.mat(X[idx_j]).T)
    return pMiu, pPi, pSigma


# In[6]:

mean = [0, 10]
cov = [[1, 0], [0, 100]]
k = 3
samples = np.random.multivariate_normal(mean, cov, 1000).T
X = samples.T


# In[7]:

labels = np.zeros(len(X), dtype=int)


# In[8]:

index = np.array(random.sample(list(range(len(X))), k))


# In[9]:

centers = np.array(X[index])


# In[10]:

Lprev = float(-10000)
pre_esp = 100000
threshold = 1e-6
maxiter = 100
pMiu, pPi, pSigma = init_params(centers, k)


# In[11]:

iter = 0
while iter < maxiter:
    Px = calc_prob(k, pMiu, pSigma)
    pGamma = np.mat(np.array(Px) * np.array(pPi))
    pGamma = pGamma / pGamma.sum(axis=1)
    Nk = pGamma.sum(axis=0)
    pMiu = np.diagflat(1/Nk) * pGamma.T * np.mat(X)
    pPi = Nk / len(samples.T)
    pSigma = np.zeros([len(X[0]), len(X[0]), k], dtype=float)
    for j in range(k):
        Xshift = np.mat(X) - pMiu[j, :]
        for i in range(len(samples.T)):
            pSigmaK = Xshift[i, :].T * Xshift[i, :]
            pSigmaK = pSigmaK * pGamma[i, j] / Nk[0, j]
            pSigma[:, :, j] = pSigma[:, :, j] + pSigmaK
    labels = pGamma.argmax(axis=1)
    if (iter+1) % 10 == 0:
        plt.clf()
        labels = np.array(labels).ravel()
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        data_colors = [colors[lbl] for lbl in labels]
        plt.scatter(X[:, 0], X[:, 1], c=data_colors, alpha=0.5)
        plt.savefig('%d.png' % (iter+1))
        plt.show()
    iter = iter + 1
    L = sum(np.log(np.mat(Px) * np.mat(pPi).T))
    cur_esp = L-Lprev
    if cur_esp < threshold or cur_esp > pre_esp:
        plt.clf()
        labels = np.array(labels).ravel()
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        data_colors = [colors[lbl] for lbl in labels]
        plt.scatter(X[:, 0], X[:, 1], c=data_colors, alpha=0.5)
        plt.savefig('%d.png' % (iter-1))
        plt.show()
        break
    pre_esp = cur_esp
    Lprev = L
    print("iter %d esp %lf" % (iter, cur_esp))


# In[ ]:



