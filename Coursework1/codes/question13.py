import numpy as np
from numpy.random import normal, uniform
from scipy.stats import multivariate_normal as mv_norm
import matplotlib.pyplot as plt

D = 100
n = 10
for i in range(1, n + 1):
    colormap = plt.cm.gist_ncar
    xs = np.linspace(-10,10,D)
    ys = np.random.multivariate_normal(np.zeros(D), np.eye(D))
    plt.plot(xs,i*ys)
    #plt.subplot(211)
    
plt.show()
plt.grid(True)
plt.title('10 samples from a %dD gaussian distribution' % D)
plt.xlabel("x")
plt.ylabel("t")

def mean(x):
    return np.zeros_like(x)

def kernel(xi, xj, sigma = 1, l = ""):
    X = np.expand_dims(xi,1) - np.expand_dims(xj,0)
    return (sigma ** 2) * np.exp(-(X/l) ** 2)

n = 10
for i in range(1, n + 1):
    colormap = plt.cm.gist_ncar
    xs = np.linspace(-10,10,100)
    ys = np.random.multivariate_normal(mean(xs), kernel(xs,xs,l=100))
    plt.plot(xs,i*ys, alpha = 0.8)
    #plt.subplot(212)

#plt.title('GP-prior with samples from a 5D Gaussian with L = 2')        
plt.grid(True)

plt.title('GP-prior with samples from a 100D Gaussian with L = 100')    
plt.xlabel("x")
plt.ylabel("t")
plt.show()
