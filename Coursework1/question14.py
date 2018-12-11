
import numpy as np
import matplotlib.pyplot as plt
import pylab
from numpy.random import normal, uniform

n = 50
X_test = np.linspace(-10, 10, n).reshape(-1,1)

def kernel(xi, xj, sigma, L):
    X = np.sum(a**2,1).reshape(-1,1)+np.sum(b**2,1)-2*np.dot(xi, xj.T)
    return np.exp(-sigma*(1/param)*X)

L = 1
num_samples
K_ss = kernel(X_test, X_test, L)

# Cholesky decomposition
Log = np.linalg.cholesky(K_ss + 1e-15*np.eye(n))
y_test = np.dot(Log, np.random.normal(size=(n,num_samples)))

plt.plot(X_test, y_test)
plt.axis([-np.pi, np.pi, -np.pi, np.pi])
plt.title('%d samples from the GP prior' % num_samples)
plt.show()

X_obs = np.linspace(-np.pi,np.pi,7).reshape(7,1)
y_obs = np.sin(X_obs)

K = kernel(X_obs, y_obs, L)
Log = np.linalg.cholesky(K + 0.00005*np.eye(len(X_obs))

K_s = kernel(X_obs, X_test, L)
Log_k = np.linalg.solve(L, Ks)
m_u = np.dot(Lk.T, np.linalg.solve(Log, y_obs)).reshape((n,))
s2 = np.diag(K_ss) - np.sum(Log_k**2, axis=0)
stdv = np.sqrt(s2)

Log = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Log_k.T, Log_k))
y_test = m_u.reshape(-1,1) + np.dot(Log, np.random.normal(size=(n,3)))

plt.plot(X_obs, y_obs, 'black', marker = 'o', ms=10, alpha = 1)
plt.plot(X_test, y_test, alpha = 0.7)
plt.gca().fill_between(X_test.flat, m_u-2*stdv, m_u+2*stdv, color="#dddddd")
plt.plot(X_test, m_u, color = 'red', linestyle = 'dashed', lw=2)
plt.title('3 samples from the GP posterior')
pylab.ylim(-np.pi,np.pi)
pylab.xlim(-np.pi,np.pi)
plt.show()
