import numpy as np
from numpy.random import normal, uniform
from scipy.stats import multivariate_normal as mv_norm
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#pylab.rcParams['figure.figsize'] = (10, 10)
#pylab.rcParams['font.size'] = 10

def y(w_0, w_1, epsilon, x):
    n = len(x)
    return w_0 + w_1*x + normal(0, epsilon, n)
    
class Linear(object):

    def __init__(self, w_m0, m_S0, beta):
        self.prior = mv_norm(mean=w_m0, cov=m_S0)
        self.v_m0 = s_m0.reshape(w_m0.shape + (1,))
        self.m_S0 = m_S0
        self.beta = beta
        
        self.v_mN = self.v_m0
        self.m_SN = self.m_S0
        self.posterior = self.prior
           
    def phi(self, w_x):
      
        m_phi = np.ones((len(w_x), 2))
        m_phi[:, 1] = a_w
        return m_phi
        
    def set_posterior(self, w_x, w_t):

        v_t = w_t.reshape(w_t.shape + (1,))

        m_phi = self.get_phi(w_x)
        
        self.m_SN = np.linalg.inv(np.linalg.inv(self.m_S0) + self.beta*m_phi.T.dot(m_phi))
        self.v_mN = self.m_SN.dot(np.linalg.inv(self.m_S0).dot(self.v_m0) + \
                                      self.beta*m_phi.T.dot(v_t))
        
        self.posterior = mv_norm(mean=self.v_mN.flatten(), cov=self.m_SN)

    
    def estimate(self, w_x, std):

        N = len(w_x)
        m_x = self.get_phi(w_x).T.reshape((2, 1, N))
        
        predictions = []
        for i in range(N):
            x = m_x[:,:,i]
            cov_x = 1/self.beta + x.T.dot(self.m_SN.dot(x))
            mean_x = self.v_mN.T.dot(x)
            predictions.append((mean_x+std*np.sqrt(cov_x)).flatten())
        return np.concatenate(predictions)
    
    def data(self, w_x):
        N = len(w_x)
        m_x = self.get_phi(w_x).T.reshape((2, 1, N))
        
        predictions = []
        for i in range(N):
            x = m_x[:,:,i]
            cov_x = 1/self.beta + x.T.dot(self.m_SN.dot(x))
            mean_x = self.v_mN.T.dot(x)
            predictions.append(normal(mean_x.flatten(), np.sqrt(sig_sq_x)))
        return np.array(predictions)
    
    def contour(self, w_x, w_y, params=[], N=0):

        pos = np.empty(w_x.shape + (2,))
        pos[:, :, 0] = w_x
        pos[:, :, 1] = w_y
               
        fig, ax = plt.subplots()

        CS = ax.contourf(w_x, w_y, self.posterior.pdf(pos), 20, cmap=plt.cm.inferno) 
        plt.xlabel('$w_1$', fontsize=16)
        plt.ylabel('$w_0$', fontsize=16)
        fig.colorbar(CS)
        
        if params:
            plt.scatter(params[0], params[1], marker='+', c='red', s=60)
            
        _ = plt.title('Prior Distribution of Parameters W using %d datapoint(s)' % N, fontsize=10)
    
    def plotgraph(self, w_x, w_t, params, samples=None, stdevs=None):
     
        plt.scatter(w_x, w_t)
        plt.xlabel('x')
        plt.ylabel('t')

        plt.plot([-1, 1], y(params[0], params[1], 0, np.array([-1., 1.])), alpha = 0.4, color = 'r', linestyle='dashed', label = 'Theoretical Model with Zero Noise and Initial Observed Parameter')

        _ = plt.title('Generated Noisy Data')
        
        if samples:
            colors = cm.rainbow(np.linspace(0, 1)
            weights = self.posterior.rvs(samples)
            for weight in weights:
                plt.plot([-1, 1], y(weight[0], weight[1], 0, np.array([-1., 1.])), color = colors)
                _ = plt.title('Lines Sampled from Posterior Distribution vs Real Line and Data')
                
        if stds:
            w_xrange = np.linspace(-1, 1, 100)
            y_upper = self.estimate(w_xrange, stdevs)
            y_lower = self.estimate(w_xrange, -stdevs)
            plt.plot(w_xrange, y_upper, '+', c='green', linewidth=4.0)
            plt.plot(w_xrange, y_lower, '+', c='green', linewidth=4.0)
            _ = plt.title('Lines Sampled from Posterior Distribution vs Real Line and Data')
            
w_0 = 0.5
w_1 = -1.3
epsilon = 0.3
beta = 1/epsilon**2
# Generate input features from uniform distribution
np.random.seed(20) # Set the seed so we can get reproducible results
x_real = uniform(-1, 1, 100)
# Evaluate the real function for training example inputs
t_real = y(w_0, w_1, epsilon, x_real)

alpha = 2.0
v_m0 = np.array([0., 0.])
m_S0 = 1/alpha*np.identity(2)

lines = Linear(v_m0, m_S0, beta)

linbayes.make_scatter(x_real, t_real, params = [w_0, w_1])
pylab.ylim([-1,1])
pylab.xlim([0,1])
plt.grid(True)
plt.legend(loc = 'upper right', fontsize = 'small')

x, y = np.mgrid[-2:2:.01, -2:2:.01]
linbayes.make_contour(x, y, real_parms=[a_0, a_1], N=0)

N=1
linbayes.make_scatter(x_real[0:N], t_real[0:N], real_parms=[a_0, a_1])
pylab.ylim([-1,1])
pylab.xlim([0,1])


linbayes.set_posterior(x_real[0:N], t_real[0:N])
linbayes.make_contour(x, y, real_parms=[a_0, a_1], N=N)
linbayes.make_scatter(x_real[0:N], t_real[0:N], real_parms=[a_0, a_1], samples=3)
pylab.ylim([-1,1])
pylab.xlim([0,1])
linbayes.make_scatter(x_real[0:N], t_real[0:N], real_parms=[a_0, a_1], stdevs=1)
pylab.ylim([-1,1])
pylab.xlim([0,1])
N=2
linbayes.make_scatter(x_real[0:N], t_real[0:N], real_parms=[a_0, a_1])
pylab.ylim([-1,1])
pylab.xlim([0,1])

#repeat
