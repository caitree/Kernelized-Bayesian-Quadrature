"""
Class implementation for general Gaussian process bandit optimization
"""

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from .methods import Methods
from .GP import GaussianProcess, unique_rows


class GPBQ:
     def __init__(self, func, bounds, acq_name):
          self.X = None # The sampled point in original domain
          self.Y = None  # original output
          self.Y_mean = None
          self.Y_std = None
          # self.X_S=None   # scaled output (The input  is scaled [0,1] in all D)
          # self.Y_S=None   # scaled inpout ( output is scaled as  (Y - mu) / sigma )
          self.bounds=bounds
          self.dim = len(bounds) # original bounds
          self.bounds_s=np.array([np.zeros(self.dim), np.ones(self.dim)]).T  # scaled bounds
          self.func = func
          self.acq_name = acq_name
          scaler = MinMaxScaler()
          scaler.fit(self.bounds.T)
          self.Xscaler=scaler

          if 'se' in self.acq_name:
               self.kernel = 'se'
          elif 'mat' in self.acq_name:
               self.kernel = 'mat'
          self.gp=GaussianProcess(self.bounds_s, verbose=0, kernel=self.kernel)

          self.beta_func = lambda x: np.sqrt(np.log(x))
          
     
     def initiate(self, X_init, Y_init=None):
          self.X = np.asarray(X_init).reshape(-1, self.dim)
          self.Y = self.func(X_init).reshape(-1,1)
          self.gp.fit(self.X, self.Y)
     

     def set_hyper(self,lengthscale,variance,nu):
          """
          Manually set the GP hyperparameters
          """
          self.gp.set_hyper(lengthscale,variance,nu)  


     def sample_new_value(self, learn=True):
          """
          Sample the next best query point based on historical observations
          """
          
          ur = unique_rows(self.X)
          # self.gp.fit(self.X, self.Y)
          if len(self.Y)%(3)==0 and learn:
               self.gp.fit(self.X, self.Y, IsOptimize=True)
               # self.gp.optimise()
          else:
               self.gp.fit(self.X, self.Y)

          y_max=max(self.Y)
          query_num=len(self.Y)

          objects = Methods(self.gp, self.acq_name, self.bounds_s, y_max, self.X, self.Y, self.beta_func(query_num))
          x_val = objects.method_val()

          # x_val=self.Xscaler.inverse_transform(np.reshape(x_val,(-1,self.dim)))
          # y_obs= self.func(x_val[0]) 
          y_obs= self.func(np.reshape(x_val,(-1,self.dim))) 

          # self.X_S = np.vstack((self.X_S, x_val.reshape((1, -1))))
          self.X=np.vstack((self.X, x_val))
          self.Y = np.append(self.Y, y_obs)

          # self.Y_mean, self.Y_std = np.mean(self.Y), np.std(self.Y)
          # self.Y_S=(self.Y-self.Y_mean)/self.Y_std

          return x_val, y_obs
