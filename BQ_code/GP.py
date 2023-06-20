"""
Created on April 2020 by @author: Vu Nguyen

Reference Link : https://github.com/ntienvu/MiniBO

Modified on December 2021 by Xu Cai

Implementation of the GP model
"""

import math
import scipy
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import scipy
#from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
#import matplotlib as mpl
import matplotlib.cm as cm
from scipy.linalg import block_diag

def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking
    :param a: array to trim repeated rows from
    :return: mask of unique rows
    """
    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]

class GaussianProcess(object):
    def __init__ (self,SearchSpace,noise_delta=1e-8,verbose=0,kernel='mat'):
        self.noise_delta=noise_delta
        self.noise_upperbound=noise_delta

        self.SearchSpace=SearchSpace
        scaler = MinMaxScaler()
        scaler.fit(SearchSpace.T)
        self.Xscaler=scaler
        self.verbose=verbose
        self.dim=SearchSpace.shape[0]

        if kernel=='mat':
            self.mycov=self.cov_MATERN
        elif kernel=='se':
            self.mycov=self.cov_SE
        else:
             raise NotImplementedError(str(kernel)+' is currently not supported')
        
        self.hyper={}
        self.hyper['var']=1 # standardise the data
        self.hyper['lengthscale']=0.1 #to be optimised
        self.nu = 1.5  # not to be optimised
   
        
    def fit(self,X,Y,IsOptimize=0):
        """
        Fit a Gaussian Process model
        X: input 2d array [N*d]
        Y: output 2d array [N*1]
        """       
        #self.X= self.Xscaler.transform(X) #this is the normalised data [0-1] in each column
        self.X=X
        # self.Y=(Y-np.mean(Y))/(np.std(Y)+1e-8) # this is the standardised output N(0,1)
        self.Y=Y
        
        if IsOptimize:
            optimized_hyper = self.optimise()
            self.hyper['lengthscale']=optimized_hyper[0]         # optimise GP hyperparameters
            self.hyper['var']=optimized_hyper[1]  
        self.KK_x_x=self.mycov(self.X,self.X,self.hyper)+np.eye(len(X))*self.noise_delta     
        if np.isnan(self.KK_x_x).any(): #NaN
            print("nan in KK_x_x !")
      
        self.L=scipy.linalg.cholesky(self.KK_x_x,lower=True)
        temp=np.linalg.solve(self.L,self.Y)
        self.alpha=np.linalg.solve(self.L.T,temp)
        
    def cov_SE(self,x1, x2,hyper):        
        """
        Radial Basic function kernel (or SE kernel)
        """
        variance=hyper['var']
        lengthscale=hyper['lengthscale']

        if x1.shape[1]!=x2.shape[1]:
            x1=np.reshape(x1,(-1,x2.shape[1]))
        Euc_dist=euclidean_distances(x1,x2)

        return variance*np.exp(-np.square(Euc_dist)/lengthscale)

    def cov_MATERN(self, x1, x2, hyper):
        """
        Matern kernel function
        """
        variance=hyper['var']
        lengthscale=hyper['lengthscale']

        if x1.shape[1]!=x2.shape[1]:
            x1=np.reshape(x1,(-1,x2.shape[1]))
        dists=euclidean_distances(x1/lengthscale, x2/lengthscale)
        
        if self.nu == 0.5:
            K=np.exp(-dists)

        elif self.nu == 1.5:
            K=dists*math.sqrt(3)
            K=(1.0+K)*np.exp(-K)

        elif self.nu == 2.5:
            K=dists*math.sqrt(5)
            K = (1.0+K+K**2/3.0)*np.exp(-K)
        else:
            raise NotImplementedError(str(self.nu)+' is currently not supported')

        return variance*K

    def log_llk(self,X,y,hyper_values):
        
        #print(hyper_values)
        hyper={}
        hyper['var']=hyper_values[1]
        hyper['lengthscale']=hyper_values[0]
        # noise_delta=self.noise_delta

        KK_x_x=self.mycov(X,X,hyper)+np.eye(len(X))*self.noise_delta     
        if np.isnan(KK_x_x).any(): #NaN
            print("nan in KK_x_x !")   

        try:
            L=scipy.linalg.cholesky(KK_x_x,lower=True)
            alpha=np.linalg.solve(KK_x_x,y)

        except: # singular
            return -np.inf
        try:
            first_term=-0.5*np.dot(self.Y.T,alpha)
            W_logdet=np.sum(np.log(np.diag(L)))
            second_term=-W_logdet

        except: # singular
            return -np.inf

        logmarginal=first_term+second_term-0.5*len(y)*np.log(2*3.14)
        logmarginal = np.array(logmarginal)
        #print(hyper_values,logmarginal)
        try:
            logmarginal = np.ndarray.item(logmarginal)
        except:
            print(logmarginal)
        return logmarginal
    
    def set_hyper (self,lengthscale,variance,nu):
        self.hyper['lengthscale']=lengthscale
        self.hyper['var']=variance
        self.nu=nu

        
    def optimise(self):
        """
        Optimise the GP kernel hyperparameters
        Returns
        x_t
        """
        opts ={'maxiter':200,'maxfun':200,'disp': False}


        bounds=np.asarray([[1e-2,1],[0.05,5]]) # bounds on Lenghtscale and keranl Variance

        init_theta = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(10, 2))
        logllk=np.array([])

        for x in init_theta:
            logllk=np.append(logllk,self.log_llk(self.X,self.Y,hyper_values=x))
            
        x0=init_theta[np.argmax(logllk)]

        res = minimize(lambda x: -self.log_llk(self.X,self.Y,hyper_values=x),x0,
                                   bounds=bounds,method="L-BFGS-B",options=opts)#L-BFGS-B
        
        if self.verbose:
            print("estimated lengthscale and variance",res.x)
            
        return res.x  
   
    def predict(self,Xtest,isOriScale=False):
        """
        ----------
        Xtest: the testing points  [N*d]
        Returns
        -------
        pred mean, pred var
        """    
        
        if isOriScale:
            Xtest=self.Xscaler.transform(Xtest)
            
        if len(Xtest.shape)==1: # 1d
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
            
        if Xtest.shape[1] != self.X.shape[1]: # different dimension
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
       
        KK_xTest_xTest=self.mycov(Xtest,Xtest,self.hyper)+np.eye(Xtest.shape[0])*self.noise_delta
        KK_xTest_x=self.mycov(Xtest,self.X,self.hyper)

        mean=np.dot(KK_xTest_x,self.alpha)
        v=np.linalg.solve(self.L,KK_xTest_x.T)
        var=KK_xTest_xTest-np.dot(v.T,v)

        std=np.reshape(np.diag(var),(-1,1))
        
        return  np.reshape(mean,(-1,1)),std


    def predict_mean_only(self,Xtest,isOriScale=False):
        """
        ----------
        Xtest: the testing points  [N*d]
        Returns
        -------
        pred mean
        """    
        
        if isOriScale:
            Xtest=self.Xscaler.transform(Xtest)
            
        if len(Xtest.shape)==1: # 1d
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
            
        if Xtest.shape[1] != self.X.shape[1]: # different dimension
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
       
        KK_xTest_x=self.mycov(Xtest,self.X,self.hyper)

        mean=np.dot(KK_xTest_x,self.alpha)

        return mean.reshape(-1,1)

   
   # sampling a point from the posterior
    def sample(self,X,size):
        m, var = self.predict(X)
        v=self.covar(X)
        def sim_one_dim(m, v):
            return np.random.multivariate_normal(m, v, size).T
        return sim_one_dim(m.flatten(), v)[:, np.newaxis, :]
    
    # Returns the covariance matrix
    def covar(self,X):
        return(self.mycov(X,X,self.hyper))