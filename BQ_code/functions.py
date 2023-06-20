"""
Implementation of all the testing functions
"""
import os
import math
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class Keane2d:
    def __init__(self, noisy=True, noise_std=0.05):
        self.dim=2
        self.bounds=np.array([[0, 1]] * self.dim)
        self.noisy=noisy
        self.noise_std = noise_std
        self.true_quad = 0.8732461935237423

    def __call__(self,X):
        X = np.array(X)
        X = X.reshape(-1, self.dim)
        X = (2*X-1)*4
        # X *= 4

        out =  np.abs((np.cos(X[:,0])**4 + np.cos(X[:,1])**4 \
                    - 2 * (np.cos(X[:,0])**2) * (np.cos(X[:,1])**2))) \
                / np.sqrt(1*X[:,0]**2 + 1.5*X[:,1]**2 + 1e-8)
        out *= 10

        return out + np.random.normal(0, self.noise_std, size=(X.shape[0], )) if self.noisy else out



class Griewank1d :
    def __init__(self, noisy=True, noise_std=0.05):
        self.dim=1
        self.bounds=np.array([[0,1]] * self.dim)
        self.noisy=noisy
        self.noise_std = noise_std
        self.true_quad = 1.06273544442227

    def __call__(self,X):
        X = np.array(X)
        X = X.reshape(-1, self.dim)

        X = X*20 - 10
            
        out = X**2/4000 - np.cos(X) + 1

        out = out.squeeze()

        return out + np.random.normal(0, self.noise_std, size=(X.shape[0], )) if self.noisy else out


class Griewank2d :
    def __init__(self, noisy=True, noise_std=0.05):
        self.dim=2
        self.bounds=np.array([[0,1]] * self.dim)
        self.noisy=noisy
        self.noise_std = noise_std
        self.true_quad = 1.022120376314382

    def __call__(self,X):
        X = np.array(X)
        X = X.reshape(-1, self.dim)

        X = X*20 - 10
            
        out = (X[:,0]**2 + X[:,1]**2) /4000 - np.cos(X[:,0])*np.cos(X[:,1]/np.sqrt(2)) + 1

        out = out.squeeze()

        return out + np.random.normal(0, self.noise_std, size=(X.shape[0], )) if self.noisy else out


class Rastrigin2d:
    def __init__(self, noisy=True, noise_std=0.05):
        self.dim=2
        self.bounds=np.array([[0,1]] * self.dim)
        self.noisy=noisy
        self.noise_std = noise_std
        self.true_quad = 3.7050684417886197

    def __call__(self,X):
        X = np.array(X)
        X = X.reshape(-1, self.dim)
        X = (2*X-1)*5.12
            
        out = (X[:,0]**2 - 10 * np.cos(2 * np.pi * X[:,0])) + (X[:,1]**2 - 10 * np.cos(2 * np.pi * X[:,1])) + 10 * self.dim

        out *= 0.1

        return out + np.random.normal(0, self.noise_std, size=(X.shape[0], )) if self.noisy else out


class Gramacy_Lee1d:
    def __init__(self, noisy=True, noise_std=0.05):
        self.dim=1
        self.bounds=np.array([[0,1]] * self.dim)
        self.noisy=noisy
        self.noise_std = noise_std
        self.true_quad = 2.2496698169513656

    def __call__(self,X):
        X = np.array(X)
        X = X.reshape(-1, self.dim)

        X = X*2 + 0.5
            
        out = 3*(np.sin(10*np.pi*X)/(2*X) + (X-1)**4).squeeze()

        return out + np.random.normal(0, self.noise_std, size=(X.shape[0], )) if self.noisy else out


class Alpine1d:
    def __init__(self, noisy=True, noise_std=0.05):
        self.dim=1
        self.bounds=np.array([[0,1]] * self.dim)
        self.noisy=noisy
        self.noise_std = noise_std

        self.true_quad = 3.0963167425409894

    def __call__(self,X):
        X = np.array(X)
        X = X.reshape(-1, self.dim)
        X = X * 10
            
        out = np.abs(X*np.sin(X)+0.1*X)
        out = out.squeeze()


        return out + np.random.normal(0, self.noise_std, size=(X.shape[0], )) if self.noisy else out


class Alpine2d:
    def __init__(self, noisy=True, noise_std=0.05):
        self.dim=2
        self.bounds=np.array([[0,1]] * self.dim)
        self.noisy=noisy
        self.noise_std = noise_std

        self.true_quad = 6.192633656214865

    def __call__(self,X):
        X = np.array(X)
        X = X.reshape(-1, self.dim)
        X = X * 10
            
        out = np.abs(X[:,0]*np.sin(X[:,0])+0.1*X[:,0]) + np.abs(X[:,1]*np.sin(X[:,1])+0.1*X[:,1])


        return out + np.random.normal(0, self.noise_std, size=(X.shape[0], )) if self.noisy else out


class Ackley1d:
    def __init__(self, noisy=True, noise_std=0.05):
        self.dim=1
        self.bounds=np.array([[0,1]] * self.dim)
        self.noisy=noisy
        self.noise_std = noise_std

        self.true_quad = 4.9682182524890015

    def __call__(self,X):
        X = np.array(X)
        X = X.reshape(-1, self.dim)
        X = (2*X-1)*2

        firstsum = X**2
        secondsum = np.cos(2*np.pi*X)

        out = -20 * np.exp(-0.2*np.sqrt(firstsum)) - np.exp(secondsum) + 20 + np.e
        out = out.squeeze()

        return out + np.random.normal(0, self.noise_std, size=(X.shape[0], )) if self.noisy else out


class Ackley2d:
    def __init__(self, noisy=True, noise_std=0.05):
        self.dim=2
        self.bounds=np.array([[0,1]] * self.dim)
        self.noisy=noisy
        self.noise_std = noise_std

        self.true_quad = 5.426723237989637

    def __call__(self,X):
        X = np.array(X)
        X = X.reshape(-1, self.dim)
        X = (2*X-1)*2

        firstsum = 0.5*(X[:,0]**2+X[:,1]**2)
        secondsum = np.cos(2*np.pi*X[:,0]) + np.cos(2*np.pi*X[:,1])

        out = -20 * np.exp(-0.2*np.sqrt(firstsum)) - np.exp(0.5*secondsum) + 20 + np.e

        return out + np.random.normal(0, self.noise_std, size=(X.shape[0], )) if self.noisy else out


class Syn_Base:
    def __init__(self, dim, noisy=True, noise_std=0.05, kernel='mat'):
        self.dim=dim
        self.n_samples = 30 * self.dim
        self.bounds=np.array([[0, 1]] * self.dim)
        self.noisy=noisy
        # self.rand_seed = 111 * self.dim
        self.rand_seed = 555 * self.dim

        if kernel=='mat':
            self.mycov=self._cov_MATERN
        elif kernel=='se':
            self.mycov=self._cov_SE
        else:
             raise NotImplementedError(str(kernel)+' is currently not supported')

        np.random.seed(self.rand_seed)
        self.weight_dir = './syn_weights'
        if not os.path.exists(self.weight_dir):
            os.mkdir(self.weight_dir)
        self.param_file = os.path.join(self.weight_dir, 'syn' + str(self.dim) + '_param_save.npy')

        
        if not os.path.exists(self.param_file):
            self.samples = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_samples, self.bounds.shape[0]))
            self.deltas = np.random.uniform(-1, 1, size=(self.n_samples))

            np.save(self.param_file, np.asarray({'samples' :  self.samples, 'deltas' :  self.deltas}))

        else:
            params = np.load(self.param_file, allow_pickle=True)[()]
            self.samples = params['samples']
            self.deltas = params['deltas']
        
        self.noise_std = noise_std
        self.var = 1
        self.l = 0.2
        self.nu = 1.5


    def _cov_MATERN(self, x1, x2):
        """
        Matern kernel function
        """

        if x1.shape[1]!=x2.shape[1]:
            x1=np.reshape(x1,(-1,x2.shape[1]))
        dists=euclidean_distances(x1/self.l, x2/self.l)
        
        if self.nu == 0.5:
            K=np.exp(-dists)

        elif self.nu == 1.5:
            K=dists*math.sqrt(3)
            K=(1.0+K)*np.exp(-K)

        elif self.nu == 2.5:
            K=dists*math.sqrt(5)
            K=(1.0+K+K**2/3.0)*np.exp(-K)
        else:
            raise NotImplementedError(str(self.nu)+' is currently not supported')

        return self.var*K

    def _cov_SE(self,x1, x2):        
        """
        Radial Basic function kernel (or SE kernel)
        """
        if x1.shape[1]!=x2.shape[1]:
            x1=np.reshape(x1,(-1,x2.shape[1]))
        dists=euclidean_distances(x1,x2)

        return self.var*np.exp(-np.square(dists)/self.l)

    def __call__(self, X):
        X = np.array(X)
        X = X.reshape(-1, self.dim)

        out = self.deltas @ self.mycov(self.samples, X) + 1

        return out + np.random.normal(0, self.noise_std, size=(X.shape[0], )) if self.noisy else out


class MAT_Synthetic1d(Syn_Base):
    def __init__(self, noisy=True, noise_std=0.05):
        Syn_Base.__init__(self, dim=1, noisy=noisy,noise_std=noise_std,kernel='mat')
        # self.true_quad = 0.933314942262178
        self.true_quad = -0.6944846134893783
        self.var = 1
        self.l = 0.2


class MAT_Synthetic2d(Syn_Base):
    def __init__(self, noisy=True, noise_std=0.05):
        Syn_Base.__init__(self, dim=2, noisy=noisy,noise_std=noise_std,kernel='mat')
        # self.true_quad = 1.3252756160898231
        self.true_quad = 0.8031351621142319
        self.var = 1
        self.l = 0.2


class MAT_Synthetic3d(Syn_Base):
    def __init__(self, noisy=True, noise_std=0.05):
        Syn_Base.__init__(self, dim=3, noisy=noisy,noise_std=noise_std,kernel='mat')
        # self.true_quad = 1.2847667646581131
        self.true_quad = 0.28315252941123614
        self.var = 1
        self.l = 0.2
   

class MAT_Synthetic4d(Syn_Base):
    def __init__(self, noisy=True, noise_std=0.05):
        Syn_Base.__init__(self, dim=4, noisy=noisy,noise_std=noise_std,kernel='mat')
        # self.true_quad = 1.1688270361821036
        self.true_quad = 1.1169133731025271
        self.var = 1
        self.l = 0.2


class SE_Synthetic1d(Syn_Base):
    def __init__(self, noisy=True, noise_std=0.05):
        Syn_Base.__init__(self, dim=1, noisy=noisy,noise_std=noise_std,kernel='se')
        # self.true_quad = 0.8251281243078049
        self.true_quad = -1.6816754811567298
        self.var = 1
        self.l = 0.2


class SE_Synthetic2d(Syn_Base):
    def __init__(self, noisy=True, noise_std=0.05):
        Syn_Base.__init__(self, dim=2, noisy=noisy,noise_std=noise_std,kernel='se')
        # self.true_quad = 1.7421534793666178
        self.true_quad = 0.5134773800128456
        self.var = 1
        self.l = 0.2


class SE_Synthetic3d(Syn_Base):
    def __init__(self, noisy=True, noise_std=0.05):
        Syn_Base.__init__(self, dim=3, noisy=noisy,noise_std=noise_std,kernel='se')
        # self.true_quad = 1.792995318777183
        self.true_quad = -0.9909837536355721
        self.var = 1
        self.l = 0.2
   

class SE_Synthetic4d(Syn_Base):
    def __init__(self, noisy=True, noise_std=0.05):
        Syn_Base.__init__(self, dim=4, noisy=noisy,noise_std=noise_std,kernel='se')
        self.true_quad = 1.4026530471385987
        self.var = 1
        self.l = 0.2



class Energy_TS:
    def __init__(self, noisy=True, noise_std=0.05):
        self.dim=1
        self.bounds=np.array([[0,1]] * self.dim)
        self.noisy=noisy
        self.noise_std = noise_std

        '''
        Uncomment the following lines on first run
        '''

        # import pandas as pd
        # self.raw_data = pd.read_csv("path_to_LCL-June2015v2_0.csv", header=0)
        # self.raw_data['date'] = pd.to_datetime(self.raw_data['DateTime'])
        # self.data = self.raw_data.loc[:, ['KWH/hh (per half hour) ']]
        # self.data = self.data.set_index(self.raw_data.date)
        # self.data['KWH/hh (per half hour) '] = pd.to_numeric(self.data['KWH/hh (per half hour) '],downcast='float',errors='coerce')
        # self.hourly_data = self.data.resample('H').sum().to_numpy().squeeze()

        self.hourly_data = np.load('./LCL-June2015v2_0.npy', allow_pickle=True)[()]
        self.total_num = self.hourly_data.size

        self.hour_region_scale = np.arange(self.total_num ) / self.total_num 

        self.true_quad = self.hourly_data.mean()


    def find_nearest_idx(self, X):
        idx = (np.abs(self.hour_region_scale - X)).argmin(axis=1)
        return idx
    
    def __call__(self,X):
        X = np.array(X).squeeze()
        X = X.reshape(-1, self.dim)

        idx = self.find_nearest_idx(X)
        out = self.hourly_data[idx] 

        return out + np.random.normal(0, self.noise_std, size=(X.shape[0], )) if self.noisy else out

