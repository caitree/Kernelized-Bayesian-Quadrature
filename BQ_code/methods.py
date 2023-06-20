"""
Implementations of acquisition functions
"""

import numpy as np
import scipy.optimize as spo
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import qmc


class Methods(object):

    def __init__(self,model,acq_name,bounds,y_max,X,Y,beta_sqrt):

        self.model=model
        self.acq_name = acq_name
        self.bounds=bounds
        self.y_max=y_max
        self.dim = len(self.bounds)
        self.sobol = qmc.Sobol(self.dim, scramble=True)
        self.lb=np.array(self.bounds)[:,0]
        self.ub=np.array(self.bounds)[:,1]
        self.X=X
        self.Y=Y
        self.beta_sqrt = beta_sqrt


    def method_val(self):
        
        if 'mvs' in self.acq_name:
            x_init, acq_max = self.acq_maximize(self.mvs_acq)
            x_return = self.multi_restart_maximize(self.mvs_acq, x_init, acq_max)
            return x_return
        
        else:
            err = "The acquisition function " \
                  "{} has not been implemented, " \
                  "please choose one from the given list".format(acq_name)
            raise NotImplementedError(err)
    
    
    def mvs_acq(self, x):
        _, var = self.model.predict(x)
        return np.sqrt(var)
    
    
    def acq_maximize(self,acq):
        x_tries = self.sobol.random_base2(m=10)
        ys = acq(x_tries)
        x_max = x_tries[np.random.choice(np.where(ys == ys.max())[0])]
        acq_max = ys.max()
        return x_max, acq_max


    # Explore the parameter space more throughly
    def multi_restart_maximize(self, acq_func, x_max, acq_max, seed_num=10):
        x_seeds = self.sobol.random(seed_num)
        for x_try in x_seeds:
            res = minimize(lambda x: -acq_func(x.reshape(1, -1)).squeeze(),\
                    x_try.reshape(1, -1),\
                    bounds=self.bounds,\
                    method="L-BFGS-B")
            if not res.success:
                continue
            if acq_max is None or -res.fun >= acq_max:
                x_max = res.x
                acq_max = -res.fun
        return x_max
    