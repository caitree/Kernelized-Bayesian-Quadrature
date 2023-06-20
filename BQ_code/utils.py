FUNC = {
        'keane' : 'Keane2d',
        'mat_syn1' : 'MAT_Synthetic1d',
        'mat_syn2' : 'MAT_Synthetic2d',
        'mat_syn3' : 'MAT_Synthetic3d',
        'mat_syn4' : 'MAT_Synthetic4d',
        'se_syn1' : 'SE_Synthetic1d',
        'se_syn2' : 'SE_Synthetic2d',
        'se_syn3' : 'SE_Synthetic3d',
        'se_syn4' : 'SE_Synthetic4d',
        'ras' : 'Rastrigin2d',
        'gl' : 'Gramacy_Lee1d',
        'gri1' : 'Griewank1d',
        'gri2' : 'Griewank2d',
        'alp1' : 'Alpine1d',
        'alp2' : 'Alpine2d',
        'ack1' : 'Ackley1d',
        'ack2' : 'Ackley2d',
        'energy' : 'Energy_TS',
        }

ALGO = {
        'mvs-mat' : 'MVS-MAT',
        'mvs-mc-mat': 'MVS-MC-MAT',
        'mvs-se' : 'MVS-SE',
        'mvs-mc-se' : 'MVS-MC-SE',
        'mc' : 'Monte-Carlo',
        }

import math
import numpy as np
from scipy import integrate
from scipy.stats import qmc

def calc_quadrature_trap_single(func, domain, steps):
    dim = domain.shape[0]

    N_per_dim = int(steps ** (1.0 / dim) + 1e-8)
    total_N = N_per_dim ** dim
    mesh_width = np.zeros((dim,))
    grid_per_dim = []
    for d in range(dim):
        grid_per_dim.append(np.linspace(domain[d][0],domain[d][1],N_per_dim))
        mesh_width[d] = grid_per_dim[d][1] - grid_per_dim[d][0]
        
    sample_grid = np.meshgrid(*grid_per_dim, indexing='ij')
    sample_grid = [s.flatten() for s in sample_grid]
    samples = np.stack((tuple(sample_grid))).T
    

    y_vals = func(samples)
    y_vals = y_vals.reshape([N_per_dim]*dim)

    cur_dim_areas = y_vals
    for cur_dim in range(dim):
        cur_dim_areas = (
            mesh_width[cur_dim]
            / 2.0
            * (cur_dim_areas[..., 0:-1] + cur_dim_areas[..., 1:])
        )
        cur_dim_areas = np.sum(cur_dim_areas, axis=dim - cur_dim - 1)
    
    return cur_dim_areas


def calc_quadrature_trap(func, domain, steps, return_his=True):
    dim = domain.shape[0]

    if return_his:
        quad_his = []
        for t in range(2**dim, steps):
            quad_t = calc_quadrature_trap_single(func, domain, t)
            quad_his.append(quad_t)

        return quad_his, None, None

    else:
        return calc_quadrature_trap_single(func, domain, steps)
        


def calc_quadrature_mc_single(func, domain, steps):
    dim = domain.shape[0]
    scales = domain[:, 1] - domain[:, 0]
    volume = np.prod(scales)

    samples = np.zeros((steps, dim))

    for d in range(dim):
        samples[:, d] = np.random.rand(steps)*scales[d] + domain[d,0]

    y_vals = func(samples)

    return volume * np.sum(y_vals) / steps


def calc_quadrature_mc(func, domain, steps, return_his=True):

    # Integral = V / N * sum(func values)
    if return_his:
        quad_his = []
        for t in range(3,steps):
            quad_his.append(calc_quadrature_mc_single(func, domain, t))
            
        return quad_his, None, None
    else:
        return calc_quadrature_mc_single(func, domain, steps)


def calc_quadrature_bq(gp_model, func, domain, steps, init_num=3, split=0.5):
    '''
    Implementation of our Two-Phase BQ Algorithm
    '''
    dim = domain.shape[0]

    if split == 0:
        return calc_quadrature_mc(func, domain, steps)
    else:
        start_t = math.ceil(init_num / split)

        quad_his = []
        mvs_last_t = init_num
        for t in range(start_t, steps):
            t1 = math.ceil(t * split)
            # t1 = math.floor(t * split)
            t2 = t - t1

            quad_t = 0

            # Run MVS t1 iterations
            for _ in range(t1 - mvs_last_t):
                _, _ = gp_model.sample_new_value(learn=False)

            mu_func = lambda x: gp_model.gp.predict_mean_only(x).squeeze()

            # if dim == 1:
            #     mu_quad = calc_quadrature_nquad(mu_func, domain)
            # else:
            mu_quad = calc_quadrature_trap_single(mu_func, domain, steps=10000)

            quad_t += mu_quad

            mvs_last_t = t1


            # Run MC t2 iterations
            if t2 > 0:

                residual_func = lambda x: func(x) - gp_model.gp.predict_mean_only(x).squeeze()

                residual_quad = calc_quadrature_mc_single(residual_func, domain, steps=t2)

                quad_t += residual_quad

            quad_his.append(quad_t)
                
        return quad_his, None, None



def calc_quadrature_nquad(func, domain):
    dim = domain.shape[0]
    def func_wrapper(*args):
        x = []
        for arg in args:
            x.append(arg)
        x = np.array(x)
        x = x.reshape(-1,dim)
        return func(x)
        
    return integrate.nquad(func_wrapper, domain, opts={'epsabs': 1e-3})[0]