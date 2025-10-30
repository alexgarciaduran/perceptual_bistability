# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:49:48 2024

@author: alexg
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib as mpl
import matplotlib.pylab as pl
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from loop_belief_prop_necker import discrete_DBN, Loopy_belief_propagation, dyn_sys_fbp
from mean_field_necker import mean_field_stim, solution_mf_sdo_euler, dyn_sys_mf, backwards
from gibbs_necker import gibbs_samp_necker, return_theta, occ_function_markov
import itertools
import pandas as pd
import seaborn as sns
from pybads import BADS
from scipy.optimize import Bounds
import glob
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from diptest import diptest
from sklearn import linear_model
from sklearn.metrics import r2_score
import warnings
# warnings.filterwarnings("ignore")

THETA = np.array([[0 ,1 ,1 ,0 ,1 ,0 ,0 ,0], [1, 0, 0, 1, 0, 1, 0, 0],
                  [1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1],
                  [0, 0, 1, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 1, 1, 0]])


DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/fitting/data/'  # Alex
SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/fitting/parameters/'  # Alex
COLORMAP = LinearSegmentedColormap.from_list('rg', ['firebrick', 'gainsboro', 'darkgreen'], N=128)

def sigmoid(x):
    return 1/(1+np.exp(-x))


class optimization:
    def __init__(self, data, n_iters, theta=THETA):
        self.theta = theta
        self.n_iters = n_iters
        self.confidence = data['confidence']
        self.coupling = data['coupling']
        self.stim_str = data['stim_str']

    
    def functions_posterior(self, exp_variables, pars, n_iters=10, burn_in=10):
        j, alpha, b1, bias, n_iter_gibbs, sigma = pars
        coupling, stim_str = exp_variables
        j = j*coupling
        b = b1*stim_str+bias
        # post_log_fbp = lambda x: sigmoid(n/alpha * np.arctanh(np.tanh(alpha*j)*np.tanh(np.log(x/(1-x))/n*(n-alpha)+b*alpha)))
        # post_log_mf = lambda x: sigmoid(2*n*j*(2*x-1)+2*b)
        
        vec_time = mean_field_stim(j, self.n_iters, b, theta=self.theta, sigma=sigma)
        # vec_time = solution_mf_sdo_euler(j, b, theta=self.theta, noise=sigma, tau=0.1, time_end=10, dt=1e-3)
        posterior_mf = np.mean(vec_time[-1])
        posterior_mf = np.max((posterior_mf, 1-posterior_mf))
        pos, neg = discrete_DBN(j, b=b, theta=self.theta, num_iter=self.n_iters,
                                thr=1e-6, alpha=alpha)
        posterior_fbp = np.mean(pos[0])
        # posterior_fbp = 10
        pos, neg, _ = Loopy_belief_propagation(j=j, stim=b, theta=self.theta, num_iter=self.n_iters,
                                               thr=1e-6)
        posterior_lbp = np.mean(pos)
        init_state = np.random.choice([-1, 1], self.theta.shape[0])
        states_mat = gibbs_samp_necker(init_state=init_state,
                                       burn_in=burn_in,
                                       n_iter=n_iter_gibbs, j=j,
                                       stim=b, theta=self.theta)
        posterior_gibbs = np.mean((states_mat+1)/2)
        return posterior_mf, posterior_fbp, posterior_lbp, posterior_gibbs


    def mse_computation(self, pars):
        n_trials = len(self.confidence)
        mse_list_mf = []
        mse_list_fbp = []
        mse_list_lbp = []
        mse_list_gibbs = []
        for n in range(n_trials):
            posterior_mf, posterior_fbp, posterior_lbp, posterior_gibbs =\
                self.functions_posterior(exp_variables=[self.coupling[n], self.stim_str[n]],
                                         pars=pars, n_iters=self.n_iters,
                                         burn_in=20)
            # define min err in case the system falls to other side
            min_err = np.min(((self.confidence[n]-posterior_mf)**2,(self.confidence[n]-1+posterior_mf)**2 ))
            mse_list_mf.append(min_err)
            min_err = np.min(((self.confidence[n]-posterior_fbp)**2,(self.confidence[n]-1+posterior_fbp)**2 ))
            mse_list_fbp.append(min_err)
            min_err = np.min(((self.confidence[n]-posterior_lbp)**2,(self.confidence[n]-1+posterior_lbp)**2 ))
            mse_list_lbp.append(min_err)
            min_err = np.min(((self.confidence[n]-posterior_gibbs)**2,(self.confidence[n]-1+posterior_gibbs)**2 ))
            mse_list_gibbs.append(min_err)
        return np.sum(mse_list_mf), np.sum(mse_list_fbp), np.sum(mse_list_lbp), np.sum(mse_list_gibbs)


    def nlh_boltzmann_lbp(self, pars, n=3.92, eps=1e-3, conts=1e-2):
        coupling, stim_str, confidence = self.coupling, self.stim_str, self.confidence
        if len(pars) == 4:
            jpar, b1par, biaspar, noise = pars
            j = jpar*np.array(coupling)
        else:
            jpar, jbiaspar, b1par, biaspar, noise = pars
            j = jpar*np.array(coupling)+jbiaspar
        b = b1par*np.array(stim_str)+biaspar
        unique_j = np.unique(j)
        unique_b = np.unique(b)
        tfconf = (0.5*np.log(confidence.values / (1-confidence.values))-b)/n
        combs = list(itertools.product(unique_j, unique_b))
        # log of e^{-2*V(M)/sigma^2}
        bmann_distro = lambda potential: -np.array(potential)*2 / (noise*noise)
        # function of LBP to be integrated, depends on x (M) and i (trial index)
        pot_lbp = lambda x, i: np.arctanh(np.tanh(j[i])*np.tanh(x*(n-1)+b[i]))-x
        # function of LBP to be integrated to compute Z,
        # depends on x (M) and i (combination index)
        pot_lbp_combs = lambda x, i: np.arctanh(np.tanh(combs[i][0])*np.tanh(x*(n-1)+combs[i][1]))-x
        min_val_integ = 0
        dm = 0.1
        # space to compute norm. cte Z
        m_vals = np.arange(-3, 3+dm, dm)
        # compute normalization constant
        norm_cte_combs = np.zeros(len(combs))
        for i in range(len(combs)):  # for all possible combinations of J&B
            norm_cte_i = []
            for i_m, m in enumerate(m_vals):
                # in positive bc V(q) = int{-F(q)}
                # then Boltzmann is ~ exp(-V) = exp{int{F(q)}}
                norm_cte_i.append(np.exp((scipy.integrate.quad(lambda x: pot_lbp_combs(x, i),
                                                           min_val_integ, m)[0])*2/ (noise*noise)))
            norm_cte_combs[i] = np.sum(norm_cte_i)*dm
        norm_cte = []
        potential_lbp = []
        for i in range(len(confidence)):
            # compute potential at M=vartransform for each trial i
            vartransform = tfconf[i]
            potential_lbp.append(-scipy.integrate.quad(lambda x: pot_lbp(x, i),
                                                           min_val_integ, vartransform)[0])
            # take norm_cte for each trial i
            idx = (np.array(combs) == (j[i], b[i])).all(axis=1)
            norm_cte.append(norm_cte_combs[idx][0])
        boltzman_lbp = bmann_distro(potential_lbp)
        # nlh_lbp = -np.sum(boltzman_lbp-np.log(norm_cte))
        # contaminants (?)
        distro = np.exp(boltzman_lbp)/norm_cte
        nlh_lbp = -np.sum(np.log(distro*(1-eps)+conts*eps))
        # iexp = 20
        # print(nlh_lbp)
        # qv = np.arange(-2, 2, 1e-2)
        # potential_fbp = []
        # for q in qv:
        #     potential_fbp.append(-scipy.integrate.quad(lambda x: pot_lbp(x, iexp),
        #                                                 min_val_integ, q)[0])
        # potential_fbp = np.array(potential_fbp)
        # plt.plot(qv, np.exp(-potential_fbp*2/noise**2)/norm_cte[iexp], label=jpar)
        # plt.axvline(tfconf[iexp], color='r')
        if np.isnan(nlh_lbp):
            print('a')
        return nlh_lbp


    def nlh_boltzmann_fbp(self, pars, n=3.92, eps=1e-3, conts=0.5):
        jpar, b1par, biaspar, noise, alpha = pars
        coupling, stim_str, confidence = self.coupling, self.stim_str, self.confidence
        # for jpar in np.arange(0.1, 2, 0.2):
        j = jpar*np.array(coupling)
        b = b1par*np.array(stim_str)+biaspar
        unique_j = np.unique(j)
        unique_b = np.unique(b)
        tfconf = (0.5*np.log(confidence / (1-confidence))-b)/n
        combs = list(itertools.product(unique_j, unique_b))
        bmann_distro = lambda potential: -np.array(potential)*2 / (noise*noise)
        pot_fbp = lambda x, i: 1/alpha * np.arctanh(np.tanh(alpha*j[i])*np.tanh(x*(n-alpha)+b[i]))-x
        pot_fbp_combs = lambda x, i: 1/alpha * np.arctanh(np.tanh(alpha*combs[i][0])*np.tanh(x*(n-alpha)+combs[i][1]))-x
        min_val_integ = 0
        dm = 0.1
        m_vals = np.arange(-3, 3+dm, dm)
        # compute normalization constant
        norm_cte_combs = np.zeros(len(combs))
        for i in range(len(combs)):  # for all possible combinations of J&B
            norm_cte_i = []
            for i_m, m in enumerate(m_vals):
                # in positive bc V(q) = int{-F(q)}
                # then Boltzmann is exp(-V) = exp{int{F(q)}}
                norm_cte_i.append(np.exp((scipy.integrate.quad(lambda x: pot_fbp_combs(x, i),
                                                           min_val_integ, m)[0])*2/ (noise*noise)))
                
            norm_cte_combs[i] = np.sum(norm_cte_i)
        norm_cte = []
        potential_fbp = []
        for i in range(len(confidence)):
            vartransform = tfconf[i]
            # compute potential at M=vartransform for each trial i
            potential_fbp.append(-scipy.integrate.quad(lambda x: pot_fbp(x, i),
                                                           min_val_integ, vartransform)[0])
            # take norm_cte for each trial i
            idx = (np.array(combs) == (j[i], b[i])).all(axis=1)
            norm_cte.append(norm_cte_combs[idx][0])
        boltzman_fbp = bmann_distro(potential_fbp)
        # nlh_fbp = -np.nansum(boltzman_fbp - np.log(norm_cte))
        distro = np.exp(boltzman_fbp)/norm_cte
        nlh_fbp = -np.nansum(np.log(distro*(1-eps)+conts*eps))
        # iexp = 90
        # print(-(boltzman_fbp - np.log(norm_cte))[iexp])
        # qv = np.arange(-2, 2, 1e-2)
        # potential_fbp = []
        # for q in qv:
        #     potential_fbp.append(-scipy.integrate.quad(lambda x: pot_fbp(x, iexp),
        #                                                 min_val_integ, q)[0])
        # potential_fbp = np.array(potential_fbp)
        # plt.plot(qv, np.exp(-potential_fbp*2/noise**2)/norm_cte[iexp], label=round(alpha, 2))
        # plt.axvline(tfconf[iexp], color='r')
        # plt.legend(title='alpha')
        # plt.ylabel('Boltzmann distro')
        # plt.xlabel('Log-belief ratio')
        return nlh_fbp


    def nlh_boltzmann_mf(self, pars, n=3.92, eps=1e-3, conts_distro=1e-2,
                         penalization_nan=0, dq=1e-2):
        coupling, stim_str, confidence = self.coupling, self.stim_str, self.confidence
        if len(pars) == 4:
            jpar, b1par, biaspar, noise = pars
            j = jpar*np.array(coupling)
        else:
            jpar, jbiaspar, b1par, biaspar, noise = pars
            j = jpar*np.array(coupling)+jbiaspar
        b = b1par*np.array(stim_str)+biaspar

        q = np.arange(0, 1, dq)  # Define q outside the loop, shape (100,)

        # Reshape j and b to broadcast correctly over q
        j = np.array(j).reshape(-1, 1)  # Reshape j to shape (500, 1)
        b = np.array(b).reshape(-1, 1)  # Reshape b to shape (500, 1)

        # Vectorized version of the potential function over all i and q
        exp_term = 2 * n * (j * (2 * q - 1)) + 2 * b  # Shape: (500, 100)
        log_term = np.log(1 + np.exp(exp_term))  # Shape: (500, 100)

        # Vectorized potential (pot_mf_i) for all i and all q
        # pot_mf = (q*q) / 2 - log_term / (4* n * j)  # Shape: (500, 100)
        pot_mf = np.where(j > 5e-2, (q*q) / 2 - log_term / (4* n * j),
                          q*q/2 - q*sigmoid(2*b))

        # Apply Boltzmann distribution function over the potential values (vectorized)
        bmann_values = np.exp(-2 * pot_mf / (noise*noise))  # Shape: (500, 100)

        # Sum over q for each i to get the normalization constant, norm_cte
        norm_cte = np.sum(bmann_values, axis=1)  # Shape: (500,)

        j = np.array(j).reshape(-1)  # Reshape j to shape (500, 1)
        b = np.array(b).reshape(-1)  # Reshape b to shape (500, 1)
        pot_mf_fun = lambda q: np.where(
                            j > 5e-2,
                            q*q/2 - np.log(1+np.exp(2*n*(j*(2*q-1))+b*2))/(4*n*j), 
                            q*q/2 - q*sigmoid(2*b))
        bmann_distro_log = lambda potential: -2*np.array(potential) / (noise*noise)
        log_likelihood = np.log((1-eps)*np.exp(bmann_distro_log(pot_mf_fun(confidence)))/norm_cte + eps*conts_distro)
        log_likelihood[np.isnan(log_likelihood)] = -penalization_nan
        nlh_mf = -np.sum(log_likelihood)
        # nlh_mf = -np.nansum(bmann_distro_log(pot_mf_fun(confidence)) - np.log(norm_cte))
        return np.max((nlh_mf, 0))  #  - log_prior(pars)*len(j)


    def nlh_gibbs(self, pars, n=3.92, eps=1e-3, conts_distro=1e-2, nan_penalty=1e-5):
        jpar, b1par, biaspar, time = pars
        coupling, stim_str, confidence = self.coupling, self.stim_str, self.confidence
        idx_c = coupling == 1
        stim_str = stim_str[idx_c]
        coupling = coupling[idx_c]
        confidence = confidence[idx_c]
        j = jpar*np.array(coupling)
        b = b1par*np.array(stim_str)+biaspar
        likelihood = []
        k_1 = 12*j + 8*b
        k_2 = 6*j + 6*b
        k_3 = 4*j + 4*b
        k_4 = 2*j + 2*b
        k_5 = 12*j - 8*b
        k_6 = 6*j - 6*b
        k_7 = 4*j - 4*b
        k_8 = 2*j - 2*b
        lamb = sigmoid(k_1-k_2)*sigmoid(k_2-k_3)*sigmoid(k_3-k_4)
        nu = sigmoid(k_5-k_6)*sigmoid(k_6-k_7)*sigmoid(k_7-k_8)
        val = occ_function_markov(lamb, nu, time, confidence*time)
        time_vals = np.arange(0, time+1, 1)
        norm_cte = np.array([np.nansum(occ_function_markov(nu[i], lamb[i], time, time_vals)) for i in range(len(confidence))])
        lh = val / norm_cte
        lh[np.isnan(norm_cte) + np.isnan(val) + (norm_cte == 0) + np.isnan(lh)] = nan_penalty
        # lh += 1e-10
        likelihood = np.array(lh)
        return -np.sum(np.log(likelihood*(1-eps) + eps*conts_distro))


    def mcmc(self, initial_params, iterations=10000, proposal_cov=None):
        """
        Multivariate Metropolis-Hastings sampler for (J1, B0, B1, sigma)
        """
        num_params = len(initial_params)
        samples = np.zeros((iterations, num_params))
        current_params = np.array(initial_params)

        # Default proposal covariance (diagonal)
        if proposal_cov is None:
            proposal_cov = np.diag([0.001, 0.001, 0.001, 0.0001])  # Tuneable step sizes

        for i in range(iterations):
            proposed_params = np.random.multivariate_normal(current_params, proposal_cov)

            # Ensure sigma is positive
            if proposed_params[3] <= 0:
                continue  # Reject negative sigma values

            # Compute Metropolis acceptance ratio
            log_p_current = -self.nlh_boltzmann_mf(current_params)
            log_p_proposed = -self.nlh_boltzmann_mf(proposed_params)
            acceptance_ratio = np.exp(log_p_proposed - log_p_current)

            if np.random.rand() < acceptance_ratio:
                current_params = proposed_params

            samples[i] = current_params

        return samples


    def optimize_nlh(self, x0, model='MF', method='nelder-mead'):
        # effective_n = np.max(np.linalg.eigvals(self.theta))
        assert model in ['MF', 'MF5', 'LBP', 'LBP5', 'FBP', 'GS'], 'Model should be either GS, MF, MF5, LBP or FBP'
        if model == 'MF':
            fun = self.nlh_boltzmann_mf
            assert len(x0) == 4, 'x0 should have 4 values (J, B1, bias, noise)'
            if method != 'BADS':
                bounds = Bounds([0., -0.2, -.7, 0.1], [2, 2, .8, 0.6])
            if method == 'BADS':
                lb = [0., -0.2, -0.8, 0.01]
                ub = [1.3, 2, 0.8, 0.5]
                plb = [0.1, 0.1, -0.6, 0.08]
                pub = [1.1, 0.9, 0.6, 0.25]
        if model == 'MF5':
            fun = self.nlh_boltzmann_mf
            assert len(x0) == 5, 'x0 should have 5 values (J1, Jbias, B1, bias, noise)'
            if method != 'BADS':
                bounds = Bounds([0., -0.4, -0.2, -.7, 0.1], [2, 0.6, 2, .8, 0.6])
            if method == 'BADS':
                lb = [0, 0., -0.2, -0.8, 0.01]
                ub = [2., 1.5, 2, 0.8, 0.5]
                plb = [0.1, 0.1, 0.1, -0.6, 0.08]
                pub = [1.4, 1.3, 0.9, 0.6, 0.25]
        if model == 'GS':
            fun = self.nlh_gibbs
            assert len(x0) == 4, 'x0 should have 4 values (J, B1, bias, time_end)'
            if method != 'BADS':
                bounds = Bounds([0., -0.2, -.7, 5], [2, 2, .8, 1e5])
            if method == 'BADS':
                lb = [0.01, -0.2, -0.8, 1]
                ub = [2., 2, 0.8, 1e5]
                plb = [0.18, 0.1, -0.6, 10]
                pub = [1.2, 0.9, 0.6, 1e4]
        if model == 'LBP':
            fun = self.nlh_boltzmann_lbp
            assert len(x0) == 4, 'x0 should have 4 values (J, B1, bias, noise)'
            if method != 'BADS':
                bounds = Bounds([1e-1, -.5, -.5, 0.05], [3, .5, .5, 0.3])
            if method == 'BADS':
                lb = [0., -0.2, -0.8, 0.01]
                ub = [1.3, 2, 0.8, 0.25]
                plb = [0.1, 0.1, -0.6, 0.08]
                pub = [1.1, 0.9, 0.6, 0.20]
        if model == 'LBP5':
            fun = self.nlh_boltzmann_lbp
            assert len(x0) == 5, 'x0 should have 5 values (J1, J0, B1, bias, noise)'
            if method != 'BADS':
                bounds = Bounds([1e-1, -.5, -.5, 0.05], [3, .5, .5, 0.3])
            if method == 'BADS':
                lb = [0, 0., -0.2, -0.8, 0.01]
                ub = [2., 1.5, 1.5, 0.8, 0.25]
                plb = [0.1, 0.1, 0.1, -0.6, 0.08]
                pub = [1.4, 1.3, 0.9, 0.6, 0.20]
        if model == 'FBP':
            fun = self.nlh_boltzmann_fbp
            assert len(x0) == 5, 'x0 should have 5 values (J, B1, bias, noise, alpha)'
            if method != 'BADS':
                bounds = Bounds([1e-1, -.1, -.1, 0.05, 0], [2., .4, .4, 0.3, 1.5])
            if method == 'BADS':
                lb = [0.01, -.2, -.8, 0.05, 0.]
                ub = [2., 2., 0.8, 0.6, 2]
                plb = [0.2, 0., -0.3, 0.1, 0.6]
                pub = [1.4, 0.6, 0.3, 0.3, 1.4]
        if method != 'BADS':
            optimizer_0 = scipy.optimize.minimize(fun, x0, method=method,
                                                  bounds=bounds)
        if method == 'BADS':
            print('BADS')
            # constraint = lambda x: np.abs(x[:, 1]+x[:, 2]) > 0.6
            optimizer_0 = BADS(fun, x0, lb, ub, plb, pub).optimize()  # non_box_cons=constraint
            print(optimizer_0.x)
        # optimizer_0 = scipy.optimize.minimize(fun, x0, method='trust-constr', bounds=bounds)
        # optimizer_0 = scipy.optimize.minimize(fun, x0, method='BFGS', bounds=bounds)
        # optimizer_0 = scipy.optimize.minimize(fun, x0, method='COBYLA', bounds=bounds)
        pars = optimizer_0.x
        if method == 'BADS':
            if optimizer_0.fval == 0:
                pars = self.optimize_nlh(x0=x0, model=model, method=method)
        # if (np.abs(pars - lb) < 1e-3).any() or (np.abs(pars - ub) < 1e-3).any():
        #     pars = self.optimize_nlh(x0=x0, model=model, method=method)
        return pars


    def mse_minimization(self):
        # define param grid
        j = np.arange(0.2, 1, 0.2)
        # alpha = np.arange(0.1, 1.4, 2e-1)
        alpha = [1]  # [0.2, 1, 1.3]
        b1 = np.arange(0, 0.25, 1e-1)
        bias = np.arange(0, 0.25, 1e-1)
        num_iters_gibbs = [50]  # np.logspace(0, 3, 1)
        combinations = list(itertools.product(j, alpha, b1, bias, num_iters_gibbs))
        mse_list_mf = []
        mse_list_fbp = []
        mse_list_lbp = []
        mse_list_gibbs = []
        n_combs = len(combinations)
        for i_par, pars in enumerate(combinations):
            if i_par > 0 and i_par % 10 == 0:
                print(str(round(i_par / n_combs*100, 2)) + ' %')
            msemf, msefbp, mselbp, msegibbs =\
                self.mse_computation(pars)
            mse_list_mf.append(msemf)
            mse_list_fbp.append(msefbp)
            mse_list_lbp.append(mselbp)
            mse_list_gibbs.append(msegibbs)
        self.mse_list_mf = mse_list_mf
        self.mse_list_fbp = mse_list_fbp
        self.mse_list_lbp = mse_list_lbp
        self.mse_list_gibbs= mse_list_gibbs
        self.combinations = combinations


def simulate_data(data, pars, opt, n_iters):
    n_trials = len(data['confidence'])
    coupling = data['coupling']
    stim_str = data['stim_str']
    confidence_mf = []
    confidence_fbp = []
    confidence_lbp = []
    confidence_gibbs = []
    for n in range(n_trials):
        posterior_mf, posterior_fbp, posterior_lbp, posterior_gibbs =\
            opt.functions_posterior(exp_variables=[data['coupling'][n], data['stim_str'][n]],
                                    pars=pars, n_iters=n_iters,
                                    burn_in=10)
        confidence_mf.append(posterior_mf)
        confidence_fbp.append(posterior_fbp)
        confidence_lbp.append(posterior_lbp)
        confidence_gibbs.append(posterior_gibbs)
    data_sim = {'stim_str': stim_str, 'coupling': coupling,
                'confidence_mf': confidence_mf, 'confidence_fbp': confidence_fbp,
                'confidence_lbp': confidence_lbp, 'confidence_gibbs': confidence_gibbs}
    return pd.DataFrame(data_sim)


def load_data(data_folder, n_participants=1, sigmoid_fit=False):
    files = glob.glob(data_folder + '*')
    if type(n_participants) != str and n_participants == 1:
        f = files[:n_participants]
        return pd.read_csv(f[0])
    if type(n_participants) != str and n_participants > 1:
        f = files[:n_participants]
        df_0 = pd.DataFrame()
        for i in range(len(f)):
            df = pd.read_csv(f[i])
            df['subject'] = 's_' + str(i+1)
            df_0 = pd.concat((df_0, df))
        return df_0
    if type(n_participants) == str:
        df_0 = pd.DataFrame()
        for i in range(len(files)):
            df = pd.read_csv(files[i])
            df = df.dropna(subset=['confidence'])
            df['subject'] = 's_' + str(i+1)
            # df['confidence'] = sigmoid(scipy.stats.zscore(df.confidence.values)*2)*2-1
            if sigmoid_fit:
                fun = lambda cp, w0, w1, w2: sigmoid(w0 + w1*cp + w2*cp**3)  # cp = 2*c-1
                df = df.dropna(subset=['confidence'])
                y = (df.confidence.values+1)/2
                x = df.evidence.values
                popt = scipy.optimize.curve_fit(f=fun, xdata=x, ydata=y)[0]
                conf_new = fun(x, *popt)
                df['confidence'] = (conf_new-0.5)*2
            df_0 = pd.concat((df_0, df))
        return df_0


def transform(x, minval=0.5, maxval=0.9999):
    maxarray = np.nanmax(x)
    minarray = np.nanmin(x)
    return (maxval-minval)/(maxarray-minarray)*(x-minarray) + minval


def fit_data(optimizer, plot=True, model='MF', n_iters=200, method='nelder-mead',
             sub=None, rec=None):
    if model in ['FBP', 'MF5', 'LBP5']:
        numpars = 5
    else:
        numpars = 4
    pars_array = np.empty((n_iters, numpars))
    pars_array[:] = np.nan
    pars_array_0 = np.empty((n_iters, numpars))
    pars_array_0[:] = np.nan
    if method == 'BADS':
        appendix = '_BADS'
    else:
        appendix = ''
    for k in range(n_iters):
        if k % 2 == 0:
            print(str(round(k/n_iters*100, 2)) + ' %')
        j0 = np.random.uniform(0.2, 0.6)
        b10 = np.random.uniform(0.4, 0.6)
        bias0 = np.random.uniform(0, 0.3)
        noise0 = np.random.uniform(0.12, 0.2)
        if model == 'FBP':
            alpha0 = np.random.uniform(0.1, 1.4)
            x0 = [j0, b10, bias0, noise0, alpha0]
        if model == 'GS':
            time = 10**(np.random.uniform(2, 4))
            x0 = [j0, b10, bias0, time]
        if model in ['LBP', 'MF']:
            x0 = [j0, b10, bias0, noise0]
        if model == 'LBP5':
            jbias0 = np.random.uniform(0.1, 0.2)
            x0 = [j0, jbias0, b10, bias0, noise0]
        if model == 'MF5':
            jbias0 = np.random.uniform(0.1, 0.2)
            if sub is None:
                x0 = [j0, jbias0, b10, bias0, noise0]
            else:
                if model == 'MF5':
                    params_i = np.load(SV_FOLDER + 'parameters_' + 'MF' + appendix + sub +  'null' + '.npy')
                if model == 'LBP5':
                    params_i = np.load(SV_FOLDER + 'parameters_' + 'LBP' + appendix + sub +  'null' + '.npy')
                x0 = [j0, params_i[0], params_i[1], params_i[2], params_i[3]]
        if rec is not None:
            x0 = np.load(SV_FOLDER + 'param_recovery/pars_prt' + str(rec) + model + '.npy') + np.random.randn(numpars)*0.02
        pars_array_0[k, :] = x0
        # df_simul = simulate_data(data_orig, x0, optimizer, 50)
        # optimizer.confidence = df_simul.confidence
        # optimizer.coupling = df_simul.coupling
        # optimizer.stim_str = df_simul.stim_str
        sols_0 = optimizer.optimize_nlh(np.array(x0), model=model, method=method)
        pars_array[k, :] = sols_0
    if plot:
        fig, ax = plt.subplots(ncols=numpars)
        xlims = [[0., 3], [0, 1], [0, 1], [0., 0.3], [0, 2]]
        for i_a, a in enumerate(ax):
            a.plot(pars_array_0[:, i_a], pars_array[:, i_a], color='k', marker='o', linestyle='', markersize=3)
            a.set_title('corr = ' + str(round(np.corrcoef(pars_array_0[:, i_a], pars_array[:, i_a])[0][1], 2)))
            a.set_xlim(xlims[i_a])
        fig, ax = plt.subplots(ncols=numpars, figsize=(14, 4))
        if model == 'FBP':
            titles = ['coupling J', 'stim weight', 'bias', 'noise', 'alpha']
        if model == 'GS':
            titles = ['coupling J', 'stim weight', 'bias', 'noise', 'time']
        if model == 'LBP' or model == 'MF':
            titles = ['coupling J', 'stim weight', 'bias', 'noise']
        for i_a, a in enumerate(ax):
            a.set_title(titles[i_a])
            a.plot(pars_array[:, i_a], np.repeat(0, n_iters)+np.random.randn(n_iters)*0.15,
                   color='k', marker='o', markersize=3, linestyle='', alpha=0.6)
            sns.violinplot(x=pars_array[:, i_a], ax=ax[i_a], color='darksalmon',
                           warn_singular=False)  # linewidth=0
            a.axvline(np.median(pars_array[:, i_a]),
                      color='k', alpha=0.5, linestyle='--')
            # a.set_xlim(xlims[i_a])
        plt.pause(0.01)
    return pars_array


def return_and_plot_simul_data(data, params, optimizer, plot=True, model='MF'):
    pars = [params[0], 1, params[1], params[2], 50, params[3]]
    df_simul = simulate_data(data, pars, optimizer, 50)
    if model == 'MF':
        appendix = '_mf'
    if model == 'LBP':
        appendix = '_lbp'
    if model == 'FBP':
        appendix = '_fbp'
    if model == 'GS':
        appendix = '_gibbs'
    if plot:
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(10, 4))
        ax = ax.flatten()
        sns.lineplot(df_simul, x='coupling', y='confidence'+appendix, hue='stim_str', ax=ax[0])
        sns.lineplot(df_simul, x='stim_str', y='confidence'+appendix, hue='coupling', ax=ax[1])
        df = pd.DataFrame(data)
        ax[2].plot(df.confidence, df_simul['confidence'+appendix], marker='o', linestyle='',
                   color='k', markersize=1)
        ax[2].set_title(model + ', corr = ' +
                        str(round(np.corrcoef(df.confidence,
                                              df_simul['confidence'+appendix])[0][1], 2)))
        ax[2].plot([0.4, 1], [0.4, 1], color='k', alpha=0.2)
        ax[2].set_xlabel('True confidence')
        ax[2].set_ylabel('Model confidence')
        fig.tight_layout()
        confidence_mf = df_simul['confidence'+appendix].values
        df_simul_mf = df_simul[['coupling', 'stim_str']]
        df_simul_mf['confidence'] = confidence_mf
        df_simul_mf['Type'] = 'Model'
        data['Type'] = 'Data'
        df_all = pd.concat((data, pd.DataFrame(df_simul_mf)))
        fig2, ax2 = plt.subplots(ncols=2)
        sns.lineplot(df_all, x='coupling', y='confidence', hue='stim_str', style='Type', ax=ax2[0],
                     estimator='mean')
        sns.lineplot(df_all, x='stim_str', y='confidence', hue='coupling', style='Type', ax=ax2[1],
                     estimator='mean')
        for a in ax2:
            a.set_ylim(0.4, 1.05)
    return df_simul


def simulate_FBP(pars, n_iters,
                 stimulus, coupling, sv_folder=SV_FOLDER,
                 n_iter=0, model='FBP', recovery=True, resimulate=False, extra='',
                 n=3.92):
    vals_conf = []
    decision = []
    if model == 'FBP':
        jpar, b1par, biaspar, noise, alpha = pars
    if model == 'LBP':
        jpar, b1par, biaspar, noise = pars
        alpha = 1
    if model == 'LBP5':
        jpar, jbiaspar, b1par, biaspar, noise = pars
        alpha = 1
    if model == 'MF':
        jpar, b1par, biaspar, noise = pars
    if model == 'MF5':
        jpar, jbiaspar, b1par, biaspar, noise = pars
    b = stimulus*b1par + biaspar
    j = coupling*jpar
    if model == 'MF5' or model == 'LBP5':
        j += jbiaspar
    if recovery:
        folder = 'param_recovery'
    else:
        folder = 'simulated_data'
    pathdata = sv_folder + folder + '/df_simul' + str(n_iter) + model +  extra + '.csv'
    os.makedirs(os.path.dirname(pathdata), exist_ok=True)
    if os.path.exists(pathdata) and not resimulate:
        data = pd.read_csv(pathdata)
    else:
        if model in ['FBP', 'LBP', 'LBP5']:
            for i in range(len(stimulus)):
                # pos, neg = discrete_DBN(j[i], b=b[i], theta=theta, num_iter=n_iters,
                #                         thr=1e-6, alpha=alpha)
                # logmess = np.random.randn()/10
                logmess = 0
                # lm = []
                for _ in range(n_iters):
                    logmess = dyn_sys_fbp(logmess, j[i], b[i], alpha=alpha, n=n, dt=1e-2, noise=noise)
                    # lm.append(logmess)
                posterior_fbp = sigmoid(2*(n*logmess+b[i]))
                # posterior_fbp = np.max((posterior_fbp, 1-posterior_fbp))
                vals_conf.append(posterior_fbp)
                decision.append(np.sign(posterior_fbp-0.5))
        else:
            for i in range(len(stimulus)):
                # q = np.random.rand()
                q = 0.5
                for _ in range(n_iters):
                    q = dyn_sys_mf(q, dt=1e-2, j=j[i], bias=b[i], n=3.92, sigma=noise,
                                   tau=0.2)
                q_final = q  #  if np.sign(q-0.5) > 0 else 1-q
                vals_conf.append(q_final)
                decision.append(np.sign(q-0.5))
        data = pd.DataFrame({'stim_str': stimulus, 'coupling': coupling,
                             'confidence': vals_conf, 'decision': decision,
                             'stim_ev_cong': decision*stimulus})
        data.to_csv(sv_folder + folder + '/df_simul' + str(n_iter) + model + extra + '.csv')
    return data


def save_params_recovery(n_pars=50, sv_folder=SV_FOLDER,
                         i_ini=0, model='MF5'):
    for i in range(i_ini, n_pars):
        if model in ['LBP', 'FBP']:
            j0 = np.random.uniform(0.3, 1.6)
            b10 = np.random.uniform(0.05, 0.3)
            bias0 = np.random.uniform(0.05, 0.3)
            noise0 = np.random.uniform(0.1, 0.2)
            alpha0 = np.random.uniform(0.3, 1.4)
            params = [j0, b10, bias0, noise0, alpha0]
            np.save(sv_folder + 'param_recovery/pars_prt' + str(i) + '.npy',
                    np.array(params))
        else:
            j0 = np.random.uniform(0.15, 1.2)
            b10 = np.random.uniform(0.15, 0.8)
            bias0 = np.random.uniform(0.05, 0.4)
            noise0 = np.random.uniform(0.15, 0.3)
            if model == 'MF5':
                jbias0 = np.random.uniform(0.2, 0.8)
                params = [j0, jbias0, b10, bias0, noise0]
            else:
                params = [j0, b10, bias0, noise0]
            np.save(sv_folder + 'param_recovery/pars_prt' + str(i) + model + '.npy',
                    np.array(params))


def simulate_subjects(sv_folder=SV_FOLDER,
                      model='MF', resimulate=True, extra='',
                      mcmc=True, method='BADS', data_augment=False,
                      sub=None, plot_subs=False):
    if model in ['MF5', 'FBP', 'LBP5']:
        numpars = 5
    else:
        numpars = 4
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    if sub is not None:
        all_df = all_df.loc[all_df.subject == sub]
    subjects = all_df.subject.unique()
    accuracies = []
    accuracies_model = []
    pright_model = []
    parmat = np.zeros((len(subjects), numpars))
    conf_model = []
    if mcmc:
        path = SV_FOLDER + extra + 'MCMC_fitted_MF_parameters.npy'
        params = np.load(path)
        extra += 'mcmc'
    for i_s, sub in enumerate(subjects):
        print(sub)
        dataframe = all_df.copy().loc[all_df['subject'] == sub]
        accuracies.append(sum(dataframe.response == dataframe.side)/len(dataframe.response))
        # for sub in dataframe.subject.unique():
        #     data = dataframe[['pShuffle', 'confidence', 'evidence']]
        unique_vals = np.sort(dataframe['pShuffle'].unique())
        if extra != 'null':
            dataframe['coupling'] = dataframe['pShuffle'].replace(to_replace=unique_vals,
                                        value= [1., 0.3, 0.])
        else:
            dataframe['coupling'] = 1
        dataframe['confidence'] = (transform(dataframe.confidence.values, -0.999, 0.999)+1)/2
        dataframe['stim_str'] = (dataframe.evidence.values)
        # dataframe['confidence'] = transform(np.abs(dataframe.confidence.values))
        # dataframe['stim_str'] = np.abs(dataframe.evidence.values)
        dataframe['stim_ev_cong'] = dataframe.response.values*dataframe.stim_str.values
        data = dataframe[['coupling', 'confidence', 'stim_str', 'stim_ev_cong', 'response']]
        if data_augment:
            data = data_augmentation(data, sigma=0.05, times_augm=4,
                                     minval=0.001)
            extra += '_data_augment_'
        print(len(data))
        if method == 'BADS':
            appendix = '_BADS'
        else:
            appendix = ''
        if not mcmc:
            pars = np.load(sv_folder + '/parameters_' + model + appendix + sub + extra + '.npy')
        else:
            pars = params[:, i_s]
        print(pars)
        parmat[i_s, :] = pars
        df = simulate_FBP(pars, 400,
                          data.stim_str.values, data.coupling.values,
                          sv_folder=SV_FOLDER, n_iter=sub, model=model,
                          recovery=False, resimulate=resimulate, extra=extra)
        accuracies_model.append(np.sum(df.decision.values == np.array(dataframe.side.values, dtype=np.float32))/len(dataframe.response))
        pright_model.append(np.sum((df.decision.values+1)/2)/len(dataframe.response))
        conf_model.append(np.nanmean(np.abs(df.confidence.values-1/2)*2))
        df['abs_confidence'] = np.abs(df.confidence-0.5)*2
        data['abs_confidence'] = np.abs(data.confidence-0.5)*2
        df['abs_stim'] = np.abs(df.stim_str)
        data['abs_stim'] = np.abs(data.stim_str)
        df['decision'] = (df['decision']+1)/2
        data['response'] = (data['response']+1)/2
        if plot_subs:
            fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(10, 13))
            ax = ax.flatten()
            cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)
            if 'null' in extra:
                data['coupling'] = dataframe['pShuffle'].replace(to_replace=unique_vals,
                                                                      value= [1., 0.3, 0.]) 
                df['coupling'] = data['coupling']
            sns.lineplot(df, x='abs_stim', y='abs_confidence', hue='coupling', ax=ax[1],
                          palette=cmap)
            sns.lineplot(data, x='abs_stim', y='abs_confidence', hue='coupling', ax=ax[0],
                          palette=cmap)
            sns.lineplot(df, x='stim_ev_cong', y='abs_confidence', hue='coupling', ax=ax[3],
                          palette=cmap)
            sns.lineplot(data, x='stim_ev_cong', y='abs_confidence', hue='coupling', ax=ax[2],
                          palette=cmap)
            sns.lineplot(df, x='stim_str', y='decision', hue='coupling', ax=ax[5],
                          palette=cmap, marker='o')
            sns.lineplot(data, x='stim_str', y='response', hue='coupling', ax=ax[4],
                          palette=cmap, marker='o')
            ax[4].set_ylabel('p(rightward)')
            ax[5].set_ylabel('p(rightward)')
            ax[4].set_xlabel('stimulus evidence')
            ax[5].set_xlabel('stimulus evidence')
            labs = [sub, 'MF', sub, 'MF', sub, 'MF', sub, 'MF']
            for i_a, a in enumerate(ax):
                a.set_title(labs[i_a])
                a.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(SV_FOLDER + 'ind_simuls/sims_'+sub+extra+'_'+ model+'.png', dpi=100, bbox_inches='tight')
            fig.savefig(SV_FOLDER + 'ind_simuls/sims_'+sub+extra+'_'+ model+'.svg', dpi=100, bbox_inches='tight')
            plt.close(fig)
    fig2, ax2 = plt.subplots(ncols=numpars, nrows=3, figsize=(16, 14))
    ax2 = ax2.flatten()
    labels = ['Coupling, J', 'Stimulus weight, B1', 'Bias, B0', 'noise', 'Alpha'][:numpars]
    if model == 'MF5' or model == 'LBP5':
        labels = ['Coupling slope, J1', 'Coupling bias, J0', 'Stimulus weight, B1', 'Bias, B0', 'noise']
    for i_a, a in enumerate(ax2):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        if i_a < numpars:
            var = pright_model
            lab = 'p(right)'
        if i_a >= numpars < 2*numpars:
            var = accuracies_model
            lab = 'p(correct)'
        if i_a >= 2*numpars:
            var = conf_model
            lab = 'Confidence'
        # corr = np.corrcoef(var, parmat[:, i_a % numpars])[0, 1]
        corr = scipy.stats.pearsonr(var, parmat[:, i_a % numpars])
        if i_a == (2*numpars-1):
            cmap = mpl.cm.Oranges
            if model == 'MF5':
                c_index = 2
            else:
                c_index = 1
            norm = mpl.colors.Normalize(vmin=0, vmax=np.max(parmat[:, c_index]))
            color = cmap(norm(parmat[:, c_index]))
            a.scatter(parmat[:, i_a % numpars], var, c=color, marker='o')
        else:
            color = 'k'
            a.plot(parmat[:, i_a % numpars], var, color=color, marker='o', linestyle='')
        a.set_ylabel(lab)
        a.set_xlabel(labels[i_a % numpars])
        a.set_title(rf'$\rho =$ {round(corr.statistic, 3)}, p={corr.pvalue:.1e}')
    fig2.tight_layout()
    fig2.savefig(SV_FOLDER + extra + model +'correlations_fitted_parameters_with_model.png', dpi=200, bbox_inches='tight')
    fig2.savefig(SV_FOLDER + extra + model +'correlations_fitted_parameters_with_model.svg', dpi=200, bbox_inches='tight')


def parameter_recovery(n_pars=50, sv_folder=SV_FOLDER,
                       theta=THETA, n_iters=2000, n_trials=5000,
                       i_ini=0, model='FBP', method='BADS'):
    # coupling = np.repeat(coupling.values, 3)
    # np.random.shuffle(coupling)
    # stimulus = np.repeat(stimulus.values, 3)
    coupling_values = [0., 0.3, 1]
    stimulus_values = [-1, -0.8, -0.4, 0., 0.4, 0.8, 1]
    coupling = np.random.choice(coupling_values, n_trials)
    stimulus = np.random.choice(stimulus_values, n_trials)
    for i in range(i_ini, n_pars):
        if model in ['MF5', 'MF']:
            pars = np.load(sv_folder + 'param_recovery/pars_prt' + str(i) + model + '.npy')
        else:
            pars = np.load(sv_folder + 'param_recovery/pars_prt' + str(i) + '.npy')
        print(pars)
        df = simulate_FBP(pars=pars, n_iters=n_iters,
                          stimulus=stimulus, coupling=coupling,
                          sv_folder=SV_FOLDER, n_iter=i,
                          model=model, resimulate=True)
        optimizer = optimization(data=df, n_iters=50, theta=theta)
        pars_array = fit_data(optimizer, model=model, n_iters=1, method=method,
                              plot=False, rec=i)
        print(pars_array)
        if method == 'BADS':
            method1 = ''
        else:
            method1 = method
        np.save(sv_folder + 'param_recovery/pars_prt_recovered' + str(i) + model + method1 + '.npy',
                np.array(pars_array))


def data_augmentation(df, times_augm=10, sigma=0.05, minval=0., maxval=0.9999):
    df_copy = df.copy()
    newdf = pd.DataFrame(np.repeat(df.values, times_augm, axis=0))
    newdf.columns = df.columns
    confidence_new = newdf.confidence.values[len(df_copy):] + np.random.randn(len(newdf)-len(df_copy))*sigma
    confidence = np.concatenate((newdf.confidence.values[:len(df_copy)], confidence_new))
    # confidence = np.random.choice(df_copy.confidence.values, len(df_copy)*(times_augm+1))
    confidence = np.clip(confidence, minval, maxval)
    newdf['confidence'] = confidence
    return newdf


def plot_parameter_recovery(sv_folder=SV_FOLDER, n_pars=50, model='FBP', method='BADS'):
    if model in ['LBP', 'GS', 'MF']:
        numpars = 4
    else:
        numpars = 5
    if method == 'BADS':
        method = ''
    orig_params = np.zeros((n_pars, numpars))
    recovered_params = np.zeros((n_pars, numpars))
    for i in range(n_pars):
        # params_recovered = np.load(sv_folder + 'param_recovery/version_2/version_2_10000/pars_prt_recovered' + str(i) + model + method + '.npy')
        # params_original = np.load(sv_folder + 'param_recovery/version_2/version_2_10000/pars_prt' + str(i) + '.npy')
        params_original = np.load(sv_folder + 'param_recovery/pars_prt' + str(i) + model + '.npy')
        params_recovered = np.load(sv_folder + 'param_recovery/pars_prt_recovered' + str(i) + model + '.npy')
        orig_params[i] = params_original[:numpars]
        recovered_params[i] = params_recovered
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(15, 9))
    ax = ax.flatten()
    if model in ['LBP', 'FBP']:
        labels = ['Coupling, J', 'Stimulus weight, B1', 'Bias, B0', 'noise', 'Alpha'][:numpars]
        xylims = [[0, 3], [0, 0.5], [0, 0.5], [0, 0.3], [0, 2]]
    if model == 'MF':
        labels = ['Coupling, J', 'Stimulus weight, B1', 'Bias, B0', 'noise']
        xylims = [[0, 3], [0, 0.5], [0, 0.5], [0, 0.3], [0, 2]]
    if model == 'MF5':
        labels = ['Coupling, J1', 'Coupling bias, J0', 'Stimulus weight, B1', 'Bias, B0', 'noise']
        xylims = [[0, 3], [0, 0.8], [0, 0.7], [0, 0.5], [0, 0.5]]
    for i_a in range(numpars):
        a = ax[i_a]
        a.plot(orig_params[:, i_a], recovered_params[:, i_a], color='k', marker='o',
               markersize=5, linestyle='')
        a.plot(xylims[i_a], xylims[i_a], color='k', alpha=0.3)
        a.set_title(labels[i_a])
        a.set_xlabel('Original parameters')
        a.set_ylabel('Recovered parameters')
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    ax[-1].axis('off')
    if model == 'LBP':
        ax[-2].axis('off')
    fig.tight_layout()
    fig2, ax2 = plt.subplots(ncols=2)
    ax2, ax = ax2
    # define correlation matrix
    corr_mat = np.empty((numpars, numpars))
    corr_mat[:] = np.nan
    for i in range(numpars):
        for j in range(numpars):
            # compute cross-correlation matrix
            corr_mat[i, j] = np.corrcoef(orig_params[:, i], recovered_params[:, j])[1][0]
    # plot cross-correlation matrix
    im = ax.imshow(corr_mat.T, cmap='bwr', vmin=-1, vmax=1)
    # tune panels
    plt.colorbar(im, ax=ax, label='Correlation')
    labels_reduced = ['J', 'B1', 'B0', r'$\sigma$', r'$\alpha$'][:numpars]
    ax.set_xticks(np.arange(numpars), labels, rotation='270', fontsize=12)
    ax.set_yticks(np.arange(numpars), labels_reduced, fontsize=12)
    ax.set_xlabel('Original parameters', fontsize=14)
    # compute correlation matrix
    mat_corr = np.corrcoef(recovered_params.T, rowvar=True)
    mat_corr *= np.tri(*mat_corr.shape, k=-1)
    # plot correlation matrix
    im = ax2.imshow(mat_corr, cmap='bwr', vmin=-1, vmax=1)
    ax2.step(np.arange(0, numpars)-0.5, np.arange(0, numpars)-0.5, color='k',
             linewidth=.7)
    ax2.set_xticks(np.arange(numpars), labels, rotation='270', fontsize=12)
    ax2.set_yticks(np.arange(numpars), labels, fontsize=12)
    ax2.set_xlabel('Inferred parameters', fontsize=14)
    ax2.set_ylabel('Inferred parameters', fontsize=14)


def fit_subjects(method='BADS', model='MF', subjects='separated',
                 data_augmen=False, n_init=1, extra='', sub=None):
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    if subjects == 'together':
        all_df['subject'] = 'all'
        all_df = all_df.reset_index()
    if sub is not None:
        all_df = all_df.loc[all_df.subject == sub]
    subjects = all_df.subject.unique()
    accuracies = []
    for sub in subjects:
        print(sub)
        dataframe = all_df.copy().loc[all_df['subject'] == sub]
        accuracies.append(sum(dataframe.response == dataframe.side)/len(dataframe.response))
        # for sub in dataframe.subject.unique():
        #     data = dataframe[['pShuffle', 'confidence', 'evidence']]
        unique_vals = np.sort(dataframe['pShuffle'].unique())
        if extra != 'null':
            dataframe['coupling'] = dataframe['pShuffle'].replace(to_replace=unique_vals,
                                        value= [1., 0.3, 0.]) 
        else:
            dataframe['coupling'] = 1
        dataframe['confidence'] = (transform(dataframe.confidence.values, -0.999, 0.999)+1)/2
        # dataframe['confidence'] = transform(np.abs(dataframe.confidence.values))
        # dataframe['stim_str'] = np.abs(dataframe.evidence.values)
        dataframe['stim_str'] = (dataframe.evidence.values)
        dataframe['stim_ev_cong'] = dataframe.stim_str * dataframe.response
        data = dataframe[['coupling', 'confidence', 'stim_str', 'stim_ev_cong']]
        if data_augmen:
            data = data_augmentation(data, sigma=0.05, times_augm=4,
                                     minval=0.001)
        print(len(data))
        # fig, ax = plt.subplots(ncols=2)
        # sns.lineplot(data, x='coupling', y='confidence', hue='stim_str', ax=ax[0])
        # sns.lineplot(data, x='stim_str', y='confidence', hue='coupling', ax=ax[1])
        # fig.suptitle(sub)
        optimizer = optimization(data=data, n_iters=50, theta=return_theta())
        if model in ['MF5', 'LBP5']:
            s = sub
        else:
            s = None
        pars_array = fit_data(optimizer, model=model, n_iters=n_init, method=method,
                              plot=False, sub=s)
        params = np.median(pars_array, axis=0)
        # params = pars_array[0]
        print(params)
        if method == 'BADS':
            appendix = '_BADS'
        else:
            appendix = ''
        np.save(SV_FOLDER + 'parameters_' + model + appendix + sub +  extra + '.npy', params)


def plot_params_LBP5_vs_MF5():
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    subjects = all_df.subject.unique()
    nsubs = len(subjects)
    parmat_lbp5 = np.zeros((nsubs, 5))
    parmat_mf5 = np.zeros((nsubs, 5))
    for i_s, sub in enumerate(subjects):
        params_lbp5 = np.load(SV_FOLDER + 'parameters_' + 'LBP5_BADS' + sub + '.npy')
        params_mf5 = np.load(SV_FOLDER + 'parameters_' + 'MF5_BADS' + sub + '.npy')
        parmat_lbp5[i_s, :] = params_lbp5
        parmat_mf5[i_s, :] = params_mf5
    fig, ax = plt.subplots(ncols=5, figsize=(14, 3.5))
    labels = ['Coupling slope, J1', 'Coupling bias, J0', 'Stimulus weight, B1', 'Bias, B0', 'noise']
    for i in range(len(ax)):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].plot(parmat_mf5[:, i], parmat_lbp5[:, i], color='k', marker='o', linestyle='',
                   markersize=4)
        corr = scipy.stats.pearsonr(parmat_mf5[:, i], parmat_lbp5[:, i])
        ax[i].set_title(labels[i] + '\n' + fr'$\rho=${round(corr.statistic, 4)}, p={corr.pvalue:.3e}', fontsize=15)
        ax[i].set_ylabel('LBP5')
        ax[i].set_xlabel('MF5')
    fig.tight_layout()


def plot_fitted_params(sv_folder=SV_FOLDER, model='LBP', method='BADS',
                       subjects='separated'):
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    if subjects == 'together':
        all_df['subject'] = 'all'
    subjects = all_df.subject.unique()
    if model in ['LBP', 'MF', 'GS']:
        numpars = 4
    else:
        numpars = 5
    if method == 'BADS':
        appendix = '_BADS'
    else:
        appendix = ''
    accuracies = []
    pright = []
    conf = []
    nsubs = len(subjects)
    parmat = np.zeros((nsubs, numpars))
    for i_s, sub in enumerate(subjects):
        params = np.load(SV_FOLDER + 'parameters_' + model + appendix + sub + '.npy')
        parmat[i_s, :] = params
        dataframe = all_df.copy().loc[all_df['subject'] == sub]
        accuracies.append(sum(dataframe.response == dataframe.side)/len(dataframe.response))
        pright.append(np.mean((dataframe.response+1)/2))
        conf.append(np.mean(np.abs(dataframe.confidence)))
    fig, ax = plt.subplots(ncols=numpars, figsize=(15,5))
    labels = ['Coupling, J', 'Stimulus weight, B1', 'Bias, B0', 'noise', 'Alpha'][:numpars]
    if model == 'MF5' or model == 'LBP5':
        labels = ['Coupling slope, J1', 'Coupling bias, J0', 'Stimulus weight, B1', 'Bias, B0', 'noise']
    for i in range(numpars):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        sns.violinplot(parmat[:, i], color='lightblue', alpha=0.3, ax=ax[i])
        ax[i].plot(np.random.randn(nsubs)*0.05, parmat[:, i], marker='o', color='k', linestyle='', markersize=4)
        ax[i].set_xticks([])
        ax[i].set_ylabel(labels[i])
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'distros_fitted_params.png', dpi=200, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'distros_fitted_params.svg', dpi=200, bbox_inches='tight')
    fig2, ax2 = plt.subplots(ncols=numpars, nrows=3, figsize=(16, 14))
    ax2 = ax2.flatten()
    for i_a, a in enumerate(ax2):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        if i_a < numpars:
            var = pright
            lab = 'p(right)'
        if i_a >= numpars < 2*numpars:
            var = accuracies
            lab = 'p(correct)'
        if i_a >= numpars*2:
            var = conf
            lab = 'Confidence'
        # corr = np.corrcoef(var, parmat[:, i_a % numpars])[0, 1]
        corr = scipy.stats.pearsonr(var, parmat[:, i_a % numpars])
        if i_a == 2*numpars-1:
            cmap = mpl.cm.Oranges
            norm = mpl.colors.Normalize(vmin=0, vmax=np.max(parmat[:, 2]))
            color = cmap(norm(parmat[:, 2]))
            a.scatter(parmat[:, i_a % numpars], var, c=color, marker='o')
        else:
            color = 'k'
            a.plot(parmat[:, i_a % numpars], var, color=color, marker='o', linestyle='')
        a.set_ylabel(lab)
        a.set_xlabel(labels[i_a % numpars])
        a.set_title(rf'$\rho =$ {round(corr.statistic, 3)}, p={corr.pvalue:.1e}')
    fig2.tight_layout()
    ax2[0].text(-0.5, -1.1, 'Fitted parameters', rotation='vertical')
    ax2[6].text(-0.1, -1.05, 'Participants data')
    fig2.savefig(SV_FOLDER + 'correlations_fitted_parameters.png', dpi=200, bbox_inches='tight')
    fig2.savefig(SV_FOLDER + 'correlations_fitted_parameters.svg', dpi=200, bbox_inches='tight')
    fig3, ax3 = plt.subplots(ncols=4, figsize=(16, 5))
    stim_list = [0, 0.4, 0.8, 1]
    colormap = pl.cm.Oranges(np.linspace(0.2, 1, len(stim_list)))
    ylabs = ['p(right)', 'p(correct)']*2
    xlabs = ['b_1 * stim', '', 'b_1 * stim + b0', '']
    for i_a, a in enumerate(ax3):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_xlabel(xlabs[i_a])
        a.set_ylabel(ylabs[i_a])
    for i_s, stim in enumerate(stim_list):
        ax3[0].plot(parmat[:, 2]*stim, pright, color=colormap[i_s], marker='o', linestyle='')
        ax3[1].plot(parmat[:, 2]*stim, accuracies, color=colormap[i_s], marker='o', linestyle='')
        ax3[2].plot(parmat[:, 2]*stim+parmat[:, 3], pright, color=colormap[i_s], marker='o', linestyle='')
        ax3[3].plot(parmat[:, 2]*stim+parmat[:, 3], accuracies, color=colormap[i_s], marker='o', linestyle='')
    fig3.tight_layout()
    fig3.savefig(SV_FOLDER + 'biases_modulation_fitted_params.png', dpi=200, bbox_inches='tight')
    fig3.savefig(SV_FOLDER + 'biases_modulation_fitted_params.svg', dpi=200, bbox_inches='tight')
    fig, ax = plt.subplots(1)
    corrmat = np.corrcoef(parmat.T)
    corrmat[corrmat > 0.99] = np.nan
    im = ax.imshow(corrmat, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    plt.colorbar(im, ax=ax, label='correlation')
    
    labs = ['J', 'B1', 'B0', 'N', 'A'][:numpars]
    if model == 'MF5':
        labs = ['J1', 'J0', 'B1', 'B0', 'N']    
    ax.set_yticks(np.arange(numpars), labs)
    ax.set_xticks(np.arange(numpars), labs)
    fig4, ax4 = plt.subplots(ncols=4, figsize=(16, 4.5))
    for i_a, a in enumerate(ax4):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylabel('p(correct)')
        a.set_xlabel('Noise')
    cmap = mpl.cm.Oranges
    norm = mpl.colors.Normalize(vmin=0, vmax=0.8)
    color = cmap(norm(parmat[:, 1]))
    ax4[0].scatter(parmat[:, 3], accuracies, c=color, marker='o')
    ax4[0].set_title('Color: Stim. weight B1')
    cmap = mpl.cm.Blues
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    color = cmap(norm(parmat[:, 0]))
    ax4[1].scatter(parmat[:, 3], accuracies, c=color, marker='o')
    ax4[1].set_title('Color: Coupling J')
    cmap = mpl.cm.Greens
    norm = mpl.colors.Normalize(vmin=0., vmax=0.5)
    color = cmap(norm(np.abs(parmat[:, 2])))
    ax4[2].scatter(parmat[:, 3], accuracies, c=color, marker='o')
    ax4[2].set_title('Color: Bias |B_0|')
    cmap = mpl.cm.Purples
    var = -np.abs(parmat[:, 2])+parmat[:, 1]
    norm = mpl.colors.Normalize(vmin=np.min(var)-0.1, vmax=np.max(var))
    color = cmap(norm(-np.abs(parmat[:, 2])+parmat[:, 1]))
    ax4[3].scatter(parmat[:, 3], accuracies, c=color, marker='o')
    ax4[3].set_title('Color: B_1 - |B_0|')
    fig4.tight_layout()


def compute_j_crit(j_list=np.arange(0., 1.005, 0.01),
                   b_list=np.arange(-1, 1, 0.01),
                   num_iter=100):
    first_j = []
    for i_b, beta in enumerate(b_list):
        for j in j_list:
            q_fin = 0.65
            for i in range(num_iter):
                q_fin = backwards(q_fin, j, beta, n_neigh=3.92)
            if ~np.isnan(q_fin):
                first_j.append(j)
                break
        if len(first_j) != (i_b+1):
            first_j.append(np.nan)
    return first_j


def plot_density(num_iter=100, extra='', method='BADS', model='MF',
                 n=3.92):
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    subjects = all_df.subject.unique()
    state = []
    jstar = []
    bstar = []
    for sub in subjects:
        print(sub)
        dataframe = all_df.copy().loc[all_df['subject'] == sub]
        unique_vals = np.sort(dataframe['pShuffle'].unique())
        dataframe['coupling'] = dataframe['pShuffle'].replace(to_replace=unique_vals,
                                    value= [1., 0.3, 0.]) 
        dataframe['confidence'] = dataframe.confidence.values
        dataframe['stim_str'] = (dataframe.evidence.values)
        if method == 'BADS':
            appendix = '_BADS'
        else:
            appendix = ''
        pars = np.load(SV_FOLDER + '/parameters_'+model+ appendix+ sub + extra + '.npy')
        if model == 'MF5':
            b_eff = dataframe.stim_str.values*pars[2]+pars[3]
            j_eff = dataframe.coupling.values*pars[0]+pars[1]
        else:
            b_eff = dataframe.stim_str.values*pars[1]+pars[2]
            j_eff = dataframe.coupling.values*pars[0]
        jstar = jstar + list(j_eff)
        bstar = bstar + list(b_eff)
        for j, b in zip(j_eff, b_eff):
            q_fin = 0.65
            for i in range(num_iter):
                q_fin = backwards(q_fin, j, b, n_neigh=3.92)
                if np.isnan(q_fin):
                    state.append('Monostable')
                    break
            if ~np.isnan(q_fin):
                state.append('Bistable')
    all_df['state'] = state
    all_df['jstar'] = jstar
    all_df['bstar'] = bstar
    all_df = all_df.reset_index()
    plt.figure()
    # all_df['abs_evidence'] = np.abs(all_df.evidence)
    sns.kdeplot(all_df.loc[all_df.evidence == 0], x='confidence', hue='state',
                alpha=1, lw=2.5, bw_adjust=0.7, common_norm=False)
    fig, ax = plt.subplots(1, figsize=(4.5, 4))
    # b_list=np.arange(-1.5, 1.5, 0.01)
    j_list = np.arange(0., 1.405, 0.01)
    delta = np.sqrt(1-1/(j_list*n))
    b_crit1 = (np.log((1-delta)/(1+delta))+2*n*j_list*delta)/2
    b_crit2 = (np.log((1+delta)/(1-delta))-2*n*j_list*delta)/2
    ax.plot(b_crit1, j_list, color='k', linewidth=2.5, label='J*')
    ax.legend(frameon=False)
    ax.plot(b_crit2, j_list, color='k', linewidth=2.5)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks([-1, 0, 1])
    ax.text(-1., 1., 'Bistable')
    ax.text(-1., 0.1, 'Monostable')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Sensory evidence, B')
    ax.set_ylabel('Coupling, J')
    fig.tight_layout()
    sns.kdeplot(all_df, x='bstar', y='jstar', fill=True, ax=ax)
    # cbar=True,
    # cbar_kws={'label': 'Density of trials',
    #           'orientation': 'horizontal',
    #           'location': 'top'})
    # cbar = fig.axes[-1]
    # cbar.tick_params(left=False, labelleft=False, right=False, labelright=False,
    #                  top=False, labeltop=False, bottom=False, labelbottom=False)
    # cbar.set_xticklabels([])
    fig.savefig(SV_FOLDER + 'j_vs_b_classification.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'j_vs_b_classification.svg', dpi=400, bbox_inches='tight')


def compute_jstar_bstar(sub, dataframe, model='MF', method='BADS',
                        extra='', num_iter=100):
    state = []
    if method == 'BADS':
        appendix = '_BADS'
    else:
        appendix = ''
    pars = np.load(SV_FOLDER + '/parameters_'+model+ appendix+ sub + extra + '.npy')
    if model == 'MF5':
        b_eff = dataframe.stim_str.values*pars[2]+pars[3]
        j_eff = dataframe.coupling.values*pars[0]+pars[1]
    else:
        if extra == 'null':
            j_eff = np.ones(len(dataframe))*pars[0]
        else:
            j_eff = dataframe.coupling.values*pars[0]
        b_eff = dataframe.stim_str.values*pars[1]+pars[2]
        
    for j, b in zip(j_eff, b_eff):
        q_fin = 0.65
        for i in range(num_iter):
            q_fin = backwards(q_fin, j, b, n_neigh=3.92)
            if np.isnan(q_fin):
                state.append('Monostable')
                break
        if ~np.isnan(q_fin):
            state.append('Bistable')
    return j_eff, b_eff, state


def plot_confidence_vs_stim(method='BADS', variable='confidence', subject='s_11', plot_all=False,
                            bw=0.5, annot=False, model_density=True):
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    data_orig, data_model_orig, data_model_null =\
        load_all_data(all_df, model='MF5', method=method, sv_folder=SV_FOLDER)
    data_orig['decision'] = (data_orig.response.values+1)/2
    data_model_orig['decision'] = (data_model_orig.decision.values+1)/2
    data_model_null['decision'] = (data_model_null.decision.values+1)/2
    df_sub_final = data_orig.dropna().reset_index()
    # Compute the mean confidence per subject for each (stim_ev_cong, coupling) pair
    df_sub_model = data_model_orig.dropna().reset_index()
    # Compute the mean confidence per subject for each (stim_ev_cong, coupling) pair
    df_sub_null = data_model_null.dropna().reset_index()
    stim_str = np.array([-1, 0, 1])
    if plot_all:
        fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(12, 10))
        ax = ax.flatten()
    else:
        fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(4, 8))
    for i_c, c in enumerate([0, 0.3, 1]):
        if subject == 'all' or subject is None:
            df_data_coup = df_sub_final.loc[(df_sub_final.coupling == c)]
            df_data_coup['confidence'] = df_data_coup.groupby('subject')['confidence'].apply(scipy.stats.zscore)
            df_model_coup = df_sub_model.loc[(df_sub_model.coupling == c)]
            df_model_coup['confidence'] = df_model_coup.groupby('subject')['confidence'].apply(scipy.stats.zscore)
            df_null_coup = df_sub_null.loc[(df_sub_null.coupling == c)]
            df_null_coup['confidence'] = df_null_coup.groupby('subject')['confidence'].apply(scipy.stats.zscore)
            s = 0.4
        else:
            df_data_coup = df_sub_final.loc[(df_sub_final.coupling == c) & (df_sub_final.subject == subject)]
            df_model_coup = df_sub_model.loc[(df_sub_model.coupling == c) & (df_sub_model.subject == subject)]
            df_null_coup = df_sub_null.loc[(df_sub_null.coupling == c) & (df_sub_null.subject == subject)]
            s = 2
        if annot:
            skewness = scipy.stats.skew(df_data_coup.loc[df_data_coup.stim_str == 0, variable])
            kurtosis = scipy.stats.kurtosis(
                df_data_coup.loc[df_data_coup.stim_str == 0, variable], fisher=True)
            n = len(df_data_coup)
            val_sum = 3*(n-1)**2 / ((n-2)*(n-3))
            beta = round((skewness**2 + 1)/(kurtosis+val_sum), 4)
            # ax[i_c].set_title(f'Bimodal coef. = {beta}', fontsize=15)
            # for i_s, stim in enumerate([-1, -0.8, -0.4, 0, 0.4, 0.8, 1]):
            #     stat, p = diptest(df_data_coup.loc[df_data_coup.stim_str == stim, variable])
            #     pval = stars_pval(p)
            #     # ho = np.max(df_data_coup.loc[df_data_coup.stim_str == stim, variable])+0.2
            #     ho = 1.3 if subject != 'all' else 2.5
            #     ax[i_c].text(i_s, ho, f"{pval}", ha='center', va='bottom', color='k',
            #                  fontsize=12)
            stat, p = diptest(df_data_coup.loc[df_data_coup.stim_str == 0, variable])
            print(p)
            ax[i_c].set_title(f'Bimodal coef. = {beta},\npval = {p:.3e}', fontsize=14)
        if model_density:
            sns.violinplot(df_model_coup, x='stim_str', y=variable, ax=ax[i_c],
                           palette=COLORMAP, hue='stim_str',
                           legend=False, inner=None, split=False, bw_adjust=bw,
                           linewidth=0, cut=0)
        else:
            sns.violinplot(df_data_coup, x='stim_str', y=variable, ax=ax[i_c],
                           palette=COLORMAP, hue='stim_str',
                           legend=False, inner=None, split=False, bw_adjust=bw,
                           linewidth=0, cut=0)
        sns.swarmplot(df_data_coup, x='stim_str', y=variable, ax=ax[i_c],
                      color='k', size=s, legend=False, alpha=0.8)
        # slope = scipy.stats.linregress(df_data_coup.stim_str, df_data_coup.confidence).slope
        # intercept = scipy.stats.linregress(df_data_coup.stim_str, df_data_coup.confidence).intercept
        # ax[i_c].plot([0, 3, 6], stim_str*slope+intercept, color='gray', linestyle='--', alpha=0.7)
        if plot_all:
            sns.violinplot(df_model_coup, x='stim_str', y=variable, ax=ax[i_c+3],
                           palette=COLORMAP, hue='stim_str',
                           legend=False, inner=None, split=False, bw_adjust=bw,
                           linewidth=0)
            sns.swarmplot(df_model_coup, x='stim_str', y=variable, ax=ax[i_c+3],
                          color='k', size=s, legend=False, alpha=0.8)
            sns.violinplot(df_null_coup, x='stim_str', y=variable, ax=ax[i_c+6],
                           palette=COLORMAP, hue='stim_str',
                           legend=False, inner=None, split=False, bw_adjust=bw,
                           linewidth=0)
            sns.swarmplot(df_null_coup, x='stim_str', y=variable, ax=ax[i_c+6],
                          color='k', size=s, legend=False, alpha=0.8)
    for i_a, a in enumerate(ax):
        # a.plot([0, 6], [0, 1], color='gray', linestyle='--', alpha=0.7)
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        if i_a < 2:
            a.set_xticks([])
            a.set_xlabel('')
        if i_a != 1:
            a.set_ylabel('')
        if subject != 'all':
            a.set_ylim(-0.2, 1.2)
            a.set_yticks([0, 0.5, 1])
            a.axhline(1, color='gray', linestyle='--', alpha=0.7)
            a.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax[2].set_xticks([0, 3, 6])
    ax[2].set_xlabel('Stimulus evidence')
    if subject == 'all':
        ax[1].set_ylabel('z-scored confidence')
    else:
        ax[1].set_ylabel('Confidence')
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'conf_vs_stim_data_density_plot_model.png', dpi=150, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'conf_vs_stim_data_density_plot_model.svg', dpi=150, bbox_inches='tight')


def plot_conf_vs_coupling_3_groups(method='BADS', model='MF5', extra='', bw=0.7,
                                   data_only=True):
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    subjects = all_df.subject.unique()
    state = []
    data_orig, data_model_orig, data_model_null =\
        load_all_data(all_df, model=model, method=method, sv_folder=SV_FOLDER)
    datastim0 = data_orig.loc[data_orig.stim_str == 0]
    datastim_model0 = data_model_orig.loc[data_model_orig.stim_str == 0]
    datastim0['state'] = 0
    datastim_model0['state'] = 0
    arr_betavals = np.zeros((3, len(subjects)))
    coupvals = np.zeros((3, len(subjects)))
    # jcritvals = np.zeros((len(subjects)))
    if method == 'BADS':
        appendix = '_BADS'
    else:
        appendix = ''
    for i_s, sub in enumerate(subjects):
        dataframe = datastim0.copy().loc[(datastim0['subject'] == sub)]
        for i_c, c in enumerate([0., 0.3, 1]):
            skewness = scipy.stats.skew(dataframe.loc[dataframe.coupling == c, 'confidence'])
            kurtosis = scipy.stats.kurtosis(
                dataframe.loc[dataframe.coupling == c, 'confidence'], fisher=True)
            n = len(dataframe)
            val_sum = 3*(n-1)**2 / ((n-2)*(n-3))
            beta = round((skewness**2 + 1)/(kurtosis+val_sum), 4)
            arr_betavals[i_c, i_s] = beta
        pars = np.load(SV_FOLDER + '/parameters_'+model+ appendix+ sub + extra + '.npy')
        if extra != 'null':
            b_eff = pars[3]
            jcrit = compute_j_crit(j_list=np.arange(1/4, 0.8, 1e-4), b_list=[b_eff], num_iter=100)[0]
            coup = np.unique(pars[0]*dataframe.coupling+pars[1])-jcrit
        else:
            b_eff = pars[2]
            jcrit = compute_j_crit(j_list=np.arange(1/4, 0.8, 1e-4), b_list=[b_eff], num_iter=100)[0]
            coup = np.unique(pars[0])-jcrit
        coupvals[:, i_s] = np.sort(coup)
        # jcritvals[i_s] = jcrit
        if (np.sign(coup) < 0).all():
            s = 0
        if (np.sign(coup) > 0).all():
            s = 2
        if np.sign(np.min(coup)) != np.sign(np.max(coup)):
            s = 1
        state.append(s)
        datastim0.loc[datastim0['subject'] == sub, 'state'] = s
        datastim_model0.loc[datastim_model0['subject'] == sub, 'state'] = s
    print('S0: monostable: ' + str(np.sum(np.array(state)==0)))
    print('S1: change: ' + str(np.sum(np.array(state)==1)))
    print('S2: bistable: ' + str(np.sum(np.array(state)==2)))
    fig, ax = plt.subplots(ncols=3, figsize=(13, 5))
    colormap = pl.cm.Oranges(np.linspace(0.3, 1, 3))
    cmap = [c for c in colormap]
    labs = ['Sub-critical', 'Transition', 'Supra-critical']
    labs_j = [0, 0.3, 1]
    legendelements = []
    for i_a, a in enumerate(ax):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        sns.kdeplot(datastim0.loc[datastim0.state == i_a],
                    x='confidence', hue='coupling', ax=a,
                    bw_adjust=bw, palette=cmap, linewidth=3,
                    legend=False)
        if not data_only:
            sns.kdeplot(datastim_model0.loc[datastim_model0.state == i_a],
                        x='confidence', hue='coupling', ax=a,
                        bw_adjust=bw, palette=cmap, linewidth=3,
                        legend=False, linestyle='--')
        legendelements.append(Line2D([0], [0], color=cmap[i_a],
                                     lw=3, label=labs_j[i_a]))
        a.set_title(labs[i_a], fontsize=15)
        a.set_ylabel('')
        a.set_xlabel('Confidence')
    ax[0].set_ylabel('Desntiy')
    ax[0].legend(frameon=False, title='Coupling, J', handles=legendelements,
                 bbox_to_anchor=(0.7, 0.75))
    fig.tight_layout()
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    # state = np.array(state)
    sns.violinplot(arr_betavals.T, ax=ax2, palette=cmap, cut=0, inner=None)
    sns.swarmplot(arr_betavals.T, color='k', size=3, legend=False, alpha=0.8)
    g = sns.lineplot(arr_betavals, color='k', alpha=0.2, legend=False, dashes=False)
    lines = g.get_lines()
    [l.set_color('black') for l in lines]
    meanvals = np.nanmean(arr_betavals, axis=1)
    ax2.plot([0, 1, 2], meanvals, linewidth=4, color='r')
    # pvals = [stars_pval(scipy.stats.ttest_1samp(arr, 5/9).pvalue) for arr in arr_betavals]
    # ax2.text(0, 1, f"{pvals[0]}", ha='center', va='bottom', color='k',
    #          fontsize=15)
    # ax2.text(1, 1, f"{pvals[1]}", ha='center', va='bottom', color='k',
    #          fontsize=15)
    # ax2.text(2, 1, f"{pvals[2]}", ha='center', va='bottom', color='k',
    #          fontsize=15)
    ax2.axhline(5/9, color='k', linestyle='--', alpha=0.3)
    ax2.set_xticks([0, 1, 2], ['100', '70', '0'])
    ax2.set_xlabel('Shuffling')
    ax2.set_ylabel('Bimodality coef.')
    fig2.tight_layout()
    fig3, ax3 = plt.subplots(ncols=2, figsize=(10, 4))
    color = ['k' if s != 1 else 'r' for s in state]
    alphas = [0.3 if s != 1 else 0.8 for s in state]
    for a in ax3:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        [a.plot([0, 0.3, 1], coupvals[:, i], color=color[i], marker='o', alpha=alphas[i]) for i in range(len(subjects))]
        a.axhline(0, color='r', linestyle='--', alpha=0.3)
        # a.axhline(1/3.92, color='r', linestyle='--', alpha=0.3)
        a.set_xlabel('J = 1-p_shuffle')
        a.set_ylabel(r'$J \beta_1  + \beta_0 - J*(B_0)$')
    ax3[1].set_ylim(-0.05, 0.05)
    fig3.tight_layout()


def plot_density_comparison(num_iter=100, method='nelder-mead',
                            kde=False, stim_ev_0=False, ax0=None, fig=None,
                            full_fig=False, variable='signed_confidence',
                            bw=0.7, model='MF5'):
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    subjects = all_df.subject.unique()
    state = [[], [], [], []]
    data_orig, data_model_orig, data_model_null =\
        load_all_data(all_df, model=model, method=method, sv_folder=SV_FOLDER)
    model_null = 'MF' if model == 'MF5' else 'LBP'
    for sub in subjects:
        dataframe = data_orig.copy().loc[data_orig['subject'] == sub]
        data_model_o_sub = data_model_orig.copy().loc[data_model_orig['subject'] == sub]
        data_model_null_sub = data_model_null.copy().loc[data_model_null['subject'] == sub]
        j_eff, b_eff, state_sub = compute_jstar_bstar(sub, dataframe, model=model_null,
                                                      method=method, extra='null',
                                                      num_iter=num_iter)
        state[0] = state[0] + state_sub
        j_eff, b_eff, state_sub = compute_jstar_bstar(sub, dataframe, model=model,
                                                      method=method, extra='',
                                                      num_iter=num_iter)
        state[1] = state[1] + state_sub
        j_eff, b_eff, state_sub = compute_jstar_bstar(sub, data_model_o_sub, model=model,
                                                      method=method, extra='',
                                                      num_iter=num_iter)
        state[2] = state[2] + state_sub
        j_eff, b_eff, state_sub = compute_jstar_bstar(sub, data_model_null_sub, model=model_null,
                                                      method=method, extra='null',
                                                      num_iter=num_iter)
        state[3] = state[3] + state_sub
    data_orig_mf_null = data_orig.copy()
    data_orig_mf = data_orig.copy()
    data_orig_mf_null['state'] = state[0]
    data_orig_mf_null = data_orig_mf_null.reset_index()
    data_orig_mf['state'] = state[1]
    data_orig_mf = data_orig_mf.reset_index()
    data_model_orig['state'] = state[2]
    data_model_orig = data_model_orig.reset_index()
    data_model_null['state'] = state[3]
    data_model_null = data_model_null.reset_index()
    if not full_fig:
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(11, 10))
        ax = ax.flatten()
        leg = [True, False, False, False]
        titles = [r'Data, $\theta_{null}$', r'Data, $\theta_{full}$',
                  r'Null data, $\theta_{null}$', r'Full data, $\theta_{full}$']
        for df, a, l, title in zip([data_orig_mf_null, data_orig_mf, data_model_null, data_model_orig],
                                   ax, leg, titles):
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            df['signed_confidence'] = 2*df.confidence-1
            df['aligned_confidence'] = (2*df.confidence-1)*np.sign(df.stim_str + np.random.randn()*1e-6)
            if not kde:
                sns.histplot(df.loc[df.stim_str == 0], x=variable, hue='state',
                             alpha=0.4, lw=1.5, common_norm=False, ax=a,
                             legend=l, stat='density', bins=10, palette=['k', 'r'])
                a.set_ylim(-0.1, 1.2)
            if kde:
                if not stim_ev_0:
                    stim = [0, 0.4, 0.8, 1]
                    colormap_r = pl.cm.Reds(np.linspace(0.3, 1, 4))
                    colormap_k = pl.cm.gist_gray(np.linspace(0., 0.7, 4))
                    for ia in range(4):
                        sns.kdeplot(df.loc[df.stim_str.abs() == stim[ia]], x=variable, hue='state',
                                     alpha=1, lw=2.5, common_norm=False, ax=a,
                                     legend=l, bw_adjust=bw, palette=[colormap_k[ia], colormap_r[ia]])
                    a.set_ylim(-0.1, 1.1)
                if stim_ev_0:
                    sns.kdeplot(df.loc[df.stim_str == 0], x=variable, hue='state',
                                 alpha=1, lw=2.5, common_norm=False, ax=a,
                                 legend=l, bw_adjust=bw, palette=['k', 'r'])
                    a.set_ylim(-0.1, 1.1)
            if variable == 'aligned_confidence':
                a.set_xlabel('Confidence aligned with stimulus')
            else:
                a.set_xlabel('Confidence')
            a.set_ylabel('Density')
            a.set_title(title)
    if ax0 is None:
        fig.tight_layout()
        fig2, ax2 = plt.subplots(ncols=2, figsize=(9, 4))
    else:
        ax2 = ax0
    stim = [0, 0.4, 0.8, 1]
    colormap_r = pl.cm.Reds(np.linspace(0.3, 1, 4))
    colormap_k = pl.cm.gist_gray_r(np.linspace(0.3, 1, 4))
    data_monost = data_orig_mf.loc[data_orig_mf.state == 'Monostable']
    data_bist = data_orig_mf.loc[data_orig_mf.state == 'Bistable']
    legendelements = []
    for ia in range(4):
        sns.kdeplot(data_monost.loc[(data_orig_mf.stim_str.abs() == stim[ia])],
                    x=variable,
                    alpha=1, lw=3., common_norm=False, ax=ax2[0],
                    legend=False, bw_adjust=bw, color=colormap_k[ia])
        sns.kdeplot(data_bist.loc[(data_orig_mf.stim_str.abs() == stim[ia])],
                    x=variable,
                    alpha=1, lw=3., common_norm=False, ax=ax2[1],
                    legend=False, bw_adjust=bw, color=colormap_k[ia])
        legendelements.append(Line2D([0], [0], color=colormap_k[ia],
                                     lw=3.5, label=stim[ia]))
    ax2[0].set_title('Monostable', fontsize=19)
    ax2[1].set_title('Bistable', fontsize=19)
    ax2[0].set_ylabel('Density of confidence')
    ax2[1].set_ylabel('')
    ax2[1].legend(frameon=False, title='Stimulus\nstrength', handles=legendelements,
                  bbox_to_anchor=(0.86, 0.6))
    for a2 in ax2:
        a2.spines['right'].set_visible(False)
        a2.spines['top'].set_visible(False)
        a2.set_yticks([])
        if variable == 'aligned_confidence':
            a2.set_ylim(-0.05, 1.6)
            a2.set_xlabel('')
        else:
            a2.set_xlabel('Confidence')
            a2.set_ylim(-0.05, 1.)
    ax2[0].set_xlabel('                                Confidence aligned with stimulus')
    fig2.tight_layout()
    fig2.savefig(SV_FOLDER + 'classification_density_confidence.png', dpi=400, bbox_inches='tight')
    fig2.savefig(SV_FOLDER + 'classification_density_confidence.svg', dpi=400, bbox_inches='tight')


def linear_regression(data_orig, data_model_orig, data_model_null):
    subjects = data_orig.subject.unique()
    weights_o = np.zeros((4, len(subjects)))
    weights_model_o = np.zeros((4, len(subjects)))
    weights_model_null = np.zeros((4, len(subjects)))
    scores_model_o = np.zeros((len(subjects)))
    scores_model_null = np.zeros((len(subjects)))
    for i_s, sub in enumerate(subjects):
        df_o = data_orig.loc[data_orig.subject == sub]
        df_model_o = data_model_orig.loc[data_model_orig.subject == sub]
        df_model_null = data_model_null.loc[data_model_null.subject == sub]
        coupling = df_o.coupling.values
        # stim_str = np.abs(df_o.stim_str.values)
        stim_str_cong_o = df_o.stim_ev_cong.values
        stim_str_cong_model_o = df_model_o.stim_ev_cong.values
        stim_str_cong_model_null = df_model_null.stim_ev_cong.values
        val_conf_yo = df_o.abs_confidence.values
        idx_non_nan = np.where(~np.isnan(val_conf_yo))[0]
        X_o = np.column_stack((coupling[idx_non_nan], stim_str_cong_o[idx_non_nan],
                               stim_str_cong_o[idx_non_nan]*coupling[idx_non_nan]))
        X_model_o = np.column_stack((coupling[idx_non_nan], stim_str_cong_model_o[idx_non_nan],
                                     stim_str_cong_model_o[idx_non_nan]*coupling[idx_non_nan]))
        X_model_null = np.column_stack((coupling[idx_non_nan], stim_str_cong_model_null[idx_non_nan],
                                        stim_str_cong_model_null[idx_non_nan]*coupling[idx_non_nan]))
        y_o = scipy.stats.zscore(val_conf_yo[idx_non_nan])
        y_model_o = scipy.stats.zscore(df_model_o.abs_confidence.values[idx_non_nan])
        y_model_null = scipy.stats.zscore(df_model_null.abs_confidence.values[idx_non_nan])
        regr_o = linear_model.LinearRegression(fit_intercept=True)
        regr_o.fit(X_o, y_o)
        weights_o[1:, i_s] = regr_o.coef_
        weights_o[0, i_s] = regr_o.intercept_
        regr_model_o = linear_model.LinearRegression(fit_intercept=True)
        regr_model_o.fit(X_model_o, y_model_o)
        weights_model_o[1:, i_s] = regr_model_o.coef_
        weights_model_o[0, i_s] = regr_model_o.intercept_
        regr_model_null = linear_model.LinearRegression(fit_intercept=True)
        regr_model_null.fit(X_model_null, y_model_null)
        weights_model_null[1:, i_s] = regr_model_null.coef_
        weights_model_null[0, i_s] = regr_model_null.intercept_
        y_o_predict_model_o = regr_o.predict(X_model_o)
        y_o_predict_model_null = regr_o.predict(X_model_null)
        scores_model_null[i_s] = r2_score(y_o_predict_model_null, regr_model_null.predict(X_model_null))
        scores_model_o[i_s] = r2_score(y_o_predict_model_o, regr_model_o.predict(X_model_o))
    return weights_o, weights_model_o, weights_model_null, scores_model_o, scores_model_null


def linear_mixed_model(data_orig, data_model_orig, data_model_null):
    md_orig = smf.mixedlm("abs_confidence ~ coupling*stim_ev_cong",
                          data_orig.dropna(), groups="subject",
                          re_formula='~coupling*stim_ev_cong')
    md_orig = md_orig.fit()
    print(md_orig.summary())
    data_mode_orig_no_nans = data_model_orig.dropna()
    md_model_orig = smf.mixedlm("abs_confidence ~ coupling*stim_ev_cong",
                          data_mode_orig_no_nans, groups="subject",
                          re_formula='~coupling*stim_ev_cong')
    md_model_orig = md_model_orig.fit()
    print(md_model_orig.summary())
    data_mode_null_no_nans = data_model_null.dropna()
    md_model_null = smf.mixedlm("abs_confidence ~ coupling*stim_ev_cong",
                                data_mode_null_no_nans, groups="subject",
                                re_formula='~coupling*stim_ev_cong')
    md_model_null = md_model_null.fit()
    print(md_model_null.summary())


def load_all_data(all_df, model='MF5', method='BADS', sv_folder=SV_FOLDER,
                  data_augment=False):
    subjects = all_df.subject.unique()
    modeln = 'MF' if model == 'MF5' else 'LBP'
    data_orig = pd.DataFrame()
    data_model_null = pd.DataFrame()
    data_model_orig = pd.DataFrame()
    if method == 'BADS':
        appendix = '_BADS'
    else:
        appendix = ''
    for sub in subjects:
        print(sub)
        dataframe = all_df.copy().loc[all_df['subject'] == sub]
        unique_vals = np.sort(dataframe['pShuffle'].unique())
        extra = 'null'
        dataframe['coupling'] = 1
        dataframe['confidence'] = (transform(dataframe.confidence.values, -0.999, 0.999)+1)/2
        dataframe['stim_str'] = (dataframe.evidence.values)
        dataframe['stim_ev_cong'] = dataframe.response.values*dataframe.stim_str.values
        data = dataframe[['coupling', 'confidence', 'stim_str', 'stim_ev_cong', 'response']]
        pars = np.load(sv_folder + '/parameters_'+modeln + appendix + sub + extra + '.npy')
        df_null = simulate_FBP(pars, 400,
                               data.stim_str.values, data.coupling.values,
                               sv_folder=SV_FOLDER, n_iter=sub, model=modeln,
                               recovery=False, resimulate=False, extra=extra)
        df_null['coupling'] = dataframe['pShuffle'].replace(to_replace=unique_vals,
                                                           value= [1., 0.3, 0.]) 
        df_null['subject'] = sub
        data_model_null = pd.concat((data_model_null, df_null))
        extra = ''
        dataframe['coupling'] = dataframe['pShuffle'].replace(to_replace=unique_vals,
                                    value= [1., 0.3, 0.]) 
        dataframe['confidence'] = (transform(dataframe.confidence.values, -0.999, 0.999)+1)/2
        dataframe['stim_str'] = (dataframe.evidence.values)
        dataframe['stim_ev_cong'] = dataframe.response.values*dataframe.stim_str.values
        data = dataframe[['coupling', 'confidence', 'stim_str', 'stim_ev_cong', 'response',
                          'subject']]
        if data_augment:
            data = data_augmentation(data, sigma=0.05, times_augm=4,
                                     minval=0.001)
        data_orig = pd.concat((data_orig, data))
        pars = np.load(sv_folder + '/parameters_'+model + appendix + sub + extra + '.npy')
        df_no_null = simulate_FBP(pars, 400,
                                  data.stim_str.values, data.coupling.values,
                                  sv_folder=SV_FOLDER, n_iter=sub, model=model,
                                  recovery=False, resimulate=False, extra=extra)
        df_no_null['coupling'] = dataframe['pShuffle'].replace(to_replace=unique_vals,
                                                               value= [1., 0.3, 0.]) 
        data_model_orig = pd.concat((data_model_orig, df_no_null))
    for datframe in [data_orig, data_model_null, data_model_orig]:
        datframe['abs_confidence'] = np.abs(datframe.confidence.values-0.5)*2
    data_model_orig['subject'] = data_model_null.subject.values
    if data_augment:
        data_orig['subject'] = data_model_null.subject.values
    if not data_augment:
        data_orig['subject'] = all_df.subject.values
    return data_orig, data_model_orig, data_model_null


def plot_regression_weights(sv_folder=SV_FOLDER, load=True, model='MF', method='BADS',
                            ax=None, fig=None):
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    subjects = all_df.subject.unique()
    modeln = 'LBP' if model == 'LBP5' else 'MF'
    if not load:
        data_orig, data_model_orig, data_model_null =\
            load_all_data(all_df, model=model, method=method, sv_folder=SV_FOLDER,
                          data_augment=False)
        data_orig.to_csv(sv_folder + 'simulated_data' + '/df_orig.csv')
        data_model_orig.to_csv(sv_folder + 'simulated_data' + '/df_simul_'+model+'_orig.csv')
        data_model_null.to_csv(sv_folder + 'simulated_data' + '/df_simul_'+modeln+'_null_model.csv')
    else:
        data_orig = pd.read_csv(sv_folder + 'simulated_data' + '/df_orig.csv')
        data_model_orig = pd.read_csv(sv_folder + 'simulated_data' + '/df_simul_'+model+'_orig.csv')
        data_model_null = pd.read_csv(sv_folder + 'simulated_data' + '/df_simul_'+modeln+'_null_model.csv')
    # linear_mixed_model(data_orig, data_model_orig, data_model_null)
    # data_orig['decision'] = data_orig.response
    # for df in ([data_orig, data_model_orig, data_model_null]):
    #     df['congruent_confidence'] = df.confidence*df.decision
    # sub by sub lienar regression
    weights_o, weights_model_o, weights_model_null, scores_model_o, scores_model_null =\
        linear_regression(data_orig, data_model_orig, data_model_null)
    savefig = False
    fig3, ax3 = plt.subplots(1)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    sns.kdeplot(scores_model_o, color='k', linewidth=3, label='Full', bw_adjust=0.5,
                cumulative=True)
    sns.kdeplot(scores_model_null, color='r', linewidth=3, label='Null', bw_adjust=0.5,
                cumulative=True)
    ax3.set_xlabel('Score from linear regression')
    ax3.set_ylabel('Cumulative density')
    ax3.legend(frameon=False)
    fig3.tight_layout()
    if ax is None:
        fig, ax = plt.subplots(ncols=4, figsize=(16, 4))
        savefig = True
    for a in ax:
        a.axhline(0, color='k', linestyle='--')
    colors = ['midnightblue', 'royalblue', 'cornflowerblue']
    # colors = []
    xlabs = ['Data', 'Model', 'Null']
    ylabs = ['Intercept', 'Coupling', 'Stim. congr.', 'Coupling:stim. congr.']
    for j in range(4):
        ax[j].spines['right'].set_visible(False)
        ax[j].spines['top'].set_visible(False)
        sns.boxplot([weights_o[j], weights_model_o[j], weights_model_null[j]], ax=ax[j],
                    palette=colors, flierprops={"marker": ""}, linecolor='grey', linewidth=1.4)
        ax[j].set_ylabel(ylabs[j])
        ax[j].set_xticks([0, 1, 2], xlabs, rotation=45)
    for i_s in range(len(subjects)):
        jitter = np.random.randn(3)*0.05
        # ax[0].plot([0+jitter[0], 1+jitter[1], 2+jitter[2]],
        #            [weights_o[0][i_s], weights_model_o[0][i_s], weights_model_null[0][i_s]],
        #            color='gray', alpha=0.5)
        # ax[1].plot([0+jitter[0], 1+jitter[1], 2+jitter[2]],
        #            [weights_o[1][i_s], weights_model_o[1][i_s], weights_model_null[1][i_s]],
        #            color='gray', alpha=0.5)
        # ax[2].plot([0+jitter[0], 1+jitter[1], 2+jitter[2]],
        #            [weights_o[2][i_s], weights_model_o[2][i_s], weights_model_null[2][i_s]],
        #            color='gray', alpha=0.5)
        # ax[3].plot([0+jitter[0], 1+jitter[1], 2+jitter[2]],
        #            [weights_o[3][i_s], weights_model_o[3][i_s], weights_model_null[3][i_s]],
        #            color='gray', alpha=0.5)
        for t in range(3):
            ax[0].plot([t+jitter[t]],
                       [weights_o[0][i_s], weights_model_o[0][i_s], weights_model_null[0][i_s]][t],
                       marker='o', color=colors[t], linestyle='',
                       markersize=5, markeredgewidth=1, markeredgecolor='white')
            ax[1].plot([t+jitter[t]],
                       [weights_o[1][i_s], weights_model_o[1][i_s], weights_model_null[1][i_s]][t],
                       marker='o', color=colors[t], linestyle='',
                       markersize=5, markeredgewidth=1, markeredgecolor='white')
            ax[2].plot([t+jitter[t]],
                       [weights_o[2][i_s], weights_model_o[2][i_s], weights_model_null[2][i_s]][t],
                       marker='o', color=colors[t], linestyle='',
                       markersize=5, markeredgewidth=1, markeredgecolor='white')
            ax[3].plot([t+jitter[t]],
                       [weights_o[3][i_s], weights_model_o[3][i_s], weights_model_null[3][i_s]][t],
                       marker='o', color=colors[t], linestyle='',
                       markersize=5, markeredgewidth=1, markeredgecolor='white')
    if savefig:
        fig.tight_layout()
    pvals_o = []
    pvals_fm = []
    pvals_null = []
    for wo, wfm, wnull in zip(weights_o, weights_model_o, weights_model_null):
        pvals_o.append(stars_pval(scipy.stats.ttest_1samp(wo, 0).pvalue))
        pvals_fm.append(stars_pval(scipy.stats.ttest_1samp(wfm, 0).pvalue))
        pvals_null.append(stars_pval(scipy.stats.ttest_1samp(wnull, 0).pvalue))
    h_o = np.max(weights_o, axis=1)
    h_fm = np.max(weights_model_o, axis=1)
    h_null = np.max(weights_model_null, axis=1)
    eps = 0.05
    for a in range(4):
        ax[a].text(0, h_o[a]+0.1, f"{pvals_o[a]}", ha='center', va='bottom', color='k',
                   fontsize=12)
        ax[a].text(1, h_fm[a]+0.1, f"{pvals_fm[a]}", ha='center', va='bottom', color='k',
                   fontsize=12)
        ax[a].text(2, h_null[a]+0.1, f"{pvals_null[a]}", ha='center', va='bottom', color='k',
                   fontsize=12)
        x1, x2 = [0+eps, 1-eps]
        p = stars_pval(scipy.stats.ttest_ind(weights_o[a],  weights_model_o[a]).pvalue)
        y, h, col = max(map(max, np.column_stack((weights_o[a],
                                                  weights_model_o[a]))))+0.4, 0.05, 'k'
        ax[a].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        ax[a].text((x1+x2)*.5, y+h, f"{p}", ha='center', va='bottom', color=col,
                   fontsize=12)
        p = stars_pval(scipy.stats.ttest_ind(weights_o[a],  weights_model_null[a]).pvalue)
        x1, x2 = [0-eps, 2+eps]
        y, h, col = max(map(max, np.column_stack((weights_o[a],
                                                  weights_model_o[a]))))+0.65, 0.05, 'k'
        ax[a].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        ax[a].text((x1+x2)*.5, y+h, f"{p}", ha='center', va='bottom', color=col,
                   fontsize=12)
        x1, x2 = [1+eps, 2-eps]
        y, h, col = max(map(max, np.column_stack((weights_o[a],
                                                  weights_model_o[a]))))+0.4, 0.05, 'k'
        ax[a].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        ax[a].text((x1+x2)*.5, y+h, f"{p}", ha='center', va='bottom', color=col,
                   fontsize=12)
    if savefig:
        fig.savefig(SV_FOLDER + 'linear_regression_analysis.png', dpi=400, bbox_inches='tight')
        fig.savefig(SV_FOLDER + 'linear_regression_analysis.svg', dpi=400, bbox_inches='tight')


def stars_pval(pval):
    s = 'ns'
    if pval < 0.05 and pval >= 0.01:
        s = '*'
    if pval < 0.01 and pval >= 0.001:
        s = '**'
    if pval < 0.001:
        s = '***'
    return s


def p_corr_vs_noise(n_trials=2000, n_iters=300, noiselist=np.arange(0.1, 0.525, 0.025),
                    j_vals=[0.1, 0.45, 0.8], b_list=[0.1, 0.4, 0.8], load_data=True):
    if load_data:
        acc_vs_j_noise = np.load(SV_FOLDER + 'acc_vs_j_noise_b.npy')
        err_acc_j_noise = np.load(SV_FOLDER + 'acc_sd_vs_j_noise_b.npy')
    # stimulus = np.random.choice([-1, -0.8, -0.4, 0, 0.4, 0.8, 1])
    else:
        acc_vs_j_noise = np.zeros((len(j_vals), len(noiselist), len(b_list)))
        err_acc_j_noise = np.zeros((len(j_vals), len(noiselist), len(b_list)))
        for i_b, b in enumerate(b_list):
            for i_j, j in enumerate(j_vals):
                accuracy = []
                err = []
                for i_n, noise in enumerate(noiselist):
                    hit = []
                    for i in range(n_trials):
                        q = np.random.rand()
                        # q = np.min((np.max((0.5 + np.random.randn()*0.1, 0)), 1))
                        for _ in range(n_iters):
                            q = dyn_sys_mf(q, dt=1e-2, j=j, bias=b, n=3.92, sigma=noise,
                                           tau=0.5)
                        hit.append(np.sign(q-0.5) == 1)
                    accuracy.append(np.nanmean(hit))
                    err.append(np.nanstd(hit) / np.sqrt(n_trials))
                acc_vs_j_noise[i_j, :, i_b] = accuracy
                err_acc_j_noise[i_j, :, i_b] = err
        np.save(SV_FOLDER + 'acc_vs_j_noise_b.npy', acc_vs_j_noise)
        np.save(SV_FOLDER + 'acc_sd_vs_j_noise_b.npy', err_acc_j_noise)
    fig, ax = plt.subplots(ncols=4, figsize=(16, 4.5))
    colormap = pl.cm.Oranges(np.linspace(0.2, 1, len(j_vals)))
    titles = ['J=0.1', 'J=0.45', 'J=0.8']
    ax[-1].spines['right'].set_visible(False)
    ax[-1].spines['top'].set_visible(False)
    for i_a, a in enumerate(ax[:-1]):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        for i in range(len(b_list)):
            a.plot(noiselist, acc_vs_j_noise[i_a, :, i], color=colormap[i], linewidth=2.5,
                   label=f'B={b_list[i]}')
            a.fill_between(noiselist, acc_vs_j_noise[i_a, :, i]-err_acc_j_noise[i_a, :, i],
                           acc_vs_j_noise[i_a, :, i]+err_acc_j_noise[i_a, :, i], color=colormap[i],
                           alpha=0.3)
        a.plot(noiselist, np.mean(acc_vs_j_noise[i_a, :, :], axis=1), color='k', linewidth=2.5,
               label='avg.', linestyle='--')
        a.set_xlabel(r'$\sigma$')
        a.set_title(titles[i_a])
        a.set_ylim(0.49, 1.01)
        ax[-1].plot(noiselist, np.mean(acc_vs_j_noise[i_a, :, :], axis=1), color='k', linewidth=2.5,
                    label='avg.', linestyle='--', alpha=0.4)
    ax[-1].set_ylim(0.49, 1.01)
    ax[-1].plot(noiselist, np.mean(np.mean(acc_vs_j_noise, axis=0), axis=1), color='k', linewidth=2.5)
    ax[-1].set_xlabel(r'$\sigma$')
    ax[0].legend(frameon=False)
    ax[0].set_ylabel('p(correct)')
    fig.tight_layout()
    # ax.set_xscale('log')


def plot_noise_vs_b_accuracy(n_trials=2000, n_iters=300, noiselist=np.arange(0.1, 0.55, 0.05),
                             j=0.5, b_list=np.arange(0, 0.9, 0.1), load_data=True):
    if load_data:
        acc_vs_noise = np.load(SV_FOLDER + 'acc_vs_noise_and_b_jbis.npy')
        err_acc_noise = np.load(SV_FOLDER + 'acc_sd_vs_noise_and_b_jbis.npy')
    # stimulus = np.random.choice([-1, -0.8, -0.4, 0, 0.4, 0.8, 1])
    else:
        acc_vs_noise = np.zeros((len(noiselist), len(b_list)))
        err_acc_noise = np.zeros((len(noiselist), len(b_list)))
        for i_b, b in enumerate(b_list):
            accuracy = []
            err = []
            for i_n, noise in enumerate(noiselist):
                hit = []
                for i in range(n_trials):
                    q = np.random.rand()
                    # q = np.min((np.max((0.5 + np.random.randn()*0.1, 0)), 1))
                    for _ in range(n_iters):
                        q = dyn_sys_mf(q, dt=1e-2, j=j, bias=b, n=3.92, sigma=noise,
                                       tau=0.5)
                    hit.append(np.sign(q-0.5) == 1)
                accuracy.append(np.nanmean(hit))
                err.append(np.nanstd(hit) / np.sqrt(n_trials))
            acc_vs_noise[:, i_b] = accuracy
            err_acc_noise[:, i_b] = err
        np.save(SV_FOLDER + 'acc_vs_noise_and_b_jbis.npy', acc_vs_noise)
        np.save(SV_FOLDER + 'acc_sd_vs_noise_and_b_jbis.npy', err_acc_noise)
    fig, ax = plt.subplots(ncols=3, figsize=(14, 4.5))
    im = ax[0].imshow(np.flipud(acc_vs_noise), aspect='auto',
                      extent=[0, 0.8, 0, 0.5], cmap='Greens')
    plt.colorbar(im, label='P(Correct)', ax=ax[0])
    ax[0].set_xlabel('B')
    ax[0].set_ylabel('Noise')
    colormap = pl.cm.Oranges(np.linspace(0.2, 1, len(b_list)))
    for i_b, b in enumerate(b_list):
        ax[1].plot(noiselist, acc_vs_noise[:, i_b], label=b, linewidth=2.5, color=colormap[i_b])
        ax[1].fill_between(noiselist, acc_vs_noise[:, i_b]-err_acc_noise[:, i_b],
                           acc_vs_noise[:, i_b]+err_acc_noise[:, i_b], color=colormap[i_b],
                           alpha=0.3)
    ax[1].plot(noiselist, np.mean(acc_vs_noise, axis=1), label='avg.', color='k',
               linestyle='--', linewidth=2.5)
    ax[1].set_xlabel('Noise')
    ax[1].set_ylabel('P(Correct)')
    mean_acc_per_noise = np.mean(acc_vs_noise, axis=1)
    cmap = mpl.cm.Oranges
    norm = mpl.colors.Normalize(vmin=-0.2, vmax=0.8)
    color = cmap(norm(b_list))
    ax[2].scatter(noiselist, mean_acc_per_noise, c=color, marker='o')
    ax[2].set_xlabel('Noise')
    ax[2].set_ylabel('P(Correct)')
    fig.tight_layout()


def log_prior(pars):
    # if prior is added, worse fits may happen, but better params distros (kind of forced?)
    if len(pars) == 4:
        jpar, b1par, biaspar, noise = pars
        lp = (np.log(scipy.stats.norm.pdf(jpar, 1/3.92/2, 1)+1e-12) +  # J1 prior
                np.log(scipy.stats.norm.pdf(b1par, 0.5, 1.)+1e-12) +  # B1 prior
                np.log(scipy.stats.norm.pdf(biaspar, 0., 0.8)+1e-12)+  # B0 prior
                np.log(scipy.stats.uniform.pdf(noise, 0.1, 0.5)+1e-12))  # Uniform prior for sigma (positive only)
    else:
        jpar, jbiaspar, b1par, biaspar, noise = pars
        lp = (np.log(scipy.stats.norm.pdf(jpar, 1/3.92/2, 1)+1e-12) +  # J1 prior
                np.log(scipy.stats.norm.pdf(jbiaspar, 0., 0.4)+1e-12) + # JBias
                np.log(scipy.stats.norm.pdf(b1par, 0.5, 1.)+1e-12) +  # B1 prior
                np.log(scipy.stats.norm.pdf(biaspar, 0., 0.8)+1e-12)+  # B0 prior
                np.log(scipy.stats.uniform.pdf(noise, 0.1, 0.5)+1e-12))  # Uniform prior for sigma (positive only)
    return lp


def plot_bic_across_models(sv_folder=SV_FOLDER,
                           bic=True, method='BADS'):
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    subjects = all_df.subject.unique()
    llh_all = np.zeros((4, len(subjects)))
    models = ['MF5', 'MF', 'LBP5', 'LBP']
    for i_e, extra in enumerate(['', 'null', '', 'null']):
        model = models[i_e]
        for i_s, sub in enumerate(subjects):
            # print(sub)
            dataframe = all_df.copy().loc[all_df['subject'] == sub]
            unique_vals = np.sort(dataframe['pShuffle'].unique())
            if extra != 'null':
                dataframe['coupling'] = dataframe['pShuffle'].replace(to_replace=unique_vals,
                                            value= [1., 0.3, 0.]) 
            else:
                dataframe['coupling'] = 1
            dataframe['confidence'] = (transform(dataframe.confidence.values, -0.999, 0.999)+1)/2
            dataframe['stim_str'] = (dataframe.evidence.values)
            dataframe['stim_ev_cong'] = dataframe.stim_str * dataframe.response
            data = dataframe[['coupling', 'confidence', 'stim_str', 'stim_ev_cong']]
            if method == 'BADS':
                appendix = '_BADS'
            else:
                appendix = ''
            pars = np.load(sv_folder + '/parameters_'+model+ appendix+ sub + extra + '.npy')
            # negative log likelihood
            if model in ['MF', 'MF5']:
                nllh = optimization(data=data, n_iters=10).nlh_boltzmann_mf(pars=pars)
            if model in ['LBP', 'LBP5']:
                nllh = optimization(data=data, n_iters=10).nlh_boltzmann_lbp(pars=pars)
            if bic:
                numpars = 5 if model in ['MF5', 'FBP', 'LBP5'] else 4
                nllh = numpars*np.log(len(dataframe))+2*(nllh)  # +log_prior(pars)*len(dataframe))
            llh_all[i_e, i_s] = nllh
    fig, ax = plt.subplots(1)
    sns.barplot(llh_all.T, ax=ax, linewidth=2.5, edgecolor=".5", facecolor=(0, 0, 0, 0))
    ax.set_xticks([0, 1, 2, 3], ['MF', 'MF-null', 'LBP', 'LBP-null'], rotation=45)
    ax.axhline(0, linestyle='--', color='k', alpha=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for i_s in range(len(subjects)):
        jitter = np.random.randn(4)*0.03
        ax.plot([jitter[0], 1+jitter[1], 2+jitter[2], 3+jitter[3]],
                [llh_all[0, i_s], llh_all[1, i_s], llh_all[2, i_s], llh_all[3, i_s]],
                         color='k', marker='o', linestyle='', markersize=4, alpha=0.6)
    if bic:
        ax.set_ylabel(r'$BIC$')
    else:
        ax.set_ylabel('Negative log-likelihood')
    fig.tight_layout()


def plot_log_likelihood_difference(sv_folder=SV_FOLDER, mcmc=False,
                                   model='MF', bic=True,
                                   plot_all=False, method='nelder-mead'):
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    subjects = all_df.subject.unique()
    llh_all = np.zeros((3, len(subjects)))
    for i_e, extra in enumerate(['', 'null']):
        if mcmc:
            path = SV_FOLDER + extra + 'MCMC_fitted_'+model+'_parameters.npy'
            params = np.load(path)
        if extra == 'null':
            model = 'MF' if model == 'MF5' else 'LBP'
        for i_s, sub in enumerate(subjects):
            # print(sub)
            dataframe = all_df.copy().loc[all_df['subject'] == sub]
            unique_vals = np.sort(dataframe['pShuffle'].unique())
            if extra != 'null':
                dataframe['coupling'] = dataframe['pShuffle'].replace(to_replace=unique_vals,
                                            value= [1., 0.3, 0.]) 
            else:
                dataframe['coupling'] = 1
            dataframe['confidence'] = (transform(dataframe.confidence.values, -0.999, 0.999)+1)/2
            dataframe['stim_str'] = (dataframe.evidence.values)
            dataframe['stim_ev_cong'] = dataframe.stim_str * dataframe.response
            data = dataframe[['coupling', 'confidence', 'stim_str', 'stim_ev_cong']]
            if mcmc:
                pars = params[:, i_s]
            if not mcmc:
                if method == 'BADS':
                    appendix = '_BADS'
                else:
                    appendix = ''
                pars = np.load(sv_folder + '/parameters_'+model+ appendix+ sub + extra + '.npy')
            # negative log likelihood
            if model in ['MF', 'MF5']:
                nllh = optimization(data=data, n_iters=10).nlh_boltzmann_mf(pars=pars)
            if model in ['LBP', 'LBP5']:
                nllh = optimization(data=data, n_iters=10).nlh_boltzmann_lbp(pars=pars)
            if bic:
                numpars = 5 if model in ['MF5', 'FBP', 'LBP5'] else 4
                nllh = numpars*np.log(len(dataframe))+2*(nllh)  # +log_prior(pars)*len(dataframe))
            llh_all[i_e, i_s] = nllh
    llh_all[2] = llh_all[1]-llh_all[0]
    print(subjects[np.where(llh_all[2]< -20)[0]])
    print('Average \Delta BIC (Null-Full):')
    print(np.mean(llh_all[2]))
    print('Median \Delta BIC (Null-Full):')
    print(np.median(llh_all[2]))
    fig, ax = plt.subplots(1)
    if plot_all:
        ax.boxplot(llh_all.T)
        ax.set_xticks([1, 2, 3], ['Full', 'Null', 'Null-Full'], rotation=45)
    else:
        ax.boxplot(llh_all[2])
    ax.axhline(0, linestyle='--', color='k', alpha=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for i_s in range(len(subjects)):
        jitter = np.random.randn(3)*0.03
        if plot_all:
            ax.plot([1+jitter[0], 2+jitter[1]], [llh_all[0, i_s], llh_all[1, i_s]],
                     color='gray', alpha=0.4)
            ax.plot([1+jitter[0], 2+jitter[1]], [llh_all[0, i_s], llh_all[1, i_s]],
                     color='k', marker='o', linestyle='', markersize=4, alpha=0.6)
            ax.plot([1+jitter[0], 2+jitter[1], 3+jitter[2]],
                    [llh_all[0, i_s], llh_all[1, i_s], llh_all[2, i_s]],
                     color='k', marker='o', linestyle='', markersize=4, alpha=0.6)
        else:
            ax.plot([1+jitter[0]], [llh_all[2, i_s]],
                     color='k', marker='o', linestyle='', markersize=4, alpha=0.6)
    if bic:
        ax.set_ylabel(r'$\Delta BIC$')
    else:
        ax.set_ylabel('Negative log-likelihood')
    p = scipy.stats.ttest_ind(llh_all[0], llh_all[1]).pvalue
    x1, x2 = 1, 2
    p = scipy.stats.ttest_1samp(llh_all[2], popmean=0).pvalue
    if plot_all:
        y, h, col = max(map(max, llh_all)) + 150, 80, 'k'
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        ax.text((x1+x2)*.5, y+h, f"p = {p:.2e}", ha='center', va='bottom', color=col)
        ax.set_ylim(np.min((np.min(llh_all[2])-200, -10)), y+200)
        ax.text(3, np.max(llh_all[2])+50, f"p = {p:.2e}", ha='center', va='bottom', color=col)
    else:
        ax.text(1, np.max(llh_all[2])+10, f"p = {p:.2e}", ha='center', va='bottom', color='k')
        if bic:
            ax.set_xticks([1], [r'$BIC(Null)-BIC(Full)$'])
        else:
            ax.set_xticks([1], [r'$NLLH(Null)-NLLH(Full)$'])
    fig.tight_layout()


def mcmc_all_subjects(plot=True, burn_in=100, iterations=20000, load_params=True,
                      extra=''):
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    subjects = all_df.subject.unique()
    params = np.zeros((4, len(subjects)))
    jvals = np.arange(-1, 2, 1e-2)
    jdist = scipy.stats.norm.pdf(jvals, 1/3.92, 1)
    b1vals = np.arange(-1, 2, 1e-2)
    b1dist = scipy.stats.norm.pdf(b1vals, 0.5, 1)
    noisevals = np.arange(0.05, 0.65, 1e-3)
    noisedist = scipy.stats.uniform.pdf(noisevals, 0.1, 0.5)
    b0vals = np.arange(-2, 2, 1e-2)
    b0dist = scipy.stats.norm.pdf(b0vals, 0., 0.6)
    vals = [jvals, b1vals, b0vals, noisevals]
    dists = [jdist, b1dist, b0dist, noisedist]
    path = SV_FOLDER + extra + 'MCMC_fitted_MF_parameters.npy'
    if not load_params:
        for i_s, sub in enumerate(subjects):
            print(sub)
            dataframe = all_df.copy().loc[all_df['subject'] == sub]
            unique_vals = np.sort(dataframe['pShuffle'].unique())
            if extra == '':
                dataframe['coupling'] = dataframe['pShuffle'].replace(to_replace=unique_vals,
                                                                      value= [1., 0.3, 0.])
            else:
                dataframe['coupling'] = 1
            dataframe['confidence'] = (transform(dataframe.confidence.values, -0.999, 0.999)+1)/2
            dataframe['stim_str'] = (dataframe.evidence.values)
            dataframe['stim_ev_cong'] = dataframe.stim_str * dataframe.response
            data = dataframe[['coupling', 'confidence', 'stim_str', 'stim_ev_cong']]
            states = optimization(data=data, n_iters=10).mcmc(initial_params=[0.5, 0.4, 0.1, 0.2],
                                                              iterations=iterations+burn_in)
            # params[:, i_s] = scipy.stats.mode(states[burn_in:]).mode
            params[:, i_s] = np.median(states[burn_in:], axis=0)
            if plot:
                plot_mcmc_individual(states, vals, dists, burn_in)
        np.save(path, params)
    else:
        params = np.load(path)
    parmat_bads = np.zeros((4, len(subjects)))
    for i_s, sub in enumerate(subjects):
        pars = np.load(SV_FOLDER + '/parameters_MF_BADS' + sub + extra + '.npy')
        parmat_bads[:, i_s] = pars
    if plot:
        numpars = 4
        nsubs = len(subjects)
        fig, ax = plt.subplots(ncols=numpars, figsize=(15,5))
        labels = ['Coupling, J', 'Stimulus weight, B1', 'Bias, B0', 'noise', 'Alpha'][:numpars]
        for i in range(numpars):
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['bottom'].set_visible(False)
            sns.violinplot(params[i, :], color='lightblue', alpha=0.3, ax=ax[i])
            ax[i].plot(np.random.randn(nsubs)*0.05, params[i, :], marker='o', color='k', linestyle='', markersize=4)
            ax[i].set_xticks([])
            ax[i].set_ylabel(labels[i])
        fig.tight_layout()
        fig2, ax2 = plt.subplots(ncols=numpars, figsize=(15,5))
        labels = ['Coupling, J', 'Stimulus weight, B1', 'Bias, B0', 'noise', 'Alpha'][:numpars]
        lims = [[0.1, 1.5], [0, 1.7], [-0.6, 1.5], [0, 0.7]]
        for i in range(numpars):
            ax2[i].spines['top'].set_visible(False)
            ax2[i].spines['right'].set_visible(False)
            ax2[i].plot(params[i, :], parmat_bads[i, :], color='k', marker='o',
                        linestyle='')
            ax2[i].set_ylabel(labels[i] + ', MCMC')
            ax2[i].set_xlabel(labels[i] + ', BADS')
            ax2[i].plot(lims[i], lims[i], color='gray', alpha=0.5)
            ax2[i].set_ylim(lims[i])
            ax2[i].set_xlim(lims[i])
        fig2.tight_layout()


def plot_mcmc_individual(states, vals, dists, burn_in):
    fig, ax = plt.subplots(4)
    for a in range(4):
        ax[a].plot(states[:, a])
    df = pd.DataFrame(states[burn_in:], columns=['J', 'B1', 'B0', 'noise'])
    axes = sns.pairplot(df)
    ax = axes.axes
    for i in range(4):
        for j in range(4):
            if i == j:
                ax[i, j].plot(vals[i], dists[i], color='k', linewidth=2.5)
                par = np.median(states[burn_in:, i])
                ax[i, j].axvline(par, color='r')
            else:
                joint_dist = np.outer(dists[i], dists[j])
                ax[i, j].contour(vals[j], vals[i], joint_dist,
                                 color='k', alpha=0.8, levels=5)
    fig, ax = plt.subplots(ncols=2, nrows=2)
    ax = ax.flatten()
    for i in range(4):
        pd.plotting.autocorrelation_plot(df[i], ax=ax[i])
    fig.tight_layout()


def plot_all_subjects(xvar='stim_ev_cong'):
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    unique_vals = np.sort(all_df['pShuffle'].unique())
    all_df['coupling'] = all_df['pShuffle'].replace(to_replace=unique_vals,
                                value= [1., 0.3, 0.])
    all_df['stim_str'] = all_df.evidence.values
    all_df['stim_ev_cong'] = all_df.stim_str * all_df.response
    subjects = all_df.subject.unique()
    fig, ax = plt.subplots(ncols=8, nrows=4, figsize=(19, 12))
    ax = ax.flatten()
    fig4, ax4 = plt.subplots(ncols=8, nrows=4, figsize=(19, 12))
    ax4 = ax4.flatten()
    fig5, ax5 = plt.subplots(ncols=8, nrows=4, figsize=(19, 12))
    ax5 = ax5.flatten()
    df_sub = pd.DataFrame({})
    # fig6, ax6 = plt.subplots(ncols=8, nrows=4, figsize=(19, 12))
    # ax6 = ax6.flatten()
    for i_s, sub in enumerate(subjects):
        ax[i_s].spines['right'].set_visible(False)
        ax[i_s].spines['top'].set_visible(False)
        ax5[i_s].spines['right'].set_visible(False)
        ax5[i_s].spines['top'].set_visible(False)
        dataframe = all_df.copy().loc[all_df['subject'] == sub]
        dataframe['confidence'] = (transform(dataframe.confidence.values, -0.999, 0.999)+1)/2
        dataframe['abs_confidence'] = np.abs(dataframe.confidence-0.5)*2
        df_sub = pd.concat((df_sub, dataframe[['stim_ev_cong', 'coupling', 'abs_confidence', 'subject', 'stim_str']]))
        l = True if i_s == 0 else False
        sns.lineplot(dataframe, x=xvar, y='abs_confidence',
                     hue='coupling', ax=ax[i_s], legend=l, errorbar=('se'))
        sns.lineplot(dataframe, x='coupling', y='abs_confidence',
                     ax=ax4[i_s], legend=l, errorbar=('se'), color='k')
        sns.histplot(dataframe, x='confidence', legend=False, ax=ax5[i_s], bins=25)
        ax5[i_s].axvline(0.5, color='r', linestyle='--', alpha=0.4)
        ax5[i_s].set_xlabel('Confidence')
        # for p_shuffle in [0., 0.3, 1.0]:
        #     ax6[i_s].plot([p_shuffle]*len(dataframe.loc[(dataframe.coupling == p_shuffle) & (dataframe.stim_str == 0)]),
        #                   dataframe.loc[(dataframe.coupling == p_shuffle) & (dataframe.stim_str == 0)].confidence,
        #                   color='k', linestyle='', marker='o')
        if i_s > 23:
            if xvar == 'stim_ev_cong':
                ax[i_s].set_xlabel('Stim. ev. cong.')
            if xvar == 'stim_str':
                ax[i_s].set_xlabel('Stimulus strength')
        else:
            ax[i_s].set_xlabel('')
        if i_s in [0, 8, 16, 24]:
            ax[i_s].set_ylabel('Confidence')
        else:
            ax[i_s].set_ylabel('')
        if xvar == 'stim_str':
            ax[i_s].set_xticks([0, 1])
        if xvar == 'stim_ev_cong':
            ax[i_s].set_xticks([-1, 0, 1])
        ax[i_s].set_ylim([0, 1])
        # ax4[i_s].set_ylim([0, 1])
        ax[i_s].set_yticks([0, 0.5, 1])
    fig.tight_layout()
    fig5.tight_layout()
    # fig6.tight_layout()
    fig4.tight_layout()
    fig.savefig(SV_FOLDER + 'all_subjects_abs_conf.png', dpi=200, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'all_subjects_abs_conf.svg', dpi=200, bbox_inches='tight')
    fig2, ax2 = plt.subplots(1)
    df_sub_final = df_sub.dropna().reset_index()
    # df_sub_final['abs_confidence'] = scipy.stats.zscore(df_sub_final.abs_confidence.values)
    # Compute the mean confidence per subject for each (stim_ev_cong, coupling) pair
    df_subject_avg = df_sub_final.groupby(['subject', 'stim_ev_cong', 'coupling'])['abs_confidence'].mean().reset_index()
    # Plot
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    sns.lineplot(data=df_subject_avg, x='stim_ev_cong', y='abs_confidence', hue='coupling', ax=ax2,
                 errorbar=('se'))
    # Perform statistical tests at each stim_ev_cong level
    alpha = 0.01  # Significance threshold
    significant_x = []  # Store x positions where significant differences occur
    significant_x2 = []
    significant_x3 = []
    df = df_subject_avg.copy()
    # stim_values = df['stim_ev_cong'].unique()
    coupling_values = df['coupling'].unique()
    
    for stim in [0, 0.4, 0.8, 1]:
        groups = []
        
        for coup in coupling_values:
            # Extract confidence values for each coupling at a given stim_ev_cong
            conf_values = df_subject_avg.loc[
                (df_subject_avg['stim_ev_cong'] == stim) & (df_subject_avg['coupling'] == coup),
                'abs_confidence'].values
            
            groups.append(conf_values)
    
        stat, p_val = scipy.stats.ttest_rel(groups[0], groups[2])  # Paired t-test across subjects
        stat, p_val2 = scipy.stats.ttest_rel(groups[0], groups[1])  # Paired t-test across subjects
        stat, p_val3 = scipy.stats.ttest_rel(groups[2], groups[1])  # Paired t-test across subjects
    
        # If p-value is below alpha, mark this stim_ev_cong as significant
        if p_val < alpha:
            significant_x.append(stim)
        if p_val2 < alpha:
            significant_x3.append(stim)
        if p_val3 < alpha:
            significant_x2.append(stim)
    # Add horizontal lines on top for significant differences
    colormap = sns.color_palette("rocket", as_cmap=True)
    cmap = colormap([0, 0.5, 1])
    if significant_x:
        for i, sig_x in enumerate([significant_x, significant_x2, significant_x3]): 
            y_max = 0.3 + i*0.04 # Position above highest confidence
            if sig_x == [0, 0.4, 0.8, 1] or sig_x == [0, 0.4, 0.8]:
                ax2.plot(sig_x, [y_max]*len(sig_x),
                         color=cmap[i], linewidth=4)
            else:
                for x in sig_x:
                    plt.plot([x - 0.2, x], [y_max, y_max],
                             color=cmap[i], linewidth=4)  # Short line above plot
    ax2.set_xlabel('Stim. ev. cong')
    ax2.set_ylabel('Absolute confidence')
    fig2.tight_layout()
    fig.savefig(SV_FOLDER + 'mean_across_all_subjects_abs_conf.png', dpi=200, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'mean_across_all_subjects_abs_conf.svg', dpi=200, bbox_inches='tight')


def plot_models_predictions(sv_folder=SV_FOLDER, model='MF5', method='Powell',
                            variable='abs_confidence'):
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    data_orig, data_model_orig, data_model_null =\
        load_all_data(all_df, model=model, method=method, sv_folder=SV_FOLDER)
    data_orig['decision'] = (data_orig.response.values+1)/2
    data_model_orig['decision'] = (data_model_orig.decision.values+1)/2
    data_model_null['decision'] = (data_model_null.decision.values+1)/2
    df_sub_final = data_orig.dropna().reset_index()
    # Compute the mean confidence per subject for each (stim_ev_cong, coupling) pair
    df_subject_avg = df_sub_final.groupby(['subject', 'coupling'])[variable].mean().reset_index()
    df_sub_model = data_model_orig.dropna().reset_index()
    # Compute the mean confidence per subject for each (stim_ev_cong, coupling) pair
    df_model_subject_avg = df_sub_model.groupby(['subject', 'coupling'])[variable].mean().reset_index()
    df_sub_null = data_model_null.dropna().reset_index()
    # Compute the mean confidence per subject for each (stim_ev_cong, coupling) pair
    df_null_subject_avg = df_sub_null.groupby(['subject', 'coupling'])[variable].mean().reset_index()
    fig2, ax2 = plt.subplots(ncols=3, nrows=2, figsize=(9, 5.5))
    ax2 = ax2.flatten()
    if variable != 'decision':
        lab = 'conf.'
    else:
        lab = 'p(right)'
    titles = ['Model '+lab, 'Null model '+lab]
    for a in ax2:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_xlim(-0.05, 1.05)
        a.set_ylim(-0.05, 1.05)
        a.plot([-0.05, 1.05], [-0.05, 1.05], color='k', alpha=0.4)
    for i_c, c in enumerate([0, 0.3, 1]):
        df_data_coup = df_subject_avg.loc[df_subject_avg.coupling == c]
        df_model_coup = df_model_subject_avg.loc[df_model_subject_avg.coupling == c]
        df_null_coup = df_null_subject_avg.loc[df_null_subject_avg.coupling == c]
        # for i_s, stim in enumerate([0, 0.4, 0.8, 1]):
        conf_data = df_data_coup[variable].values
        conf_model = df_model_coup[variable].values
        conf_null = df_null_coup[variable].values
        ro_model = scipy.stats.pearsonr(conf_data, conf_model).statistic
        ro_null = scipy.stats.pearsonr(conf_data, conf_null).statistic
        # ro_model = scipy.stats.linregress(conf_data, conf_model).slope
        # ro_null = scipy.stats.linregress(conf_data, conf_null).slope
        ax2[i_c].set_title(f'Shuffling = {100*(1-c)}%', fontsize=15)
        ax2[i_c].plot(conf_data, conf_model, marker='o', color='k', linestyle='')
        ax2[i_c].text(0.06, 0.84, rf'$\rho = $ {round(ro_model, 3)}',  # , p={ro_model.pvalue: .3e}
                      fontsize=14)
        ax2[i_c+3].plot(conf_data, conf_null, marker='o', color='r', linestyle='')
        ax2[i_c+3].text(0.06, 0.84, rf'$\rho = $ {round(ro_null, 3)}',  # , p={ro_model.pvalue: .3e}
                        fontsize=14)
        ax2[i_c+3].set_xlabel('Data ' + lab, fontsize=15)
    ax2[0].set_ylabel(titles[0], fontsize=15)
    ax2[3].set_ylabel(titles[1], fontsize=15)
    fig2.tight_layout()
    fig2.savefig(SV_FOLDER + 'full_comparison_shuffling_' + lab + '.png', dpi=200, bbox_inches='tight')
    fig2.savefig(SV_FOLDER + 'full_comparison_shuffling_' + lab + '.svg', dpi=200, bbox_inches='tight')
    # avg. psychometrics
    df_subject_avg = df_sub_final.groupby(['subject', 'coupling', 'stim_str'])['decision'].mean().reset_index()
    df_model_subject_avg = df_sub_model.groupby(['subject', 'coupling', 'stim_str'])['decision'].mean().reset_index()
    df_null_subject_avg = df_sub_null.groupby(['subject', 'coupling', 'stim_str'])['decision'].mean().reset_index()
    fig, ax = plt.subplots(ncols=3, figsize=(9, 4))
    titles = ['Data', 'Model', 'Null']
    for i_a, a in enumerate(ax):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylim(0, 1)
        a.set_title(titles[i_a], fontsize=15)
    sns.lineplot(df_subject_avg, x='stim_str', y='decision', hue='coupling', errorbar=('se'), ax=ax[0],
                 legend=True, marker='o')
    sns.lineplot(df_model_subject_avg, x='stim_str', y='decision', hue='coupling', errorbar=('se'), ax=ax[1],
                 legend=False, marker='o')
    sns.lineplot(df_null_subject_avg, x='stim_str', y='decision', hue='coupling', errorbar=('se'), ax=ax[2],
                 legend=False, marker='o')
    ax[0].set_ylabel('P(rightward)')
    ax[1].set_ylabel('')
    ax[2].set_ylabel('')
    ax[0].set_xlabel('')
    ax[1].set_xlabel('Stimulus evidence')
    ax[2].set_xlabel('')
    fig.tight_layout()


def ridgeplot_all_subs(sv_folder=SV_FOLDER, model='MF5', method='BADS',
                       band_width=0.5, together=False, sort_by_j=False):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    model_nonull = 'MF5' if model in ['MF', 'MF5'] else 'LBP5'
    data_orig, data_model_orig, data_model_null =\
        load_all_data(all_df, model=model_nonull, method=method, sv_folder=SV_FOLDER)
    if model in ['MF5', 'LBP5']:
        data_model = data_model_orig
    else:
        data_model = data_model_null
    if method == 'BADS':
        appendix = '_BADS'
    else:
        appendix = ''
    subjects = all_df.subject.unique()
    sum_js = np.zeros((len(subjects)))
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(32, rot=-.25, light=.7)
    if sort_by_j:
        data_orig['sum_js'] = 0
        data_model_orig['sum_js'] = 0
        data_model_null['sum_js'] = 0
        for i_s, sub in enumerate(subjects):
            params_i = np.load(SV_FOLDER + 'parameters_' + 'MF5' + appendix + sub +  '' + '.npy')
            sum_js[i_s] = params_i[0]+params_i[1]
            data_orig.loc[data_orig.subject == sub, 'sum_js'] = params_i[0] + params_i[1]
            data_model_orig.loc[data_model_orig.subject == sub, 'sum_js'] = params_i[0] + params_i[1]
            data_model_null.loc[data_model_null.subject == sub, 'sum_js'] = params_i[0] + params_i[1]
        data_orig = data_orig.sort_values('sum_js')
        data_model_orig = data_model_orig.sort_values('sum_js')
        data_model_null = data_model_null.sort_values('sum_js')
    g = sns.FacetGrid(data_orig, hue="subject", aspect=1, height=2.8, palette=pal,
                      row='subject', sharey=False)  # , col_wrap=4, 
    
    # Draw the densities in a few steps
    g.map(sns.kdeplot, "confidence",
          bw_adjust=band_width, clip_on=False,
          fill=True, alpha=1, linewidth=1.5, common_norm=False)
    g.map(sns.kdeplot, "confidence", clip_on=False, color="w", lw=2, bw_adjust=band_width)
    
    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    subjects_sorted = data_model_orig.subject.unique()
    for i_a, a in enumerate(g.axes):
        a[0].set_title('')
        sns.kdeplot(data_model.loc[(data_model.subject == subjects_sorted[i_a])],
                    x='confidence', ax=a[0],
                    lw=3, color='r', bw_adjust=band_width, common_norm=False)
        if together:
            if model in ['MF5', 'LBP5']:
                data_model_2 = data_model_null
            else:
                data_model_2 = data_model_orig
            sns.kdeplot(data_model_2.loc[(data_model_2.subject == data_model_2.subject.unique()[i_a])],
                        x='confidence', ax=a[0],
                        lw=3, color='r', bw_adjust=band_width, common_norm=False,
                        linestyle='--')
        if i_a not in [16]:
            a[0].set_ylabel('')
    
    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(-0.1, 0.2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    g.map(label, "confidence")
    # g.figure.tight_layout()
    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.3)
    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.axes[0][0].set_xlabel('Confidence')
    g.despine(bottom=True, left=True)
    fig = g.figure
    fig.savefig(SV_FOLDER + model + 'distros_confidence_subjects_model.png', dpi=200, bbox_inches='tight')
    fig.savefig(SV_FOLDER + model + 'distros_confidence_subjects_model.svg', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    opt_algorithm = 'BADS'  # Powell, nelder-mead, BADS, L-BFGS-B
    # plot_parameter_recovery(sv_folder=SV_FOLDER, n_pars=50, model='FBP', method='BADS')
    # fit_subjects(method=opt_algorithm, model='MF', data_augmen=False, n_init=1, extra='null')
    # fit_subjects(method=opt_algorithm, model='MF5', data_augmen=False, n_init=1, extra='')
    # fit_subjects(method=opt_algorithm, model='GS', data_augmen=False, n_init=1, extra='')
    # simulate_subjects(sv_folder=SV_FOLDER, model='MF5', resimulate=True,
    #                   extra='', mcmc=False, method=opt_algorithm, data_augment=False,
    #                   plot_subs=False)
    # simulate_subjects(sv_folder=SV_FOLDER, model='MF', resimulate=True,
    #                   extra='null', mcmc=False, method=opt_algorithm, data_augment=False,
    #                   plot_subs=False)
    # plot_fitted_params(sv_folder=SV_FOLDER, model='MF5', method=opt_algorithm,
    #                     subjects='separated')
    # plot_log_likelihood_difference(sv_folder=SV_FOLDER, mcmc=False, model='MF5', method=opt_algorithm,
    #                                bic=True)
    # plot_all_subjects()
    # plot_models_predictions(sv_folder=SV_FOLDER, model='MF5', method=opt_algorithm)
    # plot_conf_vs_coupling_3_groups(method=opt_algorithm, model='MF5', extra='', bw=0.7,
    #                                data_only=True)
    # plot_conf_vs_coupling_3_groups(method=opt_algorithm, model='MF5', extra='', bw=0.7,
    #                                 data_only=False)
    # plot_bic_across_models(sv_folder=SV_FOLDER, bic=True, method='BADS')
    # plot_density(num_iter=100, model='MF5', extra='', method=opt_algorithm)
    # plot_density(num_iter=100, model='MF', extra='null', method=opt_algorithm)
    # plot_density_comparison(num_iter=100, method=opt_algorithm, kde=False)
    # plot_density_comparison(num_iter=100, method=opt_algorithm, kde=True, stim_ev_0=True,
    #                         variable='aligned_confidence', bw=0.6, model='LBP5')
    # plot_density_comparison(num_iter=100, method=opt_algorithm, kde=True, stim_ev_0=True,
    #                         variable='aligned_confidence', bw=0.7, model='MF5')
    # plot_regression_weights(sv_folder=SV_FOLDER, load=True, model='MF5',
    #                         method=opt_algorithm)
    # ridgeplot_all_subs(sv_folder=SV_FOLDER, model='LBP5', method=opt_algorithm,
    #                     band_width=0.7)
    # ridgeplot_all_subs(sv_folder=SV_FOLDER, model='MF', method=opt_algorithm,
    #                     band_width=0.7, sort_by_j=True)
    plot_confidence_vs_stim(method='BADS', variable='confidence', subject='s_11', plot_all=False,
                            bw=0.8, annot=False, model_density=True)  # good: 11, 7, 15, 18, 23, 30
    # mcmc_all_subjects(plot=True, burn_in=100, iterations=1000, load_params=True,
    #                   extra='null')
    # mcmc_all_subjects(plot=True, burn_in=100, iterations=1000, load_params=True,
    #                   extra='')
    # plot_noise_vs_b_accuracy(n_trials=2000, n_iters=300, noiselist=np.arange(0.1, 0.55, 0.05),
    #                          j=0.5, b_list=np.arange(0, 0.9, 0.1), load_data=True)
    # p_corr_vs_noise(n_trials=2000, n_iters=300, noiselist=np.arange(0.1, 0.525, 0.025),
    #                 j_vals=[0.1, 0.45, 0.8], b_list=[0.1, 0.4, 0.8], load_data=True)
    # parameter_recovery(n_pars=50, sv_folder=SV_FOLDER,
    #                     theta=THETA, n_iters=500, n_trials=500,
    #                     model='MF5', method='BADS')
    # parameter_recovery(n_pars=50, sv_folder=SV_FOLDER,
    #                     theta=THETA, n_iters=500, n_trials=500,
    #                     model='MF', method='BADS')
