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
from sklearn import linear_model


THETA = np.array([[0 ,1 ,1 ,0 ,1 ,0 ,0 ,0], [1, 0, 0, 1, 0, 1, 0, 0],
                  [1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1],
                  [0, 0, 1, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 1, 1, 0]])


DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/fitting/data/'  # Alex
SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/fitting/parameters/'  # Alex


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


    def nlh_boltzmann_lbp(self, pars, n=3.92, eps=1e-3, conts=0.5):
        jpar, b1par, biaspar, noise  = pars
        coupling, stim_str, confidence = self.coupling, self.stim_str, self.confidence
        # for jpar in np.arange(0.1, 2, 0.2):
        j = jpar*np.array(coupling)
        b = b1par*np.array(stim_str)+biaspar
        unique_j = np.unique(j)
        unique_b = np.unique(b)
        tfconf = (0.5*np.log(confidence / (1-confidence))-b)/n
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
                # then Boltzmann is exp(-V) = exp{int{F(q)}}
                norm_cte_i.append(np.exp((scipy.integrate.quad(lambda x: pot_lbp_combs(x, i),
                                                           min_val_integ, m)[0])*2/ (noise*noise)))
                
            norm_cte_combs[i] = np.sum(norm_cte_i)
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
        # nlh_lbp = -np.nansum(boltzman_lbp-np.log(norm_cte))
        # contaminants (?)
        distro = np.exp(boltzman_lbp)/norm_cte
        nlh_lbp = -np.nansum(np.log(distro*(1-eps)+conts*eps))
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
                         penalization_nan=1e3):
        jpar, b1par, biaspar, noise = pars
        coupling, stim_str, confidence = self.coupling, self.stim_str, self.confidence
        j = jpar*np.array(coupling)
        b = b1par*np.array(stim_str)+biaspar

        q = np.arange(0, 1, 1e-2)  # Define q outside the loop, shape (100,)

        # Reshape j and b to broadcast correctly over q
        j = np.array(j).reshape(-1, 1)  # Reshape j to shape (500, 1)
        b = np.array(b).reshape(-1, 1)  # Reshape b to shape (500, 1)

        # Vectorized version of the potential function over all i and q
        exp_term = 2 * n * (j * (2 * q - 1)) + 2 * b  # Shape: (500, 100)
        log_term = np.log(1 + np.exp(exp_term))  # Shape: (500, 100)

        # Vectorized potential (pot_mf_i) for all i and all q
        # pot_mf = (q*q) / 2 - log_term / (4* n * j)  # Shape: (500, 100)
        pot_mf = np.where(j > 1e-6, (q*q) / 2 - log_term / (4* n * j),
                          q*q/2 - q*sigmoid(2*b))

        # Apply Boltzmann distribution function over the potential values (vectorized)
        bmann_values = np.exp(-2 * pot_mf / (noise*noise))  # Shape: (500, 100)

        # Sum over q for each i to get the normalization constant, norm_cte
        norm_cte = np.sum(bmann_values, axis=1)  # Shape: (500,)

        j = np.array(j).reshape(-1)  # Reshape j to shape (500, 1)
        b = np.array(b).reshape(-1)  # Reshape b to shape (500, 1)
        pot_mf_fun = lambda q: np.where(
                            j > 1e-6,
                            q*q/2 - np.log(1+np.exp(2*n*(j*(2*q-1))+b*2))/(4*n*j), 
                            q*q/2 - q*sigmoid(2*b))
        bmann_distro_log = lambda potential: -2*np.array(potential) / (noise*noise)
        log_likelihood = np.log((1-eps)*np.exp(bmann_distro_log(pot_mf_fun(confidence)))/norm_cte + eps*conts_distro)
        log_likelihood[np.isnan(log_likelihood)] = -penalization_nan
        nlh_mf = -np.sum(log_likelihood)
        # nlh_mf = -np.nansum(bmann_distro_log(pot_mf_fun(confidence)) - np.log(norm_cte))
        log_prior = (
            np.log(scipy.stats.norm.pdf(jpar, 1/3.92, 1)) +  # J1 prior
            np.log(scipy.stats.norm.pdf(b1par, 0.5, 1)) +  # B1 prior
            np.log(scipy.stats.norm.pdf(biaspar, 0., 0.6))+  # B0 prior
            np.log(scipy.stats.uniform.pdf(noise, 0.1, 0.5)+1e-15))  # Uniform prior for sigma (positive only)
        return np.max((nlh_mf, 0)) - log_prior*len(j)


    def nlh_gibbs(self, pars, n=3.92, eps=1e-3, conts_distro=1e-2):
        jpar, b1par, biaspar, time = pars
        coupling, stim_str, confidence = self.coupling, self.stim_str, self.confidence
        j = jpar*np.array(coupling)
        b = b1par*np.array(stim_str)+biaspar
        likelihood = []
        for i in range(len(confidence)):
            k_1 = 12*j[i] + 8*b[i]
            k_2 = 6*j[i] + 6*b[i]
            k_3 = 4*j[i] + 4*b[i]
            k_4 = 2*j[i] + 2*b[i]
            k_5 = 12*j[i] - 8*b[i]
            k_6 = 6*j[i] - 6*b[i]
            k_7 = 4*j[i] - 4*b[i]
            k_8 = 2*j[i] - 2*b[i]
            lamb = sigmoid(k_1-k_2)*sigmoid(k_2-k_3)*sigmoid(k_3-k_4)
            nu = sigmoid(k_5-k_6)*sigmoid(k_6-k_7)*sigmoid(k_7-k_8)
            val = occ_function_markov(lamb, nu, time, confidence[i]*time)
            time_vals = np.arange(0, time+1, 1)
            norm_cte = np.nansum(occ_function_markov(lamb, nu, time, time_vals))
            likelihood.append(val / norm_cte)
        likelihood = np.array(likelihood)
        return -np.nansum(np.log(likelihood*(1-eps) + eps*conts_distro))


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
        assert model in ['MF', 'LBP', 'FBP', 'GS'], 'Model should be either GS, MF, LBP or FBP'
        if model == 'MF':
            fun = self.nlh_boltzmann_mf
            assert len(x0) == 4, 'x0 should have 4 values (J, B1, bias, noise)'
            if method != 'BADS':
                bounds = Bounds([1e-1, -0.2, -0.2, 0.06], [1, 0.25, 0.25, 0.3])
            if method == 'BADS':
                lb = [0.01, -0.2, -0.8, 0.1]
                ub = [2., 2, 0.8, 0.8]
                plb = [0.18, 0.1, -0.6, 0.15]
                pub = [1.2, 0.9, 0.6, 0.3]
        if model == 'GS':
            fun = self.nlh_gibbs
            assert len(x0) == 4, 'x0 should have 4 values (J, B1, bias, time_end)'
            if method != 'BADS':
                bounds = Bounds([1e-1, -1, -1, 0.06, 1], [1, 0.25, 0.25, 0.3, 1e6])
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
                lb = [0.01, -.3, -.3, 0.05]
                ub = [2., 0.3, 0.3, 0.5]
                plb = [0.2, -0.1, -0.1, 0.1]
                pub = [1.4, 0.1, 0.1, 0.2]
        if model == 'FBP':
            fun = self.nlh_boltzmann_fbp
            assert len(x0) == 5, 'x0 should have 5 values (J, B1, bias, noise, alpha)'
            if method != 'BADS':
                bounds = Bounds([1e-1, -.1, -.1, 0.05, 0], [2., .4, .4, 0.3, 1.5])
            if method == 'BADS':
                lb = [0.01, -.3, -.05, 0.05, 0.]
                ub = [2., 0.3, 0.05, 0.5, 2]
                plb = [0.2, -0.1, -0.01, 0.1, 0.6]
                pub = [1.4, 0.1, 0.01, 0.2, 1.3]
        if method != 'BADS':
            optimizer_0 = scipy.optimize.minimize(fun, x0, method=method,
                                                  bounds=bounds)
        if method == 'BADS':
            print('BADS')
            constraint = lambda x: np.abs(x[:, 1]+x[:, 2]) > 0.6
            optimizer_0 = BADS(fun, x0, lb, ub, plb, pub).optimize()  # non_box_cons=constraint
            print(optimizer_0.x)
        # optimizer_0 = scipy.optimize.minimize(fun, x0, method='trust-constr', bounds=bounds)
        # optimizer_0 = scipy.optimize.minimize(fun, x0, method='BFGS', bounds=bounds)
        # optimizer_0 = scipy.optimize.minimize(fun, x0, method='COBYLA', bounds=bounds)
        pars = optimizer_0.x
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


def load_data(data_folder, n_participants=1):
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
            df['subject'] = 's_' + str(i+1)
            df_0 = pd.concat((df_0, df))
        return df_0


def transform(x, minval=0.5, maxval=0.9999):
    maxarray = np.nanmax(x)
    minarray = np.nanmin(x)
    return (maxval-minval)/(maxarray-minarray)*(x-minarray) + minval


def fit_data(optimizer, plot=True, model='MF', n_iters=200, method='nelder-mead'):
    if model == 'FBP':
        numpars = 5
    else:
        numpars = 4
    pars_array = np.empty((n_iters, numpars))
    pars_array[:] = np.nan
    pars_array_0 = np.empty((n_iters, numpars))
    pars_array_0[:] = np.nan
    for k in range(n_iters):
        if k % 2 == 0:
            print(str(round(k/n_iters*100, 2)) + ' %')
        j0 = np.random.uniform(0.6, 1.1)
        b10 = np.random.uniform(0.2, 0.4)
        bias0 = np.random.uniform(-0.15, 0.15)
        noise0 = np.random.uniform(0.15, 0.3)
        if model == 'FBP':
            alpha0 = np.random.uniform(0.1, 1.4)
            x0 = [j0, b10, bias0, noise0, alpha0]
        if model == 'GS':
            time = 10**(np.random.uniform(2, 4))
            x0 = [j0, b10, bias0, time]
        if model in ['LBP', 'MF']:
            x0 = [j0, b10, bias0, noise0]
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
                 n_iter=0, model='FBP', recovery=True, resimulate=False, extra=''):
    vals_conf = []
    decision = []
    if model != 'MF':
        jpar, b1par, biaspar, noise, alpha = pars
        if model == 'LBP':
            alpha = 1
    if model == 'MF':
        jpar, b1par, biaspar, noise = pars
    b = stimulus*b1par + biaspar
    j = coupling*jpar
    if recovery:
        folder = 'param_recovery'
    else:
        folder = 'simulated_data'
    pathdata = sv_folder + folder + '/df_simul' + str(n_iter) + model +  extra + '.csv'
    os.makedirs(os.path.dirname(pathdata), exist_ok=True)
    if os.path.exists(pathdata) and not resimulate:
        data = pd.read_csv(pathdata)
    else:
        if model in ['FBP', 'LBP']:
            for i in range(len(stimulus)):
                # pos, neg = discrete_DBN(j[i], b=b[i], theta=theta, num_iter=n_iters,
                #                         thr=1e-6, alpha=alpha)
                logmess = np.random.randn()/10
                # lm = []
                for _ in range(n_iters):
                    logmess = dyn_sys_fbp(logmess, j[i], b[i], alpha=alpha, n=3, dt=1e-2, noise=noise)
                    # lm.append(logmess)
                posterior_fbp = sigmoid(2*(3*logmess+b[i]))
                # posterior_fbp = np.max((posterior_fbp, 1-posterior_fbp))
                vals_conf.append(posterior_fbp)
                decision.append(np.sign(posterior_fbp-0.5))
        else:
            for i in range(len(stimulus)):
                # q = np.random.rand()
                q = 0.5
                for _ in range(n_iters):
                    q = dyn_sys_mf(q, dt=1e-2, j=j[i], bias=b[i], n=3.92, sigma=noise,
                                   tau=1)
                q_final = q  #  if np.sign(q-0.5) > 0 else 1-q
                vals_conf.append(q_final)
                decision.append(np.sign(q-0.5))
        data = pd.DataFrame({'stim_str': stimulus, 'coupling': coupling,
                             'confidence': vals_conf, 'decision': decision,
                             'stim_ev_cong': decision*stimulus})
        data.to_csv(sv_folder + folder + '/df_simul' + str(n_iter) + model + extra + '.csv')
    return data


def save_params_recovery(n_pars=50, sv_folder=SV_FOLDER,
                         i_ini=0):
    for i in range(i_ini, n_pars):
        j0 = np.random.uniform(0.3, 1.6)
        b10 = np.random.uniform(0.05, 0.3)
        bias0 = np.random.uniform(0.05, 0.3)
        noise0 = np.random.uniform(0.1, 0.2)
        alpha0 = np.random.uniform(0.3, 1.4)
        params = [j0, b10, bias0, noise0, alpha0]
        np.save(sv_folder + 'param_recovery/pars_prt' + str(i) + '.npy',
                np.array(params))


def simulate_subjects(sv_folder=SV_FOLDER,
                      model='MF', resimulate=True, extra='',
                      mcmc=True):
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    subjects = all_df.subject.unique()
    accuracies = []
    accuracies_model = []
    pright_model = []
    parmat = np.zeros((len(subjects), 4))
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
        # data = data_augmentation(data, sigma=0.05, times_augm=4,
        #                          minval=0.001)
        print(len(data))
        if not mcmc:
            pars = np.load(sv_folder + '/parameters_MF_BADS' + sub + extra + '.npy')
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
        fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(10, 13))
        ax = ax.flatten()
        cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)
        if extra == 'null':
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
        fig.savefig(SV_FOLDER + 'ind_simuls/sims_'+sub+extra+'_MF.png', dpi=100, bbox_inches='tight')
        fig.savefig(SV_FOLDER + 'ind_simuls/sims_'+sub+extra+'_MF.svg', dpi=100, bbox_inches='tight')
        plt.close(fig)
    fig2, ax2 = plt.subplots(ncols=4, nrows=3, figsize=(16, 14))
    ax2 = ax2.flatten()
    labels = ['Coupling, J', 'Stimulus weight, B1', 'Bias, B0', 'noise', 'Alpha'][:4]
    for i_a, a in enumerate(ax2):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        if i_a < 4:
            var = pright_model
            lab = 'p(right)'
        if i_a >= 4 < 8:
            var = accuracies_model
            lab = 'p(correct)'
        if i_a >= 8:
            var = conf_model
            lab = 'Confidence'
        # corr = np.corrcoef(var, parmat[:, i_a % numpars])[0, 1]
        corr = scipy.stats.pearsonr(var, parmat[:, i_a % 4])
        if i_a == 7:
            cmap = mpl.cm.Oranges
            norm = mpl.colors.Normalize(vmin=0, vmax=0.8)
            color = cmap(norm(parmat[:, 1]))
            a.scatter(parmat[:, i_a % 4], var, c=color, marker='o')
        else:
            color = 'k'
            a.plot(parmat[:, i_a % 4], var, color=color, marker='o', linestyle='')
        a.set_ylabel(lab)
        a.set_xlabel(labels[i_a % 4])
        a.set_title(rf'$\rho =$ {round(corr.statistic, 3)}, p={corr.pvalue:.1e}')
    fig2.tight_layout()
    fig2.savefig(SV_FOLDER + extra + 'correlations_fitted_parameters_with_model.png', dpi=200, bbox_inches='tight')
    fig2.savefig(SV_FOLDER + extra + 'correlations_fitted_parameters_with_model.svg', dpi=200, bbox_inches='tight')


def parameter_recovery(n_pars=50, sv_folder=SV_FOLDER,
                       theta=THETA, n_iters=2000, n_trials=5000,
                       i_ini=0, model='FBP', method='BADS'):
    # coupling = np.repeat(coupling.values, 3)
    # np.random.shuffle(coupling)
    # stimulus = np.repeat(stimulus.values, 3)
    coupling_values = [0.2, 0.5, 1]
    stimulus_values = [-1, -0.8, -0.4, 0., 0.4, 0.8, 1]
    coupling = np.random.choice(coupling_values, n_trials)
    stimulus = np.random.choice(stimulus_values, n_trials)
    for i in range(i_ini, n_pars):
        pars = np.load(sv_folder + 'param_recovery/pars_prt' + str(i) + '.npy')
        print(pars)
        df = simulate_FBP(pars, n_iters, theta,
                          stimulus, coupling, sv_folder=SV_FOLDER, n_iter=i,
                          model=model)
        optimizer = optimization(data=df, n_iters=50, theta=theta)
        pars_array = fit_data(optimizer, model=model, n_iters=1, method=method,
                              plot=False)[0]
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
    if model == 'LBP':
        numpars = 4
    else:
        numpars = 5
    if method == 'BADS':
        method = ''
    orig_params = np.zeros((n_pars, numpars))
    recovered_params = np.zeros((n_pars, numpars))
    for i in range(n_pars):
        params_recovered = np.load(sv_folder + 'param_recovery/version_2/version_2_10000/pars_prt_recovered' + str(i) + model + method + '.npy')
        params_original = np.load(sv_folder + 'param_recovery/version_2/version_2_10000/pars_prt' + str(i) + '.npy')
        orig_params[i] = params_original[:numpars]
        recovered_params[i] = params_recovered
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(15, 9))
    ax = ax.flatten()
    labels = ['Coupling, J', 'Stimulus weight, B1', 'Bias, B0', 'noise', 'Alpha'][:numpars]
    xylims = [[0, 3], [0, 0.5], [0, 0.5], [0, 0.3], [0, 2]]
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


def fit_subjects(method='BADS', model='GS', subjects='separated',
                 data_augmen=False, n_init=1, extra=''):
    # good sub: s_6
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    if subjects == 'together':
        all_df['subject'] = 'all'
        all_df = all_df.reset_index()
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
            data = data_augmentation(data, sigma=0.07, times_augm=4,
                                     minval=0.001)
        print(len(data))
        # fig, ax = plt.subplots(ncols=2)
        # sns.lineplot(data, x='coupling', y='confidence', hue='stim_str', ax=ax[0])
        # sns.lineplot(data, x='stim_str', y='confidence', hue='coupling', ax=ax[1])
        # fig.suptitle(sub)
        optimizer = optimization(data=data, n_iters=50, theta=return_theta())
        pars_array = fit_data(optimizer, model=model, n_iters=n_init, method=method,
                              plot=False)
        params = np.median(pars_array, axis=0)
        # params = pars_array[0]
        print(params)
        if method == 'BADS':
            appendix = '_BADS'
        else:
            appendix = ''
        np.save(SV_FOLDER + 'parameters_' + model + appendix + sub +  extra + '.npy', params)


def plot_fitted_params(sv_folder=SV_FOLDER, model='LBP', method='BADS',
                       subjects='separated'):
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    if subjects == 'together':
        all_df['subject'] = 'all'
    subjects = all_df.subject.unique()
    if model in ['LBP', 'MF']:
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
        if i_a == 7:
            cmap = mpl.cm.Oranges
            norm = mpl.colors.Normalize(vmin=0, vmax=0.8)
            color = cmap(norm(parmat[:, 1]))
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
        ax3[0].plot(parmat[:, 1]*stim, pright, color=colormap[i_s], marker='o', linestyle='')
        ax3[1].plot(parmat[:, 1]*stim, accuracies, color=colormap[i_s], marker='o', linestyle='')
        ax3[2].plot(parmat[:, 1]*stim+parmat[:, 2], pright, color=colormap[i_s], marker='o', linestyle='')
        ax3[3].plot(parmat[:, 1]*stim+parmat[:, 2], accuracies, color=colormap[i_s], marker='o', linestyle='')
    fig3.tight_layout()
    fig3.savefig(SV_FOLDER + 'biases_modulation_fitted_params.png', dpi=200, bbox_inches='tight')
    fig3.savefig(SV_FOLDER + 'biases_modulation_fitted_params.svg', dpi=200, bbox_inches='tight')
    fig, ax = plt.subplots(1)
    corrmat = np.corrcoef(parmat.T)
    corrmat[corrmat > 0.99] = np.nan
    im = ax.imshow(corrmat, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    plt.colorbar(im, ax=ax, label='correlation')
    ax.set_yticks(np.arange(4), ['J', 'B1', 'B0', 'N'])
    ax.set_xticks(np.arange(4), ['J', 'B1', 'B0', 'N'])
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


def plot_density(num_iter=100):
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
        pars = np.load(SV_FOLDER + '/parameters_MF_BADS' + sub + '.npy')
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
                alpha=1, lw=2.5, bw_adjust=0.5, common_norm=False)
    fig, ax = plt.subplots(1)
    b_list=np.arange(-1.5, 1.5, 0.01)
    first_j = compute_j_crit(j_list=np.arange(0., 1.405, 0.01),
                       b_list=b_list, num_iter=100)
    ax.plot(b_list, first_j, color='k', linewidth=2.5, label='J*')
    ax.legend(frameon=False)
    sns.kdeplot(all_df, x='bstar', y='jstar', fill=True, ax=ax)
    ax.set_xlim(-1.25, 1.25)
    ax.set_xlabel('B = B_1 * stim + B_0')
    ax.set_ylabel('J = (1-Shuffling)*J_1')


def plot_regression_weights(sv_folder=SV_FOLDER, load=True):
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    subjects = all_df.subject.unique()
    if not load:
        data_orig = pd.DataFrame()
        data_model_null = pd.DataFrame()
        data_model_orig = pd.DataFrame()
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
            pars = np.load(sv_folder + '/parameters_MF_BADS' + sub + extra + '.npy')
            df_null = simulate_FBP(pars, 400,
                                   data.stim_str.values, data.coupling.values,
                                   sv_folder=SV_FOLDER, n_iter=sub, model='MF',
                                   recovery=False, resimulate=False, extra=extra)
            df_null['coupling'] = dataframe['pShuffle'].replace(to_replace=unique_vals,
                                                               value= [1., 0.3, 0.]) 
            data_model_null = pd.concat((data_model_null, df_null))
            extra = ''
            dataframe['coupling'] = dataframe['pShuffle'].replace(to_replace=unique_vals,
                                        value= [1., 0.3, 0.]) 
            dataframe['confidence'] = (transform(dataframe.confidence.values, -0.999, 0.999)+1)/2
            dataframe['stim_str'] = (dataframe.evidence.values)
            dataframe['stim_ev_cong'] = dataframe.response.values*dataframe.stim_str.values
            data = dataframe[['coupling', 'confidence', 'stim_str', 'stim_ev_cong', 'response']]
            data_orig = pd.concat((data_orig, data))
            pars = np.load(sv_folder + '/parameters_MF_BADS' + sub + extra + '.npy')
            df_no_null = simulate_FBP(pars, 400,
                                      data.stim_str.values, data.coupling.values,
                                      sv_folder=SV_FOLDER, n_iter=sub, model='MF',
                                      recovery=False, resimulate=False, extra=extra)
            df_no_null['coupling'] = dataframe['pShuffle'].replace(to_replace=unique_vals,
                                                                   value= [1., 0.3, 0.]) 
            data_model_orig = pd.concat((data_model_orig, df_no_null))
        for datframe in [data_orig, data_model_null, data_model_orig]:
            datframe['abs_confidence'] = np.abs(datframe.confidence-0.5)*2
            datframe['subject'] = all_df.subject
        data_orig.to_csv(sv_folder + 'simulated_data' + '/df_orig.csv')
        data_model_orig.to_csv(sv_folder + 'simulated_data' + '/df_simul_MF_orig.csv')
        data_model_null.to_csv(sv_folder + 'simulated_data' + '/df_simul_MF_null_model.csv')
    else:
        data_orig = pd.read_csv(sv_folder + 'simulated_data' + '/df_orig.csv')
        data_model_orig = pd.read_csv(sv_folder + 'simulated_data' + '/df_simul_MF_orig.csv')
        data_model_null = pd.read_csv(sv_folder + 'simulated_data' + '/df_simul_MF_null_model.csv')
    # md_orig = smf.mixedlm("abs_confidence ~ coupling*stim_str",
    #                       data_orig.dropna(), groups="subject")
    # md_orig = md_orig.fit()
    # print(md_orig.summary())
    # md_model_orig = smf.mixedlm("abs_confidence ~ coupling*stim_str",
    #                       data_model_orig, groups="subject")
    # md_model_orig = md_model_orig.fit()
    # print(md_model_orig.summary())
    # md_model_null = smf.mixedlm("abs_confidence ~ coupling*stim_str",
    #                             data_model_null, groups="subject")
    # md_model_null = md_model_null.fit()
    # print(md_model_null.summary())
    # sub by sub lienar regression
    weights_o = np.zeros((4, len(subjects)))
    weights_model_o = np.zeros((4, len(subjects)))
    weights_model_null = np.zeros((4, len(subjects)))
    for i_s, sub in enumerate(subjects):
        df_o = data_orig.loc[data_orig.subject == sub]
        df_model_o = data_model_orig.loc[data_model_orig.subject == sub]
        df_model_null = data_model_null.loc[data_model_null.subject == sub]
        coupling = df_o.coupling.values
        stim_str = np.abs(df_o.stim_str.values)
        val_conf_yo = df_o.abs_confidence.values
        idx_non_nan = np.where(~np.isnan(val_conf_yo))[0]
        X = np.column_stack((coupling[idx_non_nan], stim_str[idx_non_nan],
                             stim_str[idx_non_nan]*coupling[idx_non_nan]))
        y_o = scipy.stats.zscore(val_conf_yo[idx_non_nan])
        y_model_o = scipy.stats.zscore(df_model_o.abs_confidence.values[idx_non_nan])
        y_model_null = scipy.stats.zscore(df_model_null.abs_confidence.values[idx_non_nan])
        regr_o = linear_model.LinearRegression(fit_intercept=True)
        regr_o.fit(X, y_o)
        weights_o[1:, i_s] = regr_o.coef_
        weights_o[0, i_s] = regr_o.intercept_
        regr_model_o = linear_model.LinearRegression(fit_intercept=True)
        regr_model_o.fit(X, y_model_o)
        weights_model_o[1:, i_s] = regr_model_o.coef_
        weights_model_o[0, i_s] = regr_model_o.intercept_
        regr_model_null = linear_model.LinearRegression(fit_intercept=True)
        regr_model_null.fit(X, y_model_null)
        weights_model_null[1:, i_s] = regr_model_null.coef_
        weights_model_null[0, i_s] = regr_model_null.intercept_
    fig, ax = plt.subplots(ncols=4, figsize=(16, 4))
    for a in ax:
        a.axhline(0, color='k', linestyle='--')
    xlabs = ['Data', 'Model', 'Null']
    ylabs = ['Intercept', 'Coupling', 'Stim.', 'Coupling:stim']
    for j in range(4):
        sns.boxplot([weights_o[j], weights_model_o[j], weights_model_null[j]], ax=ax[j])
        ax[j].set_ylabel(ylabs[j])
        ax[j].set_xticks([0, 1, 2], xlabs, rotation=45)
    for i_s in range(len(subjects)):
        jitter = np.random.randn(3)*0.05
        ax[0].plot([0+jitter[0], 1+jitter[1], 2+jitter[2]],
                   [weights_o[0][i_s], weights_model_o[0][i_s], weights_model_null[0][i_s]],
                   color='gray', alpha=0.5)
        ax[1].plot([0+jitter[0], 1+jitter[1], 2+jitter[2]],
                   [weights_o[1][i_s], weights_model_o[1][i_s], weights_model_null[1][i_s]],
                   color='gray', alpha=0.5)
        ax[2].plot([0+jitter[0], 1+jitter[1], 2+jitter[2]],
                   [weights_o[2][i_s], weights_model_o[2][i_s], weights_model_null[2][i_s]],
                   color='gray', alpha=0.5)
        ax[3].plot([0+jitter[0], 1+jitter[1], 2+jitter[2]],
                   [weights_o[3][i_s], weights_model_o[3][i_s], weights_model_null[3][i_s]],
                   color='gray', alpha=0.5)
        for t in range(3):
            ax[0].plot([t+jitter[t]],
                       [weights_o[0][i_s], weights_model_o[0][i_s], weights_model_null[0][i_s]][t],
                       marker='o', color=['tab:blue', 'tab:orange', 'tab:green'][t], linestyle='',
                       markersize=5, markeredgewidth=1, markeredgecolor='grey')
            ax[1].plot([t+jitter[t]],
                       [weights_o[1][i_s], weights_model_o[1][i_s], weights_model_null[1][i_s]][t],
                       marker='o', color=['tab:blue', 'tab:orange', 'tab:green'][t], linestyle='',
                       markersize=5, markeredgewidth=1, markeredgecolor='grey')
            ax[2].plot([t+jitter[t]],
                       [weights_o[2][i_s], weights_model_o[2][i_s], weights_model_null[2][i_s]][t],
                       marker='o', color=['tab:blue', 'tab:orange', 'tab:green'][t], linestyle='',
                       markersize=5, markeredgewidth=1, markeredgecolor='grey')
            ax[3].plot([t+jitter[t]],
                       [weights_o[3][i_s], weights_model_o[3][i_s], weights_model_null[3][i_s]][t],
                       marker='o', color=['tab:blue', 'tab:orange', 'tab:green'][t], linestyle='',
                       markersize=5, markeredgewidth=1, markeredgecolor='grey')
    fig.tight_layout()
    scipy.stats.ttest_ind(weights_o[3],  weights_model_o[3])
    scipy.stats.ttest_ind(weights_model_null[3],  weights_model_o[3])
    scipy.stats.ttest_ind(weights_o[3],  weights_model_null[3])


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


def plot_log_likelihood_difference(sv_folder=SV_FOLDER, mcmc=False,
                                   log_posterior=True):
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    subjects = all_df.subject.unique()
    llh_all = np.zeros((3, len(subjects)))
    for i_e, extra in enumerate(['', 'null']):
        if mcmc:
            path = SV_FOLDER + extra + 'MCMC_fitted_MF_parameters.npy'
            params = np.load(path)
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
                pars = np.load(sv_folder + '/parameters_MF_BADS' + sub + extra + '.npy')
            llh = optimization(data=data, n_iters=10).nlh_boltzmann_mf(pars=pars)
            if not log_posterior:
                llh += len(dataframe)*(
                    np.log(scipy.stats.norm.pdf(pars[0], 1/3.92, 1)) +  # J1 prior
                    np.log(scipy.stats.norm.pdf(pars[1], 0.5, 1)) +  # B1 prior
                    np.log(scipy.stats.norm.pdf(pars[2], 0., 0.6))+  # B0 prior
                    np.log(scipy.stats.uniform.pdf(pars[3], 0.1, 0.5)+1e-15))  # Uniform prior for sigma (positive only)
            llh_all[i_e, i_s] = llh
    llh_all[2] = llh_all[1]-llh_all[0]
    fig, ax = plt.subplots(1)
    ax.boxplot(llh_all.T)
    ax.axhline(0, linestyle='--', color='k', alpha=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for i_s in range(len(subjects)):
        jitter = np.random.randn(3)*0.03
        ax.plot([1+jitter[0], 2+jitter[1]], [llh_all[0, i_s], llh_all[1, i_s]],
                 color='gray', alpha=0.4)
        ax.plot([1+jitter[0], 2+jitter[1]], [llh_all[0, i_s], llh_all[1, i_s]],
                 color='k', marker='o', linestyle='', markersize=4, alpha=0.6)
        ax.plot([1+jitter[0], 2+jitter[1], 3+jitter[2]],
                [llh_all[0, i_s], llh_all[1, i_s], llh_all[2, i_s]],
                 color='k', marker='o', linestyle='', markersize=4, alpha=0.6)
    if log_posterior:
        ax.set_ylabel('Negative log-posterior')
    else:
        ax.set_ylabel('Negative log-likelihood')
    ax.set_xticks([1, 2, 3], ['Full', 'Null', 'Null-Full'], rotation=45)
    p = scipy.stats.ttest_ind(llh_all[0], llh_all[1]).pvalue
    x1, x2 = 1, 2
    y, h, col = max(map(max, llh_all)) + 150, 80, 'k'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x2)*.5, y+h, f"p = {p:.2e}", ha='center', va='bottom', color=col)
    ax.set_ylim(np.min((np.min(llh_all[2])-200, -10)), y+200)
    p = scipy.stats.ttest_1samp(llh_all[2], popmean=0).pvalue
    ax.text(3, np.max(llh_all[2])+50, f"p = {p:.2e}", ha='center', va='bottom', color=col)
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


if __name__ == '__main__':
    # plot_parameter_recovery(sv_folder=SV_FOLDER, n_pars=50, model='FBP', method='BADS')
    fit_subjects(method='BADS', model='GS', data_augmen=False, n_init=1, extra='')
    fit_subjects(method='BADS', model='GS', data_augmen=False, n_init=1, extra='null')
    # plot_log_likelihood_difference(sv_folder=SV_FOLDER, mcmc=True)
    # plot_density(num_iter=100)
    # simulate_subjects(sv_folder=SV_FOLDER, model='MF', resimulate=True, extra='')
    # simulate_subjects(sv_folder=SV_FOLDER, model='MF', resimulate=True, extra='null', mcmc=True)
    # plot_regression_weights(sv_folder=SV_FOLDER, load=False)
    # plot_fitted_params(sv_folder=SV_FOLDER, model='MF', method='BADS',
    #                     subjects='separated')
    # mcmc_all_subjects(plot=True, burn_in=100, iterations=1000, load_params=True,
    #                   extra='null')
    # mcmc_all_subjects(plot=True, burn_in=100, iterations=1000, load_params=True,
    #                   extra='')
    # plot_noise_vs_b_accuracy(n_trials=2000, n_iters=300, noiselist=np.arange(0.1, 0.55, 0.05),
    #                          j=0.5, b_list=np.arange(0, 0.9, 0.1), load_data=True)
    # p_corr_vs_noise(n_trials=2000, n_iters=300, noiselist=np.arange(0.1, 0.525, 0.025),
    #                 j_vals=[0.1, 0.45, 0.8], b_list=[0.1, 0.4, 0.8], load_data=True)
    # parameter_recovery(n_pars=50, sv_folder=SV_FOLDER,
    #                    theta=THETA, n_iters=2500, n_trials=500,
    #                    model='FBP', method='BADS')
    # parameter_recovery(n_pars=50, sv_folder=SV_FOLDER,
    #                     theta=THETA, n_iters=2500, n_trials=500,
    #                     model='LBP', method='BADS')
    # optimizer.mse_minimization()  # optimize via MSE
    # combinations = optimizer.combinations
    # mse_list_mf = optimizer.mse_list_mf
    # mse_list_fbp = optimizer.mse_list_fbp
    # mse_list_lbp = optimizer.mse_list_lbp
    # mse_list_gibbs = optimizer.mse_list_gibbs
    # best_pars_mf_idx = np.argmin(mse_list_mf)
    # best_combo_mf = combinations[best_pars_mf_idx]
    # best_pars_fbp_idx = np.argmin(mse_list_fbp)
    # best_combo_fbp = combinations[best_pars_fbp_idx]
    # best_pars_lbp_idx = np.argmin(mse_list_lbp)
    # best_combo_lbp = combinations[best_pars_lbp_idx]
    # best_pars_gibbs_idx = np.argmin(mse_list_gibbs)
    # best_combo_gibbs = combinations[best_pars_gibbs_idx]

    # df_simul = simulate_data(data, best_combo_lbp, optimizer, 50)
    # fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(14, 7))
    # ax = ax.flatten()
    # sns.lineplot(df_simul, x='coupling', y='confidence_mf', hue='stim_str', ax=ax[0])
    # sns.lineplot(df_simul, x='coupling', y='confidence_fbp', hue='stim_str', ax=ax[1])
    # sns.lineplot(df_simul, x='coupling', y='confidence_lbp', hue='stim_str', ax=ax[2])
    # sns.lineplot(df_simul, x='coupling', y='confidence_gibbs', hue='stim_str', ax=ax[3])
    # sns.lineplot(df_simul, x='stim_str', y='confidence_mf', hue='coupling', ax=ax[4])
    # sns.lineplot(df_simul, x='stim_str', y='confidence_fbp', hue='coupling', ax=ax[5])
    # sns.lineplot(df_simul, x='stim_str', y='confidence_lbp', hue='coupling', ax=ax[6])
    # sns.lineplot(df_simul, x='stim_str', y='confidence_gibbs', hue='coupling', ax=ax[7])
    # fig.tight_layout()


    # fig, ax = plt.subplots(ncols=4, figsize=(12, 3))
    # ax[0].plot(df.confidence, df_simul.confidence_mf, marker='o',
    #            linestyle='', color='k', markersize=3)
    # ax[0].set_title('MF, corr = ' + str(np.corrcoef(df.confidence, df_simul.confidence_mf)[0][1]))
    # ax[1].plot(df.confidence, df_simul.confidence_lbp, marker='o',
    #            linestyle='', color='k', markersize=3)
    # ax[1].set_title('LBP, corr = ' + str(np.corrcoef(df.confidence, df_simul.confidence_lbp)[0][1]))
    # ax[2].plot(df.confidence, df_simul.confidence_fbp, marker='o',
    #            linestyle='', color='k', markersize=3)
    # ax[2].set_title('FBP, corr = ' + str(np.corrcoef(df.confidence, df_simul.confidence_fbp)[0][1]))
    # ax[3].plot(df.confidence, df_simul.confidence_gibbs, marker='o',
    #            linestyle='', color='k', markersize=3)
    # ax[3].set_title('Gibbs, corr = ' + str(np.corrcoef(df.confidence, df_simul.confidence_gibbs)[0][1]))
    # for a in ax:
    #     a.plot([0.4, 1], [0.4, 1], color='k', alpha=0.2)
    # fig.tight_layout()
