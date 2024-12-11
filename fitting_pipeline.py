# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:49:48 2024

@author: alexg
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

from loop_belief_prop_necker import discrete_DBN, Loopy_belief_propagation, dyn_sys_fbp
from mean_field_necker import mean_field_stim, solution_mf_sdo_euler
from gibbs_necker import gibbs_samp_necker, return_theta, occ_function_markov_ch_var
import itertools
import pandas as pd
import seaborn as sns
from pybads import BADS
from scipy.optimize import Bounds
import glob
import os


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


    def nlh_boltzmann_lbp(self, pars, n=4, eps=1e-3, conts=0.5):
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
        nlh_lbp = -np.nansum(boltzman_lbp-np.log(norm_cte))
        # contaminants (?)
        # distro = np.exp(boltzman_lbp)/norm_cte
        # nlh_lbp = -np.nansum(np.log(distro*(1-eps)+conts*eps))
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


    def nlh_boltzmann_fbp(self, pars, n=3, eps=1e-3, conts=0.5):
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


    def nlh_boltzmann_mf(self, pars, n=4, eps=1e-3, conts_distro=1e-2):
        jpar, b1par, biaspar, noise = pars
        coupling, stim_str, confidence = self.coupling, self.stim_str, self.confidence
        j = jpar*np.array(coupling)
        b = b1par*np.array(stim_str)+biaspar

        q = np.arange(0, 1, 5e-2)  # Define q outside the loop, shape (100,)

        # Reshape j and b to broadcast correctly over q
        j = np.array(j).reshape(-1, 1)  # Reshape j to shape (500, 1)
        b = np.array(b).reshape(-1, 1)  # Reshape b to shape (500, 1)

        # Vectorized version of the potential function over all i and q
        exp_term = 2 * n * (j * (2 * q - 1)) + 2 * b  # Shape: (500, 100)
        log_term = np.log(1 + np.exp(exp_term))  # Shape: (500, 100)

        # Vectorized potential (pot_mf_i) for all i and all q
        pot_mf = (q*q) / 2 - log_term / (4* n * j)  # Shape: (500, 100)

        # Apply Boltzmann distribution function over the potential values (vectorized)
        bmann_values = np.exp(-2 * pot_mf / (noise*noise))  # Shape: (500, 100)

        # Sum over q for each i to get the normalization constant, norm_cte
        norm_cte = np.sum(bmann_values, axis=1)  # Shape: (500,)

        j = np.array(j).reshape(-1)  # Reshape j to shape (500, 1)
        b = np.array(b).reshape(-1)  # Reshape b to shape (500, 1)
        pot_mf_fun = lambda q: q*q/2 - np.log(1+np.exp(2*n*(j*(2*q-1))+b*2))/(4*n*j)
        bmann_distro_log = lambda potential: -2*np.array(potential) / (noise*noise)
        # nlh_mf = -np.nansum(np.log((1-eps)*np.exp(bmann_distro_log(pot_mf_fun(confidence)))/norm_cte + eps*conts_distro))
        nlh_mf = -np.nansum(bmann_distro_log(pot_mf_fun(confidence)) - np.log(norm_cte))
        return nlh_mf


    def nlh_gibbs(self, pars, n=4, eps=1e-3, conts_distro=1e-2):
        jpar, b1par, biaspar, noise, time = pars
        coupling, stim_str, confidence = self.coupling, self.stim_str, self.confidence
        j = jpar*np.array(coupling)
        b = b1par*np.array(stim_str)+biaspar
        likelihood = []
        for i in range(len(confidence)):
            k_1 = 12*j[i] + 8*b[i]
            k_u = 2*j[i] + 2*b[i]
            k_2 = 12*j[i] - 8*b[i]
            rho = np.exp(-(k_1-k_2)/2)
            alpha = np.exp(-(k_1+k_2 - k_u*2)/2)
            val = occ_function_markov_ch_var(rho, alpha, time, confidence[i]*time)
            time_vals = np.arange(0, time+1, 1)
            norm_cte = np.nansum(occ_function_markov_ch_var(rho, alpha, time, time_vals))
            likelihood.append(val / norm_cte)
        likelihood = np.array(likelihood)
        return -np.nansum(np.log(likelihood*(1-eps) + eps*conts_distro))


    def optimize_nlh(self, x0, model='MF', method='nelder-mead'):
        # effective_n = np.max(np.linalg.eigvals(self.theta))
        assert model in ['MF', 'LBP', 'FBP', 'GS'], 'Model should be either GS, MF, LBP or FBP'
        if model == 'MF':
            fun = self.nlh_boltzmann_mf
            assert len(x0) == 4, 'x0 should have 4 values (J, B1, bias, noise)'
            bounds = Bounds([1e-1, -0.2, -0.2, 0.06], [1, 0.25, 0.25, 0.3])
        if model == 'GS':
            fun = self.nlh_gibbs
            assert len(x0) == 5, 'x0 should have 5 values (J, B1, bias, noise, time_end)'
            bounds = Bounds([1e-1, -1, -1, 0.06, 1], [1, 0.25, 0.25, 0.3, 1e6])
        if model == 'LBP':
            fun = self.nlh_boltzmann_lbp
            assert len(x0) == 4, 'x0 should have 4 values (J, B1, bias, noise)'
            if method != 'BADS':
                bounds = Bounds([1e-1, -.5, -.5, 0.05], [3, .5, .5, 0.3])
            if method == 'BADS':
                lb = [0.01, -.5, -.5, 0.05]
                ub = [3., 1., 1., 0.3]
                plb = [0.5, -0.2, -0.2, 0.1]
                pub = [2.3, 0.5, 0.5, 0.2]
        if model == 'FBP':
            fun = self.nlh_boltzmann_fbp
            assert len(x0) == 5, 'x0 should have 5 values (J, B1, bias, noise, alpha)'
            if method != 'BADS':
                bounds = Bounds([1e-1, -.1, -.1, 0.05, 0], [2., .4, .4, 0.3, 1.5])
            if method == 'BADS':
                lb = [0.01, -.1, -.1, 0.05, 0.]
                ub = [2., 0.4, 0.4, 0.3, 1.5]
                plb = [0.5, -0.05, -0.05, 0.1, 0.3]
                pub = [1.4, 0.2, 0.2, 0.2, 1.2]
        if method != 'BADS':
            optimizer_0 = scipy.optimize.minimize(fun, x0, method=method,
                                                  bounds=bounds)
        if method == 'BADS':
            print('BADS')
            optimizer_0 = BADS(fun, x0, lb, ub, plb, pub).optimize()
        # optimizer_0 = scipy.optimize.minimize(fun, x0, method='trust-constr', bounds=bounds)
        # optimizer_0 = scipy.optimize.minimize(fun, x0, method='BFGS', bounds=bounds)
        # optimizer_0 = scipy.optimize.minimize(fun, x0, method='COBYLA', bounds=bounds)
        return optimizer_0.x


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


def transform(x, minval=0.4, maxval=0.9999):
    maxarray = np.nanmax(x)
    minarray = np.nanmin(x)
    return (maxval-minval)/(maxarray-minarray)*(x-minarray) + minval


def fit_data(optimizer, plot=True, model='MF', n_iters=200, method='nelder-mead'):
    if model == 'FBP' or model == 'GS':
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
        if model == 'MF':
            j0 = np.random.uniform(0.1, 0.8)
        else:
            j0 = np.random.uniform(0.3, 1.4)
        b10 = np.random.uniform(0.05, 0.1)
        bias0 = np.random.uniform(0.05, 0.1)
        noise0 = np.random.uniform(0.1, 0.2)
        if model == 'FBP':
            alpha0 = np.random.uniform(0.1, 1.4)
            x0 = [j0, b10, bias0, noise0, alpha0]
        if model == 'GS':
            time = 10**(np.random.uniform(-1, 6))
            x0 = [j0, b10, bias0, noise0, time]
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


def simulate_FBP(pars, n_iters, theta,
                 stimulus, coupling, sv_folder=SV_FOLDER,
                 n_iter=0, model='FBP'):
    vals_conf = []
    jpar, b1par, biaspar, noise, alpha = pars
    if model in ['LBP', 'MF']:
        alpha = 1
    b = stimulus*b1par + biaspar
    j = coupling*jpar
    pathdata = sv_folder + 'param_recovery/df_simul' + str(n_iter) + model + '.csv'
    os.makedirs(os.path.dirname(pathdata), exist_ok=True)
    if os.path.exists(pathdata):
        data = pd.read_csv(pathdata)
    else:
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
        data = pd.DataFrame({'stim_str': stimulus, 'coupling': coupling,
                             'confidence': vals_conf})
        data.to_csv(sv_folder + 'param_recovery/df_simul' + str(n_iter) + model + '.csv')
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


def data_augmentation(df, times_augm=10, sigma=0.05, minval=0.4, maxval=0.9999):
    df_copy = df.copy()
    confidence = np.repeat(df_copy.confidence.values, times_augm+1)
    confidence[len(df_copy):] += np.random.randn(len(df_copy)*times_augm)*sigma
    confidence = np.clip(confidence, minval, maxval)
    coupling = np.repeat(df_copy.coupling.values, times_augm+1)
    stim_str = np.repeat(df_copy.stim_str.values, times_augm+1)
    data = pd.DataFrame({'stim_str': stim_str, 'coupling': coupling,
                         'confidence': confidence})
    return data


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


def fit_subjects(method='BADS', model='FBP', subjects='separated',
                 data_augmen=False):
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
        unique_vals = dataframe['pShuffle'].unique()
        dataframe['coupling'] = dataframe['pShuffle'].replace(to_replace=unique_vals,
                                    value= [1, 0.5, 0.2])
        dataframe['confidence'] = transform(np.abs(dataframe.confidence.values))
        dataframe['stim_str'] = np.abs(dataframe.evidence)
        data = dataframe[['coupling', 'confidence', 'stim_str']]
        if data_augmen:
            data = data_augmentation(data, sigma=0.03, times_augm=20)
        print(len(data))
        # fig, ax = plt.subplots(ncols=2)
        # sns.lineplot(data, x='coupling', y='confidence', hue='stim_str', ax=ax[0])
        # sns.lineplot(data, x='stim_str', y='confidence', hue='coupling', ax=ax[1])
        # [a.set_ylim(0.4, 1.05) for a in ax]
        optimizer = optimization(data=data, n_iters=50, theta=return_theta())
        pars_array = fit_data(optimizer, model=model, n_iters=1, method=method,
                              plot=False)
        params = np.median(pars_array, axis=0)
        # params = pars_array[0]
        print(params)
        if method == 'BADS':
            appendix = '_BADS'
        else:
            appendix = ''
        np.save(SV_FOLDER + 'parameters_' + model + appendix + sub + '.npy', params)


def plot_fitted_params(sv_folder=SV_FOLDER, model='LBP', method='BADS',
                       subjects='separated'):
    all_df = load_data(data_folder=DATA_FOLDER, n_participants='all')
    if subjects == 'together':
        all_df['subject'] = 'all'
    subjects = all_df.subject.unique()
    if model == 'LBP':
        numpars = 4
    else:
        numpars = 5
    if method == 'BADS':
        appendix = '_BADS'
    else:
        appendix = ''
    accuracies = []
    pright = []
    nsubs = len(subjects)
    parmat = np.zeros((nsubs, numpars))
    for i_s, sub in enumerate(subjects):
        params = np.load(SV_FOLDER + 'parameters_' + model + appendix + sub + '.npy')
        parmat[i_s, :] = params
        dataframe = all_df.copy().loc[all_df['subject'] == sub]
        accuracies.append(sum(dataframe.response == dataframe.side)/len(dataframe.response))
        pright.append(np.mean((dataframe.response+1)/2))
    fig, ax = plt.subplots(ncols=numpars, figsize=(15,5))
    for i in range(numpars):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        sns.violinplot(parmat[:, i], color='lightblue', alpha=0.3, ax=ax[i])
        ax[i].plot(np.random.randn(nsubs)*0.05, parmat[:, i], marker='o', color='k', linestyle='', markersize=4)
        ax[i].set_xticks([])
    fig2, ax2 = plt.subplots(ncols=numpars, nrows=2, figsize=(16, 10))
    ax2 = ax2.flatten()
    labels = ['Coupling, J', 'Stimulus weight, B1', 'Bias, B0', 'noise', 'Alpha'][:numpars]
    for i_a, a in enumerate(ax2):
        if i_a < numpars:
            var = pright
            lab = 'p(right)'
        else:
            var = accuracies
            lab = 'p(correct)'
        a.plot(var, parmat[:, i_a % numpars], color='k', marker='o', linestyle='')
        a.set_xlabel(lab)
        a.set_ylabel(labels[i_a % numpars])
    fig2.tight_layout()


if __name__ == '__main__':
    plot_parameter_recovery(sv_folder=SV_FOLDER, n_pars=50, model='FBP', method='BADS')
    # fit_subjects(method='BADS', model='FBP')
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
