# -*- coding: utf-8 -*-
"""
Model fitting and analysis (+plots).

Split from original hysteresis_analysis.py. Uses common.py for shared code.
"""

# Make the parent folder (all_scripts/) importable so gibbs_necker,
# mean_field_necker and fitting_pipeline (one level up from this folder)
# resolve whether this module is imported or run directly.
import os as _os
import sys as _sys
_PARENT_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PARENT_DIR not in _sys.path:
    _sys.path.insert(0, _PARENT_DIR)

import pyddm
import pyddm.plot
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import scipy
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pylab as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import glob
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
from sklearn import manifold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.gridspec as gridspec
from gibbs_necker import rle
from mean_field_necker import colored_line
from fitting_pipeline import load_data as load_data_experiment_1
import matplotlib.patches as mpatches
from scipy.optimize import fsolve, minimize
from pybads import BADS
# import sbi
# from sbi.inference import infer
# from sbi.utils import MultipleIndependent
import torch
from torch.distributions import Uniform
import time
import pickle
# from sbi import analysis as analysis
import tqdm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, zscore
from scipy.signal import sawtooth, medfilt
from scipy.optimize import curve_fit, root_scalar, brentq, fsolve
from scipy.integrate import quad, cumulative_trapezoid, solve_bvp, solve_ivp
from scipy.interpolate import interp1d
import itertools
from pyddm import set_N_cpus
from pyddm.models.loss import LossLikelihood, LossBIC
from pyddm.functions import get_model_loss
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter, zoom
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
# import mplcairo
import matplotlib.animation as animation

mpl.rcParams['font.size'] = 16
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize']= 14
plt.rcParams['ytick.labelsize']= 14
plt.rcParams["axes.grid"] = False

from common import *  # noqa: F401,F403  (shared helpers, constants, base functions)
from common import DATA_FOLDER, SV_FOLDER, COLORMAP


def compute_poisson_negative_log_likelihood_single_trial(responses, stimulus, coupling,
                                                         j_par, b_par, j0par=0,
                                                         b0=0, sigma_fit=0.2,
                                                         n=3.92, sigma_stim=0, fps=60,
                                                         theta=0.5, tol=1e-6):
    j_eff = coupling*j_par+j0par
    dt = 1/fps
    switches = np.abs(np.diff(responses))
    log_likelihood = 0
    nsteps = 0
    b_t = b_par*stimulus
    for t in range(len(stimulus)-1):
        resp_t = responses[t]
        if resp_t == 0:
            continue
        switch_t = switches[t]
        b_eff = b0+b_t[t]
        xR, xL, x_unstable = get_unst_and_stab_fp(j_eff, b_eff, n=n)
        if np.isnan(x_unstable) or np.abs(xL-xR) < tol:  # monostable
            # mu_0 = compute_linearized_drift(xL, j_eff, n=n)
            lambda_t = compute_nu_0(mu_0=xL, sigma=sigma_fit, theta=theta)
        else:
            if resp_t == -1:
                lambda_t = k_i_to_j(j_eff, xL, x_unstable, noise=sigma_stim+sigma_fit,
                                    b=b_eff, n=n)
            if resp_t == 1:
                lambda_t = k_i_to_j(j_eff, xR, x_unstable, noise=sigma_stim+sigma_fit,
                                       b=b_eff, n=n)
            # lambda_t = lambda_t_LR if resp_t == 1 else lambda_t_RL
        if switch_t == 0:
            log_dd += -lambda_t*dt
        if switch_t == 2:
            log_likelihood += np.log(np.max([1-np.exp(-lambda_t*dt), 1e-12]))
        nsteps += 1
    return -log_likelihood/nsteps


def get_negative_log_likelihood_all_trials_single_subject(params, data,
                                                          fps=60, tFrame=20):
    j_par, j0, b0, b_par, sigma = params
    responses_2, responses_4, stim_values_2, stim_values_4, coupling_2, coupling_4 = data
    likelihood_2 = 0
    for t1 in range(len(coupling_2)):
        lh_2 = compute_poisson_negative_log_likelihood_single_trial(
            responses=responses_2[t1], stimulus=stim_values_2, coupling=coupling_2[t1],
            j_par=j_par, j0par=j0, b_par=b_par, b0=b0, sigma_fit=sigma,
            n=3.92, sigma_stim=0, fps=fps, theta=0.5)
        likelihood_2 += lh_2
    likelihood_4 = 0
    for t2 in range(len(coupling_4)):
        lh_4 = compute_poisson_negative_log_likelihood_single_trial(
            responses=responses_4[t2], stimulus=stim_values_4, coupling=coupling_4[t2],
            j_par=j_par, j0par=j0, b_par=b_par, b0=b0, sigma_fit=sigma,
            n=3.92, sigma_stim=0, fps=fps, theta=0.5)
        likelihood_4 += lh_4
    negative_log_likelihood = likelihood_2+likelihood_4
    return negative_log_likelihood


def fitting_transitions(data_folder=DATA_FOLDER, fps=60, tFrame=20):
    df = load_data(data_folder, n_participants='all')
    lb = [0, 0, -1, -0.2, 0.05]
    ub = [4., 0.4, 1, 1.5, 0.8]
    plb = [0.1, 0.02, -0.5, 0.1, 0.1]
    pub = [1.4, 0.12, 0.5, 0.9, 0.3]
    x0 = [0.3, 0.03, 0, 0.5, 0.15]  # J, B, B0, sigma
    subjects = df.subject.unique()
    for i_s, subject in enumerate(subjects):
        df_sub = df.loc[df.subject == subject]
        responses_2, responses_4, stim_values_2, stim_values_4, coupling_2, coupling_4, \
            signed_freq_2, signed_freq_4 =\
            prepare_data_for_fitting(df_sub, tFrame=tFrame, fps=fps)
        responses_2 = responses_clean(responses_2)
        responses_4 = responses_clean(responses_4)
        data = [responses_2, responses_4, stim_values_2, stim_values_4, coupling_2, coupling_4]
        optimization_function = lambda x:\
            get_negative_log_likelihood_all_trials_single_subject(x, data,
                                                                  fps=fps, tFrame=tFrame)
        # options = {'tol_fun': 1e-5}
        options = {}
        optimizer_0 = BADS(optimization_function, x0, lb, ub, plb, pub,
                           options=options).optimize()
        pars = optimizer_0.x
        print(pars)


def compute_log_likelihood_fokker_planck(probs, keypresses, times):
    prob_right_left, prob_left_right = probs
    log_likelihood = 0.0
    for i, idx in enumerate(times[1:]):
        if np.isnan(keypresses[i]):
            prob = 0
        if keypresses[i] == 1:  # Right
            if keypresses[i-1] == 0:
                prob = prob_left_right[i]
            if keypresses[i-1] == 1:
                prob = 1-prob_left_right[i]
        if keypresses[i] == 0:  # Left
            if keypresses[i-1] == 0:
                prob = 1-prob_right_left[i]
            if keypresses[i-1] == 1:
                prob = prob_right_left[i]
        prob = np.clip(prob, 1e-10, 1.0)
        log_likelihood += np.log(prob)
    return log_likelihood


def negative_log_likelihood(theta, times, stim_values, keypresses, dq=0.025):
    flux_right_left, flux_left_right, _, _ = solve_fokker_planck_1d(times, stim_values, theta,
                                                                    dx=dq, shift_ini=0)
    ll = compute_log_likelihood_fokker_planck([flux_right_left, flux_left_right], keypresses, times)
    return -ll


def get_negative_log_likelihood_all_data(theta, data, tFrame=18, fps=60,  #, dx=0.025,
                                         model='MF'):
    T = tFrame  # total duration (seconds)
    n_times = tFrame*fps
    times = np.linspace(0, T, n_times)
    negative_log_likelihood_all = 0
    for i_f, freq in enumerate([2, 4]):
        # get response_array and stimulus values
        responses, stim_values = data[i_f], data[i_f+2]
        coupling = data[i_f+4]
        prob_LR_arr = np.zeros((3, len(stim_values)))
        prob_RL_arr = np.zeros((3, len(stim_values)))
        coupling_values = np.sort(np.unique(np.round(coupling, 2)))
        for i_j, jval in enumerate(coupling_values):
            # solve Fokker-Planck for each shuffling value and freq (3x2=6 conditions)
            theta_prime = np.copy(theta)
            theta_prime[0] = jval*theta[0]
            if model == 'MF':
                dx = 0.025
                shift_ini = 0
            else:
                dx = 0.1
                shift_ini = -0.5
            flux_right_left, flux_left_right, p_right, p_left = solve_fokker_planck_1d(times, stim_values, theta_prime,
                                                                                       dx=dx, shift_ini=shift_ini,
                                                                                       model=model)
            flux_left_right = np.array(flux_left_right)
            flux_left_right[flux_left_right < 0] = 0
            flux_right_left = np.array(flux_right_left)
            flux_right_left[flux_right_left < 0] = 0
            prob_switch_L_to_R = flux_left_right/p_left/fps  # hazard_rate_L_R = flux_L_R/p(L)
            # p_switch_L_R = hazard_rate_L_R*dt
            prob_switch_R_to_L = flux_right_left/p_right/fps
            prob_LR_arr[i_j] = prob_switch_L_to_R
            prob_RL_arr[i_j] = prob_switch_R_to_L
        for i_trial, keypresses in enumerate(responses):
            idx = coupling_values == coupling[i_trial]
            nllh = compute_log_likelihood_fokker_planck([prob_LR_arr[idx][0],
                                                         prob_RL_arr[idx][0]],
                                                        keypresses, times)
            negative_log_likelihood_all += -nllh
    return negative_log_likelihood_all


def prepare_data_for_fitting_deprecated(df, tFrame=18, fps=60):
    df_freq = df.loc[df.freq == 2]
    # get response_array and stimulus values
    responses_2, stim_values_2 = get_response_and_blist_array(df_freq, fps=fps, tFrame=tFrame,
                                                              flip_responses=False)
    coupling_2 = np.round(1-df_freq.sort_values('trial_index').groupby('trial_index').pShuffle.mean().values, 2)
    signed_freq_2 = df_freq.sort_values('trial_index').groupby('trial_index').initial_side.mean().values*2
    df_freq = df.loc[df.freq == 4]
    # get response_array and stimulus values
    responses_4, stim_values_4 = get_response_and_blist_array(df_freq, fps=fps, tFrame=tFrame,
                                                              flip_responses=False)
    coupling_4 = np.round(1-df_freq.sort_values('trial_index').groupby('trial_index').pShuffle.mean().values, 2)
    signed_freq_4 = df_freq.sort_values('trial_index').groupby('trial_index').initial_side.mean().values*4
    return [responses_2, responses_4, stim_values_2, stim_values_4, coupling_2, coupling_4,
            signed_freq_2, signed_freq_4]


def fitting_fokker_planck(data_folder=DATA_FOLDER, tFrame=18, fps=60,
                          model='MF'):
    df = load_data(data_folder, n_participants='all')
    subjects = df.subject.unique()
    if model in ['MF', 'LBP']:
        lb = [0, -1, -0.2, 0.05, 0.05]
        ub = [2., 1, 1.5, 0.6, 2]
        plb = [0.1, -0.5, 0.1, 0.15, 0.3]
        pub = [1.4, 0.5, 0.9, 0.4, 0.9]
        x0 = [1, 0, 0.5, 0.2, 0.5]
    else:
        lb = [0, 0.01, -1, -0.2, 0.05, 0.05]
        ub = [2., 2, 1, 1.5, 0.6, 2]
        plb = [0.1, 0.5, -0.5, 0.1, 0.15, 0.3]
        pub = [1.4, 1.5, 0.5, 0.9, 0.4, 0.9]
        x0 = [1, 1, 0, 0.5, 0.2, 0.5]
    constraint = lambda x: 1/fps*x[:, -2]**2/(2*x[:, -2]**2) > 0.075
    # dt*sigma**2 / (2*tau**2) > 0.075 = 3*dx
    for i_s, subject in enumerate(subjects):
        df_sub = df.loc[df.subject == subject]
        data = prepare_data_for_fitting(df_sub, tFrame=tFrame, fps=fps)
        fun = lambda theta: get_negative_log_likelihood_all_data(theta, data,
                                                                 tFrame=tFrame,
                                                                 fps=fps,
                                                                 model=model)
        optimizer_0 = BADS(fun, x0, lb, ub, plb, pub,
                           non_box_cons=constraint).optimize()
        pars = optimizer_0.x
        print(pars)


def simulator_5_params_sticky_bound(params, freq, nFrame=1560, fps=60,
                                    return_choice=False,
                                    ini_cond_convergence=1):
    """
    Simulator. Takes set of `params` and simulates the system, returning summary statistics.
    Params: J_eff, B_eff, tau, threshold distance, noise
    """
    if abs(freq) == 2:
        difficulty_time_ref_2 = np.linspace(-2, 2, nFrame//2)
        stimulus = np.concatenate((difficulty_time_ref_2, -difficulty_time_ref_2))
    if abs(freq) == 4:
        difficulty_time_ref_4 = np.linspace(-2, 2, nFrame//4)
        stimulus = np.concatenate((difficulty_time_ref_4, -difficulty_time_ref_4,
                                   difficulty_time_ref_4, -difficulty_time_ref_4))
    if freq < 0:
        stimulus = -stimulus
    j_eff, b_par, th, sigma, ndt = params  # add ndt
    lower_bound, upper_bound = np.array([-1, 1])*th + 0.5
    tau = 1
    dt = 1/fps
    b_eff = stimulus*b_par
    noise = np.random.randn(nFrame)*sigma*np.sqrt(dt/tau)
    x = np.zeros(nFrame)
    x1 = (np.sign(b_eff[0])+1)/2
    for i in range(ini_cond_convergence):
        x1 = sigmoid(2 * (j_eff * (2 * x1 - 1)+b_eff[0]))  # convergence
    x1 = lower_bound if np.sign(freq) == 1 else upper_bound
    x[0] = x1
    choice = np.zeros(nFrame)
    prev_choice = 0.0
    new_choice = 0.0
    ndt_frames = int(ndt / dt)
    sticky_end = -10_000     # when sticky period ends
    is_sticky = False
    
    for t in range(1, nFrame):
        # sticky period (freeze x to the bound, wait for choice update until NDT finishes)
        if is_sticky and t < sticky_end:
            x[t] = x[t - 1]  # hold value constant
            choice[t] = prev_choice
            continue

        # end of sticky period, change choice
        if is_sticky and t == sticky_end:
            prev_choice = new_choice
            is_sticky = False  # allow dynamics

        # dynamics
        drive = sigmoid(2 * (j_eff * (2 * x[t - 1] - 1) + b_eff[t]))
        x[t] = x[t - 1] + dt * (drive - x[t - 1]) / tau + noise[t]

        # detect bound crossing
        if x[t] >= upper_bound:
            new_choice = 1.0
            if new_choice != prev_choice and not is_sticky:
                x[t] = 0.5 + th * new_choice  # set to bound
                sticky_end = t + ndt_frames   # fixed duration
                is_sticky = True
        elif x[t] <= lower_bound:
            new_choice = -1.0
            if new_choice != prev_choice and not is_sticky:
                x[t] = 0.5 + th * new_choice
                sticky_end = t + ndt_frames
                is_sticky = True

        # keep choice
        choice[t] = prev_choice
    # plt.figure()
    # plt.plot(x); plt.plot(choice/2+0.5); plt.plot(stimulus/4+0.5); plt.axhline(lower_bound, alpha=0.3, linestyle='--'); plt.axhline(upper_bound, alpha=0.3, linestyle='--')
    # asd
    if return_choice:
        return choice, x
    else:
        return return_input_output_for_network(params, choice, freq, nFrame=nFrame, fps=fps)


def parameter_recovery_5_params(n_simuls_network=100000, fps=60, tFrame=26,
                                n_pars_to_fit=50, n_sims_per_par=108,
                                sv_folder=SV_FOLDER, simulate=False,
                                load_net=True, not_plot_and_return=False, pyddmfit=True,
                                ini_par=0, transform=False):
    if not pyddmfit:
        density_estimator, _ = sbi_training_5_params(n_simuls=n_simuls_network, fps=fps, tFrame=tFrame,
                                                     data_folder=DATA_FOLDER, load_net=load_net)
        lb = np.array([-0.5, -0.4, 0.0, 0.0])
        ub = np.array([2.5, 1., 0.45, 0.35])
        plb = np.array([-0.3, -0.1, 0.01, 0.05])
        pub = np.array([1.6, 0.4, 0.3, 0.3])
        x0 = np.array([0.2, 0.1, 0.2, 0.15])
        npars = 4
    if pyddmfit:
        npars = 5
    nFrame = fps*tFrame
    orig_params = np.zeros((n_pars_to_fit, npars))
    recovered_params = np.zeros((n_pars_to_fit, npars))
    for par in tqdm.tqdm(range(ini_par, n_pars_to_fit)):
        # simulate
        if pyddmfit:
            theta = np.load(sv_folder + 'param_recovery/pars_pyddm_prt' + str(par) + '.npy')
            pars = np.load(sv_folder + f'param_recovery/recovered_params_pyddm_{par}.npy')
        else:
            theta = np.load(sv_folder + 'param_recovery/pars_5_prt' + str(par) + '.npy')
            if simulate:
                freq = np.random.choice([2, 4, -2, -4], n_sims_per_par)
                training_input_set = np.zeros((theta.shape[0]+3), dtype=np.float32)
                training_output_set = np.empty((2), dtype=np.float32)
                for i in range(n_sims_per_par):
                    input_net, output_net = simulator_5_params(params=theta, freq=freq[i],
                                                               nFrame=nFrame, fps=fps)
                    training_input_set = np.row_stack((training_input_set, input_net))
                    training_output_set = np.row_stack((training_output_set, output_net))
                condition = training_input_set[1:].astype(np.float32)
                x = training_output_set[1:].astype(np.float32)
                x = torch.tensor(x).unsqueeze(0).to(torch.float32)
                condition = torch.tensor(condition).to(torch.float32)
                fun_to_minimize = lambda parameters: \
                    fun_get_neg_log_likelihood_5pars(parameters, x,
                                                     condition[:, -3:], density_estimator)
                options = {"display" : 'off',
                           "uncertainty_handling": False}
                # x0 = prior_parameters(1)[0][0].numpy()
                # while x0[-1] < 0.05 or x0[3] < 0.01:
                #     x0 = prior_parameters(1)[0][0].numpy()
                # x0 = np.clip(theta + (pub-plb)/10*np.random.randn(len(theta)),
                #              plb+1e-2, pub-1e-2)
                optimizer = BADS(fun_to_minimize, x0=x0,
                                 lower_bounds=lb,
                                 upper_bounds=ub,
                                 plausible_lower_bounds=plb,
                                 plausible_upper_bounds=pub,
                                 options=options).optimize()
                pars = optimizer.x
                np.save(sv_folder + 'param_recovery/pars_5_prt_recovered' + str(par) +  str(n_simuls_network) + '.npy',
                        np.array(pars))
            else:
                pars = np.load(sv_folder + 'param_recovery/pars_5_prt_recovered' + str(par) + str(n_simuls_network) + '.npy')
        orig_params[par] = theta
        recovered_params[par] = pars
    # transform vars
    if transform:
        orig_params[:, 2] /= orig_params[:, 3]
        orig_params[:, 4] /= orig_params[:, 3]
        recovered_params[:, 2] /= recovered_params[:, 3]
        recovered_params[:, 4] /= recovered_params[:, 3]
        orig_params[:, 0] /= orig_params[:, 2]
        orig_params[:, 1] /= orig_params[:, 2]
        recovered_params[:, 0] /= recovered_params[:, 2]
        recovered_params[:, 1] /= recovered_params[:, 2]
        orig_params[np.isnan(orig_params)] = 0
        recovered_params[np.isnan(recovered_params)] = 0
    if not_plot_and_return:
        return orig_params, recovered_params
    else:
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(15, 9))
        ax = ax.flatten()
        # labels = ['Jeff', ' B1',  'Tau', 'Thres.', 'sigma']
        if pyddmfit:
            labels_reduced = [r'$J_1$', r'$J_0$', r'$B_1$', r'$\sigma$', r'$c$']
            labels = [r'Coupling weight, $J_1$', r'Coupling bias, $J_0$',
                      r'Stimulus weight, $B_1$', r'Noise, $\sigma$', r'Threshold, $c$']
        else:
            labels = ['Jeff', ' B1', 'Thres.', 'sigma']
        # xylims = [[0, 3], [0, 0.8], [0, 0.7], [0, 0.5], [0, 0.5]]
        for i_a in range(npars):
            a = ax[i_a]
            a.plot(orig_params[:, i_a], recovered_params[:, i_a], color='k', marker='o',
                   markersize=5, linestyle='')
            maxval = np.nanmax([orig_params[:, i_a], recovered_params[:, i_a]])
            minval = np.nanmin([orig_params[:, i_a], recovered_params[:, i_a]])
            a.set_xlim(minval-1e-1, maxval+1e-1)
            a.set_ylim(minval-1e-1, maxval+1e-1)
            a.plot([minval-1e-1, maxval+1e-1], [minval-1e-1, maxval+1e-1],
                   color='k', linestyle='--',
                   alpha=0.3, linewidth=4)
            # a.plot(xylims[i_a], xylims[i_a], color='k', alpha=0.3)
            a.set_title(labels[i_a])
            a.set_xlabel('Original parameters')
            a.set_ylabel('Recovered parameters')
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
        ax[-1].axis('off')
        fig.tight_layout()
        fig2 = plt.figure(figsize=(10, 4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.075], wspace=0.2)
            
        ax2  = fig2.add_subplot(gs[0])
        ax = fig2.add_subplot(gs[1])
        cax = fig2.add_subplot(gs[2])

        # define correlation matrix
        corr_mat = np.empty((npars, npars))
        corr_mat[:] = np.nan
        for i in range(npars):
            for j in range(npars):
                # compute cross-correlation matrix
                corr_mat[i, j] = np.corrcoef(orig_params[:, i], recovered_params[:, j])[1][0]
        # plot cross-correlation matrix
        if np.min(corr_mat) < -0.2:
            vmin = -1
            cmap = 'bwr'
        else:
            vmin = 0
            cmap = 'Reds'
        im = ax.imshow(corr_mat.T, cmap=cmap, vmin=vmin, vmax=1)
        # tune panels
        plt.colorbar(im, cax=cax, label='Correlation')
        ax.set_xticks(np.arange(npars), labels_reduced, fontsize=12)  # , rotation='270'
        ax.set_yticks(np.arange(npars), labels_reduced, fontsize=12)
        ax.set_xlabel('Original parameters', fontsize=14)
        # compute correlation matrix
        mat_corr = np.corrcoef(recovered_params.T, rowvar=True)
        mat_corr *= np.tri(*mat_corr.shape, k=-1)
        mat_corr[mat_corr == 0] = np.nan
        # plot correlation matrix
        im = ax2.imshow(mat_corr, cmap=cmap, vmin=vmin, vmax=1)
        ax2.step(np.arange(0, npars+1)-0.5, np.arange(0, npars+1)-0.5, color='k',
                 linewidth=.7)
        ax2.set_xticks(np.arange(npars), labels_reduced, fontsize=12)  # , rotation='270'
        ax2.set_yticks(np.arange(npars), labels, fontsize=12)
        ax2.set_xlabel('Inferred parameters', fontsize=14)
        ax2.set_ylabel('Inferred parameters', fontsize=14)
        fig2.tight_layout()
        fig.savefig(SV_FOLDER + 'param_recovery_all.png', dpi=400, bbox_inches='tight')
        fig.savefig(SV_FOLDER + 'param_recovery_all.pdf', dpi=400, bbox_inches='tight')
        fig2.savefig(SV_FOLDER + 'param_recovery_correlations.png', dpi=400, bbox_inches='tight')
        fig2.savefig(SV_FOLDER + 'param_recovery_correlations.pdf', dpi=400, bbox_inches='tight')


def build_prior_sample_theta(n_simuls=100, coupling_offset=False,
                             stim_offset=False):
    if coupling_offset:
        if stim_offset:
            prior =\
                MultipleIndependent([
                    Uniform(torch.tensor([0.]),
                            torch.tensor([1.])),  # coupling weight
                    Uniform(torch.tensor([-0.1]),
                            torch.tensor([0.4])),  # coupling offset
                    Uniform(torch.tensor([-0.5]),
                            torch.tensor([0.5])),  # stimulus offset
                    Uniform(torch.tensor([0.]),
                            torch.tensor([1.5])),  # stimulus weight
                    Uniform(torch.tensor([0.0]),
                            torch.tensor([0.35]))],  # noise
                    validate_args=False)
        else:
            prior =\
                MultipleIndependent([
                    Uniform(torch.tensor([0]),
                            torch.tensor([1.])),  # coupling weight
                    Uniform(torch.tensor([-0.1]),
                            torch.tensor([0.4])),  # coupling offset
                    Uniform(torch.tensor([0.]),
                            torch.tensor([1.5])),  # stimulus weight
                    Uniform(torch.tensor([0.0]),
                            torch.tensor([0.35]))],  # noise
                    validate_args=False)
    else:
        if stim_offset:
            prior =\
                MultipleIndependent([
                    Uniform(torch.tensor([0.]),
                            torch.tensor([1.])),  # coupling weight
                    Uniform(torch.tensor([-0.5]),
                            torch.tensor([0.5])),  # stimulus offset
                    Uniform(torch.tensor([0.]),
                            torch.tensor([1.5])),  # stimulus weight
                    Uniform(torch.tensor([0.0]),
                            torch.tensor([0.35]))],  # noise
                    validate_args=False)
        else:
            prior =\
                MultipleIndependent([
                    Uniform(torch.tensor([0.]),
                            torch.tensor([1.])),  # coupling weight
                    Uniform(torch.tensor([0.]),
                            torch.tensor([1.5])),  # stimulus weight
                    Uniform(torch.tensor([0.0]),
                            torch.tensor([0.35]))],  # noise
                    validate_args=False)
    theta_all = prior.sample((n_simuls,))
    return theta_all, prior


def sbi_training(n_simuls=10000, fps=60, tFrame=26, data_folder=DATA_FOLDER,
                load_net=False, plot_posterior=True, coupling_offset=False,
                stim_offset=False, plot_diagnostics=True,
                summary_statistics_fitting=False):
    """
    Function to train network to approximate likelihood/posterior.
    """
    nFrame = fps*tFrame
    if not load_net:
        # experimental conditions
        coupling = np.random.choice([0, 0.3, 1], n_simuls)  # , p=[0.2, 0.2, 0.6])
        freq = np.random.choice([2, 4, -2, -4], n_simuls)
        # sample prior
        theta, prior = build_prior_sample_theta(n_simuls=n_simuls, coupling_offset=coupling_offset,
                                                stim_offset=stim_offset)
        inference = sbi.inference.NLE(prior=prior)
        # theta = torch.column_stack((theta, torch.tensor(freq)))
        # theta[:, 0] *= torch.tensor(coupling) 
        theta_np = theta.detach().numpy()
        # simulate
        print(f'Starting {n_simuls} simulations')
        time_ini = time.time()
        if summary_statistics_fitting:
            summary_statistics = np.zeros((n_simuls, 8), dtype=np.float32)
            for i in range(n_simuls):
                summary_statistics[i, :] = simulator(theta_np[i], coupling=1,
                                                     freq=freq[i], nFrame=nFrame,
                                                     fps=fps, n=3.92, coupling_offset=coupling_offset,
                                                     stim_offset=stim_offset,
                                                     summary_stats=summary_statistics_fitting)
            sims_tensor = torch.tensor(summary_statistics)
            if plot_diagnostics:
                plot_diagnostics_training(theta_np[::1000, :], summary_statistics[::1000, :])
        else:
            training_input_set = np.zeros((theta_np.shape[1]+3), dtype=np.float32)
            training_output_set = np.empty((2), dtype=np.float32)
            for i in range(n_simuls):
                input_net, output_net = simulator(theta_np[i], coupling=1,
                                                  freq=freq[i], nFrame=nFrame,
                                                  fps=fps, n=3.92, coupling_offset=coupling_offset,
                                                  stim_offset=stim_offset,
                                                  summary_stats=summary_statistics_fitting)
                training_input_set = np.row_stack((training_input_set, input_net))
                training_output_set = np.row_stack((training_output_set, output_net))
            training_input_set = training_input_set[1:].astype(np.float32)
            sims_tensor = torch.tensor(training_output_set[1:].astype(np.float32))
            theta = torch.tensor(training_input_set)
        print('Simulations finished.\nIt took:' + str(round((time.time()-time_ini)/60)) + ' min.')
        # train density estimator
        print(f'Starting training NLE with {n_simuls} simulations')
        time_ini = time.time()
        density_estimator = inference.append_simulations(theta, sims_tensor).train()
        print('Training finished.\nIt took:' + str(round((time.time()-time_ini)/60)) + ' min.')
        posterior = inference.build_posterior(density_estimator, prior=prior)
        with open(data_folder + f"/nle_{n_simuls}.p", "wb") as fh:
            pickle.dump(dict(estimator=density_estimator,
                             num_simulations=n_simuls), fh)
        with open(data_folder + f"/posterior_{n_simuls}.p", "wb") as fh:
            pickle.dump(dict(posterior=posterior,
                             num_simulations=n_simuls), fh)
        if plot_posterior and summary_statistics_fitting:
            # example parameters to simulate
            theta_example = np.array([0.6, 0., 0.4, 0.1])
            freq = np.random.choice([2, 4, -2, -4], 90)
            # simulation, coupling=1
            x_try_all = np.zeros((90, 8), dtype=np.float32)
            for i in range(90):
                x_try = simulator(theta_example, coupling=1, freq=freq[i], nFrame=nFrame,
                                  fps=fps, coupling_offset=coupling_offset,
                                  stim_offset=stim_offset)
                x_try_all[i] = x_try
            # sample from posterior
            samples_posterior = posterior.sample((5000,), x=x_try_all)
            if coupling_offset:
                if stim_offset:
                    limits = [[-0.5, 2], [-0.2, 0.5], [-1, 1], [-0.2, 2], [0.0, 0.5]]
                    labels = ['J1', 'J0', 'B0', 'B1', r'$\sigma$']
                else:
                    limits = [[-0.5, 2], [-1, 1], [-0.2, 2], [0.0, 0.5]]
                    labels = ['J1', 'J0',  'B1', r'$\sigma$']
            else:
                if stim_offset:
                    limits = [[-0.5, 2], [-1, 1], [-0.2, 2], [0.0, 0.5]]
                    labels = ['J1', 'B0', 'B1', r'$\sigma$']
                else:
                    limits = [[-0.5, 2], [-0.2, 2], [0.0, 0.5]]
                    labels = ['J1', 'B1', r'$\sigma$']
            # plot posterior samples in pairplot
            _ = analysis.pairplot(samples_posterior, limits=limits, figsize=(9, 9),
                                  labels=labels, points=theta_example,
                                  points_offdiag={"markersize": 6}, points_colors="r")
    if load_net:
        with open(data_folder + f"/nle_{n_simuls}.p", 'rb') as f:
            density_estimator = pickle.load(f)
        with open(data_folder + f"/posterior_{n_simuls}.p", 'rb') as f:
            posterior = pickle.load(f)
        density_estimator = density_estimator['estimator']
        posterior = posterior['posterior']
    return density_estimator, posterior


def save_5_params_recovery(n_pars=50, sv_folder=SV_FOLDER, i_ini=0):
    """
    Saves samples of 5 params: J_eff, B_1, tau, threshold distance, noise
    """
    for i in range(i_ini, n_pars):
        j0 = np.random.uniform(-0.3, 1.5)
        b10 = np.random.uniform(-0.1, 0.8)
        # tau0 = np.random.uniform(0.05, 5)
        threshold0 = np.random.uniform(0., 0.25)
        noise0 = np.random.uniform(0.05, 0.34)
        params = [j0, b10, threshold0, noise0]
        np.save(sv_folder + 'param_recovery/pars_5_prt' + str(i) + '.npy',
                np.array(params))


def fun_get_neg_log_likelihood_5pars(theta, x, result, density_estimator,
                                     epsilon=1e-4, ct_distro=1/(26*60)):
    theta = np.atleast_2d(theta); result = torch.atleast_2d(result)
    x = torch.atleast_2d(x)
    params_repeated = np.column_stack(np.repeat(theta.reshape(-1, 1), result.shape[0], 1))
    condition = torch.tensor(np.column_stack((params_repeated, result))).to(torch.float32)
    likelihood_nle = torch.exp(density_estimator.log_prob(x, condition))
    full_likelihood_with_contaminant = likelihood_nle*(1-epsilon)+epsilon*ct_distro
    log_likelihood_full = torch.log(full_likelihood_with_contaminant)
    return -torch.nansum(log_likelihood_full).detach().numpy()


def fun_get_neg_log_likelihood(theta, x, result, density_estimator, coupling=1,
                               epsilon=1e-3, ct_distro=1):
    params_repeated = np.column_stack(np.repeat(theta.reshape(-1, 1), result.shape[0], 1))
    if len(theta) == 5:
        params_repeated[:, 1] = params_repeated[:, 0] + params_repeated[:, 1]*coupling
        params_repeated = params_repeated[:, 1:]
    if len(theta) == 4:
        params_repeated[:, 0] *= coupling
    condition = torch.tensor(np.column_stack((params_repeated, result))).to(torch.float32)
    likelihood_nle = torch.exp(density_estimator.log_prob(x, condition))
    full_likelihood_with_contaminant = likelihood_nle*(1-epsilon)+epsilon*ct_distro
    log_likelihood_full = torch.log(full_likelihood_with_contaminant)
    return -torch.nansum(log_likelihood_full).detach().numpy()


def parameter_recovery(n_simuls_network=100000, fps=60, tFrame=26,
                       n_pars_to_fit=50, n_sims_per_par=120,
                       model='MF', sv_folder=SV_FOLDER, simulate=False,
                       load_net=True, not_plot_and_return=False):
    density_estimator, _ = sbi_training(n_simuls=n_simuls_network, fps=fps, tFrame=tFrame, data_folder=DATA_FOLDER,
                                        load_net=load_net, plot_posterior=False, coupling_offset=False,
                                        stim_offset=True, plot_diagnostics=False,
                                        summary_statistics_fitting=False)    
    lb = [0, -0.6, -0.2, 0.05]
    ub = [1.2, 0.6, 1.5, 0.5]
    plb = [0.1, -0.45, 0.1, 0.1]
    pub = [1.1, 0.45, 1.1, 0.3]
    x0 = [0.55, 0.01, 0.6, 0.15]
    nFrame = fps*tFrame
    orig_params = np.zeros((n_pars_to_fit, 4))
    recovered_params = np.zeros((n_pars_to_fit, 4))
    for par in range(n_pars_to_fit):
        # simulate
        theta = np.load(sv_folder + 'param_recovery/pars_prt' + str(par) + model + '.npy')
        if simulate:
            freq = np.random.choice([2, 4, -2, -4], n_sims_per_par)
            training_input_set = np.zeros((theta.shape[0]+3), dtype=np.float32)
            training_output_set = np.empty((2), dtype=np.float32)
            for i in range(n_sims_per_par):
                input_net, output_net = simulator(theta, coupling=1,
                                                  freq=freq[i], nFrame=nFrame,
                                                  fps=fps, n=3.92, coupling_offset=False,
                                                  stim_offset=True, summary_stats=False)
                training_input_set = np.row_stack((training_input_set, input_net))
                training_output_set = np.row_stack((training_output_set, output_net))
            condition = training_input_set[1:].astype(np.float32)
            x = training_output_set[1:].astype(np.float32)
            x = torch.tensor(x).unsqueeze(0).to(torch.float32)
            condition = torch.tensor(condition).to(torch.float32)
            fun_to_minimize = lambda parameters: fun_get_neg_log_likelihood(parameters, x, condition[:, -3:], density_estimator)
            optimizer = BADS(fun_to_minimize, x0,  # theta+np.random.randn()*0.02
                             lb, ub, plb, pub).optimize()
            pars = optimizer.x
            np.save(sv_folder + 'param_recovery/pars_prt_recovered' + str(par) + model + str(n_simuls_network) + '.npy',
                    np.array(pars))
        else:
            pars = np.load(sv_folder + 'param_recovery/pars_prt_recovered' + str(par) + model + str(n_simuls_network) + '.npy')
        orig_params[par] = theta
        recovered_params[par] = pars
    if not_plot_and_return:
        return orig_params, recovered_params
    else:
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(15, 9))
        numpars = 4
        ax = ax.flatten()
        if model in ['LBP', 'FBP']:
            labels = ['Coupling, J', 'Stimulus weight, B1', 'Bias, B0', 'noise', 'Alpha'][:numpars]
            xylims = [[0, 3], [0, 0.5], [0, 0.5], [0, 0.3], [0, 2]]
        if model == 'MF':
            labels = ['Coupling, J', 'Bias, B0', 'Stimulus weight, B1', 'noise']
            xylims = [[-0.5, 1.5], [-0.85, 0.85], [-0.2, 1.2], [0, 0.6]]
        if model == 'MF5':
            labels = ['Coupling, J1', 'Coupling bias, J0',  'Bias, B0', 'Stimulus weight, B1', 'noise']
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
        ax.set_xticks(np.arange(numpars), labels, fontsize=12)  # , rotation='270'
        ax.set_yticks(np.arange(numpars), labels_reduced, fontsize=12)
        ax.set_xlabel('Original parameters', fontsize=14)
        # compute correlation matrix
        mat_corr = np.corrcoef(recovered_params.T, rowvar=True)
        mat_corr *= np.tri(*mat_corr.shape, k=-1)
        # plot correlation matrix
        im = ax2.imshow(mat_corr, cmap='bwr', vmin=-1, vmax=1)
        ax2.step(np.arange(0, numpars)-0.5, np.arange(0, numpars)-0.5, color='k',
                 linewidth=.7)
        ax2.set_xticks(np.arange(numpars), labels, fontsize=12)  # , rotation='270'
        ax2.set_yticks(np.arange(numpars), labels, fontsize=12)
        ax2.set_xlabel('Inferred parameters', fontsize=14)
        ax2.set_ylabel('Inferred parameters', fontsize=14)
        fig2.tight_layout()
        fig.savefig(SV_FOLDER + 'param_recovery_all.png', dpi=400, bbox_inches='tight')
        fig.savefig(SV_FOLDER + 'param_recovery_all.pdf', dpi=400, bbox_inches='tight')
        fig2.savefig(SV_FOLDER + 'param_recovery_correlations.png', dpi=400, bbox_inches='tight')
        fig2.savefig(SV_FOLDER + 'param_recovery_correlations.pdf', dpi=400, bbox_inches='tight')


def simulator(params, coupling, freq, nFrame=1200, fps=60, n=3.92, coupling_offset=False,
              stim_offset=False, summary_stats=False):
    """
    Simulator. Takes set of `params` and simulates the system, returning summary statistics.
    """
    if abs(freq) == 2:
        difficulty_time_ref_2 = np.linspace(-2, 2, nFrame//2)
        stimulus = np.concatenate(([difficulty_time_ref_2, -difficulty_time_ref_2]))
    if abs(freq) == 4:
        difficulty_time_ref_4 = np.linspace(-2, 2, nFrame//4)
        stimulus = np.concatenate(([difficulty_time_ref_4, -difficulty_time_ref_4,
                                    difficulty_time_ref_4, -difficulty_time_ref_4]))
    if freq < 0:
        stimulus = -stimulus
    if not coupling_offset:
        j0 = 0
        if stim_offset:
            j_par, b0, b_par, sigma = params
        else:
            j_par, b_par, sigma = params
            b0 = 0
    else:
        if stim_offset:
            j_par, j0, b0, b_par, sigma = params
        else:
            j_par, j0, b_par, sigma = params
            b0 = 0
    dt = 1/fps
    b_eff = stimulus*b_par+b0
    j_eff = (j_par*coupling+j0)*n
    noise = np.random.randn(nFrame)*sigma*np.sqrt(dt)
    x = np.zeros(nFrame)
    x[0] = 0.5
    for t in range(1, nFrame):
        drive = sigmoid(2 * (j_eff *(2 * x[t-1] - 1) + b_eff[t]))
        x[t] = x[t-1] + dt * (drive - x[t-1]) + noise[t]
    choice = np.zeros(nFrame)
    choice[x < 0.4] = -1.
    choice[x > 0.6] = 1.
    mid_mask = (x >= 0.4) & (x <= 0.6)
    for t in np.where(mid_mask)[0]:
        choice[t] = choice[t - 1] if t > 0 else 0.
    if summary_stats:
        return return_summary_statistics(choice=choice, stimulus=stimulus,
                                         freq=freq, dt=dt, nFrame=nFrame)
    else:
        return return_input_output_for_network(params, choice, freq, nFrame=nFrame, fps=fps)


def correlation_recovery_vs_N_simuls(fps=60, tFrame=26,
                                     n_pars_to_fit=200,
                                     n_sims_per_par=120, mse=False):
    n_sims_list=[100, 1000, 10000, 50000, 100000, 250000,
                 500000, 1000000, 2000000, 3000000, 5000000]
    corr_mat = np.zeros((4, len(n_sims_list)))
    for i_n, n_sims in enumerate(n_sims_list):
        try:
            orig_params, recovered_params =\
                parameter_recovery(n_simuls_network=n_sims, fps=fps, tFrame=tFrame,
                                   n_pars_to_fit=n_pars_to_fit, n_sims_per_par=n_sims_per_par,
                                   model='MF', sv_folder=SV_FOLDER, simulate=False,
                                   load_net=True, not_plot_and_return=True)
            if not mse:
                corr_array = np.zeros((4))
                for i in range(4):
                    corr_array[i] = np.corrcoef(orig_params[:, i],
                                                recovered_params[:, i])[1][0]
            if mse:
                corr_array = np.nansum((orig_params-recovered_params)**2, axis=0)
            corr_mat[:, i_n] = corr_array
        except FileNotFoundError:
            corr_mat[:, i_n] = np.nan
    fig, ax = plt.subplots(ncols=4, figsize=(15, 4.5))
    titles = ['J', 'B0', 'B1', 'Noise']
    for i_ax, a in enumerate(ax):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.plot(n_sims_list, corr_mat[i_ax, :], color='k', linewidth=4,
               marker='o')
        a.set_xscale('log')
        a.set_xlabel('# Simulations network')
        a.set_title(titles[i_ax])
        if not mse:
            a.set_ylim(0, 1)
    if not mse:
        ax[0].set_ylabel('Correlation recovered-original')
    if mse:
        ax[0].set_ylabel('MSE recovered-original')
    fig.tight_layout()


def correlation_recovery_vs_N_simuls(fps=60, tFrame=26,
                                     n_pars_to_fit=100,
                                     n_sims_per_par=120, mse=False):
    n_sims_list=[100, 1000, 10000, 50000, 100000, 500000,
                 1000000, 2000000, 5000000]
    corr_mat = np.zeros((4, len(n_sims_list)))
    for i_n, n_sims in enumerate(n_sims_list):
        try:
            orig_params, recovered_params =\
                parameter_recovery_5_params(n_simuls_network=n_sims, fps=fps, tFrame=tFrame,
                                   n_pars_to_fit=n_pars_to_fit, n_sims_per_par=n_sims_per_par,
                                   sv_folder=SV_FOLDER, simulate=False,
                                   load_net=True, not_plot_and_return=True)
            if not mse:
                corr_array = np.zeros((4))
                for i in range(4):
                    corr_array[i] = np.corrcoef(orig_params[:, i],
                                                recovered_params[:, i])[1][0]
            if mse:
                corr_array = np.nansum((orig_params-recovered_params)**2, axis=0)
            corr_mat[:, i_n] = corr_array
        except FileNotFoundError:
            corr_mat[:, i_n] = np.nan
    fig, ax = plt.subplots(ncols=4, figsize=(15, 4.5))
    titles = ['J', 'B1', 'Thresh', 'Sigma']
    for i_ax, a in enumerate(ax):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.plot(n_sims_list, corr_mat[i_ax, :], color='k', linewidth=4,
               marker='o')
        a.set_xscale('log')
        a.set_xlabel('# Simulations network')
        a.set_title(titles[i_ax])
        if not mse:
            a.set_ylim(-0.3, 1)
    if not mse:
        ax[0].set_ylabel('Correlation recovered-original')
    if mse:
        ax[0].set_ylabel('MSE recovered-original')
    fig.tight_layout()


def fit_data(data_folder=DATA_FOLDER, ntraining=8,
             n_simuls_network=50000, fps=60, tFrame=26,
             sv_folder=SV_FOLDER, load_net=True, use_j0=True, contaminants=True):
    """
    Fits data. Parameters:
        1. J0
        2. J1
        3. B1
        4. Threshold
        5. Sigma
    """
    density_estimator, _ = sbi_training_5_params(n_simuls=n_simuls_network, fps=fps, tFrame=tFrame,
                                                 data_folder=DATA_FOLDER, load_net=load_net)
    lb = np.array([-1, -0.5, -0.5, 0.0, 0.02])
    ub = np.array([2., 2.2, 1., 0.3, 0.4])
    plb = np.array([-0.15, 0.3, 0.05, 0.01, 0.04])
    pub = np.array([0.7, 1.8, 0.9, 0.25, 0.35])
    x0 = np.array([0.01, 1.2, 0.1, 0.05, 0.15])
    nFrame = fps*tFrame
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()
    label_j0 = '_with_j0' if use_j0 else ''
    for i_s, subject in enumerate(subjects):
        print('Fitting subject', subject)
        df_subject = df.loc[df.subject == subject]
        tensor_input, tensor_output, idx_filt = prepare_data_for_fitting(df_subject, fps=fps, tFrame=tFrame)
        pshuffle = df_subject.pShuffle.values[~idx_filt][1:]
        condition = tensor_input[1:].astype(np.float32)
        x = tensor_output[1:].astype(np.float32)
        x = torch.tensor(x).unsqueeze(0).to(torch.float32)
        condition = torch.tensor(condition).to(torch.float32)
        fun_to_minimize = lambda parameters: \
            fun_get_neg_log_likelihood_data(parameters, x, condition, density_estimator, pshuffle,
                                            use_j0=use_j0, contaminants=contaminants)
        options = {"display" : 'off',
                   "uncertainty_handling": False}
        if use_j0:
            optimizer = BADS(fun_to_minimize, x0=x0,
                             lower_bounds=lb,
                             upper_bounds=ub,
                             plausible_lower_bounds=plb,
                             plausible_upper_bounds=pub,
                             options=options).optimize()
        else:
            optimizer = BADS(fun_to_minimize, x0=x0[1:],
                             lower_bounds=lb[1:],
                             upper_bounds=ub[1:],
                             plausible_lower_bounds=plb[1:],
                             plausible_upper_bounds=pub[1:],
                             options=options).optimize()
        pars = optimizer.x
        if not use_j0:
            pars = np.concatenate(([np.random.rand()], pars))
        print('Pars.: ', np.round(pars, 2), 'f_val:', np.round(optimizer.fval, 2))
        np.save(sv_folder + '/pars_5_subject_' + subject + str(n_simuls_network) + label_j0 + '.npy',
                np.array(pars))


def plot_fitted_params(data_folder=DATA_FOLDER, n_simuls_network=50000,
                       sv_folder=SV_FOLDER, use_j0=False):
    label_j0 = '_with_j0' if use_j0 else ''
    df = load_data(data_folder, n_participants='all')
    subjects = df.subject.unique()
    pars_all = np.zeros((5, len(subjects)))
    for i_s, subject in enumerate(subjects):
        fitted_params = np.load(sv_folder + '/pars_5_subject_' + subject + str(n_simuls_network) + label_j0 + '.npy')
        pars_all[:, i_s] = fitted_params
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(9, 5)); ax=ax.flatten()
    ax2 = ax[-1]
    titles = [r'$J_0$', r'$J_1$', r'$B_1$', r'$\theta$', r'$\sigma$']
    lb = np.array([-1, -0.5, -0.5, 0.0, 0.02])
    ub = np.array([2., 2.2, 1., 0.45, 0.4])
    for i_a, a in enumerate(ax[:-1]):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.spines['left'].set_visible(False)
        a.axvline(lb[i_a], color='gray', linestyle='--', linewidth=2.5, alpha=0.6)
        a.axvline(ub[i_a], color='gray', linestyle='--', linewidth=2.5, alpha=0.6)
        sns.violinplot(pars_all[i_a], ax=a, inner=None, fill=False,
                       edgecolor='k', orient='y')
        sns.swarmplot(pars_all[i_a], ax=a, color='k', orient='y', size=3)
        a.set_xlabel(titles[i_a])
    if use_j0:
        ax[0].axis('off')
    fig.tight_layout()
    # fig2, ax2 = plt.subplots(ncols=1)
    numpars = pars_all.shape[0]
    mat_corr = np.corrcoef(pars_all, rowvar=True)
    mat_corr *= np.tri(*mat_corr.shape, k=-1)
    im = ax2.imshow(mat_corr, cmap='bwr', vmin=-1, vmax=1)
    ax2.step(np.arange(0, numpars)-0.5, np.arange(0, numpars)-0.5, color='k', linewidth=.7)
    ax2.set_xticks(np.arange(numpars), titles, fontsize=12, rotation=45)  # , rotation='270'
    ax2.set_yticks(np.arange(numpars), titles, fontsize=12)
    plt.colorbar(im, ax=ax2, label='Correlation', shrink=0.6, aspect=10)
    
    # fig2.tight_layout()


def fitting_pipeline(n_simuls_network=50000, use_j0=False, contaminants=True,
                     fit=False, plot_lmm=True, plot_pars=True, simulate=False):
    if fit:
        fit_data(data_folder=DATA_FOLDER, ntraining=8,
                  n_simuls_network=n_simuls_network, fps=60, tFrame=26,
                  sv_folder=SV_FOLDER, load_net=True, use_j0=use_j0, contaminants=contaminants)
    if plot_lmm:
        plot_j1_lmm_slopes(data_folder=DATA_FOLDER, sv_folder=SV_FOLDER,
                           n_simuls_network=n_simuls_network, use_j0=use_j0)
    if plot_pars:
        plot_fitted_params(data_folder=DATA_FOLDER, n_simuls_network=n_simuls_network,
                           sv_folder=SV_FOLDER, use_j0=use_j0)
    if simulate:
        simulated_subjects(data_folder=DATA_FOLDER, tFrame=26, fps=60,
                           sv_folder=SV_FOLDER, ntraining=8, n_simuls_network=n_simuls_network,
                           plot=True, simulate=simulate, use_j0=use_j0)


def model_pyddm(plot=False, ndt=0, n=4, t_dur=15):
    stim = lambda t, freq, phase_ini: sawtooth(2 * np.pi * abs(freq)/2 * (t+phase_ini)/26, 0.5)*2*np.sign(freq)
    x_hat = lambda prev_choice, x: x if prev_choice == -1 else x+1
    drift_function = lambda t, x, j1, j0, b, pshuffle, prev_choice, freq, phase_ini: 1/(1+np.exp(-2*(n*(j0+j1*(1-pshuffle))*(2*x_hat(prev_choice, x)-1) + b*stim(t, freq, phase_ini))))-x_hat(prev_choice, x)
    parameters = {"j1": (0., 0.4), "j0": (0., 0.4), "b": (0., 0.7), "sigma": (0.05, 0.3), "theta": (0., 0.4)}
    bound = lambda theta: 0.5+theta
    starting_position = lambda theta, prev_choice: 0.5-theta if prev_choice == -1 else -0.5+theta
    conditions = ["pshuffle", "prev_choice", "freq", "phase_ini"]
    noise = lambda sigma: sigma
    nondecision = ndt
    model = pyddm.gddm(drift=drift_function, parameters=parameters,
                       conditions=conditions, starting_position=starting_position, bound=bound, noise=noise,
                       T_dur=t_dur, dt=0.005, dx=0.005, nondecision=nondecision)
    if plot:
        pyddm.plot.model_gui(model, conditions={"pshuffle": [0, 0.3, 1], "prev_choice": [-1, 1], "freq": [2, 4], "phase_ini": [0, 6.5, 13, 19.5]})
    return model


def null_model_pyddm(plot=False, ndt=0, n=4, t_dur=15):
    stim = lambda t, freq, phase_ini: sawtooth(2 * np.pi * abs(freq)/2 * (t+phase_ini)/26, 0.5)*2*np.sign(freq)
    x_hat = lambda prev_choice, x: x if prev_choice == -1 else x+1
    drift_function = lambda t, x, j0, b, pshuffle, prev_choice, freq, phase_ini: 1/(1+np.exp(-2*(n*j0*(2*x_hat(prev_choice, x)-1) + b*stim(t, freq, phase_ini))))-x_hat(prev_choice, x)
    parameters = {"j0": (0., 0.4), "b": (0., 0.7), "sigma": (0.05, 0.3), "theta": (0., 0.4)}
    bound = lambda theta: 0.5+theta
    starting_position = lambda theta, prev_choice: 0.5-theta if prev_choice == -1 else -0.5+theta
    conditions = ["pshuffle", "prev_choice", "freq", "phase_ini"]
    noise = lambda sigma: sigma
    nondecision = ndt
    model = pyddm.gddm(drift=drift_function, parameters=parameters,
                       conditions=conditions, starting_position=starting_position, bound=bound, noise=noise,
                       T_dur=t_dur, dt=0.005, dx=0.005, nondecision=nondecision)
    if plot:
        pyddm.plot.model_gui(model, conditions={"pshuffle": [0, 0.3, 1], "prev_choice": [-1, 1], "freq": [2, 4], "phase_ini": [0, 6.5, 13, 19.5]})
    return model


def model_pyddm_reparameterized(plot=False, n=4, t_dur=15):
    stim = lambda t, freq, phase_ini: sawtooth(2 * np.pi * abs(freq)/2 * (t+phase_ini)/26, 0.5)*2*np.sign(freq)
    x_hat = lambda prev_choice, x: x if prev_choice == -1 else x+1
    drift_function = lambda t, x, j1, j0, beta, sigma, pshuffle, prev_choice, freq, phase_ini: 1/(1+np.exp(-2*(n*(j0+j1*(1-pshuffle))*(2*x_hat(prev_choice, x)-1) + beta*sigma*stim(t, freq, phase_ini))))-x_hat(prev_choice, x)
    parameters = {"j1": (-0.1, 0.4), "j0": (-0.1, 0.3), "beta": (-0.1, 0.7), "sigma": (0.05, 0.3), "tau": (0., 0.4)}  # , "j0": (-0.1, 0.3)
    bound = lambda tau, sigma: 0.5+tau*sigma
    starting_position = lambda tau, sigma, prev_choice: 0.5-tau*sigma if prev_choice == -1 else -0.5+tau*sigma
    conditions = ["pshuffle", "prev_choice", "freq", "phase_ini"]
    noise = lambda sigma: sigma
    model = pyddm.gddm(drift=drift_function, parameters=parameters,
                       conditions=conditions, starting_position=starting_position, bound=bound, noise=noise,
                       T_dur=t_dur, dt=0.001, dx=0.001)
    if plot:
        pyddm.plot.model_gui(model, conditions={"pshuffle": [0, 0.3, 1], "prev_choice": [-1, 1], "freq": [2, 4], "phase_ini": [0, 6.5, 13, 19.5]})
    return model


def null_model_known_params_pyddm(J0=0.1, B=0.4, THETA=0.1, SIGMA=0.1, NDT=0, n=4, t_dur=10):
    # First create two versions of the model, one to simulate the data, and one to fit to the simulated data.
    stim = lambda t, freq, phase_ini: sawtooth(2 * np.pi * abs(freq)/2 * (t+phase_ini)/26, 0.5)*2*np.sign(freq)
    x_hat = lambda prev_choice, x: x if prev_choice == -1 else x+1
    starting_position = lambda prev_choice: 0.5-THETA if prev_choice == -1 else -0.5+THETA
    drift_function_sim = lambda t, x, pshuffle, prev_choice, freq, phase_ini: 1/(1+np.exp(-2*(n*(J0)*(2*x_hat(prev_choice, x)-1) + B*stim(t, freq, phase_ini))))-x_hat(prev_choice, x)
    conditions = ["pshuffle", "prev_choice", "freq", "phase_ini"]
    m_sim = pyddm.gddm(drift=drift_function_sim, 
                       conditions=conditions, starting_position=starting_position, bound=THETA+0.5, noise=SIGMA, nondecision=NDT,
                       T_dur=t_dur, dt=0.005, dx=0.005)
    return m_sim


def simple_recovery_pyddm(J1=0.3, J0=0.1, B=0.4, THETA=0.1, SIGMA=0.1, NDT=0.1, ncpus=10, plot=False, idx=0):
    set_N_cpus(ncpus)
    model = model_pyddm(t_dur=15)
    m_sim = model_known_params_pyddm(J1=J1, J0=J0, B=B, SIGMA=SIGMA, THETA=THETA, NDT=NDT, t_dur=10)
    freqs = [2, 4]
    prev_choice = [-1, 1]
    pshuffle = [0, 0.3, 1]
    combs = list(itertools.product(freqs, prev_choice))
    SAMPLE_SIZE = 250
    pshuffles_i = np.random.choice(pshuffle, ncpus)
    freqs_i = np.random.choice(freqs, ncpus)
    phase_inis_i = np.random.uniform(0, 6.5, ncpus)
    for j in range(SAMPLE_SIZE):
        pshuffle_i = pshuffles_i[j % ncpus]
        freq_i = freqs_i[j % ncpus]
        phase_ini_i = phase_inis_i[j % ncpus]
        if freq_i == 2:
            prev_choice_i = -1
        else:
            if phase_ini_i <= 3.25:
                prev_choice_i = -1
            else:
                prev_choice_i = 1
        sample1 = m_sim.solve(conditions={"pshuffle": pshuffle_i, "freq": freq_i, "prev_choice": prev_choice_i, "phase_ini": phase_ini_i}).sample(1)
        if j == 0:
            sample_all = sample1
        else:
            sample_all = sample_all + sample1
    model.fit(sample_all, verbose=False, fitting_method='bads')
    params = model.get_model_parameters()
    # Convert to a numpy array for ease
    params = np.asarray(params)
    np.save(SV_FOLDER + f'param_recovery/recovered_params_pyddm_{idx}_ndt.npy', params)
    if plot:
    
        # Plot the histogram for each parameter
        plt.subplot(3,2,1)
        plt.hist(params[:,0])
        plt.axvline(J1, c='k', linewidth=3)
        plt.title(model.get_model_parameter_names()[0])
        
        plt.subplot(3,2,2)
        plt.hist(params[:,1])
        plt.axvline(J0, c='k', linewidth=3)
        plt.title(model.get_model_parameter_names()[1])
        
        plt.subplot(3,2,3)
        plt.hist(params[:,2])
        plt.axvline(B, c='k', linewidth=3)
        plt.title(model.get_model_parameter_names()[2])
        
        
        plt.subplot(3,2,4)
        plt.hist(params[:,3])
        plt.axvline(SIGMA, c='k', linewidth=3)
        plt.title(model.get_model_parameter_names()[3])
        
        plt.subplot(3,2,5)
        plt.hist(params[:,4])
        plt.axvline(THETA, c='k', linewidth=3)
        plt.title(model.get_model_parameter_names()[4])
        
        plt.tight_layout()


def save_params_pyddm_recovery(n_pars=100, i_ini=0,
                               sv_folder=SV_FOLDER):
    """
    Saves samples of 5 params: J_0, J1, B_1, threshold distance, noise
    """
    for i in range(i_ini, n_pars):
        j1 = np.random.uniform(-0.1, 0.4)
        j0 = np.random.uniform(-0.1, 0.3)
        b1 = np.random.uniform(0.1, 0.7)
        sigma = np.random.uniform(0.05, 0.3)
        theta = np.random.uniform(0.01, 0.35)
        ndt = np.random.uniform(0.1, 0.5)
        params = [j1, j0, b1, sigma, theta, ndt]
        np.save(sv_folder + 'param_recovery/pars_pyddm_prt_ndt' + str(i) + '.npy',
                np.array(params))


def recovery_pyddm(n_pars=50, sv_folder=SV_FOLDER, n_cpus=10, i_ini=0):
    for i in tqdm.tqdm(range(i_ini, n_pars)):
        J1, J0, B, SIGMA, THETA, NDT = np.load(sv_folder + 'param_recovery/pars_pyddm_prt_ndt' + str(i) + '.npy')
        simple_recovery_pyddm(J1=J1, J0=J0, B=B, THETA=THETA, SIGMA=SIGMA, NDT=NDT, ncpus=n_cpus,
                              plot=False, idx=i)

# import pandas as pd
# # Load the CSV file
# file_path_marti = "C:/Users/alexg/Downloads/roster_download_2.csv"
# df_marti = pd.read_csv(
#     file_path_marti,
#     skiprows=1,
#     sep=",",
#     quotechar='"',
#     engine="python",
#     encoding="utf-8-sig",
# )
# df_marti['full_name'] = df_marti['first_name'] + df_marti['last_name']

# file_path_alex = "C:/Users/alexg/Downloads/roster_download_5.csv"
# df_alex = pd.read_csv(
#     file_path_alex,
#     skiprows=1,
#     sep=",",
#     quotechar='"',
#     engine="python",
#     encoding="utf-8-sig",
# )
# df_alex['full_name'] = df_alex['first_name'] + df_alex['last_name']

# common_login_ids = set(df_alex['full_name']).intersection(df_marti['full_name'])
# print(f"Number of participants in both files: {len(common_login_ids)}")
# print(common_login_ids)


def fit_data_pyddm(data_folder=DATA_FOLDER, ncpus=10, ntraining=8,
                   t_dur=15, subj_ini=None, nbins=27, fitting_method='differential_evolution'):
    if ncpus is not None:
        set_N_cpus(ncpus)
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()
    bins = np.linspace(0, 26, nbins).round(2)
    ndt = np.abs(np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
    if subj_ini is not None:
        idx = np.where(subjects == subj_ini)[0][0]
        subjects = subjects[idx:]
    print('Fitting ', len(subjects), ' subjects')
    for i_s, subject in enumerate(subjects):
        print('Fitting subject ', subject)
        model = model_pyddm(t_dur=t_dur, ndt=ndt)
        df_sub = df.loc[(df.subject == subject) & (df.response > 0)]
        pshuffles = df_sub.pShuffle.values
        freqs = df_sub.freq.values*df_sub.initial_side.values
        phase_inis = df_sub.keypress_seconds_onset.values
        prev_choices = (df_sub.response.values-1)*2-1
        next_choice = -((df_sub.response.values-1)-1)
        rt = df_sub.keypress_seconds_offset.values-phase_inis
        rt_idx = (rt < t_dur) * (rt > 0.1)
        print(sum(rt_idx), ' trials')
        # not_last_change_idx = phase_inis < 25
        df_fit = pd.DataFrame({'prev_choice': prev_choices,
                               "freq": freqs, "phase_ini": bins[np.digitize(phase_inis-1e-5, bins)],
                               "pshuffle": pshuffles, "next_choice": next_choice,
                               "rt": rt})[rt_idx]
        sample_all = pyddm.Sample.from_pandas_dataframe(df_fit, rt_column_name="rt", choice_column_name="next_choice")
        print('Start actual fitting')
        try:
            model.fit(sample_all, verbose=True, fitting_method=fitting_method)
            params = model.get_model_parameters()
            print('Fitted params: ', np.round(np.asarray(params), 4))
        except Exception:
            print('Error, subject ', subject, ' could not be fitted.')
            params = np.full(6, np.nan)
        # Convert to a numpy array for ease
        params = np.asarray(params)
        np.save(SV_FOLDER + f'fitted_params/fitted_with_preprocessed_data_more_bins/fitted_params_pyddm_{subject}_fixed_ndt_preprocessed.npy', params)


def instanton_rhs(t, y, J, B, D):
    x, dx = y
    fx = drive(x, J, B)
    fxp = f_prime(x, J, B)
    fxx = f_double_prime(x, J, B)
    ddx = fx * fxp + D * fxx
    return np.vstack((dx, ddx))


def optimal_eta(x, dx, J, B, D):
    return (dx - drive(x, J, B)) / np.sqrt(2*D)


def optimal_escape_eta(j,  time, stim=None):
    fun_to_minimize = lambda q: sigmoid(2*j*(2*q-1))-q
    val1 = fsolve(fun_to_minimize, 0)+5e-3
    x = np.zeros(len(time))
    x[0] = val1
    dt = np.diff(time)[0]
    if stim is None:
        stim = np.zeros(len(time))
    for t in range(1, len(time)):
        x[t] = x[t-1] - drive(x[t-1], j, b=stim[t])*dt
    eta = -2*drive(x, j, b=stim)
    return eta


def optimal_escape_eta_with_all_functional(j=1, theta=0., D=0.025):
    B = 0.0
    if j < 1:
        x0 = 0.5-theta
    else:
        x0 = 0.01
    xT = 0.5 + theta

    # --- Initial mesh and guess --------------------------------------------
    t = np.linspace(0, 10, 200)
    y_init = np.zeros((2, t.size))
    y_init[0] = np.linspace(x0, xT, t.size)

    # --- Solve BVP ----------------------------------------------------------
    sol = solve_bvp(lambda t, y: instanton_rhs(t, y, j, B, D),
                    lambda y0, yT: bc(y0, yT, x0, xT),
                    t, y_init)

    if sol.success:
        x_opt = sol.y[0]
        dx_opt = sol.y[1]
        time_vals = sol.x
        eta_opt = (dx_opt - drive(x_opt, j, B)) / np.sqrt(2*D)

        plt.figure(figsize=(6, 3))
        plt.subplot(121)
        plt.plot(time_vals, x_opt, label="q(t)", color='k', linewidth=3)
        plt.xlabel("Time, t"); plt.ylabel("q")
        plt.legend()

        plt.subplot(122)
        plt.plot(time_vals, eta_opt, label="η(t)", color='k', linewidth=3)
        plt.xlabel("Time, t"); plt.ylabel("η(t)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("BVP failed:", sol.message)


def optimal_b_escape(J=1.5, theta=0., k=1):
    if J > 1:
        q0 = 0.05  # initial position (left attractor)
    else:
        q0 = 0.5-theta
    qT = 0.5 + theta
    
    T = 10
    t = np.linspace(0,T,500)
    
    q = q0 + (qT-q0)/(1+np.exp(-k*(t-T/2)))
    
    dq = np.gradient(q,t)
    
    B = 0.5*(np.log((q+dq)/(1-(q+dq))) - 2*J*(2*q-1))
    
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3))
    ax[0].plot(t,q,label="q(t)", color='k', linewidth=3)
    ax[1].plot(t,B,label="B(t)", color='k', linewidth=3)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('q(t)')
    ax[1].set_ylabel('B(t)')
    fig.tight_layout()


def optimal_eta_and_stimuli(J=1, sigma=0.1, alpha=1, theta=0.05, T=10,
                            timepoints=10000):
    if J > 1:
        q0 = 0.05  # initial position (left attractor)
    else:
        q0 = 0.5-theta
    
    target = 0.5 + theta
    
    
    def f(z):
        return 1/(1+np.exp(-z))
    
    def fp(z):
        return f(z)*(1-f(z))
    
    def compute_B(q,p):

        B=0
        for _ in range(100):
            z=2*J*(2*q-1)+2*B
            B=-alpha*p*fp(z)
        return B

    def dynamics(t,y):

        q,p=y
        B=compute_B(q,p)
        z=2*J*(2*q-1)+2*B
        
        dq = f(z)-q - sigma**2*p
        dp = -p*(2*J*fp(z)-1)

        return [dq,dp]

    def shoot(p0, q0=q0):

        sol = solve_ivp(
            dynamics,
            [0,T],
            [q0,p0],
            t_eval=[T]
        )
    
        qT = sol.y[0,-1]

        return qT - target

    root = root_scalar(
    shoot,
    bracket=[-10,10],
    method='brentq'
    )

    p0 = root.root
    y0=[q0, p0]
    print(y0)
    t = np.linspace(0,T,timepoints)
    sol = solve_ivp(dynamics,[0,T],y0,t_eval=t)

    q=sol.y[0]
    p=sol.y[1]

    B = np.array([compute_B(q[i],p[i]) for i in range(len(q))])

    dq = np.gradient(q, t)

    eta_check = dq - f(2*J*(2*q-1)+2*B) + q

    fig, ax = plt.subplots(ncols=3, figsize=(8, 3))
    time_switch = -sol.t[::-1]
    ax[0].plot(time_switch,q,label="q(t)", linewidth=3, color='k')
    ax[0].set_ylabel('q(t)')
    ax[1].plot(time_switch,-p*sigma**2,label=r"$-p(t)\sigma^2$", linewidth=3, color='k')
    ax[1].plot(time_switch,eta_check,label=r"$\eta(t)$", linewidth=3, color='firebrick')
    ax[1].legend(frameon=False)
    ax[1].set_ylabel('$\eta(t)$')
    ax[2].plot(time_switch,B,label="B(t)", linewidth=3, color='k')
    ax[2].set_ylabel('B(t)')
    for a in ax:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
        a.set_xlabel('Time from switch (s)')
    fig.tight_layout()


def simple_optimal_eta_quartic_potential(a=1):
    # time
    t = np.linspace(-10, 5, 400)
    # optimal escape path
    x = -np.sqrt(a) / np.sqrt(1 + np.exp(-2 * a * (t+0.5)))
    # deterministic drift
    f = x*a - x**3
    # optimal noise (\eta)
    eta = -2 * f
    # Potential
    xv = np.linspace(-1.5, 1.5, 200)
    V = 0.25*xv**4 - 0.5*xv**2*a
    fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(5, 10))
    for i_a, ax_i in enumerate(ax):
        ax_i.spines['right'].set_visible(False); ax_i.spines['top'].set_visible(False)
    ax[0].plot(xv, V, color='k', linewidth=3)
    ax[0].plot([-np.sqrt(a), 0, np.sqrt(a)],
               [V[np.searchsorted(xv, -np.sqrt(a))],
                V[np.searchsorted(xv, 0)],
                V[np.searchsorted(xv, np.sqrt(a))]], color='r',
               marker='o', linestyle='')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('Potential,\n' + r'$V(x)=-x^2/2+x^4/4$')
    ax[1].plot(t, f, color='k', linewidth=3)
    ax[1].set_ylabel(r'Drift,  $f(x)=x-x^3$')
    ax[2].plot(t, x[::-1], linewidth=3, color='k')
    ax[2].set_ylabel(r'Trajectory,  $x(t)$')
    ax[3].plot(t, eta, linewidth=3, color='k')
    ax[3].set_ylabel('Optimal escape noise, \n' + r'$\eta(t)=-2f(x)$')
    ax[3].set_xlabel('Time from switch (s)')
    fig.tight_layout()


def plot_optimal_eta_b_vs_0(ntrials=10, j=1):
    time = np.arange(0, 25, 1e-3)
    dt = np.diff(time)[0]
    noisyframes = 15 // dt // 60
    nFrame = len(time)
    time_interp = np.arange(0, nFrame+noisyframes, noisyframes)*dt
    time = np.arange(0, nFrame, 1)*dt
    noise_exp = np.random.randn(len(time_interp), ntrials)
    noise_signal = np.array([interp1d(time_interp, noise_exp[:, trial])(time) for trial in range(ntrials)]).T
    eta_0 = optimal_escape_eta(j, time+1, stim=None)
    all_etas_stim = []
    for i in range(ntrials):
        stim = noise_signal[:, i]*0.03
        eta_stim = optimal_escape_eta(j, time+1, stim=stim)
        all_etas_stim.append(eta_stim)
    eta_stim = np.nanmean(all_etas_stim, axis=0)
    fig, ax = plt.subplots(ncols=2, figsize=(7, 4))
    ax[0].plot(time, eta_0, color='k', linewidth=3)
    ax[1].plot(time, eta_stim, color='k', linewidth=3)
    for a in ax:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
        a.axhline(0, color='k', linestyle='--', linewidth=3, alpha=0.2)
    ax[1].set_ylim(ax[0].get_ylim())
    fig.tight_layout()


def plot_simulated_subjects_noise_trials(data_folder=DATA_FOLDER,
                                         shuffle_vals=[1., 0.7, 0.], ntrials=36,
                                         steps_back=120, steps_front=20, avoid_first=True,
                                         tFrame=26, window_conv=1,
                                         zscore_number_switches=False, fps=60, ax=None, hysteresis_area=False,
                                         normalize_variables=False, ratio=1, nFrame=1546, n=4,
                                         load_simulations=False,
                                         adaptation=False):
    ndt = np.abs(np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = [[n*params[0], n*params[1], params[2], params[4], params[3], ndt] for params in fitted_params_all]
    steps_back = steps_back*ratio; steps_front = steps_front*ratio
    nFrame = nFrame*ratio; fps= fps*ratio
    df = load_data(data_folder + '/noisy/', n_participants='all')
    noise_signal, choice, pshuffles = simulate_noise_subjects(df, data_folder=DATA_FOLDER, n=4, nFrame=nFrame, fps=fps,
                                                              load_simulations=load_simulations,
                                                              adaptation=adaptation)
    # print(len(df.trial_index.unique()))
    subs = df.subject.unique()[:pshuffles.shape[0]]
    print(subs, ', number:', len(subs))
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    mean_peak_latency = np.empty((len(shuffle_vals), len(subs)))
    mean_peak_latency[:] = np.nan
    mean_peak_amplitude = np.empty((len(shuffle_vals), len(subs)))
    mean_peak_amplitude[:] = np.nan
    mean_vals_noise_switch_coupling = np.empty((len(shuffle_vals), steps_back+steps_front, len(subs)))
    mean_vals_noise_switch_coupling[:] = np.nan
    err_vals_noise_switch_coupling = np.empty((len(shuffle_vals), steps_back+steps_front, len(subs)))
    err_vals_noise_switch_coupling[:] = np.nan
    mean_number_switchs_coupling = np.empty((len(shuffle_vals), len(subs)))
    mean_number_switchs_coupling[:] = np.nan
    mean_vals_noise_switch_coupling_bist_mono = np.empty((2, steps_back+steps_front, len(subs)))
    mean_vals_noise_switch_coupling_bist_mono[:] = np.nan
    err_vals_noise_switch_coupling_bist_mono = np.empty((2, steps_back+steps_front, len(subs)))
    err_vals_noise_switch_coupling_bist_mono[:] = np.nan
    zscor = scipy.stats.zscore
    all_kernels = []
    # all_noises = [[], [], []]
    for i_sub, subject in enumerate(subs):
        df_sub = df.loc[df.subject == subject]
        monostable_kernels = []
        bistable_kernels = []
        for i_sh, pshuffle in enumerate(shuffle_vals):
            df_coupling = df_sub.loc[df_sub.pShuffle == pshuffle]
            trial_index = df_coupling.trial_index.unique()
            # mean_vals_noise_switch_all_trials = np.empty((len(trial_index), steps_back+steps_front))
            mean_vals_noise_switch_all_trials = np.empty((1, steps_back+steps_front))
            mean_vals_noise_switch_all_trials[:] = np.nan
            number_switches = []
            idx_trials = np.where(pshuffles[i_sub] == pshuffle)[0]
            number_switches = []
            bistable_mask = []
            dominance_durations = []
            for i_trial, trial in enumerate(trial_index):
                is_bistable = (fitted_params_all[i_sub][0]*(1-pshuffle)+fitted_params_all[i_sub][1]) >= 1
                responses = choice[i_sub, idx_trials[i_trial]]
                chi = noise_signal[i_sub, idx_trials[i_trial]]
                # chi = chi-np.nanmean(chi)
                orders = rle(responses)  # not last one
                if avoid_first:
                    idx_1 = orders[1][1:][orders[2][1:] == 1]
                    idx_0 = orders[1][1:][orders[2][1:] == -1]
                else:
                    idx_1 = orders[1][orders[2] == 1]
                    idx_0 = orders[1][orders[2] == -1]
                number_switches.append(len(idx_1)+len(idx_0))
                dominance_durations.extend(orders[0])
                idx_1 = idx_1[(idx_1 > steps_back) & (idx_1 < (len(responses))-steps_front)]
                idx_0 = idx_0[(idx_0 > steps_back) & (idx_0 < (len(responses))-steps_front)]
                # original order
                mean_vals_noise_switch = np.empty((len(idx_1)+len(idx_0), steps_back+steps_front))
                mean_vals_noise_switch[:] = np.nan
                for i, idx in enumerate(idx_1):
                    mean_vals_noise_switch[i, :] = chi[idx - steps_back:idx+steps_front]
                    bistable_mask.append(is_bistable)
                for i, idx in enumerate(idx_0):
                    mean_vals_noise_switch[i+len(idx_1), :] =\
                        chi[idx - steps_back:idx+steps_front]*-1
                    bistable_mask.append(is_bistable)
                mean_vals_noise_switch_all_trials = np.row_stack((mean_vals_noise_switch_all_trials, mean_vals_noise_switch))
            mean_vals_noise_switch_all_trials = mean_vals_noise_switch_all_trials[1:]
            # it's better to compute afterwards, with the average peak per coupling
            # because trial by trial there is a lot of noise and that breaks the mean/latency
            # it gets averaged out
            mean_number_switchs_coupling[i_sh, i_sub] = np.nanmean(dominance_durations)/fps  # tFrame/ np.max([np.nanmean(np.array(number_switches)), 1])
            # axis=0 means average across switches (leaves time coords)
            # axis=1 means average across time (leaves switches coords)
            averaged_and_convolved_values = np.convolve(np.nanmean(mean_vals_noise_switch_all_trials, axis=0),
                                                                          np.ones(window_conv)/window_conv, 'same')
            mean_peak_latency[i_sh, i_sub] = (np.argmax(averaged_and_convolved_values) - steps_back)/fps
            mean_peak_amplitude[i_sh, i_sub] = np.nanmax(averaged_and_convolved_values)
            mean_vals_noise_switch_coupling[i_sh, :, i_sub] = averaged_and_convolved_values
            err_vals_noise_switch_coupling[i_sh, :, i_sub] = np.nanstd(mean_vals_noise_switch_all_trials, axis=0) / np.sqrt(mean_vals_noise_switch_all_trials.shape[0])

            bistable_mask = np.array(bistable_mask)
            bistable_kernels.append(mean_vals_noise_switch_all_trials[bistable_mask])
            monostable_kernels.append(mean_vals_noise_switch_all_trials[~bistable_mask])
        np.save(SV_FOLDER + f'/simulated_kernels_subjects/sim_kernel_{subject}.npy', mean_vals_noise_switch_coupling)
        np.save(SV_FOLDER + f'/simulated_kernels_subjects/sim_kernel_{subject}_error.npy', err_vals_noise_switch_coupling)
        kernel_bistable = np.nanmean(np.row_stack(bistable_kernels), axis=0)
        kernel_monostable = np.nanmean(np.row_stack(monostable_kernels), axis=0)
        all_together = np.row_stack((np.row_stack(monostable_kernels), np.row_stack(bistable_kernels)))
        all_kernels.append(np.nanmean(all_together, axis=0))
        mean_vals_noise_switch_coupling_bist_mono[0, :, i_sub] = kernel_bistable
        err_vals_noise_switch_coupling_bist_mono[0, :, i_sub] = np.nanstd(kernel_bistable, axis=0)
        mean_vals_noise_switch_coupling_bist_mono[1, :, i_sub] = kernel_monostable
        err_vals_noise_switch_coupling_bist_mono[1, :, i_sub] = np.nanstd(kernel_monostable, axis=0)

        if len(subs) > 1 and zscore_number_switches:
            label = 'z-scored '
        else:
            label = ''
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5., 4))
    fig3, ax34567 = plt.subplots(ncols=3, nrows=2, figsize=(12.5, 8))
    ax34567= ax34567.flatten()
    ax3, ax4, ax5, ax6, ax7, ax8 = ax34567

    def corrfunc(x, y, ax=None, **kws):
        """Plot the correlation coefficient in the top left hand corner of a plot."""
        r, _ = scipy.stats.pearsonr(x, y)
        ax = ax or plt.gca()
        ax.annotate(f'ρ = {r:.3f}', xy=(.1, 1.), xycoords=ax.transAxes)
    if hysteresis_area:
        hyst_width_2 = np.load(DATA_FOLDER + 'hysteresis_width_f2_sims_fitted_params.npy')
        hyst_width_4 = np.load(DATA_FOLDER + 'hysteresis_width_f4_sims_fitted_params.npy')
        datframe = pd.DataFrame({'Amplitude': mean_peak_amplitude.flatten(),
                                 'Latency': mean_peak_latency.flatten(),
                                 'Dominance': mean_number_switchs_coupling.flatten(),
                                 'Width f=2': hyst_width_2.flatten(),
                                 'Width f=4': hyst_width_4.flatten()})
    else:
        datframe = pd.DataFrame({'Amplitude': mean_peak_amplitude.flatten(),
                                 'Latency': mean_peak_latency.flatten(),
                                 'Dominance': mean_number_switchs_coupling.flatten()})
    np.save(DATA_FOLDER + 'simulated_mean_number_switches_per_subject.npy', mean_number_switchs_coupling)
    g = sns.pairplot(datframe)
    g.map_lower(corrfunc)
    for a in [ax, ax3, ax4, ax5, ax6, ax7, ax8]:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    latency = mean_peak_latency.copy()
    switches = mean_number_switchs_coupling.copy()
    amplitude = mean_peak_amplitude.copy()
    if normalize_variables:
        for var in [latency, switches, amplitude]:
            for i_s in range(len(subs)):
                var[:, i_s] = var[:, i_s] / np.abs(np.mean(var[:, i_s]))
    if zscore_number_switches:
        # zscore with respect to every p(shuffle), across subjects
        latency = zscor(latency, axis=0)
        switches = zscor(switches, axis=0)
        amplitude = zscor(amplitude, axis=0)
    for i_sub, subj in enumerate(subs):
        if len(subs) == 1:
            ax3.plot(shuffle_vals, switches, color='k', linewidth=4)
            ax4.plot(shuffle_vals, latency, color='k', linewidth=4)
            ax5.plot(shuffle_vals, amplitude, color='k', linewidth=4)
        else:
            ax3.plot(shuffle_vals, switches[:, i_sub], color='k', linewidth=2, alpha=0.3)
            ax4.plot(shuffle_vals, latency[:, i_sub], color='k', linewidth=2, alpha=0.3)
            ax5.plot(shuffle_vals, amplitude[:, i_sub], color='k', linewidth=2, alpha=0.3)
    corr = np.corrcoef(mean_peak_latency.flatten(), mean_number_switchs_coupling.flatten())[0][1]
    ax6.set_title(f'r = {corr :.3f}')
    for i_sh, sh in enumerate(shuffle_vals):
        ax6.plot(mean_peak_latency[i_sh, :], mean_number_switchs_coupling[i_sh, :],
                 color=colormap[i_sh], marker='o', linestyle='')
        ax7.plot(mean_peak_latency[i_sh, :], mean_peak_amplitude[i_sh, :],
                 color=colormap[i_sh], marker='o', linestyle='')
        ax8.plot(mean_number_switchs_coupling[i_sh, :], mean_peak_amplitude[i_sh, :],
                 color=colormap[i_sh], marker='o', linestyle='')
    for i_s in range(len(subs)):
        ax6.plot(mean_peak_latency[:, i_s], mean_number_switchs_coupling[:, i_s],
                  color='k', alpha=0.1)
        ax7.plot(mean_peak_latency[:, i_s], mean_peak_amplitude[:, i_s],
                  color='k', alpha=0.1)
        ax8.plot(mean_number_switchs_coupling[:, i_s], mean_peak_amplitude[:, i_s],
                  color='k', alpha=0.1)
    corr = np.corrcoef(mean_peak_latency.flatten(), mean_peak_amplitude.flatten())[0][1]
    ax7.set_title(f'r = {corr :.3f}')
    corr = np.corrcoef(mean_number_switchs_coupling.flatten(), mean_peak_amplitude.flatten())[0][1]
    ax8.set_title(f'r = {corr :.3f}')
    # ax3.set_yscale('')
    if len(subs) > 1:
        mean = np.nanmean(switches, axis=-1)
        err = np.nanstd(switches, axis=-1) / np.sqrt(len(subs))
        ax3.errorbar(shuffle_vals, mean, err, color='k', linewidth=4)
        mean = np.nanmean(latency, axis=-1)
        err = np.nanstd(latency, axis=-1) / np.sqrt(len(subs))
        ax4.errorbar(shuffle_vals, mean, err, color='k', linewidth=4)
        mean = np.nanmean(amplitude, axis=-1)
        err = np.nanstd(amplitude, axis=-1) / np.sqrt(len(subs))
        ax5.errorbar(shuffle_vals, mean, err, color='k', linewidth=4)
    ax3.set_xlabel('p(Shuffle)'); ax3.set_ylabel(label + 'Dominance duration'); ax4.set_xlabel('p(shuffle)'); ax4.set_ylabel(label + 'Peak latency')
    ax6.set_xlabel('Peak latency'); ax6.set_ylabel('Dominance duration'); ax7.set_xlabel('Peak latency'); ax7.set_ylabel('Peak amplitude')
    ax8.set_xlabel('Dominance duration'); ax8.set_ylabel('Peak amplitude'); ax5.set_xlabel('p(shuffle)'); ax5.set_ylabel(label + 'Peak amplitude')
    for i_sh, pshuffle in enumerate(shuffle_vals):
        x_plot = np.arange(-steps_back, steps_front, 1)/fps
        if len(subs) > 1:
            y_plot = np.nanmean(mean_vals_noise_switch_coupling[i_sh, :], axis=-1)
            err_plot = np.nanstd(mean_vals_noise_switch_coupling[i_sh, :], axis=-1) / np.sqrt(len(subs))
        else:
            y_plot = np.nanmean(mean_vals_noise_switch_coupling[i_sh, :], axis=-1)
            err_plot = err_vals_noise_switch_coupling[i_sh, :, 0]
        ax.plot(x_plot, y_plot, color=colormap[i_sh],
                label=pshuffle, linewidth=3)
        ax.fill_between(x_plot, y_plot-err_plot, y_plot+err_plot, color=colormap[i_sh],
                        alpha=0.3)
    ax.legend(title='p(shuffle)', frameon=False)
    ax.set_xlabel('Time from switch (s)')
    ax.set_ylabel('Noise')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'simulated_kernel_pshuffle.png', dpi=400)
    fig.savefig(SV_FOLDER + 'simulated_kernel_pshuffle.svg', dpi=400)
    fig3.tight_layout()
    colormap = ['peru', 'cadetblue']; labels=['Bistable', 'Monostable']
    fig, ax = plt.subplots(ncols=1, figsize=(5, 4))
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
    for regime in range(2):
        x_plot = np.arange(-steps_back, steps_front, 1)/fps
        if len(subs) > 1:
            y_plot = np.nanmean(mean_vals_noise_switch_coupling_bist_mono[regime, :], axis=-1)
            err_plot = np.nanstd(mean_vals_noise_switch_coupling_bist_mono[regime, :], axis=-1) / np.sqrt(len(subs))
        else:
            y_plot = mean_vals_noise_switch_coupling_bist_mono[regime, :, 0]
            err_plot = err_vals_noise_switch_coupling_bist_mono[regime, :, 0]
        ax.plot(x_plot, y_plot, color=colormap[regime],
                label=labels[regime], linewidth=3)
        ax.fill_between(x_plot, y_plot-err_plot, y_plot+err_plot, color=colormap[regime],
                        alpha=0.3)
    ax.legend(frameon=False); ax.set_xlabel('Time from switch(s)')
    ax.set_ylabel('Noise')
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'simulated_kernel_regime.png', dpi=400)
    fig.savefig(SV_FOLDER + 'simulated_kernel_regime.svg', dpi=400)
    
    fig, ax = plt.subplots(ncols=1, figsize=(5, 4))
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
    # for i_sub in range(len(subs)):
    #     y_plot = all_kernels[i_sub]
    #     ax.plot(x_plot, y_plot, color='k', linewidth=2, alpha=0.5)
    x_plot = np.arange(-steps_back, 0, 1)/fps
    y_plot = np.nanmean(all_kernels, axis=0)[:-steps_front]
    np.save(DATA_FOLDER + 'simulated_all_kernels_noise_switch_aligned.npy', all_kernels)
    ax.plot(x_plot, y_plot, color='k', linewidth=4, )
    err = np.nanstd(all_kernels, axis=0)[:-steps_front]/np.sqrt(len(subs))
    ax.fill_between(x_plot, y_plot-err, y_plot+err, color='k', alpha=0.2)
    ax.set_xlabel('Time from switch(s)')
    ax.set_ylabel('Noise')
    fig.tight_layout()


def plot_params_corrs():
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    fitted_subs = len(pars)
    b1s = [np.load(par)[2] for par in pars]
    sigmas = np.array([np.load(par)[3] for par in pars])
    thetas = [np.load(par)[4] for par in pars]
    j1s = np.array([np.load(par)[0] for par in pars])  # /sigmas
    j0s = np.array([np.load(par)[1] for par in pars])  # /sigmas
    dat = pd.DataFrame({'J0': j0s, 'J1': j1s,
                        'B1': b1s, 'Theta': thetas,
                        'Sigma': sigmas})
    g = sns.pairplot(dat)
    g.map_lower(corrfunc)


def plot_params_distros(ndt=False):
    label = 'ndt/' if ndt else ''
    npars = 6 if ndt else 5
    pars = glob.glob(SV_FOLDER + 'fitted_params/' + label + '*.npy')
    fitted_subs = len(pars)
    b1s = [np.load(par)[2] for par in pars]
    sigmas = np.array([np.load(par)[3] for par in pars])
    thetas = [np.load(par)[4] for par in pars]
    j1s = np.array([np.load(par)[0] for par in pars])  # /sigmas
    j0s = np.array([np.load(par)[1] for par in pars])  # /sigmas
    ndts = []
    labels = ['J1', 'J0', 'B1', 'Threshold', 'Sigma', 'NDT']
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(9, 6))
    ax = ax.flatten()
    if ndt:
        params_all = [j1s, j0s, b1s, thetas, sigmas, ndts]
        lims = [[-0.05, 0.5], [-0.05, 0.5], [0., 0.75], [0., 0.4], [0.05, 0.3], [0.1, 1.]]
    else:
        params_all = [j1s, j0s, b1s, thetas, sigmas]
        lims = [[-0.05, 0.5], [-0.05, 0.5], [0., 0.75], [0., 0.4], [0.05, 0.3]]
    for i, param in enumerate(params_all):
        ax[i].spines['right'].set_visible(False); ax[i].spines['top'].set_visible(False)
        sns.violinplot(x=param, inner=None, ax=ax[i], orient='horiz', fill=False,
                       linewidth=4)
        sns.swarmplot(x=param, ax=ax[i], orient='horiz', hue=np.arange(len(param)),
                      palette='Spectral', legend=False)
        ax[i].set_xlabel(labels[i])
        for k in range(2):
            ax[i].axvline(lims[i][k], color='r', alpha=0.4)
    if not ndt:
        ax[-1].axis('off')
    fig.tight_layout()
    print(np.sum(np.array(b1s) > 0.74))


def similarity_params(folder_2='ndt_fixed_15_tdur_good_comb'):
    pars_ndt = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    pars = glob.glob(SV_FOLDER + f'fitted_params/{folder_2}/' + '*.npy')[:len(pars_ndt)]
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = np.array([[params[0], params[1], params[2], params[4], params[3]] for params in fitted_params_all])
    fitted_params_all_ndt = [np.load(par) for par in pars_ndt]
    fitted_params_all_ndt = np.array([[params[0], params[1], params[2], params[4], params[3]] for params in fitted_params_all_ndt])
    labels = ['J1', 'J0', 'B1', 'Threshold', 'Sigma', 'NDT']
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(9, 6))
    ax = ax.flatten()
    fig2, ax2 = plt.subplots(ncols=3, nrows=2, figsize=(9, 6))
    ax2 = ax2.flatten()
    real_ndt = -np.load(DATA_FOLDER + 'kernel_latency_average.npy')
    ax[-1].spines['right'].set_visible(False); ax[-1].spines['top'].set_visible(False)
    for i, param in enumerate(fitted_params_all.T):
        ax[i].spines['right'].set_visible(False); ax[i].spines['top'].set_visible(False)
        ax2[i].spines['right'].set_visible(False); ax2[i].spines['top'].set_visible(False)
        ax[i].plot(fitted_params_all_ndt[:, i], param, color='k', linestyle='',
                   marker='o')
        sns.kdeplot(fitted_params_all_ndt[:, i], color='firebrick', ax=ax2[i], linewidth=4, label='With ndt')
        sns.kdeplot(param, color='midnightblue', ax=ax2[i], linewidth=4, label='Without ndt')
        pvalue = scipy.stats.mannwhitneyu(fitted_params_all_ndt[:, i], param).pvalue
        ax2[i].annotate(f'p = {pvalue: .2f}', xy=(.1, 1.1), xycoords=ax2[i].transAxes)
        mean_ndt = np.mean(fitted_params_all_ndt[:, i])
        mean_non_ndt = np.mean(param)
        median_ndt = np.median(fitted_params_all_ndt[:, i])
        median_non_ndt = np.median(param)
        ax[i].axvline(mean_ndt, color='k', alpha=0.3)
        ax[i].axhline(mean_non_ndt, color='k', alpha=0.3)
        ax2[i].axvline(median_ndt, color='firebrick', linewidth=2, linestyle='--')
        ax2[i].axvline(median_non_ndt, color='midnightblue', linewidth=2, linestyle='--')
        maxval = np.nanmax([fitted_params_all_ndt[:, i], param])
        minval = np.nanmin([fitted_params_all_ndt[:, i], param])
        ax[i].plot([minval, maxval], [minval, maxval],
                   color='k', linestyle='--', linewidth=3, alpha=0.3)
        ax[i].set_xlim(minval-5e-2, maxval+5e-2)
        ax[i].set_ylim(minval-5e-2, maxval+5e-2)
        ax[i].set_title(labels[i]);  ax2[i].set_xlabel(labels[i])
        ax[i].set_ylabel('Without NDT'); ax[i].set_xlabel('With NDT')
    difference_j0 =  fitted_params_all[:, 1]-fitted_params_all_ndt[:, 1]
    difference_ndt = real_ndt-np.median(np.abs(real_ndt))
    ax[-1].plot(difference_ndt, difference_j0, color='k',
                linestyle='', marker='o')
    r, p = pearsonr(difference_ndt, difference_j0)
    ax[-1].annotate(f'r = {r:.3f}\np={p:.0e}', xy=(.04, 0.8), xycoords=ax[-1].transAxes)
    ax[-1].axhline(0, color='k', linestyle='--', alpha=0.3); ax[-1].axvline(0 ,color='k', linestyle='--', alpha=0.3)
    ax[-1].set_xlabel('Subject NDT'); ax[-1].set_ylabel('J0(without) - J0(with)'); ax2[-1].axis('off');
    handles, labels = ax2[0].get_legend_handles_labels()
    ax2[-1].legend(handles, labels, frameon=False)
    fig.tight_layout();  fig2.tight_layout()


def get_likelihood(pars, n=4, data_folder=DATA_FOLDER, ntraining=8, nbins=27, t_dur=15, null=False):
    set_N_cpus(8)
    print(len(pars), ' fitted subjects')
    ndt = np.abs(np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
    fitted_params_all = [np.load(par) for par in pars]
    if not null:
        fitted_params_all = [[params[0], params[1], params[2], params[4], params[3], ndt] for params in fitted_params_all]
    if null:
        fitted_params_all = [[params[0], params[1], params[3], params[2], ndt] for params in fitted_params_all]
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()
    bins = np.linspace(0, 26, nbins).round(2)
    llhs = []
    bics = []
    for i_s, subject in enumerate(subjects):
        print('Fitting subject ', subject)
        if null:
            J0, B, THETA, SIGMA, NDT = fitted_params_all[i_s]
            model = null_model_known_params_pyddm(J0=J0, B=B, THETA=THETA, SIGMA=SIGMA, NDT=NDT, n=n, t_dur=t_dur)
        else:
            J1, J0, B, THETA, SIGMA, NDT = fitted_params_all[i_s]
            model = model_known_params_pyddm(J1=J1, J0=J0, B=B, THETA=THETA, SIGMA=SIGMA, NDT=NDT, n=n, t_dur=t_dur)
        
        df_sub = df.loc[(df.subject == subject) & (df.response > 0)]
        pshuffles = df_sub.pShuffle.values
        freqs = df_sub.freq.values*df_sub.initial_side.values
        phase_inis = df_sub.keypress_seconds_onset.values
        prev_choices = (df_sub.response.values-1)*2-1
        next_choice = -((df_sub.response.values-1)-1)
        rt = df_sub.keypress_seconds_offset.values-phase_inis
        rt_idx = (rt < t_dur)*(rt > 0.1)
        # not_last_change_idx = phase_inis < 25
        df_fit = pd.DataFrame({'prev_choice': prev_choices,
                               "freq": freqs, "phase_ini": bins[np.digitize(phase_inis-1e-5, bins)],
                               "pshuffle": pshuffles, "next_choice": next_choice,
                               "rt": rt})[rt_idx]
        sample_all = pyddm.Sample.from_pandas_dataframe(df_fit, rt_column_name="rt", choice_column_name="next_choice")
        print('Get likelihood')
        ll = get_model_loss(model, sample_all, lossfunction=LossLikelihood, method=None)
        print('Get BIC')
        bic = get_model_loss(model, sample_all, lossfunction=LossBIC, method=None)
        llhs.append(ll)
        bics.append(bic)
    return llhs, bics


def compare_likelihoods_models(load=True, loss='NLH',
                               data_folder=DATA_FOLDER,
                               ntraining=8,
                               nbins=105,
                               t_dur=20
                               ):
    if loss == 'BIC':
        df = load_data(data_folder, n_participants='all')
        df = df.loc[df.trial_index > ntraining]
        subjects = df.subject.unique()
        bins = np.linspace(0, 26, nbins).round(2)
        sample_sizes = []
        for i_s, subject in enumerate(subjects):
            df_sub = df.loc[(df.subject == subject) & (df.response > 0)]
            pshuffles = df_sub.pShuffle.values
            freqs = df_sub.freq.values*df_sub.initial_side.values
            phase_inis = df_sub.keypress_seconds_onset.values
            prev_choices = (df_sub.response.values-1)*2-1
            next_choice = -((df_sub.response.values-1)-1)
            rt = df_sub.keypress_seconds_offset.values-phase_inis
            rt_idx = (rt < t_dur)*(rt > 0.1)
            # not_last_change_idx = phase_inis < 25
            df_fit = pd.DataFrame({'prev_choice': prev_choices,
                                   "freq": freqs, "phase_ini": bins[np.digitize(phase_inis-1e-5, bins)],
                                   "pshuffle": pshuffles, "next_choice": next_choice,
                                   "rt": rt})[rt_idx]
            sample_sizes.append(len(df_fit))
        sample_sizes = np.array(sample_sizes)
    if not load:
        pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
        likelihood_with_ndt, bic_with_ndt = get_likelihood(pars, n=4, data_folder=DATA_FOLDER, ntraining=8, nbins=105, t_dur=20)
        np.save(SV_FOLDER + 'likelihood_model_with_ndt.npy', likelihood_with_ndt)
        np.save(SV_FOLDER + 'bic_model_with_ndt.npy', bic_with_ndt)
        pars2 = glob.glob(SV_FOLDER + 'fitted_params/null_model_params/' + '*.npy')
        likelihood_null_model, bic_null_model = get_likelihood(pars2, n=4, data_folder=DATA_FOLDER, ntraining=8, nbins=105, t_dur=20, null=True)
        np.save(SV_FOLDER + 'likelihood_null_model.npy', likelihood_null_model)
        np.save(SV_FOLDER + 'bic_null_model.npy', bic_null_model)
    if load:
        likelihood_null_model = np.array(np.load(SV_FOLDER + 'likelihood_null_model.npy'))
        bic_null_model = np.array(np.load(SV_FOLDER + 'bic_null_model.npy'))
        likelihood_with_ndt = np.array(np.load(SV_FOLDER + 'likelihood_model_with_ndt.npy'))
        bic_with_ndt = np.array(np.load(SV_FOLDER + 'bic_model_with_ndt.npy'))
    fig5, ax5 = plt.subplots(ncols=1, figsize=(3.5, 4))
    if loss == 'BIC':
        # loss_null_model = bic_null_model
        # loss_original_model = bic_with_ndt
        loss_null_model = 2*likelihood_null_model + 4*np.log(sample_sizes)
        loss_original_model = 2*likelihood_with_ndt + 5*np.log(sample_sizes)
        label = 'bic_difference'
    if loss == 'AIC':
        loss_null_model = 2*likelihood_null_model + 2*4
        loss_original_model = 2*likelihood_with_ndt + 2*5
        label = 'aic_difference'
    if loss == 'NLH':
        loss_null_model = likelihood_null_model
        loss_original_model = likelihood_with_ndt
        label = 'nllh_difference'
    losses = [loss_null_model, loss_original_model]
    ax5.spines['right'].set_visible(False); ax5.spines['top'].set_visible(False)
    sns.barplot(losses, palette=['mediumpurple', 'burlywood'], ax=ax5)
    pvalue = scipy.stats.ttest_rel(losses[0], losses[1]).pvalue
    print(pvalue)
    heights = [np.nanmean(losses[k]) for k in range(2)]
    barplot_annotate_brackets(0, 1, pvalue, [0, 1], heights, yerr=None, dh=.2, barh=.02, fs=10,
                              maxasterix=3, ax=ax5)
    sns.stripplot(losses, color='k', ax=ax5, size=3)
    ax5.set_xticks([0, 1], ['Null model', 'Model'])
    ylabel = loss
    ax5.set_ylabel(ylabel)
    fig5.tight_layout()
    fig5, ax5 = plt.subplots(ncols=1, figsize=(2., 4))
    losses = [loss_null_model - loss_original_model]
    ax5.spines['right'].set_visible(False); ax5.spines['top'].set_visible(False)
    sns.barplot(losses, palette=['yellowgreen'], ax=ax5)
    sns.stripplot(losses, color='k', ax=ax5, size=3)
    pvalue = scipy.stats.ttest_1samp(losses[0], 0).pvalue
    print(pvalue)
    ax5.axhline(0, color='k', linestyle='--', linewidth=2, alpha=0.4)
    if loss == 'BIC':
        ax5.text(0.8, 0.27, 'Better', rotation=90, fontsize=13, transform=ax5.transAxes)
        ax5.text(0.8, 0.04, 'Worse', rotation=90, fontsize=13, transform=ax5.transAxes)
    else:
        ax5.text(0.8, 0.35, 'Better', rotation=90, fontsize=13, transform=ax5.transAxes)
        ax5.text(0.8, 0.12, 'Worse', rotation=90, fontsize=13, transform=ax5.transAxes)
    ax5.set_xlim(-0.5, 0.8)
    ax5.set_xticks([0], ['Null - Original'])
    ax5.set_ylim(np.min(losses)-15, np.max(losses)+5)
    ylabel = r'$\Delta$' + loss
    ax5.set_ylabel(ylabel)
    fig5.tight_layout()
    fig5.savefig(SV_FOLDER + f'{label}.png', dpi=400, bbox_inches='tight')
    fig5.savefig(SV_FOLDER + f'{label}.svg', dpi=400, bbox_inches='tight')


def plot_density_regime_trials(n=4, unique_shuffle=[1., 0.7, 0.]):
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    print(len(pars), ' fitted subjects')
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = [[params[0], params[1], params[2], params[4], params[3]] for params in fitted_params_all]
    unique_shuffle = np.array(unique_shuffle)
    fitted_subs = len(pars)
    jeffs = np.zeros((3, fitted_subs))
    for i in range(fitted_subs):
        jeffs[:, i] = (fitted_params_all[i][0]*(1-unique_shuffle)+fitted_params_all[i][1])
    fig, ax = plt.subplots(1, figsize=(4, 3))
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    for i_sh, shuffle in enumerate(unique_shuffle):
        sns.kdeplot(jeffs[i_sh], color=colormap[i_sh], linewidth=5, label=shuffle,
                    bw_adjust=0.4, ax=ax, cut=0)
    ax.set_xlabel('Effective coupling');  ax.axvline(0.25, color='k', linestyle='--', alpha=0.3, linewidth=2)
    ax.legend(title='p(shuffle)', frameon=False); fig.tight_layout()


if __name__ == '__main__':
    print('Running model_fitting_analysis.py')
    # Example entry points (uncomment to run):
    # fit_data_pyddm(data_folder=DATA_FOLDER, ncpus=12, ntraining=8, t_dur=13,
    #                subj_ini=None, nbins=54, fitting_method='bads')
    # recovery_pyddm(n_pars=30, sv_folder=SV_FOLDER, n_cpus=11, i_ini=0)
    # compare_likelihoods_models(load=True, loss='BIC')
    # parameter_recovery_5_params(n_simuls_network=1, fps=60, tFrame=26,
    #                             n_pars_to_fit=100, n_sims_per_par=100,
    #                             sv_folder=SV_FOLDER, simulate=True)
