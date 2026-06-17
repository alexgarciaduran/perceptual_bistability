# -*- coding: utf-8 -*-
"""
Hysteresis analysis and plots.

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


def plot_responses_panels(responses_2, responses_4, barray_2, barray_4, coupling_levels,
                          tFrame=26, fps=60, window_conv=None,
                          ndt_list=np.arange(100)):
    """
    Make a 2-panel plot:
      - Left:  freq=2
      - Right: freq=4
    """
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(7.5, 4.))
    nFrame = tFrame*fps
    # common axis formatting
    # fig2, ax2 = plt.subplots(ncols=2, nrows=1, figsize=(6.5, 4.))
    # for a in ax2:
    #     a.spines['right'].set_visible(False)
    #     a.spines['top'].set_visible(False)
    #     a.set_xlabel('p(Shuffle)')
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylim(-0.025, 1.085)
        a.set_yticks([0, 0.5, 1])
        a.set_xlim(-2.05, 2.05)
        a.set_xticks([-2, 0, 2], [-1, 0, 1])
        a.axhline(0.5, color='k', linestyle='--', alpha=0.2)
        a.axvline(0., color='k', linestyle='--', alpha=0.2)
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.4, 0.27, 0.12, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.9, 0.27, 0.12, 0.2]
    ax4 = fig.add_axes([left, bottom, width, height])
    for a in ax2, ax4:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    hyst_width_2 = np.zeros((len(coupling_levels), len(responses_2[0])))
    hyst_width_2_difference_b = np.zeros((len(coupling_levels), len(responses_2[0])))
    nsubs = hyst_width_2.shape[1]
    if ndt_list is not None:
        # hysteresis, max_hists = \
        #     get_argmax_ndt_hyst_per_subject(responses_2, responses_4, barray_2, barray_4, coupling_levels,
        #                                     tFrame=tFrame, fps=fps, window_conv=window_conv,
        #                                     ndtlist=ndt_list)
        # hysteresis_across_coupling = np.mean(hysteresis, axis=0)
        # delay_per_subject = ndt_list[np.argmax(hysteresis_across_coupling, axis=1)]
        delay_per_subject = np.int32(60*np.load(DATA_FOLDER + 'kernel_latency_average.npy'))
    else:
        delay_per_subject = [0]*nsubs
    # --- FREQ = 2 ---
    switch_time_diff_2 = np.zeros((len(coupling_levels), len(responses_2[0])))
    switch_time_diff_4 = np.zeros((len(coupling_levels), len(responses_2[0])))
    for i_c, coupling in enumerate(coupling_levels):
        subj_means_asc, subj_means_desc = [], []
        for i_s, subj_resp in enumerate(responses_2[i_c]):
            resp_desc = subj_resp["desc"]
            resp_asc = subj_resp["asc"]
            responses_asc_desc = np.hstack((resp_asc, resp_desc))
            resp_rolled = np.roll(responses_asc_desc, delay_per_subject[i_s], axis=1)
            asc = np.nanmean(resp_rolled[:, :nFrame//2], axis=0)
            desc = np.nanmean(resp_rolled[:, nFrame//2:], axis=0)
            subj_means_asc.append(asc)
            subj_means_desc.append(desc)
            hyst_width_2[i_c, i_s] = np.nansum(np.abs(desc[::-1]-asc), axis=0) * np.diff(barray_2[:nFrame//2])[0]
            switch_time_diff_2[i_c, i_s] = np.nanmean(subj_resp['switches_diff'])
            
            # compute width - not area
            b_vals = barray_2[:nFrame//2]
            # Estimate B50 via sigmoid
            b_centers, binned_asc = bin_responses(b_vals, asc, bin_width=0.1)
            B50_asc = estimate_B50(b_centers, binned_asc)
            
            b_centers, binned_desc = bin_responses(b_vals, desc[::-1], bin_width=0.1)
            B50_desc = estimate_B50(b_centers, binned_desc)
            
            hyst_width_2_difference_b[i_c, i_s] = np.abs(B50_asc - B50_desc)
        if subj_means_asc:
            y_asc = np.nanmean(subj_means_asc, axis=0)
            y_desc = np.nanmean(subj_means_desc, axis=0)
            asc_mask = np.gradient(barray_2) > 0
            x_asc = barray_2[asc_mask]
            x_desc = barray_2[~asc_mask]
            # smoothing
            if window_conv is not None:
                y_asc = np.convolve(y_asc, np.ones(window_conv)/window_conv, mode='same')[window_conv//2:-window_conv//2]
                y_desc = np.convolve(y_desc, np.ones(window_conv)/window_conv, mode='same')[window_conv//2:-window_conv//2]
                x_asc = x_asc[window_conv//2:-window_conv//2]
                x_desc = x_desc[window_conv//2:-window_conv//2]
            ax[0].plot(x_asc, y_asc, color=colormap[i_c], linewidth=4,
                       label=f"{1-coupling:.1f}")
            ax[0].plot(x_desc, y_desc, color=colormap[i_c], linewidth=4)
    sns.barplot(hyst_width_2.T, palette=colormap, ax=ax2, errorbar='se')
    ax2.set_ylim(np.min(np.mean(hyst_width_2, axis=1))-0.25, np.max(np.mean(hyst_width_2, axis=1))+0.8)
    hyst_width_4 = np.zeros((len(coupling_levels), len(responses_2[0])))
    hyst_width_4_difference_b = np.zeros((len(coupling_levels), len(responses_2[0])))
    # --- FREQ = 4 ---
    for i_c, coupling in enumerate(coupling_levels):
        subj_means_asc, subj_means_desc = [], []
        for i_s, subj_resp in enumerate(responses_4[i_c]):
            resp_desc = subj_resp["desc"]
            resp_asc = subj_resp["asc"]
            responses_asc_desc = np.hstack((resp_asc, resp_desc))
            resp_rolled = np.roll(responses_asc_desc, delay_per_subject[i_s], axis=1)
            asc = np.nanmean(resp_rolled[:, :nFrame//4], axis=0)
            desc = np.nanmean(resp_rolled[:, nFrame//4:], axis=0)
            subj_means_asc.append(asc)
            subj_means_desc.append(desc)
            hyst_width_4[i_c, i_s] = np.nansum(np.abs(desc[::-1]-asc), axis=0) * np.diff(barray_4[:nFrame//2])[0]
            switch_time_diff_4[i_c, i_s] = np.nanmean(subj_resp['switches_diff'])
            
            # compute width - not area
            b_vals = barray_4[:nFrame//4]
            # Estimate B50 via sigmoid
            b_centers, binned_asc = bin_responses(b_vals, asc, bin_width=0.1)
            B50_asc = estimate_B50(b_centers, binned_asc)
            
            b_centers, binned_desc = bin_responses(b_vals, desc[::-1], bin_width=0.1)
            B50_desc = estimate_B50(b_centers, binned_desc)
            
            hyst_width_4_difference_b[i_c, i_s] = np.abs(B50_asc - B50_desc)
        if subj_means_asc:
            y_asc = np.nanmean(subj_means_asc, axis=0)
            y_desc = np.nanmean(subj_means_desc, axis=0)
            # only half the barray (one cycle)
            asc_mask = np.gradient(barray_4) > 0
            x_asc = barray_4[asc_mask][:nFrame//4]
            x_desc = barray_4[~asc_mask][:nFrame//4]
            # smoothing
            if window_conv is not None:
                y_asc = np.convolve(y_asc, np.ones(window_conv)/window_conv, mode='same')[window_conv//2:-window_conv//2]
                y_desc = np.convolve(y_desc, np.ones(window_conv)/window_conv, mode='same')[window_conv//2:-window_conv//2]
                x_asc = x_asc[window_conv//2:-window_conv//2]
                x_desc = x_desc[window_conv//2:-window_conv//2]
            ax[1].plot(x_asc, y_asc, color=colormap[i_c], linewidth=4)
            ax[1].plot(x_desc, y_desc, color=colormap[i_c], linewidth=4)
    sns.barplot(hyst_width_4.T, palette=colormap, ax=ax4, errorbar="se")
    ax4.set_ylim(np.min(np.mean(hyst_width_2, axis=1))-0.25, np.max(np.mean(hyst_width_2, axis=1))+0.8)
    heights = np.nanmean(hyst_width_2.T, axis=0)
    bars = np.arange(3)
    pv_sh012 = scipy.stats.ttest_rel(hyst_width_2[0], hyst_width_2[1]).pvalue
    pv_sh022 = scipy.stats.ttest_rel(hyst_width_2[0], hyst_width_2[2]).pvalue
    pv_sh122 = scipy.stats.ttest_rel(hyst_width_2[1], hyst_width_2[2]).pvalue
    pv_sh014 = scipy.stats.ttest_rel(hyst_width_4[0], hyst_width_4[1]).pvalue
    pv_sh024 = scipy.stats.ttest_rel(hyst_width_4[0], hyst_width_4[2]).pvalue
    pv_sh124 = scipy.stats.ttest_rel(hyst_width_4[1], hyst_width_4[2]).pvalue
    barplot_annotate_brackets(0, 1, pv_sh012, bars, heights, yerr=None, dh=.16, barh=.05, fs=10,
                              maxasterix=3, ax=ax2)
    barplot_annotate_brackets(0, 2, pv_sh022, bars, heights, yerr=None, dh=.39, barh=.05, fs=10,
                              maxasterix=3, ax=ax2)
    barplot_annotate_brackets(1, 2, pv_sh122, bars, heights, yerr=None, dh=.2, barh=.05, fs=10,
                              maxasterix=3, ax=ax2)
    heights = np.nanmean(hyst_width_4.T, axis=0)
    barplot_annotate_brackets(0, 1, pv_sh014, bars, heights, yerr=None, dh=.16, barh=.05, fs=10,
                              maxasterix=3, ax=ax4)
    barplot_annotate_brackets(0, 2, pv_sh024, bars, heights, yerr=None, dh=.35, barh=.05, fs=10,
                              maxasterix=3, ax=ax4)
    barplot_annotate_brackets(1, 2, pv_sh124, bars, heights, yerr=None, dh=.2, barh=.05, fs=10,
                              maxasterix=3, ax=ax4)
    for a in ax2, ax4:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_xlabel('p(shuffle)', fontsize=11); a.set_xticks([])
        a.set_ylabel('Hysteresis', fontsize=11); a.set_yticks([])
    # labels/titles
    np.save(DATA_FOLDER + 'hysteresis_width_freq_2.npy', hyst_width_2)
    np.save(DATA_FOLDER + 'hysteresis_width_freq_4.npy', hyst_width_4)
    np.save(DATA_FOLDER + 'difference_b_hysteresis_width_freq_2.npy', hyst_width_2_difference_b)
    np.save(DATA_FOLDER + 'difference_b_hysteresis_width_freq_4.npy', hyst_width_4_difference_b)
    np.save(DATA_FOLDER + 'switch_time_diff_freq_2.npy', switch_time_diff_2)
    np.save(DATA_FOLDER + 'switch_time_diff_freq_4.npy', switch_time_diff_4)
    # ax2[0].set_ylabel('Hysteresis width')
    ax[0].set_xlabel('Depth cue, c(t)')
    ax[1].set_xlabel('Depth cue, c(t)')
    ax[0].set_ylabel('P(rightward)')
    ax[0].legend(title='p(shuffle)', frameon=False,
                 bbox_to_anchor=[-0.02, 1.07], loc='upper left')
    ax[0].set_title('One cycle', fontsize=14)
    ax[1].set_title('Two cycles', fontsize=14)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'hysteresis_average.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'hysteresis_average.svg', dpi=400, bbox_inches='tight')
    # fig2.tight_layout()
    plt.show()
    fig3, ax3 = plt.subplots(1, figsize=(4, 3.5))
    ax3.plot([0, 3], [0, 3], color='k', alpha=0.4, linestyle='--', linewidth=4)
    for i_c in range(len(coupling_levels)):
        ax3.plot(hyst_width_2[i_c], hyst_width_4[i_c],
                  color=colormap[i_c], marker='o', linestyle='')
    # for i_s in range(nsubs):
    #     ax3.plot(hyst_width_2[:, i_s], hyst_width_4[:, i_s],
    #               color='k', alpha=0.1)
    
    fig4, ax4 = plt.subplots(1)
    for a in [ax3, ax4]:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
    ax4.plot([-1.5, 1.5], [-1.5, 1.5], color='k', alpha=0.4, linestyle='--', linewidth=4)
    ax4.axvline(0, color='r', alpha=0.4, linestyle='--', linewidth=2)
    ax4.axhline(0, color='r', alpha=0.4, linestyle='--', linewidth=2)
    ax4.plot(hyst_width_2[2]-hyst_width_2[0], hyst_width_4[2]-hyst_width_4[0],
              color='k', marker='o', linestyle='')
    ax4.set_ylabel('w(freq=4 | J_high) - w(freq=4 | J_low)')
    ax4.set_xlabel('w(freq=2 | J_high) - w(freq=2 | J_low)')
    ax3.set_ylabel('Hysteresis two cycles')
    ax3.set_xlabel('Hysteresis one cycle')
    fig4.tight_layout()
    fig3.tight_layout()
    fig3.savefig(SV_FOLDER + 'hysteresis_2_vs_4.png', dpi=400, bbox_inches='tight')
    fig3.savefig(SV_FOLDER + 'hysteresis_2_vs_4.svg', dpi=400, bbox_inches='tight')


def get_argmax_ndt_hyst_per_subject(responses_2, responses_4, barray_2, barray_4, coupling_levels,
                                    tFrame=26, fps=60, window_conv=None,
                                    ndtlist=np.arange(100)):
    nFrame = tFrame*fps
    hyst_widths = np.zeros((3, len(responses_2[0]), len(ndtlist)))
    # --- FREQ = 2 ---
    for i_c, coupling in enumerate(coupling_levels):
        for i_s, subj_resp in enumerate(responses_2[i_c]):
            for i_ndt, ndt in enumerate(ndtlist):
                resp_desc = subj_resp["desc"]
                resp_asc = subj_resp["asc"]
                responses_asc_desc = np.hstack((resp_asc, resp_desc))
                resp_rolled = np.roll(responses_asc_desc, ndt, axis=1)
                desc = np.nanmean(resp_rolled[:, :nFrame//2], axis=0)[::-1]
                asc = np.nanmean(resp_rolled[:, nFrame//2:], axis=0)
                hyst_widths[i_c, i_s, i_ndt] += np.nansum(desc-asc, axis=0) *\
                    np.diff(barray_2[:nFrame//2])[0]

    # --- FREQ = 4 ---
    for i_c, coupling in enumerate(coupling_levels):
        for i_s, subj_resp in enumerate(responses_4[i_c]):
            for i_ndt, ndt in enumerate(ndtlist):
                resp_desc = subj_resp["desc"]
                resp_asc = subj_resp["asc"]
                responses_asc_desc = np.hstack((resp_asc, resp_desc))
                resp_rolled = np.roll(responses_asc_desc, ndt, axis=1)
                desc = np.nanmean(resp_rolled[:, :nFrame//4], axis=0)[::-1]
                asc = np.nanmean(resp_rolled[:, nFrame//4:], axis=0)
                hyst_widths[i_c, i_s, i_ndt] += np.nansum(desc-asc, axis=0) *\
                    np.diff(barray_4[:nFrame//4])[0]
    ndts_max = np.argmax(hyst_widths/2, axis=2)
    return hyst_widths, ndts_max


def get_analytical_approximations_areas(b1=1, jeff=1, tFrame=26):
    temp = 1/(8*jeff)  # multiplied by parameter J1
    omega_2 = 2*np.pi/tFrame
    omega_4 = 4*np.pi/tFrame
    h_0 = 3*b1  # multiplied by parameter B1
    area_2 = 4*np.pi*(1/temp * np.exp(-2/temp))*h_0**2 * omega_2/(omega_2**2 + 1)
    area_4 = 4*np.pi*(1/temp * np.exp(-2/temp))*h_0**2 * omega_4/(omega_4**2 + 1)
    return area_2, area_4


def get_analytical_approximation_hysteresis_per_participant():
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    print(len(pars), ' fitted subjects')
    j1s = np.array([np.load(par)[0] for par in pars])
    j0s = np.array([np.load(par)[1] for par in pars])
    b1s = np.array([np.load(par)[2] for par in pars])
    shuffling_levels = np.array([1., 0.7, 0.])
    areas = {'f2': [], 'f4': []}
    for sub in range(len(j1s)):
        jeff = (1-shuffling_levels)*j1s[sub]+j0s[sub]+1e-10
        area_2, area_4 = get_analytical_approximations_areas(b1=b1s[sub],
                                                             jeff=jeff, tFrame=26)
        areas['f2'].append(area_2)
        areas['f4'].append(area_4)
    fig, ax = plt.subplots(ncols=2)
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    sns.barplot(np.vstack(areas['f2']), palette=colormap, ax=ax[0], errorbar="se")
    sns.barplot(np.vstack(areas['f4']), palette=colormap, ax=ax[1], errorbar="se")


def noise_bf_switch_threshold(load_sims=False, thres_vals=np.arange(0, 0.5, 1e-2),
                              j=0.4, nFrame=5000, fps=60, noisyframes=30,
                              ntrials=100):
    dt  = 1/fps
    tau = 10*dt
    label = 'Bistable' if j >= 1 else 'Monostable'
    if not load_sims:
        indep_noise = np.sqrt(dt/tau)*np.random.randn(nFrame, ntrials)*0.1
        x_arr = np.zeros((len(thres_vals), nFrame, ntrials))
        choice = np.zeros((len(thres_vals), nFrame, ntrials))
    
        time_interp = np.arange(0, nFrame+noisyframes, noisyframes)*dt
        time = np.arange(0, nFrame, 1)*dt
        noise_exp = np.random.randn(len(time_interp), ntrials)*0.2
        noise_signal = np.array([interp1d(time_interp, noise_exp[:, trial])(time) for trial in range(ntrials)]).T

        for i_j, th in enumerate(thres_vals):
            for trial in range(ntrials):
                x = np.random.rand()
                vec = [x]
                for t in range(1, nFrame):
                    x = x + dt*(sigmoid(2*j*(2*x-1)+2*noise_signal[t, trial])-x)/tau + indep_noise[t, trial]
                    vec.append(x)
                    if x < 0.5-th:
                        ch = -1.
                    if x >= 0.5+th:
                        ch = 1.
                    if 0.5-th <= x <= 0.5+th:
                        ch = choice[i_j, t-1, trial]
                    choice[i_j, t, trial] = ch
                x_arr[i_j, :, trial] = vec
        np.save(DATA_FOLDER + label +  '_noise_signal_experiment_threshold.npy', noise_signal)
        np.save(DATA_FOLDER + label + '_choice_noise_threshold.npy', choice)
        np.save(DATA_FOLDER + label + '_posterior_noise_threshold.npy', x_arr)
    else:
        noise_signal = np.load(DATA_FOLDER + label + '_noise_signal_experiment_threshold.npy')
        choice = np.load(DATA_FOLDER + label + '_choice_noise_threshold.npy')
        x_arr = np.load(DATA_FOLDER + label + '_posterior_noise_threshold.npy')
    return noise_signal, choice, x_arr


def noise_bf_switch_stim_weight(load_sims=False, stim_vals=np.arange(0, 2, 5e-2),
                                j=0.4, nFrame=5000, fps=60, noisyframes=30,
                                ntrials=100, th=0.1):
    dt  = 1/fps
    tau = 10*dt
    label = 'Bistable' if j >= 1 else 'Monostable'
    if not load_sims:
        indep_noise = np.sqrt(dt/tau)*np.random.randn(nFrame, ntrials)*0.1
        x_arr = np.zeros((len(stim_vals), nFrame, ntrials))
        choice = np.zeros((len(stim_vals), nFrame, ntrials))
    
        time_interp = np.arange(0, nFrame+noisyframes, noisyframes)*dt
        time = np.arange(0, nFrame, 1)*dt
        noise_exp = np.random.randn(len(time_interp), ntrials)*0.2
        noise_signal = np.array([interp1d(time_interp, noise_exp[:, trial])(time) for trial in range(ntrials)]).T

        for i_j, b1 in enumerate(stim_vals):
            for trial in range(ntrials):
                x = np.random.rand()
                vec = [x]
                for t in range(1, nFrame):
                    x = x + dt*(sigmoid(2*j*(2*x-1)+2*noise_signal[t, trial]*b1)-x)/tau + indep_noise[t, trial]
                    vec.append(x)
                    if x < 0.5-th:
                        ch = -1.
                    if x >= 0.5+th:
                        ch = 1.
                    if 0.5-th <= x <= 0.5+th:
                        ch = choice[i_j, t-1, trial]
                    choice[i_j, t, trial] = ch
                x_arr[i_j, :, trial] = vec
        np.save(DATA_FOLDER + label +  '_noise_signal_experiment_stim_weight.npy', noise_signal)
        np.save(DATA_FOLDER + label + '_choice_noise_stim_weight.npy', choice)
        np.save(DATA_FOLDER + label + '_posterior_noise_stim_weight.npy', x_arr)
    else:
        noise_signal = np.load(DATA_FOLDER + label + '_noise_signal_experiment_stim_weight.npy')
        choice = np.load(DATA_FOLDER + label + '_choice_noise_stim_weight.npy')
        x_arr = np.load(DATA_FOLDER + label + '_posterior_noise_stim_weight.npy')
    return noise_signal, choice, x_arr


def plot_noise_simulations_variable(load_sims=False, thres_vals=np.arange(0, 0.5, 1e-2),
                        variable='threshold',
                        j=0.4, nFrame=5000, fps=60, noisyframes=30,
                        n=4., steps_back=120, steps_front=20,
                        ntrials=100, zscore_number_switches=False, hysteresis_width=False):
    if variable == 'threshold':
        thres_vals = np.arange(0, 0.5, 1e-2)
        noise, choice, x_arr = noise_bf_switch_threshold(load_sims=load_sims,
                                                         thres_vals=thres_vals,
                                                         j=j, nFrame=nFrame, fps=fps,
                                                         noisyframes=noisyframes,
                                                         ntrials=ntrials)
    if variable == 'stim_weight':
        thres_vals = np.arange(0, 2, 5e-2)
        noise, choice, x_arr = noise_bf_switch_stim_weight(load_sims=load_sims,
                                                           stim_vals=thres_vals,
                                                           j=j, nFrame=nFrame, fps=fps,
                                                           noisyframes=noisyframes,
                                                           ntrials=ntrials)       
    # return x_arr, choice, noise_signal  
    mean_peak_latency = np.empty((len(thres_vals), ntrials))
    mean_peak_latency[:] = np.nan
    mean_peak_amplitude = np.empty((len(thres_vals), ntrials))
    mean_peak_amplitude[:] = np.nan
    mean_number_switchs_coupling = np.empty((len(thres_vals), ntrials))
    mean_number_switchs_coupling[:] = np.nan
    mean_vals_noise_switch_coupling = np.empty((len(thres_vals), steps_back+steps_front))
    mean_vals_noise_switch_coupling[:] = np.nan
    err_vals_noise_switch_coupling = np.empty((len(thres_vals), steps_back+steps_front))
    err_vals_noise_switch_coupling[:] = np.nan
    for i_j, threshold in enumerate(thres_vals):
        mean_vals_noise_switch_all_trials = np.empty((ntrials, steps_back+steps_front))
        mean_vals_noise_switch_all_trials[:] = np.nan
        number_switches = []
        latency = []
        height = []
        for trial in range(ntrials):
            responses = choice[i_j, :, trial]
            chi = noise[:, trial]
            # chi = chi-np.nanmean(chi)
            orders = rle(responses)
            idx_1 = orders[1][orders[2] == 1]
            idx_0 = orders[1][orders[2] == -1]
            idx_1 = idx_1[(idx_1 > steps_back) & (idx_1 < (len(responses))-steps_front)]
            idx_0 = idx_0[(idx_0 > steps_back) & (idx_0 < (len(responses))-steps_front)]
            number_switches.append(len(idx_1)+len(idx_0))
            # original order
            mean_vals_noise_switch = np.empty((len(idx_1)+len(idx_0), steps_back+steps_front))
            mean_vals_noise_switch[:] = np.nan
            for i, idx in enumerate(idx_1):
                mean_vals_noise_switch[i, :] = chi[idx - steps_back:idx+steps_front]
            for i, idx in enumerate(idx_0):
                mean_vals_noise_switch[i+len(idx_1), :] =\
                    chi[idx - steps_back:idx+steps_front]*-1
            mean_vals_noise_switch_all_trials[trial, :] = np.nanmean(mean_vals_noise_switch, axis=0)
            # if len(idx_0) == 0 and len(idx_1) == 0:
            #     continue
            # else:
            #     # take max values and latencies across time (axis=1) and then average across trials
            #     latency.append(np.nanmean(np.argmax(mean_vals_noise_switch, axis=1)))
            #     height.append(np.nanmean(np.nanmax(mean_vals_noise_switch, axis=1)))
        mean_number_switchs_coupling[i_j, :] = nFrame / np.array(number_switches)
        mean_peak_latency[i_j, :] = (np.nanmean(np.argmax(mean_vals_noise_switch_all_trials, axis=1)) - steps_back)/fps
        mean_peak_amplitude[i_j, :] = np.nanmean(np.nanmax(mean_vals_noise_switch_all_trials, axis=1))
        mean_vals_noise_switch_coupling[i_j, :] = np.nanmean(mean_vals_noise_switch_all_trials, axis=0)
        err_vals_noise_switch_coupling[i_j, :] = np.nanstd(mean_vals_noise_switch_all_trials, axis=0) / np.sqrt(ntrials)
    if variable == 'threshold':
        colormap = pl.cm.Oranges(np.linspace(0.3, 1, len(thres_vals)))
    if variable == 'stim_weight':
        colormap = pl.cm.Greens(np.linspace(0.3, 1, len(thres_vals)))
    fig, ax = plt.subplots(1, figsize=(5., 4))
    ax.set_xlabel('Time from switch (s)')
    ax.set_ylabel('Noise')
    fig3, ax34567 = plt.subplots(ncols=3, nrows=2, figsize=(12.5, 8))
    ax34567= ax34567.flatten()
    ax3, ax4, ax5, ax6, ax7, ax8 = ax34567
    mean_number_switchs_coupling = np.log(mean_number_switchs_coupling)
    if hysteresis_width and variable == 'threshold':
        hyst_width_2 = np.load(DATA_FOLDER + 'hysteresis_width_freq_2_simul_threshold.npy')
        hyst_width_4 = np.load(DATA_FOLDER + 'hysteresis_width_freq_4_simul_threshold.npy')
        datframe = pd.DataFrame({'Amplitude': np.nanmean(mean_peak_amplitude, axis=1).flatten(),
                                 'Latency': np.nanmean(mean_peak_latency, axis=1).flatten(),
                                 'Dominance': np.nanmean(mean_number_switchs_coupling, axis=1).flatten(),
                                 'Width f=2': hyst_width_2.flatten(),
                                 'Width f=4': hyst_width_4.flatten()})
    else:
        datframe = pd.DataFrame({'Amplitude': mean_peak_amplitude.flatten(),
                                 'Latency': mean_peak_latency.flatten(),
                                 'log Dominance': mean_number_switchs_coupling.flatten()})
    g = sns.pairplot(datframe)
    g.map_lower(corrfunc)
    for a in [ax, ax3, ax4, ax5, ax6, ax7, ax8]:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    mn, mx = [np.min(thres_vals), np.max(thres_vals)]
    ax5.plot([mn, mx], [mn, mx], color='gray', linewidth=3, linestyle='--', alpha=0.6)
    latency = mean_peak_latency.copy()
    switches = mean_number_switchs_coupling.copy()
    amplitude = mean_peak_amplitude.copy()
    if zscore_number_switches:
        # zscore with respect to every p(shuffle), across subjects
        latency = zscore(latency, axis=0)
        switches = zscore(switches, axis=0)
        amplitude = zscore(amplitude, axis=0)
        ax3.plot(thres_vals, switches, color='k', linewidth=4)
        ax4.plot(thres_vals, latency, color='k', linewidth=4)
        ax5.plot(thres_vals, amplitude, color='k', linewidth=4)
    corr = np.corrcoef(mean_peak_latency.flatten(), mean_number_switchs_coupling.flatten())[0][1]
    ax6.set_title(f'r = {corr :.3f}')
    for i_sh, th in enumerate(thres_vals):
        ax6.plot(mean_peak_latency[i_sh, :], mean_number_switchs_coupling[i_sh, :],
                 color=colormap[i_sh], marker='o', linestyle='')
        ax7.plot(mean_peak_latency[i_sh, :], mean_peak_amplitude[i_sh, :],
                 color=colormap[i_sh], marker='o', linestyle='')
        ax8.plot(mean_number_switchs_coupling[i_sh, :], mean_peak_amplitude[i_sh, :],
                 color=colormap[i_sh], marker='o', linestyle='')
    corr = np.corrcoef(mean_peak_latency.flatten(), mean_peak_amplitude.flatten())[0][1]
    ax7.set_title(f'r = {corr :.3f}')
    corr = np.corrcoef(mean_number_switchs_coupling.flatten(), mean_peak_amplitude.flatten())[0][1]
    ax8.set_title(f'r = {corr :.3f}')
    for n in range(ntrials):
        ax3.plot(thres_vals, switches[:, n], color='k', linewidth=1, alpha=0.3)
        ax4.plot(thres_vals, latency[:, n], color='k', linewidth=1, alpha=0.3)
        ax5.plot(thres_vals, amplitude[:, n], color='k', linewidth=1, alpha=0.3)
    # ax3.set_yscale('')
    mean = np.nanmean(switches, axis=-1)
    # err = np.nanstd(switches, axis=-1)
    ax3.plot(thres_vals, mean, color='k', linewidth=5)
    mean = np.nanmean(latency, axis=-1)
    # err = np.nanstd(latency, axis=-1)
    ax4.plot(thres_vals, mean, color='k', linewidth=5)
    mean = np.nanmean(amplitude, axis=-1)
    # err = np.nanstd(amplitude, axis=-1)
    ax5.plot(thres_vals, mean, color='k', linewidth=5)
    # ax5twin = ax5.twinx()
    # ax5twin.spines['top'].set_visible(False)
    # ax5twin.plot(shuffle_vals, np.exp(barriers), color='r', linewidth=4)
    label = ''
    ax3.set_xlabel(variable); ax3.set_ylabel(label + 'log Dominance duration'); ax4.set_xlabel(variable); ax4.set_ylabel(label + 'Peak latency')
    ax6.set_xlabel('Peak latency'); ax6.set_ylabel('log Dominance duration'); ax7.set_xlabel('Peak latency'); ax7.set_ylabel('Peak amplitude')
    ax8.set_xlabel('log Dominance duration'); ax8.set_ylabel('Peak amplitude'); ax5.set_xlabel(variable); ax5.set_ylabel(label + 'Peak amplitude')
    for i_sh, threshold in enumerate(thres_vals):
        x_plot = np.arange(-steps_back, steps_front, 1)/fps
        y_plot = mean_vals_noise_switch_coupling[i_sh, :]
        ax.plot(x_plot, y_plot, color=colormap[i_sh],
                label=threshold, linewidth=3)
    fig.tight_layout()
    fig3.tight_layout()


def hysteresis_simulation_threshold(j=1.2, thres_vals=np.arange(0, 0.5, 1e-2),
                                    n=4., tau=0.1, sigma=0.1, b1=0.15,
                                    tFrame=26, fps=60, nreps=500,
                                    simulate=False):
    nFrame = fps*tFrame
    n_th = len(thres_vals)
    label = 'Bistable' if j >= 1 else 'Monostable'
    if simulate:
        choice_all = np.zeros((2, nFrame, n_th, nreps))
        for freq, freqval in enumerate([2, 4]):
            for i_th, th in enumerate(thres_vals):
                for i in range(nreps):
                    params = [j, b1, tau, th, sigma]
                    # sign = (-1)**(i > nreps//2)
                    choice, x = simulator_5_params(params=params, freq=freqval, nFrame=nFrame,
                                                   fps=fps, return_choice=True)
                    choice_all[freq, :, i_th, i] = choice
        np.save(DATA_FOLDER + label + '_hysteresis_choices_changing_threshold.npy', choice_all)
    else:
        choice_all= np.load(DATA_FOLDER + label + '_hysteresis_choices_changing_threshold.npy')
    hyst_width_2 = np.zeros((n_th))
    hyst_width_4 = np.zeros((n_th))
    f2, ax2 = plt.subplots(ncols=2, figsize=(9, 4.5))
    colormap = pl.cm.Oranges(np.linspace(0.3, 1, 3))
    for freq, freqval in enumerate([2, 4]):
        color = 0
        for i_th, th in enumerate(thres_vals):
            stimulus = get_blist(freq=freqval, nFrame=nFrame)
            response_raw = choice_all[freq, :, i_th, :]
            response_raw[response_raw == 0] = np.nan
            response_raw = (response_raw+1)/2
            if freq > 0:
                choice_aligned = np.column_stack((response_raw[:nFrame//2],
                                                  response_raw[nFrame//2:]))
                response_raw = choice_aligned
                stimulus = stimulus[:nFrame//2]
            dx = np.diff(stimulus)[0]
            asc_mask = np.sign(np.gradient(stimulus)) > 0
            ascending = np.nanmean(response_raw, axis=1)[asc_mask]
            descending = np.nanmean(response_raw, axis=1)[~asc_mask]
            width = np.nansum(np.abs(descending[::-1] - ascending))*dx
            [hyst_width_2, hyst_width_4][freq][i_th] = width
            if i_th in [0, n_th//3, n_th*2//3]:
                ax2[freq].plot(stimulus[asc_mask], ascending, color=colormap[color], linewidth=4,
                               label=f'{thres_vals[i_th]}')
                ax2[freq].plot(stimulus[~asc_mask], descending, color=colormap[color], linewidth=4)
                color += 1
    np.save(DATA_FOLDER + 'hysteresis_width_freq_2_simul_threshold.npy', hyst_width_2)
    np.save(DATA_FOLDER + 'hysteresis_width_freq_4_simul_threshold.npy', hyst_width_4)
    for a in ax2:
        a.set_xlabel('Sensory evidence B(t)')
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    ax2[0].set_ylabel('P(choice = R)')
    ax2[0].legend(frameon=False, title=r'$\theta$')
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    colormap = pl.cm.Oranges(np.linspace(0.3, 1, n_th))
    ax.plot([0, 3], [0, 3], color='k', alpha=0.2, linestyle='--', linewidth=4)
    for i_c in range(n_th):
        ax.plot(hyst_width_2[i_c], hyst_width_4[i_c],
                color=colormap[i_c], marker='o', linestyle='')
    ax.set_ylabel('Width freq = 4')
    ax.set_xlabel('Width freq = 2')
    fig.tight_layout()
    f2.tight_layout()


def plot_switch_rate(tFrame=26, fps=60, data_folder=DATA_FOLDER,
                     ntraining=8, coupling_levels=[0, 0.3, 1],
                     window_conv=10, bin_size=0.1, switch_01=False):
    """
    Plots switch rate(t) across subjects, conditioning on coupling.
    Subject s_2 has a lot of switches.
    """
    only_ascending = switch_01
    nFrame = tFrame*fps
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()

    responses_2, responses_4, barray_2, barray_4 = collect_responses(
        df, subjects, coupling_levels, fps=fps, tFrame=tFrame)
    
    timebins = np.arange(0, tFrame+bin_size, bin_size)
    xvals = timebins[:-1] + bin_size/2

    fig, axes = plt.subplots(ncols=2, figsize=(7.5, 4.))
    titles = ['Freq = 2', 'Freq = 4']
    for i_ax, ax in enumerate(axes):
        ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
        ax.set_xlabel('Time (s)'); ax.axvline(tFrame/(2+2*i_ax), color='k', alpha=0.4,
                                              linestyle='--', linewidth=3)
        ax.axvline(tFrame/(4+4*i_ax), color='k', alpha=0.6, linestyle=':', linewidth=2)
        ax.axvline(3*tFrame/(4+4*i_ax), color='k', alpha=0.6, linestyle=':', linewidth=2)
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    for i_c, coupling in enumerate(coupling_levels):
    # pick one coupling level (e.g. i_c = 0) and ascending responses
        bins, mean012, sem01, mean102, sem10, per_sub_rates_01, per_sub_rates_10 =\
            average_switch_rates_dir(responses_2[i_c], fps=fps, bin_size=bin_size, join=True,
                                     only_ascending=only_ascending)
        bins, mean014, sem01, mean104, sem10, per_sub_rates_01, per_sub_rates_10 =\
            average_switch_rates_dir(responses_4[i_c], fps=fps, bin_size=bin_size/2, join=True,
                                     only_ascending=only_ascending)
        val_2 = mean012 if switch_01 else mean102
        val_4 = mean014 if switch_01 else mean104
        convolved_vals2 = np.convolve(val_2, np.ones(window_conv)/window_conv, "same")
        convolved_vals4 = np.convolve(val_4, np.ones(window_conv)/window_conv, "same")
        axes[0].plot(xvals , convolved_vals2, color=colormap[i_c], linewidth=3, label=f'{1-coupling}')
        axes[1].plot(xvals/2, convolved_vals4, color=colormap[i_c], linewidth=3)
    label_1 = 'R' if switch_01 else 'L'
    label_2 = 'L' if switch_01 else 'R'
    axes[0].legend(frameon=True, title='p(shuffle)'); axes[0].set_ylabel(fr'Switch rate {label_1}$\rightarrow${label_2}, (Hz)')
    fig.tight_layout()
    for ax in axes:
        pos_ax = ax.get_position()
        ax.set_position([pos_ax.x0, pos_ax.y0, pos_ax.width, pos_ax.height*0.75])
    pos_ax = axes[0].get_position()
    ax2 = fig.add_axes([pos_ax.x0, pos_ax.y0+pos_ax.height*1.03, pos_ax.width, pos_ax.height/4])
    pos_ax = axes[1].get_position()
    ax3 = fig.add_axes([pos_ax.x0, pos_ax.y0+pos_ax.height*1.03, pos_ax.width, pos_ax.height/4])
    for i_a, a in enumerate([ax2, ax3]):
        left, right = axes[0].get_xlim()
        a.set_xlim([left*fps, right*fps])   
        a.plot(barray_2*(-1)**(only_ascending), color='k', linewidth=4)
        a.axhline(0, color='k', linewidth=1)
        a.axvline((nFrame)/4, color='k', alpha=0.6, linestyle=':', linewidth=2)
        a.axvline(3*(nFrame)/4, color='k', alpha=0.6, linestyle=':', linewidth=2)
        a.axvline((nFrame)/2, color='k', alpha=0.4, linestyle='--', linewidth=3)
        a.set_xticks([]); a.set_yticks([-2, 0, 2], ['L', '0', 'R'])
        a.spines['right'].set_visible(False);  a.spines['top'].set_visible(False);
        a.spines['bottom'].set_visible(False)
        a.set_title(titles[i_a], fontsize=12)
    ax2.set_ylabel('Stim.(t)')
    fig.savefig(SV_FOLDER + 'switch_rate.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'switch_rate.svg', dpi=400, bbox_inches='tight')
    inc_switches = []
    fig, ax = plt.subplots(ncols=1, figsize=(5, 4))
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    for i_c, coupling in enumerate(coupling_levels):
    # pick one coupling level (e.g. i_c = 0) and ascending responses
        bins, mean012, sem01, mean102, sem10, per_sub_rates_01, per_sub_rates_10 =\
            average_switch_rates_dir(responses_2[i_c], fps=fps, bin_size=bin_size, join=True,
                                     only_ascending=only_ascending)
        idx_bins = (bins > 17.5)*(bins < 22.5)
        switches10 = np.nansum(per_sub_rates_10[:, idx_bins], axis=1)
        idx_bins = (bins > 3.5)*(bins < 8.5)
        switches01 = np.nansum(per_sub_rates_01[:, idx_bins], axis=1)
        sum_switches = switches01+switches10
        inc_switches.append(sum_switches)
    inc_switches = np.array(inc_switches)
    sns.barplot(inc_switches.T, palette=colormap, ax=ax, errorbar="se")
    sns.stripplot(inc_switches.T, color='k', ax=ax)
    ax.set_xticks([0, 1, 2], [1., 0.7, 0.])
    ax.set_xlabel('p(shuffle'); ax.set_ylabel('Incongruent switches')
    fig.tight_layout()


def average_switch_rates_dir(responses, fps=60, bin_size=1.0, join=True,
                             only_ascending=False):
    """
    Compute average 0→1 and 1→0 switch rates across subjects.
    If join=True, concatenates asc+desc before counting.
    """
    per_sub_rates_01, per_sub_rates_10 = [], []
    per_sub_counts_01, per_sub_counts_10 = [], []
    bins_ref = None

    for subj in responses:
        arr = join_trial_responses(subj, only_ascending=False) if join else subj["asc"]
        bins, r01, r10, counts_01, counts_10, _ = compute_switch_rate_from_array_dir(
            arr, fps=fps, bin_size=bin_size)
        if bins_ref is None:
            bins_ref = bins
        per_sub_rates_01.append(r01)
        per_sub_rates_10.append(r10)
        per_sub_counts_01.append(counts_01)
        per_sub_counts_10.append(counts_10)

    per_sub_rates_01 = np.vstack(per_sub_rates_01)
    per_sub_rates_10 = np.vstack(per_sub_rates_10)

    mean01 = np.nanmean(per_sub_rates_01, axis=0)
    sem01 = np.nanstd(per_sub_rates_01, axis=0, ddof=0) / np.sqrt(per_sub_rates_01.shape[0])
    mean10 = np.nanmean(per_sub_rates_10, axis=0)
    sem10 = np.nanstd(per_sub_rates_10, axis=0, ddof=0) / np.sqrt(per_sub_rates_10.shape[0])

    return bins_ref, mean01, sem01, mean10, sem10, np.vstack(per_sub_counts_01), np.vstack(per_sub_counts_10)


def compute_switch_rate_from_array_dir(response_array, fps=60, bin_size=1.0,
                                       min_interval=0.3):
    """
    Like compute_switch_rate_from_array but separates 0→1 and 1→0 switches.
    """
    n_trials, n_timepoints = response_array.shape
    duration = n_timepoints / fps
    bins = np.arange(0, duration + bin_size, bin_size)

    counts_01 = np.zeros(len(bins) - 1)
    counts_10 = np.zeros(len(bins) - 1)
    trial_bins_contrib = np.zeros(len(bins) - 1)

    for trial in response_array:
        idx_01, idx_10 = get_switch_indices_with_dir(trial)


        if idx_01.size > 0:
            times = idx_01 / fps
            h, _ = np.histogram(times, bins=bins)
            counts_01 += h
        if idx_10.size > 0:
            times = idx_10 / fps
            h, _ = np.histogram(times, bins=bins)
            counts_10 += h

        # contribution mask
        valid_frames = np.where(~np.isnan(trial))[0]
        if valid_frames.size > 0:
            valid_times = valid_frames / fps
            bin_idx = np.searchsorted(bins, valid_times, side="right") - 1
            unique_bins = np.unique(bin_idx[(bin_idx >= 0) & (bin_idx < len(trial_bins_contrib))])
            trial_bins_contrib[unique_bins] += 1

    denom = np.where(trial_bins_contrib > 0, trial_bins_contrib, np.nan)
    rate_01 = counts_01 / denom / bin_size
    rate_10 = counts_10 / denom / bin_size

    return bins[:-1], rate_01, rate_10, counts_01, counts_10, trial_bins_contrib


def plot_hysteresis_average(tFrame=26, fps=60, data_folder=DATA_FOLDER,
                            ntraining=8, coupling_levels=[0, 0.3, 1],
                            window_conv=None, ndt_list=np.arange(-50, 50)):
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()

    responses_2, responses_4, barray_2, barray_4 = collect_responses(
        df, subjects, coupling_levels, fps=fps, tFrame=tFrame)
    
    plot_responses_panels(responses_2, responses_4, barray_2, barray_4, coupling_levels,
                          tFrame=tFrame, fps=fps, window_conv=window_conv,
                          ndt_list=ndt_list)


def plot_hysteresis_by_regime(
    tFrame=26,
    fps=60,
    data_folder=DATA_FOLDER,
    sv_folder=SV_FOLDER,
    ntraining=8,
    coupling_levels=[0, 0.3, 1],
    window_conv=None,
    n=4,
    use_delay=False,
):
    """
    Like plot_hysteresis_average, but instead of averaging by coupling level,
    subjects are split into bistable / monostable groups based on their fitted
    J parameters, and the hysteresis curves are averaged within each group.
 
    Bistability condition (same as the rest of the pipeline):
        J_eff = (n·J1)·(1 − pshuffle) + (n·J0)  ≥  1
 
    Parameters
    ----------
    tFrame        : trial duration in seconds
    fps           : frames per second
    data_folder   : path to behavioural data
    sv_folder     : path to fitted_params/ndt/*.npy files
    ntraining     : number of training trials to discard
    coupling_levels : list of coupling values (1 − pshuffle)
    window_conv   : smoothing kernel width (None = no smoothing)
    n             : nonlinearity exponent used during fitting
    use_delay     : if True, load kernel_latency_average.npy to roll responses
                    (same logic as plot_responses_panels)
    """
    # ------------------------------------------------------------------
    # 1. Load behavioural data
    # ------------------------------------------------------------------
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()
 
    # ------------------------------------------------------------------
    # 2. Load fitted parameters  [n·J1, n·J0, B1, sigma, theta]
    #    Order on disk matches subjects order assumed by collect_responses.
    # ------------------------------------------------------------------
    pars_files = sorted(glob.glob(sv_folder + 'fitted_params/ndt/' + '*.npy'))
    fitted_params_all = []
    for f in pars_files:
        p = np.load(f)           # raw: [J1, J0, B1, theta, sigma, ...]
        fitted_params_all.append([
            n * p[0],   # n·J1   → index 0
            n * p[1],   # n·J0   → index 1
            p[2],       # B1     → index 2
            p[4],       # sigma  → index 3
            p[3],       # theta  → index 4
        ])
 
    n_fitted = len(fitted_params_all)
 
    # ------------------------------------------------------------------
    # 3. Collect responses (same helper as the original function)
    # ------------------------------------------------------------------
    responses_2, responses_4, barray_2, barray_4 = collect_responses(
        df, subjects, coupling_levels, fps=fps, tFrame=tFrame)
 
    nFrame = tFrame * fps
    nsubs = len(responses_2[0])
 
    # Per-subject delay (frames) used to roll the response array
    if use_delay:
        delay_per_subject = np.int32(fps * np.load(data_folder + 'kernel_latency_average.npy'))
    else:
        delay_per_subject = np.zeros(nsubs, dtype=int)
 
    # ------------------------------------------------------------------
    # 4. Stimulus x-axes
    # ------------------------------------------------------------------
    asc_mask_2 = np.gradient(barray_2) > 0
    x_asc_2    = barray_2[asc_mask_2]
    x_desc_2   = barray_2[~asc_mask_2]
 
    asc_mask_4 = np.gradient(barray_4) > 0
    x_asc_4    = barray_4[asc_mask_4][:nFrame // 4]
    x_desc_4   = barray_4[~asc_mask_4][:nFrame // 4]
 
    # ------------------------------------------------------------------
    # 5. Build regime-separated response lists, pooled across all couplings
    # ------------------------------------------------------------------
    # Each entry is one (subject × coupling) observation, a 1-D mean curve.
    regime_resp_2 = {0: [], 1: []}   # 0 = bistable, 1 = monostable
    regime_resp_4 = {0: [], 1: []}
 
    for i_c, coupling in enumerate(coupling_levels):
        pshuffle = 1.0 - coupling
 
        # ---- freq = 2 ----
        for i_s, subj_resp in enumerate(responses_2[i_c]):
            if i_s >= n_fitted:
                break
 
            j_eff = (fitted_params_all[i_s][0] * (1.0 - pshuffle)
                     + fitted_params_all[i_s][1])
            regime = 0 if j_eff >= 1.0 else 1
 
            resp_asc  = subj_resp["asc"]
            resp_desc = subj_resp["desc"]
            combined  = np.hstack((resp_asc, resp_desc))
            combined  = np.roll(combined, delay_per_subject[i_s], axis=1)
 
            asc  = np.nanmean(combined[:, :nFrame // 2], axis=0)
            desc = np.nanmean(combined[:, nFrame // 2:], axis=0)
 
            regime_resp_2[regime].append((asc, desc))
 
        # ---- freq = 4 ----
        for i_s, subj_resp in enumerate(responses_4[i_c]):
            if i_s >= n_fitted:
                break
 
            j_eff = (fitted_params_all[i_s][0] * (1.0 - pshuffle)
                     + fitted_params_all[i_s][1])
            regime = 0 if j_eff >= 1.0 else 1
 
            resp_asc  = subj_resp["asc"]
            resp_desc = subj_resp["desc"]
            combined  = np.hstack((resp_asc, resp_desc))
            combined  = np.roll(combined, delay_per_subject[i_s], axis=1)
 
            asc  = np.nanmean(combined[:, :nFrame // 4], axis=0)
            desc = np.nanmean(combined[:, nFrame // 4:], axis=0)
 
            regime_resp_4[regime].append((asc, desc))
 
    # ------------------------------------------------------------------
    # 6. Plot — one curve per regime, pooled across all coupling levels
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(7.5, 4.))
 
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylim(-0.025, 1.085)
        a.set_yticks([0, 0.5, 1])
        a.set_xlim(-2.05, 2.05)
        a.set_xticks([-2, 0, 2], [-1, 0, 1])
        a.axhline(0.5, color='k', linestyle='--', alpha=0.2)
        a.axvline(0.0, color='k', linestyle='--', alpha=0.2)
 
    regime_colors = {0: 'peru',   1: 'cadetblue'}
    regime_labels = {0: 'Bistable',  1: 'Monostable'}
 
    def _smooth(y, x, wc):
        if wc is not None and wc > 1:
            y = np.convolve(y, np.ones(wc) / wc, mode='same')[wc//2:-wc//2]
            x = x[wc//2:-wc//2]
        return y, x
 
    for regime in [0, 1]:
        color   = regime_colors[regime]
 
        # ---- freq = 2 ----
        data_2 = regime_resp_2[regime]
        if data_2:
            n_obs  = len(data_2)
            y_asc  = np.nanmean([d[0] for d in data_2], axis=0)
            y_desc = np.nanmean([d[1] for d in data_2], axis=0)
            ya, xa = _smooth(y_asc,  x_asc_2.copy(),  window_conv)
            yd, xd = _smooth(y_desc, x_desc_2.copy(), window_conv)
            ax[0].plot(xa, ya, color=color, linewidth=4,
                       label=f'{regime_labels[regime]}')
            ax[0].plot(xd, yd, color=color, linewidth=4)
 
        # ---- freq = 4 ----
        # data_4 = regime_resp_4[regime]
        # if data_4:
        #     y_asc  = np.nanmean([d[0] for d in data_4], axis=0)
        #     y_desc = np.nanmean([d[1] for d in data_4], axis=0)
        #     ya, xa = _smooth(y_asc,  x_asc_4.copy(),  window_conv)
        #     yd, xd = _smooth(y_desc, x_desc_4.copy(), window_conv)
        #     ax[1].plot(xa, ya, color=color, linewidth=4)
        #     ax[1].plot(xd, yd, color=color, linewidth=4)
 
    ax[0].set_xlabel('Depth cue, c(t)')
    ax[1].set_xlabel('Depth cue, c(t)')
    ax[0].set_ylabel('P(rightward)')
    ax[0].set_title('Freq = 2', fontsize=14)
    ax[1].set_title('Freq = 4', fontsize=14)
    ax[0].legend(frameon=False, fontsize=12, loc='upper left')
 
    fig.tight_layout()
    fig.savefig(sv_folder + 'hysteresis_by_regime_together.png', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder + 'hysteresis_by_regime_together.pdf', dpi=400, bbox_inches='tight')
    plt.show()
    return fig, ax


def plot_max_hyst_ndt_subject(tFrame=26, fps=60, data_folder=DATA_FOLDER,
                              ntraining=8, coupling_levels=[0, 0.3, 1],
                              window_conv=None, ndt_list=np.arange(100)):
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()

    responses_2, responses_4, barray_2, barray_4 = collect_responses(
        df, subjects, coupling_levels, fps=fps, tFrame=tFrame)
    
    hysteresis, max_hists = \
        get_argmax_ndt_hyst_per_subject(responses_2, responses_4, barray_2, barray_4, coupling_levels,
                                        tFrame=tFrame, fps=fps, window_conv=window_conv,
                                        ndtlist=ndt_list)
    hysteresis_across_coupling = np.mean(hysteresis, axis=0)
    f = plt.figure()
    for i in range(hysteresis_across_coupling.shape[0]):
        plt.plot(ndt_list/fps, hysteresis_across_coupling[i], color='k',
                 alpha=0.3)
    hysteresis_across_coupling_subjects = np.mean(hysteresis_across_coupling, axis=0)
    plt.plot(ndt_list/fps, hysteresis_across_coupling_subjects, color='k', linewidth=3)
    plt.ylabel('Hysteresis area'); plt.xlabel('Delay (s)'); f.tight_layout()


def hysteresis_basic_plot(coupling_levels=[0, 0.3, 1],
                          fps=60, tFrame=26, data_folder=DATA_FOLDER,
                          nbins=13, ntraining=4, arrows=False, subjects=['s_1'],
                          window_conv=None):
    nFrame = fps*tFrame
    window_conv = 1 if window_conv is None else window_conv
    df = load_data(data_folder, n_participants='all', filter_subjects=False)
    df = df.loc[df.trial_index > ntraining]
    # resp = df.prev_response.values
    # resp[np.isnan(resp)] = 0
    # df['responses'] = resp
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(7.5, 4.))
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylim(-0.025, 1.085)
        a.set_yticks([0, 0.5, 1])
        a.set_xlim(-2.05, 2.05)
        a.set_xticks([-2, 0, 2], [-1, 0, 1])
        a.axhline(0.5, color='k', linestyle='--', alpha=0.2)
        a.axvline(0., color='k', linestyle='--', alpha=0.2)
    colormap = pl.cm.Oranges(np.linspace(0.3, 1, 3))[::-1]
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    for i_s, subject in enumerate(subjects):
        df_sub = df.loc[df.subject == subject]
        for i_c, coupling in enumerate(coupling_levels):
            df_filt = df_sub.loc[df_sub.pShuffle.round(2) == round(1-coupling, 2)]
            df_freq_2 = df_filt.loc[df_filt.freq == 2]
            response_array_asc, response_array_desc, barray_2, _, _ = get_response_and_blist_array(df_freq_2, fps=fps,
                                                                                                   tFrame=tFrame)
            # response_array_2 = np.roll(response_array_2, -50, axis=1)
            # mean_response_2 = np.nanmean(response_array_2, axis=0)
            x_valsasc = barray_2[:nFrame//2]
            x_valsdesc = barray_2[nFrame//2:]
            y_vals2asc = np.nanmean(response_array_asc, axis=0)
            y_vals2desc = np.nanmean(response_array_desc, axis=0)
            if window_conv is not None:
                y_vals2asc = np.convolve(y_vals2asc, np.ones(window_conv)/window_conv, mode='same')[window_conv//2:-window_conv//2]
                y_vals2desc = np.convolve(y_vals2desc, np.ones(window_conv)/window_conv, mode='same')[window_conv//2:-window_conv//2]
                x_valsasc = x_valsasc[window_conv//2:-window_conv//2]
                x_valsdesc = x_valsdesc[window_conv//2:-window_conv//2]
            ax[0].plot(x_valsasc, y_vals2asc, color=colormap[i_c], linewidth=4,
                       linestyle='--' if arrows else 'solid', label=1-coupling)
            ax[0].plot(x_valsdesc, y_vals2desc, color=colormap[i_c], linewidth=4,
                       linestyle='--' if arrows else 'solid')
            df_freq_4 = df_filt.loc[df_filt.freq == 4]
            response_array_4_asc, response_array_4_desc, barray_4, _, _ = get_response_and_blist_array(df_freq_4, fps=fps,
                                                                                                       tFrame=tFrame)
            asc_mask = np.gradient(barray_4) > 0
            x_vals4_asc = barray_4[asc_mask][:nFrame//4]
            x_vals4_desc = barray_4[~asc_mask][:nFrame//4]
            y_vals4_asc = np.nanmean(response_array_4_asc, axis=0)
            y_vals4_desc = np.nanmean(response_array_4_desc, axis=0)
            if window_conv is not None:
                y_vals4_asc = np.convolve(y_vals4_asc, np.ones(window_conv)/window_conv, mode='same')[window_conv//2:-window_conv//2]
                y_vals4_desc = np.convolve(y_vals4_desc, np.ones(window_conv)/window_conv, mode='same')[window_conv//2:-window_conv//2]
                x_vals4_asc = x_vals4_asc[window_conv//2:-window_conv//2]
                x_vals4_desc = x_vals4_desc[window_conv//2:-window_conv//2]
            ax[1].plot(x_vals4_asc, y_vals4_asc, color=colormap[i_c], linewidth=4,
                       linestyle='--' if arrows else 'solid')
            ax[1].plot(x_vals4_desc, y_vals4_desc, color=colormap[i_c], linewidth=4,
                       linestyle='--' if arrows else 'solid')
            # if coupling == 1 and arrows:
            #     ax[0].annotate(text='', xy=(-hist_val_2+0.25, 1.06),
            #                    xytext=(hist_val_2-0.25, 1.06),
            #                    arrowprops=dict(arrowstyle='<->', color=colormap[i_c],
            #                                    linewidth=3))
            #     ax[1].annotate(text='', xy=(-hist_val_4+0.6, 1.06),
            #                    xytext=(hist_val_4-0.5, 1.06),
            #                    arrowprops=dict(arrowstyle='<->', color=colormap[i_c],
            #                                    linewidth=3))
            #     ax[0].axhline(0.5, color=colormap[i_c], alpha=0.7, linestyle='--')
            #     ax[1].axhline(0.5, color=colormap[i_c], alpha=0.7, linestyle='--')
            #     ax[0].plot([-hist_val_2+0.25, -hist_val_2+0.25],
            #                [0.5, 1.06], color=colormap[i_c], alpha=0.7, linestyle='--')
            #     ax[0].plot([hist_val_2-0.25]*2,
            #                [0.5, 1.06], color=colormap[i_c], alpha=0.7, linestyle='--')
            #     ax[1].plot([-hist_val_4+0.62]*2,
            #                [0.5, 1.06], color=colormap[i_c], alpha=0.7, linestyle='--')
            #     ax[1].plot([hist_val_4-0.53]*2,
            #                [0.5, 1.06], color=colormap[i_c], alpha=0.7, linestyle='--')
    ax[0].set_xlabel('Depth cue, c(t)')
    ax[1].set_xlabel('Depth cue, c(t)')
    ax[0].set_ylabel('Proportion of rightward responses')
    ax[0].legend(title='p(shuffle)', frameon=False)
    ax[0].set_title('Freq = 2', fontsize=14)
    ax[1].set_title('Freq = 4', fontsize=14)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'hysteresis_basic_plot.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'hysteresis_basic_plot.pdf', dpi=400, bbox_inches='tight')


def hysteresis_basic_plot_all_subjects(coupling_levels=[0, 0.3, 1],
                                      fps=60, tFrame=26, data_folder=DATA_FOLDER,
                                      nbins=13, ntraining=4, arrows=False):
    nFrame = fps*tFrame
    df = load_data(data_folder, n_participants='all', filter_subjects=True)
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()
    fig, ax = plt.subplots(ncols=6, nrows=int(np.ceil(len(subjects)/3)), figsize=(22, 23))
    ax = ax.flatten()
    fig2, ax2 = plt.subplots(ncols=4, nrows=int(np.ceil(len(subjects)/4)), figsize=(18, 20))
    ax2 = ax2.flatten()
    fig.tight_layout()
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylim(-0.025, 1.085)
        a.set_yticks([0, 0.5, 1])
        a.set_xlim(-2.05, 2.05)
        a.set_xticks([-2, 0, 2], [-1, 0, 1])
        a.axhline(0.5, color='k', linestyle='--', alpha=0.2)
        a.axvline(0., color='k', linestyle='--', alpha=0.2)
    for a in ax2:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    colormap = pl.cm.Oranges(np.linspace(0.3, 1, 3))[::-1]
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    for i_s, subject in enumerate(subjects):
        df_sub = df.loc[df.subject == subject]
        for i_c, coupling in enumerate(coupling_levels):
            df_filt = df_sub.loc[df_sub.pShuffle.round(2) == round(1-coupling, 2)]
            df_freq_2 = df_filt.loc[df_filt.freq == 2]
            response_array_asc, response_array_desc, barray_2, _, _ = get_response_and_blist_array(df_freq_2, fps=fps,
                                                                                                tFrame=tFrame)
            # response_array_2 = np.roll(response_array_2, -50, axis=1)
            # mean_response_2 = np.nanmean(response_array_2, axis=0)
            x_valsasc = barray_2[:nFrame//2]
            x_valsdesc = barray_2[nFrame//2:]
            r2asc = np.nanmean(response_array_asc, axis=0)
            r2desc = np.nanmean(response_array_desc, axis=0)
            # y_vals2asc = np.convolve(r2asc, np.ones(50)/50, mode='same')
            # y_vals2desc = np.convolve(r2desc, np.ones(50)/50, mode='same')
            ax[i_s*2].plot(x_valsasc, r2asc, color=colormap[i_c], linewidth=4,
                           linestyle='--' if arrows else 'solid', label=1-coupling)
            ax[i_s*2].plot(x_valsdesc, r2desc, color=colormap[i_c], linewidth=4,
                           linestyle='--' if arrows else 'solid')
            df_freq_4 = df_filt.loc[df_filt.freq == 4]
            response_array_4_asc, response_array_4_desc, barray_4, _, _ = get_response_and_blist_array(df_freq_4, fps=fps,
                                                                                                    tFrame=tFrame)
            asc_mask = np.gradient(barray_4) > 0
            x_vals4_asc = barray_4[asc_mask][:nFrame//4]
            x_vals4_desc = barray_4[~asc_mask][:nFrame//4]
            y_vals4_asc = np.nanmean(response_array_4_asc, axis=0)
            y_vals4_desc = np.nanmean(response_array_4_desc, axis=0)
            # y_vals4_asc_conv = np.convolve(y_vals4_asc, np.ones(1)/1, mode='same')
            # y_vals4_desc_conv = np.convolve(y_vals4_desc, np.ones(1)/1, mode='same')
            ax[i_s*2+1].plot(x_vals4_asc, y_vals4_asc, color=colormap[i_c], linewidth=4,
                             linestyle='--' if arrows else 'solid')
            ax[i_s*2+1].plot(x_vals4_desc, y_vals4_desc, color=colormap[i_c], linewidth=4,
                             linestyle='--' if arrows else 'solid')
        f, f2, f3, g, fnew = plot_noise_before_switch(data_folder=DATA_FOLDER, fps=60, tFrame=26,
                                                      steps_back=150, steps_front=20,
                                                      shuffle_vals=[1, 0.7, 0], violin=False, sub=subject,
                                                      avoid_first=False, window_conv=1, zscore_number_switches=False, ax=ax2[i_s],
                                                      fig=fig, filter_subjects=False, legend_axes=True if i_s==36 else False)
        plt.close(f2); plt.close(f3); plt.close(g.fig); plt.close(fnew)
        ax2[i_s].set_xticks([]);  ax2[i_s].set_yticks([])
        ax2[i_s].axvline(0, color='k', alpha=0.3, linestyle='--')
        ax2[i_s].set_ylim(-0.4, 0.7)
        if i_s < 32:
            ax[i_s].set_xticks([-1, 0, 1], ['', '', ''])
        if (i_s+1) % 6 != 0:
            ax[i_s].set_yticks([])
    ax2[-1].set_ylim(-0.4, 0.7)
    ax2[-1].set_ylabel('Noise')
    ax2[-1].set_xlabel('Time from switch (s)')
    ax[i_s*2].set_xlabel('Depth cue, c(t)')
    ax[i_s*2+1].set_xlabel('Depth cue, c(t)')
    ax[i_s*2-1].set_xlabel('Depth cue, c(t)')
    ax[i_s*2-2].set_xlabel('Depth cue, c(t)')
    ax[0].set_ylabel('Proportion of rightward responses')
    ax[0].legend(title='p(shuffle)', frameon=False)
    ax[0].set_title('Freq = 2', fontsize=14)
    ax[1].set_title('Freq = 4', fontsize=14)
    ax[2].set_title('Freq = 2', fontsize=14)
    ax[3].set_title('Freq = 4', fontsize=14)
    ax[4].set_title('Freq = 2', fontsize=14)
    ax[5].set_title('Freq = 4', fontsize=14)
    fig2.tight_layout()
    fig.savefig(SV_FOLDER + 'hysteresis_basic_plot_all.png', dpi=200, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'hysteresis_basic_plot_all.pdf', dpi=200, bbox_inches='tight')
    fig2.savefig(SV_FOLDER + 'noise_kernel_all.png', dpi=200, bbox_inches='tight')
    fig2.savefig(SV_FOLDER + 'nosie_kernel_all.pdf', dpi=200, bbox_inches='tight')


def hysteresis_basic_plot_simulation(coup_vals=np.array((0., 0.3, 1))*0.27+0.02,
                                     fps=60, nsubs=20,
                                     n=4, nsims=100, b_list=np.linspace(-0.5, 0.5, 501),
                                     simul=False):
    b_list_2 = np.concatenate((b_list[:-1], b_list[1:][::-1]))
    b_list_4 = np.concatenate((b_list[:-1][::2], b_list[1:][::-2], b_list[:-1][::2], b_list[1:][::-2]))
    nFrame = len(b_list_2)
    dt  = 1/fps
    time = np.arange(0, nFrame, 1)*dt
    tau = 0.01
    if simul:
        indep_noise = np.sqrt(dt/tau)*np.random.randn(nFrame, nsims, nsubs)*0.07
        choice = np.zeros((len(coup_vals), nFrame, nsims, 2, nsubs))
        for i_j, j in enumerate(coup_vals):
            for freq in range(2):
                stimulus = [b_list_2, b_list_4][freq]
                for sub in range(nsubs):
                    for sim in range(nsims):
                        x = np.random.rand()*0.5  # assume we start close to q ~ 0 (always L --> R)
                        for i in range(2):
                            x = sigmoid(2*j*n*(2*x-1)+2*stimulus[0]*0.3)
                        vec = [x]
                        for t in range(1, nFrame):
                            x = x + dt*(sigmoid(2*j*n*(2*x-1)+2*stimulus[t]*0.3)-x)/tau + indep_noise[t, sim, sub]
                            vec.append(x)
                            if x < 0.49999:
                                ch = 0.
                            if x > 0.50999:
                                ch = 1.
                            if 0.49999 <= x <= 0.50999 and t > 0:
                                ch = choice[i_j, t-1, sim, freq, sub]
                            choice[i_j, t, sim, freq, sub] = ch
        np.save(DATA_FOLDER + 'choice_hysteresis_large_tau.npy', choice)
    else:
        choice = np.load(DATA_FOLDER + 'choice_hysteresis_large_tau.npy')
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(9.5, 4))
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylim(-0.025, 1.08)
        a.set_yticks([0, 0.5, 1])
        a.set_xticks([-0.5, 0, 0.5], ['L', '0', 'R' ])
        # a.tick_params(axis='x', rotation=45)
    colormap = ['cadetblue', 'peru']
    lsts = ['solid', 'solid']
    labels = ['Monostable', 'Bistable']
    for i_c, coupling in enumerate(coup_vals):
        for freq in range(2):
            stimulus = [b_list_2, b_list_4][freq]
            response_raw = choice[i_c, :, :, freq, 0]
            if freq > 0:
                choice_aligned = np.column_stack((response_raw[:nFrame//2],
                                                  response_raw[nFrame//2:]))
                response_raw = choice_aligned
                stimulus = stimulus[:nFrame//2]
            response = np.nanmean(response_raw, axis=1)
            
            response = np.convolve(response, np.ones(1)/1, 'same')
            ax[freq].plot(stimulus, response, color=colormap[i_c], linewidth=4,
                          label=labels[i_c], linestyle=lsts[freq])
        if i_c == 2:
            ax[0].axhline(0.5, color=colormap[i_c], alpha=0.7, linestyle='--')
            ax[1].axhline(0.5, color=colormap[i_c], alpha=0.7, linestyle='--')
    ax[0].set_xlabel('Depth cue, c(t)')
    ax[1].set_yticks([0, 0.5, 1], ['', '', ''])
    ax[1].set_xlabel('Depth cue, c(t)')
    ax[0].set_ylabel('P(rightward)')
    ax[0].legend(frameon=False)
    ax[0].set_title('Freq = 2', fontsize=14)
    ax[1].set_title('Freq = 4', fontsize=14)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'hysteresis_basic_plot_simulation_v2.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'hysteresis_basic_plot_simulation_v2.pdf', dpi=400, bbox_inches='tight')


def plot_hysteresis_width_simluations(coup_vals=np.arange(0.05, 0.35, 1e-2),
                                      b_list=np.linspace(-0.5, 0.5, 501),
                                      window_conv=10):
    b_list_2 = np.concatenate((b_list[:-1], b_list[1:][::-1]))
    b_list_4 = np.concatenate((b_list[:-1][::2], b_list[1:][::-2], b_list[:-1][::2], b_list[1:][::-2]))
    # choice is a len(coup_vals) x timepoints x nsims x freqs(2)
    choice = np.load(DATA_FOLDER + 'choice_hysteresis_large_tau.npy')[:, ..., 0]
    # choice = np.load(DATA_FOLDER + 'choice_hysteresis.npy')
    n_coup, nFrame, nsims, nfreqs = choice.shape
    hyst_width_2 = np.zeros((n_coup))
    hyst_width_4 = np.zeros((n_coup))
    f0, ax0 = plt.subplots(ncols=2, figsize=(7.5, 4.))
    colormap = pl.cm.Blues(np.linspace(0.3, 1, 3))
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    left, bottom, width, height = [0.4, 0.27, 0.12, 0.2]
    ax2 = f0.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.9, 0.27, 0.12, 0.2]
    ax4 = f0.add_axes([left, bottom, width, height])
    for a in ax2, ax4:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    h2 = []; h4 = []
    for freq in range(2):
        color = 0
        for i_c, coupling in enumerate(coup_vals):
            stimulus = [b_list_2, b_list_4][freq]
            response_raw = choice[i_c, :, :, freq]
            if freq > 0:
                choice_aligned = np.column_stack((response_raw[:nFrame//2],
                                                  response_raw[nFrame//2:]))
                response_raw = choice_aligned
                stimulus = stimulus[:nFrame//2]
            dx = np.diff(stimulus)[0]
            asc_mask = np.sign(np.gradient(stimulus)) > 0
            ascending = np.nanmean(response_raw, axis=1)[asc_mask]
            descending = np.nanmean(response_raw, axis=1)[~asc_mask]
            width = np.nansum(np.abs(descending[::-1] - ascending))*dx
            [hyst_width_2, hyst_width_4][freq][i_c] = width
            descending = np.convolve(descending, np.ones(window_conv)/window_conv, "same")
            ascending = np.convolve(ascending, np.ones(window_conv)/window_conv, "same")
            if i_c in [0, 1, 2]:
                ax0[freq].plot(stimulus[asc_mask], ascending, color=colormap[color], linewidth=4,
                               label=f'{round(coup_vals[i_c], 1)}')
                ax0[freq].plot(stimulus[~asc_mask], descending, color=colormap[color], linewidth=4)
                color += 1
                [h2, h4][freq].append(width)
    sns.barplot(h2, palette=colormap, ax=ax2, errorbar="se")
    ax2.set_ylim(np.min(np.mean(h2))-0.25, np.max(np.mean(h2))+0.5)
    sns.barplot(h4, palette=colormap, ax=ax4, errorbar="se")
    ax4.set_ylim(np.min(np.mean(h4))-0.25, np.max(np.mean(h4))+0.2)
    for a in ax2, ax4:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_xlabel('Coupling, J', fontsize=11); a.set_xticks([])
        a.set_ylabel('Hysteresis', fontsize=11); a.set_yticks([])
    for a in ax0:
        a.set_xlabel('Sensory evidence B(t)'); a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
        a.axhline(0.5, color='k', linestyle='--', alpha=0.2)
        a.axvline(0., color='k', linestyle='--', alpha=0.2)
        a.set_xlim(-0.51, 0.51)
        a.set_xticks([-0.5, 0, 0.5], [-1, 0, 1])
    ax0[0].set_ylabel('P(rightward)');  ax0[0].legend(frameon=False, title='Coupling, J', loc='upper left'); ax0[0].set_yticks([0, 0.5, 1])
    f0.tight_layout()
    f0.savefig(SV_FOLDER + 'hysteresis_basic_plot_simulation_v3.png', dpi=400, bbox_inches='tight')
    f0.savefig(SV_FOLDER + 'hysteresis_basic_plot_simulation_v3.pdf', dpi=400, bbox_inches='tight')
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    colormap = pl.cm.Blues(np.linspace(0.3, 1, len(coup_vals)))
    ax.plot([0, 0.6], [0, 0.6], color='k', alpha=0.2, linestyle='--', linewidth=4)
    for i_c in range(len(coup_vals)):
        ax.plot(hyst_width_2[i_c], hyst_width_4[i_c],
                color=colormap[i_c], marker='o', linestyle='')
    ax.set_ylabel('Width freq = 4')
    ax.set_xlabel('Width freq = 2')
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    colormap = pl.cm.Blues(np.linspace(0.3, 1, len(coup_vals)))
    ax.plot([-0.05, 0.18], [-0.05, 0.18], color='k', alpha=0.2, linestyle='--', linewidth=4)
    ax.axvline(0, color='r', alpha=0.4, linestyle='--', linewidth=2)
    ax.axhline(0, color='r', alpha=0.4, linestyle='--', linewidth=2)
    for i_c in range(len(coup_vals)-5):
        ax.plot(hyst_width_2[i_c+5]-hyst_width_2[i_c], hyst_width_4[i_c+5]-hyst_width_4[i_c],
                color='k', marker='o', linestyle='')
    ax.set_ylabel('w(freq=4 | J_high) - w(freq=4 | J_low)')
    ax.set_xlabel('w(freq=2 | J_high) - w(freq=2 | J_low)')
    fig.tight_layout()
    np.save(DATA_FOLDER + 'hysteresis_width_freq_2_simul.npy', hyst_width_2)
    np.save(DATA_FOLDER + 'hysteresis_width_freq_4_simul.npy', hyst_width_4)


def noise_bf_switch_coupling(load_sims=False, coup_vals=np.arange(0.05, 0.35, 1e-2),
                             nFrame=5000, fps=60, noisyframes=30,
                             n=4., steps_back=120, steps_front=20,
                             ntrials=100, zscore_number_switches=False, hysteresis_width=False,
                             th=0.1):
    dt  = 1/fps
    tau = 10*dt
    shuffle_vals = 1-(coup_vals-np.min(coup_vals))/(np.max(coup_vals)-np.min(coup_vals))
    val_05 = 0.5
    barriers = []
    if th == 0.1:
        label = ''
    else:
        label = str(th)
    for i_j, j in enumerate(coup_vals):
        fun_to_minimize = lambda q: sigmoid(2*n*j*(2*q-1))-q
        val = fsolve(fun_to_minimize, 1)
        barrier = -potential_mf(val, j, n=4) + potential_mf(val_05, j, n=4)
        barriers.append(barrier)
    if not load_sims:
        indep_noise = np.sqrt(dt/tau)*np.random.randn(nFrame, ntrials)*0.1
        x_arr = np.zeros((len(coup_vals), nFrame, ntrials))
        choice = np.zeros((len(coup_vals), nFrame, ntrials))
    
        time_interp = np.arange(0, nFrame+noisyframes, noisyframes)*dt
        time = np.arange(0, nFrame, 1)*dt
        noise_exp = np.random.randn(len(time_interp), ntrials)*0.2
        noise_signal = np.array([interp1d(time_interp, noise_exp[:, trial])(time) for trial in range(ntrials)]).T
    
        
        for i_j, j in enumerate(coup_vals):
            for trial in range(ntrials):
                x = np.random.rand()
                vec = [x]
                for t in range(1, nFrame):
                    x = x + dt*(sigmoid(2*j*n*(2*x-1)+2*noise_signal[t, trial])-x)/tau + indep_noise[t, trial]
                    vec.append(x)
                    if x < 0.5-th:
                        ch = -1.
                    if x >= 0.5+th:
                        ch = 1.
                    if 0.5-th <= x <= 0.5+th:
                        ch = choice[i_j, t-1, trial]
                    choice[i_j, t, trial] = ch
                x_arr[i_j, :, trial] = vec
        np.save(DATA_FOLDER + f'noise_signal_experiment{label}.npy', noise_signal)
        np.save(DATA_FOLDER + f'choice_noise{label}.npy', choice)
        np.save(DATA_FOLDER + f'posterior_noise{label}.npy', x_arr)
    else:
        noise_signal = np.load(DATA_FOLDER + f'noise_signal_experiment{label}.npy')
        choice = np.load(DATA_FOLDER + f'choice_noise{label}.npy')
        x_arr = np.load(DATA_FOLDER + f'posterior_noise{label}.npy')
    # return x_arr, choice, noise_signal  
    mean_peak_latency = np.empty((len(coup_vals), ntrials))
    mean_peak_latency[:] = np.nan
    mean_peak_amplitude = np.empty((len(coup_vals), ntrials))
    mean_peak_amplitude[:] = np.nan
    mean_number_switchs_coupling = np.empty((len(coup_vals), ntrials))
    mean_number_switchs_coupling[:] = np.nan
    mean_vals_noise_switch_coupling = np.empty((len(coup_vals), steps_back+steps_front))
    mean_vals_noise_switch_coupling[:] = np.nan
    err_vals_noise_switch_coupling = np.empty((len(coup_vals), steps_back+steps_front))
    err_vals_noise_switch_coupling[:] = np.nan
    for i_j, coupling in enumerate(coup_vals):
        mean_vals_noise_switch_all_trials = np.empty((ntrials, steps_back+steps_front))
        mean_vals_noise_switch_all_trials[:] = np.nan
        number_switches = []
        latency = []
        height = []
        for trial in range(ntrials):
            responses = choice[i_j, :, trial]
            chi = noise_signal[:, trial]
            # chi = chi-np.nanmean(chi)
            orders = rle(responses)
            idx_1 = orders[1][orders[2] == 1]
            idx_0 = orders[1][orders[2] == -1]
            idx_1 = idx_1[(idx_1 > steps_back) & (idx_1 < (len(responses))-steps_front)]
            idx_0 = idx_0[(idx_0 > steps_back) & (idx_0 < (len(responses))-steps_front)]
            number_switches.append(len(idx_1)+len(idx_0))
            # original order
            mean_vals_noise_switch = np.empty((len(idx_1)+len(idx_0), steps_back+steps_front))
            mean_vals_noise_switch[:] = np.nan
            for i, idx in enumerate(idx_1):
                mean_vals_noise_switch[i, :] = chi[idx - steps_back:idx+steps_front]
            for i, idx in enumerate(idx_0):
                mean_vals_noise_switch[i+len(idx_1), :] =\
                    chi[idx - steps_back:idx+steps_front]*-1
            mean_vals_noise_switch_all_trials[trial, :] = np.nanmean(mean_vals_noise_switch, axis=0)
            # if len(idx_0) == 0 and len(idx_1) == 0:
            #     continue
            # else:
            #     # take max values and latencies across time (axis=1) and then average across trials
            #     latency.append(np.nanmean(np.argmax(mean_vals_noise_switch, axis=1)))
            #     height.append(np.nanmean(np.nanmax(mean_vals_noise_switch, axis=1)))
        mean_number_switchs_coupling[i_j, :] = np.log(nFrame / (np.array(number_switches)+1e-12))
        mean_peak_latency[i_j, :] = (np.nanmean(np.argmax(mean_vals_noise_switch_all_trials, axis=1)) - steps_back)/fps
        mean_peak_amplitude[i_j, :] = np.nanmean(np.nanmax(mean_vals_noise_switch_all_trials, axis=1))
        mean_vals_noise_switch_coupling[i_j, :] = np.nanmean(mean_vals_noise_switch_all_trials, axis=0)
        err_vals_noise_switch_coupling[i_j, :] = np.nanstd(mean_vals_noise_switch_all_trials, axis=0) / np.sqrt(ntrials)
    colormap = pl.cm.Blues(np.linspace(0.3, 1, len(coup_vals)))
    fig, ax = plt.subplots(1, figsize=(5., 4))
    ax.set_xlabel('Time from switch (s)')
    ax.set_ylabel('Noise')
    fig3, ax34567 = plt.subplots(ncols=3, nrows=2, figsize=(12.5, 8))
    ax34567= ax34567.flatten()
    ax3, ax4, ax5, ax6, ax7, ax8 = ax34567


    if hysteresis_width:
        hyst_width_2 = np.load(DATA_FOLDER + 'hysteresis_width_freq_2_simul.npy')
        hyst_width_4 = np.load(DATA_FOLDER + 'hysteresis_width_freq_4_simul.npy')
        potential_barrier = []
        for j in coup_vals:
            fun_to_minimize = lambda q: sigmoid(2*n*j*(2*q-1))-q
            val = fsolve(fun_to_minimize, 1)[0]
            barrier = -potential_mf(val, j, n=4) + potential_mf(0.5, j, n=4)
            potential_barrier.append(barrier)
        datframe = pd.DataFrame({'Amplitude': np.nanmean(mean_peak_amplitude, axis=1).flatten(),
                                 'Latency': np.nanmean(mean_peak_latency, axis=1).flatten(),
                                 'Dominance': np.nanmean(mean_number_switchs_coupling, axis=1).flatten(),
                                 'Width f=2': hyst_width_2.flatten(),
                                 'Width f=4': hyst_width_4.flatten(),
                                 r'$\Delta V$': potential_barrier})
    else:
        datframe = pd.DataFrame({'Amplitude': mean_peak_amplitude.flatten(),
                                 'Latency': mean_peak_latency.flatten(),
                                 'Dominance': mean_number_switchs_coupling.flatten()})
    g = sns.pairplot(datframe)
    g.map_lower(corrfunc)
    for a in [ax, ax3, ax4, ax5, ax6, ax7, ax8]:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    latency = mean_peak_latency.copy()
    switches = mean_number_switchs_coupling.copy()
    amplitude = mean_peak_amplitude.copy()
    zscor = scipy.stats.zscore
    if zscore_number_switches:
        # zscore with respect to every p(shuffle), across subjects
        latency = zscor(latency, axis=0)
        switches = zscor(switches, axis=0)
        amplitude = zscor(amplitude, axis=0)
        ax3.plot(shuffle_vals, switches, color='k', linewidth=4)
        ax4.plot(shuffle_vals, latency, color='k', linewidth=4)
        ax5.plot(shuffle_vals, amplitude, color='k', linewidth=4)
    corr = np.corrcoef(mean_peak_latency.flatten(), mean_number_switchs_coupling.flatten())[0][1]
    ax6.set_title(f'r = {corr :.3f}')
    for i_sh, sh in enumerate(shuffle_vals):
        ax6.plot(mean_peak_latency[i_sh, :], mean_number_switchs_coupling[i_sh, :],
                 color=colormap[i_sh], marker='o', linestyle='')
        ax7.plot(mean_peak_latency[i_sh, :], mean_peak_amplitude[i_sh, :],
                 color=colormap[i_sh], marker='o', linestyle='')
        ax8.plot(mean_number_switchs_coupling[i_sh, :], mean_peak_amplitude[i_sh, :],
                 color=colormap[i_sh], marker='o', linestyle='')
    corr = np.corrcoef(mean_peak_latency.flatten(), mean_peak_amplitude.flatten())[0][1]
    ax7.set_title(f'r = {corr :.3f}')
    corr = np.corrcoef(mean_number_switchs_coupling.flatten(), mean_peak_amplitude.flatten())[0][1]
    ax8.set_title(f'r = {corr :.3f}')
    for n in range(ntrials):
        ax3.plot(shuffle_vals, switches[:, n], color='k', linewidth=1, alpha=0.3)
        ax4.plot(shuffle_vals, latency[:, n], color='k', linewidth=1, alpha=0.3)
        ax5.plot(shuffle_vals, amplitude[:, n], color='k', linewidth=1, alpha=0.3)
    # ax3.set_yscale('')
    mean = np.nanmean(switches, axis=-1)
    # err = np.nanstd(switches, axis=-1)
    ax3.plot(shuffle_vals, mean, color='k', linewidth=5)
    mean = np.nanmean(latency, axis=-1)
    # err = np.nanstd(latency, axis=-1)
    ax4.plot(shuffle_vals, mean, color='k', linewidth=5)
    mean = np.nanmean(amplitude, axis=-1)
    # err = np.nanstd(amplitude, axis=-1)
    ax5.plot(shuffle_vals, mean, color='k', linewidth=5)
    # ax5twin = ax5.twinx()
    # ax5twin.spines['top'].set_visible(False)
    # ax5twin.plot(shuffle_vals, np.exp(barriers), color='r', linewidth=4)
    label = ''
    ax3.set_xlabel('p(Shuffle)'); ax3.set_ylabel(label + 'Dominance duration'); ax4.set_xlabel('p(shuffle)'); ax4.set_ylabel(label + 'Peak latency')
    ax6.set_xlabel('Peak latency'); ax6.set_ylabel('Dominance duration'); ax7.set_xlabel('Peak latency'); ax7.set_ylabel('Peak amplitude')
    ax8.set_xlabel('Dominance duration'); ax8.set_ylabel('Peak amplitude'); ax5.set_xlabel('p(shuffle)'); ax5.set_ylabel(label + 'Peak amplitude')
    for i_sh, pshuffle in enumerate(shuffle_vals):
        x_plot = np.arange(-steps_back, steps_front, 1)/fps
        y_plot = mean_vals_noise_switch_coupling[i_sh, :]
        ax.plot(x_plot, y_plot, color=colormap[i_sh],
                label=pshuffle, linewidth=3)
    fig.tight_layout()
    fig3.tight_layout()


def compute_incongruent_switches(df, nFrame=1560, dt=1/60):
    """
    Compute incongruent perceptual switches per subject and trial.

    Parameters
    ----------
    data : pandas.DataFrame or structured array
        Must contain columns:
        ['subject', 'trial_index', 'response', 'freq', 'initial_side', 'pShuffle']
        response: 1D np.ndarray or list of 1/2 responses per frame.
    nFrame : int
        Number of frames in each trial.

    Returns
    -------
    results : list of dict
        One dict per trial with incongruent switch information.
    """
    subjects = df['subject'].unique()
    results = []
    mean_per_sub_coupling = np.zeros((3, 35))
    for i_sub, subj in enumerate(subjects):
        df_subj = df.loc[df['subject'] == subj]
        # df_subj = df_subj.loc[df_subj.freq == 2]
        trial_index = df_subj.trial_index.unique()
        for trial in trial_index:
            df_trial = df_subj.loc[df_subj['trial_index'] == trial]
            df_trial = df_trial.loc[df_trial.keypress_seconds_onset < 26]

            # single-trial info
            freq = df_trial.freq.values[0]
            side = df_trial.initial_side.values[0]
            pShuffle = df_trial.pShuffle.values[0]
            
            # reconstruct stimulus trajectory
            stim = get_blist(freq * side, nFrame)
            time_array = np.arange(nFrame) * dt  # approximate time vector

            # compute switches between consecutive dominance periods
            n_switches = len(df_trial) - 1
            n_incongruent = 0

            for i in range(n_switches):
                prev_r = df_trial['response'].iloc[i]
                new_r = df_trial['response'].iloc[i+1]

                # stimulus change over the interval
                t_start = df_trial['keypress_seconds_onset'].iloc[i+1]
                t_end = df_trial['keypress_seconds_offset'].iloc[i+1]
                idx_start = int(np.searchsorted(time_array, t_start))
                idx_end = int(np.searchsorted(time_array, t_end))
                delta_stim = stim[idx_end-1] - stim[idx_start]

                # detect incongruency
                if prev_r == 1 and new_r == 2:
                    congruent = delta_stim > 0
                elif prev_r == 2 and new_r == 1:
                    congruent = delta_stim < 0
                else:
                    continue

                if not congruent:
                    n_incongruent += 1

            frac_incongruent = n_incongruent / n_switches if n_switches > 0 else np.nan

            results.append({
                'subject': subj,
                'trial_index': trial,
                'freq': freq,
                'initial_side': side,
                'pShuffle': pShuffle,
                'n_switches': n_switches,
                'n_incongruent': n_incongruent,
                'frac_incongruent': frac_incongruent
            })
        resdf = pd.DataFrame(results).loc[pd.DataFrame(results).subject == subj]
        pshs = resdf.groupby('pShuffle')['frac_incongruent'].mean().index.values
        inc_switchrate = resdf.groupby('pShuffle')['frac_incongruent'].mean().values
        for i_psh, pshuff in enumerate([1., 0.7, 0.]):
            idx_psh = pshs == pshuff
            mean_per_sub_coupling[idx_psh, i_sub] = inc_switchrate[idx_psh]

    return pd.DataFrame(results)


def get_correlation_consecutive_dominance_durations(data_folder=DATA_FOLDER, fps=60, tFrame=26):
    df = load_data(data_folder + '/noisy/', n_participants='all', filter_subjects=True)
    subs = df.subject.unique()
    responses_all = np.load(SV_FOLDER + 'responses_simulated_noise.npy')
    map_resps = {-1:1, 0:0, 1:2}
    corrs_all_subjects = []
    corrs_all_subjects_simul = []
    corr_per_coupling = {1.: [], 0.7: [], 0.: []}
    corr_per_coupling_simul = {1.: [], 0.7: [], 0.: []}
    pshuffs = [1., 0.7, 0.]
    for i_sub, subject in enumerate(subs):
        df_sub = df.loc[df.subject == subject]
        prev_doms = []
        nxt_doms = []
        prev_doms_simul = []
        nxt_doms_simul = []
        trial_index = df_sub.trial_index.unique()
        prev_dom_coup = {1.: [], 0.7: [], 0.: []}
        nxt_dom_coup = {1.: [], 0.7: [], 0.: []}
        prev_dom_coup_simul = {1.: [], 0.7: [], 0.: []}
        nxt_dom_coup_simul = {1.: [], 0.7: [], 0.: []}
        for i_trial, trial in enumerate(trial_index):
            df_trial = df_sub.loc[df_sub.trial_index == trial]
            responses = df_trial.responses.values
            responses_simul = [map_resps[resp] for resp in responses_all[i_sub, i_trial]]
            psh = df_trial.pShuffle.values[0]
            for i_r, r in enumerate([responses, responses_simul]):
                r = np.array(r)
                r = r[r != 0]
                try:
                    change = np.r_[True, np.diff(r) != 0, True]
                except ValueError:
                    continue
                starts = np.where(change[:-1])[0]
                ends = np.where(change[1:])[0]
                lengths = np.array(ends - starts)[:-1]
                [prev_doms, prev_doms_simul][i_r].append(lengths[:-1])
                [nxt_doms, nxt_doms_simul][i_r].append(lengths[1:])
                [prev_dom_coup, prev_dom_coup_simul][i_r][psh].append(lengths[:-1])
                [nxt_dom_coup, nxt_dom_coup_simul][i_r][psh].append(lengths[1:])
        nxt_doms = np.concatenate(nxt_doms)/fps
        prev_doms = np.concatenate(prev_doms)/fps
        # plt.plot(prev_doms, nxt_doms, marker='o', linestyle='')
        # plt.xlabel('Prev'); plt.ylabel('Next')
        nxt_doms_simul = np.concatenate(nxt_doms_simul)/fps
        prev_doms_simul = np.concatenate(prev_doms_simul)/fps
        corrs_all_subjects.append(np.corrcoef(prev_doms, nxt_doms)[0][1])
        corrs_all_subjects_simul.append(np.corrcoef(prev_doms_simul, nxt_doms_simul)[0][1])
        for ps in pshuffs:
            prev_coup = np.concatenate(prev_dom_coup[ps]) / fps 
            next_coup = np.concatenate(nxt_dom_coup[ps]) / fps 
            corr_per_coupling[ps].append(np.corrcoef(prev_coup, next_coup)[0][1])
            prev_coup_simul = np.concatenate(prev_dom_coup_simul[ps]) / fps 
            next_coup_simul = np.concatenate(nxt_dom_coup_simul[ps]) / fps 
            corr_per_coupling_simul[ps].append(np.corrcoef(prev_coup_simul, next_coup_simul)[0][1])
    fig, ax = plt.subplots(1, figsize=(4., 3.5))
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    sns.histplot(corrs_all_subjects, color='k', linewidth=3, label='Data')
    sns.histplot(corrs_all_subjects_simul, color='firebrick', linewidth=3,
                 label='Simulation')
    ax.legend(frameon=False); ax.set_xlabel('Consecutive dominance correlation')
    fig.tight_layout()
    fig, ax = plt.subplots(1, figsize=(4., 3.5))
    ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=3, zorder=1)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=3, zorder=1)
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.plot(corrs_all_subjects, corrs_all_subjects_simul, color='k', marker='o', linestyle='')
    r, p = pearsonr(corrs_all_subjects, corrs_all_subjects_simul)
    minval = np.min([corrs_all_subjects, corrs_all_subjects_simul])
    maxval = np.max([corrs_all_subjects, corrs_all_subjects_simul])
    ax.plot([minval-0.1, maxval+0.1], [minval-0.1, maxval+0.1],
            color='k', linestyle='--', alpha=0.3, linewidth=3)
    ax.set_ylim(minval-0.05, maxval+0.05); ax.set_xlim(minval-0.05, maxval+0.05)
    ax.annotate(f'r = {r:.3f}\np={p:.0e}', xy=(.04, 0.8), xycoords=ax.transAxes)
    ax.set_xlabel('r(data)'); ax.set_ylabel('r(simul)')
    fig.tight_layout()
    fig, ax = plt.subplots(ncols=2, figsize=(8., 3.5), sharex=True)
    for a in ax:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
        a.set_xlabel('Consecutive dom. correlation')
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    for i_psh, ps in enumerate(pshuffs):
        sns.histplot(corr_per_coupling[ps], color=colormap[i_psh], linewidth=3, label=ps,
                    ax=ax[0])
        sns.histplot(corr_per_coupling_simul[ps], color=colormap[i_psh], linewidth=3, label=ps,
                    ax=ax[1])
    ax[0].legend(frameon=False, title='p(shuffle)');
    fig.tight_layout()


def plot_subject_dominance_distributions(
    data_folder=DATA_FOLDER,
    fps=60,
    simulated=True,
    adaptation=True
):
    import numpy as np
    import glob
    import seaborn as sns
    import matplotlib.pyplot as plt

    # -------------------------
    # LOAD DATA
    # -------------------------
    df = load_data(
        data_folder + '/noisy/',
        n_participants='all',
        filter_subjects=True
    )
    subs = df.subject.unique()

    # -------------------------
    # LOAD PARAMETERS (only for sim)
    # -------------------------
    pars = sorted(glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy'))
    fitted_params_all = [np.load(p) for p in pars]

    # -------------------------
    # SIMULATION
    # -------------------------
    label = 'adaptation' if adaptation else ''
    responses_all = np.load(
        SV_FOLDER + f'responses_simulated_noise{label}.npy'
    )
    map_resps = {-1: 1, 0: 0, 1: 2}

    def doms(r):
        r = np.asarray(r)
        change = np.r_[True, np.diff(r) != 0, True]
        s = np.where(change[:-1])[0]
        e = np.where(change[1:])[0]
        seg = np.array([r[i] for i in s])
        L = np.array(e - s)
        return L[seg != 0] / fps

    # -------------------------
    # FIGURE
    # -------------------------
    ncols = 5
    nrows = int(np.ceil(len(subs) / ncols))

    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows),
                           sharex=True)
    ax = ax.flatten()

    bins = np.linspace(0, 25, 30)

    # -------------------------
    # SUBJECT LOOP
    # -------------------------
    for i_sub, subject in enumerate(subs):

        df_sub = df[df.subject == subject]
        trials = df_sub.trial_index.unique()

        all_data = []
        all_sim = []

        for i_t, tr in enumerate(trials):

            df_trial = df_sub[df_sub.trial_index == tr]

            r = df_trial.responses.values
            all_data.extend(doms(r))

            if simulated:
                r_sim = responses_all[i_sub, i_t]
                r_sim = [map_resps[x] for x in r_sim]
                all_sim.extend(doms(r_sim))

        # -------------------------
        # PLOT SUBJECT
        # -------------------------
        ax[i_sub].spines["right"].set_visible(False)
        ax[i_sub].spines["top"].set_visible(False)

        # DATA histogram (light gray)
        sns.histplot(
            all_data,
            bins=bins,
            stat="density",
            color="lightgray",
            alpha=0.6,
            edgecolor=None,
            ax=ax[i_sub]
        )

        # MODEL KDE (black)
        if simulated and len(all_sim) > 1:
            sns.kdeplot(
                all_sim,
                color="black",
                linewidth=2.2,
                ax=ax[i_sub],
                cut=0
            )
        #     mean_sim = np.nanmean(all_sim)
        #     ax[i_sub].axvline(mean_sim, color='k', linestyle='--',
        #                       alpha=0.8)
        # mean_data = np.nanmean(all_data)
        # ax[i_sub].axvline(mean_data, color='k',
        #                   alpha=0.8)

        ax[i_sub].set_title(subject)
        ax[i_sub].set_xlabel("Dominance (s)")

    ax[0].set_ylabel("Density")
    legend_elements = [
        Patch(facecolor="gray", alpha=0.3, label="data (hist)"),
        Line2D([0], [0], color="black", lw=2.5, label="model (KDE)")
    ]
    
    ax[0].legend(handles=legend_elements, frameon=False)
    fig.tight_layout()
    fig.savefig(
        SV_FOLDER + label +
        'noise_trials_dominance_regime_comparison_single_subject.png',
        dpi=400,
        bbox_inches='tight'
    )

    fig.savefig(
        SV_FOLDER + label +
        'noise_trials_dominance_regime_comparison_single_subject.svg',
        dpi=400,
        bbox_inches='tight'
    )


def plot_dominance_monostable_vs_bistable(
    data_folder=DATA_FOLDER,
    fps=60,
    simulated=True,
    unique_shuffle=[1., 0.7, 0.],
    regime_threshold=0.25,
    adaptation=True
):


    # -------------------------
    # LOAD DATA
    # -------------------------
    df = load_data(
        data_folder + '/noisy/',
        n_participants='all',
        filter_subjects=True
    )
    subs = df.subject.unique()

    # -------------------------
    # LOAD PARAMETERS
    # -------------------------
    pars = sorted(glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy'))
    fitted_params_all = [np.load(p) for p in pars]
    fitted_params_all = [[p[0], p[1], p[2], p[4], p[3]] for p in fitted_params_all]

    unique_shuffle = np.array(unique_shuffle)

    # -------------------------
    # REGIMES
    # -------------------------
    regimes_per_subject = {}

    for i_sub, sub in enumerate(subs):
        J = fitted_params_all[i_sub][0]
        bias = fitted_params_all[i_sub][1]

        jeffs = J * (1 - unique_shuffle) + bias

        regimes_per_subject[sub] = {
            psh: ("Bistable" if jeffs[i_psh] >= regime_threshold else "Monostable")
            for i_psh, psh in enumerate(unique_shuffle)
        }

    # -------------------------
    # SIMULATION
    # -------------------------
    label = 'adaptation' if adaptation else ''
    responses_all = np.load(
        SV_FOLDER + f'responses_simulated_noise{label}.npy'
    )
    map_resps = {-1: 1, 0: 0, 1: 2}

    # -------------------------
    # CONTAINER
    # -------------------------
    pooled = {
        "Monostable": {"data": [], "sim": []},
        "Bistable": {"data": [], "sim": []}
    }

    def doms(r):
        r = np.asarray(r)
        change = np.r_[True, np.diff(r) != 0, True]
        s = np.where(change[:-1])[0]
        e = np.where(change[1:])[0]
        seg = np.array([r[i] for i in s])
        L = np.array(e - s)
        return L[seg != 0] / fps

    # -------------------------
    # BUILD DISTRIBUTIONS
    # -------------------------
    for i_sub, subject in enumerate(subs):

        df_sub = df[df.subject == subject]
        trials = df_sub.trial_index.unique()

        for i_t, tr in enumerate(trials):

            df_trial = df_sub[df_sub.trial_index == tr]
            psh = df_trial.pShuffle.values[0]
            regime = regimes_per_subject[subject][psh]

            r = df_trial.responses.values
            pooled[regime]["data"].extend(doms(r))

            if simulated:
                r_sim = [map_resps[x] for x in responses_all[i_sub, i_t]]
                pooled[regime]["sim"].extend(doms(r_sim))

    # -------------------------
    # PLOT
    # -------------------------
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.5), sharex=True)
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)

    bins = np.linspace(0, 25, 30)
    colors = {"Monostable": "cadetblue", "Bistable": "peru"}

    for i, regime in enumerate(["Monostable", "Bistable"]):

        data = pooled[regime]["data"]
        sim = pooled[regime]["sim"]

        # DATA histogram
        sns.histplot(
            data,
            bins=bins,
            stat="density",
            color=colors[regime],
            alpha=0.3,
            edgecolor=None,
            ax=ax[i]
        )

        # MODEL KDE
        if len(sim) > 1:
            sns.kdeplot(
                sim,
                color=colors[regime],
                linewidth=2.5,
                ax=ax[i],
                cut=0,
                label="model"
            )
        #     mean_sim = np.nanmean(sim)
        #     ax[i].axvline(mean_sim, color=colors[regime], linestyle='--',
        #                   alpha=0.8)
        # mean_data = np.nanmean(data)
        # ax[i].axvline(mean_data, color=colors[regime],
        #               alpha=0.8)

        ax[i].set_title(regime, color=colors[regime])
        ax[i].set_xlabel("Dominance (s)")

    ax[0].set_ylabel("Density")
    legend_elements = [
        Patch(facecolor="gray", alpha=0.3, label="data (hist)"),
        Line2D([0], [0], color="black", lw=3.5, label="model (KDE)")
    ]
    
    ax[0].legend(handles=legend_elements, frameon=False)
    fig.tight_layout()
    fig.savefig(
        SV_FOLDER + label +
        'noise_trials_dominance_regime_comparison_model_data.png',
        dpi=400,
        bbox_inches='tight'
    )

    fig.savefig(
        SV_FOLDER + label +
        'noise_trials_dominance_regime_comparison_model_data.svg',
        dpi=400,
        bbox_inches='tight'
    )


def plot_dominance_distros_noise_trials_per_subject_regime(
        data_folder=DATA_FOLDER,
        fps=60,
        simulated=False,
        unique_shuffle=[1., 0.7, 0.],
        regime_threshold=0.25,
        adaptation=True
):
    """
    Plot dominance distributions conditioned on inferred regime.

    Regime definition:
        Jeff = J * (1 - pShuffle) + bias

    where:
        fitted_params[0] -> J
        fitted_params[1] -> baseline

    Regimes:
        Jeff < threshold  -> weak coupling
        Jeff >= threshold -> strong coupling
    """

    # =========================================================
    # LOAD DATA
    # =========================================================

    df = load_data(
        data_folder + '/noisy/',
        n_participants='all',
        filter_subjects=True
    )

    subs = df.subject.unique()

    # =========================================================
    # LOAD FITTED PARAMETERS
    # =========================================================

    pars = glob.glob(
        SV_FOLDER + 'fitted_params/ndt/' + '*.npy'
    )

    pars = sorted(pars)

    fitted_params_all = [
        np.load(par) for par in pars
    ]

    fitted_params_all = [
        [params[0], params[1], params[2], params[4], params[3]]
        for params in fitted_params_all
    ]

    unique_shuffle = np.array(unique_shuffle)

    # =========================================================
    # COMPUTE EFFECTIVE COUPLING / REGIME
    # =========================================================

    regimes_per_subject = {}

    for i_sub, sub in enumerate(subs):

        J = fitted_params_all[i_sub][0]
        bias = fitted_params_all[i_sub][1]

        jeffs = J * (1 - unique_shuffle) + bias

        regimes = {}

        for i_psh, psh in enumerate(unique_shuffle):

            regimes[psh] = (
                'Bistable'
                if jeffs[i_psh] >= regime_threshold
                else 'Monostable'
            )

        regimes_per_subject[sub] = regimes

    # =========================================================
    # SIMULATED RESPONSES
    # =========================================================
    label = 'adaptation' if adaptation else ''
    responses_all = np.load(
        SV_FOLDER + f'responses_simulated_noise{label}.npy'
    )

    map_resps = {-1: 1, 0: 0, 1: 2}

    # =========================================================
    # COLORS
    # =========================================================

    regime_colors = {
        'Monostable': 'cadetblue',
        'Bistable': 'peru'
    }

    # =========================================================
    # PER SUBJECT PLOTS
    # =========================================================

    fig, ax = plt.subplots(
        nrows=len(subs)//5,
        ncols=5,
        figsize=(15, 12),
        sharex=True,
        # sharey=True
    )

    ax = ax.flatten()

    dominance_all_regimes = {
        'Monostable': [],
        'Bistable': []
    }

    mean_dom_per_subject = {
        'Monostable': [],
        'Bistable': []
    }

    for i_sub, subject in enumerate(subs):

        ax[i_sub].spines['right'].set_visible(False)
        ax[i_sub].spines['top'].set_visible(False)

        df_sub = df.loc[df.subject == subject]

        trial_index = df_sub.trial_index.unique()

        doms_regime = {
            'Monostable': [],
            'Bistable': []
        }

        for i_trial, trial in enumerate(trial_index):

            df_trial = df_sub.loc[
                df_sub.trial_index == trial
            ]

            r = df_trial.responses.values

            if simulated:
                r = [
                    map_resps[resp]
                    for resp in responses_all[i_sub, i_trial]
                ]

            # -------------------------------------------------
            # segment dominance durations
            # -------------------------------------------------

            change = np.r_[True, np.diff(r) != 0, True]

            starts = np.where(change[:-1])[0]
            ends = np.where(change[1:])[0]

            segment_values = np.array([r[s] for s in starts])

            lengths = np.array(ends - starts)

            # exclude neutral state
            doms = lengths[segment_values != 0]/ fps
            # doms = np.array([26 / sum(segment_values != 0)])

            psh = df_trial.pShuffle.values[0]

            regime = regimes_per_subject[subject][psh]

            doms_regime[regime].extend(doms)

        # -----------------------------------------------------
        # plot per subject
        # -----------------------------------------------------

        for regime in ['Monostable', 'Bistable']:

            if len(doms_regime[regime]) > 3:

                sns.kdeplot(
                    doms_regime[regime],
                    color=regime_colors[regime],
                    linewidth=3,
                    ax=ax[i_sub],
                    label=regime,
                    cut=0
                )

                dominance_all_regimes[regime].extend(
                    doms_regime[regime]
                )

                mean_dom_per_subject[regime].append(
                    np.nanmean(doms_regime[regime])
                )

        ax[i_sub].set_title(subject)

        ax[i_sub].set_xlabel('Dominance (s)')

    ax[0].set_ylabel('Density')
    ax[0].legend(frameon=False)

    fig.tight_layout()

    label = 'simulated_' if simulated else ''

    # fig.savefig(
    #     SV_FOLDER + label +
    #     'noise_trials_dominance_regime_per_subject.png',
    #     dpi=400,
    #     bbox_inches='tight'
    # )

    # fig.savefig(
    #     SV_FOLDER + label +
    #     'noise_trials_dominance_regime_per_subject.pdf',
    #     dpi=400,
    #     bbox_inches='tight'
    # )

    # =========================================================
    # GROUP-LEVEL DISTRIBUTIONS
    # =========================================================

    fig, ax = plt.subplots(ncols=2, figsize=(7, 3.5),
                           sharex=True, sharey=False)

    for i_r, regime in enumerate(['Monostable', 'Bistable']):
    
        sns.histplot(
            dominance_all_regimes[regime],
            # stat='density',
            bins=np.linspace(0, 25, 50),
            color=regime_colors[regime],
            alpha=0.5,
            ax=ax[i_r]
        )
    
        # sns.kdeplot(
        #     dominance_all_regimes[regime],
        #     color=regime_colors[regime],
        #     linewidth=3,
        #     cut=0,
        #     ax=ax[i_r]
        # )
    
    ax[0].set_xlabel('Dominance duration (s)')
    ax[1].set_xlabel('Dominance duration (s)')
    ax[0].set_ylabel('Number of events')
    
    ax[0].set_title('Monostable', color='cadetblue', fontsize=14)
    ax[1].set_title('Bistable', color='peru', fontsize=14)
    sns.despine()
    fig.tight_layout()

    fig.savefig(
        SV_FOLDER + label +
        'noise_trials_dominance_regime_average.png',
        dpi=400,
        bbox_inches='tight'
    )

    fig.savefig(
        SV_FOLDER + label +
        'noise_trials_dominance_regime_average.svg',
        dpi=400,
        bbox_inches='tight'
    )

    # =========================================================
    # SUBJECT AVERAGES + SWARMPLOT
    # =========================================================

    f3, ax3 = plt.subplots(
        1,
        figsize=(4.5, 4)
    )

    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)

    means = [
        mean_dom_per_subject['Monostable'],
        mean_dom_per_subject['Bistable']
    ]

    positions = [0, 1]

    bar_means = [
        np.nanmean(means[0]),
        np.nanmean(means[1])
    ]

    bar_sds = [
        np.nanstd(means[0]),
        np.nanstd(means[1])
    ]

    ax3.bar(
        positions,
        bar_means,
        yerr=bar_sds,
        capsize=5,
        color=[
            regime_colors['Monostable'],
            regime_colors['Bistable']
        ]
    )

    # ---------------------------------------------------------
    # swarm points
    # ---------------------------------------------------------

    for i, regime in enumerate(['Monostable', 'Bistable']):

        x = np.random.normal(
            positions[i],
            0.04,
            size=len(mean_dom_per_subject[regime])
        )

        ax3.scatter(
            x,
            mean_dom_per_subject[regime],
            color='black',
            s=30,
            alpha=0.8
        )

    ax3.set_xticks(positions)

    ax3.set_xticklabels([
        'Monostable',
        'Bistable'
    ])

    ax3.set_ylabel('Mean dominance (s)')

    f3.tight_layout()

    # f3.savefig(
    #     SV_FOLDER + label +
    #     'noise_trials_dominance_regime_barplot.png',
    #     dpi=400,
    #     bbox_inches='tight'
    # )

    # f3.savefig(
    #     SV_FOLDER + label +
    #     'noise_trials_dominance_regime_barplot.pdf',
    #     dpi=400,
    #     bbox_inches='tight'
    # )

    # return (
    #     dominance_all_regimes,
    #     mean_dom_per_subject,
    #     regimes_per_subject
    # )


def plot_dominance_distros_noise_trials_per_subject(data_folder=DATA_FOLDER, fps=60, tFrame=26,
                                                    simulated=False):
    df = load_data(data_folder + '/noisy/', n_participants='all', filter_subjects=True)
    subs = df.subject.unique()
    fig, ax = plt.subplots(nrows=len(subs)//5, ncols=5, figsize=(14, 10), sharex=True)
    ax = ax.flatten()
    responses_all = np.load(SV_FOLDER + 'responses_simulated_noise.npy')
    map_resps = {-1:1, 0:0, 1:2}
    doms_all_subs = []
    dom_per_coupling = []
    for i_sub, subject in enumerate(subs):
        ax[i_sub].spines['right'].set_visible(False)
        ax[i_sub].spines['top'].set_visible(False)
        df_sub = df.loc[df.subject == subject]
        dominances = {0: [], 1: [], 2: []}
        trial_index = df_sub.trial_index.unique()
        dom_coup = {1.: [], 0.7: [], 0.: []}
        for i_trial, trial in enumerate(trial_index):
            df_trial = df_sub.loc[df_sub.trial_index == trial]
            r = df_trial.responses.values
            if simulated:
                r = [map_resps[resp] for resp in responses_all[i_sub, i_trial]]
            change = np.r_[True, np.diff(r) != 0, True]
            starts = np.where(change[:-1])[0]
            ends = np.where(change[1:])[0]
            segment_values = np.array([r[s] for s in starts])
            lengths = np.array(ends - starts)
            psh = df_trial.pShuffle.values[0]
            dom_coup[psh].append(np.nanmedian(lengths[segment_values != 0]))
            for i in [0, 1, 2]:
                dominances[i].append(lengths[segment_values == i])
        dominances = [np.concatenate(dominances[i])/fps for i in range(3)]
        dom_per_coupling.append(dom_coup)
        doms_all_subs.append([np.nanmean(dominances[i]) for i in range(3)])
        # sns.kdeplot(dominances[0], color='gray', linewidth=4, ax=ax[i_sub], label='No')
        sns.kdeplot(dominances[1], color='darkgreen', linewidth=4, ax=ax[i_sub], label='L',
                    cut=0)
        sns.kdeplot(dominances[2], color='firebrick', linewidth=4, ax=ax[i_sub], label='R',
                    cut=0)
        ax[i_sub].set_xlabel('Dominance (s)')
        ax[i_sub].set_ylabel('')
    ax[0].set_ylabel('Density')
    ax[0].legend(frameon=False)
    fig.tight_layout()
    label = 'simulated_' if simulated else ''
    fig.savefig(SV_FOLDER + label + 'noise_trials_dominance.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + label + 'noise_trials_dominance.pdf', dpi=400, bbox_inches='tight')
    f2, ax2 = plt.subplots(1, figsize=(4, 3.5))
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    doms_all = np.row_stack((doms_all_subs))
    sns.kdeplot(doms_all[:, 1], color='darkgreen', linewidth=4, ax=ax2, label='L',
                cut=0)
    sns.kdeplot(doms_all[:, 2], color='firebrick', linewidth=4, ax=ax2, label='R',
                cut=0)
    ax2.legend(frameon=False); ax2.set_xlabel('Dominance (s)'); f2.tight_layout()
    f3, a3 = plt.subplots(1, figsize=(4, 3.5))
    pshuffs = [1., 0.7, 0.]
    a3.spines['right'].set_visible(False)
    a3.spines['top'].set_visible(False)
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    for i in range(3):
        dominance_psh = [np.nanmean(dom_per_coupling[k][pshuffs[i]])/fps for k in range(len(subs))]
        sns.kdeplot(dominance_psh, linewidth=4, color=colormap[i], label=pshuffs[i],
                    )
    a3.legend(frameon=False, title='p(shuffle)'); a3.set_xlabel('Dominance (s)'); f3.tight_layout()
    f3.savefig(SV_FOLDER + label + 'average_noise_trials_dominance.png', dpi=400, bbox_inches='tight')
    f3.savefig(SV_FOLDER + label + 'average_noise_trials_dominance.pdf', dpi=400, bbox_inches='tight')


def plot_noise_before_switch(data_folder=DATA_FOLDER, fps=60, tFrame=18,
                             steps_back=60, steps_front=20,
                             shuffle_vals=[1, 0.7, 0], violin=False, sub='s_1',
                             avoid_first=False, window_conv=1, zscore_number_switches=False, ax=None,
                             fig=None, normalize_variables=True, hysteresis_area=False,
                             filter_subjects=True, legend_axes=True):
    nFrame = fps*tFrame
    df = load_data(data_folder + '/noisy/', n_participants='all', filter_subjects=filter_subjects)
    if sub is not None:
        df = df.loc[df.subject == sub]
    subs = df.subject.unique()
    print(subs, ', number:', len(subs))
    colormap = pl.cm.Oranges(np.linspace(0.3, 1, 3))[::-1]
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
    zscor = scipy.stats.zscore
    latency_avg = []
    fignew, axnew = plt.subplots(nrows=2, figsize=(5.5, 7.5))
    axnew[0].set_xlabel('Time from switch (s)'); axnew[0].set_ylabel('Noise')
    axnew[0].set_title('Average across trials', fontsize=13)
    all_kernels = []
    for i_sub, subject in enumerate(subs):
        df_sub = df.loc[df.subject == subject]
        mean_vals_noise_switch_all_shuffles_subject = np.empty((1, steps_back+steps_front))
        mean_vals_noise_switch_all_shuffles_subject[:] = np.nan
        dominances = {1: [], 2: []}
        for i_sh, pshuffle in enumerate(shuffle_vals):
            df_coupling = df_sub.loc[df_sub.pShuffle == pshuffle]
            trial_index = df_coupling.trial_index.unique()
            # mean_vals_noise_switch_all_trials = np.empty((len(trial_index), steps_back+steps_front))
            mean_vals_noise_switch_all_trials = np.empty((1, steps_back+steps_front))
            mean_vals_noise_switch_all_trials[:] = np.nan
            number_switches = []
            dominance_durations = []
            for i_trial, trial in enumerate(trial_index):
                df_trial = df_coupling.loc[df_coupling.trial_index == trial]
                responses = df_trial.responses.values
                r = np.copy(responses)
                change = np.r_[True, np.diff(r) != 0, True]
                starts = np.where(change[:-1])[0]
                ends = np.where(change[1:])[0]
                segment_values = np.array([r[s] for s in starts])
                lengths = np.array(ends - starts)
                for i in [1, 2]:
                    dominances[i].append(lengths[segment_values == i])
                chi = df_trial.stimulus.values
                # chi = chi-np.nanmean(chi)
                orders = rle(responses)
                if avoid_first:
                    idx_1 = orders[1][1:][orders[2][1:] == 2]
                    idx_0 = orders[1][1:][orders[2][1:] == 1]
                else:
                    idx_1 = orders[1][orders[2] == 2]
                    idx_0 = orders[1][orders[2] == 1]
                number_switches.append(len(idx_1)+len(idx_0))
                dominance_durations.extend(orders[0])
                idx_1 = idx_1[(idx_1 > steps_back) & (idx_1 < (len(responses))-steps_front)]
                idx_0 = idx_0[(idx_0 > steps_back) & (idx_0 < (len(responses))-steps_front)]
                # original order
                mean_vals_noise_switch = np.empty((len(idx_1)+len(idx_0), steps_back+steps_front))
                mean_vals_noise_switch[:] = np.nan
                for i, idx in enumerate(idx_1):
                    mean_vals_noise_switch[i, :] = chi[idx - steps_back:idx+steps_front]
                for i, idx in enumerate(idx_0):
                    mean_vals_noise_switch[i+len(idx_1), :] =\
                        chi[idx - steps_back:idx+steps_front]*-1
                mean_vals_noise_switch_all_trials = np.row_stack((mean_vals_noise_switch_all_trials, mean_vals_noise_switch))
            mean_vals_noise_switch_all_trials = mean_vals_noise_switch_all_trials[1:]
            mean_vals_noise_switch_all_shuffles_subject = np.row_stack((mean_vals_noise_switch_all_shuffles_subject, mean_vals_noise_switch_all_trials))
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
        mean_vals_noise_switch_all_shuffles_subject = mean_vals_noise_switch_all_shuffles_subject[1:]
        x_plot = np.arange(-steps_back, steps_front, 1)/fps
        latency = (np.argmax(np.nanmean(mean_vals_noise_switch_all_trials[:, :-steps_front], axis=0))-steps_back)/fps
        peakval = np.nanmax(np.nanmean(mean_vals_noise_switch_all_trials[:, :-steps_front], axis=0))
        kernel = np.nanmean(mean_vals_noise_switch_all_trials, axis=0)
        latency_avg.append(latency)
        all_kernels.append(kernel)
        axnew[0].plot(x_plot, kernel, color='k', linewidth=2, alpha=0.5, zorder=1)
        axnew[0].plot(latency, peakval, marker='*', color='firebrick', markersize=8, zorder=5)
        if len(subs) > 1 and zscore_number_switches:
            label = 'z-scored '
        else:
            label = ''
    sns.histplot(x=latency_avg, ax=axnew[1], color='firebrick', bins=15)
    sns.kdeplot(x=latency_avg, ax=axnew[1], color='firebrick', bw_adjust=0.5, linewidth=3)
    axnew[1].axvline(np.mean(latency_avg), color='k', linewidth=2, label='Mean')
    axnew[1].axvline(np.median(latency_avg), linestyle='--', color='k', linewidth=2, label='Median')
    axnew[1].legend(frameon=False)
    axnew[1].set_xlabel('Latency (s)'); axnew[1].set_xlim(axnew[0].get_xlim())
    np.save(DATA_FOLDER + 'kernel_latency_average.npy', np.array(latency_avg))
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 4))
    fig2, ax2 = plt.subplots(1, figsize=(5., 4))
    fig3, ax34567 = plt.subplots(ncols=3, nrows=2, figsize=(12.5, 8))
    ax34567= ax34567.flatten()
    ax3, ax4, ax5, ax6, ax7, ax8 = ax34567

    def corrfunc(x, y, ax=None, **kws):
        """Plot the correlation coefficient in the top left hand corner of a plot."""
        r, p = scipy.stats.pearsonr(x, y)
        ax = ax or plt.gca()
        ax.annotate(f'r = {r:.3f}\np = {p:.1e}', xy=(.1, 0.8), xycoords=ax.transAxes)

    if hysteresis_area:
        np.save(DATA_FOLDER + 'mean_peak_amplitude_per_subject.npy', mean_peak_amplitude)
        np.save(DATA_FOLDER + 'mean_peak_latency_per_subject.npy', mean_peak_latency)
        np.save(DATA_FOLDER + 'mean_number_switches_per_subject.npy', mean_number_switchs_coupling)
        hyst_width_2 = np.load(DATA_FOLDER + 'hysteresis_width_freq_2.npy')
        hyst_width_4 = np.load(DATA_FOLDER + 'hysteresis_width_freq_4.npy')
        datframe = pd.DataFrame({'Amplitude': mean_peak_amplitude.flatten(),
                                 'Latency': mean_peak_latency.flatten(),
                                 'Dominance': mean_number_switchs_coupling.flatten(),
                                 'Width f2': hyst_width_2.flatten(),
                                 'Width f4': hyst_width_4.flatten()})
        delta_dominance = (mean_number_switchs_coupling[-1]-mean_number_switchs_coupling[0])  # /mean_number_switchs_coupling[0]
        delta_hysteresis_2 = hyst_width_2[-1]-hyst_width_2[0]
        delta_hysteresis_4 = hyst_width_4[-1]-hyst_width_4[0]
        avg_hyst_2 = np.nanmean(hyst_width_2, axis=0)
        avg_hyst_4 = np.nanmean(hyst_width_4, axis=0)
        avg_dominance = np.nanmean(mean_number_switchs_coupling, axis=0)
        datframe_deltas  = pd.DataFrame({'Difference dominance': delta_dominance,
                                         'Difference H2 width': delta_hysteresis_2,
                                         'Difference H4 width': delta_hysteresis_4,
                                         '<Hyst>f2': avg_hyst_2,
                                         '<Hyst>f4': avg_hyst_4,
                                         '<T_domin>': avg_dominance})
        deltasplot = sns.pairplot(datframe_deltas)
        deltasplot.map_lower(corrfunc)
    else:
        datframe = pd.DataFrame({'Amplitude': mean_peak_amplitude.flatten(),
                                 'Latency': mean_peak_latency.flatten(),
                                 'Dominance': mean_number_switchs_coupling.flatten()})
    g = sns.pairplot(datframe)
    g.map_lower(corrfunc)
    for a in [ax, ax2, ax3, ax4, ax5, ax6, ax7, ax8, axnew[0], axnew[1]]:
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
    # mean_vals_noise_switch_coupling[:, steps_back-20:-steps_front].T
    # a, b, c = all_noises
    a, b, c = np.nanmean(mean_vals_noise_switch_coupling[:, steps_back-40:-steps_front], axis=1)
    # all_noises = [a.flatten(), b.flatten(), c.flatten()]
    pv_sh1 = stars_pval(scipy.stats.ttest_1samp(a.flatten(), 0).pvalue)
    pv_sh07 = stars_pval(scipy.stats.ttest_1samp(b.flatten(), 0).pvalue)
    pv_sh0 = stars_pval(scipy.stats.ttest_1samp(c.flatten(), 0).pvalue)
    pvs = [pv_sh1, pv_sh07, pv_sh0]
    if violin:
        # yvals = np.nanmax(mean_vals_noise_switch_coupling[:, steps_back-40:-steps_front], axis=1)
        yvals = np.nanmax(mean_vals_noise_switch_coupling[:, steps_back-40:-steps_front], axis=1)
        sns.violinplot(yvals.T, palette=colormap, ax=ax2,
                        zorder=2, fill=False, inner='point')
        # sns.swarmplot(yvals.T, color='k', ax=ax2,
        #               zorder=9, size=3, legend=False, edgecolor='')
        g = sns.lineplot(yvals, palette=colormap, ax=ax2,
                          zorder=10, markers='', alpha=0.2, legend=False)
        lines = g.get_lines()
        [l.set_color('black') for l in lines]
        [l.set_linestyle('solid') for l in lines]
        h_1, h_07, h_0 = np.nanmax(a), np.nanmax(b), np.nanmax(c)
    else:
        yvals = np.nanmean(mean_vals_noise_switch_coupling[:, steps_back-40:-steps_front], axis=-1)
        sns.barplot(yvals.T, palette=colormap, ax=ax2,
                    zorder=2)
        h_1, h_07, h_0 = np.nanmean(a), np.nanmean(b), np.nanmean(c)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
    heights = [h_1, h_07, h_0]
    offset = 0.55 if violin else 0.05
    # ax2.set_ylim(np.nanmin(a)-0.1, np.nanmax(heights)+0.2)
    for a in range(3):
        ax2.text(a, heights[a]+offset, f"{pvs[a]}", ha='center', va='bottom', color='k',
                  fontsize=12)
    ax2.set_xticks([0, 1, 2], [1, 0.7, 0])
    ax2.set_xlabel('p(Shuffle)')
    ax2.set_ylabel('Noise before switch')
    ax.set_xlabel('Time from switch (s)')
    ax.set_ylabel('Noise')
    if legend_axes:
        ax.legend(title='p(shuffle)', frameon=False)
    if not legend_axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(''); ax.set_ylabel('')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fignew.tight_layout()
    fignew.savefig(SV_FOLDER + 'latency_computation.png', dpi=400, bbox_inches='tight')
    fignew.savefig(SV_FOLDER + 'latency_computation.pdf', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'noise_before_switch_experiment.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'noise_before_switch_experiment.svg', dpi=400, bbox_inches='tight')
    figlast, ax = plt.subplots(ncols=1, figsize=(5, 4))
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
    x_plot = np.arange(-steps_back, steps_front, 1)/fps
    y_plot = np.nanmean(all_kernels, axis=0)
    np.save(DATA_FOLDER + 'all_kernels_noise_switch_aligned.npy', all_kernels)
    ax.plot(x_plot, y_plot, color='k', linewidth=4, )
    err = np.nanstd(all_kernels, axis=0)/np.sqrt(len(subs))
    ax.fill_between(x_plot, y_plot-err, y_plot+err, color='k', alpha=0.2)
    ax.set_xlabel('Time from switch(s)')
    ax.set_ylabel('Noise')
    figlast.tight_layout()
    figlast.savefig(SV_FOLDER + 'average_kernel_across_subjects.png', dpi=400, bbox_inches='tight')
    figlast.savefig(SV_FOLDER + 'average_kernel_across_subjects.svg', dpi=400, bbox_inches='tight')
    return fig, fig2, fig3, g, fignew


def plot_model_data_average_kernel(steps_back=150, steps_front=10,
                                   fps=60):
    x_time = np.arange(-steps_back, steps_front, 1)/fps
    
    kernels_data = np.load(DATA_FOLDER + 'all_kernels_noise_switch_aligned.npy')
    kernels_model = np.load(DATA_FOLDER + 'simulated_all_kernels_noise_switch_aligned.npy')
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    mean_data = np.nanmean(kernels_data, axis=0)
    err_data = np.nanstd(kernels_data, axis=0) / np.sqrt(kernels_data.shape[0])
    mean_model = np.nanmean(kernels_model, axis=0)
    err_model = np.nanstd(kernels_model, axis=0) / np.sqrt(kernels_model.shape[0])
    
    ax.plot(x_time, mean_data, color='k', linewidth=3, label='Data')
    ax.fill_between(x_time, mean_data-err_data, mean_data+err_data,
                    color='k', alpha=0.3)
    ax.plot(x_time, mean_model, color='firebrick', linewidth=3, label='Model')
    ax.fill_between(x_time, mean_model-err_model, mean_model+err_model,
                    color='firebrick', alpha=0.3)
    ax.legend(frameon=False)
    ax.set_xlabel('Time from switch (s)')
    ax.set_ylabel('Noise')
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'average_kernel_model_data.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'average_kernel_model_data.svg', dpi=400, bbox_inches='tight')


def experiment_example(nFrame=1560, fps=60, noisyframes=15):
    colormap = COLORMAP
    # coolwarm_r
    dt = 1/fps
    noise_exp = np.random.randn(nFrame // noisyframes+1)*0.1
    time_interp = np.arange(0, nFrame+1, noisyframes)*dt
    noise_signal = interp1d(time_interp, noise_exp)
    time = np.arange(0, nFrame, 1)*dt
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 4))
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Noise')
    ax[0].set_ylabel('Depth cue, c(t)')
    difficulty_time_ref_2 = np.linspace(2, -2, nFrame//2)
    stimulus = np.concatenate(([difficulty_time_ref_2, -difficulty_time_ref_2]))
    ax[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    # ax[0].plot(time, stimulus, linewidth=4, label='2', color='navy')
    line = colored_line(time, stimulus, stimulus, ax[0],
                        linewidth=4, cmap=colormap, 
                        norm=plt.Normalize(vmin=-2, vmax=2), label='2')
    ax[0].set_xlim([np.min(time)-1e-1, np.max(time)+1e-1])
    ax[0].set_yticks([-2, 0, 2], ['-1', '0', '1'])
    # difficulty_time_ref_4 = np.linspace(-2, 2, nFrame//4)
    # stimulus = np.concatenate(([difficulty_time_ref_4, -difficulty_time_ref_4,
    #                             difficulty_time_ref_4, -difficulty_time_ref_4]))
    # # ax[0].plot(time, stimulus, linewidth=4, label='4', color='navy', linestyle='--')
    # line = colored_line(time, stimulus, stimulus, ax[0],
    #                     linewidth=4, cmap=colormap, linestyle='--', 
    #                     norm=plt.Normalize(vmin=-2, vmax=2), label='4')
    # # ax[0].legend(title='Freq.', frameon=False)
    # legendelements = [Line2D([0], [0], color='k', lw=4, label='2'),
    #                   Line2D([0], [0], color='k', lw=4, label='4', linestyle='--')]
    #                   # Line2D([0], [0], color='b', lw=2, label='5')]
    # ax[0].legend(handles=legendelements, title='Freq', frameon=False)
    ax[1].axhline(0, color='lightblue', linestyle='--', alpha=1)
    # ax[1].plot(time, noise_signal(time), color='navy', linewidth=4)
    vals_max = np.max(np.abs(noise_signal(time)))
    line = colored_line(time, noise_signal(time), noise_signal(time), ax[1],
                        linewidth=4, cmap=colormap,
                        norm=plt.Normalize(vmin=-vals_max, vmax=vals_max))
    ax[1].set_ylim(-0.4, 0.4)
    ax[1].set_yticks([0])
    ax[1].set_xlim([np.min(time)-1e-1, np.max(time)+1e-1])
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'stim_example.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'stim_example.pdf', dpi=400, bbox_inches='tight')


def plot_example(theta=[0.1, 0, 0.5, 0.1, 0.5], data_folder=DATA_FOLDER,
                 fps=60, tFrame=18, model='MF', prob_flux=False, freq=4,
                 idx=0):
    theta = [1.72243614, -0.11800575, 1.35294342, 0.16528025, 0.54757462]
    # theta = [ 1.45528034e-12,  1.06029511e+00, -1.31412416e-01,  8.46116941e-01,
    #   1.56537657e-01,  1.49512790e-01]
    df = load_data(data_folder, n_participants='all')
    df_freq = df.loc[(df.freq == freq) & (df.pShuffle == 0)]
    T = tFrame  # total duration (seconds)
    n_times = tFrame*fps
    times = np.linspace(0, T, n_times)
    responses_4, stim_values_4 = get_response_and_blist_array(df_freq, fps=fps, tFrame=tFrame)
    keypresses = responses_4[idx]
    flux_right_left, flux_left_right, p_right, p_left = solve_fokker_planck_1d(times, stim_values_4, theta,
                                                                               dx=0.025, model=model, shift_ini=0)
    flux_left_right = np.array(flux_left_right)
    flux_left_right[flux_left_right < 0] = 0
    flux_right_left = np.array(flux_right_left)
    flux_right_left[flux_right_left < 0] = 0
    prob_switch_L_to_R = flux_left_right/p_left/fps  # hazard_rate_L_R = flux_L_R/p(L)
    # p_switch_L_R = hazard_rate_L_R*dt
    prob_switch_R_to_L = flux_right_left/p_right/fps
    fig, ax = plt.subplots(1, figsize=(6, 5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if prob_flux:
        ax.plot(times, flux_left_right, label=r'$S(\theta^{+}, t)$', linewidth=3)
        ax.plot(times, flux_right_left, label=r'$-S(\theta^{-}, t)$', linewidth=3)
        ax.set_ylim(-0.02, 0.45)
        ax.set_ylabel(r'Probability current $S(\theta, t)$')
    else:
        ax.plot(times, prob_switch_L_to_R, label=r'$p_{L \to R} (t)$', linewidth=3)
        ax.plot(times, prob_switch_R_to_L, label=r'$p_{R \to L} (t)$', linewidth=3)
        ax.set_ylim(-0.001, 0.06)
        ax.set_ylabel(r'Instantaneous switch probability')
    # plt.plot(times, 0.5*(stim_values_4/2+1), label='Stimulus', linewidth=3)
    ax2 = ax.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.plot(times, keypresses, label='response', color='k', linewidth=3)
    ax.set_xlabel('Time (s)')
    ax2.set_ylabel('Responses')
    
    ax.legend(frameon=False, ncol=2, bbox_to_anchor=[0.1, 1.2])
    ax2.legend(frameon=False, bbox_to_anchor=[0.72, 1.32])
    ax2.set_yticks([0, 1])
    fig.tight_layout()


def plot_example_pswitch(params=[0.1, 1e-2, 0, 0.5, 0.15], data_folder=DATA_FOLDER,
                         fps=60, tFrame=26, freq=4, idx=10, n=3.92, theta=0.5,
                         tol=1e-6, pshuffle=0):
    j_par, j0, b0, b_par, sigma = params
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > 8]
    df_freq = df.loc[df.freq == freq]
    j_eff = j_par*(1-pshuffle)+j0
    T = tFrame  # total duration (seconds)
    n_times = tFrame*fps
    responses, responses_4, stimulus, barray_4 = collect_responses(df, subjects=df.subject.unique(),
                                                                   coupling_levels=[1-pshuffle], fps=60, tFrame=26)
    responses = join_trial_responses(responses[0][idx])[idx]
    vals = np.abs(np.diff(np.sign(stimulus)))
    idx_diff = np.where(vals != 0)[0]
    keypresses = responses[idx]
    # responses = responses_clean(keypresses)
    dt = 1/fps
    switches = np.abs(np.diff(responses))
    b_t = b_par*stimulus
    lambda_list = []
    prob_list = []
    p_stay = []
    times = []
    xl = []
    xr = []
    xu = []
    uns_axvlines = []
    for t in range(len(stimulus)-1):
        resp_t = responses[t]
        if resp_t == 0:
            continue
        times.append(t*dt)
        switch_t = switches[t]
        b_eff = b0+b_t[t]
        xR, xL, x_unstable = get_unst_and_stab_fp(j_eff, b_eff, n=n)
        xr.append(xR)
        xl.append(xL)
        xu.append(x_unstable)
        # mu_0 = compute_linearized_drift(xR, j_eff, n=n)
        if np.isnan(x_unstable) or np.abs(xL-xR) < tol:  # monostable
            # mu_0 = compute_linearized_drift(xL, j_eff, n=n)
            lambda_t = compute_nu_0_real(q=xL, j=j_eff, b=b_eff, sigma=sigma,
                                         theta=theta, n=n)
            # lambda_t = compute_nu_0(mu_0=xL, sigma=sigma, theta=theta)
            # lambda_t = np.nan
        else:
            if resp_t == -1:
                lambda_t = k_i_to_j(j_eff, xL, x_unstable, noise=sigma,
                                    b=b_eff, n=n)
            if resp_t == 1:
                lambda_t = k_i_to_j(j_eff, xR, x_unstable, noise=sigma,
                                    b=b_eff, n=n)
        p_stay.append(np.exp(-lambda_t*dt))
        if switch_t == 0:
            prob_list.append(np.exp(-lambda_t*dt))
        if switch_t == 2:
            prob_list.append(np.max([1-np.exp(-lambda_t*dt), 1e-12]))
        if switch_t == 1:
            prob_list.append(np.nan)
        lambda_list.append(lambda_t)
    if j_eff > 1/n:
        delta = np.sqrt(1-1/(j_eff*n))
        b_crit1 = (np.log((1-delta)/(1+delta))+2*n*j_eff*delta)/2
        b_crit1 = np.round(b_crit1, 4)
        idx_crits = []
        counter = 1
        while len(idx_crits) == 0:
            idx_crits = np.arange(0, T, dt)[np.isclose(np.abs(stimulus), b_crit1, rtol=1e-3*counter)]
            counter += 1
    fig, ax = plt.subplots(ncols=3, figsize=(14, 5))
    ax[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        [a.axvline(index_time*dt, color='k', linestyle='--', alpha=0.3) for index_time in idx_diff]
        if j_eff > 1/n:
            [a.axvline(index_time, color='r', linestyle='--', alpha=0.5) for index_time in idx_crits]
        a.set_xlabel('Time (s)')
    ax[2].plot(times, prob_list, label=r'$S(\theta^{+}, t)$', linewidth=3)
    ax[1].plot(times, p_stay, label=r'$S(\theta^{+}, t)$', linewidth=3)
    ax[0].plot(np.arange(0, T, dt), stimulus/10, label=r'$S(\theta^{+}, t)$', linewidth=3)
    ax[2].set_ylabel(r'$P \,(\,r^t \,| \,r^{t-1}\,)$  ')
    ax[1].set_ylabel(r'P(stay)')
    ax[0].plot(times, lambda_list, label=r'$p_{L \to R} (t)$', linewidth=3)
    ax[2].plot(np.arange(0, T, dt), responses/10, label=r'$S(\theta^{+}, t)$', linewidth=3)
    ax[0].set_ylabel(r'Switch rate')
    fig.tight_layout()


def return_input_output_for_network_data(params, choice, freq, nFrame=1200, fps=60,
                                         max_number=10, coupling=1, coupling_return=False):
    dt = 1/fps
    tFrame = nFrame*dt
    # Find the indices where the value changes
    change_indices = np.where(choice[1:] != choice[:-1])[0] + 1
    
    # Start indices of epochs (always include 0)
    start_indices = np.concatenate(([0], change_indices))
    
    # Corresponding response values at those start indices
    responses = choice[start_indices]
    mask = np.where((responses != 0) + (start_indices == 0))[0]
    # for data, we discard responses=0 (no press) when time > 0

    # Combine into array of [response, start_index]
    result = np.column_stack((responses[mask], start_indices[mask]/nFrame))
    # If you want it as a list of [response, time] pairs:
    output_network_0 = np.concatenate((change_indices[mask[:-1]], [nFrame]))/nFrame
    output_network = np.column_stack((output_network_0, np.roll(responses[mask], 1)))
    params_repeated = np.column_stack(np.repeat(params.reshape(-1, 1), result.shape[0], 1))
    if len(params) == 5:
        params_repeated[:, 1] = params_repeated[:, 0] + params_repeated[:, 1]*coupling
        params_repeated = params_repeated[:, 1:]
    if len(params) == 4:
        params_repeated[:, 0] *= coupling
    input_network = np.column_stack((params_repeated, result, np.repeat(freq, result.shape[0])))
    if coupling_return:
        coup_all = np.repeat(coupling, result.shape[0])
        return input_network[1:], output_network[1:], coup_all[1:]
    else:
        return input_network[1:], output_network[1:]


def lmm_hysteresis_pshuffle(freq=2, plot_summary=True,
                            slope_random_effect=False, plot_individual=False):
    y2 = np.load(DATA_FOLDER + 'hysteresis_width_freq_2.npy')
    y4 = np.load(DATA_FOLDER + 'hysteresis_width_freq_4.npy')
    x = np.repeat(np.array([0, 0.3, 1.]), y2.shape[1]).reshape(y2.shape)
    N, M = x.shape
    if freq == 2:
        y = y2
    if freq == 4:
        y = y4
    df = pd.DataFrame({"x": x.flatten(),
                       "y": y.flatten(),
                       "subject": np.repeat(np.arange(M), N),
                       "condition": np.tile(np.arange(N), M)})
    re_formula = "~x" if slope_random_effect else "1"
    model = smf.mixedlm("y ~ x", df, groups=df["subject"],
                        re_formula=re_formula)
    result = model.fit()
    fe = result.fe_params
    re = result.random_effects
    # get intercepts/slopes per subject
    intercepts = [fe["Intercept"] + eff.get("Group", 0) for subj, eff in re.items()]
    slopes = [fe["x"] + eff.get("x", 0) for subj, eff in re.items()]
    if plot_summary:
        print(result.summary())
        # fixed effects
        intercept = result.fe_params["Intercept"]
        slope = result.fe_params["x"]
        
        x_range = np.linspace(np.min(x), np.max(x), 100)
        y_pred = intercept + slope * x_range
        fig = plt.figure()
        plt.xlabel('p(shuffle)')
        plt.ylabel('Hysteresis area')
        plt.plot(x_range, y_pred, color="black", linewidth=4, label="Fixed effect")
        if plot_individual:
            random_effects = result.random_effects
            s = 0
            for subj, re in random_effects.items():
                subj_intercept = intercepts[s]
                subj_slope = slopes[s]
                plt.plot(x_range, subj_intercept + subj_slope * x_range,
                          alpha=0.3, color='k')
                s += 1
        else:
            cov = result.cov_params()
        
            var_intercept = cov.loc["Intercept", "Intercept"]
            var_slope = cov.loc["x", "x"]
            cov_intercept_slope = cov.loc["Intercept", "x"]
            
            y_range = np.linspace(min(df["x"]), max(df["x"]), 100)
            
            intercept = result.params["Intercept"]
            slope = result.params["x"]
            
            y_pred = intercept + slope * y_range
            
            z_val = 1.96  # for 95% CI
            
            SE = np.sqrt(
                var_intercept +
                (y_range ** 2) * var_slope +
                2 * y_range * cov_intercept_slope
            )
            
            y_pred_lower = y_pred - z_val * SE
            y_pred_upper = y_pred + z_val * SE
    
            
            plt.fill_between(x_range, y_pred_lower, y_pred_upper, alpha=0.3,
                             color='k')
        plt.plot(x+np.random.randn(y2.shape[0], y2.shape[1])*0.01, y, color='k', marker='o', linestyle='')
        fig.tight_layout()


def plot_dominance_versus_hysteresis():
    dominance = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
    hyst_width_2 = np.load(DATA_FOLDER + 'hysteresis_width_freq_2.npy')
    hyst_width_4 = np.load(DATA_FOLDER + 'hysteresis_width_freq_4.npy')
    dom_mean = dominance[0, :]  # np.mean(dominance, axis=0)
    hyst_mean = hyst_width_2[0, :]  # np.mean(hyst_width_2, axis=0)
    r, p = pearsonr(dom_mean, hyst_mean)
    fig, ax = plt.subplots(1, figsize=(4, 3.5))
    ax.spines['right'].set_visible(False);  ax.spines['top'].set_visible(False)
    ax.annotate(f'r = {r:.3f}\np={p:.0e}', xy=(.04, 0.8), xycoords=ax.transAxes)
    ax.plot(dom_mean, hyst_mean, color='k', marker='o', linestyle='')
    ax.set_xlabel('Dominance in noise trials');  ax.set_ylabel('Hysteresis in freq=2 trials')
    fig.tight_layout()
    fig2, ax2 = plt.subplots(1, figsize=(4, 3.5))
    ax2.spines['right'].set_visible(False);  ax2.spines['top'].set_visible(False)
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    pshuffles = [1., 0.7, 0.]
    for i in range(3):
        sns.kdeplot(dominance[i, :], color=colormap[i], linewidth=4, label=pshuffles[i])
    ax2.set_xlabel('Dominance (s)');  ax2.legend(frameon=False, title='p(shuffle)')
    fig2.tight_layout()
    fig2, ax2 = plt.subplots(1, figsize=(4, 3.5))
    ax2.spines['right'].set_visible(False);  ax2.spines['top'].set_visible(False)
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    pshuffles = [1., 0.7, 0.]
    sns.barplot(dominance.T, ax=ax2, palette=colormap, errorbar='se')
    ax2.set_ylim(4, 9.5);  ax2.set_xticks([0, 1, 2], pshuffles)
    ax2.set_xlabel('p(shuffle'); ax2.set_ylabel('Dominance (s)')
    fig2.tight_layout()


def plot_noise_variables_versus_hysteresis():
    mean_peak_amplitude = np.load(DATA_FOLDER + 'mean_peak_amplitude_per_subject.npy')
    mean_peak_latency = np.load(DATA_FOLDER + 'mean_peak_latency_per_subject.npy')
    mean_number_switchs_coupling = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
    hyst_width_2 = np.load(DATA_FOLDER + 'hysteresis_width_freq_2.npy')
    hyst_width_4 = np.load(DATA_FOLDER + 'hysteresis_width_freq_4.npy')
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    n_shuffle = hyst_width_2.shape[0]
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(9, 5))
    ax = ax.flatten()
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    kwargs = {'linestyle': '', 'marker': 'o'}
    for ish in range(n_shuffle):
        ax[0].plot(mean_peak_amplitude[ish], hyst_width_2[ish], color=colormap[ish],
                   **kwargs)
        ax[1].plot(mean_peak_latency[ish], hyst_width_2[ish], color=colormap[ish],
                   **kwargs)
        ax[2].plot(mean_number_switchs_coupling[ish], hyst_width_2[ish], color=colormap[ish],
                   **kwargs)
        ax[3].plot(mean_peak_amplitude[ish], hyst_width_4[ish], color=colormap[ish],
                   **kwargs)
        ax[4].plot(mean_peak_latency[ish], hyst_width_4[ish], color=colormap[ish],
                   **kwargs)
        ax[5].plot(mean_number_switchs_coupling[ish], hyst_width_4[ish], color=colormap[ish],
                   **kwargs)
    xlabels = ['Amplitude', 'Latency', 'dominance']
    for ia, a in enumerate([ax[3], ax[4], ax[5]]):
        a.set_xlabel(xlabels[ia])
    ax[0].set_ylabel('Width, freq = 2')
    ax[3].set_ylabel('Width, freq = 4')
    fig2, ax2 = plt.subplots(ncols=3, nrows=2, figsize=(10, 7))
    ax2 = ax2.flatten()
    pshuffles = [1, 0.7, 0.]
    for i in range(3):
        ax2[i].set_title(f'p(shuffle) = {pshuffles[i]}', fontsize=15)
        vardom = mean_number_switchs_coupling[i]
        varhyst = hyst_width_2[i]
        y_lims = [np.min(varhyst)-0.4, np.max(varhyst)+0.4]
        x_lims = [np.min(vardom)-1, np.max(vardom)+1]
        ax2[i].set_ylim(y_lims)
        ax2[i].set_xlim(x_lims)
        ax2[i].plot(x_lims, y_lims, color='gray', alpha=0.5, linewidth=3, linestyle='--')
        r, p = pearsonr(vardom, varhyst)
        ax2[i].annotate(f'r = {r:.3f}\np={p:.0e}', xy=(.04, 0.8), xycoords=ax2[i].transAxes)
        ax2[i].plot(vardom, varhyst, color=colormap[i],
                    **kwargs)
        varhyst = hyst_width_4[i]
        y_lims = [np.min(varhyst)-0.4, np.max(varhyst)+0.4]
        ax2[i+3].set_ylim(y_lims)
        ax2[i+3].set_xlim(x_lims)
        ax2[i+3].plot(x_lims, y_lims, color='gray', alpha=0.5, linewidth=3, linestyle='--')
        r, p = pearsonr(vardom, varhyst)
        ax2[i+3].annotate(f'r = {r:.3f}\np={p:.0e}', xy=(.04, 0.8), xycoords=ax2[i+3].transAxes)
        ax2[i+3].plot(vardom, varhyst, color=colormap[i],
                    **kwargs)
    ax2 = ax2.flatten()
    for a in ax2:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    ax2[4].set_xlabel('Dominance (s)')
    ax2[0].set_ylabel('Hysteresis f=2')
    ax2[3].set_ylabel('Hysteresis f=4')
    fig2.tight_layout()


def cartoon_hysteresis_responses(data_folder=DATA_FOLDER,
                                 sv_folder=SV_FOLDER,
                                 ntraining=8, simulated_subject='s_36',
                                 fps=60, idx_trial=4, ntrials=72, n=4,
                                 nfreq=1, plot_response=False, fps_sim=200,
                                 n_trials=200):
    # for subject s_36, good trial_idx for plotting responses are 4 and 5
    # for subject s_35, good trial_idx for responses (2,3,4,5) & pupil (2,3, 5)
    nFrame = fps*26
    nFrame_sim = fps_sim*26
    df = load_data(data_folder, n_participants='all', preprocess_data=True)
    df = df.loc[df.trial_index > ntraining]
    df_subject = df.loc[df.subject == simulated_subject]
    trial_index = df_subject.trial_index.unique()
    figsize = (7.5, 8) if nfreq == 2 else (6, 8)
    fig_sim, ax_sim = plt.subplots(ncols=nfreq, nrows=3, figsize=figsize, sharex=True)
    fig_dat, ax_dat = plt.subplots(ncols=nfreq, nrows=3, figsize=figsize, sharex=True)
    fig_dat2, ax_dat2 = plt.subplots(ncols=nfreq, nrows=2, figsize=(6, 1.5), sharex=True,
                                     sharey=True)
    fig_pupil, ax_pupil = plt.subplots(ncols=nfreq, nrows=2, figsize=(6, 1.5), sharex=True,
                                       sharey=True)
    example_shuffles = [0., 1.]
    example_freqs = [2, 4][:nfreq]
    fitted_params = [0.3, 0.1, 0.3, 0., 0.0, 0.5]
    colormap = COLORMAP
    crossing_2 = [6.5, 19.5]
    crossing_4 = [3.25, 9.65, 16.15, 22.65]
    crossings = [crossing_2, crossing_4]
    if nfreq == 1:
        ax_sim = np.expand_dims(ax_sim, axis=1)
        ax_dat = np.expand_dims(ax_dat, axis=1)
    ax_sim[0, 0].text(0.005, 0.14, 'Ascending',
                      transform=ax_sim[0, 0].transAxes, rotation=35)
    ax_sim[0, 0].text(0.75, 0.07, 'Descending',
                      transform=ax_sim[0, 0].transAxes, rotation=-34)
    colors = ['peru', 'cadetblue']  # bistable, monostable
    labels = ['Bistable', 'Monostable']
    lsts = ['solid', '--']
    choice_model_all = np.zeros((nFrame, n_trials, 2))  # Bistable, Monostable
    fitted_params_noise = [0.3, 0.1, 0.3, 0.1, 0.04, 0.5]  # sigma != 0
    for i_p, ps in enumerate(example_shuffles):
        j_eff = ((1-ps)*fitted_params_noise[0] + fitted_params_noise[1])*n
        params = fitted_params_noise[1:].copy()
        params[0] = j_eff
        for trial in range(n_trials):
            choice, x = simulator_5_params(params=params, freq=2, nFrame=nFrame,
                                           fps=fps, return_choice=True, ini_cond_convergence=50,
                                           tau=0.05)
            choice_model_all[:, trial, i_p] = choice
    for i_p, ps in enumerate(example_shuffles):
        for i_fr, fr in enumerate(example_freqs):
            t = np.arange(0, 26, 1/fps)
            stim = sawtooth(2 * np.pi * abs(fr)/2 * t/26, 0.5)*2*np.sign(fr)
            line = colored_line(t, stim, stim, ax_sim[0, i_fr],
                                linewidth=4, cmap=colormap,
                                norm=plt.Normalize(vmin=-2, vmax=2))
            line = colored_line(t, stim, stim, ax_dat[0, i_fr],
                                linewidth=4, cmap=colormap,
                                norm=plt.Normalize(vmin=-2, vmax=2))
            choice_dat = df_subject.loc[(df_subject.pShuffle == ps)&(df_subject.freq == fr)&(df_subject.initial_side==1)]
            unique_tr_index = choice_dat.trial_index.unique()[idx_trial]
            plot_kws = {'linewidth': 4, 'color': colors[i_p]}
            plot_example_pupil(ax_pupil[i_p],
                               data_folder=DATA_FOLDER,
                               sub=simulated_subject, trial_idx=unique_tr_index,
                               **plot_kws)
            choices = choice_dat.loc[choice_dat.trial_index == unique_tr_index]
            t_choices = get_response_array(choices)
            j_eff = ((1-ps)*fitted_params[0] + fitted_params[1])*n
            params = fitted_params[1:].copy()
            params[0] = j_eff
            choice, x = simulator_5_params(params=params, freq=fr, nFrame=nFrame_sim,
                                           fps=fps_sim, return_choice=True, ini_cond_convergence=20,
                                           tau=0.05)

            ax_sim[i_p+1, i_fr].axhline(0.5, color='gray', linestyle=':', alpha=0.6,
                                        linewidth=2)
            for cross in crossings[i_fr]:
                ax_sim[i_p+1, i_fr].axvline(cross, color='gray', linestyle='--', alpha=0.3)
                ax_sim[0, i_fr].axvline(cross, color='gray', linestyle='--', alpha=0.3)
                ax_dat[i_p+1, i_fr].axvline(cross, color='gray', linestyle='--', alpha=0.3)
                ax_dat[0, i_fr].axvline(cross, color='gray', linestyle='--', alpha=0.3)
                ax_dat2[i_p].axvline(cross, color='gray', linestyle='--', alpha=0.3)
                ax_pupil[i_p].axvline(cross, color='gray', linestyle='--', alpha=0.3)
            choice[choice == 1] = 1
            choice[choice == -1] = 0.
            time_sim = np.arange(nFrame_sim)/fps_sim
            if plot_response:
                ax_sim[i_p+1, i_fr].plot(time_sim, choice, color='k', linewidth=3,
                                           label='choice')
            x = np.clip(x, 0, 1)
            line = colored_line(time_sim, x, x, ax_sim[i_p+1, i_fr],
                                linewidth=4, cmap=colormap,
                                norm=plt.Normalize(vmin=0, vmax=1),
                                label='q')
            ax_sim[i_p+1, i_fr].set_ylim(-0.05, 1.02)
            ax_sim[i_p+1, i_fr].set_yticks([0., 0.5, 1.])
            ax_dat[i_p+1, i_fr].set_ylim(1-0.05, 1+1.05)
            ax_dat[i_p+1, i_fr].plot(t, t_choices, color='k', linewidth=3)
            ax_dat[i_p+1, i_fr].set_xlim(-0.1, 26.1)
            ax_sim[0, i_fr].set_yticks([-2, 0, 2], ['L', '0', 'R'])
            ax_sim[0, i_fr].set_ylim(-2.05, 2.05)
            ax_dat[0, i_fr].set_ylim(-2.05, 2.05)
            t_choices = t_choices.astype(float)
            t_choices[t_choices < 1] = np.nan
            ax_dat2[i_p].plot(t, t_choices, color=colors[i_p], linewidth=5,
                              label=labels[i_p], linestyle=lsts[i_p])
        # ax_dat[0, i_fr].legend(frameon=False, title='Regime')
    for a2 in ax_dat2:
        a2.spines['right'].set_visible(False)
        a2.spines['top'].set_visible(False)
        a2.spines['left'].set_visible(False)
        a2.set_yticks([])
        a2.set_ylim(0.95, 2.05)
    for a2 in ax_pupil:
        a2.spines['right'].set_visible(False)
        a2.spines['top'].set_visible(False)
        a2.spines['left'].set_visible(False)
        a2.set_yticks([])
    ax_dat2[0].set_xticks([])
    j_eff_list = [0.4, 0.1]
    b_list =  [-0.15, 0.15, 0.5, 0.2, 0.0, -0.2, -1]
    ini_points = [0, 0, 1] # [1, 1, 1, 0]
    offsets = [1e-2, 1e-2, 2e-2]
    pos_y = [0.65, 0.95, 1.1]
    pos_x = [0.06, 0.225, 0.41, 0.425, 0.54, 0.67, 0.75, 0.9]
    for i_p in range(2):
        j_eff = j_eff_list[i_p]
        pos_idx = 0
        for offset, ini_cond, b in zip(offsets, ini_points, b_list):
            val = 1 if (pos_idx < 2 or pos_idx > 4) else 0.7
            pos = [pos_x[pos_idx], pos_y[pos_idx], 0.18, 0.51]  # +0.23*((pos_idx % 2) == 0)*1.
            sub_ax =  ax_sim[i_p+1, 0].inset_axes(
                        pos, transform=ax_sim[i_p+1, 0].transAxes)
            if pos_idx == 2:
                sub_ax2 =  ax_sim[i_p+1, 0].inset_axes(
                            [0.8, 1.2, 0.2, 0.54], transform=ax_sim[i_p+1, 0].transAxes)
                sub_ax2.set_ylabel(r'p(   )')
                sub_ax2.set_xlabel(r's(t)')
                stim = sawtooth(2 * np.pi * abs(2)/2 * t/26, 0.5)*2*np.sign(2)
                sub_ax2.plot(stim[50:][::10], np.mean(choice_model_all, axis=1)[50:, i_p][::10], linewidth=3,
                             color=colors[i_p])
                sub_ax2.spines['right'].set_visible(False)
                sub_ax2.spines['top'].set_visible(False)
                sub_ax2.set_yticks([]); sub_ax2.set_xticks([])
                sub_ax2.axvline(0., color='gray', linestyle=':', alpha=0.6)
            sub_ax.axis('off')
            q = np.arange(-0.2, 1.2, 1e-3)
            v_xj = potential_mf(q, j_eff, b, n=n)
            sub_ax.set_ylim(np.min(v_xj)-2*offset, np.max(v_xj))
            sub_ax.plot(q, v_xj, color='k', linewidth=2)
            fun_to_minimize = lambda q: sigmoid(2*n*j_eff*(2*q-1)+2*b)-q
            val = fsolve(fun_to_minimize, ini_cond)
            v_xj = potential_mf(val, j_eff, b, n=n)
            sub_ax.axvline(0.5, color='gray', linestyle=':', alpha=0.6)
            sub_ax.plot(val, v_xj+offset, color=COLORMAP(val), marker='o',
                        markersize=7)
            pos_idx += 1
    ax_sim = ax_sim.flatten()
    # Use a color emoji font. Example for Linux:
    
    y_labels_sim = ['Depth cue, c(t)', '',
                    'q(    in front)', '',
                    'q(    in front)', '',
                    ]
    y_labels_dat = ['Depth cue, c(t)', '',
                    'Response \nBistable regime', '',
                    '', 'Response \nMonostable regime',
                    ]
    if nfreq == 1:
        y_labels_sim = y_labels_sim[::2]
        y_labels_dat = y_labels_dat[::2]

    for i_a, a in enumerate(ax_sim):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylabel(y_labels_sim[i_a])
    for i_a, a in enumerate(ax_sim[1:]):
        a.set_xlim(0., 27)
    ax_dat = ax_dat.flatten()
    for i_a, a in enumerate(ax_dat):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylabel(y_labels_dat[i_a])
    # ax_sim[0].set_title('1 cycle', fontsize=15)
    if nfreq > 1:
        ax_sim[1].set_title('2 cycles', fontsize=15)
        ax_sim[-2].set_xlabel('Time (s)')
        ax_dat[1].set_title('2 cycles', fontsize=15)
        ax_dat[-2].set_xlabel('Time (s)')
    ax_sim[-1].set_xlabel('Time (s)')
    ax_dat[0].set_title('1 cycle', fontsize=15)
    ax_dat[-1].set_xlabel('Time (s)')
    ax_sim[-1].set_xticks([])
    fig_sim.tight_layout()
    fig_dat.tight_layout()
    fig_sim.savefig(SV_FOLDER + 'hysteresis_cartoon_evolution.png', dpi=200, bbox_inches='tight')
    fig_sim.savefig(SV_FOLDER + 'hysteresis_cartoon_evolution.pdf', dpi=200, bbox_inches='tight')
    fig_sim.savefig(SV_FOLDER + 'hysteresis_cartoon_evolution.svg', dpi=200, bbox_inches='tight')
    fig_dat.savefig(SV_FOLDER + 'hysteresis_cartoon_evolution_responses_data.png', dpi=200, bbox_inches='tight')
    fig_dat.savefig(SV_FOLDER + 'hysteresis_cartoon_evolution_responses_data.pdf', dpi=200, bbox_inches='tight')
    fig_dat2.savefig(SV_FOLDER + 'example_hysteresis_responses_data.png', dpi=200, bbox_inches='tight')
    fig_dat2.savefig(SV_FOLDER + 'example_hysteresis_responses_data.pdf', dpi=200, bbox_inches='tight')


def plot_dist_metrics(n_simuls_network=100000):
    
    nllhj0, aicj0, bicj0 = get_log_likelihood_all_data(data_folder=DATA_FOLDER,
                                                       sv_folder=SV_FOLDER, use_j0=True,
                                                       n_simuls_network=n_simuls_network, ntraining=8,
                                                       fps=60, tFrame=26)
    nllh_no_j0, aic_no_j0, bic_no_j0 = get_log_likelihood_all_data(data_folder=DATA_FOLDER,
                                                                   sv_folder=SV_FOLDER, use_j0=False,
                                                                   n_simuls_network=n_simuls_network, ntraining=8,
                                                                   fps=60, tFrame=26)
    fig, ax = plt.subplots(ncols=3, figsize=(10, 3))
    df_nllh = pd.DataFrame({'NLLH, J0': nllh_no_j0-nllhj0})
    df_aic = pd.DataFrame({'AIC, J0': aic_no_j0-aicj0})
    df_bic = pd.DataFrame({'BIC, J0': bic_no_j0-bicj0})
    sns.violinplot(df_nllh, ax=ax[0], inner=None, fill=False, edgecolor='k')
    sns.swarmplot(df_nllh, ax=ax[0], color='k')
    sns.violinplot(df_aic, ax=ax[1], inner=None, fill=False, edgecolor='k')
    sns.swarmplot(df_aic, ax=ax[1], color='k')
    sns.violinplot(df_bic, ax=ax[2], inner=None, fill=False, edgecolor='k')
    sns.swarmplot(df_bic, ax=ax[2], color='k')
    fig.tight_layout()


def get_rt_distro_and_incorrect_resps(data_folder=DATA_FOLDER,
                                      ntraining=8, coupling_levels=[0, 0.3, 1]):
    df = load_data(data_folder, n_participants='all', preprocess_data=False)
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()
    accuracies_1st = []
    all_rts = []
    fig2, ax2 = plt.subplots(ncols=4, nrows=int(np.ceil(len(subjects)/4)), figsize=(18, 20))
    fig3, ax3 = plt.subplots(ncols=4, nrows=int(np.ceil(len(subjects)/4)), figsize=(18, 20))
    ax2 = ax2.flatten(); ax3 = ax3.flatten()
    for i_s, subject in enumerate(subjects):
        df_filt = df.loc[df.subject == subject]
        trial_index = df_filt.trial_index.unique()
        reac_times_subject = []
        choices_subject = []
        response_0_times = np.array(())
        for ti in trial_index:
            df_ti = df_filt.loc[df_filt.trial_index == ti]
            df_resp_0 = df_ti.loc[df_ti.response == 0, ['keypress_seconds_offset', 'keypress_seconds_onset']]
            diff = (df_resp_0.keypress_seconds_offset-df_resp_0.keypress_seconds_onset).values[1:]
            response_0_times = np.concatenate((response_0_times, diff))
            rt = df_ti.keypress_seconds_offset.values[0]
            choice = df_ti.response.values[1]
            reac_times_subject.append(rt); choices_subject.append((choice-1)*2-1)
        correct_1st_choice = np.array(choices_subject) == -df_filt.groupby('trial_index')['initial_side'].min().values
        accuracy_1st_choice = np.mean(correct_1st_choice)
        accuracies_1st.append(accuracy_1st_choice)
        signed_rt = np.array(reac_times_subject)*(-1)**(~correct_1st_choice)
        sns.histplot(np.abs(signed_rt), color='k', linewidth=3, ax=ax2[i_s], bins=20)
        sns.histplot(np.array(response_0_times).flatten(), color='k', linewidth=3, ax=ax3[i_s], bins=20)
        all_rts.append(signed_rt)
        if i_s >= (int(np.ceil(len(subjects)/4))-1)*4:
            ax2[i_s].set_xlabel('RT (s)')
            ax3[i_s].set_xlabel('Time(ch=0)')
    fig2.tight_layout()
    fig3.tight_layout()
    fig, ax = plt.subplots(1)
    all_rts = np.array(all_rts)
    sns.kdeplot(all_rts[(all_rts > 0)*(all_rts < 5)], ax=ax, color='g', label='Correct')
    sns.kdeplot(-all_rts[(all_rts < 0)*(np.abs(all_rts) < 5)], ax=ax, color='r', label='Incorrect')
    ax.legend(frameon=False); ax.set_xlabel('RT (s)')
    ax.set_xlim(-0.5, 5)
    plt.figure()
    sns.histplot(accuracies_1st, color='forestgreen', bins=20)
    plt.xlabel('Accuracy first choice')


def plot_sequential_effects(data_folder=DATA_FOLDER, ntraining=8):
    df = load_data(data_folder=data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()
    df_noisy = load_data(data_folder + '/noisy/', n_participants='all')
    maxes = np.max([df.trial_index.max(), df_noisy.trial_index.max()])
    mins = np.min([df.trial_index.min(), df_noisy.trial_index.min()])
    trial_indices = np.arange(mins, maxes)
    sequential_effects_all = np.zeros((2, 2, len(subjects)))
    sequential_effects_all_counts = np.zeros((2, 2, len(subjects)))
    for i_s, subject in enumerate(subjects):
        df_filt = df.loc[df.subject == subject]
        df_filt = df_filt.loc[df_filt.response != 0]
        trial_indices_hyst = df_filt.trial_index.unique()
        df_noisy_filt = df_noisy.loc[df_noisy.subject == subject]
        trial_indices_noise = df_noisy_filt.trial_index.unique()
        sequential_effects_subject = np.zeros((2, 2))
        sequential_effects_subject_counts = np.zeros((2, 2))
        for ti in trial_indices[1:]:
            if ti in trial_indices_hyst:
                first_response = df_filt.loc[df_filt.trial_index == ti, 'response'].values[0]
                current = 1
            if ti in trial_indices_noise:
                responses = df_noisy_filt.loc[df_noisy_filt.trial_index == ti, 'responses'].values
                first_response = responses[responses != 0][0]
                current = 0
            if ti-1 in trial_indices_hyst:
                last_response = df_filt.loc[df_filt.trial_index == ti-1, 'response'].values[-1]
                last = 1
            if ti-1 in trial_indices_noise:
                last_responses = df_noisy_filt.loc[df_noisy_filt.trial_index == ti-1, 'responses'].values
                last_response = last_responses[last_responses != 0][-1]
                last = 0
            sequential_effects_subject[last, current] += (first_response == last_response)*1.
            sequential_effects_subject_counts[last, current] += 1
        sequential_effects_all[:, :, i_s] = sequential_effects_subject/sequential_effects_subject_counts
        sequential_effects_all_counts[:, :, i_s] = sequential_effects_subject_counts
    fig, ax = plt.subplots(1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axhline(0.5, color='gray', linestyle='--')
    seq_all = []
    mean_counts = np.round(np.mean(sequential_effects_all_counts, axis=-1), 1) # / (len(trial_indices)-1), 2)
    i = 0
    for last in range(2):
        for current in range(2):
            sequential_effs = sequential_effects_all[last, current]
            seq_all.append(sequential_effs)
            ax.text(i-0.45, 1.1, r'$\hat{N}$ ='+f'{mean_counts[last, current]}')
            i += 1
    sns.violinplot(seq_all, fill=False, inner=None,
                   edgecolor='k', ax=ax, cut=0)
    sns.swarmplot(seq_all, color='k', ax=ax)
    ax.set_xticks([0, 1, 2, 3], ['N-->N', 'H-->N', 'N-->H', 'H-->H'])
    ax.set_ylabel('p(ch_last = ch_current)')
    ax.set_ylim(-0.1, 1.1)
    fig.tight_layout()
    # im = ax.imshow(np.nanmean(sequential_effects_all, axis=-1), cmap='bwr',
    #                vmin=0.3, vmax=0.7)
    # ax.set_xticks([0, 1], ['Hyst.', 'Noise'])
    # ax.set_yticks([0, 1], ['Hyst.', 'Noise'])
    # ax.set_ylabel('Last trial'); ax.set_xlabel('Current trial')
    # plt.colorbar(im, ax=ax, label='p(last = same response)', shrink=0.6, aspect=10)


def plot_rt_distros_simple(J1=0.3, J0=0.1, B=0.4, THETA=0.1, SIGMA=0.1):
    m_sim = model_known_params_pyddm(J1=J1, J0=J0, B=B, SIGMA=SIGMA, THETA=THETA)
    fig = plt.figure(figsize=(12, 2))
    ax1=plt.subplot(1, 4, 1)
    ax2=plt.subplot(1, 4, 2)
    ax3=plt.subplot(1, 4, 3)
    ax4=plt.subplot(1, 4, 4)
    titles = ['L to L', 'L to R', 'R to L' , 'R to R']
    for phase in [0, 6.5, 13]:
        sol1=m_sim.solve(conditions={"freq": 2, "phase_ini": phase, "prev_choice": 1, "pshuffle": 0})
        sol0=m_sim.solve(conditions={"freq": 2, "phase_ini": phase, "prev_choice": -1, "pshuffle": 0})
        ax1.plot(sol0.pdf(0), label=phase)
        ax2.plot(sol0.pdf(1), label=phase)
        ax3.plot(sol1.pdf(0), label=phase)
        ax4.plot(sol1.pdf(1), label=phase)
    for i_a, ax in enumerate([ax1, ax2, ax3, ax4]):
        ax.set_title(titles[i_a])
    ax1.legend(title='Phase')
    ax1.set_ylabel('RT density')
    fig.tight_layout()


def plot_simulate_hysteresis_subject(data_folder=DATA_FOLDER, subject_name=None,
                                     ntraining=8, n=4, window_conv=None, fps=60,
                                     ntrials=72, shift_ndt=False, simulate=False):
    np.random.seed(50)  # 24, 42, 13, 1234, 11, **50**, 51, 100,   10with1000
    ndt = np.abs(np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    print(len(pars), ' fitted subjects')
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = [[n*params[0], n*params[1], params[2], params[4], params[3], ndt] for params in fitted_params_all]   # np.random.normal(ndt, 0.05)
    # print('J1, J0, B1, Sigma, Threshold')
    simulated_subjects(data_folder=DATA_FOLDER, tFrame=26, fps=fps,
                       sv_folder=SV_FOLDER, ntraining=ntraining, ntrials=ntrials,
                       plot=True, simulate=simulate, use_j0=True, subjects=None,
                       fitted_params_all=fitted_params_all, window_conv=window_conv,
                       shift_ndt=shift_ndt)


def plot_simulated_kernel_per_subject(data_folder=DATA_FOLDER,
                                      shuffle_vals=[1., 0.7, 0.],
                                      steps_back=150, steps_front=20, fps=60):
    df = load_data(data_folder + '/noisy/', n_participants='all')
    subjects = df.subject.unique()
    fig2, ax2 = plt.subplots(ncols=4, nrows=int(np.ceil(len(subjects)/4)), figsize=(18, 20))
    ax2 = ax2.flatten()
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    x_plot = np.arange(-steps_back, steps_front, 1)/fps
    for i_sub, subject in enumerate(subjects):
        kernel = np.load(SV_FOLDER + f'/simulated_kernels_subjects/sim_kernel_{subject}.npy')
        errors = np.load(SV_FOLDER + f'/simulated_kernels_subjects/sim_kernel_{subject}_error.npy')
        ax2[i_sub].axhline(0, color='k', alpha=0.3, linestyle='--')
        ax2[i_sub].axvline(0, color='k', alpha=0.3, linestyle='--')
        ax2[i_sub].spines['right'].set_visible(False); ax2[i_sub].spines['top'].set_visible(False)
        ax2[i_sub].set_ylim(-0.3, 0.5)
        for i_sh in range(len(shuffle_vals)):
            y_plot = kernel[i_sh, :, i_sub]
            err_plot = errors[i_sh, :, i_sub]
            ax2[i_sub].plot(x_plot, y_plot, color=colormap[i_sh],
                            linewidth=3, label=shuffle_vals[i_sh])
            ax2[i_sub].fill_between(x_plot, y_plot-err_plot, y_plot+err_plot, color=colormap[i_sh],
                                    alpha=0.3)
        ax2[i_sub].set_xticks([]); ax2[i_sub].set_yticks([])
    ax2[-1].spines['right'].set_visible(False); ax2[-1].spines['top'].set_visible(False)
    ax2[-1].set_ylim(-0.3, 0.5)
    ax2[-1].axhline(0, color='k', alpha=0.3, linestyle='--')
    ax2[-1].axvline(0, color='k', alpha=0.3, linestyle='--')
    fig2.tight_layout();  ax2[0].legend()
    ax2[-1].set_xlabel('Time from switch (s)'); ax2[-1].set_ylabel('Noise')
    fig2.savefig(SV_FOLDER + 'simulated_noise_kernel_all.png', dpi=200, bbox_inches='tight')
    fig2.savefig(SV_FOLDER + 'simulated_nosie_kernel_all.pdf', dpi=200, bbox_inches='tight')


def plot_eta():
    colormap = ['cadetblue', 'peru']
    linestyles = ['solid', '--']
    plt.figure()
    for i_j, J in enumerate([0.6, 1.4]):
        for i_th, theta in enumerate([0.05, 0.2]):
            t, x_t, eta_t, C = optimal_eta_time(J, theta, T=10, n_points=1000)
            plt.plot(t, eta_t, label=f'J={J}, θ={theta}, C={C:.3f}',
                     linewidth=3, color=colormap[i_j],
                     linestyle=linestyles[i_th])
    plt.xlabel('time (t)')
    plt.ylabel('η*(t)')
    plt.title('Optimal noise η*(t) along escape trajectory')
    plt.legend()
    plt.show()


def dependency_latency_amplitude_on_a(a_list=np.arange(0., 2., 1e-2)):
    # time
    t = np.arange(-10, 10, 1e-3)
    peak = []; peak_analytical = []
    latency = []
    auc_vals = []
    dt = np.diff(t)[0]
    for a in a_list:
        # optimal escape path
        x = -np.sqrt(a) / np.sqrt(1 + np.exp(-2 * a * (t+0.5)))
        # deterministic drift
        f = x*a - x**3
        # optimal noise (\eta)
        eta = -2 * f
        peak.append(np.max(eta))
        peak_analytical.append(eta[np.searchsorted(t, -0.5)])
        latency.append((np.argmax(eta)-len(t)//2)*dt)
        auc_vals.append(np.trapz(eta/np.max(eta), t, dx=dt))
    fig, ax = plt.subplots(ncols=3, figsize=(10, 3.5))
    variables = [peak, latency, auc_vals]
    labels = ['Peak', 'Latency', r'AUC (normalized $\eta(t)$)']
    for i_a, a in enumerate(ax):
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
        a.plot(a_list, variables[i_a], color='k', linewidth=4, label=r'Max($\eta(t)$)');  a.set_ylabel(labels[i_a])
        a.set_xlabel('a')
    ax[1].axhline(-0.5, color='r', alpha=0.3, linestyle='--', linewidth=3)
    ax[0].plot(a_list, peak_analytical, color='gray', alpha=0.5, linestyle='--', linewidth=4,
               label='Analytical')
    ax[0].legend(frameon=False)
    fig.tight_layout()


def plot_average_x_noise_trials(data_folder=DATA_FOLDER,
                                tFrame=26, fps=60,
                                steps_back=120, steps_front=20, avoid_first=True,
                                n=4, load_simulations=True, normalize=False,
                                sigma=None, pshuf_only=None, bis_mono=None,
                                adaptation=False,
                                k_steps=None):
    title = r'$\sigma = 0$' if sigma is not None else r'$\sigma \neq 0$'
    if fps == 60:
        nFrame = 1546
    else:
        nFrame = int(1546*fps/60)
    ndt = np.abs(np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    label = f'_sigma_{sigma}_' if sigma is not None else ''
    if adaptation:
        label += 'adaptation'
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = [[n*params[0], n*params[1], params[2], params[4], params[3], ndt] for params in fitted_params_all]
    df = load_data(data_folder + '/noisy/', n_participants='all')
    noise_signal, choice, pshuffles = simulate_noise_subjects(df, data_folder=DATA_FOLDER, n=4, nFrame=nFrame, fps=fps,
                                                              load_simulations=load_simulations,
                                                              sigma_predefined=sigma)
    x_all = np.load(SV_FOLDER + f'x_simulated_noise{label}.npy')
    internal_noise_all = np.load(SV_FOLDER + f'internal_noise_simulated_noise{label}.npy')
    subs = df.subject.unique()[:pshuffles.shape[0]]
    x_values_all_subjects = np.empty((len(subs), steps_back+steps_front))
    x_values_all_subjects[:] = np.nan
    stim_values_all_subjects = np.empty((len(subs), steps_back+steps_front))
    stim_values_all_subjects[:] = np.nan
    internal_noise_values_all_subjects = np.empty((len(subs), steps_back+steps_front))
    internal_noise_values_all_subjects[:] = np.nan
    surprise_value_all_subjects = np.empty((len(subs), steps_back+steps_front))
    surprise_value_all_subjects[:] = np.nan
    median_theta = []
    for i_sub, subject in enumerate(subs):
        df_sub = df.loc[df.subject == subject]
        trial_index = df_sub.trial_index.unique()
        x_vals_aligned_all_trials = np.empty((1, steps_back+steps_front))
        x_vals_aligned_all_trials[:] = np.nan
        stim_vals_aligned_all_trials = np.empty((1, steps_back+steps_front))
        stim_vals_aligned_all_trials[:] = np.nan
        internal_noise_vals_aligned_all_trials = np.empty((1, steps_back+steps_front))
        internal_noise_vals_aligned_all_trials[:] = np.nan
        surprise_value_aligned_all_trials = np.empty((1, steps_back+steps_front))
        surprise_value_aligned_all_trials[:] = np.nan
        for i_trial, trial in enumerate(trial_index):
            if pshuf_only is not None:
                if pshuffles[i_sub, i_trial] != pshuf_only:
                    continue
            if bis_mono is not None:
                psh = pshuffles[i_sub, i_trial]
                j0 = fitted_params_all[i_sub][1]
                j1 = fitted_params_all[i_sub][0]
                jeff = (1-psh)*j1+j0
                if jeff < 1 and bis_mono == 'Bistable':
                    continue
                if jeff > 1 and bis_mono == 'Monostable':
                    continue
                lower_bound = 0.5-fitted_params_all[i_sub][3]
                upper_bound = 0.5+fitted_params_all[i_sub][3]
                median_theta.append(fitted_params_all[i_sub][3])
            values_x = x_all[i_sub, i_trial]
            chi = noise_signal[i_sub, i_trial]
            responses = choice[i_sub, i_trial]
            internal_noise = internal_noise_all[i_sub, i_trial]
            orders = rle(responses)
            if avoid_first:
                idx_1 = orders[1][1:][orders[2][1:] == 1]
                idx_0 = orders[1][1:][orders[2][1:] == -1]
            else:
                idx_1 = orders[1][orders[2] == 1]
                idx_0 = orders[1][orders[2] == -1]
            idx_1 = idx_1[(idx_1 > steps_back) & (idx_1 < (len(responses))-steps_front)]
            idx_0 = idx_0[(idx_0 > steps_back) & (idx_0 < (len(responses))-steps_front)]
            # original order
            x_vals_aligned = np.empty((len(idx_1)+len(idx_0), steps_back+steps_front))
            x_vals_aligned[:] = np.nan
            stim_vals_aligned = np.empty((len(idx_1)+len(idx_0), steps_back+steps_front))
            stim_vals_aligned[:] = np.nan
            internal_noise_vals_aligned = np.empty((len(idx_1)+len(idx_0), steps_back+steps_front))
            internal_noise_vals_aligned[:] = np.nan
            surprise_value_aligned = np.empty((len(idx_1)+len(idx_0), steps_back+steps_front))
            surprise_value_aligned[:] = np.nan
            parameters_sub = fitted_params_all[i_sub]
            sigma_param = parameters_sub[4]
            k = int(fps*0.25) if k_steps is None else k_steps
            for i, idx in enumerate(idx_1):
                conf_i = values_x[idx - steps_back:idx+steps_front]
                x_vals_aligned[i, :] = conf_i
                stim_vals_aligned[i, :] = chi[idx - steps_back:idx+steps_front]
                internal_noise_vals_aligned[i, :] = internal_noise[idx - steps_back:idx+steps_front]*sigma_param
                dxdt_full = np.full_like(conf_i, np.nan)
                min_dist_vals = np.min(np.row_stack([(conf_i-upper_bound)**2, (conf_i-lower_bound)**2]), axis=0)
                q = (conf_i - lower_bound) / (upper_bound - lower_bound + 1e-4)
                dxdt_full[k:] = (q[k:]-q[:-k])/(k/fps)
                surprise_value_aligned[i, :] = dxdt_full
            for i, idx in enumerate(idx_0):
                conf_i = 1-values_x[idx - steps_back:idx+steps_front]
                x_vals_aligned[i+len(idx_1), :] = conf_i
                stim_vals_aligned[i+len(idx_1), :] = chi[idx - steps_back:idx+steps_front]*-1
                internal_noise_vals_aligned[i+len(idx_1), :] = internal_noise[idx - steps_back:idx+steps_front]*-1*sigma_param
                dxdt_full = np.full_like(conf_i, np.nan)
                min_dist_vals = np.min(np.row_stack([(conf_i-upper_bound)**2, (conf_i-lower_bound)**2]), axis=0)
                q = (conf_i - lower_bound) / (upper_bound - lower_bound + 1e-4)
                dxdt_full[k:] = (q[k:]-q[:-k])/(k/fps)
                surprise_value_aligned[i+len(idx_1), :] = dxdt_full

            x_vals_aligned_all_trials = np.row_stack((x_vals_aligned_all_trials, x_vals_aligned))
            surprise_value_aligned_all_trials = np.row_stack((surprise_value_aligned_all_trials, surprise_value_aligned))
            stim_vals_aligned_all_trials = np.row_stack((stim_vals_aligned_all_trials, stim_vals_aligned))
            internal_noise_vals_aligned_all_trials =\
                np.row_stack((internal_noise_vals_aligned_all_trials, internal_noise_vals_aligned))
            
        x_vals_aligned_all_trials = x_vals_aligned_all_trials[1:]
        x_values_all_subjects[i_sub] = np.nanmean(x_vals_aligned_all_trials, axis=0)
        stim_vals_aligned_all_trials = stim_vals_aligned_all_trials[1:]
        stim_values_all_subjects[i_sub] = np.nanmean(stim_vals_aligned_all_trials, axis=0)
        internal_noise_vals_aligned_all_trials = internal_noise_vals_aligned_all_trials[1:]
        internal_noise_values_all_subjects[i_sub] = np.nanmean(internal_noise_vals_aligned_all_trials, axis=0)
        surprise_value_aligned_all_trials = surprise_value_aligned_all_trials[1:]
        surprise_value_all_subjects[i_sub] = np.nanmean(surprise_value_aligned_all_trials, axis=0)
    if bis_mono is None:
        theta = [pars[3] for pars in fitted_params_all]
        median_theta = 0.5 + np.nanmedian(theta)*np.array([-1, 1])
    else:
        median_theta = 0.5 + np.nanmean(median_theta)*np.array([-1, 1])
    fig, ax = plt.subplots(ncols=2, nrows=4, figsize=(10, 11), sharex=True,)
                           # sharey='row')
    fig.suptitle(title, fontsize=16)
    ax = ax.flatten()
    variables = [x_values_all_subjects, stim_values_all_subjects, internal_noise_values_all_subjects,
                 surprise_value_all_subjects]
    labels = ['App. posterior q']*2 + ['Stimulus noise B(t)']*2 + ['Internal noise']*2 + [r'Surprise, $ \frac{d\hat{q}}{dt}$']*2
    var = 0
    [ax[1].axhline(val, color='r', linestyle='--', alpha=0.2, linewidth=2, zorder=1) for val in median_theta]
    ax[1].axhline(0.5, color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
    # ylims = [[0.25, 0.75], [-0.2, 0.5], [-0.5, 1.5]]
    
    func_x = lambda t, k1:  -1 / np.sqrt(1 + k1*np.exp(-2 * (t+ndt)))
    
    if normalize:
        ax[0].axhline(0., color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
    else:
        ax[0].axhline(0.5, color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
    if bis_mono is None:
        color = 'k'
        c_ind = 'firebrick'
    if bis_mono == 'Bistable':
        color = 'peru'
        c_ind = 'peru'
    if bis_mono == 'Monostable':
        color = 'cadetblue'
        c_ind = 'cadetblue'
    for i_a, a in enumerate(ax):
        # a.set_ylim(ylims[var][0], ylims[var][1])
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
        if i_a >= 2:
            a.axhline(0., color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
        a.set_xlabel('Time from switch (s)'); a.set_ylabel(labels[i_a])
        if i_a % 2 == 0:
            for i_sub, subject in enumerate(subs):
                x_plot = np.arange(-steps_back, steps_front, 1)/fps
                if i_a == 0 and normalize:
                    minvar = np.min(variables[var][i_sub])
                    maxvar = np.max(variables[var][i_sub])
                    y_plot = (variables[var][i_sub]-0.5)/(maxvar-minvar)
                else:
                    y_plot = variables[var][i_sub]
                a.plot(x_plot, y_plot, color=c_ind,
                       linewidth=2, alpha=0.1)
            continue
        y_plot = np.nanmean(variables[var], axis=0)
        y_err = np.nanstd(variables[var], axis=0)/np.sqrt(len(subs))
        a.plot(x_plot, y_plot, color=color, linewidth=4)
        a.fill_between(x_plot, y_plot-y_err, y_plot+y_err, color=color, alpha=0.2)
        var += 1
    max_val_surprise = np.nanmax(surprise_value_all_subjects, axis=1)  # max per subject
    if bis_mono == 'Monostable':
        path_max = SV_FOLDER + 'max_val_surprise_monostable.npy'
        path_full = SV_FOLDER + 'surprise_monostable.npy'
    if bis_mono == 'Bistable':
        path_max = SV_FOLDER + 'max_val_surprise_bistable.npy'
        path_full = SV_FOLDER + 'surprise_bistable.npy'
    np.save(path_max, max_val_surprise)
    np.save(path_full, surprise_value_all_subjects)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + f'simulated_variables_noise_trials{label}.png', dpi=200, bbox_inches='tight')
    fig.savefig(SV_FOLDER + f'simulated_variables_noise_trials{label}.pdf', dpi=200, bbox_inches='tight')


def plot_surprise(steps_back=240,
                  steps_front=240, fps=60,
                  negative=False):
    
    path_full_mono = SV_FOLDER + 'surprise_monostable.npy'
    path_full_bis = SV_FOLDER + 'surprise_bistable.npy'
    surprise_bis = np.load(path_full_bis)
    surprise_mono = np.load(path_full_mono)
    f, a = plt.subplots(ncols=1, figsize=(4.5, 3.5))
    variables = [surprise_mono*(-1)**negative, surprise_bis*(-1)**negative]
    x_plot = np.arange(-steps_back, steps_front, 1)/fps
    a.axvline(0, color='k', linestyle='--')
    a.axhline(0, color='k', linestyle='--')
    for var, bis_mono in enumerate(['Monostable', 'Bistable']):
        if bis_mono == 'Bistable':
            color = 'peru'
        if bis_mono == 'Monostable':
            color = 'cadetblue'
    
        # a.set_ylim(ylims[var][0], ylims[var][1])
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
        y_plot = np.nanmean(variables[var], axis=0)
        y_err = np.nanstd(variables[var], axis=0)/np.sqrt(surprise_mono.shape[0])
        a.plot(x_plot, y_plot, color=color, linewidth=4,
               label=bis_mono)
        a.fill_between(x_plot, y_plot-y_err, y_plot+y_err, color=color, alpha=0.2)
    a.set_xlim(-4.2, 4.2)
    a.set_xticks([-4, -2, 0, 2, 4])
    a.set_xlabel('Time from switch (s)')
    a.set_ylabel('Perceptual surprise')
    a.legend(frameon=False, loc='lower left' if negative else 'upper left')
    f.tight_layout()
    f.savefig(SV_FOLDER + 'surprise_aligned_switch.png', dpi=200, bbox_inches='tight')
    f.savefig(SV_FOLDER + 'surprise_aligned_switch.svg', dpi=200, bbox_inches='tight')


def plot_kernels_predicted_amplitude(steps_back=150, steps_front=10, fps=60,
                                     cumsum=False, npercentiles=4, sim_predict_dat=False):
    x_plot = np.arange(-steps_back, steps_front, 1)/fps
    kernels_data = np.load(DATA_FOLDER + 'all_kernels_noise_switch_aligned.npy')
    kernels_simul = np.load(DATA_FOLDER + 'simulated_all_kernels_noise_switch_aligned.npy')
    amplitude_simul = np.nanmax(kernels_simul, axis=1)
    amplitude_data = np.nanmax(kernels_data, axis=1)
    bins_perc = [i/npercentiles*100 for i in range(npercentiles+1)]
    extra_for_all_vals = np.array([-1e-5] + [0]*(len(bins_perc)-2)+ [1e-5])
    if sim_predict_dat:
        amplitude_prediction = amplitude_simul
        amplitude_to_plot = amplitude_data
        kernels_to_plot = kernels_data
        label_save = 'sim_to_dat_'
    else:
        amplitude_prediction = amplitude_simul
        amplitude_to_plot = amplitude_simul
        kernels_to_plot = kernels_simul
        label_save = 'sim_to_sim_'
    idxs = np.digitize(amplitude_prediction, np.percentile(amplitude_prediction, bins_perc)+extra_for_all_vals)-1
    fig, ax = plt.subplots(1, figsize=(4., 3.5))
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    nbins = len(np.unique(idxs))
    # colormap = pl.cm.Oranges(np.linspace(0.3, 1, nbins))
    colormap = pl.cm.binary(np.linspace(0.3, 1, nbins))
    if cumsum:
        label = r'$\sum_{k=1}^t Noise(k)$'
    else:
        label = 'Noise'
    for i in range(nbins):
        if cumsum:
            average_across_subjects = np.cumsum(np.nanmean(kernels_to_plot[idxs == i], axis=0))
            sem_across_subjects = np.nanstd(np.cumsum(kernels_to_plot[idxs == i], axis=1), axis=0) / np.sqrt(sum(idxs == i))
        else:
            average_across_subjects = np.nanmean(kernels_to_plot[idxs == i], axis=0)
            sem_across_subjects = np.nanstd(kernels_to_plot[idxs == i], axis=0) / np.sqrt(sum(idxs == i))
        ax.plot(x_plot, average_across_subjects, color=colormap[i],
                linewidth=5, label=f'{i+1}{enums(i+1)}')
        max_avg = np.nanmax(average_across_subjects)
        # ax.axhline(max_avg, color=colormap[i], linewidth=2, linestyle=':')
        ax.fill_between(x_plot, average_across_subjects-sem_across_subjects,
                        average_across_subjects+sem_across_subjects,
                        color=colormap[i], alpha=0.1)
    ax.set_xlabel('Time from switch (s)')
    ax.set_ylabel(label); ax.legend(title='Predicted\npercentile', frameon=False)
    fig.tight_layout()
    print('Saving images')
    fig.savefig(SV_FOLDER + label_save + 'kernel_noise_bf_switch_predicted_amplitude.png', dpi=200, bbox_inches='tight')
    fig.savefig(SV_FOLDER + label_save + 'kernel_noise_bf_switch_predicted_amplitude.svg', dpi=200, bbox_inches='tight')
    fig, ax = plt.subplots(ncols=1, figsize=(4, 3.5))
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    if cumsum:
        var_data = np.sum(kernels_data, axis=1)
        var_simul = np.sum(kernels_simul, axis=1)
        ax.set_xlabel('Cumulative sum data'); ax.set_ylabel('Cumulative sum simulation')
    else:
        var_data = amplitude_data
        var_simul = amplitude_simul
        ax.set_xlabel('Kernel peak, data'); ax.set_ylabel('Kernel peak, simulation')
    ax.plot(var_data, var_simul, marker='o', linestyle='', color='k')
    r, p = pearsonr(var_data, var_simul)
    ax.annotate(f'r = {r:.3f}\np={p:.3f}', xy=(.6, 0.3), xycoords=ax.transAxes)
    linreg = LinearRegression(fit_intercept=True).fit(var_data.reshape(-1, 1), var_simul.reshape(-1, 1))
    minmax_array = np.array([np.min(var_data)-0.1, np.max(var_data)+0.1]).reshape(-1, 1)
    pred_y = linreg.predict(minmax_array)
    ax.plot(minmax_array,
           pred_y, color='gray', linestyle='--', alpha=0.4, linewidth=3)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + label_save + 'kernel_peak_comparison.png', dpi=200, bbox_inches='tight')
    fig.savefig(SV_FOLDER + label_save + 'kernel_peak_comparison.svg', dpi=200, bbox_inches='tight')


def lmm_dominance_regime(unique_shuffle=[1., 0.7, 0.], n=4, simulations=False):
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    j1s = np.array([np.load(par)[0] for par in pars])  # /sigmas
    j0s = np.array([np.load(par)[1] for par in pars])  # /sigmas
    if simulations:
        mean_number_switchs_coupling = np.load(DATA_FOLDER + 'simulated_mean_number_switches_per_subject.npy')
        dh = 0.6
    else:
        mean_number_switchs_coupling = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
        dh = .5
    print(len(pars), ' fitted subjects')
    fitted_params_all = [np.load(par) for par in pars]
    j_coupling_0 = (j0s)
    j_coupling_03 = (j1s*0.3+j0s)
    j_coupling_1 = (j1s+j0s)
    print(len(np.where(np.sign(j_coupling_0-1/n) != np.sign(j_coupling_1-1/n))[0]))
    all_coups = np.row_stack((j_coupling_0, j_coupling_03, j_coupling_1))
    fitted_params_all = np.array([[n*params[0], n*params[1], params[2], params[4], params[3]] for params in fitted_params_all])
    unique_shuffle = np.array(unique_shuffle)
    fitted_subs = fitted_params_all.shape[0]
    jeffs = np.zeros((3, fitted_subs))
    bistable_stim_2_dominance = []
    monostable_stim_2_dominance = []
    subjects = []
    conditions = []
    for i in range(fitted_subs):
        jeffs[:, i] = (fitted_params_all[i][0]*(1-unique_shuffle)+fitted_params_all[i][1])
        for k in range(3):
            subjects.append(i)
            conditions.append(unique_shuffle[k])
    jeffs_mask = jeffs >= 1  # 1/n
    bistable_stim_2_dominance = mean_number_switchs_coupling[:, :fitted_subs].flatten()[jeffs_mask.flatten()]
    monostable_stim_2_dominance = mean_number_switchs_coupling[:, :fitted_subs].flatten()[~jeffs_mask.flatten()]
    dominances = mean_number_switchs_coupling.flatten()
    regimes = jeffs_mask.flatten()*1.
    df = pd.DataFrame({"x": regimes,
                       "y": dominances,
                       "subject": subjects})
    re_formula = "1"
    model = smf.mixedlm("y ~ x", df, groups=df["subject"],
                        re_formula=re_formula)
    result = model.fit()
    fe = result.fe_params
    re = result.random_effects
    # get intercepts/slopes per subject
    intercepts = [fe["Intercept"] + eff.get("Group", 0) for subj, eff in re.items()]
    slopes = [fe["x"] + eff.get("x", 0) for subj, eff in re.items()]
    print(result.summary())


def plot_dominance_bis_mono_data_model(n=4, ax=None,
                                       estimator='mean'):
    unique_shuffle=[1., 0.7, 0.]
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    mean_dominance_shuffle_data = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
    mean_dominance_shuffle_model = np.load(DATA_FOLDER + 'simulated_mean_number_switches_per_subject.npy')
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = np.array([[n*params[0], n*params[1], params[2], params[4], params[3]] for params in fitted_params_all])
    unique_shuffle = np.array(unique_shuffle)
    fitted_subs = fitted_params_all.shape[0]
    jeffs = np.zeros((3, fitted_subs))
    bistable_stim_dominance_data = []
    monostable_stim_dominance_data = []
    bistable_stim_dominance_model = []
    monostable_stim_dominance_model = []
    for i in range(fitted_subs):
        jeffs[:, i] = (fitted_params_all[i][0]*(1-unique_shuffle)+fitted_params_all[i][1])
        dom_bis_data = mean_dominance_shuffle_data[:, i][jeffs[:, i] >= 1]
        dom_mono_data = mean_dominance_shuffle_data[:, i][jeffs[:, i] < 1]
        dom_bis_model = mean_dominance_shuffle_model[:, i][jeffs[:, i] >= 1]
        dom_mono_model = mean_dominance_shuffle_model[:, i][jeffs[:, i] < 1]
        if not sum(np.isnan(dom_bis_data)):
            bistable_stim_dominance_data.extend(dom_bis_data)
            bistable_stim_dominance_model.extend(dom_bis_model)
        if not sum(np.isnan(dom_mono_data)):
            monostable_stim_dominance_data.extend(dom_mono_data)
            monostable_stim_dominance_model.extend(dom_mono_model)
    variables_dominance_data = [monostable_stim_dominance_data, bistable_stim_dominance_data]
    variables_dominance_model = [monostable_stim_dominance_model, bistable_stim_dominance_model]
    if ax is None:
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(6, 3.5), sharey=True)
        ax = ax.flatten()
        saveflag = True
    else:
        saveflag = False
    variables = [variables_dominance_data, variables_dominance_model]
    colormap = ['cadetblue', 'peru']
    for i_var, var in enumerate(variables):
        ax[i_var].spines['right'].set_visible(False); ax[i_var].spines['top'].set_visible(False)
        sns.barplot(var, palette=colormap, errorbar='se', ax=ax[i_var], estimator=estimator)
        sns.swarmplot(
                    data=var,
                    color="black",        # point fill
                    edgecolor="white",    # contrast on dark bars
                    linewidth=0.5,
                    size=3,
                    ax=ax[i_var],
                    zorder=10             # ensures points are on top
                    )
    if estimator == 'mean':
        dhs = [0.35, 0.35]
    if estimator == 'median':
        dhs = [0.35, 0.35]
    for i_a, a in enumerate(ax):
        if estimator == 'median':
            heights = np.nanmax(variables[i_a], axis=0)
        else:
            heights = np.nanmax(variables[i_a], axis=0)
        bars = np.arange(2)
        pv_sh01 = scipy.stats.mannwhitneyu(variables[i_a][0], variables[i_a][1]).pvalue
        barplot_annotate_brackets(0, 1, pv_sh01, bars, heights, yerr=None, dh=dhs[i_a], barh=.01, fs=10,
                                  maxasterix=3, ax=a)
        a.set_xticks([0, 1], ['Monostable', 'Bistable'])
    ax[0].set_title('Human', fontsize=15)
    ax[1].set_title('Model', fontsize=15)
    ax[0].set_ylabel('Dominance (s)')
    fig.savefig(SV_FOLDER + 'model_data_dominance_regime.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'model_data_dominance_regime.svg', dpi=400, bbox_inches='tight')


def plot_dominance_bis_mono(unique_shuffle=[1., 0.7, 0.], n=4, simulations=False):
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    j1s = np.array([np.load(par)[0] for par in pars])  # /sigmas
    j0s = np.array([np.load(par)[1] for par in pars])  # /sigmas
    if simulations:
        mean_number_switchs_coupling = np.load(DATA_FOLDER + 'simulated_mean_number_switches_per_subject.npy')
        dh = 0.45
    else:
        mean_number_switchs_coupling = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
        dh = .3
    print(len(pars), ' fitted subjects')
    fitted_params_all = [np.load(par) for par in pars]
    j_coupling_0 = (j0s)
    j_coupling_03 = (j1s*0.3+j0s)
    j_coupling_1 = (j1s+j0s)
    print(len(np.where(np.sign(j_coupling_0-1/n) != np.sign(j_coupling_1-1/n))[0]))
    all_coups = np.row_stack((j_coupling_0, j_coupling_03, j_coupling_1))
    # fig, ax = plt.subplots(1, figsize=(6, 4.5))
    # idxs = np.sign(j_coupling_0-1/n) != np.sign(j_coupling_1-1/n)
    # mean_number_switchs_coupling = mean_number_switchs_coupling[:, idxs]
    fitted_params_all = np.array([[n*params[0], n*params[1], params[2], params[4], params[3]] for params in fitted_params_all])
    unique_shuffle = np.array(unique_shuffle)
    fitted_subs = fitted_params_all.shape[0]
    jeffs = np.zeros((3, fitted_subs))
    bistable_stim_2_dominance = []
    monostable_stim_2_dominance = []
    for i in range(fitted_subs):
        jeffs[:, i] = (fitted_params_all[i][0]*(1-unique_shuffle)+fitted_params_all[i][1])
        dom_bis = np.nanmean(mean_number_switchs_coupling[:, i][jeffs[:, i] >= 1])
        dom_mono = np.nanmean(mean_number_switchs_coupling[:, i][jeffs[:, i] < 1])
        if not np.isnan(dom_bis):
            bistable_stim_2_dominance.append(dom_bis)
        if not np.isnan(dom_mono):
            monostable_stim_2_dominance.append(dom_mono)
    jeffs_mask = jeffs >= 1  # 1/n
    # mean_peak_amplitude = np.load(DATA_FOLDER + 'mean_peak_amplitude_per_subject.npy')
    # mean_peak_latency = np.load(DATA_FOLDER + 'mean_peak_latency_per_subject.npy')
    # mean_number_switchs_coupling = np.load(DATA_FOLDER + 'mean_peak_amplitude_per_subject.npy')
    # bistable_stim_2_dominance = mean_number_switchs_coupling[:, :fitted_subs].flatten()[jeffs_mask.flatten()]
    # monostable_stim_2_dominance = mean_number_switchs_coupling[:, :fitted_subs].flatten()[~jeffs_mask.flatten()]
    fig5, ax5 = plt.subplots(ncols=1, figsize=(3.5, 4))
    ax5.spines['right'].set_visible(False); ax5.spines['top'].set_visible(False)
    sns.barplot([bistable_stim_2_dominance, monostable_stim_2_dominance], palette=['peru', 'cadetblue'], ax=ax5)
    pvalue = scipy.stats.mannwhitneyu(bistable_stim_2_dominance, monostable_stim_2_dominance).pvalue
    heights = [np.nanmean([bistable_stim_2_dominance, monostable_stim_2_dominance][k]) for k in range(2)]
    barplot_annotate_brackets(0, 1, pvalue, [0, 1], heights, yerr=None, dh=dh, barh=.02, fs=10,
                              maxasterix=3, ax=ax5)
    sns.swarmplot([bistable_stim_2_dominance, monostable_stim_2_dominance], color='k', ax=ax5, size=3)
    ax5.set_xticks([0, 1], ['Bistable', 'Monostable'])
    ax5.set_ylabel('Dominance duration');
    # if simulations:
    #     ax5.set_ylim(0, 5.3)
    # else:
    #     ax5.set_ylim(0, 12.5)
    fig5.tight_layout()
    label = 'simulated_' if simulations else ''
    fig5.savefig(DATA_FOLDER + f'{label}dominance_regime.png', dpi=400, bbox_inches='tight')
    fig5.savefig(DATA_FOLDER + f'{label}dominance_regime.svg', dpi=400, bbox_inches='tight')


def plot_hysteresis_model_data():
    from matplotlib.patches import Patch
    # --- load data ---
    hyst_width_2_model = np.load(DATA_FOLDER + 'hysteresis_width_f2_sims_fitted_params.npy')
    hyst_width_4_model = np.load(DATA_FOLDER + 'hysteresis_width_f4_sims_fitted_params.npy')
    hyst_width_2_data  = np.load(DATA_FOLDER + 'hysteresis_width_freq_2.npy')
    hyst_width_4_data  = np.load(DATA_FOLDER + 'hysteresis_width_freq_4.npy')
    
    n_cond, n_subj = hyst_width_2_model.shape
    conds = ['1', '0.7', '0']
    conditions = [f'{conds[i]}' for i in range(n_cond)]
    groups = ['Data', 'Model']
    
    # Color per condition
    palette_conditions = ['lightskyblue', 'royalblue', 'midnightblue']
    
    # --- helper to build long-form DataFrame ---
    def build_df(data_array, model_array):
        values = np.concatenate([data_array.T.flatten(), model_array.T.flatten()])
        cond = np.tile(np.arange(n_cond), n_subj*2)
        cond_labels = [conditions[i] for i in cond]
        group = ['Data']*(n_cond*n_subj) + ['Model']*(n_cond*n_subj)
        return pd.DataFrame({'Condition': cond_labels, 'Group': group, 'Value': values})
    
    df_2 = build_df(hyst_width_2_data, hyst_width_2_model)
    df_4 = build_df(hyst_width_4_data, hyst_width_4_model)
    
    # --- plot ---
    fig, ax = plt.subplots(ncols=2, figsize=(7.5, 2.), sharey=False, sharex=True)
    def plot_panel(df, axis):
        g = sns.barplot(
            data=df,
            x='Condition',
            y='Value',
            hue='Group',
            errorbar='se',
            dodge=True,
            palette={'Data':'white', 'Model':'white'},  # placeholder, we'll color manually
            edgecolor='white',
            ax=axis,
            legend=False
        )
        
        # Apply condition colors and hatch
        for i, patch in enumerate(g.patches):
            cond_idx = i % n_cond  # index in 0..2
            patch.set_facecolor(palette_conditions[cond_idx])
            if i // n_cond == 1:  # second group = Model → hatch
                patch.set_hatch('//')
            else:  # Data → solid
                patch.set_hatch('')
        
    # Legend for hatch only
    hatch_handles = [
        Patch(facecolor='midnightblue', edgecolor='white', label='Data'),
        Patch(facecolor='midnightblue', edgecolor='white', hatch='//', label='Model')
    ]
    ax[0].legend(handles=hatch_handles, frameon=False, loc='upper left',
                 bbox_to_anchor=[0., 1.1])
    
    plot_panel(df_2, ax[0])
    
    plot_panel(df_4, ax[1])
    lims = [np.max(hyst_width_2_model), np.max(hyst_width_4_model)]
    for i_a, a in enumerate(ax):
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
        a.set_xlabel('p(shuffle)')
        a.set_ylim(0, lims[i_a])
    ax[0].set_ylabel('Hysteresis')
    ax[1].set_ylabel('')
    plt.tight_layout()
    fig.savefig(DATA_FOLDER + 'hysteresis_data_model_comparison.png', dpi=400, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'hysteresis_data_model_comparison.pdf', dpi=400, bbox_inches='tight')


def hyst_vs_dom_bis_mono(unique_shuffle=[1., 0.7, 0.], n=4,
                         freq=2, kde=False,
                         point_per_subject_x_shuffle=False,
                         simulated=False, zscore_vars=False, pupil=False):
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    if simulated:
        mean_dominance_shuffle = np.load(DATA_FOLDER + 'simulated_mean_number_switches_per_subject.npy')
        label = 'simulated_'
    else:
        mean_dominance_shuffle = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
        label = ''
    if freq == 4:
        if not simulated:
            file = 'hysteresis_width_freq_4.npy'
        else:
            file = 'hysteresis_width_f4_sims_fitted_params.npy'
    if freq == 2:
        if not simulated:
            file = 'hysteresis_width_freq_2.npy'
        else:
            file = 'hysteresis_width_f2_sims_fitted_params.npy'
    if pupil:
        save_path = os.path.join(DATA_FOLDER, 'aligned_eye_tracker_data','plots', 'min_pupil_noisy_trials.npy')
        var = np.load(save_path)
    else:
        var = np.load(DATA_FOLDER + file)

    if zscore_vars:
        mean_dominance_shuffle = zscore(mean_dominance_shuffle, axis=1)
        var = zscore(var, axis=1)
        lab_zscore = 'z-scored '
    else:
        lab_zscore = ''
    print(len(pars), ' fitted subjects')
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = np.array([[n*params[0], n*params[1], params[2], params[4], params[3]] for params in fitted_params_all])
    unique_shuffle = np.array(unique_shuffle)
    fitted_subs = fitted_params_all.shape[0]
    jeffs = np.zeros((3, fitted_subs))
    bistable_stim_dominance = []
    monostable_stim_dominance = []
    bistable_stim_hyst = []
    monostable_stim_hyst = []
    for i in range(fitted_subs):
        jeffs[:, i] = (fitted_params_all[i][0]*(1-unique_shuffle)+fitted_params_all[i][1])
        dom_bis = np.nanmean(mean_dominance_shuffle[:, i][jeffs[:, i] >= 1])
        dom_mono = np.nanmean(mean_dominance_shuffle[:, i][jeffs[:, i] < 1])
        hyst_bis = np.nanmean(var[:, i][jeffs[:, i] >= 1])
        hyst_mono = np.nanmean(var[:, i][jeffs[:, i] < 1])
        if not np.isnan(dom_bis):
            bistable_stim_dominance.append(dom_bis)
            bistable_stim_hyst.append(hyst_bis)
        if not np.isnan(dom_mono):
            monostable_stim_dominance.append(dom_mono)
            monostable_stim_hyst.append(hyst_mono)
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.05
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    
    # start with a square Figure
    fig = plt.figure(figsize=(5, 5))
    
    ax3 = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax3)
    ax_histy = fig.add_axes(rect_histy, sharey=ax3)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    if point_per_subject_x_shuffle:
        jeffs = jeffs.flatten() > 1
        mean_dominance_shuffle = mean_dominance_shuffle.flatten()
        var = var.flatten()
        monostable_stim_hyst = var[~jeffs]
        bistable_stim_hyst = var[jeffs]
        monostable_stim_dominance = mean_dominance_shuffle[~jeffs]
        bistable_stim_dominance = mean_dominance_shuffle[jeffs]
    alldom = np.concatenate((monostable_stim_dominance, bistable_stim_dominance))
    allhyst = np.concatenate((monostable_stim_hyst, bistable_stim_hyst))
    r, p = pearsonr(alldom, allhyst)
    ax3.annotate(f'r = {r:.2f}\np = {p:.0e}', xy=(.02, 0.85), xycoords=ax3.transAxes)
    # sns.kdeplot(x=bistable_stim_hyst, y=bistable_stim_dominance,
    #             color='peru', ax=ax3, label='Bistable')
    # sns.kdeplot(x=monostable_stim_hyst, y=monostable_stim_dominance,
    #             color='cadetblue', ax=ax3, label='Monostable')
    colormap = ['cadetblue', 'peru']; labels = ['Monostable', 'Bistable']
    
    variables_hyst = [monostable_stim_hyst, bistable_stim_hyst]
    variables_dominance = [monostable_stim_dominance, bistable_stim_dominance]
    for a in [ax_histx, ax_histy]:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
    ax_histx.spines['bottom'].set_visible(False)
    ax_histy.spines['left'].set_visible(False)
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)
    for i in range(2):
        meanx = np.mean(variables_hyst[i]); meany = np.mean(variables_dominance[i])
        errx = np.std(variables_hyst[i]);  erry = np.std(variables_dominance[i])
        ax3.errorbar(x=meanx, y=meany, xerr=errx, yerr=erry, color=colormap[i],
                     marker='o', label=labels[i], markersize=9)
        ax3.plot(variables_hyst[i], variables_dominance[i], color=colormap[i], linestyle='',
                 marker='x', alpha=0.6, markersize=6)
        if not kde:
            sns.histplot(y=variables_dominance[i], color=colormap[i], ax=ax_histy,
                        alpha=0.7, bins=np.linspace(0, 12, 20))
            sns.histplot(variables_hyst[i], color=colormap[i], ax=ax_histx,
                        alpha=0.7, bins=np.linspace(0, 3, 20))
        if kde:
            sns.kdeplot(y=variables_dominance[i], color=colormap[i], ax=ax_histy,
                        linewidth=2, bw_adjust=0.5)
            sns.kdeplot(variables_hyst[i], color=colormap[i], ax=ax_histx,
                        linewidth=2, bw_adjust=0.5)
    # ax3.set_yscale('log'); ax3.set_xscale('log')
    if not zscore_vars:
        if point_per_subject_x_shuffle:
            if not simulated:
                ax3.set_ylim(0, 15.9); ax3.set_xlim(0, 2.8)
            else:
                ax3.set_ylim(0, 6.5); ax3.set_xlim(0.3, 3.)
        else:
            ax3.set_ylim(0, 11.9); ax3.set_xlim(0, 2.9)
    if pupil:
        ax3.set_xlim(-0.9, 0.1)
    ax3.set_ylabel(lab_zscore + 'Dominance (s)')
    if pupil:
        ax3.set_xlabel(lab_zscore + 'Min. pupil');
    else:
        ax3.set_xlabel(lab_zscore + 'Hysteresis');
    ax3.legend(frameon=False, loc='lower left', bbox_to_anchor=[0.4, -0.02])
    fig.tight_layout()
    label_pup = 'pupil_' if pupil else ''
    fig.savefig(DATA_FOLDER + label + label_pup + 'dominance_vs_hysteresis_classified.png', dpi=300, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + label + label_pup +'dominance_vs_hysteresis_classified.pdf', dpi=300, bbox_inches='tight')
    # do some clustering, n_centroids=2
    # X = np.column_stack((alldom, allhyst))
    # from sklearn.cluster import KMeans
    # y_pred = KMeans(n_clusters=3, random_state=4).fit_predict(X)
    # fig, ax = plt.subplots(1)
    # ax.scatter(X[:, 0], X[:, 1], c=y_pred)


def plot_hyst_bis_mono(unique_shuffle=[1., 0.7, 0.], n=4, simulations=False,
                       freq=2):
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    if simulations:
        if freq == 2:
            hyst_width = np.load(DATA_FOLDER + 'hysteresis_width_f2_sims_fitted_params.npy')
        else:
            hyst_width = np.load(DATA_FOLDER + 'hysteresis_width_f4_sims_fitted_params.npy')
        dh = 0.45
    else:
        if freq == 2:
            hyst_width = np.load(DATA_FOLDER + 'hysteresis_width_freq_2.npy')
        else:
            hyst_width = np.load(DATA_FOLDER + 'hysteresis_width_freq_4.npy')
        dh = .3
    print(len(pars), ' fitted subjects')
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = np.array([[n*params[0], n*params[1], params[2], params[4], params[3]] for params in fitted_params_all])
    unique_shuffle = np.array(unique_shuffle)
    fitted_subs = fitted_params_all.shape[0]
    jeffs = np.zeros((3, fitted_subs))
    bistable_stim_hyst = []
    monostable_stim_hyst = []
    for i in range(fitted_subs):
        jeffs[:, i] = (fitted_params_all[i][0]*(1-unique_shuffle)+fitted_params_all[i][1])
        hyst_bis = np.nanmean(hyst_width[:, i][jeffs[:, i] >= 1])
        hyst_mono = np.nanmean(hyst_width[:, i][jeffs[:, i] < 1])
        if not np.isnan(hyst_bis):
            bistable_stim_hyst.append(hyst_bis)
        if not np.isnan(hyst_mono):
            monostable_stim_hyst.append(hyst_mono)
    fig5, ax5 = plt.subplots(ncols=1, figsize=(3.5, 4))
    ax5.spines['right'].set_visible(False); ax5.spines['top'].set_visible(False)
    sns.barplot([bistable_stim_hyst, monostable_stim_hyst], palette=['peru', 'cadetblue'], ax=ax5)
    pvalue = scipy.stats.mannwhitneyu(bistable_stim_hyst, monostable_stim_hyst).pvalue
    heights = [np.nanmean([bistable_stim_hyst, monostable_stim_hyst][k]) for k in range(2)]
    barplot_annotate_brackets(0, 1, pvalue, [0, 1], heights, yerr=None, dh=dh, barh=.02, fs=10,
                              maxasterix=3, ax=ax5)
    sns.stripplot([bistable_stim_hyst, monostable_stim_hyst], color='k', ax=ax5, size=3)
    ax5.set_xticks([0, 1], ['Bistable', 'Monostable'])
    ax5.set_ylabel('Hysteresis width');
    ax5.set_ylim(0, 2.8)
    fig5.tight_layout()


def plot_noise_variables_vs_fitted_params(n=4, variable='dominance',
                                          fitted_variable='J',
                                          full=False):
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    fitted_subs = len(pars)
    b1s = np.array([np.load(par)[2] for par in pars])
    sigmas = np.array([np.load(par)[3] for par in pars])
    thetas = np.array([np.load(par)[4] for par in pars])
    j1s = np.array([np.load(par)[0] for par in pars])  # /sigmas
    j0s = np.array([np.load(par)[1] for par in pars])  # /sigmas
    variables = {'J': j0s+j1s, 'B': b1s, 'sigma': sigmas, 'theta': thetas,
                 'J1': j1s, 'J0': j0s, 'B/sigma': b1s/sigmas,
                 'theta/sigma': thetas/sigmas}
    labels = ['J1', 'J0', 'B1', 'Threshold', 'Sigma']
    if variable == 'dominance':
        file = 'mean_number_switches_per_subject.npy'
        label = 'dominance'
    if variable == 'amplitude':
        file = 'mean_peak_amplitude_per_subject.npy'
        label = 'Noise trials peak amplitude'
    if variable == 'latency':
        file = 'mean_peak_latency_per_subject.npy'
        label = 'Noise trials peak latency'
    if variable == 'freq2':
        file = 'hysteresis_width_freq_2.npy'
        label = 'Hysteresis freq=2'
    if variable == 'freq4':
        file = 'hysteresis_width_freq_4.npy'
        label = 'Hysteresis freq=4'
    if variable == 'pupil':
        file = os.path.join('aligned_eye_tracker_data','plots', 'min_pupil_across_trials.npy')
        label = 'Min. pupil'
    if 'saccade' in variable:
        file = os.path.join('aligned_eye_tracker_data','plots', 'saccade_rate.npy')
        label = 'Mean saccade rate'
    if 'blink' in variable:
        file = os.path.join('aligned_eye_tracker_data','plots', 'blink_rate.npy')
        label = 'Mean blink rate'
    if 'speed' in variable:
        file = os.path.join('aligned_eye_tracker_data','plots', 'average_speed.npy')
        label = 'Mean eye speed'
    if 'fixation' in variable:
        file = os.path.join('aligned_eye_tracker_data','plots', 'average_fixation_break_rate.npy')
        label = 'Mean FB rate'
    var = np.load(DATA_FOLDER + file)
    mean_variable = np.nanmean(var, axis=0)[:fitted_subs]
    r, p = pearsonr(variables[fitted_variable], mean_variable)
    fig, ax = plt.subplots(1, figsize=(3.4, 3))
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    var_fitted = variables[fitted_variable]
    ax.plot(var_fitted, mean_variable, marker='o', linestyle='', color='k')
    
    X = var_fitted.reshape(-1, 1)
    y = mean_variable.reshape(-1, 1)
    linreg = LinearRegression(fit_intercept=True).fit(X, y)
    minmax_array = np.array([np.min(X)-0.1, np.max(X)+0.1]).reshape(-1, 1)
    pred_y = linreg.predict(minmax_array)
    ax.annotate(f'r = {r:.3f}\np={p:.2e}', xy=(.04, 0.8), xycoords=ax.transAxes, fontsize=14)
    ax.plot(minmax_array,
           pred_y, color='gray', linestyle='--', alpha=0.4, linewidth=3)
    
    ax.set_xlabel(fitted_variable)
    ax.set_ylabel(label)
    fig.tight_layout()
    if full:
        fig, ax = plt.subplots(ncols=4, figsize=(14, 4))
        couplings = [0, 0.3, 1]
        for i in range(4):
            ax[i].spines['right'].set_visible(False); ax[i].spines['top'].set_visible(False)
            if i < 3:
                dominance_dur_coup = var[i, :fitted_subs]
                j_coupling = (j1s*couplings[i]+j0s)*n
                rho = pearsonr(j_coupling, dominance_dur_coup).statistic
                print(pearsonr(j_coupling, dominance_dur_coup).pvalue)
                ax[i].plot(j_coupling, dominance_dur_coup,
                            color='k', marker='o', linestyle='')
                ax[i].set_title('p(sh)=' + str(1-couplings[i]) + ';   r = ' + str(round(rho, 3)))
            else:
                j_coupling_0 = (j0s)*n
                j_coupling_03 = (j1s*0.3+j0s)*n
                j_coupling_1 = (j1s+j0s)*n
                all_coups = np.concatenate((j_coupling_0, j_coupling_03, j_coupling_1))
                all_doms = var[:, :fitted_subs].flatten()
                rho = pearsonr(all_coups, all_doms).statistic
                ax[i].plot(all_coups, all_doms, color='k', marker='o', linestyle='')
                ax[i].set_title('All p(sh);   r = ' + str(round(rho, 3)))
            ax[i].set_xlabel('Fitted J = (1-p(sh))*J1 + J0')
        ax[0].set_ylabel(label)
        fig.tight_layout()


def plot_theta_by_regime():
    label = 'ndt/'
    pars = glob.glob(SV_FOLDER + 'fitted_params/' + label + '*.npy')
    n_subs = len(pars)
    thetas = [np.load(par)[4] for par in pars]
    j1s = np.array([np.load(par)[0] for par in pars])  # /sigmas
    j0s = np.array([np.load(par)[1] for par in pars])  # /sigmas
    pshuffles = [1, 0.7, 0]
    thetas_mono = []
    thetas_bis = []
    jeffs = []
    thetas_all = []
    for sub in range(n_subs):
        jeffs.append(j1s[sub]+j0s[sub])
        thetas_all.append(thetas[sub])
        for psh in pshuffles:
            jeff = j1s[sub]*(1-psh)+j0s[sub]
            if jeff < 1/4:
                thetas_mono.append(thetas[sub])
            if jeff > 1/4:
                thetas_bis.append(thetas[sub])
    fig, ax = plt.subplots(ncols=3, figsize=(14, 4.5))
    variables = [j0s, j1s, jeffs]
    for i in range(3):
        r, p = pearsonr(variables[i], thetas_all)
        ax[i].annotate(f'r = {r:.3f}\np={p:.0e}', xy=(.04, 0.8), xycoords=ax[i].transAxes)
        ax[i].plot(variables[i], thetas_all, color='k', marker='o', linestyle='')
    ax[2].set_xlabel('Max. coupling, J_0 + J_1')
    ax[1].set_xlabel('J_1')
    ax[0].set_xlabel('J_0')
    ax[0].set_ylabel('theta')
    ax[1].set_ylabel('theta')
    ax[2].set_ylabel('theta')
    fig.tight_layout()
    thetas_mono = np.unique(thetas_mono)
    thetas_bis = np.unique(thetas_bis)
    plt.figure()
    sns.histplot(thetas_mono, color='cadetblue', bins=np.arange(0, 0.3, 0.03), alpha=0.5)
    sns.histplot(thetas_bis, color='peru', bins=np.arange(0, 0.3, 0.03))


def plot_coupling_transitions(n=4, plot_regime=False,
                              bw_adjust=1):
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    j1s = np.array([np.load(par)[0] for par in pars])  # /sigmas
    j0s = np.array([np.load(par)[1] for par in pars])  # /sigmas
    couplings = [0, 0.3, 1]
    j_coupling_0 = (j0s)
    j_coupling_03 = (j1s*0.3+j0s)
    j_coupling_1 = (j1s+j0s)
    print(len(np.where(np.sign(j_coupling_0-1/n) != np.sign(j_coupling_1-1/n))[0]))
    all_coups = np.row_stack((j_coupling_0, j_coupling_03, j_coupling_1))
    fig, ax = plt.subplots(1, figsize=(6, 4.5))
    idxs = np.sign(j_coupling_0-1/n) != np.sign(j_coupling_1-1/n)
    color = ['k' if not idx else 'r' for idx in idxs]
    alphas = [0.3 if not idx else 0.8 for idx in idxs]
    for i in range(len(pars)):
        ax.plot(couplings, all_coups[:, i], marker='o',
                color=color[i], alpha=alphas[i])
    ax.axhline(1/n, color='r', linewidth=3, linestyle='--', alpha=0.5)
    ax.set_xlabel('1-p(shuffle)'); ax.set_xticks([0., 0.3, 1.])
    ax.set_ylabel('J = (1-p(shuffle))*J1+J0'); fig.tight_layout()
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    data = np.row_stack((j_coupling_0, j_coupling_03, j_coupling_1)).flatten()
    threshold = 1/n
    fig, ax = plt.subplots(1, figsize=(3.5, 3.))
    if plot_regime:
        # KDE
        kde = scipy.stats.gaussian_kde(data)
        x = np.linspace(data.min()-0.25, data.max()+0.25, 500)
        y = kde(x)
        # Plot line
        # Line — left
        plt.plot(x[x < threshold], y[x < threshold], color='cadetblue', linewidth=4)
        
        # Line — right
        plt.plot(x[x > threshold], y[x > threshold], color='peru', linewidth=4)
        
        # Fill regions
        ax.fill_between(x[x < threshold], y[x < threshold],
                         color="cadetblue", alpha=1, facecolor='cadetblue',
                         linewidth=4)
        
        ax.fill_between(x[x > threshold], y[x > threshold],
                         color="peru", alpha=1, facecolor='peru',
                         linewidth=4)
        ax.set_ylim(-0.12, np.max(y)+0.52)
        ax.annotate('Monostable', xy=(.02, 0.9), xycoords=ax.transAxes,
                    color='cadetblue')
        ax.annotate('Bistable', xy=(.6, 0.9), xycoords=ax.transAxes,
                    color='peru')
        ax.annotate('Critical J', xy=(.42, 0.12), xycoords=ax.transAxes,
                    color='k', rotation=90)
    else:
        sns.kdeplot(all_coups.T, palette=['lightskyblue', 'royalblue', 'midnightblue'],
                    linewidth=4, ax=ax, bw_adjust=bw_adjust)
        ax.legend(['0', '0.7', '1'], title='p(shuffle)',
                  frameon=False)
    ax.axvline(threshold, color="black", linewidth=3)
    ax.set_xlabel('Effective coupling, J')
    ax.set_ylabel('')
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False); ax.set_yticks([])
    fig.tight_layout()
    savelab = '' if plot_regime else 'pshuffle_'
    fig.savefig(DATA_FOLDER + f'{savelab}critical_j_bistability.png', dpi=400)
    fig.savefig(DATA_FOLDER + f'{savelab}critical_j_bistability.pdf', dpi=400)


def compare_parameters_two_experiments(ax=None):
    folder_params_experiment_1 = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/fitting/parameters/'
    folder_data_experiment_1 = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/fitting/data/'
    all_df = load_data_experiment_1(data_folder=folder_data_experiment_1, n_participants='all')
    subjects = all_df.subject.unique()
    params_experiment_1 = []
    for subject in subjects:
        params_experiment_1.append(np.load(folder_params_experiment_1 + '/parameters_MF5_BADS' + subject + '.npy'))
    params_experiment_2 = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    j1s_exp2 = np.array([np.load(par)[0] for par in params_experiment_2])
    j0s_exp2 = np.array([np.load(par)[1] for par in params_experiment_2])
    b1s_exp2 = np.array([np.load(par)[2] for par in params_experiment_2])
    pval_j1 = scipy.stats.ttest_1samp(j1s_exp2, 0).pvalue
    # print(pval_j1)
    sigmas_exp2 = np.array([np.load(par)[3] for par in params_experiment_2])
    j1s_exp1 = np.array([par[0] for par in params_experiment_1])
    j0s_exp1 = np.array([par[1] for par in params_experiment_1])
    b1s_exp1 = np.array([par[2] for par in params_experiment_1])
    sigmas_exp1 = np.array([par[4] for par in params_experiment_1])
    parameter_pairs = [[j0s_exp1, j0s_exp2], [j1s_exp1, j1s_exp2], [b1s_exp1, b1s_exp2],
                                                                    [sigmas_exp1, sigmas_exp2]]
    labels = ['J0', 'J1', 'B1', r'$\sigma$']; colors = ['mediumpurple', 'burlywood']
    if ax is None:
        fig, ax = plt.subplots(ncols=4, figsize=(10, 3))
        saveflag = True
    else:
        saveflag = False
    for i_a, a in enumerate(ax):
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
        pvalue = scipy.stats.mannwhitneyu(parameter_pairs[i_a][0], parameter_pairs[i_a][1]).pvalue
        if i_a == 1:
            pv1 = scipy.stats.ttest_1samp(parameter_pairs[i_a][0], 0).pvalue    
            print(pv1)
        # print(pvalue)
        heights = [np.nanmean(parameter_pairs[i_a][k]) for k in range(2)]
        barplot_annotate_brackets(0, 1, pvalue, [0, 1], heights, yerr=None, dh=.16, barh=.05, fs=10,
                                  maxasterix=3, ax=a)
        sns.barplot(parameter_pairs[i_a], ax=a, linewidth=3, palette=colors, errorbar='se')
        # sns.stripplot(parameter_pairs[i_a], ax=a, color='k', size=2.5)
        sns.swarmplot(
                    data=parameter_pairs[i_a],
                    color="black",        # point fill
                    edgecolor="white",    # contrast on dark bars
                    linewidth=0.5,
                    size=3,
                    ax=a,
                    zorder=10             # ensures points are on top
                    )
        a.set_xticks([]); a.set_ylabel(labels[i_a])
    # Create legend only once (not in the loop)
    handles = [mpatches.Patch(color=colors[0], label='Exp. 1'),
               mpatches.Patch(color=colors[1], label='Exp. 2')]
    ax[0].legend(handles=handles, loc='upper center', frameon=False,
                 bbox_to_anchor=[0.7, 0.8])
    if saveflag:
        fig.tight_layout()
        fig.savefig(DATA_FOLDER + 'parameter_comparison_between_experiments.png', dpi=400)
        fig.savefig(DATA_FOLDER + 'parameter_comparison_between_experiments.pdf', dpi=400)


def analytical_bimodal_coef(j, sigma=0.2, n=4):
    delta = 216*j**3 / (1-n*j)**2 * sigma**2
    return 1/(3-delta)


def comparison_between_experiments_bis_mono(unique_shuffle=[1., 0.7, 0.],
                                            estimator='mean', n=4, ax=None):
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    mean_dominance_shuffle = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
    file_4 = 'hysteresis_width_freq_4.npy'
    file_2 = 'hysteresis_width_freq_2.npy'
    hyst_2 = np.load(DATA_FOLDER + file_2)
    hyst_4 = np.load(DATA_FOLDER + file_4)
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = np.array([[n*params[0], n*params[1], params[2], params[4], params[3]] for params in fitted_params_all])
    unique_shuffle = np.array(unique_shuffle)
    fitted_subs = fitted_params_all.shape[0]
    jeffs = np.zeros((3, fitted_subs))
    bistable_stim_dominance = []
    monostable_stim_dominance = []
    bistable_stim_hyst_2 = []
    monostable_stim_hyst_2 = []
    bistable_stim_hyst_4 = []
    monostable_stim_hyst_4 = []
    for i in range(fitted_subs):
        jeffs[:, i] = (fitted_params_all[i][0]*(1-unique_shuffle)+fitted_params_all[i][1])
        dom_bis = mean_dominance_shuffle[:, i][jeffs[:, i] >= 1]
        dom_mono = mean_dominance_shuffle[:, i][jeffs[:, i] < 1]
        hyst_bis_2 = hyst_2[:, i][jeffs[:, i] >= 1]
        hyst_mono_2 = hyst_2[:, i][jeffs[:, i] < 1]
        hyst_bis_4 = hyst_4[:, i][jeffs[:, i] >= 1]
        hyst_mono_4 = hyst_4[:, i][jeffs[:, i] < 1]
        if not sum(np.isnan(dom_bis)):
            bistable_stim_dominance.extend(dom_bis)
            bistable_stim_hyst_2.extend(hyst_bis_2)
            bistable_stim_hyst_4.extend(hyst_bis_4)
        if not sum(np.isnan(dom_mono)):
            monostable_stim_dominance.extend(dom_mono)
            monostable_stim_hyst_4.extend(hyst_mono_4)
            monostable_stim_hyst_2.extend(hyst_mono_2)
    # now for exp 1
    folder_bimodal = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/fitting/parameters/'  # Alex
    bimodal_coef = np.load(folder_bimodal + 'bimodality_coefficient.npy')
    folder_params_experiment_1 = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/fitting/parameters/'
    folder_data_experiment_1 = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/fitting/data/'
    all_df = load_data_experiment_1(data_folder=folder_data_experiment_1, n_participants='all')
    subjects = all_df.subject.unique()
    sarle_coef_bis = []
    sarle_coef_mono = []
    for i_sub, subject in enumerate(subjects):
        params_experiment_1 = np.load(folder_params_experiment_1 + '/parameters_MF5_BADS' + subject + '.npy')
        j_eff_exp1 = (params_experiment_1[0]*np.array(unique_shuffle) + params_experiment_1[1])*n
        coef_bis = bimodal_coef[:, i_sub][j_eff_exp1 >= 1]
        coef_mono = bimodal_coef[:, i_sub][j_eff_exp1 < 1]
        if not np.sum(np.isnan(coef_mono)):
            sarle_coef_mono.extend(coef_mono)
        if not np.sum(np.isnan(coef_bis)):
            sarle_coef_bis.extend(coef_bis)
    colormap = ['cadetblue', 'peru']
    variables_hyst_2 = [monostable_stim_hyst_2, bistable_stim_hyst_2]
    variables_hyst_4 = [monostable_stim_hyst_4, bistable_stim_hyst_4]
    variables_dominance = [monostable_stim_dominance, bistable_stim_dominance]
    variables_coef = [sarle_coef_mono, sarle_coef_bis]
    if ax is None:
        fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(11.2, 4))
        ax = ax.flatten()
        saveflag = True
    else:
        saveflag = False
    variables = [variables_hyst_2, variables_hyst_4, variables_dominance, variables_coef]
    for i_var, var in enumerate(variables):
        sns.barplot(var, palette=colormap, errorbar='se', ax=ax[i_var], estimator=estimator)
        sns.swarmplot(
                    data=var,
                    color="black",        # point fill
                    edgecolor="white",    # contrast on dark bars
                    linewidth=0.5,
                    size=3,
                    ax=ax[i_var],
                    zorder=10             # ensures points are on top
                    )
    if estimator == 'mean':
        dhs = [0.2, 0.2, 0.32, 0.05]
    if estimator == 'median':
        dhs = [0.2, 0.2, 0.32, 0.05]
    for i_a, a in enumerate(ax):
        if estimator == 'median':
            heights = np.nanmax(variables[i_a], axis=0)
        else:
            heights = np.nanmax(variables[i_a], axis=0)
        bars = np.arange(2)
        pv_sh01 = scipy.stats.mannwhitneyu(variables[i_a][0], variables[i_a][1]).pvalue
        print(pv_sh01)
        barplot_annotate_brackets(0, 1, pv_sh01, bars, heights, yerr=None, dh=dhs[i_a], barh=.01, fs=10,
                                  maxasterix=3, ax=a)
    ax[-1].axhline(5/9, color='gray', linestyle='--', linewidth=3)
    # if estimator == 'mean':
    #     ax[3].set_ylim(0.45, 0.72);  ax[0].set_ylim(0.85, 2.);
    #     ax[1].set_ylim(1.6, 2.35); ax[2].set_ylim(5.5, 10.4)
    # if estimator == 'median':
    #     ax[3].set_ylim(0.45, 0.72);  ax[0].set_ylim(0.85, 2.);
    #     ax[1].set_ylim(1.6, 2.85); ax[2].set_ylim(5.5, 10.4)
    # ax[0].set_ylabel("Sarle's bimodality coefficient")
    ax[3].set_ylabel("Bimodality coefficient")
    ax[2].set_ylabel("Dominance duration (s)")
    ax[3].set_yticks([0.3, 0.4, 5/9, 0.7, 0.8], ['0.3', '0.4', '5/9', '0.7', '0.8'])
    ax[3].set_ylim(0.28, 1.)
    ax[0].set_ylabel('Hysteresis f=2'); ax[1].set_ylabel('Hysteresis f=4')
    handles = [mpatches.Patch(color=colormap[0], label='Monostable'),
               mpatches.Patch(color=colormap[1], label='Bistable')]
    for a in ax:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
        a.set_xticks([])
    ax[3].legend(handles=handles, loc='upper center', frameon=False,
                 bbox_to_anchor=[0.6, 1.1])
    if saveflag:
        fig.tight_layout()
        fig.savefig(DATA_FOLDER + 'comparison_between_experiments_bistable_regime.png', dpi=400)
        fig.savefig(DATA_FOLDER + 'comparison_between_experiments_bistable_regime.pdf', dpi=400)


def comparison_between_experiments(estimator='mean', data_only=False,
                                   ax=None, fig=None):
    folder_bimodal = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/fitting/parameters/'  # Alex
    bimodal_coef = np.load(folder_bimodal + 'bimodality_coefficient.npy')
    bimodal_coef_simul = np.load(folder_bimodal + 'bimodality_coefficient_simul.npy')

    # mean_peak_amplitude = np.load(DATA_FOLDER + 'mean_peak_amplitude_per_subject.npy')
    # mean_peak_latency = np.load(DATA_FOLDER + 'mean_peak_latency_per_subject.npy')
    mean_number_switchs_coupling = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
    mean_number_switchs_coupling_simul = np.load(DATA_FOLDER + 'simulated_mean_number_switches_per_subject.npy')
    hyst_width_2 = np.load(DATA_FOLDER + 'hysteresis_width_freq_2.npy')
    hyst_width_4 = np.load(DATA_FOLDER + 'hysteresis_width_freq_4.npy')
    hyst_width_2_simul = np.load(DATA_FOLDER + 'hysteresis_width_f2_sims_fitted_params.npy')
    hyst_width_4_simul = np.load(DATA_FOLDER + 'hysteresis_width_f4_sims_fitted_params.npy')
    nrows = 1 if data_only else 2
    height = 4 if data_only else 6
    if ax is None:
        fig, ax = plt.subplots(ncols=4, nrows=nrows, figsize=(11.2, height),
                               sharey='col')
        ax = ax.flatten()
        saveflag = True
    else:
        saveflag = False
    for a in ax:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    variables = [hyst_width_2.T, hyst_width_4.T, mean_number_switchs_coupling.T, bimodal_coef.T]
    for i_var, var in enumerate(variables):
        sns.barplot(var, palette=colormap, errorbar='se', ax=ax[i_var], estimator=estimator)
        sns.swarmplot(
                    data=var,
                    color="black",        # point fill
                    edgecolor="white",    # contrast on dark bars
                    linewidth=0.5,
                    size=3,
                    ax=ax[i_var],
                    zorder=10             # ensures points are on top
                    )
    if not data_only:
        variables_simul = [hyst_width_2_simul.T, hyst_width_4_simul.T, mean_number_switchs_coupling_simul.T, bimodal_coef_simul.T]
        for i_var, var in enumerate(variables_simul):
            sns.barplot(var, palette=colormap, errorbar='se', ax=ax[i_var+4], estimator=estimator)
            sns.swarmplot(
                        data=var,
                        color="black",        # point fill
                        edgecolor="white",    # contrast on dark bars
                        linewidth=0.5,
                        size=3,
                        ax=ax[i_var+4],
                        zorder=10             # ensures points are on top
                        )
    
    ax[3].axhline(5/9, color='gray', linestyle='--', linewidth=3)
    xlim_ax0 = ax[3].get_xlim() + np.array((0., 0.35))
    ax[3].set_xlim(xlim_ax0)
    if not data_only:
        ax[7].axhline(5/9, color='gray', linestyle='--', linewidth=3)
        ax[7].set_ylim(0.28, 1.05)
    # ax[3].set_ylim(0.46, 0.71);  ax[0].set_ylim(0.85, 1.68);
    # ax[1].set_ylim(1.6, 2.35); ax[2].set_ylim(5.5, 9.9)
    # if not data_only:
    #     ax[7].set_ylim(0.48, 0.64);  ax[4].set_ylim(1.2, 1.62);  ax[6].set_ylim(1, 3.5)
    ax[3].set_ylabel("Bimodality coefficient")
    ax[2].set_ylabel("Dominance duration (s)")
    ax[3].set_yticks([0.3, 0.4, 5/9, 0.7, 0.8, 0.9], ['0.3', '0.4', '5/9', '0.7', '0.8', '0.9'])
    ax[3].set_ylim(0.28, 1.05)
    titles = ['Exp. 1\n', 'Exp. 2\nHysteresis, 1-C', 'Exp. 2\nHysteresis, 2-C', 'Exp. 2\nNoise trials']
    variables = [hyst_width_2, hyst_width_4, mean_number_switchs_coupling, bimodal_coef]
    dhs = [[0.05, 0.15, 0.05],
           [0.05, 0.16, 0.05],
           [0.05, 0.15, 0.05],
           [0.06, 0.12, 0.03]]
    if not data_only:
        variables_simul = [hyst_width_2_simul, hyst_width_4_simul, mean_number_switchs_coupling_simul, bimodal_coef_simul]
        variables = variables + variables_simul
        dhs = dhs*2
        ax[0].set_yticks([0, 1, 2, 3])
        ax[4].set_yticks([0, 1, 2, 3])
    for i_a, a in enumerate(ax):
        if estimator == 'median':
            heights = np.nanmax(variables[i_a].T, axis=0)
        else:
            heights = np.nanmax(variables[i_a].T, axis=0)
        bars = np.arange(3)
        pv_sh01 = scipy.stats.ttest_rel(variables[i_a][0], variables[i_a][1]).pvalue
        pv_sh02 = scipy.stats.ttest_rel(variables[i_a][0], variables[i_a][2]).pvalue
        pv_sh12 = scipy.stats.ttest_rel(variables[i_a][1], variables[i_a][2]).pvalue
        dh1, dh2, dh3 = dhs[i_a]
        barplot_annotate_brackets(0, 1, pv_sh01, bars, heights, yerr=None, dh=dh1, barh=.01, fs=10,
                                  maxasterix=3, ax=a)
        barplot_annotate_brackets(0, 2, pv_sh02, bars, heights, yerr=None, dh=dh2, barh=.01, fs=10,
                                  maxasterix=3, ax=a)
        barplot_annotate_brackets(1, 2, pv_sh12, bars, heights, yerr=None, dh=dh3, barh=.01, fs=10,
                                  maxasterix=3, ax=a)
        a.set_xticks([])
    ax[0].set_ylabel('Hysteresis, 1-C'); ax[1].set_ylabel('Hysteresis, 2-C')
    handles = [mpatches.Patch(color=colormap[0], label='1.'),
                mpatches.Patch(color=colormap[1], label='0.7'),
                mpatches.Patch(color=colormap[2], label='0.')]
    ax[1].legend(handles=handles,frameon=False,
                  title='p(shuffle)', loc='upper right')
    ax[3].text(0.92, 0.42, 'Bimodal', rotation=90, fontsize=13, transform=ax[3].transAxes)
    ax[3].text(0.92, 0.035, 'Unimodal', rotation=90, fontsize=13, transform=ax[3].transAxes)
    if saveflag:
        fig.tight_layout()
        fig.savefig(DATA_FOLDER + 'comparison_between_experiments.png', dpi=400)
        fig.savefig(DATA_FOLDER + 'comparison_between_experiments.pdf', dpi=400)


def experiment_comparison_altogether():
    fig, ax = plt.subplots(ncols=4, nrows=3, figsize=(11.2, 9))
    ax = ax.flatten()
    compare_parameters_two_experiments(ax=ax[:4])
    comparison_between_experiments(estimator='mean', data_only=True,
                                   ax=ax[4:8], fig=fig)
    comparison_between_experiments_bis_mono(unique_shuffle=[1., 0.7, 0.],
                                            estimator='mean', n=4, ax=ax[8:])
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'full_comparison_between_experiments.png', dpi=400)
    fig.savefig(DATA_FOLDER + 'full_comparison_between_experiments.pdf', dpi=400)
    fig.savefig(DATA_FOLDER + 'full_comparison_between_experiments.svg', dpi=400)


def plot_kernel_different_regimes(data_folder=DATA_FOLDER, fps=60, tFrame=26,
                                  steps_back=60, steps_front=20,
                                  shuffle_vals=[1, 0.7, 0],
                                  avoid_first=False, window_conv=1,
                                  filter_subjects=True, n=4, sub_alone=None,
                                  ax=None, cumsum=False):
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = np.array([[n*params[0], n*params[1], params[2], params[4], params[3]] for params in fitted_params_all])
    fitted_subs = len(pars)
    # idxs = [(par[2]/par[4] > 2) for par in fitted_params_all]
    fitted_params_all = fitted_params_all  # [idxs]
    df = load_data(data_folder + '/noisy/', n_participants='all', filter_subjects=filter_subjects)
    subs = df.subject.unique()[:fitted_subs]  # [idxs]
    print(subs, ', number:', len(subs))
    mean_vals_noise_switch_coupling = np.empty((2, steps_back+steps_front, len(subs)))
    mean_vals_noise_switch_coupling[:] = np.nan
    err_vals_noise_switch_coupling = np.empty((2, steps_back+steps_front, len(subs)))
    err_vals_noise_switch_coupling[:] = np.nan
    avg_bistable = []
    monostable_kernels = []
    bistable_kernels = []
    for i_sub, subject in enumerate(subs):
        if sub_alone is not None:
            if subject != sub_alone:
                continue
        df_sub = df.loc[df.subject == subject]
        trial_index = df_sub.trial_index.unique()
        mean_vals_noise_switch_all_trials = np.empty((1, steps_back+steps_front))
        mean_vals_noise_switch_all_trials[:] = np.nan
        number_switches = []
        bistable_mask = []
        for i_trial, trial in enumerate(trial_index):
            df_trial = df_sub.loc[df_sub.trial_index == trial]
            pshuffle_trial = df_trial.pShuffle.values[0]
            is_bistable = (fitted_params_all[i_sub][0]*(1-pshuffle_trial)+fitted_params_all[i_sub][1]) >= 1
            responses = df_trial.responses.values
            chi = df_trial.stimulus.values
            # chi = chi-np.nanmean(chi)
            orders = rle(responses)
            if avoid_first:
                idx_1 = orders[1][1:][orders[2][1:] == 2]
                idx_0 = orders[1][1:][orders[2][1:] == 1]
            else:
                idx_1 = orders[1][orders[2] == 2]
                idx_0 = orders[1][orders[2] == 1]
            number_switches.append(len(idx_1)+len(idx_0))
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
        avg_bistable.append(np.sum(bistable_mask) / len(bistable_mask))
        bistable_mask = np.array(bistable_mask)
        mean_vals_noise_switch_all_trials = mean_vals_noise_switch_all_trials[1:]
        bistable_values = mean_vals_noise_switch_all_trials[bistable_mask]
        monostable_values = mean_vals_noise_switch_all_trials[~bistable_mask]
        kernel_bistable = np.nanmean(bistable_values, axis=0)
        kernel_monostable = np.nanmean(monostable_values, axis=0)
        mean_vals_noise_switch_coupling[0, :, i_sub] = kernel_bistable
        err_vals_noise_switch_coupling[0, :, i_sub] = np.nanstd(bistable_values, axis=0)
        mean_vals_noise_switch_coupling[1, :, i_sub] = kernel_monostable
        err_vals_noise_switch_coupling[1, :, i_sub] = np.nanstd(monostable_values, axis=0)

    print(avg_bistable)
    colormap = ['peru', 'cadetblue']; labels=['Bistable', 'Monostable']
    if ax is None:
        fig, ax = plt.subplots(ncols=1, figsize=(5., 4))
        saveflag = True
    else:
        saveflag = False
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
    for regime in range(2):
        x_plot = np.arange(-steps_back, steps_front, 1)/fps
        if len(subs) > 1:
            y_plot = np.nanmean(mean_vals_noise_switch_coupling[regime, :], axis=-1)
            err_plot = np.nanstd(mean_vals_noise_switch_coupling[regime, :], axis=-1) / np.sqrt(len(subs))
        else:
            y_plot = np.nanmean(mean_vals_noise_switch_coupling[regime, :], axis=-1)
            err_plot = err_vals_noise_switch_coupling[regime, :, 0]
        if cumsum:
            y_plot = np.cumsum(y_plot)
        ax.plot(x_plot, y_plot, color=colormap[regime],
                label=labels[regime], linewidth=3)
        ax.fill_between(x_plot, y_plot-err_plot, y_plot+err_plot, color=colormap[regime],
                        alpha=0.3)
    ax.set_xlabel('Time from switch(s)')
    ax.set_ylabel('Noise')
    if saveflag:
        ax.legend(frameon=False)
        fig.tight_layout()    
        fig.savefig(SV_FOLDER + 'kernel_regime_across_subjects.png', dpi=400)
        fig.savefig(SV_FOLDER + 'kernel_regime_across_subjects.svg', dpi=400)


def plot_kernel_different_parameter_values(data_folder=DATA_FOLDER, fps=60, tFrame=26,
                                           steps_back=60, steps_front=20,
                                           shuffle_vals=[1, 0.7, 0],
                                           avoid_first=False, window_conv=1,
                                           filter_subjects=True, n=4, variable='J0',
                                           simulated=False,
                                           pshuff=None, ax=None, fig=None, legend=False):
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    if variable == 'J0':
        var = [np.load(par)[1] for par in pars]
        bins = [-0.1, 0.12, 0.2, 1/2]
        # bins = np.percentile(var, (0, 25, 50, 75, 100))
        # colormap = pl.cm.Oranges(np.linspace(0.3, 1, len(bins)))
    if variable == 'B1':
        var = [np.load(par)[2] for par in pars]
        bins =  [0, 0.2, 0.6, 0.8]
        # bins = np.percentile(var, (0, 33, 66, 100))
        # colormap = pl.cm.Greens(np.linspace(0.3, 1, len(bins)))
    if variable == 'THETA':
        var = [np.load(par)[4] for par in pars]
        bins = np.percentile(var, (0, 33, 66, 100))+np.array([-1e-6, 0, 0, 1e-6])
        bins = [0., 0.02, 0.3]
        # colormap = pl.cm.Blues(np.linspace(0.3, 1, len(bins)))
    if variable == 'SIGMA':
        var = [np.load(par)[3] for par in pars]
        bins = np.percentile(var, (0, 1/3*100, 100*2/3, 100))+np.array([-1e-6, 0, 0, 1e-6])
        # bins = [0., 0.2, 0.26, 0.3]
        # colormap = pl.cm.Greens(np.linspace(0.3, 1, len(bins)))
    if variable == 'J1':
        var = [np.load(par)[0] for par in pars]
        bins = np.percentile(var, (0, 1/3*100, 100*2/3, 100))+np.array([-1e-6, 0, 0, 1e-6])
        # bins = np.percentile(var, (0, 1/4*100, 100*2/4, 100*3/4, 100))+np.array([-1e-6, 0, 0, 0, 1e-6])
        # colormap = pl.cm.Reds(np.linspace(0.3, 1, len(bins)))
    colormap = pl.cm.binary(np.linspace(0.3, 1, len(bins)))
    label_save_fig = '/kernel_vs_params/simulation' if simulated else '/kernel_vs_params/'
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = np.array([[n*params[0], n*params[1], params[2], params[4], params[3]] for params in fitted_params_all])
    fitted_subs = len(pars)
    df = load_data(data_folder + '/noisy/', n_participants='all', filter_subjects=filter_subjects)
    subs = df.subject.unique()[:fitted_subs]
    print(subs, ', number:', len(subs))
    idxs_variable = np.digitize(var, bins=bins)-1
    print(idxs_variable)
    nbins = len(np.unique(idxs_variable))
    mean_vals_noise_switch_coupling = np.empty((nbins, steps_back+steps_front, len(subs)))
    mean_vals_noise_switch_coupling[:] = np.nan
    err_vals_noise_switch_coupling = np.empty((nbins, steps_back+steps_front, len(subs)))
    err_vals_noise_switch_coupling[:] = np.nan
    stim_subject = np.load(SV_FOLDER + 'stim_subject_noise.npy')
    responses_all = np.load(SV_FOLDER + 'responses_simulated_noise.npy')
    map_resps = {-1:1, 0:0, 1:2}
    for i_sub, subject in enumerate(subs):
        df_sub = df.loc[(df.subject == subject)]

        trial_index = df_sub.trial_index.unique()
        mean_vals_noise_switch_all_trials = np.empty((1, steps_back+steps_front))
        mean_vals_noise_switch_all_trials[:] = np.nan
        number_switches = []
        for i_trial, trial in enumerate(trial_index):
            df_trial = df_sub.loc[df_sub.trial_index == trial]

            if pshuff is not None:
                if df_trial.pShuffle.values[0] != pshuff:
                    continue
            if simulated:
                chi = stim_subject[i_sub, i_trial]
                responses = [map_resps[resp] for resp in responses_all[i_sub, i_trial]]
            else:
                responses = df_trial.responses.values
                chi = df_trial.stimulus.values
            # chi = chi-np.nanmean(chi)
            orders = rle(responses)
            if avoid_first:
                idx_1 = orders[1][1:][orders[2][1:] == 2]
                idx_0 = orders[1][1:][orders[2][1:] == 1]
            else:
                idx_1 = orders[1][orders[2] == 2]
                idx_0 = orders[1][orders[2] == 1]
            number_switches.append(len(idx_1)+len(idx_0))
            idx_1 = idx_1[(idx_1 > steps_back) & (idx_1 < (len(responses))-steps_front)]
            idx_0 = idx_0[(idx_0 > steps_back) & (idx_0 < (len(responses))-steps_front)]
            # original order
            mean_vals_noise_switch = np.empty((len(idx_1)+len(idx_0), steps_back+steps_front))
            mean_vals_noise_switch[:] = np.nan
            for i, idx in enumerate(idx_1):
                mean_vals_noise_switch[i, :] = chi[idx - steps_back:idx+steps_front]
            for i, idx in enumerate(idx_0):
                mean_vals_noise_switch[i+len(idx_1), :] =\
                    chi[idx - steps_back:idx+steps_front]*-1
            mean_vals_noise_switch_all_trials = np.row_stack((mean_vals_noise_switch_all_trials, mean_vals_noise_switch))
        mean_vals_noise_switch_all_trials = mean_vals_noise_switch_all_trials[1:]
        mean_vals_noise_switch_coupling[idxs_variable[i_sub], :, i_sub] = np.nanmean(mean_vals_noise_switch_all_trials, axis=0)
    if ax is None:
        fig, ax = plt.subplots(ncols=1, figsize=(5, 4))
        ax_none_flag = True
    else:
        ax_none_flag = False
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
    for regime in range(nbins):
        x_plot = np.arange(-steps_back, steps_front, 1)/fps
        if len(subs) > 1:
            y_plot = np.nanmean(mean_vals_noise_switch_coupling[regime, :], axis=-1)
            err_plot = np.nanstd(mean_vals_noise_switch_coupling[regime, :], axis=-1) / np.sqrt(len(subs))
        else:
            y_plot = np.nanmean(mean_vals_noise_switch_coupling[regime, :], axis=-1)
            err_plot = err_vals_noise_switch_coupling[regime, :, 0]
        ax.plot(x_plot, y_plot, color=colormap[regime], linewidth=3, label=f'{regime+1}{enums(regime+1)}')
        ax.fill_between(x_plot, y_plot-err_plot, y_plot+err_plot, color=colormap[regime],
                        alpha=0.3)
    ax.set_xlabel('Time from switch(s)')
    if legend:
        ax.legend(title=variable, frameon=False)
    if ax_none_flag:
        ax.set_ylabel('Noise')
        fig.tight_layout()
        fig.savefig(DATA_FOLDER + label_save_fig + f'kernel_across_subjects_different_{variable}.png', dpi=400)
        fig.savefig(DATA_FOLDER + label_save_fig + f'kernel_across_subjects_different_{variable}.pdf', dpi=400)


def compute_ndt_per_p_binary(stim_by_p, resp_by_p, p_values, sampling_rate=60, max_lag_s=2.0):
    """
    Compute NDT per p_value using lagged logistic regression.
    Assumes responses are 0 = no response, 1 = Left, 2 = Right.
    Only uses trials with actual responses (1 or 2) and converts to binary (Right=1, Left=0).
    
    Parameters
    ----------
    stim_by_p, resp_by_p: dict[p_value] -> list of np.arrays (one per trial)
        each array shape (T,)
    p_values: list of p condition values
    sampling_rate: Hz
    max_lag_s: maximum lag to test (seconds)
    
    Returns
    -------
    results: dict
        keys = p_value
        values = dict with 'best_lag', 'ndt_s', 'aucs', 'lags'
    """
    max_lag = int(max_lag_s * sampling_rate)
    results = {}

    for p in p_values:
        if p == 'all':
            stim_trials = np.row_stack([stim_by_p[p] for p in p_values[1:]])
            resp_trials = np.row_stack([resp_by_p[p] for p in p_values[1:]])
        else:
            stim_trials = stim_by_p[p]
            resp_trials = resp_by_p[p]

        aucs = []
        lags = np.arange(max_lag + 1)

        for lag in lags:
            X_all, y_all = [], []

            # gather all trials for this lag
            for stim, resp in zip(stim_trials, resp_trials):
                if lag == 0:
                    stim_lagged = stim
                    resp_lagged = resp
                else:
                    stim_lagged = stim[:-lag]
                    resp_lagged = resp[lag:]

                # remove no-response periods
                mask = resp_lagged != 0
                if np.sum(mask) < 2:
                    continue

                X_all.append(stim_lagged[mask])
                y_all.append((resp_lagged[mask] == 2).astype(int))  # Right=1, Left=0

            if len(X_all) == 0:
                aucs.append(np.nan)
                continue

            X = np.concatenate(X_all).reshape(-1, 1)
            y = np.concatenate(y_all)

            if np.unique(y).size < 2:
                aucs.append(np.nan)
                continue

            model = LogisticRegression(max_iter=1000)
            scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            aucs.append(np.mean(scores))

        best_lag = lags[np.nanargmax(aucs)]
        ndt_s = best_lag / sampling_rate
        results[p] = dict(best_lag=best_lag, ndt_s=ndt_s, aucs=aucs, lags=lags)

    return results


def plot_ndt_logistic_regression(data_folder=DATA_FOLDER,
                                 ntraining=8, p_shuffle=[0., 0.7, 1., 'all'],
                                 fps=60, tFrame=26, compute=False,
                                 subj_exp=21, all_trials=False):
    # subject 19, 23, 24: bistable much worse performance. good modulation of NDT
    # subject 21/29: good overall performance, well separated
    # subject 7: p(shuffle)=1 highest auc
    # subject 3: good evolution on p(shuffles)
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()
    if compute:
        ndt_list = []
        aucs = {1.: [], 0.: [], 0.7: [], 'all': []}
        for i_subject, subject in enumerate(subjects):
            df_sub = df.loc[df.subject == subject]
            responses, stim, time = expand_responses(df_sub, sampling_rate=fps, trial_duration=tFrame)
            result = compute_ndt_per_p_binary(stim, responses, p_shuffle, sampling_rate=fps, max_lag_s=2.0)
            best_lags = {p: v['best_lag'] for p, v in result.items()}
            p_min = min(best_lags, key=best_lags.get)
            for p in p_shuffle:
                aucs[p].append(result[p]['aucs'])
            aucs['all'].append(result['all']['aucs'])
            # min_lag = best_lags[p_min]
            min_ndt = result[p_min]['ndt_s']
            ndt_list.append(min_ndt)
        ndts = np.array(ndt_list)
        np.save(DATA_FOLDER + 'ndts_data_argmin_R2.npy', ndts)
        np.save(DATA_FOLDER + 'aucs_data_argmin_R2.npy', aucs)
    else:
        ndts_total = np.load(DATA_FOLDER + 'ndts_data_argmin_R2.npy')
        aucs = np.load(DATA_FOLDER + 'aucs_data_argmin_R2.npy', allow_pickle=True).item()
    pars = glob.glob(SV_FOLDER + 'fitted_params/' + '*.npy')
    print(len(pars), ' fitted subjects')
    fitted_params_all = [np.load(par) for par in pars]
    j1s = np.array([np.load(par)[0] for par in pars])  # /sigmas
    j0s = np.array([np.load(par)[1] for par in pars])  # /sigmas
    b1s = [np.load(par)[2] for par in pars]
    sigmas = np.array([np.load(par)[3] for par in pars])
    thetas = [np.load(par)[4] for par in pars]
    fitted_params_all = [[params[0], params[1], params[2], params[4], params[3]] for params in fitted_params_all]
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(13, 7.))
    colormap = ['midnightblue', 'royalblue', 'lightskyblue']
    ax = ax.flatten()
    lags = np.arange(121)/60
    ax[0].set_title('Example subject', fontsize=13)
    ax[1].set_title('Average across subjects', fontsize=13)
    ax[2].set_title('Max. AUC across lags', fontsize=13)
    ax[1].set_xlabel('Lag (s)'); ax[0].set_xlabel('Lag (s)')
    ax[2].set_xlabel('Subject ID');   
    for i_a, a in enumerate(ax):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        if i_a < 4 and i_a > 0:
            a.set_ylabel('<AUC>')
            a.axhline(0.5, color='k', alpha=0.4, linestyle='--', linewidth=3)
    ax[0].set_ylabel(r'AUC(lag), $\hat{r}(t) = f(s(t-lag))$')
    ax[2].set_ylim(0.49, 1); ax[1].set_ylim(0.49, 1.0); ax[3].set_ylim(0.49, 1)
    ax[4].set_xlabel(r'$NDT = argmin_{lag} (argmax_{lag} AUC(lag, p(shuffle)))$')
    aucs_subject_exp = np.row_stack(([aucs[p][subj_exp] for p in p_shuffle]))
    mean_exp = np.nanmean(aucs_subject_exp, axis=0)
    for i_shuf, p in enumerate(p_shuffle):
        ax[0].plot(lags, aucs[p][subj_exp], linewidth=4, color=colormap[i_shuf])
        ax[0].axvline(lags[np.argmax(aucs[p][subj_exp])], color=colormap[i_shuf],
                      linestyle='--')
        ax[1].plot(lags, np.nanmean(aucs[p], axis=0),
                   label=p, linewidth=4, color=colormap[i_shuf])
        ax[1].axvline(lags[np.argmax(np.nanmean(aucs[p], axis=0))], color=colormap[i_shuf],
                      linestyle='--')
        ndts = np.argmax(aucs[p], axis=1)/60
        ax[2].plot(np.nanmax(aucs[p], axis=1), linewidth=4, color=colormap[i_shuf])
        sns.kdeplot(y=np.nanmean(aucs[p], axis=1), linewidth=4, ax=ax[3], bw_adjust=0.5,
                    cut=0, color=colormap[i_shuf])
        sns.kdeplot(x=ndts, linewidth=3, ax=ax[4], bw_adjust=0.5,
                    cut=0, color=colormap[i_shuf])
        # j_eff = j1s*(1-p)+j0s
        # ax[5].plot(thetas, ndts, marker='o', linestyle='')
    if all_trials:
        ax[1].plot(lags, np.nanmean(aucs['all'][:35], axis=0),
                   label='All trials', linewidth=4, color='firebrick')
        ax[2].plot(np.nanmean(aucs['all'][::2], axis=1), linewidth=4, color='firebrick')
        ax[1].axvline(lags[np.argmax(np.nanmean(aucs['all'][:35], axis=0))],
                      color='firebrick', linestyle='--')
        sns.kdeplot(y=np.nanmean(aucs['all'][::2], axis=1), linewidth=4, ax=ax[3], bw_adjust=0.5,
                    cut=0, color='firebrick')
        ndts_all_trials = lags[np.argmax(aucs['all'][::2], axis=1)]
        sns.kdeplot(x=ndts_all_trials, linewidth=4, ax=ax[4], bw_adjust=0.5,
                    cut=0, color='firebrick')
        fig2, ax2 = plt.subplots(1)
        ax2.plot([0, 2], [0, 2], color='k', linestyle='--', alpha=0.3, linewidth=4)
        ax2.set_ylabel('MIN delay across shuffles')
        ax2.set_xlabel('argmax delay across ALL trials')
        ax2.plot(ndts_all_trials, ndts_total, marker='o', linestyle='', color='k')
        r, p = pearsonr(ndts_all_trials, ndts_total)
        ax2.annotate(f'r = {r:.3f}\np={p:.0e}', xy=(.04, 0.8), xycoords=ax2.transAxes)
        fig2.tight_layout()
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        # ax[5].plot(j0s, ndts_all_trials, marker='>', linestyle='', color='firebrick')
    # ax[0].plot(lags, mean_exp, color='firebrick', linewidth=4)
    # ax[0].axvline(lags[np.argmax(mean_exp)], color='firebrick', linestyle='--')
    ax[5].set_xlabel('Kernel NDT');  ax[5].set_ylabel('NDT')
    ax[5].set_xlim(ax[5].get_ylim())
    ndts_comparison = ndts_all_trials if all_trials else ndts_total
    kernel_ndt = np.load(DATA_FOLDER + 'kernel_latency_average.npy')
    ax[5].plot(np.abs(kernel_ndt), ndts_comparison, marker='o', linestyle='', color='firebrick')
    r, p = pearsonr(np.abs(kernel_ndt), ndts_comparison)
    ax[5].annotate(f'r = {r:.3f}\np={p:.0e}', xy=(.04, 1.), xycoords=ax[5].transAxes)
    sns.kdeplot(x=ndts_total, linewidth=4, ax=ax[4], bw_adjust=0.5,
                cut=0, color='firebrick', linestyle='--')
    if all_trials:
        ax[1].legend(title='p(shuffle)', frameon=False, loc='lower left', ncol=2); fig.tight_layout()
    else:
        ax[1].legend(title='p(shuffle)', frameon=False); fig.tight_layout()


def peak_latency_distributions():
    vals = np.load(DATA_FOLDER + 'kernel_latency_average.npy')
    f = plt.figure(); plt.xlabel('Latency (s)')
    sns.histplot(vals, bins=20)
    plt.axvline(np.mean(vals), color='k')
    plt.axvline(np.median(vals), linestyle='--', color='k')
    f.tight_layout()


# Function to plot a curved arrow on the cylinder surface


# Function to plot a curved arrow on the cylinder surface
def curved_arrow_cylinder(ax, r, z_pos, theta_start, theta_end, color='green', lw=2, marker='<', cyl=True):
    theta = np.linspace(theta_start, theta_end, 500)
    y_arrow = r * np.sin(theta)
    if cyl:
        x_arrow = r * np.cos(theta)
    else:
        x_arrow = np.zeros(500)
        y_arrow = np.linspace(np.min(y_arrow), np.max(y_arrow), 500)
    z_arrow = np.ones_like(theta) * z_pos
    ax.plot(x_arrow, y_arrow, z_arrow, color=color, linewidth=lw)
    # Add arrow head manually at the end
    ax.plot([x_arrow[-1]],
            [y_arrow[-1]],
            [z_arrow[-1]],
            color=color, linewidth=lw, marker=marker, markersize=8)


def plot_cylinder(cyl=True, dot_prop=1., ndots=300):
    # Parameters
    n_points = ndots
    r = 1.0
    height = 2.0
    
    if cyl:
        # Random points on cylinder surface
        theta = np.random.uniform(0, 2*np.pi, n_points)
        z = np.random.uniform(-height/2, height/2, n_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
    else:
        # Random points on cylinder surface
        z = np.random.uniform(-r, r, n_points)
        x = np.random.uniform(-r, r, n_points)
        y = np.random.uniform(-r, r, n_points)
    
    # Define front/back by small random jitter in y (to simulate depth)
    # Points with y > 0 appear in front (green), y < 0 are back (red)
    front = x > 0
    back = ~front
    # Dot sizes
    size_back = 29.5
    size_front = dot_prop*size_back
    # Plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    if cyl:
        x_f, x_b = x[front], x[back]
        # Compute point size based on how front-facing the point is (y-value)
        s_min, s_max = 1, size_front
        s = s_min + (np.abs(x) / r) * (s_max - s_min)
        size_front = s[front]
        s_min, s_max = 1, size_back
        s = s_min + (np.abs(x) / r) * (s_max - s_min)
        size_back = s[back]
    else:
        x_f = x_b = 0
        z_fake = np.random.uniform(0., 0.3, z[front].shape[0])
        size_front = size_front* (1 - z_fake)
        z_fake = np.random.uniform(0., 0.3, z[back].shape[0])
        size_back = size_back* (1 - z_fake)
    ax.scatter(x_b, y[back], z[back], color='red', edgecolor='red', s=size_back, depthshade=False)
    ax.scatter(x_f, y[front], z[front], color='green', edgecolor='green', s=size_front, depthshade=False)
    # Transparent cylinder surface
    # Create meshgrid for cylinder surface
    if cyl:
        theta_surf = np.linspace(0, 2*np.pi, 50)
        z_surf = np.linspace(-height/2, height/2, 20)
        theta_surf, z_surf = np.meshgrid(theta_surf, z_surf)
        x_surf = r * np.cos(theta_surf)
        y_surf = r * np.sin(theta_surf)
        ax.plot_surface(x_surf, y_surf, z_surf, color='gray', alpha=0.1, linewidth=0, edgecolor=None)
        ax.plot(x_surf[0], y_surf[0], z_surf[0], color='k')
        ax.plot(x_surf[-1], y_surf[-1], z_surf[-1], color='k')
        ax.plot([0, 0], [-r, -r], [-height/2, height/2], color='k')
        ax.plot([0, 0], [r, r], [-height/2, height/2], color='k')
        # Add green arrow in front
        curved_arrow_cylinder(ax, r=r, z_pos=-1.3, theta_start=-np.pi/4, theta_end=np.pi/4, color='green', lw=4, marker='>')
        
        # Add red arrow in back
        curved_arrow_cylinder(ax, r=r, z_pos=1.3, theta_start=-5*np.pi/4, theta_end=-3*np.pi/4, color='red', lw=4, marker='<')
        # View straight from the front (along +y)
        ax.view_init(elev=11, azim=0)
    else:
        theta_surf = np.linspace(0, 2*np.pi, 50)
        z_surf = np.linspace(-height/2, height/2, 20)
        theta_surf, z_surf = np.meshgrid(theta_surf, z_surf)
        x_surf = 0 * np.cos(theta_surf)
        y_surf = r * np.sin(theta_surf)
        ax.plot_surface(x_surf, y_surf, z_surf, color='gray', alpha=0.05, linewidth=0, edgecolor=None)
        ax.plot([0, 0], [-r, -r], [-height/2, height/2], color='k')
        ax.plot([0, 0], [r, r], [-height/2, height/2], color='k')
        ax.plot([0, 0], [-r, r], [height/2, height/2], color='k')
        ax.plot([0, 0], [-r, r], [-height/2, -height/2], color='k')
        curved_arrow_cylinder(ax, r=r, z_pos=1.3, theta_start=5*np.pi/4, theta_end=3*np.pi/4,
                              color='red', lw=4, marker='<', cyl=cyl)
        curved_arrow_cylinder(ax, r=r, z_pos=-1.3, theta_start=-np.pi/4, theta_end=np.pi/4, color='green',
                              lw=4, marker='>', cyl=cyl)
        # View straight from the front (along +y)
        ax.view_init(elev=0, azim=0)
    
    # Remove axes, grid, and frame
    ax.set_axis_off()
    ax.grid(False)
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    plt.show()
    label = 'sfm_' if cyl else 'rdm_'
    label = label if dot_prop == 1. else label + f'dot_prop_{dot_prop}_'
    fig.savefig(SV_FOLDER + label + 'cartoon.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + label + 'cartoon.pdf', dpi=400, bbox_inches='tight')


def low_dimensional_projection():
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    ndt = np.abs(np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = np.array([[params[0], params[1], params[2], params[4], params[3]] for params in fitted_params_all])
    scaling = manifold.MDS(n_components=3, max_iter=5000, normalized_stress=False)
    S_scaling = scaling.fit_transform(fitted_params_all)
    x, y, z = S_scaling.T
    
    color = fitted_params_all[:, 1]
    # color = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy').mean(axis=0)
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter3D(x, y, z, c=color, cmap='copper')
    plt.xlabel('MDS-dim1'); plt.ylabel('MDS-dim2');
    plt.title('Colored by J1+J0')


def vector_field_cylinder(p_shuff=1):
    x = np.linspace(0, np.pi, 8)
    y = np.linspace(0, np.pi, 5)
    X, Y = np.meshgrid(x, y)
    V = 0
    U = np.sin(X); U_flat = U.flatten()
    nShuffle = int(p_shuff*U.size)
    indices = np.random.choice(U.size, size=nShuffle, replace=False)
    shuffled_values = U_flat[indices]
    np.random.shuffle(shuffled_values)
    U_flat[indices] = shuffled_values
    U_shuffled = U_flat.reshape(U.shape)
    fig, ax = plt.subplots(1)
    plt.quiver(X, Y, U_shuffled, V, color='darkgreen', width=0.009, linewidths=1.,
               edgecolor='darkgreen')
    ax.axis('off')
    ax.set_xlim(-0.2, np.pi+0.4)
    fig.savefig(SV_FOLDER + f'vf_cylinder_p_shuffle_{p_shuff}.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + f'vf_cylinder_p_shuffle_{p_shuff}.pdf', dpi=400, bbox_inches='tight')


def plot_dominance_subjects_vs_model():
    dominance_model = np.load(DATA_FOLDER + 'simulated_mean_number_switches_per_subject.npy')
    dominance_data = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
    plt.figure()
    normalized_dominance_data = dominance_data  # / np.mean(dominance_data, axis=0)
    normalized_dominance_model = dominance_model  #/ np.mean(dominance_model, axis=0)
    minval = np.min([normalized_dominance_data.flatten(), normalized_dominance_model.flatten()])-2e-2
    maxval = np.max([normalized_dominance_data.flatten(), normalized_dominance_model.flatten()])+2e-2
    plt.plot([minval, maxval], [minval, maxval], color='gray', linewidth=3, linestyle='--', alpha=0.3)
    plt.xlim(minval, maxval); plt.ylim(minval, maxval)
    plt.plot(normalized_dominance_data.flatten(), normalized_dominance_model.flatten(), color='k', marker='o',
             linestyle='')


def compute_dominance_durations(df):
    """
    Compute dominance durations for each response state (1=left, 2=right),
    grouped by subject and p_shuffle condition.
    """
    df = df.copy()
    df["duration"] = df["keypress_seconds_offset"] - df["keypress_seconds_onset"]
    df = df[df["response"] != 0]  # ignore no-press periods
    
    durations = (
        df.groupby(["subject", "pShuffle", "response"], sort=False)["duration"]
        .apply(list)
        .reset_index()
    )
    return durations


def plot_dominance_durations(data_folder=DATA_FOLDER,
                             ntraining=8, freq=2, sem=False):

    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()
    nsubs = len(subjects)
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    durations_df = compute_dominance_durations(df.loc[df.freq.abs() == freq])

    
    fig, axes = plt.subplots(nrows=nsubs//5, ncols=5, figsize=(13, 10), sharex=True)
    axes = axes.flatten()
    fig2, axes2 = plt.subplots(nrows=nsubs//5, ncols=5, figsize=(13, 10), sharex=True)
    axes2 = axes2.flatten()
    # --- Individual subject distributions
    mean_dominance_shuffle = np.zeros((3, nsubs))
    for i, subj in enumerate(subjects):
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['top'].set_visible(False)
        axes2[i].spines['right'].set_visible(False)
        axes2[i].spines['top'].set_visible(False)
        subdf = durations_df.loc[durations_df["subject"] == subj]
        duration_ch1 = np.concatenate(subdf.loc[subdf['response'] == 1, 'duration'].values)
        duration_ch2 = np.concatenate(subdf.loc[subdf['response'] == 2, 'duration'].values)
        sns.histplot(duration_ch1, ax=axes[i], color='darkgreen', linewidth=4, label='ch=L',
                     bins=np.linspace(0, 26, 20))
        sns.histplot(duration_ch2, ax=axes[i], color='firebrick', linewidth=4, label='ch=R',
                     bins=np.linspace(0, 26, 20))
        axes[i].set_title(f"Subject {subj}", fontsize=10)
        axes[i].set_xlabel("Dominance (s)")
        axes[i].set_ylabel('')
        for i_sh, shuffle in enumerate([1., 0.7, 0.]):
            dom_durs = subdf.groupby(["pShuffle"], sort=False)["duration"].apply(sum)[shuffle]
            mean_dominance_shuffle[i_sh, i] = np.nanmedian(dom_durs)
            sns.kdeplot(dom_durs, ax=axes2[i], color=colormap[i_sh], linewidth=4, label=shuffle,
                        bw_adjust=.7, cut=0)
        axes2[i].set_title(f"Subject {subj}", fontsize=10)
        axes2[i].set_xlabel("Dominance (s)")
        axes2[i].set_ylabel('')
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    axes2[0].set_ylabel("Density")
    axes2[0].legend(frameon=False, title='p(shuffle)')
    fig2.tight_layout()
    fig2.savefig(SV_FOLDER + f'dominance_durations_freq_{freq}.png', dpi=400, bbox_inches='tight')
    fig2.savefig(SV_FOLDER + f'dominance_durations_freq_{freq}.pdf', dpi=400, bbox_inches='tight')

    # --- Average distribution across subjects -- median across trials per subject
    fig2, ax2 = plt.subplots(1, figsize=(4, 3.5))
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    for i_sh, shuffle in enumerate([1., 0.7, 0.]):
        sns.kdeplot(mean_dominance_shuffle[i_sh], ax=ax2, color=colormap[i_sh], linewidth=4,
                    bw_adjust=0.6, label=shuffle, cut=0)
    ax2.set_xlabel('Dominance (s)'); ax2.legend(title='p(shuffle)', frameon=False)
    fig2.tight_layout()
    fig2.savefig(SV_FOLDER + f'average_dominance_durations_freq_{freq}.png', dpi=400, bbox_inches='tight')
    fig2.savefig(SV_FOLDER + f'average_dominance_durations_freq_{freq}.pdf', dpi=400, bbox_inches='tight')


    fig2, ax2 = plt.subplots(1, figsize=(4, 3.5))
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    sns.boxplot(mean_dominance_shuffle.T, ax=ax2, palette=colormap, linewidth=4)
    sns.stripplot(mean_dominance_shuffle.T, ax=ax2, color='k')
    for i in range(nsubs):
        ax2.plot([0, 1, 2], mean_dominance_shuffle[:, i], color='gray', alpha=0.5)
    ax2.set_ylabel('Dominance (s)')
    fig2.tight_layout()

    # mean_dominance_shuffle = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
    # fig2, ax2 = plt.subplots(1, figsize=(4, 3.5))
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    # sns.boxplot(mean_dominance_shuffle.T, ax=ax2, palette=colormap, linewidth=4)
    # sns.stripplot(mean_dominance_shuffle.T, ax=ax2, color='k')
    # for i in range(nsubs):
    #     ax2.plot([0, 1, 2], mean_dominance_shuffle[:, i], color='gray', alpha=0.5)
    # ax2.set_ylabel('Dominance (s)')
    # fig2.tight_layout()


def plot_dom_hyst_correlation():
    mean_dominance_shuffle = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
    def corrfunc(x, y, ax=None, **kws):
        """Plot the correlation coefficient in the top left hand corner of a plot."""
        r, p = scipy.stats.pearsonr(x, y)
        ax = ax or plt.gca()
        ax.annotate(f'r = {r:.3f}\np = {p:.1e}', xy=(.1, 0.8), xycoords=ax.transAxes)
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(6, 3))
    for a in ax.flatten():
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    for i_f, freq in enumerate([2, 4]):
        if freq == 4:
            file = 'hysteresis_width_freq_4.npy'
        if freq == 2:
            file = 'hysteresis_width_freq_2.npy'
        var = np.load(DATA_FOLDER + file)
        x_data = np.mean(mean_dominance_shuffle, axis=0)  # .flatten()
        y_data = np.mean(var, axis=0)  # .flatten()
        ax[i_f].plot(x_data, y_data,
                     marker='o', linestyle='', color='k')
        corrfunc(x_data, y_data, ax=ax[i_f])
        
    ax[0].set_xlabel('Dominance (s)')
    ax[1].set_xlabel('Dominance (s)')
    ax[0].set_ylabel('Hysteresis 1 cycle')
    ax[1].set_ylabel('Hysteresis 2 cycle')
    fig.tight_layout()


def plot_dominance_hyst_pshuffle(freq=2):
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.05
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    
    # start with a square Figure
    fig3 = plt.figure(figsize=(5, 5))
    
    ax3 = fig3.add_axes(rect_scatter)
    ax_histx = fig3.add_axes(rect_histx, sharex=ax3)
    ax_histy = fig3.add_axes(rect_histy, sharey=ax3)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    if freq == 4:
        file = 'hysteresis_width_freq_4.npy'
    if freq == 2:
        file = 'hysteresis_width_freq_2.npy'
    var = np.load(DATA_FOLDER + file)
    mean_dominance_shuffle = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
    variables_hyst = var
    variables_dominance = mean_dominance_shuffle
    for a in [ax_histx, ax_histy]:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
    ax_histx.spines['bottom'].set_visible(False)
    ax_histy.spines['left'].set_visible(False)
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)
    r, p = pearsonr(var.flatten(), mean_dominance_shuffle.flatten())
    ax3.annotate(f'r = {r:.2f}\np = {p:.0e}', xy=(.02, 0.85), xycoords=ax3.transAxes)
    labels = ['1', '0.7', '0']
    for i in range(3):
        meanx = np.mean(variables_hyst[i]); meany = np.mean(variables_dominance[i])
        errx = np.std(variables_hyst[i]);  erry = np.std(variables_dominance[i])
        ax3.errorbar(x=meanx, y=meany, xerr=errx, yerr=erry, color=colormap[i],
                     marker='o', label=labels[i], markersize=9)
        ax3.plot(variables_hyst[i], variables_dominance[i], color=colormap[i], linestyle='',
                 marker='x', alpha=0.6, markersize=6)
        sns.kdeplot(y=variables_dominance[i], color=colormap[i], ax=ax_histy,
                    linewidth=2, bw_adjust=0.6)
        sns.kdeplot(variables_hyst[i], color=colormap[i], ax=ax_histx,
                    linewidth=2, bw_adjust=0.6)
    # ax3.set_yscale('log'); ax3.set_xscale('log')
    ax3.set_ylim(0, 15.9); ax3.set_xlim(0, 2.8)
    ax3.set_ylabel('Dominance (s)')
    ax3.set_xlabel('Hysteresis');
    ax3.legend(frameon=False, loc='lower left', bbox_to_anchor=[0.65, -0.02],
               title='p(shuffle)')
    fig3.tight_layout()
    fig3.savefig(DATA_FOLDER + 'dominance_vs_hysteresis_pshuffle.png', dpi=300, bbox_inches='tight')
    fig3.savefig(DATA_FOLDER + 'dominance_vs_hysteresis_pshuffle.pdf', dpi=300, bbox_inches='tight')


def compute_switch_prob_group(stim, choice, freq, pshuffle, n_bins=50, T_trial=26,
                              fps=60):
    """
    Compute probability of L→R and R→L switches over time,
    averaged across subjects, grouped by frequency and pshuffle condition.
    
    Parameters
    ----------
    stim : array (n_subj, n_trials, n_time)
        Stimulus traces
    choice : array (n_subj, n_trials, n_time)
        Choices (-1, 0, 1)
    freq : array (n_subj, n_trials)
        Stimulus frequency (2 or 4)
    pshuffle : array (n_subj, n_trials)
        Condition (0., 0.7, 1.)
    n_bins : int
        Number of time bins
    
    Returns
    -------
    df : pandas.DataFrame
        Columns: freq, pshuffle, time_bin, p_LR, p_RL
    """
    results = []
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    n_subj, n_trials, n_time = stim.shape      # seconds per bin
    t_norm = np.linspace(0, 1, n_time - 1)  # assume fixed ascending-descending sweep

    for f in np.unique(freq):
        deltat = T_trial / n_bins
        for ps in np.unique(pshuffle):
            p_LR_sum = np.zeros(n_bins)
            p_RL_sum = np.zeros(n_bins)
            count_sum = np.zeros(n_bins)
            subj_count = 0

            for subj in range(n_subj):
                # Select trials for this subject and condition
                mask = (freq[subj] == f) & (pshuffle[subj] == ps)
                stim_sel = stim[subj, mask]
                choice_sel = choice[subj, mask]

                if stim_sel.size == 0:
                    continue

                p_LR_s, p_RL_s, count_s = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)

                for c in choice_sel:
                    switch_LR = (c[:-1] == -1) & (c[1:] == 1)
                    switch_RL = (c[:-1] == 1) & (c[1:] == -1)

                    p_LR, _ = np.histogram(t_norm[switch_LR], bins=bin_edges)
                    p_RL, _ = np.histogram(t_norm[switch_RL], bins=bin_edges)

                    p_LR_s += p_LR
                    p_RL_s += p_RL
                    count_s += 1

                # average over trials for this subject
                with np.errstate(divide='ignore', invalid='ignore'):
                    p_LR_s = np.divide(p_LR_s, count_s, out=np.zeros_like(p_LR_s), where=count_s > 0)
                    p_RL_s = np.divide(p_RL_s, count_s, out=np.zeros_like(p_RL_s), where=count_s > 0)

                p_LR_sum += p_LR_s/deltat
                p_RL_sum += p_RL_s/deltat
                count_sum += (count_s > 0)
                subj_count += 1

            # average across subjects (ignore subjects with no data)
            if subj_count > 0:
                p_LR_mean = p_LR_sum / subj_count
                p_RL_mean = p_RL_sum / subj_count
            else:
                p_LR_mean = np.zeros(n_bins)
                p_RL_mean = np.zeros(n_bins)

            df = pd.DataFrame({
                "freq": f,
                "pshuffle": ps,
                "time_bin": bin_centers,
                "p_LR": p_LR_mean,
                "p_RL": p_RL_mean
            })
            results.append(df)

    return pd.concat(results, ignore_index=True)


def compute_switch_rate_since_switch(data_folder=DATA_FOLDER, shuffle_vals=[1., 0.7, 0.], tFrame=26,
                                     fps=60, steps_front=400, bin_size=0.25, simulated=True, steps_back=100):
    """
    Compute and plot the switch rate as a function of time since each perceptual switch,
    separately for each pShuffle condition.
    """

    nFrame = int(fps * tFrame)
    dt = 1 / fps
    time_since_switch = np.arange(-steps_back, steps_front) * dt  # seconds

    df = load_data(data_folder + '/noisy/', n_participants='all', filter_subjects=True)
    subs = df.subject.unique()


    responses_all = np.load(SV_FOLDER + 'responses_simulated_noise.npy')
    map_resps = {-1:1, 0:0, 1:2}
    
    # Store subject-level means per pShuffle
    subject_means = {ps: [] for ps in shuffle_vals}

    for i_sub, subject in enumerate(subs):
        df_sub = df.loc[df.subject == subject]
        trial_index = df_sub.trial_index.unique()
        all_switch_segments = []
        all_pshuffle_labels = []

        for i_trial, trial in enumerate(trial_index):
            df_trial = df_sub.loc[df_sub.trial_index == trial]
            if not simulated:
                responses = df_trial.responses.values
            if simulated:
                responses = np.array([map_resps[resp] for resp in responses_all[i_sub, i_trial]])
            # get switch indices
            idx_01, idx_10 = get_switch_indices(responses)
            all_switches = np.sort(np.concatenate([idx_01, idx_10]))

            # keep only those with enough frames ahead
            all_switches = all_switches[(all_switches < len(responses) - steps_front) &
                                        (all_switches >= steps_back)]

            for idx in all_switches:
                seg = responses[idx-steps_back:idx + steps_front]
                all_switch_segments.append(seg)
                all_pshuffle_labels.append(df_trial.pShuffle.values[0])
        # === Compute subject-level mean per pShuffle ===
        all_switch_segments = np.vstack(all_switch_segments)
        all_pshuffle_labels = np.array(all_pshuffle_labels)
        for ps in shuffle_vals:
            
            mask = all_pshuffle_labels == ps
            if not np.any(mask):
                continue
            mean_seg = np.nanmean(all_switch_segments[mask], axis=0)
    
            # === Bin the signal over time for smoothing ===
            bin_frames = int(bin_size * fps)
            total_steps = steps_back + steps_front
            n_bins = total_steps // bin_frames
            binned = [np.nanmean(mean_seg[i * bin_frames:(i + 1) * bin_frames])
                      for i in range(n_bins)]
            subject_means[ps].append(binned)

    # === Group-level averaging ===
    bin_centers = (np.arange(n_bins) * bin_frames + bin_frames / 2 - steps_back) * dt
    group_mean = {ps: np.nanmean(subject_means[ps], axis=0) for ps in shuffle_vals if len(subject_means[ps]) > 0}
    group_sem = {ps: np.nanstd(subject_means[ps], axis=0) / np.sqrt(len(subject_means[ps])) for ps in shuffle_vals if len(subject_means[ps]) > 0}

    fig, ax = plt.subplots(ncols=1, figsize=(5, 4))
    # ax.set_ylim(-0.02, 1.02)
    ax.axhline(1.5, color='k', linewidth=3, linestyle='--', alpha=0.2)
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    cmap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]

    for color, ps in zip(cmap, group_mean.keys()):

        # mean_rate = np.nanmean(switch_prob[mask], axis=0) / dt  # Hz
        # sem_rate = np.nanstd(switch_prob[mask] / dt, axis=0) / np.sqrt(np.sum(mask))

        # # bin using mean
        # mean_rate_binned = [np.nanmean(mean_rate[i * bin_frames:(i + 1) * bin_frames])
        #                     for i in range(n_bins)]
        # sem_rate_binned = [np.nanmean(sem_rate[i * bin_frames:(i + 1) * bin_frames])
        #                    for i in range(n_bins)]
        plt.plot(bin_centers, group_mean[ps], lw=4, color=color, label=ps)
        plt.fill_between(bin_centers,
                          group_mean[ps] - group_sem[ps],
                          group_mean[ps] + group_sem[ps],
                          color=color, alpha=0.3)
    # plt.ylim(1.44, 1.54)
    plt.xlabel("Time since switch (s)")
    plt.ylabel("Switch rate (Hz)")
    plt.legend(title="p(shuffle)", frameon=False)
    plt.tight_layout()


def plot_switch_rate_model(data_folder=DATA_FOLDER, sv_folder=SV_FOLDER,
                           fps=60, n=4, ntraining=8, tFrame=26,
                           window_conv=5, n_bins=74):
    nFrame = tFrame*fps
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()
    choices_all_subject = np.zeros((len(subjects), 72, nFrame))
    pshuffles_per_sub = np.zeros((len(subjects), 72))
    freqs_per_sub = np.zeros((len(subjects), 72))
    all_stims = np.zeros_like(choices_all_subject)
    for i_s, subject in enumerate(subjects):
        print('Simulating subject', subject)
        df_subject = df.loc[df.subject == subject]
        pshuffles = np.round(df_subject.groupby('trial_index')['pShuffle'].mean().values, 1)
        ini_side = df_subject.groupby('trial_index')['initial_side'].mean().values
        frequencies = np.round(df_subject.groupby('trial_index')['freq'].mean().values*ini_side)
        choice_all = np.load(sv_folder + f'choice_matrix_subject_{subject}.npy')
        stimlist = np.vstack([get_blist(freq, nFrame) for freq in frequencies])
        choices_all_subject[i_s] = choice_all
        pshuffles_per_sub[i_s] = pshuffles
        freqs_per_sub[i_s] = frequencies
        all_stims[i_s] = stimlist
    df_switches = compute_switch_prob_group(all_stims, choices_all_subject, freqs_per_sub, pshuffles_per_sub, n_bins=n_bins, fps=fps)
    fig, axes = plt.subplots(ncols=2, figsize=(7.5, 4.))
    titles = ['One cycle', 'Two cycles']
    for i_ax, ax in enumerate(axes):
        ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
        ax.set_xlabel('Time (s)'); ax.axvline(tFrame/(2+2*i_ax), color='k', alpha=0.4,
                                              linestyle='--', linewidth=3)
        ax.axvline(tFrame/(4+4*i_ax), color='k', alpha=0.6, linestyle=':', linewidth=2)
        ax.axvline(3*tFrame/(4+4*i_ax), color='k', alpha=0.6, linestyle=':', linewidth=2)
        ax.set_title(titles[i_ax], fontsize=13)
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    for ipsh, ps in enumerate(sorted(df_switches['pshuffle'].unique())[::-1]):
        for i_f, f in enumerate([2, 4]):
            sel = (df_switches['pshuffle'] == ps) & (df_switches['freq'] == f)
            d = df_switches[sel]
            sr = d['p_LR'].values
            if window_conv is None:
                switch_rate = sr
            else:
                switch_rate = np.convolve(sr, np.ones(window_conv)/window_conv, mode='same')
            if f == 4:
                switch_rate = np.nanmean(np.row_stack([switch_rate[:n_bins//2], switch_rate[n_bins//2:]]), axis=0)
                timevals = d['time_bin'][::2]*tFrame/2
            else:
                timevals = d['time_bin']*tFrame
            axes[i_f].plot(timevals, switch_rate, label=f"{ps}",
                           color=colormap[ipsh], linewidth=4)
    axes[0].legend(frameon=False, title='p(shuffle)')
    axes[1].set_xlim(-0.1, 13.1)
    axes[0].set_xlabel("Time (s)"); axes[1].set_xlabel("Time (s)")
    axes[0].set_ylabel("Switch rate L-->R")
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'simulated_hysteresis_switch_rate.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'simulated_hysteresis_switch_rate.svg', dpi=400, bbox_inches='tight')


def plot_kernel_parameters_data_vs_model():
    fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(16, 6))
    ax = ax.flatten()
    i_a = 0
    for a, variable in zip(ax, ['J0', 'J1', 'B1', 'SIGMA', 'THETA']):
        plot_kernel_different_parameter_values(data_folder=DATA_FOLDER, fps=60, tFrame=26,
                                                steps_back=120, steps_front=20,
                                                shuffle_vals=[1, 0.7, 0],
                                                avoid_first=False, window_conv=1,
                                                filter_subjects=True, n=4, variable=variable,
                                                simulated=False, pshuff=None,
                                                ax=a, fig=fig, legend=True)
        plot_kernel_different_parameter_values(data_folder=DATA_FOLDER, fps=60, tFrame=26,
                                                steps_back=120, steps_front=20,
                                                shuffle_vals=[1, 0.7, 0],
                                                avoid_first=False, window_conv=1,
                                                filter_subjects=True, n=4, variable=variable,
                                                simulated=True, pshuff=None,
                                                ax=ax[i_a+5], fig=fig, legend=False)
        ax[i_a].set_title(variable, fontsize=14)
        i_a += 1
    ax[0].set_ylabel('Noise - data')
    ax[5].set_ylabel('Noise - simulation')
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'kernel_across_subjects_different_all_variables.png', dpi=400)
    fig.savefig(DATA_FOLDER + 'kernel_across_subjects_different_all_variables.pdf', dpi=400)


def compute_logistic_regression(X, y):
    clf = LogisticRegression(max_iter=1000, random_state=0).fit(X, y)
    return clf.intercept_, clf.coef_


def logistic_regression_weights_over_time(data_folder=DATA_FOLDER,
                                          fps=60, tFrame=26,
                                          steps_back=120, steps_front=120,
                                          simulated=False, pshuf_only=None,
                                          ax=None, color=None, compute_regression=False):
    """
    Compute and plot the switch rate as a function of time since each perceptual switch,
    separately for each pShuffle condition.
    """
    
    dt = 1 / fps
    time_since_switch = np.arange(-steps_back, steps_front) * dt  # seconds
    label_simul = '_simul' if simulated else ''
    if pshuf_only is not None:
        label = f'_pshuffle_{pshuf_only}' + label_simul
    else:
        label = '_all' + label_simul
    if compute_regression:
        df = load_data(data_folder + '/noisy/', n_participants='all', filter_subjects=True)
        subs = df.subject.unique()
        
        
        if simulated:
            ndt = np.abs(np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
            ndt = np.repeat(ndt, len(subs))
        else:
            ndt = -np.load(DATA_FOLDER + 'kernel_latency_average.npy')
        ndt_frames_all = np.int64(ndt / dt)
        responses_all = np.load(SV_FOLDER + 'responses_simulated_noise.npy')
        map_resps = {-1:1, 0:0, 1:2}
    
    
        weights_per_subject = np.zeros((len(time_since_switch), len(subs)))
        for i_sub, subject in enumerate(subs):
            ndt_frames = 0
            df_sub = df.loc[df.subject == subject]
            trial_index = df_sub.trial_index.unique()
            stim_all_trials = np.empty((1, steps_back+steps_front))
            stim_all_trials[:] = np.nan
            responses_all_trials = np.empty((1, steps_back+steps_front))
            responses_all_trials[:] = np.nan
            for i_trial, trial in enumerate(trial_index):
                df_trial = df_sub.loc[df_sub.trial_index == trial]
                if pshuf_only is not None:
                    if df_trial.pShuffle.values[0] != pshuf_only:
                        continue
                chi = df_trial.stimulus.values
                if not simulated:
                    responses = df_trial.responses.values
                if simulated:
                    responses = np.array([map_resps[resp] for resp in responses_all[i_sub, i_trial]])
                orders = rle(responses)
                idx_1 = orders[1][1:][orders[2][1:] == 2]
                idx_0 = orders[1][1:][orders[2][1:] == 1]
                idx_1 = idx_1[(idx_1 > (steps_back + ndt_frames)) & (idx_1 < (len(responses))-(ndt_frames+steps_front))]
                idx_0 = idx_0[(idx_0 > (steps_back+ndt_frames)) & (idx_0 < (len(responses))-(steps_front+ndt_frames))]
                # original order
                stimulus_trial = np.empty((len(idx_1)+len(idx_0), steps_back+steps_front))
                stimulus_trial[:] = np.nan
                responses_trial = np.empty((len(idx_1)+len(idx_0), steps_back+steps_front))
                responses_trial[:] = np.nan
                for i, idx in enumerate(idx_1):
                    stimulus_trial[i, :] = chi[idx - steps_back:idx+steps_front]
                    responses_trial[i, :] = responses[idx - steps_back+ndt_frames:idx+steps_front+ndt_frames]
                for i, idx in enumerate(idx_0):
                    stimulus_trial[i+len(idx_1), :] =\
                        chi[idx - steps_back:idx+steps_front]
                    responses_trial[i, :] = responses[idx - steps_back+ndt_frames:idx+steps_front+ndt_frames]
                stim_all_trials = np.row_stack((stim_all_trials, stimulus_trial))
                responses_all_trials = np.row_stack((responses_all_trials, responses_trial))
            responses_all_trials = responses_all_trials[1:]
            stim_all_trials = stim_all_trials[1:]
            for i_t, t in enumerate(time_since_switch):
                X, y = stim_all_trials[:, i_t], responses_all_trials[:, i_t]-1
                idxs = (~np.isnan(y))*(y >= 0)*(~np.isnan(X))
                X = X[idxs].reshape(-1, 1); y = y[idxs]
                try:
                    intercept, coefs = compute_logistic_regression(X, y)
                except ValueError:
                    coefs = np.nan
                weights_per_subject[i_t, i_sub] = coefs
        np.save(DATA_FOLDER + 'regression_weights' + label +'.npy', weights_per_subject)
    else:
        weights_per_subject = np.load(DATA_FOLDER + 'regression_weights' + label +'.npy')
    if ax is None:
        fig, ax = plt.subplots(ncols=1, figsize=(5, 4))
        if pshuf_only is not None:
            ax.set_title('p(shuffle)=' + str(pshuf_only), fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axhline(0, color='k', alpha=0.3, linestyle='--', linewidth=2)
    # for sub in range(len(subs)):
    #     ax.plot(time_since_switch, weights_per_subject[:, sub],
    #             color='k', linewidth=1, alpha=0.1)
    color = 'k' if color is None else color
    ax.plot(time_since_switch, np.nanmean(weights_per_subject, axis=-1),
            color=color, linewidth=4, label=pshuf_only)
    ax.set_xlabel('Time from switch (s)'); ax.set_ylabel('Stimulus weight')


def plot_regression_weights():
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharey=True)
    pshuffles = [1., 0.7, 0.]
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    for i_p, pshuffle in enumerate(pshuffles):
        logistic_regression_weights_over_time(data_folder=DATA_FOLDER,
                                                  fps=60, tFrame=26,
                                                  steps_back=120, steps_front=120,
                                                  simulated=False, pshuf_only=pshuffle,
                                                  ax=ax[0], color=colormap[i_p], compute_regression=False)
        logistic_regression_weights_over_time(data_folder=DATA_FOLDER,
                                                  fps=60, tFrame=26,
                                                  steps_back=120, steps_front=120,
                                                  simulated=True, pshuf_only=pshuffle,
                                                  ax=ax[1], color=colormap[i_p], compute_regression=False)
    ax[0].legend(title='p(shuffle)', frameon=False)
    ax[1].set_ylabel('')
    fig.tight_layout()
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
    for i_a, a in enumerate(ax):
        pshuf = pshuffles[i_a]
        label = f'_pshuffle_{pshuf}_simul'
        weights_per_subject_model = np.load(DATA_FOLDER + 'regression_weights' + label +'.npy')
        label = f'_pshuffle_{pshuf}'
        weights_per_subject_data = np.load(DATA_FOLDER + 'regression_weights' + label +'.npy')
        a.plot(weights_per_subject_data, weights_per_subject_model, color=colormap[i_a],
               marker='o', linestyle='')
    fig.tight_layout()


def kernel_different_regimes_all_subjects(data_folder=DATA_FOLDER,
                                          fps=60, tFrame=26, ntraining=8):
    nFrame = fps*tFrame
    df = load_data(data_folder, n_participants='all', filter_subjects=True)
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()
    fig2, ax2 = plt.subplots(ncols=4, nrows=int(np.ceil(len(subjects)/4)), figsize=(18, 20),
                             sharex=True, sharey=True)
    ax2 = ax2.flatten()
    for i_s, sub in enumerate(subjects):
        plot_kernel_different_regimes(data_folder=DATA_FOLDER, fps=60, tFrame=26,
                                      steps_back=150, steps_front=20,
                                      shuffle_vals=[1, 0.7, 0],
                                      avoid_first=True, window_conv=1,
                                      filter_subjects=True, n=4, sub_alone=sub,
                                      ax=ax2[i_s])
        ax2[i_s].set_xticks([]);  ax2[i_s].set_yticks([])
        ax2[i_s].set_xlabel('');  ax2[i_s].set_ylabel('')
        ax2[i_s].axvline(0, color='k', alpha=0.3, linestyle='--')
    ax2[-1].legend(frameon=False)
    ax2[-1].set_ylabel('Noise')
    ax2[-1].set_xlabel('Time from switch (s)')
    ax2[-1].axvline(0, color='k', alpha=0.3, linestyle='--')
    ax2[-1].spines['right'].set_visible(False)
    ax2[-1].spines['top'].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(SV_FOLDER + 'noise_kernel_different_regimes.png', dpi=200, bbox_inches='tight')
    fig2.savefig(SV_FOLDER + 'noise_kernel_different_regimes.svg', dpi=200, bbox_inches='tight')


def ridgeplot_all_kernels(
        data_folder=DATA_FOLDER,
        steps_back=150,
        steps_front=10,
        fps=60,
        order_by_variable=False,
        zscore_variables=False,
        n_columns=5):

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import glob

    x_plot = np.arange(-steps_back, steps_front, 1) / fps
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    kernels_data = np.load(DATA_FOLDER + 'all_kernels_noise_switch_aligned.npy')
    kernels_model = np.load(DATA_FOLDER + 'simulated_all_kernels_noise_switch_aligned.npy')

    n_subs, n_t = kernels_data.shape

    # --------------------------------------------------
    # Ordering
    # --------------------------------------------------
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')

    j1s = np.array([np.load(par)[0] for par in pars])
    j0s = np.array([np.load(par)[1] for par in pars])
    b1s = np.array([np.load(par)[0] for par in pars])
    thetas = np.array([np.load(par)[1] for par in pars])
    sigmas = np.array([np.load(par)[0] for par in pars])

    kernel_ndt = np.load(DATA_FOLDER + 'kernel_latency_average.npy')

    variables = {
        'J': j0s + j1s,
        'B': b1s,
        'sigma': sigmas,
        'theta': thetas,
        'ndt': kernel_ndt,
        'J1': j1s,
        'J0': j0s,
        'B/sigma': b1s / sigmas,
        'theta/sigma': thetas / sigmas,
        'theta/J': thetas / (j0s+j1s),
        'peak_data': np.nanmax(kernels_data, axis=1),
        'peak_model': np.nanmax(kernels_model, axis=1) 
    }

    if order_by_variable is not None:
        idxs_by_variable = np.argsort(variables[order_by_variable])
    else:
        idxs_by_variable = np.arange(n_subs)

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    n_rows = int(np.ceil(n_subs / n_columns))

    fig, axes = plt.subplots(
        1,
        n_columns,
        figsize=(3 * n_columns, 0.5 * n_rows),
        sharex=True,
        sharey=True
    )

    if n_columns == 1:
        axes = [axes]

    # vertical spacing
    offset = 6 if zscore_variables else 1.2
    offset *= np.max(np.abs(kernels_data))

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    for col in range(n_columns):

        ax = axes[col]

        start = col * n_rows
        stop = min((col + 1) * n_rows, n_subs)

        for local_i, global_i in enumerate(range(start, stop)):

            idx = (
                idxs_by_variable[global_i]
                if order_by_variable is not None
                else global_i
            )

            y0 = local_i * offset

            ax.axhline(
                y0,
                color='gray',
                alpha=0.5,
                linestyle=':'
            )

            # -------------------------
            # Data
            # -------------------------
            y_plot = (
                zscore(kernels_data[idx])
                if zscore_variables
                else kernels_data[idx]
            )

            ax.plot(
                x_plot,
                y_plot + y0,
                color="k",
                lw=4,
                label="Data" if (col == 0 and local_i == 0) else None
            )

            # -------------------------
            # Model
            # -------------------------
            y_plot = (
                zscore(kernels_model[idx])
                if zscore_variables
                else kernels_model[idx]
            )

            ax.plot(
                x_plot,
                y_plot + y0,
                color="r",
                lw=4,
                alpha=0.8,
                label="Model" if (col == 0 and local_i == 0) else None
            )

            ax.text(
                x_plot[0] - 0.15 * (x_plot[-1] - x_plot[0]),
                y0,
                f"S{idx+1}",
                va="center"
            )

        ax.axvline(0, linestyle='--', color='gray', alpha=0.3)
        ax.set_yticks([])
        ax.set_xlabel("Time (s)")

    axes[0].legend(
        loc="upper left",
        frameon=False,
        ncol=2,
        bbox_to_anchor=[0.02, 1.01]
    )

    sns.despine(left=True)
    plt.tight_layout()

    label = 'z_scored_' if zscore_variables else ''

    fig.savefig(
        SV_FOLDER + label + 'kernels_data_vs_model.png',
        dpi=200,
        bbox_inches='tight'
    )

    fig.savefig(
        SV_FOLDER + label + 'kernels_data_vs_model.svg',
        dpi=200,
        bbox_inches='tight'
    )

    return fig, axes


def low_dimensional_projection_hyst_noise_vars():
    from sklearn.decomposition import PCA
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = np.array([[params[0], params[1], params[2], params[4], params[3]] for params in fitted_params_all])
    mean_peak_amplitude = np.load(DATA_FOLDER + 'mean_peak_amplitude_per_subject.npy')
    mean_peak_latency = np.load(DATA_FOLDER + 'mean_peak_latency_per_subject.npy')
    mean_number_switchs_coupling = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
    hyst_width_2 = np.load(DATA_FOLDER + 'hysteresis_width_freq_2.npy')
    hyst_width_4 = np.load(DATA_FOLDER + 'hysteresis_width_freq_4.npy')
    shuf = np.repeat(np.array([0, 0.3, 1.]), hyst_width_2.shape[1]).reshape(hyst_width_2.shape)
    variables = np.vstack((mean_peak_amplitude.flatten(),
                           mean_number_switchs_coupling.flatten(),
                           mean_peak_latency.flatten(),
                           hyst_width_2.flatten(),
                           hyst_width_4.flatten()))
    # scaling = manifold.MDS(n_components=2, max_iter=10000, normalized_stress=False,
    #                   random_state=3)
    scaling = PCA(n_components=2)
    S_scaling = scaling.fit_transform(variables.T)
    x, y = S_scaling.T
    # color = hyst_width_2.flatten()
    color = (fitted_params_all[:, 0]*(1-shuf) + fitted_params_all[:, 1]) > 1/4
    fig, ax = plt.subplots(1)
    ax.scatter(x, y, c=color, cmap='copper')
    plt.xlabel('PC1'); plt.ylabel('PC2');
    plt.title('Colored by H(J)_2')


def plot_cool_neuron_sheet(n_examples=6, save=True, plot_different_orient=False):
    # Generate smooth low-frequency “cloth-like” undulations
    np.random.seed(21)  #41
    height = 5 if not plot_different_orient else 12
    fig = plt.figure(figsize=(4*n_examples, height))
    for n_ex in range(n_examples):
        # Grid for the sheet
        n = 400
        x = np.linspace(-2, 2, n)
        y = np.linspace(-2, 2, n)
        X, Y = np.meshgrid(x, y)
    
        Z_random = np.random.randn(*X.shape)
        Z_smooth = gaussian_filter(Z_random, sigma=90)  # larger sigma -> smoother, gentle hills
        Z_smooth *= 0.01  # amplitude of bending
    
        # Define thickness
        thickness = 0.000015
        Z_top = Z_smooth
        Z_bottom = Z_smooth - thickness
    
        # --- Create high-res mask for neurons ---
        high_res = 800   # high-resolution image for smooth circles
        x_hr = np.linspace(-2, 2, high_res)
        y_hr = np.linspace(-2, 2, high_res)
        X_hr, Y_hr = np.meshgrid(x_hr, y_hr)
        
        # Base sheet RGB
        sheet_hr = np.ones((high_res, high_res, 3))  # gray
        
        # Neuron parameters
        num_neurons_side = 4
        neuron_radius = 0.2
        neuron_x = np.linspace(-1.5, 1.5, num_neurons_side)
        neuron_y = np.linspace(-1.5, 1.5, num_neurons_side)
        
        # --- Random colors for connections ---
        colors = [
            np.array([0.1, 0.8, 0.1]),   # green
            np.array([0.8, 0.1, 0.1]),   # red
            np.array([0.2, 0.2, 0.2])    # dark gray
        ]
        
        # Thickness of connection lines in the high-res texture
        line_thickness = int(high_res * 0.007)
        
        # Neurons arranged in a square grid:
        centers = [(nx, ny) for nx in neuron_x for ny in neuron_y]
        Nside = len(neuron_x)
        
        # Convenience functions
        x_to_idx = lambda v: np.argmin(np.abs(x_hr - v))
        y_to_idx = lambda v: np.argmin(np.abs(y_hr - v))
        # Loop through neuron grid using 2D indexing
        for ix in range(Nside):
            for iy in range(Nside):
        
                x1 = neuron_x[ix]
                y1 = neuron_y[iy]
        
                # list neighbors: (dx,dy) relative indices
                neighbor_offsets = [(1,0), (-1,0), (0,1), (0,-1)]
        
                for dx_idx, dy_idx in neighbor_offsets:
        
                    jx = ix + dx_idx
                    jy = iy + dy_idx
        
                    # skip if out of bounds
                    if jx < 0 or jx >= Nside or jy < 0 or jy >= Nside:
                        continue
        
                    x2 = neuron_x[jx]
                    y2 = neuron_y[jy]
        
                    # choose random color
                    line_color = colors[np.random.choice(np.arange(len(colors)), p=[0.1, 0.1, 0.8])]
        
                    # direction
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = np.sqrt(dx*dx + dy*dy)
                    ux = dx / dist
                    uy = dy / dist
        
                    # start/end points at circle edges
                    Ax = x1 + neuron_radius * ux
                    Ay = y1 + neuron_radius * uy
                    Bx = x2 - neuron_radius * ux
                    By = y2 - neuron_radius * uy
        
                    # convert to pixel coordinates
                    Ax_i = x_to_idx(Ax)
                    Ay_i = y_to_idx(Ay)
                    Bx_i = x_to_idx(Bx)
                    By_i = y_to_idx(By)
        
                    # parametric line
                    steps = int(dist * high_res / 2)
                    t = np.linspace(0, 1, steps)
        
                    xs = Ax_i + (Bx_i - Ax_i) * t
                    ys = Ay_i + (By_i - Ay_i) * t
        
                    # draw thick line
                    for xi, yi in zip(xs.astype(int), ys.astype(int)):
                        y0 = max(0, yi - line_thickness)
                        y1_ = min(high_res, yi + line_thickness)
                        x0 = max(0, xi - line_thickness)
                        x1_ = min(high_res, xi + line_thickness)
                        sheet_hr[y0:y1_, x0:x1_] = line_color

        # plot neurons
        for nx in neuron_x:
            for ny in neuron_y:
                mask = (X_hr - nx)**2 + (Y_hr - ny)**2 < neuron_radius**2
                sheet_hr[mask] = [0.8, 0.8, 0.8]  # darker neurons
                distance = np.sqrt((X_hr - nx)**2 + (Y_hr - ny)**2)
                border_thickness = 0.05  # fraction of total range, adjust as needed
                border_mask = np.logical_and(distance >= neuron_radius - border_thickness,
                                             distance <= neuron_radius)
                sheet_hr[border_mask] = [0, 0, 0]  # black border
        # --- Interpolate down to surface grid size ---
        factor = n / high_res
        sheet = zoom(sheet_hr, (factor, factor, 1), order=1)  # bilinear interpolation
        
        
        ax = fig.add_subplot(1+plot_different_orient*1,
                             n_examples, n_ex+1, projection='3d')
        
        # Top sheet with neurons
        ax.plot_surface(X, Y, Z_top, facecolors=sheet, edgecolor=None, antialiased=True,
                        rstride=1, cstride=1, shade=False)
        # Top edge
        ax.plot(X[0, :], Y[0, :], Z_top[0, :], color='black', linewidth=4)
        
        # Bottom edge
        ax.plot(X[-1, :], Y[-1, :], Z_top[-1, :], color='black', linewidth=4)
        
        # Left edge
        ax.plot(X[:, 0], Y[:, 0], Z_top[:, 0], color='black', linewidth=4)
        
        # Right edge
        ax.plot(X[:, -1], Y[:, -1], Z_top[:, -1], color='black', linewidth=4)
        if plot_different_orient:
            ax2 = fig.add_subplot(2, n_examples, n_ex+1+n_examples, projection='3d')
            ax2.plot_surface(X, Y, Z_top, facecolors=sheet, edgecolor=None, antialiased=True,
                             rstride=1, cstride=1)
            ax2.plot_surface(X, Y, Z_bottom, color='gray', edgecolor=None, antialiased=True)
        
        # Bottom sheet
        ax.plot_surface(X, Y, Z_bottom, color='white', edgecolor=None, antialiased=True)
        
        # Black edges for thickness
        edges = [(0, slice(None)), (-1, slice(None)), (slice(None), 0), (slice(None), -1)]
        for edge in edges:
            if isinstance(edge[0], int):
                xs = X[edge[0], :]
                ys = Y[edge[0], :]
                zt = Z_top[edge[0], :]
                zb = Z_bottom[edge[0], :]
            else:
                xs = X[:, edge[1]]
                ys = Y[:, edge[1]]
                zt = Z_top[:, edge[1]]
                zb = Z_bottom[:, edge[1]]
            verts = [list(zip(xs, ys, zb)) + list(zip(xs[::-1], ys[::-1], zt[::-1]))]
            ax.add_collection3d(Poly3DCollection(verts, facecolor='black', edgecolor=None))
            if plot_different_orient:
                ax2.add_collection3d(Poly3DCollection(verts, facecolor='black', edgecolor=None))
        
        # View
        ax.view_init(elev=35, azim=35)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_box_aspect([1,1,0.3])
        plt.show()
        ax.axis('off')
        if plot_different_orient:
            ax2.view_init(elev=40, azim=35)
            ax2.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax2.set_box_aspect([1,1,0.3])
            plt.show()
            ax2.axis('off')
    if save:
        print('Saving PNG')
        fig.savefig(SV_FOLDER + 'cartoon_neural_sheet.png', dpi=200, bbox_inches='tight')
        print('Saving pdf')
        fig.savefig(SV_FOLDER + 'cartoon_neural_sheet.pdf', dpi=200, bbox_inches='tight')


def plot_cool_factor_graph():
    print('Plotting factor graph')
    # ----------------- Parameters -----------------
    n_neurons_side = 4
    neuron_radius = 0.2
    plane_z = 0.0
    stimulus_z = -5.0
    square_size = 0.1
    square_offset = -3.5  # below neuron plane
    connection_fraction = 0.18  # 20% of neighbor pairs
    np.random.seed(14)
    
    # ----------------- Figure -----------------
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    # ----------------- Neuron positions -----------------
    x_neurons = np.linspace(-1, 1, n_neurons_side)
    y_neurons = np.linspace(-1, 1, n_neurons_side)
    xx, yy = np.meshgrid(x_neurons, y_neurons)
    xx = xx.flatten()
    yy = yy.flatten()
    neuron_positions = list(zip(xx, yy))
    
    # ----------------- Draw neurons -----------------
    def draw_circle(ax, x0, y0, z0, r, n_points=50, facecolor='white', edgecolor='black'):
        theta = np.linspace(0, 2*np.pi, n_points)
        xs = x0 + r * np.cos(theta)
        ys = y0 + r * np.sin(theta)
        zs = np.full_like(xs, z0)
        verts = [list(zip(xs, ys, zs))]
        collection = Poly3DCollection(verts, facecolor=facecolor, edgecolor=edgecolor,
                                      linewidth=2)
        ax.add_collection3d(collection)
    
    for x, y in neuron_positions:
        draw_circle(ax, x, y, plane_z, neuron_radius)
    
    # ----------------- Neighbor pairs -----------------
    pairs = []
    N = n_neurons_side
    for i in range(N):
        for j in range(N):
            idx = i*N + j
            # right neighbor
            if j < N-1:
                idx2 = i*N + (j+1)
                pairs.append((idx, idx2))
            # down neighbor
            if i < N-1:
                idx2 = (i+1)*N + j
                pairs.append((idx, idx2))
    
    # ----------------- Randomly select 10% of pairs -----------------
    n_select = max(1, int(len(pairs) * connection_fraction))
    selected_pairs = np.random.choice(len(pairs), n_select, replace=False)
    selected_pairs = [pairs[i] for i in selected_pairs]
    
    
    # ----------------- Stimulus plane -----------------
    xs = np.linspace(-1.5, 1.5, 2)
    ys = np.linspace(-1.5, 1.5, 2)
    XS, YS = np.meshgrid(xs, ys)
    ZS = np.full_like(XS, stimulus_z)
    ax.plot_surface(XS, YS, ZS, color='lightgray', alpha=0.1)
    # Top edge
    ax.plot(XS[0, :], YS[0, :], ZS[0, :], color='black', linewidth=3)
    
    # Bottom edge
    ax.plot(XS[-1, :], YS[-1, :], ZS[-1, :], color='black', linewidth=3)
    
    # Left edge
    ax.plot(XS[:, 0], YS[:, 0], ZS[:, 0], color='black', linewidth=3)
    
    # Right edge
    ax.plot(XS[:, -1], YS[:, -1], ZS[:, -1], color='black', linewidth=3)
    
    # ----------------- Neurons plane -----------------
    xs = np.linspace(-1.5, 1.5, 2)
    ys = np.linspace(-1.5, 1.5, 2)
    XS, YS = np.meshgrid(xs, ys)
    ZS = np.full_like(XS, 0)
    ax.plot_surface(XS, YS, ZS, color='white', alpha=0.)
    # Top edge
    ax.plot(XS[0, :], YS[0, :], ZS[0, :], color='black', linewidth=3)
    
    # Bottom edge
    ax.plot(XS[-1, :], YS[-1, :], ZS[-1, :], color='black', linewidth=3)
    
    # Left edge
    ax.plot(XS[:, 0], YS[:, 0], ZS[:, 0], color='black', linewidth=3)
    
    # Right edge
    ax.plot(XS[:, -1], YS[:, -1], ZS[:, -1], color='black', linewidth=3)

    # ----------------- Draw squares and connecting lines -----------------
    for idx1, idx2 in selected_pairs:
        x1, y1 = neuron_positions[idx1]
        x2, y2 = neuron_positions[idx2]
    
        # midpoint square position
        xm = (x1 + x2)/2
        ym = (y1 + y2)/2
        zm = plane_z + square_offset
        xm += np.random.uniform(-0.1, 0.1)
        ym += np.random.uniform(-0.1, 0.1)
    
        # square polygon
        s = square_size / 2
        verts = [
            (xm, ym-s, zm-2*s),
            (xm, ym+s, zm-2*s),
            (xm, ym+s, zm+2*s),
            (xm, ym-s, zm+2*s)
        ]
        square = Poly3DCollection([verts], facecolor='white', edgecolor='black')
        ax.add_collection3d(square)
    
        # diagonal lines from neurons to square
        ax.plot([x1, xm], [y1, ym], [plane_z, zm], color='black', linewidth=3)
        ax.plot([x2, xm], [y2, ym], [plane_z, zm], color='black', linewidth=3)
    
        # vertical line from square to stimulus plane
        ax.plot([xm, xm], [ym, ym], [zm, stimulus_z], color='blue', linewidth=3)
    
    
    # ----------------- Figure adjustments -----------------
    ax.set_box_aspect([1,1,0.5])
    ax.view_init(elev=30, azim=120)
    ax.dist = 15
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    plt.show()
    ax.axis('off')


def cartoon_2d_factor_graph():
    from matplotlib.patches import Circle, Rectangle

    np.random.seed(4)
    
    # ---------------------
    #  Parameters
    # ---------------------
    N = 4  # neurons per side
    neuron_r = 0.15
    square_size = 0.25
    plane_y = 1.0        # top plane (neurons)
    stim_y = -1.5        # bottom plane (stimulus)
    square_offset = -0.6 # relative to top plane
    connection_fraction = 0.1  # 10% of pairs
    
    # ---------------------
    #  Neuron grid positions
    # ---------------------
    xs = np.linspace(-2, 2, N)
    ys = np.linspace(-2, 2, N)
    XX, YY = np.meshgrid(xs, ys)
    neuron_positions = list(zip(XX.flatten(), YY.flatten()))
    
    # ---------------------
    #  Build neighbor pairs
    # ---------------------
    pairs = []
    for i in range(N):
        for j in range(N):
            idx = i*N + j
            # right neighbor
            if j < N-1:
                pairs.append((idx, idx+1))
            # down neighbor
            if i < N-1:
                pairs.append((idx, idx+N))
    
    # Pick ~10% at random
    k = max(1, int(len(pairs)*connection_fraction))
    chosen = np.random.choice(len(pairs), k, replace=False)
    chosen_pairs = [pairs[i] for i in chosen]
    
    # ---------------------
    #  Plot
    # ---------------------
    fig, ax = plt.subplots(figsize=(8,8))
    
    # ---- Draw neurons ----
    for (x,y) in neuron_positions:
        circ = Circle((x, plane_y+y*0), neuron_r, 
                      facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(circ)
    
    # ---- Draw chosen pair squares & connecting lines ----
    for idx1, idx2 in chosen_pairs:
        x1, y1 = neuron_positions[idx1]
        x2, y2 = neuron_positions[idx2]
    
        # Midpoint (square position)
        xm = 0.5*(x1 + x2)
        ym = plane_y + square_offset
    
        # square
        rect = Rectangle((xm-square_size/2, ym-square_size/2),
                         square_size, square_size,
                         facecolor='lightgray', edgecolor='black')
        ax.add_patch(rect)
    
        # Diagonal lines to square
        ax.plot([x1, xm], [plane_y, ym], color='black', linewidth=3)
        ax.plot([x2, xm], [plane_y, ym], color='black', linewidth=3)
    
        # Vertical line to stimulus plane
        ax.plot([xm, xm], [ym, stim_y], color='blue', linewidth=3)
    
    # ---- Draw stimulus plane ----
    ax.plot([-2.5, 2.5], [stim_y, stim_y], color='black', linewidth=3)
    ax.text(0, stim_y-0.2, "Stimulus Plane", ha='center', fontsize=14)
    
    # ---- Cosmetics ----
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.show()


def assign_blocks(df, block_size=10):
    df = df.copy()
    block = df["trial_index"] // block_size + 1
    block[block == 12] = 11.6
    df["block"] = block
    return df


def extract_switches(df, sample_dt=0.01667):
    """
    df: behavioral dataframe with columns:
        - 'trial_index'
        - 'responses' (continuous keypress per timestep)
    sample_dt: duration of each timestep in seconds
    """
    rows = []
    # iterate over trials
    for trial_idx, trial_df in df.groupby("trial_index"):
        responses = trial_df["responses"].values
        if len(responses) < 2:
            continue
        
        # find switches
        changes = np.where(responses[1:] != responses[:-1])[0] + 1
        
        for c in changes:
            t_switch = c * sample_dt  # time relative to trial start
            rows.append({
                "trial_index": trial_idx,
                "t_switch": t_switch,
                "from": responses[c-1],
                "to": responses[c]
            })
    
    return pd.DataFrame(rows)


def plot_regression_subjects(data_folder=DATA_FOLDER,
                             freq=2, noisy=False,
                             t_before=1.25, t_after=1.25, compute=True,
                             dt=1/60, align=True, pshuffle='all',
                             pupil_col='Pupil_residual'):
    if pshuffle != 'all':
        extra_pshub = f'pshuf_{pshuffle}_'
    else:
        extra_pshub = ''
    if noisy:
        name_matrix = extra_pshub + 'noise_trials_weight_matrix_linear_regression.npy'
        name_intercept = extra_pshub + 'noise_trials_intercept_linear_regression.npy'
    else:
        name_matrix = extra_pshub + f'hysteresis_freq_{freq}_weight_matrix_linear_regression.npy'
        name_intercept = extra_pshub + f'hysteresis_freq_{freq}_intercept_linear_regression.npy'
    path_shuffle_matrix = os.path.join(
                        data_folder,
                        'aligned_eye_tracker_data',
                        name_matrix
                        )
    path_stim_matrix = os.path.join(
        data_folder,
        'aligned_eye_tracker_data',
        'regression_weights',
        'stim_' + name_matrix
        )
    path_abs_stim_matrix = os.path.join(
        data_folder,
        'aligned_eye_tracker_data',
        'regression_weights',
        'abs_stim_' + name_matrix
        )
    path_posterior_matrix = os.path.join(
        data_folder,
        'aligned_eye_tracker_data',
        'regression_weights',
        'posterior_' + name_matrix
        )
    path_intercept = os.path.join(
        data_folder,
        'aligned_eye_tracker_data',
        'regression_weights',
        name_intercept
        )
    if compute:
        if noisy:
            # --- Load behavioral data ---
            df_all = load_data(data_folder=data_folder + '/noisy/', n_participants='all', filter_subjects=True)
        else:
            df_all = load_data(data_folder=data_folder, n_participants='all', filter_subjects=True)
            if freq != 'all':
                df_all = df_all.loc[df_all.freq == freq]
        if pshuffle != 'all':
            df_all = df_all.loc[df_all.pShuffle == pshuffle]
        sublist = df_all.subject.unique()
        shuffle_weights_matrix = []
        stim_weights_matrix = []
        abs_stim_weights_matrix = []
        posterior_weights_matrix = []
        intercept_all = []
        for i_sub, sub in enumerate(sublist):
            print('Subject ' + sub)
            beta, pval, t, intercept = pupil_regression_raw_switches(
                    data_folder=DATA_FOLDER,
                    df_all=df_all,
                    sublist=[sub],
                    t_before=t_before,
                    t_after=t_after, 
                    noisy=noisy,
                    align=align,
                    regressors=('effective_coupling', 'abs_stimulus', 'stimulus'), dt=dt,
                    pupil_col=pupil_col)
            shuffle_weights_matrix.append(beta['effective_coupling'])
            stim_weights_matrix.append(beta['stimulus'])
            abs_stim_weights_matrix.append(beta['abs_stimulus'])
            # posterior_weights_matrix.append(beta['posterior'])
            intercept_all.append(intercept)
        np.save(path_shuffle_matrix, shuffle_weights_matrix)
        np.save(path_stim_matrix, stim_weights_matrix)
        np.save(path_abs_stim_matrix, abs_stim_weights_matrix)
        # np.save(path_posterior_matrix, posterior_weights_matrix)
        np.save(path_intercept, intercept_all)
    else:
        shuffle_weights_matrix = np.load(path_shuffle_matrix, allow_pickle=True)
        stim_weights_matrix = np.load(path_stim_matrix, allow_pickle=True)
        abs_stim_weights_matrix = np.load(path_abs_stim_matrix, allow_pickle=True)
        intercept_all = np.load(path_intercept, allow_pickle=True)
        # posterior_weights_matrix = np.load(path_posterior_matrix, allow_pickle=True)
    fig, ax = plt.subplots(ncols=4, figsize=(12, 4), sharey=True)
    if align:
        t_bins = np.arange(-t_before, t_after + dt, dt)
    else:
        t_bins = np.arange(0, 26 + dt, dt)
    t = t_bins[:-1] + dt / 2
    weights_matrices = np.array([intercept_all, shuffle_weights_matrix, stim_weights_matrix,
                                 abs_stim_weights_matrix])  # posterior_weights_matrix
    labels = ['Intercept', 'p(Shuffle)', 'Stimulus', '|Stimulus|', 'posterior']
    for i_a, a in enumerate(ax):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.axhline(0, color='gray', linestyle='--', alpha=0.4, linewidth=2)
        a.axvline(0, color='gray', linestyle='--', alpha=0.4, linewidth=2)
        a.plot(t, np.nanmean(weights_matrices[i_a], axis=0), color='k',
                linewidth=4)
        # for sub in range(weights_matrices[i_a].shape[0]):
        #     a.plot(t, weights_matrices[i_a][sub], color='k', alpha=0.2)
        # significant = []
        # for i in range(len(t)):
        #     weights_t = weights_matrices[i_a][:, i]
        #     _, p = scipy.stats.ttest_1samp(weights_t, popmean=0)
        #     significant.append(p < 0.05)
        # a.fill_between(t, -0.6, 0.6, where=significant, color='gray', alpha=0.2)
        n_boot = 1000
        boot_means = np.zeros((n_boot, len(t)))
        beta = weights_matrices[i_a]
        for b in range(n_boot):
            sample_idx = np.random.choice(beta.shape[0], beta.shape[0], replace=True)
            boot_means[b] = np.nanmean(beta[sample_idx], axis=0)

        lower = np.percentile(boot_means, 2.5, axis=0)
        upper = np.percentile(boot_means, 97.5, axis=0)
        a.fill_between(t, lower, upper, color='gray', alpha=0.3)

        a.set_xlabel('Time from switch (s)')
        a.set_ylabel(labels[i_a])
    fig.suptitle('Regression weights (a.u.)', fontsize=15)
    fig.tight_layout()


def linear_reg_behavior(delta=False):
    data_dominance = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
    data_hysteresis_2 = np.load(DATA_FOLDER + 'hysteresis_width_freq_2.npy')
    data_hysteresis_4 = np.load(DATA_FOLDER + 'hysteresis_width_freq_4.npy')
    save_path = os.path.join(DATA_FOLDER, 'aligned_eye_tracker_data','plots', 'min_pupil_across_trials.npy')
    min_pupil = np.load(save_path)
    variables = [data_dominance, data_hysteresis_2, data_hysteresis_4]
    pshuffle = (1-np.array([0, 0.7, 1]))
    n = len(variables)
    nshufs, nsubs = data_dominance.shape
    coef_vars = np.zeros((n, nsubs))
    for i_var, var in enumerate(variables):
        coefs = []
        for sub in range(nsubs):
            if delta:
                var_z = var[:, sub]
                coefs.append(var_z[2]-var_z[0])
            else:
                linreg = LinearRegression(fit_intercept=True)
                var_z = zscore(var[:, sub])
                coefs.append(linreg.fit(pshuffle.reshape(-1, 1), var_z.reshape(-1, 1)).coef_[0][0])

        coef_vars[i_var] = coefs

    corr_matrix = np.full((n, n), np.nan)
    p_matrix = np.full((n, n), np.nan)
    annot = np.empty((n, n), dtype=object)
    annot[:] = ""
    for i in range(n):
        var_i = coef_vars[i]
    
        for j in range(n):
            if i == j:
                continue
            r, p = pearsonr(var_i, coef_vars[j])
            corr_matrix[i, j] = r
            p_matrix[i, j] = p
            annot[i, j] = stars(p)
    fig, ax = plt.subplots(1, figsize=(8, 6))
    labs = ['Dom.', 'Hyst. f2', 'f4', 'pupil']
    ax.set_xticks(np.arange(len(labs)), labs, rotation=45)
    ax.set_yticks(np.arange(len(labs)), labs, rotation=0)
    fig.tight_layout()
    sns.heatmap(corr_matrix, cmap="bwr",
                vmin=-1, vmax=1,
                cbar_kws={"label": "Correlation"},
                fmt="", annot=annot)
    ax.set_xticks(np.arange(len(labs))+0.5, labs, rotation=45)
    ax.set_yticks(np.arange(len(labs))+0.5, labs, rotation=0)


def linear_reg_comparison(delta=False):
    data_dominance = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
    data_amplitude = np.load(DATA_FOLDER + 'mean_peak_amplitude_per_subject.npy')
    data_hysteresis_2 = np.load(DATA_FOLDER + 'hysteresis_width_freq_2.npy')
    data_hysteresis_4 = np.load(DATA_FOLDER + 'hysteresis_width_freq_4.npy')
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    min_pupil = get_eye_tracker_data_across_trials(data_folder=DATA_FOLDER,
                                                   pupil_col='Pupil_residual',
                                                   align=True, flatten=False)
    saccade_baseline = get_eye_tracker_data_across_trials(data_folder=DATA_FOLDER,
                                           pupil_col='saccade',
                                           align=False, flatten=False)
    saccade_max = get_eye_tracker_data_across_trials(data_folder=DATA_FOLDER,
                                           pupil_col='saccade',
                                           align=True, flatten=False)
    blink_baseline = get_eye_tracker_data_across_trials(data_folder=DATA_FOLDER,
                                           pupil_col='blink',
                                           align=False, flatten=False)
    blink_max = get_eye_tracker_data_across_trials(data_folder=DATA_FOLDER,
                                           pupil_col='blink',
                                           align=True, flatten=False)
    variables = [data_dominance, data_hysteresis_2, data_hysteresis_4,
                 min_pupil, saccade_baseline, saccade_max,
                 blink_baseline, blink_max]
    pshuffle = 1-np.array([0, 0.7, 1])
    n = len(variables)
    nshufs, nsubs = data_dominance.shape
    coef_vars = np.zeros((n, nsubs))
    for i_var, var in enumerate(variables):
        coefs = []
        for sub in range(nsubs):
            if delta:
                var_z = var[:, sub]
                coefs.append(var_z[2]-var_z[0])
            else:
                linreg = LinearRegression(fit_intercept=True)
                var_z = zscore(var[:, sub])
                coefs.append(linreg.fit(pshuffle.reshape(-1, 1), var_z.reshape(-1, 1)).coef_[0][0])

        coef_vars[i_var] = coefs
    
    corr_matrix = np.full((n, n), np.nan)
    p_matrix = np.full((n, n), np.nan)
    annot = np.empty((n, n), dtype=object)
    annot[:] = ""
    for i in range(n):
        var_i = coef_vars[i]
    
        for j in range(n):
            if i == j:
                continue
            r, p = pearsonr(var_i, coef_vars[j])
            corr_matrix[i, j] = r
            p_matrix[i, j] = p
            annot[i, j] = stars(p)
    fig, ax = plt.subplots(1, figsize=(8, 6))
    labs = ['Dom.', 'Hyst. f2', 'f4', 'Min pupil',
     'sacc. base', 'sacc. max', 'blink base', 'blink max']
    ax.set_xticks(np.arange(len(labs)), labs, rotation=45)
    ax.set_yticks(np.arange(len(labs)), labs, rotation=0)
    fig.tight_layout()
    sns.heatmap(corr_matrix, cmap="bwr",
                vmin=-1, vmax=1,
                cbar_kws={"label": "Correlation"},
                fmt="", annot=annot)
    ax.set_xticks(np.arange(len(labs))+0.5, labs, rotation=45)
    ax.set_yticks(np.arange(len(labs))+0.5, labs, rotation=0)


def plot_hyst_dom_correlation(data_folder=DATA_FOLDER):
    variable_eye = np.load(data_folder + 'mean_number_switches_per_subject.npy')
    xlabel = 'Dominance (s)'
    
    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(2.6, 4.6), sharex=True, sharey=False)
    for i_f, freq in enumerate([2, 4]):
        ax = axes[i_f]
        variable = np.load(data_folder + f'hysteresis_width_freq_{freq}.npy')
        ylabel = f'Hysteresis, {freq // 2}-C'
        avg_eye = np.nanmean(variable_eye, axis=0)
        avg_beh = np.nanmean(variable, axis=0)
        X = avg_eye.reshape(-1, 1)
        y = avg_beh.reshape(-1, 1)
        linreg = LinearRegression(fit_intercept=True).fit(X, y)
        minmax_array = np.array([np.min(X)-0.1, np.max(X)+0.1]).reshape(-1, 1)
        pred_y = linreg.predict(minmax_array)
        r, p = pearsonr(avg_eye, avg_beh)
        ax.annotate(f'r = {r:.3f}\np={p:.2e}', xy=(.04, 0.8), xycoords=ax.transAxes, fontsize=14)
        ax.plot(minmax_array,
               pred_y, color='gray', linestyle='--', alpha=0.4, linewidth=3)
        ax.plot(avg_eye, avg_beh, marker='o', color='k',
               linestyle='')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel(ylabel)
    axes[1].set_xlabel(xlabel)
    fig.tight_layout()
    for extension in ['.png', '.svg', '.pdf']:
        plot_name = 'hysteresis_vs_dominance' + extension
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'behavioral_correlates',  plot_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=400, bbox_inches='tight')


def analytical_hysteresis_width_degeneration(n=4, freq=2, fps=60):
    coup_vals = np.array([0., 0.3, 1.])
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    print(len(pars), ' fitted subjects')
    j1s = np.array([np.load(par)[0] for par in pars])
    j0s = np.array([np.load(par)[1] for par in pars])
    b1s = np.array([np.load(par)[2] for par in pars])
    sigmas = np.array([np.load(par)[3] for par in pars])
    diffusion = 0.5*sigmas**2
    hyst_width_analytical = []
    ds = np.diff(get_blist(freq, 1560, 3))[0] * fps
    
    if freq == 2:
        hyst_width_data = np.load(DATA_FOLDER + 'difference_b_hysteresis_width_freq_2.npy')
    else:
        hyst_width_data = np.load(DATA_FOLDER + 'difference_b_hysteresis_width_freq_4.npy')
    for i in range(len(j1s)):
        j_list = j1s[i]*coup_vals + j0s[i]
        areas_all = []
        for j in j_list:
            if j*n <= 1:
                area = 2*ds/(1-j*n)*10
                area = np.nan
            else:
                # delta = np.sqrt(1-1/(j*n))
                # area = (np.log((1-delta)/(1+delta))+2*n*j*delta)  #  + ds*0.5*fps
                area = kramers_width(j, n, diffusion[i], ds,
                              Bmin=-3*b1s[i], Bmax=3*b1s[i], nscan=400)
            areas_all.append(area)
        hyst_width_analytical.append(areas_all)
    hyst_width_analytical = np.stack(hyst_width_analytical).T
    # hyst_width_analytical[np.isnan(hyst_width_analytical)] = 0.5
    fig, ax = plt.subplots(1, figsize=(4, 3.5))
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    for i in range(3):
        plt.plot(hyst_width_analytical[i], hyst_width_data[i], marker='o',
                 linestyle='', color=colormap[i])
    an = hyst_width_analytical.flatten()
    hyst_width_data = hyst_width_data.flatten()
    idx_nan = np.isnan(an) + np.isnan(hyst_width_data) + np.isinf(an) + np.isinf(hyst_width_data)
    r, p = pearsonr(an[~idx_nan], hyst_width_data[~idx_nan])
    ax.annotate(f'r = {r:.3f}\np={p:.2e}', xy=(.04, 0.8), xycoords=ax.transAxes)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Analytical hysteresis width')
    ax.set_ylabel('Empirical hysteresis width')
    fig.tight_layout()


if __name__ == '__main__':
    print('Running hysteresis_analysis_part.py')
    # Example entry points (uncomment to run):
    # plot_hysteresis_average(tFrame=26, fps=60, data_folder=DATA_FOLDER,
    #                         ntraining=8, coupling_levels=[0, 0.3, 1],
    #                         window_conv=None, ndt_list=None)
    # hysteresis_basic_plot(coupling_levels=[0, 0.3, 1], fps=60, tFrame=26,
    #                       data_folder=DATA_FOLDER, nbins=10, ntraining=8,
    #                       arrows=False, subjects=['s_23'], window_conv=None)
    # plot_switch_rate(tFrame=26, fps=60, data_folder=DATA_FOLDER, ntraining=8,
    #                  coupling_levels=[0, 0.3, 1], window_conv=5, bin_size=0.35,
    #                  switch_01=False)
    # plot_dominance_durations(data_folder=DATA_FOLDER, ntraining=8, freq=2, sem=False)
