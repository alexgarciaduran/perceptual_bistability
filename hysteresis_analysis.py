# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 20:07:39 2025

@author: alexg
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import matplotlib as mpl
import matplotlib.pylab as pl
from matplotlib.lines import Line2D
import glob
from sklearn.metrics import roc_curve, auc
from gibbs_necker import rle
from mean_field_necker import colored_line
from scipy.optimize import fsolve, minimize
from pybads import BADS
import sbi
from sbi.inference import infer
from sbi.utils import MultipleIndependent
import torch
from torch.distributions import Uniform
import time
import pickle
from sbi import analysis as analysis

mpl.rcParams['font.size'] = 16
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize']= 14
plt.rcParams['ytick.labelsize']= 14


DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/hysteresis/data/'  # Alex
SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/hysteresis/parameters/'  # Alex


def load_data(data_folder, n_participants='all'):
    files = glob.glob(data_folder + '*.csv')
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


def get_response_and_blist_array(df, fps=60, tFrame=20):
    nFrame = fps*tFrame
    stimulus_times = np.arange(0, nFrame)
    trial_index = df.trial_index.unique()
    response_array = np.zeros((len(trial_index), nFrame))
    difficulty_time_ref_2 = np.linspace(-2, 2, nFrame//2)
    blist_freq2_ref = np.concatenate(([difficulty_time_ref_2, -difficulty_time_ref_2]))
    difficulty_time_ref_4 = np.linspace(-2, 2, nFrame//4)
    blist_freq4_ref = np.concatenate(([difficulty_time_ref_4, -difficulty_time_ref_4,
                                       difficulty_time_ref_4, -difficulty_time_ref_4]))
    freq = df.freq.unique()[0]
    map_resps = {1:1, 0:np.nan, 2:0}
    for idx_ti, ti in enumerate(trial_index):
        df_filt = df.loc[df.trial_index == ti]
        # times_change = np.int32(df_filt.keypress_seconds_offset.values/tFrame*nFrame)
        times_onset = np.int32(df_filt.keypress_seconds_onset.values/tFrame*nFrame)
        switch_times = np.concatenate((times_onset, [nFrame]))
        responses = df_filt.response.values
        responses = [map_resps[r] for r in responses]
        if df_filt.loc[df.trial_index == ti, 'initial_side'].unique()[0] > 0:
            responses = responses[::-1]
        response_array[idx_ti, :] = responses[0] if len(responses) > 1 else responses
        if len(times_onset) > 1:
            response_series = np.array(())
            for i in range(len(switch_times)-1):
                if switch_times[i+1] < 40:
                    resp_i = np.nan
                else:
                    resp_i = responses[i]
                mask = (stimulus_times >= switch_times[i]) & (stimulus_times < switch_times[i+1])
                response_series = np.concatenate((response_series, [resp_i] * np.sum(mask)))
            response_array[idx_ti, :] = response_series
    b_array = blist_freq2_ref if freq == 2 else blist_freq4_ref
    return response_array, b_array


def bin_average_response(stim_time, response_array, nbins=11):
    ascending = np.sign(np.gradient(stim_time))
    asc_mask = ascending == 1
    desc_mask = ~asc_mask
    binned_responses_trial = np.zeros((response_array.shape[0], nbins*2))
    for i_trial, responses in enumerate(response_array):
        asc_responses = responses[asc_mask]
        desc_responses = responses[desc_mask]
        df_2 = pd.DataFrame({'stimulus': stim_time, 'ascending': ascending == 1,
                             'responses': responses})
        df_2['stim_bin'] = pd.cut(df_2['stimulus'], bins=nbins)
        yvals_2 = df_2.groupby(['ascending', 'stim_bin'])['responses'].mean().values
        xvals_2 = df_2.groupby(['ascending', 'stim_bin'])['stimulus'].mean().values
        binned_responses_trial[i_trial] = yvals_2
    xvals_2[len(xvals_2)//2:] = xvals_2[len(xvals_2)//2:][::-1]
    xvals_final = xvals_2
    return xvals_2, np.mean(binned_responses_trial, axis=0)


# def plot_levelts(fps=60, tFrame=18, data_folder=DATA_FOLDER,
#                  ntraining=10):
#     nFrame = fps*tFrame
#     df = load_data(data_folder, n_participants='all')
#     df = df.loc[df.trial_index <= ntraining]
#     subjects = df.subject.unique()
#     fig, ax = plt.subplots(ncols=2, nrows=len(subjects), figsize=(8, 4.5))
#     for a in ax:
#         a.spines['right'].set_visible(False)
#         a.spines['top'].set_visible(False)
#         a.set_ylim(-0.025, 1.025)
#     colormap = pl.cm.Oranges(np.linspace(0.3, 1, 3))[::-1]
#     for i_s, subject in enumerate(subjects):
#         df_sub = df.loc[df.subject == subject]
#         evidence_values = np.sort(df_sub.signed_evidence.unique())
#         times_0 = []
#         times_1 = []
#         r_1 = []
#         for i_ev, evidence in enumerate(evidence_values):
#             df_ev = df_sub.loc[df_sub.signed_evidence == evidence]
#             responses, _ = get_response_and_blist_array(df_ev, fps=fps,
#                                                         tFrame=tFrame)
#             df_ev['times_diff'] = df_ev.keypress_seconds_offset-df_ev.keypress_seconds_onset
#             # df_ev = df_ev.loc[df_ev.keypress_seconds_offset < tFrame-2/fps]
#             times = df_ev.groupby(['trial_index', 'response']).times_diff.mean()
#             vals = times.groupby('response').mean().values[1:]
#             times_0.append(vals[0])
#             times_1.append(vals[1])
#             r_1.append(np.nanmean(responses, axis=1).mean())


def hysteresis_basic_plot(coupling_levels=[0, 0.3, 1],
                          fps=60, tFrame=18, data_folder=DATA_FOLDER,
                          nbins=13, ntraining=4, arrows=False):
    nFrame = fps*tFrame
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()
    fig, ax = plt.subplots(ncols=2, nrows=len(subjects), figsize=(7.5, 4.))
    fig2, ax2 = plt.subplots(1, figsize=(4.5, 4))
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylim(-0.025, 1.085)
        a.set_yticks([0, 0.5, 1])
        a.set_xlim(-2.05, 2.05)
        a.set_xticks([-2, 0, 2], [-1, 0, 1])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)    
    colormap = pl.cm.Oranges(np.linspace(0.3, 1, 3))[::-1]
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    for i_s, subject in enumerate(subjects):
        df_sub = df.loc[df.subject == subject]
        hvals_4 = []
        hvals_2 = []
        for i_c, coupling in enumerate(coupling_levels):
            df_filt = df_sub.loc[df_sub.pShuffle.round(2) == round(1-coupling, 2)]
            df_freq_2 = df_filt.loc[df.freq == 2]
            response_array_2, barray_2 = get_response_and_blist_array(df_freq_2, fps=fps,
                                                                      tFrame=tFrame)
            # response_array_2 = np.roll(response_array_2, -120, axis=1)
            # mean_response_2 = np.nanmean(response_array_2, axis=0)
            ascending = np.sign(np.gradient(barray_2))
            df_to_plot = pd.DataFrame(response_array_2.T)
            df_to_plot['stimulus'] = barray_2
            df_to_plot['ascending'] = ascending
            df_to_plot['stim_bin'] = pd.cut(df_to_plot['stimulus'], bins=nbins)
            df_to_plot = df_to_plot.drop('stimulus', axis=1)
            yvals_2 = np.nanmean(df_to_plot.groupby(['ascending', 'stim_bin']).mean().values, axis=1)
            df_to_plot['stimulus'] = barray_2
            xvals_2 = df_to_plot.groupby(['ascending', 'stim_bin'])['stimulus'].mean().values
            idx_close_ascending = np.argmin(np.abs(yvals_2[len(xvals_2)//2:]-0.5))
            idx_close_descending = np.argmin(np.abs(yvals_2[:len(xvals_2)//2]-0.5))
            hist_val_2 = xvals_2[len(xvals_2)//2:][idx_close_ascending]-xvals_2[:len(xvals_2)//2][idx_close_descending]
            hvals_2.append(hist_val_2)
            ax2.plot(coupling, hist_val_2, marker='o', markersize=8, color=colormap[i_c],
                     zorder=5)
            ax[0].plot(xvals_2[len(xvals_2)//2:], yvals_2[len(xvals_2)//2:], color=colormap[i_c], linewidth=4,
                       label=1-coupling)
            ax[0].plot(xvals_2[:len(xvals_2)//2], yvals_2[:len(xvals_2)//2], color=colormap[i_c], linewidth=4)
            df_freq_4 = df_filt.loc[df.freq == 4]
            response_array_4, barray_4 = get_response_and_blist_array(df_freq_4, fps=fps,
                                                                      tFrame=tFrame)
            # response_array_4 = np.roll(response_array_4, -30, axis=1)
            # mean_response_4 = np.nanmean(response_array_4, axis=0)
            # response_array_4_1st = response_array_4[:, :len(barray_4)//4]
            # mean_1st = np.nanmean(response_array_4_1st, axis=1)
            # response_array_4_2nd = response_array_4[:, len(barray_4)//4:len(barray_4)//2]
            # mean_2nd = np.nanmean(response_array_4_2nd, axis=1)
            # response_array_4_3rd = response_array_4[:, len(barray_4)//2:3*len(barray_4)//4]
            # mean_3rd = np.nanmean(response_array_4_3rd, axis=1)
            # response_array_4_4th = response_array_4[:, 3*len(barray_4)//4:]
            # mean_4th = np.nanmean(response_array_4_4th, axis=1)
            ascending = np.sign(np.gradient(barray_4))
            df_to_plot = pd.DataFrame(response_array_4.T)
            df_to_plot['stimulus'] = barray_4
            df_to_plot['ascending'] = ascending
            df_to_plot['stim_bin'] = pd.cut(df_to_plot['stimulus'], bins=nbins)
            df_to_plot = df_to_plot.drop('stimulus', axis=1)
            yvals_4 = np.nanmean(df_to_plot.groupby(['ascending', 'stim_bin']).mean().values, axis=1)
            df_to_plot['stimulus'] = barray_4
            xvals_4 = df_to_plot.groupby(['ascending', 'stim_bin'])['stimulus'].mean().values
            idx_close_ascending = np.argmin(np.abs(yvals_4[len(xvals_4)//2:]-0.5))
            idx_close_descending = np.argmin(np.abs(yvals_4[:len(xvals_4)//2]-0.5))
            hist_val_4 = xvals_4[len(xvals_4)//2:][idx_close_ascending]-xvals_4[:len(xvals_4)//2][idx_close_descending]
            hvals_4.append(hist_val_4)
            ax[1].plot(xvals_4[len(xvals_4)//2:], yvals_4[len(xvals_4)//2:],
                       color=colormap[i_c], linewidth=4, label=1-coupling,
                       linestyle='--')
            ax[1].plot(xvals_4[:len(xvals_4)//2],
                       yvals_4[:len(xvals_4)//2], color=colormap[i_c], linewidth=4,
                       linestyle='--')
            ax2.plot(coupling, hist_val_4, marker='o', markersize=8, color=colormap[i_c],
                     zorder=5)
            if coupling == 1 and arrows:
                ax[0].annotate(text='', xy=(-hist_val_2+0.25, 1.06),
                               xytext=(hist_val_2-0.25, 1.06),
                               arrowprops=dict(arrowstyle='<->', color=colormap[i_c],
                                               linewidth=3))
                ax[1].annotate(text='', xy=(-hist_val_4+0.6, 1.06),
                               xytext=(hist_val_4-0.5, 1.06),
                               arrowprops=dict(arrowstyle='<->', color=colormap[i_c],
                                               linewidth=3))
                ax[0].axhline(0.5, color=colormap[i_c], alpha=0.7, linestyle='--')
                ax[1].axhline(0.5, color=colormap[i_c], alpha=0.7, linestyle='--')
                ax[0].plot([-hist_val_2+0.25, -hist_val_2+0.25],
                           [0.5, 1.06], color=colormap[i_c], alpha=0.7, linestyle='--')
                ax[0].plot([hist_val_2-0.25]*2,
                           [0.5, 1.06], color=colormap[i_c], alpha=0.7, linestyle='--')
                ax[1].plot([-hist_val_4+0.62]*2,
                           [0.5, 1.06], color=colormap[i_c], alpha=0.7, linestyle='--')
                ax[1].plot([hist_val_4-0.53]*2,
                           [0.5, 1.06], color=colormap[i_c], alpha=0.7, linestyle='--')
            # ax[1].plot(barray_4, mean_response_4, color=colormap[i_c], linewidth=3)
        ax2.plot(coupling_levels, hvals_2, color='k', linewidth=3,
                 linestyle='--', zorder=1, label='Freq = 2')
        ax2.plot(coupling_levels, hvals_4, color='k', linewidth=3,
                 zorder=1, label='Freq = 4')
    ax2.legend(frameon=False)
    ax2.set_xlabel('Coupling, J')
    ax2.set_ylabel('Hysteresis width')
    ax[0].set_xlabel('Sensory evidence, B(t)')
    ax[1].set_xlabel('Sensory evidence, B(t)')
    ax[0].set_ylabel('Proportion of rightward responses')
    ax[0].legend(title='p(shuffle)', frameon=False)
    ax[0].set_title('Freq = 2', fontsize=14)
    ax[1].set_title('Freq = 4', fontsize=14)
    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig(SV_FOLDER + 'hysteresis_basic_plot.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'hysteresis_basic_plot.svg', dpi=400, bbox_inches='tight')


def hysteresis_basic_plot_simulation(coup_vals=np.array((0., 0.3, 1))*0.35+0.02,
                                     fps=60,
                                     n=3.92, nsims=100, b_list=np.linspace(-0.5, 0.5, 501)):
    b_list_2 = np.concatenate((b_list[:-1], b_list[::-1]))
    b_list_4 = np.concatenate((b_list[:-1][::2], b_list[::-2], b_list[:-1][::2], b_list[::-2]))[:-1]
    nFrame = len(b_list_2)
    dt  = 1/fps
    time = np.arange(0, nFrame, 1)*dt
    tau = 0.1
    indep_noise = np.sqrt(dt/tau)*np.random.randn(nFrame, nsims)*0.08
    choice = np.zeros((len(coup_vals), nFrame, nsims, 2))
    for i_j, j in enumerate(coup_vals):
        for freq in range(2):
            stimulus = [b_list_2, b_list_4][freq]
            for sim in range(nsims):
                x = 0.
                vec = [x]
                for t in range(1, nFrame):
                    x = x + dt*(sigmoid(2*j*n*(2*x-1)+2*stimulus[t])-x)/tau + indep_noise[t, sim]
                    vec.append(x)
                    if x < 0.45:
                        ch = 0.
                    if x > 0.55:
                        ch = 1.
                    if 0.45 <= x <= 0.55:
                        ch = choice[i_j, t-1, sim, freq] 
                    choice[i_j, t, sim, freq] = ch
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(7.5, 4))
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylim(-0.025, 1.08)
        a.set_yticks([0, 0.5, 1])
        a.set_xticks([-0.5, 0, 0.5], [-1, 0, 1])
        # a.tick_params(axis='x', rotation=45)
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    lsts = ['solid', '--']
    for i_c, coupling in enumerate(coup_vals):
        for freq in range(2):
            stimulus = [b_list_2, b_list_4][freq]
            response_raw = choice[i_c, :, :, freq]
            if freq > 0:
                choice_aligned = np.column_stack((response_raw[:nFrame//2],
                                                  response_raw[nFrame//2:-1]))
                response_raw = choice_aligned
                stimulus = stimulus[:nFrame//2]
            response = np.nanmean(response_raw, axis=1)
            response = np.convolve(response, np.ones(10)/10, 'same')
            ax[freq].plot(stimulus, response, color=colormap[i_c], linewidth=4,
                          label=round(coupling, 1), linestyle=lsts[freq])
        if i_c == 2:
            ax[0].axhline(0.5, color=colormap[i_c], alpha=0.7, linestyle='--')
            ax[1].axhline(0.5, color=colormap[i_c], alpha=0.7, linestyle='--')
    ax[0].set_xlabel('Sensory evidence, B(t)')
    ax[1].set_yticks([0, 0.5, 1], ['', '', ''])
    ax[1].set_xlabel('Sensory evidence, B(t)')
    ax[0].set_ylabel('Proportion of rightward responses')
    ax[0].legend(title='Coupling, J', frameon=False)
    ax[0].set_title('Freq = 2', fontsize=14)
    ax[1].set_title('Freq = 4', fontsize=14)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'hysteresis_basic_plot_simulation.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'hysteresis_basic_plot_simulation.svg', dpi=400, bbox_inches='tight')


def noise_bf_switch_coupling(coup_vals=np.array((0., 0.3, 1))*0.3+0.05,
                             nFrame=50000, fps=60, noisyframes=30,
                             n=3.92, steps_back=120, steps_front=20):
    dt  = 1/fps/2
    time_interp = np.arange(0, nFrame+noisyframes, noisyframes)*dt
    noise_exp = np.random.randn(len(time_interp))*0.15
    time = np.arange(0, nFrame, 1)*dt
    noise_signal = scipy.interpolate.interp1d(time_interp, noise_exp)(time)
    tau = 10*dt
    indep_noise = np.sqrt(dt/tau)*np.random.randn(nFrame)*0.0
    x_arr = np.zeros((len(coup_vals), nFrame))
    choice = np.zeros((len(coup_vals), nFrame))
    for i_j, j in enumerate(coup_vals):
        x = 0.5
        vec = [x]
        for t in range(1, nFrame):
            x = x + dt*(sigmoid(2*j*n*(2*x-1)+2*noise_signal[t])-x)/tau + indep_noise[t]
            vec.append(x)
            if x < 0.5:
                ch = -1.
            if x >= 0.5:
                ch = 1.
            # if 0.4 <= x <= 0.6:
            #     ch = choice[i_j, t-1] 
            choice[i_j, t] = ch
        x_arr[i_j, :] = vec
    mean_vals_noise_switch_coupling = np.empty((len(coup_vals), steps_back+steps_front))
    mean_vals_noise_switch_coupling[:] = np.nan
    err_vals_noise_switch_coupling = np.empty((len(coup_vals), steps_back+steps_front))
    err_vals_noise_switch_coupling[:] = np.nan
    for i_j, coupling in enumerate(coup_vals):
        responses = choice[i_j]
        chi = noise_signal
        chi = chi-np.nanmean(chi)
        orders = rle(responses)
        idx_1 = orders[1][orders[2] == 1]
        idx_0 = orders[1][orders[2] == -1]
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
        mean_vals_noise_switch_coupling[i_j, :] = np.nanmean(mean_vals_noise_switch, axis=0)
        err_vals_noise_switch_coupling[i_j, :] = np.nanstd(mean_vals_noise_switch, axis=0) / np.sqrt(mean_vals_noise_switch.shape[0])
    fig, ax = plt.subplots(1, figsize=(5.5, 4))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    for i_j, j in enumerate(coup_vals):
        x_plot = np.arange(-steps_back, steps_front, 1)/fps
        y_plot = mean_vals_noise_switch_coupling[i_j, :]
        # y_plot = y_plot - np.nanmean(y_plot[x_plot < -50])
        err_plot = err_vals_noise_switch_coupling[i_j, :]
        ax.plot(x_plot, y_plot, color=colormap[i_j],
                label=np.round(j, 1), linewidth=4)
        ax.fill_between(x_plot, y_plot-err_plot, y_plot+err_plot, color=colormap[i_j],
                        alpha=0.3)
    ax.legend(title='Coupling, J', frameon=False)
    ax.set_xlabel('Time from switch (s)')
    ax.set_ylabel(r'Noise $\eta(t)$')
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'noise_before_switch_simulations.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'noise_before_switch_simulations.svg', dpi=400, bbox_inches='tight')


def get_kernel(d, chi, nFrame):
    """
    """
    chi_mean = np.mean(chi, axis=0)  # (T,)
    d_mean = np.mean(d, axis=0)  # (T,)
    chi_var = np.var(chi, axis=0)    # (T,)
    kern = np.zeros((nFrame))
    for t in range(nFrame):
        cov = np.mean((chi[:, t] - chi_mean[t]) * (d[:, t] - d_mean[t]))
        if chi_var[t] > 0:
            kern[t] = cov / chi_var[t]
        else:
            kern[t] = 0  # avoid divide by zero
    return kern


def plot_psych_kernels(data_folder=DATA_FOLDER, fps=60, tFrame=15,
                       shuffle_vals=[1, 0.7, 0]):
    nFrame = fps*tFrame
    df = load_data(data_folder + '/noisy/', n_participants='all')
    colormap = pl.cm.Oranges(np.linspace(0.3, 1, 3))[::-1]
    kernels = np.zeros((len(shuffle_vals), nFrame))
    for i_sh, pshuffle in enumerate(shuffle_vals):
        df_coupling = df.loc[df.pShuffle == pshuffle]
        trial_index = df_coupling.trial_index.unique()
        chi = df_coupling.stimulus.values.reshape((len(trial_index), nFrame))
        d = df_coupling.responses.values.astype(np.float32)
        d = d.reshape((len(trial_index), nFrame))-1
        kern = get_kernel(d, chi, nFrame=nFrame)
        kernels[i_sh] = kern
    fig, ax = plt.subplots(1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for i_sh, pshuffle in enumerate(shuffle_vals):
        ax.plot(np.arange(nFrame)/fps,
                kernels[i_sh, :], color=colormap[i_sh], label=pshuffle, linewidth=3)
    ax.legend(title='p(shuffle)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('P.K.')
    ax.set_ylim(-1, 1)
    fig.tight_layout()


def plot_noise_before_switch(data_folder=DATA_FOLDER, fps=60, tFrame=18,
                             steps_back=60, steps_front=20,
                             shuffle_vals=[1, 0.7, 0]):
    nFrame = fps*tFrame
    df = load_data(data_folder + '/noisy/', n_participants='all')
    colormap = pl.cm.Oranges(np.linspace(0.3, 1, 3))[::-1]
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    mean_vals_noise_switch_coupling = np.empty((len(shuffle_vals), steps_back+steps_front))
    mean_vals_noise_switch_coupling[:] = np.nan
    err_vals_noise_switch_coupling = np.empty((len(shuffle_vals), steps_back+steps_front))
    err_vals_noise_switch_coupling[:] = np.nan
    for i_sh, pshuffle in enumerate(shuffle_vals):
        df_coupling = df.loc[df.pShuffle == pshuffle]
        trial_index = df_coupling.trial_index.unique()
        mean_vals_noise_switch_all_trials = np.empty((len(trial_index), steps_back+steps_front))
        mean_vals_noise_switch_all_trials[:] = np.nan
        for i_trial, trial in enumerate(trial_index):
            df_trial = df.loc[df.trial_index == trial]
            responses = df_trial.responses.values
            chi = df_trial.stimulus.values
            chi = chi-np.nanmean(chi)
            orders = rle(responses)
            idx_1 = orders[1][orders[2] == 2]
            idx_0 = orders[1][orders[2] == 1]
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
            mean_vals_noise_switch_all_trials[i_trial, :] =\
                np.nanmean(mean_vals_noise_switch, axis=0)
        mean_vals_noise_switch_coupling[i_sh, :] = np.convolve(np.nanmean(mean_vals_noise_switch_all_trials, axis=0), np.ones(20)/20,
                                                               'same')
        err_vals_noise_switch_coupling[i_sh, :] = np.nanstd(mean_vals_noise_switch_all_trials, axis=0) / np.sqrt(len(trial_index))
    fig, ax = plt.subplots(1, figsize=(5.5, 4))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for i_sh, pshuffle in enumerate(shuffle_vals):
        x_plot = np.arange(-steps_back, steps_front, 1)/fps
        y_plot = mean_vals_noise_switch_coupling[i_sh, :]
        err_plot = err_vals_noise_switch_coupling[i_sh, :]
        ax.plot(x_plot, y_plot, color=colormap[i_sh],
                label=pshuffle, linewidth=3)
        ax.fill_between(x_plot, y_plot-err_plot, y_plot+err_plot, color=colormap[i_sh],
                        alpha=0.3)
    ax.legend(title='p(shuffle)', frameon=False)
    ax.set_xlabel('Time from switch (s)')
    ax.set_ylabel('Noise')
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'noise_before_switch_experiment.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'noise_before_switch_experiment.svg', dpi=400, bbox_inches='tight')


def experiment_example(nFrame=1200, fps=60, noisyframes=10):
    dt = 1/fps
    noise_exp = np.random.randn(nFrame // noisyframes+1)*0.1
    time_interp = np.arange(0, nFrame+1, noisyframes)*dt
    noise_signal = scipy.interpolate.interp1d(time_interp, noise_exp)
    time = np.arange(0, nFrame, 1)*dt
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 4))
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Noise')
    ax[0].set_ylabel('Sensory evidence, B(t)')
    difficulty_time_ref_2 = np.linspace(2, -2, nFrame//2)
    stimulus = np.concatenate(([difficulty_time_ref_2, -difficulty_time_ref_2]))
    # ax[0].plot(time, stimulus, linewidth=4, label='2', color='navy')
    line = colored_line(time, stimulus, stimulus, ax[0],
                        linewidth=4, cmap='coolwarm_r', 
                        norm=plt.Normalize(vmin=-2, vmax=2), label='2')
    ax[0].set_xlim([np.min(time)-1e-1, np.max(time)+1e-1])
    ax[0].set_yticks([-2, 0, 2], ['-1', '0', '1'])
    difficulty_time_ref_4 = np.linspace(-2, 2, nFrame//4)
    stimulus = np.concatenate(([difficulty_time_ref_4, -difficulty_time_ref_4,
                                difficulty_time_ref_4, -difficulty_time_ref_4]))
    # ax[0].plot(time, stimulus, linewidth=4, label='4', color='navy', linestyle='--')
    line = colored_line(time, stimulus, stimulus, ax[0],
                        linewidth=4, cmap='coolwarm_r', linestyle='--', 
                        norm=plt.Normalize(vmin=-2, vmax=2), label='4')
    # ax[0].legend(title='Freq.', frameon=False)
    legendelements = [Line2D([0], [0], color='k', lw=4, label='2'),
                      Line2D([0], [0], color='k', lw=4, label='4', linestyle='--')]
                      # Line2D([0], [0], color='b', lw=2, label='5')]
    ax[0].legend(handles=legendelements, title='Freq', frameon=False)
    ax[1].axhline(0, color='lightblue', linestyle='--', alpha=1)
    # ax[1].plot(time, noise_signal(time), color='navy', linewidth=4)
    vals_max = np.max(np.abs(noise_signal(time)))
    line = colored_line(time, noise_signal(time), noise_signal(time), ax[1],
                        linewidth=4, cmap='coolwarm_r',
                        norm=plt.Normalize(vmin=-vals_max, vmax=vals_max))
    ax[1].set_ylim(-0.4, 0.4)
    ax[1].set_yticks([0])
    ax[1].set_xlim([np.min(time)-1e-1, np.max(time)+1e-1])
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'stim_example.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'stim_example.svg', dpi=400, bbox_inches='tight')


def sigmoid(x):
    return 1/(1+np.exp(-x))


def second_derivative_potential(q, j, b=0, n=3.92):
    expo = 2*n*(j*(2*q-1))+2*b
    return 1 - 4*n*j*sigmoid(expo)*(1-sigmoid(expo))


def potential_mf(q, j, bias=0, n=3.92):
    return q*q/2 - np.log(1+np.exp(2*n*(j*(2*q-1))+bias*2))/(4*n*j) #  + q*bias


def k_i_to_j(j, xi, xj, noise, b=0, n=3.92):
    v_2_xi = second_derivative_potential(xi, j, b=b, n=n)
    v_2_xj = second_derivative_potential(xj, j, b=b, n=n)
    v_xi = potential_mf(xi, j, b, n=n)
    v_xj = potential_mf(xj, j, b, n=n)
    return np.sqrt(np.abs(v_2_xi*v_2_xj))*np.exp(2*(v_xi - v_xj)/noise**2) / (2*np.pi)


def get_unst_and_stab_fp(j, b, n=3.92):
    x_unst = 0.5
    qst_1 = 1
    qst_0 = 0
    q1 = lambda q: sigmoid(2*n*j*(2*q-1) + b*2)
    for i in range(100):
        x_unst = backwards(x_unst, j, b, n=n)
        qst_1 = q1(qst_1)
        qst_0 = q1(qst_0)
    if x_unst < 0 or x_unst > 1:
        x_unst = np.nan
    return qst_1, qst_0, x_unst


def responses_clean(responses):
    responses = responses*2-1
    responses[np.isnan(responses)] = 0
    return responses


def backwards(q, j, b, n=3):
    if 0 <= q <= 1:
        # q_new = 0.5*(1+ 1/j *(1/(2*n_neigh) * np.log(q/(1-q)) - beta))
        q_new = 0.5*(1+ 1/(j*n) *(1/2 * np.log(q/(1-q)) - b))
    else:
        q_new = np.nan
    return q_new


def compute_linearized_drift(q, j, n=3.92):
    return q*(1-q)*4*j*n+q


def compute_nu_0(mu_0, sigma, theta):
    alpha = np.abs(theta-mu_0)/(sigma)
    return 1/(alpha/np.sqrt(np.pi)*np.exp(-alpha**2))


def compute_nu_0_real(q, j, b, sigma, theta=0.5, n=3.92):
    factor = 4*n*j*q*(1-q)
    a = factor-1
    b = q*(1-factor)
    mu = -b/a
    cap_sigma = sigma**2 / np.abs(2*a)
    integrand = lambda u: (1+scipy.special.erf(u))*np.exp(u**2)
    nu_inverse = np.sqrt(np.pi)*scipy.integrate.quad(integrand, -mu/cap_sigma, np.abs(theta-mu)/cap_sigma)[0]
    return 1/nu_inverse


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
            log_likelihood += -lambda_t*dt
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
        responses_2, responses_4, stim_values_2, stim_values_4, coupling_2, coupling_4 =\
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


def beta(x):
    return 1 / (1 + np.exp(-x))


def drift_MF(q, t, stim_t, theta):
    J, B0, B, sigma, tau = theta
    N = 3.92
    input_drive = 2 * (J * N * (2*q - 1) + B * stim_t + B0)
    return (beta(input_drive) - q) / tau


def drift_FBP(q, t, stim_t, theta):
    J, alpha, B0, B, sigma, tau = theta
    N = 3.92
    m = (q-stim_t*B-B0)/N
    fun = (1/alpha * np.arctanh(np.tanh(alpha*J)*np.tanh(m*(N-alpha) + B * stim_t + B0))-m)/tau
    return fun


def drift_LBP(q, t, stim_t, theta):
    J, B0, B, sigma, tau = theta
    N = 3.92
    alpha = 1
    m = (q-stim_t*B-B0)/N
    fun = (np.arctanh(np.tanh(J)*np.tanh(m*(N-alpha) + B * stim_t + B0))-m)/tau
    return fun


def diffusion(theta, model='MF'):
    if model in ['MF', 'LBP']:
        _, _, _, sigma, tau = theta
    else:
        _, _, _, _, sigma, tau = theta
    return (sigma**2) / (2 * tau**2)


def solve_fokker_planck_1d(times, stim_values, theta,
                           dx=0.01, shift_ini=0, model='MF',
                           thres=0.5):
    if model == 'MF':
        f = lambda q, k: drift_MF(q, times[k], stim_values[k], theta)
    if model == 'LBP':
        f = lambda q, k: drift_LBP(q, times[k], stim_values[k], theta)
    if model == 'FBP':
        f = lambda q, k: drift_FBP(q, times[k], stim_values[k], theta)
    dt = np.min(np.diff(times))
    sigma = theta[-2]
    tau = theta[-1]
    if dt > 2*tau*dx**2/(sigma**2):
        # print('Careful with dt, dx. dt > dx^2 / (2 \sigma^2)')
        # print(f'Now, dt = {dt}, dx = {dx}, dx^2 / (2 \sigma^2) = {round(dx**2 / (2 * sigma**2), 6)}')
        # dx = float(input('dx: '))
        dx = np.sqrt(dt*sigma**2/(2*tau))+1e-2
        # dt = float(input('dt: '))
    # Create the grid
    if model == 'MF':
        x = np.arange(0, 1 + dx, dx)
    else:
        x = np.arange(-5, 5 + dx, dx)
        # val_min_fbp = np.log(0.6/0.4)/2
    # Initialize probability distribution (e.g., a Gaussian at t=0)
    P = 0.5 * np.exp(-(x-shift_ini-0.5)**2 / (0.1**2))
    P /= P.sum() * dx  # Normalize

    # Precompute constants
    D = diffusion(theta, model=model)  # Diffusion coefficient

    # Finite difference coefficients
    P_new = np.zeros_like(P)
    # Parr = np.zeros((len(x), len(times)))
    p_left = []
    p_right = []
    flux_left_right = []
    flux_right_left = []
    for i_t, ti in enumerate(times):
        # Compute derivatives
        dP_dx = np.gradient(P, dx)  # First derivative
        dPf_dx = np.gradient(P*f(x, i_t), dx)  # First derivative of (p*f)
        d2P_dx2 = np.gradient(dP_dx, dx)  # Second derivative
        
        # Update using Fokker-Planck equation
        P_new = P - dt * (dPf_dx - D * d2P_dx2)
        
        # Enforce boundary conditions (e.g., zero flux)
        P_new[0], P_new[-1] = 0, 0
        
        # Update probability distribution
        P = np.abs(P_new.copy())
        P /= (P.sum() * dx)  # Re-normalize to ensure total probability is 1
        idx_threshold = np.argmin(np.abs(x-thres))
        if x[idx_threshold] > 0.5:
            idx_threshold_neg = idx_threshold-1
            idx_threshold_pos = idx_threshold
        if x[idx_threshold] < 0.5:
            idx_threshold_neg = idx_threshold
            idx_threshold_pos = idx_threshold+1
        if x[idx_threshold] == 0.5:
            idx_threshold_neg = idx_threshold-1
            idx_threshold_pos = idx_threshold+1
        # p_right = np.sum(P[x > thres])*dx 
        # p_left = np.sum(P[x < thres])*dx
        flux_t_neg = -sigma**2/(2*tau)*dP_dx[idx_threshold_neg] + f(x[idx_threshold_neg], i_t)*P[idx_threshold_neg]
        flux_t_pos = sigma**2/(2*tau)*dP_dx[idx_threshold_pos] - f(x[idx_threshold_pos], i_t)*P[idx_threshold_pos]
        flux_right_left.append(flux_t_pos)  #  / p_right
        flux_left_right.append(flux_t_neg)  #  / p_left
        if model == 'MF':
            p_left.append(np.sum(P[x < thres])*dx)  # between 0.6 and 1
            p_right.append(np.sum(P[x > thres])*dx)  # between 0 and 0.4
        else:
            p_left.append(np.sum(P[x < 0])*dx)
            p_right.append(np.sum(P[x > 0])*dx)
        # Parr[:, i_t] = P
    return flux_right_left, flux_left_right, p_right, p_left


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


def prepare_data_for_fitting(df, tFrame=18, fps=60):
    df_freq = df.loc[df.freq == 2]
    # get response_array and stimulus values
    responses_2, stim_values_2 = get_response_and_blist_array(df_freq, fps=fps, tFrame=tFrame)
    coupling_2 = np.round(1-df_freq.groupby('trial_index').pShuffle.mean().values, 2)
    df_freq = df.loc[df.freq == 4]
    # get response_array and stimulus values
    responses_4, stim_values_4 = get_response_and_blist_array(df_freq, fps=fps, tFrame=tFrame)
    coupling_4 = np.round(1-df_freq.groupby('trial_index').pShuffle.mean().values, 2)
    return [responses_2, responses_4, stim_values_2, stim_values_4, coupling_2, coupling_4]


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
                         fps=60, tFrame=18, freq=4, idx=10, n=3.92, theta=0.5,
                         tol=1e-6, pshuffle=0):
    j_par, j0, b0, b_par, sigma = params
    df = load_data(data_folder, n_participants='all')
    df_freq = df.loc[(df.freq == freq) & (df.pShuffle == pshuffle)]
    j_eff = j_par*(1-pshuffle)+j0
    T = tFrame  # total duration (seconds)
    n_times = tFrame*fps
    responses, stimulus = get_response_and_blist_array(df_freq, fps=fps, tFrame=tFrame)
    vals = np.abs(np.diff(np.sign(stimulus)))
    idx_diff = np.where(vals != 0)[0]
    keypresses = responses[idx]
    responses = responses_clean(keypresses)
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


def plot_diagnostics_training(theta_np, summary_statistics):
    npars = theta_np.shape[1]
    nrows = summary_statistics.shape[1]-1
    fig, ax = plt.subplots(nrows=npars, ncols=nrows, figsize=(16, 10))
    labs_rows = ['# switches', 'Alignment', '<ch = 1>', '<ch = -1>', 'entropy',
                 'latency 1st switch', '<T>']
    ax = ax.flatten()
    a_id = 0
    for i in range(npars):
        for r in range(nrows):
            ax[a_id].plot(theta_np[:, i], summary_statistics[:, r],
                          marker='o', linestyle='', color='k', markersize=2)
            corr = np.corrcoef(theta_np[:, i], summary_statistics[:, r])[0][1]
            ax[a_id].set_title(r'$\rho = $' + str(round(corr, 3)), fontsize=13)
            ax[a_id].set_ylabel(labs_rows[r])
            a_id += 1
    fig.tight_layout()


def sbi_training(n_simuls=10000, fps=60, tFrame=20, data_folder=DATA_FOLDER,
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
        posterior = inference.build_posterior(density_estimator)
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
                    labels = ['J1', 'J0', 'B0', 'B1', '\sigma']
                else:
                    limits = [[-0.5, 2], [-1, 1], [-0.2, 2], [0.0, 0.5]]
                    labels = ['J1', 'J0',  'B1', '\sigma']
            else:
                if stim_offset:
                    limits = [[-0.5, 2], [-1, 1], [-0.2, 2], [0.0, 0.5]]
                    labels = ['J1', 'B0', 'B1', '\sigma']
                else:
                    limits = [[-0.5, 2], [-0.2, 2], [0.0, 0.5]]
                    labels = ['J1', 'B1', '\sigma']
            # plot posterior samples in pairplot
            _ = analysis.pairplot(samples_posterior, limits=limits, figsize=(9, 9),
                                  labels=labels, points=theta_example,
                                  points_offdiag={"markersize": 6}, points_colors="r")
    if load_net:
        with open(data_folder + f"/snpe_{n_simuls}.p", 'rb') as f:
            density_estimator = pickle.load(f)
        with open(data_folder + f"/posterior_{n_simuls}.p", 'rb') as f:
            posterior = pickle.load(f)
        density_estimator = density_estimator['estimator']
        posterior = posterior['posterior']
    return density_estimator, posterior


def get_likelihood_data(params, data_folder=DATA_FOLDER, tFrame=20, fps=60, ntraining=8,
                        n_simuls=500000):
    df = load_data(data_folder, n_participants='all')
    subjects = df.subject.unique()
    for i_s, subject in enumerate(subjects):
        df_sub = df.loc[(df.subject == subject) & (df.trial_index > ntraining)]
        responses_2, responses_4, stim_values_2, stim_values_4, coupling_2, coupling_4 =\
            prepare_data_for_fitting(df_sub, tFrame=tFrame, fps=fps)
        responses_2 = responses_clean(responses_2)
        responses_4 = responses_clean(responses_4)
        responses_all = np.concatenate((responses_2, responses_4))
        coupling_all = np.concatenate((coupling_2, coupling_4))
        freqs = np.concatenate((np.repeat(2, len(coupling_2)), np.repeat(4, len(coupling_4))))
        data_all = np.column_stack((responses_all, freqs))
        estimator, posterior = sbi_training(n_simuls=n_simuls, fps=fps, tFrame=tFrame,
                                            data_folder=DATA_FOLDER, load_net=True, plot_posterior=False)
        
        theta = torch.reshape(torch.tensor(params),
                              (1, len(params))).to(torch.float32)
        llh = estimator.log_prob(theta.repeat(len(data_all), 1), torch.tensor(data_all))


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


def return_input_output_for_network(params, choice, freq, nFrame=1200, fps=60,
                                    max_number=10):
    dt = 1/fps
    tFrame = nFrame*dt
    # Find the indices where the value changes
    change_indices = np.where(choice[1:] != choice[:-1])[0] + 1
    
    # Start indices of epochs (always include 0)
    start_indices = np.concatenate(([0], change_indices))
    
    # Corresponding response values at those start indices
    responses = choice[start_indices]
    
    # Combine into array of [response, start_index]
    result = np.column_stack((responses, start_indices/nFrame))
    
    # If you want it as a list of [response, time] pairs:
    output_network_0 = np.concatenate((change_indices, [nFrame]))/nFrame
    output_network = np.column_stack((output_network_0, np.roll(responses, -1)))
    params_repeated = np.column_stack(np.repeat(params.reshape(-1, 1), result.shape[0], 1))
    input_network = np.column_stack((params_repeated, result, np.repeat(freq, result.shape[0])))
    return input_network[:-1], output_network[:-1]


def normalized_percept_entropy(resp_t):
    values, counts = np.unique(resp_t, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(values)) if len(values) > 1 else 1
    return entropy / max_entropy


def return_summary_statistics(choice, stimulus, freq, dt=1/60, nFrame=1200):
    """
    Function that will return the summary statistics of simulations.
    These will be:
        - number of switches
        - average percept aligned with stim
        - average choices 1 & -1 (0 is unnecessary)
        - freq
    """
    alignment = np.nanmean(choice == np.sign(stimulus))
    switches = np.abs(np.diff(choice))
    number_switches = np.nansum(switches > 1)
    # switches = np.concatenate((switches, [2]))
    # average_dominance_time = np.nanmean(np.diff(np.where(switches != 0)[0]))*dt
    # if np.isnan(average_dominance_time):
    #     average_dominance_time = nFrame*dt
    avg_ch_1 = np.nanmean(choice == 1)
    avg_ch_neg1 = np.nanmean(choice == -1)
    entropy = normalized_percept_entropy(choice)
    first_stimulus_change = np.where(np.abs(np.diff(np.sign(stimulus))) != 0)[0][0]*dt
    avg_dominance = np.nanmean(np.diff(np.where(switches > 0)[0]))*dt
    if np.isnan(avg_dominance):
        avg_dominance = nFrame*dt
        latency_switch = avg_dominance
    else:
        first_switch = np.where(switches > 1)[0][0]*dt
        latency_switch = first_switch - first_stimulus_change
    # first_resp = np.where(choice != 0)[0]
    # latency_first_response = first_resp[0]*dt if len(first_resp > 0) else nFrame*dt
    return [number_switches, alignment, avg_ch_1, avg_ch_neg1, entropy, latency_switch, avg_dominance,
            freq]


if __name__ == '__main__':
    # plot_example(theta=[0.1, 0, 0.5, 0.1, 0.5], data_folder=DATA_FOLDER,
    #              fps=60, tFrame=18, model='MF', prob_flux=False,
    #              freq=4, idx=2)
    hysteresis_basic_plot(coupling_levels=[0, 0.3, 1],
                          fps=60, tFrame=18, data_folder=DATA_FOLDER,
                          nbins=9, ntraining=8, arrows=True)
    plot_noise_before_switch(data_folder=DATA_FOLDER, fps=60, tFrame=18,
                             steps_back=120, steps_front=20,
                             shuffle_vals=[1, 0.7, 0])
    # plot_example_pswitch(params=[0.7, 1e-2, 0., 0.2, 0.5], data_folder=DATA_FOLDER,
    #                       fps=60, tFrame=20, freq=2, idx=5, n=3.92, theta=0.5,
    #                       tol=1e-3, pshuffle=0)
    # fitting_transitions(data_folder=DATA_FOLDER, fps=60, tFrame=18)
    # fitting_fokker_planck(data_folder=DATA_FOLDER, model='MF')
