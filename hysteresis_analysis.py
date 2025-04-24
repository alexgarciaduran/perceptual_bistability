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
                if switch_times[i+1] < 50:
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
    

def hysteresis_basic_plot(coupling_levels=[0, 0.3, 1],
                          diff_maxmin=[-2, 2],
                          fps=60, tFrame=15, data_folder=DATA_FOLDER,
                          n=3.92, eps=0.1):
    nFrame = fps*tFrame
    df = load_data(data_folder, n_participants='all')
    subjects = df.subject.unique()
    fig, ax = plt.subplots(ncols=2, nrows=len(subjects), figsize=(8, 4.5))
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylim(-0.025, 1.025)
    colormap = pl.cm.Oranges(np.linspace(0.3, 1, 3))[::-1]
    for i_s, subject in enumerate(subjects):
        df_sub = df.loc[df.subject == subject]
        for i_c, coupling in enumerate(coupling_levels):
            df_filt = df_sub.loc[df_sub.pShuffle.round(2) == round(1-coupling, 2)]
            df_freq_2 = df_filt.loc[df.freq == 2]
            response_array_2, barray_2 = get_response_and_blist_array(df_freq_2, fps=fps,
                                                                      tFrame=tFrame)
            # mean_response_2 = np.nanmean(response_array_2, axis=0)
            ascending = np.sign(np.gradient(barray_2))
            df_to_plot = pd.DataFrame(response_array_2.T)
            df_to_plot['stimulus'] = barray_2
            df_to_plot['ascending'] = ascending
            df_to_plot['stim_bin'] = pd.cut(df_to_plot['stimulus'], bins=13)
            df_to_plot = df_to_plot.drop('stimulus', axis=1)
            yvals_2 = np.nanmean(df_to_plot.groupby(['ascending', 'stim_bin']).mean().values, axis=1)
            df_to_plot['stimulus'] = barray_2
            xvals_2 = df_to_plot.groupby(['ascending', 'stim_bin'])['stimulus'].mean().values
            ax[0].plot(xvals_2[len(xvals_2)//2:], yvals_2[len(xvals_2)//2:], color=colormap[i_c], linewidth=3,
                       label=1-coupling)
            ax[0].plot(xvals_2[:len(xvals_2)//2], yvals_2[:len(xvals_2)//2], color=colormap[i_c], linewidth=3)
            df_freq_4 = df_filt.loc[df.freq == 4]
            response_array_4, barray_4 = get_response_and_blist_array(df_freq_4, fps=fps,
                                                                      tFrame=tFrame)
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
            df_to_plot['stim_bin'] = pd.cut(df_to_plot['stimulus'], bins=11)
            df_to_plot = df_to_plot.drop('stimulus', axis=1)
            yvals_4 = np.nanmean(df_to_plot.groupby(['ascending', 'stim_bin']).mean().values, axis=1)
            df_to_plot['stimulus'] = barray_4
            xvals_4 = df_to_plot.groupby(['ascending', 'stim_bin'])['stimulus'].mean().values
            ax[1].plot(xvals_4[len(xvals_4)//2:], yvals_4[len(xvals_4)//2:],
                       color=colormap[i_c], linewidth=3, label=1-coupling)
            ax[1].plot(xvals_4[:len(xvals_4)//2],
                       yvals_4[:len(xvals_4)//2], color=colormap[i_c], linewidth=3)
            # ax[1].plot(barray_4, mean_response_4, color=colormap[i_c], linewidth=3)
    ax[0].set_xlabel('Sensory evidence, B(t)')
    ax[1].set_xlabel('Sensory evidence, B(t)')
    ax[0].set_ylabel('Proportion of rightward responses')
    ax[0].legend(title='p(shuffle)', frameon=False)
    ax[0].set_title('Freq = 2', fontsize=14)
    ax[1].set_title('Freq = 4', fontsize=14)
    fig.tight_layout()


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


def plot_noise_before_switch(data_folder=DATA_FOLDER, fps=60, tFrame=15,
                             steps_back=60, steps_front=20,
                             shuffle_vals=[1, 0.7, 0]):
    nFrame = fps*tFrame
    df = load_data(data_folder + '/noisy/', n_participants='all')
    colormap = pl.cm.Oranges(np.linspace(0.3, 1, 3))[::-1]
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
        mean_vals_noise_switch_coupling[i_sh, :] = np.nanmean(mean_vals_noise_switch_all_trials, axis=0)
        err_vals_noise_switch_coupling[i_sh, :] = np.nanstd(mean_vals_noise_switch_all_trials, axis=0) / np.sqrt(len(trial_index))
    fig, ax = plt.subplots(1)
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
    ax.legend(title='p(shuffle)')
    ax.set_xlabel('Time from switch (s)')
    ax.set_ylabel('Noise')
    fig.tight_layout()
