# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 20:07:39 2025

@author: alexg
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
import tqdm
import statsmodels.formula.api as smf


mpl.rcParams['font.size'] = 16
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize']= 14
plt.rcParams['ytick.labelsize']= 14

pc_name = 'alex'
if pc_name == 'alex':
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/hysteresis/data/'  # Alex
    SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/hysteresis/parameters/'  # Alex
elif pc_name == 'alex_CRM':
    DATA_FOLDER = 'C:/Users/agarcia/Desktop/phd/necker/data_folder/hysteresis/'  # Alex CRM
    SV_FOLDER = 'C:/Users/agarcia/Desktop/phd/necker/data_folder/hysteresis/'  # Alex CRM


def load_data(data_folder, n_participants='all', filter_subjects=True):
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
            # if i in [4, 10, 15, 18, 19]:  # manually discard
            #     continue
            df = pd.read_csv(files[i])
            df['subject'] = 's_' + str(i+1)
            if filter_subjects and \
                not accept_subject(DATA_FOLDER, i,
                                   threshold_switches=3, n_training=8):
                continue
            df_0 = pd.concat((df_0, df))
        return df_0


def accept_subject(data_folder, i, threshold_switches=3, n_training=8):
    # excludes participant if less than 3 responses per trial in average
    files = glob.glob(data_folder + '*.csv')
    df = pd.read_csv(files[i])
    trial_index = df.trial_index.unique()
    avg_resp = []
    for ti in trial_index:
        if ti < n_training:
            continue
        df_ti = df.loc[df.trial_index == ti]
        responses = df_ti.response.values
        respvals, respcounts = np.unique(responses, return_counts=True)
        avg_resp.append(sum(respcounts[respvals != 0]))
    return np.mean(avg_resp) > threshold_switches


def preprocess_df_examples(data_folder, threshold_switches=3, n_training=8):
    # excludes participant if less than 1.5 switches in response in average
    df = load_data(data_folder=data_folder, n_participants='all',
                   filter_subjects=False)
    subjects = df.subject.unique()
    avg_resp_per_subject = []
    for subject in subjects:
        df_sub = df.loc[df.subject == subject]
        trial_index = df_sub.trial_index.unique()
        avg_resp = []
        for ti in trial_index:
            if ti < n_training:
                continue
            df_ti = df_sub.loc[df_sub.trial_index == ti]
            responses = df_ti.response.values
            respvals, respcounts = np.unique(responses, return_counts=True)
            avg_resp.append(sum(respcounts[respvals != 0]))
        avg_resp_per_subject.append(np.mean(avg_resp))
    plt.figure()
    plt.ylabel('Average number of percepts')
    plt.xlabel('Subject ID')
    plt.plot(np.arange(1, len(subjects)+1),
             avg_resp_per_subject)
    plt.axhline(threshold_switches)


def get_response_and_blist_array(df, fps=60, tFrame=26,
                                 flip_responses=False):
    nFrame = fps*tFrame
    stimulus_times = np.arange(0, nFrame)
    trial_index = df.trial_index.unique()
    freq = df.freq.unique()[0]
    if freq == 2:
        difficulty_time_ref = np.linspace(-2, 2, nFrame//2)
        blist_freq_ref = np.concatenate(([difficulty_time_ref, -difficulty_time_ref]))
        response_array_desc = np.zeros((len(trial_index), nFrame//2))
        response_array_asc = np.zeros((len(trial_index), nFrame//2))
    else:
        difficulty_time_ref = np.linspace(-2, 2, nFrame//4)
        blist_freq_ref = np.concatenate(([difficulty_time_ref, -difficulty_time_ref,
                                          difficulty_time_ref, -difficulty_time_ref]))
        response_array_desc = np.zeros((len(trial_index)*2, nFrame//4))
        response_array_asc = np.zeros((len(trial_index)*2, nFrame//4))

    map_resps = {1:1, 0:np.nan, 2:0}
    ini_sides = []

    for idx_ti, ti in enumerate(trial_index):
        df_filt = df.loc[df.trial_index == ti]
        times_onset = np.int32(df_filt.keypress_seconds_onset.values/tFrame*nFrame)
        switch_times = np.concatenate((times_onset, [nFrame]))
        responses = df_filt.response.values
        if np.isnan(responses).any():
            continue
        responses = [map_resps[r] for r in responses]
        response_array_asc[idx_ti, :] = responses[0] if len(responses) > 1 else responses
        response_array_desc[idx_ti, :] = responses[0] if len(responses) > 1 else responses

        ini_side_trial = df_filt['initial_side'].unique()[0]
        ini_sides.append(ini_side_trial)
        
        blist_freq_ref_trial = blist_freq_ref*(-ini_side_trial)
        siwtches_difference = []
        if len(times_onset) > 1:
            response_series = np.array(())
            for i in range(len(switch_times)-1):
                if switch_times[i+1] < 40:
                    resp_i = np.nan
                else:
                    resp_i = responses[i]
                mask = (stimulus_times >= switch_times[i]) & (stimulus_times < switch_times[i+1])
                response_series = np.concatenate((response_series, [resp_i] * np.sum(mask)))
            diff_switches = switch_times[2]-switch_times[1]
            asc_mask = np.sign(np.gradient(blist_freq_ref_trial)) > 0
            siwtches_difference.append(diff_switches)
            if freq == 2:
                response_array_asc[idx_ti, :] = response_series[asc_mask]
                response_array_desc[idx_ti, :] = response_series[~asc_mask]
            if freq == 4:
                resp_first_half = response_series[:nFrame//2]
                resp_second_half = response_series[nFrame//2:]
                response_array_asc[idx_ti, :] = resp_first_half[asc_mask[:nFrame//2]]
                response_array_desc[idx_ti, :] = resp_first_half[~asc_mask[:nFrame//2]]
                response_array_asc[idx_ti+len(trial_index), :] = resp_second_half[asc_mask[nFrame//2:]]
                response_array_desc[idx_ti+len(trial_index), :] = resp_second_half[~asc_mask[nFrame//2:]]
    return response_array_asc, response_array_desc, blist_freq_ref, ini_sides, siwtches_difference


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


def collect_responses(df, subjects, coupling_levels, fps=60, tFrame=26):
    responses_2 = [[] for _ in coupling_levels]
    responses_4 = [[] for _ in coupling_levels]
    barray_2, barray_4 = None, None

    for subject in subjects:
        df_sub = df.loc[df.subject == subject]
        for i_c, coupling in enumerate(coupling_levels):
            df_filt = df_sub.loc[df_sub.pShuffle.round(2) == round(1-coupling, 2)]

            # freq = 2
            df_freq_2 = df_filt.loc[df_filt.freq == 2]
            if not df_freq_2.empty:
                resp_asc, resp_desc, barray_2, ini_sides, switches_diff = get_response_and_blist_array(
                    df_freq_2, fps=fps, tFrame=tFrame)
                responses_2[i_c].append({"asc": resp_asc, "desc": resp_desc, "ini_side": ini_sides,
                                         'switches_diff': switches_diff})

            # freq = 4
            df_freq_4 = df_filt.loc[df_filt.freq == 4]
            if not df_freq_4.empty:
                resp_asc, resp_desc, barray_4, ini_sides, switches_diff = get_response_and_blist_array(
                    df_freq_4, fps=fps, tFrame=tFrame)
                responses_4[i_c].append({"asc": resp_asc, "desc": resp_desc, "ini_side": ini_sides,
                                         'switches_diff': switches_diff})

    return responses_2, responses_4, barray_2, barray_4



def plot_responses_panels(responses_2, responses_4, barray_2, barray_4, coupling_levels,
                          tFrame=26, fps=60, window_conv=None,
                          ndt_list=np.arange(100)):
    """
    Make a 2-panel plot:
      - Left:  freq=2
      - Right: freq=4
    Solid line = ascending, dashed line = descending
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
    nsubs = hyst_width_2.shape[1]
    if ndt_list is not None:
        hysteresis, max_hists = \
            get_argmax_ndt_hyst_per_subject(responses_2, responses_4, barray_2, barray_4, coupling_levels,
                                            tFrame=tFrame, fps=fps, window_conv=window_conv,
                                            ndtlist=ndt_list)
        hysteresis_across_coupling = np.mean(hysteresis, axis=0)
        delay_per_subject = ndt_list[np.argmax(hysteresis_across_coupling, axis=1)]
    else:
        delay_per_subject = [0]*nsubs
    # --- FREQ = 2 ---
    switch_time_diff_2 = np.zeros((len(coupling_levels), len(responses_2[0])))
    switch_time_diff_4 = np.zeros((len(coupling_levels), len(responses_2[0])))
    for i_c, coupling in enumerate(coupling_levels):
        subj_means_asc, subj_means_desc = [], []
        for i_s, subj_resp in enumerate(responses_2[i_c]):
            subj_means_asc.append(np.roll(np.nanmean(subj_resp["asc"], axis=0), delay_per_subject[i_s]))
            subj_means_desc.append(np.roll(np.nanmean(subj_resp["desc"], axis=0), delay_per_subject[i_s]))
            hyst_width_2[i_c, i_s] = np.nansum(np.abs(np.nanmean(subj_resp["desc"], axis=0)[::-1]-np.nanmean(subj_resp["asc"], axis=0)), axis=0) * np.diff(barray_2[:nFrame//2])[0]
            switch_time_diff_2[i_c, i_s] = np.nanmean(subj_resp['switches_diff'])
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
    ax2.set_ylim(np.min(np.mean(hyst_width_2, axis=1))-0.25, np.max(np.mean(hyst_width_2, axis=1))+0.25)
    hyst_width_4 = np.zeros((len(coupling_levels), len(responses_2[0])))
    # --- FREQ = 4 ---
    for i_c, coupling in enumerate(coupling_levels):
        subj_means_asc, subj_means_desc = [], []
        for i_s, subj_resp in enumerate(responses_4[i_c]):
            subj_means_asc.append(np.roll(np.nanmean(subj_resp["asc"], axis=0), delay_per_subject[i_s]))
            subj_means_desc.append(np.roll(np.nanmean(subj_resp["desc"], axis=0), delay_per_subject[i_s]))
            hyst_width_4[i_c, i_s] = np.nansum(np.abs(np.nanmean(subj_resp["desc"], axis=0)[::-1]-np.nanmean(subj_resp["asc"], axis=0)), axis=0)* np.diff(barray_4[:nFrame//2])[0]
            switch_time_diff_4[i_c, i_s] = np.nanmean(subj_resp['switches_diff'])
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
    ax4.set_ylim(np.min(np.mean(hyst_width_4, axis=1))-0.25, np.max(np.mean(hyst_width_4, axis=1))+0.2)
    for a in ax2, ax4:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_xlabel('p(shuffle)', fontsize=11); a.set_xticks([])
        a.set_ylabel('Hysteresis', fontsize=11); a.set_yticks([])
    # labels/titles
    np.save(DATA_FOLDER + 'hysteresis_width_freq_2.npy', hyst_width_2)
    np.save(DATA_FOLDER + 'hysteresis_width_freq_4.npy', hyst_width_4)
    np.save(DATA_FOLDER + 'switch_time_diff_freq_2.npy', switch_time_diff_2)
    np.save(DATA_FOLDER + 'switch_time_diff_freq_4.npy', switch_time_diff_4)
    # ax2[0].set_ylabel('Hysteresis width')
    ax[0].set_xlabel('Sensory evidence, B(t)')
    ax[1].set_xlabel('Sensory evidence, B(t)')
    ax[0].set_ylabel('P(rightward)')
    ax[0].legend(title='p(shuffle)', frameon=False,
                 bbox_to_anchor=[-0.02, 1.07], loc='upper left')
    ax[0].set_title('Freq = 2', fontsize=14)
    ax[1].set_title('Freq = 4', fontsize=14)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'hysteresis_average.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'hysteresis_average.svg', dpi=400, bbox_inches='tight')
    # fig2.tight_layout()
    plt.show()
    fig3, ax3 = plt.subplots(1, figsize=(5, 4))
    ax3.plot([0, 2.5], [0, 2.5], color='k', alpha=0.4, linestyle='--', linewidth=4)
    for i_c in range(len(coupling_levels)):
        ax3.plot(hyst_width_2[i_c], hyst_width_4[i_c],
                  color=colormap[i_c], marker='o', linestyle='')
    for i_s in range(nsubs):
        ax3.plot(hyst_width_2[:, i_s], hyst_width_4[:, i_s],
                  color='k', alpha=0.1)
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
    ax3.set_ylabel('Width freq = 4')
    ax3.set_xlabel('Width freq = 2')
    fig4.tight_layout()
    fig3.tight_layout()


def get_argmax_ndt_hyst_per_subject(responses_2, responses_4, barray_2, barray_4, coupling_levels,
                                    tFrame=26, fps=60, window_conv=None,
                                    ndtlist=np.arange(100)):
    nFrame = tFrame*fps
    hyst_widths = np.zeros((len(coupling_levels), len(responses_2[0]), len(ndtlist)))
    # --- FREQ = 2 ---
    for i_c, coupling in enumerate(coupling_levels):
        for i_s, subj_resp in enumerate(responses_2[i_c]):
            for i_ndt, ndt in enumerate(ndtlist):
                desc = np.roll(np.nanmean(subj_resp["desc"], axis=0)[::-1], -ndt)
                asc = np.roll(np.nanmean(subj_resp["asc"], axis=0), ndt)
                hyst_widths[i_c, i_s, i_ndt] += np.nansum(np.abs(desc-asc), axis=0) *\
                    np.diff(barray_2[:nFrame//2])[0]

    # --- FREQ = 4 ---
    for i_c, coupling in enumerate(coupling_levels):
        for i_s, subj_resp in enumerate(responses_4[i_c]):
            for i_ndt, ndt in enumerate(ndtlist):
                desc = np.roll(np.nanmean(subj_resp["desc"], axis=0)[::-1], -ndt)
                asc = np.roll(np.nanmean(subj_resp["asc"], axis=0), ndt)
                hyst_widths[i_c, i_s, i_ndt] += np.nansum(np.abs(desc-asc), axis=0) *\
                    np.diff(barray_4[:nFrame//2])[0]
    ndts_max = np.argmax(hyst_widths/2, axis=2)
    return hyst_widths, ndts_max
    

def get_analytical_approximations_areas(shuffling_levels=np.array([0., 0.7, 1]),
                                        b1=1, j1=1, tFrame=26):
    temp = 1/((1-np.array(shuffling_levels)+1e-1)*8*j1)  # multiplied by parameter J1
    omega_2 = 2*np.pi/tFrame
    omega_4 = 4*np.pi/tFrame
    h_0 = 3*b1  # multiplied by parameter B1
    area_2 = 4*np.pi*(1/temp * np.exp(-2/temp))*h_0**2 * omega_2/(omega_2**2 + 1)
    area_4 = 4*np.pi*(1/temp * np.exp(-2/temp))*h_0**2 * omega_4/(omega_4**2 + 1)
    return area_2, area_4


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
        noise_signal = np.array([scipy.interpolate.interp1d(time_interp, noise_exp[:, trial])(time) for trial in range(ntrials)]).T

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
        noise_signal = np.array([scipy.interpolate.interp1d(time_interp, noise_exp[:, trial])(time) for trial in range(ntrials)]).T

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
    fig, ax = plt.subplots(1, figsize=(5.5, 4))
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
    zscor = scipy.stats.zscore
    if zscore_number_switches:
        # zscore with respect to every p(shuffle), across subjects
        latency = zscor(latency, axis=0)
        switches = zscor(switches, axis=0)
        amplitude = zscor(amplitude, axis=0)
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


def get_blist(freq, nFrame):
    if abs(freq) == 2:
        difficulty_time_ref_2 = np.linspace(-2, 2, nFrame//2)
        stimulus = np.concatenate(([difficulty_time_ref_2, -difficulty_time_ref_2]))
    if abs(freq) == 4:
        difficulty_time_ref_4 = np.linspace(-2, 2, nFrame//4)
        stimulus = np.concatenate(([difficulty_time_ref_4, -difficulty_time_ref_4,
                                    difficulty_time_ref_4, -difficulty_time_ref_4]))
    if freq < 0:
        stimulus = -stimulus
    return stimulus


def plot_switch_rate(tFrame=26, fps=60, data_folder=DATA_FOLDER,
                     ntraining=8, coupling_levels=[0, 0.3, 1],
                     window_conv=10, bin_size=0.05):
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()

    responses_2, responses_4, barray_2, barray_4 = collect_responses(
        df, subjects, coupling_levels, fps=fps, tFrame=tFrame)
    
    timebins = np.arange(0, tFrame + bin_size, bin_size)
    xvals = timebins[:-1] + bin_size/2
    fig, axes = plt.subplots(ncols=2, figsize=(7.5, 4.))
    titles = ['Freq = 2', 'Freq = 4']
    for i_ax, ax in enumerate(axes):
        ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
        ax.set_xlabel('Time (s)'); ax.axvline(tFrame/(2+2*i_ax), color='k', alpha=0.4,
                                              linestyle='--', linewidth=3)
        ax.set_title(titles[i_ax], fontsize=12)
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    for i_c, coupling in enumerate(coupling_levels):
    # pick one coupling level (e.g. i_c = 0) and ascending responses
        bins, mean012, sem01, mean102, sem10, per_sub_rates_01, per_sub_rates_10 =\
            average_switch_rates_dir(responses_2[i_c], fps=fps, bin_size=bin_size, join=True)
        bins, mean014, sem01, mean104, sem10, per_sub_rates_01, per_sub_rates_10 =\
            average_switch_rates_dir(responses_4[i_c], fps=fps, bin_size=bin_size/2, join=True)
        convolved_vals2 = np.convolve(mean102, np.ones(window_conv)/window_conv,
                                     "same")
        convolved_vals4 = np.convolve(mean104, np.ones(window_conv)/window_conv,
                                     "same")
        axes[0].plot(xvals, convolved_vals2, color=colormap[i_c], linewidth=3, label=f'{1-coupling}')
        axes[1].plot(xvals/2, convolved_vals4, color=colormap[i_c], linewidth=3)
    axes[0].legend(frameon=False, title='p(shuffle)'); axes[0].set_ylabel('Switch rate (L to R, asc. start)')
    fig.tight_layout()


def join_trial_responses(subj):
    """
    subj: dict {"asc": array, "desc": array, "ini_side": list}
    Returns joined responses (n_trials, timepoints).
    """
    asc, desc, ini_sides = subj["asc"], subj["desc"], subj["ini_side"]
    joined = []
    for i in range(len(ini_sides)):
        if ini_sides[i] == -1:   # ascending first
            continue
            trial = np.concatenate([asc[i], desc[i]])
        else:                   # descending first
            trial = np.concatenate([desc[i], asc[i]])
        joined.append(trial)
    return np.array(joined)


def average_switch_rates_dir(responses, fps=60, bin_size=1.0, join=True):
    """
    Compute average 0→1 and 1→0 switch rates across subjects.
    If join=True, concatenates asc+desc before counting.
    """
    per_sub_rates_01, per_sub_rates_10 = [], []
    bins_ref = None

    for subj in responses:
        arr = join_trial_responses(subj) if join else subj["asc"]
        bins, r01, r10, _, _, _ = compute_switch_rate_from_array_dir(
            arr, fps=fps, bin_size=bin_size)
        if bins_ref is None:
            bins_ref = bins
        per_sub_rates_01.append(r01)
        per_sub_rates_10.append(r10)

    per_sub_rates_01 = np.vstack(per_sub_rates_01)
    per_sub_rates_10 = np.vstack(per_sub_rates_10)

    mean01 = np.nanmean(per_sub_rates_01, axis=0)
    sem01 = np.nanstd(per_sub_rates_01, axis=0, ddof=0) / np.sqrt(per_sub_rates_01.shape[0])
    mean10 = np.nanmean(per_sub_rates_10, axis=0)
    sem10 = np.nanstd(per_sub_rates_10, axis=0, ddof=0) / np.sqrt(per_sub_rates_10.shape[0])

    return bins_ref, mean01, sem01, mean10, sem10, per_sub_rates_01, per_sub_rates_10



def get_switch_indices_with_dir(arr):
    """
    Detect switches in a response array (0/1 with NaN).
    Returns two arrays of indices:
      - idx_01: where a 0→1 switch occurred
      - idx_10: where a 1→0 switch occurred
    """
    prev = arr[:-1]
    nxt = arr[1:]
    valid = (~np.isnan(prev)) & (~np.isnan(nxt))
    idx_01 = np.where(valid & (prev == 0) & (nxt == 1))[0] + 1
    idx_10 = np.where(valid & (prev == 1) & (nxt == 0))[0] + 1
    return idx_01, idx_10



def compute_switch_rate_from_array_dir(response_array, fps=60, bin_size=1.0):
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
    ax[0].set_xlabel('Sensory evidence, B(t)')
    ax[1].set_xlabel('Sensory evidence, B(t)')
    ax[0].set_ylabel('Proportion of rightward responses')
    ax[0].legend(title='p(shuffle)', frameon=False)
    ax[0].set_title('Freq = 2', fontsize=14)
    ax[1].set_title('Freq = 4', fontsize=14)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'hysteresis_basic_plot.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'hysteresis_basic_plot.svg', dpi=400, bbox_inches='tight')


def hysteresis_basic_plot_all_subjects(coupling_levels=[0, 0.3, 1],
                                      fps=60, tFrame=26, data_folder=DATA_FOLDER,
                                      nbins=13, ntraining=4, arrows=False):
    nFrame = fps*tFrame
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()
    fig, ax = plt.subplots(ncols=6, nrows=int(np.ceil(len(subjects)/3)), figsize=(22, 23))
    ax = ax.flatten()
    fig2, ax2 = plt.subplots(ncols=4, nrows=int(np.ceil(len(subjects)/4)), figsize=(18, 20))
    ax2 = ax2.flatten()
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
        # a.set_ylim(-0.025, 1.085)
        # a.set_yticks([0, 0.5, 1])
        # a.set_xlim(-2.05, 2.05)
        # a.set_xticks([-2, 0, 2], [-1, 0, 1])
        # a.axhline(0.5, color='k', linestyle='--', alpha=0.2)
        # a.axvline(0., color='k', linestyle='--', alpha=0.2)
    colormap = pl.cm.Oranges(np.linspace(0.3, 1, 3))[::-1]
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    for i_s, subject in enumerate(subjects):
        df_sub = df.loc[df.subject == subject]
        for i_c, coupling in enumerate(coupling_levels):
            df_filt = df_sub.loc[df_sub.pShuffle.round(2) == round(1-coupling, 2)]
            df_freq_2 = df_filt.loc[df_filt.freq == 2]
            response_array_asc, response_array_desc, barray_2, _ = get_response_and_blist_array(df_freq_2, fps=fps,
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
            response_array_4_asc, response_array_4_desc, barray_4, _ = get_response_and_blist_array(df_freq_4, fps=fps,
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
        f, f2, f3, g = plot_noise_before_switch(data_folder=DATA_FOLDER, fps=60, tFrame=26,
                                                steps_back=65, steps_front=20,
                                                shuffle_vals=[1, 0.7, 0], violin=False, sub=subject,
                                                avoid_first=False, window_conv=1, zscore_number_switches=False, ax=ax2[i_s],
                                                fig=fig)
        # ax2[i_s].set_title(subject)
        plt.close(f2)
        plt.close(f3)
        # plt.close(plt.figure(3))
        plt.close(g.fig)
        if i_s < 32:
            ax2[i_s].set_xticks([-1, 0, 1], ['', '', ''])
        if i_s < 32:
            ax[i_s].set_xticks([-1, 0, 1], ['', '', ''])
        if (i_s+1) % 4 != 0:
            ax2[i_s].set_yticks([])
        if (i_s+1) % 6 != 0:
            ax[i_s].set_yticks([])
    ax[i_s*2].set_xlabel('Sensory evidence, B(t)')
    ax[i_s*2+1].set_xlabel('Sensory evidence, B(t)')
    ax[i_s*2-1].set_xlabel('Sensory evidence, B(t)')
    ax[i_s*2-2].set_xlabel('Sensory evidence, B(t)')
    ax[0].set_ylabel('Proportion of rightward responses')
    ax[0].legend(title='p(shuffle)', frameon=False)
    ax[0].set_title('Freq = 2', fontsize=14)
    ax[1].set_title('Freq = 4', fontsize=14)
    ax[2].set_title('Freq = 2', fontsize=14)
    ax[3].set_title('Freq = 4', fontsize=14)
    ax[4].set_title('Freq = 2', fontsize=14)
    ax[5].set_title('Freq = 4', fontsize=14)
    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig(SV_FOLDER + 'hysteresis_basic_plot_all.png', dpi=200, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'hysteresis_basic_plot_all.svg', dpi=200, bbox_inches='tight')
    fig2.savefig(SV_FOLDER + 'noise_kernel_all.png', dpi=200, bbox_inches='tight')
    fig2.savefig(SV_FOLDER + 'nosie_kernel_all.svg', dpi=200, bbox_inches='tight')


def hysteresis_basic_plot_simulation(coup_vals=np.array((0., 0.3, 1))*0.27+0.02,
                                     fps=60, nsubs=20,
                                     n=4, nsims=100, b_list=np.linspace(-0.5, 0.5, 501)):
    b_list_2 = np.concatenate((b_list[:-1], b_list[1:][::-1]))
    b_list_4 = np.concatenate((b_list[:-1][::2], b_list[1:][::-2], b_list[:-1][::2], b_list[1:][::-2]))
    nFrame = len(b_list_2)
    dt  = 1/fps
    time = np.arange(0, nFrame, 1)*dt
    tau = 0.1
    indep_noise = np.sqrt(dt/tau)*np.random.randn(nFrame, nsims, nsubs)*0.08
    choice = np.zeros((len(coup_vals), nFrame, nsims, 2, nsubs))
    for i_j, j in enumerate(coup_vals):
        for freq in range(2):
            stimulus = [b_list_2, b_list_4][freq]
            for sub in range(nsubs):
                for sim in range(nsims):
                    # convergence
                    x = np.random.rand()
                    for i in range(50):
                        x = sigmoid(2*j*n*(2*x-1))
                    vec = [x]
                    for t in range(1, nFrame):
                        x = x + dt*(sigmoid(2*j*n*(2*x-1)+2*stimulus[t])-x)/tau + indep_noise[t, sim, sub]
                        vec.append(x)
                        if x < 0.4:
                            ch = 0.
                        if x > 0.6:
                            ch = 1.
                        if 0.4 <= x <= 0.6 and t > 0:
                            ch = choice[i_j, t-1, sim, freq, sub] 
                        choice[i_j, t, sim, freq, sub] = ch
    # np.save(DATA_FOLDER + 'choice_hysteresis.npy', choice)
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(7.5, 4))
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylim(-0.025, 1.08)
        a.set_yticks([0, 0.5, 1])
        a.set_xticks([-0.5, 0, 0.5], [-1, 0, 1])
        # a.tick_params(axis='x', rotation=45)
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    lsts = ['solid', 'solid']
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


def plot_hysteresis_width_simluations(coup_vals=np.arange(0.05, 0.35, 1e-2),
                                      b_list=np.linspace(-0.5, 0.5, 501)):
    b_list_2 = np.concatenate((b_list[:-1], b_list[1:][::-1]))
    b_list_4 = np.concatenate((b_list[:-1][::2], b_list[1:][::-2], b_list[:-1][::2], b_list[1:][::-2]))
    # choice is a len(coup_vals) x timepoints x nsims x freqs(2)
    choice = np.load(DATA_FOLDER + 'choice_hysteresis.npy')
    n_coup, nFrame, nsims, nfreqs = choice.shape
    hyst_width_2 = np.zeros((n_coup))
    hyst_width_4 = np.zeros((n_coup))
    f0, ax0 = plt.subplots(ncols=2, figsize=(9, 4.5))
    colormap = pl.cm.Blues(np.linspace(0.3, 1, 3))
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
            if i_c in [0, n_coup//2, n_coup-1]:
                ax0[freq].plot(stimulus[asc_mask], ascending, color=colormap[color], linewidth=4,
                               label=f'{coup_vals[i_c]}')
                ax0[freq].plot(stimulus[~asc_mask], descending, color=colormap[color], linewidth=4)
                color += 1
    for a in ax0:
        a.set_xlabel('Sensory evidence B(t)'); a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
    ax0[0].set_ylabel('P(choice = R)');  ax0[0].legend(frameon=False, title='J')
    f0.tight_layout()
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
            
    
def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = scipy.stats.pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.3f}', xy=(.1, 1.), xycoords=ax.transAxes)
            

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
        noise_signal = np.array([scipy.interpolate.interp1d(time_interp, noise_exp[:, trial])(time) for trial in range(ntrials)]).T
    
        
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
    fig, ax = plt.subplots(1, figsize=(5.5, 4))
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


def stars_pval(pval):
    s = 'ns'
    if pval < 0.05 and pval >= 0.01:
        s = '*'
    if pval < 0.01 and pval >= 0.001:
        s = '**'
    if pval < 0.001:
        s = '***'
    return s


def plot_noise_before_switch(data_folder=DATA_FOLDER, fps=60, tFrame=18,
                             steps_back=60, steps_front=20,
                             shuffle_vals=[1, 0.7, 0], violin=False, sub='s_1',
                             avoid_first=False, window_conv=1, zscore_number_switches=False, ax=None,
                             fig=None, normalize_variables=True, hysteresis_area=False):
    nFrame = fps*tFrame
    df = load_data(data_folder + '/noisy/', n_participants='all')
    if sub is not None:
        df = df.loc[df.subject == sub]
    # print(len(df.trial_index.unique()))
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
    # all_noises = [[], [], []]
    for i_sub, subject in enumerate(subs):
        df_sub = df.loc[df.subject == subject]
        for i_sh, pshuffle in enumerate(shuffle_vals):
            df_coupling = df_sub.loc[df_sub.pShuffle == pshuffle]
            trial_index = df_coupling.trial_index.unique()
            # mean_vals_noise_switch_all_trials = np.empty((len(trial_index), steps_back+steps_front))
            mean_vals_noise_switch_all_trials = np.empty((1, steps_back+steps_front))
            mean_vals_noise_switch_all_trials[:] = np.nan
            # latency = []
            # height = []
            number_switches = []
            for i_trial, trial in enumerate(trial_index):
                df_trial = df_coupling.loc[df_coupling.trial_index == trial]
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
                #     mean_vals_noise_switch_all_trials[i_trial, :] =\
                #         np.nanmean(mean_vals_noise_switch, axis=0)
                # if len(idx_0) == 0 and len(idx_1) == 0:
                #     continue
                # else:
                #     # take max values and latencies across time (axis=1) and then average across trials
                #     latency.append(np.nanmean(np.argmax(mean_vals_noise_switch, axis=1)))
                #     height.append(np.nanmean(np.nanmax(mean_vals_noise_switch, axis=1)))
            mean_vals_noise_switch_all_trials = mean_vals_noise_switch_all_trials[1:]
            # it's better to compute afterwards, with the average peak per coupling
            # because trial by trial there is a lot of noise and that breaks the mean/latency
            # it gets averaged out
            mean_number_switchs_coupling[i_sh, i_sub] = tFrame/ np.nanmean(np.array(number_switches))
            # axis=0 means average across switches (leaves time coords)
            # axis=1 means average across time (leaves switches coords)
            averaged_and_convolved_values = np.convolve(np.nanmean(mean_vals_noise_switch_all_trials, axis=0),
                                                                          np.ones(window_conv)/window_conv, 'same')
            mean_peak_latency[i_sh, i_sub] = (np.nanmean(np.argmax(mean_vals_noise_switch_all_trials, axis=1)) - steps_back)/fps
            mean_peak_amplitude[i_sh, i_sub] = np.nanmean(np.nanmax(mean_vals_noise_switch_all_trials, axis=1))
            mean_vals_noise_switch_coupling[i_sh, :, i_sub] = averaged_and_convolved_values
            err_vals_noise_switch_coupling[i_sh, :, i_sub] = np.nanstd(mean_vals_noise_switch_all_trials, axis=0) / np.sqrt(len(trial_index))
        if len(subs) > 1 and zscore_number_switches:
            # mean_number_switchs_coupling[:, i_sub] = zscor(mean_number_switchs_coupling[:, i_sub])
            # mean_peak_latency[:, i_sub] = zscor(mean_peak_latency[:, i_sub])
            # mean_peak_amplitude[:, i_sub] = zscor(mean_peak_amplitude[:, i_sub])
            label = 'z-scored '
        else:
            label = ''
        #     mean_vals_noise_switch_coupling[:, :, i_sub] = zscor(mean_vals_noise_switch_coupling[:, :, i_sub], nan_policy='omit')
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5.5, 4))
    fig2, ax2 = plt.subplots(1, figsize=(5.5, 4))
    fig3, ax34567 = plt.subplots(ncols=3, nrows=2, figsize=(12.5, 8))
    ax34567= ax34567.flatten()
    ax3, ax4, ax5, ax6, ax7, ax8 = ax34567

    def corrfunc(x, y, ax=None, **kws):
        """Plot the correlation coefficient in the top left hand corner of a plot."""
        r, _ = scipy.stats.pearsonr(x, y)
        ax = ax or plt.gca()
        ax.annotate(f'ρ = {r:.3f}', xy=(.1, 1.), xycoords=ax.transAxes)
    if hysteresis_area:
        np.save(DATA_FOLDER + 'mean_peak_amplitude_per_subject.npy', mean_peak_amplitude)
        np.save(DATA_FOLDER + 'mean_peak_latency_per_subject.npy', mean_peak_latency)
        np.save(DATA_FOLDER + 'mean_number_switches_per_subject.npy', mean_number_switchs_coupling)
        hyst_width_2 = np.load(DATA_FOLDER + 'hysteresis_width_freq_2.npy')
        hyst_width_4 = np.load(DATA_FOLDER + 'hysteresis_width_freq_4.npy')
        # switch_time_diff_2 = np.load(DATA_FOLDER + 'switch_time_diff_freq_2.npy')
        # switch_time_diff_4 = np.load(DATA_FOLDER + 'switch_time_diff_freq_4.npy')
        datframe = pd.DataFrame({'Amplitude': mean_peak_amplitude.flatten(),
                                 'Latency': mean_peak_latency.flatten(),
                                 'Dominance': mean_number_switchs_coupling.flatten(),
                                 'Width f2': hyst_width_2.flatten(),
                                 'Width f4': hyst_width_4.flatten()})
        delta_dominance = mean_number_switchs_coupling[-1]-mean_number_switchs_coupling[0]
        delta_hysteresis_2 = hyst_width_2[-1]-hyst_width_2[0]
        delta_hysteresis_4 = hyst_width_4[-1]-hyst_width_4[0]
        datframe_deltas  = pd.DataFrame({'Difference dominance': delta_dominance,
                                         'Difference H2 width': delta_hysteresis_2,
                                         'Difference H4 width': delta_hysteresis_4})
        deltasplot = sns.pairplot(datframe_deltas)
        deltasplot.map_lower(corrfunc)
    else:
        datframe = pd.DataFrame({'Amplitude': mean_peak_amplitude.flatten(),
                                 'Latency': mean_peak_latency.flatten(),
                                 'Dominance': mean_number_switchs_coupling.flatten()})
    g = sns.pairplot(datframe)
    g.map_lower(corrfunc)
    for a in [ax, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
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
        # a, b, c = mean_vals_noise_switch_coupling
        # scipy.stats.ttest_rel(a, b)
        # scipy.stats.ttest_rel(a, c)
        # scipy.stats.ttest_rel(b, c)
        h_1, h_07, h_0 = np.nanmean(a), np.nanmean(b), np.nanmean(c)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
    heights = [h_1, h_07, h_0]
    offset = 0.55 if violin else 0.05
    # ax2.set_ylim(np.nanmin(a)-0.1, np.nanmax(heights)+0.2)
    for a in range(3):
        ax2.text(a, heights[a]+offset, f"{pvs[a]}", ha='center', va='bottom', color='k',
                  fontsize=12)
        # x1 = a % 3 + 5e-2
        # x2 = (a+1) % 3 - 5e-2
        # offset_bar = 0.1 if a < 2 else 0.2
        # max_height = np.max([heights[a % 2]+offset,
        #                      heights[(a+1) % 2]+offset]) + offset_bar
        # min_height = np.min([heights[a % 2]+offset,
        #                      heights[(a+1) % 2]+offset]) + offset_bar
        # ax2.plot([x1, x1, x2, x2], [min_height,
        #                             max_height,
        #                             max_height,
        #                             min_height],
        #          linewidth=1, color='k')
    ax2.set_xticks([0, 1, 2], [1, 0.7, 0])
    ax2.set_xlabel('p(Shuffle)')
    ax2.set_ylabel('Noise before switch')
    ax.legend(title='p(shuffle)', frameon=False)
    ax.set_xlabel('Time from switch (s)')
    ax.set_ylabel('Noise')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig.savefig(SV_FOLDER + 'noise_before_switch_experiment.png', dpi=100, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'noise_before_switch_experiment.svg', dpi=100, bbox_inches='tight')
    return fig, fig2, fig3, g


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


def prior_parameters(n_simuls=100):
    prior =\
        MultipleIndependent([
            Uniform(torch.tensor([-0.5]),
                    torch.tensor([2.5])),  # J_eff / N = (J_0 + J_1*coupling) (N=4)
            Uniform(torch.tensor([-0.4]),
                    torch.tensor([1.])),  # B1
            # Uniform(torch.tensor([0.05]),
            #         torch.tensor([5])),  # tau
            Uniform(torch.tensor([0.]),
                    torch.tensor([0.45])),  # threshold distance
            Uniform(torch.tensor([0.0]),
                    torch.tensor([0.35]))],  # noise
            validate_args=False)
    theta_all = prior.sample((n_simuls,))
    return theta_all, prior


def simulator_5_params(params, freq, nFrame=1560, fps=60,
                       return_choice=False):
    """
    Simulator. Takes set of `params` and simulates the system, returning summary statistics.
    Params: J_eff, B_eff, tau, threshold distance, noise
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
    j_eff, b_par, th, sigma = params
    tau = 0.5
    dt = 1/fps
    b_eff = stimulus*b_par
    noise = np.random.randn(nFrame)*sigma*np.sqrt(dt/tau)
    x = np.zeros(nFrame)
    x[0] = 0.5
    for t in range(1, nFrame):
        drive = sigmoid(2 * (j_eff * (2 * x[t-1] - 1) + b_eff[t]))
        x[t] = x[t-1] + dt * (drive - x[t-1]) / tau + noise[t]
    choice = np.zeros(nFrame)
    choice[x < (0.5-th)] = -1.
    choice[x > (0.5+th)] = 1.
    mid_mask = (x >= (0.5-th)) & (x <= (0.5+th))
    for t in np.where(mid_mask)[0]:
        choice[t] = choice[t - 1] if t > 0 else 0.
    if return_choice:
        return choice, x
    else:
        return return_input_output_for_network(params, choice, freq, nFrame=nFrame, fps=fps)


def sbi_training_5_params(n_simuls=10000, fps=60, tFrame=26, data_folder=DATA_FOLDER,
                          load_net=False):
    """
    Function to train network to approximate likelihood/posterior.
    """
    nFrame = fps*tFrame
    if not load_net:
        # experimental conditions
        freq = np.random.choice([2, 4, -2, -4], n_simuls)
        # sample prior
        theta, prior = prior_parameters(n_simuls=n_simuls)
        inference = sbi.inference.MNLE(prior=prior)
        theta_np = theta.detach().numpy()
        # simulate
        print(f'Starting {n_simuls} simulations')
        time_ini = time.time()
        training_input_set = np.zeros((theta_np.shape[1]+3), dtype=np.float32)
        training_output_set = np.empty((2), dtype=np.float32)
        for i in range(n_simuls):
            input_net, output_net = simulator_5_params(params=theta_np[i], freq=freq[i],
                                                       nFrame=nFrame, fps=fps)
            training_input_set = np.row_stack((training_input_set, input_net))
            training_output_set = np.row_stack((training_output_set, output_net))
        training_input_set = training_input_set[1:].astype(np.float32)
        sims_tensor = torch.tensor(training_output_set[1:].astype(np.float32))
        theta = torch.tensor(training_input_set)
        print('Simulations finished.\nIt took:' + str(round((time.time()-time_ini)/60)) + ' min.')
        # train density estimator
        print(f'Starting training NLE with {sims_tensor.shape[0]} simulations')
        time_ini = time.time()
        density_estimator = inference.append_simulations(theta, sims_tensor).train()
        print('Training finished.\nIt took:' + str(round((time.time()-time_ini)/60)) + ' min.')
        posterior = inference.build_posterior(density_estimator, prior=prior)
        with open(data_folder + f"/nle_5pars_{n_simuls}.p", "wb") as fh:
            pickle.dump(dict(estimator=density_estimator,
                             num_simulations=n_simuls), fh)
        with open(data_folder + f"/posterior_5pars_{n_simuls}.p", "wb") as fh:
            pickle.dump(dict(posterior=posterior,
                             num_simulations=n_simuls), fh)
    if load_net:
        with open(data_folder + f"/nle_5pars_{n_simuls}.p", 'rb') as f:
            density_estimator = pickle.load(f)
        with open(data_folder + f"/posterior_5pars_{n_simuls}.p", 'rb') as f:
            posterior = pickle.load(f)
        density_estimator = density_estimator['estimator']
        posterior = posterior['posterior']
    return density_estimator, posterior


def parameter_recovery_5_params(n_simuls_network=100000, fps=60, tFrame=26,
                                n_pars_to_fit=50, n_sims_per_par=108,
                                sv_folder=SV_FOLDER, simulate=False,
                                load_net=True, not_plot_and_return=False):
    density_estimator, _ = sbi_training_5_params(n_simuls=n_simuls_network, fps=fps, tFrame=tFrame,
                                                 data_folder=DATA_FOLDER, load_net=load_net)
    lb = np.array([-0.5, -0.4, 0.0, 0.0])
    ub = np.array([2.5, 1., 0.45, 0.35])
    plb = np.array([-0.3, -0.1, 0.01, 0.05])
    pub = np.array([1.6, 0.4, 0.3, 0.3])
    x0 = np.array([0.2, 0.1, 0.2, 0.15])
    nFrame = fps*tFrame
    orig_params = np.zeros((n_pars_to_fit, len(x0)))
    recovered_params = np.zeros((n_pars_to_fit, len(x0)))
    for par in tqdm.tqdm(range(n_pars_to_fit)):
        # simulate
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
    if not_plot_and_return:
        return orig_params, recovered_params
    else:
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 9))
        numpars = 4
        ax = ax.flatten()
        # labels = ['Jeff', ' B1',  'Tau', 'Thres.', 'sigma']
        labels = ['Jeff', ' B1', 'Thres.', 'sigma']
        # xylims = [[0, 3], [0, 0.8], [0, 0.7], [0, 0.5], [0, 0.5]]
        for i_a in range(numpars):
            a = ax[i_a]
            a.plot(orig_params[:, i_a], recovered_params[:, i_a], color='k', marker='o',
                   markersize=5, linestyle='')
            maxval = np.max([orig_params[:, i_a], recovered_params[:, i_a]])
            minval = np.min([orig_params[:, i_a], recovered_params[:, i_a]])
            a.set_xlim(minval-1e-2, maxval+1e-2)
            a.set_ylim(minval-1e-2, maxval+1e-2)
            a.plot([minval, maxval], [minval, maxval],
                   color='k', linestyle='--',
                   alpha=0.3, linewidth=4)
            # a.plot(xylims[i_a], xylims[i_a], color='k', alpha=0.3)
            a.set_title(labels[i_a])
            a.set_xlabel('Original parameters')
            a.set_ylabel('Recovered parameters')
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
        # ax[-1].axis('off')
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
        labels_reduced = labels
        ax.set_xticks(np.arange(numpars), labels, fontsize=12, rotation=45)  # , rotation='270'
        ax.set_yticks(np.arange(numpars), labels_reduced, fontsize=12)
        ax.set_xlabel('Original parameters', fontsize=14)
        # compute correlation matrix
        mat_corr = np.corrcoef(recovered_params.T, rowvar=True)
        mat_corr *= np.tri(*mat_corr.shape, k=-1)
        # plot correlation matrix
        im = ax2.imshow(mat_corr, cmap='bwr', vmin=-1, vmax=1)
        ax2.step(np.arange(0, numpars)-0.5, np.arange(0, numpars)-0.5, color='k',
                 linewidth=.7)
        ax2.set_xticks(np.arange(numpars), labels, fontsize=12, rotation=45)  # , rotation='270'
        ax2.set_yticks(np.arange(numpars), labels, fontsize=12)
        ax2.set_xlabel('Inferred parameters', fontsize=14)
        ax2.set_ylabel('Inferred parameters', fontsize=14)
        fig2.tight_layout()
        fig.savefig(SV_FOLDER + 'param_recovery_all.png', dpi=400, bbox_inches='tight')
        fig.savefig(SV_FOLDER + 'param_recovery_all.svg', dpi=400, bbox_inches='tight')
        fig2.savefig(SV_FOLDER + 'param_recovery_correlations.png', dpi=400, bbox_inches='tight')
        fig2.savefig(SV_FOLDER + 'param_recovery_correlations.svg', dpi=400, bbox_inches='tight')


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
        fig.savefig(SV_FOLDER + 'param_recovery_all.svg', dpi=400, bbox_inches='tight')
        fig2.savefig(SV_FOLDER + 'param_recovery_correlations.png', dpi=400, bbox_inches='tight')
        fig2.savefig(SV_FOLDER + 'param_recovery_correlations.svg', dpi=400, bbox_inches='tight')


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
    output_network = np.column_stack((np.roll(responses, 1), output_network_0))
    params_repeated = np.column_stack(np.repeat(params.reshape(-1, 1), result.shape[0], 1))
    input_network = np.column_stack((params_repeated, result, np.repeat(freq, result.shape[0])))
    return input_network[1:max_number], output_network[1:max_number]


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
        - Dominance duration
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


def lmm_hysteresis_dominance(freq=2, plot_summary=False,
                             slope_random_effect=False, plot_individual=False):
    y2 = np.load(DATA_FOLDER + 'hysteresis_width_freq_2.npy')
    y4 = np.load(DATA_FOLDER + 'hysteresis_width_freq_4.npy')
    x = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
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
        plt.xlabel('Dominance')
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
        plt.plot(x, y, color='k', marker='o', linestyle='')
        fig.tight_layout()
    else:
        return intercepts, slopes, result


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


def fun_get_neg_log_likelihood_data(theta, x, result, density_estimator, pShuffle,
                                    epsilon=1e-3, ct_distro=1/(26*60), use_j0=False,
                                    contaminants=True):
    result = torch.atleast_2d(result); x = torch.atleast_2d(x)
    if not use_j0 and theta.shape[0] < 5:
        theta = np.concatenate(([np.random.rand()], theta))
    j_eff = (1-pShuffle)*theta[1] + theta[0]*use_j0
    params_repeated = np.column_stack(np.repeat(theta[1:].reshape(-1, 1), result.shape[0], 1))
    params_repeated[:, 0] = j_eff
    condition = torch.tensor(np.column_stack((params_repeated, result))).to(torch.float32)
    if contaminants:
        likelihood_nle = torch.exp(density_estimator.log_prob(x, condition))
        full_likelihood_with_contaminant = likelihood_nle*(1-epsilon)+epsilon*ct_distro
        log_likelihood_full = torch.log(full_likelihood_with_contaminant)
    else:
        log_likelihood_full = density_estimator.log_prob(x, condition)
    return -torch.nansum(log_likelihood_full).detach().numpy()


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


def get_log_likelihood_all_data(data_folder=DATA_FOLDER,
                                sv_folder=SV_FOLDER, use_j0=False,
                                n_simuls_network=100000, ntraining=8,
                                fps=60, tFrame=26):
    density_estimator, _ = sbi_training_5_params(n_simuls=n_simuls_network, fps=fps, tFrame=tFrame,
                                                 data_folder=DATA_FOLDER, load_net=True)
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()
    label_j0 = '_with_j0' if use_j0 else ''
    numpars = 4 + use_j0*1.
    bics = []; aics = []; nllh = []
    for i_s, subject in enumerate(subjects):
        df_subject = df.loc[df.subject == subject]
        tensor_input, tensor_output, idx_filt = prepare_data_for_fitting(df_subject, fps=fps, tFrame=tFrame)
        condition = tensor_input[1:].astype(np.float32)
        pshuffle = df_subject.pShuffle.values[~idx_filt][1:]
        x = tensor_output[1:].astype(np.float32)
        x = torch.tensor(x).unsqueeze(0).to(torch.float32)
        condition = torch.tensor(condition).to(torch.float32)
        params = np.load(sv_folder + '/pars_5_subject_' + subject + str(n_simuls_network) + label_j0 + '.npy')
        neg_log_likelihood_s = fun_get_neg_log_likelihood_data(params, x, condition, density_estimator, pshuffle,
                                                               use_j0=use_j0, contaminants=True)
        aic_s = 2*numpars + 2*neg_log_likelihood_s
        bic_s = numpars*np.log(tensor_input.shape[0]) + 2*neg_log_likelihood_s
        bics.append(bic_s)
        aics.append(aic_s)
        nllh.append(neg_log_likelihood_s)
    return np.array(nllh), np.array(aics), np.array(bics)


def simulated_subjects(data_folder=DATA_FOLDER, tFrame=26, fps=60,
                       sv_folder=SV_FOLDER, ntraining=8, n_simuls_network=50000,
                       plot=False, simulate=False, use_j0=False):
    label_j0 = '_with_j0' if use_j0 else ''
    nFrame = fps*tFrame
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    ntrials = 72
    subjects = df.subject.unique()
    # subjects = ['s_1']
    choices_all_subject = np.zeros((len(subjects), ntrials, nFrame))
    for i_s, subject in enumerate(subjects):
        print('Simulating subject', subject)
        fitted_params = np.load(sv_folder + '/pars_5_subject_' + subject + str(n_simuls_network) + label_j0 + '.npy')
        df_subject = df.loc[df.subject == subject]
        pshuffles = df_subject.groupby('trial_index')['pShuffle'].mean().values
        ini_side = df_subject.groupby('trial_index')['initial_side'].mean().values
        frequencies = df_subject.groupby('trial_index')['freq'].mean().values*ini_side
        if simulate:
            choice_all = np.zeros((ntrials, nFrame))
            for trial in range(ntrials):
                j_eff = (1-pshuffles[trial])*fitted_params[1] + fitted_params[0]*use_j0
                params = fitted_params[1:]
                params[0] = j_eff
                choice, _ = simulator_5_params(params=params, freq=frequencies[trial], nFrame=nFrame,
                                               fps=fps, return_choice=True)
                choice_all[trial, :] = choice
            np.save(sv_folder + f'choice_matrix_subject_{subject}.npy', choice_all)
        else:
            choice_all = np.load(sv_folder + f'choice_matrix_subject_{subject}.npy')
        choices_all_subject[i_s] = choice_all
    if plot:
        unique_shuffle = [1., 0.7, 0.]
        nshuffle = len(unique_shuffle)
        f2, ax2 = plt.subplots(ncols=2, figsize=(9, 4.5))
        colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
        # containers: [shuffle, subject, timepoints]
        ascending_subjects_2 = np.full((nshuffle, len(subjects), nFrame//2), np.nan)
        descending_subjects_2 = np.full((nshuffle, len(subjects), nFrame//2), np.nan)
        ascending_subjects_4 = np.full((nshuffle, len(subjects), nFrame//4), np.nan)
        descending_subjects_4 = np.full((nshuffle, len(subjects), nFrame//4), np.nan)
        
        for i_s, subject in enumerate(subjects):
            df_subject = df.loc[df.subject == subject]
            pshuffles = np.round(df_subject.groupby('trial_index')['pShuffle'].mean().values, 3)
            ini_side = np.round(df_subject.groupby('trial_index')['initial_side'].mean().values, 1)
            frequencies = np.round(df_subject.groupby('trial_index')['freq'].mean().values * ini_side, 2)
            choice_all = choices_all_subject[i_s]
            ntrials = len(frequencies)
            # trial-level accumulators
            trial_ascending_subjects_2 = np.full((ntrials, nshuffle, nFrame//2), np.nan)
            trial_descending_subjects_2 = np.full((ntrials, nshuffle, nFrame//2), np.nan)
            trial_ascending_subjects_4 = np.full((ntrials*2, nshuffle, nFrame//4), np.nan)
            trial_descending_subjects_4 = np.full((ntrials*2, nshuffle, nFrame//4), np.nan)
        
            for i_trial, freqval in enumerate(frequencies):
                i_sh = np.where(unique_shuffle == pshuffles[i_trial])[0][0]
                stimulus = get_blist(freq=freqval, nFrame=nFrame)
        
                response_raw = choice_all[i_trial, :]
                response_raw[response_raw == 0] = np.nan
                response_raw = (response_raw + 1) / 2
        
                if abs(freqval) == 4:
                    response_raw = np.column_stack((response_raw[:nFrame//2],
                                                    response_raw[nFrame//2:]))
                    stimulus = stimulus[:nFrame//2]
        
                asc_mask = np.sign(np.gradient(stimulus)) > 0
                ascending = response_raw[asc_mask].T
                descending = response_raw[~asc_mask].T
        
                if np.abs(freqval) == 4:
                    trial_descending_subjects_4[[i_trial, ntrials], i_sh] = descending
                    trial_ascending_subjects_4[[i_trial, ntrials], i_sh] = ascending
                else:
                    trial_descending_subjects_2[i_trial, i_sh] = descending
                    trial_ascending_subjects_2[i_trial, i_sh] = ascending
            ascending_subjects_2[:, i_s] = np.nanmean(trial_ascending_subjects_2, axis=0)
            descending_subjects_2[:, i_s] = np.nanmean(trial_descending_subjects_2, axis=0)
            ascending_subjects_4[:, i_s] = np.nanmean(trial_ascending_subjects_4, axis=0)
            descending_subjects_4[:, i_s] = np.nanmean(trial_descending_subjects_4, axis=0)

        for freq_idx, freqval in enumerate([2, 4]):
            stimulus = get_blist(freq=freqval, nFrame=nFrame)
            if freqval == 4:
                stimulus = stimulus[:nFrame//2]
        
            asc_mask = np.sign(np.gradient(stimulus)) > 0
        
            for i_sh in range(nshuffle):
                if freqval == 4:
                    ascending_subjects = ascending_subjects_4
                    descending_subjects = descending_subjects_4
                else:
                    ascending_subjects = ascending_subjects_2
                    descending_subjects = descending_subjects_2
        
                # average across subjects
                asc_vals = np.nanmean(ascending_subjects[i_sh], axis=0)
                desc_vals = np.nanmean(descending_subjects[i_sh], axis=0)
        
                # plot
                ax2[freq_idx].plot(stimulus[asc_mask],
                                   asc_vals[:asc_mask.sum()],
                                   color=colormap[i_sh], linewidth=3)
                ax2[freq_idx].plot(stimulus[~asc_mask],
                                   desc_vals[:(~asc_mask).sum()],
                                   color=colormap[i_sh], linewidth=3)
        for a in ax2:
            a.set_xlabel('Sensory evidence B(t)')
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            a.axhline(0.5, color='k', linestyle='--', alpha=0.2)
            a.axvline(0., color='k', linestyle='--', alpha=0.2)
            a.set_ylim(-0.1, 1.1)
        ax2[0].set_ylabel('P(choice = R)')
        # fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
        # ax.plot([0, 3], [0, 3], color='k', alpha=0.2, linestyle='--', linewidth=4)
        # for i_c in range(nshuffle):
        #     ax.plot(hyst_width_2[i_c], hyst_width_4[i_c],
        #             color=colormap[i_c], marker='o', linestyle='')
        # ax.set_ylabel('Width freq = 4')
        # ax.set_xlabel('Width freq = 2')
        # fig.tight_layout()
        f2.tight_layout()
        


def prepare_data_for_fitting(df_subject, fps=60, tFrame=26):
    """
    Gets the df of a subject, returns two tensors:
        - Input (without parameters): [response^t, t_onset, freq]
        - Output: [t_offset, response^{t+1}]
    """
    map_resps = {1:1, 0:0, 2:-1}
    times_onset = df_subject.keypress_seconds_onset.values/tFrame
    times_offset = df_subject.keypress_seconds_offset.values/tFrame
    freq = df_subject.freq.values
    responses = df_subject.response.values
    responses = np.array([map_resps[r] for r in responses])
    idx_filt = (responses == 0)*(times_onset != 0)
    responses_output = np.roll(responses, 1)
    tensor_input = np.column_stack((responses, times_onset, freq))
    tensor_output = np.column_stack((responses_output, times_offset))
    return tensor_input[~idx_filt], tensor_output[~idx_filt], idx_filt


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


def plot_j1_lmm_slopes(data_folder=DATA_FOLDER, sv_folder=SV_FOLDER, n_simuls_network=50000,
                       use_j0=True):
    label_j0 = '_with_j0' if use_j0 else ''
    intercepts_2, slopes_2, _ =\
        lmm_hysteresis_dominance(freq=2, plot_summary=False,
                                 slope_random_effect=True, plot_individual=False)
    intercepts_4, slopes_4, _ =\
        lmm_hysteresis_dominance(freq=4, plot_summary=False,
                                 slope_random_effect=True, plot_individual=False)
    df = load_data(data_folder, n_participants='all')
    subjects = df.subject.unique()
    pars_all = np.zeros((5, len(subjects)))
    for i_s, subject in enumerate(subjects):
        fitted_params = np.load(sv_folder + '/pars_5_subject_' + subject + str(n_simuls_network) + label_j0 + '.npy')
        pars_all[:, i_s] = fitted_params
    df_all_pars = pd.DataFrame({'J0': pars_all[0], 'J1': pars_all[1],
                                'B1': pars_all[2], '\theta': pars_all[3],
                                'sigma': pars_all[4], 'slopes f2': slopes_2,
                                'slopes f4': slopes_4, 'int f2': intercepts_2,
                                'int f4': intercepts_4})
    g = sns.pairplot(df_all_pars)
    g.map_lower(corrfunc)


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


if __name__ == '__main__':
    print('Running hysteresis_analysis.py')
    fitting_pipeline(n_simuls_network=100000, use_j0=False, contaminants=True,
                      fit=True, plot_lmm=False, plot_pars=True, simulate=True)
    fitting_pipeline(n_simuls_network=100000, use_j0=True, contaminants=True,
                      fit=True, plot_lmm=False, plot_pars=True, simulate=True)
    fitting_pipeline(n_simuls_network=50000, use_j0=False, contaminants=True,
                      fit=True, plot_lmm=False, plot_pars=True, simulate=True)
    fitting_pipeline(n_simuls_network=50000, use_j0=True, contaminants=True,
                      fit=True, plot_lmm=False, plot_pars=True, simulate=True)
    # plot_dist_metrics(n_simuls_network=100000)
    # plot_dist_metrics(n_simuls_network=50000)
    # plot_example(theta=[0.1, 0, 0.5, 0.1, 0.5], data_folder=DATA_FOLDER,
    #              fps=60, tFrame=18, model='MF', prob_flux=False,
    #              freq=4, idx=2)
    # noise_bf_switch_coupling(load_sims=True, coup_vals=np.arange(0.05, 0.35, 1e-2),  # np.array((0.13, 0.17, 0.3))
    #                          nFrame=100000, fps=60, noisyframes=30,
    #                          n=4.0, steps_back=60, steps_front=20,
    #                          ntrials=20, hysteresis_width=False,
    #                          th=0.1)
    # plot_hysteresis_width_simluations(coup_vals=np.arange(0.05, 0.35, 1e-2),
    #                                   b_list=np.linspace(-0.5, 0.5, 501))
    # hysteresis_simulation_threshold(j=1.2, thres_vals=np.arange(0, 0.5, 1e-2),
    #                                 n=4., tau=0.07, sigma=0.1, b1=0.15,
    #                                 tFrame=26, fps=60, nreps=1000,
    #                                 simulate=False)
    # hysteresis_simulation_threshold(j=0.4, thres_vals=np.arange(0, 0.5, 1e-2),
    #                                 n=4., tau=0.07, sigma=0.1, b1=0.15,
    #                                 tFrame=26, fps=60, nreps=1000,
    #                                 simulate=False)
    # plot_noise_simulations_variable(load_sims=True, thres_vals=np.arange(0, 0.5, 1e-2),
    #                     variable='stim_weight', j=1.2, nFrame=100000, fps=60,
    #                     noisyframes=30, n=4., steps_back=90, steps_front=20,
    #                     ntrials=20, zscore_number_switches=False, hysteresis_width=True)
    # plot_noise_simulations_variable(load_sims=True, thres_vals=np.arange(0, 0.5, 1e-2),
    #                     variable='stim_weight', j=0.4, nFrame=100000, fps=60,
    #                     noisyframes=30, n=4., steps_back=90, steps_front=20,
    #                     ntrials=20, zscore_number_switches=False, hysteresis_width=True)
    # hysteresis_basic_plot(coupling_levels=[0, 0.3, 1],
    #                       fps=60, tFrame=26, data_folder=DATA_FOLDER,
    #                       nbins=10, ntraining=8, arrows=False, subjects=['s_36'],
    #                       window_conv=None)
    # plot_max_hyst_ndt_subject(tFrame=26, fps=60, data_folder=DATA_FOLDER,
    #                           ntraining=8, coupling_levels=[0, 0.3, 1],
    #                           window_conv=None, ndt_list=np.arange(-20, 200))
    # plot_hysteresis_average(tFrame=26, fps=60, data_folder=DATA_FOLDER,
    #                         ntraining=8, coupling_levels=[0, 0.3, 1],
    #                         window_conv=None, ndt_list=None)
    # plot_switch_rate(tFrame=26, fps=60, data_folder=DATA_FOLDER,
    #                   ntraining=8, coupling_levels=[0, 0.3, 1],
    #                   window_conv=20, bin_size=0.1)
    # hysteresis_basic_plot_simulation(coup_vals=np.array((0., 0.3, 1))*0.27+0.02,
    #                                  fps=60, nsubs=1, n=4, nsims=1000,
    #                                  b_list=np.linspace(-0.5, 0.5, 501))
    # plot_noise_before_switch(data_folder=DATA_FOLDER, fps=60, tFrame=26,
    #                           steps_back=150, steps_front=10,
    #                           shuffle_vals=[1, 0.7, 0], violin=True, sub=None,
    #                           avoid_first=False, window_conv=1,
    #                           zscore_number_switches=False, 
    #                           normalize_variables=True,
    #                           hysteresis_area=True)
    # hysteresis_basic_plot_all_subjects(coupling_levels=[0, 0.3, 1],
    #                                     fps=60, tFrame=26, data_folder=DATA_FOLDER,
    #                                     ntraining=8, arrows=False)
    # hysteresis_basic_plot(coupling_levels=[0, 0.3, 1],
    #                       fps=60, tFrame=18, data_folder=DATA_FOLDER,
    #                       nbins=9, ntraining=8, arrows=True)*
    # plot_noise_before_switch(data_folder=DATA_FOLDER, fps=60, tFrame=18,
    #                          steps_back=120, steps_front=20,
    #                          shuffle_vals=[1, 0.7, 0])
    # save_5_params_recovery(n_pars=100, sv_folder=SV_FOLDER, i_ini=0)
    # for sims in [1000000]:
    #     parameter_recovery_5_params(n_simuls_network=sims, fps=60, tFrame=26,
    #                                 n_pars_to_fit=100, n_sims_per_par=100,
    #                                 sv_folder=SV_FOLDER, simulate=True,
    #                                 load_net=False, not_plot_and_return=False)
    #     plt.close('all')
    # plt.close('all')
    # plot_example_pswitch(params=[0.7, 1e-2, 0., 0.2, 0.5], data_folder=DATA_FOLDER,
    #                       fps=60, tFrame=26, freq=2, idx=1, n=3.92, theta=0.5,
    #                       tol=1e-3, pshuffle=0)
    # fitting_transitions(data_folder=DATA_FOLDER, fps=60, tFrame=18)
    # fitting_fokker_planck(data_folder=DATA_FOLDER, model='MF')
    # lmm_hysteresis_dominance(freq=2, plot_summary=True,
    #                          slope_random_effect=False, plot_individual=False)
    # lmm_hysteresis_dominance(freq=2, plot_summary=True,
    #                          slope_random_effect=True, plot_individual=True)
