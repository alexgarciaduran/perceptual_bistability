# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 20:07:39 2025

@author: alexg
"""

import pyddm
import pyddm.plot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pylab as pl
from matplotlib.lines import Line2D
import glob
from sklearn.metrics import roc_curve, auc
from sklearn import manifold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from gibbs_necker import rle
from mean_field_necker import colored_line
from fitting_pipeline import load_data as load_data_experiment_1
import matplotlib.patches as mpatches
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
from scipy.stats import pearsonr, zscore
from scipy.signal import sawtooth
from scipy.optimize import curve_fit
import itertools
from pyddm import set_N_cpus
from pyddm.models.loss import LossLikelihood, LossBIC
from pyddm.functions import get_model_loss
from mpl_toolkits.mplot3d import Axes3D


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
    DATA_FOLDER = 'C:/Users/agarcia/Desktop/phd/necker/data_folder/hysteresis/data/'  # Alex CRM
    SV_FOLDER = 'C:/Users/agarcia/Desktop/phd/necker/data_folder/hysteresis/'  # Alex CRM

COLORMAP = LinearSegmentedColormap.from_list('rg', ['darkgreen', 'gainsboro', 'r'], N=128)


def preprocess_keypress_data(df, no_press_threshold=0.3):
    """
    Preprocess behavioral data so that 0 (no press) rows
    shorter than a given threshold are replaced by the next response
    (1=left, 2=right), merging the two rows.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
        ['trial_index', 'conditions', 'response', 
         'keypress_seconds_onset', 'keypress_seconds_offset', 'subject']
    no_press_threshold : float
        Maximum duration (in seconds) for which a no-press period is considered
        a 'switch' and merged with the next response. Default = 0.3s (300 ms).
        
    Returns
    -------
    pandas.DataFrame
        Preprocessed DataFrame with merged short 0-responses.
    """
    df = df.copy()
    df['_row_order'] = range(len(df))  # store original order index
    processed_dfs = []

    # Group by subject and trial, preserving input order
    for (subj, trial), sub_df in df.groupby(['subject', 'trial_index'], sort=False):
        sub_df = sub_df.sort_values('keypress_seconds_onset').reset_index(drop=True)
        keep_rows = []
        i = 0

        while i < len(sub_df):
            row = sub_df.iloc[i].copy()

            # Handle short 'no press' rows followed by another response
            if row['response'] == 0 and i + 1 < len(sub_df):
                duration = row['keypress_seconds_offset'] - row['keypress_seconds_onset']
                next_row = sub_df.iloc[i + 1]

                if duration < no_press_threshold:
                    # Merge: take next response, extend offset
                    row['response'] = next_row['response']
                    row['keypress_seconds_offset'] = next_row['keypress_seconds_offset']
                    keep_rows.append(row)
                    i += 2
                    continue  # skip the next row

            keep_rows.append(row)
            i += 1

        processed_dfs.append(pd.DataFrame(keep_rows))

    # Concatenate and restore original ordering
    processed_df = pd.concat(processed_dfs, ignore_index=True)
    processed_df = processed_df.sort_values('_row_order').drop(columns='_row_order')
    processed_df.reset_index(drop=True, inplace=True)

    return processed_df


def preprocess_time_series(df, dt=1/60, no_press_threshold=0.3):
    """
    Preprocess time-evolving keypress data per subject and trial.
    Short 'no press' (response=0) segments shorter than threshold
    are replaced by the next response (1=left, 2=right).
    The output preserves the original ordering of rows.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
        ['subject', 'trial_index', 'response', 'stimulus', 'conditions']
    dt : float
        Sampling interval between consecutive timepoints (in seconds)
    no_press_threshold : float
        Maximum duration (in seconds) for which a 0-segment is merged (default=0.3s)

    Returns
    -------
    pandas.DataFrame
        DataFrame with modified 'response' column, same row order as input
    """

    df = df.copy()
    df['_row_order'] = np.arange(len(df))  # store original order

    n_thresh = int(np.round(no_press_threshold / dt))  # samples threshold
    processed_dfs = []

    # groupby preserves order if we set sort=False
    for (subj, trial), sub_df in df.groupby(['subject', 'trial_index'], sort=False):
        r = sub_df["responses"].to_numpy().copy()

        # find contiguous segments of constant response
        change = np.r_[True, np.diff(r) != 0, True]
        starts = np.where(change[:-1])[0]
        ends = np.where(change[1:])[0]
        segment_values = [r[s] for s in starts]
        lengths = ends - starts

        # replace short no-press segments with next response
        for i, (s, e, val, L) in enumerate(zip(starts, ends, segment_values, lengths)):
            if val == 0 and L <= n_thresh and i + 1 < len(segment_values):
                next_val = segment_values[i + 1]
                r[s:e+1] = next_val

        sub_df["responses"] = r
        processed_dfs.append(sub_df)

    # concatenate and restore original order
    processed_df = pd.concat(processed_dfs, ignore_index=True)
    processed_df = processed_df.sort_values('_row_order').drop(columns='_row_order')
    processed_df.reset_index(drop=True, inplace=True)
    return processed_df


def load_data(data_folder, n_participants='all', filter_subjects=True,
              preprocess_data=True):
    files = glob.glob(data_folder + '*.csv')
    if 'noisy' in data_folder:
        preprocess = preprocess_time_series
    else:
        preprocess = preprocess_keypress_data
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
            if filter_subjects and \
                not accept_subject(DATA_FOLDER, i,
                                   threshold_switches=3, n_training=8):
                continue
            df_0 = pd.concat((df_0, df))
        if preprocess_data:
            print('Preprocessing')
            df = preprocess(df_0, no_press_threshold=0.3)
            print(df.subject.unique())
            return df
        else:
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


def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=9,
                              maxasterix=3, ax=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    text = stars_pval(data)
    # print(data)

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = ax.get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax.text(*mid, text, **kwargs)


def plot_responses_panels(responses_2, responses_4, barray_2, barray_4, coupling_levels,
                          tFrame=26, fps=60, window_conv=None,
                          ndt_list=np.arange(100), unfold_time=False):
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
    ax2.set_ylim(np.min(np.mean(hyst_width_2, axis=1))-0.25, np.max(np.mean(hyst_width_2, axis=1))+0.54)
    hyst_width_4 = np.zeros((len(coupling_levels), len(responses_2[0])))
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
    ax4.set_ylim(np.min(np.mean(hyst_width_2, axis=1))-0.25, np.max(np.mean(hyst_width_2, axis=1))+0.54)
    heights = np.nanmean(hyst_width_2.T, axis=0)
    bars = np.arange(3)
    pv_sh012 = scipy.stats.ttest_ind(hyst_width_2[0], hyst_width_2[1]).pvalue
    pv_sh022 = scipy.stats.ttest_ind(hyst_width_2[0], hyst_width_2[2]).pvalue
    pv_sh122 = scipy.stats.ttest_ind(hyst_width_2[1], hyst_width_2[2]).pvalue
    # pv_sh014 = scipy.stats.ttest_ind(hyst_width_4[0], hyst_width_4[1]).pvalue
    # pv_sh024 = scipy.stats.ttest_ind(hyst_width_4[0], hyst_width_4[2]).pvalue
    # pv_sh124 = scipy.stats.ttest_ind(hyst_width_4[1], hyst_width_4[2]).pvalue
    barplot_annotate_brackets(0, 1, pv_sh012, bars, heights, yerr=None, dh=.16, barh=.05, fs=10,
                              maxasterix=3, ax=ax2)
    barplot_annotate_brackets(0, 2, pv_sh022, bars, heights, yerr=None, dh=.39, barh=.05, fs=10,
                              maxasterix=3, ax=ax2)
    barplot_annotate_brackets(1, 2, pv_sh122, bars, heights, yerr=None, dh=.2, barh=.05, fs=10,
                              maxasterix=3, ax=ax2)
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
    ax3.set_ylabel('Hysteresis freq = 4')
    ax3.set_xlabel('Hysteresis freq = 2')
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
                     window_conv=10, bin_size=0.1, switch_01=True,
                     only_ascending=False):
    nFrame = tFrame*fps
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()

    responses_2, responses_4, barray_2, barray_4 = collect_responses(
        df, subjects, coupling_levels, fps=fps, tFrame=tFrame)
    
    timebins = np.arange(0, tFrame+bin_size, bin_size)
    xvals = timebins[:-1] + bin_size/2
    # frame → time mapping
    # frame_times = np.linspace(0, tFrame, nFrame, endpoint=False)
    # # interpolate array values at bin times
    # array_at_bins_2 = np.interp(timebins, frame_times, barray_2)
    # array_at_bins_4 = np.interp(timebins, frame_times, barray_4)

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
    axes[0].legend(frameon=True, title='p(shuffle)'); axes[0].set_ylabel(r'Switch rate L$\rightarrow$R, (Hz)')
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


def join_trial_responses(subj, only_ascending=False):
    """
    subj: dict {"asc": array, "desc": array, "ini_side": list}
    Returns joined responses (n_trials, timepoints).
    """
    asc, desc, ini_sides = subj["asc"], subj["desc"], subj["ini_side"]
    joined = []
    for i in range(len(ini_sides)):
        if ini_sides[i] == -1:   # ascending first
            if not only_ascending:
                continue  # keep only one!
            trial = np.concatenate([asc[i], desc[i]])
        else:                   # descending first
            if only_ascending:
                continue
            trial = np.concatenate([desc[i], asc[i]])
        joined.append(trial)
    return np.array(joined)


def average_switch_rates_dir(responses, fps=60, bin_size=1.0, join=True,
                             only_ascending=False):
    """
    Compute average 0→1 and 1→0 switch rates across subjects.
    If join=True, concatenates asc+desc before counting.
    """
    per_sub_rates_01, per_sub_rates_10 = [], []
    bins_ref = None

    for subj in responses:
        arr = join_trial_responses(subj, only_ascending=False) if join else subj["asc"]
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
                            window_conv=None, ndt_list=np.arange(-50, 50),
                            unfold_time=False):
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()

    responses_2, responses_4, barray_2, barray_4 = collect_responses(
        df, subjects, coupling_levels, fps=fps, tFrame=tFrame)
    
    plot_responses_panels(responses_2, responses_4, barray_2, barray_4, coupling_levels,
                          tFrame=tFrame, fps=fps, window_conv=window_conv,
                          ndt_list=ndt_list, unfold_time=unfold_time)


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
    ax2[-1].set_xlabel('Time before switch (s)')
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
    indep_noise = np.sqrt(dt/tau)*np.random.randn(nFrame, nsims, nsubs)*0.07
    choice = np.zeros((len(coup_vals), nFrame, nsims, 2, nsubs))
    for i_j, j in enumerate(coup_vals):
        for freq in range(2):
            stimulus = [b_list_2, b_list_4][freq]
            for sub in range(nsubs):
                for sim in range(nsims):
                    x = 0.02  # assume we start close to q ~ 0 (always L --> R)
                    vec = [x]
                    for t in range(1, nFrame):
                        x = x + dt*(sigmoid(2*j*n*(2*x-1)+2*stimulus[t]*0.3)-x)/tau + indep_noise[t, sim, sub]
                        vec.append(x)
                        if x < 0.4:
                            ch = 0.
                        if x > 0.6:
                            ch = 1.
                        if 0.4 <= x <= 0.6 and t > 0:
                            ch = choice[i_j, t-1, sim, freq, sub]
                        choice[i_j, t, sim, freq, sub] = ch
    np.save(DATA_FOLDER + 'choice_hysteresis_large_tau.npy', choice)
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
    fig.savefig(SV_FOLDER + 'hysteresis_basic_plot_simulation_v2.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'hysteresis_basic_plot_simulation_v2.svg', dpi=400, bbox_inches='tight')


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
    f0.savefig(SV_FOLDER + 'hysteresis_basic_plot_simulation_v3.svg', dpi=400, bbox_inches='tight')
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


def stars_pval(pval):
    s = 'n.s.'
    if pval < 0.05 and pval >= 0.01:
        s = '*'
    if pval < 0.01 and pval >= 0.001:
        s = '**'
    if pval < 0.001:
        s = '***'
    return s


def decision_kernel(t, A, tau, sigma, baseline):
    return A * np.exp(-((t + tau)**2) / (2 * sigma**2)) + baseline


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
    fig.savefig(SV_FOLDER + label + 'noise_trials_dominance.svg', dpi=400, bbox_inches='tight')
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
    f3.savefig(SV_FOLDER + label + 'average_noise_trials_dominance.svg', dpi=400, bbox_inches='tight')


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
    latency_avg = []
    # all_noises = [[], [], []]
    fignew, axnew = plt.subplots(nrows=2, figsize=(5.5, 7.5))
    # figgaussian, axgaussian = plt.subplots(nrows=1, figsize=(6, 4.5))
    axnew[0].set_xlabel('Time before switch (s)'); axnew[0].set_ylabel('Noise')
    # axgaussian.set_xlabel('Time before switch (s)'); axgaussian.set_ylabel('Noise')
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
            mean_vals_noise_switch_all_shuffles_subject = np.row_stack((mean_vals_noise_switch_all_shuffles_subject, mean_vals_noise_switch_all_trials))
            # it's better to compute afterwards, with the average peak per coupling
            # because trial by trial there is a lot of noise and that breaks the mean/latency
            # it gets averaged out
            mean_number_switchs_coupling[i_sh, i_sub] = tFrame/ np.nanmean(np.array(number_switches))
            # axis=0 means average across switches (leaves time coords)
            # axis=1 means average across time (leaves switches coords)
            averaged_and_convolved_values = np.convolve(np.nanmean(mean_vals_noise_switch_all_trials, axis=0),
                                                                          np.ones(window_conv)/window_conv, 'same')
            mean_peak_latency[i_sh, i_sub] = (np.argmax(averaged_and_convolved_values) - steps_back)/fps
            mean_peak_amplitude[i_sh, i_sub] = np.nanmax(averaged_and_convolved_values)
            mean_vals_noise_switch_coupling[i_sh, :, i_sub] = averaged_and_convolved_values
            err_vals_noise_switch_coupling[i_sh, :, i_sub] = np.nanstd(mean_vals_noise_switch_all_trials, axis=0) / np.sqrt(mean_vals_noise_switch_all_trials.shape[0])
        mean_vals_noise_switch_all_shuffles_subject = mean_vals_noise_switch_all_shuffles_subject[1:]
        x_plot = np.arange(-steps_back, 0, 1)/fps
        latency = (np.argmax(np.nanmean(mean_vals_noise_switch_all_trials[:, :-steps_front], axis=0))-steps_back)/fps
        peakval = np.nanmax(np.nanmean(mean_vals_noise_switch_all_trials[:, :-steps_front], axis=0))
        kernel = np.nanmean(mean_vals_noise_switch_all_trials[:, :-steps_front], axis=0)
        # p0 = [1.0, 0.3, 0.2, 0.0]  # initial guess: A, tau(s), sigma(s), baseline
        # pars, cov = curve_fit(decision_kernel, x_plot, kernel, p0=p0)
        # A, tau, sigma, baseline = pars
        # if i_sub == 6:
        #     print(pars)
        #     ker_fitted = decision_kernel(x_plot, A, tau, sigma, baseline)
        #     axgaussian.plot(x_plot, ker_fitted, color='r', linewidth=2, alpha=0.5, zorder=1)
        #     axgaussian.plot(x_plot, kernel, color='k', linewidth=2, alpha=0.5, zorder=1)
        latency_avg.append(latency)
        all_kernels.append(kernel)
        axnew[0].plot(x_plot, kernel, color='k', linewidth=2, alpha=0.5, zorder=1)
        axnew[0].plot(latency, peakval, marker='*', color='firebrick', markersize=8, zorder=5)
        if len(subs) > 1 and zscore_number_switches:
            # mean_number_switchs_coupling[:, i_sub] = zscor(mean_number_switchs_coupling[:, i_sub])
            # mean_peak_latency[:, i_sub] = zscor(mean_peak_latency[:, i_sub])
            # mean_peak_amplitude[:, i_sub] = zscor(mean_peak_amplitude[:, i_sub])
            label = 'z-scored '
        else:
            label = ''
        #     mean_vals_noise_switch_coupling[:, :, i_sub] = zscor(mean_vals_noise_switch_coupling[:, :, i_sub], nan_policy='omit')
    sns.histplot(x=latency_avg, ax=axnew[1], color='firebrick', bins=15)
    sns.kdeplot(x=latency_avg, ax=axnew[1], color='firebrick', bw_adjust=0.5, linewidth=3)
    axnew[1].axvline(np.mean(latency_avg), color='k', linewidth=2, label='Mean')
    axnew[1].axvline(np.median(latency_avg), linestyle='--', color='k', linewidth=2, label='Median')
    axnew[1].legend(frameon=False)
    axnew[1].set_xlabel('Latency (s)'); axnew[1].set_xlim(axnew[0].get_xlim())
    np.save(DATA_FOLDER + 'kernel_latency_average.npy', np.array(latency_avg))
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5.5, 4))
    fig2, ax2 = plt.subplots(1, figsize=(5.5, 4))
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
        # switch_time_diff_2 = np.load(DATA_FOLDER + 'switch_time_diff_freq_2.npy')
        # switch_time_diff_4 = np.load(DATA_FOLDER + 'switch_time_diff_freq_4.npy')
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
    fignew.savefig(SV_FOLDER + 'latency_computation.png', dpi=100, bbox_inches='tight')
    fignew.savefig(SV_FOLDER + 'latency_computation.svg', dpi=100, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'noise_before_switch_experiment.png', dpi=100, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'noise_before_switch_experiment.svg', dpi=100, bbox_inches='tight')
    figlast, ax = plt.subplots(ncols=1, figsize=(5, 4))
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
    # for i_sub in range(len(subs)):
    #     y_plot = all_kernels[i_sub]
    #     ax.plot(x_plot, y_plot, color='k', linewidth=2, alpha=0.5)
    x_plot = np.arange(-steps_back, 0, 1)/fps
    y_plot = np.nanmean(all_kernels, axis=0)
    ax.plot(x_plot, y_plot, color='k', linewidth=4, )
    err = np.nanstd(all_kernels, axis=0)/np.sqrt(len(subs))
    ax.fill_between(x_plot, y_plot-err, y_plot+err, color='k', alpha=0.2)
    ax.set_xlabel('Time before switch(s)')
    ax.set_ylabel('Noise')
    figlast.tight_layout()
    return fig, fig2, fig3, g, fignew


def experiment_example(nFrame=1560, fps=60, noisyframes=15):
    colormap = COLORMAP
    # coolwarm_r
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
    ax[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    # ax[0].plot(time, stimulus, linewidth=4, label='2', color='navy')
    line = colored_line(time, stimulus, stimulus, ax[0],
                        linewidth=4, cmap=colormap, 
                        norm=plt.Normalize(vmin=-2, vmax=2), label='2')
    ax[0].set_xlim([np.min(time)-1e-1, np.max(time)+1e-1])
    ax[0].set_yticks([-2, 0, 2], ['-1', '0', '1'])
    difficulty_time_ref_4 = np.linspace(-2, 2, nFrame//4)
    stimulus = np.concatenate(([difficulty_time_ref_4, -difficulty_time_ref_4,
                                difficulty_time_ref_4, -difficulty_time_ref_4]))
    # ax[0].plot(time, stimulus, linewidth=4, label='4', color='navy', linestyle='--')
    line = colored_line(time, stimulus, stimulus, ax[0],
                        linewidth=4, cmap=colormap, linestyle='--', 
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
                        linewidth=4, cmap=colormap,
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


def simulator_5_params_sticky_bound(params, freq, nFrame=1560, fps=60,
                                    return_choice=False):
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
    # x1 = (np.sign(b_eff[0])+1)/2
    # for i in range(5):
    #     x1 = sigmoid(2 * (j_eff * (2 * x1 - 1)))  # convergence
    x[0] = 0.5
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


def simulator_5_params(params, freq, nFrame=1560, fps=60,
                       return_choice=False):
    """
    Simulator. Takes set of `params` and simulates the system, returning summary statistics.
    Params: J_eff, B_eff, tau, threshold distance, noise
    """
    t = np.arange(0, nFrame, 1)/fps
    stimulus = sawtooth(2 * np.pi * abs(freq)/2 * t/26, 0.5)*2*np.sign(freq)
    j_eff, b_par, th, sigma, ndt = params  # add ndt
    lower_bound, upper_bound = np.array([-1, 1])*th + 0.5
    tau = 1
    dt = 1/fps
    b_eff = stimulus*b_par
    noise = np.random.randn(nFrame)*sigma*np.sqrt(dt/tau)
    x = np.zeros(nFrame)
    x1 = (np.sign(b_eff[0])+1)/2
    x1 = sigmoid(2 * (j_eff * (2 * x1 - 1)))  # convergence
    x[0] = x1

    choice = np.zeros(nFrame)
    prev_choice = 0.0
    pending_choice = None      # choice scheduled but not yet applied
    pending_time = None        # when to apply it
    ndt_frames = int(ndt / dt)
    
    for t in range(1, nFrame):
        # apply pending choice if NDT has elapsed
        if pending_choice is not None and t >= pending_time:
            prev_choice = pending_choice
            pending_choice = None

        # evolve freely (no stickiness)
        drive = sigmoid(2 * (j_eff * (2 * x[t - 1] - 1) + b_eff[t]))
        x[t] = x[t - 1] + dt * (drive - x[t - 1]) / tau + noise[t]
    
        # bound crossing
        if x[t] >= upper_bound:
            new_choice = 1.0
        elif x[t] <= lower_bound:
            new_choice = -1.0
        else:
            new_choice = prev_choice
    
        # schedule motor choice change (after NDT delay)
        if new_choice != prev_choice and pending_choice is None:
            pending_choice = new_choice
            pending_time = t + ndt_frames

        # delayed decision
        choice[t] = prev_choice
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
        npars = 6
    nFrame = fps*tFrame
    orig_params = np.zeros((n_pars_to_fit, npars))
    recovered_params = np.zeros((n_pars_to_fit, npars))
    for par in tqdm.tqdm(range(ini_par, n_pars_to_fit)):
        # simulate
        if pyddmfit:
            theta = np.load(sv_folder + 'param_recovery/pars_pyddm_prt_ndt' + str(par) + '.npy')
            pars = np.load(sv_folder + f'param_recovery/recovered_params_pyddm_{par}_ndt.npy')
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
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(8, 5))
        ax = ax.flatten()
        # labels = ['Jeff', ' B1',  'Tau', 'Thres.', 'sigma']
        if pyddmfit:
            labels = ['J1', 'J0', ' B1', 'sigma', 'Thres.', 'NDT']
        else:
            labels = ['Jeff', ' B1', 'Thres.', 'sigma']
        # xylims = [[0, 3], [0, 0.8], [0, 0.7], [0, 0.5], [0, 0.5]]
        for i_a in range(npars):
            a = ax[i_a]
            a.plot(orig_params[:, i_a], recovered_params[:, i_a], color='k', marker='o',
                   markersize=5, linestyle='')
            maxval = np.nanmax([orig_params[:, i_a], recovered_params[:, i_a]])
            minval = np.nanmin([orig_params[:, i_a], recovered_params[:, i_a]])
            a.set_xlim(minval-1e-2, maxval+1e-2)
            a.set_ylim(minval-1e-2, maxval+1e-2)
            a.plot([minval, maxval], [minval, maxval],
                   color='k', linestyle='--',
                   alpha=0.3, linewidth=4)
            # a.plot(xylims[i_a], xylims[i_a], color='k', alpha=0.3)
            a.set_title(labels[i_a], fontsize=12)
            a.set_xlabel('Original', fontsize=12)
            a.set_ylabel('Recovered', fontsize=12)
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
        # for a in [ax[-1]]:
        #     b_over_sigma_orig = orig_params[:, 2]/orig_params[:, 3]
        #     b_over_sigma_rec = recovered_params[:, 2]/recovered_params[:, 3]
        #     a.plot(b_over_sigma_orig, b_over_sigma_rec, color='k', marker='o',
        #            markersize=5, linestyle='')
        #     notnanidx = np.isnan(b_over_sigma_orig+b_over_sigma_rec)
        #     corr = np.corrcoef(b_over_sigma_orig[~notnanidx], b_over_sigma_rec[~notnanidx])[0][1]
        #     maxval = np.nanmax([b_over_sigma_orig, b_over_sigma_rec])
        #     minval = np.nanmin([b_over_sigma_orig, b_over_sigma_rec])
        #     a.set_xlim(minval-1e-2, maxval+1e-2)
        #     a.set_ylim(minval-1e-2, maxval+1e-2)
        #     a.plot([minval, maxval], [minval, maxval],
        #            color='k', linestyle='--',
        #            alpha=0.3, linewidth=4)
        #     # a.plot(xylims[i_a], xylims[i_a], color='k', alpha=0.3)
        #     a.set_title(f'B1 / sigma, r={round(corr, 3)}', fontsize=12)
        #     a.set_xlabel('Original', fontsize=12)
        #     a.set_ylabel('Recovered', fontsize=12)
        #     a.spines['right'].set_visible(False)
        #     a.spines['top'].set_visible(False)
        # ax[-1].axis('off')
        fig.tight_layout()
        fig2, ax2 = plt.subplots(ncols=2)
        ax2, ax = ax2
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
        plt.colorbar(im, ax=ax, label='Correlation')
        labels_reduced = labels
        ax.set_xticks(np.arange(npars), labels, fontsize=12, rotation=45)  # , rotation='270'
        ax.set_yticks(np.arange(npars), labels_reduced, fontsize=12)
        ax.set_xlabel('Original parameters', fontsize=14)
        # compute correlation matrix
        mat_corr = np.corrcoef(recovered_params.T, rowvar=True)
        mat_corr *= np.tri(*mat_corr.shape, k=-1)
        mat_corr[mat_corr == 0] = np.nan
        # plot correlation matrix
        im = ax2.imshow(mat_corr, cmap=cmap, vmin=vmin, vmax=1)
        ax2.step(np.arange(0, npars)-0.5, np.arange(0, npars)-0.5, color='k',
                 linewidth=.7)
        ax2.set_xticks(np.arange(npars), labels, fontsize=12, rotation=45)  # , rotation='270'
        ax2.set_yticks(np.arange(npars), labels, fontsize=12)
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
    ax2.set_ylim(4, 7.5);  ax2.set_xticks([0, 1, 2], pshuffles)
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
                       plot=False, simulate=False, use_j0=False,
                       fitted_params_all=None, subjects=['s_1'], ntrials=72, window_conv=1):
    label_j0 = '_with_j0' if use_j0 else ''
    nFrame = fps*tFrame
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    if subjects is None:
        subjects = df.subject.unique()[:len(fitted_params_all)]
    choices_all_subject = np.zeros((len(subjects), ntrials, nFrame))
    for i_s, subject in enumerate(subjects):
        fitted_params = fitted_params_all[i_s]
        print('Simulating subject', subject)
        if fitted_params is None:
            fitted_params = np.load(sv_folder + '/pars_5_subject_' + subject + str(n_simuls_network) + label_j0 + '.npy')
        df_subject = df.loc[df.subject == subject]
        pshuffles = df_subject.groupby('trial_index')['pShuffle'].mean().values
        ini_side = df_subject.groupby('trial_index')['initial_side'].mean().values
        frequencies = df_subject.groupby('trial_index')['freq'].mean().values*ini_side
        if simulate:
            choice_all = np.zeros((ntrials, nFrame))
            for trial in range(ntrials):
                j_eff = (1-pshuffles[trial])*fitted_params[0] + fitted_params[1]*use_j0
                params = fitted_params[1:].copy()
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
        f2, ax2 = plt.subplots(ncols=2, figsize=(7.5, 4))
        colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
        # containers: [shuffle, subject, timepoints]
        ascending_subjects_2 = np.full((nshuffle, len(subjects), nFrame//2), np.nan)
        descending_subjects_2 = np.full((nshuffle, len(subjects), nFrame//2), np.nan)
        ascending_subjects_4 = np.full((nshuffle, len(subjects), nFrame//4), np.nan)
        descending_subjects_4 = np.full((nshuffle, len(subjects), nFrame//4), np.nan)
        # align choices into ascending/descending phases
        for i_s, subject in enumerate(subjects):
            df_subject = df.loc[df.subject == subject]
            pshuffles = np.round(df_subject.groupby('trial_index')['pShuffle'].mean().values, 1)
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
        
                response_raw = choice_all[i_trial, :].copy()
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
        # compute hysteresis width for f=2, f=4
        stimulus = get_blist(freq=2, nFrame=nFrame)
        dx = np.diff(stimulus)[0]
        hyst_width_2 = np.nansum(np.abs(descending_subjects_2[:, :, ::-1] - ascending_subjects_2), axis=-1)*dx
        stimulus = get_blist(freq=4, nFrame=nFrame)
        dx = np.diff(stimulus)[0]
        hyst_width_4 = np.nansum(np.abs(descending_subjects_4[:, :, ::-1] - ascending_subjects_4), axis=-1)*dx
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
                asc_vals = np.convolve(asc_vals, np.ones(window_conv)/window_conv, mode='same')[window_conv//2:-window_conv//2]
                desc_vals = np.convolve(desc_vals, np.ones(window_conv)/window_conv, mode='same')[window_conv//2:-window_conv//2]
                x_asc = stimulus[asc_mask][window_conv//2:-window_conv//2]
                x_desc = stimulus[~asc_mask][window_conv//2:-window_conv//2]
        
                # plot
                ax2[freq_idx].plot(x_asc, asc_vals,
                                   color=colormap[i_sh], linewidth=4, label=unique_shuffle[i_sh])
                ax2[freq_idx].plot(x_desc, desc_vals,
                                   color=colormap[i_sh], linewidth=4)
        for a in ax2:
            a.set_xlabel('Sensory evidence B(t)')
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            a.axhline(0.5, color='k', linestyle='--', alpha=0.2)
            a.axvline(0., color='k', linestyle='--', alpha=0.2)
            a.set_ylim(-0.025, 1.085)
            a.set_yticks([0, 0.5, 1])
            a.set_xlim(-2.05, 2.05)
            a.set_xticks([-2, 0, 2], [-1, 0, 1])
        ax2[0].set_title('Freq = 2', fontsize=14)
        ax2[0].legend(title='p(shuffle)', frameon=False,
                      bbox_to_anchor=[-0.02, 1.07], loc='upper left')
        ax2[1].set_title('Freq = 4', fontsize=14)
        ax2[0].set_ylabel('P(choice = R)')
        left, bottom, width, height = [0.4, 0.27, 0.12, 0.2]
        ax_01 = f2.add_axes([left, bottom, width, height])
        left, bottom, width, height = [0.9, 0.27, 0.12, 0.2]
        ax_11 = f2.add_axes([left, bottom, width, height])
        sns.barplot(hyst_width_2.T, palette=colormap, ax=ax_01, errorbar='se')
        ax_01.set_ylim(np.min(np.mean(hyst_width_2, axis=1))-0.25, np.max(np.mean(hyst_width_2, axis=1))+0.5)
        sns.barplot(hyst_width_4.T, palette=colormap, ax=ax_11, errorbar='se')
        ax_11.set_ylim(np.min(np.mean(hyst_width_2, axis=1))-0.25, np.max(np.mean(hyst_width_2, axis=1))+0.5)
        heights = np.nanmean(hyst_width_2.T, axis=0)
        bars = np.arange(3)
        pv_sh012 = scipy.stats.ttest_ind(hyst_width_2[0], hyst_width_2[1]).pvalue
        pv_sh022 = scipy.stats.ttest_ind(hyst_width_2[0], hyst_width_2[2]).pvalue
        pv_sh122 = scipy.stats.ttest_ind(hyst_width_2[1], hyst_width_2[2]).pvalue
        barplot_annotate_brackets(0, 1, pv_sh012, bars, heights, yerr=None, dh=.16, barh=.05, fs=10,
                                  maxasterix=3, ax=ax_01)
        barplot_annotate_brackets(0, 2, pv_sh022, bars, heights, yerr=None, dh=.39, barh=.05, fs=10,
                                  maxasterix=3, ax=ax_01)
        barplot_annotate_brackets(1, 2, pv_sh122, bars, heights, yerr=None, dh=.2, barh=.05, fs=10,
                                  maxasterix=3, ax=ax_01)
        # heights = np.nanmean(hyst_width_4.T, axis=0)
        # bars = np.arange(3)
        # pv_sh012 = scipy.stats.ttest_ind(hyst_width_4[0], hyst_width_4[1]).pvalue
        # pv_sh022 = scipy.stats.ttest_ind(hyst_width_4[0], hyst_width_4[2]).pvalue
        # pv_sh122 = scipy.stats.ttest_ind(hyst_width_4[1], hyst_width_4[2]).pvalue
        # barplot_annotate_brackets(0, 1, pv_sh012, bars, heights, yerr=None, dh=.16, barh=.05, fs=10,
        #                           maxasterix=3, ax=ax_11)
        # barplot_annotate_brackets(0, 2, pv_sh022, bars, heights, yerr=None, dh=.39, barh=.05, fs=10,
        #                           maxasterix=3, ax=ax_11)
        # barplot_annotate_brackets(1, 2, pv_sh122, bars, heights, yerr=None, dh=.2, barh=.05, fs=10,
        #                           maxasterix=3, ax=ax_11)
        for a in [ax_01, ax_11]:
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            a.set_xlabel('p(shuffle)', fontsize=11); a.set_xticks([])
            a.set_ylabel('Hysteresis', fontsize=11); a.set_yticks([])

        f2.tight_layout()
        f2.savefig(SV_FOLDER + 'simulated_hysteresis_average.png', dpi=400, bbox_inches='tight')
        f2.savefig(SV_FOLDER + 'simulated_hysteresis_average.svg', dpi=400, bbox_inches='tight')
        hyst_width_2_data = np.load(DATA_FOLDER + 'hysteresis_width_freq_2.npy')
        hyst_width_4_data = np.load(DATA_FOLDER + 'hysteresis_width_freq_4.npy')
        fitted_subs = len(subjects)
        fig, ax = plt.subplots(ncols=2, figsize=(8., 4))
        corr_2 = pearsonr(hyst_width_2_data[:, :fitted_subs].flatten(), hyst_width_2.flatten())
        corr_4 = pearsonr(hyst_width_4_data[:, :fitted_subs].flatten(), hyst_width_4.flatten())
        minmax = [np.min([hyst_width_2_data[:, :fitted_subs].flatten(), hyst_width_2.flatten()])-0.5,
                   np.max([hyst_width_2_data[:, :fitted_subs].flatten(), hyst_width_2.flatten()])+0.5]
        ax[0].set_ylim(minmax)
        ax[0].set_xlim(minmax)
        ax[0].plot(minmax, minmax, color='k', linestyle='--', alpha=0.4, linewidth=4)
        minmax = [np.min([hyst_width_4_data[:, :fitted_subs].flatten(), hyst_width_4.flatten()])-0.5,
                   np.max([hyst_width_4_data[:, :fitted_subs].flatten(), hyst_width_4.flatten()])+0.5]
        ax[1].set_ylim(minmax)
        ax[1].set_xlim(minmax)
        ax[1].plot(minmax, minmax, color='k', linestyle='--', alpha=0.4, linewidth=4)
        for i_a, a in enumerate(ax):
            a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
            a.set_ylim(0, 3.5); a.set_xlim(0, 3.5)
            a.set_yticks([0, 1, 2, 3]); a.set_xticks([0, 1, 2, 3])
        ax[0].text(0.2, 2.8, f'r={round(corr_2.statistic, 2)} \np={corr_2.pvalue: .1e}')
        ax[1].text(0.2, 2.8, f'r={round(corr_4.statistic, 2)} \np={corr_4.pvalue: .1e}')
        ax[0].set_xlabel('Hysteresis data')
        ax[1].set_xlabel('Hysteresis data')
        ax[0].set_ylabel('Hysteresis simulations')
        ax[0].set_title('Freq. = 2', fontsize=12); ax[1].set_title('Freq. = 4', fontsize=12)
        ax[0].plot(hyst_width_2_data[:, :fitted_subs].flatten(),
                   hyst_width_2.flatten(), marker='o', color='k', linestyle='')
        ax[1].plot(hyst_width_4_data[:, :fitted_subs].flatten(), hyst_width_4.flatten(), marker='o', color='k', linestyle='')
        fig.tight_layout()
        # plot deltas hysteresis
        delta_2_data = hyst_width_2_data[0, :fitted_subs]-hyst_width_2_data[-1, :fitted_subs]
        delta_4_data = hyst_width_4_data[0, :fitted_subs]-hyst_width_4_data[-1, :fitted_subs]
        delta_2_sims = hyst_width_2[0]-hyst_width_2[-1]
        delta_4_sims = hyst_width_4[0]-hyst_width_4[-1]
        fig, ax = plt.subplots(ncols=2, figsize=(8., 4))
        for i_a, a in enumerate(ax):
            a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
        corr_2 = pearsonr(delta_2_data, delta_2_sims)
        corr_4 = pearsonr(delta_4_data, delta_4_sims)
        minmax = [np.min([delta_2_data, delta_2_sims])-0.5, np.max([delta_2_data, delta_2_sims])+0.5]
        ax[0].set_ylim(minmax)
        ax[0].set_xlim(minmax)
        ax[0].plot(minmax, minmax, color='k', linestyle='--', alpha=0.4, linewidth=4)
        minmax = [np.min([delta_4_data, delta_4_sims])-0.5, np.max([delta_4_data, delta_4_sims])+0.5]
        ax[1].set_ylim(minmax)
        ax[1].set_xlim(minmax)
        ax[1].plot(minmax, minmax, color='k', linestyle='--', alpha=0.4, linewidth=4)
        ax[0].text(-1, 1, f'r={round(corr_2.statistic, 2)} \np={corr_2.pvalue: .1e}')
        ax[1].text(-1, 1, f'r={round(corr_4.statistic, 2)} \np={corr_4.pvalue: .1e}')
        ax[0].set_xlabel('Hysteresis diff. data')
        ax[1].set_xlabel('Hysteresis diff. data')
        ax[0].set_ylabel('Hysteresis diff. simulations')
        ax[0].set_title('Freq. = 2', fontsize=12); ax[1].set_title('Freq. = 4', fontsize=12)
        ax[0].plot(delta_2_data, delta_2_sims, marker='o', color='k', linestyle='')
        ax[1].plot(delta_4_data, delta_4_sims, marker='o', color='k', linestyle='')
        for k in range(2):
            means_data = np.nanmean([delta_2_data, delta_4_data][k])
            means_sims = np.nanmean([delta_2_sims, delta_4_sims][k])
            ax[k].axhline(means_sims, color='k', alpha=0.5)
            ax[k].axvline(means_data, color='k', alpha=0.5)
        fig.tight_layout()
        # unique_shuffle = np.array(unique_shuffle)
        # jeffs = np.zeros((3, fitted_subs)); n=4
        # for i in range(fitted_subs):
        #     jeffs[:, i] = (fitted_params_all[i][0]*unique_shuffle+fitted_params_all[i][1])
        # jeffs_mask = jeffs >= 1  # 1/n
        # bistable_stim_2 = hyst_width_2_data[:, :fitted_subs].flatten()[jeffs_mask.flatten()]
        # monostable_stim_2 = hyst_width_2_data[:, :fitted_subs].flatten()[~jeffs_mask.flatten()]
        # bistable_stim_2_sims = hyst_width_2[:, :fitted_subs].flatten()[jeffs_mask.flatten()]
        # monostable_stim_2_sims = hyst_width_2[:, :fitted_subs].flatten()[~jeffs_mask.flatten()]
        # bistable_stim_4 = hyst_width_4_data[:, :fitted_subs].flatten()[jeffs_mask.flatten()]
        # monostable_stim_4 = hyst_width_4_data[:, :fitted_subs].flatten()[~jeffs_mask.flatten()]
        # bistable_stim_4_sims = hyst_width_4[:, :fitted_subs].flatten()[jeffs_mask.flatten()]
        # monostable_stim_4_sims = hyst_width_4[:, :fitted_subs].flatten()[~jeffs_mask.flatten()]
        # fig5, ax5 = plt.subplots(ncols=1, figsize=(5, 4))
        # ax5.spines['right'].set_visible(False); ax5.spines['top'].set_visible(False)
        # sns.kdeplot(bistable_stim_4, color='k', ax=ax5, label='Bistable')
        # sns.kdeplot(monostable_stim_4, color='r', ax=ax5, label='Monostable')
        # sns.kdeplot(bistable_stim_4_sims, color='k', ax=ax5, linestyle='--')
        # sns.kdeplot(monostable_stim_4_sims, color='r', ax=ax5, linestyle='--')
        # ax5.set_xlabel('Hysteresis'); ax5.legend(frameon=False)
        # fig5.tight_layout()
        np.save(DATA_FOLDER + 'hysteresis_width_f2_sims_fitted_params.npy', hyst_width_2)
        np.save(DATA_FOLDER + 'hysteresis_width_f4_sims_fitted_params.npy', hyst_width_4)
        fig3, ax3 = plt.subplots(1, figsize=(5, 4))
        ax3.plot([0, 3], [0, 3], color='k', alpha=0.4, linestyle='--', linewidth=4)
        for i_c in range(len(unique_shuffle)):
            ax3.plot(hyst_width_2[i_c], hyst_width_4[i_c],
                      color=colormap[i_c], marker='o', linestyle='')
        for i_s in range(fitted_subs):
            ax3.plot(hyst_width_2[:, i_s], hyst_width_4[:, i_s],
                      color='k', alpha=0.1)
        for a in [ax3]:
            a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
        ax3.set_ylabel('Width freq = 4')
        ax3.set_xlabel('Width freq = 2')
        fig3.tight_layout()
        fig3.savefig(SV_FOLDER + 'simulated_hysteresis_f4_vs_f2.png', dpi=400, bbox_inches='tight')
        fig3.savefig(SV_FOLDER + 'simulated_hysteresis_f4_vs_f2.svg', dpi=400, bbox_inches='tight')


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


def plot_j1_lmm_slopes(data_folder=DATA_FOLDER, sv_folder=SV_FOLDER):
    intercepts_2, slopes_2, _ =\
        lmm_hysteresis_dominance(freq=2, plot_summary=False,
                                 slope_random_effect=False, plot_individual=False)
    intercepts_4, slopes_4, _ =\
        lmm_hysteresis_dominance(freq=4, plot_summary=False,
                                 slope_random_effect=False, plot_individual=False)
    df = load_data(data_folder, n_participants='all')
    subjects = df.subject.unique()
    pars_all = np.zeros((5, len(subjects)))
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    for i_s, subject in enumerate(subjects):
        fitted_params = np.load(pars[i_s])
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


def model_known_params_pyddm(J1=0.3, J0=0.1, B=0.4, THETA=0.1, SIGMA=0.1, NDT=0, n=4, t_dur=10):
    # First create two versions of the model, one to simulate the data, and one to fit to the simulated data.
    stim = lambda t, freq, phase_ini: sawtooth(2 * np.pi * abs(freq)/2 * (t+phase_ini)/26, 0.5)*2*np.sign(freq)
    x_hat = lambda prev_choice, x: x if prev_choice == -1 else x+1
    starting_position = lambda prev_choice: 0.5-THETA if prev_choice == -1 else -0.5+THETA
    drift_function_sim = lambda t, x, pshuffle, prev_choice, freq, phase_ini: 1/(1+np.exp(-2*(n*(J0+J1*(1-pshuffle))*(2*x_hat(prev_choice, x)-1) + B*stim(t, freq, phase_ini))))-x_hat(prev_choice, x)
    conditions = ["pshuffle", "prev_choice", "freq", "phase_ini"]
    m_sim = pyddm.gddm(drift=drift_function_sim, 
                       conditions=conditions, starting_position=starting_position, bound=THETA+0.5, noise=SIGMA, nondecision=NDT,
                       T_dur=t_dur, dt=0.005, dx=0.005)
    return m_sim


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
    ndt = np.abs(np.median(np.load(SV_FOLDER + 'kernel_latency_average.npy')))
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


def plot_simulate_subject(data_folder=DATA_FOLDER, subject_name=None,
                          ntraining=8, n=4, window_conv=None, fps=60):
    np.random.seed(50)  # 24, 42, 13, 1234, 11  10with1000
    ndt = np.abs(np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    print(len(pars), ' fitted subjects')
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = [[n*params[0], n*params[1], params[2], params[4], params[3], ndt] for params in fitted_params_all]
    # print('J1, J0, B1, Sigma, Threshold')
    simulated_subjects(data_folder=DATA_FOLDER, tFrame=26, fps=fps,
                       sv_folder=SV_FOLDER, ntraining=ntraining,
                       plot=True, simulate=False, use_j0=True, subjects=None,
                       fitted_params_all=fitted_params_all, window_conv=window_conv)


def simulate_noise_subjects(df, data_folder=DATA_FOLDER, n=4, nFrame=1546, fps=60,
                            load_simulations=True):
    ratio = int(nFrame/1546)
    nFrame = nFrame-ratio+1
    ndt = np.abs(np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    print(len(pars), ' fitted subjects')
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = [[n*params[0], n*params[1], params[2], params[4], params[3], ndt] for params in fitted_params_all]
    subjects = df.subject.unique()[:len(pars)]
    time_interp = np.arange(0, nFrame, 1)/fps
    time_frames = len(time_interp)
    responses_all = np.zeros((len(subjects), 36, time_frames))
    stim_subject = np.zeros((len(subjects), 36, time_frames))
    pshuffles_all = np.zeros((len(subjects), 36))
    time = np.arange(0, 1546, 1)/60
    if load_simulations:
        stim_subject = np.load(SV_FOLDER + 'stim_subject_noise.npy')
        responses_all = np.load(SV_FOLDER + 'responses_simulated_noise.npy')
        pshuffles_all = np.load(SV_FOLDER + 'stim_subject_pshuffles.npy')
    else:
        for i_s, subject in enumerate(subjects):
            print('Simulating noise trials subject ', subject)
            fitted_params_subject = fitted_params_all[i_s]
            df_sub = df.loc[df.subject == subject]
            pshuffles = df_sub.groupby("trial_index").pShuffle.mean().values
            trial_indices = df_sub.trial_index.unique()
            choices_subject = np.zeros((len(trial_indices), time_frames))
            for i_trial, trial in enumerate(trial_indices):
                j_eff = (1-pshuffles[i_trial])*fitted_params_subject[0] + fitted_params_subject[1]
                params = fitted_params_subject[1:]
                params[0] = j_eff
                df_sub_trial = df_sub.loc[df_sub.trial_index == trial]
                stimulus = df_sub_trial.stimulus[:-14].values
                stimulus = scipy.interpolate.interp1d(time, stimulus)(time_interp)
                # stimulus = np.repeat(stimulus, np.ceil(nFrame/1546))
                x = np.zeros(choices_subject.shape[1])
                j_eff, b_par, th, sigma, ndt = params
                lower_bound, upper_bound = np.array([-th, th]) + 0.5
                dt = 1/fps; tau=1
                b_eff = stimulus*b_par
                noise = np.random.randn(nFrame)*sigma*np.sqrt(dt/tau)
                x = np.zeros(time_frames)
                x[0] = 0.5
                choice = np.zeros(nFrame)
                prev_choice = 0.0
                pending_choice = None      # choice scheduled but not yet applied
                pending_time = None        # when to apply it
                ndt_frames = int(ndt / dt)
                
                for t in range(1, nFrame):
                    # apply pending choice if NDT has elapsed
                    if pending_choice is not None and t >= pending_time:
                        prev_choice = pending_choice
                        pending_choice = None
    
                    # evolve freely (no stickiness)
                    drive = sigmoid(2 * (j_eff * (2 * x[t - 1] - 1) + b_eff[t]))
                    x[t] = x[t - 1] + dt * (drive - x[t - 1]) / tau + noise[t]
                
                    # bound crossing
                    if x[t] >= upper_bound:
                        new_choice = 1.0
                    elif x[t] <= lower_bound:
                        new_choice = -1.0
                    else:
                        new_choice = prev_choice
                
                    # schedule motor choice change (after NDT delay)
                    if new_choice != prev_choice and pending_choice is None:
                        pending_choice = new_choice
                        pending_time = t + ndt_frames
    
                    # delayed decision
                    choice[t] = prev_choice
                choices_subject[i_trial, :] = choice
                stim_subject[i_s, i_trial, :] = stimulus    
            responses_all[i_s, :, :] = choices_subject
            pshuffles_all[i_s] = pshuffles
        np.save(SV_FOLDER + 'stim_subject_noise.npy', stim_subject)
        np.save(SV_FOLDER + 'responses_simulated_noise.npy', responses_all)
        np.save(SV_FOLDER + 'stim_subject_pshuffles.npy', pshuffles_all)
    return stim_subject, responses_all, pshuffles_all


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
    fig2.savefig(SV_FOLDER + 'simulated_nosie_kernel_all.svg', dpi=200, bbox_inches='tight')


def plot_simulated_subjects_noise_trials(data_folder=DATA_FOLDER,
                                         shuffle_vals=[1., 0.7, 0.], ntrials=36,
                                         steps_back=120, steps_front=20, avoid_first=True,
                                         tFrame=26, window_conv=1,
                                         zscore_number_switches=False, fps=60, ax=None, hysteresis_area=False,
                                         normalize_variables=False, ratio=1, nFrame=1546, n=4,
                                         load_simulations=False):
    ndt = np.abs(np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = [[n*params[0], n*params[1], params[2], params[4], params[3], ndt] for params in fitted_params_all]
    steps_back = steps_back*ratio; steps_front = steps_front*ratio
    nFrame = nFrame*ratio; fps= fps*ratio
    df = load_data(data_folder + '/noisy/', n_participants='all')
    noise_signal, choice, pshuffles = simulate_noise_subjects(df, data_folder=DATA_FOLDER, n=4, nFrame=nFrame, fps=fps,
                                                              load_simulations=load_simulations)
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
            for i_trial, trial in enumerate(trial_index):
                is_bistable = (fitted_params_all[i_sub][0]*(1-pshuffle)+fitted_params_all[i_sub][1]) >= 1
                responses = choice[i_sub, idx_trials[i_trial]]
                chi = noise_signal[i_sub, idx_trials[i_trial]]
                # chi = chi-np.nanmean(chi)
                orders = rle(responses)
                if avoid_first:
                    idx_1 = orders[1][1:][orders[2][1:] == 1]
                    idx_0 = orders[1][1:][orders[2][1:] == -1]
                else:
                    idx_1 = orders[1][orders[2] == 1]
                    idx_0 = orders[1][orders[2] == -1]
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
            mean_vals_noise_switch_all_trials = mean_vals_noise_switch_all_trials[1:]
            # it's better to compute afterwards, with the average peak per coupling
            # because trial by trial there is a lot of noise and that breaks the mean/latency
            # it gets averaged out
            mean_number_switchs_coupling[i_sh, i_sub] = tFrame/ np.nanmean(np.array(number_switches))
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
        fig, ax = plt.subplots(1, figsize=(5.5, 4))
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
    ax.legend(frameon=False); ax.set_xlabel('Time before switch(s)')
    ax.set_ylabel('Noise')
    fig.tight_layout()
    
    fig, ax = plt.subplots(ncols=1, figsize=(5, 4))
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
    # for i_sub in range(len(subs)):
    #     y_plot = all_kernels[i_sub]
    #     ax.plot(x_plot, y_plot, color='k', linewidth=2, alpha=0.5)
    x_plot = np.arange(-steps_back, 0, 1)/fps
    y_plot = np.nanmean(all_kernels, axis=0)[:-steps_front]
    ax.plot(x_plot, y_plot, color='k', linewidth=4, )
    err = np.nanstd(all_kernels, axis=0)[:-steps_front]/np.sqrt(len(subs))
    ax.fill_between(x_plot, y_plot-err, y_plot+err, color='k', alpha=0.2)
    ax.set_xlabel('Time before switch(s)')
    ax.set_ylabel('Noise')
    fig.tight_layout()


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
    re_formula = "~x"
    model = smf.mixedlm("y ~ x", df, groups=df["subject"],
                        re_formula=re_formula)
    result = model.fit()
    fe = result.fe_params
    re = result.random_effects
    # get intercepts/slopes per subject
    intercepts = [fe["Intercept"] + eff.get("Group", 0) for subj, eff in re.items()]
    slopes = [fe["x"] + eff.get("x", 0) for subj, eff in re.items()]
    print(result.summary())


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
    sns.stripplot([bistable_stim_2_dominance, monostable_stim_2_dominance], color='k', ax=ax5, size=3)
    ax5.set_xticks([0, 1], ['Bistable', 'Monostable'])
    ax5.set_ylabel('Dominance duration');
    if simulations:
        ax5.set_ylim(0, 5.3)
    else:
        ax5.set_ylim(0, 12.5)
    fig5.tight_layout()


def plot_noise_variables_vs_fitted_params(n=4, variable='dominance'):
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    fitted_subs = len(pars)
    b1s = [np.load(par)[2] for par in pars]
    sigmas = np.array([np.load(par)[3] for par in pars])
    thetas = [np.load(par)[4] for par in pars]
    j1s = np.array([np.load(par)[0] for par in pars])  # /sigmas
    j0s = np.array([np.load(par)[1] for par in pars])  # /sigmas
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
    var = np.load(DATA_FOLDER + file)
    mean_duration_per_sub = np.nanmean(var, axis=0)[:fitted_subs]
    rho, pval = pearsonr(j0s, mean_duration_per_sub)
    fig, ax = plt.subplots(1, figsize=(5, 4))
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.plot(j0s, mean_duration_per_sub, marker='o', linestyle='', color='k')
    ax.set_title('r = ' + str(round(rho, 3)) + '\np = ' + str(round(pval, 5)))
    ax.set_xlabel('Fitted J0')
    ax.set_ylabel(label)
    fig.tight_layout()
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


def plot_coupling_transitions(n=4):
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


def compare_parameters_two_experiments():
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
    sigmas_exp2 = np.array([np.load(par)[3] for par in params_experiment_2])
    j1s_exp1 = np.array([par[0] for par in params_experiment_1])
    j0s_exp1 = np.array([par[1] for par in params_experiment_1])
    b1s_exp1 = np.array([par[2] for par in params_experiment_1])
    sigmas_exp1 = np.array([par[4] for par in params_experiment_1])
    parameter_pairs = [[j0s_exp1, j0s_exp2], [j1s_exp1, j1s_exp2], [b1s_exp1, b1s_exp2],
                                                                    [sigmas_exp1, sigmas_exp2]]
    labels = ['J0', 'J1', 'B1', r'$\sigma$']; colors = ['mediumpurple', 'burlywood']
    fig, ax = plt.subplots(ncols=4, figsize=(10, 3))
    for i_a, a in enumerate(ax):
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
        pvalue = scipy.stats.mannwhitneyu(parameter_pairs[i_a][0], parameter_pairs[i_a][1]).pvalue
        # print(pvalue)
        heights = [np.nanmean(parameter_pairs[i_a][k]) for k in range(2)]
        barplot_annotate_brackets(0, 1, pvalue, [0, 1], heights, yerr=None, dh=.16, barh=.05, fs=10,
                                  maxasterix=3, ax=a)
        sns.barplot(parameter_pairs[i_a], ax=a, linewidth=3, palette=colors, errorbar='se')
        sns.stripplot(parameter_pairs[i_a], ax=a, color='k', size=2.5)
        a.set_xticks([]); a.set_ylabel(labels[i_a])
    # Create legend only once (not in the loop)
    handles = [mpatches.Patch(color=colors[0], label='Exp. 1'),
               mpatches.Patch(color=colors[1], label='Exp. 2')]
    fig.tight_layout()
    ax[0].legend(handles=handles, loc='upper center', frameon=False,
                 bbox_to_anchor=[0.7, 0.8])


def hysteresis_vs_bimodality():
    folder_bimodal = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/fitting/parameters/'  # Alex
    bimodal_coef = np.load(folder_bimodal + 'bimodality_coefficient.npy')
    mean_peak_amplitude = np.load(DATA_FOLDER + 'mean_peak_amplitude_per_subject.npy')
    mean_peak_latency = np.load(DATA_FOLDER + 'mean_peak_latency_per_subject.npy')
    mean_number_switchs_coupling = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
    hyst_width_2 = np.load(DATA_FOLDER + 'hysteresis_width_freq_2.npy')
    hyst_width_4 = np.load(DATA_FOLDER + 'hysteresis_width_freq_4.npy')
    fig, ax = plt.subplots(ncols=3, figsize=(9.2, 4))
    for a in ax:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    sns.barplot(bimodal_coef.T, palette=colormap, errorbar='se', ax=ax[0])
    sns.barplot(hyst_width_2.T, palette=colormap, errorbar='se', ax=ax[1])
    sns.barplot(mean_number_switchs_coupling.T, palette=colormap, errorbar='se', ax=ax[2])
    ax[0].plot(np.nanmedian(bimodal_coef, axis=1), color='firebrick', linewidth=4)
    ax[1].plot(np.nanmedian(hyst_width_2, axis=1), color='firebrick', linewidth=4)
    ax[2].plot(np.nanmedian(mean_number_switchs_coupling, axis=1), color='firebrick', linewidth=4)
    ax[0].axhline(5/9, color='gray', linestyle='--', linewidth=3)
    ax[0].set_ylim(0.5, 0.67);  ax[1].set_ylim(0.85, 1.62);  ax[2].set_ylim(5.5, 9)
    ax[0].set_ylabel("Snarle's bimodality coefficient")
    ax[2].set_ylabel("Dominance duration (s)")
    ax[0].set_yticks([0.5, 5/9, 0.6], ['0.5', '5/9', '0.6'])
    ax[0].set_xticks([0, 1, 2], [1., 0.7, 0.]); ax[0].set_xlabel('p(shuffle)');
    ax[2].set_xlabel('p(shuffle)')
    ax[1].set_xlabel('p(shuffle)');  ax[1].set_ylabel('Hysteresis f=2')
    handles = [mpatches.Patch(color=colormap[0], label='1.'),
               mpatches.Patch(color=colormap[1], label='0.7'),
               mpatches.Patch(color=colormap[2], label='0.')]
    ax[0].legend(handles=handles,frameon=False,
                 title='p(shuffle)')
    fig.tight_layout()
    

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


def plot_kernel_different_regimes(data_folder=DATA_FOLDER, fps=60, tFrame=26,
                                  steps_back=60, steps_front=20,
                                  shuffle_vals=[1, 0.7, 0],
                                  avoid_first=False, window_conv=1,
                                  filter_subjects=True, n=4):
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = np.array([[n*params[0], n*params[1], params[2], params[4], params[3]] for params in fitted_params_all])
    fitted_subs = len(pars)
    # idxs = [(par[4] < 0.25)*(par[3] < 0.1) for par in fitted_params_all]
    # fitted_params_all = fitted_params_all[idxs]
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
    fig, ax = plt.subplots(ncols=1, figsize=(5, 4))
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
        ax.plot(x_plot, y_plot, color=colormap[regime],
                label=labels[regime], linewidth=3)
        ax.fill_between(x_plot, y_plot-err_plot, y_plot+err_plot, color=colormap[regime],
                        alpha=0.3)
    ax.legend(frameon=False); ax.set_xlabel('Time before switch(s)')
    ax.set_ylabel('Noise')
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'kernel_across_subjects.png', dpi=400)
    fig.savefig(DATA_FOLDER + 'kernel_across_subjects.svg', dpi=400)



def plot_kernel_different_parameter_values(data_folder=DATA_FOLDER, fps=60, tFrame=26,
                                           steps_back=60, steps_front=20,
                                           shuffle_vals=[1, 0.7, 0],
                                           avoid_first=False, window_conv=1,
                                           filter_subjects=True, n=4, variable='J0',
                                           simulated=False,
                                           pshuff=None):
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    if variable == 'J0':
        var = [np.load(par)[1] for par in pars]
        bins = [-0.1, 0.12, 0.2, 1/2]
        # bins = np.percentile(var, (0, 25, 50, 75, 100))
        colormap = pl.cm.Oranges(np.linspace(0.3, 1, len(bins)))
    if variable == 'B1':
        var = [np.load(par)[2] for par in pars]
        bins =  [0, 0.2, 0.6, 0.8]
        # bins = np.percentile(var, (0, 33, 66, 100))
        colormap = pl.cm.Greens(np.linspace(0.3, 1, len(bins)))
    if variable == 'THETA':
        var = [np.load(par)[4] for par in pars]
        bins = np.percentile(var, (0, 33, 66, 100))+np.array([-1e-6, 0, 0, 1e-6])
        bins = [0., 0.02, 0.3]
        colormap = pl.cm.Blues(np.linspace(0.3, 1, len(bins)))
    if variable == 'SIGMA':
        var = [np.load(par)[3] for par in pars]
        bins = np.percentile(var, (0, 1/3*100, 100*2/3, 100))+np.array([-1e-6, 0, 0, 1e-6])
        # bins = [0., 0.2, 0.26, 0.3]
        colormap = pl.cm.Greens(np.linspace(0.3, 1, len(bins)))
    if variable == 'J1':
        var = [np.load(par)[0] for par in pars]
        bins = np.percentile(var, (0, 1/3*100, 100*2/3, 100))+np.array([-1e-6, 0, 0, 1e-6])
        colormap = pl.cm.Reds(np.linspace(0.3, 1, len(bins)))
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

    fig, ax = plt.subplots(ncols=1, figsize=(5, 4))
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
    ax.legend(title=variable, frameon=False); ax.set_xlabel('Time before switch(s)')
    ax.set_ylabel('Noise')
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + label_save_fig + f'kernel_across_subjects_different_{variable}.png', dpi=400)
    fig.savefig(DATA_FOLDER + label_save_fig + f'kernel_across_subjects_different_{variable}.svg', dpi=400)


def enums(idx):
    if idx == 1:
        return 'st'
    elif idx == 2:
        return 'nd'
    elif idx == 3:
        return 'rd'
    elif idx >= 4:
        return 'th'


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


def expand_responses(df, sampling_rate=60, trial_duration=26.0):
    """
    Convert event-based responses (time_onset, time_offset) into a
    continuous binary time series sampled at `sampling_rate`.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ['trial', 'response', 'time_onset', 'time_offset'].
    sampling_rate : float
        Sampling frequency in Hz.
    trial_duration : float
        Duration of each trial in seconds.

    Returns
    -------
    resp_cont : dict
        Keys = trial IDs, values = binary numpy arrays of shape (T,)
    time_vector : numpy array
        Time axis (in seconds).
    """
    nFrame = int(trial_duration*sampling_rate)
    dt = 1.0 / sampling_rate
    time_vector = np.arange(0, trial_duration, dt)
    resp_by_p = {}
    stim_by_p = {}

    for trial, df_trial in df.groupby('trial_index'):
        r = np.zeros_like(time_vector, dtype=int)
        for _, row in df_trial.iterrows():
            idx_on = np.searchsorted(time_vector, row['keypress_seconds_onset'])
            idx_off = np.searchsorted(time_vector, row['keypress_seconds_offset'])
            r[idx_on:idx_off] = int(row['response'])
        freq = df_trial.freq.values[0]*df_trial.initial_side.values[0]
        blist = get_blist(freq, nFrame=nFrame)
        # Group by p_value
        p = df_trial.pShuffle.values[0]
        if p not in resp_by_p:
            resp_by_p[p] = []
            stim_by_p[p] = []

        resp_by_p[p].append(r)
        stim_by_p[p].append(blist)

    return resp_by_p, stim_by_p, time_vector


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
    fig.savefig(SV_FOLDER + label + 'cartoon.svg', dpi=400, bbox_inches='tight')


def get_likelihood(pars, n=4, data_folder=DATA_FOLDER, ntraining=8, nbins=27, t_dur=15):
    set_N_cpus(8)
    print(len(pars), ' fitted subjects')
    ndt = np.abs(np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = [[params[0], params[1], params[2], params[4], params[3], ndt] for params in fitted_params_all]
    df = load_data(data_folder, n_participants='all')
    df = df.loc[df.trial_index > ntraining]
    subjects = df.subject.unique()
    bins = np.linspace(0, 26, nbins).round(2)
    llhs = []
    bics = []
    for i_s, subject in enumerate(subjects):
        print('Fitting subject ', subject)
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


def compare_likelihoods_models(load=True, bic=False):
    if not load:
        pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
        likelihood_with_ndt, bic_with_ndt = get_likelihood(pars, n=4, data_folder=DATA_FOLDER, ntraining=8, nbins=27, t_dur=15)
        np.save(SV_FOLDER + 'likelihood_model_with_ndt.npy', likelihood_with_ndt)
        np.save(SV_FOLDER + 'bic_model_with_ndt.npy', bic_with_ndt)
        pars2 = glob.glob(SV_FOLDER + 'fitted_params/' + '*.npy')
        likelihood_without_ndt, bic_without_ndt = get_likelihood(pars2, n=4, data_folder=DATA_FOLDER, ntraining=8, nbins=27, t_dur=15)
        np.save(SV_FOLDER + 'likelihood_model_without_ndt.npy', likelihood_without_ndt)
        np.save(SV_FOLDER + 'bic_model_without_ndt.npy', bic_without_ndt)
    if load:
        likelihood_without_ndt = np.array(np.load(SV_FOLDER + 'likelihood_model_without_ndt.npy'))
        bic_without_ndt = np.array(np.load(SV_FOLDER + 'bic_model_without_ndt.npy'))
        likelihood_with_ndt = np.array(np.load(SV_FOLDER + 'likelihood_model_with_ndt.npy'))
        bic_with_ndt = np.array(np.load(SV_FOLDER + 'bic_model_with_ndt.npy'))
    fig5, ax5 = plt.subplots(ncols=1, figsize=(3.5, 4))
    losses = [bic_without_ndt, bic_with_ndt] if bic else [likelihood_without_ndt, likelihood_with_ndt]
    ax5.spines['right'].set_visible(False); ax5.spines['top'].set_visible(False)
    sns.barplot(losses, palette=['peru', 'cadetblue'], ax=ax5)
    pvalue = scipy.stats.mannwhitneyu(losses[0], losses[1]).pvalue
    heights = [np.nanmean(losses[k]) for k in range(2)]
    barplot_annotate_brackets(0, 1, pvalue, [0, 1], heights, yerr=None, dh=.2, barh=.02, fs=10,
                              maxasterix=3, ax=ax5)
    sns.stripplot(losses, color='k', ax=ax5, size=3)
    ax5.set_xticks([0, 1], ['Without NDT', 'With NDT'])
    ylabel = 'BIC' if bic else 'NLH'
    ax5.set_ylabel(ylabel)
    fig5.tight_layout()
    fig5, ax5 = plt.subplots(ncols=1, figsize=(2., 4))
    losses = [bic_without_ndt-bic_with_ndt] if bic else [likelihood_without_ndt-likelihood_with_ndt]
    ax5.spines['right'].set_visible(False); ax5.spines['top'].set_visible(False)
    sns.barplot(losses, palette=['peru', 'cadetblue'], ax=ax5)
    sns.stripplot(losses, color='k', ax=ax5, size=3)
    ax5.axhline(0, color='k', linestyle='--', linewidth=2, alpha=0.4)
    ax5.text(0.5, 4, 'Better', rotation=90, fontsize=13)
    ax5.text(0.5, -15, 'Worse', rotation=90, fontsize=13)
    ax5.set_xlim(-0.5, 0.8)
    ax5.set_xticks([0], ['Without NDT - With NDT'])
    ax5.set_ylim(-20, 62)
    ylabel = r'$\Delta$BIC' if bic else r'$\Delta$NLH'
    ax5.set_ylabel(ylabel)
    fig5.tight_layout()


def low_dimensional_projection():
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    ndt = np.abs(np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = np.array([[params[0], params[1], params[2], params[4], params[3]] for params in fitted_params_all])
    scaling = manifold.MDS(n_components=3, max_iter=5000, normalized_stress=False)
    S_scaling = scaling.fit_transform(fitted_params_all)
    x, y, z = S_scaling.T
    color = fitted_params_all[:, 1]
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter3D(x, y, z, c=color, cmap='copper')
    plt.xlabel('MDS-dim1'); plt.ylabel('MDS-dim2');
    plt.title('Colored by J1+J0')


def vector_field_cylinder(p_shuff=1):
    x = np.linspace(0, np.pi, 12)
    y = np.linspace(0, np.pi, 12)
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
    plt.quiver(X, Y, U_shuffled, V, color='darkgreen')
    ax.axis('off')
    fig.savefig(SV_FOLDER + f'vf_cylinder_p_shuffle_{p_shuff}.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + f'vf_cylinder_p_shuffle_{p_shuff}.svg', dpi=400, bbox_inches='tight')


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
                             ntraining=8, freq=2):

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
    fig2.savefig(SV_FOLDER + f'dominance_durations_freq_{freq}.svg', dpi=400, bbox_inches='tight')

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
    fig2.savefig(SV_FOLDER + f'average_dominance_durations_freq_{freq}.svg', dpi=400, bbox_inches='tight')


    fig2, ax2 = plt.subplots(1, figsize=(4, 3.5))
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    sns.boxplot(mean_dominance_shuffle.T, ax=ax2, palette=colormap, linewidth=4)
    sns.stripplot(mean_dominance_shuffle.T, ax=ax2, color='k')
    for i in range(nsubs):
        ax2.plot([0, 1, 2], mean_dominance_shuffle[:, i], color='gray', alpha=0.5)
    ax2.set_ylabel('Dominance (s)')
    fig2.tight_layout()


    fig3, ax3 = plt.subplots(1, figsize=(4, 3.5))
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    if freq == 4:
        file = 'hysteresis_width_freq_4.npy'
    if freq == 2:
        file = 'hysteresis_width_freq_2.npy'
    var = np.load(DATA_FOLDER + file)
    r, p = pearsonr(var.flatten(), mean_dominance_shuffle.flatten())
    ax3.annotate(f'r = {r:.3f}\np = {p:.1e}', xy=(.1, 0.1), xycoords=ax3.transAxes)

    for i in range(3):
        meanx = np.mean(var[i]); meany = np.mean(mean_dominance_shuffle[i])
        errx = np.std(var[i]); erry = np.std(mean_dominance_shuffle[i])
        ax3.errorbar(x=meanx, y=meany, xerr=errx, yerr=erry, color=colormap[i],
                     marker='o')
        ax3.plot(var[i], mean_dominance_shuffle[i], color=colormap[i], linestyle='',
                 marker='x', alpha=0.2)
    ax3.set_xlabel('Hysteresis')
    ax3.set_ylabel('Dominance (s)'); fig3.tight_layout()


def compute_switch_prob_group(stim, choice, freq, pshuffle, n_bins=50, T_trial=26):
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

    n_subj, n_trials, n_time = stim.shape
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

                    p_LR, _ = np.histogram(t_norm[switch_LR], bins=bin_edges, range=(0, T_trial))
                    p_RL, _ = np.histogram(t_norm[switch_RL], bins=bin_edges, range=(0, T_trial))
                    count, _ = np.histogram(t_norm, bins=bin_edges)

                    p_LR_s += p_LR
                    p_RL_s += p_RL
                    count_s += count

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
    df_switches = compute_switch_prob_group(all_stims, choices_all_subject, freqs_per_sub, pshuffles_per_sub, n_bins=n_bins)
    fig, axes = plt.subplots(ncols=2, figsize=(7.5, 4.))
    titles = ['Freq = 2', 'Freq = 4']
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


if __name__ == '__main__':
    print('Running hysteresis_analysis.py')
    # plot_dominance_durations(data_folder=DATA_FOLDER,
    #                           ntraining=8, freq=2)
    # plot_dominance_durations(data_folder=DATA_FOLDER,
    #                           ntraining=8, freq=4)
    # plot_dominance_distros_noise_trials_per_subject(data_folder=DATA_FOLDER, fps=60, tFrame=26,
    #                                                     simulated=False)
    # plot_dominance_distros_noise_trials_per_subject(data_folder=DATA_FOLDER, fps=60, tFrame=26,
    #                                                     simulated=True)
    # plot_dominance_bis_mono(unique_shuffle=[1., 0.7, 0.], n=4)
    # plot_dominance_bis_mono(unique_shuffle=[1., 0.7, 0.], n=4, simulations=True)
    # plot_noise_variables_vs_fitted_params(n=4, variable='freq4')
    # plot_params_distros(ndt=True)
    # plot_simulate_subject(data_folder=DATA_FOLDER, subject_name=None,
    #                       ntraining=8, window_conv=1, fps=200)
    plot_switch_rate_model(data_folder=DATA_FOLDER, sv_folder=SV_FOLDER,
                          fps=200, n=4, ntraining=8, tFrame=26,
                          window_conv=5, n_bins=80)
    # plot_kernel_different_regimes(data_folder=DATA_FOLDER, fps=60, tFrame=26,
    #                               steps_back=150, steps_front=20,
    #                               shuffle_vals=[1, 0.7, 0],
    #                               avoid_first=True, window_conv=1,
    #                               filter_subjects=True, n=4)
    # compare_parameters_two_experiments()
    # plot_simulated_subjects_noise_trials(data_folder=DATA_FOLDER,
    #                                      shuffle_vals=[1., 0.7, 0.], ntrials=36,
    #                                      steps_back=150, steps_front=20, avoid_first=True,
    #                                      tFrame=26, window_conv=1,
    #                                      fps=60, ax=None, hysteresis_area=True,
    #                                      normalize_variables=True, ratio=1,
    #                                      load_simulations=True)
    # for variable in  ['B1']:
    #     plot_kernel_different_parameter_values(data_folder=DATA_FOLDER, fps=60, tFrame=26,
    #                                             steps_back=120, steps_front=20,
    #                                             shuffle_vals=[1, 0.7, 0],
    #                                             avoid_first=False, window_conv=1,
    #                                             filter_subjects=True, n=4, variable=variable,
    #                                             simulated=False, pshuff=None)
    #     plot_kernel_different_parameter_values(data_folder=DATA_FOLDER, fps=60, tFrame=26,
    #                                             steps_back=120, steps_front=20,
    #                                             shuffle_vals=[1, 0.7, 0],
    #                                             avoid_first=False, window_conv=1,
    #                                             filter_subjects=True, n=4, variable=variable,
    #                                             simulated=True, pshuff=None)
    # plot_example(theta=[0.1, 0, 0.5, 0.1, 0.5], data_folder=DATA_FOLDER,
    #              fps=60, tFrame=18, model='MF', prob_flux=False,
    #              freq=4, idx=2)
    # noise_bf_switch_coupling(load_sims=True, coup_vals=np.arange(0.05, 0.35, 1e-2),  # np.array((0.13, 0.17, 0.3))
    #                           nFrame=100000, fps=60, noisyframes=30,
    #                           n=4.0, steps_back=60, steps_front=20,
    #                           ntrials=20, hysteresis_width=False,
    #                           th=0.3)
    # plot_hysteresis_width_simluations(coup_vals=np.array((0., 0.3, 1))*0.27+0.02,
    #                                   b_list=np.linspace(-0.5, 0.5, 501),
    #                                   window_conv=1)
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
    #                       nbins=10, ntraining=8, arrows=False, subjects=['s_1'],
    #                       window_conv=None)
    # plot_max_hyst_ndt_subject(tFrame=26, fps=60, data_folder=DATA_FOLDER,
    #                           ntraining=8, coupling_levels=[0, 0.3, 1],
    #                           window_conv=None, ndt_list=np.arange(-240, 80))
    # plot_hysteresis_average(tFrame=26, fps=60, data_folder=DATA_FOLDER,
    #                         ntraining=8, coupling_levels=[0, 0.3, 1],
    #                         window_conv=None, ndt_list=None)
    # simple_recovery_pyddm(J1=0.3, J0=0.1, B=0.4, THETA=0.1, SIGMA=0.1)
    # save_params_pyddm_recovery(n_pars=100, i_ini=29, sv_folder=SV_FOLDER)
    # recovery_pyddm(n_pars=30, sv_folder=SV_FOLDER, n_cpus=11, i_ini=0)
    # fit_data_pyddm(data_folder=DATA_FOLDER, ncpus=12, ntraining=8, t_dur=13,
    #                subj_ini=None, nbins=54, fitting_method='bads')
    # parameter_recovery_5_params(n_simuls_network=1, fps=60, tFrame=26,
    #                             n_pars_to_fit=30, n_sims_per_par=100,
    #                             sv_folder=SV_FOLDER, simulate=True,
    #                             load_net=False, not_plot_and_return=False,
    #                             pyddmfit=True, transform=False)
    # plot_switch_rate(tFrame=26, fps=60, data_folder=DATA_FOLDER,
    #                   ntraining=8, coupling_levels=[0, 0.3, 1],
    #                   window_conv=5, bin_size=0.35, switch_01=False)
    # plot_sequential_effects(data_folder=DATA_FOLDER, ntraining=8)
    # get_rt_distro_and_incorrect_resps(data_folder=DATA_FOLDER,
    #                                   ntraining=8, coupling_levels=[0, 0.3, 1])
    # hysteresis_basic_plot_simulation(coup_vals=np.array((0., 0.3, 1))*0.27+0.02,
    #                                  fps=60, nsubs=1, n=4, nsims=1000,
    #                                  b_list=np.linspace(-0.5, 0.5, 501))
    # plot_noise_before_switch(data_folder=DATA_FOLDER, fps=60, tFrame=26,
    #                           steps_back=150, steps_front=10,
    #                           shuffle_vals=[1, 0.7, 0], violin=True, sub=None,
    #                           avoid_first=True, window_conv=1,
    #                           zscore_number_switches=False, 
    #                           normalize_variables=True, hysteresis_area=True)
    # hysteresis_basic_plot_all_subjects(coupling_levels=[0, 0.3, 1],
    #                                     fps=60, tFrame=26, data_folder=DATA_FOLDER,
    #                                     ntraining=8, arrows=False)
    # hysteresis_basic_plot(coupling_levels=[0, 0.3, 1],
    #                       fps=60, tFrame=18, data_folder=DATA_FOLDER,
    #                       nbins=9, ntraining=8, arrows=True)
    # save_5_params_recovery(n_pars=100, sv_folder=SV_FOLDER, i_ini=0)
    # for sims in [1000000]:
    #     parameter_recovery_5_params(n_simuls_network=sims, fps=60, tFrame=26,
    #                                 n_pars_to_fit=100, n_sims_per_par=100,
    #                                 sv_folder=SV_FOLDER, simulate=True,
    #                                 load_net=False, not_plot_and_return=False)
    #     plt.close('all')
    # lmm_hysteresis_dominance(freq=2, plot_summary=True,
    #                          slope_random_effect=False, plot_individual=False)
    # lmm_hysteresis_dominance(freq=2, plot_summary=True,
    #                          slope_random_effect=True, plot_individual=True)
