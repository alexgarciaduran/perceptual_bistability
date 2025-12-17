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
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pylab as pl
from matplotlib.lines import Line2D
import glob
from sklearn.metrics import roc_curve, auc
from sklearn import manifold
from sklearn.linear_model import LogisticRegression, LinearRegression
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
from scipy.optimize import curve_fit, root_scalar
from scipy.integrate import quad, cumulative_trapezoid, solve_bvp
import itertools
from pyddm import set_N_cpus
from pyddm.models.loss import LossLikelihood, LossBIC
from pyddm.functions import get_model_loss
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter, zoom
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import mplcairo

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

COLORMAP = LinearSegmentedColormap.from_list('rg', ['darkgreen', 'gainsboro', 'crimson'], N=128)


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
    np.save(DATA_FOLDER + 'switch_time_diff_freq_2.npy', switch_time_diff_2)
    np.save(DATA_FOLDER + 'switch_time_diff_freq_4.npy', switch_time_diff_4)
    # ax2[0].set_ylabel('Hysteresis width')
    ax[0].set_xlabel('Depth cue, s(t)')
    ax[1].set_xlabel('Depth cue, s(t)')
    ax[0].set_ylabel('P(rightward)')
    ax[0].legend(title='p(shuffle)', frameon=False,
                 bbox_to_anchor=[-0.02, 1.07], loc='upper left')
    ax[0].set_title('Freq = 2', fontsize=14)
    ax[1].set_title('Freq = 4', fontsize=14)
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'hysteresis_average.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'hysteresis_average.pdf', dpi=400, bbox_inches='tight')
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
    fig3.savefig(SV_FOLDER + 'hysteresis_2_vs_4.pdf', dpi=400, bbox_inches='tight')


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
    fig.savefig(SV_FOLDER + 'switch_rate.pdf', dpi=400, bbox_inches='tight')
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
    ax[0].set_xlabel('Depth cue, s(t)')
    ax[1].set_xlabel('Depth cue, s(t)')
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
    ax2[-1].set_xlabel('Time before switch (s)')
    ax[i_s*2].set_xlabel('Depth cue, s(t)')
    ax[i_s*2+1].set_xlabel('Depth cue, s(t)')
    ax[i_s*2-1].set_xlabel('Depth cue, s(t)')
    ax[i_s*2-2].set_xlabel('Depth cue, s(t)')
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
    ax[0].set_xlabel('Depth cue, s(t)')
    ax[1].set_yticks([0, 0.5, 1], ['', '', ''])
    ax[1].set_xlabel('Depth cue, s(t)')
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
        x_plot = np.arange(-steps_back, steps_front, 1)/fps
        latency = (np.argmax(np.nanmean(mean_vals_noise_switch_all_trials[:, :-steps_front], axis=0))-steps_back)/fps
        peakval = np.nanmax(np.nanmean(mean_vals_noise_switch_all_trials[:, :-steps_front], axis=0))
        kernel = np.nanmean(mean_vals_noise_switch_all_trials, axis=0)
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
    fignew.savefig(SV_FOLDER + 'latency_computation.pdf', dpi=100, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'noise_before_switch_experiment.png', dpi=100, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'noise_before_switch_experiment.pdf', dpi=100, bbox_inches='tight')
    figlast, ax = plt.subplots(ncols=1, figsize=(5, 4))
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
    # for i_sub in range(len(subs)):
    #     y_plot = all_kernels[i_sub]
    #     ax.plot(x_plot, y_plot, color='k', linewidth=2, alpha=0.5)
    x_plot = np.arange(-steps_back, steps_front, 1)/fps
    y_plot = np.nanmean(all_kernels, axis=0)
    np.save(DATA_FOLDER + 'all_kernels_noise_switch_aligned.npy', all_kernels)
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
    ax[0].set_ylabel('Depth cue, s(t)')
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
    x1 = (np.sign(b_eff[0])+1)/2
    for i in range(5):
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


def simulator_5_params(params, freq, nFrame=1560, fps=60,
                       return_choice=False, ini_cond_convergence=None, tau=None):
    """
    Simulator. Takes set of `params` and simulates the system, returning summary statistics.
    Params: J_eff, B_eff, tau, threshold distance, noise
    """
    t = np.arange(0, nFrame, 1)/fps
    stimulus = sawtooth(2 * np.pi * abs(freq)/2 * t/26, 0.5)*2*np.sign(freq)
    j_eff, b_par, th, sigma, ndt = params  # add ndt
    lower_bound, upper_bound = np.array([-1, 1])*th + 0.5
    tau = 1 if tau is None else tau
    dt = 1/fps
    b_eff = stimulus*b_par
    noise = np.random.randn(nFrame)*sigma*np.sqrt(dt/tau)
    x = np.zeros(nFrame)
    x1 = lower_bound if np.sign(freq) == 1 else upper_bound
    # x1 = (np.sign(b_eff[0])+1)/2
    if ini_cond_convergence is None:
        x1 = x1
    else:
        for i in range(ini_cond_convergence):
            x1 = sigmoid(2 * (j_eff * (2 * x1 - 1) + b_eff[0]))  # convergence
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




def lmm_hysteresis_pshuffle(freq=2, plot_summary=False,
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


def cartoon_hysteresis_responses(data_folder=DATA_FOLDER,
                                 sv_folder=SV_FOLDER,
                                 ntraining=8, simulated_subject='s_1',
                                 fps=60, idx_trial=1, ntrials=72, n=4,
                                 nfreq=1, plot_response=False, fps_sim=200):
    nFrame = fps*26
    nFrame_sim = fps_sim*26
    df = load_data(data_folder, n_participants='all', preprocess_data=True)
    df = df.loc[df.trial_index > ntraining]
    df_subject = df.loc[df.subject == simulated_subject]
    trial_index = df_subject.trial_index.unique()
    figsize = (7.5, 8) if nfreq == 2 else (6, 8)
    fig_sim, ax_sim = plt.subplots(ncols=nfreq, nrows=3, figsize=figsize, sharex=True)
    fig_dat, ax_dat = plt.subplots(ncols=nfreq, nrows=3, figsize=figsize, sharex=True)
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
                            [0.8, 1.2, 0.16, 0.4], transform=ax_sim[i_p+1, 0].transAxes)
                sub_ax2.set_ylabel(r'$V(q)$')
                sub_ax2.set_xlabel(r'$q$')
                sub_ax2.spines['right'].set_visible(False)
                sub_ax2.spines['top'].set_visible(False)
                sub_ax2.set_xticks([0, 0.5, 1]); sub_ax2.set_yticks([])
                sub_ax2.axvline(0.5, color='gray', linestyle=':', alpha=0.6)
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
    
    y_labels_sim = ['Depth cue, s(t)', '',
                    'q(    in front)', '',
                    'q(    in front)', '',
                    ]
    y_labels_dat = ['Depth cue, s(t)', '',
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
    fig_dat.savefig(SV_FOLDER + 'hysteresis_cartoon_evolution_responses_data.png', dpi=200, bbox_inches='tight')
    fig_dat.savefig(SV_FOLDER + 'hysteresis_cartoon_evolution_responses_data.pdf', dpi=200, bbox_inches='tight')


def simulated_subjects(data_folder=DATA_FOLDER, tFrame=26, fps=60,
                       sv_folder=SV_FOLDER, ntraining=8, n_simuls_network=50000,
                       plot=False, simulate=False, use_j0=False,
                       fitted_params_all=None, subjects=['s_1'], ntrials=72, window_conv=1,
                       shift_ndt=False):
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
        pshuffles = np.repeat(pshuffles, ntrials // 72)
        frequencies = np.repeat(frequencies, ntrials // 72)
        if simulate:
            choice_all = np.zeros((ntrials, nFrame))
            for trial in range(ntrials):
                j_eff = (1-pshuffles[trial])*fitted_params[0] + fitted_params[1]*use_j0
                params = fitted_params[1:].copy()
                params[0] = j_eff
                choice, _ = simulator_5_params(params=params, freq=frequencies[trial], nFrame=nFrame,
                                               fps=fps, return_choice=True, ini_cond_convergence=2)
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
        delay_per_subject = np.int32(200*np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
        for i_s, subject in enumerate(subjects):
            df_subject = df.loc[df.subject == subject]
            pshuffles = np.round(df_subject.groupby('trial_index')['pShuffle'].mean().values, 1)
            ini_side = np.round(df_subject.groupby('trial_index')['initial_side'].mean().values, 1)
            frequencies = np.round(df_subject.groupby('trial_index')['freq'].mean().values * ini_side, 2)
            pshuffles = np.repeat(pshuffles, ntrials // 72)
            frequencies = np.repeat(frequencies, ntrials // 72)
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
                    trial_descending_subjects_4[[i_trial, i_trial+ntrials], i_sh] = descending
                    trial_ascending_subjects_4[[i_trial, i_trial+ntrials], i_sh] = ascending
                else:
                    trial_descending_subjects_2[i_trial, i_sh] = descending
                    trial_ascending_subjects_2[i_trial, i_sh] = ascending
            if shift_ndt:
                ascending_shuf_2 = []
                descending_shuf_2 = []
                ascending_shuf_4 = []
                descending_shuf_4 = []
                for i_shuf in range(3):
                    responses_asc_desc = np.concatenate((trial_ascending_subjects_2[:, i_shuf],
                                                         trial_descending_subjects_2[:, i_shuf]),
                                                        axis=1)
                    resp_rolled = np.roll(responses_asc_desc, delay_per_subject, axis=1)
                    ascending = np.nanmean(resp_rolled[:, :nFrame//2], axis=0)
                    descending = np.nanmean(resp_rolled[:, nFrame//2:], axis=0)
                    ascending_shuf_2.append(ascending)
                    descending_shuf_2.append(descending)
                    responses_asc_desc = np.concatenate((trial_ascending_subjects_4[:, i_shuf],
                                                         trial_descending_subjects_4[:, i_shuf]), axis=1)
                    resp_rolled = np.roll(responses_asc_desc, delay_per_subject, axis=1)
                    ascending = np.nanmean(resp_rolled[:, :nFrame//4], axis=0)
                    descending = np.nanmean(resp_rolled[:, nFrame//4:], axis=0)
                    ascending_shuf_4.append(ascending)
                    descending_shuf_4.append(descending)
                ascending_subjects_2[:, i_s] = np.vstack(ascending_shuf_2)
                descending_subjects_2[:, i_s] = np.vstack(descending_shuf_2)
                ascending_subjects_4[:, i_s] = np.vstack(ascending_shuf_4)
                descending_subjects_4[:, i_s] = np.vstack(descending_shuf_4)
            else:
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
        ax_01.set_ylim(np.min(np.mean(hyst_width_2, axis=1))-0.25, np.max(np.mean(hyst_width_2, axis=1))+0.6)
        sns.barplot(hyst_width_4.T, palette=colormap, ax=ax_11, errorbar='se')
        ax_11.set_ylim(np.min(np.mean(hyst_width_2, axis=1))-0.25, np.max(np.mean(hyst_width_2, axis=1))+0.6)
        heights = np.nanmean(hyst_width_2.T, axis=0)
        bars = np.arange(3)
        pv_sh012 = scipy.stats.ttest_rel(hyst_width_2[0], hyst_width_2[1]).pvalue
        pv_sh022 = scipy.stats.ttest_rel(hyst_width_2[0], hyst_width_2[2]).pvalue
        pv_sh122 = scipy.stats.ttest_rel(hyst_width_2[1], hyst_width_2[2]).pvalue
        barplot_annotate_brackets(0, 1, pv_sh012, bars, heights, yerr=None, dh=.16, barh=.05, fs=10,
                                  maxasterix=3, ax=ax_01)
        barplot_annotate_brackets(0, 2, pv_sh022, bars, heights, yerr=None, dh=.39, barh=.05, fs=10,
                                  maxasterix=3, ax=ax_01)
        barplot_annotate_brackets(1, 2, pv_sh122, bars, heights, yerr=None, dh=.2, barh=.05, fs=10,
                                  maxasterix=3, ax=ax_01)
        heights = np.nanmean(hyst_width_4.T, axis=0)
        bars = np.arange(3)
        pv_sh012 = scipy.stats.ttest_ind(hyst_width_4[0], hyst_width_4[1]).pvalue
        pv_sh022 = scipy.stats.ttest_ind(hyst_width_4[0], hyst_width_4[2]).pvalue
        pv_sh122 = scipy.stats.ttest_ind(hyst_width_4[1], hyst_width_4[2]).pvalue
        barplot_annotate_brackets(0, 1, pv_sh012, bars, heights, yerr=None, dh=.16, barh=.05, fs=10,
                                  maxasterix=3, ax=ax_11)
        barplot_annotate_brackets(0, 2, pv_sh022, bars, heights, yerr=None, dh=.39, barh=.05, fs=10,
                                  maxasterix=3, ax=ax_11)
        barplot_annotate_brackets(1, 2, pv_sh122, bars, heights, yerr=None, dh=.2, barh=.05, fs=10,
                                  maxasterix=3, ax=ax_11)
        for a in [ax_01, ax_11]:
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            a.set_xlabel('p(shuffle)', fontsize=11); a.set_xticks([])
            a.set_ylabel('Hysteresis', fontsize=11); a.set_yticks([])

        f2.tight_layout()
        f2.savefig(SV_FOLDER + 'simulated_hysteresis_average.png', dpi=400, bbox_inches='tight')
        f2.savefig(SV_FOLDER + 'simulated_hysteresis_average.pdf', dpi=400, bbox_inches='tight')
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
        fig3.savefig(SV_FOLDER + 'simulated_hysteresis_f4_vs_f2.pdf', dpi=400, bbox_inches='tight')
        
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(10., 7))
        ax = ax.flatten()
        for i in range(3):
            corr_2 = pearsonr(hyst_width_2_data[i], hyst_width_2[i])
            corr_4 = pearsonr(hyst_width_4_data[i], hyst_width_4[i])
            minmax = [np.min([hyst_width_2_data[i], hyst_width_2[i]])-0.5,
                       np.max([hyst_width_2_data[i], hyst_width_2[i]])+0.5]
            ax[i].set_ylim(minmax)
            ax[i].set_xlim(minmax)
            ax[i].plot(minmax, minmax, color='k', linestyle='--', alpha=0.4, linewidth=4)
            minmax = [np.min([hyst_width_4_data[i], hyst_width_4[i]])-0.5,
                       np.max([hyst_width_4_data[i], hyst_width_4[i]])+0.5]
            ax[i+3].set_ylim(minmax)
            ax[i+3].set_xlim(minmax)
            ax[i+3].plot(minmax, minmax, color='k', linestyle='--', alpha=0.4, linewidth=4)
            ax[i].text(0.2, 2.8, f'r={round(corr_2.statistic, 2)} \np={corr_2.pvalue: .1e}')
            ax[i+3].text(0.2, 2.8, f'r={round(corr_4.statistic, 2)} \np={corr_4.pvalue: .1e}')
            ax[i+3].set_xlabel('Hysteresis data')
            ax[i].set_title('Freq. = 2', fontsize=12); ax[i+3].set_title('Freq. = 4', fontsize=12)
            ax[i].plot(hyst_width_2_data[i], hyst_width_2[i],
                       marker='o', color=colormap[i], linestyle='')
            ax[i].axhline(np.mean(hyst_width_2[i]), color='gray', alpha=0.5, linestyle='--')
            ax[i].axvline(np.mean(hyst_width_2_data[i]), color='gray', alpha=0.5, linestyle='--')
            ax[i+3].plot(hyst_width_4_data[i],
                         hyst_width_4[i], marker='o', color=colormap[i], linestyle='')
            ax[i+3].axhline(np.mean(hyst_width_4[i]), color='gray', alpha=0.5, linestyle='--')
            ax[i+3].axvline(np.mean(hyst_width_4_data[i]), color='gray', alpha=0.5, linestyle='--')
        ax[0].set_ylabel('Hysteresis simulations')
        ax[3].set_ylabel('Hysteresis simulations')
        for i_a, a in enumerate(ax):
            a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
            a.set_ylim(0, 3.5); a.set_xlim(0, 3.5)
            a.set_yticks([0, 1, 2, 3]); a.set_xticks([0, 1, 2, 3])
        fig.tight_layout()


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


def plot_simulate_hysteresis_subject(data_folder=DATA_FOLDER, subject_name=None,
                                     ntraining=8, n=4, window_conv=None, fps=60,
                                     ntrials=72, shift_ndt=False):
    np.random.seed(50)  # 24, 42, 13, 1234, 11, **50**, 51, 100,   10with1000
    ndt = np.abs(np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    print(len(pars), ' fitted subjects')
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = [[n*params[0], n*params[1], params[2], params[4], params[3], ndt] for params in fitted_params_all]   # np.random.normal(ndt, 0.05)
    # print('J1, J0, B1, Sigma, Threshold')
    simulated_subjects(data_folder=DATA_FOLDER, tFrame=26, fps=fps,
                       sv_folder=SV_FOLDER, ntraining=ntraining, ntrials=ntrials,
                       plot=True, simulate=False, use_j0=True, subjects=None,
                       fitted_params_all=fitted_params_all, window_conv=window_conv,
                       shift_ndt=shift_ndt)


def simulate_noise_subjects(df, data_folder=DATA_FOLDER, n=4, nFrame=1546, fps=60,
                            load_simulations=True, sigma_predefined=None):
    np.random.seed(50)  # 25, 50 (3 perc)
    ratio = int(nFrame/1546)
    nFrame = nFrame-ratio+1
    ndt = np.abs(np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    print(len(pars), ' fitted subjects')
    label = f'_sigma_{sigma_predefined}_' if sigma_predefined is not None else ''
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = [[n*params[0], n*params[1], params[2], params[4], params[3], ndt] for params in fitted_params_all]
    subjects = df.subject.unique()[:len(pars)]
    time_interp = np.arange(0, nFrame, 1)/fps
    time_frames = len(time_interp)
    responses_all = np.zeros((len(subjects), 36, time_frames))
    x_all = np.zeros((len(subjects), 36, time_frames))
    internal_noise_all = np.zeros((len(subjects), 36, time_frames))
    stim_subject = np.zeros((len(subjects), 36, time_frames))
    pshuffles_all = np.zeros((len(subjects), 36))
    time = np.arange(0, 1546, 1)/60
    if load_simulations:
        stim_subject = np.load(SV_FOLDER + f'stim_subject_noise{label}.npy')
        responses_all = np.load(SV_FOLDER + f'responses_simulated_noise{label}.npy')
        pshuffles_all = np.load(SV_FOLDER + f'stim_subject_pshuffles{label}.npy')
    else:
        for i_s, subject in enumerate(subjects):
            print('Simulating noise trials subject ', subject)
            fitted_params_subject = fitted_params_all[i_s]
            df_sub = df.loc[df.subject == subject]
            pshuffles = df_sub.groupby("trial_index").pShuffle.mean().values
            trial_indices = df_sub.trial_index.unique()
            choices_subject = np.zeros((len(trial_indices), time_frames))
            x_subject = np.zeros((len(trial_indices), time_frames))
            internal_noise_subject = np.random.randn(len(trial_indices), nFrame)
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
                if sigma_predefined is not None:
                    sigma = sigma_predefined
                lower_bound, upper_bound = np.array([-th, th]) + 0.5
                dt = 1/fps; tau=1
                # noise_sub = np.random.randn()*sigma*np.sqrt(dt)*0.1
                b_eff = stimulus*b_par
                noise = internal_noise_subject[i_trial]*sigma*np.sqrt(dt/tau)
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
                    # noise_sub = noise_sub + np.random.randn()*np.sqrt(dt/tau)*sigma*0.1  # ou process
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
                x_subject[i_trial, :] = x
                stim_subject[i_s, i_trial, :] = stimulus    
            responses_all[i_s, :, :] = choices_subject
            pshuffles_all[i_s] = pshuffles
            x_all[i_s, :, :] = x_subject
            internal_noise_all[i_s, :, :] = internal_noise_subject
        np.save(SV_FOLDER + f'stim_subject_noise{label}.npy', stim_subject)
        np.save(SV_FOLDER + f'responses_simulated_noise{label}.npy', responses_all)
        np.save(SV_FOLDER + f'x_simulated_noise{label}.npy', x_all)
        np.save(SV_FOLDER + f'internal_noise_simulated_noise{label}.npy', internal_noise_all)
        np.save(SV_FOLDER + f'stim_subject_pshuffles{label}.npy', pshuffles_all)
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
    fig2.savefig(SV_FOLDER + 'simulated_nosie_kernel_all.pdf', dpi=200, bbox_inches='tight')


def drive(q, j, b):
    return sigmoid(2*j*(2*q-1)+2*b)-q


def f_prime(q, j, b=0.0):
    """Derivative of the drift term."""
    s = sigmoid(2*j*(2*q - 1) + 2*b)
    return 4* j * s * (1 - s) - 1


def f_double_prime(x, J, B):
    sig = 1 / (1 + np.exp(-2*J*(2*x - 1) - 2*B))
    return 16*(J**2)*sig*(1 - sig)*(1 - 2*sig)


def instanton_rhs(t, y, J, B, D):
    x, dx = y
    fx = drive(x, J, B)
    fxp = f_prime(x, J, B)
    fxx = f_double_prime(x, J, B)
    ddx = fx * fxp + D * fxx
    return np.vstack((dx, ddx))


def bc(y0, yT, x0, xT):
    return np.array([y0[0] - x0, yT[0] - xT])


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


def optimal_escape_eta_with_all_functional(j=1, theta=0.):
    B = 0.0
    D = 0.01
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

        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.plot(time_vals, x_opt, label="x(t)")
        plt.title("Optimal path")
        plt.xlabel("Time, t"); plt.ylabel("x")
        plt.legend()

        plt.subplot(122)
        plt.plot(time_vals, eta_opt, label="η(t)")
        plt.title("Optimal noise")
        plt.xlabel("Time, t"); plt.ylabel("η(t)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("BVP failed:", sol.message)


def optimal_eta_time(J, theta, T=1.0, B=0.0, n_points=400):
    """
    Compute the optimal noise eta*(x) for the nonlinear logistic system.
    Parameters
    ----------
    x : array-like
        Positions to evaluate eta*(x).
    J : float
        Coupling parameter (controls bistability).
    theta : float
        Offset from threshold (target endpoint x_T = 0.5 + theta).
    T : float
        Total transition duration (arbitrary scaling, defines C).
    B : float
        External bias term (default 0).
    Returns
    -------
    eta_star : array
        Optimal noise at each x.
    C : float
        Corresponding constant of motion (depends on J, theta, T).
    """
    if J > 1:
        x0 = 0.05  # initial position (left attractor)
    else:
        x0 = 0.5-theta
    xT = 0.5 + theta
    x0, xT = 0.0, 0.5 + theta  # left attractor to right threshold
    x_grid = np.linspace(x0, xT, n_points)
    # Equation for the travel time integral
    def travel_time(C):
        integrand = lambda x: 1.0 / np.sqrt(drive(x, J, B)**2 + C)
        val, _ = quad(integrand, x0, xT, limit=200)
        return val - T  # we want this to equal zero

    # Find C numerically so that the travel time equals T
    sol = root_scalar(travel_time, bracket=[1e-10, 10.0], method='brentq')
    C = sol.root

    # Compute eta*(x)
    # compute x(t)
    fx = drive(x_grid, J, B)
    dxdt = np.sqrt(fx**2 + C)
    dt_dx = 1.0 / dxdt
    t_grid = cumulative_trapezoid(dt_dx, x_grid, initial=0)
    t_grid *= T / t_grid[-1]  # normalize to total duration T
    # 5. compute eta*(t)
    eta_t = dxdt - fx
    return t_grid, x_grid, eta_t, C


def plot_eta():
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    linestyles = ['solid', '--']
    for i_j, J in enumerate([1.5, 2, 2.5]):
        for i_th, theta in enumerate([0.05, 0.1]):
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
    labels = ['Peak', 'Latency', r'AUC (normalized $eta(t)$)']
    for i_a, a in enumerate(ax):
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
        a.plot(a_list, variables[i_a], color='k', linewidth=4, label=r'Max($\eta(t)$)');  a.set_ylabel(labels[i_a])
        a.set_xlabel('a')
    ax[1].axhline(-0.5, color='r', alpha=0.3, linestyle='--', linewidth=3)
    ax[0].plot(a_list, peak_analytical, color='gray', alpha=0.5, linestyle='--', linewidth=4,
               label='Analytical')
    ax[0].legend(frameon=False)
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
    ax[3].set_xlabel('Time before switch (s)')
    fig.tight_layout()


def plot_average_x_noise_trials(data_folder=DATA_FOLDER,
                                tFrame=26, fps=60,
                                steps_back=120, steps_front=20, avoid_first=True,
                                n=4, load_simulations=True, normalize=False,
                                sigma=None, pshuf_only=None):
    title = r'$\sigma = 0$' if sigma is not None else r'$\sigma \neq 0$'
    nFrame = 1546
    ndt = np.abs(np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    label = f'_sigma_{sigma}_' if sigma is not None else ''
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
    potential_value_all_subjects = np.empty((len(subs), steps_back+steps_front))
    potential_value_all_subjects[:] = np.nan
    for i_sub, subject in enumerate(subs):
        df_sub = df.loc[df.subject == subject]
        trial_index = df_sub.trial_index.unique()
        x_vals_aligned_all_trials = np.empty((1, steps_back+steps_front))
        x_vals_aligned_all_trials[:] = np.nan
        stim_vals_aligned_all_trials = np.empty((1, steps_back+steps_front))
        stim_vals_aligned_all_trials[:] = np.nan
        internal_noise_vals_aligned_all_trials = np.empty((1, steps_back+steps_front))
        internal_noise_vals_aligned_all_trials[:] = np.nan
        potential_vals_aligned_all_trials = np.empty((1, steps_back+steps_front))
        potential_vals_aligned_all_trials[:] = np.nan
        for i_trial, trial in enumerate(trial_index):
            if pshuf_only is not None:
                if pshuffles[i_sub, i_trial] != pshuf_only:
                    continue
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
            potential_vals_aligned = np.empty((len(idx_1)+len(idx_0), steps_back+steps_front))
            potential_vals_aligned[:] = np.nan
            parameters_sub = fitted_params_all[i_sub]
            j_eff = parameters_sub[0]*(1-pshuffles[i_sub, i_trial]) + parameters_sub[1]
            b1 = parameters_sub[2]
            for i, idx in enumerate(idx_1):
                x_vals_aligned[i, :] = values_x[idx - steps_back:idx+steps_front]
                stim_vals_aligned[i, :] = chi[idx - steps_back:idx+steps_front]
                internal_noise_vals_aligned[i, :] = internal_noise[idx - steps_back:idx+steps_front]
                potential = potential_mf(x_vals_aligned[i, :], j_eff,
                                         bias=b1*stim_vals_aligned[i, :], n=1)
                # potential_vals_aligned[i, :] =(potential-np.nanmean(potential)) / (np.max(np.abs(potential))-np.min(np.abs(potential)))
                potential_vals_aligned[i, :] = drive(x_vals_aligned[i, :], j_eff,
                                                     b1*stim_vals_aligned[i, :])
            for i, idx in enumerate(idx_0):
                x_vals_aligned[i+len(idx_1), :] =\
                    1-values_x[idx - steps_back:idx+steps_front]
                stim_vals_aligned[i, :] = chi[idx - steps_back:idx+steps_front]*-1
                internal_noise_vals_aligned[i, :] = internal_noise[idx - steps_back:idx+steps_front]*-1
                # potential = potential_mf(x_vals_aligned[i, :], j_eff,
                #                          bias=b1*stim_vals_aligned[i, :], n=1)
                # potential_vals_aligned[i, :] = (potential-np.nanmean(potential)) / (np.max(np.abs(potential))-np.min(np.abs(potential)))
                potential_vals_aligned[i, :] = drive(x_vals_aligned[i, :], j_eff,
                                                     b1*stim_vals_aligned[i, :])
            x_vals_aligned_all_trials = np.row_stack((x_vals_aligned_all_trials, x_vals_aligned))
            potential_vals_aligned_all_trials = np.row_stack((potential_vals_aligned_all_trials, potential_vals_aligned))
            stim_vals_aligned_all_trials = np.row_stack((stim_vals_aligned_all_trials, stim_vals_aligned))
            internal_noise_vals_aligned_all_trials =\
                np.row_stack((internal_noise_vals_aligned_all_trials, internal_noise_vals_aligned))
            
        x_vals_aligned_all_trials = x_vals_aligned_all_trials[1:]
        x_values_all_subjects[i_sub] = np.nanmean(x_vals_aligned_all_trials, axis=0)
        stim_vals_aligned_all_trials = stim_vals_aligned_all_trials[1:]
        stim_values_all_subjects[i_sub] = np.nanmean(stim_vals_aligned_all_trials, axis=0)
        internal_noise_vals_aligned_all_trials = internal_noise_vals_aligned_all_trials[1:]
        internal_noise_values_all_subjects[i_sub] = np.nanmean(internal_noise_vals_aligned_all_trials, axis=0)
        potential_vals_aligned_all_trials = potential_vals_aligned_all_trials[1:]
        potential_value_all_subjects[i_sub] = np.nanmean(potential_vals_aligned_all_trials, axis=0)
    theta = [pars[3] for pars in fitted_params_all]
    median_theta = 0.5 + np.nanmean(theta)*np.array([-1, 1])
    diff_theta = 2*np.array(theta)
    fig, ax = plt.subplots(ncols=2, nrows=4, figsize=(10, 11))
    fig.suptitle(title, fontsize=16)
    ax = ax.flatten()
    variables = [x_values_all_subjects, stim_values_all_subjects, internal_noise_values_all_subjects,
                 potential_value_all_subjects]
    labels = ['App. posterior q']*2 + ['Stimulus noise B(t)']*2 + ['Internal noise']*2 + ['Drive']*2
    var = 0
    [ax[1].axhline(val, color='r', linestyle='--', alpha=0.2, linewidth=2, zorder=1) for val in median_theta]
    ax[1].axhline(0.5, color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
    # ylims = [[0.25, 0.75], [-0.2, 0.5], [-0.5, 1.5]]
    
    func_x = lambda t, k1:  -1 / np.sqrt(1 + k1*np.exp(-2 * (t+ndt)))
    
    if normalize:
        ax[0].axhline(0., color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
    else:
        ax[0].axhline(0.5, color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
    for i_a, a in enumerate(ax):
        # a.set_ylim(ylims[var][0], ylims[var][1])
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
        if i_a >= 2:
            a.axhline(0., color='k', linestyle='--', alpha=0.4, linewidth=3, zorder=1)
        a.set_xlabel('Time before switch (s)'); a.set_ylabel(labels[i_a])
        if i_a % 2 == 0:
            for i_sub, subject in enumerate(subs):
                x_plot = np.arange(-steps_back, steps_front, 1)/fps
                if i_a == 0 and normalize:
                    minvar = np.min(variables[var][i_sub])
                    maxvar = np.max(variables[var][i_sub])
                    y_plot = (variables[var][i_sub]-0.5)/(maxvar-minvar)
                else:
                    y_plot = variables[var][i_sub]
                a.plot(x_plot, y_plot, color='firebrick',
                       linewidth=2, alpha=0.1)
            continue
        y_plot = np.nanmean(variables[var], axis=0)
        # if var == 1:
        #     x_plot_for_eta = np.arange(-steps_back*2, steps_front*2, 1)/60
        #     eta = optimal_escape_eta(1.2, x_plot_for_eta+0.5)
        #     ax2 = a.twinx()
        #     ax2.plot(x_plot_for_eta/10, eta, color='darkgreen', linestyle='--', linewidth=3)
        #     ax2.spines['top'].set_visible(False)
        y_err = np.nanstd(variables[var], axis=0)/np.sqrt(len(subs))
        a.plot(x_plot, y_plot, color='k', linewidth=4)
        a.fill_between(x_plot, y_plot-y_err, y_plot+y_err, color='k', alpha=0.2)
        var += 1
    fig.tight_layout()
    fig.savefig(SV_FOLDER + f'simulated_variables_noise_trials{label}.png', dpi=200, bbox_inches='tight')
    fig.savefig(SV_FOLDER + f'simulated_variables_noise_trials{label}.pdf', dpi=200, bbox_inches='tight')


def plot_optimal_eta_b_vs_0(ntrials=10, j=1):
    time = np.arange(0, 25, 1e-3)
    dt = np.diff(time)[0]
    noisyframes = 15 // dt // 60
    nFrame = len(time)
    time_interp = np.arange(0, nFrame+noisyframes, noisyframes)*dt
    time = np.arange(0, nFrame, 1)*dt
    noise_exp = np.random.randn(len(time_interp), ntrials)
    noise_signal = np.array([scipy.interpolate.interp1d(time_interp, noise_exp[:, trial])(time) for trial in range(ntrials)]).T
    eta_0 = optimal_escape_eta(2, time+1, stim=None)
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
    fig, ax = plt.subplots(1, figsize=(5.5, 4))
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
    ax.set_xlabel('Time before switch (s)')
    ax.set_ylabel(label); ax.legend(title='Percentile', frameon=False)
    fig.tight_layout()
    print('Saving images')
    fig.savefig(SV_FOLDER + label_save + 'kernel_noise_bf_switch_predicted_amplitude.png', dpi=200, bbox_inches='tight')
    fig.savefig(SV_FOLDER + label_save + 'kernel_noise_bf_switch_predicted_amplitude.pdf', dpi=200, bbox_inches='tight')
    fig, ax = plt.subplots(ncols=1, figsize=(5.5, 4))
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    if cumsum:
        var_data = np.sum(kernels_data, axis=1)
        var_simul = np.sum(kernels_simul, axis=1)
        ax.set_xlabel('Cumulative sum data'); ax.set_ylabel('Cumulative sum simulation')
    else:
        var_data = amplitude_data
        var_simul = amplitude_simul
        ax.set_xlabel('Amplitude data'); ax.set_ylabel('Amplitude simulation')
    ax.plot(var_data, var_simul, marker='o', linestyle='', color='k')
    r, p = pearsonr(var_data, var_simul)
    ax.annotate(f'r = {r:.3f}\np={p:.0e}', xy=(.04, 0.8), xycoords=ax.transAxes)
    fig.tight_layout()


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
    np.save(DATA_FOLDER + 'simulated_all_kernels_noise_switch_aligned.npy', all_kernels)
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
                         simulated=False, zscore_vars=False):
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
    ax3.set_ylabel(lab_zscore + 'Dominance (s)')
    ax3.set_xlabel(lab_zscore + 'Hysteresis');
    ax3.legend(frameon=False, loc='lower left', bbox_to_anchor=[0.4, -0.02])
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + label + 'dominance_vs_hysteresis_classified.png', dpi=300, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + label + 'dominance_vs_hysteresis_classified.pdf', dpi=300, bbox_inches='tight')
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
                                          fitted_variable='J'):
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    fitted_subs = len(pars)
    b1s = [np.load(par)[2] for par in pars]
    sigmas = np.array([np.load(par)[3] for par in pars])
    thetas = [np.load(par)[4] for par in pars]
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
    var = np.load(DATA_FOLDER + file)
    mean_duration_per_sub = np.nanmean(var, axis=0)[:fitted_subs]
    rho, pval = pearsonr(variables[fitted_variable], mean_duration_per_sub)
    fig, ax = plt.subplots(1, figsize=(5, 4))
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.plot(variables[fitted_variable], mean_duration_per_sub, marker='o', linestyle='', color='k')
    ax.set_title('r = ' + str(round(rho, 3)) + '\np = ' + str(round(pval, 5)))
    ax.set_xlabel(variable)
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
        dom_bis = np.nanmean(mean_dominance_shuffle[:, i][jeffs[:, i] >= 1])
        dom_mono = np.nanmean(mean_dominance_shuffle[:, i][jeffs[:, i] < 1])
        hyst_bis_2 = np.nanmean(hyst_2[:, i][jeffs[:, i] >= 1])
        hyst_mono_2 = np.nanmean(hyst_2[:, i][jeffs[:, i] < 1])
        hyst_bis_4 = np.nanmean(hyst_4[:, i][jeffs[:, i] >= 1])
        hyst_mono_4 = np.nanmean(hyst_4[:, i][jeffs[:, i] < 1])
        if not np.isnan(dom_bis):
            bistable_stim_dominance.append(dom_bis)
            bistable_stim_hyst_2.append(hyst_bis_2)
            bistable_stim_hyst_4.append(hyst_bis_4)
        if not np.isnan(dom_mono):
            monostable_stim_dominance.append(dom_mono)
            monostable_stim_hyst_4.append(hyst_mono_4)
            monostable_stim_hyst_2.append(hyst_mono_2)
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
        coef_bis = np.nanmean(bimodal_coef[:, i_sub][j_eff_exp1 >= 1])
        coef_mono = np.nanmean(bimodal_coef[:, i_sub][j_eff_exp1 < 1])
        if not np.isnan(coef_mono):
            sarle_coef_mono.append(coef_mono)
        if not np.isnan(coef_bis):
            sarle_coef_bis.append(coef_bis)
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
    sns.barplot(variables_coef, palette=colormap, errorbar='se', ax=ax[0], estimator=estimator)
    sns.barplot(variables_hyst_2, palette=colormap, errorbar='se', ax=ax[1], estimator=estimator)
    sns.barplot(variables_hyst_4, palette=colormap, errorbar='se', ax=ax[2], estimator=estimator)
    sns.barplot(variables_dominance, palette=colormap, errorbar='se', ax=ax[3], estimator=estimator)
    variables = [variables_coef, variables_hyst_2, variables_hyst_4, variables_dominance]
    if estimator == 'mean':
        dhs = [0.63, 0.7, 0.64, 0.64]
    if estimator == 'median':
        dhs = [0.63, 0.7, 0.72, 0.64]
    for i_a, a in enumerate(ax):
        if estimator == 'median':
            heights = np.nanmedian(variables[i_a], axis=0)
        else:
            heights = np.nanmean(variables[i_a], axis=0)
        bars = np.arange(2)
        pv_sh01 = scipy.stats.mannwhitneyu(variables[i_a][0], variables[i_a][1]).pvalue
        barplot_annotate_brackets(0, 1, pv_sh01, bars, heights, yerr=None, dh=dhs[i_a], barh=.01, fs=10,
                                  maxasterix=3, ax=a)
    ax[0].axhline(5/9, color='gray', linestyle='--', linewidth=3)
    if estimator == 'mean':
        ax[0].set_ylim(0.45, 0.72);  ax[1].set_ylim(0.85, 2.);
        ax[2].set_ylim(1.6, 2.35); ax[3].set_ylim(5.5, 10.4)
    if estimator == 'median':
        ax[0].set_ylim(0.45, 0.72);  ax[1].set_ylim(0.85, 2.);
        ax[2].set_ylim(1.6, 2.85); ax[3].set_ylim(5.5, 10.4)
    # ax[0].set_ylabel("Sarle's bimodality coefficient")
    ax[0].set_ylabel("Bimodality coefficient")
    ax[3].set_ylabel("Dominance duration (s)")
    ax[0].set_yticks([0.45, 0.5, 5/9, 0.6, 0.65, 0.7], ['0.45', '0.5', '5/9', '0.6', '0.65', '0.7'])
    ax[1].set_ylabel('Hysteresis f=2'); ax[2].set_ylabel('Hysteresis f=4')
    handles = [mpatches.Patch(color=colormap[0], label='Monostable'),
               mpatches.Patch(color=colormap[1], label='Bistable')]
    for a in ax:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
        a.set_xticks([])
    ax[0].legend(handles=handles, loc='upper center', frameon=False,
                 bbox_to_anchor=[0.6, 1.1])
    if saveflag:
        fig.tight_layout()
        fig.savefig(DATA_FOLDER + 'comparison_between_experiments_bistable_regime.png', dpi=400)
        fig.savefig(DATA_FOLDER + 'comparison_between_experiments_bistable_regime.pdf', dpi=400)


def comparison_between_experiments(estimator='median', data_only=True,
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
    height = 4 if data_only else 7
    if ax is None:
        fig, ax = plt.subplots(ncols=4, nrows=nrows, figsize=(11.2, height))
        ax = ax.flatten()
        saveflag = True
    else:
        saveflag = False
    for a in ax:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    sns.barplot(bimodal_coef.T, palette=colormap, errorbar='se', ax=ax[0], estimator=estimator)
    sns.barplot(hyst_width_2.T, palette=colormap, errorbar='se', ax=ax[1], estimator=estimator)
    sns.barplot(hyst_width_4.T, palette=colormap, errorbar='se', ax=ax[2], estimator=estimator)
    sns.barplot(mean_number_switchs_coupling.T, palette=colormap, errorbar='se', ax=ax[3], estimator=estimator)
    if not data_only:
        sns.barplot(bimodal_coef_simul.T, palette=colormap, errorbar='se', ax=ax[4], estimator=estimator)
        sns.barplot(hyst_width_2_simul.T, palette=colormap, errorbar='se', ax=ax[5], estimator=estimator)
        sns.barplot(hyst_width_4_simul.T, palette=colormap, errorbar='se', ax=ax[6], estimator=estimator)
        sns.barplot(mean_number_switchs_coupling_simul.T, palette=colormap, errorbar='se', ax=ax[7], estimator=estimator)
    # sns.swarmplot(bimodal_coef.T, color='k', ax=ax[0])
    # sns.swarmplot(hyst_width_2.T, color='k', ax=ax[1])
    # sns.swarmplot(mean_number_switchs_coupling.T, color='k', ax=ax[2])
    # if estimator == 'mean':
    #     ax[0].plot(np.nanmedian(bimodal_coef, axis=1), color='firebrick', linewidth=4)
    #     ax[1].plot(np.nanmedian(hyst_width_2, axis=1), color='firebrick', linewidth=4)
    #     ax[2].plot(np.nanmedian(hyst_width_4, axis=1), color='firebrick', linewidth=4)
    #     ax[3].plot(np.nanmedian(mean_number_switchs_coupling, axis=1), color='firebrick', linewidth=4)
    
    ax[0].axhline(5/9, color='gray', linestyle='--', linewidth=3)
    xlim_ax0 = ax[0].get_xlim() + np.array((0., 0.35))
    ax[0].set_xlim(xlim_ax0)
    if not data_only:
        ax[4].axhline(5/9, color='gray', linestyle='--', linewidth=3)
    ax[0].set_ylim(0.46, 0.71);  ax[1].set_ylim(0.85, 1.68);
    ax[2].set_ylim(1.6, 2.35); ax[3].set_ylim(5.5, 9.9)
    if not data_only:
        ax[4].set_ylim(0.48, 0.64);  ax[5].set_ylim(1.2, 1.62);  ax[7].set_ylim(1, 3.5)
    ax[0].set_ylabel("Bimodality coefficient")
    ax[3].set_ylabel("Dominance duration (s)")
    ax[0].set_yticks([0.5, 5/9, 0.6, 0.65], ['0.5', '5/9', '0.6', '0.65'])
    titles = ['Exp. 1\n', 'Exp. 2\nHysteresis (f=2)', 'Exp. 2\nHysteresis (f=4)', 'Exp. 2\nNoise trials']
    variables = [bimodal_coef, hyst_width_2, hyst_width_4, mean_number_switchs_coupling]
    for i_a, a in enumerate(ax):
        if estimator == 'median':
            heights = np.nanmedian(variables[i_a].T, axis=0)
        else:
            heights = np.nanmean(variables[i_a].T, axis=0)
        bars = np.arange(3)
        pv_sh01 = scipy.stats.ttest_rel(variables[i_a][0], variables[i_a][1]).pvalue
        pv_sh02 = scipy.stats.ttest_rel(variables[i_a][0], variables[i_a][2]).pvalue
        pv_sh12 = scipy.stats.ttest_rel(variables[i_a][1], variables[i_a][2]).pvalue
        barplot_annotate_brackets(0, 1, pv_sh01, bars, heights, yerr=None, dh=.19, barh=.01, fs=10,
                                  maxasterix=3, ax=a)
        barplot_annotate_brackets(0, 2, pv_sh02, bars, heights, yerr=None, dh=.23, barh=.01, fs=10,
                                  maxasterix=3, ax=a)
        barplot_annotate_brackets(1, 2, pv_sh12, bars, heights, yerr=None, dh=.16, barh=.01, fs=10,
                                  maxasterix=3, ax=a)
        # a.set_xticks([0, 1, 2], [1., 0.7, 0.])
        a.set_xticks([])
        # if i_a < 4:
        #     a.set_title(titles[i_a], fontsize=14, pad=15)
    # ax[0].set_xlabel('p(shuffle)');
    # ax[2].set_xlabel('p(shuffle)'); ax[3].set_xlabel('p(shuffle)')
    # ax[1].set_xlabel('p(shuffle)');  
    ax[1].set_ylabel('Hysteresis f=2'); ax[2].set_ylabel('Hysteresis f=4')
    handles = [mpatches.Patch(color=colormap[0], label='1.'),
                mpatches.Patch(color=colormap[1], label='0.7'),
                mpatches.Patch(color=colormap[2], label='0.')]
    ax[2].legend(handles=handles,frameon=False,
                  title='p(shuffle)', loc='upper right')
    ax[0].text(0.92, 0.42, 'Bimodal', rotation=90, fontsize=13, transform=ax[0].transAxes)
    ax[0].text(0.92, 0.035, 'Unimodal', rotation=90, fontsize=13, transform=ax[0].transAxes)
    if saveflag:
        fig.tight_layout()
        fig.savefig(DATA_FOLDER + 'comparison_between_experiments.png', dpi=400)
        fig.savefig(DATA_FOLDER + 'comparison_between_experiments.pdf', dpi=400)


def experiment_comparison_altogether():
    fig, ax = plt.subplots(ncols=4, nrows=3, figsize=(11.2, 9))
    ax = ax.flatten()
    compare_parameters_two_experiments(ax=ax[:4])
    comparison_between_experiments(estimator='median', data_only=True,
                                   ax=ax[4:8], fig=fig)
    comparison_between_experiments_bis_mono(unique_shuffle=[1., 0.7, 0.],
                                            estimator='median', n=4, ax=ax[8:])
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'full_comparison_between_experiments.png', dpi=400)
    fig.savefig(DATA_FOLDER + 'full_comparison_between_experiments.pdf', dpi=400)
    fig.savefig(DATA_FOLDER + 'full_comparison_between_experiments.svg', dpi=400)


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
        fig, ax = plt.subplots(ncols=1, figsize=(5, 4))
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
    ax.set_xlabel('Time before switch(s)')
    ax.set_ylabel('Noise')
    if saveflag:
        ax.legend(frameon=False)
        fig.tight_layout()    
        fig.savefig(DATA_FOLDER + 'kernel_across_subjects.png', dpi=400)
        fig.savefig(DATA_FOLDER + 'kernel_across_subjects.pdf', dpi=400)


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
    ax.set_xlabel('Time before switch(s)')
    if legend:
        ax.legend(title=variable, frameon=False)
    if ax_none_flag:
        ax.set_ylabel('Noise')
        fig.tight_layout()
        fig.savefig(DATA_FOLDER + label_save_fig + f'kernel_across_subjects_different_{variable}.png', dpi=400)
        fig.savefig(DATA_FOLDER + label_save_fig + f'kernel_across_subjects_different_{variable}.pdf', dpi=400)


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
    fig.savefig(SV_FOLDER + label + 'cartoon.pdf', dpi=400, bbox_inches='tight')


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


def compare_likelihoods_models(load=True, loss='Likelihood'):
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
        loss_null_model = bic_null_model
        loss_original_model = bic_with_ndt
    if loss == 'AIC':
        loss_null_model = likelihood_null_model + 2*4
        loss_original_model = likelihood_with_ndt + 2*4
    if loss == 'NLH':
        loss_null_model = likelihood_null_model
        loss_original_model = likelihood_with_ndt
    losses = [loss_null_model, loss_original_model]
    ax5.spines['right'].set_visible(False); ax5.spines['top'].set_visible(False)
    sns.barplot(losses, palette=['mediumpurple', 'burlywood'], ax=ax5)
    pvalue = scipy.stats.ttest_rel(losses[0], losses[1]).pvalue
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


def get_switch_indices(arr):
    """
    Detect switches in a response array (0/1 with NaN).
    Returns two arrays of indices:
      - idx_01: where a 1→2 switch occurred
      - idx_10: where a 2→1 switch occurred
    """
    prev = arr[:-1]
    nxt = arr[1:]
    valid = (~np.isnan(prev)) & (~np.isnan(nxt))
    idx_01 = np.where(valid & (prev == 1) & (nxt == 2))[0] + 1
    idx_10 = np.where(valid & (prev == 2) & (nxt == 1))[0] + 1
    return idx_01, idx_10


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
    fig.savefig(SV_FOLDER + 'simulated_hysteresis_switch_rate.pdf', dpi=400, bbox_inches='tight')


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


def get_response_array(df):
    # Define sampling
    dt = 1/60  # sampling step (seconds)
    eps = dt / 2
    t_min = 0
    t_max = df["keypress_seconds_offset"].max()
    time = np.arange(t_min, t_max + dt, dt)
    
    # Initialize response array with baseline (e.g. 0)
    response_array = np.zeros_like(time, dtype=int)
    
    # Fill in active responses
    for _, row in df.iterrows():
        onset, offset, resp = row["keypress_seconds_onset"], row["keypress_seconds_offset"], row["response"]
        mask = (time >= onset - eps) & (time < offset + eps)
        response_array[mask] = resp
    
    return response_array


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
    ax2[-1].set_xlabel('Time before switch (s)')
    ax2[-1].axvline(0, color='k', alpha=0.3, linestyle='--')
    ax2[-1].spines['right'].set_visible(False)
    ax2[-1].spines['top'].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(SV_FOLDER + 'noise_kernel_different_regimes.png', dpi=200, bbox_inches='tight')
    fig2.savefig(SV_FOLDER + 'noise_kernel_different_regimes.pdf', dpi=200, bbox_inches='tight')


def ridgeplot_all_kernels(data_folder=DATA_FOLDER, steps_back=150, steps_front=10, fps=60,
                          order_by_variable=False, zscore_variables=False):
    x_plot = np.arange(-steps_back, steps_front, 1)/fps
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    kernels_data = np.load(DATA_FOLDER + 'all_kernels_noise_switch_aligned.npy')
    kernels_model = np.load(DATA_FOLDER + 'simulated_all_kernels_noise_switch_aligned.npy')
    
    
    n_subs, n_t = kernels_data.shape
    x_plot = np.arange(-steps_back, steps_front) / fps
    
    # --------------------------------------------------
    # Ridge plot parameters
    # --------------------------------------------------
    offset = 6 if zscore_variables else 1.2
    offset = offset * np.max(np.abs(kernels_data))   # vertical spacing
    
    fig = plt.figure(figsize=(3, 2 * n_subs))
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    j1s = np.array([np.load(par)[0] for par in pars])  # /sigmas
    j0s = np.array([np.load(par)[1] for par in pars])  # /sigmas
    b1s = np.array([np.load(par)[0] for par in pars])  # /sigmas
    thetas = np.array([np.load(par)[1] for par in pars])  # /sigmas
    sigmas = np.array([np.load(par)[0] for par in pars])  # /sigmas
    kernel_ndt = np.load(DATA_FOLDER + 'kernel_latency_average.npy')
    variables = {'J': j0s+j1s, 'B': b1s, 'sigma': sigmas, 'theta': thetas,
                 'ndt': kernel_ndt, 'J1': j1s, 'J0': j0s, 'B/sigma': b1s/sigmas,
                 'theta/sigma': thetas/sigmas}
    if order_by_variable is not None:
        variable = variables[order_by_variable]
        idxs_by_variable = np.argsort(variable)
    for i in range(n_subs):
        if order_by_variable is not None:
            idx = idxs_by_variable[i]
        else:
            idx = i
        y0 = i * offset
        plt.axhline(y0, color='gray', alpha=0.5, linestyle=':')
        # data trace
        y_plot = zscore(kernels_data[idx]) if zscore_variables else kernels_data[idx]
        plt.plot(
            x_plot,
            y_plot + y0,
            color="k",
            lw=4,
            label="data" if i == 0 else None
        )

        y_plot = zscore(kernels_model[idx]) if zscore_variables else kernels_model[idx]
        # model trace
        plt.plot(
            x_plot,
            y_plot + y0,
            color="r",
            lw=4,
            alpha=0.8,
            label="model" if i == 0 else None
        )
        # subject label
        plt.text(x_plot[0] - 0.15*(x_plot[-1]-x_plot[0]),
                 y0,
                 f"S{idx+1}",
                 va="center")
    plt.axvline(0, linestyle='--', color='gray', alpha=0.3)
    plt.xlabel("Time (s)")
    plt.yticks([])
    
    plt.legend(loc="upper right", frameon=False)
    sns.despine(left=True)
    
    plt.tight_layout()
    plt.show()
    label = 'z_scored_' if zscore_variables else ''
    fig.savefig(SV_FOLDER + label + 'kernels_data_vs_model.png', dpi=200, bbox_inches='tight')
    fig.savefig(SV_FOLDER + label + 'kernels_data_vs_model.pdf', dpi=200, bbox_inches='tight')


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


if __name__ == '__main__':
    print('Running hysteresis_analysis.py')
    # experiment_example(nFrame=1560, fps=60, noisyframes=15)
    # plot_noise_variables_vs_fitted_params(n=4, variable='amplitude')
    # ridgeplot_all_kernels(data_folder=DATA_FOLDER, steps_back=150, steps_front=10, fps=60,
    #                       zscore_variables=True, order_by_variable=None)
    # kernel_different_regimes_all_subjects(data_folder=DATA_FOLDER,
    #                                       fps=60, tFrame=26, ntraining=8)
    # plot_regression_weights()
    # compute_switch_rate_since_switch(data_folder=DATA_FOLDER, shuffle_vals=[1., 0.7, 0.], tFrame=26,
    #                                   fps=60, steps_front=350, bin_size=0.3, simulated=False,
    #                                   steps_back=0)
    # compute_switch_rate_since_switch(data_folder=DATA_FOLDER, shuffle_vals=[1., 0.7, 0.], tFrame=26,
    #                                   fps=60, steps_front=350, bin_size=0.3, simulated=True,
    #                                   steps_back=0)
    # optimal_escape_eta_with_all_functional(j=1.2, theta=0.15)
    # get_correlation_consecutive_dominance_durations(data_folder=DATA_FOLDER, fps=60, tFrame=26)
    # simple_optimal_eta_quartic_potential(a=0.5)
    # simple_optimal_eta_quartic_potential(a=1.)
    # simple_optimal_eta_quartic_potential(a=1.5)
    # plot_average_x_noise_trials(data_folder=DATA_FOLDER,
    #                             tFrame=26, fps=60,
    #                             steps_back=200, steps_front=20, avoid_first=True,
    #                             n=4, load_simulations=True, normalize=False, sigma=None,
    #                             pshuf_only=None)
    # plot_average_x_noise_trials(data_folder=DATA_FOLDER,
    #                             tFrame=26, fps=60,
    #                             steps_back=200, steps_front=20, avoid_first=True,
    #                             n=4, load_simulations=True, normalize=False, sigma=0,
    #                             pshuf_only=None)
    compare_likelihoods_models(load=True, loss='AIC')
    compare_likelihoods_models(load=True, loss='BIC')
    compare_likelihoods_models(load=True, loss='NLH')
    # plot_kernel_different_regimes(data_folder=DATA_FOLDER, fps=60, tFrame=26,
    #                               steps_back=120, steps_front=20,
    #                               shuffle_vals=[1, 0.7, 0],
    #                               avoid_first=False, window_conv=1,
    #                               filter_subjects=True, n=4, sub_alone=None,
    #                               ax=None)
    # plot_dominance_durations(data_folder=DATA_FOLDER,
    #                           ntraining=8, freq=2, sem=False)
    # plot_dominance_hyst_pshuffle(freq=4)
    # plot_dominance_hyst_pshuffle(freq=2)
    # plot_dominance_durations(data_folder=DATA_FOLDER,
    #                           ntraining=8, freq=4, sem=False)
    # hyst_vs_dom_bis_mono(unique_shuffle=[1., 0.7, 0.], n=4,
    #                       freq=2, kde=True,
    #                       point_per_subject_x_shuffle=True, simulated=False,
    #                       zscore_vars=False)
    # hyst_vs_dom_bis_mono(unique_shuffle=[1., 0.7, 0.], n=4,
    #                       freq=2, kde=True,
    #                       point_per_subject_x_shuffle=True, simulated=False,
    #                       zscore_vars=True)
    # comparison_between_experiments_bis_mono(unique_shuffle=[1., 0.7, 0.],
    #                                         estimator='mean', n=4)
    # comparison_between_experiments(estimator='mean', data_only=True)
    # compare_parameters_two_experiments()
    # experiment_comparison_altogether()
    # plot_dominance_distros_noise_trials_per_subject(data_folder=DATA_FOLDER, fps=60, tFrame=26,
    #                                                     simulated=False)
    # plot_dominance_distros_noise_trials_per_subject(data_folder=DATA_FOLDER, fps=60, tFrame=26,
    #                                                     simulated=True)
    # plot_dominance_bis_mono(unique_shuffle=[1., 0.7, 0.], n=4)
    # plot_dominance_bis_mono(unique_shuffle=[1., 0.7, 0.], n=4, simulations=True)
    # plot_noise_variables_vs_fitted_params(n=4, variable='freq2')
    # plot_params_distros(ndt=True)
    # plot_simulate_hysteresis_subject(data_folder=DATA_FOLDER, subject_name=None,
    #                                   ntraining=8, window_conv=10, fps=200,
    #                                   ntrials=72, shift_ndt=False)
    # plot_hysteresis_model_data()
    # cartoon_hysteresis_responses(data_folder=DATA_FOLDER,
    #                               sv_folder=SV_FOLDER,
    #                               ntraining=8, simulated_subject='s_36',
    #                               fps=60, idx_trial=2, ntrials=72, nfreq=1,
    #                               plot_response=False)
    # plot_switch_rate_model(data_folder=DATA_FOLDER, sv_folder=SV_FOLDER,
    #                         fps=200, n=4, ntraining=8, tFrame=26,
    #                         window_conv=5, n_bins=50)
    # plot_simulated_subjects_noise_trials(data_folder=DATA_FOLDER,
    #                                       shuffle_vals=[1., 0.7, 0.], ntrials=36,
    #                                       steps_back=150, steps_front=10, avoid_first=True,
    #                                       tFrame=26, window_conv=1,
    #                                       fps=60, ax=None, hysteresis_area=True,
    #                                       normalize_variables=True, ratio=1,
    #                                       load_simulations=False)
    # plot_kernels_predicted_amplitude(steps_back=150, steps_front=10, fps=60,
    #                                   cumsum=False, npercentiles=3, sim_predict_dat=False)
    # plot_kernels_predicted_amplitude(steps_back=150, steps_front=10, fps=60,
    #                                   cumsum=False, npercentiles=3, sim_predict_dat=True)
    # for variable in ['J0', 'J1', 'B1', 'SIGMA', 'THETA']:
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
    #                  ntraining=8, coupling_levels=[0, 0.3, 1],
    #                  window_conv=5, bin_size=0.35, switch_01=False)
    # plot_sequential_effects(data_folder=DATA_FOLDER, ntraining=8)
    # get_rt_distro_and_incorrect_resps(data_folder=DATA_FOLDER,
    #                                   ntraining=8, coupling_levels=[0, 0.3, 1])
    # hysteresis_basic_plot_simulation(coup_vals=np.array((0., 1))*0.32,
    #                                  fps=150, nsubs=1, n=4, nsims=5000,
    #                                  b_list=np.linspace(-0.5, 0.5, 1001), simul=False)
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
    #                           slope_random_effect=False, plot_individual=True)
    # lmm_hysteresis_dominance(freq=2, plot_summary=True,
    #                           slope_random_effect=True, plot_individual=True)
