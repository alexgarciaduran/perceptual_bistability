# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 20:07:39 2025

@author: alexg
"""

# Make the parent folder (all_scripts/) importable so that gibbs_necker,
# mean_field_necker and fitting_pipeline (which live one level up from this
# hysteresis/ folder) can be imported as before.
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


pc_name = 'alex'
if pc_name == 'alex':
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/hysteresis/data/'  # Alex
    SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/hysteresis/parameters/'  # Alex
elif pc_name == 'alex_CRM':
    DATA_FOLDER = 'C:/Users/agarcia/Desktop/phd/necker/data_folder/hysteresis/data/'  # Alex CRM
    SV_FOLDER = 'C:/Users/agarcia/Desktop/phd/necker/data_folder/hysteresis/'  # Alex CRM

COLORMAP = LinearSegmentedColormap.from_list('rg', ['darkgreen', 'gainsboro', 'crimson'], N=128)




def sigmoid_fit_lapse(B, B0, k, gamma, lam):
    return gamma + (1 - gamma - lam) / (1 + np.exp(-k * (B - B0)))


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
                                 flip_responses=False,
                                 frames_no_resp_ini=0):
    """
    From the df extracts responses arrays,
    separating ascending from descending parts.
    """
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
                if switch_times[i+1] < frames_no_resp_ini:
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
    """
    From the df, it collects the responses nicely so that we have
    dictionaries (responses_2 and responses_4) with the response values,
    separated in asc/desc, the initial side. It also returns the stimulus array.
    """
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
                              maxasterix=3, ax=None, raw_p=False):
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

    text = f'p={data:.3e}' if raw_p else stars_pval(data)
    # print(data)

    lx, ly = center[num1]+2e-2, height[num1]
    rx, ry = center[num2]-2e-2, height[num2]

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


def estimate_B50(b_vals, responses):
    """
    Fit a sigmoid to responses and return B50 (stimulus at 50% response)
    clipped to [-2, 2] to avoid edge artifacts.
    """
    # Clip responses to avoid exactly 0 or 1
    responses = np.clip(responses, 0.001, 0.999)
    
    # Bounds for B0 and slope k
    bounds = (
        [-2, 0, 0, 0],    # B0_lower, k_lower, gamma_lower, lambda_lower
        [2, 10, 0.1, 0.1] # B0_upper, k_upper, gamma_upper, lambda_upper
    )

    try:
        popt, _ = curve_fit(sigmoid_fit_lapse, b_vals, responses, bounds=bounds, p0=[0., 1., 0.05, 0.05])
        B50 = np.clip(popt[0], -2., 2.)  # B0 is the 50% crossing
    except RuntimeError:
        # If fit fails, fallback to median
        B50 = np.median(b_vals)

    return B50


def bin_responses(b_vals, responses, bin_width=0.2):
    bins = np.arange(np.min(b_vals), np.max(b_vals)+bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width/2
    binned_resp = np.zeros(len(bin_centers))
    
    for i in range(len(bins)-1):
        idx = (b_vals >= bins[i]) & (b_vals < bins[i+1])
        if np.any(idx):
            binned_resp[i] = np.nanmean(responses[idx])
        else:
            binned_resp[i] = np.nan  # no data in bin
    
    return bin_centers, binned_resp


def get_blist(freq, nFrame, maxval=2):
    if abs(freq) == 2:
        difficulty_time_ref_2 = np.linspace(-maxval, maxval, nFrame//2)
        stimulus = np.concatenate(([difficulty_time_ref_2, -difficulty_time_ref_2]))
    if abs(freq) == 4:
        difficulty_time_ref_4 = np.linspace(-maxval, maxval, nFrame//4)
        stimulus = np.concatenate(([difficulty_time_ref_4, -difficulty_time_ref_4,
                                    difficulty_time_ref_4, -difficulty_time_ref_4]))
    if freq < 0:
        stimulus = -stimulus
    return stimulus


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


def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = scipy.stats.pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.3f}', xy=(.1, 1.), xycoords=ax.transAxes)


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
                       return_choice=False, ini_cond_convergence=None, tau=None,
                       stimulus=None):
    """
    Simulator. Takes set of `params` and simulates the system, returning summary statistics.
    Params: J_eff, B_eff, tau, threshold distance, noise
    """
    time_sim = np.arange(0, nFrame, 1)/fps
    if stimulus is None:
        stimulus = sawtooth(2 * np.pi * abs(freq)/2 * time_sim/26, 0.5)*2*np.sign(freq)
    j_eff, b_par, th, sigma, ndt = params  # add ndt
    lower_bound, upper_bound = np.array([-1, 1])*th + 0.5
    tau = 1 if tau is None else tau
    dt = 1/fps
    b_eff = stimulus*b_par
    noise_raw = np.random.randn(nFrame)
    noise = noise_raw*sigma*np.sqrt(dt/tau)
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
    
        if pending_choice is not None:
            if (pending_choice == 1.0 and x[t] <= lower_bound):
                # crossed back — cancel and reschedule opposite
                pending_choice = -1.0
                pending_time = t + ndt_frames
            elif (pending_choice == -1.0 and x[t] >= upper_bound):
                pending_choice = 1.0
                pending_time = t + ndt_frames
        elif new_choice != prev_choice:
            pending_choice = new_choice
            pending_time = t + ndt_frames
     
        choice[t] = prev_choice

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


def plot_example_pupil(ax,
                       data_folder=DATA_FOLDER,
                       sub='s_36', trial_idx=4,
                       **plot_kws):
    path_save_csv = data_folder + '/aligned_eye_tracker_data/' + sub + '_aligned_Gaze_Data.csv'
    eyetracker_data = pd.read_csv(path_save_csv)
    pupil_trial = eyetracker_data.loc[eyetracker_data.trial_index == trial_idx, 'Pupil_residual'].values
    time = eyetracker_data.loc[eyetracker_data.trial_index == trial_idx, 't_trial'].values
    ax.plot(time, pupil_trial, **plot_kws)
    ax.set_xlim(-0.05, 26.05)


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
    x_all_subject = np.zeros((len(subjects), ntrials, nFrame))
    for i_s, subject in enumerate(subjects):
        fitted_params = fitted_params_all[i_s]
        print('Simulating subject', subject)
        if fitted_params is None:
            fitted_params = np.load(sv_folder + '/pars_5_subject_' + subject + str(n_simuls_network) + label_j0 + '.npy')
        df_subject = df.loc[df.subject == subject]
        pshuffles = df_subject.groupby('trial_index')['pShuffle'].mean().values
        ini_side = df_subject.groupby('trial_index')['initial_side'].mean().values
        frequencies = df_subject.groupby('trial_index')['freq'].mean().values*ini_side
        # pshuffles = np.repeat(pshuffles, ntrials // 72)
        # frequencies = np.repeat(frequencies, ntrials // 72)
        if simulate:
            choice_all = np.zeros((ntrials, nFrame))
            x_all = np.zeros((ntrials, nFrame))
            for trial in range(ntrials):
                j_eff = (1-pshuffles[trial])*fitted_params[0] + fitted_params[1]*use_j0
                params = fitted_params[1:].copy()
                params[0] = j_eff
                choice, x = simulator_5_params(params=params, freq=frequencies[trial], nFrame=nFrame,
                                               fps=fps, return_choice=True, ini_cond_convergence=2)
                choice_all[trial, :] = choice
                x_all[trial, :] = x
            np.save(sv_folder + f'choice_matrix_subject_{subject}.npy', choice_all)
            np.save(sv_folder + f'confidence_matrix_subject_{subject}.npy', x_all)
        else:
            choice_all = np.load(sv_folder + f'choice_matrix_subject_{subject}.npy')
            x_all = np.load(sv_folder + f'confidence_matrix_subject_{subject}.npy')
        choices_all_subject[i_s] = choice_all
        x_all_subject[i_s] = x_all
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
            a.set_xlabel('Depth cue, c(t)')
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            a.axhline(0.5, color='k', linestyle='--', alpha=0.2)
            a.axvline(0., color='k', linestyle='--', alpha=0.2)
            a.set_ylim(-0.025, 1.085)
            a.set_yticks([0, 0.5, 1])
            a.set_xlim(-2.05, 2.05)
            a.set_xticks([-2, 0, 2], [-1, 0, 1])
        ax2[0].set_title('One cycle', fontsize=14)
        ax2[0].legend(title='p(shuffle)', frameon=False,
                      bbox_to_anchor=[-0.02, 1.07], loc='upper left')
        ax2[1].set_title('Two cycles', fontsize=14)
        ax2[0].set_ylabel('P(rightward)')
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
        pv_sh012 = scipy.stats.ttest_rel(hyst_width_4[0], hyst_width_4[1]).pvalue
        pv_sh022 = scipy.stats.ttest_rel(hyst_width_4[0], hyst_width_4[2]).pvalue
        pv_sh122 = scipy.stats.ttest_rel(hyst_width_4[1], hyst_width_4[2]).pvalue
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
        ax[0].set_title('One cycle', fontsize=12); ax[1].set_title('Two cycles', fontsize=12)
        ax[0].plot(hyst_width_2_data[:, :fitted_subs].flatten(),
                   hyst_width_2.flatten(), marker='o', color='k', linestyle='')
        ax[1].plot(hyst_width_4_data[:, :fitted_subs].flatten(), hyst_width_4.flatten(), marker='o', color='k', linestyle='')
        fig.tight_layout()
        fig.savefig(SV_FOLDER + 'simulated_hysteresis_vs_data.png', dpi=400, bbox_inches='tight')
        fig.savefig(SV_FOLDER + 'simulated_hysteresis_vs_data.svg', dpi=400, bbox_inches='tight')
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
        ax[0].set_title('One cycle', fontsize=12); ax[1].set_title('Two cycles', fontsize=12)
        ax[0].plot(delta_2_data, delta_2_sims, marker='o', color='k', linestyle='')
        ax[1].plot(delta_4_data, delta_4_sims, marker='o', color='k', linestyle='')
        for k in range(2):
            means_data = np.nanmean([delta_2_data, delta_4_data][k])
            means_sims = np.nanmean([delta_2_sims, delta_4_sims][k])
            ax[k].axhline(means_sims, color='k', alpha=0.5)
            ax[k].axvline(means_data, color='k', alpha=0.5)
        fig.tight_layout()
        fig.savefig(SV_FOLDER + 'simulated_hysteresis_deltas_vs_data.png', dpi=400, bbox_inches='tight')
        fig.savefig(SV_FOLDER + 'simulated_hysteresis_deltas_vs_data.svg', dpi=400, bbox_inches='tight')

        np.save(DATA_FOLDER + 'hysteresis_width_f2_sims_fitted_params.npy', hyst_width_2)
        np.save(DATA_FOLDER + 'hysteresis_width_f4_sims_fitted_params.npy', hyst_width_4)
        fig3, ax3 = plt.subplots(1, figsize=(4, 3.5))
        ax3.plot([0, 3], [0, 3], color='k', alpha=0.4, linestyle='--', linewidth=4)
        for i_c in range(len(unique_shuffle)):
            ax3.plot(hyst_width_2[i_c], hyst_width_4[i_c],
                      color=colormap[i_c], marker='o', linestyle='')
        # for i_s in range(fitted_subs):
        #     ax3.plot(hyst_width_2[:, i_s], hyst_width_4[:, i_s],
        #               color='k', alpha=0.1)
        for a in [ax3]:
            a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
        ax3.set_ylabel('Hysteresis two cycles')
        ax3.set_xlabel('Hysteresis one cycle')
        fig3.tight_layout()
        fig3.savefig(SV_FOLDER + 'simulated_hysteresis_f4_vs_f2.png', dpi=400, bbox_inches='tight')
        fig3.savefig(SV_FOLDER + 'simulated_hysteresis_f4_vs_f2.svg', dpi=400, bbox_inches='tight')

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
            ax[i].set_title('One cycle', fontsize=12); ax[i+3].set_title('Two cycles', fontsize=12)
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
        
        # alignment window (in frames)
        pre_switch = int(1.5 * fps)   # 500 ms before
        post_switch = int(1.5 * fps)  # 500 ms after
        T = pre_switch + post_switch
        t_switch = np.arange(-pre_switch, post_switch) / fps
        
        # container: [shuffle][subject][event, time]
        aligned_conf = [[[] for _ in subjects] for _ in unique_shuffle]
        
        for i_s, subject in enumerate(subjects):
            choice_all = choices_all_subject[i_s]
            x_all = x_all_subject[i_s]
            fitted_params = fitted_params_all[i_s]
            theta = fitted_params[3]
            df_subject = df.loc[df.subject == subject]
            ini_side = np.round(df_subject.groupby('trial_index')['initial_side'].mean().values, 1)
            frequencies = np.round(df_subject.groupby('trial_index')['freq'].mean().values * ini_side, 2)
            psh = np.round(df_subject.groupby('trial_index')['pShuffle'].mean().values, 1)
            psh = np.repeat(psh, ntrials // 72)
        
            for i_trial in range(ntrials):
                sh_idx = np.where(unique_shuffle == psh[i_trial])[0][0]
                if abs(frequencies[i_trial]) != 4:
                    continue
                # binary choice
                ch = choice_all[i_trial].copy()
                ch[ch == 0] = np.nan
                # ch = (ch + 1) / 2
                conf_trial = (2*x_all[i_trial]-1)*ch
                # absolute confidence (CORRECT)
                # dist_lower = (conf_trial - 0.5-theta)**2
                # dist_upper = (conf_trial - 0.5+theta)**2
                # conf = np.min(np.row_stack([dist_lower, dist_upper]), axis=0)
                conf = conf_trial
        
                switch_idx = np.where(np.abs(np.diff(ch)) > 0)[0] + 1
        
                for t0 in switch_idx:
                    if t0 > 20:
                        if t0 - pre_switch < 0 or t0 + post_switch > nFrame:
                            continue
            
                        aligned_conf[sh_idx][i_s].append(
                            conf[t0 - pre_switch:t0 + post_switch]
                        )
        
        # ------------------------------------------------------------
        # Average: events → subjects → shuffle
        # ------------------------------------------------------------
        mean_conf = np.full((len(unique_shuffle), T), np.nan)
        
        for i_sh in range(len(unique_shuffle)):
            subj_curves = []
        
            for i_s in range(len(subjects)):
                events = aligned_conf[i_sh][i_s]
        
                if len(events) == 0:
                    continue
        
                events = np.stack(events, axis=0)  # [n_events, T]
                subj_curves.append(np.nanmean(events, axis=0))
        
            if len(subj_curves) > 0:
                mean_conf[i_sh] = np.nanmean(np.stack(subj_curves, axis=0), axis=0)
        
        # ------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------
        fig_sw, ax_sw = plt.subplots(figsize=(5, 4))
        
        for i_sh, sh in enumerate(unique_shuffle):
            ax_sw.plot(
                t_switch,
                mean_conf[i_sh],
                color=colormap[i_sh],
                linewidth=3,
                label=f'p(shuffle)={sh}'
            )
        
        ax_sw.axvline(0, color='k', linestyle='--', alpha=0.4)
        ax_sw.set_xlabel('Time from switch (s)')
        ax_sw.set_ylabel(r'Absolute confidence $|x|$')
        ax_sw.set_ylim(bottom=0)
        
        ax_sw.spines['right'].set_visible(False)
        ax_sw.spines['top'].set_visible(False)
        ax_sw.legend(frameon=False)
        
        fig_sw.tight_layout()


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


def simulate_noise_subjects(df, data_folder=DATA_FOLDER, n=4, nFrame=1546, fps=60,
                            load_simulations=True, sigma_predefined=None,
                            adaptation=False,
                            ):
    np.random.seed(24)  # 24 (3 perc)
    np.random.seed(60)  # 25, 50 (3 perc)
    np.random.seed(90)  # 25, 50 (3 perc)
    np.random.seed(23)  # 25, 50 (3 perc)
    np.random.seed(123)
    # adaptation parameters
    # ---
    tau_a = 100
    ga = 0.6
    tau_ou = 1
    # ---
    ratio = int(nFrame/1546)
    nFrame = nFrame-ratio+1
    # load ndt
    ndt = np.abs(np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
    # load parameters
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    print(len(pars), ' fitted subjects')
    label = f'_sigma_{sigma_predefined}_' if sigma_predefined is not None else ''
    if adaptation:
        label += 'adaptation'
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
                j_eff = (1 - pshuffles[i_trial]) * fitted_params_subject[0] + fitted_params_subject[1]
                params = fitted_params_subject[1:]
                params[0] = j_eff
                df_sub_trial = df_sub.loc[df_sub.trial_index == trial]
                stimulus = df_sub_trial.stimulus[:-14].values
                stimulus = interp1d(time, stimulus)(time_interp)
        
                j_eff, b_par, th, sigma, ndt = params
                if sigma_predefined is not None:
                    sigma = sigma_predefined
        
                lower_bound, upper_bound = np.array([-th, th]) + 0.5
                dt = 1 / fps
                tau = 1
                b_eff = stimulus * b_par
        
                # --- noise setup ---
                if adaptation:
                    # convert fitted white-noise sigma to OU sigma (same stationary variance)
                    sigma_ou = sigma # / np.sqrt(tau_ou / 2)
                    ou_decay = np.exp(-dt / tau_ou)
                    ou_sig   = sigma_ou * np.sqrt(1.0 - ou_decay**2)
                    # pre-draw OU noise for this trial
                    eta = np.zeros(nFrame)
                    for t in range(1, nFrame):
                        eta[t] = ou_decay * eta[t-1] + ou_sig * internal_noise_subject[i_trial, t]
                    noise = eta*dt
                else:
                    noise = internal_noise_subject[i_trial] * sigma * np.sqrt(dt / tau)
        
                x = np.zeros(nFrame)
                x[0] = 0.5
                a = np.zeros(nFrame)
        
                choice = np.zeros(nFrame)
                prev_choice = 0.0
                pending_choice = None
                pending_time = None
                ndt_frames = int(ndt / dt)
        
                for t in range(1, nFrame):
                    if pending_choice is not None and t >= pending_time:
                        prev_choice = pending_choice
                        pending_choice = None
        
                    s = 2 * x[t - 1] - 1
        
                    if adaptation:
                        drive = sigmoid(2 * (j_eff * (s - ga * a[t - 1]) + b_eff[t]))
                    else:
                        drive = sigmoid(2 * (j_eff * s + b_eff[t]))
        
                    x[t] = x[t - 1] + dt * (drive - x[t - 1]) / tau + noise[t]
        
                    if adaptation:
                        a[t] = a[t - 1] + (s - a[t - 1]) / tau_a * dt
        
                    
                    # bound crossing detection
                    if x[t] >= upper_bound:
                        new_choice = 1.0
                    elif x[t] <= lower_bound:
                        new_choice = -1.0
                    else:
                        new_choice = prev_choice
                
                    # schedule switch — but if one is already pending,
                    # check if x has crossed to the OPPOSITE bound (cancels pending)
                    # if new_choice != prev_choice and pending_choice is None:
                    if pending_choice is not None:
                        if (pending_choice == 1.0 and x[t] <= lower_bound):
                            # crossed back — cancel and reschedule opposite
                            pending_choice = -1.0
                            pending_time = t + ndt_frames
                        elif (pending_choice == -1.0 and x[t] >= upper_bound):
                            pending_choice = 1.0
                            pending_time = t + ndt_frames
                    elif new_choice != prev_choice:
                        pending_choice = new_choice
                        pending_time = t + ndt_frames
                
                    choice[t] = prev_choice
        
                choices_subject[i_trial, :] = choice[:time_frames]
                x_subject[i_trial, :] = x[:time_frames]
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


def drive(q, j, b):
    return sigmoid(2*j*(2*q-1)+2*b)-q


def f_prime(q, j, b=0.0):
    """Derivative of the drift term."""
    s = sigmoid(2*j*(2*q - 1) + 2*b)
    return 4* j * s * (1 - s) - 1


def f_double_prime(x, J, B):
    sig = 1 / (1 + np.exp(-2*J*(2*x - 1) - 2*B))
    return 16*(J**2)*sig*(1 - sig)*(1 - 2*sig)


def bc(y0, yT, x0, xT):
    return np.array([y0[0] - x0, yT[0] - xT])


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


def enums(idx):
    if idx == 1:
        return 'st'
    elif idx == 2:
        return 'nd'
    elif idx == 3:
        return 'rd'
    elif idx >= 4:
        return 'th'


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


def exp_fun(t, tau, k1, k2):
    return np.exp(-t/tau)*k1+k2


def expand_mask(mask, n_pre=3, n_post=3):
    """
    Expand a boolean mask in time.
    mask: boolean array
    n_pre: samples before
    n_post: samples after
    """
    mask = mask.copy()
    idx = np.where(mask)[0]
    for i in idx:
        mask[max(0, i-n_pre):min(len(mask), i+n_post+1)] = True
    return mask


def map_responses_to_posneg_1(responses):
    """
    Maps responses from 0 (nothing), 1 (left), 2 (right)
    to NaN, -1, 1
    """
    mapping = np.array([np.nan, -1, 1])
    return mapping[responses]


def plot_pupil_across_subjects_simple(data_folder=DATA_FOLDER,
                                      sublist=None,
                                      n_training=8,
                                      dt=1/60,
                                      t_before=4,
                                      condition='pShuffle',
                                      t_after=4,
                                      smooth_window=11,
                                      polyorder=2,
                                      pupil_col='Pupil_average_clean',
                                      save_plot=True, freq=2, noisy=False,
                                      plot_name='pupil_switch_avg.png',
                                      velocity=False, n=4, zscore_values=False,
                                      downsample_to=20, null=False,
                                      align=True, region_interval=[-2, 2]):
    """
    Plot pupil size aligned to all response switches, averaged per subject and then across subjects.

    Parameters
    ----------
    data_folder : str
        Folder containing behavioral and eye-tracker data
    sublist : list of str
        List of subjects to include. If None, uses all subjects
    n_training : int
        Number of initial trials to exclude
    dt : float
        Time step (seconds)
    t_before, t_after : float
        Time window before and after switch (seconds)
    smooth_window : int
        Window for Savitzky-Golay smoothing
    polyorder : int
        Polynomial order for Savitzky-Golay smoothing
    pupil_cols : tuple
        Pupil columns to plot ('LeftEye_Pupil','RightEye_Pupil')
    save_plot : bool
        Whether to save the figure
    plot_name : str
        File name for saving the figure
    """
    if downsample_to is not None:
        dt_eff = 1 / downsample_to
    else:
        dt_eff = dt
    if noisy:
        # --- Load behavioral data ---
        df_all = load_data(data_folder=data_folder + '/noisy/', n_participants='all', filter_subjects=True)
    else:
        df_all = load_data(data_folder=data_folder, n_participants='all', filter_subjects=True)
        if freq != 'all':
            df_all = df_all.loc[df_all.freq == freq]
    if sublist is None:
        sublist = df_all.subject.unique()
    
    if 'X' in pupil_col or 'Y' in pupil_col:
        xy_flag = True
    else:
        xy_flag = False
    if velocity:
        plot_name = 'velocity_' + plot_name
    if null:
        null_appendix = '_null_'
    else:
        null_appendix = ''
    plot_name = condition + null_appendix + '_' + plot_name
    if not align:
        plot_name = 'non_aligned_' + plot_name
    # --- Dictionary to store per-subject averages ---
    per_sub_avg = {pupil_col: {}}
    per_sub_avg_region = {pupil_col: {}}
    if not align:
        per_sub_choice = {pupil_col: {}}

    # --- Helper functions ---
    def get_switch_indices(responses):
        responses = np.array(responses)
        switches = np.where(responses[1:] != responses[:-1])[0] + 1
        return switches

    def extract_epoch(trial_eye, t_switch, pupil_col):
        if align:
            mask = ((trial_eye['t_trial'] > t_switch - t_before) & (trial_eye['t_trial'] < t_switch + t_after) & 
                    (trial_eye['t_trial'] >= 2) & (trial_eye['t_trial'] <= 26))
            t_epoch = trial_eye.loc[mask, 't_trial'].values - t_switch
        else:
            mask = (
                (trial_eye['t_trial'] >= 2) &
                (trial_eye['t_trial'] <= 26))
            t_epoch = trial_eye.loc[mask, 't_trial'].values
        epoch = trial_eye.loc[mask, pupil_col].values
        if len(epoch) > 1:
            if zscore_values:
                epoch = zscore(epoch, nan_policy='omit')
            return pd.DataFrame({'t': t_epoch, 'pupil': epoch})
        else:
            return None
    # --- Loop over subjects ---
    if not null:
        pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    if null:
        pars = glob.glob(SV_FOLDER + 'fitted_params/null_model_params/' + '*.npy')
    fitted_params_all = [np.load(par) for par in pars]
    j0s = [n*params[1] for params in fitted_params_all]
    j1s = [n*params[0] for params in fitted_params_all]
    for i_sub, sub in enumerate(sublist):
        df = df_all.loc[(df_all.trial_index > n_training) & (df_all.subject == sub)]
        if condition == 'regime':
            df[condition] = (((1-df['pShuffle'])*j1s[i_sub] + j0s[i_sub]) > 1)*2-1
        conditions = np.sort(df[condition].unique())
        path_save_csv = os.path.join(data_folder, 'aligned_eye_tracker_data', f'{sub}_aligned_Gaze_Data.csv')
        if not os.path.exists(path_save_csv):
            continue
        eyetracker_data = pd.read_csv(path_save_csv)

        # --- Loop over conditions ---
        for cond in conditions:
            df_cond = df[df[condition]==cond]
            aligned_epochs = []
            if not align:
                aligned_choices = []

            for ti in df_cond.trial_index.unique():
                trial_beh = df_cond[df_cond.trial_index==ti]
                if noisy:
                    trial_resp = trial_beh.responses.values
                else:
                    trial_resp = get_response_array(trial_beh)
                trial_eye = eyetracker_data[eyetracker_data.trial_index==ti]
                if len(trial_resp)==0 or trial_eye.empty:
                    continue
                if align:
                    switch_idx = get_switch_indices(trial_resp)
                    for swi in switch_idx:
                        t_switch = swi * dt
                        epoch_df = extract_epoch(trial_eye, t_switch, pupil_col)
                        if epoch_df is not None:
                            aligned_epochs.append(epoch_df)
                else:
                    epoch_df = extract_epoch(trial_eye, 0, pupil_col)
                    aligned_epochs.append(epoch_df)
                    resps = map_responses_to_posneg_1(trial_resp)
                    if noisy:
                        aligned_choices.append(resps*np.sign(trial_beh.stimulus.iloc[0] + np.random.randn()*1e-6))
                    else:
                        aligned_choices.append(resps*trial_beh.initial_side.iloc[0])

            # --- Concatenate all epochs for this subject & condition ---
            if len(aligned_epochs)==0:
                continue
            all_epochs = pd.concat(aligned_epochs, ignore_index=True)

            # Bin and average
            if align:
                t_bins = np.arange(-t_before, t_after+dt_eff, dt_eff)
            else:
                t_bins = np.arange(0, 26 + dt_eff, dt_eff)
                mean_responses = np.nanmean((np.array(aligned_choices)+1)/2, axis=0)
                if cond not in per_sub_choice[pupil_col]:
                    per_sub_choice[pupil_col][cond] = []
                per_sub_choice[pupil_col][cond].append(mean_responses)


            t_bin_centers = t_bins[:-1] + dt_eff/2
            bin_idx = np.digitize(all_epochs['t'], t_bins) - 1
            valid = (bin_idx >= 0) & (bin_idx < len(t_bin_centers))
            
            if pupil_col in ['blink', 'saccade']:
                # --- Sum events per bin ---
                binned_sum = (
                    pd.DataFrame({
                        'bin': bin_idx[valid],
                        'pupil': all_epochs['pupil'].values[valid]
                    })
                    .groupby('bin')['pupil']
                    .agg(np.nansum)
                    .reindex(np.arange(len(t_bin_centers)), fill_value=0)
                )
            
                # --- Count contributing epochs per bin ---
                epoch_ids = np.repeat(np.arange(len(aligned_epochs)), [len(e) for e in aligned_epochs])
                df_valid = pd.DataFrame({
                    'bin': bin_idx[valid],
                    'epoch': epoch_ids,
                })
                epochs_per_bin = df_valid.groupby('bin')['epoch'].nunique().reindex(np.arange(len(t_bin_centers)), fill_value=0)
            
                # --- Compute rate (events/sec) ---
                pupil_smooth = binned_sum.values / (epochs_per_bin.values * dt_eff)
                # pupil_smooth = scipy.signal.savgol_filter(
                #     pupil_smooth, window_length=min(smooth_window, len(pupil_smooth) | 1),
                #     polyorder=polyorder)
            
            else:
                # --- Continuous pupil: mean across epochs per bin ---
                binned_mean = (
                    pd.DataFrame({
                        'bin': bin_idx[valid],
                        'pupil': all_epochs['pupil'].values[valid]
                    })
                    .groupby('bin')['pupil']
                    .agg(np.nanmean)
                    .reindex(np.arange(len(t_bin_centers)), fill_value=np.nan)
                )
                pupil_smooth = binned_mean.values
            if velocity:
                pupil_smooth = scipy.signal.savgol_filter(
                        pupil_smooth.values,
                        window_length=min(smooth_window, len(pupil_smooth) | 1),
                        polyorder=polyorder,
                        deriv=1,
                        delta=dt_eff
                    )
            if cond not in per_sub_avg_region[pupil_col]:
                per_sub_avg_region[pupil_col][cond] = []
            if cond not in per_sub_avg[pupil_col]:
                per_sub_avg[pupil_col][cond] = []

            per_sub_avg[pupil_col][cond].append(pupil_smooth)
            idx_region = (t_bin_centers >= region_interval[0])*(t_bin_centers <= region_interval[1])
            if 'Pupil' in pupil_col:
                if align:
                    per_sub_avg_region[pupil_col][cond].append([np.nanmax(pupil_smooth[idx_region]),
                                                                np.nanmean(pupil_smooth[idx_region]),
                                                                np.nanmin(pupil_smooth[idx_region])])
                else:
                    per_sub_avg_region[pupil_col][cond].append([np.nanmax(pupil_smooth),
                                                                np.nanmean(pupil_smooth),
                                                                np.nanmin(pupil_smooth)])
            else:
                if align:
                    per_sub_avg_region[pupil_col][cond].append([np.nanmax(pupil_smooth[idx_region]),
                                                                np.nanmean(pupil_smooth[idx_region]),
                                                                np.nanmin(pupil_smooth[idx_region])])
                else:
                    per_sub_avg_region[pupil_col][cond].append([np.nanmax(pupil_smooth),
                                                                np.nanmean(pupil_smooth),
                                                                np.nanmin(pupil_smooth)])
    # --- Average across subjects ---
    if align:
        t_bins = np.arange(-t_before, t_after+dt_eff, dt_eff)
    else:
        t_bins = np.arange(0, 26 + dt_eff, dt_eff)
    t_bin_centers = t_bins[:-1] + dt_eff/2
    fig, axes = plt.subplots(1, 1, figsize=(6, 4.5))
    if condition == 'pShuffle':
        colormap = ['midnightblue','royalblue','lightskyblue'][::-1]
    if condition == 'regime':
        colormap = ['cadetblue', 'peru'][::-1]

    axes = [axes]
    if 'Pupil' in pupil_col and align and condition == 'regime':
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'min_pupil_across_trials_regime_noise.npy')
        np.save(save_path, per_sub_avg_region)
    for i_p, pup_col in enumerate([pupil_col]):
        for i_c, cond in enumerate(reversed(sorted(per_sub_avg[pupil_col].keys()))):
            # Convert to array (subjects x time)
            data_array = np.array(per_sub_avg[pupil_col][cond])
            grand_avg = np.nanmean(data_array, axis=0)
            grand_error = np.nanstd(data_array, axis=0) / np.sqrt(len(sublist))
            label_dict = {-1: 'Monostable', 1: 'Bistable'}
            if condition == 'regime':
                label = label_dict.get(cond, str(cond))
            else:
                label = str(cond)
            axes[i_p].plot(t_bin_centers, grand_avg, color=colormap[i_c], linewidth=3, label=label)
            axes[i_p].fill_between(t_bin_centers, grand_avg-grand_error,
                                   grand_avg+grand_error, color=colormap[i_c], alpha=0.3)
            if not align:
                grand_avg = np.nanmean(np.array(per_sub_choice[pupil_col][cond]), axis=0)
                # axes[i_p].plot(np.arange(len(grand_avg))/60, grand_avg, color=colormap[i_c], linewidth=3,
                #                 linestyle='--')
        axes[i_p].axvline(0, color='k', linestyle='--')
        axes[i_p].axhline(0, color='k', linestyle='--')
        if align:
            axes[i_p].set_xlabel('Time from switch (s)')
        else:
            axes[i_p].set_xlabel('Time (s)')
        axes[i_p].set_title(pupil_col, fontsize=15)
        axes[i_p].spines['right'].set_visible(False)
        axes[i_p].spines['top'].set_visible(False)
    if not noisy and not align:
        stim = get_blist(freq=freq, nFrame=1560, maxval=1)
        axes[i_p].plot(np.arange(1560)/60, 0.5*(stim+1)-0.5, color='k', linewidth=3)
    if xy_flag:
        axes[0].set_ylabel('Distance')
    else:
        if velocity:
            axes[0].set_ylabel('Pupil size velocity')
        else:
            axes[0].set_ylabel('Pupil size')
    if pupil_col == 'vergence_angle':
        axes[0].set_ylabel('Vergence angle (º)')
    if pupil_col == 'blink':
        axes[0].set_ylabel('Blink rate (Hz)')
    if pupil_col == 'saccade':
        axes[0].set_ylabel('Saccade rate (Hz)')
    axes[0].legend(title='p(shuffle)', frameon=False)
    # axes[0].set_ylim(-0.65, 0.65)
    fig.tight_layout()
    if save_plot:
        align_label = 'aligned_' if align else 'full_time_'
        if condition == 'pShuffle' and not velocity:
            if 'Pupil' in pupil_col:
                name = 'pupil'
            if 'blink' in pupil_col:
                name = 'blink'
            if 'saccade' in pupil_col:
                name = 'saccade'
            if 'speed' in pupil_col:
                name = 'speed'
            if 'fixation_break' in pupil_col:
                name = 'fixation_break'
            max_all_data = np.zeros((len(conditions), len(sublist)))
            min_all_data = np.zeros((len(conditions), len(sublist)))
            mean_all_data = np.zeros((len(conditions), len(sublist)))
            # reversed so that it's saved as p(shuffle): 1, 0.7, 0
            for i_c, cond in enumerate(reversed(sorted(per_sub_avg_region[pupil_col].keys()))):
                data_array = np.array(per_sub_avg_region[pupil_col][cond])
                max_all_data[i_c] = data_array[:, 0]
                mean_all_data[i_c] = data_array[:, 1]
                min_all_data[i_c] = data_array[:, 2]
            if noisy:
                min_name_file = f'{align_label}min_{name}_noisy_trials.npy'
                max_name_file = f'{align_label}max_{name}_noisy_trials.npy'
                mean_name_file = f'{align_label}mean_{name}_noisy_trials.npy'
            else:
                min_name_file = f'{align_label}min_{name}_hysteresis_trials_freq_{freq}.npy'
                max_name_file = f'{align_label}max_{name}_hysteresis_trials_freq_{freq}.npy'
                mean_name_file = f'{align_label}mean_{name}_hysteresis_trials_freq_{freq}.npy'
            min_save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', min_name_file)
            max_save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', max_name_file)
            mean_save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', mean_name_file)
            np.save(max_save_path, max_all_data)
            np.save(min_save_path, min_all_data)
            np.save(mean_save_path, mean_all_data)
        if noisy:
            label_noisy = 'noise_trials_'
            fig.suptitle('Noise trials', fontsize=16)
        else:
            label_noisy = 'hysteresis_trials_' + f'freq_{freq}_'
            fig.suptitle('Hysteresis trials', fontsize=16)
        fig.tight_layout()
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', label_noisy + plot_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=400, bbox_inches='tight')


def pupil_regression_raw_switches(
    data_folder,
    df_all,
    sublist,
    pupil_col='Pupil_residual',  # Pupil_average_clean
    regressors=('pShuffle', 'stimulus', 'abs_stimulus', 'time'),
    n_training=8,
    dt=1/60,
    align=False,
    t_before=0.5,
    t_after=0.5,
    noisy=False,
    velocity=False,
    n=4
):

    def get_switch_indices(responses):
        responses = np.asarray(responses)
        return np.where(responses[1:] != responses[:-1])[0] + 1

    def extract_epoch(trial_eye, t_switch, pupil_col, align=True):
        if align:
            mask = ((trial_eye['t_trial'] >= 0) & (trial_eye['t_trial'] <= 26) & 
                (trial_eye['t_trial'] >= t_switch - t_before) &
                (trial_eye['t_trial'] <= t_switch + t_after)
            )
        else:
            mask = (
                (trial_eye['t_trial'] >= 0) &
                (trial_eye['t_trial'] <= 26))
        if mask.sum() <= 1:
            return None

        pupil = trial_eye.loc[mask, pupil_col].values
        stim = trial_eye.loc[mask, 'stimulus'].values
        t_abs = trial_eye.loc[mask, 't_trial'].values
        if align:
            t_rel = trial_eye.loc[mask, 't_trial'].values - t_switch
            # ker = np.exp(-np.arange(10)[::-1])
            # pupil = np.convolve(pupil, ker/ker.sum(), mode='same')
        else:
            t_rel = t_abs.copy()


        if velocity:
            pupil = np.gradient(pupil)

        return t_rel, pupil, stim, t_abs
    
    def mask_q(q, t_switch):
        time_trial = np.arange(1560)/60
        if align:
            mask = (
                (time_trial >= t_switch - t_before) &
                (time_trial <= t_switch + t_after)
            )
        else:
            mask = (
                (time_trial >= 0) &
                (time_trial <= 26))
        return q[mask]

    raw_rows = []
    if align:
        t_bins = np.arange(-t_before, t_after + dt, dt)
    else:
        t_bins = np.arange(0, 26 + dt, dt)
    t_centers = t_bins[:-1] + dt / 2
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    ndt = np.abs(np.median(np.load(DATA_FOLDER + 'kernel_latency_average.npy')))
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    fitted_params_all = [np.load(par) for par in pars]
    fitted_params_all = [[n*params[0], n*params[1], params[2], params[4], params[3], ndt] for params in fitted_params_all]
    j0s = [params[1] for params in fitted_params_all]
    j1s = [params[0] for params in fitted_params_all]
    for i_sub, sub in enumerate(sublist):
        fitted_params = fitted_params_all[i_sub]
        df = df_all.loc[
            (df_all.subject == sub) &
            (df_all.trial_index > n_training)
        ]

        path_eye = os.path.join(
            data_folder,
            'aligned_eye_tracker_data',
            f'{sub}_aligned_Gaze_Data.csv'
        )
        if not os.path.exists(path_eye):
            continue

        eyetracker_data = pd.read_csv(path_eye)
        for ti in df.trial_index.unique():
            trial_beh = df[df.trial_index == ti]
            if noisy:
                trial_resp = trial_beh.responses.values
            else:
                trial_resp = get_response_array(trial_beh)
            # simulate to get posterior
            pshuf = trial_beh['pShuffle'].iloc[0]
            j_eff = ((1-pshuf)*fitted_params[0] + fitted_params[1])*n
            params = fitted_params[1:].copy()
            params[0] = j_eff
            if not noisy:
                choice, q = simulator_5_params(params=params, freq=trial_beh.freq.iloc[0], nFrame=1560,
                                               fps=60, return_choice=True, ini_cond_convergence=2)
            if noisy:
                choice, q = simulator_5_params(params=params, freq=2, nFrame=1560,
                                               fps=60, return_choice=True, ini_cond_convergence=2, stimulus=trial_beh.stimulus.values)
            if len(trial_resp) == 0:
                continue

            trial_eye = eyetracker_data[
                eyetracker_data.trial_index == ti
            ]
            if trial_eye.empty:
                continue
            if align:
                switch_idx = get_switch_indices(trial_resp)
    
                for swi in switch_idx:
                    t_switch = swi * dt
    
                    out_avg = extract_epoch(trial_eye, t_switch, pupil_col)
                    
                    if out_avg is None or out_avg is None:
                        continue
    
                    t, pupil, stim, t_abs = out_avg
    
                    pupil = zscore(pupil, nan_policy='omit')
    
                    bin_idx = np.digitize(t, t_bins) - 1
                    
                    q_masked = mask_q(q, t_switch)
    
                    for i, b in enumerate(bin_idx):
                        if b < 0 or b > len(t_centers) or i >= len(q_masked):
                            continue
                        # get regime: 1 if bistable 0 if monostable
                        jeff = (1-pshuf)*j1s[i_sub] + j0s[i_sub]
                        regime = (jeff > 1)*1
                        
                        raw_rows.append({
                            't_bin': b,
                            'time': t_abs[i],
                            'pupil': pupil[i],
                            'pShuffle': pshuf,
                            'effective_coupling': jeff,
                            'regime': regime,
                            'stimulus': stim[i],
                            'abs_stimulus': abs(stim[i]),
                            'posterior': q_masked[i],
                            'subject': sub,
                            'abs_stim_regime': regime*abs(stim[i]),
                            'dstim_dt': np.gradient(stim, dt)[i],
                            'dstim_dt_regime': np.gradient(stim)[i]/dt * regime
                        })
            else:
                out_avg = extract_epoch(trial_eye, 0, pupil_col, align=align)
                if out_avg is None or out_avg is None:
                    continue

                t, pupil, stim, t_abs = out_avg

                # pupil = zscore(pupil, nan_policy='omit')

                bin_idx = np.digitize(t, t_bins) - 1
                for i, b in enumerate(bin_idx):
                    if b < 0 or b > len(t_centers) or i >= len(q)-1:
                        continue
                    pshuf = trial_beh['pShuffle'].iloc[0]
                    # get regime: 1 if bistable 0 if monostable
                    jeff = (1-pshuf)*j1s[i_sub] + j0s[i_sub]
                    regime = (jeff > 1)*1
                    raw_rows.append({
                        't_bin': b,
                        'time': t_abs[i],
                        'pupil': pupil[i],
                        'pShuffle': pshuf,  #-np.mean([0, 0.7, 1]),
                        'effective_coupling': jeff,
                        'regime': regime,
                        'stimulus': stim[i],
                        'posterior': q[i],
                        # add some random epsilon because otherwise is just constant
                        # at t=0
                        'abs_stimulus': abs(stim[i])+np.random.randn()*1e-6,
                        'subject': sub,
                        'abs_stim_regime': regime*abs(stim[i]),
                        'dstim_dt': np.gradient(stim)[i]/dt,
                        'dstim_dt_regime': np.gradient(stim)[i]/dt * regime
                        
                    })

    raw_df = pd.DataFrame(raw_rows)

    beta = {r: np.zeros(len(t_centers)) for r in regressors}
    pval = {r: np.zeros(len(t_centers)) for r in regressors}
    intercept = np.zeros(len(t_centers))

    for b in range(len(t_centers)):
        df_b = raw_df[raw_df.t_bin == b]
        if len(df_b) < len(regressors) + 2:
            for r in regressors:
                beta[r][b] = np.nan
                pval[r][b] = np.nan
            intercept[b] = np.nan
            continue

        y = zscore(df_b['pupil'].values, nan_policy='omit')
        X = df_b[list(regressors)].values
        X = sm.add_constant(X)

        res = sm.OLS(y, X, missing='drop').fit()

        intercept[b] = res.params[0]
        for i, r in enumerate(regressors):
            beta[r][b] = res.params[i + 1]
            pval[r][b] = res.pvalues[i + 1]

    return beta, pval, t_centers, intercept


def get_corr_p_matrix_specific_trials(data_folder=DATA_FOLDER, mean=False, include_h4=False):
    # examples:  Hysteresis - Dominance, Dominance - Pupil, Dominance - Saccade
    data_dominance = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
    data_amplitude = np.load(DATA_FOLDER + 'mean_peak_amplitude_per_subject.npy')
    data_hysteresis_2 = np.load(DATA_FOLDER + 'hysteresis_width_freq_2.npy')
    data_hysteresis_4 = np.load(DATA_FOLDER + 'hysteresis_width_freq_4.npy')
    pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
    print(len(pars), ' fitted subjects')
    j1s = np.array([np.load(par)[0] for par in pars])
    j0s = np.array([np.load(par)[1] for par in pars])
    b1s = np.array([np.load(par)[2] for par in pars])
    # thetas = np.array([np.load(par)[4] for par in pars])
    sigmas = np.array([np.load(par)[3] for par in pars])
    
    if mean:
        data_dominance = np.nanmean(data_dominance, axis=0)
        data_amplitude = np.nanmean(data_amplitude, axis=0)
        data_hysteresis_2 = np.nanmean(data_hysteresis_2, axis=0)
        data_hysteresis_4 = np.nanmean(data_hysteresis_4, axis=0)
    else:
        data_amplitude = data_amplitude.flatten()
        data_dominance = data_dominance.flatten()
        data_hysteresis_2 = data_hysteresis_2.flatten()
        data_hysteresis_4 = data_hysteresis_4.flatten()
        
    variables = [

        # --- Behavior ---
        dict(name="dominance", kind="behavior", data=data_dominance),
        dict(name="hyst2", kind="behavior", data=data_hysteresis_2),
        dict(name="hyst4", kind="behavior", data=data_hysteresis_4),
        dict(name='param J0+J1', kind='behavior', data=j0s+j1s),
        dict(name='param sigma', kind='behavior', data=sigmas),
        dict(name='param b1', kind='behavior', data=b1s),
    
        # --- Eye measures ---
        dict(name="min pupil", kind="eye",
             pupil_col="Pupil_residual",
             align=True,
             measure="min"),
    
        dict(name="fixation_break baseline", kind="eye",
             pupil_col="fixation_break",
             align=False,
             measure="mean"),
    
        dict(name="fixation_break max", kind="eye",
             pupil_col="fixation_break",
             align=True,
             measure="max"),
        
        dict(name="saccade baseline", kind="eye",
             pupil_col="saccade",
             align=False,
             measure="mean"),
    
        dict(name="saccade max", kind="eye",
             pupil_col="saccade",
             align=True,
             measure="max"),

        dict(name="speed baseline", kind="eye",
             pupil_col="speed",
             align=False,
             measure="mean"),
    
        dict(name="speed max", kind="eye",
             pupil_col="speed",
             align=True,
             measure="max"),
    
        dict(name="blink baseline", kind="eye",
             pupil_col="blink",
             align=False,
             measure="mean"),
    
        dict(name="blink max", kind="eye",
             pupil_col="blink",
             align=True,
             measure="max"),
    ]
    
    if not include_h4:
        variables.pop(2)

    def load_eye_data(var, beh_name=None):
        return get_eye_tracker_data_across_trials(
            pupil_col=var["pupil_col"],
            align=var["align"],
            var_beh=beh_name,
            measure=var["measure"],
            mean=mean, flatten=True
        )
    n = len(variables)

    corr_matrix = np.full((n, n), np.nan)
    p_matrix = np.full((n, n), np.nan)
    annot = np.empty((n, n), dtype=object)
    annot[:] = ""
    
    
    for i in range(n):
        var_i = variables[i]
    
        for j in range(n):
            if i == j:
                continue
    
            var_j = variables[j]
    
            # ---------- Behavior ↔ Behavior ----------
            if var_i["kind"] == "behavior" and var_j["kind"] == "behavior":
                x = var_i["data"]
                y = var_j["data"]
    
            # ---------- Behavior ↔ Eye ----------
            elif var_i["kind"] == "behavior" and var_j["kind"] == "eye":
                x = var_i["data"]
                y = load_eye_data(var_j, beh_name=var_i["name"])
    
            elif var_i["kind"] == "eye" and var_j["kind"] == "behavior":
                x = load_eye_data(var_i, beh_name=var_j["name"])
                y = var_j["data"]
    
            # ---------- Eye ↔ Eye (ALL trials) ----------
            else:
                x = load_eye_data(var_i, beh_name=None)
                y = load_eye_data(var_j, beh_name=None)
    
            r, p = pearsonr(x, y)
    
            corr_matrix[i, j] = r
            p_matrix[i, j] = p
            annot[i, j] = stars(p)
    return corr_matrix, p_matrix, annot


def get_corr_p_matrix_all_trials(data_folder=DATA_FOLDER,
                                 mean=False, include_h4=False):
    data_dominance = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
    data_amplitude = np.load(DATA_FOLDER + 'mean_peak_amplitude_per_subject.npy')
    data_hysteresis_2 = np.load(DATA_FOLDER + 'hysteresis_width_freq_2.npy')
    data_hysteresis_4 = np.load(DATA_FOLDER + 'hysteresis_width_freq_4.npy')
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
    if not include_h4:
        variables.pop(2)
    if mean:
        variables = [np.nanmean(var, axis=0) for var in variables]
    else:
        variables = [var.flatten() for var in variables]
    n = len(variables)

    corr_matrix = np.full((n, n), np.nan)
    p_matrix = np.full((n, n), np.nan)
    annot = np.empty((n, n), dtype=object)
    annot[:] = ""
    for i in range(n):
        var_i = variables[i]
    
        for j in range(n):
            if i == j:
                continue
            r, p = pearsonr(var_i, variables[j])
            corr_matrix[i, j] = r
            p_matrix[i, j] = p
            annot[i, j] = stars(p)
    return corr_matrix, p_matrix, annot


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def get_eye_tracker_data_across_trials(data_folder=DATA_FOLDER,
                                       pupil_col='Pupil_residual',
                                       align=False,
                                       var_beh=None, measure='mean',
                                       flatten=False, mean=False):
    if var_beh is None or 'param' in var_beh:
        if 'Pupil' in pupil_col and align:
            save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'min_pupil_across_trials.npy')
        label_save_max = 'max' if align else 'baseline'
        if 'saccade' in pupil_col:
            save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', f'{label_save_max}_saccade_rate_across_trials.npy')
        if 'blink' in pupil_col:
            save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', f'{label_save_max}_blink_rate_across_trials.npy')
        if 'fixation' in pupil_col:
            save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', f'{label_save_max}_fixation_break_rate_across_trials.npy')
        if 'speed' in pupil_col:
            save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', f'{label_save_max}_eye_speed_across_trials.npy')
    else:
        align_label = 'aligned_' if align else 'full_time_'
        if 'Pupil' in pupil_col:
            name = 'pupil'
        if 'blink' in pupil_col:
            name = 'blink'
        if 'saccade' in pupil_col:
            name = 'saccade'
        elif 'speed' in pupil_col:
            name = 'speed'
        if pupil_col == 'fixation_break':
            name = 'fixation_break'
        if var_beh in ['dominance', 'amplitude', 'latency']:
            save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', f'{align_label}{measure}_{name}_noisy_trials.npy')
        if var_beh == 'hyst2':
            save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', f'{align_label}{measure}_{name}_hysteresis_trials_freq_2.npy')
        if var_beh == 'hyst4':
            save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', f'{align_label}{measure}_{name}_hysteresis_trials_freq_4.npy')
    variable = np.load(save_path)
    if mean:
        variable = np.nanmean(np.load(save_path), axis=0)
    if flatten:
        variable = variable.flatten()
    return variable


def sigma(u):
    return 1.0/(1.0 + np.exp(-u))


def sigma_prime(u):
    s = sigma(u)
    return s*(1-s)


def u_of(x, B, J, N):
    return 2*J*N*(2*x - 1) + 2*B


def f(x, B, J, N):
    return sigma(u_of(x, B, J, N)) - x


def fprime(x, B, J, N):
    return 4*J*N*sigma_prime(u_of(x, B, J, N)) - 1.0


def V_of(x, B, J, N):
    return x*x/2.0 - np.log(1+np.exp(2*N*(J*(2*x-1)) + 2*B))/(4*N*J)


def find_roots(B, J, N, ngrid=400):
    xs = np.linspace(0.001, 0.999, ngrid)
    vals = f(xs, B, J, N)

    roots = []
    for i in range(len(xs)-1):
        if vals[i]*vals[i+1] < 0:
            root = brentq(lambda x: f(x, B, J, N),
                          xs[i], xs[i+1])
            roots.append(root)

    return roots


def barrier_prefactor(B, J, N):
    roots = find_roots(B, J, N)

    if len(roots) < 3:
        return None  # monostable

    # classify by stability
    eqs = []
    for x in roots:
        fp = fprime(x, B, J, N)
        eqs.append((x, fp))

    # sort by x position
    eqs.sort(key=lambda t: t[0])

    # stable equilibria: f'(x) < 0
    stable = [(x, fp) for x, fp in eqs if fp < 0]
    unstable = [(x, fp) for x, fp in eqs if fp > 0]

    if len(stable) < 2 or len(unstable) < 1:
        return None  # not truly bistable

    # left well = smallest stable
    x_min = stable[0][0]

    # saddle = unstable between the two wells
    # choose unstable closest to x_min
    saddles = [u for u in unstable if u[0] > x_min]
    if len(saddles) == 0:
        return None

    x_sad = saddles[0][0]

    # compute barrier
    deltaV = V_of(x_sad, B, J, N) - V_of(x_min, B, J, N)

    fp_min = fprime(x_min, B, J, N)
    fp_sad = fprime(x_sad, B, J, N)

    A = np.sqrt(abs(fp_min * fp_sad)) / (2*np.pi)

    return deltaV, A


def kramers_width(J, N, D, alpha,
                  Bmin=-3, Bmax=3, nscan=400):

    Bs = np.linspace(Bmin, Bmax, nscan)

    # detect bistable interval
    bistable = [len(find_roots(B,J,N))>=3 for B in Bs]

    if not any(bistable):
        return 0.0

    idx = np.where(bistable)[0]
    B_lo, B_hi = Bs[idx[0]], Bs[idx[-1]]

    def rate_minus_alpha(B):
        val = barrier_prefactor(B,J,N)
        if val is None:
            return np.nan
        deltaV, A = val
        return A*np.exp(-deltaV/D) - alpha

    # forward switch
    B_forward = None
    grid = np.linspace(B_lo, B_hi, 1000)
    vals = [rate_minus_alpha(B) for B in grid]

    for i in range(len(vals)-1):
        if np.sign(vals[i]) != np.sign(vals[i+1]):
            B_forward = brentq(rate_minus_alpha,
                               grid[i], grid[i+1])
            break

    # backward switch (search reversed)
    B_backward = None
    for i in reversed(range(len(vals)-1)):
        if np.sign(vals[i]) != np.sign(vals[i+1]):
            B_backward = brentq(rate_minus_alpha,
                                grid[i], grid[i+1])
            break

    if B_forward is None or B_backward is None:
        return 0.0

    return B_forward - B_backward
