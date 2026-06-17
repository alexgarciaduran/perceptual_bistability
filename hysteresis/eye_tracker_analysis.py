# -*- coding: utf-8 -*-
"""
Eye-tracker analysis and plots.

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


def max_surprise_vs_min_pupil():
    
    save_path = os.path.join(DATA_FOLDER, 'aligned_eye_tracker_data','plots', 'min_pupil_across_trials_regime.npy')
    min_pupil = np.load(save_path, allow_pickle=True)
    
    min_pupil_bis = np.float32(min_pupil[0][0])
    min_pupil_mono = np.float32(min_pupil[1][0])
    # pupil_mono = min_pupil.item()['Pupil_residual'][-1]
    # pupil_bis = min_pupil.item()['Pupil_residual'][1]
    # each has max, mean, min (in inidices 0 1 2 respectively)
    # min_pupil_mono = np.array([pup[idx] for pup in pupil_mono])
    # min_pupil_bis = np.array([pup[idx] for pup in pupil_bis])
    path_max_mono = SV_FOLDER + 'max_val_surprise_monostable.npy'
    max_surprise_mono = np.load(path_max_mono)
    max_surprise_mono = np.log(max_surprise_mono[~np.isnan(max_surprise_mono)])
    path_max_bis = SV_FOLDER + 'max_val_surprise_bistable.npy'
    max_surprise_bis = np.load(path_max_bis)
    max_surprise_bis = np.log(max_surprise_bis[~np.isnan(max_surprise_bis)])
    
    
    fig, ax = plt.subplots(ncols=1, figsize=(3.2, 3))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    colormap = ['cadetblue', 'peru']
    data_mono, data_bis = max_surprise_mono, max_surprise_bis
    sns.barplot([data_mono, data_bis], palette=colormap, ax=ax)
    sns.swarmplot([data_mono, data_bis],
                  color="black",        # point fill
                  edgecolor="white",    # contrast on dark bars
                  linewidth=0.5,
                  size=3,
                  ax=ax,
                  zorder=10             # ensures points are on top
                )
    t_stat, pval = scipy.stats.ttest_ind(data_bis, data_mono, equal_var=False,
                                         alternative='greater')
    c1 = 0; c2 = 1
    cte = 0.2
    heights = np.max((np.zeros(2), [np.nanmax(data_bis)+cte, np.nanmax(data_mono)+cte]), axis=0)
    bars = [0, 1]
    barh = 0.03
    barplot_annotate_brackets(c1, c2, pval, bars, heights,
                              yerr=None, dh=0.08, barh=barh, fs=10,
                              maxasterix=3, ax=ax)
    ax.set_xlabel('')
    ax.set_xticks([])
    ax.set_ylabel('Maximum log-surprise\nat switch')
    fig.tight_layout()
    fig.savefig(SV_FOLDER + 'max_surprise_switch.png', dpi=200, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'max_surprise_switch.svg', dpi=200, bbox_inches='tight')

    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5), sharex=True, sharey=True)
    for a in ax:
        a.spines['right'].set_visible(False); a.spines['top'].set_visible(False)
    ax[0].plot(min_pupil_mono, max_surprise_mono, 'o',
               color='cadetblue')
    r, p = pearsonr(min_pupil_mono, max_surprise_mono)
    ax[0].annotate(f'r = {r:.3f}\np={p:.2e}', xy=(.04, 0.8), xycoords=ax[0].transAxes)
    linreg = LinearRegression(fit_intercept=True).fit(min_pupil_mono.reshape(-1, 1), max_surprise_mono.reshape(-1, 1))
    minmax_array = np.array([np.min(min_pupil_mono)-0.1, np.max(min_pupil_mono)+0.1]).reshape(-1, 1)
    pred_y = linreg.predict(minmax_array)
    ax[0].plot(minmax_array,
               pred_y, color='gray', linestyle='--', alpha=0.4, linewidth=3)
    
    ax[1].plot(min_pupil_bis, max_surprise_bis, 'o',
               color='peru')
    r, p = pearsonr(min_pupil_bis, max_surprise_bis)
    ax[1].annotate(f'r = {r:.3f}\np={p:.2e}', xy=(.04, 0.8), xycoords=ax[1].transAxes)
    linreg = LinearRegression(fit_intercept=True).fit(min_pupil_bis.reshape(-1, 1), max_surprise_bis.reshape(-1, 1))
    minmax_array = np.array([np.min(min_pupil_bis)-0.1, np.max(min_pupil_bis)+0.1]).reshape(-1, 1)
    pred_y = linreg.predict(minmax_array)
    ax[1].plot(minmax_array,
               pred_y, color='gray', linestyle='--', alpha=0.4, linewidth=3)
    ax[0].set_xlabel('Minimum pupil')
    ax[1].set_xlabel('Minimum pupil')
    ax[0].set_ylabel('Log(Surprise)')
    ax[0].set_title('Monostable', color='cadetblue', fontsize=14)
    ax[1].set_title('Bistable', color='peru', fontsize=14)
    fig.tight_layout()
    f, a = plt.subplots(1)
    pup_all = np.concatenate((min_pupil_mono, min_pupil_bis))
    surp_all = np.concatenate((max_surprise_mono, max_surprise_bis))
    a.plot(pup_all, surp_all, 'o', color='k')


def eye_tracker_save_align_data(data_folder=DATA_FOLDER, ntraining=8):
    df_hyst = load_data(data_folder=data_folder, n_participants='all',
                        filter_subjects=False)
    df_noisy = load_data(data_folder=data_folder + '/noisy/', n_participants='all',
                         filter_subjects=False)
    subjects = df_hyst.subject.unique()
    for i_s, sub in enumerate(subjects):
        print(sub)
        sub_path = 'sub' + sub[1:]
        labs = [str(i) for i in range(12)] + ['11.6']
        gaze_blocks = []
        df_noisy_sub = df_noisy.loc[df_noisy.subject == sub]
        df_hyst_sub = df_hyst.loc[df_hyst.subject == sub]
        for i in range(1, 13):
            path = DATA_FOLDER + 'all_data/HYSTERESIS_EXPERIMENT/' + sub_path + '/eye_tracker/' + sub_path + '_block_' + labs[i] + '_Gaze_Data.csv'
            gaze_block = pd.read_csv(path)
            gaze_blocks.append(gaze_block)
        path_save_csv = DATA_FOLDER + '/aligned_eye_tracker_data/' + sub + '_aligned_Gaze_Data.csv'
        _ = align_eye_tracking_blocks_hyst_noisy(gaze_blocks, df_noisy_sub, df_hyst_sub,
                                                 trial_duration=27.,
                                                 dt_noisy=1/60,
                                                 save_csv=path_save_csv)


def compute_vergence(x_left, x_right, screen_width=1920,
                     eye_to_screen_distance=700):
    """
    Compute vergence angle from gaze coordinates.

    Args:
        x_left, y_left: Gaze coordinates of left eye on screen (pixels)
        x_right, y_right: Gaze coordinates of right eye on screen (pixels)
        screen_width, screen_height: Screen dimensions (pixels)
        eye_to_screen_distance: Distance from eyes to screen (mm)

    Returns:
        vergence_angle: Vergence angle in degrees
    """
    # Convert pixel coordinates to mm (assuming screen dimensions are in mm)
    # You may need to adjust this based on your screen's physical size and resolution
    screen_width_mm = 500  # Example: 500mm (adjust as needed)

    # Convert pixel coordinates to mm
    x_left_mm = (x_left) * screen_width_mm
    x_right_mm = (x_right) * screen_width_mm

    # Calculate horizontal angles for each eye
    angle_left = np.degrees(np.arctan((x_left_mm - screen_width_mm/2) / eye_to_screen_distance))
    angle_right = np.degrees(np.arctan((x_right_mm - screen_width_mm/2) / eye_to_screen_distance))

    # Vergence is the difference between the two angles
    vergence_angle = angle_left - angle_right

    return vergence_angle


def detect_blinks(pupil, min_diam=0.1):
    return np.isnan(pupil) | (pupil < min_diam)


def blink_onset_boolean(pupil):
    blink = np.isnan(pupil)

    onset = np.zeros_like(blink, dtype=bool)
    onset[1:] = (~blink[:-1]) & (blink[1:])

    return onset


def engbert_velocity(x, dt):
    v = (x[4:] + x[3:-1] - x[1:-3] - x[:-4]) / (6*dt)
    v = np.pad(v, (2,2), mode='edge')
    return v


def detect_saccades(x, y, dt=1/60, n_stds=5):
    
    vx = engbert_velocity(x, dt)
    vy = engbert_velocity(y, dt)
    
    speed = np.sqrt(vx**2 + vy**2)
    # Median Absolute Deviation
    vel_thresh = n_stds * 1.4826 * np.median(np.abs(speed - np.median(speed)))
    return (speed > vel_thresh), [speed, vx, vy]


def saccade_onset_boolean(x, y, dt=1/60, vel_thresh=30):  # deg/s equivalent
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    speed = np.sqrt(vx**2 + vy**2)
    
    saccade = (speed > vel_thresh)
    onset = np.zeros_like(saccade, dtype=bool)
    onset[1:] = (~saccade[:-1]) & (saccade[1:])

    return onset


def detect_fixation_break(x, y, max_dist=0.1, min_samples=3):
    dist = np.sqrt(x**2+y**2)
    breaks = (dist > max_dist)
    return breaks


def align_eye_tracking_blocks_hyst_noisy(
    gaze_blocks,
    df_noisy_sub,
    df_hyst_sub,
    trial_duration=27.,  # 26s trial + 1s trial delay
    dt_noisy=1/60,
    save_csv=None, threshold_detection=0.2, n_trials_block=10
):
    """
    Align Tobii gaze samples to trial start times.
    Ensures consistent trial_index (1..n_trials) and t_trial per sample.
    """

    all_aligned = []
    def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 3):
        sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
        filtered_data = scipy.signal.sosfiltfilt(sos, data)
        return filtered_data
    
    def interp_nans(x):
        x = np.asarray(x, float)
        nans = np.isnan(x)
        if nans.any():
            x[nans] = np.interp(
                np.flatnonzero(nans),
                np.flatnonzero(~nans),
                x[~nans]
            )
        return x
    trial_idx_hyst = df_hyst_sub.trial_index.unique()
    trial_idx_noisy = df_noisy_sub.trial_index.unique()
    # --- Normalize gaze timestamps to seconds ---
    first_tobii = min([g['SystemTS'].iloc[0] for g in gaze_blocks])
    first_tobii_sec = first_tobii * 1e-6
    t_index = 1
    for block_id, gaze_block in enumerate(gaze_blocks):
        gaze_block = gaze_block.copy()
        gaze_block['t_tobii'] = gaze_block['SystemTS'] * 1e-6 - first_tobii_sec
        n_samples = len(gaze_block)
        indices_change_trial = np.concatenate(([0], np.where(np.diff(gaze_block['t_tobii']) > threshold_detection)[0]+1))
        trial_index = np.zeros(n_samples, dtype=int)
        for i, start_idx in enumerate(indices_change_trial):
            if i < len(indices_change_trial) - 1:
                end_idx = indices_change_trial[i+1]
            else:
                end_idx = n_samples  # last trial goes to end of block
            trial_index[start_idx:end_idx] = t_index
            t_index = t_index + 1
        gaze_block['trial_index'] = trial_index
        t_trial = np.zeros(n_samples)
        stimulus = np.zeros(n_samples)
        blinks = np.zeros((2, n_samples))
        saccades = np.zeros((2, n_samples))
        raw_blinks = np.zeros((2, n_samples))
        raw_saccades = np.zeros((2, n_samples))
        condition = np.zeros((n_samples), dtype=object)
        responses = np.full(n_samples, np.nan)
        switches = np.full(n_samples, np.nan)
        pshuffle = np.full(n_samples, np.nan)
        for trial_id, start_idx in enumerate(indices_change_trial):
            trial_idx = gaze_block['trial_index'][start_idx]
            if trial_idx in trial_idx_hyst:
                df_trial = df_hyst_sub.loc[df_hyst_sub.trial_index == trial_idx]
                ini_side = df_trial['initial_side'].iloc[0]
                freq = df_trial['freq'].iloc[0]*ini_side
                if freq != 0:
                    stim = get_blist(freq=freq, nFrame=1560, maxval=3)
                else:
                    stim = np.zeros(1560)
                condition[start_idx:end_idx] = f"hysteresis_{df_trial['freq'].iloc[0]}"
                resp_trial = get_response_array(df_trial)
                pshuf = df_trial['pShuffle'].iloc[0]
            elif trial_idx in trial_idx_noisy:
                df_noisy_trial = df_noisy_sub.loc[df_noisy_sub.trial_index == trial_idx]
                stim = df_noisy_trial['stimulus']
                resp_trial = df_noisy_trial['responses'].values
                condition[start_idx:end_idx] = "noise_stim"
                pshuf = df_noisy_trial['pShuffle'].iloc[0]
            if trial_id < len(indices_change_trial) - 1:
                end_idx = indices_change_trial[trial_id+1]
            else:
                end_idx = n_samples
            # get actual timestamps
            t_actual = gaze_block['t_tobii'].iloc[start_idx:end_idx].values
            pshuffle[start_idx:end_idx] = pshuf
            # trial end is the last sample
            t_end = t_actual[-1]
            time_trial = trial_duration - (t_end - t_actual)
            # compute backward trial time
            t_trial[start_idx:end_idx] = time_trial
            first0 = np.where(time_trial >= 0)[0][0]
            start_idx_new = start_idx + first0
            stimulus[start_idx_new:start_idx_new+len(stim)] = stim
            responses[start_idx_new:start_idx_new+len(stim)] = resp_trial
            switches[start_idx_new+1:start_idx_new+len(stim)] = resp_trial[1:] != resp_trial[:-1]
            
            
            # create mask for valid trial time
            valid_mask = (time_trial >= 0) & (time_trial <= 26)
            valid_idx = np.where(valid_mask)[0]  # indices relative to start_idx
            non_valid_idx = np.where(~valid_mask)[0]
            # --- Preprocess pupil trial by trial ---
            
            for left_right, eye in enumerate(['LeftEye_Pupil', 'RightEye_Pupil']):
                pupil = gaze_block[eye].iloc[start_idx:end_idx].values

                # restrict to valid indices
                pupil_valid = pupil[valid_idx]
                time_valid = time_trial[valid_idx]
                # masking pupil in blinks/saccades
                blink_mask = detect_blinks(pupil_valid)
                blink_mask = expand_mask(blink_mask, n_pre=3, n_post=3)
                if left_right == 1:
                    x = gaze_block['RightEye_X'].iloc[start_idx:end_idx].values    
                    y = gaze_block['RightE_Y'].iloc[start_idx:end_idx].values
                else:
                    x = gaze_block['LeftEye_X'].iloc[start_idx:end_idx].values
                    y = gaze_block['LeftEye_Y'].iloc[start_idx:end_idx].values
                saccade_mask, _ = detect_saccades(x[valid_idx], y[valid_idx], dt=dt_noisy, n_stds=5)
                bad = blink_mask | saccade_mask
                pupil_valid[bad] = np.nan
                
                # interpolate
                pupil_valid = interp_nans(pupil_valid)
                
                # exponential fit and subtraction
                p0 = [np.median(time_valid), np.max(pupil_valid)-np.min(pupil_valid), np.min(pupil_valid)]
                try:
                    pars, _ = curve_fit(exp_fun, time_valid, pupil_valid, p0=p0)
                    pupil_valid = pupil_valid - exp_fun(time_valid, *pars)
                except:
                    pass
                # lowpass filter (cutoff: 6 Hz)
                pupil_valid = lowpass(pupil_valid, 6, 1/dt_noisy)
                # z-score
                pupil_valid = (pupil_valid - np.nanmean(pupil_valid)) / np.nanstd(pupil_valid)
                pupil_valid[bad] = np.nan
                gaze_block.loc[start_idx + valid_idx, eye+'_clean'] = pupil_valid
                gaze_block.loc[start_idx + non_valid_idx, eye+'_clean'] = 0
                blink_mask = blink_onset_boolean(pupil)
                # blink_mask = detect_blinks(pupil)
                # blink_mask = expand_mask(blink_mask, n_pre=3, n_post=3)
                _, speed_all = detect_saccades(x, y, dt=dt_noisy, n_stds=5)
                saccade_mask = saccade_onset_boolean(x, y, dt=1/60, vel_thresh=2)
                # saccade_mask = expand_mask(saccade_mask, n_pre=1, n_post=1)
                
                blinks[left_right, start_idx:end_idx] = blink_mask
                saccades[left_right, start_idx:end_idx] = saccade_mask
                
                blink_mask = detect_blinks(pupil)
                # blink_mask = expand_mask(blink_mask, n_pre=3, n_post=3)
                saccade_mask, speed_all = detect_saccades(x, y, dt=dt_noisy, n_stds=5)
                # saccade_mask = expand_mask(saccade_mask, n_pre=1, n_post=1)
                raw_blinks[left_right, start_idx:end_idx] = blink_mask
                raw_saccades[left_right, start_idx:end_idx] = saccade_mask
                
        x_pos_right = gaze_block['RightEye_X'].values
        x_pos_left = gaze_block['LeftEye_X'].values
        y_pos_right = gaze_block['RightE_Y'].values
        y_pos_left = gaze_block['LeftEye_Y'].values
        
        
        x_raw = np.mean(np.c_[x_pos_left, x_pos_right], axis=1) - 0.5
        y_raw = np.mean(np.c_[y_pos_left, y_pos_right], axis=1) - 0.5
        
        # get fixation breaks
        position = np.sqrt(x_raw**2+y_raw**2)
        fixation_breaks = detect_fixation_break(x_raw, y_raw, max_dist=0.1)
        
        # interpolate NaNs
        x_pos_left = interp_nans(x_pos_left)
        x_pos_right = interp_nans(x_pos_right)
        y_pos_left = interp_nans(y_pos_left)
        y_pos_right = interp_nans(y_pos_right)

        x = np.mean(np.c_[x_pos_left, x_pos_right], axis=1) - 0.5
        y = np.mean(np.c_[y_pos_left, y_pos_right], axis=1) - 0.5

        # get saccade mask, sacc_mask
        sacc_mask, _ = detect_saccades(x, y, dt=dt_noisy, n_stds=5)

        # apply medfilt to position (x, y)
        x = scipy.signal.medfilt(x, 5)
        y = scipy.signal.medfilt(y, 5)

        # now compute speed
        _, speed_all = detect_saccades(x, y, dt=dt_noisy, n_stds=5)
        for speed in speed_all:
            sacc_mask = sacc_mask | (np.abs(speed) > 2)
            speed[sacc_mask] = np.nan

        # save data
        gaze_block['blink'] = np.logical_or(blinks[0], blinks[1])
        gaze_block['saccade'] = np.logical_or(saccades[0], saccades[1])
        gaze_block['fixation_break'] = fixation_breaks
        gaze_block['x_position'] = x
        gaze_block['y_position'] = y
        gaze_block['position'] = position
        gaze_block['raw_blink'] = np.logical_or(raw_blinks[0], raw_blinks[1])
        gaze_block['raw_saccade'] = np.logical_or(raw_saccades[0], raw_saccades[1])
        gaze_block['speed'] = speed_all[0]
        gaze_block['y_speed'] = speed_all[2]
        gaze_block['x_speed'] = speed_all[1]
        gaze_block['t_trial'] = np.clip(t_trial, -2, trial_duration)
        gaze_block['stimulus'] = stimulus
        gaze_block['abs_stimulus'] = np.abs(stimulus)
        gaze_block['condition'] = condition
        gaze_block['response'] = responses
        gaze_block['switch'] = switches
        gaze_block['pShuffle'] = pshuffle
        
        gaze_block['vergence_angle'] = compute_vergence(
            gaze_block['LeftEye_X'], gaze_block['RightEye_X'],
            screen_width=1920, eye_to_screen_distance=700)

        all_aligned.append(gaze_block)

    gaze_aligned = pd.concat(all_aligned, ignore_index=True)
    # Add average pupil
    gaze_aligned['Pupil_average_clean'] = np.nanmean(
        np.row_stack([gaze_aligned['LeftEye_Pupil_clean'], gaze_aligned['RightEye_Pupil_clean']]), axis=0
    )

    gaze_aligned['Pupil_residual'] = gaze_aligned['Pupil_average_clean'].copy()


    dt = 1 / 60        # time bin (seconds)
    t_min, t_max = 0, 26
    min_trials = 10    # minimum samples per bin for regression

    # get time bins
    gaze_aligned['t_bin'] = np.round(gaze_aligned['t_trial'] / dt) * dt
    
    # Keep only bins in analysis window
    gaze_aligned.loc[
        (gaze_aligned['t_bin'] < t_min) |
        (gaze_aligned['t_bin'] > t_max),
        't_bin'] = np.nan

    for t in np.sort(gaze_aligned['t_bin'].dropna().unique()):
        mask = (
            (gaze_aligned['t_bin'] == t) &
            # (gaze_aligned['condition'] != 'noise_stim') &
            (~np.isnan(gaze_aligned['Pupil_average_clean'])) &
            (~np.isnan(gaze_aligned['abs_stimulus']))
        )
    
        if mask.sum() < min_trials:
            continue
    
        y = gaze_aligned.loc[mask, 'Pupil_average_clean'].values
        X = gaze_aligned.loc[mask, 'abs_stimulus'].values
    
        # Optional intercept (recommended unless already baseline-corrected)
        X = sm.add_constant(X)
    
        res = sm.OLS(y, X).fit()
    
        gaze_aligned.loc[mask, 'Pupil_residual'] = (
            y - res.predict(X))

    pup_residual = (
                    gaze_aligned
                    .groupby('trial_index')['Pupil_residual']
                    .transform(lambda x: (x - np.nanmean(x)) / np.nanstd(x)))
    gaze_aligned['Pupil_residual'] = interp_nans(pup_residual)
    if save_csv:
        gaze_aligned.to_csv(save_csv, index=False)

    return gaze_aligned


def plot_examples_subs_eye_tracker(data_folder=DATA_FOLDER,
                                   ntrials=5, n_training=8, sub='s_31'):
    df_all = load_data(data_folder=data_folder + '/noisy/', n_participants='all',
                   filter_subjects=False)
    df = df_all.loc[(df_all.trial_index > n_training) & (df_all.subject == sub)]
    t_index_unique = df.trial_index.unique()
    ti_unique = np.random.choice(t_index_unique, ntrials, replace=False)
    path_save_csv = data_folder + '/aligned_eye_tracker_data/' + sub + '_aligned_Gaze_Data.csv'
    eyetracker_data = pd.read_csv(path_save_csv)
    fig, ax = plt.subplots(ncols=ntrials, figsize=(5 + ntrials*4, 5))
    if ntrials == 1:
        ax = [ax]
    for i_t, ti in enumerate(ti_unique):
        eyetracker_data_ti = eyetracker_data.loc[eyetracker_data.trial_index == ti]
        # choices_hyst = get_response_array(df.loc[df.trial_index == ti])
        choices_hyst = df.loc[df.trial_index == ti, 'responses']
        time_choices = np.arange(0, 26, 1/60)
        mean_raw = np.nanmean(
            np.row_stack([eyetracker_data_ti['LeftEye_Pupil'], eyetracker_data_ti['RightEye_Pupil']]), axis=0
        )
        ax[i_t].plot(eyetracker_data_ti.t_trial, zscore(mean_raw, nan_policy='omit'),
                     label='Raw', linewidth=3)
        ax[i_t].plot(eyetracker_data_ti.t_trial, eyetracker_data_ti.Pupil_average_clean,
                     label='Clean', linewidth=3)
        ax2 = ax[i_t].twinx()
        ax2.plot(time_choices, choices_hyst, color='k', linewidth=3)
        ax2.set_yticks([])
        ax[i_t].set_xlabel('Time (s)')
        for a in [ax2, ax[i_t]]:
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
    ax[0].legend()
    fig.tight_layout()


def plot_pupil_around_switches_single_sub(data_folder=DATA_FOLDER,
                                          sublist=['s_1'], n_training=8,
                                          dt=1/60, t_before=1.5, t_after=1.5,
                                          smooth_window=10):
    # --- Identify switches ---
    def get_switch_indices(responses):
        responses = np.array(responses)
        switches = np.where(responses[1:] != responses[:-1])[0] + 1
        return switches

    df_all = load_data(data_folder=data_folder, n_participants='all',
                   filter_subjects=False)
    if sublist is None:
        sublist = df_all.subject.unique()
        close_fig_flag = True
    else:
        close_fig_flag = False
    conditions = np.sort(df_all['pShuffle'].unique())
    all_subjects_epochs = {cond: [] for cond in conditions}
    for sub in sublist:
        df = df_all.loc[(df_all.trial_index > n_training) & (df_all.subject == sub)]
        path_save_csv = data_folder + '/aligned_eye_tracker_data/' + sub + '_aligned_Gaze_Data.csv'
        eyetracker_data = pd.read_csv(path_save_csv)
        fig, axes = plt.subplots(1, ncols=1, figsize=(5, 4), sharey=True)
        axes = [axes]
        colormap = ['midnightblue', 'royalblue', 'lightskyblue']
        for i_c, cond in enumerate(conditions):
            df_cond = df[df['pShuffle'] == cond]
            aligned_epochs = []
    
            # Loop over trials
            for ti in df_cond.trial_index.unique():
                trial_beh = df_cond[df_cond.trial_index == ti]
                trial_resp = get_response_array(trial_beh)
                trial_eye = eyetracker_data[eyetracker_data.trial_index == ti]
                if len(trial_resp) == 0 or trial_eye.empty:
                    continue
                # Find switches
                switch_indices = get_switch_indices(trial_resp)
                for swi in switch_indices:
                    t_switch = swi * dt
                    mask = (trial_eye['t_trial'] > t_switch - t_before) & (trial_eye['t_trial'] < t_switch + t_after)
                    # epoch_right = zscore(trial_eye.loc[mask, 'LeftEye_Pupil'].values)
                    # epoch_left = zscore(trial_eye.loc[mask, 'RightEye_Pupil'].values)
                    epoch = trial_eye.loc[mask, 'Pupil_average_clean'].values
                    t_epoch = trial_eye.loc[mask, 't_trial'].values - t_switch
                    if len(epoch) > 0:
                        aligned_epochs.append(pd.DataFrame({'t': t_epoch, 'pupil': epoch}))
            all_epochs = pd.concat(aligned_epochs, ignore_index=True)
            all_subjects_epochs[cond].append(aligned_epochs)
            # Bin by time
            t_bins = np.arange(-t_before, t_after + dt, dt)
            binned = all_epochs.groupby(np.digitize(all_epochs['t'], t_bins))['pupil'].mean()
            t_bin_centers = t_bins[:-1] + dt/2
            pupil_smooth = binned.values
    
            axes[0].plot(t_bin_centers, pupil_smooth, label=f'{cond}', color=colormap[i_c],
                         linewidth=4)
        # axes.axvline(0, color='k', linestyle='--', label='Switch')
        for a in axes:
            a.axhline(0, color='k', linestyle='--')
            a.axvline(0, color='k', linestyle='--')
            a.set_xlabel('Time from switch (s)')
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
        axes[0].set_ylabel('Pupil size')
        axes[0].legend(frameon=False, title='p(shuffle)')
        fig.tight_layout()
        fig.savefig(data_folder + '/aligned_eye_tracker_data/plots/' + sub + '_pupil_at_switch.png', dpi=400, bbox_inches='tight')
        fig.savefig(data_folder + '/aligned_eye_tracker_data/plots/' + sub + '_pupil_at_switch.pdf', dpi=400, bbox_inches='tight')
        if close_fig_flag:
            plt.close(fig)


def plot_pupil_across_all_trials(data_folder=DATA_FOLDER,
                                 sublist=None,
                                 n_training=8,
                                 dt=1/60,
                                 t_before=4,
                                 condition='pShuffle',
                                 t_after=4,
                                 smooth_window=11,
                                 polyorder=2,
                                 pupil_col='Pupil_average_clean',
                                 save_plot=True,
                                 plot_name='all_pupil_switch_avg.png',
                                 velocity=False, n=4,
                                 downsample_to=20, null=False, align=True,
                                 region_interval=[-2, 2]):
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
    if region_interval is None:
        region_interval = [-t_before, t_after]
    if downsample_to is not None:
        dt_eff = 1 / downsample_to
    else:
        dt_eff = dt
    df_all_noisy = load_data(data_folder=data_folder + '/noisy/', n_participants='all', filter_subjects=True)
    df_all_hysteresis = load_data(data_folder=data_folder, n_participants='all', filter_subjects=True)

    if sublist is None:
        sublist = df_all_hysteresis.subject.unique()
    
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
        plot_name = 'full_time_' + plot_name
    # --- Dictionary to store per-subject averages ---
    per_sub_avg = {pupil_col: {}}
    per_sub_avg_region = {pupil_col: {}}

    # --- Helper functions ---
    def get_switch_indices(responses):
        responses = np.array(responses)
        switches = np.where(responses[1:] != responses[:-1])[0] + 1
        return switches[1:]

    def extract_epoch(trial_eye, t_switch, pupil_col):
        if align:
            mask = ((trial_eye['t_trial'] > t_switch - t_before)
                    & (trial_eye['t_trial'] < t_switch + t_after) &
                    (trial_eye['t_trial'] >= 2) &
                    (trial_eye['t_trial'] <= 26))
            t_epoch = trial_eye.loc[mask, 't_trial'].values - t_switch
        else:
            mask = (
                (trial_eye['t_trial'] >= 2) &
                (trial_eye['t_trial'] <= 26))
            t_epoch = trial_eye.loc[mask, 't_trial'].values
        epoch = trial_eye.loc[mask, pupil_col].values
        if len(epoch) > 1:
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
        df_noisy_sub = df_all_noisy.loc[(df_all_noisy.trial_index > n_training) & (df_all_noisy.subject == sub)]
        df_hyst_sub = df_all_hysteresis.loc[(df_all_hysteresis.trial_index > n_training) & (df_all_hysteresis.subject == sub)]
        if condition == 'regime':
            df_hyst_sub[condition] = (((1-df_hyst_sub['pShuffle'])*j1s[i_sub] + j0s[i_sub]) > 1)*2-1
            df_noisy_sub[condition] = (((1-df_noisy_sub['pShuffle'])*j1s[i_sub] + j0s[i_sub]) > 1)*2-1
        conditions = np.sort(df_hyst_sub[condition].unique())
        path_save_csv = os.path.join(data_folder, 'aligned_eye_tracker_data', f'{sub}_aligned_Gaze_Data.csv')
        if not os.path.exists(path_save_csv):
            continue
        eyetracker_data = pd.read_csv(path_save_csv)

        # --- Loop over conditions ---
        for cond in conditions:
            df_hyst_cond = df_hyst_sub[df_hyst_sub[condition]==cond]
            df_noisy_cond = df_noisy_sub[df_noisy_sub[condition]==cond]
            aligned_epochs = []
            trial_index_noisy = df_noisy_cond.trial_index.unique()
            trial_index_hyst = df_hyst_cond.trial_index.unique()
            all_tr_ind = np.concatenate((trial_index_noisy, trial_index_hyst))
            for ti in all_tr_ind:
                if ti in trial_index_hyst:
                    trial_beh = df_hyst_cond[df_hyst_cond.trial_index==ti]
                    trial_resp = get_response_array(trial_beh)
                if ti in trial_index_noisy:
                    trial_beh = df_noisy_cond[df_noisy_cond.trial_index==ti]
                    trial_resp = trial_beh.responses.values
                trial_eye = eyetracker_data[eyetracker_data.trial_index==ti]
                if len(trial_resp)==0 or trial_eye.empty:
                    continue
                if align:
                    switch_idx = get_switch_indices(trial_resp)
                    # exclusion_frames = int(0.25 / dt)
                    
                    # isolated_switches = [
                    #     swi for swi in switch_idx
                    #     if np.all(np.abs(switch_idx - swi)[np.abs(switch_idx - swi) > 0] > exclusion_frames)
                    # ]
                    for swi in switch_idx:
                        t_switch = swi * dt
                        epoch_df = extract_epoch(trial_eye, t_switch, pupil_col)
                        if epoch_df is not None:
                            # subtract mean of first 10 timepoints
                            # epoch_df['pupil'] -= np.nanmean(epoch_df['pupil'][:20])
                            # subtract first point
                            # epoch_df['pupil'] -= epoch_df['pupil'][0]
                            # z-score
                            # if zscore_values:
                            # if pupil_col == 'Pupil_average_clean':
                            #     epoch_df['pupil'] = zscore(epoch_df['pupil'], nan_policy='omit')
                            # if xy_flag:
                            #     if noisy:
                            #         resp_before = (trial_resp[swi-1]*2-1)
                            #     else:
                            #         resp_before = (trial_resp[swi-1]*2-1)*(trial_beh.initial_side)
                            #     epoch_df['pupil'] *= resp_before
                            aligned_epochs.append(epoch_df)
                else:
                    epoch_df = extract_epoch(trial_eye, 0, pupil_col)
                    aligned_epochs.append(epoch_df)

            # --- Concatenate all epochs for this subject & condition ---
            if len(aligned_epochs)==0:
                continue
            all_epochs = pd.concat(aligned_epochs, ignore_index=True)

            # Bin and average
            if align:
                t_bins = np.arange(-t_before, t_after+dt_eff, dt_eff)
            else:
                t_bins = np.arange(0, 26 + dt_eff, dt_eff)
            t_bin_centers = t_bins[:-1] + dt_eff/2
            bin_idx = np.digitize(all_epochs['t'], t_bins) - 1
            valid = (bin_idx >= 0) & (bin_idx < len(t_bin_centers))
            
            if pupil_col in ['blink', 'saccade', 'fixation_break']:
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
                if not align:
                    pupil_smooth = scipy.signal.savgol_filter(
                        pupil_smooth, window_length=min(smooth_window, len(pupil_smooth) | 1),
                        polyorder=polyorder)
            
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
                        pupil_smooth,
                        window_length=min(smooth_window, len(pupil_smooth) | 1),
                        polyorder=polyorder,
                        deriv=1,
                        delta=dt_eff
                    )
            if cond not in per_sub_avg[pupil_col]:
                per_sub_avg[pupil_col][cond] = []
            if cond not in per_sub_avg_region[pupil_col]:
                per_sub_avg_region[pupil_col][cond] = []
            per_sub_avg[pupil_col][cond].append(pupil_smooth)
            idx_region = (t_bin_centers >= region_interval[0])*(t_bin_centers <= region_interval[1])
            if pupil_col in ['blink', 'saccade', 'raw_blink', 'raw_saccade'] or 'speed' in pupil_col:
                if align:
                    per_sub_avg_region[pupil_col][cond].append(np.nanmax(pupil_smooth[idx_region]))
                if not align:
                    per_sub_avg_region[pupil_col][cond].append(np.nanmean(pupil_smooth))
            else:
                if align:
                    per_sub_avg_region[pupil_col][cond].append(np.nanmin(pupil_smooth[idx_region]))
                if not align:
                    per_sub_avg_region[pupil_col][cond].append(np.nanmin(pupil_smooth))
    # --- Average across subjects ---
    if align:
        t_bins = np.arange(-t_before, t_after+dt_eff, dt_eff)
    else:
        t_bins = np.arange(0, 26 + dt_eff, dt_eff)
    t_bin_centers = t_bins[:-1] + dt_eff/2
    fig, axes = plt.subplots(1, 1, figsize=(4.5, 3.5))
    if condition == 'pShuffle':
        colormap = ['midnightblue','royalblue','lightskyblue'][::-1]
    if condition == 'regime':
        colormap = ['peru', 'cadetblue']

    if condition == 'pShuffle':
        all_data = np.zeros((len(conditions), len(sublist), len(t_bin_centers)))
    else:
        all_data = [[], []]
    vals_to_iterate = reversed(sorted(per_sub_avg[pupil_col].keys()))
    for i_c, cond in enumerate(vals_to_iterate):
        # reversed loop so order becomes 1 (bistable) and -1 (monostable)
        # Convert to array (subjects x time)
        data_array = np.array(per_sub_avg[pupil_col][cond])
        if condition == 'pShuffle':
            all_data[i_c] = data_array
        else:
            all_data[i_c].append(data_array.T)
        # subject means
        subj_mean = np.nanmean(data_array, axis=1, keepdims=True)
        
        # grand mean
        grand_mean = np.nanmean(data_array)
        
        # normalized data (remove subject offsets)
        data_corr = data_array - subj_mean + grand_mean
        
        # SEM
        grand_avg = np.nanmean(data_array, axis=0)
        grand_error = np.nanstd(data_corr, axis=0, ddof=1) / np.sqrt(data_array.shape[0])
        label_dict = {-1: 'Monostable', 1: 'Bistable'}
        if condition == 'regime':
            label = label_dict.get(cond, str(cond))
        else:
            label = str(cond)
        axes.plot(t_bin_centers, grand_avg, color=colormap[i_c], linewidth=3, label=label)
        axes.fill_between(t_bin_centers, grand_avg-grand_error,
                          grand_avg+grand_error, color=colormap[i_c], alpha=0.3)
    axes.axvline(0, color='k', linestyle='--')
    axes.axhline(0, color='k', linestyle='--')
    if align:
        axes.set_xlabel('Time from switch (s)')
    else:
        axes.set_xlabel('Time (s)')
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    if pupil_col == 'blink' and not align and condition == 'pShuffle':
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'blink_rate.npy')
        np.save(save_path, np.nanmean(all_data, axis=-1))
    if pupil_col == 'speed' and not align and condition == 'pShuffle':
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'average_speed.npy')
        np.save(save_path, np.nanmean(all_data, axis=-1))
    if pupil_col == 'fixation_break' and not align and condition == 'pShuffle':
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'average_fixation_break_rate.npy')
        np.save(save_path, np.nanmean(all_data, axis=-1))
    if pupil_col == 'saccade' and not align and condition == 'pShuffle':
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'saccade_rate.npy')
        np.save(save_path, np.nanmean(all_data, axis=-1))
    if pupil_col == 'raw_blink' and not align and condition == 'pShuffle':
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'raw_blink.npy')
        np.save(save_path, np.nanmean(all_data, axis=-1))
    if pupil_col == 'raw_saccade' and not align and condition == 'pShuffle':
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'raw_saccade.npy')
        np.save(save_path, np.nanmean(all_data, axis=-1))
    pvals = []
    significance = []
    significance_where = []
    for i_t, tim in enumerate(t_bin_centers):
        if condition == 'pShuffle':
            pv = scipy.stats.ttest_rel(all_data[0, :, i_t], all_data[2, :, i_t]).pvalue
        else:
            pv = scipy.stats.ttest_ind(all_data[1][0][i_t], all_data[0][0][i_t], equal_var=False).pvalue
        pvals.append(pv)
        significance.append(pv < 0.05)
        if pv < 0.05:
            significance_where.append(tim)
    # f2, a2 = plt.subplots(1)
    # a2.plot(pvals)
    # a2.set_yscale('log')
    # a2.axhline(0.05)
    if pupil_col == 'saccade':
        y_max = 1.28 # Position above highest confidence
        axes.set_ylim(0.3, 1.3)
    if 'speed' in pupil_col:
        y_max = 0.075 # Position above highest confidence
        axes.set_ylim(-0.075, 0.105)
    if 'Pupil' in pupil_col:
        y_max = 0.25
    if pupil_col == 'blink':
        y_max = 0.9
        axes.set_ylim(0.15, 0.905)
    if pupil_col == 'fixation_break':
        y_max = 0.03
        axes.set_ylim(0.02, 0.06)
    if velocity:
        y_max = 0.225
    for x in significance_where:
        axes.plot([x - dt_eff/2, x+dt_eff/2], [y_max, y_max],
                     color='k', linewidth=4)
    if xy_flag:
        axes.set_ylabel('Distance')
    else:
        if velocity:
            axes.set_ylabel('Pupil size velocity')
        else:
            axes.set_ylabel('Pupil size')
    if pupil_col == 'vergence_angle':
        axes.set_ylabel('Vergence angle (º)')
    if 'speed' in pupil_col:
        axes.set_ylabel('Eye movement speed')
    if pupil_col == 'blink':
        axes.set_ylabel('Blink rate (Hz)')
    if pupil_col == 'fixation_break':
        axes.set_ylabel('Fixation break rate (Hz)')
    if pupil_col == 'saccade':
        axes.set_ylabel('Saccade rate (Hz)')
    if pupil_col == 'raw_blink':
        axes.set_ylabel('Proportion of eyes closed')
    if pupil_col == 'raw_saccade':
        axes.set_ylabel('Proportion of fixation breaks')
    
    if condition == 'pShuffle':
        axes.legend(title='p(shuffle)', frameon=False)
    if condition == 'regime':
        axes.legend(frameon=False)

    fig.tight_layout()
    if save_plot:
        fig.tight_layout()
        for extension in ['.png', '.svg']:
            save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'final_plots', plot_name + extension)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=400, bbox_inches='tight')
    f2, ax2 = plt.subplots(1, figsize=(3.5, 3.5))
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    if condition == 'pShuffle':
        all_data = np.zeros((len(conditions), len(sublist)))
    else:
        all_data = [[], []]
    for i_c, cond in enumerate(reversed(sorted(per_sub_avg_region[pupil_col].keys()))):
        data_array = np.array(per_sub_avg_region[pupil_col][cond])
        if condition == 'pShuffle':
            all_data[i_c] = data_array
        else:
            all_data[i_c].append(data_array)
    if condition == 'pShuffle':
        if 'Pupil' in pupil_col and align:
            save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'min_pupil_across_trials.npy')
            np.save(save_path, all_data)
        label_save_max = 'max' if align else 'baseline'
        if 'saccade' in pupil_col:
            save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', f'{label_save_max}_saccade_rate_across_trials.npy')
            np.save(save_path, all_data)
        if 'blink' in pupil_col:
            save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', f'{label_save_max}_blink_rate_across_trials.npy')
            np.save(save_path, all_data)
        if 'fixation' in pupil_col:
            save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', f'{label_save_max}_fixation_break_rate_across_trials.npy')
            np.save(save_path, all_data)
        if 'speed' in pupil_col:
            save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', f'{label_save_max}_eye_speed_across_trials.npy')
            np.save(save_path, all_data)
        sns.barplot(all_data.T, fill=True, palette=colormap, ax=ax2)
        sns.swarmplot(
                        data=all_data.T,
                        color="black",        # point fill
                        edgecolor="white",    # contrast on dark bars
                        linewidth=0.5,
                        size=3,
                        ax=ax2,
                        zorder=10             # ensures points are on top
                    )
        # sns.swarmplot(all_data.T, color='k', ax=ax2)
        # sns.lineplot(all_data, color='k', ax=ax2, alpha=0.2, linestyle='solid')
        if not velocity:
            if 'Pupil' in pupil_col:
                barh = 0.03
                dhs = [0.08, 0.08, 0.17]
            else:
                barh = 0.03
                dhs = [0.08, 0.08, 0.17]
        if velocity:
            dhs = [0.12, 0.12, 0.18]
        c = 0
        for c1, c2 in zip([0, 1, 0], [1, 2, 2]):
            pval = scipy.stats.ttest_rel(all_data[c1], all_data[c2]).pvalue
            cte = 0.4*('Pupil' not in pupil_col and 'speed' not in pupil_col) + 0.11*('speed' in pupil_col)
            heights = np.max((np.zeros(3), np.nanmean(all_data, axis=1)+cte), axis=0)
            bars = [0, 1, 2]
            print(pval)
            barplot_annotate_brackets(c1, c2, pval, bars, heights,
                                      yerr=None, dh=dhs[c], barh=barh, fs=10,
                                      maxasterix=3, ax=ax2)
            c += 1
    else:
        if 'Pupil' in pupil_col and align:
            save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'min_pupil_across_trials_regime.npy')
            np.save(save_path, all_data)
        # pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
        # fitted_params_all = [np.load(par) for par in pars]
        # j0s = np.array([n*params[1] for params in fitted_params_all])
        # j1s = np.array([n*params[0] for params in fitted_params_all])
        # pshuff_vals = np.tile([1., 0.7, 0.], 35).reshape(3, 35)
        # bis_mono_boolean = (pshuff_vals*j1s+j0s) > 1
        # colormap = ['cadetblue', 'peru']
        # data_bis = all_data[bis_mono_boolean]
        # data_mono = all_data[~bis_mono_boolean]
        # df = pd.DataFrame({
        #         "value": list(data_bis) + list(data_mono),
        #         "group": (["Bistable"] * np.sum(bis_mono_boolean)) + (["Monostable"] * np.sum(~bis_mono_boolean))
        #     })
        # sns.barplot(data=df, x="group", y="value", palette=colormap, ax=ax2,
        #             order=['Monostable', 'Bistable'])
        # sns.swarmplot(data=df,
        #               order=['Monostable', 'Bistable'],
        #               x='group',
        #               y='value',
        #               color="black",        # point fill
        #               edgecolor="white",    # contrast on dark bars
        #               linewidth=0.5,
        #               size=3,
        #               ax=ax2,
        #               zorder=10             # ensures points are on top
        #             )
        # t_stat, pval = scipy.stats.mannwhitneyu(data_bis, data_mono, equal_var=False,
        #                                         alternative='greater')
        # c = 0; c1 = 0; c2 = 1
        # cte = 0.
        # heights = np.max((np.zeros(2), [np.nanmean(data_bis)+cte, np.nanmean(data_mono)+cte]), axis=0)
        # bars = [0, 1]
        # barh = 0.03
        # dhs = [0.08, 0.08, 0.17]
        # barplot_annotate_brackets(c1, c2, pval, bars, heights,
        #                           yerr=None, dh=dhs[c], barh=barh, fs=10,
        #                           maxasterix=3, ax=ax2)
        # c += 1
        colormap = ['cadetblue', 'peru']
        df = pd.DataFrame({
                "value": list(all_data[0][0]) + list(all_data[1][0]),
                "group": (["Bistable"] * len(all_data[0][0])) + (["Monostable"] * len(all_data[1][0]))
            })
        sns.barplot(data=df, x="group", y="value", palette=colormap, ax=ax2,
                    order=['Monostable', 'Bistable'])
        sns.swarmplot(data=df,
                      order=['Monostable', 'Bistable'],
                      x='group',
                      y='value',
                      color="black",        # point fill
                      edgecolor="white",    # contrast on dark bars
                      linewidth=0.5,
                      size=3,
                      ax=ax2,
                      zorder=10             # ensures points are on top
                    )
        pval = scipy.stats.mannwhitneyu(all_data[1][0], all_data[0][0], equal_var=False).pvalue
        c = 0; c1 = 0; c2 = 1
        # plt.figure()
        # sns.kdeplot(all_data[0][0], color='cadetblue')
        # sns.kdeplot(all_data[1][0], color='peru')
        cte = 0.4*('Pupil' not in pupil_col and 'speed' not in pupil_col) + 0.11*('speed' in pupil_col)
        heights = np.max((np.zeros(2), [np.nanmean(all_data[0][0])+cte, np.nanmean(all_data[1][0])+cte]), axis=0)
        bars = [0, 1]
        if not velocity:
            if 'Pupil' in pupil_col:
                barh = 0.03
                dhs = [0.08, 0.08, 0.17]
            else:
                barh = 0.03
                dhs = [0.08, 0.08, 0.17]
        if velocity:
            dhs = [0.12, 0.12, 0.18]
        barplot_annotate_brackets(c1, c2, pval, bars, heights,
                                  yerr=None, dh=dhs[c], barh=barh, fs=10,
                                  maxasterix=3, ax=ax2)
        c += 1
        sns.swarmplot(data=df, x="group", y="value", color='k', ax=ax2)
    if condition == 'pShuffle':
        ax2.set_xticks([0, 1, 2], [0., 0.7, 1.][::-1])
        ax2.set_xlabel('p(Shuffle)')
    if condition == 'regime':
        ax2.set_xticks([0, 1], ['Monostable', 'Bistable'])
        ax2.set_xlabel('')
    if "Pupil" in pupil_col:
        ax2.set_ylabel('Minimum pupil')
    if pupil_col == 'blink' and align:
        ax2.set_ylabel('Maximum blink rate (Hz)')
    if pupil_col == 'blink' and not align:
        ax2.set_ylabel('Baseline blink rate (Hz)')
    if pupil_col == 'fixation_break' and align:
        ax2.set_ylabel('Maximum fixation break rate (Hz)')
    if pupil_col == 'fixation_break' and not align:
        ax2.set_ylabel('Baseline fixation break rate (Hz)')
    if pupil_col == 'saccade' and align:
        ax2.set_ylabel('Maximum saccade rate (Hz)')
    if pupil_col == 'saccade' and not align:
        ax2.set_ylabel('Baseline saccade rate (Hz)')
    if pupil_col == 'raw_blink':
        ax2.set_ylabel('Maximum proportion of closed eye')
    if pupil_col == 'speed':
        ax2.set_ylabel('Maximum eye speed')
    if pupil_col == 'raw_saccade':
        ax2.set_ylabel('Maximum proportion of eye in saccade')
    if save_plot:
        f2.tight_layout()
        if align:
            align_label = ''
        else:
            align_label = 'full_time_'
        plot_name = align_label + condition + null_appendix + '_' + pupil_col + '_' 'average_window_v2.png'
        if velocity:
            plot_name = 'velocity_' + plot_name
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'final_plots', plot_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        f2.savefig(save_path, dpi=400, bbox_inches='tight')
        plot_name = align_label + condition + null_appendix + '_' + pupil_col + '_' + 'average_window_v2.svg'
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'final_plots', plot_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        f2.savefig(save_path, dpi=400, bbox_inches='tight')


def plot_bars_eye_tracker(data_folder=DATA_FOLDER,
                          condition='pShuffle', n=4,
                          pupil_col='Pupil_residual',
                          align=True, measure='min',
                          ax=None):
    all_data = get_eye_tracker_data_across_trials(data_folder=DATA_FOLDER,
                                                  pupil_col=pupil_col,
                                                  align=align, flatten=False,
                                                  measure=measure)
    if pupil_col == 'Pupil_residual':
        ylabel = 'Min. pupil at switch'
    if pupil_col in ['saccade', 'blink', 'fixation_break']:
        lab = 'fixation break' if pupil_col == 'fixation_break' else pupil_col
        ylabel = f'Max. {lab} rate \nat switch' if align else f'Baseline {lab} rate'
    if pupil_col == 'speed':
        ylabel = 'Max. eye speed at switch' if align else 'Baseline eye speed'
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(ncols=1, figsize=(3.2, 3))
        created_fig = True
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    print(pupil_col)
    if condition == 'pShuffle':
        colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
        sns.barplot(all_data.T, fill=True, palette=colormap, ax=ax)
        sns.swarmplot(
                        data=all_data.T,
                        color="black",        # point fill
                        edgecolor="white",    # contrast on dark bars
                        linewidth=0.5,
                        size=3,
                        ax=ax,
                        zorder=10             # ensures points are on top
                    )
        # sns.swarmplot(all_data.T, color='k', ax=ax2)
        # sns.lineplot(all_data, color='k', ax=ax2, alpha=0.2, linestyle='solid')
        barh = 0.03
        dhs = [0.08, 0.08, 0.17]
        c = 0
        for c1, c2 in zip([0, 1, 0], [1, 2, 2]):
            pval = scipy.stats.ttest_rel(all_data[c1], all_data[c2]).pvalue
            cte = 0.
            heights = np.max((np.zeros(3), np.nanmean(all_data, axis=1)+cte), axis=0)
            bars = [0, 1, 2]
            print(pval)
            barplot_annotate_brackets(c1, c2, pval, bars, heights,
                                      yerr=None, dh=dhs[c], barh=barh, fs=10,
                                      maxasterix=3, ax=ax, raw_p=False)
            c += 1
    else:
        pars = glob.glob(SV_FOLDER + 'fitted_params/ndt/' + '*.npy')
        fitted_params_all = [np.load(par) for par in pars]
        j0s = np.array([n*params[1] for params in fitted_params_all])
        j1s = np.array([n*params[0] for params in fitted_params_all])
        pshuff_vals = np.tile([1., 0.7, 0.], 35).reshape(3, 35)
        bis_mono_boolean = (pshuff_vals*j1s+j0s) > 1
        colormap = ['cadetblue', 'peru']
        data_bis = all_data[bis_mono_boolean]
        data_mono = all_data[~bis_mono_boolean]
        df = pd.DataFrame({
                "value": list(data_bis) + list(data_mono),
                "group": (["Bistable"] * np.sum(bis_mono_boolean)) + (["Monostable"] * np.sum(~bis_mono_boolean))
            })
        sns.barplot(data=df, x="group", y="value", palette=colormap, ax=ax,
                    order=['Monostable', 'Bistable'])
        sns.swarmplot(data=df,
                      order=['Monostable', 'Bistable'],
                      x='group',
                      y='value',
                      color="black",        # point fill
                      edgecolor="white",    # contrast on dark bars
                      linewidth=0.5,
                      size=3,
                      ax=ax,
                      zorder=10             # ensures points are on top
                    )
        alternative = 'less' if 'Pupil' in pupil_col else 'greater'
        t_stat, pval = scipy.stats.ttest_ind(data_bis, data_mono, equal_var=False,
                                             alternative=alternative)
        print(pval)
        c = 0; c1 = 0; c2 = 1
        cte = 0.
        heights = np.max((np.zeros(2), [np.nanmean(data_bis)+cte, np.nanmean(data_mono)+cte]), axis=0)
        bars = [0, 1]
        barh = 0.03
        dhs = [0.08, 0.08, 0.17]
        barplot_annotate_brackets(c1, c2, pval, bars, heights,
                                  yerr=None, dh=dhs[c], barh=barh, fs=10,
                                  maxasterix=3, ax=ax, raw_p=True)
        c += 1
    ax.set_xlabel('')
    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    if created_fig:
        fig.tight_layout()
        for termination in ['.png', '.svg', '.pdf']:
            plot_name = f'clean_{pupil_col}_' + condition + '_' + f'average_window{termination}'
            save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'final_plots', 'bars_all',  plot_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=400, bbox_inches='tight')


def plot_eye_tracker_multipanel(data_folder=DATA_FOLDER,
                                measure='min',
                                save=True):

    # ALIGN = TRUE (2 x 5)

    pupil_cols_align = [
        'speed',
        'blink',
        'saccade',
        'Pupil_residual'
    ]

    conditions_align = ['pShuffle', 'regime']

    fig1, axs1 = plt.subplots(
        2, len(pupil_cols_align),
        figsize=(2.5*len(pupil_cols_align), 6),
        constrained_layout=True
    )
    
    fig1.tight_layout()

    for row, condition in enumerate(conditions_align):
        for col, pupil_col in enumerate(pupil_cols_align):

            plot_bars_eye_tracker(
                data_folder=data_folder,
                condition=condition,
                n=4,
                pupil_col=pupil_col,
                align=True,
                measure=measure,
                ax=axs1[row, col]
            )

    if save:
        for termination in ['.png', '.svg', '.pdf']:
            save_path = os.path.join(
                data_folder,
                'aligned_eye_tracker_data',
                'plots',
                'final_plots',
                'bars_all',
                f'eye_tracker_align_true_multipanel{termination}'
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig1.savefig(save_path, dpi=400, bbox_inches='tight')

    # ALIGN = FALSE (2 x 4)

    pupil_cols_noalign = [
        'speed',
        'blink',
        'saccade',
    ]

    conditions_noalign = ['pShuffle', 'regime']

    fig2, axs2 = plt.subplots(
        2, len(pupil_cols_noalign),
        figsize=(2.5*len(pupil_cols_noalign), 6),
        constrained_layout=True
    )
    fig2.tight_layout()

    for row, condition in enumerate(conditions_noalign):
        for col, pupil_col in enumerate(pupil_cols_noalign):

            plot_bars_eye_tracker(
                data_folder=data_folder,
                condition=condition,
                n=4,
                pupil_col=pupil_col,
                align=False,
                measure=measure,
                ax=axs2[row, col]
            )

    if save:
        for termination in ['.png', '.svg', '.pdf']:
            save_path = os.path.join(
                data_folder,
                'aligned_eye_tracker_data',
                'plots',
                'final_plots',
                'bars_all',
                f'eye_tracker_align_false_multipanel{termination}'
            )
            fig2.savefig(save_path, dpi=400, bbox_inches='tight')

    return fig1, fig2


def plot_xy_heatmap_fixation(data_folder=DATA_FOLDER, sub='s_1'):
    path_eye = os.path.join(
        data_folder,
        'aligned_eye_tracker_data',
        f'{sub}_aligned_Gaze_Data.csv'
    )
    eyetracker_data = pd.read_csv(path_eye)
    x_right = eyetracker_data['RightEye_X']
    y_right = eyetracker_data['RightE_Y']
    x_left = eyetracker_data['LeftEye_X']
    y_left = eyetracker_data['LeftEye_Y']
    x = np.nanmean([x_left, x_right], axis=0)
    y = np.nanmean([y_left, y_right], axis=0)
    x_centered = x - 0.5
    y_centered = y - 0.5
    idx_non_nan = (~np.isnan(x_centered))*(~np.isnan(y_centered))
    x_centered = x_centered[idx_non_nan]
    y_centered = y_centered[idx_non_nan]
    heatmap, xedges, yedges = np.histogram2d(x_centered, y_centered, bins=100,
                                             range=[[-1, 1], [-1, 1]], normed=True)
    plt.figure()
    plt.imshow(heatmap.T, origin='lower', cmap='hot',
               extent = [-1, 1, -1, 1],
               aspect='auto')
    plt.xlabel('X (a.u.)')
    plt.ylabel('Y (a.u.)')
    plt.title('Gaze heatmap')
    plt.tight_layout()
    plt.colorbar(label='Gaze density')
    plt.show()


def plot_xy_heatmap_switch(data_folder=DATA_FOLDER, pshuffle='all',
                           t_before=1.0, t_after=1.0, dt=1/60,
                           sublist=None, noisy=False, freq=2):
    if noisy:
        df_all = load_data(data_folder=data_folder + '/noisy/', n_participants='all', filter_subjects=True)
    else:
        df_all = load_data(data_folder=data_folder, n_participants='all', filter_subjects=True)
        if freq != 'all':
            df_all = df_all.loc[df_all.freq == freq]
    if sublist is None:
        sublist = df_all.subject.unique()

    # --- Helper functions ---
    def get_switch_indices(responses):
        responses = np.array(responses)
        return np.where(responses[1:] != responses[:-1])[0] + 1

    heatmaps_per_cond = {}

    for sub in sublist:
        df = df_all.loc[df_all.subject == sub]
        if pshuffle != 'all':
            df = df.loc[df.pShuffle == pshuffle]
        path_eye = os.path.join(data_folder, 'aligned_eye_tracker_data', f'{sub}_aligned_Gaze_Data.csv')
        if not os.path.exists(path_eye):
            continue
        trial_data = pd.read_csv(path_eye)

        conditions = np.sort(df['pShuffle'].unique())

        for cond in conditions:
            df_cond = df[df['pShuffle'] == cond]
            # --- Collect all gaze points around switches ---
            x_all = []
            y_all = []

            for ti in df_cond.trial_index.unique():
                trial_beh = df_cond[df_cond.trial_index == ti]
                trial_eye = trial_data[trial_data.trial_index == ti]
                if trial_eye.empty:
                    continue

                # responses
                if noisy:
                    trial_resp = trial_beh.responses.values
                else:
                    trial_resp = get_response_array(trial_beh)

                switch_idx = get_switch_indices(trial_resp)
                for swi in switch_idx:
                    t_switch = swi * dt
                    mask = (trial_eye['t_trial'] >= t_switch - t_before) & (trial_eye['t_trial'] <= t_switch + t_after)
                    if mask.sum() == 0:
                        continue

                    x_trial = np.nanmean([trial_eye.loc[mask, 'LeftEye_X'],
                                          trial_eye.loc[mask, 'RightEye_X']], axis=0)
                    y_trial = np.nanmean([trial_eye.loc[mask, 'LeftEye_Y'],
                                          trial_eye.loc[mask, 'RightE_Y']], axis=0)

                    x_all.append(x_trial)
                    y_all.append(y_trial)

            if len(x_all) == 0:
                continue

            # --- Concatenate all trials for this condition ---
            x_all = np.concatenate(x_all)
            y_all = np.concatenate(y_all)
            idx_valid = (~np.isnan(x_all)) & (~np.isnan(y_all))
            x_all = x_all[idx_valid]
            y_all = y_all[idx_valid]

            # --- Compute heatmap ---
            heatmap, xedges, yedges = np.histogram2d(x_all, y_all, bins=100, range=[[0,1],[0,1]])
            heatmaps_per_cond[cond] = heatmaps_per_cond.get(cond, []) + [heatmap]
    # --- Average across subjects ---
    avg_heatmaps = {}
    for cond, heatmaps in heatmaps_per_cond.items():
        avg_heatmaps[cond] = np.mean(np.stack(heatmaps), axis=0)

    nconds = len(avg_heatmaps)
    fig, axes = plt.subplots(1, nconds, figsize=(5*nconds, 5))

    if nconds == 1:
        axes = [axes]

    for i, (cond, heatmap) in enumerate(sorted(avg_heatmaps.items())):
        ax = axes[i]
        im = ax.imshow(
            heatmap.T, origin='lower', cmap='hot',
            extent=[0,1,0,1], aspect='equal'
        )
        ax.set_title(f'Condition: {cond}')
        ax.set_xlabel('X gaze')
        ax.set_ylabel('Y gaze')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Gaze density')

    plt.tight_layout()
    plt.show()


def plot_gaze_heatmap_video(
    data_folder=DATA_FOLDER,
    sublist=['s_1'],
    pshuffle='all',
    dt=1/60,
    t_before=1.0,
    t_after=1.0,
    bins=100,
    smooth_sigma=1,
    save_name='gaze_video.gif',
    fps=30,
    condition='pShuffle',
    freq=2,
    noisy=False
):
    """
    Plot and save a heatmap video of gaze over time aligned to perceptual switches.
    Averaging across subjects after computing per-subject heatmaps.
    """
    # --- Load behavioral data ---
    if noisy:
        df_subs = load_data(data_folder=data_folder + '/noisy/', n_participants='all', filter_subjects=True)
    else:
        df_subs = load_data(data_folder=data_folder, n_participants='all', filter_subjects=True)
        if freq != 'all':
            df_subs = df_subs.loc[df_subs.freq == freq]
    if sublist is None:
        sublist = df_subs.subject.unique()
    if not noisy:
        save_path = data_folder + '/aligned_eye_tracker_data/plots/' f'hysteresis_pshuf_{pshuffle}_freq_{freq}_' + save_name
    if noisy: 
        save_path = data_folder + '/aligned_eye_tracker_data/plots/' f'noise_pshuf_{pshuffle}_' + save_name
    # Time frames
    t_frames = np.arange(-t_before, t_after, dt)
    n_frames = len(t_frames)

    # Storage per subject
    subject_heatmaps = []

    # --- Loop over subjects ---
    for sub in sublist:
        df_all = df_subs.loc[df_subs.subject == sub]
        if pshuffle != 'all':
            df_all = df_all[df_all['pShuffle'] == pshuffle]

        # Load eye-tracker data
        path_eye = os.path.join(data_folder, 'aligned_eye_tracker_data', f'{sub}_aligned_Gaze_Data.csv')
        if not os.path.exists(path_eye):
            continue
        eye_data = pd.read_csv(path_eye)

        # Initialize heatmaps per frame for this subject
        heatmaps = np.zeros((n_frames, bins, bins))

        n_epochs = 0
        for trial_idx in df_all.trial_index.unique():
            trial_beh = df_all[df_all.trial_index == trial_idx]
            trial_eye = eye_data[eye_data.trial_index == trial_idx]
            if trial_eye.empty:
                continue

            # Responses
            if noisy:
                trial_resp = trial_beh.responses.values
            else:
                trial_resp = get_response_array(trial_beh)

            switch_idx = np.where(np.diff(trial_resp) != 0)[0] + 1
            for swi in switch_idx:
                t_switch = swi * dt
                mask = (trial_eye['t_trial'] >= t_switch - t_before) & (trial_eye['t_trial'] <= t_switch + t_after)
                if mask.sum() == 0:
                    continue

                # Average eyes
                x = np.nanmean([trial_eye['LeftEye_X'][mask], trial_eye['RightEye_X'][mask]], axis=0)
                y = np.nanmean([trial_eye['LeftEye_Y'][mask], trial_eye['RightE_Y'][mask]], axis=0)
                t_rel = trial_eye['t_trial'][mask].values - t_switch

                # Interpolate to uniform frames
                x_interp = np.interp(t_frames, t_rel, x)
                y_interp = np.interp(t_frames, t_rel, y)

                # Compute heatmap per frame
                for i_frame in range(n_frames):
                    valid = (~np.isnan(x_interp[i_frame])) & (~np.isnan(y_interp[i_frame]))
                    if valid:
                        h, _, _ = np.histogram2d(
                            [x_interp[i_frame]], [y_interp[i_frame]], bins=bins, range=[[0,1],[0,1]]
                        )
                        heatmaps[i_frame] += h
                n_epochs += 1

        if n_epochs > 0:
            # Normalize per subject
            heatmaps /= n_epochs
            # Smooth
            # heatmaps = gaussian_filter(heatmaps, sigma=(0, smooth_sigma, smooth_sigma))
            subject_heatmaps.append(heatmaps)

    # --- Average across subjects ---
    avg_heatmaps = np.mean(subject_heatmaps, axis=0)

    # --- Plot video ---
    fig, ax = plt.subplots(figsize=(6,6))
    vmin, vmax = 0, np.max(avg_heatmaps)

    def update(frame_idx):
        ax.clear()
        im = ax.imshow(avg_heatmaps[frame_idx].T, origin='lower', cmap='hot',
                       extent=[0,1,0,1], vmin=vmin, vmax=vmax)
        ax.set_title(f'Time: {t_frames[frame_idx]:.2f}s, p(shuffle)={pshuffle}', fontsize=12)
        ax.set_xlabel('X gaze')
        ax.set_ylabel('Y gaze')
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)

    ani.save(save_path, fps=fps, dpi=150, writer='pillow')
    plt.close(fig)
    print(f'Video saved to {save_path}')


def eye_tracker_correlates(data_folder=DATA_FOLDER, min_pupil=True, freq=2,
                           average_trials=True):
    if min_pupil:
        pre_label = 'min_'
    else:
        pre_label = 'max_'
    if freq is None:
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', pre_label+'pupil_noisy_trials.npy')
        pupil = np.load(save_path)
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', pre_label+'saccade_noisy_trials.npy')
        saccades = np.load(save_path)
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', pre_label+'blink_noisy_trials.npy')
        blinks = np.load(save_path)
    if freq is not None:
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', pre_label+f'pupil_hysteresis_trials_freq_{freq}.npy')
        pupil = np.load(save_path)
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', pre_label+f'blink_hysteresis_trials_freq_{freq}.npy')
        blinks = np.load(save_path)
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', pre_label+f'saccade_hysteresis_trials_freq_{freq}.npy')
        saccades = np.load(save_path)
    if average_trials:
        df = pd.DataFrame({pre_label+'blinks': np.nanmean(blinks, axis=0), pre_label+'saccades': np.nanmean(saccades, axis=0), pre_label+'pupil': np.nanmean(pupil, axis=0)})
    else:
        df = pd.DataFrame({pre_label+'blinks': blinks.flatten(), pre_label+'saccades': saccades.flatten(), pre_label+'pupil': pupil.flatten()})
    g = sns.pairplot(df)
    g.map_lower(corrfunc)


def plot_full_behavioral_vs_eye_tracker(data_folder=DATA_FOLDER, specific=True,
                                        mean=False, mask=False, include_h4=False):
    if specific:
        corr_matrix, p_matrix, annot = \
            get_corr_p_matrix_specific_trials(data_folder=data_folder,
                                              mean=mean, include_h4=include_h4)
    else:
        corr_matrix, p_matrix, annot = \
            get_corr_p_matrix_all_trials(data_folder=data_folder,
                                         mean=mean, include_h4=include_h4)
            
    alpha = 0.05
    fig, ax = plt.subplots(1, figsize=(8, 6))
    labs = ['Dominance', 'Hysteresis 1-cycle', 'Hysteresis 2-cycle', 'Fitted J=J0+J1', r'Fitted $\sigma$', 'Fitted B1',
     'Min. pupil', 'FB baseline', 'FB max',
     'sacc. baseline', 'sacc. max.',
     'speed base.', 'speed max.',
     'blink baseline', 'blink max.']
    labs_reduced = ['Dom.', 'Hyst. 1-c', 'Hyst. 2-c', 'Fit J', r'Fitted $\sigma$', 'Fit B1',
                    'Min. pupil',
                    'FB baseline', 'FB max',
                    'sacc. base.', 'sacc. max.', 'speed base.', 'speed max.',
                    'blink base.', 'blink max.']
    if not include_h4:
        labs.pop(2)
        labs_reduced.pop(2)
    ax.set_xticks(np.arange(len(labs)), labs, rotation=45)
    ax.set_yticks(np.arange(len(labs)), labs, rotation=0)
    fig.tight_layout()
    mask_mat = p_matrix > alpha if mask else False
    if not mask:
        corr_matrix *= np.tri(*corr_matrix.shape, k=-1)
        corr_matrix[corr_matrix == 0] = np.nan
        npars = len(labs)
        ax.step(np.arange(0, npars+1), np.arange(0, npars+1), color='k',
                linewidth=.7)
    sns.heatmap(corr_matrix, cmap="bwr",
                vmin=-1, vmax=1,
                mask=mask_mat,
                cbar_kws={"label": "Correlation"},
                annot=annot, fmt="")
    ax.set_xticks(np.arange(len(labs))+0.5, labs_reduced, rotation=45)
    ax.set_yticks(np.arange(len(labs))+0.5, labs, rotation=0)
    fig.savefig(SV_FOLDER + 'corr_matrix_all.png', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'corr_matrix_all.pdf', dpi=400, bbox_inches='tight')


def plot_behavioral_vs_eye_tracker(data_folder=DATA_FOLDER, var_beh='hyst2', var_eye='pupil',
                                   min_pupil=True,
                                   magnitude_eye='mean_',
                                   return_vars=False, align=False, full=False):
    if var_eye == 'blink':
        xlabel = 'Base. blink rate (Hz)' if min_pupil else 'Max. blink rate (Hz)'
    if var_eye == 'fixation_break':
        xlabel = 'Base. FB rate (Hz)' if min_pupil else 'Max. FB rate (Hz)'
    if var_eye == 'saccade':
        xlabel = 'Base. saccade rate (Hz)' if min_pupil else 'Max. saccade rate (Hz)'
    if var_eye == 'speed':
        xlabel = 'Base. speed' if min_pupil else 'Max. speed'
    if var_eye == 'pupil':
        xlabel = 'Min. pupil' if min_pupil else 'Max. pupil peak'
        # average across trials
        # save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'min_pupil.npy')
        # variable_eye = np.load(save_path)
    align_label = 'aligned_' if align else 'full_time_'
    if return_vars:
        pre_label = magnitude_eye
    else:
        if min_pupil:
            if align:
                pre_label = align_label + 'min_'
            else:
                pre_label = align_label + 'mean_'
        else:
            pre_label = align_label + 'max_'
    if var_beh in ['dominance', 'amplitude', 'latency']:
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', pre_label+f'{var_eye}_noisy_trials.npy')
        variable_eye = np.load(save_path)
    if var_beh == 'hyst2':
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', pre_label+f'{var_eye}_hysteresis_trials_freq_2.npy')
        variable_eye = np.load(save_path)
    if var_beh == 'hyst4':
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', pre_label+f'{var_eye}_hysteresis_trials_freq_4.npy')
        variable_eye = np.load(save_path)
    if var_beh == 'dominance':
        variable = np.load(DATA_FOLDER + 'mean_number_switches_per_subject.npy')
        ylabel = 'Dominance (s)'
    if var_beh == 'amplitude':
        variable = np.load(DATA_FOLDER + 'mean_peak_amplitude_per_subject.npy')
        ylabel = 'Noise peak amplitude'
    if var_beh == 'hyst2':
        variable = np.load(DATA_FOLDER + 'hysteresis_width_freq_2.npy')
        ylabel = 'Hysteresis, 1-C'
    if var_beh == 'hyst4':
        variable = np.load(DATA_FOLDER + 'hysteresis_width_freq_4.npy')
        ylabel = 'Hysteresis, 2-C'
    if return_vars:
        return variable, variable_eye
    if full:
        fig, ax = plt.subplots(ncols=5, figsize=(14, 3), sharex=True, sharey=True)
    else:
        fig, ax = plt.subplots(ncols=1, figsize=(2.6, 2.4), sharex=True, sharey=True)
        ax = [ax]
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    labels = ['p(shuffle)=1', 'p(shuffle)=0.7', 'p(shuffle)=0', 'All', 'Average across trials']
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    for i_a, a in enumerate(ax):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        if full:
            if i_a < 3:
                r, p = pearsonr(variable_eye[i_a], variable[i_a])
                a.annotate(f'r = {r:.3f}\np={p:.2e}', xy=(.04, 0.8), xycoords=a.transAxes)
                a.plot(variable_eye[i_a], variable[i_a], color=colormap[i_a],
                       marker='o', linestyle='')
            a.set_title(labels[i_a], fontsize=14)
            if i_a == 3:
                avg_eye = variable_eye.flatten()
                avg_beh = variable.flatten()
                r, p = pearsonr(avg_eye, avg_beh)
                a.annotate(f'r = {r:.3f}\np={p:.2e}', xy=(.04, 0.8), xycoords=a.transAxes)
                a.plot(avg_eye, avg_beh, marker='o', color='k',
                       linestyle='')
        if i_a == 4 or not full:
            avg_eye = np.nanmean(variable_eye, axis=0)
            avg_beh = np.nanmean(variable, axis=0)
            X = avg_eye.reshape(-1, 1)
            y = avg_beh.reshape(-1, 1)
            linreg = LinearRegression(fit_intercept=True).fit(X, y)
            minmax_array = np.array([np.min(X)-0.1, np.max(X)+0.1]).reshape(-1, 1)
            pred_y = linreg.predict(minmax_array)
            r, p = pearsonr(avg_eye, avg_beh)
            a.annotate(f'r = {r:.3f}\np={p:.2e}', xy=(.04, 0.8), xycoords=a.transAxes)
            a.plot(minmax_array,
                   pred_y, color='gray', linestyle='--', alpha=0.4, linewidth=3)
            a.plot(avg_eye, avg_beh, marker='o', color='k',
                   linestyle='')
    fig.tight_layout()
    # now the \Delta = (pupil(pShuffle = 0) - pupil(pShuffle = 1))
    # delta_beh = variable[-1]-variable[0]
    # delta_eye = variable_eye[-1]-variable_eye[0]
    # r, p = pearsonr(delta_eye, delta_beh)
    # f, a = plt.subplots(1, figsize=(4, 3.5))
    # a.spines['right'].set_visible(False)
    # a.spines['top'].set_visible(False)
    # a.annotate(f'r = {r:.3f}\np={p:.2e}', xy=(.04, 0.8), xycoords=a.transAxes)
    # a.plot(delta_eye, delta_beh, marker='o', color='k',
    #        linestyle='')
    # a.set_xlabel(r'$\Delta$' + ' pupil')
    # a.set_ylabel(r'$\Delta$' + ' behavior')
    # f.tight_layout()
    
    for extension in ['.png', '.svg']:
        plot_name = 'behavioral_correlates_'+ var_beh + '_' + pre_label + var_eye + extension
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'behavioral_correlates',  plot_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=400, bbox_inches='tight')


def plot_histograms_saccades(data_folder=DATA_FOLDER):
    nsubs = 39
    for sub_idx in range(1, nsubs+1):
        sub = f's_{sub_idx}'
        path_eye = os.path.join(data_folder, 'aligned_eye_tracker_data', f'{sub}_aligned_Gaze_Data.csv')
        data_eyetracker = pd.read_csv(path_eye)
        data_eyetracker_valid = data_eyetracker.loc[(data_eyetracker.t_trial >= 0) & (data_eyetracker.t_trial <= 26)]
        saccades = data_eyetracker_valid['saccade'].values
        fixation = data_eyetracker_valid['fixation_break'].values
        x_pos_right = data_eyetracker_valid['RightEye_X'].values
        x_pos_left = data_eyetracker_valid['LeftEye_X'].values
        y_pos_right = data_eyetracker_valid['RightE_Y'].values
        y_pos_left = data_eyetracker_valid['LeftEye_Y'].values
        x = np.nanmean(np.c_[x_pos_left, x_pos_right], axis=1)
        y = np.nanmean(np.c_[y_pos_left, y_pos_right], axis=1)
        s = saccades.astype(int)
        onsets = np.where((s[1:] == 1) & (s[:-1] == 0))[0] + 1
        offsets = np.where((s[1:] == 0) & (s[:-1] == 1))[0] + 1
        if s[0] == 1:
            onsets = np.r_[0, onsets]
        if s[-1] == 1:
            offsets = np.r_[offsets, len(s) - 1]
        valid = (
            ~np.isnan(x[onsets]) &
            ~np.isnan(y[onsets]) &
            ~np.isnan(x[offsets]) &
            ~np.isnan(y[offsets])
        )
        
        onsets = onsets[valid]
        offsets = offsets[valid]
        dx = x[offsets] - x[onsets]
        dy = y[offsets] - y[onsets]
        
        amplitude = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        dt = np.median(np.diff(data_eyetracker_valid['t_trial'].values))
        duration = (offsets - onsets) * dt
        saccade_df = pd.DataFrame({
                                'onset_idx': onsets,
                                'offset_idx': offsets,
                                'dx': dx,
                                'dy': dy,
                                'amplitude': amplitude,
                                'angle': angle,
                                'duration': duration
                                })
        t = data_eyetracker_valid['t_trial'].values
        saccade_df['t_onset'] = t[onsets]
        fig = plt.figure(figsize=(5,5))
        plt.hist2d(
            saccade_df['dx'],
            saccade_df['dy'],
            bins=60,
            density=True
        )
        plt.axhline(0, color='k', lw=0.5)
        plt.axvline(0, color='k', lw=0.5)
        plt.axis('equal')
        plt.colorbar(label='Density')
        plt.title('2D histogram of saccade vectors')
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'saccade_histograms', f'{sub}_saccade_vectors_histogram.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)
        fig = plt.figure(figsize=(5,5))
        ax = plt.subplot(111, projection='polar')
        ax.hist(
            saccade_df['angle'],
            bins=36,
            density=True
        )
        ax.set_title('Saccade direction distribution')
        plt.show()
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'saccade_histograms', f'{sub}_saccade_vectors_direction_histogram.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)
        fig, ax = plt.subplots(ncols=3, figsize=(10,5))
        ax[0].hist(
            saccade_df['amplitude'],
            bins=36,
            density=True
        )
        ax[0].set_title('Saccade distance (end-start)')
        ax[1].hist(
            saccade_df['dx'],
            bins=36,
            density=True
        )
        ax[1].set_title('Saccade x-distance (end-start)')
        ax[2].hist(
            saccade_df['dy'],
            bins=36,
            density=True
        )
        ax[2].set_title('Saccade y-distance (end-start)')
        plt.show()
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'saccade_histograms', f'{sub}_saccade_distance_histogram.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)
        fig, ax = plt.subplots(ncols=1, figsize=(5,5))
        all_df = pd.DataFrame({'x': x,
                               'y': y,
                               'amplitude': np.sqrt((x-0.5)**2+(y-0.5)**2),
                               'fixation': fixation,
                               'saccade': saccades
                                })
        sns.histplot(data=all_df, x='amplitude', hue='fixation', multiple='stack')
        plt.show()
        save_path = os.path.join(data_folder, 'aligned_eye_tracker_data','plots', 'saccade_histograms', f'{sub}_all_distance_histogram.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)


def count_blinks_saccades_per_condition(data_folder=DATA_FOLDER, condition='pShuffle'):
    nsubs = 39
    num_saccades_all_subs = []
    num_blinks_all_subs = []
    num_switch_all_subs = []
    for sub_idx in range(1, nsubs+1):
        sub = f's_{sub_idx}'
        path_eye = os.path.join(data_folder, 'aligned_eye_tracker_data', f'{sub}_aligned_Gaze_Data.csv')
        data_eyetracker = pd.read_csv(path_eye)
        data_eyetracker_valid = data_eyetracker.loc[(data_eyetracker.t_trial >= 0) & (data_eyetracker.t_trial <= 26)]
        avg_saccades = data_eyetracker_valid.groupby(condition)['saccade'].sum()
        avg_blinks = data_eyetracker_valid.groupby(condition)['blink'].sum()
        avg_switch = data_eyetracker_valid.groupby(condition)['fixation_break'].sum()
        num_saccades_all_subs.append(avg_saccades)
        num_blinks_all_subs.append(avg_blinks)
        num_switch_all_subs.append(avg_switch)
    fig, ax = plt.subplots(ncols=3, figsize=(10, 4.5))
    ylims = [[400, 900], [600, 1100], [1000, 3200]]
    for i_a, a in enumerate(ax):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylim(ylims[i_a])
    colormap = ['midnightblue', 'royalblue', 'lightskyblue']
    blinks = pd.concat(num_blinks_all_subs).reset_index()
    saccades = pd.concat(num_saccades_all_subs).reset_index()
    switches = pd.concat(num_switch_all_subs).reset_index()
    sns.barplot(blinks, y='blink', hue=condition, palette=colormap,
                ax=ax[0], errorbar='se', legend=False)
    sns.barplot(saccades, y='saccade', hue=condition, palette=colormap,
                ax=ax[1], errorbar='se', legend=False)
    sns.barplot(switches, y='fixation_break', hue=condition, palette=colormap,
                ax=ax[2], errorbar='se', legend=True)
    ax[2].legend(frameon=False, title='p(shuffle)', loc='upper right',
                 bbox_to_anchor=(1, 1.2))
    ax[0].set_ylabel('Total number of blinks')
    ax[1].set_ylabel('Total number of saccades')
    ax[2].set_ylabel('Fixation breaks')
    fig.tight_layout()
    all_df = pd.concat([switches, blinks['blink'], saccades['saccade']], axis=1)
    subs = np.array([[f's_{sub_idx}']*3 for sub_idx in range(1, nsubs+1)]).flatten()
    all_df['subject'] = subs
    avg_per_sub = all_df.groupby('subject').mean().reset_index()
    new_df = avg_per_sub.drop(condition, axis=1)
    new_df = new_df.drop('subject', axis=1)
    # g = sns.pairplot(all_df, hue='pShuffle', palette=colormap)
    g = sns.pairplot(new_df)
    g.map_lower(corrfunc)


def plot_all_eye_tracker():
    for ds, col in zip([5, 20, 5, 5], ['saccade', 'Pupil_residual', 'blink', 'speed']):
        for align in [False, True]:
            for cond in ['regime', 'pShuffle']:
                plot_pupil_across_all_trials(data_folder=DATA_FOLDER,
                                                  sublist=None,
                                                  n_training=8,
                                                  dt=1/60,
                                                  t_before=4,
                                                  condition=cond,
                                                  t_after=4,
                                                  smooth_window=5,
                                                  polyorder=1,
                                                  pupil_col=col,
                                                  save_plot=True,
                                                  plot_name=f'all_{col}_switch_avg.png',
                                                  velocity=False, n=4,
                                                  downsample_to=ds, null=False, align=align,
                                                  region_interval=[-2, 2])
                plt.close('all')


def plot_pupil_traces():
    #  'blink', 'speed',
    pup_cols = ['speed', 'blink', 'saccade', 'fixation_break']
    for pupil_col in pup_cols:
        for align in [True]:
            for cond in ['regime', 'pShuffle']:
                plot_pupil_across_all_trials(data_folder=DATA_FOLDER,
                                             sublist=None,
                                             n_training=8,
                                             dt=1/60,
                                             t_before=4,
                                             condition=cond,
                                             t_after=4,
                                             smooth_window=5,
                                             polyorder=1,
                                             pupil_col=pupil_col,
                                             save_plot=True,
                                             plot_name=f'all_{pupil_col}_avg_last_vf.png',
                                             velocity=False, n=4,
                                             downsample_to=10, null=False, align=align,
                                             region_interval=[-2, 2])
                # plt.close('all')


if __name__ == '__main__':
    print('Running eye_tracker_analysis.py')
    # Example entry points (uncomment to run):
    # eye_tracker_save_align_data(data_folder=DATA_FOLDER, ntraining=8)
    # plot_pupil_across_subjects_simple(data_folder=DATA_FOLDER)
    # plot_eye_tracker_multipanel(data_folder=DATA_FOLDER, measure='min')
    # plot_full_behavioral_vs_eye_tracker(data_folder=DATA_FOLDER, specific=True,
    #                                     mean=True, mask=False, include_h4=True)
    # plot_all_eye_tracker()
