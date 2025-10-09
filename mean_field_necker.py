# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 08:50:14 2023

@author: alexg
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import itertools
import os
import gibbs_necker as gn
from scipy.optimize import fsolve, curve_fit
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pylab as pl
import matplotlib as mpl
from matplotlib.transforms import Affine2D
from matplotlib.markers import MarkerStyle
import sympy
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import pandas as pd
import cv2
import torch
# import fastkde

mpl.rcParams['font.size'] = 16
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize']= 14
plt.rcParams['ytick.labelsize']= 14
plt.rcParams["axes.grid"] = False


# ---GLOBAL VARIABLES
pc_name = 'alex'
if pc_name == 'alex':
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/mean_field_necker/data_folder/'  # Alex
    # C matrix:
    # c_data = DATA_FOLDER + 'c_mat.npy'
    # C = np.load(c_data, allow_pickle=True)
elif pc_name == 'alex_CRM':
    DATA_FOLDER = 'C:/Users/agarcia/Desktop/phd/necker/data_folder/'  # Alex CRM



# theta matrix
theta = gn.THETA


def mean_field(J, num_iter, sigma=1):
    #initialize random state of the cube
    vec = np.random.rand(8)
    vec_time = np.empty((num_iter, 8))
    vec_time[:] = np.nan
    for i in range(num_iter):
        for q in range(8):
            neighbours = theta[q].astype(dtype=bool)
            vec[q] = gn.sigmoid(2*J*sum(2*vec[neighbours]-1)+np.random.randn()*sigma) 
        vec_time[i, :] = vec
    return vec_time


def mean_field_stim(J, num_iter, stim, sigma=1, theta=theta, val_init=None, sxo=0.):
    #initialize random state of the cube
    if val_init is None:
        vec = np.random.rand(theta.shape[0])
    else:
        vec = np.repeat(val_init, theta.shape[0]) + np.random.randn()*sxo
    vec_time = np.empty((num_iter, theta.shape[0]))
    vec_time[:] = np.nan
    vec_time[0, :] = vec
    for i in range(1, num_iter):
        for q in range(theta.shape[0]):
            neighbours = theta[q].astype(dtype=bool)
            # th_vals = theta[q][theta[q] != 0]
            if isinstance(stim, np.ndarray):
                b = stim[q]
            else:
                b = stim
            vec[q] = gn.sigmoid(2*(np.sum(J*(2*vec[neighbours]-1))+b))+np.random.randn()*sigma
        vec_time[i, :] = vec
    return vec_time


def mean_field_stim_matmul(J, num_iter, stim, sigma=1, theta=theta, val_init=None, sxo=0.):
    #initialize random state of the cube
    if val_init is None:
        vec = np.random.rand(theta.shape[0])
    else:
        vec = np.repeat(val_init, theta.shape[0]) + np.random.randn()*sxo
    vec_time = np.empty((num_iter, theta.shape[0]))
    vec_time[:] = np.nan
    vec_time[0, :] = vec
    for i in range(1, num_iter):
        vec = gn.sigmoid(2*J*np.matmul(theta, 2*vec-1) + 2*stim)+np.random.randn()*sigma
        vec_time[i, :] = vec
    return vec_time


def mean_field_neg_stim_back(j, num_iter, stim, theta=theta, val_init=None):
    #initialize random state of the cube
    if val_init is None:
        vec = np.random.rand(theta.shape[0])
    else:
        vec = np.repeat(val_init, theta.shape[0])
    vec_time = np.empty((num_iter, theta.shape[0]))
    vec_time[:] = np.nan
    vec_time[0, :] = vec
    for i in range(1, num_iter):
        for q in range(theta.shape[0]):
            neighbours = theta[q].astype(dtype=bool)
            # th_vals = theta[q][theta[q] != 0]
            vec[q] = gn.sigmoid(2*(sum(j*(2*vec[neighbours]-1))+stim*(-1)**(q > (theta.shape[0]//2-1))))
        vec_time[i, :] = vec
    return vec_time


def plot_mf_sols(j_list, b, theta=theta, num_iter=100):
    sol1 = np.empty((theta.shape[0], len(j_list)))
    sol2 = np.empty((theta.shape[0], len(j_list)))
    for i_j, j in enumerate(j_list):
        vec_time = mean_field_stim(j, num_iter, b, sigma=0, theta=theta, val_init=0.9)
        valspos = np.max((vec_time[-1], 1-vec_time[-1]), axis=0)
        sol1[:, i_j] = valspos
        sol2[:, i_j] = 1-valspos
    fig, ax = plt.subplots(1)
    for i in range(theta.shape[0]):
        if sum(theta[i]) == 2:
            color = 'r'
        else:
            color = 'k'
        ax.plot(j_list, sol1[i], color=color)
        ax.plot(j_list, sol2[i], color=color)
    ax.set_xlabel('Coupling, J')
    ax.set_ylabel('Approximate posterior, q(x=1)')


def plot_mean_field_neg_stim_fixed_points(j_list, b=0, theta=theta, num_iter=100):
    sol_final = np.empty((theta.shape[0], len(j_list)))
    sol_final_09 = np.empty((theta.shape[0], len(j_list)))
    for i_j, j in enumerate(j_list):
        vec = mean_field_neg_stim_back(j=j, num_iter=num_iter, stim=b, theta=theta, val_init=0.05)
        vec_final = vec[-1, :]
        # vec_final = [np.max((vec_final[i], 1-vec_final[i])) for i in range(theta.shape[0])]
        sol_final[:, i_j] = vec_final
        vec = mean_field_neg_stim_back(j=j, num_iter=num_iter, stim=b, theta=theta, val_init=0.95)
        vec_final = vec[-1, :]
        # vec_final = [np.max((vec_final[i], 1-vec_final[i])) for i in range(theta.shape[0])]
        sol_final_09[:, i_j] = vec_final
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    diff_sols_01 = sol_final[:4] - sol_final[4:]
    diff_sols_01 = (np.mean(sol_final[:4], axis=0) + np.mean(sol_final[4:], axis=0))/2
    diff_sols_09 = (np.mean(sol_final_09[:4], axis=0) + np.mean(sol_final_09[4:], axis=0))/2
    # diff_sols_09 = sol_final_09[:4] - sol_final_09[4:] 
    for i in range(8):
        if i > 3:
            color = 'r'
            label = '- B'
        if i <= 3:
            color = 'k'
            label = '+ B'
        if i == 0 or i == 4:
            ax[0].plot(j_list, sol_final[i, :], color=color,
                       label=label)
        else:
            ax[0].plot(j_list, sol_final[i, :], color=color)
        ax[0].plot(j_list, sol_final_09[i, :], color=color)
    ax[1].plot(j_list, diff_sols_01, color='k')
    ax[1].plot(j_list, diff_sols_09, color='k')
        # ax[1].plot(j_list, diff_sols_09[i, :], color='r')
    ax[1].set_xlabel('Coupling, J')
    ax[1].set_ylabel('Mean of solutions')
    ax[0].set_xlabel('Coupling, J')
    ax[0].set_ylabel('Approximate posterior, q')
    ax[0].legend()
    
    
def mean_field_fixed_points(j_list=np.arange(0., 1.005, 0.001), stim=0, num_iter=100):
    qvls_01 = []
    qvls_07 = []
    qvls_bckw = []
    for j in j_list:
        q_val_01 = 0.1
        q_val_07 = 0.7
        q_val_bckw = 0.7
        all_q = np.empty((num_iter))
        all_q[:] = np.nan
        for i in range(num_iter):
            q_val_01 = gn.sigmoid(6*(j*(2*q_val_01-1)+stim))
            q_val_07 = gn.sigmoid(6*(j*(2*q_val_07-1)+stim))
            q_val_bckw = backwards(q_val_bckw, j, stim)
        qvls_01.append(q_val_01)
        qvls_07.append(q_val_07)
        qvls_bckw.append(q_val_bckw)
    qvls_01 = np.array(qvls_01)
    qvls_07 = np.array(qvls_07)
    qvls_bckw = np.array(qvls_bckw)
    plt.figure()
    plt.plot(j_list, qvls_07, color='r')
    plt.plot(j_list[np.abs(qvls_01 - qvls_07) > 1e-5],
             qvls_01[np.abs(qvls_01 - qvls_07) > 1e-5], color='r')
    plt.plot(j_list, qvls_bckw, color='r', linestyle='--')
    plt.xlabel('J')
    plt.ylabel('q')
    # plt.plot(j_list, 1-np.array(qvls), color='k')


def plot_mf_sol_stim_bias(j_list, stim=0.1, num_iter=500, theta=theta):
    pos = []
    neg = []
    for j in j_list:
        v = mean_field_stim(j, num_iter=num_iter, stim=stim, sigma=0,
                            theta=theta)
        v_mn = v[-1, 0]
        pos.append(v_mn)
        neg.append(1-v_mn)
        # plt.figure()
        # plt.plot(v[-1, :])
        # gn.plot_cylinder(q=v[-1, :].reshape(5, 10, 2),
        #                   columns=5, rows=10, layers=2, offset=0.4,
        #                   minmax_norm=True)
    plt.figure()
    plt.plot(j_list, pos, color='r', markersize=2)
    plt.plot(j_list, neg, color='k', markersize=2)
    plt.xlabel('J')
    plt.ylabel('q')


def find_repulsor(j=0.1, num_iter=20, q_i=0.001, q_f=0.999,
                  stim=0, threshold=1e-10, theta=theta, neigh=3):
    # q = q_i
    # v_upper = mean_field_stim(j, num_iter=num_iter, stim=stim, sigma=0,
    #                           theta=theta, val_init=q_f)[-1, :]
    # neighs = np.sum(theta, axis=1)
    # val_upper = v_upper[neighs == neigh][0]
    # it = 0
    # while epsilon >= threshold:
    #     v = mean_field_stim(j, num_iter=num_iter, stim=stim, sigma=0,
    #                         theta=theta, val_init=q)[-1, :]
    #     val = v[neighs == neigh][0]
    #     if val == val_upper and it == 0:
    #         q = np.nan
    #         break
    #     if np.abs(val-val_upper) >= threshold:
    #         q = q+epsilon
    #     else:
    #         epsilon = epsilon/10
    #         q = find_repulsor(j=j, num_iter=num_iter, epsilon=epsilon,
    #                           q_i=q-epsilon*10, q_f=q+epsilon,
    #                           stim=stim, threshold=threshold, theta=theta,
    #                           neigh=neigh)
    #     if q > 1:
    #         q = np.nan
    #         break
    #     it += 1
    diff = 1
    neighs = np.sum(theta, axis=1)
    f_a = mean_field_stim(j, num_iter=num_iter, stim=stim, sigma=0,
                          theta=theta, val_init=q_f)[-1, neighs == neigh]
    f_b = mean_field_stim(j, num_iter=num_iter, stim=stim, sigma=0,
                          theta=theta, val_init=q_i)[-1, neighs == neigh]
    if np.abs(f_a-f_b) < threshold:
        return np.nan
    while diff >= threshold*1e-2:
        c = (q_i+q_f)/2
        f_c = mean_field_stim(j, num_iter=num_iter, stim=stim, sigma=0,
                              theta=theta, val_init=c)[-1, neighs == neigh]
        if np.abs(f_a-f_c) < threshold and np.abs(f_b-f_c) < threshold:
            return np.nan
        if np.abs(f_a-f_c) < threshold:
            q_f = c
            f_a = mean_field_stim(j, num_iter=num_iter, stim=stim, sigma=0,
                                  theta=theta, val_init=q_f)[-1, neighs == neigh]
        elif np.abs(f_b-f_c) < threshold:
            q_i = c
            f_b = mean_field_stim(j, num_iter=num_iter, stim=stim, sigma=0,
                                  theta=theta, val_init=q_i)[-1, neighs == neigh]
        else:
            return c
        diff = np.abs(q_i-q_f)
    return c


def plot_mf_sol_stim_bias_different_sols(j_list, stim=0.1, num_iter=500,
                                         theta=theta):
    vals_all_1 = np.empty((theta.shape[0], len(j_list)))
    vals_all_1[:] = np.nan
    vals_all_0 = np.empty((theta.shape[0], len(j_list)))
    vals_all_0[:] = np.nan
    vals_all_backwards = np.empty((3, len(j_list)))
    vals_all_backwards[:] = np.nan
    neighbors = np.unique(np.ceil(np.sum(theta, axis=1))).astype(int)
    for i_j, j in enumerate(j_list):
        v = mean_field_stim(j, num_iter=num_iter, stim=stim, sigma=0,
                            theta=theta, val_init=0.95)
        vals_all_1[:, i_j] = v[-1, :]
        v = mean_field_stim(j, num_iter=num_iter, stim=stim, sigma=0,
                            theta=theta, val_init=0.05)
        vals_all_0[:, i_j] = v[-1, :]
        for i in range(len(neighbors)):
            vals_all_backwards[i, i_j] = \
                find_repulsor(j=j, num_iter=50, q_i=0.01,
                              q_f=0.95, stim=stim, threshold=1e-2,
                              theta=theta, neigh=neighbors[i])  # + 1e-2*i
        # gn.plot_cylinder(q=v[-1, :].reshape(5, 10, 2),
        #                   columns=5, rows=10, layers=2, offset=0.4,
        #                   minmax_norm=True)
    plt.figure()
    neighs = np.sum(theta, axis=1, dtype=int)
    neighs -= np.min(neighs)
    colors = ['k', 'r', 'b']
    for i_v, vals in enumerate(vals_all_1):
        plt.plot(j_list, vals, color=colors[neighs[i_v]], alpha=0.1)
        plt.plot(j_list[vals_all_0[i_v] != vals],
                 vals_all_0[i_v][vals_all_0[i_v] != vals],
                 color=colors[neighs[i_v]], alpha=0.1)
    for i in range(len(neighbors)):
        plt.plot(j_list, vals_all_backwards[i, :],
                 color=colors[neighbors[i]-np.min(neighbors)],
                 linestyle='--', alpha=1)
    plt.xlabel('J')
    legendelements = [Line2D([0], [0], color='k', lw=2, label='3'),
                      Line2D([0], [0], color='r', lw=2, label='4')]
                      # Line2D([0], [0], color='b', lw=2, label='5')]
    plt.legend(handles=legendelements, title='Neighbors')
    plt.ylabel('q')


def backwards(q, j, beta, n_neigh=3):
    if 0 <= q <= 1:
        # q_new = 0.5*(1+ 1/j *(1/(2*n_neigh) * np.log(q/(1-q)) - beta))
        q_new = 0.5*(1+ 1/(j*n_neigh) *(1/2 * np.log(q/(1-q)) - beta))
    else:
        q_new = np.nan
    return q_new


def plot_mf_sol(j_list, num_iter=500):
    for j in j_list:
        v = mean_field(J=j, num_iter=num_iter, sigma=1)
        v_mn = np.nanmean(v[100:, 0])
        plt.plot(j, v_mn, marker='o', color='g', markersize=2)
        plt.plot(j, 1-v_mn, marker='o', color='g', markersize=2)


def plot_mf_sol_sigma_j(data_folder, j_list, sigma_list, num_iter=2000):
    matrix_loc = data_folder + 'q_sigma_j_mat.npy'
    os.makedirs(os.path.dirname(matrix_loc), exist_ok=True)
    if os.path.exists(matrix_loc):
        m_sol = np.load(matrix_loc, allow_pickle=True)
    else:
        m_sol = np.empty((len(j_list), len(sigma_list)))
        m_sol[:] = np.nan
        plt.figure()
        for i_j, j in enumerate(j_list):
            for i_s, sigma in enumerate(sigma_list):
                v = mean_field(J=j, num_iter=num_iter, sigma=sigma)
                v_mn = np.nanmean(v[100:, 0])
                v_mn_1 = 1-v_mn
                m_sol[i_j, i_s] = np.max([v_mn, v_mn_1])
    im = plt.imshow(np.flipud(m_sol), extent=[min(sigma_list), max(sigma_list),
                                              min(j_list), max(j_list)], aspect='auto')
    plt.colorbar(im, label='q*(x=1)')
    plt.xlabel(r'$\sigma$')
    plt.ylabel('J')


def plot_stim_effect(stim_list=np.linspace(0, 0.04, 4), j=0.4, N=3):
    q = np.arange(0, 1.001, 0.001)
    colormap = pl.cm.Oranges(np.linspace(0.2, 1, len(stim_list)))[::-1]
    plt.axhline(0, color='k', alpha=0.4)
    for i_s, stim in enumerate(stim_list[::-1]):
        vals = gn.sigmoid(2*N*j*(2*q-1)+ stim*2*N) - q
        plt.plot(q, vals, color=colormap[i_s], label=stim)
    plt.xlabel('q')
    plt.legend(title='B')
    plt.ylabel('f(q)')


def plot_solutions_mfield(j_list=np.arange(0., 1.005, 0.001), stim=0, N=3, plot_approx=False):
    fig, ax = plt.subplots(1, figsize=(5, 4))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    l = []
    plt.axvline(1/N, color='r', alpha=0.2, linewidth=2)
    plt.text(1/N-0.085, 0.12,  r'$J^{\ast}=1/3$', rotation='vertical')
    for j in j_list:
        q = lambda q: gn.sigmoid(2*N*j*(2*q-1)+ stim*2*N) - q 
        l.append(np.clip(fsolve(q, 0.9), 0, 1))
    plt.plot([1/3, 1], [0.5, 0.5], color='grey', alpha=1, linestyle='--',
             label='Unstable FP', linewidth=3)
    plt.plot(j_list, 1-np.array(l), color='k', linewidth=3)
    plt.plot(j_list, l, color='k', label='Stable FP', linewidth=3)
    plt.xlabel(r'Coupling $J$')
    plt.ylabel(r'Approximate posterior $q(x=1)$')
    # plt.title('Solutions of the dynamical system')
    if plot_approx:
        j_list1 = np.arange(1/N, 1, 0.001)
        r = (j_list1*N-1)*3/(4*(j_list1*N)**3)
        plt.plot(j_list1, np.sqrt(r)+0.5, color='b', linestyle='--')
        plt.plot(j_list1, -np.sqrt(r)+0.5, label=r'$q=0.5 \pm \sqrt{r}$',
                 color='b', linestyle='--')
        plt.plot([0, 1/N], [0.5, 0.5], color='r', label=r'$q=0.5$',
                 linestyle='--')
    # xtcks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # xtcks = np.sort(np.unique([0, 0.2, 0.4, 1/N, 0.6, 0.8, 1]))
    # labs = [x for x in xtcks]
    # pos = np.where(xtcks == 1/N)[0][0]
    # labs[pos] = r'$J^{\ast}$'  # '1/'+str(N)
    # plt.xticks(xtcks, labs)
    # ax.text(0.0, 0.8, 'Monostable')
    # ax.text(0.55, 0.8, 'Bistable')
    plt.legend(frameon=False, bbox_to_anchor=[0.4, 0.8])
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'mf_solutions.png', dpi=400, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'mf_solutions.svg', dpi=400, bbox_inches='tight')


def plot_solutions_mfield_neighbors(ax, j_list, color='k', stim=0, N=3):
    l = []
    for j in j_list:
        q = lambda q: gn.sigmoid(2*N*j*(2*q-1)+ stim*2) - q 
        l.append(np.clip(fsolve(q, 0.9), 0, 1))
    ax.plot(j_list, l, label=N, color=color)


def plot_solutions_mfield_for_N_neighbors(j_list, n_list=np.arange(1, 6), stim=0):
    fig, ax = plt.subplots(1)
    colormap = pl.cm.Blues(np.linspace(0.2, 1, len(n_list)+1))
    for n in n_list:
        plot_solutions_mfield_neighbors(ax, j_list, color=colormap[n],
                                        stim=stim, N=n)
    ax.legend(title='Neighbors')
    ax.set_xlabel('J')
    ax.set_ylabel('q')


def posterior_comparison_MF(stim_list=np.linspace(-2, 2, 1000), j=0.1):
    true_posterior = gn.true_posterior_stim(stim_list=stim_list, j=j)
    mf_post = []
    for stim in stim_list:
        q = lambda q: gn.sigmoid(6*j*(2*q-1)+ stim*6) - q 
        mf_post.append(np.clip(fsolve(q, 0.98), 0, 1))
    fig, ax = plt.subplots(1)
    ax.plot(true_posterior, mf_post, color='k')
    ax.plot([0, 1], [0, 1], color='grey', alpha=0.5)
    ax.set_xlabel(r'True posterior $p(x_i=1 | B)$')
    ax.set_ylabel(r'Mean-field posterior $q(x_i=1|B)$')
    ax.set_title('J = 0.1')



def plot_sols_mf_bias_stim_changing_j(j_list=[0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
                                      beta=1e-1):
    fig, ax = plt.subplots(ncols=3, nrows=2)
    ax = ax.flatten()
    q = np.arange(0, 1, 0.001)
    for i_j, j in enumerate(j_list):
        func = gn.sigmoid(6*(j*(2*q-1)+beta))
        ax[i_j].plot(q, func, color='k')
        ax[i_j].plot(q, q, color='r')
        ax[i_j].set_title('J='+str(round(j, 2)))
        ax[i_j].set_ylabel(r'$f(q, J)$')
        ax[i_j].set_xlabel(r'$q$')


def plot_q_bifurcation_vs_JB(j_list, stim_list=np.arange(-0.5, 0.5, 0.001)):
    fig, ax = plt.subplots(1, figsize=(5, 4))
    N = 3
    first_val = []
    for i_b, b in enumerate(stim_list):
        for j in j_list:
            q1 = lambda q: gn.sigmoid(2*N*j*(2*q-1)+ b*2*N) - q
            i_conds = np.linspace(0, 1, 10)
            sols, _, flag, _ = fsolve(q1, i_conds, full_output=True,
                                      xtol=1e-10)
            if flag != 1:
                continue
            un_sols = np.sort(np.unique(np.round(sols, 6)))
            if len(un_sols) == 3:
                first_val.append(un_sols[1])
                break
            else:
                continue
        if len(first_val) != (i_b+1):
            first_val.append(np.nan)
    ax.plot(stim_list, first_val, color='k')
    ax.set_ylabel('q*')
    ax.set_xlabel('B')


def plot_crit_J_vs_B_neigh(j_list, num_iter=200,
                           neigh_list=np.arange(3, 11),
                           dim3=False):
    if dim3:
        ax = plt.figure().add_subplot(projection='3d')
    else:
        fig, ax = plt.subplots(1, figsize=(5, 4))
        colormap = pl.cm.Blues(np.linspace(0.2, 1, len(neigh_list)))
    for n_neigh in neigh_list:
        print(n_neigh)
        delta = np.sqrt(1-1/(j_list*n_neigh))
        b_crit1 = (np.log((1-delta)/(1+delta))+2*n_neigh*j_list*delta)/2
        b_crit2 = (np.log((1+delta)/(1-delta))-2*n_neigh*j_list*delta)/2
        beta_list = b_crit1
        z = np.repeat(n_neigh, len(j_list))
        if dim3:
            ax.plot3D(z, beta_list, j_list, color='k')
        else:
            ax.plot(b_crit2, j_list, color=colormap[int(n_neigh-min(neigh_list))],
                    label=n_neigh, linewidth=3.5)
            ax.plot(b_crit1, j_list, color=colormap[int(n_neigh-min(neigh_list))],
                    label=n_neigh, linewidth=3.5)
    vals_b0 = 1 / neigh_list
    if dim3:
        ax.plot3D(neigh_list, np.repeat(0, len(neigh_list)), vals_b0,
                  color='r', linestyle='--')
        ax.set_xlabel('N')
        ax.set_ylabel('Sensory evidence B')
        ax.set_zlabel('Critical coupling J*')
    else:
        ax.set_xlabel(r'Sensory evidence $B$')
        ax.set_ylabel(r'Critical coupling $J^{\ast}$')
        ax.set_xlim(-1.05, 1.05)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.text()
        fig.tight_layout()
        ax_pos = ax.get_position()
        # ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*1.05, ax_pos.y0+ax_pos.height*0.2,
        #                         ax_pos.width*0.06, ax_pos.height*0.5])
        ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*0.3, ax_pos.y0+ax_pos.height*0.9,
                                ax_pos.width*0.4, ax_pos.height*0.05])
        newcmp = mpl.colors.ListedColormap(colormap)
        mpl.colorbar.ColorbarBase(ax_cbar, cmap=newcmp, label='Neighbors N',
                                  orientation='horizontal')
        ax_cbar.set_xticks([0, 0.5, 1], [np.min(neigh_list),
                                         int(np.mean(neigh_list)),
                                         np.max(neigh_list)])
        fig.savefig(DATA_FOLDER+'/J_vs_NB_MF_vf.png', dpi=400, bbox_inches='tight')
        fig.savefig(DATA_FOLDER+'/J_vs_NB_MF_vf.svg', dpi=400, bbox_inches='tight')



def plot_crit_J_vs_B(j_list, num_iter=200, beta_list=np.arange(-0.5, 0.5, 0.001)):
    first_j = []
    for i_b, beta in enumerate(beta_list):
        for j in j_list:
            q_fin = 0.65
            for i in range(num_iter):
                q_fin = backwards(q_fin, j, beta)
            if ~np.isnan(q_fin):
                first_j.append(j)
                break
        if len(first_j) != (i_b+1):
            first_j.append(np.nan)
    plt.figure()
    plt.plot(beta_list, first_j, color='k', linewidth=2)
    plt.fill_between(beta_list, first_j, 1, color='mistyrose', alpha=0.6)
    first_j = np.array(first_j)
    plt.text(-0.15, 0.8, '1 repulsor, 2 attractor')
    plt.text(-0.15, 0.2, '1 attractor')
    plt.fill_between(beta_list, 0, first_j, color='lightcyan', alpha=0.6)
    idx_neg = np.isnan(first_j) * (beta_list < 0)
    plt.fill_between(beta_list[idx_neg],
                      np.repeat(0, np.sum(idx_neg)), 1, color='lightcyan', alpha=0.6)
    idx_pos = np.isnan(first_j) * (beta_list > 0)
    plt.fill_between(beta_list[idx_pos],
                      np.repeat(0, np.sum(idx_pos)), 1, color='lightcyan', alpha=0.6)
    plt.ylabel('J*')
    plt.xlabel('B')
    plt.ylim(0, max(j_list))
    plt.yticks([0, 1/3, 0.5, 1], ['0', '1/3', '0.5', '1'])
    plt.axhline(1/3, color='r', linestyle='--', alpha=0.5)
    plt.xlim(min(beta_list), max(beta_list))


def plot_sols_mf_bias_stim_changing_beta(
        beta_list=[0, 0.02, 0.04, 0.06, 0.08, 0.1], j=0.5):
    fig, ax = plt.subplots(ncols=3, nrows=2)
    ax = ax.flatten()
    q = np.arange(0, 1, 0.001)
    for i_j, beta in enumerate(beta_list):
        func = gn.sigmoid(6*(j*(2*q-1)+beta))
        ax[i_j].plot(q, func, color='k')
        ax[i_j].plot(q, q, color='r')
        ax[i_j].set_title('B='+str(round(beta, 2)))
        ax[i_j].set_ylabel(r'$f(q, J)$')
        ax[i_j].set_xlabel(r'$q$')


def sol_bias_stim(j, beta):
    k = 6*j
    beta = 2*beta
    return (-beta+k+np.log(k-np.sqrt(k*(k-2))-1))/(2*k),\
        (-beta+k+np.log(k+np.sqrt(k*(k-2))-1))/(2*k)


def sol_mf_stim_taylor(j, beta):
    k = 6*j
    b = 2*beta
    return (1+np.exp(b)-np.exp(-b)*k)/(np.cosh(b)*2-2*k+2)


def sol_taylor_beta(j, beta):
    k = 6*j
    beta = 2*beta
    return (((np.exp(beta)+1)**2)/(np.exp(-beta)+1)-np.exp(beta)*k)/\
        ((np.exp(beta)+1)**2 -2*np.exp(beta)*k)


def sigma(x, k, b):
    return x + x*np.exp(-k * (2*x - 1) - b) - 1


def find_solution_sigma(k, b, x0=0.4):
    solution = fsolve(sigma, x0=x0, args=(k, b), xtol=1e-16)
    return solution


def solution_mf_sigma(ax, j_list, b):
    colors = ['r', 'k', 'g']
    sol_04 = []
    sol_005 = []
    sol_09 = []
    for j in j_list:
        sol_005.append(find_solution_sigma(j*6, b*6, 0.05))
        sol_04.append(find_solution_sigma(j*6, b*6, 0.4))
        sol_09.append(find_solution_sigma(j*6, b*6, 0.9))
    ax.plot(j_list, sol_005, label='b ='+str(b)+', x0=0.05',
    color=colors[0])
    ax.plot(j_list, sol_04, label='b ='+str(b)+', x0=0.4',
    color=colors[1])
    ax.plot(j_list, sol_09, label='b ='+str(b)+', x0=0.9',
    color=colors[2])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_xlabel('J')
    ax.set_ylabel('q')


def plot_solution_mf_numeric_solver(j_list=np.arange(0.001, 1, 0.001),
                                    b_list=[0, 0.025, 0.05,
                                            0.075, 0.1, 0.2]):
    fig, ax = plt.subplots(ncols=3, nrows=2)
    ax = ax.flatten()
    for a, b in zip(ax, b_list):
        solution_mf_sigma(ax=a, j_list=j_list, b=b)


def solution_for_sigmoid_taylor_order2(j, beta):
    k = 6*j
    b = 2*beta
    numerator = (((2 * np.exp(b) * k**2) / (np.exp(b) + 1)**3) 
                 - ((2 * np.exp(2 * b) * k**2) / (np.exp(b) + 1)**3) 
                 - np.sqrt(((4 * np.exp(b) * k**2) / (np.exp(b) + 1)**3) 
                           - ((4 * np.exp(2 * b) * k**2) / (np.exp(b) + 1)**3) 
                           - ((8 * np.exp(b) * k**2) / ((np.exp(-b) + 1) * (np.exp(b) + 1)**3)) 
                           + ((8 * np.exp(2 * b) * k**2) / ((np.exp(-b) + 1) * (np.exp(b) + 1)**3)) 
                           + ((4 * np.exp(2 * b) * k**2) / (np.exp(b) + 1)**4) 
                           - ((4 * np.exp(b) * k) / (np.exp(b) + 1)**2) + 1) - ((2 * np.exp(b) * k) / (np.exp(b) + 1)**2) + 1)
    denominator = 2 * (((2 * np.exp(b) * k**2) / (np.exp(b) + 1)**3) - ((2 * np.exp(2 * b) * k**2) / (np.exp(b) + 1)**3))
    return numerator / denominator


def dyn_sys_mf(q, dt, j, sigma=1, bias=0, tau=1, n=3):
    return np.clip(q + dt*(gn.sigmoid(2*n*j*(2*q-1)+2*bias)-q)/tau+
        np.random.randn()*np.sqrt(dt/tau)*sigma, 0, 1)


def plot_q_evol_time_noise(j=0.7, dt=1e-3, num_iter=100, sigma=1):
    q = np.random.rand()
    q_l = []
    # time = np.arange(num_iter)*dt
    for i in range(num_iter):
        q = np.clip(dyn_sys_mf(q, dt, j, sigma=sigma), 0, 1)
        q_l.append(q)
    plt.plot(q_l)
    plt.xlabel('Steps')
    plt.ylabel('q')
    plt.ylim(-0.05, 1.05)


def plot_psych_kernel(noise_list=[0.05, 0.2, 0.3], j=0.6, b=0,
                      dt=1e-2, tau=0.5, t_end=20, n_its=10000,
                      nboots=100):
    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(10, 10))
    ax = ax.flatten()
    for ia, a in enumerate(ax):
        a.set_ylim(-0.025, .05)
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        if ia > 5:
            a.set_xlabel('Time (s)')
        if ia % 3 != 0:
            a.set_yticks([])
        else:
            a.set_ylabel('Impact of sitmulus')
    titles = ['Perfect', 'Absorbing', 'Reflecting']
    for ia, a in enumerate(ax[:3]):
        a.set_title(titles[ia])
    for i_n, noise in enumerate(noise_list):
        aux_kernel_normal, aux_kernel_abs, aux_kernel_ref =\
            get_psych_kernel(j=j, b=b, n_its=n_its, t_end=t_end, dt=dt, tau=tau,
                             noise=noise, nboots=nboots)
        # aux_kernel_normal = np.convolve(aux_kernel_normal, np.ones(10)/10, mode='valid')
        # aux_kernel_abs = np.convolve(aux_kernel_abs, np.ones(10)/10, mode='valid')
        # aux_kernel_ref = np.convolve(aux_kernel_ref, np.ones(10)/10, mode='valid')
        time = np.arange(0, t_end, dt)
        time_2 = np.arange(0, len(time), 5)*dt
        ax[i_n*3].plot(time_2, aux_kernel_normal-0.5, color='k')
        ax[i_n*3+1].plot(time_2, aux_kernel_abs-0.5, color='r')
        ax[i_n*3+2].plot(time_2, aux_kernel_ref-0.5, color='b')


def get_psych_kernel(j=0.8, b=0.1, n_its=100, t_end=10, dt=1e-3, tau=0.1,
                     noise=0.1, nboots=1):
    """
    Plots psychophysical kernel for perfect integrator (normal),
    reflecting and absorbing boudns.
    The higher the noise, the clearer the shape.
    """
    d_normal = np.zeros(n_its)
    d_reflecting = np.zeros(n_its)
    d_absorbing = np.zeros(n_its)
    time = np.arange(0, t_end, dt)
    chi = np.zeros((n_its, len(time)))
    for it in range(n_its):
        absorb = False
        q = np.random.rand()
        q2 = np.copy(q)
        ql = []
        for t in range(len(time)):
            chi[it, t] = np.random.randn()*noise
            chi_t = chi[it, t]
            q = q + dt*(gn.sigmoid(6*j*(2*q-1)+2*b)-q)/tau +\
                chi_t*np.sqrt(dt/tau)
            q2 = q2 + dt*(gn.sigmoid(6*j*(2*q2-1)+2*b)-q2)/tau +\
                chi_t*np.sqrt(dt/tau)
            q2 = np.clip(q2, 0, 1)
            ql.append(q)
            if (q > 0.9 or q < 0.1) and not absorb:
                absorb = True
                d_absorbing[it] = np.sign(q - 0.5)
        if not absorb:
            d_absorbing[it] = np.sign(q - 0.5)
        d_normal[it] = np.sign(q - 0.5)
        d_reflecting[it] = np.sign(q2 - 0.5)
    time_2 = np.arange(0, len(time), 5)
    aux_kernel_normal = get_kernel_bootstrap(d_normal, chi, time_2, nboot=nboots)
    aux_kernel_abs = get_kernel_bootstrap(d_absorbing, chi, time_2, nboot=nboots)
    aux_kernel_ref = get_kernel_bootstrap(d_reflecting, chi, time_2, nboot=nboots)
    return aux_kernel_normal, aux_kernel_abs, aux_kernel_ref


def get_kernel_bootstrap(d, chi, time, nboot=100):
    """
    Computes psychophysical kernel using bootstrap, with 'nboot' partitions.
    """
    aux_kernel = np.zeros((len(time), nboot))
    indexs = np.random.randint(0, len(d),(nboot, len(d)))
    if nboot == 1:
        indexs = [np.arange(len(d))]
    for iboot in range(nboot):
        if iboot % 100 == 0:
            print(iboot)
        for iframe in range(len(time)):
            fpr,tpr, _ = roc_curve(d[indexs[iboot]], chi[indexs[iboot], iframe])
            aux_kernel[iframe][iboot] = auc(fpr, tpr)
    kernel = np.mean(aux_kernel, axis=1)
    return kernel


def PK_slope(kernel):
    '''
    Compute the slope of the PK:
    PRR=integral( kernel*f(t))
    with f(t)=1-a*t with a such as f(T)=-1 T stimulus duration
    positive recency
    zero flat
    negative primacy
    '''
    aux=np.linspace(1,-1,len(kernel))
    kernel=kernel-0.5
    aux_kernel=(kernel)/(np.sum(kernel))
    return -np.sum(aux_kernel*aux)


def total_area_kernel(kernel):
    '''
    Compute the PK area unnormalized
    '''
    nframes=len(kernel)
    area_pi = nframes*(0.5+2/np.pi*np.arctan(1/np.sqrt(2*nframes-1))) -0.5*nframes
    return np.sum(kernel-0.5)/area_pi


def potential_mf(q, j, bias=0):
    return q*q/2 - np.log(1+np.exp(6*(j*(2*q-1))+bias*2))/(12*j) #  + q*bias


def potential_mf_neighs(q, j, bias=0, neighs=3):
    return q*q/2 - np.log(1+np.exp(2*neighs*(j*(2*q-1))+bias*2))/(4*neighs*j)


def plot_potentials_different_beta(j=0.5, beta_list=[-0.1, -0.05, 0, 0.05, 0.1]):
    fig, ax = plt.subplots(1)
    colormap = pl.cm.copper(np.linspace(0.2, 1, len(beta_list)))
    q = np.arange(0, 1, 0.001)
    minvals = []
    potminvals = []
    qminvals = []
    pos_ax = ax.get_position()
    for i_j, beta in enumerate(beta_list):
        pot = potential_mf(q, j, beta)
        minvals.append(np.argmin(pot))
        if i_j % 2 == 0:
            label = np.round(beta, 3)
        else:
            label = ' '
        ax.plot(q, pot, label=label, color=colormap[i_j])  
        potminvals.append(pot[minvals[i_j]])
        qminvals.append(q[minvals[i_j]])
        ax.plot(q[minvals[i_j]], pot[minvals[i_j]], marker='o', color=colormap[i_j])
    ax.plot(qminvals, potminvals, color='k', linestyle='--', alpha=0.2)
    # ax.arrow(0.6, 0.01, 0.1, 0, color='k', head_length=0.03,
    #          width=0.00001, head_width=0.0006)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.legend(title='Sensory evidence, B', frameon=False, labelspacing=0.1)
    ax.set_xlabel(r'Approximate posterior $q$')
    ax.set_ylabel(r'Potential $V(q)$')
    inset = ax.inset_axes([pos_ax.x0+pos_ax.width/1.2, pos_ax.y0+pos_ax.height/1.5,
                           pos_ax.width*1.2/3, pos_ax.height/1.8])
    inset.plot(beta_list, potminvals, color='k', alpha=0.2, linestyle='--')
    # twininset = inset.twinx()
    # twininset.plot(beta_list, qminvals, color='r', alpha=0.2, linestyle='--')
    for i_j, beta in enumerate(beta_list):
        inset.plot(beta_list[i_j], potminvals[i_j], color=colormap[i_j],
                   marker='o')
        # twininset.plot(beta_list[i_j], qminvals[i_j], color=colormap[i_j],
        #                marker='o')
    inset.spines['right'].set_visible(False)
    inset.spines['top'].set_visible(False)
    inset.set_ylabel('Min. potential')
    # twininset.set_ylabel('Fixed point')
    inset.set_yticks([])
    # twininset.set_yticks([])
    inset.set_xticks([])
    # twininset.set_xticks([])
    inset.set_xlabel('Sensory evidence, B')
    fig.savefig(DATA_FOLDER + 'potential_vs_B.png', dpi=100,
                bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'potential_vs_B.svg', dpi=100,
                bbox_inches='tight')


def plot_energy_barrier_vs_B(j=0.7, beta_list=np.arange(-0.05, 0.06, 0.01)):
    fig, ax = plt.subplots(1)
    energy_barrier1 = []
    energy_barrier2 = []
    pos_ax = ax.get_position()
    for i_j, beta in enumerate(beta_list):
        x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j, beta)
        vstable1, vstable2, vunstable =\
            potential_mf(np.array([x_stable_1, x_stable_2, x_unstable]),
                         j, beta)
        energy_barrier1.append(vunstable-vstable1)
        energy_barrier2.append(vunstable-vstable2)
        if i_j == len(beta_list)-1:
            points = [x_stable_1, x_stable_2, x_unstable]
            potvals = [vstable1, vstable2, vunstable]
    # energy_barrier = np.array(energy_barrier)
    # ax.plot(beta_list, energy_barrier1, marker='o', color='k')
    ax.plot(energy_barrier1, energy_barrier2, marker='o', color='k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    # ax.set_xlabel(r'Sensory evidence, B')
    ax.set_ylabel(r'Energy barrier $\Delta V_2 (B)$')
    ax.set_xlabel(r'Energy barrier $\Delta V_1 (B)$')
    inset = ax.inset_axes([pos_ax.x0+pos_ax.width/1.6, pos_ax.y0+pos_ax.height/1.8,
                           pos_ax.width*1.7/3, pos_ax.height/1.4])
    q = np.arange(-0.2, 1.21, 0.01)
    pot = potential_mf(q, j, beta_list[-1])
    inset.plot(q, pot, color='k')
    inset.plot(points, potvals, color='k', linestyle='', marker='o')
    inset.annotate(text='', xy=(points[0], potvals[0]+3e-3),
                   xytext=(points[0], potvals[2]), arrowprops=dict(arrowstyle='<->'))
    inset.annotate(text='', xy=(points[1], potvals[1]+3e-3),
                   xytext=(points[1], potvals[2]), arrowprops=dict(arrowstyle='<->'))
    inset.text(points[1]-2e-1, potvals[1]+2.2e-2, r'$\Delta V_2$', rotation='vertical')
    inset.text(points[0]-2e-1, potvals[1]+2.2e-2, r'$\Delta V_1$', rotation='vertical')
    inset.spines['right'].set_visible(False)
    inset.spines['top'].set_visible(False)
    inset.set_yticks([])
    inset.set_xticks([])
    inset.set_ylabel('Potential V(q)')
    inset.set_xlabel('Approx. posterior q')
    fig.savefig(DATA_FOLDER + 'barrier_vs_B.png', dpi=200, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'barrier_vs_B.svg', dpi=200, bbox_inches='tight')


def mclaurin_expansion_pot(q, j ,b):
    k = 6*(j+b)
    order_0 = np.log(1+np.exp(-k))
    order_1 = (2*k*q)/(np.exp(k)+1)
    order_2 = (2*np.exp(k)*k*k*q*q)/(np.exp(k)+1)**2
    order_3 = (4*np.exp(k)*(np.exp(k)-1)*k*k*k*q*q*q)/(3*(np.exp(k)+1)**3)
    order_4 = (2*np.exp(k)*(-4*np.exp(k)+np.exp(2*k)+1)*k*k*k*k*q*q*q*q)/(3*(np.exp(k)+1)**4)
    expansion = 0.5*q**2 - (order_0+order_1+order_2+order_3+order_4) / (2*k)
    return expansion


def potential_expansion_at_any_point_order_4(q, j, b, point):
    k = 6*(j+b)
    q_centered = q-point
    expo = np.exp(k*(2*point-1))
    f_0 = np.log(1+expo)
    f_1 = 2*k*expo / (1+expo)
    f_2 = 4*(k**2)*expo / (1+expo)**2
    f_3 = 8*(k**3) * expo*(expo-1) / (1+expo)**3
    f_4 = 16 * (k**4) * expo * (np.exp(2*k*(2*point-1))-4*expo+1) / (1+expo)**4
    expansion = f_0 + f_1*q_centered + f_2*(q_centered**2) / 2 +\
        f_3*(q_centered**3)/6 + f_4*(q_centered**4)/24
    return -expansion/(2*k) + 0.5*q**2


def crit_val_J(b, N):
    j_ini = 1/N
    e2bn = np.exp(2*b*N)
    first_val_sum = 4*(e2bn) / (e2bn+1)**2
    second_val_sum = -2*np.sqrt(-(8*e2bn*(e2bn-1)/(e2bn+1)**3)*(1/(1+e2bn) - 1/2))
    return j_ini/(first_val_sum+second_val_sum)


def f_expansion_any_order_any_point(q, j, b, order, point):
    x = sympy.symbols('x')
    b = sympy.symbols('b')
    alpha = 6*j*(2*x-1) + 2*b
    func_0 = -x + 1/(1+sympy.exp(-alpha))
    funcs_diffs = [func_0]
    for i in range(order):
        funcs_diffs.append(sympy.diff(funcs_diffs[i], x))
    coefs = [funcs_diffs[n].subs(x, point) / fact(n) for n in range(order+1)]
    q_vals = [(q-point)**n for n in range(order+1)]
    expansion = 0
    for i in range(order+1):
        expansion = expansion + (coefs[i]*q_vals[i])
    return expansion


def potential_expansion_any_order_any_point(q, j, b, order, point):
    x = sympy.symbols('x')
    k = 6*(j+b)
    func_0 = 0.5*x**2-sympy.log(1+sympy.exp(k*(2*x-1)))/(2*k)
    funcs_diffs = [func_0]
    for i in range(order):
        funcs_diffs.append(sympy.diff(funcs_diffs[i], x))
    coefs = [funcs_diffs[n].subs(x, point) / fact(n) for n in range(order+1)]
    q_vals = [(q-point)**n for n in range(order+1)]
    expansion = 0
    for i in range(order+1):
        expansion = expansion + (coefs[i]*q_vals[i])
    return expansion
    

def fact(n):
    return np.prod(np.arange(n)+1)


def taylor_expansion_pot_at_05(q, j ,b):
    k = 6*(j+b)
    order_0 = np.log(2)
    order_1 = k*(q-0.5)
    order_2 = 0.5 * (k**2) * (q-0.5) ** 2
    order_3 = 0
    order_4 = -1/12 * (k**4) * (q-0.5) ** 4
    expansion = 0.5*q**2 - (order_0+order_1+order_2+order_3+order_4) / (2*k)
    return expansion


def plot_potentials_mf(j_list, bias=0, neighs=3):
    # colormap = pl.cm.Blues(np.linspace(0.2, 1, len(j_list)))
    # colormap_1 = pl.cm.Purples(np.linspace(0.4, 1, len(j_list)))
    Blues = pl.cm.get_cmap('Blues', 100)
    Purples = pl.cm.get_cmap('Purples', 100)
    newcolors = Blues(np.linspace(0.1, 1, len(j_list)))
    newcolors_purples = Purples(np.linspace(0.1, 1, len(j_list)))
    red = np.array([1, 0, 0, 1])
    newcolors[len(j_list)//3, :] = red
    newcolors[(len(j_list)//3+1):, :] = newcolors_purples[(len(j_list)//3+1):]
    newcmp = mpl.colors.ListedColormap(newcolors)
    q = np.arange(0, 1, 0.001)
    fig, ax = plt.subplots(1, figsize=(5, 4))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    change_colormap = False
    for i_j, j in enumerate(j_list):
        pot = potential_mf_neighs(q, j, bias=bias, neighs=neighs)
        # pot = potential_expansion_any_order_any_point(q, j, b=0, order=8, point=0.5)
        # norm_cte = np.max(np.abs(pot))
        if abs(j - 0.333) < 0.001 and not change_colormap:
            color = 'r'
            change_colormap = True
            lab = r'$J^* = 1/3$'
        else:
            color = newcolors[i_j]
            lab = np.round(j, 2)
        ax.plot(q, pot-np.mean(pot), color=color, label=lab)
    # ax_pos = ax.get_position()
    # ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*1.02, ax_pos.y0,
    #                         ax_pos.width*0.04, ax_pos.height*0.9])
    # mpl.colorbar.ColorbarBase(ax_cbar, cmap=newcmp, label=r'Coupling $J$')
    # ax_cbar.set_yticks([0, max(j_list)/3, max(j_list)/2, max(j_list)], [0, 0.33, 0.4, max(j_list)])
    # ax_cbar.set_title(r'Coupling $J$')
    ax.set_xlabel(r'Approximate posterior $q$')
    ax.set_ylabel(r'Potential $V(q)$')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    fig.tight_layout()
    # ax.legend(title='J:')
    ax.legend(title='Coupling, J', bbox_to_anchor=(1., 0.9),
              frameon=False, labelspacing=0.2)
    fig.savefig(DATA_FOLDER + 'potentials_vs_q_v2.png', dpi=400, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'potentials_vs_q_v2.svg', dpi=400, bbox_inches='tight')


def saddle_node_bifurcation(j=0.4):
    q = np.arange(0, 1, 0.001)
    fun_q = lambda b: gn.sigmoid(6*(j*(2*q-1)+b))-q
    potential = lambda bias: q*q/2 - np.log(1+np.exp(6*(j*(2*q-1)+bias)))/(12*j)
    fig, ax = plt.subplots(nrows=2, figsize=(6, 10))
    for b in [0, 0.01, 0.02, 0.04]:
        ax[0].plot(q, fun_q(b), label=b)
        ax[1].plot(q, potential(b)-np.mean(potential(b)))
    ax[0].legend(title='B')
    ax[0].axhline(0, color='k')
    ax[1].axhline(np.min(potential(0)-np.mean(potential(0))),
                  color='k')
    ax[1].set_xlabel('Approximate posterior, q(x=1)')
    ax[1].set_ylabel('Potential')
    ax[0].set_ylabel('f(q)')
    fig.tight_layout()


def solutions_fixing_j_changing_b(j=0.6, num_iter=200):
    b_list = np.arange(-0.5, 0.5, 5e-4)
    qvls_01 = []
    qvls_07 = []
    qvls_bckw = []
    for b in b_list:
        q_val_0 = 0
        q_val_1 = 1
        q_val_bckw = 0.7
        for i in range(num_iter):
            q_val_0 = gn.sigmoid(6*(j*(2*q_val_0-1)+b))
            q_val_1 = gn.sigmoid(6*(j*(2*q_val_1-1)+b))
            q_val_bckw = backwards(q_val_bckw, j, b)
        qvls_01.append(q_val_0)
        qvls_07.append(q_val_1)
        qvls_bckw.append(q_val_bckw)
    fig, ax = plt.subplots(1, figsize=(5, 4))
    qvls_01 = np.array(qvls_01)
    qvls_07 = np.array(qvls_07)
    q_val_bckw = np.array(q_val_bckw)
    plt.plot(b_list[qvls_01 <= 0.5], qvls_01[qvls_01 <= 0.5], color='k')
    plt.plot(b_list[qvls_07 >= 0.5], qvls_07[qvls_07 >= 0.5], color='k')
    plt.plot(b_list, qvls_bckw, color='k', linestyle='--')
    plt.ylabel('Approximate posterior, q(x=1)')
    plt.xlabel('Sensory evidence, B')
    legendelements = [Line2D([0], [0], color='k', lw=2, label='Stable'),
                      Line2D([0], [0], color='k', linestyle='--', lw=2, label='Unstable')]
    legend2 = plt.legend(handles=legendelements, title='Fixed point')
    ax.add_artist(legend2)
    fig.tight_layout()


def plot_pot_evolution_mfield(j, num_iter=10, sigma=0.1, bias=1e-3):
    fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(16, 10))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.075, right=0.98,
                        hspace=0.3, wspace=0.5)
    # fig.suptitle('J = '+ str(j))
    ax = ax.flatten()
    # m_field = mean_field(j, num_iter=num_iter, sigma=sigma)
    q = np.random.rand()
    m_field = []
    # time = np.arange(num_iter)*dt
    for i in range(num_iter):
        q = dyn_sys_mf(q, dt=1, j=j, sigma=sigma, bias=bias)
        m_field.append(q)
    q = np.arange(0, 1., 0.001)
    # q = np.arange(0, 1, 0.001)
    pot = potential_mf(q, j, bias)
    for i in range(len(ax)):
        ax[i].plot(q, pot, color='k')
        q_ind = m_field[i*num_iter//len(ax)]
        pot_ind = potential_mf(q_ind, j, bias)
        ax[i].plot(q_ind, pot_ind, marker='o', color='r')
        ax[i].set_title('iter #' + str(i*num_iter//len(ax)+1))
        if i != 0 and i != (2*len(ax)//3) and i != (len(ax)//3):
            ax[i].set_yticks([])
        else:
            ax[i].set_ylabel('Potential')
        if i < 10:
            ax[i].set_xticks([0, 0.5, 1], ['', '', ''])
        else:
            ax[i].set_xticks([0, 0.5, 1])
            ax[i].set_xlabel('q')


def potential_2d_faces(x1, x2, j=1, b=0):
    return 0.5*x1**2 + 0.5*x2**2 -\
            np.log(1+np.exp(a_ij(x2, x1, j=j, b=b)))/(12*j) -\
                    np.log(1+np.exp(a_ij(x1, x2, j=j, b=b)))/(12*j)


def a_ij(xi, xj, j=1, b=0):
    return 2*(j*(4*xi+2*xj-3)+b)


# def f_i(xi, xj, j=1, b=0):
#     return gn.sigmoid(a_ij(xi, xj, j=j, b=b)) - xi


# def f_i_both(x, j=1, b=0):
#     xi, xj = x
#     return [gn.sigmoid(a_ij(xi, xj, j=j, b=b)) - xi,
#             gn.sigmoid(a_ij(xj, xi, j=j, b=b)) - xj]


def f_i(x1, x2, j, b):
    return gn.sigmoid(2*j*3*(2*x2-1) + 2*b)-x1


def f_i_both(x, j, b=0):
    x1, x2 = x
    return [gn.sigmoid(2*j*3*(2*x2-1) + 2*b)-x1,
            gn.sigmoid(2*j*3*(2*x1-1) + 2*b)-x2]


def f_i_diagonal(x, j=1, b=0):
    return gn.sigmoid(a_ij(x, x, j=j, b=b)) - x


def f_i_diagonal_neg(x, j=1, b=0):
    y = x[1]
    x = x[0]
    return [gn.sigmoid(2*(j*(2*x-1)+b)) - x,
            gn.sigmoid(2*(j*(-2*y+1)+b)) + y - 1]


def plot_transition_time(j_list=[0.1, 0.5, 1, 2], b=0, noise=0.1, tau=1, time_end=10000, dt=1e-2):
    transition_rate = []
    transition_rate_analytical = []
    for j in j_list:
        x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j, b)
        transition_rate_analytical.append(k_i_to_j(j, x_unstable, x_stable_1, noise, b))
        init_cond = np.random.rand(2)
        m_solution = solution_mf_sdo_2_faces_euler(j, b, theta, noise, tau, init_cond,
                                                   time_end=time_end, dt=dt)
        x, y = m_solution.y[0], m_solution.y[1]
        states = [0]
        for i in range(len(x)):
            if x[i] > 1-y[i] and x[i] > 0.75 and y[i] > 0.75:
                states.append(1)
            elif x[i] < 1-y[i] and x[i] < 0.25 and y[i] < 0.25:
                states.append(-1)
            else:
                states.append(states[i-1])
            if j > 1:
                if x[i] < y[i] and x[i] > 0.55 and y[i] < 0.55:
                    states.append(-2)
                if x[i] > y[i] and x[i] < 0.55 and y[i] > 0.55:
                    states.append(2)
        orders = gn.rle(states)
        time_transition = orders[0]
        time_transition = time_transition[time_transition > 1]*dt
        # plt.plot(x, y)
        # plt.plot([0, 1], [0, 1], color='r')
        # plt.plot([0, 1], [1, 0], color='r')
        transition_rate.append(1/np.mean(time_transition))
    plt.figure()
    plt.plot(j_list, transition_rate)
    plt.plot(j_list, transition_rate_analytical)
    plt.yscale('log')
    plt.ylabel('Transition rate')
    plt.xlabel('Coupling, J')


def plot_density_map_2d_mf(j=1, b=0, noise=0.15, tau=1, time_end=10000, dt=1e-2):
    def f_i(x1, x2, j, b):
        return gn.sigmoid(2*j*3*(2*x2-1) + 2*b)-x1
    init_cond = np.random.rand(2)
    m_solution = solution_mf_sdo_2_faces_euler(j, b, theta, noise, tau, init_cond,
                                               time_end=time_end, dt=dt)
    fig, ax = plt.subplots(1)
    # df_sol = pd.DataFrame({"q_1": m_solution.y[0], "q_2": m_solution.y[1]})
    # sns.kdeplot(df_sol, x="q_1", y="q_2", common_norm=False, cmap='Blues',
    #             fill=True)
    # Peform the kernel density estimate
    xx, yy = np.mgrid[-0.2:1.2:100j, -0.2:1.2:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([m_solution.y[0], m_solution.y[1]])
    kernel = scipy.stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('white_blue', ['white', 'royalblue'])
    # Contourf plot
    cfset = ax.contourf(xx, yy, f, cmap=cmap)  # , norm=mpl.colors.LogNorm()
    plt.colorbar(cfset, ax=ax, label='Density')
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel(r'$q_1$')
    ax.set_ylabel(r'$q_2$')
    fig.savefig(DATA_FOLDER + '2d_density_map.png', dpi=300, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + '2d_density_map.svg', dpi=300, bbox_inches='tight')
    fig2, ax2 = plt.subplots(1)
    x1 = np.arange(-0.2, 1.2, 5e-2)
    x2 = np.arange(-0.2, 1.2, 5e-2)
    x, y = np.meshgrid(x1, x2)
    u1 = f_i(x, y, j=j, b=b)
    u2 = f_i(y, x, j=j, b=b)
    cfset = ax2.contourf(xx, yy, f, cmap=cmap)  # , norm=mpl.colors.LogNorm()
    plt.colorbar(cfset, ax=ax2, label='Density')
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax2.set_ylabel(r'$q_1$')
    ax2.set_xlabel(r'$q_2$')
    ax2.quiver(x, y, u1, u2)
    fig2.savefig(DATA_FOLDER + '2d_density_map_vector.png', dpi=300, bbox_inches='tight')
    fig2.savefig(DATA_FOLDER + '2d_density_map_vector.svg', dpi=300, bbox_inches='tight')


def plot_potential_and_vector_field_2d(j=1, b=0, noise=0, tau=1, time_end=50, dt=5e-2,
                                       plot_simulation=False):
    ax = plt.figure().add_subplot(projection='3d')
    x1 = np.arange(0, 1, 1e-3)
    x2 = np.arange(0, 1, 1e-3)
    x, y = np.meshgrid(x1, x2)
    V_x = potential_2d_faces(x, y, j=j, b=b)
    ax.plot_surface(x, y, V_x, alpha=0.4)
    ax.set_xlabel(r'$q_1$')
    ax.set_ylabel(r'$q_2$')
    ax.set_zlabel(r'Potential, $V(\vec{x})$')
    # init_cond = [0.1, 0.9]
    if plot_simulation:
        init_cond = np.random.rand(2)
        m_solution = solution_mf_sdo_2_faces_euler(j, b, theta, noise, tau, init_cond,
                                                   time_end=time_end, dt=dt)
        ax.plot3D(m_solution.y[0], m_solution.y[1],
                  potential_2d_faces(m_solution.y[0], m_solution.y[1], j=j, b=b),
                  color='r')
        ax.plot3D(init_cond[0], init_cond[1],
                  potential_2d_faces(init_cond[0], init_cond[1], j=j, b=b),
                  marker='o', color='r')
        ax.plot3D(m_solution.y[0][-1], m_solution.y[1][-1],
                  potential_2d_faces(m_solution.y[0][-1], m_solution.y[1][-1], j=j, b=b),
                  marker='x', color='b')
    # fig2, ax_2 = plt.subplots(ncols=1, figsize=(5, 4))
    # im = ax_2.imshow(np.flipud(V_x), cmap='jet', extent=[np.min(x1), np.max(x1),
    #                                                      np.min(x2), np.max(x2)])
    # plt.colorbar(im)
    fig2, ax_2 = plt.subplots(ncols=2, figsize=(8, 4))
    ax2, ax3 = ax_2
    x1 = np.arange(-0.2, 1.2, 5e-2)
    x2 = np.arange(-0.2, 1.2, 5e-2)
    x, y = np.meshgrid(x1, x2)
    u1 = f_i(x, y, j=j, b=b)
    u2 = f_i(y, x, j=j, b=b)
    ax2.quiver(x, y, u1, u2)
    x1 = np.arange(-0.2, 1.2, 1e-3)
    x2 = np.arange(-0.2, 1.2, 1e-3)
    # lam2 = get_eigen_2(j, x1, b=b)
    # idx_0_lam2 = np.argsort(np.abs(lam2))[:2]
    # lamneg2 = get_eigen_2_neg(j, x1, b=b)
    # idx_0_lamneg2 = np.argsort(np.abs(lamneg2))[:2]
    # for i in range(2):
    #     ax2.plot(x1[idx_0_lam2[i]], x1[idx_0_lam2[i]], marker='o',
    #              color='b', markersize=8)
    #     ax2.plot(x1[idx_0_lamneg2[i]], 1-x1[idx_0_lamneg2[i]], marker='o',
    #              color='b', markersize=8)
    ax2.set_xlabel(r'$q_1$')
    ax2.set_ylabel(r'$q_2$')
    if plot_simulation:
        ax2.plot(m_solution.y[0], m_solution.y[1], color='r')
        ax2.plot(init_cond[0], init_cond[1], marker='o', color='r')
        ax2.plot(m_solution.y[0][-1], m_solution.y[1][-1], marker='x', color='b')
    # modulo = np.sqrt(u1**2 + u2**2)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlim(-0.1, 1.1)
    x, y = np.meshgrid(x1, x2)
    u1 = f_i(x, y, j=j, b=b)
    u2 = f_i(y, x, j=j, b=b)
    modulo = np.sqrt(u1**2 + u2**2)
    image = ax3.imshow(np.flipud(modulo), extent=[np.min(x1), np.max(x1),
                                                  np.min(x2), np.max(x2)],
                       cmap='gist_gray')
    fig2.tight_layout()
    plt.subplots_adjust(wspace=0.35)
    ax_pos = ax3.get_position()
    ax_cbar = fig2.add_axes([ax_pos.x0+ax_pos.width*1.13, ax_pos.y0+ax_pos.height*0.2,
                             ax_pos.width*0.05, ax_pos.height*0.7])
    plt.colorbar(image, label=r'speed, $||f(q_1, q_2)||$', cax=ax_cbar)
    ax3.set_xlabel(r'$q_1$')
    ax3.set_ylabel(r'$q_2$')


def plot_eigenvalues_at_diagonals(j, b):
    # eigenvalues
    x = np.arange(0, 1+1e-3, 1e-3)
    lam_2 = gn.sigmoid(6*j*(2*x-1))*(1-gn.sigmoid(6*j*(2*x-1)))*12*j-1
    lam_1 = gn.sigmoid(6*j*(2*x-1))*(1-gn.sigmoid(6*j*(2*x-1)))*4*j-1
    plt.figure()
    plt.title('Eigenvalues at y=x')
    plt.plot(x, lam_2, label='a+b', color='r')
    plt.plot(x, lam_1, label='a-b', color='k')
    plt.axhline(0, linestyle='--', color='b')
    plt.xlabel('x')
    plt.legend()
    plt.ylabel(r'$\lambda$')
    lam_2 = get_eigen_1_neg(j, x, b)
    lam_1 = get_eigen_2_neg(j, x, b)
    plt.figure()
    plt.title('Eigenvalues at y=1-x')
    plt.plot(x, lam_2, label='l1, 1-x', color='r')
    plt.plot(x, lam_1, label='l2, 1-x', color='k')
    plt.axhline(0, linestyle='--', color='b')
    plt.xlabel('x')
    plt.legend()
    plt.ylabel(r'$\lambda$')


def plot_mf_evolution_2_faces(j=1, b=0, noise=0, tau=1, time_end=50, dt=5e-2):
    # time evolution of q1, q2
    init_cond = np.random.rand(2)
    m_solution = solution_mf_sdo_2_faces_euler(j, b, theta, noise, tau, init_cond,
                                               time_end=time_end, dt=dt)
    fig4, ax4 = plt.subplots(1)
    ax4.plot(m_solution.t, m_solution.y[0], label='q_1')
    ax4.plot(m_solution.t, m_solution.y[1], label='q_2')
    ax4.legend()
    ax4.set_ylabel('q')
    ax4.set_xlabel('Time (s)')


def plot_mf_evolution_all_nodes(j=1, b=0, noise=0, tau=1, time_end=50, dt=5e-2,
                                ax=None, ylabel=False, time_min=0, theta=theta,
                                avg=False, conv=True):
    time, vec, _ = solution_mf_sdo_euler_OU_noise(j, b, theta=theta, noise=noise, tau=tau,
                                                  time_end=time_end, dt=dt,
                                                  tau_n=tau)
    if ax is None:
        fig, ax = plt.subplots(1)
    if not avg:
        for q in vec.T:
            # q = np.convolve(q, np.ones(10)/10, mode='same')
            ax.plot(time[time >= time_min], q[time >= time_min])
    else:
        vals = np.nanmean(vec[time >= time_min].T, axis=0)
        if conv:
            # vals = np.convolve(vals, np.ones(100)/100, mode='valid')
            x = time[:len(vals)][::400]
            y = vals[::400]
        else:
            x = time[:len(vals)][::50]
            y = vals[::50]
        line = colored_line(x, y, y, ax, linewidth=2, cmap='coolwarm_r', 
                            norm=plt.Normalize(vmin=0,vmax=1))
        # ax.plot(time[:len(vals)][::400], vals[::400], color='k')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-1, 101)
    ax.set_xticks([0, 100])
    ax.set_title('J = ' + str(j) + ', B = ' + str(b), fontsize=12)
    if ylabel:
        ax.set_ylabel(r'$q(x=1)$')
        ax.set_yticks([0, 0.5, 1])
    else:
        ax.set_yticks([])
    if avg:
        return line


def plot_eigenvals_and_behavior_MF_2_faces(b=0, diag_neg=True):
    if not diag_neg:
        eigval_fun_1 = get_eigen_1
        eigval_fun_2 = get_eigen_2
        f_i_fun = f_i_diagonal
        ini_conds = [0.1, 0.9, 0.48]
    else:
        eigval_fun_1 = get_eigen_1_neg
        eigval_fun_2 = get_eigen_2_neg
        f_i_fun = f_i_diagonal_neg  # 2d input
        ini_conds = [[0.1, 0.9], [0.48, 0.52], [0.9, 0.1]]
    # plots eigenvalues and solutions
    fig2, ax_2 = plt.subplots(ncols=2, figsize=(9, 4))
    ax2, ax3 = ax_2
    j_list = np.arange(0, 4+1e-3, 1e-3)
    sol_j_1 = []
    sol_j_2 = []
    sol_j_3 = []
    for j in j_list:
        sol_j_1.append(fsolve(f_i_fun, ini_conds[0], args=(j, b)))
        sol_j_2.append(fsolve(f_i_fun, ini_conds[1], args=(j, b)))
        sol_j_3.append(fsolve(f_i_fun, ini_conds[2], args=(j, b)))
    for ax in ax_2:
        ax.plot(sol_j_1, j_list, color='k')
        ax.plot(sol_j_2, j_list, color='k')
        ax.plot(sol_j_3, j_list, color='k')
    x = np.arange(0, 1+1e-3, 1e-3)
    x_g, j_g = np.meshgrid(x, j_list)
    # lam_1 = 1*(get_eigen_1(j_g, x_g) > 0)*2-1
    # lam_2 = 1*(get_eigen_2(j_g, x_g) > 0)*2-1
    lam_1 = eigval_fun_1(j_g, x_g, b)
    lam_2 = eigval_fun_2(j_g, x_g, b)
    v_min = np.min((lam_1, lam_2))
    v_max = np.max((lam_1, lam_2))
    v_abs_max = np.max((np.abs(v_min), np.abs(v_max)))
    # v_abs_max = 1
    ax2.set_xlabel('x')
    ax3.set_xlabel('x')
    ax3.set_yticks([])
    ax2.set_ylabel('J')
    ax2.imshow(np.flipud(lam_1), cmap='coolwarm', vmin=-v_abs_max,
               vmax=v_abs_max, extent=[0, 1, 0, np.max(j_list)], aspect='auto')
    im_2 = ax3.imshow(np.flipud(lam_2), cmap='coolwarm', vmin=-v_abs_max,
                      vmax=v_abs_max, extent=[0, 1, 0, np.max(j_list)], aspect='auto')
    plt.colorbar(im_2, ax=ax3, label=r'$\lambda$')
    # try to compact everything in single figure
    fig, ax = plt.subplots(1)
    lam_1 = 1*(eigval_fun_1(j_g, x_g, b) > 0)*2-1
    lam_2 = 1*(eigval_fun_2(j_g, x_g, b) > 0)*2-1
    lam_1_lam_2 = lam_1 * lam_2
    ax.imshow(np.flipud(lam_1_lam_2 + lam_2), cmap='Pastel1',
              extent=[0, 1, 0, np.max(j_list)], aspect='auto')
    ax.text(0.15, 0.15, 'Stable fixed point')
    ax.text(0.35, 0.6, 'Saddle\nnode')
    ax.text(0.52, 1.17, 'Unstable fixed point', rotation='vertical')
    if diag_neg:
        label = 'Fixed points in y=1-x'
    else:
        label = 'Fixed points in y=x'
    ax.plot(sol_j_1, j_list, color='k')
    legendelements = [Line2D([0], [0], color='k', lw=2, label=label)]
    ax.legend(bbox_to_anchor=(1., 1.12), frameon=False, handles=legendelements)
    ax.plot(sol_j_2, j_list, color='k')
    ax.plot(sol_j_3, j_list, color='k')
    ax.set_ylabel('J')
    ax.set_xlabel('x')
    if b == 0:
        ax.set_yticks([0, 1/3, 0.5, 1, 1.5, 2])
        ax.axhline(1/3, color='r', linestyle='--', alpha=0.6)
    else:
        ax.set_yticks([0, 0.5, 1, 1.5, 2])
    

def get_eigen_1(j, x, b=0):
    return gn.sigmoid(6*j*(2*x-1)+2*b)*(1-gn.sigmoid(6*j*(2*x-1)+2*b))*12*j-1


def get_eigen_2(j, x, b=0):
    return gn.sigmoid(6*j*(2*x-1)+2*b)*(1-gn.sigmoid(6*j*(2*x-1)+2*b))*4*j-1


def k_1(x, j, b=0):
    return gn.sigmoid(2*(j*(2*x-1)+b))*(1-gn.sigmoid(2*(j*(2*x-1)+b)))


def k_2(x, j, b=0):
    return gn.sigmoid(2*(j*(-2*x+1)+b))*(1-gn.sigmoid(2*(j*(-2*x+1)+b)))


def get_eigen_1_neg(j, x, b=0):
    k1 = k_1(x, j, b)
    k2 = k_2(x, j, b)
    return -1 + 4*j * (k1+k2 + np.sqrt(k1**2 + k2**2 - k1*k2))


def get_eigen_2_neg(j, x, b=0):
    k1 = k_1(x, j, b)
    k2 = k_2(x, j, b)
    return -1 + 4*j * (k1+k2 - np.sqrt(k1**2 + k2**2 - k1*k2))


def eigens_neg_2_var(j, b=0):
    x = fsolve(f_i_diagonal_neg, [0.1, 0.9], args=(j[0], b))
    return get_eigen_1_neg(j, x[0], b=0)


def mf_sdo(t, x, j, b, theta, noise, tau):
    x = gn.sigmoid(2*j*(2*np.matmul(theta, x)-3) + 2*b) - x + np.random.randn(8)*noise
    return x / tau


def mf_sdo_2_faces(t, x, j, b, noise, tau):
    # q_idx = np.arange(8)
    # np.random.shuffle(q_idx)
    # for q in q_idx:
    #     neighbours = theta[q].astype(dtype=bool)
    #     x[q] = gn.sigmoid(2*j*sum(2*x[neighbours]-1) + 2*b) -\
    #         x[q] + np.random.rand()*noise
    xi = x[0]
    xj = x[1]
    xi1 = f_i(xi, xj, j=j, b=b) + np.random.randn()*noise
    xj1 = f_i(xj, xi, j=j, b=b) + np.random.randn()*noise
    return np.array((xi1, xj1)) / tau


def solution_mf_sdo(j, b, theta, noise, tau):
    init_cond = np.random.rand(8)
    t_eval = np.arange(0, 200, 1)
    y0 = np.copy(init_cond)
    m_solution = solve_ivp(fun=mf_sdo, t_span=[0, 200],
                           t_eval = t_eval, y0=y0,
                           args=(j, b, theta, noise, tau))
    return m_solution.t, m_solution.y.T


def solution_mf_sdo_euler(j, b, theta, noise, tau, time_end=50, dt=1e-2,
                          ini_cond=None):
    time = np.arange(0, time_end+dt, dt)
    if ini_cond is None:
        x = np.random.rand(theta.shape[0])  # initial_cond
    else:
        x = ini_cond
    x_vec = np.empty((len(time), theta.shape[0]))
    x_vec[:] = np.nan
    x_vec[0, :] = x
    t_cte_noise = np.sqrt(dt/tau)
    noise_vec = np.random.randn(time.shape[0], theta.shape[0])*t_cte_noise*noise
    for t in range(1, time.shape[0]):
        x = x + dt*(gn.sigmoid(2*j*(np.matmul(theta, 2*x-1)) + 2*b) - x)/ tau +\
            noise_vec[t] + np.random.randn()*0.05*t_cte_noise        # x = np.clip(x, 0, 1)
        x_vec[t, :] = x  # np.clip(x, 0, 1)
    return time, x_vec


def solution_mf_sdo_euler_OU_noise(j, b, theta, noise, tau, time_end=50, dt=1e-2,
                                   tau_n=1, approx_init=False):
    # print('Start simulating MF with OU noise w/o adaptation')
    time = np.arange(0, time_end+dt, dt)
    if approx_init:
        x = 0.5 + np.random.randn(theta.shape[0])*0.1
    else:    
        x = np.random.rand(theta.shape[0])  # initial_cond
    x_vec = np.empty((len(time), theta.shape[0]))
    ou_vec = np.empty((len(time), theta.shape[0]))
    x_vec[:] = np.nan
    x_vec[0, :] = x
    ou_vec[:] = np.nan
    ou_val = np.random.rand(theta.shape[0])
    ou_vec[0, :] = ou_val
    n_neighs = np.matmul(theta, np.ones(theta.shape[0]))
    for t in range(1, time.shape[0]):
        x_n = (dt*(gn.sigmoid(2*j*(2*np.matmul(theta, x)-n_neighs) + 2*b) - x)) / tau
        ou_val = dt*(-ou_val / tau_n) + (np.random.randn(theta.shape[0])*noise*np.sqrt(dt/tau_n))
        x = x + x_n + ou_val
        # x = np.clip(x, 0, 1)
        x_vec[t, :] = x  # np.clip(x, 0, 1)
        ou_vec[t, :] = ou_val
    return time, x_vec, ou_vec


def solution_mf_sdo_euler_OU_noise_adaptation(j, b, theta, noise, tau, gamma_adapt=0.1,
                                              time_end=50, dt=1e-2,
                                              tau_n=1):
    print('Start simulating MF with OU noise and adaptation')
    time = np.arange(0, time_end+dt, dt)
    x = np.random.rand(theta.shape[0])
    x_vec = np.empty((len(time), theta.shape[0]))
    ou_vec = np.empty((len(time), theta.shape[0]))
    adapt_vec = np.empty((len(time), theta.shape[0]))
    x_vec[:] = np.nan
    x_vec[0, :] = x
    ou_vec[:] = np.nan
    ou_val = np.random.randn(theta.shape[0])/100
    ou_vec[0, :] = ou_val
    adapt_vec[:] = np.nan
    adapt_val = gamma_adapt*x*dt
    adapt_vec[0, :] = adapt_val
    n_neighs = np.matmul(theta, np.ones(theta.shape[0]))
    for t in range(1, time.shape[0]):
        ou_val_diff = dt*(-ou_val / tau_n) + (np.random.randn(theta.shape[0])*noise*np.sqrt(2*dt/tau_n))
        adapt_val = adapt_val + dt*(-adapt_val + gamma_adapt*x)/tau
        x_n = (dt*(gn.sigmoid(2*j*(2*np.matmul(theta, x)-n_neighs) + 2*b-adapt_val) - x)) / tau
        x = x + x_n + ou_val_diff
        # r_neuron = gn.sigmoid(x-adapt_val)-x
        x_vec[t, :] = x
        ou_val = ou_val + ou_val_diff
        ou_vec[t, :] = ou_val
        adapt_vec[t, :] = adapt_val
    return time, x_vec, ou_vec, adapt_vec


def plot_adaptation_mf(j, b, theta=theta, noise=0.1, gamma_adapt=0.1,
                       tau=0.008, time_end=1000, dt=1e-3):
    time, vec, ouvals, adapt =\
        solution_mf_sdo_euler_OU_noise_adaptation(j, b, theta, noise, tau,
                                                  time_end=time_end, dt=dt,
                                                  gamma_adapt=gamma_adapt,
                                                  tau_n=tau)
    fig, ax = plt.subplots(nrows=3, figsize=(10, 8))
    ax = ax.flatten()
    # mean_states = np.mean(vec, axis=1)  # np.clip(np.mean(vec, axis=1), 0, 1)
    # mean_adapt = np.mean(adapt, axis=1)  # np.clip(np.mean(adapt, axis=1), 0, 1)
    [ax[0].plot(time, vec[:, i]) for i in range(8)]
    [ax[1].plot(time, adapt[:, i]) for i in range(8)]
    [ax[2].plot(time, ouvals[:, i]) for i in range(8)]
    ax[0].set_ylabel('Approx. posterior')
    ax[1].set_ylabel('Adaptation')
    ax[2].set_ylabel('OU-noise')
    ax[2].set_xlabel('Time (s)')
    fig.tight_layout()
    # ax[1].plot(mean_adapt)


def plot_adaptation_1d(j, b, noise, tau, gamma_adapt=0.1,
                       time_end=50, dt=1e-2, tau_n=0.1):
    time = np.arange(0, time_end+dt, dt)
    q = np.random.randn()
    q_vec = np.empty((len(time)))
    ou_vec = np.empty((len(time)))
    adapt_vec = np.empty((len(time)))
    q_vec[:] = np.nan
    q_vec[0] = q
    ou_vec[:] = np.nan
    ou_val = np.random.randn()/1000
    ou_vec[0] = ou_val
    adapt_vec[:] = np.nan
    adapt_val = gamma_adapt*q*dt
    adapt_vec[0] = adapt_val
    for t in range(1, time.shape[0]):
        ou_val_diff = dt*(-ou_val / tau_n) + (np.random.randn(1)*noise*np.sqrt(2*dt/tau_n))
        q_n = dt*(j*3*np.tanh(q-adapt_val)+b-q) / tau
        adapt_val = adapt_val + dt*(-adapt_val + gamma_adapt*q)/tau
        q = q + q_n + ou_val_diff
        q_vec[t] = q
        ou_val = ou_val + ou_val_diff
        ou_vec[t] = ou_val
        adapt_vec[t] = adapt_val
    fig, ax = plt.subplots(nrows=3, figsize=(10, 8))
    ax = ax.flatten()
    ax[0].plot(time, q_vec)
    ax[1].plot(time, adapt_vec)
    ax[2].plot(time, ou_vec)
    ax[0].set_ylabel('Approx. posterior')
    ax[1].set_ylabel('Adaptation')
    ax[2].set_ylabel('OU-noise')
    ax[2].set_xlabel('Time (s)')
    fig.tight_layout()
    

def solution_mf_sdo_2_faces_euler(j, b, theta, noise, tau, init_cond,
                                  time_end=50, dt=1e-2):
    def f_i(x1, x2, j, b):
        return gn.sigmoid(2*j*3*(2*x2-1) + 2*b)-x1
    time = np.arange(0, time_end+dt, dt)
    x1 = init_cond[0]  # initial_cond for q1
    x2 = init_cond[1]  # initial_cond for q2
    x_vec1 = np.empty((len(time)))
    x_vec2 = np.empty((len(time)))
    x_vec1[:] = np.nan
    x_vec2[:] = np.nan
    x_vec1[0] = x1
    x_vec2[0] = x2
    for t in range(1, time.shape[0]):
        x1 = np.copy(x_vec1[t-1])
        x2 = np.copy(x_vec2[t-1])
        x1_temp = x1 + (dt*(f_i(x1, x2, j=j, b=b)) +\
            np.sqrt(dt)*noise*np.random.randn()) / tau
        x2_temp = x2 + (dt*(f_i(x2, x1, j=j, b=b)) +\
            np.sqrt(dt)*noise*np.random.randn()) / tau
        # x1_temp = np.clip(x1_temp, 0, 1)
        # x2_temp = np.clip(x2_temp, 0, 1)
        x_vec1[t] = x1_temp
        x_vec2[t] = x2_temp
        
    result = type('result', (object,), {})
    result.y = np.row_stack((x_vec1, x_vec2))
    result.t = time
    return result


def plot_occupancy_distro(j, noise=0.3, tau=1, dt=1e-1, theta=theta, b=0,
                          t=100, burn_in=0.1, n_sims=20):
    fig, ax = plt.subplots(1)
    time_end = t*(1+burn_in)
    vec = np.array(())
    for n in range(n_sims):
        time, vec_sims = solution_mf_sdo_euler(j, b, theta, noise, tau,
                                               time_end=time_end, dt=dt)
        vec_sims = vec_sims[time > t*burn_in]
        # vec_sims = np.nanmean(vec_sims, axis=1)
        vec_sims = vec_sims[:, 0]
        vec = np.concatenate((vec, vec_sims))
    print(len(vec))
    kws = dict(histtype= "stepfilled", linewidth = 1)
    ax.hist(vec, label=int(t*n_sims), cumulative=True, bins=150, density=True,
            color='mistyrose', edgecolor='k', **kws)
    # sns.kdeplot(vec, label='simulation', bw_adjust=0.05, cumulative=True, color='k')
    plot_boltzmann_distro(j, noise, b=b, ax=ax)
    plt.legend()
    ax.set_ylabel('CDF(x)')
    ax.set_xlabel('x')
    ax.set_xlim(-0.05, 1.05)
    ax.set_title('J =' + str(j) + ', B =' + str(b) + r', $\sigma$=' + str(noise))


def transition_rate_analysis(j_list=np.arange(0.35, 2, 2e-1),
                             b=0, theta=theta, noise=0.1, tau=0.1, dt=1e-2, t_end=30000):
    transition_rate = []
    transition_rate_analytical_list = []
    for i_j, j in enumerate(j_list):
        x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j, b)
        transition_rate_analytical = k_i_to_j(j, x_stable_1, x_unstable, noise, b)
        transition_rate_analytical_list.append(transition_rate_analytical)
        print('J = ' + str(j))
        t, vec = solution_mf_sdo_euler(j, b, theta, noise, tau, time_end=t_end, dt=dt)
        mean_states = np.clip(np.mean(vec, axis=1), 0, 1)
        # mean_states = np.convolve(mean_states, np.ones(1000)/1000, mode='valid')
        p_thr = 0.5
        mean_states[mean_states > p_thr] = 1
        mean_states[mean_states < (1-p_thr)] = 0
        mean_states = mean_states[mean_states != p_thr]
        orders = gn.rle(mean_states)
        transition_time = orders[0]
        transition_rate.append(1/np.mean(transition_time))
    plt.figure()
    plt.plot(j_list[:len(transition_rate_analytical_list)], transition_rate_analytical_list, label='1D analytical',
             color='k', linewidth=2.5)
    plt.plot(j_list[:len(transition_rate_analytical_list)], np.array(transition_rate), label='Simulation', color='r', linewidth=2.5)
    plt.legend()
    plt.xlabel('Coupling, J')
    plt.ylabel('Jumps')
    plt.yscale('log')


def escape_time(j, xi, xj, noise, b=0):
    v_2_xi = second_derivative_potential(xi, j, b=b)
    v_2_xj = second_derivative_potential(xj, j, b=b)
    v_xi = potential_mf(xi, j, b)
    v_xj = potential_mf(xj, j, b)
    return 1/np.sqrt(np.abs(v_2_xi*v_2_xj))*np.exp(2*(v_xi - v_xj)/noise**2) * np.pi


def analyze_jump_dynamics(j, b, theta, noise=0.35, tau=0.1, dt=1e-3, t_end=1000,
                          steps_back=250, steps_front=250):
    idx_0 = []
    idx_1 = []
    while len(idx_0) < 1 and len(idx_1) < 1:
        t, vec = solution_mf_sdo_euler(j, b, theta, noise, tau, time_end=t_end, dt=dt)
        mean_states = np.clip(np.mean(vec, axis=1), 0, 1)
        mean_states = np.convolve(mean_states, np.ones(1000)/1000, mode='same')
        p_thr = 0.5
        mean_states[mean_states > p_thr] = 1
        mean_states[mean_states < (1-p_thr)] = 0
        mean_states = mean_states[mean_states != p_thr]
        orders = gn.rle(mean_states)
        idx_1 = orders[1][orders[2] == 1]
        idx_0 = orders[1][orders[2] == 0]
    idx_1 = orders[1][orders[2] == 1]
    idx_1 = idx_1[(idx_1 > steps_back) & (idx_1 < (len(mean_states))-steps_front)]
    idx_0 = orders[1][orders[2] == 0]
    idx_0 = idx_0[(idx_0 > steps_back) & (idx_0 < (len(mean_states))-steps_front)]
    # order depending on jump
    mean_vals_1_array = np.empty((theta.shape[0], len(idx_1), steps_back+steps_front))
    mean_vals_1_array[:] = np.nan
    # original order
    mean_vals_nor_array = np.empty((theta.shape[0], len(idx_1), steps_back+steps_front))
    mean_vals_nor_array[:] = np.nan
    conv_window = 5
    fig, ax = plt.subplots(ncols=5, nrows=4)
    ax = ax.flatten()
    for i in range(len(idx_1)):
        idx = idx_1[i]
        # vals_idx = np.argsort(np.sum(vec[idx - steps_back:idx, :] -
        #                              np.mean(vec[idx - steps_back:idx, :], axis=0), axis=0))
        vals_idx = np.argsort(np.sum(vec[idx- steps_back//4:idx + steps_front//4, :], axis=0))
        for i_k, k in enumerate(vals_idx):
        # for k in range(8):
            mean_vals_1_array[i_k, i, :] = vec[idx - steps_back:idx+steps_front, k]
            # np.convolve(vec[idx - steps_back:idx+steps_front, k],
            #                                     np.ones(conv_window)/conv_window,
            #                                     mode='same')
        for k in range(8):
            mean_vals_nor_array[k, i, :] = np.convolve(vec[idx - steps_back:idx+steps_front, k],
                                                np.ones(conv_window)/conv_window,
                                                mode='same')
        if i < len(ax):
            for k in range(8):
                ax[i].plot(t[idx_1[i]-steps_back:idx_1[i]+steps_front], mean_vals_1_array[k, i, :])
                ax[i].set_xlabel('Time (s)')
                ax[i].set_ylabel('Approx. post.')
    fig.tight_layout()
    fig.suptitle('J = ' + str(j) + ', B = ' + str(b))
    mean_vals_mean = np.nanmean(mean_vals_1_array, axis=1)
    x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j, b)
    plt.figure()
    plt.axhline(x_unstable, color='k', linestyle='--', alpha=0.4)
    for i in range(theta.shape[0]):
        plt.plot(np.arange(-steps_back, steps_front-1, 1)*dt,
                  mean_vals_mean[i, :-1])
    plt.xlabel('Time from switch (s)')
    plt.ylabel('Approximate posterior of ordered nodes')
    plt.title('J = ' + str(j) + ', B = ' + str(b))
    plt.figure()
    mean_vals_mean = np.nanmean(mean_vals_nor_array, axis=1)
    plt.axhline(x_unstable, color='k', linestyle='--', alpha=0.4)
    for i in range(theta.shape[0]):
        plt.plot(np.arange(-steps_back, steps_front-1, 1)*dt,
                  mean_vals_mean[i, :-1])
    plt.xlabel('Time from switch (s)')
    plt.ylabel('Approximate posterior of node i')
    plt.title('J = ' + str(j) + ', B = ' + str(b))
    # eigenvectors projections
    fig, ax = plt.subplots(ncols=2)
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    # mean_vals_1_array is 8 x n_jumps x timepoints
    eigvects = np.linalg.eig(theta)[1].T
    eigvals = np.round(np.linalg.eig(theta)[0], 4)
    eigvects_p = eigvects[(eigvals == 3) + (eigvals == 1)]
    eigvals_p = eigvals[(eigvals == 3) + (eigvals == 1)]
    eigvects_n = eigvects[(eigvals == -3) + (eigvals == -1)]
    eigvects_n = np.row_stack((np.array(sympy.Matrix(theta).eigenvects()[0][2]).reshape(1, -1)/np.sqrt(8),
                               np.array(sympy.Matrix(theta).eigenvects()[1][2]).reshape(3, -1)/np.sqrt(4)))
    eigvects_p = np.row_stack((np.ones(8)/np.sqrt(8), np.array(sympy.Matrix(theta).eigenvects()[-2][2]).reshape(3, -1)/np.sqrt(4)))
    eigvals_n = eigvals[(eigvals == -3) + (eigvals == -1)]
    projections_p = np.zeros((4, len(idx_1), steps_back+steps_front))
    projections_n = np.zeros((4, len(idx_1), steps_back+steps_front))
    for jump in range(len(idx_1)):
        for i in range(4):
            projections_p[i, jump, :] = np.dot(mean_vals_1_array[:, jump].T, eigvects_p[i])
            projections_n[i, jump, :] = np.dot(mean_vals_1_array[:, jump].T, eigvects_n[i])
    avg_proj_per_jump_p = np.mean(projections_p, axis=1)
    avg_proj_per_jump_n = np.mean(projections_n, axis=1)
    norm = np.mean(np.sqrt(np.sum(projections_p[1:]**2, axis=0)), axis=0)
    for proj in range(4):
        ax[0].plot((np.arange(-steps_back, steps_front, 1))*dt, avg_proj_per_jump_p[proj, :],
                 label=eigvals_p[proj])
        ax[1].plot((np.arange(-steps_back, steps_front, 1))*dt, avg_proj_per_jump_n[proj, :],
                   label=eigvals_n[proj])
    fig4, ax4 = plt.subplots(ncols=4)
    colors = ['darkorange', 'seagreen', 'purple']
    for i in range(1, 4):
        for j in range(len(idx_1)):
            ax4[i-1].plot((np.arange(-steps_back, steps_front, 1))*dt, projections_p[i, j], color=colors[i-1],
                         alpha=0.2)
            ax4[-1].plot((np.arange(-steps_back, steps_front, 1))*dt, projections_p[i, j], color=colors[i-1],
                         alpha=0.2)
        ax4[i-1].set_title('Lambda_{}=1'.format(i))
        ax4[i-1].set_xlabel('Time from switch (s)')
        ax4[i-1].set_ylabel('Projection of activity')
    f2, ax2 = plt.subplots(ncols=1)
    # ax2.plot((np.arange(-steps_back, steps_front, 1))*dt, norm, color='k')
    for j in range(len(idx_1)):
        norm = np.sqrt(np.sum(projections_p[1:, j]**2, axis=0))
        ax2.plot((np.arange(-steps_back, steps_front, 1))*dt, norm, color='k', alpha=0.3)
    norm = np.mean(np.sqrt(np.sum(projections_p[1:]**2, axis=0)), axis=0)
    ax2.plot((np.arange(-steps_back, steps_front, 1))*dt, norm, color='r', alpha=1)
    ax2.set_xlabel('Time from switch (s)')
    ax2.set_ylabel('Projection of activity - Norm')
    ax[0].legend(title='Eigenvalue')
    ax[1].legend(title='Eigenvalue')
    ax[0].set_xlabel('Time from switch (s)')
    ax[1].set_xlabel('Time from switch (s)')
    ax[0].set_ylabel('Projection of activity')
    ax[1].set_ylim(-0.6, 0.6)


def vector_proj(u, v):
    return (np.dot(u, v)/np.dot(v, v))*v 


def get_n_eigenvects(n, theta):
    eigvals, eigvects = np.linalg.eig(theta)
    sorted_evals_idx = np.argsort(np.linalg.eig(theta)[0])[::-1]
    return eigvects[sorted_evals_idx[:n]]


def projection_mf_plot(theta, j=1, b=0, noise=0, tau=0.01):
    t, X = solution_mf_sdo_euler(j, b, theta, noise, tau, time_end=100, dt=1e-3)
    # from sklearn.decomposition import PCA
    # PCA = PCA(n_components=4)
    # components = PCA.fit_transform(X)
    # PCA.components_
    fig, ax = plt.subplots(1)
    for i in range(X.shape[1]):
        ax.plot(t, X[:, i], label='Node ' + str(i), alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('q')
    evects = get_n_eigenvects(4, theta).T
    val_act = np.matmul(X, evects)
    ax.plot(-val_act[:, 0], label='D1', color='r')
    ax.legend()
    fig, ax = plt.subplots(1)
    ax.plot(np.nanmean(X, axis=1), -val_act[:, 0], color='k')
    ax.set_xlabel(r'$<\vec{x}>$')
    ax.set_ylabel('proj. to evec')
    fig, ax = plt.subplots(1)
    ax.plot(val_act[:, 0], label='D1')
    ax.plot(val_act[:, 1], label='D2')
    ax.plot(val_act[:, 2], label='D3')
    ax.plot(val_act[:, 3], label='D4')
    ax.set_xlabel('Time')
    ax.set_ylabel('Projection to evec')
    ax.legend()
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot3D(val_act[:, 0], val_act[:, 1], val_act[:, 2], color='r', label='D1')
    ax.plot3D(-np.nanmean(X, axis=1), val_act[:, 1], val_act[:, 2], color='k', alpha=0.3,
              label=r'$<\vec{x}>$')
    ax.legend()
    ax.set_xlabel('D1')
    ax.set_ylabel('D2')
    ax.set_zlabel('D3')
    ax.plot3D(val_act[0, 0], val_act[0, 1], val_act[0, 2], color='r', marker='o')
    ax.plot3D(val_act[-1, 0], val_act[-1, 1], val_act[-1, 2], color='r', marker='x')


def predictions_boltzmann_distro(j_list=[0.15, 0.5], n=3.92, noise=0.15,
                                 b_list=[0, 0.4, 0.8, 1],
                                 ntrials=1000, tmax=1, dt=0.01, tau=0.1, bw=0.8):
    time = np.arange(0, tmax, dt)
    signed_confidence_array = np.zeros((ntrials, len(b_list), len(j_list)))
    for i_j, j in enumerate(j_list):
        for i_b, b in enumerate(b_list):
            for trial in range(ntrials):
                weight_stim = 2*np.abs(np.random.rand()-0.5)
                j_eff = j+np.random.rand()*0.1
                b_signed = b*np.random.choice([-1, 1])*weight_stim
                bias = np.random.randn()*0.1*2
                x = np.random.rand()
                for i in range(len(time)):
                    x = x + dt*(gn.sigmoid(2*j_eff*n*(2*x-1)+2*b_signed + bias)-x)/tau + np.random.randn()*noise*np.sqrt(dt/tau)
                signed_confidence_array[trial, i_b, i_j] = (2*x-1)*np.sign(b_signed+1e-6*np.random.randn())
    colormap = pl.cm.gist_gray_r(np.linspace(0.3, 1, len(b_list)))
    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.set_xlim(-1.7, 1.7)
        a.set_xticks([-1, 0, 1])
        a.set_ylim(-0.05, 1.6)
        a.set_xlabel('')
    legendelements = []
    for ia in range(4):
        sns.kdeplot(signed_confidence_array[:, ia, 0],
                    alpha=1, lw=3., common_norm=False, ax=ax[0],
                    legend=False, bw_adjust=bw, color=colormap[ia])
        sns.kdeplot(signed_confidence_array[:, ia, 1],
                    alpha=1, lw=3., common_norm=False, ax=ax[1],
                    legend=False, bw_adjust=bw, color=colormap[ia])
        legendelements.append(Line2D([0], [0], color=colormap[ia],
                                     lw=3.5, label=b_list[ia]))
    ax[0].legend(frameon=False, title='Stimulus\nstrength', handles=legendelements,
                 bbox_to_anchor=(0.4, 0.4))
    ax[1].set_ylabel('')
    ax[0].set_ylabel('Density of confidence', fontsize=19)
    ax[0].set_xlabel('                                                 Confidence aligned with stimulus',
                     fontsize=19)
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[0].set_title('Monostable', fontsize=19)
    ax[1].set_title('Bistable', fontsize=19)
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'density_stim_str_prediction.png', dpi=400, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'density_stim_str_prediction.svg', dpi=400, bbox_inches='tight')


def potential_stim_str(stims=[0, 0.4, 0.8, 1], j_list=[0.15, 0.5]):
    q = np.arange(-0.3, 1.3, 1e-3)
    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.set_xlim(-1.7, 1.7)
        a.set_xticks([-1, 0, 1])
        a.set_xlabel('')
        a.set_yticks([])
    colormap = pl.cm.gist_gray_r(np.linspace(0.3, 1, len(stims)))
    for i_j, j in enumerate(j_list):
        for i_s, s in enumerate(stims):
            pot = potential_mf_neighs(q, j, bias=s*0.5, neighs=3.92)
            ax[i_j].plot(q*2-1, pot-np.mean(pot), color=colormap[i_s], linewidth=3.5, label=s)
    ax[0].legend(frameon=False, title='Stimulus\nstrength')
    ax[0].set_ylabel(r'Potential $V(q)$', fontsize=19)
    ax[0].set_title('Monostable', fontsize=19)
    ax[1].set_title('Bistable', fontsize=19)
    ax[0].set_xlabel('                                                 Confidence aligned with stimulus',
                     fontsize=19)
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'potentials_stim_str_prediction.png', dpi=400, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'potentials_stim_str_prediction.svg', dpi=400, bbox_inches='tight')


def mutual_inh_cartoon(inh=2.1, exc=2.1, n_its=10000, noise=0.025, tau=0.2,
                       skip=25):
    x1, x2 = 0.5, 0.5
    x1l = []
    x2l = []
    dt = 0.05/tau
    time = np.arange(0, n_its+1, skip)*dt
    for i in range(n_its+skip+1):
        x1 += dt*(-x1 + gn.sigmoid(exc*x1 - inh*x2)) + np.random.randn()*np.sqrt(dt)*noise
        x2 += dt*(-x2 + gn.sigmoid(exc*x2 - inh*x1)) + np.random.randn()*np.sqrt(dt)*noise
        if i % skip == 0 and i > 0:
            x1l.append(x1)
            x2l.append(x2)
    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(7, 6))
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    ax[0].plot(time, x1l, color='firebrick', label=r'$r_A (t)$',
               linewidth=2.4)
    ax[0].set_ylim(0.1, 1.0)
    ax[0].plot(time, x2l, color='navy', label=r'$r_B (t)$',
               linewidth=2.4)
    ax[0].set_ylabel('Firing rate')
    ax[0].legend(frameon=False, ncol=2, loc='upper center')
    for y in [-0.375, 0, 0.375]:
        ax[1].axhline(y, color='grey', linestyle='--', alpha=0.7)
    ax[1].text(time[-1]+90, -0.52, 'Vase', color='gray')
    ax[1].text(time[-1]+90, 0.45, 'Faces', color='gray')
    ax[1].set_ylim(-0.6, 0.6)
    ax[1].set_xlim(0, np.max(time)+150)
    ax[0].set_xlim(0, np.max(time)+150)
    ax[1].plot(time, np.array(x1l)-np.array(x2l), color='grey',
               linewidth=2.4)
    ax[1].set_ylabel(r'$\Delta r (t)= r_A (t) - r_B (t)$')
    ax[1].set_xlabel('Time (s)')
    fig.tight_layout()
    # plt.ylim(0, 1)



def plot_boltzmann_distro(j, noise, b=0, ax=None, color='r', cumsum=False,
                          n=3):
    if ax is None:
        fig, ax = plt.subplots(1)
    q = np.arange(0, 1.001, 0.001)
    pot = potential_mf_neighs(q, j, bias=b, neighs=n)
    distro = np.exp(-2*pot/noise**2)
    if cumsum:
        yvals = np.cumsum(distro)
    else:
        yvals = distro
    ax.plot(q,  yvals / np.sum(distro),
            color=color, label='analytical')


def second_derivative_potential(q, j, b=0, n=3.):
    expo = 2*n*(j*(2*q-1))+2*b
    return 1 - 4*n*j*gn.sigmoid(expo)*(1-gn.sigmoid(expo))


def k_i_to_j(j, xi, xj, noise, b=0):
    v_2_xi = second_derivative_potential(xi, j, b=b)
    v_2_xj = second_derivative_potential(xj, j, b=b)
    v_xi = potential_mf(xi, j, b)
    v_xj = potential_mf(xj, j, b)
    return np.sqrt(np.abs(v_2_xi*v_2_xj))*np.exp(2*(v_xi - v_xj)/noise**2) / (2*np.pi)


def get_unst_and_stab_fp(j, b, tol=1e-10):
    diff_1 = 0
    diff_2 = 0
    init_cond_unst = 0.5
    q1 = lambda q: gn.sigmoid(6*j*(2*q-1)+ b*2) - q
    sol_1, _, flag, _ =\
        fsolve(q1, 1, full_output=True)
    x_stable_1 = sol_1[0]
    sol_2, _, flag, _ =\
        fsolve(q1, 0, full_output=True)
    x_stable_2 = sol_2[0]
    while np.abs(diff_1) <= tol or np.abs(diff_2) <= tol:
        if np.abs(x_stable_1-x_stable_2) <= tol:
            x_unstable = np.nan
            break
        sol_unstable, _, flag, _ =\
            fsolve(q1, init_cond_unst, full_output=True)
        if flag == 1:
            x_unstable = sol_unstable[0]
        else:
            x_unstable = np.nan
            break
        diff_1 = np.abs(x_unstable - x_stable_1)
        diff_2 = np.abs(x_unstable - x_stable_2)
        init_cond_unst = np.random.rand()
    return x_stable_1, x_stable_2, x_unstable


def plot_dominance_distro_approx(j=0.6, b=0, t_dur=100, noise=0.1,
                                 fit_gamma=False):
    """
    Plots P(T_eff) = P(k_eff) dk_eff/dT_eff,
    with dk_eff/dT_eff = 1/T_eff^2
    P(k_eff) is just taken as a normal distribution (for large T though...)
    """
    if t_dur < 10:
        t_f = t_dur*2
        dt = 1e-2
    else:
        t_f = t_dur
        dt = 1e-1
    t = np.arange(1e-1, t_f, dt)
    x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j, b)
    k_2 = k_i_to_j(j, x_stable_1, x_unstable, noise, b)
    k_1 = k_i_to_j(j, x_stable_2, x_unstable, noise, b)
    k = (k_1+k_2)
    pinf = k_1/k
    p1 = (pinf)*(1-np.exp(-k*t_dur))
    p2 = (1-pinf)*(1-np.exp(-k*t_dur))
    mu = k_1*p2 + k_2*p1  # k_eff, average transition rate
    sig = noise**2 / t_dur
    p_t_eff = 1/t**3 * np.exp(-(1/(1/t-mu))**2 / (2*sig))
    p_t_eff = p_t_eff/np.sum(p_t_eff)
    fig, ax = plt.subplots(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(t, p_t_eff, label='Analytical distro', color='k', linewidth=2.5)
    ax.set_ylabel('Density')
    ax.set_xlabel(r'Effective duration $(1/k_{e})$')
    if fit_gamma:
        # Fit the gamma distribution to the computed P(T_eff)
        fit_alpha, fit_loc, fit_beta = scipy.stats.gamma.fit(p_t_eff)
        y = scipy.stats.gamma.pdf(t, a=fit_alpha, scale=fit_beta, loc=fit_loc)
        # fix_loc_exp, fit_lmb = scipy.stats.expon.fit(p_t_eff)
        # y_exp = scipy.stats.expon.pdf(t, loc=fix_loc_exp, scale=fit_lmb)
        ax.text(75, 0.02, r'$\alpha = ${}'.format(np.round(fit_alpha, 2)) 
                + '\n'+ r'$\beta = ${}'.format(np.round(fit_beta, 2)),
                fontsize=11)
        ax.plot(t, y, label='Gamma distro. fit', color='r', linestyle='--',
                alpha=0.7)
        # plt.plot(t, y_exp, label='Expo. distro. fit', color='b', linestyle='--',
        #          alpha=0.7)
        ax.legend()
    ax.set_yticks([])
    fig.tight_layout()


def alternation_rate_vs_coupling(t_dur=10000, tol=1e-8,
                                 j_list=np.arange(0.6, 1, 0.001),
                                 b=0, noise=0.1):
    k_weighted = np.zeros((len(j_list)))
    for i_j, j in enumerate(j_list):
        x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j, b)
        k_2 = k_i_to_j(j, x_stable_1, x_unstable, noise, b)
        k_1 = k_i_to_j(j, x_stable_2, x_unstable, noise, b)
        k = (k_1+k_2)
        pinf = k_1/k
        p1 = (pinf)*(1-np.exp(-k*t_dur))
        p2 = (1-pinf)*(1-np.exp(-k*t_dur))
        k_weighted[i_j] = k_1*p2 + k_2*p1
    fig, ax = plt.subplots(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(j_list, k_weighted, color='k', linewidth=3)
    ax.set_xlabel('Coupling, J')
    ax.set_ylabel('Alternation rate')
    fig.tight_layout()


def alternation_rate_vs_accuracy(t_dur=10000, tol=1e-8,
                                 j_list=np.arange(0.6, 1, 0.001),
                                 b=0, noise_list=[0.1, 0.2, 0.3]):
    k_weighted = np.zeros((len(j_list), len(noise_list)))
    accuracy = np.zeros((len(j_list), len(noise_list)))
    init_cond = 0.5
    for i_j, j in enumerate(j_list):
        for i_n, noise in enumerate(noise_list):
            x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j, b)
            if b < 0:
                x_stable_2, x_stable_1 = x_stable_1, x_stable_2
            if np.abs(x_stable_1 - x_stable_2) <= tol:
                continue
            # P_{C, 0} = int_{x_E, x_0} exp(2V(x)/sigma^2) dx /
            #            int_{x_E, x_C} exp(2V(x)/sigma^2) dx
            pc0_numerator = scipy.integrate.quad(lambda q: np.exp(2*potential_mf(q, j, b)/noise**2),
                                                 x_stable_2, init_cond)[0]
            pc0_denom = scipy.integrate.quad(lambda q: np.exp(2*potential_mf(q, j, b)/noise**2),
                                             x_stable_2, x_stable_1)[0]
            pc0 = pc0_numerator/pc0_denom
            # compute error transition rates k_CE, K_EC
            k_EC = k_i_to_j(j, x_stable_1, x_unstable, noise, b)
            k_CE = k_i_to_j(j, x_stable_2, x_unstable, noise, b)
            k = k_EC+k_CE
            pCS = k_CE/k  # stationary correct
            # correct to error transition
            pEC = (1-pCS)*(1-np.exp(-k*t_dur))
            # error to correct transition
            pCE = pCS*(1-np.exp(-k*t_dur))
            # correct to correct
            pCC = pCS*(1-np.exp(-k*t_dur))+np.exp(-k*t_dur)
            # probability of correct
            pC = pc0*pCC + (1-pc0)*pCE
            accuracy[i_j, i_n] = pC  # np.max((pC, 1-pC))
            pinf = pCS
            p1 = pCE
            p2 = pEC
            k_weighted[i_j, i_n] = k_CE*p2 + k_EC*p1
    fig, ax = plt.subplots(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    colormap = pl.cm.Blues(np.linspace(0.2, 1, len(noise_list)))
    for i_n in range(len(noise_list)):
        ax.plot(accuracy[:, i_n], k_weighted[:, i_n],
                color=colormap[i_n], linewidth=3)
    ax.set_xlabel('Accuracy')
    ax.set_yscale('log')
    ax.set_ylabel('Alternation rate')
    fig.tight_layout()


def levelts_analytical(t_dur=10000, tol=1e-8,
                       b_list=np.arange(-0.2, 0.201, 0.001),
                       j=0.8, noise=0.1):
    kvals = np.zeros((len(b_list)))
    kCE_vals = np.zeros((len(b_list)))
    k_weighted = np.zeros((len(b_list)))
    kEC_vals = np.zeros((len(b_list)))
    pinfvals = np.zeros((len(b_list)))
    stabvals = np.zeros((len(b_list)))
    init_cond = .5
    for i_b, b in enumerate(b_list):
        x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j, b)
        # if b < 0:
        #     x_stable_2, x_stable_1 = x_stable_1, x_stable_2
        # compute error transition rates k_CE, K_EC
        k_2 = k_i_to_j(j, x_stable_1, x_unstable, noise, b)
        k_1 = k_i_to_j(j, x_stable_2, x_unstable, noise, b)
        k = (k_1+k_2)
        pinf = k_1/k
        kCE_vals[i_b] = k_1
        kEC_vals[i_b] = k_2
        kvals[i_b] = k
        p1 = (pinf)*(1-np.exp(-k*t_dur))
        p2 = (1-pinf)*(1-np.exp(-k*t_dur))
        k_weighted[i_b] = k_1*p2 + k_2*p1
        pinfvals[i_b] = pinf
        pc0_numerator = scipy.integrate.quad(lambda q: np.exp(2*potential_mf(q, j, b)/noise**2),
                                             x_stable_2, init_cond)[0]
        pc0_denom = scipy.integrate.quad(lambda q: np.exp(2*potential_mf(q, j, b)/noise**2),
                                         x_stable_2, x_stable_1)[0]
        pc0 = pc0_numerator/pc0_denom
        p = pc0*(1-p2)+(1-pc0)*p1
        # stabvals[i_b, 0] = x_stable_1*p + x_stable_2*(1-p)
        stabvals[i_b] = p
    fig, ax = plt.subplots(ncols=3, figsize=(17, 5))
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    ax[0].plot(b_list, stabvals, color='g', linewidth=2.5, label='State 1')
    ax[0].plot(b_list, 1-stabvals, color='r', linewidth=2.5, label='State 2')
    ax[0].legend(frameon=False)
    ax[0].set_ylabel('Perceptual predominance')
    ax[0].set_xlabel('Sensory evidence, B')
    xf = stabvals
    entropy = -xf*np.log(xf) - (1-xf)*np.log(1-xf)
    ax[2].plot(b_list, kEC_vals, color='g', linewidth=2.5, alpha=0.4)
    ax[2].plot(b_list, kCE_vals, color='r', linewidth=2.5, alpha=0.4)
    ax[2].plot(b_list, kvals, color='b', linewidth=2.5, linestyle='--',
               label=r'$K_1+K_2$', alpha=0.3)
    ax[2].legend(frameon=False, bbox_to_anchor=[0.4, 1.15])
    ax[2].set_ylabel('Transition rate (1/s)')
    ax[2].set_xlabel('Sensory evidence, B')
    ax2 = ax[2].twinx()
    ax2.plot(b_list, k_weighted,
                   color='k', linewidth=2.5, label=r'$P_1  K_1 + P_2  K_2$')
    ax2.set_ylabel(r'$P_2  K_1 + P_1  K_2$')
    ax2.spines['top'].set_visible(False)
    ax[1].plot(b_list, 1/kCE_vals, color='r', linewidth=2.5,
             alpha=1)
    ax[1].plot(b_list, 1/kEC_vals, color='g', linewidth=2.5,
               alpha=1)
    ax[1].set_ylabel('Average predominance (s)')
    ax[1].set_xlabel('Sensory evidence, B')
    ax[1].set_yscale('log')
    fig.tight_layout()
    fig2, ax_2 = plt.subplots(1)
    ax3 = ax_2.twinx()
    ax3.spines['top'].set_visible(False)
    ax_2.spines['top'].set_visible(False)
    ax3.set_ylabel('Entropy')
    ax_2.set_xlabel('Perceptual predominance')
    ax_2.set_ylabel('Alternation rate')
    ax_2.plot(stabvals, k_weighted, color='k', linewidth=2.5)
    ax3.plot(xf, entropy, color='gray', linewidth=2.5, linestyle='--')
    f4, a4 = plt.subplots(1)
    a4.spines['top'].set_visible(False)
    a4.spines['right'].set_visible(False)
    list_time_q2 = 1/kCE_vals[::12]
    list_time_q1 = 1/kEC_vals[::12]
    x_vals = np.arange(1, np.max((max(list_time_q2), max(list_time_q1)))*2, 5000)
    linereg = scipy.stats.linregress(np.log(list_time_q1), np.log(list_time_q2))
    y = np.log(x_vals)*linereg.slope + linereg.intercept
    a4.plot(np.log(list_time_q2), np.log(list_time_q1),
            color='k', linewidth=2.5,
            marker='o', label='Analytical')
    a4.plot(np.log(x_vals), y, color='b', linestyle='--', alpha=0.4,
            label=f'y ~ log(x), slope={round(linereg.slope, 3)}')
    a4.set_xlabel('log T(x=1)')
    a4.set_ylabel('log T(x=-1)')
    a4.set_ylim(np.log(np.min((np.min(list_time_q2), np.min(list_time_q1))))-1,
                np.log(np.max((np.max(list_time_q2), np.max(list_time_q1))))+1)
    a4.set_xlim(np.log(np.min((np.min(list_time_q2), np.min(list_time_q1))))-1,
                np.log(np.max((np.max(list_time_q2), np.max(list_time_q1))))+1)
    a4.legend(frameon=False)
    trans_rate_vs_noise(noise_list = np.arange(0.05, 0.5, 1e-2),
                        b=0.0, j=j, t_dur=t_dur)


def plot_entropy_approximation():
    q = np.arange(1e-4, 1, 1e-4)
    fig, ax = plt.subplots(1)
    ax.plot(q, -(q*(q-1) + (q-1)*(q)), linewidth=2.5,color='b',
            label='Approximation')
    ax.legend(frameon=False)
    # ax.set_ylabel(r'Approx., $H(q) \approx -q(q-1) - (1-q)(-q)$')
    ax.set_ylabel(r'Approximate entropy, $\hat{H}(q)$')
    ax2 = ax.twinx()
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.plot(q,-q*np.log(q)-(1-q)*np.log(1-q), color='k', linewidth=2.5,
             label='True')
    ax2.legend(frameon=False)
    ax2.set_ylabel(r'True entropy, $H(q)$')
    # ax2.set_ylabel(r'True, $H(q)=-q \log(q) - (1-q) \log(1-q)$')
    ax.set_xlabel('q')
    fig.tight_layout()


def trans_rate_vs_coupling(j_list = np.arange(0.34, 1, 1e-3),
                           b=0.0, noise=0.1):
    xstb = []
    xustb = []
    xstb2 = []
    for j in j_list:
        x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j, b)
        xstb.append(x_stable_1)
        xustb.append(x_unstable)
        xstb2.append(x_stable_2)
    plt.figure()
    plt.plot(j_list, k_i_to_j(j_list, np.array(xstb), np.array(xustb), noise, b), label='error')
    plt.plot(j_list, k_i_to_j(j_list, np.array(xstb2), np.array(xustb), noise, b), label='correct')
    plt.yscale('log')
    plt.xlabel('Coupling, J')
    plt.ylabel('K')
    plt.legend(title='transition')


def trans_rate_vs_stim(b_list = np.arange(-0.2, 0.2, 1e-3),
                       j=0.5, noise=0.1):
    xstb = []
    xustb = []
    xstb2 = []
    for b in b_list:
        x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j, b)
        xstb.append(x_stable_1)
        xustb.append(x_unstable)
        xstb2.append(x_stable_2)
    plt.figure()
    plt.plot(b_list, k_i_to_j(j, np.array(xstb), np.array(xustb), noise, b_list),
             label='S2', color='r')
    plt.plot(b_list, k_i_to_j(j, np.array(xstb2), np.array(xustb), noise, b_list),
             label='S1', color='g')
    plt.yscale('log')
    plt.xlabel('Coupling, J')
    plt.ylabel('Transition rate')
    plt.legend(title='transition')


def trans_rate_vs_noise(noise_list = np.arange(0.05, 0.5, 1e-2),
                        b=0.0, j=0.7, t_dur=100):
    """
    IV Levelt law: alternation rate vs stimulus strength (noise)
    """
    x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j, b)
    k_2 = k_i_to_j(j, x_stable_1, x_unstable, noise_list, b)
    k_1 = k_i_to_j(j, x_stable_2, x_unstable, noise_list, b)
    k = (k_1+k_2)
    pinf = k_1/k
    p1 = (pinf)*(1-np.exp(-k*t_dur))
    p2 = (1-pinf)*(1-np.exp(-k*t_dur))
    k_weighted = k_1*p2 + k_2*p1
    fig, ax = plt.subplots(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.plot(noise_list, k_i_to_j(j, x_stable_1, x_unstable, noise_list, b), label='error')
    ax.plot(noise_list, k_weighted, label='correct',
             color='k', linewidth=2.5)
    ax.set_xlabel(r'Noise, $\sigma$')
    ax.set_ylabel('Transition rate')
    # plt.legend(title='transition')


def trans_rate_vs_noise_2d(noise_list=np.arange(0.03, 0.2, 1e-3),
                           b=0.0, j=0.7, t_dur=100):
    """
    Plots duration of state some noise values
    """
    x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j, b)
    noise1, noise2 = np.meshgrid(noise_list, noise_list)
    k_2 = k_i_to_j(j, x_stable_1, x_unstable, np.sqrt(noise1**2+noise2**2), b)
    # k_1 = k_i_to_j(j, x_stable_2, x_unstable, np.sqrt(noise1**2+noise2**2), b)
    # k = (k_1+k_2)
    # pinf = k_1/k
    # p1 = (pinf)*(1-np.exp(-k*t_dur))
    # p2 = (1-pinf)*(1-np.exp(-k*t_dur))
    # k_weighted = k_1*p2 + k_2*p1
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(noise1, noise2, np.log(1/(k_2)), color='gray',
                    edgecolor='k', lw=0.5, rstride=8, cstride=8,
                    alpha=0.3)
    ax.set_xlabel(r'Strength X')
    ax.set_ylabel(r'Strength Y')
    ax.set_zlabel('Duration X')
    ax.set_yticks([0.01, np.max(noise_list)+1e-2], ['low', 'high'])
    ax.set_xticks([0.01, np.max(noise_list)+1e-2], ['low', 'high'])
    ax.set_zticks([0., np.max(np.log(1/(k_1)))+1e-2], ['low', 'high'])
    # plt.legend(title='transition')


def duration_X_vs_stim_2d(b_list=np.arange(0.0, 0.125, 2.5e-3),
                          noise=0.1, j=0.7, t_dur=100):
    """
    Plots duration_X vs B, Brascamp 2015 3D style.
    """
    def return_k_all(b_list, j):
        karr = np.zeros((len(b_list), len(b_list)))
        for i_b1, b1 in enumerate(b_list):
            for i_b2, b2 in enumerate(-b_list):
                b = b1+b2
                x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j, b)
                k_2 = k_i_to_j(j, x_stable_1, x_unstable, noise, b)
                karr[i_b1, i_b2] = k_2
        return karr
    b1, b2 = np.meshgrid(b_list, -b_list)
    k_2 = return_k_all(b_list, j)
    # k_1 = k_i_to_j(j, x_stable_2, x_unstable, np.sqrt(noise1**2+noise2**2), b)
    # k = (k_1+k_2)
    # pinf = k_1/k
    # p1 = (pinf)*(1-np.exp(-k*t_dur))
    # p2 = (1-pinf)*(1-np.exp(-k*t_dur))
    # k_weighted = k_1*p2 + k_2*p1
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(b1, b2, (1/(k_2)), color='gray',
                    edgecolor='k', lw=0.5, rstride=8, cstride=8,
                    alpha=0.3)
    ax.set_xlabel(r'Strength Y')
    ax.set_ylabel(r'Strength X')
    ax.set_zlabel('Duration X')
    ax.set_xticks([0.0, np.max(b1)+1e-2], ['low', 'high'])
    ax.set_yticks([-np.max(b1)+1e-2, 0], ['high', 'low'])
    ax.set_zticks([0., np.max(1/k_2)+1e-2], ['low', 'high'])
    # plt.legend(title='transition')


def convergence_time_vs_j(j_list=np.arange(1e-3, 2.5, 1e-2),
                          b_list=[0, 0.05, 0.1], noise=0.1, theta=theta, tol=1e-9,
                          dt=5e-3, time_end=5, nreps=100):
    niters_conv_array = np.zeros((len(j_list), len(b_list)))
    disam_array = np.zeros((len(j_list), len(b_list)))
    kldiv_array = np.zeros((len(j_list), len(b_list)))
    for i_b, b in enumerate(b_list):
        n_iters_convergence = []
        disambiguity = []
        kldiv = []
        for j in j_list:
            p = gn.true_posterior(j=j, stim=b)
            vec = mean_field_stim(j, num_iter=1000, stim=b, sigma=0, theta=theta,
                                  val_init=0.6, sxo=0.)
            mse = np.sum(np.diff(vec, axis=0)**2, axis=1)
            first_min_mse = np.where(mse <= tol)[0][0]
            n_iters_convergence.append(first_min_mse)
            disambiguity_j = []
            kldiv_j = []
            for n in range(nreps):
                time, vec_sims = solution_mf_sdo_euler(j, b, theta, noise, tau=0.5,
                                                       time_end=time_end, dt=dt)
                disambiguity_single = np.abs(np.mean(vec_sims[-1])-0.5)
                disambiguity_j.append(disambiguity_single)
                q = np.mean(vec_sims[-1])
                kldiv_j.append(q*np.log(q/p)+(1-q)*np.log((1-q)/(1-p)))
            kldiv.append(np.nanmean(kldiv_j))
            disambiguity.append(np.nanmean(disambiguity_j))
        niters_conv_array[:, i_b] = n_iters_convergence
        disam_array[:, i_b] = disambiguity
        kldiv_array[:, i_b] = kldiv
    fig, ax = plt.subplots(ncols=3, figsize=(13, 4.5))
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    ax[0].set_xlabel('Coupling, J')
    ax[0].set_ylabel('Ambiguity')
    colormap = pl.cm.Oranges(np.linspace(0.2, 1, len(b_list)))
    for i in range(len(b_list)):
        ax[0].plot(j_list, 1-disam_array[:, i], color=colormap[i], linewidth=2.5,
                   label=round(b_list[i], 3))
        ax[1].plot(j_list, kldiv_array[:, i], color=colormap[i], linewidth=2.5)
        ax[2].plot(j_list, niters_conv_array[:, i], color=colormap[i], linewidth=2.5)
    ax[0].legend(title='Stim., B', frameon=False)
    ax[1].set_xlabel('Coupling, J')
    ax[1].set_ylabel('KL divergence')
    ax[2].axvline(1/3, color='r', alpha=0.2, linestyle='--')
    ax[2].set_yscale('log')
    ax[2].set_xlabel('Coupling, J')
    ax[2].set_ylabel('Iterations for convergence')
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'coupling_use.png', dpi=300, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'coupling_use.svg', dpi=300, bbox_inches='tight')


def accuracy_vs_noise(t_dur, noiselist=np.arange(0.001, 0.5, 1e-3),
                      j_list=[0.6, 0.8, 1],
                      tol=1e-8, b=0.1):
    init_cond = 0.5
    accuracy = np.zeros((len(j_list), len(noiselist)))
    accuracy_0 = np.zeros((len(j_list), len(noiselist)))
    prob_trans_corr = np.zeros((len(j_list), len(noiselist)))
    prob_trans_incorr = np.zeros((len(j_list), len(noiselist)))
    jlist2 = np.arange(0.5, 1, 5e-3)
    peakvals = np.zeros((len(jlist2)))
    for i_n, noise in enumerate(noiselist):
        for ij, j in enumerate(j_list):
            x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j, b)
            if b < 0:
                x_stable_2, x_stable_1 = x_stable_1, x_stable_2
            if np.abs(x_stable_1 - x_stable_2) <= tol:
                continue
            # P_{C, 0} = int_{x_E, x_0} exp(2V(x)/sigma^2) dx /
            #            int_{x_E, x_C} exp(2V(x)/sigma^2) dx
            pc0_numerator = scipy.integrate.quad(lambda q: np.exp(2*potential_mf(q, j, b)/noise**2),
                                                 x_stable_2, init_cond)[0]
            pc0_denom = scipy.integrate.quad(lambda q: np.exp(2*potential_mf(q, j, b)/noise**2),
                                             x_stable_2, x_stable_1)[0]
            pc0 = pc0_numerator/pc0_denom
            # compute error transition rates k_CE, K_EC
            k_EC = k_i_to_j(j, x_stable_1, x_unstable, noise, b)
            k_CE = k_i_to_j(j, x_stable_2, x_unstable, noise, b)
            k = k_EC+k_CE
            pCS = k_CE/k  # stationary correct
            # correct to error transition
            pEC = (1-pCS)*(1-np.exp(-k*t_dur))
            # error to correct transition
            pCE = pCS*(1-np.exp(-k*t_dur))
            # correct to correct
            pCC = pCS*(1-np.exp(-k*t_dur))+np.exp(-k*t_dur)
            # probability of correct
            pC = pc0*pCC + (1-pc0)*pCE
            accuracy[ij, i_n] = pC  # np.max((pC, 1-pC))
            accuracy_0[ij, i_n] = pc0
            prob_trans_corr[ij, i_n] = pCE  # np.max((pC, 1-pC))
            prob_trans_incorr[ij, i_n] = pEC
        deltap = []
        for ij2, j2 in enumerate(jlist2):
            x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j, b)
            if b < 0:
                x_stable_2, x_stable_1 = x_stable_1, x_stable_2
            if np.abs(x_stable_1 - x_stable_2) <= tol:
                continue
            # P_{C, 0} = int_{x_E, x_0} exp(2V(x)/sigma^2) dx /
            #            int_{x_E, x_C} exp(2V(x)/sigma^2) dx
            pc0_numerator = scipy.integrate.quad(lambda q: np.exp(2*potential_mf(q, j, b)/noise**2),
                                                 x_stable_2, init_cond)[0]
            pc0_denom = scipy.integrate.quad(lambda q: np.exp(2*potential_mf(q, j, b)/noise**2),
                                             x_stable_2, x_stable_1)[0]
            pc0 = pc0_numerator/pc0_denom
            # compute error transition rates k_CE, K_EC
            k_EC = k_i_to_j(j, x_stable_1, x_unstable, noise, b)
            k_CE = k_i_to_j(j, x_stable_2, x_unstable, noise, b)
            k = k_EC+k_CE
            pCS = k_CE/k  # stationary correct
            # correct to error transition
            pEC = (1-pCS)*(1-np.exp(-k*t_dur))
            # error to correct transition
            pCE = pCS*(1-np.exp(-k*t_dur))
            # correct to correct
            pCC = pCS*(1-np.exp(-k*t_dur))+np.exp(-k*t_dur)
            # probability of correct
            pC = pc0*pCC + (1-pc0)*pCE
            deltap.append(pCE-pEC)
        peakvals[ij2] = np.argmax(deltap)
    fig, ax = plt.subplots(ncols=3, figsize=(12, 5))
    colors = ['k', 'r', 'b']
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    legendelements = []
    for c in range(len(colors)):
        ax[0].plot(noiselist, accuracy[c], color=colors[c], linewidth=2.5)
        ax[0].plot(noiselist, accuracy_0[c], color=colors[c], linewidth=2.5, linestyle='--',
                alpha=0.4)
        ax[1].plot(noiselist, prob_trans_corr[c], color=colors[c], linewidth=2.5)
        ax[1].plot(noiselist, prob_trans_incorr[c], color=colors[c], linewidth=2.5, linestyle='--',
                   alpha=0.4)
        ax[2].plot(noiselist, prob_trans_corr[c]-prob_trans_incorr[c], color=colors[c], linewidth=2.5)
        legendelements.append(Line2D([0], [0], color=colors[c], lw=2.5, label='J = ' +str(j_list[c])))
    legendelements_2 = legendelements + [Line2D([0], [0], color='k', lw=2.5, label=r'$P$'),
                        Line2D([0], [0], color='k', linestyle='--', lw=2.5, 
                               alpha=0.4, label=r'$P_{0}$')]
    ax[0].legend(handles=legendelements_2, frameon=False, loc='upper right')
    legendelements_3 = [Line2D([0], [0], color='k', lw=2.5, label=r'Correct, C'),
                        Line2D([0], [0], color='k', linestyle='--', lw=2.5, 
                               alpha=0.4, label=r'Incorrect, I')]
    ax[1].legend(handles=legendelements_3, frameon=False, title='Transition')
    ax[0].set_xlabel(r'Noise $\sigma$')
    ax[0].set_ylabel(r'Accuracy')
    ax[1].set_xlabel(r'Noise $\sigma$')
    ax[1].set_ylabel(r'Probability of transition')
    ax[2].set_xlabel(r'Noise $\sigma$')
    ax[2].set_ylabel(r'$P_{t,C}-P_{t, I}$')
    
    
def psychometric_mf_analytical(t_dur, noiselist=[0.05, 0.1, 0.2, 0.3],
                               j_list=np.arange(0., 1.1, 0.2),
                               b_list=np.arange(0, 0.2, 1e-2),
                               tol=1e-8):
    init_cond = 0.5
    accuracy = np.zeros((len(j_list), len(b_list), len(noiselist)))
    for i_n, noise in enumerate(noiselist):
        for ib, b in enumerate(b_list):  # for each b, compute P(C) = P_{C,0}*P_{C,C} + (1-P_{C,0})*P_{C,E}
        # prob of correct is prob of correct at beginning times prob of staying at correct atractor
        # plus prob of incorrect at beginning times prob of going from incorrect to correct
            for ij, j in enumerate(j_list):
                x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j, b)
                if b < 0:
                    x_stable_2, x_stable_1 = x_stable_1, x_stable_2
                if np.abs(x_stable_1 - x_stable_2) <= tol:
                    continue
                # P_{C, 0} = int_{x_E, x_0} exp(2V(x)/sigma^2) dx /
                #            int_{x_E, x_C} exp(2V(x)/sigma^2) dx
                pc0_numerator = scipy.integrate.quad(lambda q: np.exp(2*potential_mf(q, j, b)/noise**2),
                                                     x_stable_2, init_cond)[0]
                pc0_denom = scipy.integrate.quad(lambda q: np.exp(2*potential_mf(q, j, b)/noise**2),
                                                 x_stable_2, x_stable_1)[0]
                pc0 = pc0_numerator/pc0_denom
                # compute error transition rates k_CE, K_EC
                k_EC = k_i_to_j(j, x_stable_1, x_unstable, noise, b)
                k_CE = k_i_to_j(j, x_stable_2, x_unstable, noise, b)
                k = k_EC+k_CE
                pCS = k_CE/k  # stationary correct
                # error to correct transition
                pCE = pCS*(1-np.exp(-k*t_dur))
                # correct to correct
                pCC = pCS*(1-np.exp(-k*t_dur))+np.exp(-k*t_dur)
                # probability of correct
                pC = pc0*pCC + (1-pc0)*pCE
                accuracy[ij, ib, i_n] = pC  # np.max((pC, 1-pC))
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
    axes = axes.flatten()
    for iax, ax in enumerate(axes):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        colormap = pl.cm.Oranges(np.linspace(0.2, 1, len(j_list)))
        for ij in range(len(j_list)):
            ax.plot(b_list, accuracy[ij, :, iax], color=colormap[ij],
                    label=round(j_list[ij], 2))
        ax.set_xlabel('Sensory evidence, B')
        ax.set_ylabel('Accuracy')
        ax.set_title(r'Noise, $\sigma$ = ' + str(noiselist[iax]))
        ax.set_ylim(0.45, 1.05)
    axes[0].legend(title='Coupling, J')
    fig.tight_layout()


def transition_probs_j(t_dur, noise,
                       j_list=np.arange(0.001, 3.01, 0.005),
                       b=0, tol=1e-10):
    trans_prob_s2_to_s1 = []
    trans_prob_s1_to_s2 = []
    trans_prob_s2_to_s2 = []
    trans_prob_s1_to_s1 = []
    # trans_prob_i_to_j = []
    # trans_prob_j_to_i = []
    probs_well_s1 = []
    probs_well_s2 = []
    for j in j_list:
        diff_1 = 0
        diff_2 = 0
        init_cond_unst = 0.5
        q1 = lambda q: gn.sigmoid(6*j*(2*q-1)+ b*6) - q
        sol_1, _, flag, _ =\
            fsolve(q1, 1, full_output=True)
        if flag == 1:
            x_stable_1 = sol_1[0]
        else:
            x_stable_1 = np.nan
        sol_2, _, flag, _ =\
            fsolve(q1, 0, full_output=True)
        if flag == 1:
            x_stable_2 = sol_2[0]
        else:
            x_stable_2 = np.nan
        while np.abs(diff_1) <= tol or np.abs(diff_2) <= tol:
            if np.abs(x_stable_1-x_stable_2) <= tol:
                x_unstable = np.nan
                break
            sol_unstable, _, flag, _ =\
                fsolve(q1, init_cond_unst, full_output=True)
            if flag == 1:
                x_unstable = sol_unstable[0]
            else:
                x_unstable = np.nan
                break
            diff_1 = x_unstable - x_stable_1
            diff_2 = x_unstable - x_stable_2
            init_cond_unst = np.random.rand()
        k_x_s1_to_s2 = k_i_to_j(j, x_stable_1, x_unstable, noise, b)
        k_x_s2_to_s1 = k_i_to_j(j, x_stable_2, x_unstable, noise, b)
        k = k_x_s1_to_s2 + k_x_s2_to_s1
        p_is = k_x_s1_to_s2 / k
        p_js = k_x_s2_to_s1 / k
        # P_is is prob to stay in i at end of trial given by t_dur
        # P_js is prob to stay in j at end of trial given by t_sdur
        trans_prob_s1_to_s2.append(p_is*(1-np.exp(-k*t_dur))) # prob of at some point going from s1 to s2
        trans_prob_s2_to_s1.append(p_js*(1-np.exp(-k*t_dur))) # prob of at some point going from s2 to s1
        trans_prob_s2_to_s2.append(p_is)
        trans_prob_s1_to_s1.append(p_js)
        v_unst = potential_mf(x_unstable, j, b)
        v_s1 = potential_mf(x_stable_1, j, b)
        v_s2 = potential_mf(x_stable_2, j, b)
        p_s1 = np.exp((v_unst-v_s1)/noise**2)
        p_s2 = np.exp((v_unst-v_s2)/noise**2)
        probs_well_s1.append(p_s1/(p_s1+p_s2))
        probs_well_s2.append(p_s2/(p_s1+p_s2))
        # trans_prob_i_to_j.append(1-np.exp(-(k_x_s_to_u_2+k_x_u_to_s_1)*t_dur))
        # trans_prob_j_to_i.append(1-np.exp(-(k_x_s_to_u_1+k_x_u_to_s_2)*t_dur))
        # 1 - np.exp(-k*t_dur) is prob to change from i->j and vice-versa
    colors = ['k', 'k', 'r', 'r']
    lst = ['-', '--', '-', '--']
    plt.figure()
    plt.plot(j_list, trans_prob_s1_to_s2, label='P_s1_to_s2', color=colors[0],
             linestyle=lst[0])
    plt.plot(j_list, trans_prob_s2_to_s1, label='P_s2_to_s1', color=colors[2],
             linestyle=lst[2])
    # plt.plot(j_list, trans_prob_s1_to_s1, label='P_s2_to_s2', color=colors[1],
    #          linestyle=lst[1])
    # plt.plot(j_list, trans_prob_s2_to_s2, label='P_s1_to_s1', color=colors[3],
    #          linestyle=lst[3])
    plt.xlim(-0.05, np.max(j_list)+0.05)
    plt.xlabel('J')
    plt.ylabel('Transition probability')
    plt.title('B =' + str(b))
    plt.legend()
    plt.figure()
    plt.plot(j_list, probs_well_s1, label='p(x=1)', color='k')
    plt.plot(j_list, probs_well_s2, label='p(x=-1)', color='r')
    plt.legend()
    plt.xlabel('J')
    plt.ylabel(r'$p \propto e^{\Delta V / \sigma^2}$')
    plt.xlim(np.min(j_list)-0.02, np.max(j_list)+0.02)


def splitting_prob_a_to_c_vs_sigma(j, b=0, noiselist=np.logspace(-2, -0.3, 21), ax=None,
                                   eps=1e-2, color='k'):
    if ax is None:
        fig, ax = plt.subplots(1)
    q1 = lambda q: gn.sigmoid(6*j*(2*q-1)+ b*2) - q
    flag = 0
    x0 = 0
    n_iters = 0
    while flag != 1 or n_iters > 100:
        sol_1, _, flag, _ =\
            fsolve(q1, x0, full_output=True)
        if flag == 1:
            x_stable_1 = sol_1[0]
        else:
            x0 = np.random.uniform(0, 0.4)
            n_iters += 1
        sol_2, _, flag, _ =\
            fsolve(q1, 1-x0, full_output=True)
        if flag == 1:
            x_stable_2 = sol_2[0]
        else:
            x0 = np.random.uniform(0, 0.4)
        n_iters += 1
    x_stable_1 = sol_1[0]
    x_stable_2 = sol_2[0]
    splitting_prob_x_through_a = []
    for noise in noiselist:
        n_cte = scipy.integrate.quad(lambda x: np.exp(2*potential_mf(x, j=j, bias=b)/noise**2),
                                     x_stable_1, x_stable_2, epsabs=1e-10, epsrel=1e-10)[0]
        splitting_prob_x_through_a.append(scipy.integrate.quad(lambda x: np.exp(2*potential_mf(x, j=j, bias=b)/noise**2),
                                                  x_stable_1, x_stable_1+eps, epsabs=1e-10, epsrel=1e-10)[0]/ n_cte)
    ax.plot(noiselist, splitting_prob_x_through_a, color=color)


def prob_jump_vs_j(jlist=np.round(np.arange(0.5, 1.1, 5e-2), 3)):
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 8))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.3)
    colormap = pl.cm.Oranges(np.linspace(0.2, 1, len(jlist)))
    ax = ax.flatten()
    for i_j, j in enumerate(jlist):
        splitting_prob_a_to_c_vs_sigma(j, b=0, noiselist=np.logspace(-2, -0.1, 51), ax=ax[0],
                                       eps=1e-2, color=colormap[i_j])
        splitting_prob_a_to_c_vs_sigma(j, b=0.1, noiselist=np.logspace(-2, -0.1, 51), ax=ax[1],
                                       eps=1e-2, color=colormap[i_j])
        splitting_prob_a_to_c_vs_sigma(j, b=0, noiselist=np.logspace(-2, -0.1, 51), ax=ax[2],
                                       eps=2e-1, color=colormap[i_j])
        splitting_prob_a_to_c_vs_sigma(j, b=0.1, noiselist=np.logspace(-2, -0.1, 51), ax=ax[3],
                                       eps=2e-1, color=colormap[i_j])
    epslist = [0.01, 0.01, 0.2, 0.2]
    blist = [0, 0.1, 0, 0.1]
    for a, eps, b in zip(ax, epslist, blist):
        a.set_xscale('log')
        # a.set_yscale('log')
        a.set_title(r'$\epsilon = $' + str(eps) + ', B = ' + str(b))
        ax_pos = a.get_position()
        a.set_position([ax_pos.x0, ax_pos.y0,
                        ax_pos.width*0.9, ax_pos.height])
    ax[2].set_xlabel(r'Noise variance, $\sigma$')
    ax[3].set_xlabel(r'Noise variance, $\sigma$')
    ax[0].set_ylabel(r'Prob($a+\epsilon \to c$)')
    ax[2].set_ylabel(r'Prob($a+\epsilon \to c$)')
    
    ax_pos = ax[1].get_position()
    ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*1.05, ax_pos.y0-ax_pos.height*0.5,
                            ax_pos.width*0.06, ax_pos.height*0.7])
    newcmp = mpl.colors.ListedColormap(colormap)
    mpl.colorbar.ColorbarBase(ax_cbar, cmap=newcmp, label='Coupling J')
    ax_cbar.set_yticks([0, 0.5, 1], [jlist[0], jlist[len(jlist) // 2], jlist[len(jlist)-1]])


def plot_noise_changes(j=0.7, b=0, noise_list=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.075, right=0.98,
                        hspace=0.3, wspace=0.5)
    ax = ax.flatten()
    colormap = pl.cm.Oranges(np.linspace(0.2, 1, len(noise_list)))
    legend = True
    der_splitprob = []
    for i_n, noise in enumerate(noise_list):
        if i_n > 0:
            legend = False
        splitprob = plot_exit_time(j=0.7, b=0, noise=noise, ax=ax, color=colormap[i_n], fig=fig,
                                   legend=legend)
        der_splitprob.append(splitprob)
    ax[2].set_yscale('log')
    fig2, ax2 = plt.subplots(1)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.plot(noise_list, der_splitprob, color='k')


def plot_exit_time(j=0.7, b=0, noise=0.1, ax=None, color=None, fig=None, legend=True):
    q = np.arange(0, 1, 1e-2)
    v2_a = second_derivative_potential(q, j=j, b=b)
    q1 = lambda q: gn.sigmoid(6*j*(2*q-1)+ b*2) - q
    flag = 0
    x0 = 0
    n_iters = 0
    while flag != 1 or n_iters > 100:
        sol_1, _, flag, _ =\
            fsolve(q1, x0, full_output=True)
        if flag == 1:
            x_stable_1 = sol_1[0]
        else:
            x0 = np.random.uniform(0, 0.4)
            n_iters += 1
        sol_2, _, flag, _ =\
            fsolve(q1, 1-x0, full_output=True)
        if flag == 1:
            x_stable_2 = sol_2[0]
        else:
            x0 = np.random.uniform(0, 0.4)
        n_iters += 1
    x_stable_1 = sol_1[0]
    x_stable_2 = sol_2[0]
    q_val_bckw = 0.5
    for i in range(50):
        q_val_bckw = backwards(q_val_bckw, j, b)
    v2_a = second_derivative_potential(q, j=j, b=b)
    v2_b = second_derivative_potential(q_val_bckw, j=j, b=b)
    if v2_b < 0:
        print('it is unstable')
    time_from_a_to_b = []
    time_from_a_to_b_approx = []
    splitting_prob_x_through_a = []
    for q_val in q:
        # n_cte_ps = scipy.integrate.quad(lambda x: np.exp(-2*potential_mf(x, j=j, bias=b)/noise**2),
        #                                 0, 1)[0]
        time_from_a_to_b_approx.append(np.abs(1/noise * scipy.integrate.quad(lambda x: np.exp(2*potential_mf(x, j=j, bias=b)/noise**2),
                                                                             q_val, x_stable_1)[0] *
                                              scipy.integrate.quad(lambda x: np.exp(-2*potential_mf(x, j=j, bias=b)/noise**2),
                                                                   0, q_val_bckw)[0]))
        time_from_a_to_b.append(np.abs(1/noise *  scipy.integrate.quad(lambda x: np.exp(2*potential_mf(x, j=j, bias=b)/noise**2)*
                                                                       scipy.integrate.quad(
                                                                           lambda y: np.exp(-2*potential_mf(y, j=j, bias=b)/noise**2),
                                                                           0, x)[0],
                                                            q_val, x_stable_1)[0]))
        n_cte = scipy.integrate.quad(lambda x: np.exp(2*potential_mf(x, j=j, bias=b)/noise**2),
                                     x_stable_1, x_stable_2, epsabs=1e-10, epsrel=1e-10)[0]
        splitting_prob_x_through_a.append(np.abs(scipy.integrate.quad(lambda x: np.exp(2*potential_mf(x, j=j, bias=b)/noise**2),
                                                  q_val, x_stable_2, epsabs=1e-10, epsrel=1e-10)[0] / n_cte))
    # time_from_a_to_b = np.pi / np.sqrt(np.abs(v2_b)*v2_a)*np.exp((v2_b-v2_a)/noise)
    if ax is None:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.075, right=0.98,
                            hspace=0.3, wspace=0.5)
        ax = ax.flatten()
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    if color is None:
        color = 'k'
    q_pot = np.arange(0, 1, 1e-3)
    pot = potential_mf(q_pot, j, b)
    ax[0].plot(q_pot, pot-np.min(pot), color=color, linewidth=2.5)
    ax[0].set_ylabel('Potential V(q)')
    distro = np.exp(-2*pot/noise**2)
    ax[1].plot(q_pot, distro / np.sum(distro), color=color, linewidth=2.5)
    val_unst = potential_mf(q_val_bckw, j, b)
    ax[1].plot(q_val_bckw, np.exp(-2*val_unst/noise**2)/np.sum(distro),
               marker='o', color='b', linestyle='', label=r'$q^*_{unstable}$',
               markersize=8)
    ax[1].axvline(x_stable_1, color='r', alpha=0.5)
    ax[1].axvline(x_stable_2, color='g', alpha=0.5)
    ax[1].set_ylabel(r'Stationary distribution $p_s(q)$')
    ax[2].plot(q, time_from_a_to_b, color=color, linewidth=2.5, label='Exact')
    ax[2].plot(q, time_from_a_to_b_approx, color='magenta', linewidth=2, linestyle='--',
               label='Approximation')
    if legend:
        ax[2].legend(frameon=False)
    ax[2].set_ylabel(r'Time from $a$ to $q$, $T(a \rightarrow q)$')
    ax[2].set_xlabel('Approximate posterior q')
    pot_unst = potential_mf(q_val_bckw, j, b)-np.min(pot)
    ax[0].axvline(x_stable_1, color='r', alpha=0.5, label=r'$a = q^*_{stable, L}$')
    ax[0].plot(q_val_bckw, pot_unst, marker='o', color='b', linestyle='',
               label=r'$b = q^*_{unstable}$', markersize=8)
    ax[0].axvline(x_stable_2, color='g', alpha=0.5, label=r'$c = q^*_{stable, R}$')
    if legend:
        ax[0].legend(frameon=False)
    idx = np.where((q-round(q_val_bckw, 2)) == 0)
    ax[2].plot(q_val_bckw, time_from_a_to_b[idx[0][0]], marker='o',
               linestyle='', color='b', markersize=8)
    ax[2].axvline(x_stable_1, color='r', alpha=0.5)
    ax[2].axvline(x_stable_2, color='g', alpha=0.5)
    distro_split = np.array(splitting_prob_x_through_a)
    ax[3].plot(q, distro_split, color=color, linewidth=2.5)
    ax[3].plot(q_val_bckw, distro_split[idx[0][0]], marker='o',
               linestyle='', color='b', markersize=8)
    ax[3].axvline(x_stable_1, color='r', alpha=0.5)
    ax[3].axvline(x_stable_2, color='g', alpha=0.5)
    ax[3].set_xlabel('Approximate posterior q')
    ax[3].set_ylabel(r'Splitting probability $\pi_{a}(x)$')
    ax[3].set_ylim(-0.05, 1.05)
    fig.tight_layout()
    return np.diff(splitting_prob_x_through_a)[idx[0][0]]/np.diff(q)[0]


def plot_2d_exit_time(var_change='noise', noise=0.1,
                      j=0.4, b=0.01):
    """
    var_change can be:
        - noise
        - coupling
        - sensory_ev
    """
    if var_change == 'noise':
        var_list = np.logspace(-3, 0, 61)
        varlabel = r'Noise, $\sigma$'
    if var_change == 'coupling':
        var_list = np.arange(1e-4, 1, 5e-3)
        varlabel = r'Coupling, $J$'
    if var_change == 'sensory_ev':
        var_list = np.arange(-0.08, 0, 1e-3)
        varlabel = r'Sensory evidence, $B$'
    q = np.arange(0, 1, 5e-3)
    potential_array = np.zeros((len(var_list), len(q)))
    time_escape_array = np.zeros((len(var_list), len(q)))
    for i_var, var in enumerate(var_list):
        if var_change == 'noise':
            noise = var
        if var_change == 'coupling':
            j = var
        if var_change == 'sensory_ev':
            b = var
        q1 = lambda q: gn.sigmoid(6*j*(2*q-1)+ b*6) - q
        flag = 0
        x0 = 0
        n_iters = 0
        while flag != 1 or n_iters > 10:
            sol_1, _, flag, _ =\
                fsolve(q1, x0, full_output=True)
            if flag == 1:
                x_stable_1 = sol_1[0]
            else:
                x0 = np.random.uniform(0, 0.4)
                n_iters += 1
        x_stable_1 = sol_1[0]
        q_val_bckw = 0.5
        for i in range(50):
            q_val_bckw = backwards(q_val_bckw, j, b)
        time_from_a_to_b = []
        for q_val in q:
            time_from_a_to_b.append(np.abs(1/noise * scipy.integrate.quad(lambda x: np.exp(potential_mf(x, j=j, bias=b)/noise),
                                                      q_val, x_stable_1)[0] *
                                            scipy.integrate.quad(lambda x: np.exp(-potential_mf(x, j=j, bias=b)/noise),
                                                                0, q_val_bckw)[0]))
        pot = potential_mf(q, j, b)
        potential_array[i_var, :] = (pot-np.min(pot))/(np.max(pot)-np.min(pot))
        time_escape_array[i_var, :] = np.array(time_from_a_to_b) / np.sum(time_from_a_to_b)
    fig, ax = plt.subplots(nrows=2, figsize=(7, 10))
    ax[0].set_ylabel(varlabel)
    ax[1].set_ylabel(varlabel)
    ax[0].set_xticks([])
    ax[1].set_xlabel('Approximate posterior q')
    cmap_pot = 'Oranges_r'
    im_pot = ax[0].imshow(np.flipud(potential_array), cmap=cmap_pot, aspect='auto',
                          extent=[0, 1, np.min(var_list), np.max(var_list)])
    if var_change == 'noise':
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
    plt.colorbar(im_pot, ax=ax[0], label='Potential V(q)')
    im_time = ax[1].imshow(np.flipud(time_escape_array), cmap='Blues', aspect='auto',
                          extent=[0, 1, np.min(var_list), np.max(var_list)])
                          # norm=mpl.colors.LogNorm())
    plt.colorbar(im_time, ax=ax[1], label='Escape time T(q --> a)')



def dummy_potential(noise=0.1):
    fig, ax = plt.subplots(nrows=2, figsize=(6, 11))
    q = np.arange(-10, 10, 1e-2)
    pot = .01*q**4 - q**2
    x_stable_1 = q[np.where(pot == np.min(pot))[0][0]]
    pot2 = pot[100:-100]
    q_val_bckw = q[100+np.where(pot2 == np.max(pot2))[0][0]]
    time_from_a_to_b = []
    for q_val in q:
        time_from_a_to_b.append(np.abs(1/noise * scipy.integrate.quad(lambda x: np.exp(.01*x**4 - x**2/noise),
                                                  q_val, x_stable_1)[0] *
                                        scipy.integrate.quad(lambda x: np.exp(-.01*x**4 + x**2/noise),
                                                            -.1, q_val_bckw)[0]))
    ax[0].plot(q, pot)
    ax[1].plot(q, time_from_a_to_b)


def transition_probs_j_and_b(t_dur, noise,
                             j_list=np.arange(0.001, 3.01, 0.005),
                             b_list=[0], tol=1e-10):
    mat_trans_prob_s1_to_s2 = np.zeros((len(j_list), len(b_list)))
    mat_trans_prob_s2_to_s1 = np.zeros((len(j_list), len(b_list)))
    mat_probs_well_s1 = np.zeros((len(j_list), len(b_list)))
    mat_probs_well_s2 = np.zeros((len(j_list), len(b_list)))
    for i_j, j in enumerate(j_list):
        trans_prob_s2_to_s1 = []
        trans_prob_s1_to_s2 = []
        trans_prob_s2_to_s2 = []
        trans_prob_s1_to_s1 = []
        # trans_prob_i_to_j = []
        # trans_prob_j_to_i = []
        probs_well_s1 = []
        probs_well_s2 = []
        for i_b, b in enumerate(b_list):
            diff_1 = 0
            diff_2 = 0
            init_cond_unst = 0.5
            q1 = lambda q: gn.sigmoid(6*j*(2*q-1)+ b*6) - q
            sol_1, _, flag, _ =\
                fsolve(q1, 1, full_output=True)
            if flag == 1:
                x_stable_1 = sol_1[0]
            else:
                x_stable_1 = np.nan
            sol_2, _, flag, _ =\
                fsolve(q1, 0, full_output=True)
            if flag == 1:
                x_stable_2 = sol_2[0]
            else:
                x_stable_2 = np.nan
            while np.abs(diff_1) <= tol or np.abs(diff_2) <= tol:
                if np.abs(x_stable_1-x_stable_2) <= tol:
                    x_unstable = np.nan
                    break
                sol_unstable, _, flag, _ =\
                    fsolve(q1, init_cond_unst, full_output=True)
                if flag == 1:
                    x_unstable = sol_unstable[0]
                else:
                    x_unstable = np.nan
                    break
                diff_1 = x_unstable - x_stable_1
                diff_2 = x_unstable - x_stable_2
                init_cond_unst = np.random.rand()
            k_x_s1_to_s2 = k_i_to_j(j, x_stable_1, x_unstable, noise, b)
            k_x_s2_to_s1 = k_i_to_j(j, x_stable_2, x_unstable, noise, b)
            k = k_x_s1_to_s2 + k_x_s2_to_s1
            p_is = k_i_to_j(j, x_stable_1, x_unstable, noise, b) / k
            p_js = k_i_to_j(j, x_stable_2, x_unstable, noise, b) / k
            # P_is is prob to stay in i at end of trial given by t_dur
            # P_js is prob to stay in j at end of trial given by t_sdur
            trans_prob_s1_to_s2.append(np.exp(-k_x_s1_to_s2*t_dur)) # prob of at some point going from s1 to s2
            trans_prob_s2_to_s1.append(np.exp(-k_x_s2_to_s1*t_dur)) # prob of at some point going from s2 to s1
            trans_prob_s2_to_s2.append(p_is)
            trans_prob_s1_to_s1.append(p_js)
            v_unst = potential_mf(x_unstable, j, b)
            v_s1 = potential_mf(x_stable_1, j, b)
            v_s2 = potential_mf(x_stable_2, j, b)
            p_s1 = np.exp((v_unst-v_s1)/noise**2)
            p_s2 = np.exp((v_unst-v_s2)/noise**2)
            probs_well_s1.append(p_s1/(p_s1+p_s2))
            probs_well_s2.append(p_s2/(p_s1+p_s2))
            # trans_prob_i_to_j.append(1-np.exp(-(k_x_s_to_u_2+k_x_u_to_s_1)*t_dur))
            # trans_prob_j_to_i.append(1-np.exp(-(k_x_s_to_u_1+k_x_u_to_s_2)*t_dur))
            # 1 - np.exp(-k*t_dur) is prob to change from i->j and vice-versa
        mat_trans_prob_s1_to_s2[i_j, :] = trans_prob_s1_to_s2
        mat_trans_prob_s2_to_s1[i_j, :] = trans_prob_s1_to_s2
        mat_probs_well_s1[i_j, :] = probs_well_s1
        mat_probs_well_s2[i_j, :] = probs_well_s2
    fig, ax = plt.subplots(1)
    im = ax.imshow(np.flipud(mat_trans_prob_s1_to_s2), cmap='Oranges',
                   extent=[min(b_list), max(b_list), min(j_list), max(j_list)],
                   aspect='auto')
    plt.colorbar(im, ax=ax, label='Transition probability from S1 to S2')
    ax.set_ylabel('J')
    ax.set_xlabel('B')
    fig, ax = plt.subplots(1)
    im = ax.imshow(np.flipud(mat_probs_well_s1), cmap='Oranges',
                   extent=[min(b_list), max(b_list), min(j_list), max(j_list)],
                   aspect='auto')
    plt.colorbar(im, ax=ax, label=r'$p \propto e^{\Delta V / \sigma^2}$')
    ax.set_ylabel('J')
    ax.set_xlabel('B')


def plot_3_examples_mf_evolution(avg=False):
    j_list = [0.25, 0.25, 0.4]
    b_list = [0, 0.1, 0]
    fig, ax = plt.subplots(ncols=3, figsize=(6, 2.5))
    i = 0
    times = [100, 100, 100]
    time_min = [0, 0, 0]
    dt_list = [1e-3, 1e-3, 1e-3]
    noise_list = [0.15, 0.15, 0.15]
    convlist = [True, True, False]
    for j, b, t_end, dt, noise, t_min, conv in zip(j_list, b_list, times, dt_list,
                                              noise_list, time_min, convlist):
        ax[i].axhline(0.5, color='k', alpha=0.4, linestyle='--')
        line = plot_mf_evolution_all_nodes(j=j, b=b, noise=noise, tau=0.008, time_end=t_end,
                                           dt=dt, ax=ax[i], ylabel=i==0,
                                           time_min=t_min, avg=avg, conv=conv)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        i += 1
    ax[1].set_xlabel('Time (s)')
    fig.tight_layout()
    ax_pos = ax[2].get_position()
    ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*1.13, ax_pos.y0+ax_pos.height*0.2,
                            ax_pos.width*0.1, ax_pos.height*0.7])
    fig.colorbar(line, cax=ax_cbar, pad=0.3, aspect=7.5).set_label(label=r'$q(x=1)$', size=14) # add a color legend
    fig.savefig(DATA_FOLDER + 'example_dynamics.png', dpi=300, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'example_dynamics.svg', dpi=300, bbox_inches='tight')
    # plt.subplots_adjust(wspace=0.2, bottom=0.16, top=0.88)


def plot_3d_solution_mf_vs_j_b(j_list, b_list, N=3,
                               num_iter=50, tol=1e-6,
                               dim3d=False):
    if dim3d:
        ax = plt.figure().add_subplot(projection='3d')
    else:
        fig, ax = plt.subplots(1, figsize=(5, 4))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    solutions = np.empty((len(j_list), len(b_list), 3))
    for i_j, j in enumerate(j_list):
        for i_b, b in enumerate(b_list):
            q_val_01 = 0.
            q_val_07 = 1
            q_val_bckw = 0.7
            for i in range(num_iter):
                q_val_01 = gn.sigmoid(6*(j*(2*q_val_01-1))+2*b)
                q_val_07 = gn.sigmoid(6*(j*(2*q_val_07-1))+b*2)
            # q1 = lambda q: gn.sigmoid(6*j*(2*q-1)+ b*6) - q
            # q_val_01, _, flag, _ =\
            #     fsolve(q1, q_val_01, full_output=True, xtol=1e-10)
            # q_val_07, _, flag, _ =\
            #     fsolve(q1, q_val_07, full_output=True, xtol=1e-10)
            for i in range(num_iter*20):
                q_val_bckw = backwards(q_val_bckw, j, b)
                if q_val_bckw < 0 or q_val_bckw > 1:
                        q_val_bckw = np.nan
                        break
            if np.abs(q_val_01 - q_val_bckw) <= tol:
                q_val_bckw = np.nan
            if np.abs(q_val_01 - q_val_07) <= tol:
                q_val_01 = np.nan
            solutions[i_j, i_b, 0] = q_val_01
            solutions[i_j, i_b, 1] = q_val_07
            solutions[i_j, i_b, 2] = q_val_bckw
    if dim3d:
        x, y = np.meshgrid(j_list, b_list)
        ax.plot_surface(x, y, solutions[:, :, 0].T, alpha=0.4, color='b')
        ax.plot_surface(x, y, solutions[:, :, 1].T, alpha=0.4, color='b')
        ax.plot_surface(x, y, solutions[:, :, 2].T, alpha=0.4, color='r')
        ax.set_xlabel('Coupling, J')
        ax.set_ylabel('Sensory evidence, B')
        ax.set_zlabel('Approximate posterior, q(x=1)')
    else:
        colormap = pl.cm.copper(np.linspace(0.2, 1, len(b_list)))
        for i_b, b in enumerate(b_list):
            ax.plot(j_list, solutions[:, i_b, 0], color=colormap[i_b])
            ax.plot(j_list, solutions[:, i_b, 1], color=colormap[i_b])
            ax.plot(j_list, solutions[:, i_b, 2], color=colormap[i_b], linestyle='--')
        labels = ['0', ' ', '0.05', ' ', '0.1']
        legendelements = [Line2D([0], [0], color=colormap[k], lw=2.5, label=labels[k]) for k in range(len(labels))]
        legend1 = plt.legend(loc=0,
                             title='Stimulus, B', handles=legendelements,
                             frameon=False, labelspacing=0.1,
                             bbox_to_anchor=(0.34, .8))
        legendelements = [Line2D([0], [0], color='k', lw=2.5, label='Stable'),
                          Line2D([0], [0], color='k', linestyle='--', lw=2.5, label='Unstable')]
        legend2 = plt.legend(handles=legendelements, title='Fixed point',
                             frameon=False)
        ax.add_artist(legend1)
        ax.add_artist(legend2)
        ax.set_xlabel('Coupling, J')
        ax.set_ylabel('Approximate posterior, q(x=1)')
        fig.tight_layout()
        fig.savefig(DATA_FOLDER + 'solutions_diff_B_J.png', dpi=400, bbox_inches='tight')
        fig.savefig(DATA_FOLDER + 'solutions_diff_B_J.svg', dpi=400, bbox_inches='tight')


def plot_slope_wells_vs_B(j_list=np.arange(0.6, 1.01, 0.1),
                          b_list=np.arange(-.3, .3, 0.01)):
    fig, ax = plt.subplots(ncols=1)
    colormap = pl.cm.Oranges(np.linspace(0.2, 1, len(j_list)))
    for i_j, j in enumerate(j_list):
        slope = []
        for b in b_list:
            q1 = lambda q: gn.sigmoid(6*j*(2*q-1)+ b*6) - q
            sol_1, _, flag, _ =\
                fsolve(q1, 1, full_output=True, xtol=1e-10)
            if flag == 1:
                x_stable_1 = sol_1[0]
            else:
                x_stable_1 = np.nan
            sol_2, _, flag, _ =\
                fsolve(q1, 0, full_output=True, xtol=1e-10)
            if flag == 1:
                x_stable_2 = sol_2[0]
            else:
                x_stable_2 = np.nan
            val_min_0 = potential_mf(q=x_stable_1, j=j, bias=b)
            val_min_1 = potential_mf(q=x_stable_2, j=j, bias=b)
            slope.append((val_min_0-val_min_1) / (x_stable_1-x_stable_2))
        ax.plot(b_list, slope, color=colormap[i_j], label=np.round(j, 1))
    ax.legend(title='J')
    ax.set_xlabel('Sensory evidence, B')
    ax.set_ylabel('Slope of wells')


def solutions_high_j_lambert(j_list=np.arange(0.1, 2.01, 0.001),
                             b=0):
    q1 = []
    q2 = []
    q3 = []
    q = [q1, q2, q3]
    sol_real_0 = []
    sol_real_05 = []
    for j in j_list:
        q0 = lambda q: gn.sigmoid(6*j*(2*q-1)+ b*6) - q
        sol_real_0.append(fsolve(q0, 0))
        sol_real_05.append(fsolve(q0, 0.5))
        for k in range(len(q)):
            z = -12*j*np.exp(b*6-6*j)
            q[k].append(scipy.special.lambertw(z, k=k, tol=1e-8)/(-12*j))
    plt.figure()
    plt.plot(j_list, sol_real_0, label='solution', color='k')
    plt.plot(j_list, sol_real_05, color='k')
    for k in range(len(q)):
        plt.plot(j_list, q[k], label='k = {}'.format(k))
    plt.ylim(0, 1)
    plt.xlabel('J')
    plt.ylabel('q(x=1)')
    plt.legend()


def slopes_high_j_lambert(j_list=np.arange(0.7, 2.01, 0.1),
                          b_list=np.arange(-.3, .3, 0.01)):
    k = 0
    colormap = pl.cm.Oranges(np.linspace(0.2, 1, len(j_list)))
    fig, ax = plt.subplots(1)
    for i_j, j in enumerate(j_list):
        v_0 = []
        v_1 = []
        pot_0_list = []
        pot_1_list = []
        for b in b_list:
            z = -12*j*np.exp(b-6*j)
            val = scipy.special.lambertw(z, k=k, tol=1e-8)/(-12*j)
            v_0.append(np.real(val))
            v_1.append(1-np.real(val))
            pot_0 = potential_mf(q=np.real(val), j=j, bias=b)
            pot_1 = potential_mf(q=1-np.real(val), j=j, bias=b)
            pot_0_list.append(pot_0)
            pot_1_list.append(pot_1)
        v_1 = np.array(v_1)
        v_0 = np.array(v_0)
        pot_1_list = np.array(pot_1_list)
        pot_0_list = np.array(pot_0_list)
        dV_dq = (pot_0_list - pot_1_list)/(v_0-v_1)
        plt.plot(b_list, dV_dq,
                 color=colormap[i_j], label=np.round(j, 1))
    ax.legend(title='J')
    ax.set_xlabel('Sensory evidence, B')
    ax.set_ylabel('Slope of wells')


def response_time_vs_b(j=0.6, b_list=np.arange(0, 0.5, 0.01), theta=theta, noise=0.1,
                       tau=0.1, time_end=2, dt=1e-3,
                       nreps=1000, approx_init=True):
    err_rt = []
    mean_rt = []
    for b in b_list:
        rt = []
        for n in range(nreps):
            time, vec, _ = solution_mf_sdo_euler_OU_noise(j, b, theta, noise, tau,
                                                          time_end=time_end, dt=dt,
                                                          tau_n=tau, approx_init=approx_init)
            mean_states = np.clip(np.mean(vec, axis=1), 0, 1)
            mean_states[mean_states >= 0.9] = 1
            mean_states[mean_states <= (1-0.9)] = 0
            mean_states[(mean_states > 1-0.9)*(mean_states < 0.9)] = 2
            orders = gn.rle(mean_states)
            idx_1 = orders[1][orders[2] == 1]
            idx_0 = orders[1][orders[2] == 0]
            if len(idx_1) == 0:
                time_1 = np.nan
            else:
                time_1 = idx_1[0]
            if len(idx_0) == 0:
                time_0 = np.nan
            else:
                time_0 = idx_0[0]
            rt.append(np.nanmin((time_0, time_1))*dt*1000)
        err_rt.append(np.nanstd(rt)/np.sqrt(nreps))
        mean_rt.append(np.nanmean(rt))
    mean_rt = np.array(mean_rt)
    err_rt = np.array(err_rt)
    plt.figure()
    plt.plot(b_list, mean_rt, color='k', linewidth=2.5)
    plt.fill_between(b_list, mean_rt-err_rt, mean_rt+err_rt, color='k', alpha=0.3)
    plt.xlabel('Sensory evidence (B)')
    plt.ylabel('Response time (ms)')
    plt.ylim(100, 500)


def plot_dominance_duration_mean_field(j, b, theta=theta, noise=0,
                                       tau=1, time_end=10, dt=1e-1):
    time, vec, _ = solution_mf_sdo_euler_OU_noise(j, b, theta, noise, tau,
                                                       time_end=time_end, dt=dt,
                                                       tau_n=tau*100)
    # plt.figure()
    mean_states = np.clip(np.mean(vec, axis=1), 0, 1)
    # plt.plot(mean_states)
    # plt.axhline(5/8, color='k')
    # plt.axhline(1-5/8, color='k')
    # plt.ylim(0, 1)
    filt_post = scipy.signal.medfilt(mean_states, 1001)
    # plt.plot(filt_post)
    gn.plot_dominance_duration(j, b=b, n_nodes_th=5,
                               gibbs=False, mean_states=filt_post)


def plot_psychometric_mf(j_list=[0.1, 0.2, 0.4, 0.6],
                         blist=np.arange(-0.2, 0.2, 5e-2), theta=theta, noise=0.1,
                         tau=0.1, time_end=0.5, dt=1e-3, nreps=1000):
    fig, ax = plt.subplots(1)
    prightmat = np.empty((len(j_list), len(blist)))
    errmat = np.empty((len(j_list), len(blist)))
    for i_j, j in enumerate(j_list):
        pright_mean = []
        pright_err = []
        for b in blist:
            p_right = []
            for n in range(nreps):
                _, vec, _ = solution_mf_sdo_euler_OU_noise(j, b, theta, noise, tau,
                                                           time_end=time_end, dt=dt,
                                                           tau_n=tau,
                                                           approx_init=True)
                p_right_val = np.nanmean(vec[-1])
                p_right.append((np.sign(p_right_val-0.5)+1)/2)
            pright_mean.append(np.nanmean(p_right))
            pright_err.append(np.nanstd(p_right))
        prightmat[i_j, :] = pright_mean
        errmat[i_j, :] = pright_err
        ax.plot(blist, pright_mean, label=j)
        # ax.errorbar(blist, pright_mean, pright_err, label=j)
        plt.pause(0.01)
    ax.legend()


def plot_peak_noise_vs_j(j_list=np.arange(0.34, 0.45, 5e-3),
                         b=0, theta=theta, noise=0.1,
                         tau=0.01, time_end=1000, dt=1e-3, p_thr=0.5,
                         steps_back=2000, steps_front=500):
    fig, ax = plt.subplots(1)
    peaklist = []
    for i_j, j in enumerate(j_list):
        time, vec, ou_vals = solution_mf_sdo_euler_OU_noise(j, b, theta, noise, tau,
                                                            time_end=time_end, dt=dt,
                                                            tau_n=tau)
        mean_states = np.clip(np.mean(vec, axis=1), 0, 1)
        mean_ou = np.mean(ou_vals, axis=1)
        conv_window = 50
        mean_states = np.convolve(mean_states, np.ones(1000)/1000, mode='same')
        mean_states[mean_states > p_thr] = 1
        mean_states[mean_states < (1-p_thr)] = 0
        mean_states = mean_states[mean_states != p_thr]
        # mean_states[(mean_states > (1-p_thr)) & (mean_states < p_thr)] = 0
        orders = gn.rle(mean_states)
        idx_1 = orders[1][orders[2] == 1]
        idx_1 = idx_1[(idx_1 > steps_back) & (idx_1 < (len(mean_ou))-steps_front)]
        idx_0 = orders[1][orders[2] == 0]
        idx_0 = idx_0[(idx_0 > steps_back) & (idx_0 < (len(mean_ou))-steps_front)]
        ou_vals_1_array = np.empty((len(idx_1), steps_back+steps_front))
        ou_vals_1_array[:] = np.nan
        for i, idx in enumerate(idx_1):
            ou_vals_1_array[i, :] = np.convolve(mean_ou[idx - steps_back:idx+steps_front],
                                                np.ones(conv_window)/conv_window,
                                                mode='same')
        ou_vals_0_array = np.empty((len(idx_0), steps_back+steps_front))
        ou_vals_0_array[:] = np.nan
        for i, idx in enumerate(idx_0):
            ou_vals_0_array[i, :] = np.convolve(mean_ou[idx - steps_back:idx+steps_front],
                                                np.ones(conv_window)/conv_window,
                                                mode='same')*(-1)
        ou_vals_all = np.row_stack((ou_vals_1_array, ou_vals_0_array))
        ou_vals_mean = np.nanmean(ou_vals_all, axis=0)
        peaklist.append(np.nanmax(ou_vals_mean))
    ax.set_ylabel(r'Peak noise')
    ax.set_xlabel('Coupling, J')
    ax.plot(j_list, peaklist, color='k', linewidth=2.5)


def plot_noise_before_switch(j, b, theta=theta, noise=0.1,
                             tau=0.01, time_end=1000, dt=1e-3, p_thr=0.5,
                             steps_back=2000, steps_front=500, gibbs=False):
    if not gibbs:
        if j is None:
            j = 0.39
        time, vec, ou_vals = solution_mf_sdo_euler_OU_noise(j, b, theta, noise, tau,
                                                            time_end=time_end, dt=dt,
                                                            tau_n=tau)
        mean_states = np.clip(np.mean(vec, axis=1), 0, 1)
        mean_ou = np.mean(ou_vals, axis=1)
        conv_window = 50
        mean_states = np.convolve(mean_states, np.ones(1000)/1000, mode='same')
        lab = r'Noise $n(t)$'
    if gibbs:
        if j is None:
            j = 0.8
        burn_in = 100
        chain_length = int(time_end/dt)
        init_state = np.random.choice([-1, 1], theta.shape[0])
        states_mat =\
            gn.gibbs_samp_necker(init_state=init_state, burn_in=burn_in,
                                 n_iter=chain_length+burn_in,
                                 j=j, stim=b, theta=theta)
        mean_states = np.mean((states_mat+1)/2, axis=1)
        mean_ou = [gn.k_val(config, theta*j, stim=b) for config in states_mat]
        # mean_ou = np.exp(-np.gradient(mean_ou))
        conv_window = 1
        # ax.set_ylabel(r'$e^{\Delta k(\vec{x})}$')
        lab = r'$k(\vec{x})$'
        steps_back = steps_front =  100
    mean_states[mean_states > p_thr] = 1
    mean_states[mean_states < (1-p_thr)] = 0
    mean_states = mean_states[mean_states != p_thr]
    # mean_states[(mean_states > (1-p_thr)) & (mean_states < p_thr)] = 0
    orders = gn.rle(mean_states)
    idx_1 = orders[1][orders[2] == 1]
    idx_1 = idx_1[(idx_1 > steps_back) & (idx_1 < (len(mean_ou))-steps_front)]
    idx_0 = orders[1][orders[2] == 0]
    idx_0 = idx_0[(idx_0 > steps_back) & (idx_0 < (len(mean_ou))-steps_front)]
    ou_vals_1_array = np.empty((len(idx_1), steps_back+steps_front))
    ou_vals_1_array[:] = np.nan
    for i, idx in enumerate(idx_1):
        ou_vals_1_array[i, :] = np.convolve(mean_ou[idx - steps_back:idx+steps_front],
                                            np.ones(conv_window)/conv_window,
                                            mode='same')
    ou_vals_0_array = np.empty((len(idx_0), steps_back+steps_front))
    ou_vals_0_array[:] = np.nan
    for i, idx in enumerate(idx_0):
        ou_vals_0_array[i, :] = np.convolve(mean_ou[idx - steps_back:idx+steps_front],
                                            np.ones(conv_window)/conv_window,
                                            mode='same')*((-1)**(~gibbs))
    ou_vals_all = np.row_stack((ou_vals_1_array, ou_vals_0_array))
    ou_vals_mean = np.nanmean(ou_vals_all, axis=0)
    # ou_vals_std = np.nanstd(ou_vals_all, axis=0)
    time = np.arange(-steps_back, steps_front)*dt
    fig, ax = plt.subplots(1, figsize=(5, 4))
    ax.set_xlabel('Time to switch (s)')
    ax.set_ylabel(lab)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ou_vals_mean_filt = np.convolve(ou_vals_mean, np.ones(50)/50, mode='same')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.6)
    ax.plot(time, ou_vals_mean, color='k', linewidth=2.5)
    # ax.fill_between(time, ou_vals_mean-ou_vals_std, ou_vals_mean+ou_vals_std,
    #                 color='gray', alpha=0.3)
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'noise_before_switch.png', dpi=400, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'noise_before_switch.svg', dpi=400, bbox_inches='tight')


def mean_field_stim_change(j, b_list,
                           val_init=None, theta=theta,
                           burn_in=10):
    num_iter = len(b_list)+burn_in
    b_list = np.concatenate((np.repeat(b_list[0], burn_in),
                             b_list))
    if val_init is None:
        vec = np.random.rand(theta.shape[0])
    else:
        vec = np.repeat(val_init, theta.shape[0])
    vec_time = np.empty((num_iter, theta.shape[0]))
    vec_time[:] = np.nan
    for i in range(num_iter):
        for q in range(theta.shape[0]):
            neighbours = theta[q].astype(dtype=bool)
            # th_vals = theta[q][theta[q] != 0]
            vec[q] = gn.sigmoid(2*(sum(j*(2*vec[neighbours]-1))+b_list[i]))
        vec_time[i, :] = vec
    return vec_time[burn_in:]


def plot_posterior_vs_stim(j_list=[0.01, 0.2, 0.41],
                           b_list=np.linspace(-0.5, 0.5, 1001),
                           theta=theta):
    plt.figure()
    # colormap = pl.cm.Oranges(np.linspace(0.4, 1, len(j_list)))
    colormap = ['navajowhite', 'orange', 'saddlebrown']
    for i_j, j in enumerate(reversed(j_list)):
        vec_vals = []
        for b in b_list:
            vec = mean_field_stim(j, stim=b, num_iter=20, val_init=0.5,
                                  theta=theta, sigma=0)
            vec_vals.append(np.nanmean(vec[-1]))
        plt.plot(b_list, vec_vals, color=colormap[i_j],
                 label=np.round(j, 1), linewidth=4)
    plt.xlabel('Stimulus strength, B')
    plt.ylabel('Confidence')
    plt.legend(title='J')


def mean_field_simul_discrete(j_list, b, theta, num_iter=200):
    fig, ax = plt.subplots(ncols=2)
    vals_final = np.empty((len(j_list), theta.shape[0]))
    mean_neighs = 1/((1/3+1/4)/2)
    # mean_neighs = 3+1/3
    mean_neighs = np.max(np.linalg.eigvals(theta))
    vals_mean = []
    for i_j, j in enumerate(j_list):
        vec = mean_field_stim(j, stim=b, num_iter=num_iter, val_init=None,
                              theta=theta, sigma=0.)
        vals_final[i_j, :] = np.nanmax((vec[-1], 1-vec[-1]), axis=0)
        vec = np.random.rand()
        for t in range(1, num_iter):
            vec = gn.sigmoid(2*mean_neighs*(j*(2*vec-1))+2*b)
        vals_mean.append(np.max((vec, 1-vec)))
    vals_mean = np.array(vals_mean)
    mean_neighs = (1/3+1/4)/2
    mean_neighs = 1/np.max(np.linalg.eigvals(theta))
    for i in range(theta.shape[0]):
        if np.sum(theta[i]) == 3:
            color = 'r'
        else:
            color = 'k'
        ax[0].plot(j_list, vals_final[:, i], color=color,
                   linewidth=1.5)
        ax[0].plot(j_list, 1-vals_final[:, i], color=color,
                   linewidth=1.5)
    ax[0].plot(j_list, 1-np.mean(vals_final, axis=1), color='g',
               linewidth=1.5)
    ax[0].plot(j_list, np.mean(vals_final, axis=1), color='g',
               linewidth=1.5, label='mean all vals')
    ax[1].plot(j_list, 1-np.mean(vals_final, axis=1), color='g',
               linewidth=1.5, label='mean all vals')
    ax[1].plot(j_list, np.mean(vals_final, axis=1), color='g',
               linewidth=1.5)
    ax[1].plot(j_list, vals_mean, color='b',
               linewidth=1.5, alpha=0.4, label='f(N_bif = (1/(1/3+1/4)*2))')
    ax[1].plot(j_list, 1-vals_mean, color='b',
               linewidth=1.5, alpha=0.4)
    ax[0].axvline(mean_neighs, color='b', linestyle='--', alpha=0.2)
    ax[0].legend()
    ax[1].legend()
    ax[1].axvline(mean_neighs, color='b', linestyle='--', alpha=0.2)


def mean_field_mean_neighs(J, num_iter, stim, theta=theta, val_init=None):
    #initialize random state of the cube
    if val_init is None:
        vec = np.random.rand()
    else:
        vec = val_init
    vec_time = np.empty((num_iter))
    vec_time[:] = np.nan
    vec_time[0] = vec
    mean_neighs = 3.7
    for i in range(1, num_iter):
        vec = gn.sigmoid(2*mean_neighs*(J*(2*vec-1))+2*stim)
        vec_time[i] = vec
    return vec_time


def mean_nodes_simulation_comparison(b, j, theta=theta):
    vec = mean_field_stim(j, stim=b, num_iter=10, val_init=0.9,
                          theta=theta, sigma=0)
    mean_vec_time = np.mean(vec, axis=1)
    vec_hyp =\
        mean_field_mean_neighs(j, num_iter=10, stim=b, theta=theta, val_init=0.9)
    plt.figure()
    plt.plot(mean_vec_time, label='<q>')
    plt.plot(vec_hyp, label='f(<N>), approx')
    plt.legend()
    plt.figure()
    plt.plot(np.abs(mean_vec_time-vec_hyp))


def plot_mf_hysteresis(j_list=[0.1, 0.25, 0.41],
                       b_list=np.linspace(-0.5, 0.5, 1001)):
    b_list = np.concatenate((b_list[:-1], b_list[::-1]))
    fig, ax = plt.subplots(1, figsize=(5., 3.5))
    colormap = ['navajowhite', 'orange', 'saddlebrown']
    colormap = pl.cm.binary(np.linspace(0.2, 1, len(j_list)))[::-1]
    colormap = ['midnightblue', 'royalblue', 'lightskyblue'][::-1]
    ax.axvline(0, color='k', alpha=0.3, linestyle='--', zorder=1)
    for i_j, j in enumerate(j_list):
        x = 0.1
        dt = 0.1
        tau = 0.6
        for i in range(5):
            x = gn.sigmoid(2*j*3*(2*x-1)+2*b_list[0])
        vec = [x]
        for i in range(len(b_list)-1):
            x = x + dt*(gn.sigmoid(2*j*3*(2*x-1)+2*b_list[i])-x)/tau
            vec.append(x)
        ax.plot(b_list, vec, color=colormap[i_j],
                label=np.round(j, 1), linewidth=4,
                zorder=2)
    ax.arrow(-0.145, 0.6, -0.005, -0.13,
              color=colormap[-1], zorder=3, head_width=0.05)
    ax.arrow(0.145, 0.4, 0.005, 0.13,
              color=colormap[-1], zorder=3, head_width=0.05)
    ax.set_xlabel('Sensory evidence, B(t)')
    ax.set_ylabel('Approximate posterior q(x=1)')
    ax.set_ylim(-0.05, 1.05)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(title='Coupling, J', frameon=False)
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'hysteresis_cartoon.png', dpi=400,
                bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'hysteresis_cartoon.svg', dpi=400,
                bbox_inches='tight')
    # plt.legend(title='J')


def boltzmann_2d_change_j(noise=0.1, b=0):
    # noise_vals = np.arange(0, 0.3, 0.001)
    q = np.arange(0, 1, 1e-3)
    j_vals = np.arange(0, 1, 1e-3)
    x, y = np.meshgrid(j_vals, q)
    pot = potential_mf(y, x, bias=b)
    distro = np.exp(-2*pot/noise**2)
    for i in range(len(distro)):
        distro[:, i] = distro[:, i] / np.nansum(distro[:, i])
        # distro[:, i] = (distro[:, i] - np.min(distro[:, i]))/(np.max(distro[:, i])- np.min(distro[:, i]))
    fig, ax = plt.subplots(1)
    im = ax.imshow(np.flipud(distro), cmap='Oranges', extent=[0, 1, 0, 1])
    plt.colorbar(im, ax=ax, label='$p(q \; | \; J, B, \sigma)$')
    ax.set_xlabel('Coupling, J')
    ax.set_ylabel('Approximate posterior q(x=1)')


def boltzmann_2d_change_sigma(j=0.3, b=0):
    # noise_vals = np.arange(0, 0.3, 0.001)
    q = np.arange(0, 1, 1e-3)
    sigma_vals = np.arange(1e-2, 0.05, 1e-4)
    sigma_vals = np.logspace(-2, 0, 51)
    x, y = np.meshgrid(sigma_vals, q)
    pot = potential_mf(y, np.repeat(j, y.shape[0]*y.shape[1]).reshape(y.shape[0],
                                                          y.shape[1]), bias=b)
    distro = np.exp(-2*pot/x**2)
    for i in range(y.shape[1]):
        distro[:, i] = distro[:, i] / np.nansum(distro[:, i])
    fig, ax = plt.subplots(1)
    im = ax.imshow(np.flipud(distro), cmap='Oranges', aspect='auto', extent=[1e-2, 0.05, 0, 1])
    plt.colorbar(im, ax=ax, label='p(q)')
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'$q(x=1)$')


def boltzmann_2d_change_stim(j=0.3, noise=0.03):
    # noise_vals = np.arange(0, 0.3, 0.001)
    q = np.arange(0, 1, 1e-3)
    b_vals = np.arange(-0.1, 0.1, 1e-3)
    x, y = np.meshgrid(b_vals, q)
    pot = potential_mf(y, j, bias=x)
    distro = np.exp(-2*pot/noise**2)
    for i in range(y.shape[1]):
        distro[:, i] = distro[:, i] / np.nansum(distro[:, i])
    fig, ax = plt.subplots(1)
    im = ax.imshow(np.flipud(distro), cmap='Oranges', aspect='auto', extent=[-0.1, 0.1, 0, 1])
    plt.colorbar(im, ax=ax, label='p(q)')
    ax.set_xlabel(r'$b$')
    ax.set_ylabel(r'$q(x=1)$')


def plot_boltzmann_distro_pdf(j, noise, b=0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    q = np.arange(0, 1.001, 0.001)
    pot = potential_mf(q, j, bias=b)
    distro = np.exp(-2*pot/noise**2)
    ax.plot(q, distro / np.sum(distro),
            color='r', label='analytical')
    ax.set_xlabel('Approximate posterior q(x=1)')
    ax.set_ylabel('p(q(x=1))')


def plot_hysteresis_different_taus(j=0.36,
                                   b_list=np.linspace(-0.53, 0.53, 501),
                                   save_folder=DATA_FOLDER,
                                   tau_list=[0.1,  1], sigma=0,
                                   dt=0.1):
    b_list = np.concatenate((b_list[:-1], b_list[::-1]))
    fig, ax = plt.subplots(1, figsize=(5, 3.5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # colormap = pl.cm.binary(np.linspace(0.2, 1, len(tau_list)))
    # colormap = ['midnightblue', 'midnightblue']
    lsts = ['solid', '--']
    fig2, ax2 = plt.subplots(1, figsize=(4.5, 4))
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    for i_t, tau in enumerate(tau_list):
        x = 0.1
        for i in range(5):
            x = gn.sigmoid(2*j*3*(2*x-1)+2*b_list[0])
        vec = [x]
        for i in range(len(b_list)-1):
            x = x + dt*(gn.sigmoid(2*j*3*(2*x-1)+2*b_list[i])-x)/tau + sigma*np.random.randn()*np.sqrt(dt/tau)
            vec.append(x)
        ax.plot(b_list, vec, linewidth=4, color='midnightblue', label=tau,
                linestyle=lsts[i_t])
    taulist_2 = [0.2, 1]
    # hyst_dist_analytic = []
    # n = 3
    # delta = np.sqrt(1-1/(j*n))
    # b_crit1 = (np.log((1-delta)/(1+delta))+2*n*j*delta)/2
    # b_crit2 = (np.log((1+delta)/(1-delta))-2*n*j*delta)/2
    j_list = np.arange(0, 0.4, 1e-2)
    for i_t, tau in enumerate(taulist_2):
        hyst_dist_simul = []
        for j in j_list:
            x = 0.1
            for i in range(5):
                x = gn.sigmoid(2*j*3*(2*x-1)+2*b_list[0])
            vec = [x]
            for i in range(len(b_list)-1):
                x = x + dt*(gn.sigmoid(2*j*3*(2*x-1)+2*b_list[i])-x)/tau + sigma*np.random.randn()*np.sqrt(dt/tau)
                vec.append(x)
            idx_asc = np.argmin(np.abs(np.array(vec)[:len(vec)//2]-0.5))
            idx_desc = np.argmin(np.abs(np.array(vec)[len(vec)//2:]-0.5))
            hystval = b_list[:len(vec)//2][idx_asc]-b_list[len(vec)//2:][idx_desc]
            hyst_dist_simul.append(hystval)
        ax2.plot(j_list, hyst_dist_simul, color='k', linewidth=3,
                 linestyle=['--', 'solid'][i_t], label=['Slow', 'Fast'][i_t])
    ax.set_xlabel('Sensory evidence, B(t)')
    ax.legend(title=r'$\tau$', frameon=False)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Approximate posterior q(x=1)')
    ax2.set_xlabel('Coupling, J')
    ax2.legend(title=r'$\tau$', frameon=False)
    ax2.set_ylabel('Hysteresis width')
    # ax2.set_yscale('log')
    fig.tight_layout()
    fig.savefig(save_folder + 'hysteresis_taus.png', dpi=200, bbox_inches='tight')
    fig.savefig(save_folder + 'hysteresis_taus.svg', dpi=200, bbox_inches='tight')
    fig2.tight_layout()
    fig2.savefig(save_folder + 'hysteresis_distance_taus.png', dpi=200, bbox_inches='tight')
    fig2.savefig(save_folder + 'hysteresis_distance_taus.svg', dpi=200, bbox_inches='tight')


def save_images_potential_hysteresis(j=0.39,
                                     b_list=np.linspace(-0.2, 0.2, 501),
                                     save_folder=DATA_FOLDER, tau=0.8,
                                     sigma=0.):
    b_list = np.concatenate(([-0.2, -0.2], b_list[:-1], b_list[::-1]))
    x = 0.0931
    vec = [x]
    dt = 0.1
    for i in range(len(b_list)-1):
        x = x + dt*(gn.sigmoid(2*j*3*(2*x-1)+2*b_list[i])-x)/tau +\
            np.random.randn()*sigma*np.sqrt(dt/tau)
        vec.append(x)
    # plt.figure()
    # plt.plot(b_list, vec)
    if tau >= 0.5:
        lab = '/fast/' if sigma == 0 else '/fast_noisy/'
    else:
        lab = '/slow/' if sigma == 0 else '/slow_noisy/'
    q = np.arange(0, 1, 0.001)
    for i in range(len(vec)):
        fig, ax = plt.subplots(nrows=2, figsize=(6, 10))
        ax[0].plot(b_list[:i], vec[:i], color='navajowhite',
                   linewidth=4)
        ax[0].plot(b_list[i], vec[i],
                   marker='o', markersize=8, color='k')
        ax[0].set_ylabel('Approximate posterior, q(x=1)')
        ax[0].set_xlabel('Stimulus strength, B')
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].set_xlim(-0.21, 0.21)
        ax[0].set_ylim(0, 1)
        pot = potential_mf(q, j, b_list[i])
        val_particle = potential_mf(vec[i], j, b_list[i])
        ax[1].plot(q, pot-np.nanmean(pot), color='purple',
                   linewidth=4)
        ax[1].plot(vec[i], val_particle-np.nanmean(pot),
                   marker='o', markersize=8, color='k')
        ax[1].set_ylabel('Potential')
        ax[1].set_xlabel('Approximate posterior, q(x=1)')
        ax[1].spines['left'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].set_yticks([])
        ax[1].set_xticks([])
        fig.savefig(save_folder + '/images_video_hyst' + lab + str(i) + '.png',
                    dpi=100)
        plt.close(fig)

    
def create_video_from_images(image_folder=DATA_FOLDER+'/images_video_hyst/fast/'):
    video_name = image_folder + 'hysteresis_fast.mp4'
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = [images[i].replace('.png', '') for i in range(len(images))]
    images.sort(key=float)
    images = [images[i] + '.png' for i in range(len(images))]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'),
                            50, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()


def mean_field_both_eyes(j=0.45, theta=theta, b_list=np.arange(0, 0.25, 0.001)):
    max_val = []
    min_val = []
    unst_val = []
    for i_b, b in enumerate(b_list[b_list >= 0]):
        b_both_eyes = np.repeat(b, theta.shape[0])
        b_both_eyes[theta.shape[0]//2:] = -b
        vec_min = mean_field_stim(J=j, num_iter=20, stim=b_both_eyes, sigma=0, theta=theta,
                                  val_init=0.1)
        vec_max = mean_field_stim(J=j, num_iter=20, stim=b_both_eyes, sigma=0, theta=theta,
                                  val_init=0.9)
        unstable = find_repulsor(j=j, num_iter=20, q_i=0.001, q_f=0.999,
                                 stim=b_both_eyes, threshold=1e-10, theta=theta, neigh=3)
        min_states = np.mean(vec_min[-1])
        max_states = np.mean(vec_max[-1])
        max_val.append(max_states)
        min_val.append(min_states)
        unst_val.append(unstable)
    plt.figure()
    plt.plot(b_list, min_val, color='k')
    plt.plot(b_list, unst_val, color='k', linestyle='--')
    plt.plot(b_list, max_val, color='k')
    plt.ylabel('Approximate posterior q(x=1)')
    plt.xlabel('Stimulus magnitude, B')


def levelts_laws(noise=0.1, j=0.39, b_list=np.arange(0, 0.25, 0.01),
                 theta=theta, time_end=10000, dt=1e-3, tau=0.008,
                 n_nodes_th=75):
    p_thr = n_nodes_th/100
    list_time_q1 = []
    list_time_q2 = []
    list_predom_q1 = []
    list_predom_q2 = []
    alt_rate = []
    alt_rate_both_eyes = []
    for i_b, b in enumerate(b_list):
        print(round(100*(i_b+1)/len(b_list), 2))
        print('%')
        init_state = np.random.choice([-1, 1], theta.shape[0])
        time, vec, _ = solution_mf_sdo_euler_OU_noise(j, b, theta, noise, tau,
                                                      time_end=time_end, dt=dt,
                                                      tau_n=tau)
        mean_states = np.clip(np.mean(vec, axis=1), 0, 1)
        mean_states2 = scipy.signal.medfilt(mean_states, 1001)
        # mean_states2 = np.copy(mean_states)
        mean_states[mean_states > p_thr] = 1
        mean_states[mean_states < (1-p_thr)] = -1
        if n_nodes_th == 50:
            mean_states = mean_states[mean_states != p_thr]
        mean_states[(mean_states > (1-p_thr)) & (mean_states < p_thr)] = 0
        orders = gn.rle(mean_states)
        time_q1 = orders[0][orders[2] == 1]
        time_q2 = orders[0][orders[2] == -1]
        mean_states2[mean_states2 > 0.5] = 1
        mean_states2[mean_states2 < 0.5] = 0
        mean_states2 = mean_states2[mean_states2 != 0.5]
        list_predom_q1.append(np.mean(mean_states2))
        list_predom_q2.append(1-np.mean(mean_states2))
        list_time_q1.append(np.mean(time_q1))
        list_time_q2.append(np.mean(time_q2))
        alt_rate.append(np.sum(np.diff(mean_states2) != 0))
    for i_b, b in enumerate(b_list[b_list >= 0]):
        b_both_eyes = np.repeat(b, theta.shape[0])
        b_both_eyes[theta.shape[0]//2:] = -b
        time, vec, _ = solution_mf_sdo_euler_OU_noise(j, b, theta, noise, tau,
                                                      time_end=time_end, dt=dt,
                                                      tau_n=tau)
        mean_states = np.clip(np.mean(vec, axis=1), 0, 1)
        mean_states2 = np.copy(mean_states)
        mean_states2[mean_states2 > 0.5] = 1
        mean_states2[mean_states2 < 0.5] = 0
        mean_states2 = mean_states2[mean_states2 != 0.5]
        alt_rate_both_eyes.append(np.sum(np.diff(mean_states2) != 0))
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(12, 10))
    ax = ax.flatten()
    ax[0].plot(b_list, list_predom_q1, label='q(x=1)', color='green', linewidth=2.5)
    ax[0].plot(b_list, list_predom_q2, label='q(x=-1)', color='r', linewidth=2.5)
    ax[0].legend()
    ax[0].set_xlabel('Sensory evidence, B')
    ax[0].set_ylabel('Perceptual predominance (<q(x=i)>')
    ax[1].plot(b_list, list_time_q1, label='q(x=1)', color='green', linewidth=2.5)
    ax[1].plot(b_list, list_time_q2, label='q(x=-1)', color='r', linewidth=2.5)
    ax[1].set_yscale('log')
    # ax[1].legend()
    ax[1].set_xlabel('Sensory evidence, B')
    ax[1].set_ylabel('Avg. perceptual dominance, T(x=1)')
    ax[2].plot(list_predom_q1, alt_rate, color='k', linewidth=2.5)
    axtwin = ax[2].twinx()
    f = np.array(list_predom_q1)
    xf = np.arange(1e-6, 0.999, 1e-3)
    entropy = -xf*np.log(xf) - (1-xf)*np.log(1-xf)
    axtwin.plot(xf, entropy, color='grey', linestyle='--')
    axtwin.set_ylabel('Entropy')
    ax[2].set_xlabel('Fraction of q(x=1)')
    ax[2].set_ylabel('Alternation rate')
    ax[3].plot(f[b_list >= 0], alt_rate_both_eyes, color='k',
               linewidth=2.5)
    ax[3].set_xlabel('Fraction of q(x=1)')
    ax[3].set_ylabel('Alternation rate')
    x_vals = np.arange(1, np.max((max(list_time_q2), max(list_time_q1))), 1)
    linereg = scipy.stats.linregress(np.log(list_time_q1), np.log(list_time_q2))
    y = np.log(x_vals)*linereg.slope + linereg.intercept
    ax[4].plot(list_time_q2, list_time_q1, label='q(x=1)', color='k', linewidth=2.5,
               marker='o')
    ax[4].set_ylabel('T(x=1)')
    ax[4].set_xlabel('T(x=-1)')
    ax[5].plot(np.log(list_time_q2), np.log(list_time_q1),
               color='k', linewidth=2.5,
               marker='o', label='Simulation')
    ax[5].plot(y, np.log(x_vals), color='b', linestyle='--', alpha=0.4,
               label=f'y ~ log(x), slope={round(linereg.slope, 2)}')
    ax[5].legend()
    ax[5].set_ylabel('log T(x=1)')
    ax[5].set_xlabel('log T(x=-1)')
    # ax[5].set_yscale('log')
    # ax[5].set_xscale('log')
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))
    ax = ax.flatten()
    ax[0].plot(b_list, list_predom_q1, label='q(x=1)', color='k', linewidth=2.5)
    ax[0].plot(b_list, list_predom_q2, label='q(x=-1)', color='r', linewidth=2.5)
    ax[0].legend()
    ax[0].set_xlabel('Sensory evidence, B')
    ax[0].set_ylabel('Perceptual predominance (<q(x=i)>')
    ax[1].plot(b_list, list_time_q1, label='q(x=1)', color='k', linewidth=2.5)
    ax[1].plot(b_list, list_time_q2, label='q(x=-1)', color='r', linewidth=2.5)
    # ax[1].legend()
    ax[1].set_xlabel('Sensory evidence, B')
    ax[1].set_ylabel('Average perceptual dominance, T(x=1)')
    ax[2].plot(b_list, alt_rate, color='k', linewidth=2.5)
    ax[2].set_xlabel('Sensory evidence, B')
    ax[2].set_ylabel('Alternation rate')
    ax[3].plot(b_list[b_list >= 0], alt_rate_both_eyes, color='k',
               linewidth=2.5)
    ax[3].set_xlabel('Sensory evidence, B')
    ax[3].set_ylabel('Alternation rate')
    fig3, ax3 = plt.subplots(1)
    ax3.plot(list_predom_q1, list_time_q1, label='q(x=1)', color='green', linewidth=2.5)
    ax3.plot(list_predom_q2, list_time_q2, label='q(x=-1)', color='r', linewidth=2.5)
    ax.legend()
    ax3.set_xlabel('Perceptual predominance (<q(x=i)>')
    ax3.set_ylabel('Avg. perceptual dominance, T(x=1)')


def mean_field_kernel(J, num_iter, stim, sigma=1, theta=theta, val_init=None, sxo=0.):
    #initialize random state of the cube
    if val_init is None:
        vec = np.random.rand(theta.shape[0])
    else:
        vec = np.repeat(val_init, theta.shape[0]) + np.random.randn()*sxo
    vec_time = np.empty((num_iter, theta.shape[0]))
    vec_time[:] = np.nan
    vec_time[0, :] = vec
    for i in range(1, num_iter):
        for q in range(theta.shape[0]):
            if isinstance(stim, np.ndarray):
                b = stim[q]
            else:
                b = stim
            vec[q] = gn.sigmoid(2*(np.sum(J*theta[q]*(2*vec-1))+b))+np.random.randn()*sigma
        vec_time[i, :] = vec
    return vec_time


def plot_circle_dynamics(j=1, b=0., noise=0.1, ini_cond=None):
    colors = ['k', 'b', 'r']
    taulist = [0.1, 1, 2]
    fig, ax = plt.subplots(ncols=3, figsize=(6, 5))
    for i_tau, tau in enumerate(taulist):
        kernel = exp_kernel(tau=tau, x=np.arange(10))
        theta = scipy.linalg.circulant(kernel).T
        # stim = np.zeros((n_iters, theta.shape[0]))
        # stim[n_iters//2-n_iters//4:n_iters//2-n_iters//4+n_iters//8, ::2] = b
        # stim[n_iters//2-+n_iters//4+n_iters//8+1:1+n_iters//2-n_iters//4+2*n_iters//8, 1::2] = -b
        ini_cond = 0.5 + np.random.randn(theta.shape[0])*0.01
        time, vec_time =  solution_mf_sdo_euler(j=j, b=b, theta=theta, noise=noise,
                                                tau=1, time_end=10, dt=1e-3, ini_cond=ini_cond)
        for i in range(theta.shape[0]):
            ax[i_tau].plot(time, vec_time[:, i], color=colors[i_tau], alpha=0.4)
        


def mf_dyn_sys_circle(n_iters=200, b=0.):
    colors = ['k', 'b', 'r']
    taulist = [0.1, 1, 2]
    fig, ax = plt.subplots(1, figsize=(6, 5))
    for i_tau, tau in enumerate(taulist):
        kernel = exp_kernel(tau=tau, x=np.arange(10))
        theta = scipy.linalg.circulant(kernel).T
        # stim = np.zeros((n_iters, theta.shape[0]))
        # stim[n_iters//2-n_iters//4:n_iters//2-n_iters//4+n_iters//8, ::2] = b
        # stim[n_iters//2-+n_iters//4+n_iters//8+1:1+n_iters//2-n_iters//4+2*n_iters//8, 1::2] = -b
        jlist = np.linspace(0.01, 1, 1000)
        vecarr = np.empty((len(jlist), theta.shape[0]))
        for t in range(len(jlist)):
            vec_time = mean_field_kernel(jlist[t], n_iters, b, val_init=None, theta=theta, sigma=0)
            vecarr[t] = np.max((vec_time[-1], 1-vec_time[-1]), axis=0)
        for i in range(theta.shape[0]):
            ax.plot(jlist, vecarr[:, i], color=colors[i_tau], label=tau)
            ax.plot(jlist, 1-vecarr[:, i], color=colors[i_tau])
        ax.axvline(1/np.sum(kernel), color=colors[i_tau])
    ax.set_ylim(-0.05, 1.05)
    # ax.legend()
    ax.set_xlabel('Coupling, J')
    ax.set_ylabel('Approximate posterior (q(x=1))')
        # ax[1].plot(stim[:, i])
    # ax[0].set_ylabel('Approx. posterior')
    # ax[1].set_ylabel('Bias')
    # ax[1].set_xlabel('Timestep')


def plot_max_eigval():
    eig1 = []
    for tau in np.logspace(-5, 5, 100):
        kernel = exp_kernel(tau=tau)
        theta = scipy.linalg.circulant(kernel).T
        eig1.append(np.max(np.linalg.eigvals(theta)))
    plt.figure()
    plt.plot(np.logspace(-5, 5, 100), eig1)
    plt.xscale('log')
    plt.xlabel('Tau of the kernel')
    plt.ylabel('Max. eigenvalue')
    plt.axhline(39, color='k', linestyle='--')


def analytical_eigval_circulant(taulist=np.logspace(-5, 5, 100)):
    max_eigvals = []
    # for circulant matrix, eigvals are Discrete Fourier Transform of the kernel!!
    for tau in taulist:
        ker = exp_kernel(tau=tau)
        n = len(ker)
        # eigvals = []
        # for k in range(n):
        #     sumker = 2*np.sum([ker[j]*np.cos(-2*np.pi*k*j/n) for j in range(1, n//2)])
        #     eigval = ker[n // 2]*np.cos(k*np.pi) +  sumker
        #     eigvals.append(eigval)
        # max_eigvals.append(np.max(eigvals))
        # argmax.append(np.argmax(eigvals))
        maxeigval = np.exp(-(n/2-1)/tau) + 2*np.sum([np.exp(-(j-1)/tau) for j in range(1, n//2)])
        maxeigval = np.sum(ker)
        max_eigvals.append(maxeigval)
    fig, ax = plt.subplots(ncols=2)
    for a in ax:
        a.set_xscale('log')
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    ax[0].plot(taulist, max_eigvals)
    ax[1].plot(taulist, 1/np.array(max_eigvals))
    ax[0].set_xlabel('Tau of the kernel')
    ax[1].set_xlabel('Tau of the kernel')
    ax[0].set_ylabel('Max. eigenvalue')
    ax[1].set_ylabel('Critical coupling J*')
    ax[0].axhline(39, color='k', linestyle='--')
    ax[1].axhline(1/39, color='k', linestyle='--')


def mean_field_stim_change_node(j, b_list,
                                val_init=None, theta=theta):
    num_iter = b_list.shape[0]
    if val_init is None:
        vec = np.random.rand(theta.shape[0])
    else:
        vec = np.repeat(val_init, theta.shape[0])
    vec_time = np.empty((num_iter, theta.shape[0]))
    vec_time[:] = np.nan
    for i in range(num_iter):
        for q in range(theta.shape[0]):
            neighbours = theta[q].astype(dtype=bool)
            vec[q] = gn.sigmoid(2*(np.sum(j*theta[q, neighbours]*(2*vec[neighbours]-1))+b_list[i, q]))
        vec_time[i, :] = vec
    return vec_time


def exp_kernel(x=np.arange(40), tau=4):
    kernel = np.concatenate((np.exp(-(x-1)[:len(x)//2]/tau), (np.exp(-x[:len(x)//2]/tau))[::-1]))
    kernel[0] = 0
    return kernel


def gauss_kernel(x=np.arange(40), sigma=1):
    kernel = np.roll(np.exp(-((x-len(x)//2)**2)/sigma), 20)
    kernel[0] = 0
    return kernel


def confidence_matrix(jlist=np.arange(0, 0.5, 1.25e-2),
                      blist=np.arange(-0.2, 0.22, 1e-2), n=3):
    matrix = np.zeros((len(jlist), len(blist)))
    for ij, j in enumerate(jlist):
        for ib, stim in enumerate(blist):
            q = lambda q: gn.sigmoid(2*n*j*(2*q-1)+ stim*2) - q 
            init_cond = 0.9 if stim >= 0 else 0.1
            matrix[ij, ib] = fsolve(q, init_cond)[0]
    fig, ax = plt.subplots(1)
    ax.imshow(np.flipud(matrix), cmap='PRGn', aspect='auto')


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    from matplotlib.collections import LineCollection
    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def examples_pot():
    b_list = [-0.05, 0, 0.05]
    j_list = [0.25, 0.38, 0.5]
    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(6, 6))
    ax = ax.flatten()
    k = 0
    for j in j_list:
        if j < 0.4:
            q = np.arange(0.03, .97, 1e-3)
        else:
            q = np.arange(-0.12, 1.12, 1e-3)
        for b in b_list:
            ax[k].plot(q, potential_mf(q, j, b), color='k', linewidth=4)
            ax[k].axis('off')
            k += 1
    fig.savefig(DATA_FOLDER + 'examples_potentials.png', dpi=400,
                bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'examples_potentials.svg', dpi=400,
                bbox_inches='tight')


def lambda_k_eigval(kernel, k):
    n = len(kernel)
    return np.sum([np.real(np.exp(1j*2*np.pi*i*k/n))*kernel[i] for i in range(n)])


def mu_k_eigval(j, b, kernel, k):
    sol1, sol2, solunst = get_unst_and_stab_fp(j, b, tol=1e-10)
    if np.isnan(solunst):
        sol = sol1
        lambda_k = lambda_k_eigval(kernel, k)
        critj = 1/(lambda_k_eigval(kernel, k=0)*4*sol*(1-sol))
        muk = lambda_k*4*sol*(1-sol)*j
        return muk
    else:
        if b > 0:
            bif_att = np.min([sol1, sol2])
        if b < 0:
            bif_att = np.max([sol1, sol2])


def get_path_between(x, j, b, noise, theta, steps=1000, step_size=1e-1, dt=1,
                     num_prints=15, num_stashes=80, tol_stop=1e-2, neg=False):
    t = np.linspace(0, len(x) - 1, len(x)) * dt
    print_on = np.linspace(0, int(np.sqrt(steps)), num_prints).astype(np.int32) ** 2  # print more early on
    stash_on = np.linspace(0, int(np.sqrt(steps)), num_stashes).astype(np.int32) ** 2
    xs = []
    action_vals = []
    for i in range(steps):
        # Compute action gradient with respect to x
        x.requires_grad_(True)  # Ensure x has requires_grad=True
        lag = action(x, j, b, noise, theta, dt=dt, n=3, neg=neg)
        grad_x = torch.autograd.grad(lag, x, allow_unused=False)[0]  # Only take the gradient wrt x

        grad_x[[0, -1]] *= 0  # Fix first and last coordinates by zeroing their gradients
        x = x - grad_x * step_size  # Update x

        if i in print_on:
            print('step={:04d}, S={:.4e}'.format(i, lag.item()))
        if i in stash_on:
            xs.append(x.clone().detach().numpy())
        if i > 0 and np.abs(lag.item() - action_vals[-1]) <= tol_stop:
            print('Reached stop tolerance ' + str(tol_stop))
            print('Minimum action: ' + str(lag.item()))
            # print(action_vals[-1])
            break
        action_vals.append(lag.item())
    return t, x, np.stack(xs), action_vals


def lagrangian(x, xdot, j, b, noise, theta, tau=0.1, dt=1e-2, n=3, neg=False):
    fx = 1 / (1 + torch.exp(-2 * j * (2 * torch.matmul(theta, x.T) - n) - 2 * b)) * (-1)**neg
    lag = torch.norm(xdot.T - fx)/(2*noise**2)
    return lag


def action(x, j, b, noise, theta, dt=1e-2, n=3, neg=False):
    xdot = (x[1:] - x[:-1]) / dt
    xdot = torch.cat([xdot, xdot[-1:]], axis=0)
    lag = lagrangian(x, xdot, j, b, noise, theta, neg=neg)
    return lag


def minimum_action_path(j, b, noise, theta, numiters=100, dt=1e-2, steps=1000,
                        tol_stop=1e-2, neg=False):
    x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j, b)
    slope = (x_stable_2-x_stable_1)/numiters
    # line from A to B with gaussian noise
    x0 = torch.randn((numiters, theta.shape[0]), requires_grad=True)*0.05 +\
        torch.arange(numiters, dtype=torch.float, requires_grad=True).reshape(-1, 1)*(slope) +\
            x_stable_1-x_stable_2
    x_stable_1 = np.repeat(x_stable_1, theta.shape[0])  # +np.random.randn(theta.shape[0])*1e-3
    x_stable_2 = np.repeat(x_stable_2, theta.shape[0])
    x_unstable = np.repeat(x_unstable, theta.shape[0])
    x0[0] = torch.tensor(x_stable_1)
    x0[-1] = torch.tensor(x_stable_2)
    theta = torch.tensor(theta * 1., dtype=torch.float)
    t, x, xs, action_vals = get_path_between(x0.clone(), j, b, noise, theta, steps=steps, step_size=1e-5, dt=dt,
                                             tol_stop=tol_stop, neg=neg)
    return t, x, xs, x_unstable, action_vals


def reduced_symmetric_2d_system(j, b, t_end=10, dt=5e-3, noise=0.1,
                                tau=0.1):
    x = 0.51
    y = 0.8
    # x, y = np.random.rand(2)
    x_l = [x]
    y_l = [y]
    time = np.arange(0, t_end+dt, dt)
    for t in time[:-1]:
        x_n = gn.sigmoid(2*3*j*(2*y-1)+2*b)-x
        y_n = gn.sigmoid(2*3*j*(2*x-1)+2*b)-y
        x = x + x_n*dt/tau + noise*np.sqrt(dt/tau)*np.random.randn()
        y = y + y_n*dt/tau + noise*np.sqrt(dt/tau)*np.random.randn()
        x_l.append(x)
        y_l.append(y)
    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(time, x_l, color='r')
    ax[0].plot(time, y_l, color='k')
    ax[1].plot(x_l, y_l, color='k')
    fig, ax = plt.subplots(1)
    xv = np.arange(0, 1.1, 5e-2)
    yv = np.arange(0, 1.1, 5e-2)
    xx, yy = np.meshgrid(xv, yv)
    uu = gn.sigmoid(2*3*j*(2*yy-1)+2*b)-xx
    vv = gn.sigmoid(2*3*j*(2*xx-1)+2*b)-yy
    ax.quiver(xx, yy, uu, vv)
    ax.plot(x_l, y_l, color='r')


def plot_mf_sims_8d(j=2, b=0, theta=theta, noise=0,
                    ini_cond=None, step_line_plot=500):
    t, x = solution_mf_sdo_euler(j=j, b=b, theta=theta, noise=noise,
                                 tau=1, time_end=10, dt=1e-3, ini_cond=ini_cond)
    fig, ax = plt.subplots(ncols=7, nrows=4, figsize=(19, 10))
    ax = ax.flatten()
    combs = list(itertools.combinations(np.arange(8), 2))
    n = 0
    colors = ['k', 'r']  # non-neighbor, neighbor,
    mat_diagonal_inside_cube = np.flipud(np.identity(theta.shape[0], dtype=int))
    for i, j in combs:
        color = colors[theta[i, j]]
        if mat_diagonal_inside_cube[i, j]:
            color = 'g'
        ax[n].plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
        ax[n].plot(x[:, i], x[:, j], color=color)
        # for x1, y1 in zip(x[500:-1:500, i], x[500:-1:500, j]):
        ax[n].plot(x[step_line_plot:-step_line_plot:step_line_plot, i],
                   x[step_line_plot:-step_line_plot:step_line_plot, j], color=color, linestyle='',
                   marker=(2, 0, 45),  # np.arctan(y1/x1)*180/np.pi
                   markersize=10)
        ax[n].plot(x[0, i], x[0, j], color=color,
                   marker='o')
        ax[n].plot(x[-1, i], x[-1, j], color=color,
                   marker='x')
        ax[n].set_xlabel('q_'+ str(i+1))
        ax[n].set_ylabel('q_'+str(j+1))
        ax[n].set_ylim(-0.05, 1.05)
        ax[n].set_xlim(-0.05, 1.05)
        n += 1
    legendelements = [Line2D([0], [0], color='k', lw=2, label='diagonal same face'),
                      Line2D([0], [0], color='g', lw=2, label='diagonal cube'),
                      Line2D([0], [0], color='r', lw=2, label='neighbors')]
    ax[0].legend(handles=legendelements, frameon=False)
    fig.tight_layout()


def calc_min_action_path_and_plot(j=0.5, b=0, noise=0.1, theta=theta, steps=20000,
                                  tol_stop=1e-2, numiters=500, dt=5e-3):
    t, x, xs, x_unstable, action_vals = minimum_action_path(j, b, noise, theta=theta, numiters=numiters, dt=dt,
                                                            steps=steps, tol_stop=tol_stop)
    J = j
    plt.figure()
    plt.title('J = ' + str(j) + ', B = ' + str(b))
    plt.plot(action_vals)
    plt.yscale('log')
    plt.ylabel('Action S')
    plt.xlabel('Step')
    plt.figure()
    plt.title('J = ' + str(j) + ', B = ' + str(b))
    for i in range(8):
        plt.plot(t, x[:, i].detach().numpy(), label='node' + str(i))
    plt.axhline(x_unstable[0], linestyle='--', color='k', alpha=0.5, label='Unstable FP')
    plt.legend(ncol=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Approximate posterior MAP')
    plt.figure()
    plt.title('J = ' + str(j) + ', B = ' + str(b))
    stash_on = np.linspace(0, int(np.sqrt(len(action_vals))), len(xs)).astype(np.int32) ** 2
    plot_idxs = stash_on[0:-1:3]
    idxs = np.arange(0, len(xs)+1, 3)
    colormap = pl.cm.Greens(np.linspace(0.1, 1, len(idxs)))
    for i in range(len(plot_idxs)):
        plt.plot(t, xs[idxs[i], :, 0], label=plot_idxs[i],
                 color=colormap[i])
    plt.legend(title='Grad. desc. step')
    plt.xlabel('Time (s)')
    plt.ylabel('Approximate posterior MAP')
    plt.ylim(-0.02, 1.02)
    plt.figure()
    plt.title('J = ' + str(j) + ', B = ' + str(b))
    stash_on = np.linspace(0, int(np.sqrt(len(action_vals))), len(xs)).astype(np.int32) ** 2
    plot_idxs = stash_on[0:-1:3]
    idxs = np.arange(0, len(xs)+1, 3)
    colormap = pl.cm.Greens(np.linspace(0.2, 1, len(idxs)))
    for i in range(len(plot_idxs)):
        plt.plot(xs[idxs[i], :, 1], xs[idxs[i], :, 0], label=plot_idxs[i],
                 color=colormap[i])
    plt.legend(title='Grad. desc. step')
    plt.xlabel('q_2(x=1)')
    plt.ylabel('q_1(x=1)')
    plt.ylim(-0.02, 1.02)
    ax = plt.figure().add_subplot(projection='3d')
    for j in range(2, 8):
        ax.plot(x[:, 0].detach().numpy(), x[:, 1].detach().numpy(), x[:, j].detach().numpy(),
                color='k')
    for j in range(2, 8):
        ax.plot(x[:, 0].detach().numpy(), x[:, j].detach().numpy(), x[:, 2].detach().numpy(),
                color='k')
    for j in range(2, 8):
        ax.plot(x[:, j].detach().numpy(), x[:, 1].detach().numpy(), x[:, 2].detach().numpy(),
                color='k')
    for j in range(2, 8):
        ax.plot(x[:, 5].detach().numpy(), x[:, 6].detach().numpy(), x[:, j].detach().numpy(),
                color='k')
    for j in range(2, 8):
        ax.plot(x[:, 5].detach().numpy(), x[:, j].detach().numpy(), x[:, 7].detach().numpy(),
                color='k')
    for j in range(2, 8):
        ax.plot(x[:, j].detach().numpy(), x[:, 6].detach().numpy(), x[:, 7].detach().numpy(),
                color='k')
    for j in range(2, 8):
        ax.plot(x[:, 3].detach().numpy(), x[:, 4].detach().numpy(), x[:, j].detach().numpy(),
                color='k')
    for j in range(2, 8):
        ax.plot(x[:, 3].detach().numpy(), x[:, j].detach().numpy(), x[:, 5].detach().numpy(),
                color='k')
    for j in range(2, 8):
        ax.plot(x[:, j].detach().numpy(), x[:, 4].detach().numpy(), x[:, 5].detach().numpy(),
                color='k')
    for j in range(2, 8):
        ax.plot(x[:, 7].detach().numpy(), x[:, 0].detach().numpy(), x[:, j].detach().numpy(),
                color='k')
    for j in range(2, 8):
        ax.plot(x[:, 7].detach().numpy(), x[:, j].detach().numpy(), x[:, 3].detach().numpy(),
                color='k')
    for j in range(2, 8):
        ax.plot(x[:, j].detach().numpy(), x[:, 0].detach().numpy(), x[:, 3].detach().numpy(),
                color='k')
    ax.set_xlabel(r'$q_i$')
    ax.set_ylabel(r'$q_j$')
    ax.set_zlabel(r'$q_k$')
    # ax.plot(x[:, 0].detach().numpy(), x[:, 1].detach().numpy(), x[:, 3].detach().numpy())
    # ax.plot(x[:, 0].detach().numpy(), x[:, 5].detach().numpy(), x[:, 4].detach().numpy())
    # ax.plot(x[:, 4].detach().numpy(), x[:, 6].detach().numpy(), x[:, 2].detach().numpy())
    # ax.plot(x[:, 5].detach().numpy(), x[:, 6].detach().numpy(), x[:, 7].detach().numpy())
    fig, ax = plt.subplots(ncols=7, nrows=4, figsize=(15, 10))
    fig2, ax2 = plt.subplots(1)
    ax = ax.flatten()
    combs = list(itertools.combinations(np.arange(8), 2))
    n = 0
    colors = ['k', 'r']  # non-neighbor, neighbor,
    mat_diagonal_inside_cube = np.flipud(np.identity(theta.shape[0], dtype=int))
    fig3, ax3 = plt.subplots(1)
    xv = np.arange(0, 1.1, 5e-2)
    yv = np.arange(0, 1.1, 5e-2)
    xx, yy = np.meshgrid(xv, yv)
    uu = gn.sigmoid(2*3*J*(2*yy-1)+2*b)-xx
    vv = gn.sigmoid(2*3*J*(2*xx-1)+2*b)-yy
    ax3.quiver(xx, yy, uu, vv)
    for i, j in combs:
        color = colors[theta[i, j]]
        if mat_diagonal_inside_cube[i, j]:
            color = 'g'
        ax[n].plot(x[:, i].detach().numpy(), x[:, j].detach().numpy(), color=color)
        ax[n].set_xlabel('q_'+ str(i+1))
        ax[n].set_ylabel('q_'+str(j+1))
        n += 1
        ax2.plot(x[:, i].detach().numpy(), x[:, j].detach().numpy(), color='k')
        if theta[i, j]:
            ax3.plot(x[:, i].detach().numpy(), x[:, j].detach().numpy(), color='k')
    ax2.set_xlabel('q_i')
    ax2.set_ylabel('q_j')
    ax3.set_xlabel('q_1')
    ax3.set_ylabel('q_3')
    legendelements = [Line2D([0], [0], color='k', lw=2, label='diagonal same face'),
                      Line2D([0], [0], color='g', lw=2, label='diagonal cube'),
                      Line2D([0], [0], color='r', lw=2, label='neighbors')]
    ax[0].legend(handles=legendelements, frameon=False)
    fig.tight_layout()
    # eigenvectors projections
    fig, ax = plt.subplots(ncols=2)
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    # mean_vals_1_array is 8 x n_jumps x timepoints
    eigvects = np.linalg.eig(theta)[1].T
    eigvals = np.round(np.linalg.eig(theta)[0], 4)
    eigvects_p = eigvects[(eigvals == 3) + (eigvals == 1)]
    eigvals_p = eigvals[(eigvals == 3) + (eigvals == 1)]
    eigvects_n = eigvects[(eigvals == -3) + (eigvals == -1)]
    eigvals_n = eigvals[(eigvals == -3) + (eigvals == -1)]
    steps_back = steps_front = 250
    projections_p = np.zeros((4, steps_back+steps_front))
    projections_n = np.zeros((4, steps_back+steps_front))
    for i in range(4):
        projections_p[i, :] = np.dot(x.detach().numpy(), eigvects_p[i])
        projections_n[i, :] = np.dot(x.detach().numpy(), eigvects_n[i])
    avg_proj_per_jump_p = projections_p
    avg_proj_per_jump_n = projections_n
    norm = np.sqrt(np.sum(projections_p[1:]**2, axis=0))
    for proj in range(4):
        ax[0].plot((np.arange(-steps_back, steps_front, 1))*dt, avg_proj_per_jump_p[proj, :],
                 label=eigvals_p[proj])
        ax[1].plot((np.arange(-steps_back, steps_front, 1))*dt, avg_proj_per_jump_n[proj, :],
                   label=eigvals_n[proj])
    f2, ax2 = plt.subplots(1)
    ax2.plot((np.arange(-steps_back, steps_front, 1))*dt, norm, color='k', label='Nrom')
    ax2.set_xlabel('Time from switch (s)')
    ax2.set_ylabel('Projection of activity - Norm')
    ax[0].legend(title='Eigenvalue')
    ax[1].legend(title='Eigenvalue')
    ax[0].set_xlabel('Time from switch (s)')
    ax[1].set_xlabel('Time from switch (s)')
    ax[0].set_ylabel('Projection of activity')
    ax[1].set_ylim(-0.6, 0.6)


def calc_potential_diff(j=0.5, b=0, noise=0.1, theta=theta, steps=20000,
                        tol_stop=1e-2, numiters=500, dt=5e-3):
    theta = np.array(((4, 2), (2, 4)))
    f_i_fun = f_i_diagonal
    ini_conds = [0.1, 0.9, 0.48]
    x_stable_1 = fsolve(f_i_fun, ini_conds[0], args=(j, b))
    x_stable_2 = fsolve(f_i_fun, ini_conds[1], args=(j, b))
    x_unstable = fsolve(f_i_fun, ini_conds[2], args=(j, b))
    slope = (x_stable_2-x_stable_1)/numiters
    # line from A to B with gaussian noise
    x0 = torch.rand((numiters, theta.shape[0]), requires_grad=True)*0.05 +\
        torch.arange(numiters, dtype=torch.float, requires_grad=True).reshape(-1, 1)*(slope) +\
            x_stable_1-x_stable_2
    x_stable_1 = np.repeat(x_stable_1, theta.shape[0])  # +np.random.randn(theta.shape[0])*1e-3
    x_stable_2 = np.repeat(x_stable_2, theta.shape[0])
    x_unstable = np.repeat(x_unstable, theta.shape[0])
    x0[0] = torch.tensor(x_stable_1)
    x0[-1] = torch.tensor(x_stable_2)
    theta = torch.tensor(theta * 1., dtype=torch.float)
    t, x, xs, action_vals = get_path_between(x0.clone(), j, b, noise, theta, steps=steps, step_size=1e-5, dt=dt,
                                             tol_stop=tol_stop, neg=0)


def pot_mf_illustration(j_list=[0.2, 0.4, 0.6], b_list=[-0.05, 0., 0.05]):
    q = np.arange(-0.2, 1.2, 1e-3)
    fig, ax = plt.subplots(ncols=2, figsize=(8, 3.6))
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['left'].set_visible(False)
        a.set_yticks([])
        a.set_xlabel('Approximate posterior q')
    ax[0].set_ylabel('Potential V(q)')
    colormap = pl.cm.Greens(np.linspace(0.4, 1, len(j_list)))
    for i_j, j in enumerate(j_list):
        pot = potential_mf(q, j, bias=0)
        ax[0].plot(q, pot-np.mean(pot), color=colormap[i_j], label=j)
    ax[0].legend(title='Coupling, J', frameon=False, labelspacing=0.1)
    colormap = pl.cm.Oranges(np.linspace(0.4, 1, len(b_list)))[::-1]
    for i_b, b in enumerate(b_list):
        pot = potential_mf(q, 0.5, bias=b)
        ax[1].plot(q, pot, color=colormap[i_b], label=b)
    ax[1].legend(title='Stim., B', frameon=False, labelspacing=0.1)
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'mf_potentials.png', dpi=400, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'mf_potentials.svg', dpi=400, bbox_inches='tight')


def potential_approx_2d(j, minval=-0.25, maxval=0.25):
    p1 = np.arange(minval, maxval, 1e-3)
    q1 = np.arange(minval, maxval, 1e-3)
    p, q = np.meshgrid(p1, q1)
    potential = - j*(p**2+q**2+q*p) + 4*j**3 * (q**3*p/3+p**2 * q**2 + 2/3*(q**4+p**4) + 1/3*p**3*q)
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(p+0.5, q+0.5, potential, alpha=0.4, rstride=8, cstride=8, lw=0.5, edgecolor='royalblue')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'Potential, $V(\vec{x})$')
    return potential


def plot_2d_mean_passage_time(J=2, B=0, sigma=0.1):
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    import matplotlib.pyplot as plt
    
    domain = [-0.5, 1.5, -0.5, 1.5]  # [x_min, x_max, y_min, y_max]
    nx, ny = 200, 200          # Number of grid points in x and y
    dx = (domain[1] - domain[0]) / (nx - 1)
    dy = (domain[3] - domain[2]) / (ny - 1)
    
    x_vals = np.linspace(domain[0], domain[1], nx)
    y_vals = np.linspace(domain[2], domain[3], ny)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Define F1(x, y) and F2(x, y)
    sigmoid = lambda x: 1 / (1+np.exp(-x))
    # def F1(x, y):
    #     return sigmoid(2 * J * (4 * x + 2 * y - 3) + 2 * B) - x
    
    # def F2(x, y):
    #     return F1(y, x)
    def F1(x, y):
        return sigmoid(2 * J * (6 * y - 3) + 2 * B) - x
    
    def F2(x, y):
        return F1(y, x)
    
    F1_grid = F1(X, Y)
    F2_grid = F2(X, Y)
    
    # Sparse matrix setup for the backward Kolmogorov equation
    n_points = nx * ny
    A = sp.lil_matrix((n_points, n_points))
    b = -np.ones(n_points)
    
    # Helper function to get the linear index of grid (i, j)
    def idx(i, j):
        return i * ny + j
    
    # Construct the finite-difference scheme
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            k = idx(i, j)
            # Central differences for spatial derivatives
            A[k, k] += -2 * (sigma**2 / dx**2 + sigma**2 / dy**2)
            A[k, idx(i+1, j)] += sigma**2 / dx**2 + F1_grid[i, j] / (2 * dx)
            A[k, idx(i-1, j)] += sigma**2 / dx**2 - F1_grid[i, j] / (2 * dx)
            A[k, idx(i, j+1)] += sigma**2 / dy**2 + F2_grid[i, j] / (2 * dy)
            A[k, idx(i, j-1)] += sigma**2 / dy**2 - F2_grid[i, j] / (2 * dy)
    
    # vals_tr = np.round(fsolve(f_i_both, [1, 1], args=(J, B)), 4)
    # vals_lr = np.round(fsolve(f_i_both, [1, 0], args=(J, B)), 4)
    # vals_tl = np.round(fsolve(f_i_both, [0, 1], args=(J, B)), 4)
    # vals_ll = np.round(fsolve(f_i_both, [0, 0], args=(J, B)), 4)
    # # Map attractors to the nearest grid points
    # attractors = [vals_tr, vals_lr, vals_tl, vals_ll]
    # attractor_indices = []
    # for x_a, y_a in attractors:
    #     i_a = np.argmin(np.abs(x_vals - x_a))
    #     j_a = np.argmin(np.abs(y_vals - y_a))
    #     # A[idx(i_a, j_a), idx(i_a, j_a)] = 1
    #     # b[idx(i_a, j_a)] = 0
    #     attractor_indices.append((i_a, j_a))
    
    # Apply boundary conditions (absorbing boundaries)
    for i in range(nx):
        for j in [0, ny-1]:  # Bottom and top
            A[idx(i, j), idx(i, j)] = 1.0
            b[idx(i, j)] = 0.0
    for j in range(ny):
        for i in [0, nx-1]:  # Left and right
            A[idx(i, j), idx(i, j)] = 1.0
            b[idx(i, j)] = 0.0
    
    
    # Convert to CSR format for efficient solving
    A = A.tocsr()
    
    # Solve the linear system
    T = spla.spsolve(A, b)
    
    # Reshape the solution to the grid
    T_grid = T.reshape((nx, ny))
    
    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, T_grid, cmap='Blues')
    vals = fsolve(f_i_both, [1, 1], args=(J, B))
    plt.plot(vals[0], vals[1], marker='o', color='r')
    vals = fsolve(f_i_both, [0, 0], args=(J, B))
    plt.plot(vals[0], vals[1], marker='o', color='r')
    vals = fsolve(f_i_both, [1, 0], args=(J, B))
    plt.plot(vals[0], vals[1], marker='o', color='r')
    vals = fsolve(f_i_both, [0, 1], args=(J, B))
    plt.plot(vals[0], vals[1], marker='o', color='r')
    plt.colorbar(label="Mean First Passage Time (T)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Mean First Passage Time for 2D Stochastic System")
    plt.show()
    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, T_grid, cmap='Blues')
    plt.colorbar(label="Mean First Passage Time (T)")
    x1 = np.arange(-0.5, 1.6, 1e-1)
    x2 = np.arange(-0.5, 1.6, 1e-1)
    x, y = np.meshgrid(x1, x2)
    u1 = f_i(x, y, j=J, b=B)
    u2 = f_i(y, x, j=J, b=B)
    plt.quiver(x, y, u1, u2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Mean First Passage Time")
    plt.show()


def example_dynamics_hierarchical_theta(theta=gn.THETA_HIER):
    # bifurcation happens at 1/\lambda_max ~ 0.212766
    j_list = [0.1, 0.1, 0.222]
    b_list = [0, 0.2, 0]
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(6, 4.5))
    ax = ax.flatten()
    i = 0
    times = [50, 50, 50]
    time_min = [0, 0, 0]
    dt_list = [1e-3, 1e-3, 1e-3]
    noise_list = [0.05, 0.05, 0.05]
    convlist = [True, True, False]
    tau = 0.008
    for j, b, t_end, dt, noise, t_min, conv in zip(j_list, b_list, times, dt_list,
                                              noise_list, time_min, convlist):
        time, vec, _ = \
            solution_mf_sdo_euler_OU_noise(j, b, theta=theta, noise=noise, tau=tau,
                                           time_end=t_end, dt=dt, tau_n=tau)
        vals = vec[time >= t_min, -1].T
        x = time[:len(vals)][::100]
        y = vals[::100]
        ax[i+3].axhline(0.5, color='k', alpha=0.4, linestyle='--')
        ax[i].axhline(0.5, color='k', alpha=0.4, linestyle='--')
        line = colored_line(x, y, y, ax[i], linewidth=2, cmap='coolwarm_r', 
                            norm=plt.Normalize(vmin=0,vmax=1))
        for it in range(8):
            ax[i+3].plot(time[10:][::300], vec[10:, it][::300])
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].set_ylim(0, 1)
        ax[i+3].set_ylim(0, 1)
        ax[i].set_xlim(0, t_end)
        ax[i+3].spines['right'].set_visible(False)
        ax[i+3].spines['top'].set_visible(False)
        i += 1
    ax[4].set_xlabel('Time (s)')
    ax[0].set_ylabel('Percept')
    ax[3].set_ylabel(r'$q_i(x=1)$')
    fig.tight_layout()
    ax_pos = ax[2].get_position()
    ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*1.13, ax_pos.y0+ax_pos.height*0.2,
                            ax_pos.width*0.1, ax_pos.height*0.7])
    fig.colorbar(line, cax=ax_cbar, pad=0.3, aspect=7.5).set_label(label=r'$q(x=1)$', size=14) # add a color legend
    fig.savefig(DATA_FOLDER + 'example_dynamics_hierarchical.png', dpi=300, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'example_dynamics_hierarchical.svg', dpi=300, bbox_inches='tight')


def eigenvalue_matrix_desc_asc(varlist=np.arange(0, 2.01, 0.01)):
    alist = dlist = varlist
    combs = list(itertools.product(alist, dlist))
    eigvals = np.zeros((len(varlist), len(varlist)))
    c = 0
    for a, d in combs:
        theta = gn.theta_hierarchical(a=a, d=d)
        jcrit = 1/np.max(np.real(np.linalg.eigvals(theta)))
        i, j = np.unravel_index(c, eigvals.shape)
        eigvals[i, j] = jcrit
        c += 1
    # eigvals_mat = np.array(eigvals).reshape(len(varlist), len(varlist))
    plt.figure()
    im = plt.imshow(eigvals, cmap='Oranges',
                    extent=[np.min(varlist), np.max(varlist), np.min(varlist),
                            np.max(varlist)], vmin=0)
    plt.colorbar(im, label=r'$J^*$')
    plt.xlabel('Descending loops')
    plt.ylabel('Ascending loops')


def bifurcation_hierarchical(b=0, varchange='descending'):
    if varchange == 'coupling':
        lab = 'Coupling, J'
        d = 1
        a = 1
        var_list=np.arange(0, 0.5, 5e-3)
    if varchange == 'ascending':
        lab = 'Ascending coupling'
        d = 1
        j = 0.2
        var_list=np.arange(0, 2, 0.01)
    if varchange == 'descending':
        lab = 'Descending coupling'
        a = 1
        j = 0.2
        var_list=np.arange(0, 2, 0.01)
    i0 = 0
    i1 = 1
    v0 = []
    v1 = []
    v_unstable = []
    for i_j, var in enumerate(var_list):
        if varchange == 'coupling':
            j = var
        if varchange == 'ascending':
            a = var
        if varchange == 'descending':
            d = var
        theta = gn.theta_hierarchical(a=a, d=d)
        vals_0 = mean_field_stim_matmul(j, 400, stim=b, sigma=0, theta=theta, val_init=i0, sxo=0.1)
        vals_1 = mean_field_stim_matmul(j, 400, stim=b, sigma=0, theta=theta, val_init=i1, sxo=0.1)
        v1.append(vals_1[-1][-1])
        # qb = find_repulsor(j=j, num_iter=50, q_i=0.0,
        #                    q_f=1, stim=b, threshold=1e-3,
        #                    theta=theta, neigh=8)
        qb = np.nan
        if np.abs(vals_1[-1][-1] - vals_0[-1][-1]) < 1e-7:
            v0.append(np.nan)
            v_unstable.append(np.nan)
        else:
            v0.append(vals_0[-1][-1])
            v_unstable.append(qb)
    plt.figure()
    plt.plot(var_list, v0, color='k')
    plt.plot(var_list, v1, color='k')
    plt.plot(var_list, v_unstable, color='k', linestyle='--')
    plt.xlabel(lab)
    plt.ylabel('Percept')


def bcrit(j_list=np.arange(0, 1, 1e-3), n=3.92):
    delta = np.sqrt(1-1/(j_list*n))
    b_crit1 = (np.log((1-delta)/(1+delta))+2*n*j_list*delta)/2
    b_crit2 = (np.log((1+delta)/(1-delta))-2*n*j_list*delta)/2
    plt.figure()
    plt.plot(b_crit1, j_list, color='k')
    plt.plot(b_crit2, j_list, color='k')
    plt.ylabel('J')
    plt.xlabel('B*')
    plt.ylim(0, 1)
    plt.xlim(-0.5, 0.5)
    plt.figure()
    plt.plot(j_list, b_crit1, color='k')
    plt.plot(j_list, b_crit2, color='k')
    plt.xlabel('J')
    plt.xlim(0, 1)
    plt.ylabel('B*')


def cp_vs_coupling_noise(j_list=np.arange(0, 0.6, 0.05), noise_list=[0.05, 0.1, 0.15, 0.2],
                         nsimuls=10000, load_sims=True, inset=True, cylinder=False, barplot=False):
    if cylinder:
        theta = get_regular_graph()
        add_theta = np.random.randn(theta.shape[0], theta.shape[1])*0.25
        theta = theta + add_theta
        lab_cylin = 'cylinder'
        j_list = np.arange(0, 0.5, 0.05)
        jcrit = 1/3.92
    else:
        lab_cylin = ''
        jcrit = 1/3
    if not load_sims:
        cp_matrix = np.zeros((len(j_list), len(noise_list)))
        for i_n, noise in enumerate(noise_list):
            for i_j, j in enumerate(j_list):
                print(str(round(i_j / len(j_list)*100, 2))+' %')
                mean_CP = choice_probability_mean_field(j=j, b=0, theta=theta,
                                                        noise=noise, tau=0.1,
                                                        time_end=1.5,
                                                        nsimuls=nsimuls, ou_noise=False)
                cp_matrix[i_j, i_n] = mean_CP
        np.save(DATA_FOLDER + f'mean_CP_vs_coupling_sigma{lab_cylin}_common_noise.npy', cp_matrix)
    else:
        cp_matrix = np.load(DATA_FOLDER + f'mean_CP_vs_coupling_sigma{lab_cylin}_common_noise.npy')
    if not barplot:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # colors = ['k', 'firebrick', 'forestgreen']
        colormap = pl.cm.Oranges(np.linspace(0.2, 1, len(noise_list)))[::-1]
        if not inset:
            ax.axhline(0.67, color='k', linestyle='--', alpha=1, zorder=1)
            ax.axhline(0.56, color='k', linestyle='--', alpha=0.4, zorder=1)
            if not cylinder:
                ax.text(0.42, 0.69, 'SFM (Dodd 2001)', fontsize=12)
                ax.text(0.42, 0.57, 'RDM (Britten 1996)', fontsize=12, alpha=0.4)
                ax.text(0.4, 0.75, 'Experiments (MT/V5)', fontsize=13)
            if cylinder:
                ax.text(0.32, 0.69, 'SFM (Dodd 2001)', fontsize=12)
                ax.text(0.32, 0.57, 'RDM (Britten 1996)', fontsize=12, alpha=0.4)
                ax.text(0.29, 0.75, 'Experiments (MT/V5)', fontsize=13)
        if len(noise_list) == 1:
            colormap = ['k']
        for i_n, noise in enumerate(noise_list):
            if len(noise_list) == 1:
                ax.plot(j_list, cp_matrix[:, i_n], color='gray', alpha=0.7,
                        linewidth=4, zorder=2)
            else:
                ax.plot(j_list, cp_matrix[:, i_n], color=colormap[i_n], alpha=0.7,
                        linewidth=4, zorder=2)
            ax.scatter(j_list[j_list <= jcrit], cp_matrix[j_list <= jcrit, i_n], color=colormap[i_n],
                    marker='o', label=noise, s=80,
                    facecolor='white', zorder=10)
            ax.scatter(j_list[j_list > jcrit], cp_matrix[j_list > jcrit, i_n], color=colormap[i_n],
                       marker='o', label=noise, s=80, zorder=10)
        legendelements = [Line2D([0], [0], color='k', lw=2, label='Monostable', marker='s', linestyle='',
                                 mfc='white', markersize=9),
                          Line2D([0], [0], color='k', lw=2, label='Bistable', linestyle='', marker='s', markersize=9)]
        ax.legend(handles=legendelements, frameon=False, ncol=1, fontsize=12, loc='upper left')
        if cylinder:
            ax.axvline(jcrit, linestyle=':', color='gray')
        else:
            ax.axvline(jcrit, linestyle=':', color='gray')
        # ax.text(0.25, 0.55, 'Monostable', rotation='vertical', color='gray')
        # ax.text(0.36, 0.55, 'Bistable', rotation='vertical', color='gray')
        if len(noise_list) > 1:
            ax.legend(title=r'$\sigma$', frameon=False)
        ax.set_xlabel('Coupling, J')
        ax.set_ylabel('Choice probability (CP)')
        ax.axhline(0.5, color='gray', linestyle=':')
        ax.set_ylim(0.45, 0.8)
        fig.tight_layout()
        if inset:
            pos = ax.get_position()
            ax2 = fig.add_axes([pos.x0+pos.width/1.3, pos.y0+pos.height/3, pos.width/3, pos.height/3])
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.plot(1, 0.67, '^', color='k', markersize=12)
            ax2.plot(0, 0.56, '^', color='k', mfc='none', markersize=12)
            ax2.set_ylim(0.49, 0.73)
            ax2.set_xlim(-0.8, 1.8)
            ax2.set_title('Experiments (V5/MT)', fontsize=12)
            # ax.plot(0.9, 0.67, '^', color='k', markersize=10)
            # ax.plot(0.8, 0.56, '^', color='k', mfc='none', markersize=10)
            # ax.text(0.6, 0.38, 'Data - Bistable', fontsize=12)
            # ax.text(0.6, 0.19, 'Data - Monostable', fontsize=12)
            ax2.text(-0.6, 0.6, 'Britten 1996', fontsize=10)
            ax2.text(0.4, 0.71, 'Dodd 2001', fontsize=10)
            ax2.set_xticks([0, 1], ['RDM', 'SFM'], fontsize=12, rotation=45)
            ax2.set_ylabel('CP')
        # ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 0.9], [0, 0.2, 0.4, 0.6, '', ''])
        # ax.text(0.47, 0.655, 'Data - Bistable', fontsize=12)
        # ax.text(0.47, 0.545, 'Data - Monostable', fontsize=12)
        fig.savefig(DATA_FOLDER + 'choice_probs_vs_coupling_and_noise_v2.png', dpi=400, bbox_inches='tight')
        fig.savefig(DATA_FOLDER + 'choice_probs_vs_coupling_and_noise_v2.svg', dpi=400, bbox_inches='tight')
    if barplot:
        fig, ax = plt.subplots(figsize=(3.5, 3.2))
        colormap = pl.cm.Greens(np.linspace(0.3, 1, 2))
        colormap = ['mediumvioletred', 'dimgrey', 'forestgreen']*2
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.text(0.02, 0.52, 'Experiments (MT/V5)\nWashmut et al. 2019', fontsize=12)
        vals_bar = [np.nanmean(cp_matrix[np.argmin(np.abs(j_list - 0.11)), 0]),
                    np.nanmean(cp_matrix[np.argmin(np.abs(j_list - 0.33)), 0])]
        types = ['RDM', 'SFM (B=0)']*2
        classes = ['Data']*2 + ['Model']*2
        vals_all = [0.56, 0.67] + vals_bar
        df = pd.DataFrame({
            "Type": types,
            "Classes": classes,
            "Choice probability": vals_all})
        
        # Define colors — one per Type
        pair_colors = ['dimgrey', 'dimgrey']
        ax.set_ylim(0.5, 0.75)
        plt.axhline(0.5, color='k', alpha=0.3, linestyle='--')
        # Draw bars
        bar = sns.barplot(
            data=df, x="Type", y="Choice probability", hue="Classes",
            palette=pair_colors, ax=ax, legend=False)
        ax.grid(False)
        ax.set_xlabel('')
        # Apply hatching to 'Model' bars only
        hatch_map = {'Data': '', 'Model': '///'}
        for container, class_name in zip(bar.containers, df['Classes'].unique()):
            hatch = hatch_map[class_name]
            for patch in container:
                patch.set_hatch(hatch)
        plt.xticks(rotation=30)
        # Beautify
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.tight_layout()
        fig.savefig(DATA_FOLDER + 'choice_probability_barplot.png', dpi=400, bbox_inches='tight')
        fig.savefig(DATA_FOLDER + 'choice_probability_barplot.svg', dpi=400, bbox_inches='tight')
        


def cp_vs_coupling_random_neurons(j_list=np.arange(0, 0.6, 0.05), rand_neur_list=[False, 2, 4, 8],
                                  nsimuls=1000, noise=0.15, load_sims=False):
    if not load_sims:
        cp_matrix = np.zeros((len(j_list), len(rand_neur_list)))
        for i_n, rand_neurons in enumerate(rand_neur_list):
            for i_j, j in enumerate(j_list):
                mean_CP = choice_probability_mean_field(j=j, b=0, theta=theta,
                                                        noise=noise, tau=0.1,
                                                        time_end=2,
                                                        nsimuls=nsimuls, ou_noise=False,
                                                        add_random_neurons=rand_neurons)
                cp_matrix[i_j, i_n] = mean_CP
        np.save(DATA_FOLDER + 'mean_CP_vs_coupling_random_neurons.npy', cp_matrix)
    else:
        cp_matrix = np.load(DATA_FOLDER + 'mean_CP_vs_coupling_random_neurons.npy')
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # colors = ['k', 'firebrick', 'forestgreen']
    colormap = pl.cm.Oranges(np.linspace(0.2, 1, len(rand_neur_list)))[::-1]
    if len(rand_neur_list) == 1:
        colormap = ['k']
    for i_n, noise in enumerate(rand_neur_list):
        ax.plot(j_list, cp_matrix[:, i_n], color=colormap[i_n],
                linewidth=3, marker='o', label=noise)
    ax.axvline(1/3, linestyle='--', color='gray')
    if len(rand_neur_list) > 1:
        ax.legend(title='# of random neurons', frameon=False)
    ax.set_xlabel('Coupling, J')
    # ax.set_ylabel('CPs')
    ax.axhline(0.5, color='gray', linestyle='--')
    fig.savefig(DATA_FOLDER + 'choice_probs_vs_coupling_and_number_of_random_neurons.png', dpi=400, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'choice_probs_vs_coupling_and_number_of_random_neurons.svg', dpi=400, bbox_inches='tight')


def compute_time_resolved_cp(X, y, time_axis=2, cv_splits=10):
    """
    Compute CP across time from neural activity: shape (trials, neurons, time)

    Returns:
    - cp_time: (n_timepoints,) CP at each time point
    """
    n_trials, n_neurons, n_timepoints = X.shape
    cp_time = np.zeros(n_timepoints)

    for t in range(n_timepoints):
        X_t = X[:, :, t]  # (n_trials, n_neurons) at time t
        cp_t, _ = compute_choice_probability(X_t, y, cv_splits=cv_splits)
        cp_time[t] = cp_t

    return cp_time


def plot_rsc_matrix_vs_b_list_and_coupling(b_list=np.arange(0, 1.02, 0.02),
                                           j_list=np.arange(0, 1.01, 0.02),
                                           nsims=50, load_data=True, sigma=0.1, long=True,
                                           cylinder=False, theta=theta, inset=True, barplot=False):
    if cylinder:
        theta = get_regular_graph(d=4, n=100)
        lab_cylin = 'cylinder'
        b_list=np.arange(0, 0.2, 0.1)
        # j_list=np.arange(0, 0.62, 0.02)
        # j_list=np.arange(0, 0.55, 0.02)
        j_list=np.arange(0, 0.42, 0.02)
    else:
        lab_cylin = ''
    if long:
        print('Long stim.')
        if cylinder:
            time_end = 20
        else:
            time_end = 20
        tau = 0.1
        label = ''
    else:
        print('Short stim.')
        time_end = 2
        tau = 0.05
        label = 'short'
    if load_data:
        rsc_matrix = np.load(DATA_FOLDER + f'rsc_matrix_{sigma}{label}{lab_cylin}.npy')
    else:
        rsc_matrix = np.zeros((len(j_list), len(b_list), nsims))
        for i_j, j in enumerate(j_list):
            print(str(round(i_j / len(j_list)*100, 2))+' %')
            for i_b, b in enumerate(b_list):
                for n in range(nsims):
                    rsc = local_correlation_across_time(j=j, b=b, theta=theta,
                                                        noise=sigma, tau=tau,
                                                        time_end=time_end, ou_noise=False,
                                                        cylinder=cylinder)
                    rsc_matrix[i_j, i_b, n] = rsc
        np.save(DATA_FOLDER + f'rsc_matrix_{sigma}{label}{lab_cylin}.npy', rsc_matrix)
    # fig, ax = plt.subplots(1)
    # im = ax.imshow(np.flipud(np.nanmean(rsc_matrix, axis=-1)), cmap='Reds', interpolation='gaussian',
    #                extent=[0, np.max(b_list), 0, np.max(j_list)], vmax=1)
    # plt.colorbar(im, ax=ax, label='<rSC>')
    # n = 3
    # delta = np.sqrt(1-1/(j_list*n))
    # b_crit1 = (np.log((1-delta)/(1+delta))+2*n*j_list*delta)/2
    # ax.plot(b_crit1, j_list, linewidth=3, color='k')
    # ax.set_ylabel('Coupling, J')
    # ax.set_xlim(0,  b_list[-1])
    # ax.set_xlabel('Sensory evidence, B')
    B_targets = [0.0, 0.1]
    B_indices = [np.argmin(np.abs(b_list - B)) for B in B_targets]
    if not barplot:
        
        # Plot correlation vs J for the selected B values
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        colormap = pl.cm.Greens(np.linspace(0.3, 1, 2))
        colormap = ['mediumvioletred', 'dimgrey', 'green']
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if not cylinder:
            ax.axvline(1/3, color='mediumblue', alpha=0.6, linestyle='--', linewidth=2)
        if cylinder:
            ax.axvline(1/3.92, color='mediumblue', alpha=0.6, linestyle='--', linewidth=2)
        p = 0
        if not inset:
            ax.axhline(0.42, color=colormap[0], linestyle='--', alpha=1, linewidth=3)
            ax.axhline(0.28, color=colormap[1], linestyle='--', alpha=1, linewidth=3)
            ax.axhline(0.23, color=colormap[2], linestyle='--', alpha=0.7, linewidth=3)
            ax.text(0.02, 0.44, 'SFM (B=0)', fontsize=12, color=colormap[0])
            ax.text(0.02, 0.30, 'SFM (B>0)', fontsize=12, color=colormap[1])
            ax.text(0.02, 0.16, 'RDM', fontsize=12, color=colormap[2])
            ax.text(0.02, 0.52, 'Experiments (MT/V5)\nWashmut et al. 2019', fontsize=12)
        colors_lines = ['mediumblue', 'lightskyblue']
        for B_val, idxB in zip(B_targets, B_indices):
            plt.plot(j_list[j_list < 0.6], np.nanmean(rsc_matrix, axis=-1)[:, idxB][j_list < 0.6], label=f'{B_val}',
                     color=colors_lines[p], linewidth=4)
            p += 1
        plt.xlabel('Coupling, J')
        plt.legend(frameon=False, title='B', loc='lower right')
        # ax.text(0.25, 0.02, 'Monostable', rotation='vertical', color='gray')
        # ax.text(0.36, 0.02, 'Bistable', rotation='vertical', color='gray')
        plt.ylabel('Interneuronal correlation (IC)')
        if not long:
            plt.ylim(-0.05, 0.7)
        if long:
            plt.ylim(-0.05, 0.7)
        fig.tight_layout()
        if inset:
            pos = ax.get_position()
            ax2 = fig.add_axes([pos.x0+pos.width/1.4, pos.y0+pos.height/1.6, pos.width/3, pos.height/3])
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.set_ylim(0.15, 0.45)
            ax2.set_xlim(-0.3, 2.3)
            ax2.set_title('Experiments (V5/MT)\n Washmut 2019', fontsize=13)
            ax2.plot(2, 0.42, '^', color=colormap[0], markersize=10)
            ax2.plot(1, 0.28, '^', color=colormap[1], markersize=10)
            ax2.plot(0, 0.23, '^', color='k', mfc='none', markersize=10)
            ax2.set_xticks([0, 1, 2], ['RDM', 'SFM (B>0)', 'SFM (B=0)'], rotation=45, fontsize=12)
            ax2.set_ylabel('IC')
        # ax.text(0.6, 0.38, 'Data - Bistable', fontsize=12)
        # ax.text(0.6, 0.19, 'Data - Monostable', fontsize=12)
        # ax2.set_xticks([0, 1], ['Monostable', 'Bistable'], fontsize=12, rotation=45)
        fig.savefig(DATA_FOLDER + 'rsc_vs_coupling_B_simuls.png', dpi=400, bbox_inches='tight')
        fig.savefig(DATA_FOLDER + 'rsc_vs_coupling_B_simuls.svg', dpi=400, bbox_inches='tight')
    if barplot:
        fig, ax = plt.subplots(figsize=(3.5, 3.2))
        colormap = pl.cm.Greens(np.linspace(0.3, 1, 2))
        colormap = ['mediumvioletred', 'dimgrey', 'forestgreen']*2
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.text(0.02, 0.52, 'Experiments (MT/V5)\nWashmut et al. 2019', fontsize=12)
        vals_bar = [np.nanmean(rsc_matrix, axis=-1)[:, 0][np.argmin(np.abs(j_list - 0.1))]]
        for B_val, idxB in zip(B_targets[::-1], B_indices[::-1]):
            val = np.nanmean(rsc_matrix, axis=-1)[:, idxB][np.argmin(np.abs(j_list - 0.33))]
            vals_bar.append(val)
        types = ['RDM', 'SFM (B>0)', 'SFM (B=0)']*2
        classes = ['Data']*3 + ['Model']*3
        vals_all = [0.23, 0.28, 0.42] + vals_bar
        df = pd.DataFrame({
            "Type": types,
            "Classes": classes,
            "Interneuronal correlation": vals_all})
        
        # Define colors — one per Type
        pair_colors = ['dimgrey', 'dimgrey']
        ax.set_ylim(0, 0.7)
        # Draw bars
        bar = sns.barplot(
            data=df, x="Type", y="Interneuronal correlation", hue="Classes",
            palette=pair_colors, ax=ax)
        ax.grid(False)
        ax.set_xlabel('')
        # Apply hatching to 'Model' bars only
        hatch_map = {'Data': '', 'Model': '///'}
        for container, class_name in zip(bar.containers, df['Classes'].unique()):
            hatch = hatch_map[class_name]
            for patch in container:
                patch.set_hatch(hatch)
        plt.xticks(rotation=30)
        # Beautify
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(title='', loc='best', frameon=False)
        
        # Make sure legend shows hatches too
        for legend_patch, class_name in zip(ax.legend_.get_patches(), df['Classes'].unique()):
            legend_patch.set_hatch(hatch_map[class_name])
        fig.tight_layout()
        fig.savefig(DATA_FOLDER + 'interneuronal_correlation_barplot.png', dpi=400, bbox_inches='tight')
        fig.savefig(DATA_FOLDER + 'interneuronal_correlation_barplot.svg', dpi=400, bbox_inches='tight')


def plot_example_correlation(j0=0.1, j1=0.33, t_dur=0.5, dt=1e-3, sigma=0.1,
                             shift=0, jump=20):
    np.random.seed(100)
    from scipy.stats import zscore
    time = np.arange(0, t_dur+dt, dt)
    theta = get_regular_graph() + np.random.randn()*0.0
    vec1 = np.random.rand(theta.shape[0])
    vec2 = np.random.rand(theta.shape[0])
    vec1_arr = np.zeros((theta.shape[0], len(time)))
    noise1 = np.random.randn(len(time)+shift, theta.shape[0])
    noise2 = np.random.randn(len(time)+shift, theta.shape[0])
    vec2_arr = np.zeros_like(vec1_arr)
    for t in range(len(time)+shift):
        vec1 = vec1 + dt*(gn.sigmoid(2*j1*np.matmul(theta, 2*vec1-1)+2*0.1*np.random.rand())-vec1)/0.1 + sigma*np.sqrt(dt/0.1)*noise1[t]
        vec2 = vec2 + dt*(gn.sigmoid(2*j0*np.matmul(theta, 2*vec2-1)+2*0.1*np.random.rand())-vec2)/0.1 + sigma*np.sqrt(dt/0.1)*noise2[t]
        if t >= shift:
            vec1_arr[:, t-shift] = vec1
            vec2_arr[:, t-shift] = vec2
    vec2_arr = vec2_arr[::jump], vec1_arr = vec1_arr[::jump]
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(5, 4.5))
    colormap = ['mediumvioletred', 'forestgreen']
    ax[0, 1].plot(time, zscore(np.matmul(theta, vec1_arr)[0]), color=colormap[0], linewidth=3)
    ax[0, 1].plot(time, zscore(vec1_arr[0]), color=colormap[1], linewidth=3)
    ax[0, 0].plot(time, zscore(np.matmul(theta, vec2_arr)[0]), color=colormap[0], linewidth=3,
                  label='Neighbors')
    ax[0, 0].plot(time, zscore(vec2_arr[0]), color=colormap[1], linewidth=3,
                  label='Neuron i')
    ax[0, 0].legend(frameon=False)
    ax[1, 1].plot(zscore(vec1_arr[0]), zscore(np.matmul(theta, vec1_arr)[0]),
                  color='k', linestyle='', marker='o', markersize=3)
    ax[1, 0].plot(zscore(vec2_arr[0]), zscore(np.matmul(theta, vec2_arr)[0]),
                  color='k', linestyle='', marker='o', markersize=3)
    corr1 = round(np.corrcoef(zscore(vec1_arr[0]), zscore(np.matmul(theta, vec1_arr)[0]))[0][1], 3)
    corr2 = round(np.corrcoef(zscore(vec2_arr[0]), zscore(np.matmul(theta, vec2_arr)[0]))[0][1], 3)
    corrs = [corr1, corr2]
    for i_a, a in enumerate([ax[1, 1], ax[1, 0]]):
        a.plot([-3, 3], [-3, 3], color='gray', linestyle='--', alpha=0.7, linewidth=4)
        a.text(1.2, -1.8, rf'$\rho = $ {corrs[i_a]}', fontsize=12)
        a.set_xlabel('Single unit')
    ax[0, 0].set_ylabel('Activity')
    ax[0, 0].set_xlabel('Time')
    ax[0, 1].set_xlabel('Time')
    ax[1, 0].set_ylabel('Neighbors')
    for a in ax.flatten():
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_yticks([])
        a.set_xticks([])
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'interneuronal_correlation_cartoon_example_v2.png', dpi=400, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'interneuronal_correlation_cartoon_example_v2.svg', dpi=400, bbox_inches='tight')


def analytical_correlation_rsc(sigma=0.15, theta=theta):
    from scipy.linalg import solve_continuous_lyapunov
    def find_fixed_point(J, B, theta, max_iter=500, tol=1e-9):
        n = theta.shape[0]
        x = np.full(n, 0.9)
        for _ in range(max_iter):
            z = 2*J * theta @ (2*x - 1) + 2*B
            x_new = gn.sigmoid(z)
            if np.linalg.norm(x_new - x) < tol:
                return x_new
            x = x_new
        raise RuntimeError("Fixed point did not converge")


    def compute_correlation(J, B, theta, sigma=0.1, i=0):
        n = theta.shape[0]
        x_bar = find_fixed_point(J, B, theta)
        D = np.diag(x_bar * (1 - x_bar))
        A = 4 * J * D @ theta - np.eye(n)
        Q = sigma**2 * np.eye(n)
        C = solve_continuous_lyapunov(A, -Q)
        # C = -scipy.linalg.inv(A)*sigma**2/ 2
        var_i = C[i, i]
        cov_i_sum_others = (np.sum(C[i, :]) - C[i, i])
        var_sum_others = (np.sum(C) - np.sum(C[i, :]) - np.sum(C[:, i]) + C[i, i])
        corr = cov_i_sum_others / np.sqrt(var_i * var_sum_others)
        return corr

    # Parameters
    i = 0
    
    J_values = np.arange(0, 1, 0.01)
    B_values = np.arange(0, 1, 0.01)
    
    corr_grid = np.zeros((len(J_values), len(B_values)))
    for idxJ, J in enumerate(J_values):
        for idxB, B in enumerate(B_values):
            try:
                corr_grid[idxJ, idxB] = compute_correlation(J, B, theta, sigma=sigma, i=i)
            except RuntimeError:
                corr_grid[idxJ, idxB] = np.nan

    # Plot heatmap of correlation vs J and B
    fig, ax = plt.subplots(figsize=(8,6))
    im = plt.imshow(corr_grid, origin='lower', aspect='auto', 
                    extent=[B_values[0], B_values[-1], J_values[0], J_values[-1]],
                    cmap='Reds')
    plt.colorbar(im, label='Correlation ρ')
    plt.xlabel('Bias B')
    plt.ylabel('Coupling J')
    n = 3
    j_list = J_values
    delta = np.sqrt(1-1/(j_list*n))
    b_crit1 = (np.log((1-delta)/(1+delta))+2*n*j_list*delta)/2
    plt.plot(b_crit1, j_list, linewidth=3, color='k')
    plt.tight_layout()
    plt.xlim(0,  B_values[-1])
    plt.show()

    # Indices for B = 0, 0.1, 0.2
    B_targets = [0.0, 0.1, 0.2]
    B_indices = [np.argmin(np.abs(B_values - B)) for B in B_targets]

    # Plot correlation vs J for the selected B values
    fig, ax = plt.subplots(figsize=(5, 4))
    colormap = pl.cm.Greens(np.linspace(0.3, 1, 3))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axvline(1/3, color=colormap[0], alpha=0.9, linestyle='--', linewidth=3)
    p = 0
    for B_val, idxB in zip(B_targets, B_indices):
        plt.plot(J_values, corr_grid[:, idxB], label=f'B = {B_val}',
                 color=colormap[p], linewidth=4)
        p += 1
    plt.xlabel('Coupling J')
    plt.ylabel('Correlation ρ')
    # ax.text(0.25, 0.05, 'Monostable', rotation='vertical', color='gray', alpha=0.9)
    # ax.text(0.36, 0.05, 'Bistable', rotation='vertical', color='gray', alpha=0.9)
    plt.ylim(0, 1)
    plt.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'rsc_vs_coupling_B_analytical.png', dpi=400, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'rsc_vs_coupling_B_analytical.svg', dpi=400, bbox_inches='tight')


def plot_rsc_vs_b_list_and_coupling(b_list=[0, 0.025, 0.05, 0.075],
                                    j_list=[0.1, 0.4],
                                    nsims=500):
    rsc_matrix = np.zeros((len(j_list), len(b_list), nsims))
    for i_j, j in enumerate(j_list):
        for i_b, b in enumerate(b_list):
            for n in range(nsims):
                rsc = local_correlation_across_time(j=j, b=b, theta=theta,
                                                    noise=0.1, tau=0.1,
                                                    time_end=20, ou_noise=False)
                rsc_matrix[i_j, i_b, n] = rsc
    fig, ax = plt.subplots(ncols=len(j_list), figsize=(4*len(j_list), 4))
    ax_iter = ax if len(j_list) > 1 else [ax]
    for i_ax, a in enumerate(ax_iter):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        if len(b_list) > 10:
            mean_rsc_mat =  np.nanmean(rsc_matrix[i_ax], axis=1)
            err_rsc_mat =  np.nanstd(rsc_matrix[i_ax], axis=1)
            a.plot(b_list, mean_rsc_mat, color='k', linewidth=4)
            a.fill_between(b_list, mean_rsc_mat-err_rsc_mat, mean_rsc_mat+err_rsc_mat,
                           color='k', alpha=0.1)
        else:
            sns.violinplot(rsc_matrix[i_ax].T, ax=a, palette='coolwarm')
            a.set_xticks(np.arange(len(b_list)), b_list)
        a.set_ylim(0., 1.1)
        a.set_xlabel('Disparity (B)')
        a.set_title(f'J = {j_list[i_ax]}')
    if len(j_list) > 1:
        ax[0].set_ylabel('rSC')
    else:
        ax.set_ylabel('rSC')
    fig.tight_layout()
    if len(b_list) > 10:
        fig.savefig(DATA_FOLDER + 'rSC_correlations_vs_disparity.png', dpi=400, bbox_inches='tight')
        fig.savefig(DATA_FOLDER + 'rSC_correlations_vs_disparity.svg', dpi=400, bbox_inches='tight')
    else:
        fig.savefig(DATA_FOLDER + 'rSC_correlations_vs_disparity_mono_bistable_stim.png', dpi=400, bbox_inches='tight')
        fig.savefig(DATA_FOLDER + 'rSC_correlations_vs_disparity_mono_bistable_stim.svg', dpi=400, bbox_inches='tight')


def local_correlation_across_time(j=0.5, b=0, theta=theta,
                                  noise=0.15, tau=0.1, time_end=20,
                                  ou_noise=True, cylinder=False):
    if ou_noise:
        time, x_vec, _ = solution_mf_sdo_euler_OU_noise(j, b, theta, noise, tau,
                                                        time_end=time_end, dt=1e-2)
    else:
        time, x_vec = solution_mf_sdo_euler(j, b, theta, noise, tau,
                                            time_end=time_end, dt=1e-2)
    correlation_array = np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        su_activity = x_vec[:, i]
        if not cylinder:  # MU: all neurons except SU
            mu_activity = x_vec[:, np.arange(theta.shape[0]) != i].sum(axis=1)  # [:, np.arange(theta.shape[0]) != i]
        if cylinder:  # MU: all neighbors of SU
            mu_activity = np.matmul(x_vec, theta).T
            # mu_activity = 
        correlation_array[i] = np.corrcoef(su_activity, mu_activity)[0][1]
    # counts, vals = np.histogram(correlation_array, 20)
    # overall_corr = vals[np.argmax(counts)]
    return np.median(correlation_array)


def get_analytical_correlation(theta=theta):
    # from scipy.linalg import eigh
    # Parameters
    j = 0.4
    sigma = 0.15
    b = 0.
    N = 8
    q = lambda q: gn.sigmoid(2*N*j*(2*q-1)+ b*2) - q
    x_bar = np.clip(fsolve(q, 0.9), 0, 1)
    x_bar = np.eye(N)*x_bar

    # Linearization
    a = 4 * j * x_bar * (1 - x_bar)
    A = np.matmul(a, theta) - np.eye(N)
    Ainv = np.linalg.inv(A)
    # Solve Lyapunov equation: AC + CAᵀ = σ² I
    # Since A is symmetric, use C = (σ²/2) * A⁻¹
    C = Ainv*(sigma**2 / 2)

    v = np.diag(C)[0]
    c = (np.sum(C) - np.trace(C)) / (N * (N - 1))  # average off-diagonal
    
    # Correlation
    rho = 7 * c / np.sqrt(v * (7 * v + 42 * c))
    print("Correlation ρ:", rho)


def choice_probability_mean_field(j=0.5, b=0, theta=theta,
                                  noise=0.05, tau=0.1, time_end=20,
                                  nsimuls=1000, ou_noise=True,
                                  add_random_neurons=False):
    from sklearn.metrics import roc_auc_score
    X = np.zeros((nsimuls, theta.shape[0]+add_random_neurons))
    y = np.zeros(nsimuls)
    # cps = np.zeros(nsimuls)
    rates_all = np.zeros((theta.shape[0], int(time_end/1e-2)+1, nsimuls))
    for n in range(nsimuls):
        if ou_noise:
            time, x_vec, _ = solution_mf_sdo_euler_OU_noise(j, b, theta, noise, tau,
                                                            time_end=time_end, dt=1e-2)
        else:
            time, x_vec = solution_mf_sdo_euler(j, b, theta, noise, tau,
                                                time_end=time_end, dt=1e-2)
        choice = (np.sign(np.nanmean(x_vec, axis=1)-0.5)+1)[-1]/2
        if add_random_neurons:
            x_vec = np.column_stack((x_vec, 0.5+np.random.randn(x_vec.shape[0], add_random_neurons)*noise))
        # r_vec = x_vec*(1+np.random.randn(theta.shape[0])) + np.random.rand(x_vec.shape[0], x_vec.shape[1])*0.5
        rates_all[:, :, n] = x_vec.T
        # X[n] = np.mean(x_vec, axis=0)
        y[n] = choice
    CPs = []
    for i in range(theta.shape[0]):
        r = rates_all[i, :, :]  # shape: (T, K)
        r_mean = np.mean(r, axis=0)  # average over time: shape (K,)

        cp = roc_auc_score(y, r_mean)
        CPs.append(cp)
    # mean_CP, _ = compute_choice_probability(X, y, cv_splits=5, random_state=0)
    return np.nanmean(CPs)


def compute_choice_probability(X, y, cv_splits=5, random_state=0):
    """
    Compute choice probability (CP) from neural activity.

    Parameters:
    - X: (n_trials, n_neurons) numpy array of neural activity
    - y: (n_trials,) numpy array of binary choices (0 or 1)
    - cv_splits: Number of cross-validation folds
    - random_state: Random seed for reproducibility

    Returns:
    - mean_cp: Choice probability (area under ROC curve)
    - all_cps: CP from each fold
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    cps = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(solver='liblinear')  # Use linear decoder
        clf.fit(X_train, y_train)

        s_test = clf.decision_function(X_test)  # Linear projection
        cp = roc_auc_score(y_test, s_test)
        cps.append(cp)

    return np.mean(cps), cps


def coordinate_ascent_cartoon():
    # Define the objective function
    def f(x, y):
        return (x - 2)**2 + 1.1*(y - 2)**2 + x*2
    
    def grad_f(x, y):
        df_dx = 2 * (x - 2) + 2
        df_dy = 2 * (y - 2) * 1.1
        return df_dx, df_dy

    # Coordinate descent parameters
    x0, y0 = 1, 1.6
    step_size = 0.3
    n_steps = 7
    
    # Store path
    path = [(x0, y0)]
    x, y = x0, y0
    
    for i in range(n_steps):
        df_dx, df_dy = grad_f(x, y)
    
        if i % 2 == 0:
            # Step in x-direction
            x -= step_size * df_dx
        else:
            # Step in y-direction
            y -= step_size * df_dy
    
        path.append((x, y))
    
    # Extract path
    xs, ys = zip(*path)
    
    # Grid for contour plot
    xgrid = np.linspace(0.8, 2.2, 400)
    ygrid = np.linspace(1.4, 2.6, 400)
    X, Y = np.meshgrid(xgrid, ygrid)
    Z = f(X, Y)
    
    # Plot
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    contours = plt.contour(X, Y, Z, levels=10, cmap='binary_r')
    # plt.clabel(contours, inline=True, fontsize=8)
    
    # Plot path
    plt.plot(xs, ys, 'ro-', label='Coordinate Descent Path')
    # plt.scatter([xs[0]], [ys[0]], color='green', s=80, label='Start')
    # plt.scatter([xs[-1]], [ys[-1]], color='red', s=80, label='End')
    
    # Final touches
    plt.xlabel(r"$q_i$")
    plt.ylabel(r"$q_j$")
    plt.yticks([])
    plt.xticks([])
    # plt.legend()
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'cartoon_descent.png', dpi=400)


def cartoon_j_velocity_q():
    jl=[0.2, 0.4, 0.6]
    colormap = pl.cm.Blues(np.linspace(0.2, 1, 3))
    fig, ax = plt.subplots(1, figsize=(4, 3.5))
    ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
    ax.plot([0, 1], [0, 1], color='gray', linewidth=3, alpha=1, linestyle='--')
    ax.set_xlabel(r'$q$')
    ax.set_ylabel(r'$f(q)$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.05, 1.05)
    q = np.arange(0, 1, 1e-3)
    for i_j, j in enumerate(jl):
        ax.plot(q, gn.sigmoid(2*3*j*(2*q-1)),
    color=colormap[i_j], linewidth=2, label=j)
    ax.legend(frameon=False, title='J')
    fig.tight_layout()


def cartoon_b_velocity_q():
    bl=[0., 0.1, 0.2]
    colormap = pl.cm.Reds(np.linspace(0.2, 1, 3))
    fig, ax = plt.subplots(1, figsize=(4, 3.5))
    ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
    ax.plot([0, 1], [0, 1], color='gray', linewidth=3, alpha=1, linestyle='--')
    ax.set_xlabel(r'$q$')
    ax.set_ylabel(r'$f(q)$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.05, 1.05)
    q = np.arange(0, 1, 1e-3)
    j = 0.2
    for i_b, b in enumerate(bl):
        ax.plot(q, gn.sigmoid(2*3*j*(2*q-1)+2*b),
    color=colormap[i_b], linewidth=2, label=b)
    ax.legend(frameon=False, title='B')
    fig.tight_layout()


def get_regular_graph(d=4, n=100):
    G = gn.nx.random_regular_graph(d, n)
    A = gn.nx.to_numpy_array(G, dtype=int)
    return A


def pca_cylinder_predictions(j=0.29, b=0, noise=0.08, tau=0.1, dt=0.01):
    theta = get_regular_graph()
    t, vec = solution_mf_sdo_euler(j, b, theta, noise, tau, time_end=1000,
                                   dt=dt, ini_cond=None)
    pca_vals = PCA(n_components=2)
    # pca_vals.explained_variance_ratio_
    components = pca_vals.fit_transform(vec)
    plt.figure()
    plt.plot(t[::50], components[::50, 0] / np.max(components[::50, 0]))
    plt.plot(t[::50], np.nanmean(vec[::50], axis=1)/np.max(vec))
    fig, ax = plt.subplots(ncols=1, figsize=(6, 5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(components[200:, 0][::10], components[200:, 1][::10], color='k')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    fig.tight_layout()


def neuron_activity_cylinder_predictions(j=0.29, b=0, noise=0.08, tau=0.1, dt=0.01):
    theta = get_regular_graph()
    theta = theta + np.random.randn(theta.shape[0], theta.shape[1])*0.05
    # theta = (theta + theta.T)/2
    t, vec = solution_mf_sdo_euler(j, b, theta, noise, tau, time_end=1000,
                                   dt=dt, ini_cond=None)
    idxs = np.random.choice(np.arange(theta.shape[0]), 2, replace=False)
    components = vec[:, idxs]
    plt.figure()
    plt.plot(t[::50], components[::50, 0] / np.max(components[::50, 0]))
    plt.plot(t[::50], np.nanmean(vec[::50], axis=1)/np.max(vec))
    fig, ax = plt.subplots(ncols=1, figsize=(6, 5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(components[200:, 0][::5], components[200:, 1][::5], color='k')
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    fig.tight_layout()


def matrix_permutation_to_get_blockwise(M):
    # Column permutation: normalization column first (26), then params 0 to 25
    col_perm = [26] + list(range(26))
    
    # Then rows corresponding to each parameter index 0..25, i.e. configuration rows
    # where only that parameter is 1 (we find them by scanning)
    
    param_rows = []
    for param_idx in range(26):
        for r in range(M.shape[0]):
            # Row has a 1 in param_idx column, zero in normalization column
            if M[r, param_idx] == 1 and M[r, 26] == -1:
                param_rows.append(r)
                break
    
    # Compose row permutation: normalization row first, then param rows
    row_perm = [0] + param_rows
    
    # Permute M
    M_perm = M[np.ix_(row_perm, col_perm)]
    return M_perm, row_perm


def matrix_ring():
    # State encoding: x=0 (reference), y=1, z=2
    # Map states to indices
    state_to_idx = {'x':0, 'y':1, 'z':2}
    
    # Number of configurations
    n_states = 3
    n_config = n_states ** 3  # 27
    
    # Parameter indices as per your table:
    param_indices = {
        'singleton': {
            'v_i-1': { 'y': 0, 'z': 1 },
            'v_i':   { 'y': 2, 'z': 3 },
            'v_i+1': { 'y': 4, 'z': 5 }
        },
        'pairwise': {
            '(v_i-1,v_i)': { ('y','y'): 6,  ('y','z'): 7,  ('z','y'): 8,  ('z','z'): 9 },
            '(v_i,v_i+1)': { ('y','y'): 10, ('y','z'): 11, ('z','y'): 12, ('z','z'): 13 },
            '(v_i-1,v_i+1)': { ('y','y'): 14, ('y','z'): 15, ('z','y'): 16, ('z','z'): 17 }
        },
        'triplet': {  # All 8 triplet combos in {y,z}^3, lex order:
            ('y','y','y'): 18, ('y','y','z'): 19, ('y','z','y'): 20, ('y','z','z'): 21,
            ('z','y','y'): 22, ('z','y','z'): 23, ('z','z','y'): 24, ('z','z','z'): 25
        },
        'normalization': 26
    }
    
    # Enumerate all configurations in lexicographic order:
    # v_{i-1} varies fastest, then v_i, then v_{i+1}
    configs = list(itertools.product(['x','y','z'], repeat=3))  # tuples (v_{i-1}, v_i, v_{i+1})
    
    # Initialize design matrix M (27 rows × 27 params)
    M = np.zeros((n_config, 27), dtype=float)
    
    for row_idx, (v_im1, v_i, v_ip1) in enumerate(configs):
        # 1) Singleton terms
        for var, state in zip(['v_i-1', 'v_i', 'v_i+1'], [v_im1, v_i, v_ip1]):
            if state in ['y', 'z']:
                col_idx = param_indices['singleton'][var][state]
                M[row_idx, col_idx] = 1
        
        # 2) Pairwise terms
        pairs = [
            ('(v_i-1,v_i)', (v_im1, v_i)),
            ('(v_i,v_i+1)', (v_i, v_ip1)),
            ('(v_i-1,v_i+1)', (v_im1, v_ip1))
        ]
        for pair_name, (s1, s2) in pairs:
            if s1 in ['y','z'] and s2 in ['y','z']:
                col_idx = param_indices['pairwise'][pair_name][(s1, s2)]
                M[row_idx, col_idx] = 1
    
        # 3) Triplet terms
        triplet = (v_im1, v_i, v_ip1)
        if all(s in ['y','z'] for s in triplet):
            col_idx = param_indices['triplet'][triplet]
            M[row_idx, col_idx] = 1
    
        # 4) Normalization constant
        M[row_idx, param_indices['normalization']] = -1

    # M is now your design matrix mapping params -> log p(s|v)
    # Given vector log p(s|v), you solve: x = M^{-1} @ log_p
    M_perm, row_permutation = matrix_permutation_to_get_blockwise(M)
    configs_perm = tuple(map(tuple, np.array(configs)[row_permutation]))
    M_inv = np.linalg.inv(M_perm)
    plt.figure()
    plt.imshow(M_perm, cmap='bwr')
    plt.title(r'$M$')
    plt.figure()
    plt.imshow(M_inv, cmap='bwr')
    plt.title(r'$M^{-1}$')


def compute_M_ising():
    # 2 variables, each with 2 states: 0 (reference), 1
    states = [0, 1]
    M_2vars = np.zeros((4, 4))  # 2 singleton terms, 1 pairwise term, 1 normalization term
    
    def singleton_idx_2v(pos):  # pos in [0,1]
        return pos  # index 0 or 1
    
    def pairwise_idx_2v():
        return 2  # fixed index for pairwise term
    
    # Build matrix
    for row_idx, (s1, s2) in enumerate([(0, 0), (1, 1), (0, 1), (1, 0)]):
        if s1 != 0:
            M_2vars[row_idx, singleton_idx_2v(0)] = 1
        if s2 != 0:
            M_2vars[row_idx, singleton_idx_2v(1)] = 1
        if s1 != 0 and s2 != 0:
            M_2vars[row_idx, pairwise_idx_2v()] = 1
        M_2vars[row_idx, 3] = -1  # log-normalizer term

    # M_2vars = np.array([[1, 1, 1, -1],
    #                     [-1, -1, 1, -1],
    #                     [1, -1, -1, -1],
    #                     [-1, 1, -1, -1]])
    M_2vars_inv = np.linalg.pinv(M_2vars)
    plt.figure()
    plt.imshow(M_2vars, cmap='bwr')
    plt.title(r'$M$')
    plt.figure()
    plt.imshow(M_2vars_inv, cmap='bwr')
    plt.title(r'$M^{-1}$')


def compute_lyapunov_exponent(j_list=np.arange(0, 1, 1e-3),
                              time_steps=1000, dt=1e-2, epsilon=4e-2,
                              q_list=[0.25, 0.48, 0.75]):
    total_time = time_steps*dt
    lyapunov_exps = []
    for j in j_list:
        lyapunov_exps_qs = []
        for q_1 in q_list:
            q_2 = q_1+epsilon
            d0 = q_1-q_2
            distances = [epsilon]
            l_list = []
            for i in range(time_steps):
                q_1 = q_1 + dt*(gn.sigmoid(6*j*(2*q_1-1))-q_1)
                q_2 = q_2 + dt*(gn.sigmoid(6*j*(2*q_2-1))-q_2)
                d_dt = q_1-q_2
                distances.append(d_dt)
                l_list.append(np.abs(d_dt)/np.abs(d0))
                d0 = d_dt
            lyapunov_e = 1/total_time * np.sum(np.log(l_list))
            lyapunov_exps_qs.append(lyapunov_e)
        lyapunov_exps.append(np.max(lyapunov_exps_qs))
    fig = plt.figure()
    lyapunov_exps = np.array(lyapunov_exps)
    plt.axhline(0, color='r', linestyle='--', alpha=0.2)
    plt.axvline(1/3, color='k', linestyle='--', alpha=0.5)
    idx_pos = lyapunov_exps > 0
    plt.plot(j_list[idx_pos], lyapunov_exps[idx_pos], color='r', linewidth=4)
    plt.plot(j_list[~idx_pos], lyapunov_exps[~idx_pos], color='k', linewidth=4)
    plt.ylabel(r'$\lambda$')
    plt.xlabel('Coupling, J')
    fig.tight_layout()


def predictions_hysteresis_coupling(b_list=[0, 0.1, 0.2], sigma=0.05,
                                    ini_cond=None):
    j_array = np.concatenate((np.arange(0, 1, 1e-4), np.arange(0, 1, 1e-4)[::-1]))
    q_all = np.zeros((len(b_list), len(j_array)))
    dt = 1e-2
    tau = 0.1
    timescale = dt/tau
    for i_b, b in enumerate(b_list):
        q = ini_cond if ini_cond is not None else np.random.randn()*0.01+0.5
        q_list = [q]
        for t in range(len(j_array)-1):
            q = q + (gn.sigmoid(8*j_array[t]*(2*q-1)+2*b)-q)*timescale + np.random.randn()*np.sqrt(timescale)*sigma
            q_list.append(q)
        q_all[i_b] = q_list
    fig, ax = plt.subplots(ncols=3, figsize=(12, 5))
    for i_a, a in enumerate(ax):
        half_len = len(j_array)//2
        a.plot(j_array[:half_len], q_all[i_a][:half_len], color='k', linewidth=4,
               label='Ascending')
        a.plot(j_array[half_len:], q_all[i_a][half_len:], color='r', linewidth=4,
               label='Descending')
        a.set_title(f'B = {b_list[i_a]}')
        a.set_xlabel('Coupling, J(t)')
        a.set_ylim(-0.1, 1.1)
    ax[0].legend(frameon=False)
    fig.tight_layout()


if __name__ == '__main__':
    print('Mean-Field inference')
    # mf_dyn_sys_circle(n_iters=100, b=0.)
    # plot_2d_mean_passage_time(J=2, B=0., sigma=0.1)
    # plot_density_map_2d_mf(j=5, b=0, noise=0.1, tau=0.02, time_end=3000, dt=5e-3)
    # plot_potential_and_vector_field_2d(j=2, b=0, noise=0.1, tau=0.1,
    #                                    time_end=1, dt=1e-3)
    # plot_potentials_mf(j_list=np.arange(0.001, 1.01, 0.1), bias=0.15)
    # plot_pot_evolution_mfield(j=0.9, num_iter=15, sigma=0.1, bias=0)
    # plot_occupancy_distro(j=0.36, noise=0.08, tau=1, dt=5e-1, theta=theta, b=0,
    #                       t=10000, burn_in=0.001, n_sims=500)
    # q_list = []
    # for j in np.arange(0.01, 1, 0.01):
    #     q_list.append(find_repulsor(j=j, num_iter=30, epsilon=1e-1, q_i=0.01,
    #                                 q_f=0.95, stim=0.1, threshold=1e-5, theta=theta,
    #                                 neigh=3))
    # plt.plot(np.arange(0.01, 1, 0.01), q_list)
    # plot_mf_sol_stim_bias_different_sols(j_list=np.arange(0.001, 1, 0.001),
    #                                       stim=0.,
    #                                       num_iter=40,
    #                                       theta=theta)
    # plot_solutions_mfield(j_list=np.arange(0.001, 1.01, 0.001), stim=0, N=3,
    #                       plot_approx=False)
    # plot_3_examples_mf_evolution(avg=True)
    # examples_pot()
    # plot_crit_J_vs_B_neigh(j_list=np.arange(0., 1.005, 0.001),
    #                        num_iter=200, neigh_list=np.arange(3, 11),
    #                        dim3=False)
    # plot_noise_before_switch(j=0.395, b=0, theta=theta, noise=0.15,
    #                          tau=0.1, time_end=120000, dt=5e-3, p_thr=0.5,
    #                          steps_back=2000, steps_front=1000, gibbs=False)
    # mutual_inh_cartoon(inh=2.1, exc=2.1, n_its=10000, noise=0.025, tau=0.15,
    #                    skip=25)
    # plt.title('J=0.395')
    # plot_peak_noise_vs_j(j_list=np.arange(0.34, 0.55, 5e-3),
    #                      b=0, theta=theta, noise=0.12,
    #                      tau=0.1, time_end=50000, dt=5e-3, p_thr=0.5,
    #                      steps_back=2000, steps_front=500)
    # plot_q_bifurcation_vs_JB(j_list=np.arange(1/3, 1, 0.0001),
    #                          stim_list=np.arange(-0.1, 0.1, 0.001))
    # plot_potentials_mf(j_list=[0.1, 0.2, 1/3, 0.4, 0.6, 0.8, 1],
    #                    bias=0, neighs=3)
    # plot_potentials_mf(j_list=[0.35, 0.355, 0.36, 0.365, 0.37, 0.375],
    #                     bias=0, neighs=3)
    # plot_mf_sol_stim_bias(j_list=np.arange(0.00001, 1, 0.001), stim=-0.1,
    #                       num_iter=10)
    # transition_probs_j(t_dur=1, noise=0.3,
    #                     j_list=np.arange(0.001, 3.01, 0.005),
    #                     b=.2, tol=1e-10)
    # plot_exit_time(j=0.5, b=0, noise=0.1)
    # transition_probs_j_and_b(t_dur=1, noise=0.3,
    #                          j_list=np.linspace(0.001, 2, 200),
    #                          b_list=np.linspace(-0.2, 0.2, 100),
    #                          tol=1e-10)
    # plot_dominance_duration_mean_field(j=.39, b=0, theta=theta, noise=0.1,
    #                                     tau=0.008, time_end=15000, dt=1e-3)
    # plot_slope_wells_vs_B(j_list=np.arange(0.6, 1.01, 0.1),
    #                       b_list=np.arange(-.3, .3, 0.01))
    # plot_posterior_vs_stim(j_list=[0.05, 0.2, 0.41],
    #                        b_list=np.linspace(0, 0.25, 101),
    #                        theta=gn.return_theta(rows=10, columns=5,
    #                                              layers=2, factor=1))
    # boltzmann_2d_change_j(noise=0.1)
    # mean_field_simul_discrete(j_list=np.arange(0, 1, 1e-3), b=0, theta=gn.theta_rubin(),
    #                           num_iter=400)
    # psychometric_mf_analytical(t_dur=1000000, noiselist=[0.05, 0.08, 0.1, 0.15],
    #                            j_list=np.arange(0.6, 1.3, 0.1),
    #                            b_list=np.arange(-0.2, 0.2, 5e-3))
    # calc_min_action_path_and_plot(j=2, b=0, noise=0.1, theta=theta, steps=400000,
    #                               tol_stop=1e-30)
    # example_dynamics_hierarchical_theta(theta=gn.THETA_HIER)
    # bifurcation_hierarchical(b=0, varchange='descending')
    # bifurcation_hierarchical(b=0, varchange='ascending')
    # bifurcation_hierarchical(b=0, varchange='coupling')
    # alternation_rate_vs_accuracy(t_dur=1, tol=1e-8,
    #                              j_list=np.arange(0.4, 2, 0.01),
    #                              b=0.3, noise_list=[0.1, 0.15, 0.2, 0.25, 0.3])
    # calc_min_action_path_and_plot(j=0.8, b=0, noise=0.1, theta=theta, steps=400000,
    #                               tol_stop=1e-30)
    # boltzmann_2d_change_sigma(j=0.5, b=0)
    # levelts_laws(noise=0.1, j=0.39,
    #              b_list=np.round(np.arange(-0.01, 0.012, 0.0005), 4),
    #              theta=theta, time_end=12000, dt=1e-3, tau=0.008,
    #              n_nodes_th=50)
    # plot_3d_solution_mf_vs_j_b(j_list=np.arange(0.01, 1.01, 0.00025),
    #                             b_list=np.arange(0, 0.125, 0.025), N=3,
    #                             num_iter=200, tol=1e-3, dim3d=False)
    # plot_adaptation_mf(j=0.6, b=0.5, theta=theta, noise=0.1, gamma_adapt=3,
    #                    tau=1, time_end=100, dt=1e-2)
    # save_images_potential_hysteresis(j=0.39,
    #                                  b_list=np.linspace(-0.2, 0.2, 501),
    #                                  save_folder=DATA_FOLDER, tau=0.8,
    #                                  sigma=0.)
    # save_images_potential_hysteresis(j=0.39,
    #                                  b_list=np.linspace(-0.2, 0.2, 501),
    #                                  save_folder=DATA_FOLDER, tau=0.1,
    #                                  sigma=0.)
    # plot_adaptation_1d(j=0.5, b=0., noise=0.0, gamma_adapt=0.1,
    #                    tau=1, time_end=100, dt=1e-3)
    # for b in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    #     plot_mean_field_neg_stim_fixed_points(j_list=np.arange(0, 1, 0.005),
    #                                           b=b, theta=theta, num_iter=20)
    # cp_vs_coupling_random_neurons(j_list=np.arange(0, 0.6, 0.05), rand_neur_list=[False, 8, 16, 32],
    #                               nsimuls=2000, noise=0.15, load_sims=True)
    cp_vs_coupling_noise(j_list=np.arange(0, 0.6, 0.05), noise_list=[0.15],
                          nsimuls=200, load_sims=True, inset=False, cylinder=True,
                          barplot=True)
    plot_rsc_matrix_vs_b_list_and_coupling(b_list=np.arange(0, 1.02, 0.02),
                                            j_list=np.arange(0, 1.01, 0.02),
                                            nsims=20, load_data=True, sigma=0.2,
                                            long=True, cylinder=True, inset=False,
                                            barplot=True)
    # analytical_correlation_rsc(sigma=0.1, theta=get_regular_graph())
