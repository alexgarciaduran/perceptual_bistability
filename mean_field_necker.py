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
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pylab as pl
import matplotlib as mpl
import sympy
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import seaborn as sns
import cv2
import torch

mpl.rcParams['font.size'] = 16
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize']= 14
plt.rcParams['ytick.labelsize']= 14


# ---GLOBAL VARIABLES
pc_name = 'alex'
if pc_name == 'alex':
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/mean_field_necker/data_folder/'  # Alex

elif pc_name == 'alex_CRM':
    DATA_FOLDER = 'C:/Users/agarcia/Desktop/phd/necker/data_folder/'  # Alex CRM



# C matrix:
c_data = DATA_FOLDER + 'c_mat.npy'
C = np.load(c_data, allow_pickle=True)


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
    
    


def mean_field_fixed_points(j_list, stim, num_iter=100):
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
                          theta=theta, val_init=q_f)[-1, neighs == neigh][0]
    f_b = mean_field_stim(j, num_iter=num_iter, stim=stim, sigma=0,
                          theta=theta, val_init=q_i)[-1, neighs == neigh][0]
    if np.abs(f_a-f_b) < threshold:
        return np.nan
    while diff >= threshold*1e-2:
        c = (q_i+q_f)/2
        f_c = mean_field_stim(j, num_iter=num_iter, stim=stim, sigma=0,
                              theta=theta, val_init=c)[-1, neighs == neigh][0]
        if np.abs(f_a-f_c) < threshold and np.abs(f_b-f_c) < threshold:
            return np.nan
        if np.abs(f_a-f_c) < threshold:
            q_f = c
            f_a = mean_field_stim(j, num_iter=num_iter, stim=stim, sigma=0,
                                  theta=theta, val_init=q_f)[-1, neighs == neigh][0]
        elif np.abs(f_b-f_c) < threshold:
            q_i = c
            f_b = mean_field_stim(j, num_iter=num_iter, stim=stim, sigma=0,
                                  theta=theta, val_init=q_i)[-1, neighs == neigh][0]
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


def plot_solutions_mfield(j_list, stim=0, N=3, plot_approx=False):
    fig = plt.figure(figsize=(6, 4))
    l = []
    for j in j_list:
        q = lambda q: gn.sigmoid(2*N*j*(2*q-1)+ stim*2*N) - q 
        l.append(np.clip(fsolve(q, 0.9), 0, 1))
    plt.plot([1/3, 1], [0.5, 0.5], color='grey', alpha=1, linestyle='--',
             label='Unstable FP')
    plt.plot(j_list, 1-np.array(l), color='k')
    plt.plot(j_list, l, color='k', label='Stable FP')
    plt.xlabel(r'Coupling $J$')
    plt.ylabel(r'Posterior $q$')
    # plt.title('Solutions of the dynamical system')
    if plot_approx:
        j_list1 = np.arange(1/N, 1, 0.001)
        r = (j_list1*N-1)*3/(4*(j_list1*N)**3)
        plt.plot(j_list1, np.sqrt(r)+0.5, color='b', linestyle='--')
        plt.plot(j_list1, -np.sqrt(r)+0.5, label=r'$q=0.5 \pm \sqrt{r}$',
                 color='b', linestyle='--')
        plt.plot([0, 1/N], [0.5, 0.5], color='r', label=r'$q=0.5$',
                 linestyle='--')
    plt.axvline(1/N, color='r', alpha=0.2)
    plt.text(1/N-0.065, 0.12,  r'$J^{\ast}=1/3$', rotation='vertical')
    # xtcks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # xtcks = np.sort(np.unique([0, 0.2, 0.4, 1/N, 0.6, 0.8, 1]))
    # labs = [x for x in xtcks]
    # pos = np.where(xtcks == 1/N)[0][0]
    # labs[pos] = r'$J^{\ast}$'  # '1/'+str(N)
    # plt.xticks(xtcks, labs)
    plt.legend()
    plt.tight_layout()
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
                           beta_list=np.arange(-0.5, 0.5, 0.001),
                           neigh_list=np.arange(3, 11),
                           dim3=False):
    if dim3:
        ax = plt.figure().add_subplot(projection='3d')
    else:
        fig, ax = plt.subplots(1, figsize=(5, 4))
        colormap = pl.cm.Blues(np.linspace(0.2, 1, len(neigh_list)))
    for n_neigh in neigh_list:
        print(n_neigh)
        first_j = []
        for i_b, beta in enumerate(beta_list):
            for j in j_list:
                q_fin = 0.65
                for i in range(num_iter):
                    q_fin = backwards(q_fin, j, beta, n_neigh)
                if ~np.isnan(q_fin):
                    first_j.append(j)
                    break
            if len(first_j) != (i_b+1):
                first_j.append(np.nan)
        z = np.repeat(n_neigh, len(first_j))
        if dim3:
            ax.plot3D(z, beta_list, first_j, color='k')
        else:
            ax.plot(beta_list, first_j, color=colormap[int(n_neigh-min(neigh_list))],
                    label=n_neigh)
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
        ax_pos = ax.get_position()
        ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*1.05, ax_pos.y0+ax_pos.height*0.2,
                                ax_pos.width*0.06, ax_pos.height*0.5])
        newcmp = mpl.colors.ListedColormap(colormap)
        mpl.colorbar.ColorbarBase(ax_cbar, cmap=newcmp, label='Neighbors N')
        ax_cbar.set_yticks([0, 0.5, 1], [np.min(neigh_list),
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
    beta = 6*beta
    return (-beta+k+np.log(k-np.sqrt(k*(k-2))-1))/(2*k),\
        (-beta+k+np.log(k+np.sqrt(k*(k-2))-1))/(2*k)


def sol_mf_stim_taylor(j, beta):
    k = 6*j
    b = 6*beta
    return (1+np.exp(b)-np.exp(-b)*k)/(np.cosh(b)*2-2*k+2)


def sol_taylor_beta(j, beta):
    k = 6*j
    beta = 6*beta
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
    b = 6*beta
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


def dyn_sys_mf(q, dt, j, sigma=1, bias=0):
    return np.clip(q + dt*(gn.sigmoid(6*j*(2*q-1)+6*bias)-q)+
        np.random.randn()*np.sqrt(dt)*sigma, 0, 1)


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


def potential_mf(q, j, bias=0):
    return q*q/2 - np.log(1+np.exp(6*(j*(2*q-1))+bias*2))/(12*j) #  + q*bias


def potential_mf_neighs(q, j, bias=0, neighs=3):
    return q*q/2 - np.log(1+np.exp(2*neighs*(j*(2*q-1)+bias)))/(4*neighs*j)


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
    alpha = 6*j*(2*x-1) + 6*b
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


def f_i(xi, xj, j=1, b=0):
    return gn.sigmoid(a_ij(xi, xj, j=j, b=b)) - xi


def f_i_both(x, j=1, b=0):
    xi, xj = x
    return [gn.sigmoid(a_ij(xi, xj, j=j, b=b)) - xi,
            gn.sigmoid(a_ij(xj, xi, j=j, b=b)) - xj]


def f_i_diagonal(x, j=1, b=0):
    return gn.sigmoid(a_ij(x, x, j=j, b=b)) - x


def f_i_diagonal_neg(x, j=1, b=0):
    y = x[1]
    x = x[0]
    return [gn.sigmoid(2*(j*(2*x-1)+b)) - x,
            gn.sigmoid(2*(j*(-2*y+1)+b)) + y - 1]


def plot_potential_and_vector_field_2d(j=1, b=0, noise=0, tau=1, time_end=50, dt=5e-2):
    ax = plt.figure().add_subplot(projection='3d')
    x1 = np.arange(0, 1, 1e-3)
    x2 = np.arange(0, 1, 1e-3)
    x, y = np.meshgrid(x1, x2)
    V_x = potential_2d_faces(x, y, j=j, b=b)
    ax.plot_surface(x, y, V_x, alpha=0.4)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'Potential, $V(\vec{x})$')
    init_cond = [0.1, 0.9]
    # init_cond = np.random.rand(2)
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
    fig2, ax_2 = plt.subplots(ncols=1, figsize=(5, 4))
    im = ax_2.imshow(np.flipud(V_x), cmap='jet', extent=[np.min(x1), np.max(x1),
                                                         np.min(x2), np.max(x2)])
    plt.colorbar(im)
    fig2, ax_2 = plt.subplots(ncols=2, figsize=(9, 4))
    ax2, ax3 = ax_2
    x1 = np.arange(-0.2, 1.2, 5e-2)
    x2 = np.arange(-0.2, 1.2, 5e-2)
    x, y = np.meshgrid(x1, x2)
    u1 = f_i(x, y, j=j, b=b)
    u2 = f_i(y, x, j=j, b=b)
    ax2.quiver(x, y, u1, u2)
    x1 = np.arange(-0.2, 1.2, 1e-3)
    x2 = np.arange(-0.2, 1.2, 1e-3)
    # lam1 = get_eigen_1(j, x1, b=b)
    # idx_0_lam1 = np.argsort(np.abs(lam1))[:2]
    lam2 = get_eigen_2(j, x1, b=b)
    idx_0_lam2 = np.argsort(np.abs(lam2))[:2]
    # lamneg1 = get_eigen_1_neg(j, x1, b=b)
    # idx_0_lamneg1 = np.argsort(np.abs(lamneg1))[:2]
    lamneg2 = get_eigen_2_neg(j, x1, b=b)
    idx_0_lamneg2 = np.argsort(np.abs(lamneg2))[:2]
    for i in range(2):
        ax2.plot(x1[idx_0_lam2[i]], x1[idx_0_lam2[i]], marker='o',
                 color='b', markersize=8)
        ax2.plot(x1[idx_0_lamneg2[i]], 1-x1[idx_0_lamneg2[i]], marker='o',
                 color='b', markersize=8)
    ax2.set_xlabel(r'$x_1$')
    ax2.set_ylabel(r'$x_2$')
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
    plt.colorbar(image, label=r'speed, $||f(x_1, x_2)||$')
    ax3.set_xlabel(r'$x_1$')
    ax3.set_ylabel(r'$x_2$')


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
    return gn.sigmoid(6*j*(2*x-1)+6*b)*(1-gn.sigmoid(6*j*(2*x-1)+6*b))*12*j-1


def get_eigen_2(j, x, b=0):
    return gn.sigmoid(6*j*(2*x-1)+6*b)*(1-gn.sigmoid(6*j*(2*x-1)+6*b))*4*j-1


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


def solution_mf_sdo_euler(j, b, theta, noise, tau, time_end=50, dt=1e-2):
    time = np.arange(0, time_end+dt, dt)
    x = np.random.rand(theta.shape[0])  # initial_cond
    x_vec = np.empty((len(time), theta.shape[0]))
    x_vec[:] = np.nan
    x_vec[0, :] = x
    for t in range(1, time.shape[0]):
        x = x + (dt*(gn.sigmoid(2*j*(2*np.matmul(theta, x)-3) + 6*b) - x)/ tau +\
            np.random.randn(theta.shape[0])*noise*np.sqrt(dt/tau)) 
        # x = np.clip(x, 0, 1)
        x_vec[t, :] = x  # np.clip(x, 0, 1)
    return time, x_vec


def solution_mf_sdo_euler_OU_noise(j, b, theta, noise, tau, time_end=50, dt=1e-2,
                                   tau_n=1, approx_init=False):
    # print('Start simulating MF with OU noise w/o adaptation')
    time = np.arange(0, time_end+dt, dt)
    if approx_init:
        x = 0.5 + np.random.randn(theta.shape[0])*1e-1
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
        x_n = (dt*(gn.sigmoid(2*j*(2*np.matmul(theta, x)-n_neighs) + 2*b) - x)) / tau
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
        x1_temp = np.clip(x1_temp, 0, 1)
        x2_temp = np.clip(x2_temp, 0, 1)
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


def analyze_jump_dynamics(j, b, theta, noise=0.35, tau=0.1, dt=1e-3, t_end=1000,
                          steps_back=250, steps_front=250):
    idx_0 = []
    idx_1 = []
    while len(idx_0) == 0 or len(idx_1):
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
    mean_vals_1_array = np.empty((theta.shape[0], len(idx_1), steps_back+steps_front))
    mean_vals_1_array[:] = np.nan
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
            mean_vals_1_array[i_k, i, :] = np.convolve(vec[idx - steps_back:idx+steps_front, k],
                                                np.ones(conv_window)/conv_window,
                                                mode='same')
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
    

def plot_boltzmann_distro(j, noise, b=0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    q = np.arange(0, 1.001, 0.001)
    pot = potential_mf(q, j, bias=b)
    distro = np.exp(-2*pot/noise**2)
    ax.plot(q, np.cumsum(distro) / np.sum(distro),
            color='r', label='analytical')


def second_derivative_potential(q, j, b=0):
    expo = 6*(j*(2*q-1))+2*b
    return 1 - 12*j*gn.sigmoid(expo)*(1-gn.sigmoid(expo))


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


def plot_exit_time(j, b=0, noise=0.01):
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
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.075, right=0.98,
                        hspace=0.3, wspace=0.5)
    ax = ax.flatten()
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    q_pot = np.arange(0, 1, 1e-3)
    pot = potential_mf(q_pot, j, b)
    ax[0].plot(q_pot, pot-np.min(pot), color='k', linewidth=2.5)
    ax[0].set_ylabel('Potential V(q)')
    distro = np.exp(-2*pot/noise**2)
    ax[1].plot(q_pot, distro / np.sum(distro), color='k', linewidth=2.5)
    val_unst = potential_mf(q_val_bckw, j, b)
    ax[1].plot(q_val_bckw, np.exp(-2*val_unst/noise**2)/np.sum(distro),
               marker='o', color='b', linestyle='', label=r'$q^*_{unstable}$',
               markersize=8)
    ax[1].axvline(x_stable_1, color='r', alpha=0.5)
    ax[1].axvline(x_stable_2, color='g', alpha=0.5)
    ax[1].set_ylabel(r'Stationary distribution $p_s(q)$')
    ax[2].plot(q, time_from_a_to_b, color='k', linewidth=2.5, label='Exact')
    ax[2].plot(q, time_from_a_to_b_approx, color='magenta', linewidth=2, linestyle='--',
               label='Approximation')
    ax[2].legend(frameon=False)
    ax[2].set_ylabel(r'Time from $a$ to $q$, $T(a \rightarrow q)$')
    ax[2].set_xlabel('Approximate posterior q')
    pot_unst = potential_mf(q_val_bckw, j, b)-np.min(pot)
    ax[0].axvline(x_stable_1, color='r', alpha=0.5, label=r'$a = q^*_{stable, L}$')
    ax[0].plot(q_val_bckw, pot_unst, marker='o', color='b', linestyle='',
               label=r'$b = q^*_{unstable}$', markersize=8)
    ax[0].axvline(x_stable_2, color='g', alpha=0.5, label=r'$c = q^*_{stable, R}$')
    ax[0].legend(frameon=False)
    idx = np.where((q-round(q_val_bckw, 2)) == 0)
    ax[2].plot(q_val_bckw, time_from_a_to_b[idx[0][0]], marker='o',
               linestyle='', color='b', markersize=8)
    ax[2].axvline(x_stable_1, color='r', alpha=0.5)
    ax[2].axvline(x_stable_2, color='g', alpha=0.5)
    distro_split = np.array(splitting_prob_x_through_a)
    ax[3].plot(q, distro_split, color='k', linewidth=2.5)
    ax[3].plot(q_val_bckw, distro_split[idx[0][0]], marker='o',
               linestyle='', color='b', markersize=8)
    ax[3].axvline(x_stable_1, color='r', alpha=0.5)
    ax[3].axvline(x_stable_2, color='g', alpha=0.5)
    ax[3].set_xlabel('Approximate posterior q')
    ax[3].set_ylabel(r'Splitting probability $\pi_{a}(x)$')
    ax[3].set_ylim(-0.05, 1.05)
    fig.tight_layout()


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
    fig, ax = plt.subplots(1)
    ax.set_xlabel('Time to switch (s)')
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
        ax.set_ylabel(r'Noise $n(t)$')
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
        ax.set_ylabel(r'$k(\vec{x})$')
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
    # ou_vals_mean_filt = np.convolve(ou_vals_mean, np.ones(50)/50, mode='same')
    ax.plot(time, ou_vals_mean, color='k', linewidth=2.5)
    # ax.fill_between(time, ou_vals_mean-ou_vals_std, ou_vals_mean+ou_vals_std,
    #                 color='gray', alpha=0.3)


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
            vec[q] = gn.sigmoid(2*(sum(j*(2*vec[neighbours]-1)+b_list[i])))
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


def plot_mf_hysteresis(j_list=[0.01, 0.2, 0.41],
                       b_list=np.linspace(-0.5, 0.5, 1001),
                       theta=theta):
    b_list = np.concatenate((b_list[:-1], b_list[::-1]))
    plt.figure()
    colormap = ['navajowhite', 'orange', 'saddlebrown']
    for i_j, j in enumerate(reversed(j_list)):
        vec = mean_field_stim_change(j, b_list,
                                     val_init=None, theta=theta)
        plt.plot(b_list, vec[:, 0], color=colormap[i_j],
                 label=np.round(j, 1), linewidth=4)
    plt.xlabel('Sensory evidence, B')
    plt.ylabel('Approximate posterior q(x=1)')
    plt.legend(title='J')


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


def save_images_potential_hysteresis(j=0.39,
                                     b_list=np.linspace(-0.2, 0.2, 501),
                                     theta=theta,
                                     save_folder=DATA_FOLDER):
    b_list = np.concatenate((b_list[:-1], b_list[::-1]))
    vec = mean_field_stim_change(j, b_list,
                                 val_init=None, theta=theta)
    q = np.arange(0, 1, 0.001)
    for i in range(vec.shape[0]):
        fig, ax = plt.subplots(nrows=2, figsize=(6, 10))
        ax[0].plot(b_list[:i], vec[:i, 0], color='navajowhite',
                   linewidth=4)
        ax[0].set_ylabel('Approximate posterior, q(x=1)')
        ax[0].set_xlabel('Stimulus strength, B')
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].set_xlim(-0.21, 0.21)
        ax[0].set_ylim(0, 1)
        pot = potential_mf(q, j, b_list[i])
        val_particle = potential_mf(vec[i, 0], j, b_list[i])
        ax[1].plot(q, pot-np.nanmean(pot), color='purple',
                   linewidth=4)
        ax[1].plot(vec[i, 0], val_particle-np.nanmean(pot),
                   marker='o', markersize=8, color='k')
        ax[1].set_ylabel('Potential')
        ax[1].set_xlabel('Approximate posterior, q(x=1)')
        ax[1].spines['left'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].set_yticks([])
        ax[1].set_xticks([])
        fig.savefig(save_folder + '/images_video_hyst/' + str(i) + '.png',
                    dpi=100)
        plt.close(fig)

    
def create_video_from_images(image_folder=DATA_FOLDER+'/images_video_hyst/'):
    video_name = image_folder + 'hysteresis.mp4'
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


def mf_dyn_sys_circle(j, n_iters=100, b=0.):
    kernel = exp_kernel(tau=4)
    theta = scipy.linalg.circulant(kernel).T
    # stim = np.zeros((n_iters, theta.shape[0]))
    # stim[n_iters//2-n_iters//4:n_iters//2-n_iters//4+n_iters//8, ::2] = b
    # stim[n_iters//2-+n_iters//4+n_iters//8+1:1+n_iters//2-n_iters//4+2*n_iters//8, 1::2] = -b
    jlist = np.linspace(0.01, 0.06, 500)
    fig, ax = plt.subplots(1, figsize=(6, 5))
    vecarr = np.empty((len(jlist), theta.shape[0]))
    for t in range(len(jlist)):
        vec_time = mean_field_stim(jlist[t], n_iters, b, val_init=None, theta=theta, sigma=0)
        vecarr[t] = np.max((vec_time[-1], 1-vec_time[-1]), axis=0)
    for i in range(theta.shape[0]):
        ax.plot(jlist, vecarr[:, i], color='k')
        ax.plot(jlist, 1-vecarr[:, i], color='k')
    ax.set_ylim(-0.05, 1.05)
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
        # maxeigval = np.sum(ker)
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
            print(tol_stop)
            print(lag.item())
            print(action_vals[-1])
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
                                             tol_stop=tol_stop, neg=neg)
    return t, x, xs, x_unstable, action_vals


def calc_min_action_path_and_plot(j=0.5, b=0, noise=0.1, theta=theta, steps=20000,
                                  tol_stop=1e-2, numiters=500, dt=5e-3):
    t, x, xs, x_unstable, action_vals = minimum_action_path(j, b, noise, theta=theta, numiters=numiters, dt=dt,
                                                            steps=steps, tol_stop=tol_stop)
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
    nlines = 4
    for i in range(1, nlines+1):
        plt.plot(t, xs[(xs.shape[0]//(nlines+1))*i, :, 0], label=((xs.shape[0]//(nlines+1))*i)**2*80)
    plt.legend(title='Gradient descend step')
    plt.xlabel('Time (s)')
    plt.ylabel('Approximate posterior MAP')
    plt.ylim(-0.02, 1.02)
    plt.figure()
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x[:, 0].detach().numpy(), x[:, 2].detach().numpy(), x[:, 4].detach().numpy())
    ax.set_xlabel('q_0')
    ax.set_ylabel('q_2')
    ax.set_zlabel('q_4')
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x[:, 0].detach().numpy(), x[:, 1].detach().numpy(), x[:, 3].detach().numpy())
    ax.set_xlabel('q_0')
    ax.set_ylabel('q_1')
    ax.set_zlabel('q_3')


def calc_potential_diff(j=0.5, b=0, noise=0.1, theta=theta, steps=20000,
                        tol_stop=1e-2, numiters=500, dt=5e-3):
    theta = np.array(((4, 2), (2, 4)))
    eigval_fun_1 = get_eigen_1
    eigval_fun_2 = get_eigen_2
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
    q = np.arange(-0.1, 1.1, 1e-3)
    fig, ax = plt.subplots(ncols=2, figsize=(7, 3.4))
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
    colormap = pl.cm.Oranges(np.linspace(0.4, 1, len(b_list)))
    for i_b, b in enumerate(b_list):
        pot = potential_mf(q, 0.5, bias=b)
        ax[1].plot(q, pot-np.mean(pot), color=colormap[i_b], label=b)
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
        

if __name__ == '__main__':
    # plot_potential_and_vector_field_2d(j=2, b=0, noise=0., tau=1,
    #                                     time_end=50, dt=5e-2)
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
    #                         num_iter=200,
    #                         beta_list=np.arange(-1, 1, 0.001),
    #                         neigh_list=np.arange(3, 12),
    #                         dim3=False)
    # plot_noise_before_switch(j=0.395, b=0, theta=theta, noise=0.15,
    #                          tau=0.1, time_end=50000, dt=5e-3, p_thr=0.5,
    #                          steps_back=2000, steps_front=500, gibbs=False)
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
    # calc_min_action_path_and_plot(j=1.5, b=0, noise=0.1, theta=theta, steps=400000,
    #                               tol_stop=-1)
    # boltzmann_2d_change_sigma(j=0.5, b=0)
    # levelts_laws(noise=0.1, j=0.39,
    #              b_list=np.round(np.arange(-0.01, 0.012, 0.0005), 4),
    #              theta=theta, time_end=12000, dt=1e-3, tau=0.008,
    #              n_nodes_th=50)
    # plot_3d_solution_mf_vs_j_b(j_list=np.arange(0.01, 1.01, 0.00025),
    #                             b_list=np.arange(0, 0.125, 0.025), N=3,
    #                             num_iter=200, tol=1e-3, dim3d=False)
    # plot_adaptation_mf(j=0.4, b=0., theta=theta, noise=0.0, gamma_adapt=1,
    #                    tau=1, time_end=100, dt=1e-3)
    plot_adaptation_1d(j=0.5, b=0., noise=0.0, gamma_adapt=0.1,
                       tau=1, time_end=100, dt=1e-3)
    # for b in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    #     plot_mean_field_neg_stim_fixed_points(j_list=np.arange(0, 1, 0.005),
    #                                           b=b, theta=theta, num_iter=20)
