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
import seaborn as sns




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


def mean_field_stim(J, num_iter, stim, sigma=1, theta=theta, val_init=None):
    #initialize random state of the cube
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
            vec[q] = gn.sigmoid(2*(sum(J*(2*vec[neighbours]-1)+stim))+np.random.randn()*sigma) 
        vec_time[i, :] = vec
    return vec_time


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
        q_new = 0.5*(1+ 1/j *(1/(2*n_neigh) * np.log(q/(1-q)) - beta))
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


def plot_solutions_mfield(j_list, stim=0, N=3):
    l = []
    for j in j_list:
        q = lambda q: gn.sigmoid(2*N*j*(2*q-1)+ stim*2*N) - q 
        l.append(np.clip(fsolve(q, 0.9), 0, 1))
    plt.plot([1/3, 1], [0.5, 0.5], color='grey', alpha=1, linestyle='--',
             label='Unstable FP')
    plt.plot(j_list, 1-np.array(l), color='k')
    plt.plot(j_list, l, color='k', label='Stable FP')
    plt.xlabel('J')
    plt.ylabel('q')
    # plt.title('Solutions of the dynamical system')
    plt.legend()



def plot_solutions_mfield_neighbors(ax, j_list, color='k', stim=0, N=3):
    l = []
    for j in j_list:
        q = lambda q: gn.sigmoid(2*N*j*(2*q-1)+ stim*2*N) - q 
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



def plot_crit_J_vs_B_neigh(j_list, num_iter=200,
                           beta_list=np.arange(-0.5, 0.5, 0.001),
                           neigh_list=np.arange(3, 11),
                           dim3=False):
    if dim3:
        ax = plt.figure().add_subplot(projection='3d')
    else:
        fig, ax = plt.subplots(1)
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
        ax.set_ylabel('B')
        ax.set_zlabel('J*')
    else:
        ax.set_xlabel('B')
        ax.set_ylabel('J*')
        ax_pos = ax.get_position()
        ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*1.05, ax_pos.y0+ax_pos.height*0.2,
                                ax_pos.width*0.06, ax_pos.height*0.5])
        newcmp = mpl.colors.ListedColormap(colormap)
        mpl.colorbar.ColorbarBase(ax_cbar, cmap=newcmp)
        ax_cbar.set_title('N')
        ax_cbar.set_yticks([0, 0.5, 1], [np.min(neigh_list),
                                         int(np.mean(neigh_list)),
                                         np.max(neigh_list)])
        fig.savefig(DATA_FOLDER+'/J_vs_NB_MF.png', dpi=400, bbox_inches='tight')



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
    return q*q/2 - np.log(1+np.exp(6*(j*(2*q-1)+bias)))/(12*j) #  + q*bias


def potential_mf_neighs(q, j, bias=0, neighs=3):
    return q*q/2 - np.log(1+np.exp(2*neighs*(j*(2*q-1)+bias)))/(4*neighs*j)


def plot_potentials_different_beta(j=0.5, beta_list=[-0.1, -0.05, 0, 0.05, 0.1]):
    plt.figure()
    colormap = pl.cm.Blues(np.linspace(0.2, 1, len(beta_list)))
    q = np.arange(0, 1, 0.001)
    for i_j, beta in enumerate(beta_list):
        pot = potential_mf(q, j, beta)
        plt.plot(q, pot-np.mean(pot), label=np.round(beta, 1), color=colormap[i_j])
    plt.legend(title='B+b')


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
    newcolors = Blues(np.linspace(0.2, 1, len(j_list)))
    newcolors_purples = Purples(np.linspace(0.4, 1, len(j_list)))
    red = np.array([1, 0, 0, 1])
    newcolors[len(j_list)//3, :] = red
    newcolors[(len(j_list)//3+1):, :] = newcolors_purples[(len(j_list)//3+1):]
    newcmp = mpl.colors.ListedColormap(newcolors)
    q = np.arange(0, 1, 0.001)
    fig, ax = plt.subplots(1)
    change_colormap = False
    for i_j, j in enumerate(j_list):
        pot = potential_mf_neighs(q, j, bias=bias, neighs=neighs)
        # pot = potential_expansion_any_order_any_point(q, j, b=0, order=8, point=0.5)
        # norm_cte = np.max(np.abs(pot))
        if abs(j - 0.333) < 0.01 and not change_colormap:
            color = 'r'
            change_colormap = True
        else:
            color = newcolors[i_j]
        ax.plot(q, pot-np.mean(pot), color=color, label=np.round(j, 2))
    ax_pos = ax.get_position()
    ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*1.02, ax_pos.y0,
                            ax_pos.width*0.04, ax_pos.height*0.9])
    mpl.colorbar.ColorbarBase(ax_cbar, cmap=newcmp)
    ax_cbar.set_yticks([0, max(j_list)/3, max(j_list)/2, max(j_list)], [0, 0.33, max(j_list)/2, max(j_list)])
    ax_cbar.set_title('J')
    ax.set_xlabel('q')
    ax.set_ylabel(r'Mean-centered potential $V_J(q)$')
    # ax.legend(title='J:')


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
    # init_cond = [0.25, 0.75]
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
    fig2, ax_2 = plt.subplots(ncols=2, figsize=(9, 4))
    ax2, ax3 = ax_2
    x1 = np.arange(0, 1.1, 5e-2)
    x2 = np.arange(0, 1.1, 5e-2)
    x, y = np.meshgrid(x1, x2)
    u1 = f_i(x, y, j=j, b=b)
    u2 = f_i(y, x, j=j, b=b)
    ax2.quiver(x, y, u1, u2)
    ax2.set_xlabel(r'$x_1$')
    ax2.set_ylabel(r'$x_2$')
    ax2.plot(m_solution.y[0], m_solution.y[1], color='r')
    ax2.plot(init_cond[0], init_cond[1], marker='o', color='r')
    ax2.plot(m_solution.y[0][-1], m_solution.y[1][-1], marker='x', color='b')
    # modulo = np.sqrt(u1**2 + u2**2)
    x1 = np.arange(0, 1, 1e-3)
    x2 = np.arange(0, 1, 1e-3)
    # ax2.plot(x1, f_i(x1, x1, j=j, b=b))
    x, y = np.meshgrid(x1, x2)
    u1 = f_i(x, y, j=j, b=b)
    u2 = f_i(y, x, j=j, b=b)
    modulo = np.sqrt(u1**2 + u2**2)
    image = ax3.imshow(np.flipud(modulo), extent=[0, 1, 0, 1],
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


def plot_mf_evolution_all_nodes(j=1, b=0, noise=0, tau=1, time_end=50, dt=5e-2):
    time, vec = solution_mf_sdo_euler(j, b, theta, noise, tau,
                                      time_end=time_end, dt=dt)
    plt.figure()
    for q in vec.T:
        plt.plot(time, q)
    plt.ylim(-0.05, 1.05)


def plot_eigenvals_and_behavior_MF_2_faces(b=0, diag_neg=False):
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
    j_list = np.arange(0, 2+1e-3, 1e-3)
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
        x = x + (dt*(gn.sigmoid(2*j*(2*np.matmul(theta, x)-3) + 6*b) - x) +\
            np.random.randn(theta.shape[0])*noise*np.sqrt(dt)) / tau
        # x = np.clip(x, 0, 1)
        x_vec[t, :] = x  # np.clip(x, 0, 1)
    return time, x_vec


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
        vec_sims = np.nanmean(vec_sims, axis=1)
        vec = np.concatenate((vec, vec_sims))
    print(len(vec))
    kws = dict(histtype= "stepfilled", linewidth = 1)
    ax.hist(vec, label=int(t*n_sims), cumulative=True, bins=150, density=True,
            color='mistyrose', edgecolor='k', **kws)
    # sns.kdeplot(vec, label=t, bw_adjust=0.05, cumulative=True)
    plot_boltzmann_distro(j, noise, b=b, ax=ax)
    plt.legend()
    ax.set_ylabel('CDF(x)')
    ax.set_xlabel('x')
    ax.set_xlim(-0.05, 1.05)


def vector_proj(u, v):
    return (np.dot(u, v)/np.dot(v, v))*v 


def get_n_eigenvects(n, theta):
    eigvals, eigvects = np.linalg.eig(theta)
    sorted_evals_idx = np.argsort(np.linalg.eig(theta)[0])[::-1]
    return eigvects[sorted_evals_idx[:n]]


def projection_mf_plot(theta, j=1, b=0, noise=0, tau=1):
    t, X = solution_mf_sdo(j, b, theta, noise, tau)
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
    expo = 6*(j*(2*q-1)+b)
    return 1 - 12*j*gn.sigmoid(expo)*(1-gn.sigmoid(expo))


def k_i_to_j(j, xi, xj, noise, b=0):
    v_2_xi = second_derivative_potential(xi, j, b=0)
    v_2_xj = second_derivative_potential(xj, j, b=0)
    v_xi = potential_mf(xi, j, b)
    v_xj = potential_mf(xj, j, b)
    return np.sqrt(np.abs(v_2_xi*v_2_xj))*np.exp(-2*(v_xi - v_xj)/noise**2) / (2*np.pi)


def transition_probs_j(t_dur, noise,
                       j_list=np.arange(0.001, 3.01, 0.005),
                       b=0, tol=1e-10):
    trans_prob_u_to_s_1 = []
    trans_prob_u_to_s_2 = []
    trans_prob_s_to_u_1 = []
    trans_prob_s_to_u_2 = []
    # trans_prob_i_to_j = []
    # trans_prob_j_to_i = []
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
        k_x_u_to_s_1 = k_i_to_j(j, x_stable_1, x_unstable, noise, b)
        k_x_u_to_s_2 = k_i_to_j(j, x_stable_2, x_unstable, noise, b)
        k_x_s_to_u_1 = k_i_to_j(j, x_unstable, x_stable_1, noise, b)
        k_x_s_to_u_2 = k_i_to_j(j, x_unstable, x_stable_2, noise, b)
        # k = k_x_ij + k_x_ji
        # p_is = k_i_to_j(j, xi, xj, noise, b) / k
        # p_js = k_i_to_j(j, xj, xi, noise, b) / k
        # P_is is prob to stay in i at end of trial given by t_dur
        # P_js is prob to stay in j at end of trial given by t_sdur
        # trans_prob_ij.append(p_is * (1 - np.exp(-k*t_dur)))  # prob of at some point going from j to i
        # trans_prob_ji.append(p_js * (1 - np.exp(-k*t_dur))) # prob of at some point going from i to j
        trans_prob_u_to_s_1.append(1-np.exp(-k_x_u_to_s_1*t_dur))
        trans_prob_u_to_s_2.append(1-np.exp(-k_x_u_to_s_2*t_dur))
        trans_prob_s_to_u_1.append(1-np.exp(-k_x_s_to_u_1*t_dur))
        trans_prob_s_to_u_2.append(1-np.exp(-k_x_s_to_u_2*t_dur))
        # trans_prob_i_to_j.append(1-np.exp(-(k_x_s_to_u_2+k_x_u_to_s_1)*t_dur))
        # trans_prob_j_to_i.append(1-np.exp(-(k_x_s_to_u_1+k_x_u_to_s_2)*t_dur))
        # 1 - np.exp(-k*t_dur) is prob to change from i->j and vice-versa
    trans_prob_j_to_i = np.array(trans_prob_u_to_s_1)*np.array(trans_prob_s_to_u_2)
    trans_prob_i_to_j = np.array(trans_prob_u_to_s_2)*np.array(trans_prob_s_to_u_1)
    plt.figure()
    plt.plot(j_list, trans_prob_u_to_s_1, label='P_u_to_s_1')
    plt.plot(j_list, trans_prob_u_to_s_2, label='P_u_to_s_2')
    plt.plot(j_list, trans_prob_s_to_u_1, label='P_s_to_u_1')
    plt.plot(j_list, trans_prob_s_to_u_2, label='P_s_to_u_2')
    plt.xlabel('J')
    plt.ylabel('Transition probability')
    plt.legend()
    plt.figure()
    plt.plot(j_list, trans_prob_i_to_j, label='P_ij')
    plt.plot(j_list, trans_prob_j_to_i, label='P_ji')
    plt.xlabel('J')
    plt.ylabel('Transition probability')
    plt.legend()
    plt.figure()
    plt.plot(trans_prob_i_to_j, trans_prob_j_to_i)
    plt.xlabel('T_ij')
    plt.ylabel('T_ji')


if __name__ == '__main__':
    # plot_potential_and_vector_field_2d(j=1, b=0, noise=0., tau=1,
    #                                     time_end=50, dt=5e-2)
    # plot_potentials_mf(j_list=np.arange(0.001, 1.01, 0.1), bias=0.05)
    # plot_pot_evolution_mfield(j=0.9, num_iter=15, sigma=0.1, bias=0)
    # plot_occupancy_distro(j=0.6, noise=0.2, tau=1, dt=5e-2, theta=theta, b=0,
    #                       t=5000, burn_in=0.001, n_sims=50)
    plot_mf_evolution_all_nodes(j=.1, b=0., noise=0.02, tau=1, time_end=20,
                                dt=5e-2)
    # q_list = []
    # for j in np.arange(0.01, 1, 0.01):
    #     q_list.append(find_repulsor(j=j, num_iter=30, epsilon=1e-1, q_i=0.01,
    #                                 q_f=0.95, stim=0.1, threshold=1e-5, theta=theta,
    #                                 neigh=3))
    # plt.plot(np.arange(0.01, 1, 0.01), q_list)
    # plot_mf_sol_stim_bias_different_sols(j_list=np.arange(0.001, 1, 0.01),
    #                                      stim=0.0,
    #                                      num_iter=40,
    #                                      theta=gn.return_theta(columns=5, rows=10))
    # plot_crit_J_vs_B_neigh(j_list=np.arange(0.01, 1, 0.001),
    #                        num_iter=200,
    #                        beta_list=np.arange(-0.5, 0.5, 0.001),
    #                        neigh_list=np.arange(3, 12),
    #                        dim3=False)
    # plot_mf_sol_stim_bias(j_list=np.arange(0.00001, 1, 0.001), stim=-0.1,
    #                       num_iter=10)
