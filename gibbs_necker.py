# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:45:48 2023

@author: alexg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy
import itertools
import seaborn as sns
import os
import scipy.stats as stats
import matplotlib.pylab as pl
import networkx as nx
import matplotlib as mpl
import cv2
from matplotlib.lines import Line2D
from joblib import Parallel, delayed
from sbi.inference import SNLE
from sbi.utils import MultipleIndependent
import torch
from torch.distributions import Beta, Binomial, Gamma, Uniform



"""
8x8 matrix, left-right order first front:
    
Front:
        
    0    1
    
    
    2    3

Back:
    
    4    5
    
    
    6    7
    
Connections:
    0: 1, 2, 4
    1: 0, 3, 5
    2: 0, 3, 6
    3: 1, 2, 7
    4: 0, 5, 6
    5: 1, 4, 7
    6: 2, 4, 7
    7: 3, 5, 6
    
"""


mpl.rcParams['font.size'] = 14
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['xtick.labelsize']= 12
plt.rcParams['ytick.labelsize']= 12

# ---GLOBAL VARIABLES
pc_name = 'alex'
if pc_name == 'alex':
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/gibbs_sampling_necker/data_folder/'  # Alex

elif pc_name == 'alex_CRM':
    DATA_FOLDER = 'C:/Users/agarcia/Desktop/phd/necker/data_folder/'  # Alex CRM

# THETA mat

THETA = np.array([[0 ,1 ,1 ,0 ,1 ,0 ,0 ,0], [1, 0, 0, 1, 0, 1, 0, 0],
                  [1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1],
                  [0, 0, 1, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 1, 1, 0]])


def return_theta(rows=10, columns=5, layers=2, factor=1):
    graph = nx.grid_graph(dim=(columns, rows, layers))
    graph_array = nx.to_numpy_array(graph)
    for i in range(rows*columns*layers):
        if (i % columns) == 0 or ((i+1) % columns) == 0:
            # if i < rows*columns:
            #     graph_array[i, i+rows*columns] = factor
            #     graph_array[i+rows*columns, i] = factor
            # else:
            #     graph_array[i, i-rows*columns] = factor
            #     graph_array[i-rows*columns, i] = factor
            continue
        else:
            if i < rows*columns:
                graph_array[i, i+rows*columns] = 0
                graph_array[i+rows*columns, i] = 0
            else:
                graph_array[i, i-rows*columns] = 0
                graph_array[i-rows*columns, i] = 0
    # graph = nx.from_numpy_matrix(graph_array)
    return graph_array


def lattice_1d(columns=5, rows=5, layers=1):
    return nx.to_numpy_array(nx.grid_graph(dim=(columns, rows, layers)))

def get_theta_signed(j, theta):
    th = np.copy(theta)
    for t in range(8):
        ind = get_connections(t)
        th[t, ind] = [-1 if ((i >= 4) and (t <= 3)
                             or (i <= 3) and (t >= 4)) else 1 for i in ind]
    return th*j


def get_connections(node):
    if node == 0:
        return [1, 2, 4]
    if node == 1:
        return [0, 3, 5]
    if node == 2:
        return [0, 3, 6]
    if node == 3:
        return [1, 2, 7]
    if node == 4:
        return [0, 5, 6]
    if node == 5:
        return [1, 4, 7]
    if node == 6:
        return [2, 4, 7]
    if node == 7:
        return [3, 5, 6]


def sigmoid(x):
    return 1/(1+np.exp(-x))


def k_val(x_vec, j_mat, stim=0):
    return np.matmul(np.matmul(x_vec.T, j_mat), x_vec)/2 + np.sum(stim*x_vec)


def change_prob(x_vect, x_vect1, j, stim=0, theta=THETA):
    return sigmoid((-k_val(x_vect, j*theta, stim=stim) +
                    k_val(x_vect1, j*theta, stim=stim)))


def mat_theta(x_vect_1, x_vect_2, j):
    mat = np.zeros((len(x_vect_1), len(x_vect_2)))
    for i in range(len(x_vect_1)):
        connections = get_connections(i)
        for con in connections:
            mat[i, con] = x_vect_1[i] * x_vect_2[con] * j
    return mat


def transition_matrix(J, C, b=0):
    state_ks = [12*J,6*J,4*J,0,0,2*J,-2*J,-6*J,4*J,0,4*J,-4*J,-4*J,-12*J,-6*J,-2*J,2*J,0,0,4*J,6*J,12*J]
    state_ks = np.array(state_ks) + np.array(state_ks)/(3*J) *b  # 
    T = np.zeros((22,22))

    for i, k_int_state in enumerate(state_ks):
        for j, k_next_state in enumerate(state_ks):
            T[i,j] = (C[i,j]/8)*sigmoid(state_ks[j] - state_ks[i]) #k estat final menys inicial
            
    np.fill_diagonal(T, [1-sum(row) for row in T])
    
    return T


def gibbs_samp_necker(init_state, burn_in, n_iter, j, stim=0, theta=THETA,
                      save_step=1):
    x_vect = init_state
    states_mat = np.empty((n_iter-burn_in, theta.shape[0]))
    states_mat[:] = np.nan
    for i in range(n_iter):
        node = np.random.choice(np.arange(1, theta.shape[0]+1),
                                p=np.repeat(1/theta.shape[0], theta.shape[0]))
        x_vect1 = np.copy(x_vect)
        x_vect1[node-1] = -x_vect1[node-1]
        prob = change_prob(x_vect, x_vect1, j, stim=stim, theta=theta)
        # val_bool = np.random.choice([False, True], p=[1-prob, prob])
        val_bool = np.random.binomial(1, prob, size=None)
        if val_bool:
            x_vect[node-1] = -x_vect[node-1]
        if i >= burn_in:
            states_mat[i-burn_in, :] = x_vect
    return states_mat


def mean_prob_gibbs(j, ax=None, burn_in = 1000, n_iter = 10000, wsize=100,
                    node=None, stim=0):
    init_state = np.random.choice([-1, 1], 8)
    states_mat = gibbs_samp_necker(init_state=init_state,
                                   burn_in=burn_in, n_iter=n_iter, j=j,
                                   stim=stim)
    # states_mat = (states_mat + 1) / 2
    conv_states_mat = np.copy(states_mat)
    if wsize != 1:
        for i in range(8):
            conv_states_mat[:, i] = np.convolve(conv_states_mat[:, i],
                                                np.ones(wsize)/wsize, mode='same')
    if node is None:
        mean_acc_nodes = np.nanmean(conv_states_mat, axis=1)
    else:
        mean_acc_nodes = conv_states_mat[:, node]
    if ax is not None:
        ax.plot(mean_acc_nodes, label=j)
    if ax is None:
        return mean_acc_nodes


def plot_mean_prob_gibbs(j_list=np.arange(0, 1.05, 0.05), burn_in=1000, n_iter=10000,
                         wsize=1, stim=0, node=None, j_ex=0.8, f_all=False,
                         theta=THETA):
    fig, ax1 = plt.subplots(ncols=2, figsize=(11, 3))
    init_state = np.random.choice([-1, 1], theta.shape[0])
    states_mat = gibbs_samp_necker(init_state=init_state,
                                   burn_in=burn_in, n_iter=n_iter, j=j_ex,
                                   stim=stim, theta=theta)
    mean_states = np.nanmean(states_mat, axis=1)
    # states_mat = np.column_stack((mean_states, states_mat))
    # states_mat = (states_mat+1)/2
    # ticks_labels = [str(i) if i > 0 else r'$<\vec{x}^{\,t}>$' for i in range(9)]
    for i_ax, ax in enumerate(ax1):
        ax_pos = ax.get_position()
        if i_ax == 0:
            ax.set_position([ax_pos.x0, ax_pos.y0-ax_pos.height*0.12,
                             ax_pos.width, ax_pos.height])
            ax_new = fig.add_axes([ax_pos.x0, ax_pos.y0+ax_pos.height*0.92,
                                   ax_pos.width, ax_pos.height/7.5])
        else:
            ax.set_position([ax_pos.x0, ax_pos.y0-ax_pos.height*0.12,
                             ax_pos.width/2, ax_pos.height])
            ax_new = fig.add_axes([ax_pos.x0, ax_pos.y0+ax_pos.height*0.92,
                                   ax_pos.width/2, ax_pos.height/7.5])
        ax_new.plot(mean_states, color='k')
        ax_new.set_xlim(-1, len(mean_states)+1)
        ax_new.set_ylabel(r'$<\vec{x}^{\,t}>$')
        ax_new.set_xticks([])
        ax.imshow(states_mat.T, aspect='auto', cmap='seismic',  # PuOr
                  interpolation='none')
        ax.set_ylabel(r'Node $i$')
        ax.set_xlabel('Sample')
        if theta.shape[0] == 8:
            ax.set_yticks(np.arange(theta.shape[0]), np.arange(theta.shape[0])+1)
        else:
            ax.set_yticks(np.arange(10, theta.shape[0], 10))
    legendelements = [Line2D([0], [0], color='b', lw=2, label=r'$x_i=-1$'),
                      Line2D([0], [0], color='firebrick', lw=2, label=r'$x_i=1$')]
    ax1[0].legend(bbox_to_anchor=(1, 1.25), handles=legendelements,
                  frameon=False)
    fig.savefig(DATA_FOLDER + 'gibbs_simulation_fixed_j_all_nodes' + '.png',
                dpi=400, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'gibbs_simulation_fixed_j_all_nodes' + '.svg',
                dpi=400, bbox_inches='tight')
    if f_all:
        fig, ax = plt.subplots(ncols=1, figsize=(5.5, 3))
        mean_nod = np.empty((len(j_list), n_iter-burn_in))
        mean_nod[:] = np.nan
        allmeans = np.empty((len(j_list)))
        for ind_j, j in enumerate(j_list):
            mean_nod[ind_j, :] = mean_prob_gibbs(j, ax=None, burn_in=burn_in, n_iter=n_iter,
                                                 wsize=wsize, node=node, stim=stim)
            allmeans[ind_j] = np.nanmean(np.abs(mean_nod[ind_j, :]*2-1))
        im = ax.imshow(np.flipud(mean_nod), aspect='auto', cmap='seismic',
                       interpolation='none')
        ax.set_yticks(np.arange(0, len(j_list), len(j_list)//2),
                      j_list[np.arange(0, len(j_list), len(j_list)//2)][::-1])
        ax_pos = ax.get_position()
        ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*1.05, ax_pos.y0+ax_pos.height*0.1,
                                ax_pos.width*0.06, ax_pos.height*0.7])
        # ax_cbar.set_title(r'  $\frac{1}{8}\sum_i^8 {x_i}$')
        ax_cbar.set_title(r'  $<\vec{x}^{\,t}>$')
        plt.colorbar(im, cax=ax_cbar, orientation='vertical')
        ax.set_ylabel(r'Coupling $J$')
        ax.set_xlabel('Sample')
        # ax.plot(j_list, allmeans, color='k')  # np.abs(allmeans*2-1)
        # # ax.plot(j_list, 1-allmeans, color='r')
        # ax.set_xlabel('J')
        # ax.set_ylabel(r'$<P(state \in \{-1, 1\})>_t$', fontsize=10)
        fig.savefig(DATA_FOLDER + 'gibbs_simulation_all' + '.png',
                    dpi=400, bbox_inches='tight')
        fig.savefig(DATA_FOLDER + 'gibbs_simulation_all' + '.svg',
                    dpi=400, bbox_inches='tight')


def plot_duration_dominance_gamma_fit(j, burn_in=1000, n_iter=100000):
    plt.figure()
    vals_gibbs = mean_prob_gibbs(j, ax=None, burn_in=burn_in, n_iter=n_iter,
                                 wsize=1, node=None)
    orders = rle(vals_gibbs)
    time = orders[0][(orders[2] <= 0.05) + (orders[2] >= 0.95)]
    sns.histplot(time, kde=True, label='Simulations', stat='density', fill=False,
                 color='k', bins=np.arange(1, max(time)+10, 2))
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(time)
    fix_loc_exp, fit_lmb = stats.expon.fit(time)
    x = np.linspace(min(time), max(time), 1000)
    y = stats.gamma.pdf(x, a=fit_alpha, scale=fit_beta, loc=1)
    y_exp = stats.expon.pdf(x, loc=fix_loc_exp, scale=fit_lmb)
    plt.text(75, 0.02, r'$\alpha = ${}'.format(np.round(fit_alpha, 2)) 
             + '\n'+ r'$\beta = ${}'.format(np.round(fit_beta, 2)),
             fontsize=11)
    plt.plot(x, y, label='Gamma distro. fit', color='k', linestyle='--')
    plt.plot(x, y_exp, label='Expo. distro. fit', color='r', linestyle='--')
    plt.xlabel('Dominance duration')
    plt.xlim(-5, 105)
    plt.legend()


def rle(inarray):
        """ run length encoding. Partial credit to R rle function. 
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        ia = np.asarray(inarray)                # force numpy
        n = len(ia)
        if n == 0: 
            return (None, None, None)
        else:
            y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            return(z, p, ia[i])


def get_mu(x_vec):
    y = np.copy(x_vec)
    y[4:-1] *= -1
    return np.sum(y)


def get_mu_v2(x_vec):
    return np.sum(x_vec)


def get_analytical_prob(x_vec, j):
    mat = np.abs(mat_theta(x_vec, x_vec, j))
    exponent = k_val(x_vec, mat)
    return np.exp(exponent*j)


def get_mu_from_mat(mat):
    mu_vec = [get_mu(x) for x in mat]
    return np.array(mu_vec)


def get_mu_from_mat_v2(mat):
    mu_vec = [get_mu_v2(x) for x in mat]
    return np.array(mu_vec)


def num_configs():
    return np.array((1, 8, 12, 12, 4, 24, 24, 8,
                     8, 24, 6, 24, 6, 2, # start mu=0
                     8, 24, 24, 4, 12, 12, 8, 1))[::-1]


def mu_states():
    return np.array((-8, -6, -4, -4, -4, -2, -2, -2,
                     0, 0, 0, 0, 0, 0,
                     2, 2, 2, 4, 4, 4, 6, 8))


def plot_nconf_mu_gibbs(classes, clas_values = np.arange(0, 23)):
    plt.figure()
    mu = mu_states()
    nc = num_configs()
    counts, _ = np.histogram(classes, bins=clas_values)
    for i_c, c in enumerate(classes):
        if i_c == 0:
            continue
        plt.plot([mu[classes[i_c-1]], mu[c]], [nc[classes[i_c-1]], nc[c]],
                 color='r', linewidth=0.3)
    for n in range(22):
        plt.plot(mu[n], nc[n], marker='o', linestyle='', color='k',
                 markersize=counts[n]/sum(counts)*30+10)
    plt.ylabel('# configs')
    plt.xlabel(r'$\mu$')


def plot_k_vs_mu(states_mat, j):
    plt.figure()
    mu_vec = np.round(get_mu_from_mat_v2(states_mat), 1)
    klist = []
    for x_vect in states_mat:
        j_mat = np.abs(mat_theta(x_vect, x_vect, j))/j
        k = k_val(x_vect, j_mat)
        klist.append(np.round(k, 2))
    # classes = get_classes(states_mat)
    # for t in range(1, len(klist)):
    #     plt.plot([mu_vec[t-1], mu_vec[t]], [klist[t-1], klist[t]], color='r',
    #              linewidth=C[classes[t-1], classes[t]])
    plt.plot(mu_vec, klist, color='r')
    nc = num_configs() / 256
    arr_conj = np.unique(np.column_stack((mu_vec, klist)), axis=0)
    for n in range(len(arr_conj)):
        ind = np.where((mu_vec == arr_conj[n, 0]) & (np.array(klist) == arr_conj[n, 1]))[0][0]
        state = states_mat[ind]
        classe = check_class(state)
        muval = get_mu_v2(state)
        plt.plot(muval, arr_conj[n, 1], marker='o', linestyle='',
                 markersize=nc[classe]*60+7, color='k')
    plt.ylabel(r'$k = \frac{1}{2} \vec{x}^T \theta_{ij} \vec{x}$', fontsize=12)
    plt.xlabel(r'$\mu$', fontsize=12)


def plot_k_vs_mu_analytical(stim=0, eps=6e-2, plot_arist=False, plot_cubes=False):
    nc = np.sum(np.sign(np.copy(C) / 8), axis=1) / 8
    nc = num_configs() / 8
    combs = list(itertools.product([-1, 1], repeat=8))
    combs = np.array(combs, dtype=np.float64)
    class_count = []
    klist = []
    muvec = []
    for i_x, x_vec in enumerate(combs):
        x_vec = np.array(x_vec, dtype=np.float64)
        classes = check_class(x_vec)
        k = k_val(x_vec, THETA, stim=stim)
        if classes in class_count:
            continue
        else:
            muvec.append(get_mu_v2(x_vec))
            klist.append(k)
            class_count.append(classes)
        if len(class_count) == 22:
            break            
    fig, ax = plt.subplots(1, figsize=(5, 4))
    if plot_arist:
        for ic, cl in enumerate(class_count):
            for ic2, cl2 in enumerate(class_count):
                plt.plot([muvec[ic]+eps, muvec[ic2]+eps], [klist[ic]+eps, klist[ic2]+eps], color='r',
                         linewidth=np.sign(C[class_count[ic], class_count[ic]])/2)
                plt.plot([muvec[ic]-eps, muvec[ic2]-eps], [klist[ic]-eps, klist[ic2]-eps], color='r',
                         linewidth=np.sign(C[class_count[ic2], class_count[ic]])/2)
    msize = 4
    for i_c, classe in enumerate(class_count):
        # if plot_arist:
        msize = nc[classe]*2+4
        plt.plot(muvec[i_c], klist[i_c], marker='o', linestyle='', color='k',
                 markersize=msize)  # nc[classe]*55+
    # plt.annotate(text='', xy=(0, 0.5), xytext=(0, 3.5),
    #              arrowprops=dict(arrowstyle='<->'))
    # plt.text(-1, 1.5, r'$\Delta k_u$')
    plt.ylabel(r'$k=\frac{J}{2} \, \vec{y}^{\, T} \, \mathcal{V}_{ij} \, \vec{y} + B \mu$', fontsize=12)
    plt.xlabel(r'$\mu$', fontsize=12)
    # plt.title('B = {}'.format(stim))
    if plot_cubes:
        plot_necker_cubes(ax=ax, mu=-8)
        plot_necker_cubes(ax=ax, mu=-6)
        plot_necker_cubes(ax=ax, mu=8)
        plot_necker_cubes(ax=ax, mu=0, bot=False)
        plot_necker_cubes(ax=ax, mu=0)
    if plot_arist:
        plt.plot([-8, -2], [2, 2], color='k', linestyle='--', alpha=0.4)
        plt.plot([8, 2], [2, 2], color='k', linestyle='--', alpha=0.4)
        x_vec_1 = np.repeat(-1, 8)
        k1 = k_val(x_vec_1, THETA, stim=stim)
        x_vec_2 = np.repeat(1, 8)
        k2 = k_val(x_vec_2, THETA, stim=stim)
        plt.annotate(text='', xy=(-8, k1-0.5), xytext=(-8, 2),
                     arrowprops=dict(arrowstyle='<->'))
        plt.text(-7.6, 4, r'$\Delta k_1$')
        plt.annotate(text='', xy=(8, k2-0.5), xytext=(8, 2),
                     arrowprops=dict(arrowstyle='<->'))
        plt.text(6.2, 4, r'$\Delta k_2$')
        plt.xticks([-8, -4, 0, 4, 8])
    else:
        plt.xticks([-8, -4, 0, 4, 8])
    fig.savefig(DATA_FOLDER + 'k_vs_mu.png', dpi=400, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'k_vs_mu.svg', dpi=400, bbox_inches='tight')
    

def compute_C(data_folder):
    combs = list(itertools.product([-1, 1], repeat=8))
    combs = np.array(combs, dtype=np.float64)
    c_mat = np.zeros((22, 22))
    class_count = []
    for i, x_vec in enumerate(combs):
        for j, x_vec_2 in enumerate(combs):
            cl1 = check_class(x_vec)
            if cl1 not in class_count:
                cl2 = check_class(x_vec_2)
                if sum(x_vec * x_vec_2 == -1) == 1:
                    c_mat[cl1, cl2] += 1
    for j in range(22):
        c_mat[j, :] = c_mat[j, :] / np.sum(c_mat[j, :]) * 8
    np.save(data_folder + 'c_mat.npy', c_mat)


def comptue_C_with_vecs():
    init_vec = np.repeat(1, 8)
    # class_0 = check_class(init_vec)
    c_mat = np.zeros((22, 22))
    for cl in range(22):
        pos_vecs = np.empty((8, 8))
        for j in range(8):
            vec_1 = np.copy(init_vec)
            vec_1[j] = -vec_1[j]
            pos_vecs[j, :] = vec_1
            cl_1 = check_class(vec_1)
            c_mat[cl, cl_1] += 1
        init_ind = np.random.choice(8)
        init_vec = pos_vecs[init_ind]


def check_class(x_vec):
    mat_th = np.sum(np.triu(mat_theta(x_vec, x_vec, 1)))
    mat = mat_theta(x_vec, x_vec, 1)
    if np.sum(x_vec) == 8:
        return 0
    if np.sum(x_vec) == -8:
        return 21
    if np.sum(x_vec) == -6:
        return 20
    if np.sum(x_vec) == 6:
        return 1
    if np.sum(x_vec) == -4 and (mat_th == -4 or mat_th == 4):
        return 19
    if np.sum(x_vec) == 4 and (mat_th == 4 or mat_th == -4):
        return 2
    if np.sum(x_vec) == -4 and mat_th == 0:
        if (np.sum(mat, axis=0) == 3).any():
            return 18
        else:
            return 17
    if np.sum(x_vec) == 4 and mat_th == 0:
        if (np.sum(mat, axis=0) == 3).any():
            return 3
        else:
            return 4
    if np.sum(x_vec) == -2 and mat_th == -2:
        return 16
    if np.sum(x_vec) == 2 and mat_th == 2:
        return 5
    if np.sum(x_vec) == -2 and (mat_th == -2 or mat_th == 2):
        return 15
    if np.sum(x_vec) == 2 and (mat_th == -2 or mat_th == 2):
        return 6
    if np.sum(x_vec) == -2 and (mat_th == -6 or mat_th == 6):
        return 14
    if np.sum(x_vec) == 2 and (mat_th == -6 or mat_th == 6):
        return 7
    if np.sum(x_vec) == 0:
        if mat_th == 0:
            if (np.sum(mat, axis=0) == 3).any():
                return 8
            else:
                return 9
        if mat_th == 4:
            return 10
        if mat_th == -4:
            if (np.sum(mat, axis=0) == -3).any():
                return 11
            else:
                return 12
        if mat_th == -12:
            return 13


def get_classes(x_vec):
    classes = np.array([check_class(x) for x in x_vec])
    return classes


def plot_probs_gibbs(data_folder, j_list=np.round(np.arange(0, 1, 0.0005), 4)):
    # j_list = np.arange(0, 1, 0.01)
    # j_list = np.round(np.arange(0, 1, 0.0005), 4) 
    # j_list = [0, 0.25, 0.7, 0.9]   
    # fig, ax = plt.subplots(nrows=2, ncols=2)
    # ax = ax.flatten()
    # figmu, axmu = plt.subplots(ncols=len(j_list))
    init_state = np.random.choice([-1, 1], 8)
    # init_state = np.random.uniform(0, 1, 8)
    burn_in = 5000
    n_iter = 50000
    probs = np.empty((len(j_list), 22))
    clas_values = np.arange(0, 23)
    probs_data = data_folder + 'probsmat.npy'
    os.makedirs(os.path.dirname(probs_data), exist_ok=True)
    if os.path.exists(probs_data):
        probs = np.load(probs_data, allow_pickle=True)
    else:
        for j_ind, j in enumerate(j_list):
            if j_ind % 10 == 0:
                print('J = ' + str(j) + ' , {}%'.format(j_ind / len(j_list)))
            # j = 0.3
            states_mat = gibbs_samp_necker(init_state=init_state,
                                           burn_in=burn_in, n_iter=n_iter, j=j)
            # mu_vec = get_mu_from_mat(states_mat)
            # possible_states, states_count = np.unique(states_mat, axis=0, return_counts=True)
            # possible_mu, mu_count = np.unique(mu_vec, return_counts=True)
            # axmu[j_ind].plot(possible_mu, mu_count / (n_iter-burn_in), marker='o', linestyle='')
            # axmu[j_ind].set_title('J = ' + str(j))
            # axmu[j_ind].set_xlabel(r'$\mu$')
            # axmu[j_ind].set_ylabel(r'$P(\mu)$')
            classes = get_classes(states_mat)
            # possible_states, states_count = np.unique(classes, return_counts=True)
            states_count, _ = np.histogram(classes, bins=clas_values)
            probs[j_ind, :] = states_count / (n_iter-burn_in)
            # ax[j_ind].plot(possible_states, states_count / (n_iter-burn_in), marker='o', linestyle='')
            # ax[j_ind].set_title('J = ' + str(j))
            # ax[j_ind].set_xlabel(r'state')
            # ax[j_ind].set_ylabel(r'prob')
        np.save(data_folder + 'probsmat_binom.npy', probs)
    plt.figure()
    im = plt.imshow(np.flipud(probs), aspect='auto', cmap='inferno', vmax=1)
    plt.yticks(np.arange(0, len(j_list), len(j_list)/10),
               j_list[np.arange(0, len(j_list), len(j_list)//10)][::-1])
    plt.ylabel(r'J')
    plt.xticks(clas_values[:-1])
    plt.xlabel('States')
    plt.colorbar(im, fraction=0.04, shrink=0.6, orientation='vertical',
                 label='State prob.')
    plt.title('Simulation')


def c_vs_mu(j_list, ax, data_folder):
    init_state = np.random.choice([-1, 1], 8)
    # init_state = np.random.uniform(0, 1, 8)
    burn_in = 5000
    n_iter = 50000
    cmat = np.empty((len(j_list), 22))
    clas_values = np.arange(0, 23)
    cmat_data = data_folder + 'c_mat.npy'
    os.makedirs(os.path.dirname(cmat_data), exist_ok=True)
    if os.path.exists(cmat_data):
        probs = np.load(cmat_data, allow_pickle=True)
    else:
        for j_ind, j in enumerate(j_list):
            if j_ind % 10 == 0:
                print('J = ' + str(j) + ' , {}%'.format(j_ind / len(j_list)))
            # j = 0.3
            states_mat = gibbs_samp_necker(init_state=init_state,
                                           burn_in=burn_in, n_iter=n_iter, j=j)
            mu_vec = get_mu_from_mat(states_mat)
            # possible_states, states_count = np.unique(states_mat, axis=0, return_counts=True)
            # possible_mu, mu_count = np.unique(mu_vec, return_counts=True)
            # axmu[j_ind].plot(possible_mu, mu_count / (n_iter-burn_in), marker='o', linestyle='')
            # axmu[j_ind].set_title('J = ' + str(j))
            # axmu[j_ind].set_xlabel(r'$\mu$')
            # axmu[j_ind].set_ylabel(r'$P(\mu)$')
            classes = get_classes(states_mat)
            # possible_states, states_count = np.unique(classes, return_counts=True)
            states_count, _ = np.histogram(classes, bins=clas_values)
            cmat[j_ind, :] = states_count / (n_iter-burn_in)
            # ax[j_ind].plot(possible_states, states_count / (n_iter-burn_in), marker='o', linestyle='')
            # ax[j_ind].set_title('J = ' + str(j))
            # ax[j_ind].set_xlabel(r'state')
            # ax[j_ind].set_ylabel(r'prob')
        np.save(data_folder + 'c_mat.npy', cmat)


def get_analytical_probs_all(j_list):
    analytical_probs = np.zeros((len(j_list), 22))
    combs = list(itertools.product([-1, 1], repeat=8))
    combs = np.array(combs, dtype=np.float64)
    for j_ind, j in enumerate(j_list):
        cte = np.zeros((22))
        for i_x, x_vec in enumerate(combs):
            x_vec = np.array(x_vec, dtype=np.float64)
            classes = check_class(x_vec)
            cte[classes] += 1
        class_count = []
        for i_x, x_vec in enumerate(combs):
            x_vec = np.array(x_vec, dtype=np.float64)
            classes = check_class(x_vec)
            if classes in class_count:
                continue
            else:
                class_count.append(classes)
                prob_an = get_analytical_prob(x_vec, j)*cte[classes]
                analytical_probs[j_ind, classes] = prob_an
        cte_norm = np.sum(analytical_probs[j_ind, :])
        analytical_probs[j_ind, :] /= cte_norm
    return analytical_probs


def plot_analytical_prob(data_folder, j_list = np.round(np.arange(0, 1, 0.0005), 4)):
    analytical_probs_data =\
        data_folder + 'probsmat_analytical.npy'
    clas_values = np.arange(0, 23)
    os.makedirs(os.path.dirname(analytical_probs_data), exist_ok=True)
    if os.path.exists(analytical_probs_data):
        analytical_probs = np.load(analytical_probs_data, allow_pickle=True)
    else:
        analytical_probs = get_analytical_probs_all(j_list)
            # cte[classes] += prob_an
        np.save(data_folder + 'probsmat_analytical.npy',
                analytical_probs)
    fig, ax = plt.subplots(1)
    im = ax.imshow(np.flipud(analytical_probs), aspect='auto', cmap='inferno',
                   vmax=1)
    ax.set_yticks(np.arange(0, len(j_list), len(j_list)/10),
                  j_list[np.arange(0, len(j_list), len(j_list)//10)][::-1])
    ax.set_ylabel(r'J')
    ax.set_xticks(clas_values[:-1])
    ax.set_xlabel('States')
    plt.colorbar(im, fraction=0.04, shrink=0.6, orientation='vertical',
                 label='State prob.', ax=ax)
    ax.set_title('Analytical solution')


def tanh_act_bistab(n_iter, weight_prev_state, weight_noise, stim_state):
    # SIMPLEST MODEL FOR BISTABILITY INDEP. OF STIM.
    state = [0.5]
    vals = [0.5]
    dsdt = np.gradient(stim_state)
    for j in range(1, n_iter):
        x = np.random.randn()*weight_noise\
            + weight_prev_state*state[j-1] + dsdt[j]
        val = sigmoid(x)  # x/np.sqrt(1+x**2)
        vals.append(val)
        state.append(np.sign(val))
    return state, vals


def plot_prob_basic_model_coupling(n_iter, wpslist=np.linspace(0, 3, 100),
                                   stim_state=1, weight_noise=0):
    """
    Stim. independent bistability generation. Single neuron with recurrent connection
    and external noise.
    P_{t+1} = (P_t) · w + w_n · N(0, 1) + dS/dt , --> dS/dt = 0
    P_{t+1} = F(S, P_t) = g(S) + f(P_t) --> S cte. --> offset (baseline) cte
    
    if w_n = 0 --> mean field model
    """
    fig, ax = plt.subplots(ncols=1)
    stim_state = np.repeat(stim_state, n_iter)
    valslist = []
    stdlist = []
    for iw, weight_prev_state in enumerate(wpslist):
        _, vals = tanh_act_bistab(n_iter=n_iter,
                                  weight_prev_state=weight_prev_state,
                                  weight_noise=weight_noise, stim_state=stim_state)
        valslist.append(vals[-1])  # np.abs(vals)
        vals = np.array(vals)
        stdlist.append(np.nanstd((vals+1)/2))
        # vals = np.round(vals, 5)
        # plt.plot(np.repeat(iw, len(np.unique(vals))), np.unique(vals), color='k',
        #          marker='o', linestyle='', markersize=1)
    indw = \
        np.where(np.abs(np.array(valslist)-0.5) ==
                 np.min(np.abs(np.array(valslist)-0.5)))[0][0]
    ax.axvline(wpslist[indw], color='grey', linestyle='--')
    ax.text(1.5, 0.5, r'$w_{change} =$' + str(np.abs(np.round(wpslist[indw], 2))))
    # ax[1].axvline(wpslist[indw], color='grey', linestyle='--')
    ax.plot(wpslist, valslist, color='k')
    ax.plot(wpslist, 1-np.array(valslist), color='r')
    ax.set_ylabel('Prob. x=1')
    ax.set_xlabel(r'Coupling strength, $w$')
    ax.set_title(r'$P_{t+1} = tanh(P_t * w + \xi), \;\; \xi \sim \mathcal{N}(\mu=0, \sigma=1)$')
    # ax[1].set_xlabel(r'Coupling strength, $w$')
    # ax[1].set_ylabel('Std(P(x=1))')
    # ax[1].plot(wpslist, stdlist, color='k')


def prob_markov_chain_between_states(n_iter_list=np.logspace(0, 4, 5),
                                     tau=8000, iters_per_len=1000):
    # init_state = np.random.choice([-1, 1])
    # tau equivalent to J=?, tau=8000 equivalent to J=1
    eps = 1-np.exp(-1/tau)
    ps = 1-eps
    pt = eps
    mu_vals_all = np.empty((len(n_iter_list), iters_per_len))
    mu_vals_all[:] = np.nan
    for i_n, n_iter in enumerate(n_iter_list):
        mu_list = []
        for j in range(iters_per_len):
            transition_matrix = np.array(((ps, pt), (pt, ps)))
            init_state = np.random.choice([0, 1])            
            chain = [init_state]
            # p_list = []
            for i in range(1, int(n_iter)):
                if chain[i-1] == 0:
                    change = np.random.choice([0, 1], p=transition_matrix[0])
                    # p_list.append(transition_matrix[1][change])
                if chain[i-1] == 1:
                    change = np.random.choice([0, 1], p=transition_matrix[1])
                    # p_list.append(transition_matrix[1][change])
                chain.append(change)
            mu_list.append(np.mean(chain))
        mu_vals_all[i_n, :] = mu_list
    # dict_pd = {}
    # for i_n, n_iter in enumerate(n_iter_list):
    #     dict_pd[str(n_iter)] = mu_vals_all[i_n, :]
    # df = pd.DataFrame(dict_pd)
    colormap = pl.cm.Blues(np.linspace(0.2, 1, len(n_iter_list)))
    fig, ax = plt.subplots(1)
    f2, ax2 = plt.subplots(1)
    sdlist = []
    sdlist_beta = []
    for j in range(len(n_iter_list)):
        # ax2 = ax.twinx()
        vals = np.random.beta((n_iter_list[j])/tau,
                              (n_iter_list[j])/tau,
                              iters_per_len)
        sns.kdeplot(vals, label='beta: ' + str(n_iter_list[j]),
                    color=colormap[j], linestyle='--', ax=ax2)
        sdlist.append(np.nanstd(mu_vals_all[j, :]))
        # sdlist_norm.append(1/(4*(2*(n_iter_list[j])/tau+1)))
        sdlist_beta.append(np.nanstd(vals))
        sns.kdeplot(mu_vals_all[j, :],
                    color=colormap[j],
                    common_norm=False,
                    ax=ax, bw_adjust=1,
                    label=str(n_iter_list[j]))
        # ax2.spines['right'].set_visible(False)
        # ax2.spines['top'].set_visible(False)
    ax.legend(title='N')
    for a in [ax, ax2]:
        a.set_xlim(-0.05, 1.05)
        a.axvline(0.5, color='r', linestyle='--', alpha=0.4)
        a.set_xlabel('')
    ax2.set_title('Beta')
    # p_N(mu) = p_N(mu, x_N=0) + p_N(mu, x_N=1)
    # p_N+1(mu, x_N=1) = p_N(mu-1, x_N=1)*(1-eps) + p_N(mu-1, x_N=0)*eps


def magnetization_2d(J):
    return (1-np.sinh(2*J*np.sinh(2*J))**(-2))**(1/8)


def calc_exponents(x_vects_1, x_vects_0, j, stim, j_mat):
    exponent_1 = 0
    exponent_0 = 0
    for x_vec_1, x_vec_0 in zip(x_vects_1, x_vects_0):
        x_vec_1 = np.array(x_vec_1)
        x_vec_0 = np.array(x_vec_0)
        # x_vec_0[4:] *= -1
        # x_vec_1[4:] *= -1
        exponent_1 += np.exp(0.5*j*np.matmul(np.matmul(x_vec_1.T, j_mat), x_vec_1) + np.sum(stim*x_vec_1))
        exponent_0 += np.exp(0.5*j*np.matmul(np.matmul(x_vec_0.T, j_mat), x_vec_0) + np.sum(stim*x_vec_0))
    return exponent_1, exponent_0


def true_posterior(theta=THETA, j=0.1, stim=0):
    combs_7_vars = list(itertools.product([-1, 1], repeat=theta.shape[0]-1))
    x_vects_1 = [combs_7_vars[i] + (1,) for i in range(len(combs_7_vars))]
    x_vects_0 = [combs_7_vars[i] + (-1,) for i in range(len(combs_7_vars))]
    j_mat = theta
    # len_xv1 = len(x_vects_1)
    # exponent_1, exponent_0 =\
    #     Parallel(n_jobs=n_jobs)(delayed(calc_exponents)(x_vects_1[step*i:step*(i+1)], x_vects_0[step*i:step*(i+1)], j, stim, j_mat) for i in range(n_jobs))
    exponent_1, exponent_0 = calc_exponents(x_vects_1, x_vects_0, j, stim, j_mat)
    prob_x_1 = (exponent_1) / ((exponent_1)+(exponent_0))
    return prob_x_1


def sol_magnetization_hex_lattice(j_list, b):
    sol = []
    alpha = 6
    for j in j_list:
        numerator = np.exp(2*j*alpha)*np.cosh(b/2*alpha) - np.cosh(b/6*alpha)
        denominator = np.exp(2*j*alpha)*np.cosh(b/2*alpha) + 3*np.cosh(b/6*alpha)
        sol.append(np.sqrt(numerator/denominator))
    plt.plot(j_list, sol)
    


def true_posterior_stim(stim_list=np.linspace(-2, 2, 1000), j=0.5,
                        theta=THETA, data_folder=DATA_FOLDER, load_data=False,
                        save_data=True):
    posterior_stim = data_folder + str(theta.shape[0]) + '_' + str(j) + '_post_stim.npy'
    os.makedirs(os.path.dirname(posterior_stim), exist_ok=True)
    if os.path.exists(posterior_stim) and load_data:
        post_stim = np.load(posterior_stim, allow_pickle=True)
    else:
        post_stim = []
        for stim in stim_list:
            post = true_posterior(theta=theta, j=j, stim=stim)
            post_stim.append(post)
        if save_data:
            np.save(data_folder + 'post_stim.npy', np.array(post_stim))    
    return post_stim
    # plt.plot(stim_list, post_stim)
    

def true_posterior_plot_j(ax, color='b', j_list=np.arange(0.001, 1, 0.001), stim=0):
    post_stim = []
    for j in j_list:
        post = true_posterior(theta=THETA, j=j, stim=stim)
        post_stim.append(post)
    ax.plot(j_list, post_stim, color=color)


# def occupancy_distro(state, ps, k, n=1, m=1000):
#     # ps: prob of stay
#     p_kn_k = scipy.special.binom(m, k)
#     val = 0
#     for j in range(k):
#         val += scipy.special.binom(k, j)*((-1)**(j-k))*(1-ps*(m-j)/m)**n
#     return p_kn_k*val



def plot_cylinder_true_posterior(j, stim, theta=THETA):
    q = np.repeat(true_posterior(theta=theta, j=j, stim=stim), 8).reshape(2, 2, 2)
    plot_cylinder(q=q, columns=2, rows=2, layers=2, offset=0.4, minmax_norm=False)


def plot_cylinder(q=None, states=False, columns=5, rows=10, layers=2, offset=0.4, minmax_norm=False,
                  save_fig=False, n_fig=0):
    fig, ax = plt.subplots(1, figsize=(5, 10))
    nodes = np.zeros((rows, columns, layers))
    if q is None:
        q = np.copy(nodes)
    q_0 = q[:, :, 0].flatten()
    q_1 = q[:, :, 1].flatten()
    if minmax_norm:
        colormap_array_0 = (q_0-np.min(q_0))/(np.max(q_0)-np.min(q_0))
        colormap_array_1 = (q_1-np.min(q_1))/(np.max(q_1)-np.min(q_1))
    else:
        colormap_array_0 = q_0
        colormap_array_1 = q_1
    colormap_back = pl.cm.copper(colormap_array_0)
    colormap_front = pl.cm.copper(colormap_array_1)
    if states:
        colormap_back = pl.cm.bwr(colormap_array_0)
        colormap_front = pl.cm.bwr(colormap_array_1)
    x_nodes_front = nodes[:, :, 0] + np.arange(columns)
    y_nodes_front = (nodes[:, :, 0].T + np.arange(rows)).T
    x_nodes_back = nodes[:, :, 1] + np.arange(columns) + offset
    y_nodes_back = (nodes[:, :, 1].T + np.arange(rows)).T + offset
    for i in range(rows):
        for j in range(columns):
            if j % columns == 0 or j == (columns-1):
                ax.plot([x_nodes_front[i, j], x_nodes_back[i, j]],
                        [y_nodes_front[i, j], y_nodes_back[i, j]],
                        color='grey')
            if (j+1) < columns:
                ax.plot([x_nodes_front[i, j], x_nodes_front[i, j+1]],
                        [y_nodes_front[i, j], y_nodes_front[i, j+1]],
                        color='grey')
                ax.plot([x_nodes_back[i, j], x_nodes_back[i, j+1]],
                        [y_nodes_back[i, j], y_nodes_back[i, j+1]],
                        color='grey')
            if (i+1) < rows:
                ax.plot([x_nodes_front[i, j], x_nodes_front[i+1, j]],
                        [y_nodes_front[i, j], y_nodes_front[i+1, j]],
                        color='grey')
                ax.plot([x_nodes_back[i, j], x_nodes_back[i+1, j]],
                        [y_nodes_back[i, j], y_nodes_back[i+1, j]],
                        color='grey')
    i = 0
    for x_b, y_b, x_f, y_f in zip(x_nodes_back.flatten(), y_nodes_back.flatten(),
                                  x_nodes_front.flatten(), y_nodes_front.flatten()):
        ax.plot(x_b, y_b, marker='o', linestyle='', color=colormap_back[i],
                markersize=8)
        ax.plot(x_f, y_f, marker='o', linestyle='', color=colormap_front[i],
                markersize=8)
        i += 1
    if np.sum(q) != 0 or states:
        ax_pos = ax.get_position()
        ax.set_position([ax_pos.x0-ax_pos.width*0.1, ax_pos.y0,
                         ax_pos.width, ax_pos.height])
        ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*0.92, ax_pos.y0+ax_pos.height*0.2,
                                ax_pos.width*0.06, ax_pos.height*0.5])
        mpl.colorbar.ColorbarBase(ax_cbar, cmap='copper')
        if states:
            mpl.colorbar.ColorbarBase(ax_cbar, cmap='bwr')
        ax_cbar.set_title('J')
        ax_cbar.set_yticks([0, 0.5, 1], [np.round(np.min(q), 4),
                                         np.round((np.min(q)+np.max(q))/2, 4),
                                         np.round(np.max(q), 4)])
    # ax.plot(x_nodes_back, y_nodes_back, marker='o', linestyle='', color='k',
    #         markersize=8)
    # ax.plot(x_nodes_front, y_nodes_front, marker='o', linestyle='', color='k',
    #         markersize=8)
    ax.axis('off')
    if save_fig:
        fig.savefig(DATA_FOLDER + "/gibbs_video_0495_zoom/" + str(n_fig) + '.png',
                    dpi=100)


def plot_states_cylinder(j_ex, stim, n_iter=201000, theta=THETA, burn_in=1000):
    init_state = np.random.choice([-1, 1], theta.shape[0])
    states_mat = gibbs_samp_necker(init_state=init_state,
                                   burn_in=burn_in, n_iter=n_iter, j=j_ex,
                                   stim=stim, theta=theta)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    # mu = np.sum(states_mat, axis=1)
    plt.imshow(states_mat.T, cmap='bwr', aspect='auto')
    fig.savefig(DATA_FOLDER + "/gibbs_video_0495_zoom/" + 'whole_sim.png',
                dpi=100)
    n_fig = 0
    index_min = int(input('index min: '))  # 0
    index_max = int(input('index max: '))  # 50000
    step = int(input('step: '))  # 100
    for i in np.arange(index_min, index_max, step):
        # save each 1 iterations, total of 5000 images
        plot_cylinder(q=states_mat[i].reshape(5, 10, 2),
                      states=True, columns=5, rows=10, layers=2, offset=0.4, minmax_norm=False,
                      save_fig=True, n_fig=n_fig)
        plt.close('all')
        n_fig += 1


def video_create(image_folder=DATA_FOLDER + '/gibbs_video_0495_zoom/',
                 fps=50, videoname='cylinder_0495.mp4'):
    video_name = image_folder + videoname
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = [images[i].replace('.png', '') for i in range(len(images))]
    images.sort(key=float)
    images = [images[i] + '.png' for i in range(len(images))]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'),
                            fps, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()


def plot_tau_T_mat(C, ax=None, color='k',
                   j_list=np.arange(0.0001, 1.01, 0.01), b=0):
    if ax is None:
        fig, ax = plt.subplots(1)
        ax.set_xlabel('J')
        ax.set_ylabel(r'$\tau$')
    eig_vals = []
    for j in j_list:
        tmat = transition_matrix(j, C, b)
        eig_val = np.sort(np.linalg.eig(tmat)[0])[-2]
        eig_vals.append(-1/np.log(eig_val))
    ax.plot(j_list, eig_vals, color=color, label=b)


def plot_tau_vs_J_changing_B(C, b_list=[0, 0.1, 0.2]):
    fig, ax = plt.subplots(1)
    ax.set_xlabel('J')
    ax.set_ylabel(r'$\tau$')
    colors = pl.cm.copper(np.linspace(0, 1, len(b_list)))
    for i_b, b in enumerate(b_list):
        plot_tau_T_mat(C, ax=ax, color=colors[i_b],
                           j_list=np.arange(0.001, 1.01, 0.01), b=b)
    ax.legend()
    ax.set_yscale('log')


def occ_markov_discrete(p1, p2, n, x):
    # eps = 1-np.exp(-1/tau)
    # ps = 1-eps
    # pt = eps
    # p1 = ps
    q1 = 1-p1
    # p2 = ps
    q2 = 1-p2
    p01 = 0.5
    p02 = 0.5
    f = scipy.special.hyp2f1
    lamb = q1*q2 / (p1*p2)
    d = p1 - q2
    p_n = (p1 ** x) * (p2 ** (n-x)) * f(-n+x, -x, 1, lamb) - \
        p01*d*(p1 ** x) * (p2 ** (n-x-1)) * f(-n+x+1, -x, 1, lamb) - \
        p02*d*(p1 ** (x-1)) * (p2 ** (n-x)) * f(-n+x, -x+1, 1, lamb)
    # plt.plot(x/n, p_n)
    return p_n


def occ_function_markov(a, b, t, x):
    p1 = 0.5
    p2 = 0.5
    def i_n(n, x):
        return np.exp(x)*(1+(1-4*n**2)/(8*x)) / (2*np.pi*x)
    first_exp = np.exp(-a*x - b*(t-x))
    first_comp = p2*0**(x) + p1*0**(t-x)
    x_arg = 2*np.sqrt(a*b*x*(t-x))
    # i_0 = i_n(0, x_arg)
    # i_1 = i_n(1, x_arg)
    i_0 = scipy.special.i0(x_arg)
    i_1 = scipy.special.i1(x_arg)
    second_comp = (p1*np.sqrt(a*b*x/(t-x)) + p2*np.sqrt(a*b*(t-x)/x))*i_1
    third_comp = (p1*a+p2*b)*i_0
    eq_1 = first_exp*(second_comp + third_comp)
    eq_2 = first_exp*first_comp
    eq = np.nansum((eq_1, eq_2), axis=0)*t
    return  eq


def occ_function_markov_ch_var(rho, alpha, t, x):
    p1 = 0.5
    p2 = 0.5
    # rho = np.sqrt(a/b)
    # alpha = np.sqrt(a*b)
    t_p = t*alpha
    p = x/t
    first_exp = np.exp(-rho*p*t_p - t_p*(1-p)/rho)
    first_comp = (p2*0**(p) + p1*0**(1-p))*1e5  # 1e10 is just to illustrate the delta distro...
    x_arg = 2*np.sqrt(p*(1-p)*t_p*t_p)
    i_0 = scipy.special.i0(x_arg)
    i_1 = scipy.special.i1(x_arg)
    second_comp = (p1*np.sqrt(p/(1-p)) + p2*np.sqrt((1-p)/p))*i_1
    third_comp = (p1*rho+p2/rho)*i_0
    eq_1 = first_exp*(second_comp + third_comp)
    eq_2 = first_exp*first_comp
    eq = np.nansum((eq_1, eq_2), axis=0)*t_p
    return eq


def plot_distro_markov_j(t_list, stim=0, j=1):
    plt.figure()
    k_2 = 12*j + 8*stim
    k_1 = 12*j - 8*stim
    k_u = 2*j
    rho = np.exp(-(k_1-k_2)/2)
    alpha = np.exp(-(k_1+k_2 - k_u*2)/2)
    for t in t_list:
        x = np.arange(0, t+1, 1)
        vals = occ_function_markov_ch_var(rho, alpha, t, x)
        plt.plot(x/t, vals, label=t)
    plt.legend()


def plot_distro_markov_discrete(n_list, stim=0, j=1):
    k_1 = 12*j - 8*stim
    k_u = 2*j
    k_2 = 12*j + 8*stim
    p1 = 1-np.exp(-k_1+k_u)
    p2 = 1-np.exp(-k_2+k_u)
    plt.figure()
    for n in n_list:
        x = np.arange(1, n, 1)
        vals = occ_markov_discrete(p1, p2, n, x)
        plt.plot(x/n, vals, label=n)
    plt.legend(title='N')



def trans_probs_well_height(theta, j, stim=0, burn_in=1000, n_iter=21000):
    init_state = np.random.choice([-1, 1], 8)
    states_mat = gibbs_samp_necker(init_state=init_state,
                                   burn_in=burn_in, n_iter=n_iter, j=j,
                                   stim=stim)
    # states_mat = (states_mat + 1) / 2
    # k_1 = k_val(np.repeat(-1, 8), theta*j, stim)
    k_1 = 12*j - 8*stim
    k_u = 2*j
    # k_2 = k_val(np.repeat(1, 8), theta*j, stim)
    k_2 = 12*j + 8*stim
    prod_sigms = sigmoid(-12*j+6*j)*sigmoid(-6*j+4*j)*sigmoid(-4*j+2*j)  # *sigmoid(-2*j+4*j)
    curr_state = np.sign(get_mu_from_mat_v2(states_mat))
    trans_pos_neg = 0
    trans_neg_pos = 0
    curr_state = curr_state[curr_state != 0]
    change_signs = np.diff(curr_state)
    trans_neg_pos = sum(change_signs == 2)/len(states_mat)
    trans_pos_neg = sum(change_signs == -2)/len(states_mat)
    return trans_neg_pos, trans_pos_neg, np.exp(-k_1+k_u), np.exp(-k_2+k_u), prod_sigms
    # for i_st in range(len(curr_state)):
    #     if curr_state[i_st] == 0:
    #         if curr_state[i_st-1] == curr_state[i_st+1]:
    #             continue
    #         if curr_state[i_st] == curr_state[i_st+1]:
    #             continue
    #         if curr_state[i_st] == curr_state[i_st-1]:
    #             continue
    #         if curr_state[i_st-1] == -1 and curr_state[i_st+1] == 1:
    #             trans_neg_pos += 1
    #             continue
    #         if curr_state[i_st-1] == 1 and curr_state[i_st+1] == -1:
    #             trans_pos_neg += 1
    #             continue
    # trans_neg_pos = trans_neg_pos / (n_iter-burn_in)
    # trans_pos_neg = trans_pos_neg / (n_iter-burn_in)


def plot_trans_probs_comparison(j_list, theta, stim):
    p1l = []
    p2l = []
    pt1l = []
    pt2l = []
    psgsl = []
    for j in j_list:
        p1, p2, pt1, pt2, psigs =\
            trans_probs_well_height(theta, j, stim=stim, burn_in=1000, n_iter=21000)
        p1l.append(p1)
        p2l.append(p2)
        pt1l.append(pt1)
        pt2l.append(pt2)
        psgsl.append(psigs)
    plt.figure()
    plt.plot(j_list, p1l, label='p_t1')
    plt.plot(j_list, p2l, label='p_t2')
    plt.plot(j_list, pt1l, label='exp(-k_1 + k_u)', linestyle='--')
    plt.plot(j_list, pt2l, label='exp(-k_2 + k_u)', linestyle='--')
    plt.plot(j_list, psgsl, label='p_sigs', linestyle=':')
    plt.ylabel(r'$p_t$')
    plt.xlabel('J')
    plt.legend()
    tau_vals_theory_psgsl = 1/-np.log(1-np.array(psgsl))
    plt.figure()
    plt.plot(j_list[1:], tau_vals_theory_psgsl[1:], label='Tau theory sigmoid prod')
    tau_vals_theory = []
    eig_vals = []    
    for j in j_list[1:]:
        k_1 = 12*j - 8*stim
        k_u = 2*j
        tau_vals_theory.append(-1/np.log(1-np.exp(-k_1+k_u)))
        tmat = transition_matrix(j, C, stim)
        eig_val = np.sort(np.linalg.eig(tmat)[0])[-2]
        eig_vals.append(-1/np.log(eig_val))
    plt.plot(j_list[1:], tau_vals_theory, label='Tau theory expo')
    plt.plot(j_list[1:], eig_vals, label='tau Transition')
    plt.legend()
    plt.ylabel(r'$\tau$')
    plt.xlabel('J')
    plt.yscale('log')


def plot_occ_probs_gibbs(data_folder,
                         n_iter_list=np.logspace(2, 5, 4, dtype=int),
                         j=1, stim=0, n_repetitions=100, theta=THETA,
                         burn_in=0.1):
    burn_in = int(burn_in*n_iter_list[0])
    colormap = pl.cm.Blues(np.linspace(0.2, 1, len(n_iter_list)))
    probs_data = data_folder + 'mu_mat_stim_' + str(stim) + '_j_' + str(j) + '_n_reps_' + str(n_repetitions) + '.npy'
    os.makedirs(os.path.dirname(probs_data), exist_ok=True)
    if os.path.exists(probs_data):
        mu_vals_all = np.load(probs_data, allow_pickle=True)
    else:
        mu_vals_all = np.empty((len(n_iter_list), n_repetitions))
        for i_n, n_iter in enumerate(n_iter_list):
            print(n_iter)
            mu_list = []
            for i in range(n_repetitions):
                init_state = np.random.choice([-1, 1], 8)
                states_mat = gibbs_samp_necker(init_state=init_state,
                                               burn_in=burn_in, n_iter=n_iter+burn_in, j=j,
                                               stim=stim)
                curr_state = np.sign(get_mu_from_mat_v2(states_mat))
                curr_state = (curr_state[curr_state != 0]+1)/2
                mu_list.append(np.mean(curr_state))
            mu_vals_all[i_n, :] = mu_list
        np.save(probs_data, mu_vals_all)
    fig, ax = plt.subplots(1, figsize=(5, 4))
    for i_mu, mu_list in enumerate(mu_vals_all):
        sns.kdeplot(mu_list, color=colormap[i_mu], label=n_iter_list[i_mu],
                    bw_adjust=0.1, cumulative=True)
        # plt.hist(mu_list, bins=40, color=colormap[i_n], label=n_iter)
    # plt.legend(title='N')
    plt.ylabel(r'CDF of $X(T)$')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel(r'Approximate posterior $q$')
    # plt.figure()
    k_1 = 12*j + 8*stim
    k_u = 2*j + 2*stim
    k_2 = 12*j - 8*stim
    rho = np.exp(-(k_1-k_2)/2)
    alpha = np.exp(-(k_1+k_2 - k_u*2)/2)
    for i_t, t in enumerate(n_iter_list):
        x = np.arange(0, t+1, 1)
        vals = occ_function_markov_ch_var(rho, alpha, t, x)
        cumsum = np.nancumsum(vals)
        plt.plot(x/t, cumsum / np.nanmax(cumsum),
                 label='analytical' + str(t), color=colormap[i_t],
                 linestyle='--')
    legendelements = [Line2D([0], [0], color='k', lw=2, label='Simulation'),
                      Line2D([0], [0], color='k', lw=2, linestyle='--',
                             label='Analytical'),
                      Line2D([0], [0], color=colormap[0], lw=2, label='T=1e2'),
                      Line2D([0], [0], color=colormap[1], lw=2, label='T=1e3'),
                      Line2D([0], [0], color=colormap[2], lw=2, label='T=1e4'),
                      Line2D([0], [0], color=colormap[3], lw=2, label='T=1e5'),
                      Line2D([0], [0], color=colormap[4], lw=2, label='T=1e6'),
                      ]
    if stim == 0.1:
        plt.legend(bbox_to_anchor=(1., 1.), handles=legendelements)
    # plt.title('B = ' + str(stim))
    fig.savefig(DATA_FOLDER + 'CDF_gibbs_stim_' + str(stim) + '_j_' + str(j) + '.png',
                dpi=400, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'CDF_gibbs_stim_' + str(stim) + '_j_' + str(j) + '.svg',
                dpi=400, bbox_inches='tight')


def plot_necker_cube_faces(interp_sfa=True, offset=0.25, whole=False,
                           index=0, color='grey'):
    fig, ax = plt.subplots(1, figsize=(4, 3.5))
    ax.plot([0, 1], [0, 0], color='k', zorder=1)
    ax.plot([0, 0], [0, 1], color='k', zorder=1)
    ax.plot([0, 1], [1, 1], color='k', zorder=1)
    ax.plot([0, offset], [1, 1+offset], color='k', zorder=1)
    ax.plot([1, 1+offset], [0+offset, 0+offset], color='k', zorder=1)
    ax.plot([offset, 1+offset], [1+offset, 1+offset], color='k', zorder=1)
    ax.plot([1+offset, 1+offset], [0+offset, 1+offset], color='k', zorder=1)
    ax.plot([1, 1+offset], [1, 1+offset], color='k', zorder=1)
    ax.plot([offset, 1+offset], [offset, offset], color='k', zorder=1)
    ax.plot([0, offset], [0, offset], color='k', zorder=1)
    ax.plot([1, 1+offset], [0, offset], color='k', zorder=1)
    if not whole:
        if interp_sfa:
            ax.fill_between([0, 1], [1, 1], color=color, alpha=1, zorder=2)
            ax.plot([1, 1], [0, 1], color='k', zorder=1)
        if not interp_sfa:
            ax.fill_between([offset, 1+offset],
                            [1+offset, 1+offset], [offset, offset],
                            color=color, alpha=1, zorder=2)
            ax.plot([offset, offset], [offset, 1+offset], color='k', zorder=1)
    else:
        ax.plot([offset, offset], [offset, 1+offset], color='k', zorder=1)
        ax.plot([1, 1], [0, 1], color='k', zorder=1)
    plt.axis('off')
    img_path = DATA_FOLDER + '/necker_images/fig_' + str(index) + '.png'
    fig.savefig(img_path, dpi=25, bbox_inches='tight')
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # imgrot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) / 255
    # img_path = DATA_FOLDER + '/necker_images/fig_rot_' + str(index) + '.png'
    # cv2.imwrite(img_path, imgrot)


def save_necker_cubes(offset=np.arange(0.1, 0.45, 0.01)):
    index = 0
    conds = np.array(list(itertools.product([True, False], repeat=2)))
    color = 'silver'
    for of in offset:
        for arr in conds:
            sfa, whole = arr
            plot_necker_cube_faces(interp_sfa=sfa, offset=of, whole=whole,
                                   index=index, color=color)
            index += 1
            plt.close('all')


def plot_necker_cubes(ax, mu, bot=True, offset=0.6, factor=1.5, msize=4):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(4, 3.2))
    mu_None = False
    if mu is not None:
        if np.round(mu) == -8:
            color_back = ['k']*4
            color_front = ['k']*4
            val_off_x = 0.75
            val_off_y = 4
        if mu == -6:
            color_back = ['k']*4
            color_front = ['white'] + ['k']*3
            val_off_x = 0.75
            val_off_y = .8
        if np.round(mu) == 8:
            color_back = ['white']*4
            color_front = ['white']*4
            val_off_x = -2.6
            val_off_y = 4
        if mu == 0 and not bot:
            color_back = ['k']*4
            color_front = ['white']*4
            val_off_x = -1
            val_off_y = 5.4
        if mu == 0 and bot:
            color_back = ['k', 'white', 'k', 'white']
            color_front = ['white', 'k', 'white', 'k']
            val_off_x = 1.2
            val_off_y = -12.5
    else:
        val_off_x = 0.75
        val_off_y = 4
        colors_tab = list(matplotlib.colors.TABLEAU_COLORS.keys())
        color_back = colors_tab[4:9]
        color_front = colors_tab[:4]
        mu = -8
        msize = 12
        ax.axis('off')
        mu_None = True
    nodes = np.zeros((2, 2, 2))
    x_nodes_front = (nodes[:, :, 0] + np.arange(2))*factor + mu + val_off_x
    y_nodes_front = ((nodes[:, :, 0].T + 2.2*np.arange(2)).T)*factor + np.abs(mu) + val_off_y
    x_nodes_back = (nodes[:, :, 1] + np.arange(2))*factor + offset + mu + val_off_x
    y_nodes_back = ((nodes[:, :, 1].T + 2.2*np.arange(2)).T)*factor + offset + np.abs(mu) + val_off_y
    for i in range(2):
        for j in range(2):
            if j % 2 == 0 or j == 1:
                ax.plot([x_nodes_front[i, j], x_nodes_back[i, j]],
                        [y_nodes_front[i, j], y_nodes_back[i, j]],
                        color='grey',
                        alpha=0.6)
            if (j+1) < 2:
                ax.plot([x_nodes_front[i, j], x_nodes_front[i, j+1]],
                        [y_nodes_front[i, j], y_nodes_front[i, j+1]],
                        color='grey',
                        alpha=0.6)
                ax.plot([x_nodes_back[i, j], x_nodes_back[i, j+1]],
                        [y_nodes_back[i, j], y_nodes_back[i, j+1]],
                        color='grey',
                        alpha=0.6)
            if (i+1) < 2:
                ax.plot([x_nodes_front[i, j], x_nodes_front[i+1, j]],
                        [y_nodes_front[i, j], y_nodes_front[i+1, j]],
                        color='grey',
                        alpha=0.6)
                ax.plot([x_nodes_back[i, j], x_nodes_back[i+1, j]],
                        [y_nodes_back[i, j], y_nodes_back[i+1, j]],
                        color='grey',
                        alpha=0.6)
    i = 0
    for x_b, y_b, x_f, y_f in zip(x_nodes_back.flatten(), y_nodes_back.flatten(),
                                  x_nodes_front.flatten(), y_nodes_front.flatten()):
        ax.plot(x_b, y_b, marker='o', linestyle='', color=color_back[i],
                markersize=msize, markeredgecolor='k')
        ax.plot(x_f, y_f, marker='o', linestyle='', color=color_front[i],
                markersize=msize, markeredgecolor='k')
        i += 1
    if mu_None:
        fig.savefig(DATA_FOLDER + 'necker_color_nodes.png',
                    dpi=400, bbox_inches='tight')
        fig.savefig(DATA_FOLDER + 'necker_color_nodes.svg',
                    dpi=400, bbox_inches='tight')


def plot_k_vs_mu_cylinder_simulations(j, b, chain_length=int(1e6),
                                      theta=return_theta(), data_folder=DATA_FOLDER):
    path = data_folder + 'simulation_15e6.npy'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        states_mat = np.load(path, allow_pickle=True)
    else:
        init_state = np.random.choice([-1, 1], theta.shape[0])
        burn_in = 500
        states_mat =\
            gibbs_samp_necker(init_state=init_state, burn_in=burn_in,
                              n_iter=chain_length+burn_in,
                              j=j, stim=b, theta=theta)
        np.save(path, states_mat)
    mu = get_mu_from_mat_v2(states_mat)
    k_vals = [k_val(config, theta*j, stim=b) for config in states_mat]
    plt.figure()
    plt.plot(mu, k_vals, marker='o', linestyle='',
              markersize=2, color='k')
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$k(\mu)$')


def dominance_duration_vs_stim(j=0.49, b_list=np.arange(0, 0.25, 0.01),
                               theta=THETA, chain_length=int(1e4), n_nodes_th=75):
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
        burn_in = 500
        states_mat =\
            gibbs_samp_necker(init_state=init_state, burn_in=burn_in,
                              n_iter=chain_length+burn_in,
                              j=j, stim=b, theta=theta)
        # mu = get_mu_from_mat_v2(states_mat)
        # mu_signed = np.sign(mu[mu != 0])
        mean_states = np.mean((states_mat+1)/2, axis=1)
        mean_states2 = np.copy(mean_states)
        mean_states[mean_states > p_thr] = 1
        mean_states[mean_states < (1-p_thr)] = -1
        if n_nodes_th == 50:
            mean_states = mean_states[mean_states != p_thr]
        mean_states[(mean_states > (1-p_thr)) & (mean_states < p_thr)] = 0
        orders = rle(mean_states)
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
        burn_in = 500
        b_both_eyes = np.repeat(b, theta.shape[0])
        b_both_eyes[theta.shape[0]//2:] = -b
        states_mat =\
            gibbs_samp_necker(init_state=init_state, burn_in=burn_in,
                              n_iter=chain_length+burn_in,
                              j=j, stim=b_both_eyes, theta=theta)
        mean_states = np.mean((states_mat+1)/2, axis=1)
        mean_states2 = np.copy(mean_states)
        mean_states2[mean_states2 > 0.5] = 1
        mean_states2[mean_states2 < 0.5] = 0
        mean_states2 = mean_states2[mean_states2 != 0.5]
        alt_rate_both_eyes.append(np.sum(np.diff(mean_states2) != 0))
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))
    ax = ax.flatten()
    ax[0].plot(b_list, list_predom_q1, label='q(x=1)', color='green', linewidth=2.5)
    ax[0].plot(b_list, list_predom_q2, label='q(x=-1)', color='r', linewidth=2.5)
    ax[0].legend()
    ax[0].set_xlabel('Sensory evidence, B')
    ax[0].set_ylabel('Perceptual predominance (<q(x=i)>')
    ax[1].plot(b_list, list_time_q1, label='q(x=1)', color='green', linewidth=2.5)
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
    # xf = np.arange(1e-6, 0.999, 1e-3)
    entropy = -f*np.log(f) - (1-f)*np.log(1-f)
    axtwin.plot(f, entropy, color='grey', linestyle='--')
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


def plot_dominance_duration(j, b=0, chain_length=int(1e4), n_nodes_th=7,
                            gibbs=True, mean_states=[], theta=THETA):
    p_thr = n_nodes_th/8
    if gibbs and len(mean_states) == 0:
        init_state = np.random.choice([-1, 1], theta.shape[0])
        burn_in = 100
        states_mat =\
            gibbs_samp_necker(init_state=init_state, burn_in=burn_in,
                              n_iter=chain_length+burn_in,
                              j=j, stim=b, theta=theta)
        # mu = get_mu_from_mat_v2(states_mat)
        # mu_signed = np.sign(mu[mu != 0])
        mean_states = np.mean((states_mat+1)/2, axis=1)
    mean_states[mean_states >= p_thr] = 1
    mean_states[mean_states <= (1-p_thr)] = 1
    mean_states[(mean_states > (1-p_thr)) & (mean_states < p_thr)] = 0
    orders = rle(mean_states)
    time = orders[0][orders[2] == 1]

    # plotting
    plt.figure()
    if gibbs:
        hist_bins = np.arange(-4, 200, 8)
    else:
        hist_bins = np.arange(-10, max(time)/2, 400)
    time = time[time <= max(hist_bins)]
    plt.hist(time, bins=hist_bins, label='Simulation', density=True)
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(time)
    fix_loc_exp, fit_lmb = stats.expon.fit(time)
    x = np.linspace(0, max(hist_bins), 1000)
    y = stats.gamma.pdf(x, a=fit_alpha, scale=fit_beta, loc=fit_loc)
    y_exp = stats.expon.pdf(x, loc=fix_loc_exp, scale=fit_lmb)
    plt.plot(x, y, label='Gamma distro. fit', color='k', linestyle='--')
    plt.plot(x, y_exp, label='Expo. distro. fit', color='r', linestyle='--')
    plt.xlabel('Dominance duration')
    plt.legend()


def hysteresis_necker(b_list=np.arange(-0.5, 0.5, 1e-2),
                      j_list=[0.1, 0.3, 0.6, 0.8], burn_in=10,
                      n_iter=100, n_sims=100, data_folder=DATA_FOLDER):
    fig, ax = plt.subplots(1)
    b_list = np.concatenate((b_list[:-1], b_list[::-1]))
    posterior_matrix = np.empty((len(j_list), len(b_list), n_sims))
    posterior_matrix[:] = np.nan
    colormap = pl.cm.Oranges(np.linspace(0.4, 1, len(j_list)))
    print('Plotting hysteresis Gibbs')
    for i_j, j in enumerate(j_list):
        for n in range(n_sims):
            init_state = np.random.choice([-1, 1], 8)
            posterior_list = []
            for i_b, b in enumerate(b_list):
                states_mat =\
                    gibbs_samp_necker(init_state, burn_in, n_iter+burn_in,
                                      j, stim=b, theta=THETA)
                states_mat_01 = (states_mat+1)/2
                posterior_list.append(np.nanmean(states_mat_01))
                init_state = states_mat[-1]
            posterior_matrix[i_j, :, n] = np.array(posterior_list)
        posterior_b = np.nanmean(posterior_matrix[i_j], axis=1)
        ax.plot(b_list, posterior_b, color=colormap[i_j], label=np.round(j, 1))
    ax.legend()
    plt.xlabel('Sensory evidence, B')
    plt.ylabel('Approximate posterior q(x=1)')
    plt.legend(title='J')


def plot_posterior_vs_b_diff_js(j_list=[0.05, 0.4, 0.7],
                                b_list=np.linspace(0, 0.25, 101),
                                theta=THETA, burn_in=100, n_iter=int(1e4)):
    plt.figure()
    # colormap = pl.cm.Oranges(np.linspace(0.4, 1, len(j_list)))
    colormap = ['navajowhite', 'orange', 'saddlebrown']
    for i_j, j in enumerate(reversed(j_list)):
        vec_vals = []
        for b in b_list:
            init_state = np.random.choice([-1, 1], theta.shape[0])
            val = gibbs_samp_necker_post(init_state, burn_in, n_iter, j,
                                         stim=b, theta=theta)
            val = np.max((val, 1-val))
            vec_vals.append(val)
        plt.plot(b_list, vec_vals, color=colormap[i_j],
                 label=np.round(j, 1), linewidth=4)
    plt.xlabel('Stimulus strength, B')
    plt.ylabel('Confidence')
    plt.legend(title='J')


def gibbs_samp_necker_post(init_state, burn_in, n_iter, j, stim=0, theta=THETA):
    x_vect = init_state
    mean_vector = np.sum((np.copy(x_vect)+1)/2)
    for i in range(n_iter):
        node = np.random.choice(np.arange(1, theta.shape[0]+1),
                                p=np.repeat(1/theta.shape[0], theta.shape[0]))
        x_vect1 = np.copy(x_vect)
        x_vect1[node-1] = -x_vect1[node-1]
        prob = change_prob(x_vect, x_vect1, j, stim=stim, theta=theta)
        # val_bool = np.random.choice([False, True], p=[1-prob, prob])
        val_bool = np.random.binomial(1, prob, size=None)
        if val_bool:
            x_vect[node-1] = -x_vect[node-1]
        if i >= burn_in:
            x_vect_mean = np.sum((np.copy(x_vect)+1)/2)
            mean_vector += x_vect_mean
    return mean_vector/theta.shape[0]/(n_iter-burn_in)


def get_b_mat(coh=0, b=0.2, rows=5, columns=5):
    perc = (coh + 1)/2
    b_mat = np.random.choice([b, -b], p=[perc, 1-perc],
                             size=[columns, rows])
    return b_mat.flatten()


def simulate_2d_lattice(b_mat=get_b_mat(coh=0, b=0.1, rows=5, columns=5),
                        j=0.1, theta=lattice_1d(columns=5, rows=5, layers=1),
                        n_iter=1000):
    init_state = np.random.choice([-1, 1], theta.shape[0])
    burn_in = 100
    states_mat = gibbs_samp_necker(init_state=init_state,
                                   burn_in=burn_in, n_iter=n_iter, j=j,
                                   stim=b_mat, theta=theta)
    mean_states = np.nanmean(states_mat, axis=1)
    # plt.figure()
    # plt.plot(mean_states)
    # print(np.mean(mean_states))
    fig, ax = plt.subplots(ncols=3, figsize=(10, 6))
    ax[0].imshow(np.flipud(b_mat.reshape(7, 7)), cmap='bwr')
    ax[0].set_title('Stimulus')
    mean_states = np.nanmean(states_mat, axis=0)
    ax[1].imshow(np.flipud(mean_states.reshape(7, 7)), cmap='bwr',
                 vmin=-1, vmax=1)
    ax[1].set_title(r'$<x_i>_t$')
    mean_states = np.nanmean((states_mat+1)/2, axis=0)
    # gauss = gkern(nsig=1.35)  #  + 0.5
    # gauss /= np.max(gauss)
    filtered = mean_states.reshape(7, 7)  # *gauss
    im = ax[2].imshow(np.flipud(filtered), cmap='PuOr',
                      vmin=0, vmax=1)
    ax[2].set_title(r'$q(x_i=1)$')
    plt.colorbar(im, fraction=0.05)


def plot_2d_lattice_conf_vs_coh(coh_list=np.arange(-0.5, 0.5, 0.01),
                        j_list=np.arange(0.01, 0.42, 0.2),
                        theta=lattice_1d(columns=5, rows=5, layers=1),
                        n_iter=1000):
    fig, ax = plt.subplots(1)
    colors = ['navajowhite', 'orange', 'saddlebrown']
    for i_j, j in enumerate(j_list):
        mean_st_list = []
        for coh in coh_list:
            b_mat=get_b_mat(coh=coh, b=0.1, rows=5, columns=5)
            init_state = np.random.choice([-1, 1], theta.shape[0])
            burn_in = 100
            states_mat = gibbs_samp_necker(init_state=init_state,
                                           burn_in=burn_in, n_iter=n_iter, j=j,
                                           stim=b_mat, theta=theta)
            mean_states = np.mean(np.nanmean((states_mat+1)/2, axis=1))
            mean_st_list.append(mean_states)
        ax.plot(coh_list, mean_st_list, color=colors[i_j],
                label=str(round(j, 1)))
    ax.set_xlabel('coherence')
    ax.set_ylabel('confidence')
    ax.legend(title='J')


def gkern(kernlen=20, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


def create_priors_j_b(jmin=0, jmax=2,
                      bmin=-5, bmax=5,
                      num_simulations=int(15e6)):
    prior =\
        MultipleIndependent([
            Uniform(torch.tensor([jmin]),
                    torch.tensor([jmax])),  # j
            Uniform(torch.tensor([bmin]),
                    torch.tensor([bmax]))])  # b
    return prior, prior.sample((num_simulations,))


def mnle_cylinder(path):
    x = np.load(path, allow_pickle=True)
    x = torch.tensor(x).to(torch.float32)
    prior, theta_all_inp = create_priors_j_b(
        jmin=0, jmax=2, bmin=-5, bmax=5,
        num_simulations=int(15e6))
    trainer = SNLE(prior=prior)
    print('Starting network training')
    # network training
    trainer = trainer.append_simulations(theta_all_inp, x)
    estimator = trainer.train(show_train_summary=True)
    


if __name__ == '__main__':
    # C matrix:\
    c_data = DATA_FOLDER + 'c_mat.npy'
    C = np.load(c_data, allow_pickle=True)
    # hysteresis_necker(b_list=np.arange(-0.5, 0.5, 1e-2),
    #                       j_list=[0.1, 0.5, 0.8], burn_in=0,
    #                       n_iter=300, n_sims=300)
    # plot_probs_gibbs(data_folder=DATA_FOLDER)
    # plot_analytical_prob(data_folder=DATA_FOLDER)
    # save_necker_cubes(offset=np.arange(0.1, 0.55, 0.01))
    # plot_k_vs_mu_analytical(eps=0, stim=0., plot_arist=True, plot_cubes=False)
    # plot_necker_cubes(ax=None, mu=None, bot=True, offset=0.6, factor=1.5, msize=4)
    # plot_mean_prob_gibbs(j_list=np.arange(0, 1.05, 0.05), burn_in=1000,
    #                       n_iter=200000, wsize=1, stim=0, j_ex=0.495, f_all=False,
    #                       theta=return_theta(rows=10, columns=5, layers=2))
    # plot_states_cylinder(j_ex=0.495,
    #                      stim=0, n_iter=201000,
    #                      theta=return_theta(rows=10, columns=5, layers=2),
    #                      burn_in=1000)
    # video_create(image_folder=DATA_FOLDER + '/gibbs_video_0495_zoom/',
    #              fps=80, videoname='cylinder_0495.mp4')
    # t = transition_matrix(0.2, C)
    # prob_markov_chain_between_states(tau=100, iters_per_len=200,
    #                                  n_iter_list=np.logspace(0, 4, 5))
    # plot_cylinder_true_posterior(j=0.2, stim=0.05, theta=THETA)

    # plot_occ_probs_gibbs(data_folder=DATA_FOLDER,
    #                       n_iter_list=np.logspace(2, 6, 5, dtype=int),
    #                       j=1, stim=0., n_repetitions=100, theta=THETA,
    #                       burn_in=0.1)
    # plot_dominance_duration(j=.9, b=0, chain_length=int(1e6), n_nodes_th=5,
    #                         theta=THETA)
    # dominance_duration_vs_stim(j=0.495,
    #                            b_list=np.round(
    #                                np.arange(-0.02, 0.025, 0.005), 4),
    #                            theta=return_theta(),
    #                            chain_length=int(7e6), n_nodes_th=50)
    # plot_k_vs_mu_cylinder_simulations(j=0.495, b=0, chain_length=int(1.5e7),
    #                                   theta=return_theta())
    # b_mat = get_b_mat(coh=0.2, b=.2, rows=7, columns=7)
    # for j in [0.01, 0.1, 0.2, 0.3]:
    #     simulate_2d_lattice(b_mat=b_mat,
    #                         j=j, theta=lattice_1d(columns=7, rows=7, layers=1),
    #                         n_iter=20000)
    plot_2d_lattice_conf_vs_coh(coh_list=np.arange(-1, 1, 0.1),
                            j_list=np.arange(0.1, 0.31, 0.1),
                            theta=lattice_1d(columns=5, rows=5, layers=1),
                            n_iter=200000)
    # plot_posterior_vs_b_diff_js(j_list=[0.05, 0.21, 0.45],
    #                             b_list=np.linspace(0, 0.25, 11),
    #                             theta=return_theta(rows=10, columns=5, layers=2),
    #                             burn_in=100, n_iter=int(1e3))
