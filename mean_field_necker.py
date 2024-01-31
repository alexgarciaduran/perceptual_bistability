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


def plot_solutions_mfield(j_list):
    l = []
    for j in j_list:
        q = lambda q: gn.sigmoid(6*j*(2*q-1)) - q
        l.append(fsolve(q,1))
    plt.axhline(0.5, color='grey', alpha=1, linestyle='--')
    plt.plot(j_list, 1-np.array(l), color='k')
    plt.plot(j_list, l, color='r')
    plt.xlabel('J')
    plt.ylabel('q')
    plt.title('Solutions of the dynamical system')


def dyn_sys_mf(q, dt, j, sigma=1):
    return np.clip(q + dt*(gn.sigmoid(6*j*(2*q-1))-q)+
        np.random.randn()*np.sqrt(dt)*sigma, 0, 1)


def plot_q_evol_time_noise(j=0.7, dt=1e-3, num_iter=100):
    q = np.random.rand()
    q_l = []
    # time = np.arange(num_iter)*dt
    for i in range(num_iter):
        q = np.clip(dyn_sys_mf(q, dt, j, sigma=0.1), 0, 1)
        q_l.append(q)
    plt.plot(q_l)
    plt.xlabel('Steps')
    plt.ylabel('q')
    plt.ylim(-0.05, 1.05)


def potential_mf(q, j):
    return q*q/2 - np.log(1+np.exp(6*j*(2*q-1)))/(12*j)


def plot_potentials_mf(j_list):
    colormap = pl.cm.Blues(np.linspace(0.2, 1, len(j_list)))
    colormap_1 = pl.cm.Purples(np.linspace(0.4, 1, len(j_list)))
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
        pot = potential_mf(q, j)
        norm_cte = np.max(np.abs(pot))
        if abs(j - 0.333) < 0.01 and not change_colormap:
            color = 'r'
            change_colormap = True
        else:
            color = newcolors[i_j]
        ax.plot(q, pot-np.mean(pot), color=color, label=np.round(j, 2))
    ax_pos = ax.get_position()
    ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*1.02, ax_pos.y0,
                            ax_pos.width*0.04, ax_pos.height*0.9])
    cb = mpl.colorbar.ColorbarBase(ax_cbar, cmap=newcmp)
    ax_cbar.set_yticks([0, 1/3, 0.5, 1], [0, 0.33, 0.5, 1])
    ax_cbar.set_title('J')
    ax.set_xlabel('q')
    ax.set_ylabel(r'Mean-centered potential $V_J(q)$')
    # ax.legend(title='J:')


def plot_pot_evolution_mfield(j, num_iter=10, sigma=0.1):
    fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(16, 6))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.075, right=0.98,
                        hspace=0.3, wspace=0.5)
    # fig.suptitle('J = '+ str(j))
    ax = ax.flatten()
    # m_field = mean_field(j, num_iter=num_iter, sigma=sigma)
    q = np.random.rand()
    m_field = []
    # time = np.arange(num_iter)*dt
    for i in range(num_iter):
        q = dyn_sys_mf(q, dt=1, j=j, sigma=sigma)
        m_field.append(q)
    q = np.arange(-.25,1.25, 0.001)
    # q = np.arange(0, 1, 0.001)
    pot = potential_mf(q, j)
    for i in range(len(ax)):
        ax[i].plot(q, pot, color='k')
        q_ind = m_field[i*num_iter//len(ax)]
        pot_ind = potential_mf(q_ind, j)
        ax[i].plot(q_ind, pot_ind, marker='o', color='r')
        ax[i].set_title('iter' + str(i*num_iter//len(ax)+1))
        if i != 0 and i != (len(ax)//2):
            ax[i].set_yticks([])
        else:
            ax[i].set_ylabel('Potential')
        if i < 5:
            ax[i].set_xticks([0, 0.5, 1], ['', '', ''])
        else:
            ax[i].set_xticks([0, 0.5, 1])
            ax[i].set_xlabel('q')


if __name__ == '__main__':
    plot_solutions_mfield(j_list=np.arange(0.00001, 2, 0.005))
    # for num_iter in [200, 1000, 5000, 10000, 20000, 50000]:
    #     plt.figure()
    #     plot_mf_sol(j_list=np.arange(0.25, 0.65, 0.004), num_iter=num_iter)
    #     plt.xlabel('J')
    #     plt.ylabel('q*')
    #     plt.title('num_iter: ' + str(num_iter))
    # plot_mf_sol_sigma_j(data_folder=DATA_FOLDER, j_list=np.arange(0.0001, 1, 0.005),
    #                     sigma_list=np.arange(0, 2, 0.01),
    #                     num_iter=2000)
    # plot_potentials_mf(j_list=np.arange(0.001, 1, 0.01))
