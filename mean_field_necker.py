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
            th_vals = theta[q][theta[q] != 0]
            vec[q] = gn.sigmoid(2*(sum(J*th_vals*(2*vec[neighbours]-1)+stim))+np.random.randn()*sigma) 
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
                                         theta=theta, ind_list=[4, 12, 2]):
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
        # for i in range(len(neighbors)):
        #     vals_all_backwards[i, i_j] = \
        #         find_repulsor(j=j, num_iter=50, q_i=0.01,
        #                       q_f=0.95, stim=stim, threshold=1e-2,
        #                       theta=theta, neigh=neighbors[i])  # + 1e-2*i
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
                      Line2D([0], [0], color='r', lw=2, label='4'),
                      Line2D([0], [0], color='b', lw=2, label='5')]
    plt.legend(handles=legendelements, title='Neighbors')
    plt.ylabel('q')


def backwards(q, j, beta):
    if 0 <= q <= 1:
        q_new = 0.5*(1+ 1/j *(1/6 * np.log(q/(1-q)) - beta))
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
    plt.axhline(0.5, color='grey', alpha=1, linestyle='--')
    plt.plot(j_list, 1-np.array(l), color='k')
    plt.plot(j_list, l, color='r')
    plt.xlabel('J')
    plt.ylabel('q')
    plt.title('Solutions of the dynamical system')



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
    return np.clip(q + dt*(gn.sigmoid(6*j*(2*q-1))-q+bias)+
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


def plot_potentials_mf(j_list, bias=0):
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
        pot = potential_mf(q, j, bias=bias)
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
    pot = potential_mf(q, j, -bias)
    for i in range(len(ax)):
        ax[i].plot(q, pot, color='k')
        q_ind = m_field[i*num_iter//len(ax)]
        pot_ind = potential_mf(q_ind, j, -bias)
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


if __name__ == '__main__':
    # plot_potentials_mf(j_list=np.arange(0.001, 1.01, 0.1), bias=0.05)
    # plot_pot_evolution_mfield(j=0.9, num_iter=15, sigma=0.1, bias=0)
    # q_list = []
    # for j in np.arange(0.01, 1, 0.01):
    #     q_list.append(find_repulsor(j=j, num_iter=30, epsilon=1e-1, q_i=0.01,
    #                                 q_f=0.95, stim=0.1, threshold=1e-5, theta=theta,
    #                                 neigh=3))
    # plt.plot(np.arange(0.01, 1, 0.01), q_list)
    plot_mf_sol_stim_bias_different_sols(j_list=np.arange(0.001, 1, 0.005),
                                         stim=0.01,
                                         num_iter=20,
                                         theta=gn.return_theta(columns=5, rows=10),
                                         ind_list=[0, 12, 5])
    # plot_mf_sol_stim_bias(j_list=np.arange(0.00001, 1, 0.001), stim=-0.1,
    #                       num_iter=10)
