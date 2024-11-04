# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:49:30 2023

@author: alexg
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import itertools
import os
import gibbs_necker as gn
import mean_field_necker as mfn
from scipy.optimize import fsolve, bisect, root, newton
from scipy.integrate import solve_ivp
import matplotlib as mpl
from skimage.transform import resize
from matplotlib.lines import Line2D
import matplotlib.pylab as pl

THETA = gn.THETA


mpl.rcParams['font.size'] = 18
plt.rcParams['legend.title_fontsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize']= 16
plt.rcParams['ytick.labelsize']= 16


pc_name = 'alex'
if pc_name == 'alex':
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/belief_propagation_necker/data_folder/'  # Alex


def jneigbours(j,i, theta=THETA):
    """
    return the neighbours of j except i

    Input:
    - i {integer}: index of our current node
    - j {integer}: index of j
    Output:
    - return a list with all the neighbours of neighbour of j except our 
      current node
    """
    neigbours = np.array(np.where(theta[j,:] != 0))
    return neigbours[neigbours!=i]


def Loopy_belief_propagation(theta, num_iter, j, thr=1e-5, stim=0):
    """
    Computes the approximate probability of having x=1 or x=-1
    (e.g. depth), for any configuration given by the matrix of connections theta.
    
    Input:
    - theta {matrix}: matrix that defines any nodes combination (connectivity)
    - j {double}: coupling strenght
    - num_iter {integer}: number of iterations for the algorithm to run
    - thr {double}: threshold to stop iterating, i.e. convergence
    - stim {double}: sensory evidence
    
    Output:
    - returns approximate posterior of x=1 and x=-1, and number of iterations needed
    """
    # multiply arrays element wise
    mu_y_1 = np.multiply(theta, np.random.rand(theta.shape[0], theta.shape[1]))
    mat_memory_1 = np.copy(mu_y_1)
    mu_y_neg1 = np.multiply(theta, np.random.rand(theta.shape[0], theta.shape[1]))
    mat_memory_neg1 = np.copy(mu_y_neg1)
    for n in range(num_iter):
        mat_memory_1 = np.copy(mu_y_1)
        mat_memory_neg1 = np.copy(mu_y_neg1)
        # for all the nodes that i is connected
        for i in range(theta.shape[0]):
            for t in np.where(theta[i, :] != 0)[0]:
                # positive y_i
                mu_y_1[t, i] = np.exp(j*theta[i, t]+stim) *\
                        np.prod(mu_y_1[jneigbours(t, i, theta=theta), t]) \
                        + np.exp(-j*theta[i, t]-stim) *\
                        np.prod(mu_y_neg1[jneigbours(t, i, theta=theta), t])
                # mu_y_1 += np.random.rand(8, 8)*1e-3
                # negative y_i
                mu_y_neg1[t, i] = np.exp(-j*theta[i, t]+stim) *\
                    np.prod(mu_y_1[jneigbours(t, i, theta=theta), t])\
                    + np.exp(j*theta[i, t]-stim) *\
                    np.prod(mu_y_neg1[jneigbours(t, i, theta=theta), t])

                m_y_1_memory = np.copy(mu_y_1[t, i])
                mu_y_1[t, i] = mu_y_1[t, i]/(m_y_1_memory+mu_y_neg1[t, i])
                mu_y_neg1[t, i] = mu_y_neg1[t, i]/(m_y_1_memory+mu_y_neg1[t, i])
                # mu_y_neg1 += np.random.rand(8, 8)*1e-3
        if np.sqrt(np.sum(mat_memory_1 - mu_y_1)**2) and\
            np.sqrt(np.sum(mat_memory_neg1 - mu_y_neg1)**2) <= thr:
            break
    q_y_1 = np.zeros(theta.shape[0])
    q_y_neg1 = np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        q1 = np.prod(mu_y_1[np.where(theta[:, i] != 0), i]) * np.exp(stim)
        qn1 = np.prod(mu_y_neg1[np.where(theta[:, i] != 0), i]) * np.exp(-stim)
        q_y_1[i] = q1/(q1+qn1)
        q_y_neg1[i] = qn1/(q1+qn1)
    # gn.plot_cylinder(q=q_y_1.reshape(5, 10, 2),
    #                  columns=5, rows=10, layers=2, offset=0.4, minmax_norm=True)
    return q_y_1, q_y_neg1, n+1



def Fractional_loopy_belief_propagation(theta, num_iter, j, alpha, thr=1e-5, stim=0):
    """
    Computes the exact approximate probability of having a front perception for
    the depth of the 8 nodes.
    Input:
    - theta {matrix}: matrix that defines the Necker cube
    - J {double}: coupling strengs
    - num_iter {integer}: number of iterations for the algorithm to run
    Output:
    - return a list of lenght 8 with the approximate depth probabilities
    """
    # multiply arrays element wise
    mu_y_1 = np.multiply(theta, np.random.rand(theta.shape[0], theta.shape[1]))
    mat_memory_1 = np.copy(mu_y_1)
    mu_y_neg1 = np.multiply(theta, np.random.rand(theta.shape[0], theta.shape[1]))
    mat_memory_neg1 = np.copy(mu_y_neg1)
    for n in range(num_iter):
        mat_memory_1 = np.copy(mu_y_1)
        mat_memory_neg1 = np.copy(mu_y_neg1)
        # for all the nodes that i is connected
        for i in range(theta.shape[0]):
            for t in np.where(theta[i, :] != 0)[0]:
                # positive y_i
                mu_y_1[t, i] = np.exp(j*theta[i, t]+stim)**(alpha) *\
                        (mu_y_1[jneigbours(t, i, theta=theta)[0], t]*
                         mu_y_1[jneigbours(t, i, theta=theta)[1], t]) *\
                        mu_y_1[i, t]**(1-alpha) + \
                        + np.exp(-j*theta[i, t]-stim)**(alpha) *\
                        (mu_y_neg1[jneigbours(t, i, theta=theta)[0], t] *
                         mu_y_neg1[jneigbours(t, i, theta=theta)[1], t]) *\
                        mu_y_neg1[i, t]**(1-alpha)
                # mu_y_1 += np.random.rand(8, 8)*1e-3
                # negative y_i
                mu_y_neg1[t, i] = np.exp(-j*theta[i, t]+stim)**(alpha) *\
                    (mu_y_1[jneigbours(t, i, theta=theta)[0], t] *\
                     mu_y_1[jneigbours(t, i, theta=theta)[1], t])*\
                    mu_y_1[i, t]**(1-alpha) + \
                    + np.exp(j*theta[i, t]-stim)**(alpha) *\
                    (mu_y_neg1[jneigbours(t, i, theta=theta)[0], t] *
                     mu_y_neg1[jneigbours(t, i, theta=theta)[1], t])*\
                    mu_y_neg1[i, t]**(1-alpha)

                m_y_1_memory = np.copy(mu_y_1[t, i])
                mu_y_1[t, i] = mu_y_1[t, i]/(m_y_1_memory+mu_y_neg1[t, i])
                mu_y_neg1[t, i] = mu_y_neg1[t, i]/(m_y_1_memory+mu_y_neg1[t, i])
                # mu_y_neg1 += np.random.rand(8, 8)*1e-3
        if np.sqrt(np.sum(mat_memory_1 - mu_y_1)**2) and\
            np.sqrt(np.sum(mat_memory_neg1 - mu_y_neg1)**2) <= thr:
            break
    q_y_1 = np.zeros(theta.shape[0])
    q_y_neg1 = np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        q1 = np.prod((mu_y_1[np.where(theta[:, i] != 0), i])) * np.exp(stim)
        qn1 = np.prod((mu_y_neg1[np.where(theta[:, i] != 0), i])) * np.exp(-stim)
        q_y_1[i] = q1/(q1+qn1)
        q_y_neg1[i] = qn1/(q1+qn1)
    # gn.plot_cylinder(q=q_y_1.reshape(5, 10, 2),
    #                  columns=5, rows=10, layers=2, offset=0.4, minmax_norm=True)
    return q_y_1, q_y_neg1, n+1



def Frac_belief_propagation(theta, num_iter, j, alpha, thr=1e-5, stim=0):
    """
    Computes the exact approximate probability of having a front perception for
    the depth of the 8 nodes.
    Input:
    - theta {matrix}: matrix that defines the Necker cube
    - J {double}: coupling strengs
    - num_iter {integer}: number of iterations for the algorithm to run
    Output:
    - return a list of lenght 8 with the approximate depth probabilities
    """
    # multiply arrays element wise
    mu_y_1 = np.multiply(theta, np.random.rand(theta.shape[0], theta.shape[1]))
    mat_memory_1 = np.copy(mu_y_1)
    mu_y_neg1 = np.multiply(theta, np.random.rand(theta.shape[0], theta.shape[1]))
    mat_memory_neg1 = np.copy(mu_y_neg1)
    for n in range(num_iter):
        mat_memory_1 = np.copy(mu_y_1)
        mat_memory_neg1 = np.copy(mu_y_neg1)
        # for all the nodes that i is connected
        for i in range(theta.shape[0]):
            for t in np.where(theta[i, :] != 0)[0]:
                # positive y_i
                mu_y_1[t, i] = np.exp(j*theta[i, t]+stim)**(alpha) *\
                        (mu_y_1[jneigbours(t, i, theta=theta)[0], t]*
                         mu_y_1[jneigbours(t, i, theta=theta)[1], t])\
                        + np.exp(-j*theta[i, t]-stim)**(alpha) *\
                        (mu_y_neg1[jneigbours(t, i, theta=theta)[0], t] *
                         mu_y_neg1[jneigbours(t, i, theta=theta)[1], t])
                # mu_y_1 += np.random.rand(8, 8)*1e-3
                # negative y_i
                mu_y_neg1[t, i] = np.exp(-j*theta[i, t]+stim)**(alpha) *\
                    (mu_y_1[jneigbours(t, i, theta=theta)[0], t] *\
                     mu_y_1[jneigbours(t, i, theta=theta)[1], t])\
                    + np.exp(j*theta[i, t]-stim)**(alpha) *\
                    (mu_y_neg1[jneigbours(t, i, theta=theta)[0], t] *
                     mu_y_neg1[jneigbours(t, i, theta=theta)[1], t])

                m_y_1_memory = np.copy(mu_y_1[t, i])
                mu_y_1[t, i] = mu_y_1[t, i]/(m_y_1_memory+mu_y_neg1[t, i])
                mu_y_neg1[t, i] = mu_y_neg1[t, i]/(m_y_1_memory+mu_y_neg1[t, i])
                # mu_y_neg1 += np.random.rand(8, 8)*1e-3
        if np.sqrt(np.sum(mat_memory_1 - mu_y_1)**2) and\
            np.sqrt(np.sum(mat_memory_neg1 - mu_y_neg1)**2) <= thr:
            break
    q_y_1 = np.zeros(theta.shape[0])
    q_y_neg1 = np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        q1 = np.prod((mu_y_1[np.where(theta[:, i] != 0), i])**(1-alpha)) * np.exp(stim)
        qn1 = np.prod((mu_y_neg1[np.where(theta[:, i] != 0), i])**(1-alpha)) * np.exp(-stim)
        q_y_1[i] = q1/(q1+qn1)
        q_y_neg1[i] = qn1/(q1+qn1)
    # gn.plot_cylinder(q=q_y_1.reshape(5, 10, 2),
    #                  columns=5, rows=10, layers=2, offset=0.4, minmax_norm=True)
    return q_y_1, q_y_neg1, n+1


def posterior_vs_b(stim_list=np.linspace(-2, 2, 10001),
                   j=0.1, theta=THETA, data_folder=DATA_FOLDER,
                   compute_posterior=True):
    if compute_posterior:
        true_posterior = gn.true_posterior_stim(stim_list=stim_list, j=j, theta=theta,
                                                data_folder=data_folder,
                                                load_data=True, save_data=False)
    else:
        true_posterior = np.nan
    mf_post = []
    bp_post = []
    N = 3
    init_cond = 0
    init_conds_bp = [10, 5, 20, 1, 0]
    n_its = len(init_conds_bp)
    for stim in stim_list:
        q1 = lambda q: gn.sigmoid(2*N*j*(2*q-1) + stim*2) - q
        sol, _, flag, _ =\
            fsolve(q1, init_cond, full_output=True)
        if stim >= 0:
            init_cond = 1
        if flag != 1:
            sol = np.nan
            mf_post.append(sol)
        if flag == 1:
            if stim < 0 and sol < 0.5:
                mf_post.append(sol)
            if stim < 0 and sol >= 0.5:
                mf_post.append(1-sol)
            if stim >= 0:
                mf_post.append(sol)
        g_fun = lambda q: g(q, b=stim, j=j, N=3)
        flag = 2
        its = 0
        while flag != 1 and its < n_its:
            sol, _, flag, _ =\
                fsolve(g_fun, init_conds_bp[its], full_output=True)    
            its += 1
        if flag != 1:
            sol = np.nan
            bp_post.append(sol)
        if flag == 1:
            sol = (sol**N) * np.exp(stim) / (np.exp(-stim) + np.exp(stim) * (sol**N))
            if stim < 0 and sol < 0.5:
                bp_post.append(sol)
            if stim < 0 and sol >= 0.5:
                bp_post.append(1-sol)
            if stim >= 0 and sol >= 0.5:
                bp_post.append(sol)
            if stim >= 0 and sol < 0.5:
                bp_post.append(1-sol)
    return np.array(true_posterior), np.array(mf_post), np.array(bp_post)


def plot_overconfidence_vs_j(j_list=np.arange(0, 1.05, 0.01),
                             stim_list=np.linspace(0, 2, 101),
                             data_folder=DATA_FOLDER, theta=THETA,
                             gibbs=True):
    mse_mf = []
    mse_bp = []
    for j in j_list:
        true_posterior, mf_post, bp_post =\
            posterior_vs_b(stim_list=stim_list,
                           j=j, theta=theta, data_folder=data_folder)
        mse_mf.append(np.trapz(mf_post.T-true_posterior, true_posterior))
        mse_bp.append(np.trapz(bp_post.T-true_posterior, true_posterior))
    fig, ax = plt.subplots(1, figsize=(4, 3))
    ax.plot(j_list, mse_bp, color='k', label='LBP')
    ax.plot(j_list, mse_mf, color='r', label='MF')
    ax.legend()
    ax.set_xlabel(r'Coupling $J$')
    ax.set_ylabel('Over-confidence')
    fig.tight_layout()
    fig.savefig(data_folder + 'over_confidence.png')
    fig.savefig(data_folder + 'over_confidence.svg')


def plot_over_conf_mf_bp_gibbs(data_folder=DATA_FOLDER, j_list=np.arange(0., 1.005, 0.005),
                               b_list_orig=np.arange(-.5, .5005, 0.005), theta=THETA):
    b_list = np.arange(-.5, 0.5005, 0.005)
    # ind_0 = np.where(abs(b_list_orig) < 1e-15)[0][0]
    ind_0 = 0
    # True posterior
    matrix_true_file = data_folder + 'true_vs_JB_05.npy'
    mat_true = np.load(matrix_true_file, allow_pickle=True)
    # Gibbs for 3 different T
    # T=100
    matrix_gn_file = data_folder + '100_gibbs_posterior_vs_JB_05.npy'
    mat_gn_1e2 = (np.load(matrix_gn_file, allow_pickle=True)+1)/2
    # T=1000
    matrix_gn_file = data_folder + '1000_gibbs_posterior_vs_JB_05.npy'
    mat_gn_1e3 = (np.load(matrix_gn_file, allow_pickle=True)+1)/2
    # T=10000
    matrix_gn_file = data_folder + '10000_gibbs_posterior_vs_JB_05.npy'
    mat_gn_1e4 = (np.load(matrix_gn_file, allow_pickle=True)+1)/2
    # Confidence
    conf_mf = []
    conf_lbp = []
    conf_g1e2 = []
    conf_g1e3 = []
    conf_g1e4 = []
    for i_j, j in enumerate(j_list):
        # true posterior
        true_posterior = mat_true[i_j, ind_0:]
        # MF/LBP
        _, mf_post, bp_post =\
            posterior_vs_b(stim_list=b_list,
                           j=j, theta=theta, data_folder=data_folder,
                           compute_posterior=False)
        # MF
        conf_mf.append(np.trapz(abs(mf_post.T-true_posterior), true_posterior))
        # LBP
        conf_lbp.append(np.trapz(abs(bp_post.T-true_posterior), true_posterior))
        # Gibbs 100
        vals_gibs_100 = mat_gn_1e2[i_j, ind_0:]
        conf_g1e2.append(np.trapz(abs(vals_gibs_100-true_posterior), true_posterior))
        # Gibbs 1000
        vals_gibs_1000 = mat_gn_1e3[i_j, ind_0:]
        conf_g1e3.append(np.trapz(abs(vals_gibs_1000-true_posterior), true_posterior))
        # Gibbs 10000
        vals_gibs_10000 = mat_gn_1e4[i_j, ind_0:]
        conf_g1e4.append(np.trapz(abs(vals_gibs_10000-true_posterior), true_posterior))
    fig, ax = plt.subplots(1, figsize=(5, 3.4))
    ax.plot(j_list, conf_lbp, color='k', label='LBP')
    ax.plot(j_list, conf_mf, color='r', label='MF')
    colormap = pl.cm.Blues(np.linspace(0.2, 1, 3))
    wsize = 1
    ax.plot(j_list, np.convolve(conf_g1e2, np.ones(wsize)/wsize, 'same'), color=colormap[0], label='Gibbs 1e2')
    ax.plot(j_list, np.convolve(conf_g1e3, np.ones(wsize)/wsize, 'same'), color=colormap[1], label='Gibbs 1e3')
    ax.plot(j_list, np.convolve(conf_g1e4, np.ones(wsize)/wsize, 'same'), color=colormap[2], label='Gibbs 1e4')
    ax.legend()
    ax.set_xlabel(r'Coupling $J$')
    ax.set_ylabel('Over-confidence')
    fig.tight_layout()
    fig.savefig(data_folder + 'over_confidence_all.png')
    fig.savefig(data_folder + 'over_confidence_all.svg')


def posterior_comparison_MF_BP(stim_list=np.linspace(-2, 2, 1000), j=0.1,
                               num_iter=100, thr=1e-12, theta=THETA,
                               data_folder=DATA_FOLDER):
    true_posterior, mf_post, bp_post = posterior_vs_b(
        stim_list=stim_list, j=j, theta=theta,
        data_folder=data_folder)
        # q = mfn.mean_field_stim(j, num_iter=num_iter, stim=stim, sigma=0,
        #                        theta=theta)
        # mf_post.append(q[-1, 0])
        # q_bp, _, _= Loopy_belief_propagation(theta=theta,
        #                                      num_iter=num_iter,
        #                                      j=j, thr=thr, stim=stim)
        # bp_post.append(q_bp[0])
    
    fig, ax = plt.subplots(1, figsize=(5, 3.5))
    ax.plot(true_posterior, bp_post, color='k', label='LBP',
            linestyle='--')
    ax.plot(true_posterior, mf_post, color='r', label='MF',
            linestyle='--')
    ax.fill_between(true_posterior, true_posterior, mf_post.T[0],
                    color='r', alpha=0.08)
    ax.plot([0, 1], [0, 1], color='grey', alpha=0.5, label='y=x, True')
    ax.set_xlabel(r'True posterior $p(x|B)$')
    ax.set_ylabel(r'Approximated posterior $q(x|B)$')
    ax.text(0.5, 0.1, 'Over-confidence')
    ax.arrow(0.35, 0.195, 0.11, -0.06, head_width=0.02, color='k')
    # ax.set_title('J = '+str(j))
    ax.legend()
    fig.tight_layout()
    fig.savefig(data_folder + 'mf_BP_necker_posterior_comparison_last.png',
                dpi=400, bbox_inches='tight')
    fig.savefig(data_folder + 'mf_BP_necker_posterior_comparison_last.svg',
                dpi=400, bbox_inches='tight')


def plot_loopy_b_prop_sol_difference(theta, num_iter, j_list=np.arange(0, 1, 0.1),
                                     thr=1e-15):
    diff = []
    plt.figure()
    for j in j_list:
        pos, neg, n = Loopy_belief_propagation(theta=theta,
                                               num_iter=num_iter,
                                               j=j, thr=thr)
        diff.append(pos[0]-neg[0])
    plt.plot(j_list, diff, color='k')


def plot_loopy_b_prop_sol(theta, num_iter, j_list=np.arange(0, 1, 0.001),
                          thr=1e-15, stim=0.1, alpha=1):
    lp = []
    ln = []
    nlist = []
    fig, ax = plt.subplots(1)
    vals_all = np.empty((theta.shape[0], len(j_list)))
    vals_all[:] = np.nan
    for i_j, j in enumerate(j_list):
        pos, neg, n = Loopy_belief_propagation(theta=theta,
                                               num_iter=num_iter,
                                               j=j, thr=thr,
                                               stim=stim)
        # gn.plot_cylinder(q=pos.reshape(5, 10, 2),
        #                   columns=5, rows=10, layers=2, offset=0.4, minmax_norm=True)
        lp.append(pos[0])
        ln.append(neg[0])
        # to get upper attractor:
        vals_all[:, i_j] = [np.max((p, n)) for p, n in zip(pos, neg)]
        nlist.append(n)
    # ax.plot(j_list, lp, color='k')
    neighs = np.sum(theta, axis=1, dtype=int)
    min_neighs = np.min(neighs)
    neighs -= min_neighs
    colors = ['k', 'r', 'b']
    for i_v, vals in enumerate(vals_all):
        ax.plot(j_list, vals, color=colors[neighs[i_v]], alpha=1)
        ax.plot(j_list, 1-vals, color=colors[neighs[i_v]], alpha=1)
    
    # plt.plot(j_list, ln, color='r')
    # plot_sol_LBP(j_list=np.arange(0, max(j_list), 0.00001), stim=stim)
    # gn.true_posterior_plot_j(ax=ax, stim=stim)
    ax.set_title('Loopy-BP solution (symmetric)')
    ax.set_xlabel('J')
    ax.set_ylabel('q')
    # plt.plot([np.log(3)/2, 1], [0.5, 0.5], color='grey', alpha=1, linestyle='--',
    #          label='Unstable FP')
    legendelements = []
    legendelements = [Line2D([0], [0], color='k', lw=2, label='3'),
                      Line2D([0], [0], color='r', lw=2, label='4')]
    # legendelements = [Line2D([0], [0], color='k', lw=2, label='Stable FP'),
    #                   Line2D([0], [0], color='k', lw=2, label='Unstable FP')]
    plt.legend(handles=legendelements, title='Neighbors')
    # plt.legend(title='Neighbors')
    ax.set_ylim(-0.05, 1.05)
    plt.figure()
    plt.plot(j_list, nlist, color='k')
    plt.xlabel('J')
    plt.ylabel('n_iter for convergence, thr = {}'.format(thr))
    plt.figure()
    plt.plot(nlist, lp, color='k')
    plt.plot(nlist, ln, color='r')
    plt.xlabel('n_iter for convergence, thr = {}'.format(thr))
    plt.ylabel('q')
    # plot_bp_solution(ax, j_list, b=stim, tol=1e-10,
    #                  min_r=0, max_r=15,
    #                  w_size=0.05, n_neigh=3,
    #                  color='b')


def solutions_bp(j_list=np.arange(0.00001, 2, 0.000001), stim=0.1):
    q0_l = []
    q1_l = []
    q2_l = []
    for j in j_list:
        j += stim
        # q0 = (np.exp(-j)+np.exp(j))**(-3)
        q0 = 1 / 2
        q0_l.append(q0)
        if j >= np.log(3)/2:
            r_1 = ((np.exp(2*j)-1+np.sqrt((np.exp(2*j)-1)*(np.exp(2*j)-1)-4))/2)**3
            r_2 = ((np.exp(2*j)-1-np.sqrt((np.exp(2*j)-1)*(np.exp(2*j)-1)-4))/2)**3
        else:
            r_1 = np.nan
            r_2 = np.nan
        # q1 = (np.exp(-j)/r_1**2+np.exp(j))**(-3)
        # q2 = (np.exp(-j)/r_2**2+np.exp(j))**(-3)
        q1 = r_1 / (1+r_1)
        q2 = r_2 / (1+r_2)
        q1_l.append(q1)
        q2_l.append(q2)
    return q0_l, q1_l, q2_l


def plot_sol_LBP(j_list=np.arange(0.00001, 2, 0.000001), stim=0.1):
    fig = plt.figure(figsize=(6, 4))
    q0_l, q1_l, q2_l = solutions_bp(j_list=j_list, stim=stim)
    # plt.plot(j_list, q0_l, color='grey', linestyle='--')
    plt.plot([0, np.log(3)/2], [0.5, 0.5], color='k', alpha=1, label='Stable FP')
    plt.plot(j_list, q1_l, color='k')
    plt.plot(j_list, q2_l, color='k')
    plt.xlabel(r'Coupling $J$')
    plt.plot([np.log(3)/2, 1], [0.5, 0.5], color='grey', alpha=1, linestyle='--',
             label='Unstable FP')
    plt.axvline(np.log(3)/2, color='r', alpha=0.2)
    # xtcks = np.sort(np.unique([0, 0.25, 0.75, 0.5, np.log(3)/2, 1]))
    # labs = [x for x in xtcks]
    # pos = np.where(xtcks == np.log(3)/2)[0][0]
    # labs[pos] = r'$J^{\ast}$'
    # plt.xticks(xtcks, labs)
    plt.text(np.log(3)/2 - 0.05, 0.08, r'$J^{\ast} = \log{(3)}/2$',
             rotation='vertical')
    plt.ylabel(r'Posterior $q$')
    # plt.title('Solutions of the dynamical system')
    plt.legend()
    plt.tight_layout()
    fig.savefig(DATA_FOLDER + 'bp_solutions.png', dpi=400, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'bp_solutions.svg', dpi=400, bbox_inches='tight')


def plot_solution_BP(j_list, stim):
    j = np.exp(j_list*2)
    b = np.exp(stim*2)
    
    # Define the constants
    c1 = j / 3
    c2 = 2**(1/3) * (3 * b * j - b**2 * j**2) / (3 * b * (27 * b**2 - 9 * b**2 * j**2 + 2 * b**3 * j**3 + 3 * np.sqrt(3) * np.sqrt(27 * b**4 - 18 * b**4 * j**2 + 4 * b**3 * j**3 + 4 * b**5 * j**3 - b**4 * j**4))**(1/3))
    c3 = (27 * b**2 - 9 * b**2 * j**2 + 2 * b**3 * j**3 + 3 * np.sqrt(3) * np.sqrt(27 * b**4 - 18 * b**4 * j**2 + 4 * b**3 * j**3 + 4 * b**5 * j**3 - b**4 * j**4))**(1/3) / (3 * 2**(1/3) * b)
    
    # Define the equations
    eq1 = c1 - c2 + c3
    eq2 = c1 + ((1 + 1j * np.sqrt(3)) * (3 * b * j - b**2 * j**2)) / (3 * 2**(2/3) * b * (27 * b**2 - 9 * b**2 * j**2 + 2 * b**3 * j**3 + 3 * np.sqrt(3) * np.sqrt(27 * b**4 - 18 * b**4 * j**2 + 4 * b**3 * j**3 + 4 * b**5 * j**3 - b**4 * j**4))**(1/3)) - ((1 - 1j * np.sqrt(3)) * c3) / (6 * 2**(1/3) * b)
    eq3 = c1 + ((1 - 1j * np.sqrt(3)) * (3 * b * j - b**2 * j**2)) / (3 * 2**(2/3) * b * (27 * b**2 - 9 * b**2 * j**2 + 2 * b**3 * j**3 + 3 * np.sqrt(3) * np.sqrt(27 * b**4 - 18 * b**4 * j**2 + 4 * b**3 * j**3 + 4 * b**5 * j**3 - b**4 * j**4))**(1/3)) - ((1 + 1j * np.sqrt(3)) * c3) / (6 * 2**(1/3) * b)

    plt.plot(j_list, np.exp(stim)*eq1**3 / (np.exp(-stim)*+np.exp(stim)*eq1**3), color='b')
    # plt.plot(j_list, 1 / (1+eq1**3), color='b')
    # plt.plot(j_list, eq2**3 / (1+eq2**3), color='b')
    # plt.plot(j_list, 1 / (1+eq2**3), color='b')
    # plt.plot(j_list, eq3**3 / (1+eq3**3), color='b')
    # plt.plot(j_list, 1 / (1+eq3**3), color='b')


def r_stim(x, j_e, b_e, n_neigh=3, alpha=1):
    return b_e*x**(n_neigh) - (j_e**alpha) * b_e * x**(n_neigh-alpha) + (j_e**alpha)* x**alpha - 1


def r_stim_prime(x, j_e, b_e):
    return 3*x**2 - 2 * j_e * b_e * x + j_e


def cubic(a,b,c,d):
    n = -b**3/27/a**3 + b*c/6/a**2 - d/2/a
    s = (n**2 + (c/3/a - b**2/9/a**2)**3)**0.5
    r0 = (n-s)**(1/3)+(n+s)**(1/3) - b/3/a
    r1 = (n+s)**(1/3)+(n+s)**(1/3) - b/3/a
    r2 = (n-s)**(1/3)+(n-s)**(1/3) - b/3/a
    return (r0,r1,r2)


def find_solution_bp(j, b, min_r=-10, max_r=10, w_size=0.1,
                     tol=1e-2, n_neigh=3, alpha=1):
    """
    Searches for roots using bisection method in a interval of r, given by
    [min_r, max_r], using a sliding window of of size w_size.
    
    """
    j_e = np.exp(2*j)
    b_e = np.exp(2*b)
    count = 0
    sols = []
    while (min_r + (count+1) * w_size) <= max_r:
        a = min_r+count*w_size
        b = min_r+(count+1)*w_size
        if np.sign(r_stim(a, j_e, b_e, n_neigh=n_neigh, alpha=alpha)*
                   r_stim(b, j_e, b_e, n_neigh=n_neigh, alpha=alpha)) > 0:
            count += 1
        else:
            solution_bisection = bisect(r_stim, a=a, b=b,
                                        args=(j_e, b_e, n_neigh, alpha),
                                        xtol=1e-12)
            if len(sols) > 0:
                if (np.abs(np.array(sols) - solution_bisection).any()> tol):
                    sols.append(solution_bisection)
            else:
                sols.append(solution_bisection)
            if len(sols) == 3:
                count += 1000
                break
            count += 1
        
    # solution = fsolve(r_stim, fprime=r_stim_prime,
    #                   args=(j_e, b_e), x0=x0, xtol=1e-16,
    #                   maxfev=1000)
    # solution = root(r_stim, args=(j_e, b_e), x0=x0, tol=1e-12,
    #                 method='broyden2').x
    return sols


def plot_j_b_crit_BP_vs_N(j_list=np.arange(0.001, 1.01, 0.01),
                          b_list=np.arange(-0.5, 0.5, 0.01),
                          tol=1e-12, min_r=-20, max_r=20,
                          w_size=0.1, neigh_list=np.arange(3, 11),
                          dim3=False):
    if dim3:
        ax = plt.figure().add_subplot(projection='3d')
    else:
        fig, ax = plt.subplots(1, figsize=(5, 4))
        fig2, ax2 = plt.subplots(1)
        ax2.set_xlabel('B')
        ax2.set_ylabel(r'$( J^{*}_{sim.} - J^{*}_{app.})^2$')
        colormap = pl.cm.Blues(np.linspace(0.2, 1, len(neigh_list)))
    for n_neigh in neigh_list:
        print(n_neigh)
        first_j = []
        for i_b, b in enumerate(b_list):
            for j in j_list:
                sol = find_solution_bp(j, b=b, min_r=min_r, max_r=max_r, w_size=w_size,
                                       tol=tol, n_neigh=n_neigh)
                if len(sol) > 1:
                    first_j.append(j)
                    break
            if len(first_j) != (i_b+1):
                first_j.append(np.nan)
        z = np.repeat(n_neigh, len(first_j))
        if dim3:
            ax.plot3D(z, b_list, first_j, color='k')
        else:
            sol_list = []
            for b in b_list:
                solution = fsolve(equation_for_g_derivative_at_1_eps_no_small,
                                  args=(b, n_neigh), x0=3, xtol=1e-10,
                                  maxfev=1000)
                sol_list.append(np.log(solution[0])/2)
            first_j_arr = np.array(first_j)
            ax2.plot(b_list, (np.array(sol_list)-first_j_arr)**2,
                     color=colormap[int(n_neigh-np.min(neigh_list))])
            ax.plot(b_list, first_j, color=colormap[int(n_neigh-min(neigh_list))],
                    label=n_neigh)
            # ax.plot(0, np.log(n_neigh/(n_neigh-2))/2, marker='o', color='k')
    vals_b0 = np.log(neigh_list / (neigh_list - 2)) / 2
    # vals = solution_of_g_with_stim(-0.01, neigh_list, pos_sqrt=False)
    # ax.plot3D(neigh_list, np.repeat(-0.01, len(neigh_list)), vals, color='b')
    if dim3:
        ax.plot3D(neigh_list, np.repeat(0, len(neigh_list)), vals_b0,
                  color='r', linestyle='--')
        ax.set_xlabel('N')
        ax.set_ylabel('B')
        ax.set_zlabel('J*')
    else:
        ax.set_xlabel(r'Sensory evidence $B$')
        ax.set_ylabel(r'Critical coupling $J^{\ast}$')
        ax_pos = ax.get_position()
        ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*1.05, ax_pos.y0+ax_pos.height*0.2,
                                ax_pos.width*0.06, ax_pos.height*0.5])
        newcmp = mpl.colors.ListedColormap(colormap)
        mpl.colorbar.ColorbarBase(ax_cbar, cmap=newcmp, label='Neighbors N')
        # ax_cbar.set_title('N')
        ax_cbar.set_yticks([0, 0.5, 1], [np.min(neigh_list),
                                         int(np.mean(neigh_list)),
                                         np.max(neigh_list)])
        ax_pos = ax2.get_position()
        ax_cbar = fig2.add_axes([ax_pos.x0+ax_pos.width*1.05, ax_pos.y0+ax_pos.height*0.2,
                                 ax_pos.width*0.06, ax_pos.height*0.5])
        newcmp = mpl.colors.ListedColormap(colormap)
        mpl.colorbar.ColorbarBase(ax_cbar, cmap=newcmp, label='Neighbors N')
        # ax_cbar.set_title('N')
        ax_cbar.set_yticks([0, 0.5, 1], [np.min(neigh_list),
                                         np.mean(neigh_list),
                                         np.max(neigh_list)])
        fig.savefig(DATA_FOLDER+'/J_vs_NB_BP.png', dpi=400, bbox_inches='tight')
        fig.savefig(DATA_FOLDER+'/J_vs_NB_BP.svg', dpi=400, bbox_inches='tight')
        fig2.savefig(DATA_FOLDER+'/J_vs_NB_BP_error.png', dpi=400, bbox_inches='tight')


def plot_j_b_crit_BP(j_list=np.arange(0.001, 1, 0.001),
                     b_list=np.arange(-0.5, 0.5, 0.001),
                     tol=1e-12, min_r=-20, max_r=20,
                     w_size=0.1):
    first_j = []
    for i_b, b in enumerate(b_list):
        for j in j_list:
            sol = find_solution_bp(j, b=b, min_r=min_r, max_r=max_r, w_size=w_size,
                                   tol=tol)
            if len(sol) > 1:
                first_j.append(j)
                break
        if len(first_j) != (i_b+1):
            first_j.append(np.nan)
            
    plt.figure()
    plt.plot(b_list, first_j, color='k', linewidth=2)
    plt.fill_between(b_list, first_j, 1, color='mistyrose', alpha=0.6)
    first_j = np.array(first_j)
    plt.text(-0.075, 0.9, '1 repulsor,\n2 attractor')
    plt.text(-0.075, 0.2, '1 attractor')
    plt.fill_between(b_list, 0, first_j, color='lightcyan', alpha=0.6)
    idx_neg = np.isnan(first_j) * (b_list < 0)
    plt.axhline(np.log(3)/2, color='r', linestyle='--', alpha=0.5)
    plt.fill_between(b_list[idx_neg],
                      np.repeat(0, np.sum(idx_neg)), 1, color='lightcyan', alpha=0.6)
    idx_pos = np.isnan(first_j) * (b_list > 0)
    plt.fill_between(b_list[idx_pos],
                      np.repeat(0, np.sum(idx_pos)), 1, color='lightcyan', alpha=0.6)
    plt.ylabel('J*')
    plt.xlabel('B')
    plt.ylim(0, max(j_list))
    plt.yticks([0, 0.5, np.log(3)/2, 1], ['0', '0.5', 'log(3)/2', '1'])
    plt.xlim(min(b_list), max(b_list))
            

def plot_bp_solution(ax, j_list, b, tol=1e-12, min_r=-20, max_r=20,
                     w_size=0.01, n_neigh=3, color='r'):
    sols = []
    for ind_j, j in enumerate(j_list):
        sol = []
        sol=find_solution_bp(j, b=b, min_r=min_r, max_r=max_r, w_size=w_size,
                             tol=tol, n_neigh=n_neigh)
        # sol = np.array(sol)
        # plt.plot(j_list, sol**3 / (1+sol**3))
        sols.append(sol)
        # sols = np.unique((np.round(sol, 4)))
    shape_array = max([len(x) for x in sols])
    solutions_array = np.empty((shape_array, len(j_list)))
    solutions_array[:] = np.nan
    for i in range(len(j_list)):
        if len(sols[i]) == 1:
            solutions_array[0, i] = sols[i][0]
            solutions_array[1, i] = np.nan
            solutions_array[2, i] = np.nan
        if len(sols[i]) == 2:
            solutions_array[0, i] = sols[i][1]
            solutions_array[1, i] = sols[i][0]
            solutions_array[2, i] = np.nan
        if len(sols[i]) == 3:
            solutions_array[0, i] = sols[i][2]
            solutions_array[1, i] = sols[i][1]
            solutions_array[2, i] = sols[i][0]
        if len(sols[i]) == 4:
            solutions_array[0, i] = sols[i][3]
            solutions_array[1, i] = sols[i][2]
            solutions_array[2, i] = sols[i][1]
            solutions_array[3, i] = sols[i][0]
        if len(sols[i]) == 5:
            solutions_array[0, i] = sols[i][4]
            solutions_array[1, i] = sols[i][3]
            solutions_array[2, i] = sols[i][2]
            solutions_array[3, i] = sols[i][1]
            solutions_array[4, i] = sols[i][0]
    linestyles = ['-', '--', '-', '-', '-']
    ax.set_ylim(-0.05, 1.05)
    solutions_array[(np.abs(solutions_array) >= 1) * (solutions_array < 0)] = np.nan
    for i_s, sol in enumerate(solutions_array):
        ax.plot(j_list,
                np.exp(b)*sol**3 / (np.exp(-b)+np.exp(b)*sol**3),
                color=color, linestyle=linestyles[i_s])
        if b == 0:
            ax.plot(j_list,
                    1-np.exp(b)*sol**3 / (np.exp(-b)+np.exp(b)*sol**3),
                    color=color, linestyle=linestyles[i_s])
    ax.set_xlabel('J')
    ax.set_ylabel('q')


def dynamical_system_BP_euler(j, stim, theta=THETA, noise=0, t_end=10, dt=1e-2,
                              ax=None, ylabel=True, plot=False, return_all=False):
    time = np.arange(0, t_end, dt)
    mu_y_1 = np.multiply(theta, np.random.rand(theta.shape[0], theta.shape[1]))
    mu_y_neg1 = np.multiply(theta, np.random.rand(theta.shape[0], theta.shape[1]))
    theta = theta*j
    q_y_1 = np.zeros((len(time), theta.shape[0]))
    q_y_neg1 = np.zeros((len(time), theta.shape[0]))
    if return_all:
        mu_y_1_all = np.zeros((len(time), theta.shape[0], theta.shape[1]))
        mu_y_neg1_all = np.zeros((len(time), theta.shape[0], theta.shape[1]))
    for i_t, t in enumerate(time):
        if return_all:
            mu_y_1_all[i_t, :, :] = mu_y_1
            mu_y_neg1_all[i_t, :, :] = mu_y_neg1
        for i in range(theta.shape[0]):
            q1 = np.prod(mu_y_1[np.where(theta[:, i] != 0), i]) * np.exp(stim)
            qn1 = np.prod(mu_y_neg1[np.where(theta[:, i] != 0), i]) * np.exp(-stim)
            q_y_1[i_t, i] = q1/(q1+qn1)
            q_y_neg1[i_t, i] = qn1/(q1+qn1)
        for i in range(theta.shape[0]):
            for m in np.where(theta[i, :] != 0)[0]:
                # positive y_i
                mu_y_1[m, i] += (np.exp(theta[i, m]+stim) *\
                        np.prod(mu_y_1[jneigbours(t, i, theta=theta), t])\
                        + np.exp(-theta[i, m]+stim) *\
                        np.prod(mu_y_neg1[jneigbours(t, i, theta=theta), t]) -
                        mu_y_1[m, i])*dt +\
                    np.sqrt(dt)*noise*np.random.randn()
                # negative y_i
                mu_y_neg1[m, i] += (np.exp(-theta[i, m]-stim) *\
                    np.prod(mu_y_1[jneigbours(t, i, theta=theta), t])\
                    + np.exp(theta[i, m]-stim) *\
                    np.prod(mu_y_neg1[jneigbours(t, i, theta=theta), t])-
                    mu_y_neg1[m, i])*dt +\
                np.sqrt(dt)*noise*np.random.randn()
                m_y_1_memory = np.copy(mu_y_1[m, i])
                mu_y_1[m, i] = mu_y_1[m, i]/(m_y_1_memory+mu_y_neg1[m, i])
                mu_y_neg1[m, i] = mu_y_neg1[m, i]/(m_y_1_memory+mu_y_neg1[m, i])
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1)
        for q in q_y_1.T:
            ax.plot(time, q, alpha=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title('J = ' + str(j) + ', B = ' + str(stim))
        ax.set_xlabel('Time (s)')
        if ylabel:
            ax.set_ylabel(r'$q_i(x=1)$')
        else:
            ax.set_yticks([])
    if not return_all:    
        return q_y_1, q_y_neg1
    if return_all:
        return q_y_1, q_y_neg1, mu_y_1_all, mu_y_neg1_all


def sols(b, jp, N):
    """
    returns value of epsilon ,from solution r=1+epsilon from BP
    with bias (after expanding exp(2B) to 1+2B)
    """
    a = 2*b
    j = np.exp(2*jp)
    return a*(j-1) / (N-j*(N-1)+j)


def g(r, b, j, N):
    return np.exp(2*b)*r**N - np.exp(2*(j+b))*r**(N-1) + np.exp(2*j)*r-1


def dg_dr(j, b, r, N):
    return N*np.exp(2*b)*r**(N-1) - (N-1)*np.exp(2*(j+b))*r**(N-2) + np.exp(2*j)


def g_tay(r, b, j, N):
    return r**N * (1+2*b)- np.exp(2*(j))*r**(N-1)*(1+2*b) + np.exp(2*j)*r-1


def g_der(b, jp, N):
    """
    Returns value of derivative of solution of the polynomial from
    the solution of the BP with bias, evaluated at 1+epsilon and 
    with exp(2B) expanded as 1+2B.
    """
    j = np.exp(2*jp)
    a = 2*b
    return N*(1+a)*(1+(N-1)*sols(b,jp,N))-(N-1)*j*(1+a)*(1+(N-2)*sols(b,jp,N))+j


def solution_of_g_with_stim(b, N, pos_sqrt=False):
    # b = np.exp(2*b)
    b = 2*b
    if pos_sqrt:
        val = 1
    else:
        val = -1
    return np.log((b*N-2*b-2
                    + val* np.sqrt((-b*N+2*b+2)**2 - 4 *
                              (-N*b*b+b*b+b+N) *
                              (b*b*N-b*b +b*N - b - N +2))) /
                  (2 * (b*b*N - b*b + b*N - b - N + 2)))/2
    # numerator = (-np.sqrt((-b * N**2 + 2 * b * N - N**2 + 3 * N)**2 - 4 * (b**2 * N - b**2 + b * N - b - N + 2) * (b**2 * (-N) + b**2 + b * N**2 - b * N + b + N**2)) + b * N**2 - 2 * b * N + N**2 - 3 * N)
    # denominator = 2 * (b**2 * N - b**2 + b * N - b - N + 2)
    # x = numerator / denominator
    # return np.log(x)/2


def equation_for_g_derivative_at_1_eps(x, b, N):
    lhs = sols(b, np.log(x)/2, N)
    b = 2*b
    rhs_numerator = x*(b+2) - (b+1)*(N)*(x-1)
    rhs_denominator = (b+1)*(N-1)*(N*(x-1)-2*x)
    return lhs*rhs_denominator - rhs_numerator


def equation_for_g_derivative_at_1_eps_no_small(x, b, N):
    eps = sols(b, np.log(x)/2, N)
    b = 2*b
    eq = N*(1+b)*(1+eps)**(N-1) - (N-1)*(1+b)*x*(1+eps)**(N-2) + x
    return eq


def solve_equation_g_derivative(neigh_list = np.arange(3, 10),
                                b_list=np.arange(-0.1, 0.1, 0.001),
                                fun_approx=False):
    colormap = pl.cm.Blues(np.linspace(0.2, 1, len(neigh_list)))
    fig, ax = plt.subplots(1)
    if fun_approx:
        fun = equation_for_g_derivative_at_1_eps
    else:
        fun = equation_for_g_derivative_at_1_eps_no_small
    for n in neigh_list:
        sol_list = []
        for b in b_list:
            solution = fsolve(fun,
                                args=(b, n), x0=3, xtol=1e-10,
                                maxfev=1000)
            sol_list.append(np.log(solution)/2)
        plt.plot(b_list, sol_list, color=colormap[int(n-np.min(neigh_list))])
        plt.plot(0, np.log(n/(n-2))/2, marker='o', color='k')
    plt.xlabel('B')
    plt.ylabel('J*')
    ax_pos = ax.get_position()
    ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*1.05, ax_pos.y0+ax_pos.height*0.2,
                            ax_pos.width*0.06, ax_pos.height*0.5])
    newcmp = mpl.colors.ListedColormap(colormap)
    mpl.colorbar.ColorbarBase(ax_cbar, cmap=newcmp)
    ax_cbar.set_title('N')
    ax_cbar.set_yticks([0, 0.5, 1], [np.min(neigh_list),
                                     np.mean(neigh_list),
                                     np.max(neigh_list)])
    fig.savefig(DATA_FOLDER + 'J_vs_NB_numerical_poly_solution_BP.png')


def plot_sol_j_vs_N(stim_list=[0, .01, 0.05, 0.1],
                    N_list=np.arange(3, 20)):
    plt.figure()
    colormap = pl.cm.Blues(np.linspace(0.2, 1, len(stim_list)))
    for i_s, st in enumerate(stim_list):
        plt.plot(N_list, solution_of_g_with_stim(st, N_list), alpha=1, color=colormap[i_s],
                 label=st)
    plt.plot(N_list, 0.5*np.log(N_list/(N_list-2)), label='orig b=0',
             color='r', linestyle='--')
    plt.legend(title='B')
    plt.xlabel('N')
    plt.ylabel('J*')


def plot_g_and_j_approx_sol(N, j, b_val):
    r=np.arange(-0.5, 1.5, 1e-4)
    colors = ['b', 'g', 'k']
    plt.figure()
    for i_b, b in enumerate([0, b_val]):
        plt.plot(r,g(r, b, j, N), label=b, color=colors[i_b])
        plt.axvline(1+sols(b,j,N), color=colors[i_b])
    plt.legend(title='B')
    plt.xlabel('r')
    plt.ylabel(r'$g(r,J,B,N)$')
    plt.title('J = {}, N = {}'.format(j, N))
    plt.axhline(0, color='r')
    eps = np.abs(sols(b,j,N))
    plt.xlim(1-4*eps, 1+4*eps)
    plt.ylim(-4*eps, 4*eps)


def plot_g_der_sols(N, plot_der=False):
    if plot_der:
        func = g_der
        title = 'derivative at r=1+e'
    else:
        func = sols
        title = 'epsilon'
    plt.figure()
    bvals = np.linspace(-0.025, 0.025, 1001)
    jvals = np.linspace(0., 1.0, 1001)
    bv, jv = np.meshgrid(bvals, jvals)
    z = func(bv, jv, N)
    im = plt.imshow(np.flipud(z), cmap='coolwarm', vmin=-2, vmax=2)
    cb = plt.colorbar(im)
    cb.ax.set_title(title)
    plt.yticks(np.arange(0, 1001, 100), np.round(jvals[::100], 2)[::-1])
    plt.ylabel('J')
    plt.xticks(np.arange(0, 1001, 100), np.round(bvals[::100], 3))
    plt.xlabel('B')
    plt.axhline(1000-np.log(N/(N-2))*0.5*1e3, color='k')
    plt.title(title + ', N = '+str(N))


def plot_solutions_BP_depending_neighbors(j_list=np.arange(0.001, 1, 0.001),
                                          neigh_list=[3, 4, 5], b=0):
    fig, ax = plt.subplots(ncols=3, figsize=(10, 6))
    colors = ['k', 'r', 'b']
    neigh_list = [3, 4, 5]
    for n_neigh in neigh_list:
        plot_bp_solution(ax[n_neigh-min(neigh_list)], j_list, b=b, tol=1e-10,
                         min_r=0, max_r=15,
                         w_size=0.05, n_neigh=n_neigh,
                         color=colors[n_neigh-min(neigh_list)])
        ax[n_neigh-min(neigh_list)].set_title(str(n_neigh) + ' neighbors')



def plot_lbp_3_examples():
    j_list = [0.25, 0.25, 0.58]
    b_list = [0, 0.1, 0]
    fig, ax = plt.subplots(ncols=3, figsize=(10, 3.5))
    i = 0
    dtlist = [5e-2, 5e-2, 5e-1]
    time_end = [30, 30, 1000]
    for j, b, dt, t_end in zip(j_list, b_list, dtlist, time_end):
        dynamical_system_BP_euler(j=j, stim=b, theta=THETA, noise=0.08,
                                  t_end=t_end, dt=dt, ax=ax[i],
                                  ylabel=i==0)
        i += 1
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, bottom=0.16, top=0.88)


def plot_lbp_explanation(eps=2e-1):
    fig, ax = plt.subplots(1, figsize=(4, 3))
    gn.plot_necker_cubes(ax, mu=8, bot=True, offset=0.6, factor=1.5,
                         msize=10)
    plt.axis('off')
    x_pos_1 = 5.4
    y_pos_1 = 15.3
    y_pos_3 = 12
    y_pos_5 = 15.9
    x_pos_5 = 6
    x_pos_2 = 6.9
    ax.arrow(x_pos_5-eps, y_pos_5+eps, x_pos_1-x_pos_5+eps, y_pos_1-y_pos_5+eps,
             head_width=eps/2, color='k', head_length=eps/2)
    ax.arrow(x_pos_1-eps/2, y_pos_3+10*eps, 0, y_pos_1-y_pos_3-12*eps,
             head_width=eps/2, color='k', head_length=eps/2)
    ax.arrow(x_pos_2-5*eps, y_pos_1-eps, x_pos_1-x_pos_2+6*eps, 0,
             head_width=eps/2, color='k', head_length=eps/2)
    
def mse(p, q):
    return np.sqrt(np.sum((p-q)**2))


def all_comparison_together(j_list=np.arange(0., 1.005, 0.01),
                            b_list=np.arange(-1, 1, 0.01),
                            data_folder=DATA_FOLDER,
                            theta=THETA, dist_metric='mse', nrows=2):
    if dist_metric is not None:
        if dist_metric == 'mse':
            dist = mse
            label_0 = ', MSE = '
        if dist_metric == 'kl':
            dist = kl
            label_0 = ', KL = '
    if nrows == 2:
        figsize = (6.5, 4.5)
        ncols = 3
    else:
        figsize = (12, 2)
        ncols = 6
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.2)
    ax = ax.flatten()
    # true posterior
    matrix_true_file = data_folder + 'true_vs_JB_05.npy'
    os.makedirs(os.path.dirname(matrix_true_file), exist_ok=True)
    print('True posterior')
    if os.path.exists(matrix_true_file):
        mat_true = np.load(matrix_true_file, allow_pickle=True)
    else:
        mat_true = np.empty((len(j_list), len(b_list)))
        mat_true[:] = np.nan
        for i_j, j in enumerate(j_list):
            true_posterior = gn.true_posterior_stim(stim_list=b_list, j=j,
                                                    theta=theta,
                                                    data_folder=data_folder,
                                                    load_data=False)
            mat_true[i_j, :] = true_posterior
        np.save(matrix_true_file, mat_true)
    ax[0].imshow(np.flipud(mat_true), aspect='auto', interpolation=None,
                 extent=[-.5, .5, 0, 1], cmap='coolwarm_r', vmin=0, vmax=1)
    ax[0].set_title('True posterior', fontsize=14)
    # Mean-field
    matrix_mf_file = data_folder + 'mf_posterior_vs_JB_sim_05.npy'
    jcrit_mf_file = data_folder + 'mf_jcrit_vs_b_n3_v2.npy'
    jcrit_mf = np.load(jcrit_mf_file, allow_pickle=True)
    os.makedirs(os.path.dirname(matrix_mf_file), exist_ok=True)
    print('MF')
    if os.path.exists(matrix_mf_file):
        mat_mf = np.load(matrix_mf_file, allow_pickle=True)
    else:
        mat_mf = np.empty((len(j_list), len(b_list)))
        mat_mf[:] = np.nan
        # N=3
        for i_b, b in enumerate(b_list):
            l = []
            for j in j_list:
                # q1 = lambda q: gn.sigmoid(2*N*j*(2*q-1)+ b*2*N) - q 
                # sol_1, _, flag1, _ =\
                #     fsolve(q1, 1, full_output=True)
                # if flag1 != 1:
                #     sol_1, _, flag, _ =\
                #         fsolve(q1, 0, full_output=True)
                # l.append(sol_1)
                q = mfn.mean_field_stim(j, num_iter=40, stim=b, sigma=0,
                                        theta=theta)
                l.append(q[-1, 0])
                
            mat_mf[:, i_b] = l
        np.save(matrix_mf_file, mat_mf)
    ax[1].imshow(np.flipud(mat_mf), aspect='auto', interpolation='none',
                 extent=[-.5, .5, 0, 1], cmap='coolwarm_r', vmin=0, vmax=1)
    first_j = jcrit_mf
    b_list_1 = np.arange(-1, 1, 0.01)
    ax[1].plot(b_list_1, first_j, color='k')
    ax[1].set_xlim(-0.5, 0.5)
    if dist_metric is not None:
        label = label_0 + str(dist(mat_true, mat_mf))
    else:
        label = ''
    ax[1].set_title('Mean-field'+ label, fontsize=14)
    # belief propagation
    matrix_bp_file = data_folder + 'lbp_posterior_vs_JB_05.npy'
    jcrit_bp_file = data_folder + 'jcrit_vs_b_n3.npy'
    os.makedirs(os.path.dirname(matrix_bp_file), exist_ok=True)
    jcrit_bp = np.load(jcrit_bp_file, allow_pickle=True)
    print('BP')
    if os.path.exists(matrix_bp_file):
        mat_lbp = np.load(matrix_bp_file, allow_pickle=True)
    else:
        mat_lbp = np.empty((len(j_list), len(b_list)))
        mat_lbp[:] = np.nan
        for i_b, b in enumerate(b_list):
            l = []
            for j in j_list:
                pos, neg, n = Loopy_belief_propagation(theta=theta,
                                                       num_iter=100,
                                                       j=j, thr=1e-5, stim=b)
                l.append(pos[0])
            mat_lbp[:, i_b] = l
        np.save(matrix_bp_file, mat_lbp)
    ax[2].imshow(np.flipud(mat_lbp), aspect='auto', interpolation='none',
                 extent=[-.5, .5, 0, 1], cmap='coolwarm_r', vmin=0, vmax=1)
    if dist_metric is not None:
        label = label_0 + str(dist(mat_true, mat_lbp))
    else:
        label = ''
    ax[2].set_title('Loopy\nbelief propagation' + label, fontsize=14)
    ax[2].plot(b_list_1, jcrit_bp, color='k', label=r'$J^{\ast}$')
    ax[2].set_xlim(-0.5, 0.5)
    ax[2].legend(bbox_to_anchor=(0, 1.47), frameon=False)
    
    # Gibbs for 3 different T
    # T=100
    print('GS T=100')
    matrix_gn_file = data_folder + '100_gibbs_posterior_vs_JB_05.npy'
    os.makedirs(os.path.dirname(matrix_gn_file), exist_ok=True)
    if os.path.exists(matrix_gn_file):
        mat_gn = (np.load(matrix_gn_file, allow_pickle=True)+1)/2
    else:
        mat_gn = np.empty((len(j_list), len(b_list)))
        mat_gn[:] = np.nan
        for i_b, b in enumerate(b_list):
            if (i_b+1) % 10 == 0:
                print(i_b+1)
            l = []
            for j in j_list:
                init_state = np.random.choice([-1, 1], 8)
                states_mat = gn.gibbs_samp_necker(init_state=init_state,
                                                  burn_in=1000,
                                                  n_iter=1100, j=j,
                                                  stim=b)
                l.append(np.nanmean(states_mat))
            mat_gn[:, i_b] = l
        np.save(matrix_gn_file, mat_gn)
    if dist_metric is not None:
        label = label_0 + str(dist(mat_true, mat_gn))
    else:
        label = ''
    ax[3].imshow(np.flipud(mat_gn), aspect='auto',
                 extent=[-.5, .5, 0, 1], cmap='coolwarm_r', vmin=0, vmax=1)
    ax[3].set_title('Gibbs sampling\nT=1e2' + label, fontsize=14)
    # ax[3].plot(b_list, (np.log(100)+8*b_list*np.sign(b_list))/10, color='k')
    # T=10000
    matrix_gn_file = data_folder + '1000_gibbs_posterior_vs_JB_05.npy'
    os.makedirs(os.path.dirname(matrix_gn_file), exist_ok=True)
    print('GS T=1000')
    if os.path.exists(matrix_gn_file):
        mat_gn = (np.load(matrix_gn_file, allow_pickle=True)+1)/2
    else:
        mat_gn = np.empty((len(j_list), len(b_list)))
        mat_gn[:] = np.nan
        for i_b, b in enumerate(b_list):
            if (i_b+1) % 10 == 0:
                print(i_b+1)
            l = []
            for j in j_list:
                init_state = np.random.choice([-1, 1], 8)
                states_mat = gn.gibbs_samp_necker(init_state=init_state,
                                                  burn_in=1000,
                                                  n_iter=2000, j=j,
                                                  stim=b)
                l.append(np.nanmean(states_mat))
            mat_gn[:, i_b] = l
        np.save(matrix_gn_file, mat_gn)
    if dist_metric is not None:
        label = label_0 + str(dist(mat_true, mat_gn))
    else:
        label = ''
    ax[4].imshow(np.flipud(mat_gn), aspect='auto',
                 extent=[-.5, .5, 0, 1], cmap='coolwarm_r', vmin=0, vmax=1)
    ax[4].set_title('Gibbs sampling\nT=1e3' + label, fontsize=14)
    # ax[4].plot(b_list, (np.log(1000)+8*b_list*np.sign(b_list))/10, color='k')
    ax[4].set_ylim(0, 1)
    # T=100000
    matrix_gn_file = data_folder + '10000_gibbs_posterior_vs_JB_05.npy'
    os.makedirs(os.path.dirname(matrix_gn_file), exist_ok=True)
    print('GS T=10000')
    if os.path.exists(matrix_gn_file):
        mat_gn = (np.load(matrix_gn_file, allow_pickle=True)+1)/2
    else:
        mat_gn = np.empty((len(j_list), len(b_list)))
        mat_gn[:] = np.nan
        for i_b, b in enumerate(b_list):
            if (i_b+1) % 10 == 0:
                print(i_b+1)
            l = []
            for j in j_list:
                init_state = np.random.choice([-1, 1], 8)
                states_mat = gn.gibbs_samp_necker(init_state=init_state,
                                                  burn_in=1000,
                                                  n_iter=11000, j=j,
                                                  stim=b)
                l.append(np.nanmean(states_mat))
            mat_gn[:, i_b] = l
        np.save(matrix_gn_file, mat_gn)
    # mat_gn = resize(mat_gn, (201, 200))
    if dist_metric is not None:
        label = label_0 + str(dist(resize(mat_true, (41, 40)), mat_gn))
    else:
        label = ''
    im = ax[5].imshow(np.flipud(mat_gn), aspect='auto',
                      extent=[-.5, .5, 0, 1], cmap='coolwarm_r', vmin=0, vmax=1,
                      interpolation=None)
    ax[5].set_title('Gibbs sampling\nT=1e4' + label, fontsize=14)
    # ax[5].plot(b_list, (np.log(10000)+8*b_list*np.sign(b_list))/10, color='k')
    ax[5].set_ylim(0, 1)
    ax_pos = ax[5].get_position()
    ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*1.05, ax_pos.y0+ax_pos.height*0.1,
                            ax_pos.width*0.06, ax_pos.height*0.7])
    plt.colorbar(im, cax=ax_cbar, orientation='vertical', label='Posterior')
    # ax[0].set_ylabel(r'Coupling $J$')
    if nrows != 2:
        for i in range(1, 6):
            ax[i].set_yticks([])
        for i in range(6):
            ax[i].set_xlabel(r'Stimulus $B$')
    else:
        ax[1].set_yticks([])
        ax[2].set_yticks([])
        ax[1].set_xticks([])
        ax[0].set_xticks([])
        ax[2].set_xticks([])
        ax[4].set_yticks([])
        ax[5].set_yticks([])
        for a in [ax[3], ax[4], ax[5]]:
            a.set_xticks([-0.5, 0, 0.5])
            a.tick_params(axis='x', labelrotation=45)
        ax[3].set_ylabel(r'                                  Coupling $J$')
        # ax[3].set_xlabel(r'Stimulus $B$')
        ax[4].set_xlabel(r'Sensory evidence, $B$')
        # ax[5].set_xlabel(r'Stimulus $B$')
    fig.savefig(data_folder+'/comparison_all_v1.png', dpi=400, bbox_inches='tight')
    fig.savefig(data_folder+'/comparison_all_v1.svg', dpi=400, bbox_inches='tight')


def kl(p, q, eps=1e-10):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    
    return np.round(np.sum(np.where(p != 0, p * np.log(p / q), 0)), 2)


def potential_lbp(r, j, b, n, q1=True):
    z = np.exp(2*j)
    y = np.exp(2*b)
    f = scipy.special.hyp2f1
    # if not q1:
    #     q = 1-q
    # r = (1/(y*(1/q-1)))**(1/n)
    if q1:
        r = (1/(y*(1/r-1)))**(1/n)
    vals = (z-1/z)*f(1, 1/(1-n), 1+1/(1-n), -z/y*r**(1-n))
    pot_0 = 0.5*r**2
    pot_1 =  - r/z
    pot_2 = - r*vals
    return pot_0 + pot_1 + pot_2


def ratio_evolution(r_0, j, b, n=3, n_iter=20, dt=1e-3):
    r = r_0
    e2j = np.exp(2*j)
    e2b = np.exp(2*b)
    r_list = [r]
    v = e2b*r_0**n
    time = np.arange(0, n_iter, dt)
    for i in range(1, len(time)):
        r += dt*((1 + e2j*e2b*r**(n-1))/(e2j + e2b*r**(n-1))-r)
        r_list.append(r)
    v_list = [v]
    e2bn = np.exp(2*b/n)
    for k in range(1, len(time)):
        v += dt*(-v*n + e2bn * n * v**(1-1/n) * (1+e2bn*e2j*v**(1-1/n)) / (e2j +e2bn*v**(1-1/n)))
        v_list.append(v)
    r_list = np.array(r_list)
    # plt.figure()
    # plt.plot(time, r_list)
    # plt.ylabel('r')
    plt.figure()
    plt.plot(time, np.exp(b)*r_list**n / (np.exp(b)*r_list**n + np.exp(-b)), label='q1')
    plt.xlabel('time')
    plt.ylabel('approximated posterior q')
    plt.legend()
    plt.figure()
    v_list = np.array(v_list)
    plt.plot(time, 1/(1+v_list), label='v-sys')
    plt.plot(time, np.exp(-b) / (np.exp(b)*r_list**n + np.exp(-b)), linestyle='--',
             label='q2')
    plt.xlabel('time')
    plt.ylabel('approximated posterior q')
    plt.legend()
    # plt.figure()
    # v_list = np.array(v_list)
    # plt.plot(time, 1/(1+v_list))


def plot_potential_lbp(q=np.arange(0.0001, 1, 0.0001),
                       j_list=[0.1, 0.4, 0.5, np.log(3)/2, 0.57, 0.585, 0.6, .65],
                       b=0, n=3):
    # colormap = pl.cm.Purples(np.linspace(0.1, 1, len(j_list)))
    plt.figure()
    for i_j, j in enumerate(j_list):
        pot = pot_lbp(q, j, b, n)
        pot_min = pot-np.min(pot)
        if b >= 0:
            ind = q > 0.5
        else:
            ind = q > 0.8
        val_norm = np.max(pot_min[ind])
        if j == np.log(3)/2:
            label = r'$J^{\ast}=log(3)/2$'
        else:
            label = str(j)
        plt.plot(q, pot_min/val_norm, label=label)  # , color=colormap[i_j]
    plt.ylim(-.05, 1.1)
    plt.legend(title='Coupling J')
    plt.xlabel(r'$\nu=1/q(x=-1)-1$')
    # plt.xlabel('Approximate posterior q(x=-1)')
    plt.ylabel(r'Normalized potential $V(\nu)$')
    plt.title('B = ' + str(b))


def pot_lbp(v, j, b, n, q=False):
    f = scipy.special.hyp2f1
    k = np.exp(2*j)
    c = np.exp(2*b/n)
    if q:
        v = (1/v - 1)
    pot_0 = 0.5*n*v**2
    pot_1 = c*k*n*(v**(1-1/n)) / (1-2*n)
    pot_2 = (k**2-1)*f(1, 1/(1/n - 1), 1/(1-n), -k/c*v**(1/n-1))
    return pot_0 + n*v*(pot_1 + pot_2)


def plot_sol_from_potential(q=np.arange(0.0001, 1, 0.001),
                            j_list=np.arange(0.001, 1, 0.001),
                            b=0, n=3):
    sol = []
    for j in j_list:
        ind = np.where(pot_lbp(q,j,b,n) == np.min(pot_lbp(q,j,b,n)))[0][0]
        sol.append(q[ind])
    plt.figure()
    plt.xlabel('Coupling J')
    plt.ylabel('Approximate posterior q(x=1)')
    plt.plot(j_list, 1-np.array(sol), color='k')
    if b == 0:
        plt.plot(j_list, sol, color='k')
    plt.ylim(-0.05, 1.05)


def plot_potentials_lbp(j_list, b=0, neighs=3, q1=True):
    # colormap = pl.cm.Blues(np.linspace(0.2, 1, len(j_list)))
    # colormap_1 = pl.cm.Purples(np.linspace(0.4, 1, len(j_list)))
    Blues = pl.cm.get_cmap('Blues', 100)
    Purples = pl.cm.get_cmap('Purples', 100)
    newcolors = Blues(np.linspace(0.2, 1, len(j_list)))
    newcolors_purples = Purples(np.linspace(0.4, 1, len(j_list)))
    red = np.array([1, 0, 0, 1])
    ind_red = np.where(np.round(j_list, 1) == np.round(np.log(3)/2, 1))[0][0]
    newcolors[ind_red, :] = red
    newcolors[(ind_red+1):, :] = newcolors_purples[(ind_red+1):]
    newcmp = mpl.colors.ListedColormap(newcolors)
    q = np.arange(0.001, 10., 0.001)
    fig, ax = plt.subplots(1, figsize=(6, 4))
    change_colormap = False
    for i_j, j in enumerate(j_list):
        pot = potential_lbp(q, j, b=b, n=neighs, q1=q1)
        # pot = potential_expansion_any_order_any_point(q, j, b=0, order=8, point=0.5)
        # norm_cte = np.max(np.abs(pot))
        if j == ind_red and not change_colormap:
            color = 'r'
            change_colormap = True
        else:
            color = newcolors[i_j]
        ax.plot(q, pot-np.mean(pot), color=color, label=np.round(j, 2))
    ax_pos = ax.get_position()
    ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*1.02, ax_pos.y0,
                            ax_pos.width*0.04, ax_pos.height*0.9])
    mpl.colorbar.ColorbarBase(ax_cbar, cmap=newcmp, label=r'Coupling $J$')
    ax_cbar.set_yticks([0, max(j_list)/2, np.log(3)/2, max(j_list)], [0, max(j_list)/2, 'J*', max(j_list)])
    # ax_cbar.set_title(r'Coupling $J$')
    # if q1:
    #     ax.set_xlabel(r'Approx posterior $q(x=1)$')
    # else:
    #     ax.set_xlabel(r'Approx posterior $q(x=-1)$')
    ax.set_xlabel(r'Message ratio $r$')
    ax.set_ylabel(r'Mean-centered potential $V_J(q)$')
    # ax.set_ylim(-4, 1)
    # ax.legend(title='J:')
    fig.savefig(DATA_FOLDER + 'potentials_vs_q.png', dpi=400, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'potentials_vs_q.svg', dpi=400, bbox_inches='tight')


def plot_sols_FLBP(alphalist=np.linspace(0, 1, 10),
                   j_list=np.arange(0, 3, 0.001), theta=THETA,
                   num_iter=100, stim=0):
    plt.figure()
    colormap = pl.cm.Blues(np.linspace(0.2, 1, len(alphalist)))
    for ia, alpha in enumerate(alphalist):
        sols_pos = []
        sols_neg = []
        for j in j_list:
            pos, neg = discrete_DBN(j, b=stim, theta=THETA, num_iter=100, thr=1e-10,
                                    alpha=alpha)
            sols_pos.append(np.max((pos[0], neg[0])))
            sols_neg.append(np.min((pos[0], neg[0])))
        n = 3
        plt.axvline(np.log(n/(n-2*alpha))/(2*alpha), color=colormap[ia],
                    alpha=0.3, linestyle='--')
        plt.plot(j_list, sols_pos, color=colormap[ia], label=alpha)
        plt.plot(j_list, sols_neg, color=colormap[ia])
    plt.xlabel('Coupling J')
    plt.legend()
    plt.ylabel('Approximated posterior q')
    # n = 3
    # plt.axvline(np.log(n/(n-2))/(2), color='r', alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.05, j_list[-1]+5e-2)


def plot_lbp_hysteresis(j_list=[0.05, 0.4, 0.7],
                        b_list=np.linspace(-0.5, 0.5, 1001),
                        theta=THETA):
    b_list = np.concatenate((b_list[:-1], b_list[::-1]))
    plt.figure()
    colormap = ['navajowhite', 'orange', 'saddlebrown']
    for i_j, j in enumerate(reversed(j_list)):
        vec = lbp_changing_stim(j, b_list,
                                theta=theta)
        plt.plot(b_list, vec[:, 0], color=colormap[i_j],
                 label=np.round(j, 1), linewidth=4)
    plt.xlabel('Sensory evidence, B')
    plt.ylabel('Approximate posterior q(x=1)')
    plt.legend(title='J')


def lbp_changing_stim(j, b_list, theta=THETA,
                      burn_in=10):
    num_iter = len(b_list)+burn_in
    b_list = np.concatenate((np.repeat(b_list[0], burn_in),
                             b_list))
    mu_y_1 = np.multiply(theta, np.random.rand(theta.shape[0], theta.shape[1]))
    mu_y_neg1 = np.multiply(theta, np.random.rand(theta.shape[0], theta.shape[1]))
    q_y_1 = np.zeros((num_iter, theta.shape[0]))
    q_y_neg1 = np.zeros((num_iter, theta.shape[0]))
    for n in range(num_iter):
        stim = b_list[n]
        # for all the nodes that i is connected
        for i in range(theta.shape[0]):
            for t in np.where(theta[i, :] != 0)[0]:
                # positive y_i
                mu_y_1[t, i] = np.exp(j*theta[i, t]+stim) *\
                        (mu_y_1[jneigbours(t, i, theta=theta)[0], t]*
                         mu_y_1[jneigbours(t, i, theta=theta)[1], t])\
                        + np.exp(-j*theta[i, t]-stim) *\
                        (mu_y_neg1[jneigbours(t, i, theta=theta)[0], t] *
                         mu_y_neg1[jneigbours(t, i, theta=theta)[1], t])
                # mu_y_1 += np.random.rand(8, 8)*1e-3
                # negative y_i
                mu_y_neg1[t, i] = np.exp(-j*theta[i, t]+stim) *\
                    (mu_y_1[jneigbours(t, i, theta=theta)[0], t] *\
                     mu_y_1[jneigbours(t, i, theta=theta)[1], t])\
                    + np.exp(j*theta[i, t]-stim) *\
                    (mu_y_neg1[jneigbours(t, i, theta=theta)[0], t] *
                     mu_y_neg1[jneigbours(t, i, theta=theta)[1], t])

                m_y_1_memory = np.copy(mu_y_1[t, i])
                mu_y_1[t, i] = mu_y_1[t, i]/(m_y_1_memory+mu_y_neg1[t, i])
                mu_y_neg1[t, i] = mu_y_neg1[t, i]/(m_y_1_memory+mu_y_neg1[t, i])
                # mu_y_neg1 += np.random.rand(8, 8)*1e-3
            q1 = np.prod(mu_y_1[np.where(theta[:, i] != 0), i]) * np.exp(stim)
            qn1 = np.prod(mu_y_neg1[np.where(theta[:, i] != 0), i]) * np.exp(-stim)
            q_y_1[n, i] = q1/(q1+qn1)
            q_y_neg1[n, i] = qn1/(q1+qn1)
    # gn.plot_cylinder(q=q_y_1.reshape(5, 10, 2),
    #                  columns=5, rows=10, layers=2, offset=0.4, minmax_norm=True)
    return q_y_1[burn_in:]


def potential_LBP_v0(j, b, N, r=np.linspace(0, 10, 1000)):
    return (np.exp(2*b)*r**(N+1)/(N+1) - np.exp(2*(j+b))*r**(N)/(N) + np.exp(2*j)*r**2/2-r)


def plot_m1_m2_vector_field(j, b, n=3):
    x = np.linspace(0, .6, 20)
    y = np.linspace(0, .6, 20)
    xx, yy = np.meshgrid(x, y)
    uu = (np.exp(j+b)*xx**(n-1)+np.exp(-j-b)*yy**(n-1)-xx)
    vv = (np.exp(-j+b)*xx**(n-1)+np.exp(j-b)*yy**(n-1)-yy)
    fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
    ax[0].quiver(xx, yy, uu, vv)
    ax[0].set_xlim(0, max(x))
    ax[0].set_ylim(0, max(x))
    x = np.linspace(0, max(x), 5000)
    y = x
    ax[0].plot(y, ((x-np.exp(j+b)*x**2)*np.exp(j+b))**(1/(n-1)),
               color='k')
    ax[0].plot(((x-np.exp(j-b)*x**2)*np.exp(j-b))**(1/(n-1)), x,
               color='k', label='Nullclines')
    ax[0].legend(frameon=False, bbox_to_anchor=(0.7, 1.1))
    ax[0].set_xlabel(r'$m(x=1)$')
    ax[0].set_ylabel(r'$m(x=-1)$')
    x = np.linspace(0, .6, 5000)
    y = np.linspace(0, .6, 5000)
    xx, yy = np.meshgrid(x, y)
    uu = (np.exp(j+b)*xx**(n-1)+np.exp(-j-b)*yy**(n-1)-xx)
    vv = (np.exp(-j+b)*xx**(n-1)+np.exp(j-b)*yy**(n-1)-yy)
    modulo = np.sqrt(vv**2 + uu**2)
    idx_m1 = np.where(modulo < 1e-4)[0]
    idx_m2 = np.where(modulo < 1e-4)[1]
    for i in range(len(idx_m1)):
        ax[1].plot(x[idx_m2[i]], y[idx_m1[i]], color='r', marker='x',
                   markersize=9)
    image = ax[1].imshow(np.flipud(modulo), extent=[0, max(y), 0, max(x)],
                         cmap='gist_gray', aspect='auto',
                         norm=mpl.colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                    vmin=1e-8,
                                                    vmax=np.nanmax(modulo), base=10))
    ax[1].set_ylim(-0.01, 0.61)
    ax[1].set_xlim(-0.01, 0.61)
    plt.colorbar(image, label=r'speed, $||f(x_1, x_2)||$', ax=ax[1])
    ax[1].set_xlabel(r'$m(x=1)$')
    ax[1].set_ylabel(r'$m(x=-1)$')
    ax3d = plt.figure().add_subplot(projection='3d')
    ax3d.plot_surface(xx, yy, modulo, cmap='pink',
                      norm=mpl.colors.LogNorm(vmin=1e-3,
                                              vmax=np.nanmax(modulo)))
    # pot = np.exp(j+b)*xx**(n)/n+np.exp(-j-b)*yy**(n)/n-xx**2 / 2 - \
    #         (np.exp(-j+b)*xx**(n)/n+np.exp(j-b)*yy**(n)/n-yy ** 2 / 2)
    # ax3d.plot_surface(xx, yy, pot, cmap='pink')


def plot_posterior_vs_stim(j_list=[0.05, 0.4, 0.7],
                           b_list=np.linspace(0, 0.25, 101),
                           theta=THETA, thr=1e-8, num_iter=100):
    plt.figure()
    # colormap = pl.cm.Oranges(np.linspace(0.4, 1, len(j_list)))
    colormap = ['navajowhite', 'orange', 'saddlebrown']
    for i_j, j in enumerate(reversed(j_list)):
        vec_vals = []
        for b in b_list:
            pos, neg, n = Loopy_belief_propagation(theta=theta,
                                                   num_iter=num_iter,
                                                   j=j, thr=thr, stim=b)
            val = np.max((pos[0], neg[0]))
            vec_vals.append(val)
        plt.plot(b_list, vec_vals, color=colormap[i_j],
                 label=np.round(j, 1), linewidth=4)
    plt.xlabel('Stimulus strength, B')
    plt.ylabel('Confidence')
    plt.legend(title='J')


def plot_posterior_vs_stim_cylinder(j_list=[0.05, 0.4, 0.64],
                                    b_list=np.linspace(0, 0.25, 51),
                                    theta=THETA, thr=1e-8, num_iter=200,
                                    data_folder=DATA_FOLDER):
    fig, ax = plt.subplots(ncols=1, figsize=(4.8, 3.7))
    colormap = ['navajowhite', 'orange', 'saddlebrown'][::-1]
    for i_j, j in enumerate(j_list):
        vec_vals_lbp = []
        for b in b_list:
            pos, neg, n = Loopy_belief_propagation(theta=theta,
                                                   num_iter=num_iter,
                                                   j=j, thr=thr, stim=b)
            val = np.max((pos[0], neg[0]))
            vec_vals_lbp.append(val)
        ax.plot(b_list, vec_vals_lbp, color=colormap[i_j],
                label=np.round(j, 1), linewidth=4)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Stimulus strength, B')
    ax.set_ylim(0.45, 1.02)
    ax.set_yticks([0.6, 0.8, 1])
    ax.set_xticks([0., 0.1, 0.2])
    ax.set_ylabel('Confidence')
    fig.tight_layout()
    ax.legend(loc=0, title='Coupling, J', labelspacing=0.1, frameon=False,
              bbox_to_anchor=(0.9, 1.2), ncol=3)
    fig.savefig(data_folder + 'post_cyl_lbp.png', bbox_inches='tight')
    fig.savefig(data_folder + 'post_cyl_lbp.svg', bbox_inches='tight')


def plot_posterior_vs_stim_all3(j_list=[0.05, 0.4, 0.64],
                                b_list=np.linspace(0, 0.25, 51),
                                theta=THETA, thr=1e-8, num_iter=200,
                                data_folder=DATA_FOLDER):
    fig, ax = plt.subplots(ncols = 3, figsize=(12, 4))
    colormap = ['navajowhite', 'orange', 'saddlebrown']
    j_list_mf = [0.05, 0.3, 0.35][::-1]
    for i_j, j in enumerate(reversed(j_list)):
        vec_vals_lbp = []
        vec_vals_mf = []
        vec_vals_gibbs = []
        for b in b_list:
            pos, neg, n = Loopy_belief_propagation(theta=theta,
                                                   num_iter=num_iter,
                                                   j=j, thr=thr, stim=b)
            val = np.max((pos[0], neg[0]))
            vec_vals_lbp.append(val)
            init_state = np.random.choice([-1, 1], theta.shape[0])
            val = gn.gibbs_samp_necker_post(init_state, burn_in=100, n_iter=int(1e5),
                                            j=j, stim=b, theta=theta)
            val = np.max((val, 1-val))
            vec_vals_gibbs.append(val)
            vec = mfn.mean_field_stim(j_list_mf[i_j], stim=b, num_iter=50, val_init=0.9,
                                      theta=theta, sigma=0)
            vec_vals_mf.append(np.nanmean(vec[-1]))
        ax[2].plot(b_list, vec_vals_lbp, color=colormap[i_j],
                   label=np.round(j, 1), linewidth=4)
        ax[1].plot(b_list, vec_vals_mf, color=colormap[i_j],
                   label=np.round(j, 1), linewidth=4)
        ax[0].plot(b_list, vec_vals_gibbs, color=colormap[i_j],
                   label=np.round(j, 1), linewidth=4)
    titles = ['Gibbs sampling', 'Mean-field', 'Loopy belief propagation']
    for i_a, a in enumerate(ax):
        a.set_xlabel('Stimulus strength, B')
        a.set_title(titles[i_a])
        a.set_ylim(0.45, 1.02)
    ax[0].set_ylabel('Confidence')
    ax[0].legend(title='J')
    fig.tight_layout()
    fig.savefig(data_folder + 'post_cyl.png')
    fig.savefig(data_folder + 'post_cyl.svg')


def plot_loopy_b_prop_sol_j_ast_circle(j_star_list, num_iter, j_list=np.arange(0, 1, 0.001),
                                       thr=1e-15, stim=0.):
    
    vals_bif = []
    fig, ax = plt.subplots(1)
    for j_star in j_star_list:
        theta = gn.theta_circle(j_star=j_star)
        vals_all = np.empty((theta.shape[0], len(j_list)))
        vals_all[:] = np.nan
        for i_j, j in enumerate(j_list):
            pos, neg, n = Loopy_belief_propagation(theta=theta,
                                                   num_iter=num_iter,
                                                   j=j, thr=thr,
                                                   stim=stim)
            vals_all[:, i_j] = [np.max((p, n)) for p, n in zip(pos, neg)]
        vals_mean = np.round(np.mean(vals_all, axis=0), 3)
        idx_05 = np.where(vals_mean == 0.5)[0][-1]
        vals_bif.append(j_list[idx_05])
    ax.plot(j_star_list, vals_bif, color='k')
    ax.set_xlabel('Proportion (pJ) of J, J* = J*pJ')
    ax.set_ylabel('Critical coupling')


def plot_loopy_b_prop_circle_kernel(num_iter, j_list=np.arange(0, 1, 0.001),
                                    thr=1e-15, stim=0.):
    
    fig, ax = plt.subplots(1)
    x = np.arange(8)
    kernel = cos_kernel(x)
    vals_all = []
    for i_j, j in enumerate(j_list):
        theta = scipy.linalg.circulant(kernel).T*j
        pos, neg, n = Loopy_belief_propagation(theta=theta,
                                               num_iter=num_iter,
                                               j=j, thr=thr,
                                               stim=stim)
        vals_all.append(np.max((pos[0], neg[0])))
    vals_all = np.array(vals_all)
    ax.plot(j_list, vals_all, color='k')
    ax.plot(j_list, 1-vals_all, color='k')
    ax.set_xlabel('Coupling, J')
    ax.set_ylabel('Approximate posterior')


def cos_kernel(x, plot=False):
    # kernel = (np.cos((x)*(len(x)-1)/np.pi)+1)/2
    kernel = np.concatenate((np.exp(-(x-1)[:3]), np.exp(-x[:3])[::-1]))
    kernel[0] = 0
    if plot:
        plt.figure()
        plt.plot(x+1, kernel)
        plt.xlabel('Node index')
        plt.ylabel(r'Kernel, $k(x) = (\cos((x-4)/(7*\pi))+1)/2$')
    return kernel


def f_j(xj, b):
    return np.exp(xj*b)


def f_ij(xi, xj, j):
    return np.exp(xi*xj*j)


def discrete_DBN(j, b, theta=THETA, num_iter=100, thr=1e-10,
                 alpha=1):
    # multiply arrays element wise
    mat_1 = np.random.rand(theta.shape[0], theta.shape[1])
    mat_neg1 = np.random.rand(theta.shape[0], theta.shape[1])
    mu_y_1 = np.multiply(theta, mat_1)
    mat_memory_1 = np.copy(mu_y_1)
    mu_y_neg1 = np.multiply(theta, mat_neg1)
    mat_memory_neg1 = np.copy(mu_y_neg1)
    for n in range(num_iter):
        mat_memory_1 = np.copy(mu_y_1)
        mat_memory_neg1 = np.copy(mu_y_neg1)
        # for all the nodes that i is connected
        for i in range(theta.shape[0]):
            for t in np.where(theta[i, :] != 0)[0]:
                # positive y_i
                mu_y_1[t, i] = (np.exp(j*alpha+b) *\
                        np.prod(mu_y_1[jneigbours(t, i, theta=theta), t]) *\
                            mu_y_1[i, t]**(1-alpha) \
                        + np.exp(-j*alpha-b) *\
                        np.prod(mu_y_neg1[jneigbours(t, i, theta=theta), t])*\
                            mu_y_neg1[i, t]**(1-alpha))**(1/alpha)
                # mu_y_1 += np.random.rand(8, 8)*1e-3
                # negative y_i
                mu_y_neg1[t, i] = (np.exp(-j*alpha+b) *\
                    np.prod(mu_y_1[jneigbours(t, i, theta=theta), t])*\
                    mu_y_1[i, t]**(1-alpha)
                    + np.exp(j*alpha-b) *\
                    np.prod(mu_y_neg1[jneigbours(t, i, theta=theta), t])*\
                    mu_y_neg1[i, t]**(1-alpha))**(1/alpha)

                m_y_1_memory = np.copy(mu_y_1[t, i])
                mu_y_1[t, i] = mu_y_1[t, i]/(m_y_1_memory+mu_y_neg1[t, i])
                mu_y_neg1[t, i] = mu_y_neg1[t, i]/(m_y_1_memory+mu_y_neg1[t, i])
                # mu_y_neg1 += np.random.rand(8, 8)*1e-3
        if np.sqrt(np.sum(mat_memory_1 - mu_y_1)**2) and\
            np.sqrt(np.sum(mat_memory_neg1 - mu_y_neg1)**2) <= thr:
            break
    q_y_1 = np.zeros(theta.shape[0])
    q_y_neg1 = np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        q1 = np.prod(mu_y_1[np.where(theta[:, i] != 0), i]) * np.exp(b)
        qn1 = np.prod(mu_y_neg1[np.where(theta[:, i] != 0), i]) * np.exp(-b)
        q_y_1[i] = q1/(q1+qn1)
        q_y_neg1[i] = qn1/(q1+qn1)
    # gn.plot_cylinder(q=q_y_1.reshape(5, 10, 2),
    #                  columns=5, rows=10, layers=2, offset=0.4, minmax_norm=True)
    return q_y_1, q_y_neg1


def General_belief_propagation(j, b, theta=THETA, num_iter=100, thr=1e-10,
                               counting_numbers=[1, 1, 1, 1]):
    alpha, beta, kappa, gamma = counting_numbers
    # multiply arrays element wise
    mu_y_1 = np.multiply(theta, np.random.rand(theta.shape[0], theta.shape[1]))
    mat_memory_1 = np.copy(mu_y_1)
    mu_y_neg1 = np.multiply(theta, np.random.rand(theta.shape[0], theta.shape[1]))
    mat_memory_neg1 = np.copy(mu_y_neg1)
    for n in range(num_iter):
        mat_memory_1 = np.copy(mu_y_1)
        mat_memory_neg1 = np.copy(mu_y_neg1)
        # for all the nodes that i is connected
        for i in range(theta.shape[0]):
            for t in np.where(theta[i, :] != 0)[0]:
                # positive y_i
                mu_y_1[t, i] = (np.exp(j*alpha*beta+b*gamma*kappa) *\
                       (np.prod(mu_y_1[jneigbours(t, i, theta=theta), t]) *\
                            mu_y_1[i, t]**(1-alpha/kappa))**kappa \
                        + np.exp(-j*alpha*beta-b*gamma*kappa) *\
                        (np.prod(mu_y_neg1[jneigbours(t, i, theta=theta), t])*\
                            mu_y_neg1[i, t]**(1-alpha/kappa))**kappa)**(1/alpha)
                # mu_y_1 += np.random.rand(8, 8)*1e-3
                # negative y_i
                mu_y_neg1[t, i] = (np.exp(-j*alpha*beta+b*gamma*kappa) *\
                       (np.prod(mu_y_1[jneigbours(t, i, theta=theta), t]) *\
                            mu_y_1[i, t]**(1-alpha/kappa))**kappa \
                        + np.exp(j*alpha*beta-b*gamma*kappa) *\
                        (np.prod(mu_y_neg1[jneigbours(t, i, theta=theta), t])*\
                            mu_y_neg1[i, t]**(1-alpha/kappa))**kappa)**(1/alpha)

                m_y_1_memory = np.copy(mu_y_1[t, i])
                mu_y_1[t, i] = mu_y_1[t, i]/(m_y_1_memory+mu_y_neg1[t, i])
                mu_y_neg1[t, i] = mu_y_neg1[t, i]/(m_y_1_memory+mu_y_neg1[t, i])
                # mu_y_neg1 += np.random.rand(8, 8)*1e-3
        if np.sqrt(np.sum(mat_memory_1 - mu_y_1)**2) and\
            np.sqrt(np.sum(mat_memory_neg1 - mu_y_neg1)**2) <= thr:
            break
    q_y_1 = np.zeros(theta.shape[0])
    q_y_neg1 = np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        q1 = (np.prod(mu_y_1[np.where(theta[:, i] != 0), i]) * np.exp(b*gamma))**kappa
        qn1 = (np.prod(mu_y_neg1[np.where(theta[:, i] != 0), i]) * np.exp(-b*gamma))**kappa
        q_y_1[i] = q1/(q1+qn1)
        q_y_neg1[i] = qn1/(q1+qn1)
    # gn.plot_cylinder(q=q_y_1.reshape(5, 10, 2),
    #                  columns=5, rows=10, layers=2, offset=0.4, minmax_norm=True)
    return q_y_1, q_y_neg1    


def potential_2d_changing_j(b, alpha=1, n=3, noise=0.2):
    max_val = 1.25
    min_val = -1.25
    min_val_integ = -2
    epsilon = 5e-3
    vals = np.arange(min_val, max_val, epsilon)
    j_list = np.arange(0, 1, 5e-3)
    boltzmann_distro = np.empty((len(vals), len(j_list)))
    boltzmann_distro[:] = np.nan
    for i_j, j in enumerate(j_list):
        pot_0 = lambda x: 1/alpha * np.arctanh(np.tanh(alpha*j)*np.tanh(x*(n-alpha)+b)) - x
        potential = []
        for i in range(len(vals)):
            potential.append(-scipy.integrate.quad(pot_0, min_val_integ, vals[i])[0])
        pot = np.array(potential)
        distro = np.exp(-2*pot/noise**2)
        distro = distro / np.nansum(distro)
        boltzmann_distro[:, i_j] = distro
    fig, ax = plt.subplots(1)
    im = ax.imshow(np.flipud(boltzmann_distro), cmap='Oranges',
                   extent=[0, 1, -1.25, 1.25], aspect='auto')
    plt.colorbar(im, ax=ax, label='Boltzmann distribution')
    ax.set_xlabel('Coupling, J')
    ax.set_ylabel('Log-message ratio')


def log_potential(b, j_list=None, alpha=1, n=3,
                  ax=None, labels=True, norm=True, transform=True):
    max_val = 1.25
    min_val = -1.25
    min_val_integ = -10
    epsilon = 1e-3
    vals = np.arange(min_val+epsilon, max_val, epsilon)
    if j_list is None:
        j_crit = np.log(n / (n-2*alpha)) / (2*alpha)
        j_list = [round(j_crit-0.2, 2), round(j_crit-0.1, 2), j_crit, round(j_crit+0.1, 2), round(j_crit+0.2, 1)]
    if ax is None:
        fig, ax = plt.subplots(1)
        colormap = pl.cm.Purples(np.linspace(0.2, 1, len(j_list)))
    else:
        colormap = ['k']
    for i_j, j in enumerate(j_list):
        pot = lambda x: 1/alpha * np.arctanh(np.tanh(alpha*j)*np.tanh(x*(n-alpha)+b)) - x
        potential = []
        for i in range(len(vals)):
            potential.append(-scipy.integrate.quad(pot, min_val, vals[i])[0])
        potential = np.array(potential)
        if norm:
            norm_pot = (potential - np.min(potential))/ (np.max(potential) - np.min(potential))
        else:
            norm_pot = potential
        color = colormap[i_j] if i_j != 2 else 'red'
        label = str(j)  if i_j != 2 else r'$J^* = \frac{1}{2\alpha}\log{\frac{N}{N-2\alpha}}$'
        if transform:
            ax.plot(sigmoid(vals*n+b), norm_pot,
                    label=label, color=color)
        else:
            ax.plot(vals, norm_pot,
                    label=label, color=color)
    if labels:
        plt.legend(title='Coupling, J')
        if not transform:
            plt.xlabel(r'Log-message ratio, $M_{i\rightarrow j} = \frac{1}{2} \log \left[ \frac{m_{i\rightarrow j (+1)}}{m_{i\rightarrow j (+1)}}\right]$',
                       fontsize=12)
        if transform:
            plt.xlabel(r'Approximate posterior, $q_i(x=1)$',
                       fontsize=12)
        plt.ylabel(r'Potential on log-message ratio, $V(M_{i\rightarrow j})$')
        if norm:
            plt.ylim(-0.015, 0.15)


def dyn_sys_fbp(logmess, j, b, alpha=1, n=3, dt=1e-1, noise=0.1):
    logmess = logmess + dt*(1/alpha * np.arctanh(np.tanh(alpha*j)*np.tanh(logmess*(n-alpha)+b)) - logmess) +\
        np.sqrt(dt)*noise*np.random.randn()
    return logmess


def plot_derivative_dyn_sys():
    plt.figure()
    jl = np.arange(0, 1, 1e-3)
    plt.axhline(0, color='k')
    for al in [0.1, 0.5, 1, 1.4]:
        plt.plot(jl, ((3-al)*np.tanh(jl*al)/al-1),
        color='r', alpha=al/1.8+0.1, label=al)
        plt.axvline(np.log(3/(3-al*2))/(2*al),
                    color='r', alpha=al/1.8+0.1)
    plt.xlabel('coupling J')
    plt.legend()
    plt.ylabel("g'(x=0)")


def plot_fbp_dynamics(j, b, alpha=1):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(6, 4))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.075, right=0.98,
                        hspace=0.3, wspace=0.5)
    ax = ax.flatten()
    logmessages = []
    logmess = np.random.randn()/10
    potential_vals = []
    for i in range(200):
        logmess = dyn_sys_fbp(logmess, j, b, alpha=1, n=3, dt=1e-1, noise=.3)
        logmessages.append(logmess)
    ax[0].plot(logmessages)


def plot_pot_evolution_FBP(j=0.75, b=0, alpha=1, n=3, num_iter=10):
    fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(16, 10))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.075, right=0.98,
                        hspace=0.3, wspace=0.5)
    ax = ax.flatten()
    logmessages = []
    logmess = np.random.randn()/10
    pot = lambda x: 1/alpha * np.arctanh(np.tanh(alpha*j)*np.tanh(x*(n-alpha)+b)) - x
    potential_vals = []
    for i in range(num_iter):
        logmess = dyn_sys_fbp(logmess, j, b, alpha=alpha, n=3, dt=1, noise=0.2)
        logmessages.append(gn.sigmoid(logmess*n+b))
        val_pot = -scipy.integrate.quad(pot, -1.25, logmess)[0]
        potential_vals.append(val_pot)
    for i_a, a in enumerate(ax):
        log_potential(b, j_list=[j], alpha=1, n=3, ax=a, labels=False,
                      norm=False)
        a.plot(logmessages[i_a], potential_vals[i_a], marker='o', color='r')


def dyn_sys_ql(ql, ra, rb, j, b, dt=1e-2, n=3, alpha=1, noise=0.1,
               ipta=0, iptb=0, tau=0.08):
    qlt = ql + dt*(n/alpha * np.arctanh(np.tanh(j*alpha)*np.tanh((ql*(n-alpha) + b*alpha)/n))-ql + np.random.randn()*noise)/tau
    rat = ra + dt*(10*sigmoid(qlt + ipta+ np.random.randn()*noise) -ra)/tau
    rbt = rb + dt*(10*sigmoid(-qlt + iptb+ np.random.randn()*noise) - rb)/tau
    return qlt, rat, rbt


def plot_log_ratio(j, b, n=3, alpha=1):
    pot = lambda x: 1/alpha * np.arctanh(np.tanh(alpha*j)*np.tanh(x*(n-alpha)+b)) - x
    pot_taylor = lambda x: 1/alpha * (np.arctanh(np.tanh(j*alpha)*np.tanh(b))+
                                      np.tanh(j*alpha)*(1-np.tanh(b)**2)*x*(n-alpha)
                                      - np.tanh(j*alpha)/3 * (1-np.tanh(b)**2)**2 * (x**3)*(n-alpha)**3)-x
    x = np.arange(-1.5, 1.5, 1e-3)
    plt.figure()
    plt.plot(x, pot(x), color='k', label='True')
    plt.plot(x, pot_taylor(x), color='r', label='Approx.')
    plt.legend()


def plot_rates_neurons(j, b, alpha=1, n=3, dt=1e-3, noise=5,
                       t_end=1):
    ql = np.random.randn()
    ra = np.abs(np.random.randn()*5)
    rb = np.abs(np.random.randn()*5)
    qlt, rat, rbt = [], [], []
    time = np.arange(0, t_end, dt)
    for i in range(len(time)):
        ql, ra, rb = dyn_sys_ql(ql, ra, rb, j, b, dt=dt, n=n, tau=0.05,
                                alpha=alpha, noise=noise, ipta=0, iptb=0)
        qlt.append(ql)
        rat.append(ra)
        rbt.append(rb)
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))
    ax = ax.flatten()
    ax[0].plot(time, qlt)
    ax[0].set_ylabel('Log-ratio of beliefs')
    ax[0].set_xlabel('Time (s)')
    ax[1].plot(time, rat, label=r'$r_A$', color='r')
    ax[1].plot(time, rbt, label=r'$r_B$', color='b')
    ax[1].legend()
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Firing rate (Hz)')
    gamma = np.array(rat)-np.array(rbt)
    ax[2].plot(time, gamma, label=r'$\Delta r$',
               color='k')
    ax[2].axhline(0, color='g', linestyle='--')
    ax[2].set_xlabel('Time (s)')
    ax[2].legend()
    ax[3].plot(rbt, rat, color='k', alpha=0.5)
    ax[3].plot([3, 8], [3, 8], color='gray', linestyle='--')
    ax[3].set_xlabel(r'Firing rate (Hz) $r_B$')
    ax[3].set_ylabel(r'Firing rate (Hz) $r_A$')
    fig.tight_layout()
    plt.figure()
    plt.plot([-2, 2], [-7, 7], color='r')
    plt.plot(qlt, gamma, color='k', marker='o', linestyle='',
             markersize=1, alpha=0.5)
    plt.xlabel('Log-ratio of beliefs')
    plt.ylabel(r'$\Delta r$')


def plot_L_different_init(alpha=1, n=3, dt=1e-3, t_end=2, noise=0):
    ql = np.random.randn()
    ra = np.abs(np.random.randn()*5)
    rb = np.abs(np.random.randn()*5)
    time = np.arange(0, t_end, dt)
    qlt = np.empty((len(time), 4, 6))
    count = 0
    for j in [0.3, 0.7]:
        for b in [0, 0.2]:
            for i_q, ql in enumerate([-2, -1, -0.5, 0.5, 1, 2]):
                for i in range(len(time)):
                    ql, _, _ = dyn_sys_ql(ql, ra, rb, j, b, dt=dt, n=n, tau=0.05,
                                          alpha=alpha, noise=noise, ipta=0, iptb=0)
                    qlt[i, count, i_q] = ql
            count += 1
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(5, 4))
    ax = ax.flatten()
    for i_a, a in enumerate(ax):
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.axhline(0.5, color='r', alpha=0.6, linestyle='--')
        [a.plot(time, gn.sigmoid(2*qlt[:, i_a, t]), color='k') for t in range(6)]
        if i_a > 1:
            a.set_xlabel('Time (s)')
        else:    
            a.set_xticks([])
        if (i_a-1) % 2 == 0:
            a.set_yticks([])
    ax[2].set_ylabel('                          Approx. posterior q(x=1)')
    ax[0].set_title('B = 0')
    ax[1].set_title('B > 0')
    axtwin2 = ax[1].twinx()
    axtwin2.set_ylabel('J < J*')
    axtwin2.spines['right'].set_visible(False)
    axtwin2.spines['top'].set_visible(False)
    axtwin2.set_yticks([])
    axtwin2 = ax[3].twinx()
    axtwin2.set_ylabel('J > J*')
    axtwin2.spines['right'].set_visible(False)
    axtwin2.spines['top'].set_visible(False)
    axtwin2.set_yticks([])
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'examples_nonoise.png', dpi=400)
    fig.savefig(DATA_FOLDER + 'examples_nonoise.svg', dpi=400)
    

def sigmoid(x):
    return 1/(1+np.exp(-x))


def pot_expr(q, j, b, n=3, a=1):
    return -q*np.arctanh(np.tanh(a*j)*np.tanh(q*(n-1))+b)/a - np.sinh(a*j)/(a*4*(n-1)) + 0.5*q**2


def pot_potential_taylor(q, j, b, n=3, a=1):
    a1 = -np.arctanh(np.tanh(b)*np.tanh(a*j))*q 
    a2 = - 0.5*q**2 * (n-a) * np.sinh(2*a*j) / (np.cosh(2*b)+np.cosh(2*a*j)) + 0.5*q**2
    a3 = 1/3 * q**3 * (np.sinh(2*b)*(n-a)**2 * np.sinh(2*a*j)) / (np.cosh(2*b)+np.cosh(2*a*j))**2
    a4 = - 1/4 * q**4 * (np.sinh(2*j*a)*(a-n)**3 * (np.cosh(2*(b-a*j)) + np.cosh(2*(b+a*j)) - np.cosh(4*b)+3) ) / (3*(np.cosh(2*b)+np.cosh(2*a*j))**3)
    a5 = + (1/5*q**5 * np.sinh(2 * b) * (a - n)**4 * np.sinh(2 * a*j) * (-2 * (2 * np.cosh(2 * (b - a*j)) + 2 * np.cosh(2 * (b + a*j)) + 5) + np.cosh(4 * b) + np.cosh(4 * a*j))) / (6 * (np.cosh(2 * b) + np.cosh(2 * a*j))**4)
    return a1 + a2 + a3 + a4 + a5


def crit_j_frac(n_list=np.arange(3, 10, 1e-3), alpha_list=np.arange(0.1, 1.5, 0.15)):
    fig, ax = plt.subplots(1)
    colormap = pl.cm.Blues(np.linspace(0.2, 1, len(alpha_list)))[::-1]
    ax.plot(n_list, 0.5*np.log(n_list/(n_list-2)), color='k')
    for i_a, a in enumerate(reversed(alpha_list)):
        if round(a, 2) != 1:
            color = colormap[i_a]
            if i_a == 0 or round(a, 2) == 0.55 or round(a, 2) == 0.1:
                label = round(a, 2)
            else:
                label = ' '
            lw = 1
        else:
            color = 'orange'
            label = r'BP, $\alpha = 1$'
            lw = 3.5
        ax.plot(n_list, 0.5*np.log(n_list/(n_list-2*a))/a, color=color, label=label,
                linewidth=lw)
    ax.plot(n_list, 1/n_list, color='r', linestyle='--', label=r'MF, $\alpha \to 0$', linewidth=3)
    ax.legend(title=r'$\alpha$')
    ax.set_xlabel('# neighbors, N')
    ax.set_ylabel(r'Critical coupling, $J^{\ast}$')


def performance_vs_alpha(j=0.5, alpha_list=np.arange(0.1, 1.45, 0.3),
                         b_list=np.round(np.arange(-.25, .2525, 0.05), 5),
                         num_reps=500):
    fig, ax = plt.subplots(1, figsize=(4	, 3.2))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    mat_acc = np.empty((len(alpha_list), len(b_list)))
    matrix_true_file = DATA_FOLDER + 'accuracy_alpha_b.npy'
    os.makedirs(os.path.dirname(matrix_true_file), exist_ok=True)
    if os.path.exists(matrix_true_file):
        mat_acc = np.load(matrix_true_file, allow_pickle=True)
    else:
        for ia, alpha in enumerate(alpha_list):
            accuracy = np.empty((len(b_list), num_reps))
            accuracy[:] = np.nan
            alpha = np.round(alpha, 4)
            for i_b, stim in enumerate(b_list):
                acclist = []
                for n in range(num_reps):
                    pos, _ = discrete_DBN(j, b=stim, theta=THETA, num_iter=50,
                                          thr=1e-8, alpha=alpha)
                    choice = np.sign(pos[0]-0.5)
                    if stim == 0:
                        stim2 = np.random.randn()*1e-4
                    else:
                        stim2 = stim
                    acclist.append(choice == np.sign(stim2))
                accuracy[i_b, :] = acclist
            # err_alpha.append(np.nanstd(accuracy, axis=1))
            vals = np.nanmean(accuracy, axis=1)
            mat_acc[ia, :] = vals
        np.save(matrix_true_file, mat_acc)
    accuracy = np.empty((len(b_list), num_reps))
    accuracy[:] = np.nan
    for i_b, stim in enumerate(b_list):
        acclist = []
        for n in range(num_reps):
            pos = mfn.mean_field_stim(j, num_iter=20, stim=stim, sigma=0)
            pos = pos[-1]
            choice = np.sign(pos[0]-0.5)
            if stim == 0:
                stim2 = np.random.randn()*1e-4
            else:
                stim2 = stim
            acclist.append(choice == np.sign(stim2))
        accuracy[i_b, :] = acclist
    colormap = pl.cm.Blues(np.linspace(0.2, 1, len(alpha_list)+1))
    vals_mf = np.nanmean(accuracy, axis=1)
    ax.plot(b_list, vals_mf, color=colormap[0], label='0, MF', linewidth=2.5)
    for ia, alpha in enumerate(alpha_list):
        vals = mat_acc[ia, :]
        if alpha == 1:
            color = colormap[ia+1]
            label = '1, LBP'
        else:
            color = colormap[ia+1]
            label = alpha
        ax.plot(b_list, vals, color=color,
                label=label, linewidth=2.5)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Sensory evidence, B')
    fig.tight_layout()
    ax.legend(loc=0, title=r'$\alpha$', frameon=False,
              bbox_to_anchor=(0.95, 1.2), labelspacing=0.15)
    fig.savefig(DATA_FOLDER + 'performance_vs_alpha_j08_v2.png',
                dpi=400, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'performance_vs_alpha_j08_v2.svg',
                dpi=400, bbox_inches='tight')


def f_q_vs_alpha(j=0.5, alpha_list=np.arange(0.1, 2, 0.1), b=0,
                 n=3):
    fig, ax = plt.subplots(1)
    colormap = pl.cm.Blues(np.linspace(0.1, 1, len(alpha_list)))
    q = np.arange(0, 1, 1e-4)
    q_log = 0.5*np.log(q/(1-q))
    m_log = q_log/3
    vals_mf = np.tanh(n*m_log+b)*j-m_log
    ax.plot(m_log, vals_mf, color='red', label=r'MF, $\alpha \to 0$',
            linewidth=3)
    for ia, alpha in enumerate(alpha_list):
        fun = lambda x: 1/alpha * np.arctanh(np.tanh(alpha*j)*np.tanh(x*(n-alpha)+b))-x
        alpha = np.round(alpha, 4)
        vals = fun(m_log)
        if round(alpha, 2) != 1 and round(alpha, 2) != 1.5:
            color = colormap[ia]
            label = alpha
            lw = 1
        if round(alpha, 2) == 1.5:
            color = 'green'
            label = r'No bistab., $\alpha = N/2$'
            lw = 3.5
        if round(alpha, 2) == 1:
            color = 'orange'
            label = r'BP, $\alpha = 1$'
            lw = 3.5
        ax.plot(m_log, vals, color=color, label=label, linewidth=lw)
    ax.legend()
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_ylabel('f(M)-M')
    ax.set_xlabel('M')
    

def second_derivative_potential(x, j, b, alpha, n=3):
    dertanh = 1/(1-(np.tanh(alpha*j)*np.tanh(x*(n-alpha)+b))**2)
    return -np.tanh(alpha*j)*(n-alpha)*dertanh/alpha/(np.cosh(x*(n-alpha)+b))**2+1


def k_i_to_j(j, xi, xj, noise, b=0, alpha=1, n=3):
    v_2_xi = second_derivative_potential(xi, j, b=b, alpha=alpha)
    v_2_xj = second_derivative_potential(xj, j, b=b, alpha=alpha)
    negpotder = lambda x: 1/alpha * np.arctanh(np.tanh(alpha*j)*np.tanh(x*(n-alpha)+b)) - x
    v_xi = -scipy.integrate.quad(negpotder, 0, xi)[0]
    v_xj = -scipy.integrate.quad(negpotder, 0, xj)[0]
    # k_IJ = sqrt(|V''(xi)*V''(xj|)*exp{2*(V(x_i)-V(x_j))/sigma^2}/(2*pi)
    return np.sqrt(np.abs(v_2_xi*v_2_xj))*np.exp(2*(v_xi - v_xj)/noise**2) / (2*np.pi)


def get_unst_and_stab_fp(j, b, alpha=1, n=3, tol=1e-4):
    diff_1 = 0
    diff_2 = 0
    init_cond_unst = 0.
    q1 = lambda x: 1/alpha * np.arctanh(np.tanh(alpha*j)*np.tanh(x*(n-alpha)+b)) - x
    sol_1, _, flag, _ =\
        fsolve(q1, 5., full_output=True)
    x_stable_1 = sol_1[0]
    sol_2, _, flag, _ =\
        fsolve(q1, -5., full_output=True)
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
        init_cond_unst = np.random.rand()*np.sign(-b)
    return x_stable_1, x_stable_2, x_unstable


def psychometric_fbp_analytical(t_dur, noiselist=[0.05, 0.1, 0.2, 0.3],
                                j_list=np.arange(0., 1.1, 0.2),
                                b_list=np.arange(0, 0.2, 1e-2),
                                alphalist=[0.1, 0.5, 1, 1.4], n=3,
                                tol=1e-8, varchange='alpha'):
    if varchange == 'alpha':
        noise = 0.3
        varlist = alphalist
        title = r'Alpha, $\alpha$ = '
    else:
        alpha = 1
        varlist = noiselist
        title = r'Noise, $\sigma$ = '
    init_cond = 0.
    accuracy = np.zeros((len(j_list), len(b_list), len(varlist)))
    for i_n, var in enumerate(varlist):
        if varchange == 'alpha':
            alpha = var
        else:
            noise = var
        for ib, b in enumerate(b_list):  # for each b, compute P(C) = P_{C,0}*P_{C,C} + (1-P_{C,0})*P_{C,E}
        # prob of correct is prob of correct at beginning times prob of staying at correct atractor
        # plus prob of incorrect at beginning times prob of going from incorrect to correct
            for ij, j in enumerate(j_list):
                # x_stable_1 = "correct" attractor (with sign of B)
                # x_stable_2 = "incorrect" attractor (with sign of B)
                x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(j=j, b=b, alpha=alpha, n=n,
                                                                          tol=tol)
                if b < 0:  # change which attractor is the correct one depending on b
                    x_stable_2, x_stable_1 = x_stable_1, x_stable_2
                if np.abs(x_stable_1 - x_stable_2) <= tol:
                    accuracy[ij, ib, i_n] = 1
                    continue
                # q1(x) = grad(V(x))
                q1 = lambda x: -1/alpha * np.arctanh(np.tanh(alpha*j)*np.tanh(x*(n-alpha)+b)) + x
                # potential = V(x)-V(0) = int_[0, x] q1(x*) dx*
                potential = lambda x: scipy.integrate.quad(q1, 0., x)[0]
                # P_{C, 0} = int_{x_E, x_0} exp(2V(x)/sigma^2) dx /
                #            int_{x_E, x_C} exp(2V(x)/sigma^2) dx
                pc0_numerator = scipy.integrate.quad(lambda x: np.exp(2*potential(x)/noise**2),
                                                     x_stable_2, init_cond)[0]
                pc0_denom = scipy.integrate.quad(lambda x: np.exp(2*potential(x)/noise**2),
                                                 x_stable_2, x_stable_1)[0]
                pc0 = pc0_numerator/pc0_denom
                # compute error transition rates k_CE, K_EC
                k_EC = k_i_to_j(j, x_stable_1, x_unstable, noise, b, alpha, n=n)
                k_CE = k_i_to_j(j, x_stable_2, x_unstable, noise, b, alpha, n=n)
                k = k_EC+k_CE
                pCS = k_CE/k  # stationary correct
                # error to correct transition
                pCE = pCS*(1-np.exp(-k*t_dur))
                # correct to correct
                pCC = pCS*(1-np.exp(-k*t_dur))+np.exp(-k*t_dur)
                # probability of correct
                # P(C) = P_{C,0}*P_{C,C} + (1-P_{C,0})*P_{C,E}
                pC = pc0*pCC + (1-pc0)*pCE
                accuracy[ij, ib, i_n] = pC  # np.max((pC, 1-pC))
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
    axes = axes.flatten()
    colormap = pl.cm.Oranges(np.linspace(0.2, 1, len(j_list)))
    for iax, ax in enumerate(axes):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for ij in range(len(j_list)):
            ax.plot(b_list, accuracy[ij, :, iax], color=colormap[ij],
                    label=round(j_list[ij], 2))
        ax.set_xlabel('Sensory evidence, B')
        ax.set_ylabel('Accuracy')
        ax.set_title(title + str(varlist[iax]))
        ax.set_ylim(0.45, 1.05)
    axes[0].legend(title='Coupling, J')
    fig.tight_layout()


if __name__ == '__main__':
    # for stim in [0]:
    #     plot_loopy_b_prop_sol(theta=THETA, num_iter=200,
    #                           j_list=np.arange(0.00001, 1, 0.01),
    #                           thr=1e-10, stim=stim)
    # plot_loopy_b_prop_sol_j_ast_circle(j_star_list=np.arange(0, 1.1, 0.05),
    #                                    num_iter=400,
    #                                    j_list=np.arange(0, 1, 0.005),
    #                                    thr=1e-10, stim=0.)
    # plot_posterior_vs_stim_all3(j_list=[0.05, 0.4, 0.64],
    #                             b_list=np.linspace(0, 0.25, 21),
    #                             theta=gn.return_theta(),
    #                             thr=1e-8, num_iter=200)
    # plot_posterior_vs_stim_cylinder(j_list=[0.05, 0.4, 0.6],
    #                                 b_list=np.linspace(0, 0.25, 21),
    #                                 theta=gn.return_theta(),
    #                                 thr=1e-8, num_iter=300)
    # plot_sols_FLBP(alphalist=[0.1, 0.3, 0.6, 1, 1.2, 1.4],
    #                 j_list=np.arange(0, 2, 0.01), theta=THETA,
    #                 num_iter=200, stim=0.)
    # plot_sols_FLBP(alphalist=[1.8],
    #                j_list=np.arange(0, 20, .1), theta=THETA,
    #                num_iter=1000, stim=0.)
    # plot_over_conf_mf_bp_gibbs(data_folder=DATA_FOLDER, j_list=np.arange(0., 1.005, 0.005),
    #                             b_list_orig=np.arange(-.5, .5005, 0.005), theta=THETA)
    # all_comparison_together(j_list=np.arange(0., 1.005, 0.005),
    #                         b_list=np.arange(-.5, .5005, 0.005),
    #                         data_folder=DATA_FOLDER,
    #                         theta=THETA, dist_metric=None, nrows=2)
    # performance_vs_alpha(j=0.8, alpha_list=[0.1, 0.4, 0.7, 1. , 1.1],
    #                       b_list=np.round(np.arange(0, .08, 0.005), 5),
    #                       num_reps=4000)
    # plot_L_different_init(alpha=1, n=3, dt=1e-3, t_end=2, noise=0)
    # log_potential(b=0, j_list=None, alpha=1, n=3,
    #               ax=None, labels=True, norm=False, transform=False)
    psychometric_fbp_analytical(t_dur=1, noiselist=[0.1, 0.2, 0.3, 0.4],
                                j_list=np.arange(0.6, 2.3, 0.2),
                                b_list=np.arange(0, 0.2, 1e-2),
                                alphalist=[0.01, 0.1, 0.5, 1], n=5,
                                tol=1e-5, varchange='noise')
    # plot_sol_LBP(j_list=np.arange(0.00001, 1, 0.0001), stim=0.)
    # plot_potentials_lbp(j_list=np.arange(0., 1.1, 0.1), b=-0., neighs=3, q1=False)
    # plot_potential_lbp(q=np.arange(0.0001, 4, 0.01),
    #                    j_list=[0.1, 0.4, 0.5, np.log(3)/2, 0.57, 0.585, 0.6, .65],
    #                    b=0, n=3)
    # posterior_comparison_MF_BP(stim_list=np.linspace(-2, 2, 1001), j=0.28,
    #                             num_iter=40, thr=1e-8, theta=THETA)
    # plot_j_b_crit_BP_vs_N(j_list=np.arange(0.001, 1.01, 0.005),
    #                       b_list=np.arange(-0.5, 0.5, 0.01),
    #                       tol=1e-8, min_r=0, max_r=20,
    #                       w_size=0.05, neigh_list=np.arange(3, 12),
    #                       dim3=False)
    # plt.figure()
    # solve_equation_g_derivative()
    # j_list=np.arange(0.001, 1, 0.001)
    # fig, ax = plt.subplots(ncols=1)
    # plot_bp_solution(ax, j_list, b=0.05, tol=1e-10,
    #                  min_r=-15, max_r=15,
    #                  w_size=0.01, n_neigh=3,
    #                  color='r')
    # plot_lbp_3_examples()
    # plot_lbp_explanation()
    # plot_m1_m2_vector_field(j=.65, b=0., n=3)
