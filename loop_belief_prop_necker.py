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
from scipy.optimize import fsolve
import numpy as np

THETA = gn.THETA


def jneigbours(j,i):
    """
    return the neighbours of j except i

    Input:
    - i {integer}: index of our current node
    - j {integer}: index of j
    Output:
    - return a list with all the neighbours of neighbour of j except our 
      current node
    """
    neigbours = np.array(np.where(THETA[j,:] != 0))
    return neigbours[neigbours!=i]


def Loopy_belief_propagation(theta, num_iter, j, thr=1e-5, stim=0):
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
    mu_y_1 = np.multiply(theta, np.random.rand(8, 8))
    mat_memory_1 = np.copy(mu_y_1)
    mu_y_neg1 = np.multiply(theta, np.random.rand(8, 8))
    mat_memory_neg1 = np.copy(mu_y_neg1)
    theta = theta*j
    for n in range(num_iter):
        mat_memory_1 = np.copy(mu_y_1)
        mat_memory_neg1 = np.copy(mu_y_neg1)
        # for all the nodes that i is connected
        for i in range(8):
            for t in np.where(theta[i, :] != 0)[0]:
                # positive y_i
                mu_y_1[t, i] = np.exp(theta[i, t]+stim) *\
                        (mu_y_1[jneigbours(t, i)[0], t]*mu_y_1[jneigbours(t, i)[1], t])\
                        + np.exp(-theta[i, t]-stim) *\
                        (mu_y_neg1[jneigbours(t, i)[0], t] *
                         mu_y_neg1[jneigbours(t, i)[1], t])
                # mu_y_1 += np.random.rand(8, 8)*1e-3
                # negative y_i
                mu_y_neg1[t, i] = np.exp(-theta[i, t]-stim) *\
                    (mu_y_1[jneigbours(t, i)[0], t] * mu_y_1[jneigbours(t, i)[1], t])\
                    + np.exp(theta[i, t]+stim) *\
                    (mu_y_neg1[jneigbours(t, i)[0], t] *
                     mu_y_neg1[jneigbours(t, i)[1], t])

                m_y_1_memory = np.copy(mu_y_1[t, i])
                mu_y_1[t, i] = mu_y_1[t, i]/(m_y_1_memory+mu_y_neg1[t, i])
                mu_y_neg1[t, i] = mu_y_neg1[t, i]/(m_y_1_memory+mu_y_neg1[t, i])
                # mu_y_neg1 += np.random.rand(8, 8)*1e-3
        if np.sqrt(np.sum(mat_memory_1 - mu_y_1)**2) and\
            np.sqrt(np.sum(mat_memory_neg1 - mu_y_neg1)**2) <= thr:
            break

    q_y_1 = np.zeros(8)
    q_y_neg1 = np.zeros(8)
    for i in range(8):
        q1 = np.prod(mu_y_1[np.where(theta[:, i] != 0), i])
        qn1 = np.prod(mu_y_neg1[np.where(theta[:, i] != 0), i])
        q_y_1[i] = q1/(q1+qn1)
        q_y_neg1[i] = qn1/(q1+qn1)
    return q_y_1, q_y_neg1, n+1


def nonsym_Loopy_belief_propagation(theta, num_iter, j, thr=1e-5):
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
    mu_y_1 = np.multiply(theta, np.random.rand(8, 8))
    mat_memory = np.copy(mu_y_1)
    mu_y_neg1 = np.multiply(theta, np.random.rand(8, 8))
    theta = theta*j
    for n in range(num_iter):
        mat_memory = np.copy(mu_y_1)
        # for all the nodes that i is connected
        for i in range(8):
            for t in np.where(theta[i, :] != 0)[0]:
                # positive y_i
                mu_y_1[t, i] = np.exp(theta[i, t]) *\
                        (mu_y_1[jneigbours(t, i)[0], t]*mu_y_1[jneigbours(t, i)[1], t])\
                        + np.exp(-theta[i, t]) *\
                        (mu_y_neg1[jneigbours(t, i)[0], t] *
                         mu_y_neg1[jneigbours(t, i)[1], t])

                # negative y_i
                mu_y_neg1[t, i] = np.exp(-theta[i, t]) *\
                    (mu_y_1[jneigbours(t, i)[0], t] * mu_y_1[jneigbours(t, i)[1], t])\
                    + np.exp(theta[i, t]) *\
                    (mu_y_neg1[jneigbours(t, i)[0], t] *
                     mu_y_neg1[jneigbours(t, i)[1], t])

                mu_y_1[t, i] = mu_y_1[t, i]/(mu_y_1[t, i]+mu_y_neg1[t, i])
                mu_y_neg1[t, i] = mu_y_neg1[t, i]/(mu_y_1[t, i]+mu_y_neg1[t, i])
        # mu_y_1 += np.random.rand(8, 8)*1e-2
        # mu_y_neg1 += np.random.rand(8, 8)*1e-2
        if np.sqrt(np.sum(mat_memory - mu_y_1)**2) <= thr:
            break

    q_y_1 = np.zeros(8)
    q_y_neg1 = np.zeros(8)
    for i in range(8):
        q1 = np.prod(mu_y_1[np.where(theta[:, i] != 0), i])
        qn1 = np.prod(mu_y_neg1[np.where(theta[:, i] != 0), i])
        q_y_1[i] = q1/(q1+qn1)
        q_y_neg1[i] = qn1/(q1+qn1)
    return q_y_1, q_y_neg1, n+1


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


def plot_loopy_b_prop_sol(theta, num_iter, j_list=np.arange(0, 1, 0.1),
                          thr=1e-15, stim=0.1):
    lp = []
    ln = []
    nlist = []
    plt.figure()
    for j in j_list:
        pos, neg, n = Loopy_belief_propagation(theta=theta,
                                               num_iter=num_iter,
                                               j=j, thr=thr,
                                               stim=stim)
        lp.append(pos[0])
        ln.append(neg[0])
        nlist.append(n)
    plt.plot(j_list, lp, color='k')
    plt.plot(j_list, ln, color='r')
    plot_sol_LBP(j_list=np.arange(0, max(j_list), 0.00001), stim=stim)
    plt.title('Loopy-BP solution (symmetric)')
    plt.xlabel('J')
    plt.ylabel('q')
    plt.figure()
    plt.plot(j_list, nlist, color='k')
    plt.xlabel('J')
    plt.ylabel('n_iter for convergence, thr = {}'.format(thr))
    plt.figure()
    plt.plot(nlist, lp, color='k')
    plt.plot(nlist, ln, color='r')
    plt.xlabel('n_iter for convergence, thr = {}'.format(thr))
    plt.ylabel('q')
    # plt.figure()
    # q0_l, q1_l, q2_l = solutions_bp(j_list=j_list)
    # val_stop = np.log(3)/2
    # ind_stop = np.where(j_list >= val_stop)[0][0]
    # plt.plot(j_list[:ind_stop],
    #           np.array(ln[:ind_stop]) - np.array(q0_l[:ind_stop]), color='r')
    # plt.plot(j_list, np.array(ln)-np.array(q1_l), color='r')
    # plt.plot(j_list[:ind_stop],
    #           np.array(lp[:ind_stop]) - np.array(q0_l[:ind_stop]), color='k')
    # plt.plot(j_list, np.array(lp)-np.array(q2_l), color='k')
    # plt.xlabel('J')
    # plt.ylabel(r'$y_{sol} - y_{sim}$')


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
    q0_l, q1_l, q2_l = solutions_bp(j_list=j_list, stim=stim)
    plt.plot(j_list, q0_l, color='b')
    plt.plot(j_list, q1_l, color='b')
    plt.plot(j_list, q2_l, color='b')
    plt.xlabel('J')
    plt.ylabel('q')
    plt.title('Solutions of the dynamical system')


if __name__ == '__main__':
    plot_loopy_b_prop_sol(theta=THETA, num_iter=1000,
                          j_list=np.arange(0.00001, 1, 0.01),
                          thr=1e-12, stim=0.4)
