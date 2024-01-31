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


def Loopy_belief_propagation(theta, num_iter, j, thr=1e-5):
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
    M_y_1 = np.multiply(theta, np.random.rand(8, 8))
    mat_memory_1 = np.copy(M_y_1)
    M_y_neg1 = np.multiply(theta, np.random.rand(8, 8))
    mat_memory_neg1 = np.copy(M_y_neg1)
    theta = theta*j  # *(1-np.random.rand(8, 8))
    for n in range(num_iter):
        mat_memory_1 = np.copy(M_y_1)
        mat_memory_neg1 = np.copy(M_y_1)
        # for all the nodes that i is connected
        for i in range(8):
            for t in np.where(theta[i, :] != 0)[0]:
                # positive y_i
                M_y_1[t, i] = np.exp(theta[i, t]) *\
                        (M_y_1[jneigbours(t, i)[0], t]*M_y_1[jneigbours(t, i)[1], t])\
                        + np.exp(-theta[i, t]) *\
                        (M_y_neg1[jneigbours(t, i)[0], t] *
                         M_y_neg1[jneigbours(t, i)[1], t])
                # M_y_1 += np.random.rand(8, 8)*1e-3
                # negative y_i
                M_y_neg1[t, i] = np.exp(-theta[i, t]) *\
                    (M_y_1[jneigbours(t, i)[0], t] * M_y_1[jneigbours(t, i)[1], t])\
                    + np.exp(theta[i, t]) *\
                    (M_y_neg1[jneigbours(t, i)[0], t] *
                     M_y_neg1[jneigbours(t, i)[1], t])

                M_y_1[t, i] = M_y_1[t, i]/(M_y_1[t, i]+M_y_neg1[t, i])
                M_y_neg1[t, i] = M_y_neg1[t, i]/(M_y_1[t, i]+M_y_neg1[t, i])
                # M_y_neg1 += np.random.rand(8, 8)*1e-3
        if np.sqrt(np.sum(mat_memory_1 - M_y_1)**2) <= thr:
            # and\
            # np.sqrt(np.sum(mat_memory_neg1 - M_y_neg1)**2) <= thr:
            break

    q_y_1 = np.zeros(8)
    q_y_neg1 = np.zeros(8)
    for i in range(8):
        q1 = np.prod(M_y_1[np.where(theta[:, i] != 0), i])
        qn1 = np.prod(M_y_neg1[np.where(theta[:, i] != 0), i])
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
    M_y_1 = np.multiply(theta, np.random.rand(8, 8))
    mat_memory = np.copy(M_y_1)
    M_y_neg1 = np.multiply(theta, np.random.rand(8, 8))
    theta = theta*j
    for n in range(num_iter):
        mat_memory = np.copy(M_y_1)
        # for all the nodes that i is connected
        for i in range(8):
            for t in np.where(theta[i, :] != 0)[0]:
                # positive y_i
                M_y_1[t, i] = np.exp(theta[i, t]) *\
                        (M_y_1[jneigbours(t, i)[0], t]*M_y_1[jneigbours(t, i)[1], t])\
                        + np.exp(-theta[i, t]) *\
                        (M_y_neg1[jneigbours(t, i)[0], t] *
                         M_y_neg1[jneigbours(t, i)[1], t])

                # negative y_i
                M_y_neg1[t, i] = np.exp(-theta[i, t]) *\
                    (M_y_1[jneigbours(t, i)[0], t] * M_y_1[jneigbours(t, i)[1], t])\
                    + np.exp(theta[i, t]) *\
                    (M_y_neg1[jneigbours(t, i)[0], t] *
                     M_y_neg1[jneigbours(t, i)[1], t])

                M_y_1[t, i] = M_y_1[t, i]/(M_y_1[t, i]+M_y_neg1[t, i])
                M_y_neg1[t, i] = M_y_neg1[t, i]/(M_y_1[t, i]+M_y_neg1[t, i])
        # M_y_1 += np.random.rand(8, 8)*1e-2
        # M_y_neg1 += np.random.rand(8, 8)*1e-2
        if np.sqrt(np.sum(mat_memory - M_y_1)**2) <= thr:
            break

    q_y_1 = np.zeros(8)
    q_y_neg1 = np.zeros(8)
    for i in range(8):
        q1 = np.prod(M_y_1[np.where(theta[:, i] != 0), i])
        qn1 = np.prod(M_y_neg1[np.where(theta[:, i] != 0), i])
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
                          thr=1e-15):
    lp = []
    ln = []
    nlist = []
    plt.figure()
    for j in j_list:
        pos, neg, n = Loopy_belief_propagation(theta=theta,
                                               num_iter=num_iter,
                                               j=j, thr=thr)
        lp.append(pos[0])
        ln.append(neg[0])
        nlist.append(n)
    plt.plot(j_list, lp, color='k')
    plt.plot(j_list, ln, color='r')
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


def bp_sol(j, x=1):
    return 1/j * np.arctanh(np.tanh(j) * np.tanh(x*j))


def plot_bp_sol(j_list):
    plt.figure()
    sol = []
    for j in j_list:
        sol.append(bp_sol(j))
    plt.plot(j_list, sol)


if __name__ == '__main__':
    plot_loopy_b_prop_sol(theta=THETA, num_iter=100,
                          j_list=np.arange(0.0000001, 2, 0.01),
                          thr=1e-10)
