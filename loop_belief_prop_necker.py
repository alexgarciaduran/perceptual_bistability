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
from scipy.optimize import fsolve, bisect, root
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
                        (mu_y_1[jneigbours(t, i)[0], t]*
                         mu_y_1[jneigbours(t, i)[1], t])\
                        + np.exp(-theta[i, t]+stim) *\
                        (mu_y_neg1[jneigbours(t, i)[0], t] *
                         mu_y_neg1[jneigbours(t, i)[1], t])
                # mu_y_1 += np.random.rand(8, 8)*1e-3
                # negative y_i
                mu_y_neg1[t, i] = np.exp(-theta[i, t]-stim) *\
                    (mu_y_1[jneigbours(t, i)[0], t] * mu_y_1[jneigbours(t, i)[1], t])\
                    + np.exp(theta[i, t]-stim) *\
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
    fig, ax = plt.subplots(1)
    for j in j_list:
        pos, neg, n = Loopy_belief_propagation(theta=theta,
                                               num_iter=num_iter,
                                               j=j, thr=thr,
                                               stim=stim)
        lp.append(pos[0])
        ln.append(neg[0])
        nlist.append(n)
    ax.plot(j_list, lp, color='k')
    # plt.plot(j_list, ln, color='r')
    # plot_sol_LBP(j_list=np.arange(0, max(j_list), 0.00001), stim=stim)
    gn.true_posterior_plot_j(ax=ax, stim=stim)
    ax.set_title('Loopy-BP solution (symmetric)')
    ax.set_xlabel('J')
    ax.set_ylabel('q')
    ax.set_ylim(-0.05, 1.05)
    # plt.figure()
    # plt.plot(j_list, nlist, color='k')
    # plt.xlabel('J')
    # plt.ylabel('n_iter for convergence, thr = {}'.format(thr))
    # plt.figure()
    # plt.plot(nlist, lp, color='k')
    # plt.plot(nlist, ln, color='r')
    # plt.xlabel('n_iter for convergence, thr = {}'.format(thr))
    # plt.ylabel('q')
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


def plot_solution_BP(j_list, stim):
    j = np.exp(j_list*2)
    b = np.exp(stim*2)
    c1 = (b * j) / 3
    c2 = 2**(1/3) / (3 * (27 * b - 9 * b * j**2 + 2 * b**3 * j**3 + 3 * np.sqrt(3) * np.sqrt(27 * b**2 - 18 * b**2 * j**2 + 4 * j**3 + 4 * b**4 * j**3 - b**2 * j**4))**(1/3))
    c3 = (27 * b - 9 * b * j**2 + 2 * b**3 * j**3 + 3 * np.sqrt(3) * np.sqrt(27 * b**2 - 18 * b**2 * j**2 + 4 * j**3 + 4 * b**4 * j**3 - b**2 * j**4))**(1/3) / (3 * 2**(1/3))
    
    # Define the equations
    eq1 = c1 - c2 * (3 * j - b**2 * j**2) / (3 * c3) + c3 / (3 * 2**(1/3))
    # eq2 = c1 + ((1 + 1j * np.sqrt(3)) * (3 * j - b**2 * j**2)) / (3 * 2**(2/3) * c3) - ((1 - 1j * np.sqrt(3)) * c3) / (6 * 2**(1/3))
    # eq3 = c1 + ((1 - 1j * np.sqrt(3)) * (3 * j - b**2 * j**2)) / (3 * 2**(2/3) * c3) - ((1 + 1j * np.sqrt(3)) * c3) / (6 * 2**(1/3))

    plt.plot(j_list, eq1**3 / (1+eq1**3), color='b')
    plt.plot(j_list, 1 / (1+eq1**3), color='b')
    # plt.plot(j_list, eq2**3 / (1+eq2**3), color='b')
    # plt.plot(j_list, 1 / (1+eq2**3), color='b')
    # plt.plot(j_list, eq3**3 / (1+eq3**3), color='b')
    # plt.plot(j_list, 1 / (1+eq3**3), color='b')


def r_stim(x, j_e, b_e):
    return x**3 - j_e * b_e * x**2 + j_e * x - b_e


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
                     tol=1e-2):
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
        if np.sign(r_stim(a, j_e, b_e)*r_stim(b, j_e, b_e)) > 0:
            count += 1
        else:
            solution_bisection = bisect(r_stim, a=a, b=b,
                                        args=(j_e, b_e), xtol=1e-8)
            if len(sols) > 0:
                if (np.abs(np.array(sols) - solution_bisection).any()> tol):
                    sols.append(solution_bisection)
            else:
                sols.append(solution_bisection)
            count += 1
        
    # solution = fsolve(r_stim, fprime=r_stim_prime,
    #                   args=(j_e, b_e), x0=x0, xtol=1e-16,
    #                   maxfev=1000)
    # solution = root(r_stim, args=(j_e, b_e), x0=x0, tol=1e-12,
    #                 method='broyden2').x
    return sols


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
                     w_size=0.01):
    sols = []
    for ind_j, j in enumerate(j_list):
        sol = []
        sol=find_solution_bp(j, b=b, min_r=min_r, max_r=max_r, w_size=w_size,
                             tol=tol)
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
    linestyles = ['-', '--', '-']
    for i_s, sol in enumerate(solutions_array):
        ax.plot(j_list, sol**3 / (1+sol**3), color='r', linestyle=linestyles[i_s])
    ax.set_xlabel('J')
    ax.set_ylabel('q')


def backwards_bp(j, b, num_iter=100):
    m1 = np.random.rand()
    m_neg1 = np.random.rand()
    m1_list = []
    m_neg1_list = []
    m1_mem = np.copy(m1)
    m1 = m1 / (m1 + m_neg1)
    m_neg1 = m_neg1 / (m1_mem + m_neg1)
    for i in range(num_iter):
        if np.isnan(np.sqrt((m1-np.exp(b-j)*m_neg1**2) / (np.exp(j+b)))):
            break
        m1 = np.sqrt((m1-np.exp(b-j)*m_neg1**2) / (np.exp(j+b)))
        m_neg1 = np.sqrt((m_neg1-np.exp(-b-j)*m1**2) / (np.exp(j-b)))
        m1_mem = np.copy(m1)
        m1 = m1 / (m1 + m_neg1)
        m_neg1 = m_neg1 / (m1_mem + m_neg1)
        m1_list.append(m1)
        m_neg1_list.append(m_neg1)
    q = np.array(m1_list)**3 / (np.array(m1_list)**3+np.array(m_neg1_list)**3)
    q_neg1 = np.array(m_neg1_list)**3  / (np.array(m1_list)**3+np.array(m_neg1_list)**3)


if __name__ == '__main__':
    for stim in [-0.1, 0, 0.05]:
        plot_loopy_b_prop_sol(theta=THETA, num_iter=1000,
                              j_list=np.arange(0.00001, 1, 0.01),
                              thr=1e-12, stim=stim)
    j_list=np.arange(0.001, 1, 0.001)
    fig, ax = plt.subplots(1)
    plot_bp_solution(ax, j_list, b=0.1, tol=1e-12, min_r=-20, max_r=20,
                     w_size=0.01)
    