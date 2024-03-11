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
from scipy.optimize import fsolve, bisect, root
from scipy.integrate import solve_ivp
from matplotlib.lines import Line2D
import matplotlib.pylab as pl

THETA = gn.THETA

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


def posterior_comparison_MF_BP(stim_list=np.linspace(-2, 2, 1000), j=0.1,
                               num_iter=100, thr=1e-12, theta=THETA,
                               data_folder=DATA_FOLDER):
    true_posterior = gn.true_posterior_stim(stim_list=stim_list, j=j, theta=theta,
                                            data_folder=data_folder)
    mf_post = []
    bp_post = []
    for stim in stim_list:
        q = mfn.mean_field_stim(j, num_iter=num_iter, stim=stim, sigma=0,
                                theta=theta)
        mf_post.append(q[-1, 0])
        q_bp, _, _= Loopy_belief_propagation(theta=theta,
                                             num_iter=num_iter,
                                             j=j, thr=thr, stim=stim)
        bp_post.append(q_bp[0])
    fig, ax = plt.subplots(1)
    ax.plot(true_posterior, bp_post, color='k', label='Belief propagation')
    ax.plot(true_posterior, mf_post, color='r', label='Mean-field',
            linestyle='--')
    ax.plot([0, 1], [0, 1], color='grey', alpha=0.5)
    ax.set_xlabel(r'True posterior $p(x_i=1 | B)$')
    ax.set_ylabel(r'Approximated posterior $q(x_i=1|B)$')
    ax.set_title('J = '+str(j))
    ax.legend()


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
    neighs -= np.min(neighs)
    colors = ['k', 'r', 'b']
    for i_v, vals in enumerate(vals_all):
        ax.plot(j_list, vals, color=colors[neighs[i_v]], alpha=0.1)
    
    # plt.plot(j_list, ln, color='r')
    # plot_sol_LBP(j_list=np.arange(0, max(j_list), 0.00001), stim=stim)
    # gn.true_posterior_plot_j(ax=ax, stim=stim)
    ax.set_title('Loopy-BP solution (symmetric)')
    ax.set_xlabel('J')
    ax.set_ylabel('q')
    legendelements = [Line2D([0], [0], color='k', lw=2, label='3'),
                      Line2D([0], [0], color='r', lw=2, label='4'),
                      Line2D([0], [0], color='b', lw=2, label='5')]
    plt.legend(handles=legendelements, title='Neighbors')
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


def r_stim(x, j_e, b_e, n_neigh=3):
    return b_e*x**n_neigh - j_e * b_e * x**(n_neigh-1) + j_e * x - 1


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
                     tol=1e-2, n_neigh=3):
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
        if np.sign(r_stim(a, j_e, b_e, n_neigh=n_neigh)*
                   r_stim(b, j_e, b_e, n_neigh=n_neigh)) > 0:
            count += 1
        else:
            solution_bisection = bisect(r_stim, a=a, b=b,
                                        args=(j_e, b_e, n_neigh),
                                        xtol=1e-8)
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
    ax.set_xlabel('J')
    ax.set_ylabel('q')


def dynamical_system_BP(t, m, j, stim, theta):
    m = m.reshape(16, 8)
    mu_y_1 = m[:8, :]
    mu_y_neg1 = m[8:, :]
    theta = theta*j
    for i in range(8):
        for m in np.where(theta[i, :] != 0)[0]:
            # positive y_i
            mu_y_1[m, i] = np.exp(theta[i, m]+stim) *\
                    (mu_y_1[jneigbours(m, i)[0], m]*
                     mu_y_1[jneigbours(m, i)[1], m])\
                    + np.exp(-theta[i, m]+stim) *\
                    (mu_y_neg1[jneigbours(m, i)[0], m] *
                     mu_y_neg1[jneigbours(m, i)[1], m])
            # mu_y_1 += np.random.rand(8, 8)*1e-3
            # negative y_i
            mu_y_neg1[m, i] = np.exp(-theta[i, m]-stim) *\
                (mu_y_1[jneigbours(m, i)[0], m] * mu_y_1[jneigbours(m, i)[1], m])\
                + np.exp(theta[i, m]-stim) *\
                (mu_y_neg1[jneigbours(m, i)[0], m] *
                 mu_y_neg1[jneigbours(m, i)[1], m])

            m_y_1_memory = np.copy(mu_y_1[m, i])
            mu_y_1[m, i] = mu_y_1[m, i]/(m_y_1_memory+mu_y_neg1[m, i])
            mu_y_neg1[m, i] = mu_y_neg1[m, i]/(m_y_1_memory+mu_y_neg1[m, i])
    m = np.concatenate((mu_y_1.flatten(),
                        mu_y_neg1.flatten()))
    m = np.clip(m, 0, np.max(m))
    return m


def plot_dyn_sys_BP_solution():
    j_list=np.arange(0.01, 1, 0.01)
    q1_list = []
    for j in j_list:
        theta = THETA
        mu_y_1 = np.multiply(theta, np.random.rand(8, 8))
        mu_y_neg1 = np.multiply(theta, np.random.rand(8, 8))
        init_cond = np.concatenate((mu_y_1, mu_y_neg1))
        m_solution = solve_ivp(fun=dynamical_system_BP, t_span=[0, 10],
                                y0=init_cond.flatten(),
                                args=(j, 0.05, theta))
        m = m_solution.y[:, -1].reshape(16, 8)
        mu_y_1 = m[:8, :]
        mu_y_1_memory = np.copy(mu_y_1)
        mu_y_neg1 = m[8:, :]
        mu_y_neg1_memory = np.copy(mu_y_1)
        q_y_1 = np.zeros(8)
        q_y_neg1 = np.zeros(8)
        for i in range(8):
            q1 = np.prod(mu_y_1[np.where(theta[:, i] != 0), i])
            qn1 = np.prod(mu_y_neg1[np.where(theta[:, i] != 0), i])
            q_y_1[i] = q1/(q1+qn1)
            q_y_neg1[i] = qn1/(q1+qn1)
        q_1 = q_y_1[np.abs(q_y_1) <= 1][0]
        q1_list.append(q_1)
    plt.plot(j_list, q1_list)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('J')
    plt.ylabel('q')
    

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
    # return np.log((b*N-2*b-2
    #                 + val* np.sqrt((-b*N+2*b+2)**2 - 4 *
    #                           (-N*b*b+b*b+b+N) *
    #                           (b*b*N-b*b +b*N - b - N +2))) /
    #               (2 * (b*b*N - b*b + b*N - b - N + 2)))/2
    return np.log((-1-b
                    + val* np.sqrt((1+b)**2 -4*(2+b)*(-2*N*b**2+2*(2+b)*N**2))) /
                  (2 * (2+b)))/2
    # numerator = (-np.sqrt((-b * N**2 + 2 * b * N - N**2 + 3 * N)**2 - 4 * (b**2 * N - b**2 + b * N - b - N + 2) * (b**2 * (-N) + b**2 + b * N**2 - b * N + b + N**2)) + b * N**2 - 2 * b * N + N**2 - 3 * N)
    # denominator = 2 * (b**2 * N - b**2 + b * N - b - N + 2)
    # x = numerator / denominator
    # return np.log(x)/2


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
                                          neigh_list=[3, 4, 5]):
    fig, ax = plt.subplots(ncols=3, figsize=(10, 6))
    colors = ['k', 'r', 'b']
    neigh_list = [3, 4, 5]
    for n_neigh in neigh_list:
        plot_bp_solution(ax[n_neigh-min(neigh_list)], j_list, b=0.05, tol=1e-10,
                         min_r=-15, max_r=15,
                         w_size=0.01, n_neigh=n_neigh,
                         color=colors[n_neigh-min(neigh_list)])
        ax[n_neigh-min(neigh_list)].set_title(str(n_neigh) + ' neighbors')


if __name__ == '__main__':
    # for stim in [0.1]:
    #     plot_loopy_b_prop_sol(theta=gn.return_theta(), num_iter=50,
    #                           j_list=np.arange(0.00001, 1, 0.01),
    #                           thr=1e-12, stim=stim)
    
    posterior_comparison_MF_BP(stim_list=np.linspace(-2, 2, 1000), j=0.2,
                               num_iter=40, thr=1e-8, theta=gn.return_theta())
    # j_list=np.arange(0.001, 1, 0.001)
    # fig, ax = plt.subplots(ncols=1)
    # plot_bp_solution(ax, j_list, b=0.05, tol=1e-10,
    #                  min_r=-15, max_r=15,
    #                  w_size=0.01, n_neigh=3,
    #                  color='r')

