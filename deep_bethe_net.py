# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:35:47 2024

@author: alexg
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools


THETA = np.array([[0 ,1 ,1 ,0 ,1 ,0 ,0 ,0], [1, 0, 0, 1, 0, 1, 0, 0],
                  [1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1],
                  [0, 0, 1, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 1, 1, 0]])

THETA = np.array([[0, .4, 5, 0, 0], [.8, 0, 0, .8, 20],
                  [1.4, 0, 0, 0, 0], [0, 10, 0, 0, 0],
                  [0, -11, 0, 0, 0]])


class DBN:
    def __init__(self, tau_m, tau_n, tau_x, learning_rate,
                 learning_rate_plast, theta=THETA,
                 stimulus=0, j=0.4, beta_val=1):
        num_variables = theta.shape[0]
        self.num_variables = num_variables
        self.tau_m = tau_m
        self.tau_n = tau_n
        self.tau_x = tau_x
        self.learning_rate = learning_rate
        # chi_ij = psi_{ij}(x_i, x_j) = signed_theta_{ij} * J
        self.chi_ij = np.dstack((theta*j, theta*j)).T
        # self.get_theta_signed(j, theta))
        self.chi_i = np.empty((2, num_variables))
        self.chi_i[0] = stimulus
        self.chi_i[1] = -stimulus
        self.M = np.random.randn(2, num_variables)  # random init of beliefs
        self.M = np.log(np.exp(self.M)/(np.sum(np.exp(self.M), axis=0)))  # normalize marginal beliefs
        self.X = np.multiply(np.random.randn(2, num_variables, num_variables), theta)  # random init of messages
        self.N = np.multiply(np.random.randn(2, num_variables, num_variables), theta)  # random init of joint belief
        self.N = np.log(np.exp(self.N)/(np.sum(np.exp(self.N), axis=0)))  # normalize joint beliefs
        self.beta = np.random.rand(num_variables, num_variables)*beta_val
        self.beta[:] = 1
        self.theta = theta
        self.eps_i = np.random.rand(num_variables)/10
        self.eps_i[:] = 1
        self.eps_ij = np.random.rand(num_variables, num_variables)/10
        self.eps_ij[:] = 1


    def get_theta_signed(self, j, theta):
        th = np.copy(theta)
        for t in range(8):
            ind = self.get_connections(t)
            th[t, ind] = [-1 if ((i >= 4) and (t <= 3)
                                 or (i <= 3) and (t >= 4)) else 1 for i in ind]
        return th*j
    
    
    def get_connections(self, node):
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


    def forward(self, dt=1e-3):
        # M_t = self.M + dt*(np.sum(np.matmul(self.X, self.theta), axis=2)
        #                    - self.M)/self.tau_m
        # N_t = self.N + dt*(np.log(self.mu_ij) + self.M + self.M.T
        #                    - np.matmul(self.beta, self.X+self.X.T)
        #                    - self.N)/self.tau_n
        # X_t = self.X + dt*(np.log(self.mu_i) + 
        #                    np.log(np.sum(np.exp(self.N), axis=0)) -
        #                    self.M)/self.tau_n
        M_t = np.zeros(self.M.shape)
        X_t = np.zeros(self.X.shape)
        N_t = np.zeros(self.N.shape)
        for i in range(self.theta.shape[0]):
            # \tau_M dM_i(y_i)/dt = \sum_j X(z_j, i) - M_i (y_i) + \eps_i
            M_t[:, i] = self.M[:, i] +\
            dt*(np.sum(self.X[:, i], axis=1) - self.M[:, i] + self.eps_i[i])/self.tau_m
            for j in np.where(self.theta[i, :] != 0)[0]:
                # \tau_N dN_{ij}(y_i, z_j)/dt = \chi_{ij}(y_i, z_j) + M_i(y_i) + M_j(z_j)
                # - \beta_{ij} * (X_{ij}(y_i, j) + X_{ji}(z_j, i)) - N_{ij}(y_i, z_j) + \eps_{ij}
                N_t[:, i, j] = self.N[:, i, j] +\
                    dt*(self.beta[i, j]*self.chi_ij[:, i, j] + self.M[:, i] + self.M[:, j] +
                        self.chi_i[:, i] + self.chi_i[:, j] + 
                        - self.beta[i, j] * (self.X[:, i, j] + self.X[:, j, i])
                        - self.N[:, i, j] + self.eps_ij[i, j])/self.tau_n
                # \tau_X dX_{ij}(z_i, j)/dt = \chi_{i}(z_i) + log(\sum_j exp(N_{ij}(z_i, z_j))) - M_i (z_i)
                X_t[:, i, j] = self.X[:, i, j] +\
                dt*(self.chi_i[:, i] + np.log(np.sum(np.exp(self.N[:, i]), axis=1))-
                    self.M[:, i])/self.tau_x
        self.M = M_t
        self.X = X_t
        self.N = N_t


    def simulate(self, time_end, dt=1e-3):
        time = np.arange(0, time_end, dt)
        mat_m = np.zeros((2, self.num_variables, len(time)))
        for t in range(len(time)):
            self.forward(dt=dt)
            mat_m[:, :, t] = self.M
        return time, mat_m


    def calc_true_post(self, j, stim):
        vals_pos = []
        vals_neg = []
        for i in range(self.theta.shape[0]):
            combs_7_vars = list(itertools.product([-1, 1], repeat=self.theta.shape[0]))
            # over variable i
            x_vect_1 = [item for item in combs_7_vars if item[i] == 1]
            x_vect_0 = [item for item in combs_7_vars if item[i] == -1]
            exponent_1, exponent_0 = self.calc_exponents(x_vect_1, x_vect_0,
                                                         j=j, stim=stim,
                                                         j_mat=self.theta)
            prob_x1 = exponent_1 / (exponent_0 + exponent_1)
            prob_x0 = 1-prob_x1
            vals_pos.append(prob_x1)
            vals_neg.append(prob_x0)
        return vals_pos, vals_neg


    def calc_exponents(self, x_vects_1, x_vects_0, j, stim, j_mat):
        exponent_1 = 0
        exponent_0 = 0
        for x_vec_1, x_vec_0 in zip(x_vects_1, x_vects_0):
            x_vec_1 = np.array(x_vec_1)
            x_vec_0 = np.array(x_vec_0)
            # x_vec_1[4:] *= -1
            exponent_1 += np.exp(0.5*j*np.matmul(np.matmul(x_vec_1.T, j_mat), x_vec_1) + np.sum(stim*x_vec_1))
            exponent_0 += np.exp(0.5*j*np.matmul(np.matmul(x_vec_0.T, j_mat), x_vec_0) + np.sum(stim*x_vec_0))
        return exponent_1, exponent_0



if __name__ == '__main__':
    inputs = int(input('Inputs?' ))
    if inputs:
        stimulus = float(input('Sensory evidence, B: '))
        coupling = float(input('Coupling, J: '))
    else:
        stimulus = 0.2
        coupling = 0.1
    dbn = DBN(tau_m=0.5, tau_n=0.5, tau_x=0.5, learning_rate=0.1,
              learning_rate_plast=0.1, theta=THETA, stimulus=stimulus,
              j=coupling, beta_val=1)
    time, m = dbn.simulate(2)
    beliefs1 = np.exp(m[0])*np.exp(stimulus)
    beliefs2 = np.exp(m[1])*np.exp(-stimulus)
    cte_belief = np.sum((beliefs2, beliefs1), axis=0)
    beliefs1 /= cte_belief
    beliefs2 /= cte_belief
    plt.figure()
    vals_pos, _ = dbn.calc_true_post(j=coupling, stim=stimulus)
    colors = ['c', 'slateblue', 'k', 'r', 'seagreen', 'orange',
              'darkred', 'chocolate']
    for i in range(THETA.shape[0]):
        plt.axhline(vals_pos[i], linestyle='--', color=colors[i])
    # plt.axhline(0.5, color='k', linestyle='--')
    plt.ylim(-0.02, 1.02)
    [plt.plot(time, beliefs1[i], color=colors[i]) for i in range(THETA.shape[0])]
    plt.ylabel(r'Belief, $q_i(x=1)$')
    plt.xlabel('Time (s)')
    legendelements = [Line2D([0], [0], color='k', lw=2, label=r'Belief of node $i$'),
                      Line2D([0], [0], color='k', linestyle='--', lw=2, label='True posterior')]
    plt.legend(handles=legendelements)
