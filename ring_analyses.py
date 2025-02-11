# -*- coding: utf-8 -*-
"""
Created on Mon Feb 04 10:15:24 2025

@author: alexg
"""


import numpy as np
from scipy.linalg import circulant
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib as mpl
import itertools

mpl.rcParams['font.size'] = 16
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize']= 14
plt.rcParams['ytick.labelsize']= 14


class ring:
    def __init__(self, n_dots=6, epsilon=0.05):
        ring.ndots = n_dots
        ring.eps = epsilon
    
    def compute_likelihood_vector(self, s, z, s_t=[1]):
        """
        Compute the likelihood P(s_i = 1 | s, z) for all N positions.
        
        Parameters:
        - s: List or array of observed dot presences at time t-1 (length 6, values 0 or 1).
        - z: List or array of latent motion states at time t-1 (length 6, values -1 (CCW), 0 (NM), or 1 (CW)).
        - p_CW, p_NM, p_CCW, p_0: Probabilities defining likelihoods.
        
        Returns:
        - A numpy array of length 6 containing the likelihood for each position.
        """
        epsilon = self.eps
        p = 1-epsilon
        N = len(s)
        p_s = np.full(N, epsilon)  # Initialize with baseline probability
        p_CW = p_CCW = p_NM = p-epsilon
        for i in range(N):
            # Define neighboring indices with wrap-around (ring structure)
            i_prev = (i - 1) % N
            i_next = (i + 1) % N
    
            # Compute contributions from motion influences
            if s[i_prev] == 1 and z[i_prev] == 1:  # CW influence from left
                p_s[i] += p_CW
            if s[i_next] == 1 and z[i_next] == -1:  # CCW influence from right
                p_s[i] += p_CCW
            if s[i] == 1 and z[i] == 0:  # Stationary influence
                p_s[i] += p_NM
    
            # Apply inclusion-exclusion correction
            if s[i_prev] == 1 and s[i_next] == 1 and z[i_prev] == 1 and z[i_next] == -1:
                p_s[i] += -p_CW * p_CCW
            if s[i_prev] == 1 and s[i] == 1 and z[i_prev] == 1 and z[i] == 0:
                p_s[i] += -p_CW * p_NM
            if s[i_next] == 1 and s[i] == 1 and z[i_next] == -1 and z[i] == 0:
                p_s[i] += -p_CCW * p_NM
            if s[i_prev] == 1 and s[i] == 1 and s[i_next] == 1 and z[i_prev] == 1 and z[i] == 0 and z[i_next] == -1:
                p_s[i] += p_CW * p_CCW * p_NM
    
        # Ensure probabilities remain valid
        p_s = np.clip(p_s, 0, 1)
        # p_s = [p_s[i] if s_t[i] else 1-p_s[i] for i in range(N)]
        return p_s

    def exp_kernel(self, tau=0.8):
        x = np.arange(self.ndots)
        kernel = np.concatenate((np.exp(-(x-1)[:len(x)//2]/tau), (np.exp(-x[:len(x)//2]/tau))[::-1]))
        kernel[0] = 0
        return kernel / np.max(kernel)
    
    
    def kdelta(a, b):
        return 1 if a == b else 0

    
    def compute_expectation_log_likelihood(self, s, q_z_prev, s_t):
        """
        Compute the likelihood expectation over q_{i-1}(z_{i-1}) and q_{i+1}(z_{i+1})
        using the CPT (likelihood, Conditional Probabilities Table) and
        the approx. posterior over latent states.
    
        Parameters:
        - s (array): Stimulus.
        - q_z_prev (array-like): Belief on previous latent states probabilities (q_{z_{t-1}}).
        - z (array): latent scene (not very useful, just to have some structure)
          
        Returns:
        - likelihood_c_all (array): The computed log-likelihood expectation.
        """
    
        # Number of possible latent states (e.g., 3 for CW, NM, CCW)
        num_states_z = q_z_prev.shape[1]
        num_variables = q_z_prev.shape[0]
        # we want to compute \sum_{i \in N(j)} E_{q_{i-1}, q_{i+1}}[\log p(s_i | z_{i-1}, z_i, z_{i+1}, s)]
        # \sum_{i \in N(j)} E_{q_{i-1}, q_{i+1}}[\log p(s_i | z, s)]: num_variables x num_states_z
        likelihood_c_all = np.zeros((num_variables, num_states_z))
        z_states = [-1, 0, 1]
        combinations = list(itertools.product(z_states, z_states))
        for i in range(num_variables):  # for all dots
            # Iterate over all possible latent states z_{t-1}
            for z_i_index in range(num_states_z):  # for each possible state of z_i (columns)
                likelihood_contribution = 0
                # for startpoint in [0]:  # to get extra components \sum_{j \in N(i)}
                startpoint = 0
                # iterate over all possible combinations of neighbors
                for comb in combinations:  # for all combinations of z_{i-1}, z_{i+1}
                    i_prev = (i+startpoint-1) % num_variables
                    i_next = (i+startpoint+1) % num_variables
                    idx = (i+startpoint) % num_variables
                    zn = np.ones(num_variables)
                    zn[i_prev] = comb[0]  # z_{i-1}
                    zn[i_next] = comb[1]  # z_{i+1}
                    zn[idx] = z_states[z_i_index]  # z_i
                    # Get the probability of z from q_z_prev (approx. posterior)
                    # q_z_prev: num_variables x num_states_z (6 rows x 3 columns)
                    q_z_p = q_z_prev[i_prev, comb[0]+1]
                    q_z_n = q_z_prev[i_next, comb[1]+1]
    
                    # Get the CPT value for p(s_i | s, z)
                    p_s_given_z = self.compute_likelihood_vector(s, zn, s_t)[idx]  # CPT lookup based on s and z, takes p(s_i | s, z)
    
                    # Add q_{i-1} · q_{i+1} · p(s_i | s, z)
                    likelihood_contribution += np.log(p_s_given_z + 1e-10)*q_z_p*q_z_n
                    # print(likelihood_contribution)
                likelihood_c_all[i, z_i_index] = likelihood_contribution
        return likelihood_c_all


    def stim_creation(self, s_init=[0, 1], n_iters=100, true='NM'):
        """
        Create stimulus given a 'true' structure. true is a string with 4 possible values:
        - 'CW': clockwise rotation
        - 'CCW': counterclockwise rotation (note that it is 100% bistable)
        - 'NM': not moving
        - 'combination': start with CW and then jump to NM
        """
        s_init = np.repeat(np.array(s_init).reshape(-1, 1), self.ndots//len(s_init), axis=1).T.flatten()
        if true == 'NM':
            roll = 0
        if true == 'CW':
            roll = 1
        if true == 'CCW':
            roll = -1
        if true == 'combination':
            s = s_init
            sv = [s]
            for t in range(n_iters-1):
                if t < n_iters // 2:
                    roll = 1
                else:
                    roll = 0
                s = np.roll(s, roll)
                sv.append(s)
        else:
            s = s_init
            sv = [s]
            for _ in range(n_iters-1):
                s = np.roll(s, roll)
                sv.append(s)
        return np.row_stack((sv))    


    def mean_field_ring(self, j=2, n_iters=50, nstates=3, b=np.zeros(3),
                        true='NM', plot=False):
        # bifurcation at j ~ 0.554
        kernel = self.exp_kernel()
        n_dots = self.ndots
        # crit_j = 1/np.sum(kernel)
        # print(f'J* = {round(crit_j, 4)}')
        s = [1, 0]
        stim = self.stim_creation(s_init=s, n_iters=n_iters, true=true)
        s = np.repeat(np.array(s).reshape(-1, 1), n_dots//len(s), axis=1).T.flatten()
        # q_mf = np.repeat(np.array([[0.25], [0.3], [0.2]]), 6, axis=-1).T
        q_mf = np.ones((n_dots, nstates))/3 + np.random.randn(n_dots, 3)*0.05
        q_mf = (q_mf.T / np.sum(q_mf, axis=1)).T
        z = [np.random.choice([-1, 0, 1], p=q_mf[a]) for a in range(n_dots)]
        j_mat = circulant(kernel)*j
        np.fill_diagonal(j_mat, 0)
        q_mf_arr = np.zeros((n_dots, nstates, n_iters))
        q_mf_arr[:, :, 0] = q_mf
        s_arr = np.zeros((n_dots, n_iters))
        s_arr[:, 0] = s
        z_arr = np.zeros((n_dots, n_iters))
        z_arr[:, 0] = z
        for t in range(1, n_iters):
            stim_likelihood = self.compute_likelihood_vector(stim[t-1], z, stim[t])
            s = np.array([np.random.choice([0, 1], p=[1-stim_likelihood[a], stim_likelihood[a]]) for a in range(n_dots)])
            # if J*(2*Q-1), then it means repulsion between different z's, i.e. 2\delta(z_i, z_j) - 1
            # if J*Q, then it means just attraction to same, i.e. \delta(z_i, z_j)
            likelihood = self.compute_expectation_log_likelihood(stim[t-1], q_mf, stim[t])
            var_m1 = np.exp(np.matmul(j_mat, q_mf*2-1) + np.ones(3)*b + likelihood)  # np.random.randn(6, 3)
            q_mf = (var_m1.T / np.sum(var_m1, axis=1)).T
            q_mf_arr[:, :, t] = q_mf
            z = [np.random.choice([-1, 0, 1], p=q_mf[a]) for a in range(n_dots)]
            z_arr[:, t] = z
            s_arr[:, t] = s
        print('Percept: ' + str(['CCW', 'NM', 'CW'][int(np.mean(z_arr.T[-1]))+1]))
        if plot:
            plt.figure()
            colors = ['b', 'k', 'r']
            labels = ['CCW', 'NM', 'CW']
            for i in range(nstates):
                plt.plot(q_mf_arr[0, i, :], color=colors[i], label=labels[i])
            plt.legend(frameon=True)
    
            fig, ax = plt.subplots(nrows=4, figsize=(11, 16))
            title = ['CCW', 'NM', 'CW'][int(np.mean(z_arr.T[-1]))+1]
            ax[0].set_title('Percept: ' + title + ', True: ' + true + '\nR=q(z_i=CW), G=0, B=q(z_i=CCW)' )
            q_mf_plot = np.array((q_mf_arr[:, 0, :].T, q_mf_arr[:, 1, :].T,
                                  q_mf_arr[:, 2, :].T)).T
            q_mf_plot[:, :, 1] = 0
            ax[0].imshow(q_mf_plot, aspect='auto', interpolation='none',
                         vmin=0, vmax=1)
            # plt.colorbar(im1, ax=ax[0], label='sampled hidden state, z')
            # ax[0].set_title('Percept: ' + str(labels[int(np.mean(z_arr[:, :-10]))+2]))
            im2 = ax[1].imshow(stim.T, cmap='binary', aspect='auto', interpolation='none')
            plt.colorbar(im2, ax=ax[1], label='true stimulus, s')
            im2 = ax[2].imshow(s_arr, cmap='binary', aspect='auto', interpolation='none')
            plt.colorbar(im2, ax=ax[2], label='sampled (expected)\n stimulus, s*')
            im2 = ax[3].imshow(stim.T-s_arr, cmap='PiYG', aspect='auto', interpolation='none')
            plt.colorbar(im2, ax=ax[3], label='difference in stimulus')
            ax_0_pos = ax[0].get_position()
            ax_1_pos = ax[1].get_position()
            ax[0].set_position([ax_0_pos.x0, ax_0_pos.y0, ax_1_pos.width, ax_1_pos.height])
            ax = plt.figure().add_subplot(projection='3d')
            lab1 = 'NM'
            lab2 = 'CW'
            for i in range(n_dots):
                ax.plot(np.arange(n_iters), q_mf_arr[i, 0, :], q_mf_arr[i, 1, :],
                        color='k', label=lab1)
                ax.plot(np.arange(n_iters), q_mf_arr[i, 0, :], q_mf_arr[i, 2, :],
                        color='r', label=lab2)
                if i == 0:
                    ax.legend()
            ax.set_xlabel('Iterations')
            ax.set_zlabel('q(z_i = CW), q(z_i = NM)')
            ax.set_ylabel('q(z_i = CCW)')

    def compute_likelihood_contribution_BP(self, s, messages, stim_i):
        z_states = [-1, 0, 1]
        combinations = list(itertools.product(z_states, z_states))
        likelihood_vector = np.zeros(len(z_states))
        for idx, z_i in enumerate(z_states):
            llh_cont = 0
            for comb in combinations:
                z_neighbors = [comb[0], z_i, comb[1]]
                s_neighbors = s
                p_s = self.compute_likelihood_vector(s_neighbors, z_neighbors, stim_i)[1]
                message_im1 = messages[0, comb[0]+1]
                message_ip1 = messages[1, comb[1]+1]
                llh_cont += message_ip1*message_im1*p_s
            likelihood_vector[idx] = llh_cont
        return likelihood_vector


    def belief_propagation(self, j=2, n_iters=50, nstates=3, b=np.zeros(3),
                           true='NM', plot=False):
        kernel = self.exp_kernel()
        n_dots = self.ndots
        s = [1, 0]
        stim = self.stim_creation(s_init=s, n_iters=n_iters, true=true)
        s = np.repeat(np.array(s).reshape(-1, 1), n_dots//len(s), axis=1).T.flatten()
        q_mf = np.ones((n_dots, nstates))/3 + np.random.randn(6, 3)*0.05
        q_mf = (q_mf.T / np.sum(q_mf, axis=1)).T
        z = [np.random.choice([-1, 0, 1], p=q_mf[a]) for a in range(n_dots)]
        j_mat = circulant(kernel)*j
        np.fill_diagonal(j_mat, 0)
        q_mf_arr = np.zeros((n_dots, nstates, n_iters))
        q_mf_arr[:, :, 0] = q_mf
        s_arr = np.zeros((n_dots, n_iters))
        s_arr[:, 0] = s
        z_arr = np.zeros((n_dots, n_iters))
        z_arr[:, 0] = z
        states_z = np.array([-1, 0, 1])
        nstates = len(states_z)
        messages = np.ones((n_dots, n_dots, nstates))*np.random.rand(n_dots, n_dots, nstates)
        # Belief propagation iterations
        for t in range(n_iters):
            new_messages = np.ones_like(messages)
    
            for i in range(n_dots):  # for all nodes
                for j in np.where(j_mat[i] != 0)[0]:  # to compute m_{j \to i}
                    # compute prod(m_{k \to j}) for k \in N(j) different from {i-1, i, i+1}
                    incoming_messages = np.prod([messages[k, j] for k in range(n_dots) if k not in [j-1, j, j+1] and j_mat[i, k] > 0], axis=0)
                    # get s_{N(j)} = [s_{j-1}, s_{j}, s_{j+1}]
                    s_neighbors = [stim[t-1][(j - 1) % n_dots], stim[t-1][j], stim[t-1][(j + 1) % n_dots]]
                    s_t_neighbors = [stim[t][(j - 1) % n_dots], stim[t][j], stim[t][(j + 1) % n_dots]]
                    # get messages m_{j-1 \to j} and m_{j+1 \to j}
                    mes_input = np.row_stack((messages[(j-1) % n_dots, j],
                                              messages[(j+1) % n_dots, j]))
                    # now get expectation of p(s_j | z_{N(j)}, s_{N(j)}) over the messages for z_{j-1} and z_{j+1}
                    likelihood_vector = self.compute_likelihood_contribution_BP(s_neighbors, mes_input, s_t_neighbors)
                    # compute exp[J_{ij} \delta(z_i, z_j)] --> nxn
                    # we can do also 2*\delta(z_i, z_j)-1
                    compatibility = np.exp(j_mat[i, j] * (2*(states_z[:, None] == states_z)-1))
                    # compute new message, sum_{z_j} exp[B(z_j)] * E[LH](z_j) * \prod_{k} m_{k \to j}(z_j) @ exp[J \delta(z_i, z_j)]
                    new_messages[i, j] = np.exp(b) * likelihood_vector * (incoming_messages @ compatibility)
                    # normalize
                    new_messages[i, j] /= np.sum(new_messages[i, j])

            messages = new_messages.copy()

            # Compute final marginals q(z_i)
            marginals = np.prod([messages[j] for j in range(n_dots)], axis=0) * np.exp(b)
            for i in range(n_dots):
                marginals[i] /= np.sum(marginals[i])
            q_mf_arr[:, :, t] = marginals
            stim_likelihood = self.compute_likelihood_vector(stim[t-1], z, stim[t])
            s = np.array([np.random.choice([0, 1], p=[1-stim_likelihood[a], stim_likelihood[a]]) for a in range(n_dots)])
            z = [np.random.choice([-1, 0, 1], p=marginals[a]) for a in range(n_dots)]
            z_arr[:, t] = z
            s_arr[:, t] = s
        if plot:
            plt.figure()
            colors = ['b', 'k', 'r']
            labels = ['CCW', 'NM', 'CW']
            for i in range(nstates):
                plt.plot(q_mf_arr[0, i, :], color=colors[i], label=labels[i])
            plt.legend(frameon=True)
    
            fig, ax = plt.subplots(nrows=4, figsize=(11, 16))
            title = ['CCW', 'NM', 'CW'][int(np.mean(z_arr.T[-1]))+1]
            ax[0].set_title('Percept: ' + title + ', True: ' + true + '\nR=q(z_i=CW), G=0, B=q(z_i=CCW)' )
            q_mf_plot = np.array((q_mf_arr[:, 0, :].T, q_mf_arr[:, 1, :].T,
                                  q_mf_arr[:, 2, :].T)).T
            q_mf_plot[:, :, 1] = 0
            ax[0].imshow(q_mf_plot, aspect='auto', interpolation='none',
                         vmin=0, vmax=1)
            # plt.colorbar(im1, ax=ax[0], label='sampled hidden state, z')
            # ax[0].set_title('Percept: ' + str(labels[int(np.mean(z_arr[:, :-10]))+2]))
            im2 = ax[1].imshow(stim.T, cmap='binary', aspect='auto', interpolation='none')
            plt.colorbar(im2, ax=ax[1], label='true stimulus, s')
            im2 = ax[2].imshow(s_arr, cmap='binary', aspect='auto', interpolation='none')
            plt.colorbar(im2, ax=ax[2], label='sampled (expected)\n stimulus, s*')
            im2 = ax[3].imshow(stim.T-s_arr, cmap='PiYG', aspect='auto', interpolation='none')
            plt.colorbar(im2, ax=ax[3], label='difference in stimulus')
            ax_0_pos = ax[0].get_position()
            ax_1_pos = ax[1].get_position()
            ax[0].set_position([ax_0_pos.x0, ax_0_pos.y0, ax_1_pos.width, ax_1_pos.height])
            ax = plt.figure().add_subplot(projection='3d')
            lab1 = 'NM'
            lab2 = 'CW'
            for i in range(n_dots):
                ax.plot(np.arange(n_iters), q_mf_arr[i, 0, :], q_mf_arr[i, 1, :],
                        color='k', label=lab1)
                ax.plot(np.arange(n_iters), q_mf_arr[i, 0, :], q_mf_arr[i, 2, :],
                        color='r', label=lab2)
                if i == 0:
                    ax.legend()
            ax.set_xlabel('Iterations')
            ax.set_zlabel('q(z_i = CW), q(z_i = NM)')
            ax.set_ylabel('q(z_i = CCW)')

            

if __name__ == '__main__':
    # ring().mean_field_ring(true='combination', j=0.1, b=[0., 0., 0.], plot=True)
    ring(epsilon=0.001).belief_propagation(plot=True, true='combination', j=0.1)

