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
import scipy
import os
import sympy

mpl.rcParams['font.size'] = 16
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize']= 14
plt.rcParams['ytick.labelsize']= 14


pc_name = 'alex'
if pc_name == 'alex':
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/ring_analyses/data_folder/'  # Alex

elif pc_name == 'alex_CRM':
    DATA_FOLDER = 'C:/Users/agarcia/Desktop/phd/necker/data_folder/'  # Alex CRM


class ring:
    def __init__(self, n_dots=6, epsilon=0.05):
        ring.ndots = n_dots
        ring.eps = epsilon
    
    def compute_likelihood_vector(self, s, z, s_t=np.ones(6)):
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
            # if s[i_prev] == 0 and z[i_prev] == 1:
            #     p_s[i] *= (1-p)
            # if s[i_next] == 0 and z[i_next] == -1:
            #     p_s[i] *= (1-p)

        # Ensure probabilities remain valid
        p_s = np.clip(p_s, 0, 1)
        p_s = [p_s[i] if s_t[i] == 1 else 1-p_s[i] for i in range(N)]
        return p_s


    def compute_likelihood_continuous_stim(self, s, z, s_t=np.ones(6), noise=0.1,
                                           penalization=0):
        phi = np.roll(s, -1)*(np.roll(z, -1) == -1) + np.roll(s, 1)*(np.roll(z, 1) == 1) + s*(z == 0)
        norms = 1*(np.roll(z, 1) == 1) + 1*(z==0) + 1*(np.roll(z, -1) == -1)
        phi[norms == 0] = penalization
        norms[norms == 0] = 1
        likelihood = 1/(noise*np.sqrt(2*np.pi))*np.exp(-((s_t-phi/norms)**2)/noise**2/2)
        # likelihood = scipy.stats.norm.pdf(s_t, phi/3, noise)
        # likelihood = np.min((likelihood, np.ones(len(likelihood))), axis=0)
        return likelihood+1e-12


    def exp_kernel(self, tau=0.8):
        x = np.arange(self.ndots)
        kernel = np.concatenate((np.exp(-(x-1)[:len(x)//2]/tau), (np.exp(-x[:len(x)//2]/tau))[::-1]))
        kernel[0] = 0
        return kernel / np.max(kernel)

    
    def kdelta(a, b):
        return 1 if a == b else 0


    def compute_expectation_log_likelihood(self, s, q_z_prev, s_t, discrete_stim=True,
                                           noise=0.1):
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
                for startpoint in [-1, 0, 1]:  # to get extra components \sum_{j \in N(i)}
                # np.arange(-self.ndots//2, self.ndots//2, 1)
                    # startpoint = 0
                    # if startpoint == 0:
                    #     continue
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
                        # q_z_prev: num_variables x num_states_z (n_dots rows x 3 columns)
                        q_z_p = q_z_prev[i_prev, comb[0]+1]
                        q_z_n = q_z_prev[i_next, comb[1]+1]
        
                        # Get the CPT value for p(s_i | s, z)
                        if discrete_stim:
                            p_s_given_z = self.compute_likelihood_vector(s, zn, s_t)[idx]  # CPT lookup based on s and z, takes p(s_i | s, z)
                        else:
                            # based on a normal distribution centered in the expectation of the stimulus given the combination of z
                            p_s_given_z = self.compute_likelihood_continuous_stim(s, zn, s_t, noise=noise)[idx]
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
        - '2combination': start with CW and then jump to NM and jump again to CW
        - 'combination_reverse': start with NM and jump to CW
        """
        s_init = np.repeat(np.array(s_init).reshape(-1, 1), self.ndots//len(s_init), axis=1).T.flatten()
        if true == 'NM':
            roll = 0
        if true == 'CW':
            roll = 1
        if true == 'CCW':
            roll = -1
        if 'combination' in true:
            s = s_init
            sv = [s]
            for t in range(n_iters-1):
                if true == '2combination':
                    if t < n_iters // 3:
                        roll = 1
                    if n_iters //3 <= t < 2*n_iters // 3:
                        roll = 0
                    else:
                        roll = 1
                if true == 'combination':
                    if t < n_iters // 2:
                        roll = 1
                    else:
                        roll = 0
                if true == 'combination_reverse':
                    if t < n_iters // 3:
                        roll = 0
                    else:
                        roll = 1
                s = np.roll(s, roll)
                sv.append(s)
        else:
            s = s_init
            sv = [s]
            for _ in range(n_iters-1):
                s = np.roll(s, roll)
                sv.append(s)
        return np.row_stack((sv)) 
    
    
    def stim_creation_ou_process(self, s_init=[0, 1], n_iters=100, true='NM',
                                 dt=1e-2, tau=0.6, noise=0.1, change=1e-2):
        """
        Create stimulus given a 'true' structure. true is a string with 4 possible values:
        - 'CW': clockwise rotation
        - 'CCW': counterclockwise rotation (note that it is 100% bistable)
        - 'NM': not moving
            - 'combination': start with CW and then jump to NM
            - '2combination': start with CW and then jump to NM and jump again to CW
            - 'combination_reverse': start with NM and jump to CW
        """
        s_init = np.repeat(np.array(s_init).reshape(-1, 1), self.ndots//len(s_init), axis=1).T.flatten()
        if true == 'NM':
            roll = 0
        if true == 'CW':
            roll = 1
        if true == 'CCW':
            roll = -1
        sv = np.zeros((self.ndots, n_iters))
        sv[:, 0] = s_init
        for it in range(1, n_iters):
            if true != 'NM':
                r = 1 if it % 2 == 0 else 0
            else:
                r = 0
            sv[:, it] = sv[:, it-1] + dt*(np.roll(s_init, r)-sv[:, it-1])/tau + np.random.randn(self.ndots)*np.sqrt(dt/tau)*noise
            sv[:, it] = np.roll(sv[:, it], roll)
        return sv.T


    def mean_field_ring(self, j=2, n_iters=50, nstates=3, b=np.zeros(3),
                        true='NM', plot=False, noise=0, ini_cond=None):
        # bifurcation at j ~ 0.554
        kernel = self.exp_kernel()
        n_dots = self.ndots
        # crit_j = 1/np.sum(kernel)
        # print(f'J* = {round(crit_j, 4)}')
        s = [1, 0]
        stim = self.stim_creation(s_init=s, n_iters=n_iters, true=true)
        # stim[:, :-1] = 0
        s = np.repeat(np.array(s).reshape(-1, 1), n_dots//len(s), axis=1).T.flatten()
        # q_mf = np.repeat(np.array([[0.25], [0.3], [0.2]]), 6, axis=-1).T
        if ini_cond is None:
            q_mf = np.ones((n_dots, nstates))/3 + np.random.randn(n_dots, 3)*0.05
        else:
            ini_cond = np.array(ini_cond)
            q_mf = np.repeat(ini_cond.reshape(-1, 1), self.ndots, axis=1).T
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
            stim_likelihood = self.compute_likelihood_vector(stim[t-1], z, np.ones(n_dots))
            s = np.array([np.random.choice([0, 1], p=[1-stim_likelihood[a], stim_likelihood[a]]) for a in range(n_dots)])
            # if J*(2*Q-1), then it means repulsion between different z's, i.e. 2\delta(z_i, z_j) - 1
            # if J*Q, then it means just attraction to same, i.e. \delta(z_i, z_j)
            # likelihood = self.compute_expectation_log_likelihood(s=stim[t-1], q_z_prev=q_mf_arr[:, :, t-1],
            #                                                      s_t=stim[t])
            # print(likelihood)
            var_m1 = np.exp(np.matmul(j_mat, q_mf*2-1) + b + np.random.randn(n_dots, nstates)*noise)
            q_mf = (var_m1.T / np.sum(var_m1, axis=1)).T
            q_mf_arr[:, :, t] = q_mf
            z = [np.random.choice([-1, 0, 1], p=q_mf[a]) for a in range(n_dots)]
            z_arr[:, t] = z
            s_arr[:, t] = s
        print('Percept: ' + str(['CCW', 'NM', 'CW'][int(np.mean(z_arr.T[-1]))+1]))
        if plot:
            plt.figure()
            colors = ['r', 'k', 'b']
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
            im2 = ax[1].imshow(stim.T, cmap='binary', aspect='auto', interpolation='none',
                               vmin=0, vmax=1)
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
            lab2 = 'CCW'
            for i in range(n_dots):
                ax.plot(np.arange(n_iters), q_mf_arr[i, 0, :], q_mf_arr[i, 1, :],
                        color='k', label=lab1)
                ax.plot(np.arange(n_iters), q_mf_arr[i, 0, :], q_mf_arr[i, 2, :],
                        color='b', label=lab2)
                if i == 0:
                    ax.legend()
            ax.set_xlabel('Iterations')
            ax.set_zlabel('q(z_i = CCW), q(z_i = NM)')
            ax.set_ylabel('q(z_i = CW)')


    def mean_field_sde(self, dt=0.001, tau=1, n_iters=100, j=2, nstates=3, b=np.zeros(3),
                       true='NM', noise=0.2, plot=False, stim_weight=1, ini_cond=None,
                       discrete_stim=True, s=[1, 0], noise_stim=0.05, coh=None):
        t_end = n_iters*dt
        kernel = self.exp_kernel()
        n_dots = self.ndots
        if discrete_stim:
            if coh is None:
                stim = self.stim_creation(s_init=s, n_iters=n_iters, true=true)
            if coh is not None:
                stim = self.dummy_stim_creation(n_iters=n_iters, true=true, coh=coh)
        else:
            stim = self.stim_creation_ou_process(true=true, noise=noise_stim, s_init=s,
                                                 n_iters=n_iters, dt=dt)
        if discrete_stim and coh is not None:
            discrete_stim = False
        s = np.repeat(np.array(s).reshape(-1, 1), n_dots//len(s), axis=1).T.flatten()
        # q_mf = np.repeat(np.array([[0.25], [0.3], [0.2]]), 6, axis=-1).T
        # q_mf = np.ones((n_dots, nstates))/3 + np.random.randn(n_dots, 3)*0.05
        if ini_cond is None:
            q_mf = np.random.rand(n_dots, nstates)
        else:
            ini_cond = np.array(ini_cond)
            q_mf = np.repeat(ini_cond.reshape(-1, 1), self.ndots, axis=1).T
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
            # if J*(2*Q-1), then it means repulsion between different z's, i.e. 2\delta(z_i, z_j) - 1
            # if J*Q, then it means just attraction to same, i.e. \delta(z_i, z_j)
            likelihood = self.compute_expectation_log_likelihood(stim[t-1], q_mf, stim[t],
                                                                  discrete_stim=discrete_stim,
                                                                  noise=noise_stim)
            var_m1 = np.exp(np.matmul(j_mat, q_mf*2-1) + b + likelihood*stim_weight)
            q_mf = q_mf + dt/tau*(var_m1.T / np.sum(var_m1, axis=1) - q_mf.T).T + np.random.randn(n_dots, nstates)*noise*np.sqrt(dt/tau)
            q_mf_arr[:, :, t] = q_mf
        if not plot:
            return q_mf
        if plot:
            time = np.arange(0, t_end, dt)
            fig, ax = plt.subplots(nrows=4, figsize=(8, 12))
            for i_a, a in enumerate(ax.flatten()):
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)
                if i_a < 3:
                    a.set_xticks([])
                if i_a > 0:
                    a.set_ylim(-0.15, 1.15)
            ax[3].set_xlabel('Time (s)')
            ax[0].imshow(stim.T, cmap='binary', aspect='auto', interpolation='none',
                         vmin=0, vmax=1)
            ax[0].set_ylabel('Stimulus')
            ax[1].set_ylabel('q(z_i=CW)')
            ax[2].set_ylabel('q(z_i=NM)')
            ax[3].set_ylabel('q(z_i=CCW)')
            ax[1].axhline(1/3, color='r', alpha=0.4, linestyle='--', linewidth=2)
            ax[1].text(t_end+70*dt, 1/3-0.02, '1/3', color='r')
            ax[1].text(t_end+70*dt, 1/2+0.02, '1/2', color='r')
            ax[1].axhline(1/2, color='r', alpha=0.4, linestyle=':', linewidth=2)
            ax[2].axhline(1/3, color='k', alpha=0.4, linestyle='--', linewidth=2)
            ax[3].axhline(1/3, color='b', alpha=0.4, linestyle='--', linewidth=2)
            ax[3].axhline(1/2, color='b', alpha=0.4, linestyle=':', linewidth=2)
            if true == '2combination':
                title = r'$CW \longrightarrow NM \longrightarrow CW$'
            if true == 'combination':
                title = r'$CW \longrightarrow NM$'
            if true == 'combination_reverse':
                title = r'$NM \longrightarrow CW$'
            if true not in ['combination', '2combination', 'combination_reverse']:
                title = true
            ax[0].set_title(title)
            for dot in range(n_dots):
                ax[1].plot(time, q_mf_arr[dot, 0, :], color='r', linewidth=2.5)
                ax[2].plot(time, q_mf_arr[dot, 1, :], color='k', linewidth=2.5)
                ax[3].plot(time, q_mf_arr[dot, 2, :], color='b', linewidth=2.5)
            fig.suptitle(f'Coupling J = {j}', fontsize=16)
            plt.figure()
            plt.plot(np.max(np.abs(np.diff(stim.T, axis=0)), axis=0))
            plt.ylim(0, 2)
    
    
    def mean_field_sde_ising(self, dt=0.001, tau=1, n_iters=100, j=2, nstates=2, b=np.zeros(2),
                            true='NM', noise=0.2, plot=False, stim_weight=1, ini_cond=None,
                            discrete_stim=True, s=[1, 0], noise_stim=0.05, coh=None,
                            stim_stamps=50):
        t_end = n_iters*dt
        kernel = self.exp_kernel()
        n_dots = self.ndots
        if discrete_stim:
            if coh is None:
                stim = self.stim_creation(s_init=s, n_iters=n_iters, true=true)
            if coh is not None:
                stim = self.dummy_stim_creation(n_iters=n_iters, true=true, coh=coh,
                                                timesteps_between=stim_stamps)
        else:
            stim = self.stim_creation_ou_process(true=true, noise=noise_stim, s_init=s,
                                                 n_iters=n_iters, dt=dt)
        if discrete_stim and coh is not None:
            discrete_stim = False
        s = np.repeat(np.array(s).reshape(-1, 1), n_dots//len(s), axis=1).T.flatten()
        # q_mf = np.repeat(np.array([[0.25], [0.3], [0.2]]), 6, axis=-1).T
        # q_mf = np.ones((n_dots, nstates))/3 + np.random.randn(n_dots, 3)*0.05
        if ini_cond is None:
            q_mf = 0.5+np.random.randn(n_dots, nstates)*0.01
        else:
            ini_cond = np.array(ini_cond)
            q_mf = np.repeat(ini_cond.reshape(-1, 1), self.ndots, axis=1).T
        q_mf = (q_mf.T / np.sum(q_mf, axis=1)).T
        z = [np.random.choice([-1, 1], p=q_mf[a]) for a in range(n_dots)]
        j_mat = circulant(kernel)*j
        np.fill_diagonal(j_mat, 0)
        # j_mat = j_mat - (np.roll(np.eye(self.ndots), -1, axis=0) + np.roll(np.eye(self.ndots), 1, axis=0))*np.log(self.eps)/4
        q_mf_arr = np.zeros((n_dots, nstates, n_iters))
        q_mf_arr[:, :, 0] = q_mf
        s_arr = np.zeros((n_dots, n_iters))
        s_arr[:, 0] = s
        z_arr = np.zeros((n_dots, n_iters))
        z_arr[:, 0] = z
        for t in range(1, n_iters):
            # if J*(2*Q-1), then it means repulsion between different z's, i.e. 2\delta(z_i, z_j) - 1
            # if J*Q, then it means just attraction to same, i.e. \delta(z_i, z_j)
            # likelihood = self.compute_expectation_log_likelihood(stim[t-1], q_mf, stim[t],
            #                                                      discrete_stim=discrete_stim,
            #                                                      noise=noise_stim)
            jarr, mum1arr, mup1arr = compute_jstim_biases(stim[t], stim[t-1], sigma=noise_stim, include_m11=True, logm11=1e-2)
            jaddmat = (np.roll(np.eye(self.ndots), -2, axis=0) + np.roll(np.eye(self.ndots), 2, axis=0))*np.array(jarr, dtype=np.float64)
            biases = np.row_stack((mum1arr, mup1arr)).T.astype(np.float64)
            var_m1 = np.exp(np.matmul(j_mat+jaddmat, q_mf*2-1) + b + biases)  #  + likelihood*stim_weight
            q_mf = q_mf + dt/tau*(var_m1.T / np.sum(var_m1, axis=1) - q_mf.T).T + np.random.randn(n_dots, nstates)*noise*np.sqrt(dt/tau)
            q_mf_arr[:, :, t] = q_mf
        if not plot:
            return q_mf
        if plot:
            time = np.arange(0, t_end, dt)
            fig, ax = plt.subplots(nrows=3, figsize=(6, 9))
            for i_a, a in enumerate(ax.flatten()):
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)
                if i_a < 3:
                    a.set_xticks([])
                if i_a > 0:
                    a.set_ylim(-0.15, 1.15)
            ax[2].set_xlabel('Time (s)')
            ax[0].imshow(stim.T, cmap='binary', aspect='auto', interpolation='none',
                         vmin=0, vmax=1)
            ax[0].set_ylabel('Stimulus')
            ax[1].set_ylabel('q(z_i=CW)')
            ax[2].set_ylabel('q(z_i=CCW)')
            ax[1].axhline(1/3, color='r', alpha=0.4, linestyle='--', linewidth=2)
            ax[1].text(t_end+70*dt, 1/3-0.02, '1/3', color='r')
            ax[1].text(t_end+70*dt, 1/2+0.02, '1/2', color='r')
            ax[1].axhline(1/2, color='r', alpha=0.4, linestyle=':', linewidth=2)
            ax[2].axhline(1/3, color='b', alpha=0.4, linestyle='--', linewidth=2)
            ax[2].axhline(1/2, color='b', alpha=0.4, linestyle=':', linewidth=2)
            if true == '2combination':
                title = r'$CW \longrightarrow NM \longrightarrow CW$'
            if true == 'combination':
                title = r'$CW \longrightarrow NM$'
            if true == 'combination_reverse':
                title = r'$NM \longrightarrow CW$'
            if true not in ['combination', '2combination', 'combination_reverse']:
                title = true
            ax[0].set_title(title)
            for dot in range(n_dots):
                if dot % 2 == 0:
                    linestyle = ':'
                else:
                    linestyle = 'solid'
                ax[1].plot(time, q_mf_arr[dot, 0, :], color='r', linewidth=2.5,
                           linestyle=linestyle)
                ax[2].plot(time, q_mf_arr[dot, 1, :], color='k', linewidth=2.5,
                           linestyle=linestyle)
            fig.suptitle(f'Coupling J = {j}', fontsize=16)
            plt.figure()
            plt.plot(np.max(np.abs(np.diff(stim.T, axis=0)), axis=0))
            # plt.plot(q_mf_arr[:, 0, :], q_mf_arr[:, 1, :], color='k', linewidth=2.5)
            plt.ylim(0, 2)


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
            colors = ['r', 'k', 'b']
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
            lab2 = 'CCW'
            for i in range(n_dots):
                ax.plot(np.arange(n_iters), q_mf_arr[i, 0, :], q_mf_arr[i, 1, :],
                        color='k', label=lab1)
                ax.plot(np.arange(n_iters), q_mf_arr[i, 0, :], q_mf_arr[i, 2, :],
                        color='b', label=lab2)
                if i == 0:
                    ax.legend()
            ax.set_xlabel('Iterations')
            ax.set_zlabel('q(z_i = CCW), q(z_i = NM)')
            ax.set_ylabel('q(z_i = CW)')


    def mean_field_fixed_points_vs_j_different_epsilons(self, epslist=[0.1, 0.01, 0.001],
                                                        j_list=np.arange(0, 2, 0.02),
                                                        true='CW'):
        
        ini_conds = [[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.1, 0.7, 0.2],
                      [0.3, 0.3, 0.4], [0.4, 0.3, 0.3], [0.8, 0.1, 0.1],
                      [0.1, 0.1, 0.8], [0.1, 0.2, 0.7], [0.7, 0.2, 0.1],
                      [0.2, 0.1, 0.7], [0.7, 0.1, 0.2], [0.3, 0.35, 0.35],
                      [0.35, 0.35, 0.3], [0.35, 0.3, 0.35], [0.5, 0.3, 0.2],
                      [0.1, 0.6, 0.3]]
        path = DATA_FOLDER + 'fixed_points_eps_jlist_'+ true + '_random_ini_conds_z_1.npy'
        ini_conds = ini_conds + [None]*100
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print('Computing fixed points')
        if true == 'none':
            stim_weight = 0
            epslist = [0.001]
        else:
            stim_weight = 1
        if os.path.exists(path):
            q_eps_jlist = np.load(path)
            j_list=np.arange(0, 2, 0.02)
            epslist=[0.1, 0.01, 0.001]
            if true == 'none':
                epslist = [0.001]
                true = 'NM'
        else:
            if true == 'none':
                stim_weight = 0
                epslist = [0.001]
                true = 'NM'
            else:
                stim_weight = 1
            q_eps_jlist = np.zeros((len(epslist), len(j_list), len(ini_conds), 3))
            for i_e, eps in enumerate(epslist):
                print(r' $\varepsilon$ = {}'.format(eps))
                self.eps = eps
                q_jlist = np.zeros((len(j_list), len(ini_conds), 3))
                for i_j, j in enumerate(j_list):
                    q_initializations = np.zeros((len(ini_conds), 3))
                    for initialization in range(len(ini_conds)):
                        q_mf = self.mean_field_sde(dt=0.01, tau=0.06, n_iters=600, j=j,
                                                   true=true, noise=0., plot=False,
                                                   ini_cond=ini_conds[initialization],
                                                   stim_weight=stim_weight)
                        # q_mf = np.round(np.nanmean(q_mf, axis=0), 8)
                        q_mf = np.round(q_mf[0], 8)
                        q_initializations[initialization, :] = q_mf
                    q_jlist[i_j, :] = q_initializations
                q_eps_jlist[i_e] = q_jlist
            np.save(path, q_eps_jlist)
        j_crit = 1/np.sum(self.exp_kernel())
        if len(epslist) == 1:
            figsize = (9, 5)
        else:
            figsize = (9, 10)
        fig, ax = plt.subplots(nrows=len(epslist), ncols=3, figsize=figsize)
        colors = ['r', 'k', 'b']
        for i_a, a in enumerate(ax.flatten()):
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            if len(epslist) > 1:
                if i_a < 6:
                    a.set_xticks([])
            a.set_ylim(-0.15, 1.15)
            # a.axvline(j_crit)
            if i_a in [0, 3, 6]:
                color = 'r'
            if i_a in [1, 4, 7]:
                color = 'k'
            if i_a in [2, 5, 8]:
                color = 'b'
            a.axhline(1/3, color=color, alpha=0.4, linestyle='--', linewidth=2)
            a.axhline(1/2, color=color, alpha=0.4, linestyle=':', linewidth=2)
            if len(epslist) == 1:
                a.axvline(j_crit, color='k', alpha=0.4, linewidth=2.5)
        for i_e, eps in enumerate(epslist):
            for state in range(3):
                # vals, idx = np.unique(q_eps_jlist[i_e, :, :, state], return_index=True)
                
                if len(epslist) == 1:
                    axis = ax[state]
                else:
                    axis = ax[i_e, state]
                axis.plot(j_list, q_eps_jlist[i_e, :, :, state], color=colors[state],
                          linewidth=2, marker='o', linestyle='', markersize=2)
        if len(epslist) == 1:
            ax[0].set_xlabel('Coupling, J')
            ax[2].set_xlabel('Coupling, J')
            ax[1].set_xlabel('Coupling, J')
            ax[0].set_ylabel(r'q(z_i), $\varepsilon$ = {}'.format(epslist[0]))
            ax[0].set_title('q(z_i=CW)', fontsize=17)
            ax[1].set_title('Stim: ' + true + '\nq(z_i=NM)', fontsize=17)
            ax[2].set_title('q(z_i=CCW)', fontsize=17)
        else:
            ax[2, 0].set_xlabel('Coupling, J')
            ax[2, 2].set_xlabel('Coupling, J')
            ax[2, 1].set_xlabel('Coupling, J')
            ax[0, 0].set_ylabel(r'q(z_i), $\varepsilon$ = {}'.format(epslist[0]))
            ax[1, 0].set_ylabel(r'q(z_i), $\varepsilon$ = {}'.format(epslist[1]))
            ax[2, 0].set_ylabel(r'q(z_i), $\varepsilon$ = {}'.format(epslist[2]))
            ax[0, 0].set_title('q(z_i=CW)', fontsize=17)
            ax[0, 1].set_title('Stim: ' + true + '\nq(z_i=NM)', fontsize=17)
            ax[0, 2].set_title('q(z_i=CCW)', fontsize=17)
        fig.tight_layout()


    def prob_nm_vs_max_difference_continuous_stim(self, nreps=100, resimulate=False):
        all_s = np.arange(0, 0.52, 2e-2)
        ss = [[all_s[i], 1-all_s[i]] for i in range(len(all_s))]
        # ss = [[0.5, 0.5]]
        # b_list = np.arange(0, 1.01, 0.02)
        true_diffs = np.diff(ss, axis=1)
        path = DATA_FOLDER + 'nm_vs_diff_vdef' + str(nreps) + '.npy'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not resimulate:
            q_mns_vs_diff = np.load(path)
        else:
            q_mns_vs_diff = np.zeros((len(ss), nreps))
            for i in range(len(ss)):
                for n in range(nreps):
                    q = ring(epsilon=0.001).mean_field_sde(dt=0.01, tau=0.1, n_iters=100, j=0.6,
                                                           true='CW', noise=0.01, plot=False,
                                                           discrete_stim=False, s=ss[i],
                                                           b=[0., 0., 0.], noise_stim=0.01,
                                                           ini_cond=np.ones(3))
                    q_nm = np.round(np.mean(q, axis=0), 6)[1]
                    q_mns_vs_diff[i, n] = q_nm
            np.save(path, q_mns_vs_diff)
        # plt.figure()
        # plt.imshow(q_mns_vs_diff, aspect='auto',
        #            extent=[np.min(b_list), np.max(b_list), np.min(true_diffs), np.max(true_diffs)])
        plt.figure()
        mn = np.mean(q_mns_vs_diff, axis=1)
        err = np.std(q_mns_vs_diff, axis=1)
        plt.plot(true_diffs, mn, color='k', linewidth=3)
        # plt.fill_between(true_diffs.T[0], mn-err, mn + err, color='k', alpha=0.4)
        for n in range(nreps):
            plt.plot(true_diffs, q_mns_vs_diff[:, n], color='k', linewidth=1.5,
                      alpha=0.2)
        plt.xlabel('Signal difference')
        plt.ylabel(r'$q(z_i = NM) = 1-q(z_i=CCW)-q(z_i=CW)$')
        

    def dummy_stim_creation(self, n_iters=100, timesteps_between=10, true='NM', coh=0):
        """
        Create stimulus given a 'true' structure. true is a string with 4 possible values:
        - 'CW': clockwise rotation
        - 'CCW': counterclockwise rotation (note that it is 100% bistable)
        - 'NM': not moving
        - 'combination': start with CW and then jump to NM
        - '2combination': start with CW and then jump to NM and jump again to CW
        - 'combination_reverse': start with NM and jump to CW
        """
        s_init = np.array([coh, 1, 0, 1-coh, coh, 1, 0, 1-coh])
        if true == 'NM':
            roll = 0
        if true == 'CW':
            roll = 1
        if true == 'CCW':
            roll = -1
        s = s_init
        sv = [s]
        for i in range(n_iters-1):
            if i != 0 and i % timesteps_between == 0:
                roll_n = roll
            else:
                if timesteps_between == 1:
                    roll_n = roll
                else:
                    roll_n = 0
            s = np.roll(s, roll_n)
            sv.append(s)
        return np.row_stack((sv)) 
        

def psychometric_curve_ring(dt=0.01, tau=0.1, n_iters=200, j_list=[0, 0.4, 0.8],
                            noise=0.01, cohlist=np.arange(0, 0.5, 1e-2),
                            nreps=50, true='CW', noise_stim=0.1):
    choice = np.zeros((len(cohlist), nreps, len(j_list)))
    choice2 = np.zeros((len(cohlist)-1, nreps, len(j_list)))
    ring_object = ring(epsilon=0.001, n_dots=8)
    for i_t, true in enumerate(['CCW', 'CW']):
        for i_j, j in enumerate(j_list):
            for ic, coh in enumerate(cohlist):
                if i_t == 1 and coh == 0:
                    continue
                print(coh)
                for n in range(nreps):
                    q = ring_object.mean_field_sde_ising(dt=dt, tau=tau, n_iters=n_iters, j=j,
                                                         true=true, noise=noise, plot=False,
                                                         discrete_stim=True, coh=coh,
                                                         stim_stamps=1, noise_stim=noise_stim)
                    # q_nm = np.mean(q, axis=0)[0]  # q[0][0]
                    q_nm = q[0][0]
                    if i_t == 0:
                        choice[ic, n, i_j] = (np.sign(q_nm-0.5)+1)/2
                    else:
                        choice2[ic-1, n, i_j] = (np.sign(q_nm-0.5)+1)/2
    fig, ax = plt.subplots(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    colormap = pl.cm.copper(np.linspace(0.2, 1, len(j_list)))
    ax.set_xlabel('a')
    # ax.set_ylabel('Approximate posterior q(z_i = CW)')
    ax.set_ylabel('p(CW report)')
    final_coh = np.concatenate((-cohlist[::-1], cohlist[1:]))
    for i_j, j in enumerate(j_list):
        mn = np.mean(choice[:, :, i_j], axis=1)
        err = np.std(choice[:, :, i_j], axis=1)/nreps
        mn2 = np.mean(choice2[:, :, i_j], axis=1)
        err2 = np.std(choice2[:, :, i_j], axis=1)/nreps
        mnfinal = np.concatenate((mn[::-1], mn2))
        errfinal = np.concatenate((err[::-1], err2))
        ax.plot(final_coh, mnfinal, color=colormap[i_j], marker='o', label=j)
        ax.fill_between(final_coh, mnfinal-errfinal, mnfinal+errfinal, color=colormap[i_j], alpha=0.6)
    plt.legend(title='Coupling, J', frameon=False)


def plot_all():
    # discrete stim
    ring(epsilon=0.001).mean_field_ring(true='combination', j=0.1, b=[0., 0., 0.], plot=True,
                                        n_iters=300, noise=0)
    ring(epsilon=0.001).mean_field_fixed_points_vs_j_different_epsilons(true='NM')
    ring(epsilon=0.001).mean_field_fixed_points_vs_j_different_epsilons(true='CW')
    ring(epsilon=0.001).mean_field_fixed_points_vs_j_different_epsilons(true='none')
    ring(epsilon=0.001).mean_field_sde(dt=0.01, tau=0.2, n_iters=1000, j=0.1,
                                       true='CW', noise=0., plot=True)
    ring(epsilon=0.001).mean_field_sde(dt=0.01, tau=0.2, n_iters=1000, j=0.7,
                                        true='combination_reverse', noise=0., plot=True)
    ring(epsilon=0.001).mean_field_sde(dt=0.01, tau=0.2, n_iters=1000, j=1.1,
                                        true='combination_reverse', noise=0., plot=True)
    ring(epsilon=0.001).mean_field_sde(dt=0.01, tau=0.2, n_iters=1000, j=1.4,
                                        true='combination_reverse', noise=0., plot=True)
    ring(epsilon=0.001).mean_field_sde(dt=0.01, tau=0.2, n_iters=1000, j=0.5,
                                        true='2combination', noise=0., plot=True)
    ring(epsilon=0.001).mean_field_sde(dt=0.01, tau=0.2, n_iters=1000, j=0.7,
                                        true='2combination', noise=0., plot=True)
    ring(epsilon=0.001).mean_field_sde(dt=0.01, tau=0.2, n_iters=1000, j=1.1,
                                        true='2combination', noise=0., plot=True)
    ring(epsilon=0.001).mean_field_sde(dt=0.01, tau=0.2, n_iters=1000, j=1.4,
                                        true='2combination', noise=0., plot=True)


def print_jstim_biases(include_m11=False, logm11=False, a=None, sigma=None):
    if a is None:
        a = sympy.symbols('a')
    if sigma is None:
        sigma = sympy.symbols('sigma')
    stim = [1-a, 1, a]
    nd = len(stim)
    asym = sympy.symbols('a')
    stimsym = [1-asym, 1, asym]
    for i in range(nd):
        i_prev = (i-1) % nd
        i_post = (i+1) % nd
        print('stim_i: ' + str([stimsym[i_prev], stimsym[i], stimsym[i_post]]))
        s_vec = {
            "s_im1": stim[i_prev],
            "s_i": 1,
            "s_ip1": stim[i_post]}
        s_hat_11 = s_vec['s_im1']
        s_hat_m1m1 = s_vec['s_ip1']
        s_hat_m11 = s_vec['s_i']
        s_hat_1m1 = (s_vec['s_ip1']+s_vec['s_im1'])/2
        si = s_vec['s_i']
        J_stim = 1/(4*2*sigma**2)*((si-s_hat_1m1)**2 - (si-s_hat_11)**2 - (si-s_hat_m1m1)**2)
        if include_m11:
            if logm11:
                J_stim += -np.log(logm11)/4
            else:
                J_stim +=  (si-s_hat_m11)**2 /(4*2*sigma**2)
        print('J_stim = ' + str(sympy.simplify(J_stim)))
    
        mum1 = 1/(4*2*sigma**2)*(-(si-s_hat_1m1)**2  - (si-s_hat_11)**2 + (si-s_hat_m1m1)**2)
        if include_m11:
            if logm11:
                mum1 += np.log(logm11)/4
            else:
                mum1 +=  (si-s_hat_m11)**2 /(4*2*sigma**2)
        print('mu_{i-1}(z_{i-1}=1) = ' + str(sympy.simplify(mum1)))
    
        mup1 = 1/(4*2*sigma**2)*(-(si-s_hat_1m1)**2  + (si-s_hat_11)**2 - (si-s_hat_m1m1)**2)
        if include_m11:
            if logm11:
                mup1 += np.log(logm11)/4
            else:
                mup1 +=  (si-s_hat_m11)**2 /(4*2*sigma**2)
        print('mu_{i+1}(z_{i+1}=-1) = ' + str(sympy.simplify(mup1)))


def compute_jstim_biases(s_t, s_tm1, sigma=0.1, include_m11=False, logm11=False):
    nd = len(s_t)
    jarr = []
    mum1arr = []
    mup1arr = []
    for i in range(nd):
        i_prev = (i-1) % nd
        i_post = (i+1) % nd
        s_vec = {
            "s_im1": s_tm1[i_prev],
            "s_i": s_t[i],
            "s_ip1": s_tm1[i_post]}
    
        s_hat_11 = s_vec['s_im1']
        s_hat_m1m1 = s_vec['s_ip1']
        s_hat_m11 = s_vec['s_i']
        s_hat_1m1 = (s_vec['s_ip1']+s_vec['s_im1'])/2
        si = s_vec['s_i']
        J_stim = 1/(4*2*sigma**2)*((si-s_hat_1m1)**2- (si-s_hat_11)**2 - (si-s_hat_m1m1)**2)
        if include_m11:
            if logm11:
                J_stim += -np.log(logm11)/4
            else:
                J_stim +=  (si-s_hat_m11)**2 /(4*2*sigma**2)
        jarr.append(J_stim)
    
        mum1 = 1/(4*2*sigma**2)*(-(si-s_hat_1m1)**2  - (si-s_hat_11)**2 + (si-s_hat_m1m1)**2)
        if include_m11:
            if logm11:
                mum1 += np.log(logm11)/4
            else:
                mum1 +=  (si-s_hat_m11)**2 /(4*2*sigma**2)
        mum1arr.append(mum1)
    
        mup1 = 1/(4*2*sigma**2)*(-(si-s_hat_1m1)**2  + (si-s_hat_11)**2 - (si-s_hat_m1m1)**2)
        if include_m11:
            if logm11:
                mup1 += np.log(logm11)/4
            else:
                mup1 +=  (si-s_hat_m11)**2 /(4*2*sigma**2)
        mup1arr.append(mup1)
    return jarr, mum1arr, mup1arr


def sigmoid(x):
    return 1/(1+np.exp(-x))


def plot_j_stim_biases_vs_a(cohlist=np.arange(0, 0.51, 1e-2).round(4),
                            sigmalist=np.arange(0.05, 0.2, 1e-3).round(4),
                            plot_matrix=True):
    true = 'CW'
    r = ring(n_dots=8)
    jlist = np.zeros((len(cohlist), len(sigmalist)))
    mm1list = np.zeros((len(cohlist), len(sigmalist)))
    mp1list = np.zeros((len(cohlist), len(sigmalist)))
    for i_s, sigma in enumerate(sigmalist):
        for icoh, coh in enumerate(cohlist):
            stim = r.dummy_stim_creation(n_iters=2, true=true, coh=coh,
                                         timesteps_between=1)
            j, mu_zm1, mu_zp1 = compute_jstim_biases(stim[1], stim[0],
                                                     sigma=sigma, include_m11=False,
                                                     logm11=1e-3)
            jlist[icoh, i_s] = j[0]
            mm1list[icoh, i_s] = mu_zm1[0]
            mp1list[icoh, i_s] = mu_zp1[0]
    labels = [r'Stim-induced coupling, $J_s$', r'CW bias, $\mu_i(z_{i-1}=CW)$',
              r'CCW bias, $\mu_i(z_{i+1}=CCW)$']
    if plot_matrix:
        fig, ax = plt.subplots(ncols=3, figsize=(16, 4))
        ax[1].set_title('True: ' + true, fontsize=15)
        norm = mpl.colors.TwoSlopeNorm(vmin=np.min(jlist), vcenter=0, vmax=np.max(jlist))
        im0 = ax[0].imshow(np.flipud(jlist.T), cmap='bwr', aspect='auto', norm=norm)
        plt.colorbar(im0, ax=ax[0], label=labels[0])
        im1 = ax[1].imshow(np.flipud(mm1list.T), cmap='Reds', aspect='auto')
        plt.colorbar(im1, ax=ax[1], label=labels[1])
        # norm = mpl.colors.TwoSlopeNorm(vmin=np.min(mp1list), vcenter=0, vmax=np.max(mp1list))
        im2 = ax[2].imshow(np.flipud(mp1list.T), cmap='Reds', aspect='auto')
        plt.colorbar(im2, ax=ax[2], label=labels[2])
        for a in ax:
            a.set_yticks([0, 50, 100, 150][::-1], sigmalist[np.arange(0, 160, 50)])
            a.set_xticks([0, 10, 20, 30, 40, 50], cohlist[np.arange(0, 60, 10)])
            a.set_xlabel('a')
            a.set_ylabel(r'$\sigma$')
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
    else:
        fig, ax = plt.subplots(ncols=4, figsize=(16.5, 3.5))
        colormap = pl.cm.Blues(np.linspace(0.2, 1, len(sigmalist)))
        for i, idx in enumerate([0, 50, 100, 150]):
            ax[0].plot(cohlist, jlist[:, idx], color=colormap[idx], linewidth=4,
                       label=sigmalist[idx])
            ax[1].plot(cohlist, mm1list[:, idx], color=colormap[idx], linewidth=4)
            ax[2].plot(cohlist, mp1list[:, idx], color=colormap[idx], linewidth=4)
            ax[3].plot(cohlist, mm1list[:, idx]-mp1list[:, idx], color=colormap[idx], linewidth=4)
        ax[0].legend(title=r'$\sigma$', frameon=False)
        labels = labels + [r'$\mu_i(z_{i-1}=CW)-\mu_i(z_{i+1}=CCW)$']
        for i_a, a in enumerate(ax):
            a.set_xlabel('a')
            a.set_ylabel(labels[i_a])
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
    fig.tight_layout()


if __name__ == '__main__':
    # ring().prob_nm_vs_max_difference_continuous_stim(nreps=10, resimulate=False)
    # ring(epsilon=0.001, n_dots=8).mean_field_ring(true='CW', j=0.4, b=[0., 0., 0.], plot=True,
    #                                               n_iters=100, noise=0)
    # ring(epsilon=0.001, n_dots=8).mean_field_ring(true='NM', j=0.4, b=[0., 0., 0.], plot=True,
    #                                               n_iters=100, noise=0)
    # ss = [[0.8, 0.2], [0.5, 0.5], [0.9, 0.9]]
    # for i in range(len(ss)):
    #     ring(epsilon=0.001).mean_field_sde(dt=0.01, tau=0.1, n_iters=200, j=0.38,
    #                                         true='CW', noise=0., plot=True,
    #                                         discrete_stim=False, s=ss[i],
    #                                         b=[0., 0., 0.], noise_stim=0.01)
    # for i in range(5):
    #     ring(epsilon=0.001, n_dots=8).mean_field_sde_ising(dt=0.01, tau=0.1, n_iters=100, j=0.,
    #                                                         true='CW', noise=0.1, plot=True,
    #                                                         discrete_stim=True, noise_stim=0.1, coh=0.,
    #                                                         stim_stamps=1)
    # ring(epsilon=0.001, n_dots=8).mean_field_sde(dt=0.01, tau=0.1, n_iters=200, j=0.5,
    #                                               true='CCW', noise=0.01, plot=True,
    #                                               discrete_stim=True, s=[0.55, 0.45],
    #                                               b=[0., 0., 0.], noise_stim=0.0, coh=0)
    # # # ring(epsilon=0.001).mean_field_sde(dt=0.01, tau=0.2, n_iters=1000, j=0.7,
    # # #                                     true='CW', noise=0.01, plot=True,
    # # #                                     discrete_stim=False, s=[0.9, 0.9],
    # # #                                     b=[0., 0., 0.], noise_stim=0.01)
    # r = ring(epsilon=0.001)
    # dt = 1e-2
    # n_iters= 100
    # # # scw = r.stim_creation_ou_process(true='CW', noise=0.01, s_init=[0.2, 0.8],
    # # #                                   n_iters=n_iters, dt=dt)
    # snm = r.stim_creation_ou_process(true='NM', noise=0.1, s_init=[0.8, 0.2],
    #                                   n_iters=n_iters, dt=dt)
    # scw = r.stim_creation_ou_process(true='CW', noise=0.1, s_init=[0.8, 0.2],
    #                                   n_iters=n_iters, dt=dt)
    # fig, ax = plt.subplots(nrows=2, figsize=(10, 5))
    # ax[0].imshow(snm.T, aspect='auto', cmap='binary', vmin=0, vmax=1)
    # ax[1].imshow(scw.T, aspect='auto', cmap='binary', vmin=0, vmax=1)
    # psychometric_curve_ring(dt=0.01, tau=0.1, n_iters=120, j=0.7,
    #                         true='CW', noise=0.1,
    #                         cohlist=np.arange(0, 0.6, 0.1),
    #                         nreps=5)
    psychometric_curve_ring(dt=0.01, tau=0.1, n_iters=120, j_list=[0., 4],
                            noise=0.1, cohlist=np.arange(0, 0.22, 2e-2),
                            nreps=1000, noise_stim=0.2)
