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

mpl.rcParams['font.size'] = 16
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize']= 14
plt.rcParams['ytick.labelsize']= 14


def compute_likelihood_vector(s, z, p_CW=0.999, p_NM=0.999, p_CCW=0.999, p_0=0.0001):
    """
    Compute the likelihood P(s_i = 1 | s, z) for all 6 positions.
    
    Parameters:
    - s: List or array of observed dot presences at time t-1 (length 6, values 0 or 1).
    - z: List or array of latent motion states at time t-1 (length 6, values -1 (CCW), 0 (NM), or 1 (CW)).
    - p_CW, p_NM, p_CCW, p_0: Probabilities defining likelihoods.
    
    Returns:
    - A numpy array of length 6 containing the likelihood for each position.
    """
    N = len(s)
    p_s = np.full(N, p_0)  # Initialize with baseline probability

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

    return p_s


def exp_kernel(x=np.arange(6), tau=0.8):
    kernel = np.concatenate((np.exp(-(x-1)[:len(x)//2]/tau), (np.exp(-x[:len(x)//2]/tau))[::-1]))
    kernel[0] = 0
    return kernel / np.max(kernel)


def kdelta(a, b):
    return 1 if a == b else 0


def compute_likelihood_contribution(s, z, s_prev, q_z_prev):
    """
    Compute the likelihood contribution using the CPT and expectation over latent states.

    Parameters:
    - s (float): Current value of the continuous contrast.
    - s_prev (float): Previous value of the continuous contrast (s_{t-1}).
    - q_z_prev (array-like): Belief on previous latent states probabilities (q_{z_{t-1}}).
    - cpt (2D array-like): Conditional Probability Table (CPT) for p(s_t | s_{t-1}, z_{t-1}).
      Each row corresponds to a state of s_{t-1}, and each column corresponds to a state of z_{t-1}.
      CPT should have the form: p(s_t | s_{t-1}, z_{t-1}).
      
    Returns:
    - likelihood_contribution (float): The computed likelihood contribution.
    """

    # Number of possible latent states (e.g., 3 for CW, NM, CCW)
    num_states_z = q_z_prev.shape[1]
    num_variables = q_z_prev.shape[0]
    likelihood_c_all = np.zeros((num_states_z))
    # Iterate over all possible latent states z_{t-1}
    for z_prev_i in range(num_states_z):
        likelihood_contribution = 0
        for i in range(num_variables):
            zn = np.copy(z)
            zn[i] = z_prev_i-1
            # Get the probability of the previous state z_{t-1} from q_z_prev
            q_z = q_z_prev[i, z_prev_i]
            
            # Get the CPT value for p(s_t | s_{t-1}, z_{t-1})
            # Assuming s_prev corresponds to the previous state of s_t
            p_s_given_z = compute_likelihood_vector(s_prev, zn)[i]  # CPT lookup based on s_{t-1} and z_{t-1}
            
            # Compute the contribution for this z_{t-1}^i
            likelihood_contribution += q_z * np.log(p_s_given_z)
        likelihood_c_all[z_prev_i] = likelihood_contribution
    return likelihood_c_all


def mean_field_ring(kernel=exp_kernel(), j=2, n_iters=150, nstates=3, b=np.zeros(3),
                    true='NM', n_dots=6, plot=False):
    # bifurcation at j ~ 0.314
    def stim_creation(s_init=[0, 1], n_iters=100, true='NM'):
        s_init = np.repeat(np.array(s_init).reshape(-1, 1), n_dots//2, axis=1).T.flatten()
        if true == 'NM':
            roll = 0
        if true == 'CW':
            roll = 1
        if true == 'CCW':
            roll = -1
        if true != 'combination':
            s = s_init
            sv = [s]
            for _ in range(n_iters-1):
                s = np.roll(s, roll)
                sv.append(s)
        else:
            s = s_init
            sv = [s]
            for t in range(n_iters-1):
                if t < n_iters // 2:
                    roll = 1
                else:
                    roll = 0
                s = np.roll(s, roll)
                sv.append(s)
        return np.row_stack((sv))
    crit_j = 1/np.sum(kernel)
    print(f'J* = {round(crit_j, 4)}')
    # init_s = np.random.choice([0, 1], 6)
    s = [0, 1]
    stim = stim_creation(s_init=s, n_iters=n_iters, true=true)
    s = np.repeat(np.array(s).reshape(-1, 1), n_dots//2, axis=1).T.flatten()
    # z = np.random.choice([-1, 0, 1], 6)
    z = np.repeat(np.array([1, 0]).reshape(-1, 1), n_dots//2, axis=1).T.flatten()
    q_mf = np.ones((n_dots, nstates))/3 + np.random.randn(3)*0.05
    # q_mf = np.repeat(np.array([[0.25], [0.25], [0.5]]), 6, axis=-1).T
    q_mf = (q_mf.T / np.sum(q_mf, axis=1)).T
    j_mat = circulant(kernel)*j
    np.fill_diagonal(j_mat, 0)
    q_mf_arr = np.zeros((n_dots, nstates, n_iters))
    q_mf_arr[:, :, 0] = q_mf
    s_arr = np.zeros((n_dots, n_iters))
    s_arr[:, 0] = s
    z_arr = np.zeros((n_dots, n_iters))
    z_arr[:, 0] = z
    for t in range(1, n_iters):
        # stim_likelihood_cw = compute_likelihood_vector(stim[t], np.ones(n_dots))
        # stim_likelihood_nm = compute_likelihood_vector(stim[t], np.zeros(n_dots))
        # stim_likelihood_ccw = compute_likelihood_vector(stim[t], -np.ones(n_dots))
        # st_lh = np.column_stack(([stim_likelihood_cw,
        #                           stim_likelihood_nm,
        #                           stim_likelihood_ccw]))
        stim_likelihood = compute_likelihood_vector(s, z)
        s = np.array([np.random.choice([0, 1], p=[1-stim_likelihood[a], stim_likelihood[a]]) for a in range(n_dots)])
        # if J*(2*Q-1), then it means repulsion between different z's
        # if J*Q, then it means just attraction to same
        # likelihood = compute_likelihood_contribution(s, z, s_arr[:, t-1], q_mf)
        mstim = np.mean(np.abs(stim-stim[t-1]))
        if mstim > 0:
            arr = np.array([0.4975, 0.005, 0.4975])   # prob z if the abs(diff of stim) is > 0
        if mstim == 0:
            arr = np.array([0.0025, 0.995, 0.0025])  # prob z if the difference of stim is 0
        # likelihood = np.sum(q_mf.T*np.log(stim_likelihood), axis=1)
        var_m1 = np.exp(np.matmul(j_mat, q_mf*2-1) + np.ones(3)*b + 10*arr)  # np.random.randn(6, 3)
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
        # fig, ax = plt.subplots(nrows=1)
        # ax.imshow(q_mf_arr[0], cmap='PRGn', extent=[0, n_iters, -3, 3],
        #           aspect='auto')
        # ax.set_yticks([-2, 0, 2], ['q(z_i=CCW)', 'q(z_i=NM)', 'q(z_i=CW)'])
        fig, ax = plt.subplots(nrows=4, figsize=(11, 16))
        title = ['CCW', 'NM', 'CW'][int(np.mean(z_arr.T[-1]))+1]
        ax[0].set_title('Percept: ' + title + ', True: ' + true)
        im1 = ax[0].imshow(z_arr, cmap='coolwarm', aspect='auto', interpolation='none',
                           vmin=-1, vmax=1)
        plt.colorbar(im1, ax=ax[0], label='sampled hidden state, z')
        # ax[0].set_title('Percept: ' + str(labels[int(np.mean(z_arr[:, :-10]))+2]))
        im2 = ax[1].imshow(stim.T, cmap='binary', aspect='auto', interpolation='none')
        plt.colorbar(im2, ax=ax[1], label='true stimulus, s')
        im2 = ax[2].imshow(s_arr, cmap='binary', aspect='auto', interpolation='none')
        plt.colorbar(im2, ax=ax[2], label='sampled (expected)\n stimulus, s*')
        im2 = ax[3].imshow(stim.T-s_arr, cmap='PiYG', aspect='auto', interpolation='none')
        plt.colorbar(im2, ax=ax[3], label='difference in stimulus')
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
    mean_field_ring(true='CW', j=2, b=[0., 0, 0], plot=True)

