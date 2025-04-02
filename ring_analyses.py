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
from scipy.optimize import root
import sympy
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


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


    def compute_likelihood_continuous_stim(self, s_t_1, z, s_t=np.ones(6), noise=0.1):
        phi = np.sum(1*(z[0] == 1)*s_t_1[0] + 1*(z[1] == 0)*s_t_1[1] + 1*(z[2] == -1)*s_t_1[2])
        norms = np.sum((z[0] == 1)*1 + 1*(z[1] == 0) + (z[2] == -1)*1)
        if norms != 0:
            likelihood = 1/(noise*np.sqrt(2*np.pi))*np.exp(-(s_t[1]-phi/norms)**2/noise**2/2)
        else:
            likelihood = self.eps
        return likelihood


    # def compute_likelihood_continuous_stim(self, s, z, s_t=np.ones(6), noise=0.1,
    #                                        epsilon=1e-2):
    #     phi = np.roll(s, -1)*(np.roll(z, -1) == -1) + np.roll(s, 1)*(np.roll(z, 1) == 1) + s*(z == 0)
    #     norms = 1*(np.roll(z, 1) == 1) + 1*(z==0) + 1*(np.roll(z, -1) == -1)
    #     phi[norms == 0] = penalization
    #     norms[norms == 0] = 1
    #     likelihood = 1/(noise*np.sqrt(2*np.pi))*np.exp(-((s_t-phi/norms)**2)/noise**2/2)
    #     # likelihood = scipy.stats.norm.pdf(s_t, phi/3, noise)
    #     # likelihood = np.min((likelihood, np.ones(len(likelihood))), axis=0)
    #     return likelihood+1e-12


    def compute_likelihood_continuous_stim_ising(self, s_t_1, z, s_t=np.ones(3), noise=0.1,
                                                 epsilon=1e-2):
        if z[0] == -1:
            if z[2] == 1:
                return np.log(epsilon)
            else:
                return -(s_t[1]-s_t_1[2])**2 / (2*noise**2)
        if z[0] == 1:
            if z[2] == 1:
                return -(s_t[1]-s_t_1[0])**2 / (2*noise**2)
            else:
                return -(s_t[1]-s_t_1[0]*0.5-s_t_1[2]*0.5)**2 / (2*noise**2)


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
            M = np.zeros((len(combinations), num_states_z))  # Ensure correct shape

            for z_i_index in range(num_states_z):  # Iterate over possible states of z_i (columns)
                for ic, (z_prev, z_next) in enumerate(combinations):  # Iterate over (z_{i-1}, z_{i+1}) pairs
                    zn = np.ones(num_variables)
                    i_prev = (i - 1) % num_variables  # Ensure correct wrapping
                    i_next = (i + 1) % num_variables  # Ensure correct wrapping
                    zn[i_prev] = z_prev
                    zn[i_next] = z_next
                    zn[i] = z_states[z_i_index]  # Assign z_i
                    idxs = [i_prev, i, i_next]
            
                    # Get the CPT value for p(s_i | s, z)
                    if discrete_stim:
                        p_s_given_z = self.compute_likelihood_vector(s, zn, s_t)[i]  # CPT lookup based on s and z, takes p(s_i | s, z)
                    else:
                        # based on a normal distribution centered in the expectation of the stimulus given the combination of z
                        p_s_given_z =\
                            self.compute_likelihood_continuous_stim(s[idxs], zn[idxs], s_t[idxs], noise=noise)
            
                    # Store log-likelihood in matrix with clipping for stability
                    M[ic, z_i_index] = np.log(np.clip(p_s_given_z, 1e-10, None))
            i_1prev = (i-1) % num_variables
            i_1next = (i+1) % num_variables
            q_z_1p = q_z_prev[i_1prev, :]
            q_z_1n = q_z_prev[i_1next, :]
            # Reshape M to a 3D tensor (z_{i-1}, z_i, z_{i+1})
            M_tensor = M.reshape(num_states_z, num_states_z, num_states_z)
            
            # Compute expectation using q distributions
            v = np.tensordot(q_z_1p, M_tensor, axes=(0, 0))  # Sum over z_{i-1}
            v = np.tensordot(v, q_z_1n, axes=(1, 0))  # Sum over z_{i+1}

            # Final expectation over q_i
            # expectation = v*q_z_prev[i, 0]
            # U = np.tensordot(q_z_1p, mat, axes=(0, 0))
            # V = np.tensordot(U, q_z_1n, axes=(1, 0))
            likelihood_c_all[i, :] = v
        return likelihood_c_all


    def compute_expectation_log_likelihood_original(self, s_t_1, q_z_prev, s_t, discrete_stim=True,
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
        # we want to compute E_{q_{i-1}, q_{i+1}}[\log p(s_i | z_{i-1}, z_i, z_{i+1}, s)]
        #  \sum_{z_{i-1}} \sum_{z_{i+1}} \log p(s_i^t | s_{i-1, i, i+1}^{t-1}, z_{i-1}^{t-1}, z_i^{t-1}, z_{i+1}^{t-1}) + other terms
        # with \sum_{z_{i}} \sum_{z_{i-2}} and \sum_{z_{i+2}} \sum_{z_{i}}
        # E_{q \ i}[\log p(s_i | z, s)] = likelihood_c_all: num_variables (i) x num_states_z (3)
        likelihood_c_all = np.zeros((num_variables, num_states_z))
        z_states = [1, 0, -1]
        combinations = list(itertools.product(z_states, z_states))
        idxmap = {-1:2, 0:1, 1:0}  # state to index mapping
        for i in range(num_variables):  # for all dots
            # Iterate over all possible latent states z_{t-1}
            for z_i_index in range(num_states_z):  # for each possible state of z_i (columns)
                likelihood_contribution = 0
                for startpoint in [-1, 0, 1]:  # to get i-2 and i+2
                    i_prev = (i+startpoint-1) % num_variables
                    i_next = (i+startpoint+1) % num_variables
                    idx = (i+startpoint) % num_variables
                    # iterate over all possible combinations of neighbors
                    for comb in combinations:  # for all combinations of z_{i-1}, z_{i+1}
                        if startpoint == -1:
                            zn = np.ones(num_variables)
                            zn[i_prev] = comb[0]  # z_{i-2}
                            zn[idx] = comb[1]  # z_{i-1}
                            zn[i_next] = z_states[z_i_index]  # z_i
                            q_z_p = q_z_prev[i_prev, idxmap[comb[0]]]  # extract q_{i-2}(z_{i-2}=comb[0])
                            q_z_n = q_z_prev[idx, idxmap[comb[1]]]  # extract q_{i-1}(z_{i-1}=comb[1])
                        if startpoint == 0:
                            zn = np.ones(num_variables)
                            zn[i_prev] = comb[0]  # z_{i-1}
                            zn[i_next] = comb[1]  # z_{i+1}
                            zn[idx] = z_states[z_i_index]  # z_i
                            q_z_p = q_z_prev[i_prev, idxmap[comb[0]]]  # extract q_{i-1}(z_{i-1}=comb[0])
                            q_z_n = q_z_prev[i_next, idxmap[comb[1]]]  # extract q_{i+1}(z_{i+1}=comb[1])
                        if startpoint == 1:
                            zn = np.ones(num_variables)
                            zn[idx] = comb[0]  # z_{i+1}
                            zn[i_next] = comb[1]  # z_{i+2}
                            zn[i_prev] = z_states[z_i_index]  # z_i
                            q_z_p = q_z_prev[idx, idxmap[comb[0]]]  # extract q_{i+1}(z_{i+1}=comb[0])
                            q_z_n = q_z_prev[i_next, idxmap[comb[1]]]  # extract q_{i+2}(z_{i+2}=comb[1])
                        idxs = [i_prev, idx, i_next]
                        # Get the probability of z from q_z_prev (approx. posterior)
                        # q_z_prev: num_variables x num_states_z (n_dots rows x 3 columns)
    
                        # Get the CPT value for p(s_i | s, z)
                        if discrete_stim:
                            p_s_given_z = self.compute_likelihood_vector(s_t_1, zn, s_t)[idx]  # CPT lookup based on s and z, takes p(s_i | s, z)
                        else:
                            # based on a normal distribution centered in the expectation of the stimulus given the combination of z
                            p_s_given_z =\
                                self.compute_likelihood_continuous_stim(s_t_1[idxs], zn[idxs], s_t[idxs], noise=noise)
                        # add p(s_i | z_{i-1}, z_i, z_{i+1})*q_{i-1}*q_{i+1}
                        likelihood_contribution += np.log(p_s_given_z + 1e-12)*q_z_p*q_z_n
                likelihood_c_all[i, z_i_index] = likelihood_contribution
        return likelihood_c_all


    def compute_expectation_log_likelihood_ising(self, s_t_1, q_z_prev, s_t, discrete_stim=True,
                                                 noise=0.1):
        # Number of possible latent states (e.g., 3 for CW, NM, CCW)
        num_states_z = q_z_prev.shape[1]
        num_variables = q_z_prev.shape[0]
        # we want to compute \sum_{i \in N(j)} E_{q_{i-1}, q_{i+1}}[\log p(s_i | z_{i-1}, z_i, z_{i+1}, s)]
        # \sum_{i \in N(j)} E_{q_{i-1}, q_{i+1}}[\log p(s_i | z, s)]: num_variables x num_states_z
        likelihood_c_all = np.zeros((num_variables, num_states_z))
        z_states = [-1, 1]
        # rows: i+1
        # columns: i-1
        combinations = [[1, 1], [1, -1],
                        [-1, 1], [-1, -1]]
        idxmap = {-1:1, 1:0}  # state to index mapping
        for i in range(num_variables):  # for all dots
            # Iterate over all possible latent states z_{t-1}
            lh = 0
            mat = np.zeros((2, 2)).flatten()
            for startpoint in [-1, 1]:
                i_prev = (i-1+startpoint) % num_variables
                i_next = (i+1+startpoint) % num_variables
                idx = (i) % num_variables
                for ic, comb in enumerate(combinations):  # for all combinations of z_{i-1}, z_{i+1}
                    if startpoint == -1:
                        zn = np.ones(num_variables)
                        zn[i_prev] = comb[0]  # z_{i-2}
                        zn[idx] = comb[1]  # z_{i-1}
                        q_z_p = q_z_prev[i_prev, idxmap[comb[0]]]  # extract q_{i-2}(z_{i-2}=comb[0])
                        q_z_n = q_z_prev[idx, idxmap[comb[1]]]  # extract q_{i-1}(z_{i-1}=comb[1])
                    if startpoint == 1:
                        zn = np.ones(num_variables)
                        zn[idx] = comb[0]  # z_{i+1}
                        zn[i_next] = comb[1]  # z_{i+2}
                        q_z_p = q_z_prev[idx, idxmap[comb[0]]]  # extract q_{i+1}(z_{i+1}=comb[0])
                        q_z_n = q_z_prev[i_next, idxmap[comb[1]]]  # extract q_{i+2}(z_{i+2}=comb[1])
                    zn = np.ones(num_variables)
                    zn[i_prev] = comb[0]  # z_{i-1}
                    zn[i_next] = comb[1]  # z_{i+1}
                    idxs = [i_prev, i, i_next]
                    # Get the probability of z from q_z_prev (approx. posterior)
                    # q_z_prev: num_variables x num_states_z (n_dots rows x 3 columns)
    
                    # Get the CPT value for p(s_i | s, z)
                    if discrete_stim:
                        p_s_given_z = self.compute_likelihood_vector(s_t_1, zn, s_t)[idx]  # CPT lookup based on s and z, takes p(s_i | s, z)
                    else:
                        # based on a normal distribution centered in the expectation of the stimulus given the combination of z
                        p_s_given_z = self.compute_likelihood_continuous_stim_ising(s_t_1[idxs], zn[idxs],
                                                                                    s_t[idxs], noise=noise)
                    mat[ic] = p_s_given_z
            mat = mat.reshape((2, 2))
            i_prev = (i-2) % num_variables
            i_next = (i+2) % num_variables
            q_z_p = q_z_prev[i_prev, :]
            q_z_n = q_z_prev[i_next, :]
            likelihood_c_all[i, :] = mat @ q_z_p + mat.T @ q_z_n
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
                       discrete_stim=True, s=[1, 0], noise_stim=0.05, coh=None,
                       stim_stamps=1, noise_gaussian=0.1):
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
        if ini_cond is None:
            q_mf = np.random.rand(n_dots, nstates)
        else:
            ini_cond = np.array(ini_cond)
            q_mf = np.repeat(ini_cond.reshape(-1, 1), self.ndots, axis=1).T
        q_mf = (q_mf.T / np.sum(q_mf, axis=1)).T
        j_mat = circulant(kernel)*j
        np.fill_diagonal(j_mat, 0)
        q_mf_arr = np.zeros((n_dots, nstates, n_iters))
        q_mf_arr[:, :, 0] = q_mf
        s_arr = np.zeros((n_dots, n_iters))
        s_arr[:, 0] = s
        for t in range(1, n_iters):
            # if J*(2*Q-1), then it means repulsion between different z's, i.e. 2\delta(z_i, z_j) - 1
            # if J*Q, then it means just attraction to same, i.e. \delta(z_i, z_j)
            if nstates == 2:
                likelihood = self.compute_expectation_log_likelihood_ising(stim[t-1], q_mf, stim[t],
                                                                           discrete_stim=discrete_stim,
                                                                           noise=noise_stim)
            if nstates > 2:
                likelihood = self.compute_expectation_log_likelihood_original(stim[t-1], q_mf, stim[t],
                                                                             discrete_stim=discrete_stim,
                                                                             noise=noise_gaussian)
            var_m1 = np.exp(np.matmul(j_mat, q_mf*2-1) + b + likelihood*stim_weight)
            q_mf = q_mf + dt/tau*(var_m1.T / np.sum(var_m1, axis=1) - q_mf.T).T + np.random.randn(n_dots, nstates)*noise*np.sqrt(dt/tau)
            q_mf_arr[:, :, t] = q_mf
        if not plot:
            return q_mf
        if plot:
            time = np.arange(0, t_end, dt)
            fig, ax = plt.subplots(nrows=nstates+2, figsize=(8, 12))
            for i_a, a in enumerate(ax.flatten()):
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)
                if i_a < 3:
                    a.set_xticks([])
                if i_a > 1:
                    a.set_ylim(-0.15, 1.15)
            ax[nstates].set_xlabel('Time (s)')
            ax[0].imshow(stim.T, cmap='binary', aspect='auto', interpolation='none',
                         vmin=0, vmax=1)
            ax[0].set_ylabel('Stimulus')
            ax[1].set_ylabel('R=q(z_i=CW), G=0,\n B=q(z_i=CCW)' )
            q_mf_plot = np.array((q_mf_arr[:, 0, :].T, q_mf_arr[:, 1, :].T,
                                  q_mf_arr[:, 2, :].T)).T
            q_mf_plot[:, :, 1] = 0
            ax[1].imshow(q_mf_plot, aspect='auto', interpolation='none',
                         vmin=0, vmax=1)
            ax[2].set_ylabel('q(z_i=CW)')
            ax[3].set_ylabel('q(z_i=NM)')
            if nstates == 3:
                ax[4].set_ylabel('q(z_i=CCW)')
                ax[4].axhline(1/3, color='b', alpha=0.4, linestyle='--', linewidth=2)
                ax[4].axhline(1/2, color='b', alpha=0.4, linestyle=':', linewidth=2)
                ax[2].axhline(1/3, color='r', alpha=0.4, linestyle='--', linewidth=2)
                ax[2].text(t_end+n_iters/20*dt, 1/3-0.02, '1/3', color='r')
                ax[2].text(t_end+n_iters/20*dt, 1/2+0.02, '1/2', color='r')
                ax[3].axhline(1/3, color='k', alpha=0.4, linestyle='--', linewidth=2)
            ax[2].axhline(1/2, color='r', alpha=0.4, linestyle=':', linewidth=2)
            ax[3].axhline(1/2, color='k', alpha=0.4, linestyle=':', linewidth=2)
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
                ax[2].plot(time, q_mf_arr[dot, 0, :], color='r', linewidth=2.5,
                           linestyle=linestyle)
                ax[3].plot(time, q_mf_arr[dot, 1, :], color='k', linewidth=2.5,
                           linestyle=linestyle)
                if nstates == 3:
                    ax[4].plot(time, q_mf_arr[dot, 2, :], color='b', linewidth=2.5,
                               linestyle=linestyle)
            fig.suptitle(f'Coupling J = {j}', fontsize=16)
            # plt.figure()
            # plt.plot(np.max(np.abs(np.diff(stim.T, axis=0)), axis=0))
            # plt.ylim(0, 2)
    
    
    def mean_field_sde_ising(self, dt=0.001, tau=1, n_iters=100, j=2, nstates=2, b=np.zeros(2),
                            true='NM', noise=0.2, plot=False, stim_weight=1, ini_cond=None,
                            discrete_stim=True, s=[1, 0], noise_stim=0.05, coh=None,
                            stim_stamps=50, sigma_lh=0.1):
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
            if t % stim_stamps == 1 and t > stim_stamps or stim_stamps == 1:
                jarr, mum1arr, mup1arr = compute_jstim_biases(stim[t], stim[t-1], sigma=sigma_lh, epsilon=self.eps)
                jaddmat = (np.roll(np.eye(self.ndots), -2, axis=0) + np.roll(np.eye(self.ndots), 2, axis=0))*np.array(jarr, dtype=np.float64)
                biases = np.row_stack((mum1arr, mup1arr)).T.astype(np.float64)
            if t <= stim_stamps:
                biases = 0
                jaddmat = 0
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


    def fractional_belief_propagation_ising(self, j, b=0, noise=0, n_iters=200, dt=0.01,
                                            tau=1, discrete_stim=True, s=[1, 0], noise_stim=0.05, coh=None,
                                            stim_stamps=50, sigma_lh=0.1, true='CW', plot=True):
        t_end = n_iters*dt
        kernel = self.exp_kernel()
        n_dots = self.ndots
        q_y_1 = np.zeros((n_iters, n_dots))
        q_y_neg1 = np.zeros((n_iters, n_dots))
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
        j_mat = circulant(kernel)*j
        theta = j_mat*1
        np.fill_diagonal(j_mat, 0)
        mu_y_1 = np.multiply(np.ones((n_dots, n_dots)), np.random.rand(theta.shape[0], theta.shape[1]))*0.5
        mu_y_neg1 = np.multiply(np.ones((n_dots, n_dots)), np.random.rand(theta.shape[0], theta.shape[1]))*0.5
        for t in range(1, n_iters):
            jarr, mum1arr, mup1arr = compute_jstim_biases(stim[t], stim[t-1], sigma=sigma_lh, epsilon=self.eps)
            jaddmat = (np.roll(np.eye(self.ndots), -2, axis=0) + np.roll(np.eye(self.ndots), 2, axis=0))*np.array(jarr, dtype=np.float64)
            biases = np.row_stack((mum1arr, mup1arr)).astype(np.float64)
            theta = j_mat + jaddmat
            for i in range(theta.shape[0]):
                q1 = np.prod(mu_y_1[np.where(theta[:, i] != 0), i]) * np.exp(biases[0][i])
                qn1 = np.prod(mu_y_neg1[np.where(theta[:, i] != 0), i]) * np.exp(biases[1][i])
                q_y_1[t, i] = q1/(q1+qn1)
                q_y_neg1[t, i] = qn1/(q1+qn1)
            for i in range(theta.shape[0]):
                for m in np.where(theta[i, :] != 0)[0]:
                    # positive y_i
                    mu_y_1[m, i] += (np.exp(theta[i, m]+biases[0][m]) *\
                            np.prod(mu_y_1[jneigbours(m, i, theta=theta), m])\
                            + np.exp(-theta[i, m]+biases[1][m]) *\
                            np.prod(mu_y_neg1[jneigbours(m, i, theta=theta), m]) -
                            mu_y_1[m, i])*dt/tau +\
                        np.sqrt(dt/tau)*noise*np.random.randn()
                    # negative y_i
                    mu_y_neg1[m, i] += (np.exp(-theta[i, m]+biases[0][m]) *\
                        np.prod(mu_y_1[jneigbours(m, i, theta=theta), m])\
                        + np.exp(theta[i, m]+biases[1][m]) *\
                        np.prod(mu_y_neg1[jneigbours(m, i, theta=theta), m])-
                        mu_y_neg1[m, i])*dt/tau +\
                    np.sqrt(dt/tau)*noise*np.random.randn()
                    m_y_1_memory = np.copy(mu_y_1[m, i])
                    mu_y_1[m, i] = mu_y_1[m, i]/(m_y_1_memory+mu_y_neg1[m, i])
                    mu_y_neg1[m, i] = mu_y_neg1[m, i]/(m_y_1_memory+mu_y_neg1[m, i])
        if not plot:
            return q_y_1[-1]
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
                ax[1].plot(time, q_y_1[:, dot], color='r', linewidth=2.5,
                           linestyle=linestyle)
                ax[2].plot(time, q_y_neg1[:, dot], color='k', linewidth=2.5,
                           linestyle=linestyle)
            fig.suptitle(f'Coupling J = {j}', fontsize=16)
            plt.figure()
            plt.plot(np.max(np.abs(np.diff(stim.T, axis=0)), axis=0))
            # plt.plot(q_mf_arr[:, 0, :], q_mf_arr[:, 1, :], color='k', linewidth=2.5)
            plt.ylim(0, 2)


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


def bifurcations_different_bias(eps=0.1, nreps=50):
    all_s = np.arange(0.25, 0.501, 1e-2).round(4)
    ss = np.array([[all_s[i], 1-all_s[i]] for i in range(len(all_s))]).round(4)
    true_diffs = np.diff(ss, axis=1)
    colormap = pl.cm.Blues(np.linspace(0.2, 1, 3))
    filenames = ['bifurcations_difference_stim_def_eps01_bias_025.npy',
                 'bifurcations_difference_stim_def_eps01_bias.npy',
                 'bifurcations_difference_stim_def_eps01_bias_075.npy']
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(4, 12))
    biases = [0.25, 0.5, 0.75]
    col_labels = [r'$q(z_i = CW)$', r'$ q(z_i=NM)$', r'$q(z_i=CCW)$']
    for i_f, file in enumerate(filenames):
        path = DATA_FOLDER + file
        data = np.load(path)
        for row in range(3):
            for n in range(nreps):
                ax[row].plot(true_diffs, data[:, 0, n, row], color=colormap[i_f],
                             marker='o', linestyle='', markersize=2, label=biases[i_f])
            ax[row].set_xlabel('Signal difference')
            ax[row].set_ylabel(col_labels[row])
    ax[0].set_title(fr'$\varepsilon = ${eps}')
    legendelements = [Line2D([0], [0], color=colormap[0], lw=2, label=biases[0], marker='o'),
                      Line2D([0], [0], color=colormap[1], lw=2, label=biases[1], marker='o'),
                      Line2D([0], [0], color=colormap[2], lw=2, label=biases[2], marker='o')]
    ax[0].legend(title='NM prior', frameon=False, handles=legendelements)
    for a in ax.flatten():
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.set_ylim(-0.02, 1.02)
    fig.tight_layout()


def plot_bifurcations_all_bias(eps=0.1, nreps=30, biases=np.arange(0., 0.6, 0.1).round(4),
                               smallds=False, j=0):
    if smallds:
        ds = 1e-3
        if j == 0:
            extra_lab = '_small_ds'
        else:
            extra_lab = '_small_ds_coupling_' + str(j)
    else:
        ds = 2.5e-3
        if j == 0:
            extra_lab = ''
        else:
            extra_lab = '_coupling_' + str(j)
    all_s = np.arange(0.25, 0.502, ds).round(4)
    ss = np.array([[all_s[i], 1-all_s[i]] for i in range(len(all_s))]).round(4)
    true_diffs = np.diff(ss, axis=1)
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4.))
    colormap = pl.cm.Blues(np.linspace(0.2, 1, len(biases)))
    col_labels = [r'$q(z_i = CW)$', r'$ q(z_i=NM)$', r'$q(z_i=CCW)$']
    for i_b, biasval in enumerate(biases):
        path = DATA_FOLDER + 'bifurcations_difference_stim_def_eps01_bias_' + str(biasval) + extra_lab + '.npy'
        data = np.load(path)
        for row in range(3):
            for n in range(nreps):
                ax[row].plot(true_diffs, data[:, 0, n, row], color=colormap[i_b],
                             marker='o', linestyle='', markersize=2)
            ax[row].set_xlabel('Signal difference')
            ax[row].set_ylabel(col_labels[row])
            # noise = 0.1
            # d_sq = -2*noise**2*np.log(noise*eps*np.sqrt(2*np.pi))
            # biasterm = biasval*2*noise**2
            # dist_sq = np.sqrt((2*noise**2 * np.log(2) + biasterm + d_sq*0.6944444444444444)/1.777777777777778)
            # ax[row].axvline(dist_sq, color=colormap[i_b])
    ax[0].set_title(fr'$\varepsilon = ${eps}')
    ax[1].set_title(fr'Coupling $ J = ${j}')
    legendelements = [Line2D([0], [0], color=colormap[i], lw=2, label=biases[i], marker='o', linestyle='')
                      for i in range(len(biases))]
    ax[0].legend(title='NM prior', frameon=False, handles=legendelements)
    for a in ax.flatten():
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.set_ylim(-0.02, 1.02)
    fig.tight_layout()


def bifurcations_difference_stim_epsilon(nreps=100, resimulate=False, epslist=[0.1],
                                         biasval=0, j=0, smallds=False):
    if smallds:
        ds = 1e-3
        if j == 0:
            extra_lab = '_small_ds'
        else:
            extra_lab = '_small_ds_coupling_' + str(j)
    else:
        ds = 2.5e-3
        if j == 0:
            extra_lab = ''
        else:
            extra_lab = '_coupling_' + str(j)
    all_s = np.arange(0.25, 0.502, ds).round(4)
    ss = np.array([[all_s[i], 1-all_s[i]] for i in range(len(all_s))]).round(4)
    true_diffs = np.diff(ss, axis=1)
    path = DATA_FOLDER + 'bifurcations_difference_stim_def_eps01_bias_' + str(biasval) + extra_lab + '.npy'
    print(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not resimulate:
        q_mns_vs_diff = np.load(path)
    else:
        q_mns_vs_diff = np.zeros((len(ss), len(epslist), nreps, 3))
        for i_e, eps in enumerate(epslist):
            print('Epsilon: ' + str(eps))
            for i in range(len(ss)):
                if i % 10 == 0:
                    print(str(i/len(ss)*100) + ' %')
                for n in range(nreps):
                    q = ring(epsilon=eps).mean_field_sde(dt=0.1, tau=0.2, n_iters=210, j=j,
                                                         true='CW', noise=0., plot=False,
                                                         discrete_stim=False, s=ss[i],
                                                         b=[0., biasval, 0.], noise_stim=0.,
                                                         noise_gaussian=0.1)
                    q_mns_vs_diff[i, i_e, n, :] = q[1]
        np.save(path, q_mns_vs_diff)
    fig, ax = plt.subplots(nrows=3, ncols=len(epslist), figsize=(len(epslist)*4, 12))
    col_labels = [r'$q(z_i = CW)$', r'$ q(z_i=NM)$', r'$q(z_i=CCW)$']
    for i_e, eps in enumerate(epslist):
        if len(epslist) > 2:
            for row in range(3):
                for n in range(nreps):
                    ax[row, i_e].plot(true_diffs, q_mns_vs_diff[:, i_e, n, row], color='k',
                                      marker='o', linestyle='', markersize=1)
                ax[row, i_e].set_xlabel('Signal difference')
                ax[row, i_e].set_ylabel(col_labels[row])
            ax[0, i_e].set_title(fr'$\varepsilon = ${eps}')
        else:
            for row in range(3):
                for n in range(nreps):
                    ax[row].plot(true_diffs, q_mns_vs_diff[:, i_e, n, row], color='k',
                                 marker='o', linestyle='', markersize=1)
                ax[row].set_xlabel('Signal difference')
                ax[row].set_ylabel(col_labels[row])
        ax[0].set_title(fr'$\varepsilon = ${eps}')
    for a in ax.flatten():
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.set_ylim(-0.02, 1.02)
    fig.tight_layout()


def n_objects_ring_stim_epsilon(nreps=100, resimulate=False):
    all_s = np.arange(0, 0.2501, 1e-2).round(4)
    epslist = np.logspace(-5, -1, 25)
    ss = np.array([[all_s[i], 1-all_s[i]] for i in range(len(all_s))]).round(4)
    true_diffs = np.diff(ss, axis=1)
    path = DATA_FOLDER + 'n_objects_ring_stim_eps.npy'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not resimulate:
        nfps = np.load(path)
    else:
        nfps = np.zeros((len(epslist), len(all_s), 3))
        for i_e, eps in enumerate(epslist):
            print('Epsilon: ' + str(eps))
            for i in range(len(ss)):
                for n in range(nreps):
                    q = ring(epsilon=eps).mean_field_sde(dt=0.1, tau=0.2, n_iters=110, j=0.,
                                                         true='CW', noise=0., plot=False,
                                                         discrete_stim=False, s=ss[i],
                                                         b=[0., 0., 0.], noise_stim=0.)
                    for v in range(3):
                        nvals = len(np.unique(q[:, v].round(4)))
                        nfps[i_e, i, v] += nvals
        nfps = nfps/nreps
        np.save(path, nfps)
    fig, ax = plt.subplots(ncols=3, figsize=(14, 5))
    titles = [r'$q(z_i = CW)$', r'$ q(z_i=NM)$', r'$q(z_i=CCW)$']
    X, Y = np.meshgrid(true_diffs, epslist)
    for v in range(3):
        im = ax[v].pcolormesh(X, Y, np.flipud((nfps[:, :, v])), cmap='gist_stern',
                              vmin=1, vmax=8)
        ax[v].set_yscale('log')
        ax[v].set_title(titles[v])
        ax[v].set_xlabel('a')
        ax[v].set_ylabel('Epsilon')
        plt.colorbar(im, ax=ax[v], label='Average # FPs')
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'number_fixed_points_per_simulation_ring_stim_eps.png', dpi=200,
                bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'number_fixed_points_per_simulation_stim_eps.svg', dpi=200,
                bbox_inches='tight')


def n_objects_ring_stim_eps01_bias(nreps=100, resimulate=False, eps=0.1):
    all_s = np.arange(0.25, 0.501, 1e-2).round(4)
    biaslist = np.arange(0, 1.04, 4e-2)
    ss = np.array([[all_s[i], 1-all_s[i]] for i in range(len(all_s))]).round(4)
    true_diffs = np.diff(ss, axis=1)
    path = DATA_FOLDER + 'n_objects_ring_stim_blist.npy'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not resimulate:
        nfps = np.load(path)
    else:
        nfps = np.zeros((len(biaslist), len(all_s), 3))
        for i_b, bias in enumerate(biaslist):
            print('Bias: ' + str(bias))
            for i in range(len(ss)):
                for n in range(nreps):
                    q = ring(epsilon=eps).mean_field_sde(dt=0.1, tau=0.2, n_iters=150, j=0.,
                                                         true='CW', noise=0., plot=False,
                                                         discrete_stim=False, s=ss[i],
                                                         b=[0., bias, 0.], noise_stim=0.)
                    for v in range(3):
                        nvals = len(np.unique(q[:, v].round(4)))
                        nfps[i_b, i, v] += nvals
        nfps = nfps/nreps
        np.save(path, nfps)
    fig, ax = plt.subplots(ncols=3, figsize=(14, 5))
    titles = [r'$q(z_i = CW)$', r'$ q(z_i=NM)$', r'$q(z_i=CCW)$']
    X, Y = np.meshgrid(biaslist, true_diffs)
    for v in range(3):
        im = ax[v].imshow(np.fliplr(np.flipud(nfps[:, :, v])), cmap='gist_stern',
                          vmin=1, vmax=8, extent=[np.min(true_diffs), np.max(true_diffs),
                                                  np.min(biaslist), np.max(biaslist)],
                          aspect='auto')
        ax[v].set_title(titles[v])
        ax[v].set_xlabel('Signal difference')
        ax[v].set_ylabel('Bias towards NM')
        plt.colorbar(im, ax=ax[v], label='Average # FPs')
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'number_fixed_points_per_simulation_ring_stim_bias.png', dpi=200,
                bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'number_fixed_points_per_simulation_stim_bias.svg', dpi=200,
                bbox_inches='tight')


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


def compute_jstim_biases(s_t, s_tm1, sigma=0.1, epsilon=0.1):
    nd = len(s_t)
    jarr = []
    mum1arr = []
    mup1arr = []
    logm11 = -np.log(epsilon)
    for i in range(nd):
        i_prev = (i-1) % nd
        i_post = (i+1) % nd
        s_vec = {
            "s_im1": s_tm1[i_prev],
            "s_i": s_t[i],
            "s_ip1": s_tm1[i_post]}
        s_hat_11 = s_vec['s_im1']
        s_hat_m1m1 = s_vec['s_ip1']
        # s_hat_m11 = s_vec['s_i']
        s_hat_1m1 = (s_vec['s_ip1']+s_vec['s_im1'])/2
        si = s_vec['s_i']
        J_stim = 1/(4*2*sigma**2)*((si-s_hat_1m1)**2- (si-s_hat_11)**2 - (si-s_hat_m1m1)**2) + logm11/4
        jarr.append(J_stim)
    
        mum1 = 1/(4*2*sigma**2)*(-(si-s_hat_1m1)**2  - (si-s_hat_11)**2 + (si-s_hat_m1m1)**2)+ logm11/4
        mum1arr.append(mum1)
    
        mup1 = 1/(4*2*sigma**2)*(-(si-s_hat_1m1)**2  + (si-s_hat_11)**2 - (si-s_hat_m1m1)**2) + logm11/4
        mup1arr.append(mup1)
    return jarr, mum1arr, mup1arr


def sigmoid(x):
    return 1/(1+np.exp(-x))


def plot_j_stim_biases_vs_a(cohlist=np.arange(0, 0.51, 1e-2).round(4),
                            sigmalist=np.arange(0.05, 0.2, 1e-3).round(4),
                            plot_matrix=False, epsilon=0.2):
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
                                                     sigma=sigma, epsilon=0.2)
            jlist[icoh, i_s] = -3*coh**2/(32*sigma**2) - np.log(epsilon)/4
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
        ax[0].axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=3)
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


def sols_vs_a_j0(alist=np.arange(0, 0.5, 1e-2),
                 nreps=10, dt=0.01, tau=0.1, n_iters=300, true='CW', noise_stim=0.1):
    ring_object = ring(epsilon=0.001, n_dots=8)
    j = 0
    sols = np.zeros((len(alist), nreps, 8))
    for icoh, coh in enumerate(alist):
        print(coh)
        for n in range(nreps):
            q = ring_object.mean_field_sde_ising(dt=dt, tau=tau, n_iters=n_iters, j=j,
                                                 true=true, noise=0, plot=False,
                                                 discrete_stim=True, coh=coh,
                                                 stim_stamps=1, noise_stim=noise_stim)
            sols[icoh, n, :] = q[0, 0].round(6)
    fig, ax = plt.subplots(1)
    for dot in range(1):
        # if dot % 2 == 0:
        #     marker = 'o'
        # else:
        #     marker = 'x'
        for i_coh, coh in enumerate(alist):
            unique_vals = np.unique(sols[i_coh, :, dot])
            ax.plot(np.repeat(coh, len(unique_vals)),
                    unique_vals, color='k', marker='o', linestyle='',
                    markersize=3)
        # for n in range(nreps):
        #     ax.plot(alist, sols[:, n, dot], color='k', marker='o', linestyle='',
        #             markersize=3)
    ax.set_xlabel('a')
    ax.set_ylabel('q(z_i = CW)')


def sols_vs_j_cond_on_a(alist=[0, 0.05, 0.1, 0.2], j_list=np.arange(-1, 1.02, 4e-2).round(5),
                        nreps=50, dt=0.01, tau=0.1, n_iters=250, true='CW', noise_stim=0.1,
                        eps=0.2, sigma=0.2):
    ring_object = ring(epsilon=eps, n_dots=8)
    j = 0
    sols = np.zeros((len(alist), len(j_list), nreps))
    for icoh, coh in enumerate(alist):
        print(coh)
        for i_j, j in enumerate(j_list):
            for n in range(nreps):
                q = ring_object.mean_field_sde_ising(dt=dt, tau=tau, n_iters=n_iters, j=j,
                                                     true=true, noise=0, plot=False,
                                                     discrete_stim=True, coh=coh,
                                                     stim_stamps=1, noise_stim=noise_stim,
                                                     sigma_lh=sigma)
                sols[icoh, i_j, n] = q[0, 0].round(6)
    fig, ax = plt.subplots(ncols=len(alist), figsize=(len(alist)*4, 4))
    d0 = np.sqrt(-2*sigma**2 * np.log(np.sqrt(2*np.pi)*eps*sigma))
    jcrit = 0.25*(4-d0**2/sigma**2)
    ax[0].axvline(jcrit)
    ax[0].axvline(-jcrit)
    # colormap = pl.cm.copper(np.linspace(0.2, 1, len(alist)))
    # legendelements = [Line2D([0], [0], color=colormap[0], lw=2, label=alist[0], marker='o'),
    #                   Line2D([0], [0], color=colormap[1], lw=2, label=alist[1], marker='o'),
    #                   Line2D([0], [0], color=colormap[2], lw=2, label=alist[2], marker='o')]
    for icoh, coh in enumerate(alist):
        for i_j, j in enumerate(j_list):
            # js = -coh**2/(16*noise_stim**2) - np.log(6e-2)/4
            # ax[icoh].axvline(-js, color='r', linestyle='--', linewidth=2, alpha=0.4)
            unique_vals = np.unique(sols[icoh, i_j])
            ax[icoh].plot(np.repeat(j, len(unique_vals)),
                          unique_vals, color='k', marker='o', linestyle='',
                          markersize=3)
        ax[icoh].set_title('a = ' + str(coh))
        ax[icoh].set_xlabel('Coupling, J')
        ax[icoh].set_ylim(-0.05, 1.05)
    # ax.legend(handles=legendelements, title='a', frameon=False)
    ax[0].set_ylabel('q(z_i = CW)')
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'solutions_MF_ising_ring_diff_a_j_eps02.png', dpi=200)
    fig.savefig(DATA_FOLDER + 'solutions_MF_ising_ring_diff_a_j_eps02.svg', dpi=200)


def sols_vs_j_cond_on_a_beleif_prop(alist=[0, 0.05, 0.1, 0.2], j_list=np.arange(0, 1.02, 2e-2).round(5),
                                    nreps=50, dt=0.01, tau=0.1, n_iters=250, true='CW', noise_stim=0.1):
    ring_object = ring(epsilon=0.001, n_dots=8)
    j = 0
    sols = np.zeros((len(alist), len(j_list), nreps))
    for icoh, coh in enumerate(alist):
        print(coh)
        for i_j, j in enumerate(j_list):
            for n in range(nreps):
                q = ring_object.fractional_belief_propagation_ising(dt=dt, tau=tau, n_iters=n_iters, j=j,
                                                                    true=true, noise=0, plot=False,
                                                                    discrete_stim=True, coh=coh,
                                                                    stim_stamps=1, noise_stim=noise_stim)
                sols[icoh, i_j, n] = q[0].round(6)
    fig, ax = plt.subplots(ncols=len(alist), figsize=(len(alist)*4, 4))
    # colormap = pl.cm.copper(np.linspace(0.2, 1, len(alist)))
    # legendelements = [Line2D([0], [0], color=colormap[0], lw=2, label=alist[0], marker='o'),
    #                   Line2D([0], [0], color=colormap[1], lw=2, label=alist[1], marker='o'),
    #                   Line2D([0], [0], color=colormap[2], lw=2, label=alist[2], marker='o')]
    for icoh, coh in enumerate(alist):
        for i_j, j in enumerate(j_list):
            # js = -coh**2/(16*noise_stim**2) - np.log(6e-2)/4
            # ax[icoh].axvline(-js, color='r', linestyle='--', linewidth=2, alpha=0.4)
            unique_vals = np.unique(sols[icoh, i_j])
            ax[icoh].plot(np.repeat(j, len(unique_vals)),
                          unique_vals, color='k', marker='o', linestyle='',
                          markersize=3)
        ax[icoh].set_title('a = ' + str(coh))
        ax[icoh].set_xlabel('Coupling, J')
        ax[icoh].set_ylim(-0.05, 1.05)
    # ax.legend(handles=legendelements, title='a', frameon=False)
    ax[0].set_ylabel('q(z_i = CW)')
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'solutions_LBP_ising_ring_diff_a_j.png', dpi=200)
    fig.savefig(DATA_FOLDER + 'solutions_LBP_ising_ring_diff_a_j.svg', dpi=200)


def ising_1d(a=0, eps=0.1, sigma=0.1, j=0, niters=50, alpha=2.76):
    d0 = np.sqrt(-2*sigma**2 * np.log(np.sqrt(2*np.pi)*eps*sigma))
    variables = np.random.rand(2)
    x, y = variables/np.sum(variables)
    for n in range(niters):
        f_cw = y*(x+y)*(-a**2/4-d0**2) / (2*sigma**2) + alpha*j*(2*x-1)
        f_ccw = (-d0**2*(x**2+x*y) - a**2*(1/4*(x**2+x*y) + 2*y**2 + 2*x*y)) / (2*sigma**2) + alpha*j*(2*y-1)
        denom = np.exp(f_cw) + np.exp(f_ccw)
        x, y = np.exp(f_cw)/denom, np.exp(f_ccw)/denom
    return x


def sols_ising_1d_vs_j_cond_a(eps=0.2, sigma=0.2, niters=50, j_list=np.arange(-1.01, 1.01, 5e-3),
                              nreps=10, alist=[0, 0.05, 0.1, 0.2]):
    sols = np.zeros((len(j_list), len(alist), nreps))
    alpha = kernel_alpha()
    for i_a, a in enumerate(alist):
        for i_j, j in enumerate(j_list):
            for n in range(nreps):
                sols[i_j, i_a, n] = ising_1d(a=a, eps=eps, sigma=sigma, j=j, niters=niters, alpha=alpha)
    d0 = np.sqrt(-2*sigma**2 * np.log(np.sqrt(2*np.pi)*eps*sigma))
    jcrit = 0.25*(4/alpha-d0**2/sigma**2)
    fig, ax = plt.subplots(1)
    colormap = pl.cm.copper(np.linspace(0.2, 1, len(alist)))
    for i_a, a in enumerate(alist):
        for n in range(nreps):
            ax.plot(j_list, sols[:, i_a, n], color=colormap[i_a], linestyle='', marker='o', markersize=1.5)
    ax.axvline(jcrit, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Coupling, J')
    ax.set_ylabel('q(CW)')


def number_fps_vs_a_j(alist=np.arange(0, 0.525, 2.5e-2).round(4),
                      jlist=np.arange(-1.4, 1.05, 0.1).round(4),
                      nreps=50, dt=0.01, tau=0.1, n_iters=500, true='CW', noise_stim=0.1,
                      load_data=True):
    ring_object = ring(epsilon=0.001, n_dots=8)
    path = DATA_FOLDER + 'number_fps_all_smalleps_bigsigma.npy'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if load_data and os.path.exists(path):
        nfps = np.load(path)
    else:
        nfps = np.zeros((len(alist), len(jlist)))
        for icoh, coh in enumerate(alist):
            print(coh)
            for i_j, j in enumerate(jlist):
                for n in range(nreps):
                    q = ring_object.mean_field_sde_ising(dt=dt, tau=tau, n_iters=n_iters, j=j,
                                                         true=true, noise=0, plot=False,
                                                         discrete_stim=True, coh=coh,
                                                         stim_stamps=1, noise_stim=noise_stim,
                                                         sigma_lh=0.2)
                    nvals = len(np.unique(q[:, 0].round(4)))
                    nfps[icoh, i_j] += nvals
        nfps = nfps/nreps
        np.save(path, nfps)
    fig, ax = plt.subplots(1)
    im = ax.imshow(np.flipud((nfps)), aspect='auto', cmap='gist_stern', extent=[-1.4, 1.05, 0, 0.525])
    ax.set_xlabel('Coupling, J')
    ax.set_ylabel('a')
    plt.colorbar(im, ax=ax, label='Average # FPs')
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'number_fixed_points_per_simulation_smalleps_bigsigma.png', dpi=200,
                bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'number_fixed_points_per_simulation_smalleps_bigsigma.svg', dpi=200,
                bbox_inches='tight')
    return fig, ax


def number_fps_vs_a_j_bprop(alist=np.arange(0, 0.525, 2.5e-2).round(4),
                      jlist=np.arange(0, 1.05, 0.1).round(4),
                      nreps=50, dt=0.01, tau=0.1, n_iters=500, true='CW', noise_stim=0.1,
                      load_data=True):
    ring_object = ring(epsilon=0.001, n_dots=8)
    path = DATA_FOLDER + 'number_fps_all_smalleps_bprop.npy'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if load_data and os.path.exists(path):
        nfps = np.load(path)
    else:
        nfps = np.zeros((len(alist), len(jlist)))
        for icoh, coh in enumerate(alist):
            print(coh)
            for i_j, j in enumerate(jlist):
                for n in range(nreps):
                    q = ring_object.fractional_belief_propagation_ising(dt=dt, tau=tau, n_iters=n_iters, j=j,
                                                                        true=true, noise=0, plot=False,
                                                                        discrete_stim=True, coh=coh,
                                                                        stim_stamps=1, noise_stim=noise_stim,
                                                                        sigma_lh=noise_stim)
                    nvals = len(np.unique(q.round(4)))
                    nfps[icoh, i_j] += nvals
        nfps = nfps/nreps
        np.save(path, nfps)
    fig, ax = plt.subplots(1)
    im = ax.imshow(np.flipud((nfps)), aspect='auto', cmap='gist_stern', extent=[0, 1.05, 0, 0.525])
    ax.set_xlabel('Coupling, J')
    ax.set_ylabel('a')
    plt.colorbar(im, ax=ax, label='Average # FPs')
    fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'number_fixed_points_per_simulation_bprop.png', dpi=200,
                bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'number_fixed_points_per_simulation_bprop.svg', dpi=200,
                bbox_inches='tight')
    return fig, ax


def plot_all_jeff(alist=np.arange(0, 0.525, 1e-3).round(4),
                  jlist=np.arange(-1.4, 1.05, 1e-2).round(4),
                  noise_stim=0.1):
    j, a = np.meshgrid(jlist, alist)
    jeff = -3*a**2/(32*noise_stim**2) - np.log(1e-2)/4 + j
    # bias_cw = 3*a**2/(32*noise_stim**2) - np.log(6e-2)/4
    # bias_ccw = -3*a**2/(32*noise_stim**2) + np.log(6e-2)/4
    # beff = bias_cw-bias_ccw
    jstim = -3*a**2/(32*noise_stim**2) - np.log(1e-2)/4 
    jprior = j
    x, tau = np.arange(8), 0.8
    kernel = np.concatenate((np.exp(-(x-1)[:len(x)//2]/tau), (np.exp(-x[:len(x)//2]/tau))[::-1]))
    kernel[0] = 0
    alpha = np.sum(kernel)
    avals = alist[np.where(np.abs((alpha*jprior+2*jstim) - 1) < 0.01)[0]]
    jvals = jlist[np.where(np.abs((alpha*jprior+2*jstim) - 1) < 0.01)[1]]
    avals1 = alist[np.where(np.abs((-alpha*jprior+2*jstim) - 1) < 0.01)[0]]
    jvals1 = jlist[np.where(np.abs((-alpha*jprior+2*jstim) - 1) < 0.01)[1]]
    avals2 = alist[np.where(np.abs((alpha*jprior-2*jstim) - 1) < 0.01)[0]]
    jvals2 = jlist[np.where(np.abs((alpha*jprior-2*jstim) - 1) < 0.01)[1]]
    avals3 = alist[np.where(np.abs((-alpha*jprior-2*jstim) - 1) < 0.01)[0]]
    jvals3 = jlist[np.where(np.abs((-alpha*jprior-2*jstim) - 1) < 0.01)[1]]
    fig, ax = number_fps_vs_a_j(alist=np.arange(0, 0.525, 1e-2).round(4),
                                jlist=np.arange(-1.4, 1.05, 0.1).round(4),
                                nreps=50, dt=0.05, tau=0.1, n_iters=500, true='CW', noise_stim=0.1,
                                load_data=True)
    # norm = mpl.colors.TwoSlopeNorm(vmin=np.min(jeff), vcenter=0, vmax=np.max(jeff))
    
    ax.plot(jvals, avals, color='white')
    ax.plot(jvals1, avals1, color='white')
    ax.plot(jvals2, avals2, color='white')
    ax.plot(jvals3, avals3, color='white')
    # im = ax.imshow(np.flipud(1/np.abs(jeff)>0.6), cmap='bwr', aspect='auto', extent=[-1.4, 1.05, 0, 0.525],
    #                vmin=np.min(jeff), vmax=np.max(np.abs(jeff)))
    ax.set_xlabel('Coupling, J')
    ax.set_ylabel('a')
    fig2, ax2 = plt.subplots(1)
    im = ax2.imshow(np.flipud(jeff), cmap='bwr', aspect='auto', extent=[-1.4, 1.05, 0, 0.525],
                    vmin=np.min(jeff), vmax=np.max(np.abs(jeff)))
    plt.colorbar(im, ax=ax2, label='J_eff')
    ax2.set_xlabel('Coupling, J')
    ax2.set_ylabel('a')
    # ax[1].imshow(np.flipud(beff), cmap='Reds', aspect='auto', extent=[-1.4, 1.05, 0, 0.525])


def jneigbours(j,i, theta):
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


def fractional_belief_prop(j, b, theta, num_iter=100, thr=1e-10,
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
                mu_y_1[t, i] = (np.exp(j*alpha*theta[i, t]+b) *\
                        np.prod(mu_y_1[jneigbours(t, i, theta=theta), t]) *\
                            mu_y_1[i, t]**(1-alpha) \
                        + np.exp(-j*alpha*theta[i, t]-b) *\
                        np.prod(mu_y_neg1[jneigbours(t, i, theta=theta), t])*\
                            mu_y_neg1[i, t]**(1-alpha))**(1/alpha)
                # mu_y_1 += np.random.rand(8, 8)*1e-3
                # negative y_i
                mu_y_neg1[t, i] = (np.exp(-j*alpha*theta[i, t]+b) *\
                    np.prod(mu_y_1[jneigbours(t, i, theta=theta), t])*\
                    mu_y_1[i, t]**(1-alpha)
                    + np.exp(j*alpha*theta[i, t]-b) *\
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
    return q_y_1, q_y_neg1


def mean_field_solutions(eps=0.1, nreps=30, sigma=0.1, j=0, dd=1e-4):
    # epsilon = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    d = np.arange(0, 0.5, dd)
    biasval = np.arange(0, 0.6, 0.1).round(4)
    qarray = np.zeros((len(d), len(biasval), nreps, 3))
    for i_b, bias in enumerate(biasval):
        for i_d, dval in enumerate(d):
            for n in range(nreps):
                if n > nreps - 3:
                    ini_cond = [1/3]*3
                else:
                    ini_cond = None
                qcw, qnm, qccw = mean_field_1d(sigma=sigma, d=dval, epsilon=eps,
                                               n_iters=1000, j=j, ndots=8, tau=.8,
                                               biases=[0, bias, 0], ini_cond=ini_cond,
                                               threshold=1e-12)
                qarray[i_d, i_b, n, :] = [qcw, qnm, qccw]
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4.))
    colormap = pl.cm.Blues(np.linspace(0.2, 1, len(biasval)))
    col_labels = [r'$q(z_i = CW)$', r'$ q(z_i=NM)$', r'$q(z_i=CCW)$']
    for i_b, bias in enumerate(biasval):
        for row in range(3):
            for n in range(nreps):
                ax[row].plot(d, qarray[:, i_b, n, row], color=colormap[i_b],
                             marker='o', linestyle='', markersize=1.2)
            ax[row].set_xlabel('Signal difference')
            ax[row].set_ylabel(col_labels[row])
    ax[0].set_title(fr'$\varepsilon = ${eps}')
    ax[1].set_title(fr'Coupling $ J = ${j}')
    legendelements = [Line2D([0], [0], color=colormap[i], lw=2, label=biasval[i], marker='o', linestyle='')
                      for i in range(len(biasval))]
    ax[0].legend(title='NM prior', frameon=False, handles=legendelements)
    for a in ax.flatten():
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.set_ylim(-0.02, 1.02)
    fig.tight_layout()


def functions_mf_1d():
    fun_1_nm = lambda qnm, qcw, qccw, d, d0:\
        -d**2*(qcw*qccw + qnm + 1/4*(qcw**2 + qccw**2 + qnm*(1-qnm)) + 1/9*qcw*qccw)
    fun_2_nm = lambda qnm, qcw, qccw, d, d0: -d0**2*(qccw+qnm*qcw) - d**2*(qnm*(1-qcw) + 1/4*qcw*qnm)
    fun_3_nm = lambda qnm, qcw, qccw, d, d0: -d0**2*(qcw+qnm*qccw) - d**2*(qnm*(1-qccw) + 1/4*qccw*qnm)
    fun_1_ccw = lambda qnm, qcw, qccw, d, d0: -d0**2*(qccw*qcw+qnm)
    fun_2_ccw = lambda qnm, qcw, qccw, d, d0: - d**2*(qnm*(1-qcw)/4 + 1/9*qcw*qnm)
    fun_3_ccw = fun_3_nm
    fun_1_cw = lambda qnm, qcw, qccw, d, d0: -d0**2*(qccw*qcw+qnm)
    fun_2_cw = fun_2_nm
    fun_3_cw = lambda qnm, qcw, qccw, d, d0: - d**2*(qnm*(1-qccw)/4 + 1/9*qccw*qnm)
    fun_nm = lambda qnm, qcw, qccw, d, d0: fun_1_nm(qnm, qcw, qccw, d, d0) +\
        fun_2_nm(qnm, qcw, qccw, d, d0) + fun_3_nm(qnm, qcw, qccw, d, d0)
    fun_ccw = lambda qnm, qcw, qccw, d, d0: fun_1_ccw(qnm, qcw, qccw, d, d0) +\
        fun_2_ccw(qnm, qcw, qccw, d, d0) + fun_3_ccw(qnm, qcw, qccw, d, d0)
    fun_cw = lambda qnm, qcw, qccw, d, d0: fun_1_cw(qnm, qcw, qccw, d, d0) +\
        fun_2_cw(qnm, qcw, qccw, d, d0) + fun_3_cw(qnm, qcw, qccw, d, d0)
    return fun_cw, fun_nm, fun_ccw


def functions_mf_1d_bias():
    fun_1_nm = lambda qnm, qcw, qccw, d, d0, a:\
        - 4/9*(d-a)**2*qcw*qccw - (d-a/2)**2 *qcw*qnm - (-2*d+a)**2 *(qccw*qcw+qnm)-\
            (d-a)**2 * qccw**2 -(d-a)**2*qnm*qccw - (d-a/2)**2 *qcw*qcw
    fun_2_nm = lambda qnm, qcw, qccw, d, d0, a: -d0**2*(qccw+qnm*qcw) - 4*(d-a)**2*(qccw*qnm+qnm*qnm) - (d-a)**2 * qcw*qnm
    fun_3_nm = lambda qnm, qcw, qccw, d, d0, a: -d0**2*(qcw+qnm*qccw) - a**2 * (qcw*qccw + qccw*qccw) - 4*d**2 * (qnm**2 + qnm*qcw) -(d+a/2)**2*qnm*qccw
    fun_1_ccw = lambda qnm, qcw, qccw, d, d0, a: -d0**2*(qccw*qcw+qnm) -a**2/4*qcw*qccw -a**2*qnm*qccw - a**2*qccw**2
    fun_2_ccw = lambda qnm, qcw, qccw, d, d0, a: - 1/4*a**2 * (qcw**2+qcw*qccw)-a**2*(qccw+qnm*qcw) -(-2/3*d+a)**2 * qcw*qnm -(-d+3*a/2)**2 *(qccw*qnm+qnm**2)
    fun_3_ccw = fun_3_nm
    fun_1_cw = fun_1_ccw
    fun_2_cw = fun_2_nm
    fun_3_cw = lambda qnm, qcw, qccw, d, d0, a: - d**2*(qnm**2+qnm*qcw) - a**2/4*(qcw*qccw+qccw**2) - (2/3*d+a/3)**2*qnm*qccw
    fun_nm = lambda qnm, qcw, qccw, d, d0, a: fun_1_nm(qnm, qcw, qccw, d, d0, a) +\
        fun_2_nm(qnm, qcw, qccw, d, d0, a) + fun_3_nm(qnm, qcw, qccw, d, d0, a)
    fun_ccw = lambda qnm, qcw, qccw, d, d0, a: fun_1_ccw(qnm, qcw, qccw, d, d0, a) +\
        fun_2_ccw(qnm, qcw, qccw, d, d0, a) + fun_3_ccw(qnm, qcw, qccw, d, d0, a)
    fun_cw = lambda qnm, qcw, qccw, d, d0, a: fun_1_cw(qnm, qcw, qccw, d, d0, a) +\
        fun_2_cw(qnm, qcw, qccw, d, d0, a) + fun_3_cw(qnm, qcw, qccw, d, d0, a)
    return fun_cw, fun_nm, fun_ccw


def jacobian(q, sigma=0.1, eps=0.1):
    fun_cw, fun_nm, fun_ccw = functions_mf_1d()
    # qnm = sympy.symbols('qnm')
    qcw = sympy.symbols('qcw')
    # qccw = sympy.symbols('qccw')
    d0 = sympy.symbols('d0')
    # sigma = sympy.symbols('sigma')
    d = sympy.symbols('d')
    fun_exp_cw = sympy.exp((fun_cw(1-2*qcw, qcw, qcw, d, d0)-fun_nm(1-2*qcw, qcw, qcw, d, d0))/(2*sigma**2))
    fun_exp_nm = 1
    fun_exp_ccw = sympy.exp((fun_ccw(1-2*qcw, qcw, qcw, d, d0)-fun_nm(1-2*qcw, qcw, qcw, d, d0))/(2*sigma**2))
    norm = fun_exp_ccw + fun_exp_nm + fun_exp_cw
    fun_cw_final = (fun_exp_cw / norm).simplify()
    # fun_nm_final = (fun_exp_nm / norm).simplify()
    fun_ccw_final = (fun_exp_ccw / norm).simplify()
    F = sympy.Matrix([fun_cw_final, fun_ccw_final])
    jac = F.jacobian([qcw, qcw])
    d0_val = np.sqrt(-2*sigma**2 * np.log(np.sqrt(2*np.pi)*eps*sigma))
    evals = jac.subs(dict(zip([d0_val, d], [0.257, 0]))).eigenvals()


def quiver_plots_bias_nm(biaslist=[0, 0.2, 0.4], dlist=[0., 0.1, 0.2], eps=0.2, a=0, j=0,
                         sigma=0.2):
    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(12, 10))
    fig.suptitle('a = '+ str(a))
    combs = list(itertools.product(dlist, biaslist))
    for i_a, axi in enumerate(ax.flatten()):
        axi.set_title('d = ' + str(combs[i_a][0]) + ', B_nm = ' + str(combs[i_a][1]))
        axi.set_xlabel('p')
        axi.set_ylabel('m')
    fig.tight_layout()
    for i_b, bias in enumerate(biaslist):
        for i_d, d in enumerate(dlist):
            leg = True if i_b == 0 and i_d == 0 else False
            quiver_plots_1d_mf(eps=eps, sigma=sigma, d=d, biasnm=bias, ax=ax[i_d, i_b],
                               legend=leg, a=a, j=j)

def kernel_alpha(ndots=8, tau=.8):
    x = np.arange(ndots)
    kernel = np.concatenate((np.exp(-(x-1)[:len(x)//2]/tau), (np.exp(-x[:len(x)//2]/tau))[::-1]))
    kernel[0] = 0
    alpha = np.sum(kernel)
    return alpha


def ising_1d_fps(eps=0.1, sigma=0.1, j=0, d=0.1, biasnm=0, a=0, niters=50):
    d0 = np.sqrt(-2*sigma**2 * np.log(np.sqrt(2*np.pi)*eps*sigma))
    # alpha = kernel_alpha()
    q_cw = np.arange(0, 1.01, 5e-2)
    q_ccw = np.arange(0, 1.01, 5e-2)
    def system(q, d0=d0, d=d, sigma=sigma, alpha=1, j=j, a=a):
        fun_cw, fun_nm, fun_ccw = functions_mf_1d_bias()
        q1, q2 = q
        q0 = 1-q1-q2
        fcw = fun_cw(q0, q1, q2, d, d0, a)/(2*sigma**2)+j*alpha*q1
        fccw = fun_ccw(q0, q1, q2, d, d0, a)/(2*sigma**2)+j*alpha*q2
        fnm = fun_nm(q0, q1, q2, d, d0, a)/(2*sigma**2)+j*alpha*q0+biasnm
        maxf = np.max([fcw, fccw, fnm])
        norm = (np.exp(fcw-maxf) + np.exp(fccw-maxf) + np.exp(fnm - maxf))
        f1 = np.exp(fcw - maxf)/norm
        f2 = np.exp(fccw - maxf) / norm
        return f1, f2
    fps = []
    qualitative_sol = []
    for q1 in q_cw:
        for q2 in q_ccw:
            if q1+q2 > 1.25:
                continue
            q = [q1, q2]
            for n in range(niters):
                q = system(q, d0=d0, d=d, sigma=sigma, alpha=1, j=j, a=a)
            sol = np.round(q, 4)
            if not any(np.allclose(sol, fp) for fp in fps):
                fps.append(sol)
                label = np.nan
                if sol[0] > 0.5:
                    label = 0
                if sol[1] > 0.5:
                    label = 2
                if 1-(sol[0]+sol[1]) > 0.5:
                    label = 1
                if sol[0] <= 0.5 and sol[1] <= 0.5 and 1-(sol[0]+sol[1]) <= 0.5:
                    label = 7
                qualitative_sol.append(label)
    qualitative_sol = np.unique(np.array(qualitative_sol))
    return qualitative_sol[~np.isnan(qualitative_sol)].astype(int)


def find_fps(eps=0.1, sigma=0.1, j=0, d=0.1, biasnm=0, tol=1e-4, a=0):
    d0 = np.sqrt(-2*sigma**2 * np.log(np.sqrt(2*np.pi)*eps*sigma))
    # alpha = kernel_alpha()
    q_cw = np.arange(0, 1.01, 5e-2)
    q_ccw = np.arange(0, 1.01, 5e-2)
    def system(q, d0=d0, d=d, sigma=sigma, alpha=1, j=j, a=a):
        fun_cw, fun_nm, fun_ccw = functions_mf_1d_bias()
        q1, q2 = q
        q0 = 1-q1-q2
        fcw = fun_cw(q0, q1, q2, d, d0, a)/(2*sigma**2)+j*alpha*q1
        fccw = fun_ccw(q0, q1, q2, d, d0, a)/(2*sigma**2)+j*alpha*q2
        fnm = fun_nm(q0, q1, q2, d, d0, a)/(2*sigma**2)+j*alpha*q0+biasnm
        maxf = np.max([fcw, fccw, fnm])
        norm = (np.exp(fcw-maxf) + np.exp(fccw-maxf) + np.exp(fnm - maxf))
        f1 = np.exp(fcw - maxf)/norm - q1
        f2 = np.exp(fccw - maxf) / norm - q2
        return f1, f2
    fps = []
    qualitative_sol = []
    # bounds = [(0, 1), (0, 1)]
    for q1 in q_cw:
        for q2 in q_ccw:
           solroot = root(system, [q1, q2])
           sol = np.round(solroot.x, 5)
           fval = solroot.fun
           if any(np.abs(fval) > tol):
               continue
           if sol[0] < 0 or sol[0] > 1 or sol[1] < 0 or sol[1] > 1 or sol[0]+sol[1] > 1:
               continue
           if not any(np.allclose(sol, fp) for fp in fps):
               fps.append(sol)
               label = np.nan
               if sol[0] > 0.5:
                   label = 0
               if sol[1] > 0.5:
                   label = 2
               if 1-(sol[0]+sol[1]) > 0.5:
                   label = 1
               if sol[0] <= 0.5 and sol[1] <= 0.5 and 1-(sol[0]+sol[1]) <= 0.5:
                   label = 7
               qualitative_sol.append(label)
    qualitative_sol = np.unique(np.array(qualitative_sol))
    return qualitative_sol[~np.isnan(qualitative_sol)].astype(int)
    # vals = np.array(qualitative_sol)[np.array(qualitative_sol) != '']


def qualbehav(sols):
    # l = 0 --> CW
    # l = 1 --> NM
    # l = 2 --> CCW
    # l = 3 --> CW & NM
    # l = 4 --> CW & CCW
    # l = 5 --> CCW & NM
    # l = 6 --> NM & CW & CCW
    # l += 0.5 --> any config + undef
    add = 0.5 if 7 in sols and len(sols) > 1 else 0
    if any(sols != 7):
        sols = sols[sols != 7]
    if len(sols) == 1:
        val = sols[0] + add
    if len(sols) == 2:
        if 0 not in sols:
            val = 5  + add
        if 1 not in sols:
            val = 4 + add
        if 2 not in sols:
            val = 3 + add
    if len(sols) == 3:
        val = 6 + add
    return val


def dict_sols():
    d = {0: 'CW', 0.5: '+udf.', 1: 'NM', 1.5: '+udf.',
         2: 'CCW', 2.5: '+udf.', 3: 'CW & NM', 3.5: '+udf.',
         4: 'CW & CCW', 4.5: '+udf.', 5: 'CCW & NM', 5.5: '+udf.',
         6: 'NM & CW & CCW', 6.5: '+udf.', 7: 'udf.'}
    return d


def phase_diagram_d_b(dlist=np.arange(0, 1, 1e-2),
                      blist=np.arange(0, 10.1, 0.1),
                      j=0):
    if j == 0:
        path = DATA_FOLDER + 'qual_behavior_vs_contrast_prior_bias.npy'
    else:
        path = DATA_FOLDER + 'qual_behavior_vs_contrast_prior_bias_coupling' + str(j) + '.npy'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        behav = np.load(path)
    else:
        behav = np.zeros((len(dlist), len(blist)))
        for i_d, d in enumerate(dlist):
            for i_b, b in enumerate(blist):
                sols = find_fps(eps=0.1, sigma=0.1, j=j, d=d, biasnm=b)
                behav[i_d, i_b] = qualbehav(sols)
        np.save(path, behav)
    plt.figure()
    cmap = plt.get_cmap('Set2', 7)
    plt.imshow(np.flipud(behav), cmap=cmap,
               extent=[np.min(blist), np.max(blist), np.min(dlist), np.max(dlist)],
               aspect='auto', vmin=0.5, vmax=6.5)
    plt.text(0.7, 0.7, dict_sols()[4], color='k')
    plt.text(6, 0.1, dict_sols()[1], color='k')
    plt.xlabel('Bias towards NM')
    plt.ylabel('Contrast difference d')


def phase_diagram_d_biasccw_a(dlist=np.arange(0, 0.505, 1e-2),
                              alist=np.arange(0, 0.505, 1e-2), biasnm=0,
                              resimulate=False, ax=None, cbar=False, fig=None, j=0,
                              plot=False):
    if biasnm == 0:
        lab = ''
    else:
        lab = '_bias_nm_' + str(biasnm)
    if j == 0:
        path = DATA_FOLDER + 'qual_behavior_vs_contrast_prior_contrast_bias_CW' + lab + '_eps_02_simuls.npy'
    else:
        path = DATA_FOLDER + 'qual_behavior_vs_contrast_prior_contrast_bias_CW' + lab + '_coupling_' + str(j) + '_eps_02_simuls.npy'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not resimulate:
        behav = np.load(path)
    else:
        behav = np.zeros((len(dlist), len(alist)))
        for i_d, d in enumerate(dlist):
            for i_a, a in enumerate(alist):
                sols = ising_1d_fps(eps=0.2, sigma=0.2, j=j, d=d, biasnm=biasnm, a=a, niters=80)
                behav[i_d, i_a] = qualbehav(sols)
        np.save(path, behav)
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(7, 4))
        cmap = plt.get_cmap('tab20', 15)
        # Define colors for phases
        colors = ['firebrick', 'darksalmon',
                  'royalblue', 'lightsteelblue',
                  'orange', 'navajowhite',
                  'darkorchid', 'plum',
                  'forestgreen', 'palegreen',
                  'grey', 'lightgray',
                  'goldenrod', 'gold',
                  'black']
        cmap = ListedColormap(colors)
        extent = [np.min(alist), np.max(alist), 2*np.min(dlist), 2*np.max(dlist)]
        im = ax.imshow(np.flipud(behav), cmap=cmap, extent=extent,
                       aspect='auto', vmin=-0.25, vmax=7.25)
        if cbar:
            ax_pos = ax.get_position()
            ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*1.18, ax_pos.y0-ax_pos.height*1.15,
                                    ax_pos.width*0.15, ax_pos.height*3])
            cb = plt.colorbar(im, cax=ax_cbar)
            cb.ax.set_yticks(np.arange(0., 7.5, 0.5), dict_sols().values())
        ax.set_xlabel('Bias towards CW movement, a')
        ax.set_ylabel('Contrast difference, d')
        ax.set_title('Bias towards NM = ' + str(biasnm), fontsize=14)


def plot_phase_diagrams_vs_biasnm(biasnmlist=[0, 0.5, 1, 1.5, 2],
                                  jlist=[0, 0.4, 0.8, 1.2, 1.6]):
    fig, ax = plt.subplots(ncols=len(biasnmlist), nrows=len(jlist),
                           figsize=(4*len(biasnmlist), 3.5*len(jlist)))
    for i_b, bias in enumerate(biasnmlist):
        for i_j, j in enumerate(jlist):
            phase_diagram_d_biasccw_a(dlist=np.arange(0, 0.505, 1e-2),
                                      alist=np.arange(0, 0.505, 1e-2), biasnm=bias,
                                      resimulate=False, ax=ax[i_j, i_b], cbar=(i_b == len(biasnmlist)-1)*(i_j == 2),
                                      fig=fig, j=j, plot=True)
            if i_b == len(biasnmlist)-1:
                ax2 = ax[i_j, i_b].twinx()
                ax2.set_yticks([])
                ax2.set_ylabel('J = ' + str(j))
            if i_b != 0:
                ax[i_j, i_b].set_ylabel('')
                # ax[i_j, i_b].set_yticks([0, 0.5, 1])
            if i_j < len(jlist)-1:
                ax[i_j, i_b].set_xlabel('')
                # ax[i_j, i_b].set_xticks([0, 0.25, 0.5])
            ax[i_j, i_b].plot([0, 0.5], [0, 1], color='white')
            if i_j > 0:
                ax[i_j, i_b].set_title('')
            if i_b == len(biasnmlist)-2:
                fig.tight_layout()
    fig.savefig(DATA_FOLDER + 'qualitative_behavior_j0.png', dpi=200, bbox_inches='tight')
    fig.savefig(DATA_FOLDER + 'qualitative_behavior_j0.svg', dpi=200, bbox_inches='tight')


def quiver_plots_1d_mf(eps=0.1, sigma=0.1, j=0, d=0.1, biasnm=0, ax=None, legend=False,
                       plot_modulo=False, prec=1e-3, a=0, plot_regions=True):
    d0 = np.sqrt(-2*sigma**2 * np.log(np.sqrt(2*np.pi)*eps*sigma))
    fun_cw, fun_nm, fun_ccw = functions_mf_1d_bias()
    q_cw = np.arange(-0.1, 1.1, 7.5e-2)
    q_ccw = np.arange(-0.1, 1.1, 7.5e-2)
    X, Y = np.meshgrid(q_ccw, q_cw)
    U = fun_ccw(1-X-Y, Y, X, d, d0, a)/(2*sigma**2) + j*X
    V = fun_cw(1-X-Y, Y, X, d, d0, a)/(2*sigma**2) + j*Y
    NM = fun_nm(1-X-Y, Y, X, d, d0, a)/(2*sigma**2)+ biasnm + j*(1-X-Y)
    norm = (np.exp(U) + np.exp(V) + np.exp(NM+ biasnm))
    Up = np.exp(U) / norm - X
    Vp = np.exp(V) / norm - Y
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.quiver(X, Y, Up, Vp)
    q_cw = np.arange(-0.11, 1.11, prec)
    q_ccw = np.arange(-0.11, 1.11, prec)
    X, Y = np.meshgrid(q_ccw, q_cw)
    U = fun_ccw(1-X-Y, Y, X, d, d0, a)/(2*sigma**2) + j*X
    V = fun_cw(1-X-Y, Y, X, d, d0, a)/(2*sigma**2) + j*Y
    NM = fun_nm(1-X-Y, Y, X, d, d0, a)/(2*sigma**2) + biasnm  + j*(1-X-Y)
    norm = (np.exp(U) + np.exp(V) + np.exp(NM+biasnm))
    Up = np.exp(U) / norm - X
    Vp = np.exp(V) / norm - Y
    ax.contour(X, Y, Up, levels=[0], colors='r', linewidths=2)
    ax.contour(X, Y, Vp, levels=[0], colors='b', linewidths=2)
    if plot_regions:
        q = np.arange(0, 1, 1e-2)
        ax.plot(q, 1-q, color='k', alpha=0.5)
        q = np.arange(0, 0.5, 1e-2)
        ax.plot(q, 1/2-q, color='k', alpha=0.5)
        ax.plot([0, 0.5], [0.5, 0.5], color='k', alpha=0.5)
        ax.plot([0.5, 0.5], [0, 0.5], color='k', alpha=0.5)
    if legend:
        legendelements = [Line2D([0], [0], color='r', lw=2, label=r"$\dot{q(CCW)}=0$"),
                          Line2D([0], [0], color='b', lw=2, label=r"$\dot{q(CCW)}=0$")]
        ax.legend(frameon=False, handles=legendelements, bbox_to_anchor=[0.4, 1.1], ncol=2)
    ax.set_xlabel('q(CCW)')
    ax.set_ylabel('q(CW)')
    if plot_modulo:
        plt.figure()
        modulo = np.sqrt(Vp**2+Up**2)
        im = plt.imshow(np.log(np.flipud(modulo)), extent=[np.min(q_ccw), np.max(q_ccw),
                                                           np.min(q_ccw), np.max(q_ccw)],
                        cmap='binary')
        plt.colorbar(im, label=r'$\log \; ||F(q(CW), q(CCW))||$')
        plt.contour(X, Y, Up, levels=[0], colors='r', linewidths=2)
        plt.contour(X, Y, Vp, levels=[0], colors='b', linewidths=2)
        idxs = np.where(modulo < 1e-3)
        plt.plot(q_ccw[idxs[0]], q_cw[idxs[1]], marker='x', color='r',
                  linestyle='', markersize=5)
        plt.xlabel('q(CCW)')
        plt.ylabel('q(CW)')


def mean_field_1d(sigma=0.1, d=0., epsilon=0.1, n_iters=100, j=0, ndots=8, tau=.8, biases=np.zeros(3),
                  ini_cond=None, threshold=None):
    d0 = np.sqrt(-2*sigma**2 * np.log(np.sqrt(2*np.pi)*epsilon*sigma))
    x = np.arange(ndots)
    kernel = np.concatenate((np.exp(-(x-1)[:len(x)//2]/tau), (np.exp(-x[:len(x)//2]/tau))[::-1]))
    kernel[0] = 0
    alpha = np.sum(kernel)
    fun_cw, fun_nm, fun_ccw = functions_mf_1d()
    # qnm = 1/3
    # qcw = 1/3
    # qccw = 1/3
    if ini_cond is None:
        qnm, qcw, qccw = np.random.rand(3)
        norm = qnm + qcw + qccw
        qnm /= norm
        qcw /= norm
        qccw /= norm
    else:
        qnm, qcw, qccw = ini_cond
    bias_cw, bias_nm, bias_ccw = biases
    for t in range(n_iters):
        norm = np.exp(bias_nm + j*alpha*(2*qnm-1) + fun_nm(qnm, qcw, qccw, d, d0)/(2*sigma**2))+\
                np.exp(bias_cw + j*alpha*(2*qcw-1) + fun_ccw(qnm, qcw, qccw, d, d0)/(2*sigma**2))+\
                np.exp(bias_ccw + j*alpha*(2*qccw-1) + fun_cw(qnm, qcw, qccw, d, d0)/(2*sigma**2))
        qnmn = np.exp(bias_nm + j*alpha*(2*qnm-1) + fun_nm(qnm, qcw, qccw, d, d0)/(2*sigma**2)) / norm
        qcwn = np.exp(bias_cw + j*alpha*(2*qcw-1) + fun_cw(qnm, qcw, qccw, d, d0)/(2*sigma**2)) / norm
        qccwn = np.exp(bias_ccw + j*alpha*(2*qccw-1) + fun_ccw(qnm, qcw, qccw, d, d0)/(2*sigma**2)) / norm
        if threshold is not None:
            sum_diffs = np.abs(qnmn-qnm) + np.abs(qccwn-qccw)+ np.abs(qcwn-qcw)
            if sum_diffs < threshold:
                break
        qnm = qnmn
        qcw = qcwn
        qccw = qccwn
    return qcw, qnm, qccw


def d0_sigma_maxvals(eps=0.1, sigma=0.1):
    sigmalist = np.logspace(-5, -0.3, 150)
    d0list = np.logspace(-3, 0, 150)
    X, Y = np.meshgrid(sigmalist, d0list)
    fun = (Y**2 < (2-np.log(np.sqrt(2*np.pi)*X)) * 2*X**2)*1
    plt.figure()
    plt.pcolormesh(X, Y, fun)
    d0 = np.sqrt(-2*sigma**2 * np.log(np.sqrt(2*np.pi)*eps*sigma))
    plt.axhline(d0)
    plt.axvline(sigma)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Sigma')
    plt.ylabel('d0')
    plt.tight_layout()


if __name__ == '__main__':
    # number_fps_vs_a_j_bprop(alist=np.arange(0, 0.525, 2.5e-2).round(4),
    #                         jlist=np.arange(0, 1.05, 0.05).round(4),
    #                         nreps=20, dt=0.05, tau=0.1, n_iters=2500, true='CW', noise_stim=0.1,
    #                         load_data=True)
    # ring(epsilon=0.001, n_dots=8).mean_field_ring(true='CW', j=0.4, b=[0., 0., 0.], plot=True,
    #                                               n_iters=100, noise=0)
    # ring(epsilon=0.001, n_dots=8).mean_field_ring(true='NM', j=0.4, b=[0., 0., 0.], plot=True,
    #                                               n_iters=100, noise=0)
    # bifurcations_difference_stim_epsilon(nreps=100, resimulate=False, epslist=[0.1])
    # n_objects_ring_stim_epsilon(nreps=50, resimulate=False)
    # n_objects_ring_stim_eps01_bias(nreps=100, resimulate=False, eps=0.1)
    # plot_bifurcations_all_bias(eps=0.1, nreps=30, biases=np.arange(0., 0.11, 0.02).round(4),
    #                             smallds=False)
    # plot_bifurcations_all_bias(eps=0.1, nreps=30, biases=np.arange(0., 0.6, 0.1).round(4),
    #                             smallds=False, j=0.)
    plot_phase_diagrams_vs_biasnm(biasnmlist=[0, 0.25, 0.5, 0.75, 1],
                                  jlist=[0, 0.4, 0.8, 1.2, 1.6])
    # for biasnm in [1.5, 2]:
    #     for j in [0., 0.4, 0.8, 1.2, 1.6]:
    #         phase_diagram_d_biasccw_a(dlist=np.arange(0, 0.505, 1e-2),
    #                                   alist=np.arange(0, 0.505, 1e-2),
    #                                   biasnm=biasnm, resimulate=True,
    #                                   ax=None, cbar=False, fig=None, j=j,
    #                                   plot=False)
    # ss = [[0., 1.]]*2
    # ss = [[0.47, 0.53]]*5
    # for i in range(len(ss)):
    #     ring(epsilon=0.001, n_dots=8).mean_field_sde(dt=0.06, tau=0.15, n_iters=150, j=0.,
    #                                                 true='CW', noise=0., plot=True,
    #                                                 discrete_stim=True, s=ss[i],
    #                                                 b=[0, 0, 0], noise_stim=0.0,
    #                                                 coh=0.1, nstates=3, stim_stamps=1)
    # sols_vs_j_cond_on_a_beleif_prop(alist=[0, 0.05, 0.1, 0.2], j_list=np.arange(0, 1.02, 2e-2).round(5),
    #                                 nreps=50, dt=0.05, tau=0.1, n_iters=250, true='CW', noise_stim=0.1)
    # for i in range(2):
    #     ring(epsilon=0.001, n_dots=8).fractional_belief_propagation_ising(dt=0.05, tau=0.1, n_iters=500, j=0.1,
    #                                                                       true='CW', noise=0.0, plot=True,
    #                                                                       discrete_stim=True, 
    #                                                                       noise_stim=0.1, s=[0.5, 0.5],
    #                                                                       stim_stamps=1, sigma_lh=0.1,
    #                                                                       coh=0.)
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
    # psychometric_curve_ring(dt=0.01, tau=0.1, n_iters=120, j_list=[0., 4],
    #                         noise=0.1, cohlist=np.arange(0, 0.22, 2e-2),
    #                         nreps=1000, noise_stim=0.2)
