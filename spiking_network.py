# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 22:12:42 2025

@author: alexg
"""
import numpy as np
import matplotlib.pyplot as plt
import gibbs_necker as gn

# Parameters
num_dots = 8
num_states = 2  # x=-1, x=+1

J0 = 1.3              # Ising interaction strength
alpha = 1              # Observation influence
lambda_0 = 0.3         # Baseline firing rate (Hz)
beta = 0.2             # Sensitivity of firing rate to input
delta_t = 0.001        # Time step (1 ms)
T = 10.0                # Total simulation time (1 second)
gamma = 6
num_steps = int(T / delta_t)
refractory_period = 10e-3  # refractory period, 10ms
# Randomly generate observations: 1 (supports motion) or 0 (doesn't support)
np.random.seed(42)

# n_iter = num_steps+1
# burn_in = 100

recurrent_weights = np.abs([[0, 1, 1, 0 ,-1 ,0 ,0 ,0], [1, 0, 0, 1, 0, -1, 0, 0],
                            [1, 0, 0, 1, 0, 0, -1, 0], [0, 1, 1, 0, 0, 0, 0, -1],
                            [-1, 0, 0, 0, 0, 1, 1, 0], [0, -1, 0, 0, 1, 0, 0, 1],
                            [0, 0, -1, 0, 1, 0, 0, 1], [0, 0, 0, -1, 0, 1, 1, 0]])*J0
# np.fill_diagonal(recurrent_weights, J0)

# Function to compute observation potential h_i(z_i, s_i)
def observation_potential(z_state, s_obs):
    if s_obs == 1:
        return alpha if z_state == 0 else -alpha
    else:
        return -alpha if z_state == 0 else alpha


def eta(t, t_spikes, tau=0.01):
    return np.sum(np.exp(-(t-t_spikes)/tau)/tau)

# Use desired neurons per state
neurons_per_state = 10
total_neurons = num_dots * num_states * neurons_per_state


# Clear previous spike data
spike_times = []
spike_times_inh = []

# Simulate spiking dynamics with updated parameters
spikes = np.zeros((num_dots, num_states, neurons_per_state))  # Set spikes array
n_inh_neurons = 50
inh_spikes = np.zeros((n_inh_neurons))  # Inhibitrory spikes

n_total = n_inh_neurons + neurons_per_state*num_states
# init_state = np.random.choice([-1, 1], 8)
# states_mat = gn.gibbs_samp_necker(init_state=init_state,
#                                   burn_in=burn_in, n_iter=n_iter+burn_in, j=0.85, stim=0)

# Initialize last spike time tracker
last_spike_time = np.zeros((num_dots, num_states, neurons_per_state))

for t in range(num_steps):
    current_time = t * delta_t
    new_spikes = np.zeros_like(spikes)
    new_inh_spikes = np.zeros_like(inh_spikes)
    for inh in range(n_inh_neurons):
        inhibitory_input = gamma

        # Compute firing rate
        firing_rate = lambda_0 * np.exp(inhibitory_input)

        # Generate Poisson spike
        p_spike = 1 - np.exp(-firing_rate * delta_t)
        if np.random.rand() < p_spike:
            spike_times_inh.append((inh, current_time))
            new_inh_spikes[inh] = 1
    inh_input = inh_spikes.sum()
    for i in range(num_dots):  # Loop over dots
        for k in range(num_states):  # Loop over states
            for n in range(neurons_per_state):  # Loop over neurons per state
                if (current_time - last_spike_time[i, k, n]) >= refractory_period:
                    # Compute input current: observation + recurrent inputs
                    obs_input = 0  # np.random.randn()*10
    
                    # Recurrent input from neighboring neurons with same state
                    recurrent_input = 0
                    for j in range(num_dots):
                        if i != j:
                            recurrent_input += recurrent_weights[i, j] *\
                                eta(current_time, last_spike_time[i, k]) -\
                                0.3*recurrent_weights[i, j] * eta(current_time, last_spike_time[i, 1-k])
                    I_input = obs_input + recurrent_input/np.sqrt(n_total) - inh_input
    
                    # Compute firing rate
                    firing_rate = lambda_0 * np.exp(beta * I_input)
                    
                    # Generate Poisson spike
                    # p_spike = 1 - np.exp(-firing_rate * delta_t)
                    p_spike = firing_rate * delta_t  # * np.exp(-firing_rate * delta_t)
                    if np.random.rand() < p_spike:
                        neuron_id = i * num_states * neurons_per_state + k * neurons_per_state + n
                        spike_times.append((neuron_id, current_time))
                        new_spikes[i, k, n] = 1
                        last_spike_time[i, k, n] = current_time

    # Update spikes for recurrent connections in the next timestep
    spikes = new_spikes
    inh_spikes = new_inh_spikes

# Define colors for each state:
state_colors = ['r', 'b']

# Prepare data for colored raster plot
neuron_ids_colored = []
times_colored = []
colors_colored = []

for neuron_id, time in spike_times:
    dot_idx = neuron_id // (num_states * neurons_per_state)
    state_idx = (neuron_id % (num_states * neurons_per_state)) // neurons_per_state
    color = state_colors[state_idx]

    neuron_ids_colored.append(neuron_id)
    times_colored.append(time)
    colors_colored.append(color)


state_labels = ['x=+1', 'x=-1']
# Sliding window parameters
window_size = 0.2  # 200 ms window
step_size = 0.01    # 10 ms step
num_windows = int((T - window_size) / step_size) + 1

# Time bins
time_bins = np.arange(0, T - window_size + step_size, step_size)
firing_rate_evolution = np.zeros((num_states, len(time_bins)))

# Compute firing rate for each time window
for idx, start_time in enumerate(time_bins):
    end_time = start_time + window_size
    
    # Count spikes in the window for each state
    spikes_in_window = [0] * num_states
    for neuron_id, spike_time in spike_times:
        if start_time <= spike_time < end_time:
            state_idx = (neuron_id % (num_states * neurons_per_state)) // neurons_per_state
            spikes_in_window[state_idx] += 1
    
    # Convert spike counts to firing rates (Hz)
    for state_idx in range(num_states):
        firing_rate_evolution[state_idx, idx] = spikes_in_window[state_idx] / (neurons_per_state * num_dots * window_size)

# Plotting firing rate evolution over time
fig, ax = plt.subplots(nrows=3, figsize=(12, 10))
# ax[0].imshow(states_mat.T, cmap='coolwarm', aspect='auto', interpolation='none',
#              extent=[0, T, 8, -1])
# ax[0].set_yticks(np.arange(8)[::2], np.arange(1, 9)[::-1][::2])
# ax[0].set_ylabel('Node index i')
neuron_ids_inh, times_inh = zip(*spike_times_inh)
ax[0].scatter(times_inh, neuron_ids_inh, s=1, color='black')
ax[0].set_ylabel('Inh. neuron ID')
for state_idx, color in enumerate(state_colors):
    ax[2].plot(time_bins + window_size / 2, firing_rate_evolution[state_idx], label=state_labels[state_idx], color=color)

ax[2].set_ylabel('Firing Rate (Hz)')
ax[2].legend()
ax[2].set_xlabel('Time (s)')

# Plotting the colored raster plot
ax[1].scatter(times_colored, neuron_ids_colored, s=1, c=colors_colored)
ax[1].set_ylabel('Neuron ID')
for a in ax:
    a.set_xlim(-0.2, T+0.2)
