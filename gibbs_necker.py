# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:45:48 2023

@author: alexg
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import itertools
import os


"""
8x8 matrix, left-right order first front:
    
Front:
        
    0    1
    
    
    2    3

Back:
    
    4    5
    
    
    6    7
    
Connections:
    0: 1, 2, 4
    1: 0, 3, 5
    2: 0, 3, 6
    3: 1, 2, 7
    4: 0, 5, 6
    5: 1, 4, 7
    6: 2, 4, 7
    7: 3, 5, 6
    
"""

# C matrix:
    
C = np.load('C:/Users/alexg/Onedrive/Escritorio/phd/gibbs_sampling_necker/data_folder/c_mat.npy')


# THETA mat

THETA = np.array([[0 ,1 ,1 ,0 ,1 ,0 ,0 ,0], [1, 0, 0, 1, 0, 1, 0, 0],
                  [1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1],
                  [0, 0, 1, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 1, 1, 0]])


def get_connections(node):
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


def sigmoid(x):
    return 1/(1+np.exp(-x))


def k_val(x_vec, j_mat):
    return np.matmul(np.matmul(x_vec.T, j_mat), x_vec)/2


def change_prob(x_vect, x_vect1, j):
    j_mat = np.abs(mat_theta(x_vect, x_vect, j))
    j_mat1 = np.abs(mat_theta(x_vect1, x_vect1, j))
    return sigmoid(j*(-k_val(x_vect, j_mat) + k_val(x_vect1, j_mat1)))


def mat_theta(x_vect_1, x_vect_2, j):
    mat = np.zeros((len(x_vect_1), len(x_vect_2)))
    for i in range(len(x_vect_1)):
        connections = get_connections(i)
        for con in connections:
            mat[i, con] = x_vect_1[i] * x_vect_2[con] * j
    return mat


def gibbs_samp_necker(init_state, burn_in, n_iter, j):
    x_vect = init_state
    states_mat = np.empty((n_iter-burn_in, 8))
    states_mat[:] = np.nan
    for i in range(n_iter):
        node = np.random.choice(np.arange(1, 9), p=np.repeat(1/8, 8))
        x_vect1 = np.copy(x_vect)
        x_vect1[node-1] = -x_vect1[node-1]
        prob = change_prob(x_vect, x_vect1, j)
        # val_bool = np.random.choice([False, True], p=[1-prob, prob])
        val_bool = np.random.binomial(1, prob, size=None)
        if val_bool:
            x_vect[node-1] = -x_vect[node-1]
        if i >= burn_in:
            states_mat[i-burn_in, :] = x_vect
    return states_mat


def get_mu(x_vec):
    y = np.copy(x_vec)
    y[4:-1] *= -1
    return np.sum(y)


def get_mu_v2(x_vec):
    return np.sum(x_vec)


def get_analytical_prob(x_vec, j):
    mat = np.abs(mat_theta(x_vec, x_vec, j))
    exponent = k_val(x_vec, mat)
    return np.exp(exponent*j)


def get_mu_from_mat(mat):
    mu_vec = [get_mu(x) for x in mat]
    return np.array(mu_vec)


def get_mu_from_mat_v2(mat):
    mu_vec = [get_mu_v2(x) for x in mat]
    return np.array(mu_vec)


def num_configs():
    return np.array((1, 8, 12, 12, 4, 24, 24, 8,
                     8, 24, 6, 24, 6, 2, # start mu=0
                     8, 24, 24, 4, 12, 12, 8, 1))[::-1]


def mu_states():
    return np.array((-8, -6, -4, -4, -4, -2, -2, -2,
                     0, 0, 0, 0, 0, 0,
                     2, 2, 2, 4, 4, 4, 6, 8))


def plot_nconf_mu_gibbs(classes, clas_values = np.arange(0, 23)):
    plt.figure()
    mu = mu_states()
    nc = num_configs()
    counts, _ = np.histogram(classes, bins=clas_values)
    for i_c, c in enumerate(classes):
        if i_c == 0:
            continue
        plt.plot([mu[classes[i_c-1]], mu[c]], [nc[classes[i_c-1]], nc[c]],
                 color='r', linewidth=0.3)
    for n in range(22):
        plt.plot(mu[n], nc[n], marker='o', linestyle='', color='k',
                 markersize=counts[n]/sum(counts)*30+10)
    plt.ylabel('# configs')
    plt.xlabel(r'$\mu$')


def plot_k_vs_mu(states_mat, j):
    plt.figure()
    mu_vec = np.round(get_mu_from_mat_v2(states_mat), 1)
    klist = []
    for x_vect in states_mat:
        j_mat = np.abs(mat_theta(x_vect, x_vect, j))/j
        k = k_val(x_vect, j_mat)
        klist.append(np.round(k, 2))
    # classes = get_classes(states_mat)
    # for t in range(1, len(klist)):
    #     plt.plot([mu_vec[t-1], mu_vec[t]], [klist[t-1], klist[t]], color='r',
    #              linewidth=C[classes[t-1], classes[t]])
    plt.plot(mu_vec, klist, color='r')
    nc = num_configs() / 256
    arr_conj = np.unique(np.column_stack((mu_vec, klist)), axis=0)
    for n in range(len(arr_conj)):
        ind = np.where((mu_vec == arr_conj[n, 0]) & (np.array(klist) == arr_conj[n, 1]))[0][0]
        state = states_mat[ind]
        classe = check_class(state)
        muval = get_mu_v2(state)
        plt.plot(muval, arr_conj[n, 1], marker='o', linestyle='',
                 markersize=nc[classe]*60+7, color='k')
    plt.ylabel(r'$k = \frac{1}{2} \vec{x}^T \theta_{ij} \vec{x}$', fontsize=12)
    plt.xlabel(r'$\mu$', fontsize=12)


def plot_k_vs_mu_analytical(eps=6e-2):
    nc = num_configs() / 256
    combs = list(itertools.product([-1, 1], repeat=8))
    combs = np.array(combs, dtype=np.float64)
    pmat = np.zeros((22, 22))
    class_count = []
    klist = []
    muvec = []
    for i_x, x_vec in enumerate(combs):
        x_vec = np.array(x_vec, dtype=np.float64)
        classes = check_class(x_vec)
        k = k_val(x_vec, THETA)
        if classes in class_count:
            continue
        else:
            muvec.append(get_mu_v2(x_vec))
            klist.append(k)
            class_count.append(classes)
            class_count2 = []
            for i_x2, x_vec2 in enumerate(combs):
                classes2 = check_class(x_vec2)
                if classes2 in class_count2:
                    continue
                else:
                    class_count2.append(classes2)
                    pmat[classes2, classes] = change_prob(x_vec, x_vec2, j=1)
    cte = np.sum(pmat, axis=1)
    pmat /= cte
    plt.figure()
    for ic, cl in enumerate(class_count):
        for ic2, cl2 in enumerate(class_count):
            plt.plot([muvec[ic]+eps, muvec[ic2]+eps], [klist[ic]+eps, klist[ic2]+eps], color='r',
                     linewidth=C[class_count[ic], class_count[ic2]]/5)
            plt.plot([muvec[ic]-eps, muvec[ic2]-eps], [klist[ic]-eps, klist[ic2]-eps], color='r',
                     linewidth=C[class_count[ic2], class_count[ic]]/5)
    for i_c, classe in enumerate(class_count):
        plt.plot(muvec[i_c], klist[i_c], marker='o', linestyle='', color='k',
                 markersize=nc[classe]*55+7)
    plt.ylabel(r'$k = \frac{1}{2} \vec{x}^T \theta_{ij} \vec{x}$', fontsize=12)
    plt.xlabel(r'$\mu$', fontsize=12)


def compute_C():
    combs = list(itertools.product([-1, 1], repeat=8))
    combs = np.array(combs, dtype=np.float64)
    c_mat = np.zeros((22, 22))
    class_count = []
    for i, x_vec in enumerate(combs):
        for j, x_vec_2 in enumerate(combs):
            cl1 = check_class(x_vec)
            if cl1 not in class_count:
                cl2 = check_class(x_vec_2)
                if sum(x_vec * x_vec_2 == -1) == 1:
                    c_mat[cl1, cl2] += 1
    for j in range(22):
        c_mat[j, :] = c_mat[j, :] / np.sum(c_mat[j, :]) * 8
    np.save('C:/Users/alexg/Onedrive/Escritorio/phd/gibbs_sampling_necker/data_folder/c_mat.npy', c_mat)


def comptue_C_with_vecs():
    init_vec = np.repeat(1, 8)
    # class_0 = check_class(init_vec)
    c_mat = np.zeros((22, 22))
    for cl in range(22):
        pos_vecs = np.empty((8, 8))
        for j in range(8):
            vec_1 = np.copy(init_vec)
            vec_1[j] = -vec_1[j]
            pos_vecs[j, :] = vec_1
            cl_1 = check_class(vec_1)
            c_mat[cl, cl_1] += 1
        init_ind = np.random.choice(8)
        init_vec = pos_vecs[init_ind]


def check_class(x_vec):
    mat_th = np.sum(np.triu(mat_theta(x_vec, x_vec, 1)))
    mat = mat_theta(x_vec, x_vec, 1)
    if np.sum(x_vec) == 8:
        return 0
    if np.sum(x_vec) == -8:
        return 21
    if np.sum(x_vec) == -6:
        return 20
    if np.sum(x_vec) == 6:
        return 1
    if np.sum(x_vec) == -4 and (mat_th == -4 or mat_th == 4):
        return 19
    if np.sum(x_vec) == 4 and (mat_th == 4 or mat_th == -4):
        return 2
    if np.sum(x_vec) == -4 and mat_th == 0:
        if (np.sum(mat, axis=0) == 3).any():
            return 18
        else:
            return 17
    if np.sum(x_vec) == 4 and mat_th == 0:
        if (np.sum(mat, axis=0) == 3).any():
            return 3
        else:
            return 4
    if np.sum(x_vec) == -2 and mat_th == -2:
        return 16
    if np.sum(x_vec) == 2 and mat_th == 2:
        return 5
    if np.sum(x_vec) == -2 and (mat_th == -2 or mat_th == 2):
        return 15
    if np.sum(x_vec) == 2 and (mat_th == -2 or mat_th == 2):
        return 6
    if np.sum(x_vec) == -2 and (mat_th == -6 or mat_th == 6):
        return 14
    if np.sum(x_vec) == 2 and (mat_th == -6 or mat_th == 6):
        return 7
    if np.sum(x_vec) == 0:
        if mat_th == 0:
            if (np.sum(mat, axis=0) == 3).any():
                return 8
            else:
                return 9
        if mat_th == 4:
            return 10
        if mat_th == -4:
            if (np.sum(mat, axis=0) == -3).any():
                return 11
            else:
                return 12
        if mat_th == -12:
            return 13


def get_classes(x_vec):
    classes = np.array([check_class(x) for x in x_vec])
    return classes


def plot_probs_gibbs(j_list = np.round(np.arange(0, 1, 0.0005), 4)):
    # j_list = np.arange(0, 1, 0.01)
    # j_list = np.round(np.arange(0, 1, 0.0005), 4) 
    # j_list = [0, 0.25, 0.7, 0.9]   
    # fig, ax = plt.subplots(nrows=2, ncols=2)
    # ax = ax.flatten()
    # figmu, axmu = plt.subplots(ncols=len(j_list))
    init_state = np.random.choice([-1, 1], 8)
    # init_state = np.random.uniform(0, 1, 8)
    burn_in = 5000
    n_iter = 50000
    probs = np.empty((len(j_list), 22))
    clas_values = np.arange(0, 23)
    probs_data = 'C:/Users/alexg/Onedrive/Escritorio/phd/gibbs_sampling_necker/data_folder/probsmat.npy'
    os.makedirs(os.path.dirname(probs_data), exist_ok=True)
    if os.path.exists(probs_data):
        probs = np.load(probs_data, allow_pickle=True)
    else:
        for j_ind, j in enumerate(j_list):
            if j_ind % 10 == 0:
                print('J = ' + str(j) + ' , {}%'.format(j_ind / len(j_list)))
            # j = 0.3
            states_mat = gibbs_samp_necker(init_state=init_state,
                                           burn_in=burn_in, n_iter=n_iter, j=j)
            # mu_vec = get_mu_from_mat(states_mat)
            # possible_states, states_count = np.unique(states_mat, axis=0, return_counts=True)
            # possible_mu, mu_count = np.unique(mu_vec, return_counts=True)
            # axmu[j_ind].plot(possible_mu, mu_count / (n_iter-burn_in), marker='o', linestyle='')
            # axmu[j_ind].set_title('J = ' + str(j))
            # axmu[j_ind].set_xlabel(r'$\mu$')
            # axmu[j_ind].set_ylabel(r'$P(\mu)$')
            classes = get_classes(states_mat)
            # possible_states, states_count = np.unique(classes, return_counts=True)
            states_count, _ = np.histogram(classes, bins=clas_values)
            probs[j_ind, :] = states_count / (n_iter-burn_in)
            # ax[j_ind].plot(possible_states, states_count / (n_iter-burn_in), marker='o', linestyle='')
            # ax[j_ind].set_title('J = ' + str(j))
            # ax[j_ind].set_xlabel(r'state')
            # ax[j_ind].set_ylabel(r'prob')
        np.save('C:/Users/alexg/Onedrive/Escritorio/phd/gibbs_sampling_necker/data_folder/probsmat_binom.npy', probs)
    plt.figure()
    im = plt.imshow(np.flipud(probs), aspect='auto', cmap='inferno', vmax=1)
    plt.yticks(np.arange(0, len(j_list), len(j_list)/10),
               j_list[np.arange(0, len(j_list), len(j_list)//10)][::-1])
    plt.ylabel(r'J')
    plt.xticks(clas_values[:-1])
    plt.xlabel('States')
    plt.colorbar(im, fraction=0.04, shrink=0.6, orientation='vertical',
                 label='State prob.')
    plt.title('Simulation')


def c_vs_mu(j_list, ax):
    init_state = np.random.choice([-1, 1], 8)
    # init_state = np.random.uniform(0, 1, 8)
    burn_in = 5000
    n_iter = 50000
    cmat = np.empty((len(j_list), 22))
    clas_values = np.arange(0, 23)
    cmat_data = 'C:/Users/alexg/Onedrive/Escritorio/phd/gibbs_sampling_necker/data_folder/c_mat.npy'
    os.makedirs(os.path.dirname(cmat_data), exist_ok=True)
    if os.path.exists(cmat_data):
        probs = np.load(cmat_data, allow_pickle=True)
    else:
        for j_ind, j in enumerate(j_list):
            if j_ind % 10 == 0:
                print('J = ' + str(j) + ' , {}%'.format(j_ind / len(j_list)))
            # j = 0.3
            states_mat = gibbs_samp_necker(init_state=init_state,
                                           burn_in=burn_in, n_iter=n_iter, j=j)
            mu_vec = get_mu_from_mat(states_mat)
            # possible_states, states_count = np.unique(states_mat, axis=0, return_counts=True)
            # possible_mu, mu_count = np.unique(mu_vec, return_counts=True)
            # axmu[j_ind].plot(possible_mu, mu_count / (n_iter-burn_in), marker='o', linestyle='')
            # axmu[j_ind].set_title('J = ' + str(j))
            # axmu[j_ind].set_xlabel(r'$\mu$')
            # axmu[j_ind].set_ylabel(r'$P(\mu)$')
            classes = get_classes(states_mat)
            # possible_states, states_count = np.unique(classes, return_counts=True)
            states_count, _ = np.histogram(classes, bins=clas_values)
            cmat[j_ind, :] = states_count / (n_iter-burn_in)
            # ax[j_ind].plot(possible_states, states_count / (n_iter-burn_in), marker='o', linestyle='')
            # ax[j_ind].set_title('J = ' + str(j))
            # ax[j_ind].set_xlabel(r'state')
            # ax[j_ind].set_ylabel(r'prob')
        np.save('C:/Users/alexg/Onedrive/Escritorio/phd/gibbs_sampling_necker/data_folder/cmat_data.npy', cmat)


def get_analytical_probs_all(j_list):
    analytical_probs = np.zeros((len(j_list), 22))
    combs = list(itertools.product([-1, 1], repeat=8))
    combs = np.array(combs, dtype=np.float64)
    for j_ind, j in enumerate(j_list):
        cte = np.zeros((22))
        for i_x, x_vec in enumerate(combs):
            x_vec = np.array(x_vec, dtype=np.float64)
            classes = check_class(x_vec)
            cte[classes] += 1
        class_count = []
        for i_x, x_vec in enumerate(combs):
            x_vec = np.array(x_vec, dtype=np.float64)
            classes = check_class(x_vec)
            if classes in class_count:
                continue
            else:
                class_count.append(classes)
                prob_an = get_analytical_prob(x_vec, j)*cte[classes]
                analytical_probs[j_ind, classes] = prob_an
        cte_norm = np.sum(analytical_probs[j_ind, :])
        analytical_probs[j_ind, :] /= cte_norm
    return analytical_probs


def plot_analytical_prob(j_list = np.round(np.arange(0, 1, 0.0005), 4)):
    analytical_probs_data =\
        'C:/Users/alexg/Onedrive/Escritorio/phd/gibbs_sampling_necker/data_folder/probsmat_analytical.npy'
    clas_values = np.arange(0, 23)
    os.makedirs(os.path.dirname(analytical_probs_data), exist_ok=True)
    if os.path.exists(analytical_probs_data):
        analytical_probs = np.load(analytical_probs_data, allow_pickle=True)
    else:
        analytical_probs = get_analytical_probs_all(j_list)
            # cte[classes] += prob_an
        np.save('C:/Users/alexg/Onedrive/Escritorio/phd/gibbs_sampling_necker/data_folder/probsmat_analytical.npy',
                analytical_probs)
    fig, ax = plt.subplots(1)
    im = ax.imshow(np.flipud(analytical_probs), aspect='auto', cmap='inferno',
                   vmax=1)
    ax.set_yticks(np.arange(0, len(j_list), len(j_list)/10),
                  j_list[np.arange(0, len(j_list), len(j_list)//10)][::-1])
    ax.set_ylabel(r'J')
    ax.set_xticks(clas_values[:-1])
    ax.set_xlabel('States')
    plt.colorbar(im, fraction=0.04, shrink=0.6, orientation='vertical',
                 label='State prob.', ax=ax)
    ax.set_title('Analytical solution')


if __name__ == '__main__':
    plot_probs_gibbs()
    plot_analytical_prob()
    plot_k_vs_mu_analytical(eps=0)
