# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:45:48 2023

@author: alexg
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import itertools
import seaborn as sns
import os
import scipy.stats as stats
import matplotlib.pylab as pl



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




# ---GLOBAL VARIABLES
pc_name = 'alex'
if pc_name == 'alex':
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/gibbs_sampling_necker/data_folder/'  # Alex

elif pc_name == 'alex_CRM':
    DATA_FOLDER = 'C:/Users/agarcia/Desktop/phd/necker/data_folder/'  # Alex CRM

# THETA mat

THETA = np.array([[0 ,1 ,1 ,0 ,1 ,0 ,0 ,0], [1, 0, 0, 1, 0, 1, 0, 0],
                  [1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1],
                  [0, 0, 1, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 1, 1, 0]])


def return_theta(rows=10, columns=5, layers=2):
    theta = np.zeros(((rows)*(columns)*(layers), (rows)*(columns)*(layers)))
    for i in range((rows)*(columns)*(layers)):
        theta[i, i+int((-i)**(i == (rows*columns*layers-1)))] = 1
        theta[i, i-1] = 1
        if (i % columns == 0 and i != 0) or (i > (rows*columns*layers - columns)):
            theta[i, 0] = 1
            theta[i, columns + i - rows*columns*layers] = 1
        else:
            theta[i, i+1] = 1
            theta[i, i+columns] = 1
                
        # theta[i, i+int((-1)**(i >= (rows*columns*layers-columns)))*columns] = 1
        # theta[i, i+int((-1)**(i >= columns))*columns] = 1
        # theta[i, i+int(rows*columns*layers/2*(-1)**(i > rows*columns/2))] = 1
    plt.imshow(theta)
    # TODO: finish (connections for any graph with rowsxcolumnsxlayers structure)
                
    


def get_theta_signed(j):
    th = np.copy(THETA)
    for t in range(8):
        ind = get_connections(t)
        th[t, ind] = [-1 if ((i >= 4) and (t <= 3)
                             or (i <= 3) and (t >= 4)) else 1 for i in ind]
    return th*j


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


def k_val(x_vec, j_mat, stim=0):
    return np.matmul(np.matmul(x_vec.T, j_mat), x_vec)/2 + np.sum(stim*x_vec)


def change_prob(x_vect, x_vect1, j, stim=0):
    j_mat = np.abs(mat_theta(x_vect, x_vect, j))
    j_mat1 = np.abs(mat_theta(x_vect1, x_vect1, j))
    return sigmoid(j*(-k_val(x_vect, j_mat, stim=stim) +
                      k_val(x_vect1, j_mat1, stim=stim)))


def mat_theta(x_vect_1, x_vect_2, j):
    mat = np.zeros((len(x_vect_1), len(x_vect_2)))
    for i in range(len(x_vect_1)):
        connections = get_connections(i)
        for con in connections:
            mat[i, con] = x_vect_1[i] * x_vect_2[con] * j
    return mat


def transition_matrix(J, C):
    state_ks = [12*J,6*J,4*J,0,0,2*J,-2*J,-6*J,4*J,0,4*J,-4*J,-4*J,-12*J,-6*J,-2*J,2*J,0,0,4*J,6*J,12*J]
    T = np.zeros((22,22))

    for i, k_int_state in enumerate(state_ks):
        for j, k_next_state in enumerate(state_ks):
            T[i,j] = (C[i,j]/8)*sigmoid(state_ks[j] - state_ks[i]) #k estat final menys inicial
            
    np.fill_diagonal(T, [1-sum(row) for row in T])
    
    return T


def gibbs_samp_necker(init_state, burn_in, n_iter, j, stim=0):
    x_vect = init_state
    states_mat = np.empty((n_iter-burn_in, 8))
    states_mat[:] = np.nan
    for i in range(n_iter):
        node = np.random.choice(np.arange(1, 9), p=np.repeat(1/8, 8))
        x_vect1 = np.copy(x_vect)
        x_vect1[node-1] = -x_vect1[node-1]
        prob = change_prob(x_vect, x_vect1, j, stim=stim)
        # val_bool = np.random.choice([False, True], p=[1-prob, prob])
        val_bool = np.random.binomial(1, prob, size=None)
        if val_bool:
            x_vect[node-1] = -x_vect[node-1]
        if i >= burn_in:
            states_mat[i-burn_in, :] = x_vect
    return states_mat


def mean_prob_gibbs(j, ax=None, burn_in = 1000, n_iter = 10000, wsize=100,
                    node=None, stim=0):
    init_state = np.random.choice([-1, 1], 8)
    states_mat = gibbs_samp_necker(init_state=init_state,
                                   burn_in=burn_in, n_iter=n_iter, j=j,
                                   stim=stim)
    states_mat = (states_mat + 1) / 2
    conv_states_mat = np.copy(states_mat)
    if wsize != 1:
        for i in range(8):
            conv_states_mat[:, i] = np.convolve(conv_states_mat[:, i],
                                                np.ones(wsize)/wsize, mode='same')
    if node is None:
        mean_acc_nodes = np.nanmean(conv_states_mat, axis=1)
    else:
        mean_acc_nodes = conv_states_mat[:, node]
    if ax is not None:
        ax.plot(mean_acc_nodes, label=j)
    if ax is None:
        return mean_acc_nodes


def plot_mean_prob_gibbs(j_list=np.arange(0, 1.05, 0.05), burn_in=1000, n_iter=10000,
                         wsize=1, stim=0, node=None):
    fig, ax_tot = plt.subplots(ncols=2)
    ax = ax_tot[0]
    mean_nod = np.empty((len(j_list), n_iter-burn_in))
    mean_nod[:] = np.nan
    allmeans = np.empty((len(j_list)))
    for ind_j, j in enumerate(j_list):
        mean_nod[ind_j, :] = mean_prob_gibbs(j, ax=None, burn_in=burn_in, n_iter=n_iter,
                                             wsize=wsize, node=node, stim=stim)
        allmeans[ind_j] = np.nanmean(np.abs(mean_nod[ind_j, :]*2-1))
    im = ax.imshow(np.flipud(mean_nod), aspect='auto', cmap='seismic')
    ax.set_yticks(np.arange(0, len(j_list), len(j_list)//2),
                  j_list[np.arange(0, len(j_list), len(j_list)//2)][::-1])
    plt.colorbar(im, label=r'$\frac{1}{8}\sum_i^8 {P(x_i = 1, t)}$', ax=ax,
                 orientation='horizontal')
    ax.set_ylabel('J')
    ax.set_xlabel('Iter (time)')
    ax = ax_tot[1]
    ax.plot(j_list, allmeans, color='k')  # np.abs(allmeans*2-1)
    # ax.plot(j_list, 1-allmeans, color='r')
    ax.set_xlabel('J')
    ax.set_ylabel(r'$<P(state \in \{-1, 1\})>_t$', fontsize=10)


def plot_duration_dominance_gamma_fit(j, burn_in=1000, n_iter=100000):
    plt.figure()
    vals_gibbs = mean_prob_gibbs(j, ax=None, burn_in=burn_in, n_iter=n_iter,
                                 wsize=1, node=None)
    orders = rle(vals_gibbs)
    time = orders[0][(orders[2] <= 0.05) + (orders[2] >= 0.95)]
    sns.histplot(time, kde=True, label='Simulations', stat='density', fill=False,
                 color='k', bins=np.arange(1, max(time)+10, 2))
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(time)
    fix_loc_exp, fit_lmb = stats.expon.fit(time)
    x = np.linspace(min(time), max(time), 1000)
    y = stats.gamma.pdf(x, a=fit_alpha, scale=fit_beta, loc=1)
    y_exp = stats.expon.pdf(x, loc=fix_loc_exp, scale=fit_lmb)
    plt.text(75, 0.02, r'$\alpha = ${}'.format(np.round(fit_alpha, 2)) 
             + '\n'+ r'$\beta = ${}'.format(np.round(fit_beta, 2)),
             fontsize=11)
    plt.plot(x, y, label='Gamma distro. fit', color='k', linestyle='--')
    plt.plot(x, y_exp, label='Expo. distro. fit', color='r', linestyle='--')
    plt.xlabel('Dominance duration')
    plt.xlim(-5, 105)
    plt.legend()


def rle(inarray):
        """ run length encoding. Partial credit to R rle function. 
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        ia = np.asarray(inarray)                # force numpy
        n = len(ia)
        if n == 0: 
            return (None, None, None)
        else:
            y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            return(z, p, ia[i])


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


def compute_C(data_folder):
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
    np.save(data_folder + 'c_mat.npy', c_mat)


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


def plot_probs_gibbs(data_folder, j_list=np.round(np.arange(0, 1, 0.0005), 4)):
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
    probs_data = data_folder + 'probsmat.npy'
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
        np.save(data_folder + 'probsmat_binom.npy', probs)
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


def c_vs_mu(j_list, ax, data_folder):
    init_state = np.random.choice([-1, 1], 8)
    # init_state = np.random.uniform(0, 1, 8)
    burn_in = 5000
    n_iter = 50000
    cmat = np.empty((len(j_list), 22))
    clas_values = np.arange(0, 23)
    cmat_data = data_folder + 'c_mat.npy'
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
        np.save(data_folder + 'c_mat.npy', cmat)


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


def plot_analytical_prob(data_folder, j_list = np.round(np.arange(0, 1, 0.0005), 4)):
    analytical_probs_data =\
        data_folder + 'probsmat_analytical.npy'
    clas_values = np.arange(0, 23)
    os.makedirs(os.path.dirname(analytical_probs_data), exist_ok=True)
    if os.path.exists(analytical_probs_data):
        analytical_probs = np.load(analytical_probs_data, allow_pickle=True)
    else:
        analytical_probs = get_analytical_probs_all(j_list)
            # cte[classes] += prob_an
        np.save(data_folder + 'probsmat_analytical.npy',
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


def tanh_act_bistab(n_iter, weight_prev_state, weight_noise, stim_state):
    # SIMPLEST MODEL FOR BISTABILITY INDEP. OF STIM.
    state = [0.5]
    vals = [0.5]
    dsdt = np.gradient(stim_state)
    for j in range(1, n_iter):
        x = np.random.randn()*weight_noise\
            + weight_prev_state*state[j-1] + dsdt[j]
        val = sigmoid(x)  # x/np.sqrt(1+x**2)
        vals.append(val)
        state.append(np.sign(val))
    return state, vals


def plot_prob_basic_model_coupling(n_iter, wpslist=np.linspace(0, 3, 100),
                                   stim_state=1, weight_noise=0):
    """
    Stim. independent bistability generation. Single neuron with recurrent connection
    and external noise.
    P_{t+1} = (P_t) · w + w_n · N(0, 1) + dS/dt , --> dS/dt = 0
    P_{t+1} = F(S, P_t) = g(S) + f(P_t) --> S cte. --> offset (baseline) cte
    
    if w_n = 0 --> mean field model
    """
    fig, ax = plt.subplots(ncols=1)
    stim_state = np.repeat(stim_state, n_iter)
    valslist = []
    stdlist = []
    for iw, weight_prev_state in enumerate(wpslist):
        _, vals = tanh_act_bistab(n_iter=n_iter,
                                  weight_prev_state=weight_prev_state,
                                  weight_noise=weight_noise, stim_state=stim_state)
        valslist.append(vals[-1])  # np.abs(vals)
        vals = np.array(vals)
        stdlist.append(np.nanstd((vals+1)/2))
        # vals = np.round(vals, 5)
        # plt.plot(np.repeat(iw, len(np.unique(vals))), np.unique(vals), color='k',
        #          marker='o', linestyle='', markersize=1)
    indw = \
        np.where(np.abs(np.array(valslist)-0.5) ==
                 np.min(np.abs(np.array(valslist)-0.5)))[0][0]
    ax.axvline(wpslist[indw], color='grey', linestyle='--')
    ax.text(1.5, 0.5, r'$w_{change} =$' + str(np.abs(np.round(wpslist[indw], 2))))
    # ax[1].axvline(wpslist[indw], color='grey', linestyle='--')
    ax.plot(wpslist, valslist, color='k')
    ax.plot(wpslist, 1-np.array(valslist), color='r')
    ax.set_ylabel('Prob. x=1')
    ax.set_xlabel(r'Coupling strength, $w$')
    ax.set_title(r'$P_{t+1} = tanh(P_t * w + \xi), \;\; \xi \sim \mathcal{N}(\mu=0, \sigma=1)$')
    # ax[1].set_xlabel(r'Coupling strength, $w$')
    # ax[1].set_ylabel('Std(P(x=1))')
    # ax[1].plot(wpslist, stdlist, color='k')


def prob_markov_chain_between_states(n_iter=int(1e6)):
    # init_state = np.random.choice([-1, 1])
    init_state = 0
    tau = 8000  # equivalent to J=?, tau=8000 equivalent to J=1
    eps = 1-np.exp(-1/tau)
    ps = 1-eps
    pt = eps
    chain = [init_state]
    # p_list = []
    transition_matrix = np.array(((ps, pt), (pt, ps)))
    mu_list = []
    for i in range(1, n_iter):
        if chain[i-1] == 0:
            change = np.random.choice([0, 1], p=transition_matrix[0])
            # p_list.append(transition_matrix[1][change])
        if chain[i-1] == 1:
            change = np.random.choice([0, 1], p=transition_matrix[1])
            # p_list.append(transition_matrix[1][change])
        chain.append(change)
    mu_list = np.cumsum(chain)
    mu_list_norm = mu_list / np.arange(len(mu_list))
    # p_mu_N_1 = []
    # chain = np.array(chain)
    # for i_c, stat in enumerate(chain):
    #     # p(X_N=1)*eps if prev state was 0, else p(X_N=1)*(1-eps)
    #     p_mu_N_x1 = np.mean(chain[:(i_c+1)] == 1)*transition_matrix[1][chain[i_c]]
    #     # p(X_N=0)*eps if prev state was 1, else p(X_N=0)*(1-eps)
    #     p_mu_N_x0 = np.mean(chain[:(i_c+1)] == 0)*transition_matrix[0][chain[i_c]]
    #     p_mu_N_1.append(p_mu_N_x1+p_mu_N_x0)
    p_mu_N_1_x1 = []
    chain = np.array(chain)
    for i_c, stat in enumerate(chain):
        # p(X_{N+1}=1)*eps if prev state was 0, if was 1 p(X_N=1)*(1-eps)
        p_mu_N_x1 = np.mean(chain[:(i_c+1)] == 1)*(1-eps)
        p_mu_N_x0 = np.mean(chain[:(i_c+1)] == 0)*eps
        p_mu_N_1_x1.append(p_mu_N_x1+p_mu_N_x0)
    vals_to_plot = np.logspace(4, 6, 9, dtype=int)
    colormap = pl.cm.Blues(np.linspace(0.08, 1, len(vals_to_plot)))
    fig, ax = plt.subplots(1)
    for j in range(len(vals_to_plot)):
        # ax2 = ax.twinx()
        sns.kdeplot(mu_list_norm[:vals_to_plot[j]],
                    common_norm=False, color=colormap[j],
                    ax=ax, bw_adjust=3, label=vals_to_plot[j])
        # ax2.spines['right'].set_visible(False)
        # ax2.spines['top'].set_visible(False)
    # p_N(mu) = p_N(mu, x_N=0) + p_N(mu, x_N=1)
    # p_N+1(mu, x_N=1) = p_N(mu-1, x_N=1)*(1-eps) + p_N(mu-1, x_N=0)*eps
        


def true_posterior(theta=THETA, j=0.1, stim=0):
    combs_7_vars = list(itertools.product([-1, 1], repeat=7))
    np.random.shuffle(combs_7_vars)
    x_vects_1 = [combs_7_vars[i] + (1,) for i in range(len(combs_7_vars))]
    x_vects_0 = [combs_7_vars[i] + (-1,) for i in range(len(combs_7_vars))]
    j_mat = theta
    exponent_1 = 0
    exponent_0 = 0
    for x_vec_1, x_vec_0 in zip(x_vects_1, x_vects_0):
        x_vec_1 = np.array(x_vec_1)
        x_vec_0 = np.array(x_vec_0)
        # x_vec_0[4:] *= -1
        # x_vec_1[4:] *= -1
        exponent_1 += np.exp(0.5*j*np.matmul(np.matmul(x_vec_1.T, j_mat), x_vec_1) + np.sum(stim*x_vec_1))
        exponent_0 += np.exp(0.5*j*np.matmul(np.matmul(x_vec_0.T, j_mat), x_vec_0) + np.sum(stim*x_vec_0))
    prob_x_1 = (exponent_1) / ((exponent_1)+(exponent_0))
    return prob_x_1


def sol_magnetization_hex_lattice(j_list, b):
    sol = []
    alpha = 6
    for j in j_list:
        numerator = np.exp(2*j*alpha)*np.cosh(b/2*alpha) - np.cosh(b/6*alpha)
        denominator = np.exp(2*j*alpha)*np.cosh(b/2*alpha) + 3*np.cosh(b/6*alpha)
        sol.append(np.sqrt(numerator/denominator))
    plt.plot(j_list, sol)
    


def true_posterior_stim(stim_list=np.linspace(-0.05, 0.05, 1000)):
    post_stim = []
    for stim in stim_list:
        post = true_posterior(theta=THETA, j=0.5, stim=stim)
        post_stim.append(post)
    plt.plot(stim_list, post_stim)
    

def true_posterior_plot_j(ax, color='b', j_list=np.arange(0.001, 1, 0.001), stim=0):
    post_stim = []
    for j in j_list:
        post = true_posterior(theta=THETA, j=j, stim=stim)
        post_stim.append(post)
    ax.plot(j_list, post_stim, color=color)


def occupancy_distro(state, ps, k, n=1, m=1000):
    # ps: prob of stay
    p_kn_k = scipy.special.binom(m, k)
    val = 0
    for j in range(k):
        val += scipy.special.binom(k, j)*((-1)**(j-k))*(1-ps*(m-j)/m)**n
    return p_kn_k*val


if __name__ == '__main__':
    # C matrix:\
    c_data = DATA_FOLDER + 'c_mat.npy'
    C = np.load(c_data, allow_pickle=True)

    # plot_probs_gibbs(data_folder=DATA_FOLDER)
    # plot_analytical_prob(data_folder=DATA_FOLDER)
    # plot_k_vs_mu_analytical(eps=0)
    plot_mean_prob_gibbs(j_list=np.arange(0, 1.05, 0.05), burn_in=1000, n_iter=10000,
                          wsize=1, stim=-0.1)
    t = transition_matrix(0.2, C)
