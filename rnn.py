# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 16:09:32 2025

@author: alexg
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pylab as pl
import ops
import seaborn as sns
import scipy
import scipy.ndimage as ndimage
from scipy.signal import correlate2d
from skimage.metrics import structural_similarity as ssim

DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/fitting/data/'  # Alex
SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/fitting/parameters/'  # Alex


THETA = np.array([[0 ,1 ,1 ,0 ,1 ,0 ,0 ,0], [1, 0, 0, 1, 0, 1, 0, 0],
                  [1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1],
                  [0, 0, 1, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 1, 1, 0]])


def load_data(data_folder, sub='s_11'):
    files = glob.glob(data_folder + '*')
    df_0 = pd.DataFrame()
    for i in range(len(files)):
        df = pd.read_csv(files[i])
        df = df.dropna(subset=['confidence'])
        df['subject'] = 's_' + str(i+1)
        df_0 = pd.concat((df_0, df))
    all_df = df_0
    if sub != 'all':
        all_df = all_df.loc[all_df.subject == sub]
    return all_df


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 ncontexts=3):
        super(Net, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases for RNN
        self.W_ih = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1 + 0.5)
        self.W_hh = nn.Parameter((torch.randn(hidden_size, hidden_size) * 0.4))
        # self.m_1 = nn.Parameter(torch.randn(hidden_size) * 0.1)
        # self.m_2 = nn.Parameter(torch.randn(hidden_size) * 0.1)
        # self.m_3 = nn.Parameter(torch.randn(hidden_size) * 0.1)
        # self.m_4 = nn.Parameter(torch.randn(hidden_size) * 0.1)
        # self.W_hh = torch.outer(self.m_1, self.m_2)+torch.outer(self.m_3, self.m_4)
        # self.W_hh = nn.Parameter(torch.outer(self.m_1, self.m_2) + torch.randn(hidden_size, hidden_size)*0.1)
        self.W_hh_bias = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1 + 0.1)
        # self.b_ctext_bias = nn.Parameter(torch.randn(hidden_size) * 0.1)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        # self.noise = torch.tensor([0.1]*hidden_size)  # nn.Parameter(torch.rand(hidden_size)*0.1)

        # Initialize weights and biases for Linear layer
        self.W = nn.Parameter(torch.randn(output_size, hidden_size) * 0.3)
        self.b = nn.Parameter(torch.zeros(output_size))

    def forward(self, x, context=0):
        seq_len, batch_size = x.size()
        h = torch.randn(batch_size, self.hidden_size)*0.1 # Initial hidden state
        # self.W_hh = torch.outer(self.m_1, self.m_2)+torch.outer(self.m_3, self.m_4)
        # Apply the RNN step by step
        out = torch.zeros((batch_size, seq_len, self.hidden_size))
        for t in range(seq_len):
            x_t = x[t, :]
            x_t = x_t.unsqueeze(-1)
            c0 = torch.matmul(x_t, self.W_ih.T)
            c1 = (context*torch.matmul(h, self.W_hh_bias.T).T).T
            h = torch.tanh(torch.matmul(h, self.W_hh.T) + c0 + c1 + self.b_h) + torch.randn(batch_size, self.hidden_size)*0.1
            out[:, t, :] = h

        # Apply the linear layer
        x = nn.functional.sigmoid(torch.matmul(out, self.W.T) + self.b)
        return x[:, :, 0], out


def create_stim(evidence, sigma=0.03, dt=0.1, trial_len=4):
    time = np.arange(0, trial_len, dt)
    evvals = np.zeros((len(evidence), len(time)))
    for i_e, e in enumerate(evidence):
        evvals[i_e] = e + np.random.randn(len(time))*sigma
    return evvals.T


def quick_simul_mf(dataset, ctext, num_simuls, dt=0.1):
    sims = np.zeros_like(dataset)
    for i in range(num_simuls):
        x = np.random.rand()
        xl = [x]
        for n in range(dataset.shape[0]-1):
            x = x+dt*(sigmoid(2*ctext[i]*3*(2*x-1)+2*dataset[n, i])-x)/0.5 + np.random.randn()*np.sqrt(dt/0.5)*0.025
            xl.append(x)
        sims[:, i] = xl
    labs = np.clip(sims, 0, 1)
    return labs


def training(hidden_size=8, training_kwargs={'dt': 0.1,
                                             'lr': 1e-2,
                                             'n_epochs': 8000,
                                             'batch_size': 200,
                                             'seq_len': 10},
             num_simuls=5000):
    # Define network instance from the Net class
    net = Net(input_size=1,
              hidden_size=hidden_size,
              output_size=1)
    # Define loss: instance of the CrossEntropyLoss class
    criterion = nn.MSELoss()
    # criterion = nn.KLDivLoss()
    # criterion = torch.nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=training_kwargs['lr'],
                                 weight_decay=1e-4)
    num_epochs = training_kwargs['n_epochs']
    batch_size = training_kwargs['batch_size']

    # It is initialized to zero and then monitored over training interations
    running_loss = 0.0
    # df = load_data(data_folder=DATA_FOLDER, sub='s_11')
    # df['coupling'] = df['pShuffle'].replace(to_replace=[0, 70, 100],
    #                                         value= [1, 0.3, 0])
    # evidence = df.evidence.values/10
    dt = training_kwargs['dt']
    seqlen = training_kwargs['seq_len']
    evidence = np.random.choice([-1, -0.5, -0.25, 0, 0.25, 0.5, 1], num_simuls)
    dataset = create_stim(evidence, sigma=0.05, dt=training_kwargs['dt'],
                          trial_len=training_kwargs['seq_len'])
    ctext = np.random.choice(np.arange(0, 0.8, 0.05), num_simuls)
    # conf = np.clip((df.confidence.values+1)/2, 0.001, 0.999)
    # labs = 0.5*np.log(conf / (1-conf))
    labs = quick_simul_mf(dataset, ctext, num_simuls, dt=dt)
    losslist = []
    mat_ctx_list = []
    mat_hidden_list = []
    mat_input_list = []
    # net.W_m2.requires_grad = False
    # net.W_m1.requires_grad = False
    # net.W_hh.requires_grad = False
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(11.5, 8))
    ax = ax.flatten()
    ax[0].set_xlabel('Epoch'); ax[1].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss'); ax[1].set_ylabel('<W_ctext>')
    ax[2].set_xlabel('Loss'); ax[2].set_ylabel('<W_ctext>')
    ax[3].set_xlabel('Epoch'); ax[4].set_xlabel('Epoch')
    ax[3].set_ylabel('<W_input>'); ax[4].set_ylabel('<W_hidden>')
    ax[5].set_xlabel('Loss'); ax[5].set_ylabel('<W_input>')
    ax[5].set_title('Correlation'); ax[2].set_title('Correlation')
    fig.tight_layout()
    for i in range(num_epochs):
        idx = np.random.choice(np.arange(0, len(labs), 1), batch_size, replace=True)
        # get inputs and labels and pass them to the GPU
        inputs = dataset[:, idx]
        inputs = torch.from_numpy(inputs).type(torch.float)
        # inputs = inputs.view(-1, len(labels))
        context = torch.from_numpy(ctext[idx]).type(torch.float)
        labels = torch.from_numpy(labs[:, idx]).type(torch.float).T
        if i == 0:
            print('inputs shape: ', inputs.shape)
            print('labels shape: ', labels.shape)
            print('Max labels: ', labels.max())
        # we need zero the parameter gradients to re-initialize and avoid they accumulate across epochs
        optimizer.zero_grad()

        # FORWARD PASS: get the output of the network for a given input
        outputs, _ = net.forward(inputs, context)
        # compute loss with respect to the labels
        loss = criterion(outputs, labels)
        l1_lambda = 1e-4
        l1_norm = torch.sum(torch.abs(net.W_hh))/hidden_size**2
        l2_lambda = 1e-4
        l2_norm = torch.sum(net.W_ih**2)/hidden_size

        loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm

        # compute gradients
        loss.backward()

        # update weights
        optimizer.step()
        # print average loss over last 200 training iterations and save the current network
        running_loss += loss.item()
        mat_ctx_list.append(net.W_hh_bias.detach().numpy().mean())
        mat_input_list.append(net.W_ih.detach().numpy().mean())
        mat_hidden_list.append(net.W_hh.detach().numpy().mean())
        losslist.append(loss.item())
        if i % 200 == 199:
            print('{:d} loss: {:0.5f}'.format(i + 1, running_loss / 200))
            running_loss = 0.0
            ax[0].plot(losslist, color='k')
            ax[1].plot(mat_ctx_list, color='k')
            ax[3].plot(mat_input_list, color='k')
            ax[4].plot(mat_hidden_list, color='k')
            ax[5].plot(losslist, mat_input_list, color='k', marker='o', linestyle='', alpha=0.3)
            ax[2].plot(losslist, mat_ctx_list, color='k', marker='o', linestyle='', alpha=0.3)
            corr = np.corrcoef(losslist, mat_ctx_list)[0][1]
            ax[2].set_title('Correlation: ' + f'{corr: .3f}')
            ax[0].set_yscale('log')
            ax[2].set_xscale('log')
            ax[5].set_xscale('log')
            corr = np.corrcoef(losslist, mat_input_list)[0][1]
            ax[5].set_title('Correlation: ' + f'{corr: .3f}')
            plt.pause(0.05)
            # save current state of network's parameters
            torch.save(net.state_dict(), SV_FOLDER + 'net.pth')
    np.save(SV_FOLDER + 'stimulus.npy', dataset)
    print('Finished Training')


def test_network_simulations(sv_folder=SV_FOLDER, hidden_size=12,
                             training_kwargs={'dt': 0.1, 'lr': 1e-2,
                                              'n_epochs': 8000,
                                              'batch_size': 200,
                                              'seq_len': 10},
                             num_simuls=2000):
    # load configuration file - we might have run the training on the cloud and might now open the results locally
    with torch.no_grad():
        net = Net(input_size=1,
                  hidden_size=hidden_size,
                  output_size=1)
    
        # load the trained network's weights from the saved file
        net.load_state_dict(torch.load(sv_folder + 'net.pth', weights_only=True))
        dt = training_kwargs['dt']
        seqlen = training_kwargs['seq_len']
        evidence = np.random.choice([-1, -0.5, -0.25, 0, 0.25, 0.5, 1], num_simuls)
        dataset = create_stim(evidence, sigma=0.05, dt=training_kwargs['dt'],
                              trial_len=training_kwargs['seq_len'])
        ctext = np.random.choice([0.05, 0.3, 0.5], num_simuls)
        # conf = np.clip((df.confidence.values+1)/2, 0.001, 0.999)
        labs = quick_simul_mf(dataset, ctext, num_simuls, dt=dt)
        activity = []
        conf_rnn = []
        accuracy = []
        for i in range(num_simuls):
            inputs = dataset[:, i]
            inputs = torch.from_numpy(inputs).type(torch.float)
            # context = torch.from_numpy(np.array(ctext[i])).type(torch.int)
            context = torch.from_numpy(np.array(ctext[i])).type(torch.float)
            h = torch.randn(1, net.hidden_size)*0.1  # Initial hidden state

            # Apply the RNN step by step
            outputs = []
            for t in range(len(inputs)):
                x_t = inputs[t]
                h = torch.tanh(x_t*net.W_ih.T + torch.matmul(h, net.W_hh.T)+
                    torch.matmul(h, net.W_hh_bias.T)*context + net.b_h) + torch.randn(hidden_size)*0.1  # + torch.matmul(h, net.W_hh_bias.T)*context
                outputs.append(h.unsqueeze(0))

            out = torch.cat(outputs, dim=0)

            # Apply the linear layer
            x = nn.functional.sigmoid(torch.matmul(out[-1], net.W.T) + net.b)
            action_pred, hidden = x[0].detach().numpy()[0], out.detach().numpy()[:, 0]
            activity.append(np.array(hidden))
            conf_rnn.append(action_pred)
            accuracy.append(np.sign(action_pred-0.5) == np.sign(labs[-1, i]-0.5))
        activity = np.array(activity)
        plt.figure(); plt.xlabel('Conf. data'); plt.ylabel('Conf. RNN'); plt.ylim(0, 1); plt.xlim(0, 1)
        plt.plot(labs[-1], conf_rnn, marker='o', color='k', linestyle='')
        ax = plt.figure().add_subplot(projection='3d')
        # fig, ax = plt.subplots(1)
        colors = ['k', 'b', 'r']
        fig3, ax3 = plt.subplots(ncols=3, figsize=(12, 4))
        for i_ct, ctx in enumerate(np.sort(np.unique(ctext))):
            sns.kdeplot(x=np.array(conf_rnn)[evidence == 0], hue=ctext[evidence == 0], ax=ax3[i_ct], linestyle='--')
            sns.kdeplot(x=labs[-1, evidence == 0], hue=ctext[evidence == 0], ax=ax3[i_ct])
            idx_ctext = (ctext == ctx)*(evidence == 0)  # *(np.sign(labs[i]-0.5) != 0)
            print(sum(idx_ctext))
            activity_reshape = np.reshape(activity[idx_ctext], (-1, activity[idx_ctext].shape[1]))
            pca = PCA(n_components=3)
            pca.fit(activity_reshape)
            # a = activity[info[condition] == value].mean(axis=0)
            a = pca.transform(activity_reshape)  # (N_time, N_PC)
            print(a.shape)
            ax.plot(a[5:, 0], a[5:, 1], a[5:, 2], color=colors[i_ct], alpha=0.3)
            ax.plot(a[5, 0], a[5, 1], a[5, 2], color=colors[i_ct], alpha=0.3,
                    marker='o')
            ax.plot(a[-1, 0], a[-1, 1], a[-1, 2], color=colors[i_ct], alpha=0.3,
                    marker='x')
        plt.figure()
        sns.lineplot(x=evidence, y=accuracy, hue=ctext, color='k', marker='o')
        plt.xlabel('Sensory evidence')
        plt.ylabel('Accuracy')
        fig, ax = plt.subplots(ncols=2)
        im0 = ax[0].imshow(net.W_hh.detach().numpy())
        plt.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(net.W_hh_bias.detach().numpy())
        plt.colorbar(im1, ax=ax[1])


def load_network(sv_folder=SV_FOLDER, hidden_size=12, plot_coup=1):
    # load configuration file - we might have run the training on the cloud and might now open the results locally
    with torch.no_grad():
        net = Net(input_size=1,
                  hidden_size=hidden_size,
                  output_size=1)
    
        # load the trained network's weights from the saved file
        net.load_state_dict(torch.load(sv_folder + 'net.pth'))
        dataset = np.load(sv_folder + 'stimulus.npy')
        df = load_data(data_folder=DATA_FOLDER, sub='s_11')
        df['coupling'] = df['pShuffle'].replace(to_replace=[0, 70, 100],
                                                value= [1, 0.3, 0])
        evidence = df.evidence.values/10
        real_coh = np.mean(dataset, axis=0)
        gt = np.sign(real_coh)
        ctext = df.coupling.values
        # conf = np.clip((df.confidence.values+1)/2, 0.001, 0.999)
        # labs = 0.5*np.log(conf / (1-conf))
        labs = (df.confidence.values+1)/2
        activity = []
        conf_rnn = []
        accuracy = []
        for i in range(len(gt)):
            inputs = dataset[:, i]
            inputs = torch.from_numpy(inputs).type(torch.float)
            # context = torch.from_numpy(np.array(ctext[i])).type(torch.int)
            context = torch.from_numpy(np.array(ctext[i])).type(torch.float)
            h = torch.randn(1, net.hidden_size)*0.0 + 0.5  # Initial hidden state

            # Apply the RNN step by step
            outputs = []
            for t in range(len(inputs)):
                x_t = inputs[t]
                h = nn.functional.relu(
                    nn.functional.relu(x_t*net.W_ih.T + torch.matmul(h, net.W_hh.T))+
                    torch.matmul(h, net.W_hh_bias.T)*context + net.b_h
                    + torch.randn(hidden_size)*0.05)  # + torch.matmul(h, net.W_hh_bias.T)*context
                outputs.append(h.unsqueeze(0))

            out = torch.cat(outputs, dim=0)

            # Apply the linear layer
            x = nn.functional.sigmoid(torch.matmul(out, net.W.T) + net.b)
            action_pred, hidden = x.T.detach().numpy()[0, 0], out.detach().numpy()[:, 0]
            activity.append(np.array(hidden))
            conf_rnn.append(action_pred[-1])
            accuracy.append(np.sign(action_pred[-1]-0.5) == gt[i])
        activity = np.array(activity)
        plt.figure()
        plt.xlabel('Conf. data'); plt.ylabel('Conf. RNN'); plt.ylim(0, 1); plt.xlim(0, 1)
        plt.plot(labs, conf_rnn, marker='o', color='k', linestyle='')
        idx_ctext = np.where((ctext == plot_coup)*(evidence != 0))[0]
        plt.plot(labs[idx_ctext], np.array(conf_rnn)[idx_ctext], marker='o', color='r', linestyle='')
        idx_ctext = np.where((ctext == plot_coup)*(evidence == 0))[0]
        plt.plot(labs[idx_ctext], np.array(conf_rnn)[idx_ctext], marker='x', color='r', linestyle='')
        ax = plt.figure().add_subplot(projection='3d')
        # fig, ax = plt.subplots(1)
        colors = ['k', 'b', 'r']
        for i_ev, ev in enumerate([-1/10, 0, 1/10]):
            idx_ctext = np.where((ctext == plot_coup)*(evidence == ev))[0]
            activity_reshape = np.reshape(activity[idx_ctext], (-1, activity[idx_ctext].shape[-1]))
            pca = PCA(n_components=3)
            pca.fit(activity_reshape)
            # a = activity[info[condition] == value].mean(axis=0)
            for idx in idx_ctext:
                a = pca.transform(activity[idx])  # (N_time, N_PC)
                ax.plot(a[:, 0], a[:, 1], a[:, 2], color=colors[i_ev], alpha=0.3)
                ax.plot(a[0, 0], a[0, 1], a[0, 2], color=colors[i_ev], alpha=0.3,
                        marker='o')
                ax.plot(a[-1, 0], a[-1, 1], a[-1, 2], color=colors[i_ev], alpha=0.3,
                        marker='x')
        plt.figure()
        sns.lineplot(x=evidence, y=accuracy, color='k', marker='o')
        fig, ax = plt.subplots(ncols=2)
        im0 = ax[0].imshow(net.W_hh.detach().numpy())
        plt.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(net.W_hh_bias.detach().numpy())
        plt.colorbar(im1, ax=ax[1])


def sigmoid(x):
    return 1/(1+np.exp(-x))


def stimulus_creation(coh=0.5, p_shuffle=0, rows=6, cols=4, sigma=0):
    x = np.arange(0, cols)
    n_shuffle = int(p_shuffle * cols)
    y = np.arange(0, rows)
    X, Y = np.meshgrid(x, y)
    V = np.sin(X / (cols-1) * np.pi) + np.random.randn(X.shape[0], X.shape[1])*sigma
    if n_shuffle != 0:
        for j in range(rows):
            indices = np.random.choice(cols, n_shuffle, replace=False)
            # Shuffle selected elements
            shuffled_values = V[j, indices].copy()
            np.random.shuffle(shuffled_values)
            # Create new array with shuffled values
            shuffled_arr = V[j].copy()
            shuffled_arr[indices] = shuffled_values
            V[j, :] = shuffled_arr
    U = np.zeros_like(Y)
    return np.array([V, U]), np.array([-V, U])


def gain_computation(coh=0.5, p_shuffle=0, rows=6, cols=4, sigma=1):
    """
    Computes gain matrix as correlations between stimulus.
    Stimulus should be a (2 x rows x cols) velocity field.
    """
    (v0, u0), (v1, u1) = stimulus_creation(coh=coh, p_shuffle=p_shuffle,
                                           rows=rows, cols=cols, sigma=0.)
    # coherence_v0 = structure_tensor(v0, v0)
    # coherence_v1 = structure_tensor(v1, v1)
    coherence_v0 = compute_local_ssim(v0, window_size=5, sigma=sigma)
    coherence_v1 = compute_local_ssim(v1, window_size=5, sigma=sigma)
    corrs = np.concatenate((coherence_v0.flatten(), coherence_v1.flatten()))
    return np.round(np.outer(corrs, corrs), 4)


def structure_tensor(Vx, Vy, sigma=1):
    """
    Compute the structure tensor for local motion coherence.
    """
    # Compute spatial derivatives
    Ix = ndimage.sobel(Vx, axis=1)  # dVx/dx
    Iy = ndimage.sobel(Vx, axis=0)  # dVx/dy
    Jx = ndimage.sobel(Vy, axis=1)  # dVy/dx
    Jy = ndimage.sobel(Vy, axis=0)  # dVy/dy
    
    # Structure tensor elements
    Txx = ndimage.gaussian_filter(Ix**2, sigma)
    Tyy = ndimage.gaussian_filter(Jy**2, sigma)
    Txy = ndimage.gaussian_filter(Ix * Jy, sigma)
    
    # Compute coherence measure (eigenvalues of structure tensor)
    trace = Txx + Tyy
    det = Txx * Tyy - Txy**2
    coherence = np.sqrt(trace**2 - 4 * det) / (trace + 1e-8)  # Avoid division by 0
    
    return coherence


def exp_kernel(n_units, tau=4):
    x = np.arange(n_units//2)
    kernel = np.concatenate((np.exp(-(x-1)[:len(x)//2]/tau), (np.exp(-x[:len(x)//2]/tau))[::-1]))
    kernel[0] = 0
    return kernel


def rnn_connectivity_learning_sims(time_end=10, dt=1e-2, eta=0.1, rows=6, columns=5,
                                   p_shuffle=0, coh=0.1, sigma=0.05, noise=0.1,
                                   lr=0.1, n_memory=4, gain=None, loadmat=False,
                                   plot=False, feedback=False):
    if gain is None:
        gain = gain_computation(p_shuffle=p_shuffle, rows=rows, cols=columns, sigma=1)
        if plot:
            plt.figure()
            im = plt.imshow(gain, vmin=0, vmax=1)
            plt.colorbar(im)
    n_units = gain.shape[0]
    np.random.seed(4)
    time = np.arange(0, time_end+dt, dt)
    stim = np.random.randn(n_units, len(time))*sigma + coh
    J_ij = np.random.randn(n_units, n_units)
    ker = exp_kernel(n_units, tau=10)
    ker = ker/np.sum(ker)
    m_11 = scipy.linalg.circulant(ker)
    m_12 = m_11
    m_21 = m_12
    m_22 = m_11
    M_12 = np.concatenate((np.column_stack((m_11, m_12)), np.column_stack((m_21, m_22))))
    M_12 = (M_12 + M_12.T)/2
    C_mat = eta*J_ij + M_12
    np.fill_diagonal(C_mat, 0)
    act_ker = np.concatenate((np.ones(n_units//2), -np.ones(n_units//2)))
    # if loadmat:
    #     C_mat = np.load(SV_FOLDER + 'cmat.npy')
    x = np.random.randn(n_units)*0.1
    x_all = np.zeros((n_units, len(time)))
    cmat_memory = np.zeros((n_units, n_units, n_memory))
    cmat_memory[:, :, 0] = C_mat
    tv = [0]
    c = 0
    indices = np.linspace(0, len(time)-1, n_memory, dtype=int)
    for i_t, t in enumerate(time):
        xm = 2*np.matmul(gain*C_mat, x) + 2*stim[:, i_t]
        if feedback:
            fbck = -feedback*np.sign(x[0])
            if t >= time_end//2 and t < time_end//2+5:
                xm += fbck
        x += dt*(sigmoid(xm)-(x+1)/2) + np.random.randn(n_units)*np.sqrt(dt)*noise
        x_all[:, i_t] = x*act_ker
        # hebbian
        if lr > 0:
            C_mat = C_mat + oja_update(W=C_mat, W0=M_12, x=x, eta=lr)*dt
            np.fill_diagonal(C_mat, 0)
        # C_mat = C_mat / np.sum(np.abs(C_mat), axis=0)
        if t > 0 and i_t in indices:
            c += 1
            cmat_memory[:, :, c] = C_mat
            tv.append(t)
    # np.save(SV_FOLDER + 'cmat.npy', C_mat)
    if not plot:
        return x_all
    if plot:
        if lr == 0:
            fig, ax = plt.subplots(ncols=1)
            im = ax.imshow(C_mat, cmap='coolwarm')
        if lr > 0:
            fig, ax = plt.subplots(ncols=n_memory)
            vmin_neg = np.min(C_mat)
            vmax_pos = np.max(C_mat)
            vmin = np.min((vmin_neg, -vmax_pos))
            vmax = np.max((vmax_pos, -vmin_neg))
            for ia, a in enumerate(ax):
                im = a.imshow(cmat_memory[:, :, ia], cmap='coolwarm', vmin=vmin, vmax=vmax)
                a.set_title(f't = {tv[ia]}')
            ax_pos = ax[-1].get_position()
            ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*1.05, ax_pos.y0+ax_pos.height*0.1,
                                    ax_pos.width*0.15, ax_pos.height*0.8])
            plt.colorbar(im, cax=ax_cbar)
        fig, ax = plt.subplots(ncols=1, figsize=(6, 5))
        for n in range(n_units):
            if n < n_units // 2:
                col = 'r'
            else:
                col = 'b'
            ax.plot(time, x_all[n], alpha=0.2, color=col)
        ax.plot(time, np.mean(x_all[n_units//2:], axis=0), color='b', linewidth=3)
        ax.plot(time, np.mean(x_all[:n_units//2], axis=0), color='r', linewidth=3)
        ax.plot(time, np.mean(x_all, axis=0), color='k', linewidth=3)
        ax.set_ylim(-1.5, 1.5)
        fig.tight_layout()
        # Center the data
        X_centered = x_all - np.mean(x_all, axis=1, keepdims=True)
        # Perform PCA using SVD
        pca = PCA(n_components=10)  # Keep 10 principal components
        X_pca = pca.fit_transform(X_centered.T).T  # Projected data (shape: 10 x 1000)
        fig2, ax2 = plt.subplots(1)
        ax2.plot(X_pca[0], X_pca[1], alpha=0.7, color='k', linewidth=2.5)
        ax2.plot(X_pca[0][0], X_pca[1][0], alpha=1, color='k', marker='o')
        ax2.plot(X_pca[0][-1], X_pca[1][-1], alpha=1, color='k', marker='x')


def low_rank_rnn_sims(time_end=10, dt=1e-2, eta=0.1, rows=6, columns=5,
                      p_shuffle=0, coh=0.1, sigma=0.05, noise=0.1, random_matrix=False, lr=0.1,
                      n_memory=4, gain=None, loadmat=False):
    if gain is None:
        gain = gain_computation(p_shuffle=p_shuffle, rows=rows, cols=columns)
        plt.figure()
        plt.imshow(gain)
    n_units = gain.shape[0]
    # np.random.seed(4)
    stim = np.random.randn(n_units)*sigma + coh
    J_ij = np.random.randn(n_units, n_units)
    if random_matrix:
        m1 = np.random.randn(n_units)
        m1 = m1-np.mean(m1)
        m2 = np.random.randn(n_units)
        m2 = m2-np.mean(m2)
        M_12 = np.outer(m1, m2)
    else:
        m1 = np.concatenate((np.ones(n_units//2), -np.ones(n_units//2)))
        m2 = -m1
        M_12 = -np.outer(m1, m2)
        # m1 = np.cos(np.arange(n_units))
        # m2 = np.sin(np.arange(n_units))
    C_mat = eta*J_ij + M_12
    np.fill_diagonal(C_mat, 0)
    if loadmat:
        C_mat = np.load(SV_FOLDER + 'cmat.npy')
    # np.linalg.eigvals(C_mat)
    readout_vec = np.random.randn(n_units)*0.2
    readout_vec -= np.mean(readout_vec)
    time = np.arange(0, time_end+dt, dt)
    # np.concatenate((np.ones(n_units//2), -np.ones(n_units//2)))
    x = np.random.randn(n_units)*0.05
    x1_list = []
    x2_list = []
    x_all = np.zeros((n_units, len(time)))
    x_readout = np.zeros((len(time)))
    cmat_memory = np.zeros((n_units, n_units, n_memory))
    cmat_memory[:, :, 0] = C_mat
    tv = [0]
    c = 0
    indices = np.linspace(0, len(time)-1, n_memory, dtype=int)
    for i_t, t in enumerate(time):
        xm = 2*np.matmul(gain*C_mat, x) + 2*stim
        x += dt*(sigmoid(xm)-(x+1)/2) + np.random.randn(n_units)*np.sqrt(dt)*noise
        x_1 = np.dot(x, m1)
        x_2 = np.dot(x, m2)
        x1_list.append(x_1)
        x2_list.append(x_2)
        x_all[:, i_t] = x
        x_readout[i_t] = np.matmul(readout_vec, x)
        # hebbian
        C_mat += gain*oja_update(W=C_mat, x=x, eta=lr)*dt
        np.fill_diagonal(C_mat, 0)
        # C_mat = C_mat / np.sum(np.abs(C_mat), axis=0)
        if t > 0 and i_t in indices:
            c += 1
            cmat_memory[:, :, c] = C_mat
            tv.append(t)
    np.save(SV_FOLDER + 'cmat.npy', C_mat)
    fig, ax = plt.subplots(ncols=n_memory)
    vmin_neg = np.min(C_mat)
    vmax_pos = np.max(C_mat)
    vmin = np.min((vmin_neg, -vmax_pos))-3/2
    vmax = np.max((vmax_pos, -vmin_neg))+3/2
    for ia, a in enumerate(ax):
        im = a.imshow(cmat_memory[:, :, ia], cmap='coolwarm', vmin=vmin, vmax=vmax)
        a.set_title(f't = {tv[ia]}')
    ax_pos = ax[-1].get_position()
    ax_cbar = fig.add_axes([ax_pos.x0+ax_pos.width*1.05, ax_pos.y0+ax_pos.height*0.1,
                            ax_pos.width*0.15, ax_pos.height*0.8])
    plt.colorbar(im, cax=ax_cbar)
    plt.figure()
    plt.plot(x1_list, x2_list)
    plt.plot(x1_list[0], x2_list[0], marker='o')
    plt.plot(x1_list[-1], x2_list[-1], marker='x')
    plt.figure()
    plt.plot(time, x2_list)
    plt.plot(time, x1_list)
    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
    for n in range(n_units):
        ax[0].plot(time, x_all[n], alpha=0.2)
    ax[0].plot(time, np.mean(x_all, axis=0), color='k', linewidth=3)
    ax[0].plot(time, x_readout, color='k', linewidth=3, linestyle='--')
    ax[0].set_ylim(-1.5, 1.5)
    # ylim = plt.ylim()
    gamma = np.array(x2_list)-np.array(x1_list)
    ax[1].plot(time, gamma, color='r', linewidth=3)
    ax[1].plot(time, x_readout, color='k', linewidth=3, linestyle='--')
    # plt.ylim(ylim)
    fig.tight_layout()


def compute_autocorrelation(image, neighborhood_size=5):
    """Compute local autocorrelation of an image."""
    kernel = np.ones((neighborhood_size, neighborhood_size))
    auto_corr = correlate2d(image, kernel, mode='same', boundary='symm')
    return auto_corr / np.max(auto_corr)  # Normalize


def oja_update(W, W0, x, eta=0.01, alpha=10):
    """
    Vectorized Oja's learning rule.
    
    W: (N, N) weight matrix
    x: (N,) activity vector
    eta: learning rate
    """
    # Outer product term (Hebbian learning)
    hebbian_term = np.outer(x, x)

    # Weight decay term (Oja's normalization)
    decay_term = np.diag(W @ x)[:, None] * x  # (N, 1) * (1, N) = (N, N)

    # Update weights
    deltaW = eta * W0 * (hebbian_term - decay_term[:, 0])

    return deltaW


def compute_local_entropy(image, window_size=5):
    """
    Compute local entropy using a moving window.
    
    Parameters:
        image (ndarray): Grayscale image.
        window_size (int): Size of the local window.
    
    Returns:
        entropy_img (ndarray): Local entropy map.
    """
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    return entropy(image, disk(window_size // 2))


def compute_local_ssim(image, window_size=7, sigma=1):
    """
    Compute local SSIM map of an image.
    
    Parameters:
        image (ndarray): Grayscale image.
        window_size (int): Size of the local comparison window.
    
    Returns:
        ssim_map (ndarray): SSIM values computed over local windows.
    """
    from scipy.ndimage import gaussian_filter
    # Compute SSIM against a blurred version of the image itself, which gives a self-similarity map
    blurred_image = gaussian_filter(image, sigma=sigma)
    _, ssim_map = ssim(image, blurred_image, full=True, win_size=window_size)
    return ssim_map


def max_eigenvalue_gain_matrix_plot(p_sh_list=np.arange(0, 1, 1e-2), nreps=100,
                                    rows=8, columns=7):
    mateigvalsgain = np.zeros((len(p_sh_list), nreps))
    mateigvalsm12 = np.zeros((len(p_sh_list), nreps))
    mateigvalsmult = np.zeros((len(p_sh_list), nreps))
    ker = exp_kernel(rows*columns*2, tau=10)
    ker = ker/np.sum(ker)
    m_11 = scipy.linalg.circulant(ker)
    m_12 = m_11
    m_21 = m_12
    m_22 = m_11
    M_12 = np.concatenate((np.column_stack((m_11, m_12)), np.column_stack((m_21, m_22))))
    M_12 = (M_12 + M_12.T)/2
    for i_p, p_shuffle in enumerate(p_sh_list):
        for n in range(nreps):
            gain = gain_computation(p_shuffle=p_shuffle, rows=rows, cols=columns, sigma=1)
            evgain = np.real(np.linalg.eigvals(gain)).max()
            evm12 = np.real(np.linalg.eigvals(M_12)).max()
            evmult = np.real(np.linalg.eigvals(gain*M_12)).max()
            mateigvalsgain[i_p, n] = evgain
            mateigvalsm12[i_p, n] = evm12
            mateigvalsmult[i_p, n] = evmult
    plt.figure()
    mngain = np.mean(mateigvalsgain, axis=1)
    # errgain = np.std(mateigvalsgain, axis=1)/nreps
    mnm12 = np.mean(mateigvalsm12, axis=1)
    # errm12 = np.std(mateigvalsm12, axis=1)/nreps
    mnmult = np.mean(mateigvalsmult, axis=1)
    # errmult = np.std(mateigvalsmult, axis=1)/nreps
    plt.plot(p_sh_list, mngain, color='k', linewidth=3, label='gain')
    plt.plot(p_sh_list, mnm12, color='b', linewidth=3, label='kernel matrix')
    plt.plot(p_sh_list, mnmult, color='r', linewidth=3, label='multiplication')
    plt.plot(p_sh_list, mnm12*mngain, color='r', linewidth=3, label='multiplication',
             linestyle='--')
    plt.legend(frameon=False)
    # plt.fill_between(p_sh_list, mn-err, mn+err, color='k', alpha=0.5)
    plt.ylabel('Maximum eigenvalue of gain matrix')
    plt.xlabel('Pshuffle')


def pca_different_stims(b_list=np.arange(-0.5, 0.6, 0.25).round(3), p_sh = [0, 1],
                        time_end=80, dt=1e-2, noise_stim=0, noise_simul=0,
                        noise_matrix=0, fbk=0):
    fig2, ax2 = plt.subplots(ncols=len(p_sh), figsize=(15, 5))
    fig, ax = plt.subplots(ncols=len(p_sh), figsize=(15, 5))
    colormap = pl.cm.coolwarm(np.linspace(0., 1, len(b_list)))
    K = len(b_list)
    for p, pshuf in enumerate(p_sh):
        print('Shuffle: ' + str(pshuf))
        for i_b, b in enumerate(b_list):
            x_all = rnn_connectivity_learning_sims(time_end=time_end,
                                                   dt=dt, eta=noise_matrix,
                                                   rows=8, columns=7,
                                                   p_shuffle=pshuf, coh=b,
                                                   sigma=noise_stim, noise=noise_simul,
                                                   lr=0., n_memory=4, plot=False,
                                                   feedback=fbk)
            # x_all = x_all[:x_all.shape[0] // 2]
            X_centered = x_all - np.mean(x_all, axis=0, keepdims=True)
            if i_b == 0:
                arr_act = X_centered
            else:
                arr_act = np.column_stack((arr_act, X_centered))
        # Perform PCA
        pca = PCA(n_components=10)  # Keep 10 principal components
        X_pca = pca.fit_transform(arr_act.T).T  # Projected data
        X_pca_conditions = np.split(X_pca, K, axis=1)
        time = np.arange(0, time_end+dt, dt)
        plt.figure()
        plt.bar(np.arange(10), pca.explained_variance_ratio_)
        plt.yscale('log')
        for k in range(K):
            ax2[p].plot(X_pca_conditions[k][0], X_pca_conditions[k][1],
                        color=colormap[k])
            for i in range(0, len(time) // 4, 100):
                ax2[p].plot(X_pca_conditions[k][0][i], X_pca_conditions[k][1][i],
                            color=colormap[k], marker='o', markersize=(i*0.002+0.8))
            ax[p].plot(time, X_pca_conditions[k][0], color=colormap[k])
            ax2[p].plot(X_pca_conditions[k][0][0], X_pca_conditions[k][1][0],
                        color=colormap[k], marker='o')
            ax2[p].plot(X_pca_conditions[k][0][-1], X_pca_conditions[k][1][-1],
                        color=colormap[k], marker='x')
            # ax2[p].plot(X_pca[0], X_pca[1], alpha=0.7, color=colormap[i_b],
            #             linewidth=3)
            # ax2[p].plot(X_pca[0][0], X_pca[1][0], alpha=1, color=colormap[i_b], marker='o')
            # ax2[p].plot(X_pca[0][-1], X_pca[1][-1], alpha=1, color=colormap[i_b], marker='x')
        ax2[p].set_title('p_shuffle: ' + str(pshuf))
        ax[p].set_title('p_shuffle: ' + str(pshuf))
        ax2[p].set_xlabel('PC-1')
        ax2[p].set_ylabel('PC-2')
        ax[p].set_ylabel('PC-1')
        ax[p].set_xlabel('Time')
    fig.tight_layout()
    fig2.tight_layout()


if __name__ == '__main__':
    training_dict = {'dt': 1e-1, 'lr': 1e-3,
                      'n_epochs': 5000, 'batch_size': 100, 'seq_len': 4}
    training(hidden_size=40, training_kwargs=training_dict,
              num_simuls=50000)
    test_network_simulations(sv_folder=SV_FOLDER, hidden_size=40,
                                 training_kwargs=training_dict,
                                 num_simuls=10000)
    # load_network(sv_folder=SV_FOLDER, hidden_size=8)
    # load_network(sv_folder=SV_FOLDER, hidden_size=40, plot_coup=0)
    # load_network(sv_folder=SV_FOLDER, hidden_size=40, plot_coup=1)
    # low_rank_rnn_sims(time_end=400, dt=1e-2, eta=0.0, gain=np.random.randn(8, 8)*0.1+4,
    #                   coh=0., sigma=0.0, noise=0., random_matrix=False, lr=0.1)
    # low_rank_rnn_sims(time_end=400, dt=1e-2, eta=0.0, gain=np.random.randn(8, 8)*0.1+0.1,
    #                   coh=0., sigma=0.0, noise=0., random_matrix=False, lr=0.1)

    # low_rank_rnn_sims(time_end=300, dt=1e-2, eta=0., rows=6, columns=5,
    #                   p_shuffle=1, coh=0., sigma=0.0, noise=0., random_matrix=False,
    #                   lr=0.5, n_memory=4)
    # rnn_connectivity_learning_sims(time_end=100, dt=1e-2, eta=0., rows=8, columns=7,
    #                                p_shuffle=0, coh=0., sigma=0.0, noise=0.1,
    #                                lr=0., n_memory=4, plot=True, feedback=1)
    # pca_different_stims(b_list=np.arange(-0.5, 0.6, 0.25).round(3),
    #                     time_end=80, dt=1e-2, noise_stim=0.0, noise_simul=0.,
    #                     noise_matrix=0., p_sh=[0, 0.2, 0.5, 0.7, 1])
    # rnn_connectivity_learning_sims(time_end=400, dt=1e-2, eta=0.05, rows=6, columns=6,
    #                                p_shuffle=1, coh=0., sigma=0.0, noise=0.1,
    #                                lr=0.1, n_memory=4)
