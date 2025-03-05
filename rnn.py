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
import ops
import seaborn as sns

DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/fitting/data/'  # Alex
SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/fitting/parameters/'  # Alex


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
        self.W_ih = nn.Parameter(torch.randn(hidden_size, input_size) * 0.05+0.4)
        self.W_hh = nn.Parameter((torch.randn(hidden_size, hidden_size) * 0.2))
        # self.m_1 = nn.Parameter(torch.randn(hidden_size) * 0.1+0.3)
        # self.m_2 = nn.Parameter(torch.randn(hidden_size) * 0.1+0.3)
        # self.W_hh = nn.Parameter(torch.outer(self.m_1, self.m_2) + torch.randn(hidden_size, hidden_size)*0.1)
        self.W_hh_bias = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01+0.5)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        # Initialize weights and biases for Linear layer
        self.W = nn.Parameter(torch.randn(output_size, hidden_size) * 0.1+0.5)
        self.b = nn.Parameter(torch.zeros(output_size))

    def forward(self, x, context=0):
        seq_len, batch_size = x.size()
        h = torch.randn(batch_size, self.hidden_size)*0.01 + 0.5 # Initial hidden state

        # Apply the RNN step by step
        out = torch.zeros((batch_size, seq_len, self.hidden_size))
        for t in range(seq_len):
            x_t = x[t, :]
            x_t = x_t.unsqueeze(-1)
            # shape_mtmul = (torch.matmul(x_t, self.W_ih[:, 0, context].T)).shape
            c0 = torch.matmul(x_t, self.W_ih.T)
            # c1 = torch.bmm(h.unsqueeze(1), self.W_hh[:, :, context].T).squeeze(1)
            c1 = torch.matmul(2*h-1, self.W_hh.T)
            c2 = (context*torch.matmul(2*h-1, self.W_hh_bias.T).T).T
            h =  nn.functional.sigmoid(c0 +\
                 c1 + c2 +  self.b_h)  #   + torch.randn(self.hidden_size)*0.05
            out[:, t, :] = h

        # Apply the linear layer
        x = torch.matmul(out, self.W.T) + self.b
        return x[:, :, 0], out


def create_stim(evidence, sigma=0.03, dt=0.1, trial_len=4):
    time = np.arange(0, trial_len, dt)
    evvals = np.zeros((len(evidence), len(time)))
    for i_e, e in enumerate(evidence):
        evvals[i_e] = e + np.random.randn(len(time))*sigma
    return evvals.T


def training(hidden_size=8, training_kwargs={'dt': 0.1,
                                             'lr': 1e-2,
                                             'n_epochs': 8000,
                                             'batch_size': 540,
                                             'seq_len': 100}):
    # Define network instance from the Net class
    net = Net(input_size=1,
              hidden_size=hidden_size,
              output_size=1)
    # Define loss: instance of the CrossEntropyLoss class
    criterion = nn.MSELoss()

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=training_kwargs['lr'])
    num_epochs = training_kwargs['n_epochs']
    batch_size = training_kwargs['batch_size']

    # It is initialized to zero and then monitored over training interations
    running_loss = 0.0
    df = load_data(data_folder=DATA_FOLDER, sub='s_11')
    df['coupling'] = df['pShuffle'].replace(to_replace=[0, 70, 100],
                                            value= [1, 0.3, 0])
    evidence = df.evidence.values/10
    dataset = create_stim(evidence, sigma=0.05, dt=training_kwargs['dt'],
                          trial_len=training_kwargs['seq_len'])
    ctext = df.coupling.values
    # conf = np.clip((df.confidence.values+1)/2, 0.001, 0.999)
    # labs = 0.5*np.log(conf / (1-conf))
    labs = (df.confidence.values+1)/2
    losslist = []
    plt.figure()
    plt.yscale('log')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # net.W_m2.requires_grad = False
    # net.W_m1.requires_grad = False
    # net.W_hh.requires_grad = False
    for i in range(num_epochs):
        # if i < 4000:
        #     idx = np.random.choice(np.where(ctext != 1)[0], batch_size)
        #     net.W_hh_bias.requires_grad = False
        # else:
        #     # idx = np.random.choice(np.arange(0, len(labs), 1), batch_size)
        #     idx = np.random.choice(np.where(ctext != 0)[0], batch_size)
        #     net.W_hh_bias.requires_grad = True
        #     net.W_hh.requires_grad = False
        idx = np.random.choice(np.arange(0, len(labs), 1), batch_size)
        # get inputs and labels and pass them to the GPU
        inputs = dataset[:, idx]
        inputs = torch.from_numpy(inputs).type(torch.float)
        # inputs = inputs.view(-1, len(labels))
        if i > -1:
            # context = torch.from_numpy(ctext[idx]).type(torch.int)
            context = torch.from_numpy(ctext[idx]).type(torch.float)
        # else:
        #     context = torch.from_numpy(np.repeat(0, batch_size)).type(torch.int)
        labels = torch.from_numpy(labs[idx] + np.random.randn(batch_size)*0.02).type(torch.float)
        # labels = (labels.view(-1, 1) + torch.randn(inputs.shape[0]).view(-1, inputs.shape[0])*0.05).T
        # print shapes of inputs and labels
        if i == 0:
            print('inputs shape: ', inputs.shape)
            print('labels shape: ', labels.shape)
            print('Max labels: ', labels.max())
        # we need zero the parameter gradients to re-initialize and avoid they accumulate across epochs
        optimizer.zero_grad()

        # FORWARD PASS: get the output of the network for a given input
        outputs, _ = net(inputs, context)

        # compute loss with respect to the labels
        loss = criterion(outputs[:, -1], labels)

        # compute gradients
        loss.backward()

        # update weights
        optimizer.step()
        # print average loss over last 200 training iterations and save the current network
        running_loss += loss.item()
        losslist.append(loss.item())
        if i % 200 == 199:
            print('{:d} loss: {:0.5f}'.format(i + 1, running_loss / 200))
            running_loss = 0.0
            plt.plot(losslist, color='k')
            plt.pause(0.05)
            # save current state of network's parameters
            torch.save(net.state_dict(), SV_FOLDER + 'net.pth')
    np.save(SV_FOLDER + 'stimulus.npy', dataset)
    print('Finished Training')


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
                h = nn.functional.sigmoid(x_t*net.W_ih.T + torch.matmul(2*h-1, net.W_hh_bias.T)*context +
                    torch.matmul(2*h-1, net.W_hh.T) + net.b_h
                    + torch.randn(hidden_size)*0.0)  # + torch.matmul(h, net.W_hh_bias.T)*context
                outputs.append(h.unsqueeze(0))

            out = torch.cat(outputs, dim=0)

            # Apply the linear layer
            x = torch.matmul(out, net.W.T) + net.b
            action_pred, hidden = x.T.detach().numpy()[0, 0], out.detach().numpy()[:, 0]
            activity.append(np.array(hidden))
            conf_rnn.append(action_pred[-1])
            accuracy.append(np.sign(action_pred[-1]-0.5) == gt[i])
        activity = np.array(activity)
        plt.figure()
        plt.xlabel('Conf. data')
        plt.ylabel('Conf. RNN')
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
        plt.figure()
        sns.lineplot(x=evidence, y=accuracy, color='k', marker='o')
        fig, ax = plt.subplots(ncols=2)
        im0 = ax[0].imshow(net.W_hh)
        plt.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(net.W_hh_bias)
        plt.colorbar(im1, ax=ax[1])


def sigmoid(x):
    return 1/(1+np.exp(-x))


def low_rank_rnn_sims(n_units=20, time_end=10, dt=1e-2, eta=0.1, gain=1,
                      coh=0.1, sigma=0.05, noise=0.1, random_matrix=False):
    stim = np.random.randn(n_units)*sigma + coh
    J_ij = np.random.randn(n_units, n_units)
    if random_matrix:
        m1 = np.random.randn(n_units)
        m1 = m1-np.mean(m1)
        m2 = np.random.randn(n_units)
        m2 = m2-np.mean(m2)
    else:
        m1 = np.concatenate((np.ones(n_units//2), -np.ones(n_units//2)))
        m2 = -m1
        # m1 = np.cos(np.arange(n_units))
        # m2 = np.sin(np.arange(n_units))
    M_12 = np.outer(m1, m2)
    C_mat = eta*J_ij + M_12
    np.fill_diagonal(C_mat, 0)
    readout_vec = np.random.randn(n_units)*0.2
    readout_vec -= np.mean(readout_vec)
    time = np.arange(0, time_end, dt)
    # np.concatenate((np.ones(n_units//2), -np.ones(n_units//2)))
    x = np.random.randn(n_units)*0.05
    x1_list = []
    x2_list = []
    x_all = np.zeros((n_units, len(time)))
    x_readout = np.zeros((len(time)))
    for i_t, t in enumerate(time):
        x += dt*(sigmoid(2*gain*np.matmul(C_mat, x) + 2*stim)-(x+1)/2) + np.random.randn(n_units)*np.sqrt(dt)*noise
        x_1 = np.dot(x, m1)
        x_2 = np.dot(x, m2)
        x1_list.append(x_1)
        x2_list.append(x_2)
        x_all[:, i_t] = x
        x_readout[i_t] = np.matmul(readout_vec, x)
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


if __name__ == '__main__':
    training(hidden_size=8, training_kwargs={'dt': 1e-1,
                                             'lr': 1e-3,
                                             'n_epochs': 10000,
                                             'batch_size': 10,
                                             'seq_len': 4})
    load_network(sv_folder=SV_FOLDER, hidden_size=8)
    # load_network(sv_folder=SV_FOLDER, hidden_size=8, plot_coup=0)
