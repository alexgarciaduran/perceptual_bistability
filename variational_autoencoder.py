# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 16:02:01 2024

@author: alexg
"""

import torch
import torch.nn as nn

import numpy as np
import glob
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.optim import Adam


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# ---GLOBAL VARIABLES
pc_name = 'alex'
if pc_name == 'alex':
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/gibbs_sampling_necker/data_folder/'  # Alex

elif pc_name == 'alex_CRM':
    DATA_FOLDER = 'C:/Users/agarcia/Desktop/phd/necker/data_folder/'  # Alex CRM

dataset_path = DATA_FOLDER + '/necker_images/'

cuda = False
DEVICE = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} 



class Encoder(nn.Module):
    """
        A simple implementation of Gaussian MLP Encoder and Decoder
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
        
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var, z

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


def data_load(dataset_path, batch_size=10):
    images = glob.glob(dataset_path + '*')
    np.random.shuffle(images)
    size_x = len(images)//batch_size + 0**(len(images) % batch_size == 0)
    train_data = np.empty((size_x, batch_size, 72, 82))
    b_ind = 0
    for im_ind, image in enumerate(images):
        img = plt.imread(image)[:, :, 0]
        img[img == 1] = 0
        img[img > 0] = 1
        if im_ind % batch_size == 0 and im_ind != 0:
            b_ind += 1
        train_data[b_ind, im_ind % batch_size, :, :] = img
    train_data = torch.from_numpy(train_data).to(torch.float32)
    return train_data[:-1], train_data[-1]


def VAE(dataset_path, batch_size=10, x_dim=5904, hidden_dim=200,
        latent_dim=8, lr=1e-3, epochs=30):
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)
    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
    BCE_loss = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    print("Start training VAE...")
    model.train()
    train_data, test_data = data_load(dataset_path, batch_size=10)
    print(f'Train data shape (# batches, batch size, x, y): {train_data.shape}')
    for epoch in range(epochs):
        overall_loss = 0
        # zarr = np.empty((batch_size, latent_dim))
        for batch_idx, x in enumerate(train_data):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()
    
            x_hat, mean, log_var, z = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
    print("Finish!!")
    # model.eval()
    # _, test_data = data_load(dataset_path, batch_size=10)
    x = test_data.view(batch_size, x_dim)
    x_hat, _, _, z = model(x)
    for i in range(4):
        show_image(x, idx=i)
        show_image(x_hat, idx=i)
    plt.figure()
    z = z.detach().numpy()
    im = plt.imshow(1/(1+np.exp(-z)))
    plt.ylabel('Image index')
    plt.xlabel(r'Hidden variable index, $i$')
    plt.colorbar(im, label=r'$\sigma(z_i)$')


def show_image(x, idx, batch_size=10):
    x = x.view(batch_size, 72, 82)
    plt.figure()
    plt.imshow(x[idx].detach().numpy(),
               cmap='gist_gray')
    plt.title(f'Test image index: {idx}')



if __name__ == '__main__':
    VAE(dataset_path, batch_size=10, x_dim=5904, hidden_dim=1200,
        latent_dim=8, lr=1e-4, epochs=50)
    # lr was 1e-3
