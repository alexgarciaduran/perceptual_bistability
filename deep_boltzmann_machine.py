# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:41:19 2024

@author: alexg
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
# ---GLOBAL VARIABLES
pc_name = 'alex'
if pc_name == 'alex':
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/gibbs_sampling_necker/data_folder/'  # Alex

elif pc_name == 'alex_CRM':
    DATA_FOLDER = 'C:/Users/agarcia/Desktop/phd/necker/data_folder/'  # Alex CRM

dataset_path = DATA_FOLDER + '/necker_images/'

 
class RBM:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.weights = np.random.randn(num_visible, num_hidden)
        self.fast_weights = np.random.randn(num_visible, num_hidden)
        self.visible_bias = np.zeros(num_visible) + np.random.randn()*0.01
        self.fast_visible_bias = np.zeros(num_visible) + np.random.randn()*0.01
        self.hidden_bias = np.zeros(num_hidden) + np.random.randn()*0.01
        self.fast_hidden_bias = np.zeros(num_hidden) + np.random.randn()*0.01
 
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
 
    def gibbs_sampling(self, visible_data, k=1):
        """
        Samples hidden probs from visible data and then samples
        visible data from hidden state.
        """
        for _ in range(k):
            hidden_probs = self.sigmoid(np.dot(visible_data, self.weights) + self.hidden_bias)
            hidden_states = np.random.rand(len(visible_data), self.num_hidden) < hidden_probs
            visible_probs = self.sigmoid(np.dot(hidden_states, self.weights.T) + self.visible_bias)
            visible_data = np.random.rand(len(visible_data), self.num_visible) < visible_probs
        return visible_data, hidden_probs
 
    def contrastive_divergence_k(self, data, learning_rate=0.1, k=1, epochs=10):
        """
        Performs CD-k.
        """
        for _ in range(epochs):
            positive_hidden_probs = self.sigmoid(np.dot(data, self.weights) + self.hidden_bias)
            positive_hidden_states = np.random.rand(len(data), self.num_hidden) < positive_hidden_probs
            positive_associations = np.dot(data.T, positive_hidden_probs)
 
            recon_data, recon_hidden_probs = self.gibbs_sampling(data, k)
            negative_visible_probs = recon_data
            negative_hidden_probs = recon_hidden_probs
            negative_associations = np.dot(recon_data.T, negative_hidden_probs)
 
            self.weights += learning_rate * (positive_associations - negative_associations)
            self.visible_bias += learning_rate * np.mean(data - negative_visible_probs, axis=0)
            self.hidden_bias += learning_rate * np.mean(positive_hidden_probs - negative_hidden_probs, axis=0)


def data_load(dataset_path, batch_size=10):
    images = glob.glob(dataset_path + '*')
    np.random.shuffle(images)
    images = images[:batch_size]
    train_data = np.zeros((batch_size, 5904))
    for im_ind, image in enumerate(images):
        img = plt.imread(image)[:, :, 0]
        img[img == 1] = 0
        img[img > 0] = 1
        train_data[im_ind, :] = img.flatten()
    # train_data = torch.from_numpy(train_data).to(torch.float32)
    return train_data


train_data = data_load(dataset_path, batch_size=100)
print(f'Train data shape (# batches, batch size, x, y): {train_data.shape}')


rbm = RBM(train_data.shape[1], train_data.shape[1])
rbm.contrastive_divergence_k(train_data, k=1, epochs=20)
visible, hidden_probs = rbm.gibbs_sampling(train_data, k=1)
plt.figure()
plt.imshow(visible[-1, :].reshape(72, 82))
plt.figure()
plt.imshow(hidden_probs[-1, :].reshape(72, 82))
