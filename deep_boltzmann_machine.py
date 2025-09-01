# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:41:19 2024

@author: alexg
"""

import numpy as np
import seaborn as sns
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import itertools
from sklearn import svm


mpl.rcParams['font.size'] = 18
plt.rcParams['legend.title_fontsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize']= 16
plt.rcParams['ytick.labelsize']= 16


# ---GLOBAL VARIABLES
pc_name = 'alex'
if pc_name == 'alex':
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/dbm/'  # Alex

elif pc_name == 'alex_CRM':
    DATA_FOLDER = 'C:/Users/agarcia/Desktop/phd/necker/data_folder/'  # Alex CRM

dataset_path = DATA_FOLDER + '/necker_images/'

 
class RBM:
    def __init__(self, num_visible, num_hidden, load_data=False):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        if load_data:
            self.weights = np.load(DATA_FOLDER + 'weights_rbm.npy')
            self.visible_bias = np.load(DATA_FOLDER + 'visible_bias_rbm.npy')
            self.hidden_bias = np.load(DATA_FOLDER + 'hidden_bias_rbm.npy')
        else:
            self.weights = np.random.randn(num_visible, num_hidden)*0.2
            self.visible_bias = np.zeros(num_visible) + np.random.randn()*0.01
            self.hidden_bias = np.zeros(num_hidden) + np.random.randn()*0.01  # np.random.randn(num_hidden)*0.1
        self.fast_weights = np.random.randn(num_visible, num_hidden)
        self.fast_visible_bias = np.zeros(num_visible) + np.random.randn()*0.01
        self.fast_hidden_bias = np.zeros(num_hidden) + np.random.randn()*0.01
        self.mean_weight = []
 
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def gibbs_sampling(self, visible_data, k=1, adapt=0., fast=False):
        """
        Samples hidden probs from visible data and then samples
        visible data from hidden state.
        """
        if fast:
            w = self.fast_weights
            b_h = self.fast_hidden_bias
            b_v = self.fast_visible_bias
        else:
            w = self.weights
            b_h = self.hidden_bias
            b_v = self.visible_bias
        for _ in range(k):
            hidden_probs = self.sigmoid(np.dot(visible_data, w) + b_h)
            hidden_states = (np.random.rand(len(visible_data), self.num_hidden) < hidden_probs)*1
            visible_probs = self.sigmoid(np.dot(hidden_states, w.T) + b_v - 
                                         adapt*visible_data)
            visible_data = (np.random.rand(len(visible_data), self.num_visible) < visible_probs)*1
        return visible_data, hidden_probs, w.T

    def contrastive_divergence_k(self, data, learning_rate=0.01, k=1, epochs=10,
                                 save_weights=False):
        """
        Performs CD-k.
        """
        for _ in range(epochs):
            positive_hidden_probs = self.sigmoid(np.dot(data, self.weights) + self.hidden_bias)
            positive_hidden_states = (np.random.rand(len(data), self.num_hidden) < positive_hidden_probs)*1
            positive_associations = np.dot(data.T, positive_hidden_probs)
 
            recon_data, recon_hidden_probs, _ = self.gibbs_sampling(data, k)
            negative_visible_probs = recon_data
            negative_hidden_probs = recon_hidden_probs
            negative_associations = np.dot(recon_data.T, negative_hidden_probs)
 
            self.weights += learning_rate * (positive_associations - negative_associations)
            self.visible_bias += learning_rate * np.mean(data - negative_visible_probs, axis=0)
            self.hidden_bias += learning_rate * np.mean(positive_hidden_probs - negative_hidden_probs, axis=0)
            if save_weights:
                self.mean_weight.append(np.mean(self.weights))

    def rates_FPCD(self, data, epsilon=1e-2, k=1, alpha=1, numiters=100):
        visible_data, hidden_probs, _ = self.gibbs_sampling(data, k)
        hidden_probs = self.sigmoid(np.dot(data, self.weights) + self.hidden_bias)
        hidden_states = np.random.rand(len(visible_data), self.num_hidden) < hidden_probs
        w_hat = np.dot(data.T, hidden_states)/data.shape[0]
        b_hat = np.mean(hidden_states, axis=0)
        c_hat = np.mean(data, axis=0)
        self.fast_weights = np.copy(self.weights)
        self.fast_visible_bias = np.copy(self.visible_bias)
        self.fast_hidden_bias = np.copy(self.hidden_bias)
        visible_all = np.copy(visible_data)
        hidden_all = np.copy(hidden_probs)
        mean_weights = []
        for _ in range(numiters):
            visdat_mean = np.zeros((visible_data.shape))
            hidden_mean = np.zeros((hidden_probs.shape))
            for _ in range(k):
                hidden_states = (np.random.rand(len(visible_data), self.num_hidden) < hidden_probs)*1
                visible_data, hidden_probs, _ = self.gibbs_sampling(visible_data, k)
                visdat_mean += visible_data/k
                hidden_mean += hidden_states/k
            self.fast_weights = alpha*self.fast_weights +\
                epsilon*(w_hat - np.dot(visible_data.T, hidden_states))
            self.fast_visible_bias = alpha*self.fast_visible_bias +\
                epsilon*np.mean(c_hat - visible_data, axis=0)
            self.fast_hidden_bias = alpha*self.fast_hidden_bias +\
                epsilon*np.mean(b_hat - hidden_states, axis=0)
            visible_all = np.row_stack((visible_all, visible_data))
            hidden_all = np.row_stack((hidden_all, hidden_probs))
            mean_weights.append(np.mean(self.fast_weights))
        return visible_all, hidden_all, mean_weights
        

def get_digit(a):
    digits = {
    "0": [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ],
    "1": [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
    ],
    "2": [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ],
    "3": [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ],
    "4": [
        [0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
    ],
    "5": [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ],
    "6": [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ],
    "7": [
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ],
    "8": [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ],
    "9": [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ],
}
    return digits[a]


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


def jaccard(digit_0, digit_1):
    overlap = np.logical_and(digit_0, digit_1)
    overlap_sum = np.sum(overlap)
    union = np.logical_or(digit_0, digit_1)
    union_sum = np.sum(union)
    return overlap_sum / union_sum


def data_digits(batch_size=10):
    dig = [str(i) for i in range(10)]
    combs = list(itertools.combinations(dig, 2))
    combs = np.array(combs)
    digits = list(combs[np.random.randint(0, len(combs), batch_size)])
    train_data = np.zeros((batch_size, 35))
    test_data = np.zeros((batch_size, 35))
    truelabs_train = np.zeros((batch_size), dtype=np.int32)
    overlap_test = []
    truelabs_test = np.zeros((2, len(digits)))
    for i_p, digit_pair in enumerate(digits):
        digit_0 = np.array(get_digit(digit_pair[0]))
        digit_1 = np.array(get_digit(digit_pair[1]))
        combination = np.clip(digit_0 + digit_1, 0, 1)
        overlap_test.append(jaccard(digit_0, digit_1))
        idx_pair = np.random.choice([0, 1])
        digs = [digit_0.flatten(), digit_1.flatten()][idx_pair]
        train_data[i_p] = digs
        test_data[i_p] = combination.flatten()
        truelabs_train[i_p] = int(digit_pair[idx_pair])
        truelabs_test[:, i_p] = digit_pair
    return train_data, test_data, truelabs_train, overlap_test, truelabs_test

#%% Get data
# train_data = data_load(dataset_path, batch_size=100)
train_data, test_data, ground_truth, overlap_test, truelabs_test = data_digits(batch_size=1000)
print(f'Train data shape (# batches, batch size, x, y): {train_data.shape}')

#%% Train RBM
rbm = RBM(train_data.shape[1], 10, load_data=True)
rbm.contrastive_divergence_k(train_data, k=1, epochs=1000, learning_rate=0.1,
                             save_weights=True)
plt.figure()
plt.plot(rbm.mean_weight)
plt.ylabel('Mean weight (assess convergence)')
visible, hidden_probs, weights = rbm.gibbs_sampling(train_data, k=1)
classifier = svm.SVC()
classifier.fit(hidden_probs, ground_truth)
print('Classifier accuracy')
print(classifier.score(hidden_probs, ground_truth))
predictions = classifier.predict(hidden_probs)

#%% Plot results
fig, ax = plt.subplots(ncols=5, nrows=5)
fig.suptitle('Train data')
ax = ax.flatten()
for i_a, a in enumerate(ax):
    a.imshow(train_data[i_a, :].reshape(7, 5), cmap='binary')  # 72, 82
fig, ax = plt.subplots(ncols=5, nrows=5)
fig.suptitle('Train data, sample from RBM')
ax = ax.flatten()
for i_a, a in enumerate(ax):
    a.set_title('T: ' + str(ground_truth[i_a]) + ', P: ' + str(predictions[i_a]), fontsize=12)
    a.imshow(visible[i_a, :].reshape(7, 5), cmap='binary')  # 72, 82
fig2, ax2 = plt.subplots(ncols=5, nrows=5)
fig2.suptitle('Train data, probs from RBM')
ax2 = ax2.flatten()
for i_a, a in enumerate(ax2):
    a.plot(hidden_probs[i_a])
visible, hidden_probs, weights = rbm.gibbs_sampling(test_data, k=20)
predictions = classifier.predict(hidden_probs)
fig, ax = plt.subplots(ncols=5, nrows=5)
fig.suptitle('Test data')
ax = ax.flatten()
for i_a, a in enumerate(ax):
    a.imshow(test_data[i_a, :].reshape(7, 5), cmap='binary')  # 72, 82
fig, ax = plt.subplots(ncols=5, nrows=5)
fig.suptitle('Test data, sample from RBM')
ax = ax.flatten()
for i_a, a in enumerate(ax):
    a.set_title('Prediction: ' + str(predictions[i_a]), fontsize=12)
    a.imshow(visible[i_a, :].reshape(7, 5), cmap='binary')  # 72, 82
fig2, ax2 = plt.subplots(ncols=5, nrows=5)
fig2.suptitle('Test data, probs from RBM')
ax2 = ax2.flatten()
for i_a, a in enumerate(ax2):
    a.plot(hidden_probs[i_a])
plt.figure()
im = plt.imshow(weights, aspect='auto')
plt.colorbar(im, label='Weights')
plt.ylabel('Hidden variable index')
plt.xlabel('Input pixel index')


#%% Test_error vs overlap
# overlap_test_prediction = []
test_per_sim = []
overlap_per_sim = []
overlap_test_vs_pred = []
# overlap_bins = np.linspace(0.15, 0.9, test_simuls)
train_data, test_data, ground_truth, overlap_test, truelabs_test = data_digits(batch_size=20000)
overlap_vals = np.sort(np.unique(overlap_test))
for n in range(len(overlap_vals)):
    train_data, test_data, ground_truth, overlap_test, truelabs_test = data_digits(batch_size=5000)
    overlap_test = np.array(overlap_test)
    test_err = []
    overlap_test_pred = []
    visible, hidden_probs, weights = rbm.gibbs_sampling(test_data, k=1)
    predictions = classifier.predict(hidden_probs)
    idxs = overlap_test == overlap_vals[n]
    print(sum(idxs))
    if sum(idxs) == 0:
        continue
    for i_im, image in enumerate(test_data[idxs]):
        if i_im == 50:
            break
        test_err.append(predictions[i_im] in truelabs_test[:, i_im])
        overlap_test_pred.append(jaccard(image, visible[idxs][i_im]))
    overlap_test_vs_pred.append(np.nanmean(overlap_test_pred))
    test_per_sim.append(np.nanmean(test_err))
    overlap_per_sim.append(np.nanmean(overlap_test[idxs]))

fig, ax = plt.subplots(ncols=3, figsize=(15, 4.5))
ax[0].plot(overlap_per_sim, test_per_sim, color='k', marker='o', linestyle='')
ax[1].plot(overlap_test_vs_pred, test_per_sim, color='k', marker='o', linestyle='')
ax[2].plot(overlap_test_vs_pred, overlap_per_sim, color='k', marker='o', linestyle='')
ax[0].set_ylabel('Test accuracy')
corr = np.corrcoef(overlap_per_sim, test_per_sim)[0, 1]
ax[0].set_title(f'Correlation: {corr: .3f}')
corr = np.corrcoef(overlap_test_vs_pred, test_per_sim)[0, 1]
ax[1].set_title(f'Correlation: {corr: .3f}')
corr = np.corrcoef(overlap_test_vs_pred, overlap_per_sim)[0, 1]
ax[2].set_title(f'Correlation: {corr: .3f}')
fig.tight_layout()
#%% Sample across time
# change across time
fig, ax = plt.subplots(ncols=1)
# ax = ax.flatten()
numiters = 10000
visible, hidden_probs, weights = rbm.gibbs_sampling(test_data[:100], k=1)  # train_data
c = 0
# ax[c].imshow(test_data[0].reshape(7, 5), cmap='binary', aspect='auto')
preds = np.zeros((numiters, 100))
# ax[0].set_title('Test data')
for i in range(numiters):
    predictions = classifier.predict(hidden_probs)
    # if i % 50 == 0 and i != 0:
    #     c += 1
    #     ax[c].imshow(visible[0].reshape(7, 5), cmap='binary', aspect='auto')
    #     ax[c].set_title('Pred: ' + str(predictions[0]))
    visible, hidden_probs, weights = rbm.gibbs_sampling(visible, k=1, adapt=0)
    preds[i] = predictions[:100]
ax.plot(preds)
ax.set_xlabel('Iteration')
ax.set_ylabel('Predicted number')
fig, ax = plt.subplots(ncols=1)
ax.plot(preds[:, 19])
#%% rates-FPCD
# change across time
fig, ax = plt.subplots(ncols=1)
numiters = 1000
preds = np.zeros((numiters))
vals, hidden_probs, mw = rbm.rates_FPCD(np.atleast_2d(train_data[0]), epsilon=1e-2, k=20, alpha=1, numiters=numiters)
for r in range(numiters):
    predictions = classifier.predict(np.atleast_2d(hidden_probs[r]))
    preds[r] = predictions
ax.plot(preds)
ax.set_xlabel('Iteration')
ax.set_ylabel('Predicted number')

idx_change = np.where(np.diff(preds) != 0)[0]
fig, ax = plt.subplots(ncols=len(idx_change)+2, figsize=((len(idx_change)+2)*4, 5))
ax[0].imshow(train_data[0].reshape(7, 5), cmap='binary')
ax[0].set_title('Original image')
ax[1].imshow(vals[0].reshape(7, 5), cmap='binary')
ax[1].set_title('1st percept')
labs = ['2nd', '3rd', '4th']
for i_c, c in enumerate(idx_change):
    ax[i_c+2].imshow(vals[c+1].reshape(7, 5), cmap='binary')
    ax[i_c+2].set_title(labs[i_c] + ' percept')

plt.figure()
im = plt.imshow(rbm.fast_weights.T, aspect='auto')
plt.colorbar(im, label='Fast weights')
plt.ylabel('Hidden variable index')
plt.xlabel('Input pixel index')

plt.figure()
plt.plot(mw)
