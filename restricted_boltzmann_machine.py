# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:52:42 2024

@author: alexg
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from skimage.transform import resize

pc_name = 'alex'
if pc_name == 'alex':
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/gibbs_sampling_necker/data_folder/'  # Alex

elif pc_name == 'alex_CRM':
    DATA_FOLDER = 'C:/Users/agarcia/Desktop/phd/necker/data_folder/'  # Alex CRM

dataset_path = DATA_FOLDER + '/necker_images_clean/'
train_path = DATA_FOLDER + '/necker_images_white/'
test_path = DATA_FOLDER + '/necker_images_ambiguous/'

#%% RBM
class RBM:
  
  def __init__(self, num_visible, num_hidden):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.debug_print = False
    self.debug_plot = True

    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    # a uniform distribution between -sqrt(6. / (num_hidden + num_visible))
    # and sqrt(6. / (num_hidden + num_visible)). One could vary the 
    # standard deviation by multiplying the interval with appropriate value.
    # Here we initialize the weights with mean 0 and standard deviation 0.1. 
    # Reference: Understanding the difficulty of training deep feedforward 
    # neural networks by Xavier Glorot and Yoshua Bengio
    np_rng = np.random.RandomState(1234)

    self.weights = np.asarray(np_rng.uniform(
			low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	size=(num_visible, num_hidden)))


    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)


  def train(self, data, max_epochs = 1000, learning_rate = 0.1):
    """
    Train the machine.

    Parameters
    ----------
    data: A matrix where each row is a training example consisting of the states of visible units.    
    """

    num_examples = data.shape[0]

    # Insert bias units of 1 into the first column.
    data = np.insert(data, 0, 1, axis = 1)
    error_list = []
    w_100_in = []
    w_100_out = []
    for epoch in range(max_epochs):      
      # Clamp to the data and sample from the hidden units. 
      # (This is the "positive CD phase", aka the reality phase.)
      pos_hidden_activations = np.dot(data, self.weights)      
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_probs[:,0] = 1 # Fix the bias unit.
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
      # Note that we're using the activation *probabilities* of the hidden states, not the hidden states       
      # themselves, when computing associations. We could also use the states; see section 3 of Hinton's 
      # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
      pos_associations = np.dot(data.T, pos_hidden_probs)

      # Reconstruct the visible units and sample again from the hidden units.
      # (This is the "negative CD-1 phase", aka the daydreaming phase.)
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:, 0] = 1 # Fix the bias unit.
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      # Note, again, that we're using the activation *probabilities* when computing associations, not the states 
      # themselves.
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

      # Update weights.
      self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

      error = np.sum((data - neg_visible_probs) ** 2)
      if self.debug_print:
          print("Epoch %s: error is %s" % (epoch, error))
      if self.debug_plot:
          error_list.append(error)
          w_100_in.append(np.mean(self.weights[18]))
          w_100_out.append(np.mean(self.weights[:, 18]))
    if self.debug_plot:
        return error_list, w_100_in, w_100_out

  def run_visible(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of visible units, to get a sample of the hidden units.
    
    Parameters
    ----------
    data: A matrix where each row consists of the states of the visible units.
    
    Returns
    -------
    hidden_states: A matrix where each row consists of the hidden units activated from the visible
    units in the data matrix passed in.
    """
    
    num_examples = data.shape[0]
    
    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden + 1))
    
    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights)
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = self._logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Always fix the bias unit to 1.
    # hidden_states[:,0] = 1
  
    # Ignore the bias units.
    hidden_states = hidden_states[:,1:]
    return hidden_states
    
  # TODO: Remove the code duplication between this method and `run_visible`?
  def run_hidden(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of hidden units, to get a sample of the visible units.

    Parameters
    ----------
    data: A matrix where each row consists of the states of the hidden units.

    Returns
    -------
    visible_states: A matrix where each row consists of the visible units activated from the hidden
    units in the data matrix passed in.
    """

    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    # sampled from a training example.
    visible_states = np.ones((num_examples, self.num_visible + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the visible units.
    visible_activations = np.dot(data, self.weights.T)
    # Calculate the probabilities of turning the visible units on.
    visible_probs = self._logistic(visible_activations)
    # Turn the visible units on with their specified probabilities.
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    # Always fix the bias unit to 1.
    # visible_states[:,0] = 1

    # Ignore the bias units.
    visible_states = visible_states[:,1:]
    return visible_states
    
  def daydream(self, num_samples, test_data):
    """
    Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
    (where each step consists of updating all the hidden units, and then updating all of the visible units),
    taking a sample of the visible units at each step.
    Note that we only initialize the network *once*, so these samples are correlated.

    Returns
    -------
    samples: A matrix, where each row is a sample of the visible units produced while the network was
    daydreaming.
    """

    # Create a matrix, where each row is to be a sample of of the visible units 
    # (with an extra bias unit), initialized to all ones.
    samples = np.ones((num_samples, self.num_visible + 1))

    # Take the first sample from a uniform distribution.
    samples[0, 1:] = test_data

    # Start the alternating Gibbs sampling.
    # Note that we keep the hidden units binary states, but leave the
    # visible units as real probabilities. See section 3 of Hinton's
    # "A Practical Guide to Training Restricted Boltzmann Machines"
    # for more on why.
    visible_probs_all = np.zeros((num_samples, self.num_hidden+1))
    for i in range(1, num_samples):
      visible = samples[i-1,:]

      # Calculate the activations of the hidden units.
      hidden_activations = np.dot(visible, self.weights)
      # Calculate the probabilities of turning the hidden units on.
      hidden_probs = self._logistic(hidden_activations)
      # Turn the hidden units on with their specified probabilities.
      hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
      # Always fix the bias unit to 1.
      hidden_states[0] = 1

      # Recalculate the probabilities that the visible units are on.
      visible_activations = np.dot(hidden_states, self.weights.T)
      visible_probs = self._logistic(visible_activations)
      visible_states = visible_probs > np.random.rand(self.num_visible + 1)
      samples[i,:] = visible_states
      visible_probs_all[i, :] = visible_probs
    # Ignore the bias units (the first column), since they're always set to 1.
    return samples[:, 1:], visible_probs_all[:, 1:]
      
  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))


def data_load(dataset_path, batch_size=10, imgdesiredsize=26,
              thres=0.92, roll=40):
    images = glob.glob(dataset_path + '*')
    # np.random.shuffle(images)
    images = np.roll(images, roll)
    images = images[:batch_size]
    data = np.zeros((batch_size, int(imgdesiredsize**2)))
    for im_ind, image in enumerate(images):
        orig_img = plt.imread(image)[:, :, 0]
        img = resize(orig_img, (imgdesiredsize, imgdesiredsize))
        img[img >= thres] = 0
        img[(img > 0)] = 1
        data[im_ind, :] = img.flatten()
    # train_data = torch.from_numpy(train_data).to(torch.float32)
    return data


def plot_random_examples_training_data(training_data,
                                       s0=26, s1=26, n_examples=8, nrows=2):
    fig, ax = plt.subplots(ncols=n_examples//nrows, nrows=nrows)
    ax = ax.flatten()
    for i in range(n_examples):
        ax[i].imshow(training_data[i].reshape(s0, s1), cmap='gist_gray_r')
        ax[i].axis('off')
    fig.tight_layout()


if __name__ == '__main__':
    important_idxs = [17, 7]
    training_data = data_load(train_path, batch_size=40, roll=60)
    plot_random_examples_training_data(training_data, s0=26, s1=26, n_examples=10, nrows=2)
    r = RBM(num_visible = training_data.shape[1], num_hidden = training_data.shape[1])
    # training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
    error_list, w_in, w_out = r.train(training_data,
                                      max_epochs=1000)
    test_data = data_load(test_path, batch_size=1, thres=0.9, roll=60)
    fig, ax = plt.subplots(ncols=2, nrows=2)
    ax = ax.flatten()
    ax[0].plot(error_list)
    ax[0].set_yscale('log')
    ax2 = ax[0].twinx()
    ax2.plot(w_in, color='r', label='In')
    ax2.plot(w_out, color='k', label='Out')
    ax2.set_ylabel('Weights')
    ax2.legend()
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Error')
    ax[1].imshow(test_data.reshape(26, 26))
    ax[1].set_title('Test set')
    ax[2].imshow(r.run_visible(test_data).reshape(26, 26))
    ax[2].set_title('Sample of hidden variables')
    ax[3].imshow(r.run_hidden(test_data).reshape(26, 26))
    ax[3].set_title('Sample of visible variables')
    fig, ax2 = plt.subplots(ncols=3)
    daydream, visible_probs_all = r.daydream(100000, test_data)
    max_variance_index = np.argmax(np.std(daydream, axis=0))
    aprox_05 = np.argmin(np.abs(np.mean(daydream, axis=0)-0.5))
    index_to_tuple = np.unravel_index(max_variance_index, [26, 26])
    ax[1].scatter(index_to_tuple[1], index_to_tuple[0], color='r', marker='x')
    ax[1].scatter(important_idxs[0], important_idxs[1], color='r')
    idx_important = np.ravel_multi_index(important_idxs, [26,26])
    ax2[0].plot(daydream[:, max_variance_index])
    ax2[1].plot(visible_probs_all[:, max_variance_index-10:max_variance_index+10])
    ax2[2].plot(np.sum((test_data - daydream) ** 2, axis=1))
    ax2[2].set_ylabel('Reconstruction error')
    fig, axtime = plt.subplots(ncols=6, figsize=(14, 6))
    indices = np.linspace(0, 100000-1, len(axtime), dtype=int)
    for i, ind in enumerate(indices):
        axtime[i].imshow(daydream[ind, :].reshape(26, 26))
        axtime[i].scatter(index_to_tuple[1], index_to_tuple[0], color='r',
                          marker='x')
    fig, axtime = plt.subplots(ncols=11, figsize=(14, 6))
    indices = np.arange(100, 111, 1)
    for i, ind in enumerate(indices):
        axtime[i].imshow(daydream[ind, :].reshape(26, 26))
        axtime[i].scatter(index_to_tuple[1], index_to_tuple[0], color='r',
                          marker='x')

