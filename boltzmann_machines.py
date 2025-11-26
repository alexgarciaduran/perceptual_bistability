# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 09:29:27 2025

@author: alexg
"""

import numpy as np
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import itertools
from tqdm import tqdm

# Optional sklearn utilities (for the classifier & demo)
try:
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import load_digits
    SKLEARN_AVAILABLE = True
except Exception as e:
    SKLEARN_AVAILABLE = False
    print("sklearn not available in this environment. Classifier/demo parts will be skipped.")


mpl.rcParams['font.size'] = 16
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize']= 14
plt.rcParams['ytick.labelsize']= 14


# ------------------------ Utilities ------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def binarize(X, threshold=0.5):
    return (X > threshold).astype(np.float32)

# ------------------------ Restricted Boltzmann Machine ------------------------
class RBM:
    """
    Binary-binary RBM with CD-k training.
    Visible units: binary (0/1)
    Hidden units: binary (0/1)
    """
    def __init__(self, n_visible, n_hidden, lr=0.01, momentum=0.5, weight_decay=1e-4, cd_k=1, seed=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.cd_k = cd_k
        rng = np.random.RandomState(seed)
        # Initialize weights: small gauss
        self.W = 0.01 * rng.randn(n_visible, n_hidden).astype(np.float32)
        self.b = np.zeros(n_visible, dtype=np.float32)  # visible biases
        self.c = np.zeros(n_hidden, dtype=np.float32)   # hidden biases
        # momentum terms
        self.W_inc = np.zeros_like(self.W)
        self.b_inc = np.zeros_like(self.b)
        self.c_inc = np.zeros_like(self.c)

    def propup(self, v):
        return sigmoid(np.dot(v, self.W) + self.c)

    def propdown(self, h):
        return sigmoid(np.dot(h, self.W.T) + self.b)

    def sample_h(self, v):
        ph = self.propup(v)
        return ph, (np.random.rand(*ph.shape) < ph).astype(np.float32)

    def sample_v(self, h):
        pv = self.propdown(h)
        return pv, (np.random.rand(*pv.shape) < pv).astype(np.float32)

    def cd_step(self, v0):
        ph0, h0 = self.sample_h(v0)
        h = h0
        for _ in range(self.cd_k):
            pv, v_sample = self.sample_v(h)
            ph, h = self.sample_h(v_sample)
        # positive and negative associations
        pos = np.dot(v0.T, ph0)
        neg = np.dot(v_sample.T, ph)
        return pos, neg, v0, v_sample, ph0, ph

    def train(self, X, batch_size=64, n_epochs=10, verbose=True):
        n_samples = X.shape[0]
        for epoch in range(n_epochs):
            perm = np.random.permutation(n_samples)
            losses = []
            for i in range(0, n_samples, batch_size):
                batch = X[perm[i:i+batch_size]]
                pos, neg, v0, vk, ph0, phk = self.cd_step(batch)
                # Update increments
                self.W_inc = self.momentum * self.W_inc + self.lr * ((pos - neg) / batch.shape[0] - self.weight_decay * self.W)
                self.b_inc = self.momentum * self.b_inc + self.lr * np.mean(v0 - vk, axis=0)
                self.c_inc = self.momentum * self.c_inc + self.lr * np.mean(ph0 - phk, axis=0)
                # Apply updates
                self.W += self.W_inc
                self.b += self.b_inc
                self.c += self.c_inc
                # reconstruction error for monitoring
                recon = vk
                loss = np.mean((v0 - recon)**2)
                losses.append(loss)
            if verbose:
                print(f"RBM epoch {epoch+1}/{n_epochs} - recon mse: {np.mean(losses):.5f}")
        return self

    def transform(self, X, use_mean=True):
        ph = self.propup(X)
        return ph if use_mean else (np.random.rand(*ph.shape) < ph).astype(np.float32)

    def sample(self, n_gibbs=100, init=None):
        if init is None:
            v = (np.random.rand(1, self.n_visible) < 0.5).astype(np.float32)
        else:
            v = init.copy()
        for _ in range(n_gibbs):
            ph, h = self.sample_h(v)
            pv, v = self.sample_v(h)
        return v

# ------------------------ Fully-connected Boltzmann Machine (non-restricted) ------------------------
class BoltzmannMachine:
    """
    Fully-connected binary Boltzmann Machine with symmetric weight matrix (including visible-visible and hidden-hidden).
    We partition units into visible (first n_visible indices) and hidden (remaining indices).
    Training implemented approximately via Persistent Contrastive Divergence / Gibbs sampling over all units.
    Notes:
     - This is computationally heavy for many units because full-state Gibbs sampling is required.
     - Units are binary (0/1).
    """
    def __init__(self, n_visible, n_hidden, lr=0.005, momentum=0.5, weight_decay=1e-4, persistent_chain_size=100, seed=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_units = n_visible + n_hidden
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        rng = np.random.RandomState(seed)
        # Weight matrix: symmetric, zeros on diagonal
        W_init = 0.01 * rng.randn(self.n_units, self.n_units).astype(np.float32)
        W_init = (W_init + W_init.T) / 2.0
        np.fill_diagonal(W_init, 0.0)
        self.W = W_init
        # biases for all units
        self.b = np.zeros(self.n_units, dtype=np.float32)
        # momentum
        self.W_inc = np.zeros_like(self.W)
        self.b_inc = np.zeros_like(self.b)
        # persistent chains for negative phase
        self.chain = (rng.rand(persistent_chain_size, self.n_units) < 0.5).astype(np.float32)
        self.persistent_chain_size = persistent_chain_size

    def energy(self, s):
        # s: shape (batch, n_units) with binary states
        # E = -0.5 * s^T W s - b^T s  (0.5 factor because W counts pairs twice)
        term = -0.5 * np.sum(np.dot(s, self.W) * s, axis=1) - np.dot(s, self.b)
        return term

    def prop_prob(self, s):
        # compute probabilities for each unit given the other units: P(s_i=1 | s_-i) = sigmoid(sum_j W_ij s_j + b_i)
        net = np.dot(s, self.W) + self.b  # shape (batch, n_units)
        return sigmoid(net)

    def gibbs_step(self, s):
        # Update all units in parallel (blocked Gibbs). This is valid for binary units where we sample each unit conditioned on others.
        p = self.prop_prob(s)
        return (np.random.rand(*p.shape) < p).astype(np.float32)

    def negative_phase(self, n_steps=1):
        # run persistent chains for n_steps
        for _ in range(n_steps):
            self.chain = self.gibbs_step(self.chain)
        return self.chain

    def train(self, X, batch_size=32, n_epochs=10, n_negative_steps=5, verbose=True):
        # X : array shape (n_samples, n_visible) binary
        n_samples = X.shape[0]
        # embed X into full unit space (visible first, hidden zeros)
        for epoch in range(n_epochs):
            perm = np.random.permutation(n_samples)
            losses = []
            for i in range(0, n_samples, batch_size):
                batch_vis = X[perm[i:i+batch_size]]
                # positive phase: clamp visible units, sample hidden units conditioned on visibles
                batch_size_actual = batch_vis.shape[0]
                s_pos = np.zeros((batch_size_actual, self.n_units), dtype=np.float32)
                s_pos[:, :self.n_visible] = batch_vis
                # sample hidden units given visible (one-shot)
                hidden_prob = sigmoid(np.dot(s_pos, self.W) + self.b)[:, self.n_visible:]
                # we want hidden conditioned on visible, so zero out visible->visible contributions
                # but since we use full W, the prop_prob sums contributions from both visible and hidden; sampling once is OK
                h_sample = (np.random.rand(batch_size_actual, self.n_hidden) < hidden_prob).astype(np.float32)
                s_pos[:, self.n_visible:] = h_sample
                # compute positive associations (full outer product over all units)
                pos_assoc = np.einsum('bi,bj->ij', s_pos, s_pos) / batch_size_actual
                pos_bias = np.mean(s_pos, axis=0)
                # negative phase using persistent chains
                neg_chain = self.negative_phase(n_steps=n_negative_steps)
                # sample a subset of persistent chains equal to batch size to compute negative associations
                idx = np.random.randint(0, neg_chain.shape[0], size=batch_size_actual)
                s_neg = neg_chain[idx]
                neg_assoc = np.einsum('bi,bj->ij', s_neg, s_neg) / batch_size_actual
                neg_bias = np.mean(s_neg, axis=0)
                # Weight/bias updates (gradient approx: pos - neg)
                dW = pos_assoc - neg_assoc - self.weight_decay * self.W
                db = pos_bias - neg_bias
                self.W_inc = self.momentum * self.W_inc + self.lr * dW
                self.b_inc = self.momentum * self.b_inc + self.lr * db
                self.W += self.W_inc
                self.b += self.b_inc
                # force symmetry & zero diagonal
                self.W = (self.W + self.W.T) / 2.0
                np.fill_diagonal(self.W, 0.0)
                losses.append(np.mean((self.energy(s_pos) - self.energy(s_neg))**2))
            if verbose:
                print(f"BM epoch {epoch+1}/{n_epochs} - energy-diff mse: {np.mean(losses):.6f}")
        return self

    def transform(self, X, use_mean=True, n_gibbs_for_hidden=10):
        # Given visible X, sample hidden units (conditional). We'll run some Gibbs steps starting from visible clamped.
        batch_size = X.shape[0]
        s = np.zeros((batch_size, self.n_units), dtype=np.float32)
        s[:, :self.n_visible] = X
        # initialize hidden randomly
        s[:, self.n_visible:] = (np.random.rand(batch_size, self.n_hidden) < 0.5).astype(np.float32)
        # alternate updates but keep visible clamped
        for _ in range(n_gibbs_for_hidden):
            # compute hidden probabilities conditioned on current state (including visibles)
            net = np.dot(s, self.W) + self.b
            hidden_prob = sigmoid(net[:, self.n_visible:])
            s[:, self.n_visible:] = (hidden_prob if use_mean else (np.random.rand(*hidden_prob.shape) < hidden_prob).astype(np.float32))
        return hidden_prob if use_mean else s[:, self.n_visible:]

# ------------------------ Classifier helper ------------------------
def train_classifier(features_train, y_train, features_val=None, y_val=None, classifier='svm', **kwargs):
    """
    Train a classifier on RBM/BM features. Returns the trained model and optionally validation accuracy.
    classifier: 'svm' or 'logreg'
    kwargs passed to the classifier constructor (C, kernel, etc.)
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("sklearn required for train_classifier")

    if classifier == 'svm':
        clf = SVC(probability=True, **kwargs)
    else:
        clf = LogisticRegression(max_iter=200, **kwargs)
    clf.fit(features_train, y_train)
    val_acc = None
    if features_val is not None and y_val is not None:
        preds = clf.predict(features_val)
        val_acc = accuracy_score(y_val, preds)
    return clf, val_acc

# ------------------------ Example demo (sklearn digits) ------------------------
def demo_with_sklearn_digits():
    if not SKLEARN_AVAILABLE:
        print("sklearn not available; demo skipped.")
        return
    data = load_digits()
    X = data.data  # shape (n_samples, 64), integers 0..16
    y = data.target
    # scale to [0,1] and binarize
    X = X / 16.0
    Xb = binarize(X, threshold=0.5)  # tweak threshold as desired
    X_train, X_test, y_train, y_test = train_test_split(Xb, y, test_size=0.2, random_state=0, stratify=y)
    # RBM training
    print("Training RBM...")
    rbm = RBM(n_visible=X_train.shape[1], n_hidden=128, lr=0.05, cd_k=1)
    rbm.train(X_train, batch_size=32, n_epochs=15)
    feat_train = rbm.transform(X_train, use_mean=True)
    feat_test = rbm.transform(X_test, use_mean=True)
    clf, val_acc = train_classifier(feat_train, y_train, feat_test, y_test, classifier='svm', C=1.0, kernel='rbf')
    print(f"RBM + SVM test accuracy: {val_acc:.4f}")

    # BM training (careful: slow). Use fewer hidden units for demo.
    print("Training fully-connected Boltzmann Machine (this may be slow)...")
    bm = BoltzmannMachine(n_visible=X_train.shape[1], n_hidden=128, lr=0.01, persistent_chain_size=200)
    bm.train(X_train, batch_size=32, n_epochs=15, n_negative_steps=3)
    feat_train_bm = bm.transform(X_train, use_mean=True, n_gibbs_for_hidden=5)
    feat_test_bm = bm.transform(X_test, use_mean=True, n_gibbs_for_hidden=5)
    clf_bm, val_acc_bm = train_classifier(feat_train_bm, y_train, feat_test_bm, y_test, classifier='svm', C=1.0, kernel='rbf')
    print(f"BM + SVM test accuracy: {val_acc_bm:.4f}")

    return {'rbm': rbm, 'rbm_clf': clf, 'rbm_acc': val_acc, 'bm': bm, 'bm_clf': clf_bm, 'bm_acc': val_acc_bm}


def plot_hidden_activations(rbm_or_bm, X, n_samples=100, use_mean=True):
    """
    Plot hidden activations for a batch of data.
    rbm_or_bm: trained RBM or BoltzmannMachine object
    X: input data, shape (n_samples_total, n_visible)
    """
    X_batch = X[:n_samples]
    hidden_act = rbm_or_bm.transform(X_batch, use_mean=use_mean)
    
    # Plot histogram of activations
    plt.figure(figsize=(8,4))
    plt.hist(hidden_act.flatten(), bins=20, color='skyblue')
    plt.title("Histogram of hidden unit activations")
    plt.xlabel("Activation")
    plt.ylabel("Frequency")
    plt.show()
    
    # Optional: heatmap of activations (samples x hidden units)
    plt.figure(figsize=(10,6))
    plt.imshow(hidden_act, aspect='auto', cmap='viridis')
    plt.colorbar(label='Activation')
    plt.xlabel("Hidden units")
    plt.ylabel("Samples")
    plt.title("Hidden unit activations")
    plt.show()

def plot_receptive_fields(rbm, img_shape, n_cols=16):
    """
    Plot receptive fields of hidden units as images.
    rbm: trained RBM (only RBM has clear receptive fields)
    img_shape: tuple of image height and width (e.g., (8,8) for sklearn digits)
    """
    n_hidden = rbm.n_hidden
    n_rows = int(np.ceil(n_hidden / n_cols))
    
    plt.figure(figsize=(n_cols, n_rows))
    for i in range(n_hidden):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(rbm.W[:,i].reshape(img_shape), cmap='gray', interpolation='nearest')
        plt.axis('off')
    plt.suptitle("RBM hidden unit receptive fields")
    plt.show()


def plot_hidden_activations_bm(bm, X, n_samples=100, use_mean=True, n_gibbs_for_hidden=10):
    """
    Plot hidden activations for a batch of data in a BM.
    """
    X_batch = X[:n_samples]
    hidden_act = bm.transform(X_batch, use_mean=use_mean, n_gibbs_for_hidden=n_gibbs_for_hidden)
    
    # Histogram
    plt.figure(figsize=(8,4))
    plt.hist(hidden_act.flatten(), bins=20, color='lightcoral')
    plt.title("Histogram of BM hidden unit activations")
    plt.xlabel("Activation")
    plt.ylabel("Frequency")
    plt.show()
    
    # Heatmap
    plt.figure(figsize=(10,6))
    plt.imshow(hidden_act, aspect='auto', cmap='viridis')
    plt.colorbar(label='Activation')
    plt.xlabel("Hidden units")
    plt.ylabel("Samples")
    plt.title("BM hidden unit activations")
    plt.show()

def plot_receptive_fields_bm(bm, img_shape, n_cols=16):
    """
    Plot receptive fields of hidden units in BM.
    We visualize visible->hidden weights: W[:n_visible, n_visible+i]
    """
    n_hidden = bm.n_hidden
    n_rows = int(np.ceil(n_hidden / n_cols))
    
    plt.figure(figsize=(n_cols, n_rows))
    for i in range(n_hidden):
        rf = bm.W[:bm.n_visible, bm.n_visible + i]
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(rf.reshape(img_shape), cmap='gray', interpolation='nearest')
        plt.axis('off')
    plt.suptitle("BM hidden unit receptive fields (visible→hidden weights)")
    plt.show()



def hysteresis_test(rbm_or_bm, clf, images_sequence, digit0, use_mean=True, n_gibbs_for_hidden=10):
    """
    images_sequence: array of images (n_steps, n_pixels)
    target_labels: list of the "true" digit labels for each image
    rbm_or_bm: trained RBM or BM
    clf: classifier (SVM or logistic regression)
    """
    n_steps = images_sequence.shape[0]
    probs = []
    for i in range(n_steps):
        x = images_sequence[i:i+1]
        if isinstance(rbm_or_bm, RBM):
            h = rbm_or_bm.transform(x, use_mean=use_mean)
        else:
            h = rbm_or_bm.transform(x, use_mean=use_mean, n_gibbs_for_hidden=n_gibbs_for_hidden)
        p = clf.predict_proba(h)[0]  # shape: n_classes
        probs.append(p[digit0])  # probability of the "true" class
    probs = np.array(probs)
    return probs


def merge_digits(digit0, digit1, n_steps=11, alpha_values=None):
    """
    Create a sequence of merged images from digit0 to digit1.
    
    Parameters
    ----------
    digit0 : int
        First digit (0-9)
    digit1 : int
        Second digit (0-9)
    n_steps : int
        Number of intermediate steps (including endpoints)
    alpha_values : array-like, optional
        Fractions of digit1 in the mixture. If None, linear spacing 0..1 is used.
        
    Returns
    -------
    images_merged : np.ndarray, shape (n_steps, n_pixels)
        Merged images
    labels_merged : np.ndarray, shape (n_steps,)
        Corresponding "true" labels (digit0 or digit1). For simplicity, assign digit0 if alpha<0.5 else digit1.
    """
    data = load_digits()
    X = data.data / 16.0  # scale to [0,1]
    y = data.target

    # Get one random image per digit (could also average multiple examples)
    img0 = X[y == digit0][0]
    img1 = X[y == digit1][0]
    
    if alpha_values is None:
        alpha_values = np.linspace(0, 1, n_steps)
    
    images_merged = []
    labels_merged = []
    
    for alpha in alpha_values:
        mixed = (1 - alpha) * img0 + alpha * img1
        images_merged.append(mixed)
        # label: assign to the "dominant" digit
        labels_merged.append(digit0 if alpha < 0.5 else digit1)
    
    images_merged = np.array(images_merged)
    labels_merged = np.array(labels_merged)
    
    return images_merged, labels_merged


def plot_example_hysteresis(digit0, digit1, rbm, clf):
    # ------------------ Plot example ------------------
    images_0_to_1, labels_merged = merge_digits(digit0, digit1, n_steps=100)
    # Suppose we have images_0_to_8 and labels_0_to_8
    # forward sequence
    probs_forward = hysteresis_test(rbm, clf, images_0_to_1, digit0)
    # reverse sequence
    probs_reverse = hysteresis_test(rbm, clf, images_0_to_1[::-1], digit0)
    
    plt.figure(figsize=(8,4))
    plt.plot(probs_forward, label=f"{digit0}→{digit1}")
    plt.plot(probs_reverse[::-1], label=f"{digit1}→{digit0}")
    plt.xlabel("Step")
    plt.ylabel("Decoder probability")
    plt.title("Hysteresis of decoder probabilities")
    plt.legend()
    plt.show()


def mixed_image_bistability(rbm_or_bm, clf, image1, image2, n_samples=50, use_mean=False, n_gibbs_for_hidden=10):
    """
    image1, image2: two images (1D arrays)
    """
    mixed = (image1+image2)
    mixed = np.clip(mixed, 0, 1)  # keep valid range
    
    probs_sum = np.zeros(clf.classes_.shape[0])
    for _ in range(n_samples):
        if isinstance(rbm_or_bm, RBM):
            h = rbm_or_bm.transform(mixed[None,:], use_mean=use_mean)
        else:
            h = rbm_or_bm.transform(mixed[None,:], use_mean=use_mean, n_gibbs_for_hidden=n_gibbs_for_hidden)
        p = clf.predict_proba(h)[0]
        probs_sum += p
    return probs_sum/n_samples


def jaccard(digit_0, digit_1):
    overlap = np.logical_and(digit_0, digit_1)
    overlap_sum = np.sum(overlap)
    union = np.logical_or(digit_0, digit_1)
    union_sum = np.sum(union)
    return overlap_sum / union_sum


def prob_vs_overlap(rbm, clf, ntimes=50, use_mean=True):
    digits_possible = np.arange(10, dtype=np.int64)
    combs = list(itertools.product(digits_possible, digits_possible))
    prob1 = []
    prob2 = []
    overlap_mean = []
    for digit1, digit2 in tqdm(combs):
        probs_all = probs_after_many_overlaps(rbm, clf, digit1, digit2, ntimes=ntimes, use_mean=use_mean)
        probs = np.nanmean(probs_all, axis=0)
        prob1.append(probs[digit1]); prob2.append(probs[digit2])
        overlap = []
        for n in range(ntimes):
            image1, image2 = get_digit_images(digit1, digit2)
            overlap.append(jaccard(image1, image2))
        overlap_mean.append(np.nanmean(overlap))
    fig, ax = plt.subplots(1)
    ax.plot(overlap_mean, prob1, label='Digit 1',
            marker='o', linestyle='')
    ax.plot(overlap_mean, prob2, label='Digit 2',
            marker='o', linestyle='')
    ax.set_ylabel('P(y=label)')
    ax.set_xlabel('Overlap')
    

def overlap_bistability(rbm, clf, digit1, digit2, ntimes=50, use_mean=False):
    probs = probs_after_many_overlaps(rbm, clf, digit1, digit2, ntimes=ntimes, use_mean=use_mean)
    sns.barplot(probs, errorbar='se')
    plt.xlabel("Digit class")
    plt.ylabel("Probability")
    plt.title("Bistability distribution for mixed image")
    plt.show()


def probs_after_many_overlaps(rbm, clf, digit1, digit2, ntimes=50, use_mean=True):
    probs_sum = np.zeros((ntimes, clf.classes_.shape[0]))
    for n in range(ntimes):
        image1, image2 = get_digit_images(digit1, digit2)
        probs = mixed_image_bistability(rbm, clf, image1, image2, n_samples=100, use_mean=use_mean)
        probs_sum[n] = probs
    return probs_sum


def get_digit_images(digit1, digit2):
    """
    Return one example image for each of the two specified digits.

    Parameters
    ----------
    digit1 : int
        First digit (0-9)
    digit2 : int
        Second digit (0-9)

    Returns
    -------
    image1 : np.ndarray, shape (n_pixels,)
        Image of digit1, scaled to [0,1]
    image2 : np.ndarray, shape (n_pixels,)
        Image of digit2, scaled to [0,1]
    """
    data = load_digits()
    X = data.data / 16.0  # scale to [0,1]
    X = binarize(X, threshold=0.5)
    y = data.target

    # pick the first example for each digit
    idx1 = y == digit1
    idx2 = y == digit2
    image1 = X[np.random.choice(np.where(idx1)[0])]
    image2 = X[np.random.choice(np.where(idx2)[0])]

    return image1, image2


if __name__ == "__main__":
    results = demo_with_sklearn_digits()
    bm = results['bm']
    rbm = results['rbm']
    data = load_digits()
    X = data.data / 16.0  # scale to [0,1]
    Xb = (X > 0.5).astype(np.float32)
    ## Suppose rbm is your trained RBM object
    # plot hidden activations
    # plot_hidden_activations(rbm, Xb, n_samples=100, use_mean=True)
    
    ## plot receptive fields
    # plot_receptive_fields(rbm, img_shape=(8,8), n_cols=16)
    # # now for BM
    # plot_hidden_activations_bm(bm, Xb, n_samples=100, use_mean=True, n_gibbs_for_hidden=10)
    # plot_receptive_fields_bm(bm, img_shape=(8,8), n_cols=16)
    # plot_example_hysteresis(3, 8, bm, results['bm_clf'])
    # prob_vs_overlap(bm, results['bm_clf'], ntimes=50, use_mean=True)
    overlap_bistability(bm, results['bm_clf'], digit1=1, digit2=2, use_mean=True)
