# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:04:17 2025

@author: alexg
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse
import itertools
import matplotlib as mpl


mpl.rcParams['font.size'] = 18
plt.rcParams['legend.title_fontsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize']= 16
plt.rcParams['ytick.labelsize']= 16


def get_digit(a):
    digits = {
        "0": [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        "1": [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0]
        ],
        "2": [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1]
        ],
        "3": [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        "4": [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0]
        ],
        "5": [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0]
        ],
        "6": [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        "7": [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ],
        "8": [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ],
        "9": [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ]
    }
    return digits[a]


def psi(n_plus, p):
    n_minus = 1-n_plus
    return n_plus*np.log(p) + n_minus*np.log(1-p)


def figure_ground(po=0.5, pa=0.5, pb=0.5, digits=['0', '1']):
    digit_0 = np.round(np.clip(get_digit(digits[0]) + 0.*np.random.randn(5, 5), 0, 1))
    digit_1 = np.round(np.clip(get_digit(digits[1]) + 0.*np.random.randn(5, 5), 0, 1))
    combination = np.clip(digit_0 + digit_1, 0, 1)
    n_plus = combination.flatten().sum()
    n_minus = (1-combination.flatten())
    n_a = digit_0.flatten()
    n_b = digit_1.flatten()
    n_c = (1*(digit_0 == digit_1)*(digit_0 == 1)*(digit_1 == 1)).flatten()

    # compute probabilities as proportions (?)
    # po = np.sum(combination.flatten()) / len(combination.flatten())
    # pa = n_a.sum() / len(digit_0.flatten())
    # pb = n_b.sum() / len(digit_0.flatten())
    # pc = n_c.sum() / len(digit_0.flatten())
    # pc = 1-(1-pa)*(1-pc)
    pc = (pa+pb)/2

    p = [po, pc, pa, pb]
    q = [1-ps for ps in p]
    p0 = p[0]*p[1]/(p[2]*p[3])
    q0 = q[0]*q[1]/(q[2]*q[3])
    j_eff = 0.5*(n_c*np.log(p0) + (1-n_c)*np.log(q0))
    b1 = 0.5*((psi(n_a, pa)-psi(n_a, po))+
              (psi(n_c, pb)+psi(n_c, po)-psi(n_c, pc)-psi(n_c, pa)))
    b2 = 0.5*((psi(n_b, pb)-psi(n_b, po))+
              (psi(n_c, pa)+psi(n_c, po)-psi(n_c, pc)-psi(n_c, pb)))
    # plt.figure()
    # im = plt.imshow(j_eff.reshape(5,5), cmap='viridis', aspect='auto')
    # plt.colorbar(im)
    # plt.title('Jeff values')
    # plt.figure()
    # im = plt.imshow(b1.reshape(5,5), cmap='viridis', aspect='auto')
    # plt.colorbar(im)
    # plt.title('b1 values')
    # plt.figure()
    # im = plt.imshow(b2.reshape(5,5), cmap='viridis', aspect='auto')
    # plt.colorbar(im)
    # plt.title('b2 values')
    # plt.figure()
    # plt.imshow(combination.reshape(5,5), cmap='binary', aspect='auto')
    # plt.title(f'Combination between {digits[0]} and {digits[1]}')
    return j_eff, b1, b2


def j_eff_vs_mse(pa=0.8, pb=0.8, po=0.5):
    dig = [str(a) for a in range(10)]
    combs = list(itertools.combinations(dig, 2))
    mselist = []
    jefflist = []
    b2list = []
    b1list = []
    for digits in combs:
        digit_0 = np.array(get_digit(digits[0]))
        digit_1 = np.array(get_digit(digits[1]))
        jeff_vals, b1vals, b2vals = figure_ground(po=po, pa=pa, pb=pb, digits=digits)
        jeff = np.sum(jeff_vals)
        b2 = np.sum(b2vals)
        b1 = np.sum(b1vals)
        jefflist.append(jeff)
        b2list.append(b2)
        b1list.append(b1)
        mselist.append(mse(digit_0, digit_1))
    fig, ax = plt.subplots(ncols=3, figsize=(14, 4.5))
    ax[0].plot(jefflist, mselist, marker='o', linestyle='', color='k')
    ax[0].set_xlabel(r'Effective coupling, $\sum_i \; J_i^{eff} \, (x, y)$')
    ax[0].set_ylabel('MSE(digit x, digit y)')
    ax[1].plot(b1list, mselist, marker='o', linestyle='', color='k')
    ax[1].set_xlabel(r'Effective bias tw X, $\sum_i \; B_{1, i}^{eff} \, (x, y)$')
    ax[1].set_ylabel('MSE(digit x, digit y)')
    ax[2].plot(b2list, mselist, marker='o', linestyle='', color='k')
    ax[2].set_xlabel(r'Effective bias tw Y, $\sum_i \; B_{2, i}^{eff} \, (x, y)$')
    ax[2].set_ylabel('MSE(digit x, digit y)')
    fig.tight_layout()


if __name__ == '__main__':
    j_eff_vs_mse()
    