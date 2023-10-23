# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 08:50:14 2023

@author: alexg
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import itertools
import os
import gibbs_necker as gn
from scipy.optimize import fsolve
import numpy as np

C = gn.C
theta = gn.THETA


def mean_field(J, num_iter):
    #initialize random state of the cube
    vec = np.random.rand(8)
    vec_time = np.empty((num_iter, 8))
    vec_time[:] = np.nan
    for i in range(num_iter):
        for q in range(8):
            neighbours = theta[q].astype(dtype=bool)
            vec[q] = gn.sigmoid(2*J*sum(2*vec[neighbours]-1))
        vec_time[i, :] = vec
    return vec_time


def plot_solutions_mfield(j_list):
    l = []
    for j in j_list:
        q = lambda q: gn.sigmoid(6*j*(2*q-1)) - q
        l.append(fsolve(q,1))
        
    plt.plot(j_list, 1-np.array(l), color='k')
    plt.plot(j_list, l, color='r')
    plt.xlabel('J')
    plt.ylabel('q')
      
    plt.title('Solutions of the dynamical system')

if __name__ == '__main__':
    plot_solutions_mfield(j_list=np.arange(0.00001, 2, 0.01))
