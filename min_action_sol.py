# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 23:35:37 2024

@author: alexg
"""

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from mean_field_necker import get_unst_and_stab_fp

# Parameters
J = 1.   # Interaction strength
B = 0.   # Bias
sigma = 0.1  # Noise intensity


def sigmoid(x):
    return 1/(1+np.exp(-x))

# Define the sigmoid drift function f_i(X) with connectivity matrix theta
def drift_function(X, theta, J, B):
    # X is (d, N) where d is the dimension and N is the number of time points
    d, N = X.shape
    weighted_sum = 2*np.matmul(theta, X)-3  # Matrix multiplication for weighted sum
    f = sigmoid(2 * J * weighted_sum + 2 * B) - X
    return f


# Define the system of ODEs for solve_bvp
def ode_system(t, X):
    d = X.shape[0] // 2  # Separate x and y components
    X_dot = np.zeros_like(X)  # X_dot will store dx/dt and dy/dt

    # First d components of X_dot represent dx/dt, which are simply y_i values
    X_dot[:d, :] = X[d:, :]

    # Second d components of X_dot represent dy/dt
    X_ddot = np.zeros((d, X.shape[1]))
    f = drift_function(X[:d, :], theta, J, B)  # (d, N)

    for i in range(d):
        for j in range(d):
            X_ddot[i, :] += -(X[d + j, :] - f[j, :]) * sigmoid_gradient(X[:d, :], j, i, theta, J, B) +\
                sigmoid_gradient(X[:d, :], i, j, theta, J, B) * X[d + j, :]
    
    X_dot[d:, :] = X_ddot  # Set second d components to dy/dt
    return X_dot


# Define the boundary conditions
def boundary_conditions(Xa, Xb):
    return np.concatenate((Xa[:d] - x0[:d],
                           Xb[:d] - xf[:d]))


# Sigmoid gradient with respect to X

def sigmoid_gradient(X, i, j, theta, J, B):
    # Compute the weighted sum term for component i across all neighbors
    weighted_sum = np.sum((2 * X[theta[i, :].astype(bool)] - 1), axis=0)  # Shape: (N,)
    # Calculate the sigmoid function for component i at each time point
    sigmoid_x = sigmoid(2 * J * weighted_sum + 2 * B)  # Shape: (N,)
    # Compute the gradient of the sigmoid with respect to X_j
    grad = 4 * J * theta[i, j] * sigmoid_x * (1 - sigmoid_x)  # Shape: (N,)
    return grad


# Define parameters for the path and neighbors
d = 8         # Dimension of the system

# connectivity matrix
theta = np.array([[0 ,1 ,1 ,0 ,1 ,0 ,0 ,0], [1, 0, 0, 1, 0, 1, 0, 0],
                  [1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1],
                  [0, 0, 1, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 1, 1, 0]])

x_stable_1, x_stable_2, x_unstable = get_unst_and_stab_fp(J, B)
x0 = np.concatenate((np.ones(d) * x_stable_1, np.zeros(d)))  # Initial state
xf = np.concatenate((np.ones(d) * x_stable_2, np.zeros(d))) # Final state
t_end = 1.0            # End time
N = 1000              # Number of time points for discretization

# Initial guess for X and X_dot (needed by solve_bvp)
t = np.linspace(0, t_end, N)
X_guess = np.zeros((2 * d, N))
X_guess[:d, :] = np.outer(xf[:d] - x0[:d], t / t_end) + x0[:d, np.newaxis]
X_guess[d:, :] = (xf[:d] - x0[:d]).reshape(-1, 1) / t_end

# Solve the boundary value problem
solution = solve_bvp(ode_system, boundary_conditions, t, X_guess)

# Check if the solver converged
if solution.success:
    print("Solution found!")
else:
    print("Solution not found.")

# Plot the solution
plt.figure()
for i in range(d):
    plt.plot(solution.x, solution.y[i, :], label=f'$x_{i+1}(t)$')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.title('Most Probable Path Solution')
plt.show()

# Plot the solution
plt.figure()
for i in range(d):
    plt.plot(solution.x, solution.y[i+d, :], label=f'$x_{i+1}(t)$')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.title('Most Probable Path Solution')
plt.show()
