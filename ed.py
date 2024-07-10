# -*- coding: utf-8 -*-
"""energy_distance.py"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from IPython.display import Image
import plotly.graph_objs as go
import plotly.io as pio
from matplotlib.animation import FuncAnimation

# Import functions from functions.py
from functions import (weighted_energy_distance, rotation_matrix_z, 
                       pairwise_distances, energy_distance, optimize_rotation,
                       euclidean_distance, generate_elliptical_cloud)

# Example data
x = generate_elliptical_cloud([0, 0, 0], [[1, 0.5, 0.2], [0.5, 2, 0.3], [0.2, 0.3, 1]], 500)
y = generate_elliptical_cloud([0, 0, 0], [[1, 0.5, 0.2], [0.5, 2, 0.3], [0.2, 0.3, 1]], 500)

# Define the optimizer, learning rate, and number of steps
R = rotation_matrix_z(45)
R.requires_grad_(True)
optimizer_class = optim.Adam
lr = 0.01
num_steps = 300

# Optimize the rotation matrix
R_optimized, loss_values, intermediate_rotations = optimize_rotation(optimizer_class, R, lr, num_steps, x, y)

# Plot the loss values over the optimization steps
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Energy Distance Loss')
plt.xlabel('Optimization Steps')
plt.ylabel('Loss')
plt.title('Energy Distance Loss Over Optimization Steps')
plt.legend()
plt.show()

