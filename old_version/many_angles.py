import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import imageio
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import Image
from matplotlib.animation import FuncAnimation
from data_generation import x, alpha_normalized, beta_normalized
from optimization_functions import optimize_rotation, optimize_rotation_lbfgs, optimize_rotation_multiple_angles, optimize_rotation_multiple_angles_lbfgs
from imports_and_utils import rotation_matrix_z, energy_distance, weighted_energy_distance, generate_elliptical_cloud



angles = np.linspace(-90, 90, num=181)
weighted_energy_distances = []

for angle in angles:
    R = rotation_matrix_z(angle)
    x_rotated = x @ R.T
    distance = weighted_energy_distance(x_rotated, x, alpha_normalized, beta_normalized)
    weighted_energy_distances.append(distance.item())
    
plt.figure(figsize=(10, 6))
plt.plot(angles, weighted_energy_distances, label='Weighted Energy Distance')
plt.xlabel('Rotation Angle (radians)')
plt.ylabel('Weighted Energy Distance')
plt.title('Weighted Energy Distance vs. Rotation Angle')
plt.legend()
plt.grid(True)
plt.savefig('weighted_energy_distance_plot.png')
plt.show()

