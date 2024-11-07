import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from IPython.display import Image
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import imageio
from matplotlib.animation import FuncAnimation
from data_generation import x, alpha_normalized, beta_normalized
from optimization_functions import optimize_rotation, optimize_rotation_lbfgs, optimize_rotation_multiple_angles, optimize_rotation_multiple_angles_lbfgs
from imports_and_utils import rotation_matrix_z, energy_distance, weighted_energy_distance, generate_elliptical_cloud

R = rotation_matrix_z(45.0)
R.requires_grad_(True)
num_steps = 300
lr = 0.01

# Run optimization
#R_optimized, loss_values, intermediate_rotations = optimize_rotation(torch.optim.SGD, R, lr, num_steps, x, x, use_weighted=True, alpha=alpha_normalized, beta=beta_normalized)
#R_optimized, loss_values, intermediate_rotations = optimize_rotation(torch.optim.Adam, R, lr, num_steps, x, x, use_weighted=True, alpha=alpha_normalized, beta=beta_normalized)
#R_optimized, loss_values, intermediate_rotations = optimize_rotation(torch.optim.RMSprop, R, lr, num_steps, x, x, use_weighted=True, alpha=alpha_normalized, beta=beta_normalized)
R_optimized, loss_values, intermediate_rotations = optimize_rotation_lbfgs(R, lr, num_steps, x, x, use_weighted=True, alpha=alpha_normalized, beta=beta_normalized)

# Plot loss values
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Energy Distance Loss')
plt.xlabel('Optimization Steps')
plt.ylabel('Loss')
plt.title('Energy Distance Loss Over Optimization Steps')
plt.legend()
#plt.savefig('loss_plot_weighted_sgd.png')
#plt.savefig('loss_plot_weighted_rmsprop.png')
#plt.savefig('loss_plot_weighted_adam.png')
plt.savefig('loss_plot_weighted_lbfgs.png')
plt.show()

# Create animation
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

def update_plot(frame):
    ax.clear()
    x_rotated = (x @ intermediate_rotations[frame].t()).numpy()
    ax.scatter(x_rotated[:, 0], x_rotated[:, 1], x_rotated[:, 2], c='red', s=3, alpha=0.8, label='Rotated Points')
    ax.scatter(x[:, 0].numpy(), x[:, 1].numpy(), x[:, 2].numpy(), c='green', s=3, alpha=0.8, label='Original Points')
    ax.set_title(f'Optimization Step: {frame + 1}')
    ax.legend()

ani = FuncAnimation(fig, update_plot, frames=len(intermediate_rotations), interval=50)
#ani.save('rotation_optimization_weighted_sgd.gif', writer='imagemagick')
#ani.save('rotation_optimization_weighted_rmsprop.gif', writer='imagemagick')
#ani.save('rotation_optimization_weighted_adam.gif', writer='imagemagick')
ani.save('rotation_optimization_weighted_lbfgs.gif', writer='imagemagick')

