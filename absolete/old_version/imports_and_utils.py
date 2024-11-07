import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

from matplotlib.animation import FuncAnimation
from IPython.display import Image
from mpl_toolkits.mplot3d import Axes3D

def rotation_matrix_z(angle_degrees):
    angle_radians = np.radians(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    R = torch.tensor([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    return R

def pairwise_distances(x, y):
    n, m = x.size(0), y.size(0)
    xx = torch.sum(x ** 2, dim=1, keepdim=True).expand(n, m)
    yy = torch.sum(y ** 2, dim=1, keepdim=True).expand(m, n).t()
    dist = xx + yy - 2 * torch.mm(x, y.t())
    dist = torch.clamp(dist, min=1e-12)  # Clamp to avoid negative values
    return torch.sqrt(dist)  # No need to add epsilon inside sqrt

def energy_distance(x, y):
    n, m = x.size(0), y.size(0)
    d_xy = pairwise_distances(x, y)
    d_xx = pairwise_distances(x, x)
    d_yy = pairwise_distances(y, y)
    e_xy = torch.mean(d_xy)
    e_xx = torch.mean(d_xx)
    e_yy = torch.mean(d_yy)
    energy_dist = 2 * e_xy - e_xx - e_yy
    return energy_dist

def euclidean_distance(x, y):
    assert x.shape == y.shape, "Shapes of x and y must be the same"
    distances = torch.norm(x - y, dim=1)
    return torch.mean(distances)

def weighted_energy_distance(x, y, alpha, beta):
    n, m = x.size(0), y.size(0)
    d_xy = pairwise_distances(x, y)
    d_xx = pairwise_distances(x, x)
    d_yy = pairwise_distances(y, y)
    alpha_x = alpha.view(-1, 1)
    beta_y = beta.view(1, -1)
    e_xy = torch.sum(alpha_x * d_xy * beta_y)
    e_xx = torch.sum(alpha_x * d_xx * alpha_x.t())
    e_yy = torch.sum(beta_y * d_yy * beta_y.t())
    energy_dist = 2 * e_xy - e_xx - e_yy
    return energy_dist

def generate_elliptical_cloud(mean, cov, num_points):
    points = np.random.multivariate_normal(mean, cov, num_points)
    return torch.tensor(points)
