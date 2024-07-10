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


def weighted_energy_distance(x, y, alpha, beta):
    n, m = x.size(0), y.size(0)

    # Pairwise distances
    d_xy = pairwise_distances(x, y)
    d_xx = pairwise_distances(x, x)
    d_yy = pairwise_distances(y, y)

    # Weighted expectations
    e_xy = torch.sum(alpha[:, None] * d_xy * beta[None, :])
    e_xx = torch.sum(alpha[:, None] * d_xx * alpha[None, :])
    e_yy = torch.sum(beta[:, None] * d_yy * beta[None, :])

    # Energy distance
    energy_dist = 2 * e_xy - e_xx - e_yy
    return energy_dist


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

    # Pairwise distances
    d_xy = pairwise_distances(x, y)
    d_xx = pairwise_distances(x, x)
    d_yy = pairwise_distances(y, y)

    # Calculate the expectations
    e_xy = torch.mean(d_xy)
    e_xx = torch.mean(d_xx)
    e_yy = torch.mean(d_yy)

    # Energy distance
    energy_dist = 2 * e_xy - e_xx - e_yy
    return energy_dist





def euclidean_distance(x, y):
    # Ensure x and y have the same shape
    assert x.shape == y.shape, "Shapes of x and y must be the same"

    # Calculate pairwise Euclidean distances
    distances = torch.norm(x - y, dim=1)

    # Return the mean distance
    return torch.mean(distances)




    
# Function to generate cloud-like points in a deformed ellipse
def generate_elliptical_cloud(mean, cov, num_points):
    points = np.random.multivariate_normal(mean, cov, num_points)
    return torch.tensor(points)






def optimize_rotation(optimizer_class, R, lr, num_steps, x, y):
    # Initialize the optimizer
    optimizer = optimizer_class([R], lr=lr)

    # Store loss values
    loss_values = []
    
    # List to store intermediate rotations
    intermediate_rotations = []

    for step in range(num_steps):
        optimizer.zero_grad()

        # Rotate x using the rotation matrix R
        x_rotated = x @ R
        y = x_rotated

        # Calculate energy distance between x_rotated and y
        loss = energy_distance(x, y)

        # Backpropagate the loss
        loss.backward()

        # Store the loss value
        loss_values.append(loss.item())

        # Optimize the rotation matrix
        optimizer.step()

        # Ensure R remains a valid rotation matrix
        with torch.no_grad():
            U, _, V = torch.svd(R)
            R.copy_(U @ V.t())
            
        # Store the intermediate rotation matrix
        intermediate_rotations.append(R.detach().clone())
        
    print(f"Final Loss: {loss.item()}")    
    print(f"Optimized Rotation Matrix R:\n{R.detach()}")
    return R, loss_values, intermediate_rotations



def optimize_rotation_lbfgs(R, lr, num_steps, x, y):
    # Initialize the optimizer
    optimizer = optim.LBFGS([R], lr=lr)

    # Store loss values
    loss_values = []
    
    # List to store intermediate rotations
    intermediate_rotations = []

    def closure():
        optimizer.zero_grad()

        # Rotate x using the rotation matrix R
        x_rotated = x @ R
        y = x_rotated

        # Calculate energy distance between x_rotated and y
        loss = energy_distance(x, y)

        # Backpropagate the loss
        loss.backward()

        return loss

    for step in range(num_steps):
        optimizer.step(closure)

        # Calculate the loss after the optimization step
        with torch.no_grad():
            x_rotated = x @ R
            loss = energy_distance(x_rotated, y)

        # Store the loss value
        loss_values.append(loss.item())

        # Ensure R remains a valid rotation matrix
        with torch.no_grad():
            U, _, V = torch.svd(R)
            R.copy_(U @ V.t())
            
        # Store the intermediate rotation matrix
        intermediate_rotations.append(R.detach().clone())

    print(f"Final Loss: {loss.item()}")
    print(f"Optimized Rotation Matrix R:\n{R.detach()}")

    return R, loss_values, intermediate_rotations
    
    
 
   
def optimize_rotation_multiple_angles(optimizer_class, lr, num_steps, x, initial_angles):
    # Ensure tensors do not require gradients
    x.requires_grad_(False)

    # Store all losses for analysis
    all_loss_values = []
    all_rotations_and_predictions = []

    for angle in initial_angles:
        # Initialize rotation matrix R with a specific initial angle
        R = rotation_matrix_z(angle)
        R.requires_grad_(True)  # Enable gradient tracking for R

        # Define the optimizer
        optimizer = optimizer_class([R], lr=lr)

        # Store loss values for this run
        loss_values = []
        rotations_and_predictions = []

        for step in range(num_steps):
            optimizer.zero_grad()

            # Rotate x using the rotation matrix R
            x_rotated = x @ R

            # Calculate energy distance between x_rotated and x
            loss = energy_distance(x_rotated, x)

            # Backpropagate the loss
            loss.backward()

            # Store the loss value
            loss_values.append(loss.item())

            # Store the current rotation matrix every 20 steps and the last step
            if step % 20 == 0 or step == num_steps - 1:
                rotations_and_predictions.append((R.detach().clone().numpy(), x_rotated.detach().clone().numpy()))

            # Optimize the rotation matrix
            optimizer.step()

            # Ensure R remains a valid rotation matrix
            with torch.no_grad():
                U, _, V = torch.svd(R)
                R.copy_(U @ V.t())

        all_loss_values.append((angle, loss_values))
        all_rotations_and_predictions.append((angle, rotations_and_predictions))

    return all_loss_values, all_rotations_and_predictions






  
def optimize_rotation_multiple_angles_lbfgs(lr, num_steps, x, initial_angles):
    # Ensure tensors do not require gradients
    x.requires_grad_(False)

    # Store all losses for analysis
    all_loss_values = []
    all_rotations_and_predictions = []

    for angle in initial_angles:
        # Initialize rotation matrix R with a specific initial angle
        R = rotation_matrix_z(angle)
        R.requires_grad_(True)  # Enable gradient tracking for R

        # Define the optimizer
        optimizer = optim.LBFGS([R], lr=lr)

        # Store loss values for this run
        loss_values = []
        rotations_and_predictions = []

        def closure():
            optimizer.zero_grad()

            # Rotate x using the rotation matrix R
            x_rotated = x @ R

            # Calculate energy distance between x_rotated and x
            loss = energy_distance(x_rotated, x)

            # Backpropagate the loss
            loss.backward()

            return loss

        for step in range(num_steps):
            # Optimize the rotation matrix
            optimizer.step(closure)

            # Calculate the loss after the optimization step
            with torch.no_grad():
                x_rotated = x @ R
                loss = energy_distance(x_rotated, x)

            # Store the loss value
            loss_values.append(loss.item())
            
            # Store the current rotation matrix every 20 steps and the last step
            if step % 20 == 0 or step == num_steps - 1:
                rotations_and_predictions.append((R.detach().clone().numpy(), x_rotated.detach().clone().numpy()))

        all_loss_values.append((angle, loss_values))
        all_rotations_and_predictions.append((angle, rotations_and_predictions))

    return all_loss_values
