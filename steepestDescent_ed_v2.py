import torch
import pymanopt
from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.optimizers import SteepestDescent
import numpy as np
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d import Axes3D

# Import functions from functions.py
from functions import (weighted_energy_distance, rotation_matrix_z, 
                       pairwise_distances, energy_distance, optimize_rotation,
                       euclidean_distance, generate_elliptical_cloud)



def create_cost_and_derivates(manifold, A, B, intermediate_rotations, intermediate_losses):
    # Convert input matrices A and B to PyTorch tensors and ensure they are of type float
    A_ = torch.from_numpy(A).float()
    B_ = torch.from_numpy(B).float()

    @pymanopt.function.pytorch(manifold)
    def cost(X):
        # Reshape X to match the shape of rotation matrices
        X_ = X.view(A_.shape[0], A_.shape[1], A_.shape[2]).float()
        total_cost = 0.0
        for a, b, x in zip(A_, B_, X_):
            # Apply rotation matrix x to a and compute the energy distance to b
            a_rotated = x @ a  # Apply rotation matrix x to a
            total_cost += energy_distance(a_rotated, b)
        intermediate_rotations.append(X_.detach().clone().numpy())
        intermediate_losses.append(total_cost.item())
        return total_cost

    return cost, None
    
def plot_losses(intermediate_losses, filename='loss_plot.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(intermediate_losses, label='Weighted Energy Distance Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Intermediate Losses During Optimization')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)  # Save the plot as an image file
    plt.show() 

def create_gif(intermediate_rotations, A, B, filename='intermediate_rotations.gif'):
    images = []
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for X_ in intermediate_rotations:
        ax.clear()
        X_ = torch.from_numpy(X_).float()
        A_ = torch.from_numpy(A).float()
        B_ = torch.from_numpy(B).float()
        for a, b, x in zip(A_, B_, X_):
            a_rotated = x @ a
            ax.scatter(a_rotated[:, 0], a_rotated[:, 1], a_rotated[:, 2], c='r', marker='o')
            ax.scatter(b[:, 0], b[:, 1], b[:, 2], c='g', marker='^')
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([-3, 3])
        ax.set_title('Intermediate Rotations')
        plt.draw()

        # Convert plot to an image
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

    # Save the images as a GIF
    imageio.mimsave(filename, images, fps=5)

def run(quiet=True):
    x, y = 10, 3  # Define the dimensions of the matrices

    # Generate random 2D matrices A and B
    A = np.random.normal(size=(x, y, y))
    B = np.random.normal(size=(x, y, y))

    print(f"A shape: {A.shape}, B shape: {B.shape}")

    # Initialize lists to store intermediate rotations and losses
    intermediate_rotations = []
    intermediate_losses = []

    # Define the manifold for 3D rotations
    manifold = SpecialOrthogonalGroup(y, k=x)  # Ensure y matches the dimension of the rotation matrix
    cost, euclidean_gradient = create_cost_and_derivates(manifold, A, B, intermediate_rotations, intermediate_losses)
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)

    optimizer = SteepestDescent(verbosity=2 * int(not quiet))
    X = optimizer.run(problem).point

    print(f"X shape: {X.shape}")

    if not quiet:
        print("Optimized Rotation Matrix:\n", X)

    # Return the optimized rotation matrix, intermediate rotations, and intermediate losses
    return X, intermediate_rotations, intermediate_losses

if __name__ == "__main__":
    X, intermediate_rotations, intermediate_losses = run(quiet=False)
    print(f"Number of intermediate rotations saved: {len(intermediate_rotations)}")
    print(f"Number of intermediate losses saved: {len(intermediate_losses)}")
    plot_losses(intermediate_losses, filename='loss_plot.png')
    create_gif(intermediate_rotations, A, B)
