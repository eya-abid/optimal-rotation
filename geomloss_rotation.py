import torch
import pymanopt
from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.optimizers import ConjugateGradient
import numpy as np
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d import Axes3D
from geomloss import SamplesLoss
import os
import pyvista as pv


def geom_energy_distance(x, y):
    loss = SamplesLoss("energy")
    return loss(x, y)

def geom_sinkhorn_distance(x, y, epsilon=0.01, p=2):
    loss = SamplesLoss("sinkhorn", blur=epsilon, p=p)
    return loss(x, y)

def geom_gaussian_distance(x, y, epsilon=0.01, p=2):
    loss = SamplesLoss("gaussian", blur=epsilon)
    return loss(x, y)

def create_cost_and_derivates(manifold, A, B, distance_type, intermediate_rotations, intermediate_losses):
    A_ = torch.from_numpy(A).float()
    B_ = torch.from_numpy(B).float()
    @pymanopt.function.pytorch(manifold)
    def cost(X):
        X_ = X.view(A_.shape[0], A_.shape[2], A_.shape[2]).float()
        total_cost = 0.0
        for a, b, x in zip(A_, B_, X_):
            a_rotated = a @ x
            if distance_type == "energy":
                total_cost += geom_energy_distance(a_rotated, b)
            elif distance_type == "sinkhorn":
                total_cost += geom_sinkhorn_distance(a_rotated, b)
            elif distance_type == "gaussian":
                total_cost += geom_gaussian_distance(a_rotated, b)
        intermediate_rotations.append(X_.detach().clone().numpy())
        intermediate_losses.append(total_cost.item())
        return total_cost
    return cost, None

def plot_losses(intermediate_losses, filename='loss_plot.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(intermediate_losses, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Intermediate Losses During Optimization')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

    
def visualize_point_cloud(cloud, title="Point Cloud Visualization", color="red"):
    # Create a PyVista plotter for this point cloud
    plotter = pv.Plotter()

    # Ensure the cloud is in the correct 3D shape and add it to the plotter
    for sub_cloud in cloud:
        plotter.add_points(sub_cloud, color=color, point_size=10, label=title)
    
    # Add grid and labels
    plotter.add_legend()
    plotter.show_grid()

    # Set plot title and show
    plotter.view_isometric()
    plotter.show(title=title)


def create_gif(intermediate_rotations, A, B, filename='intermediate_rotations_ConjugateGradient.gif'):
    images = []
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for i, X_ in enumerate(intermediate_rotations):
        ax.clear()
        X_ = torch.from_numpy(X_).float()
        A_ = torch.from_numpy(A).float()
        B_ = torch.from_numpy(B).float()
        for a, b, x in zip(A_, B_, X_):
            a_rotated = a @ x  # Apply the rotation
            ax.scatter(a_rotated[:, 0], a_rotated[:, 1], a_rotated[:, 2], c='r', marker='o')
            ax.scatter(b[:, 0], b[:, 1], b[:, 2], c='g', marker='^')
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([-3, 3])
        ax.set_title(f'Intermediate Rotations - Frame {i+1}')
        plt.draw()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

    imageio.mimsave(filename, images, fps=5)
    
    
def generate_elliptical_cloud(mean, cov, num_points):
    points = np.random.multivariate_normal(mean, cov, num_points)
    return torch.tensor(points, dtype=torch.float32)

def run(distance_type, quiet=True):
    num_points = 50
    dim = 3

    mean_A = np.zeros(dim)
    mean_B = np.ones(dim)
    cov_A = np.eye(dim) * 0.5
    cov_B = np.eye(dim) * 0.5
    

    A = np.array([generate_elliptical_cloud(mean_A, cov_A, num_points).numpy() for _ in range(num_points)])
    B = np.array([generate_elliptical_cloud(mean_B, cov_B, num_points).numpy() for _ in range(num_points)])
    
    # A = A - A.mean(0) (moyenne sur les colomnes) ===> A = A - np.mean(A, axis=0)
    
    # Centering the point clouds A and B around the origin
    
    A_centered = A - np.mean(A, axis=(1, 0), keepdims=True)
    B_centered = B - np.mean(B, axis=(1, 0), keepdims=True)


    print(f"A shape: {A.shape}, B shape: {B.shape}")
    print("Visualizing Point Cloud A...")
    visualize_point_cloud(A, title=f"Point Cloud A", color="red")
    print("Visualizing Point Cloud B...")
    visualize_point_cloud(B, title=f"Point Cloud B", color="green")

    intermediate_rotations = []
    intermediate_losses = []

    manifold = SpecialOrthogonalGroup(dim, k=num_points)
    cost, euclidean_gradient = create_cost_and_derivates(manifold, A_centered, B_centered, distance_type, intermediate_rotations, intermediate_losses)
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)

    optimizer = ConjugateGradient(verbosity=2 * int(not quiet))
    X = optimizer.run(problem).point

    print(f"X shape: {X.shape}")

    if not quiet:
        print("Optimized Rotation Matrix:\n", X)

    return X, intermediate_rotations, intermediate_losses, A_centered, B_centered

if __name__ == "__main__":
    for distance_type in ["energy", "sinkhorn", "gaussian"]:
        print(f"Running optimization for {distance_type} distance")
        X, intermediate_rotations, intermediate_losses, A, B = run(distance_type, quiet=False)
        print(f"Number of intermediate rotations saved: {len(intermediate_rotations)}")
        print(f"Number of intermediate losses saved: {len(intermediate_losses)}")
        plot_losses(intermediate_losses, filename=f'loss_plot_50_pts_CG_{distance_type}.png')
#        create_gif(intermediate_rotations, A, B, filename=f'intermediate_rotations_20_pts_SD_{distance_type}.gif')


"""if __name__ == "__main__":
    # Example usage after running the optimization
    for distance_type in ["energy", "sinkhorn", "gaussian"]:
        print(f"Running optimization for {distance_type} distance")
        X, intermediate_rotations, intermediate_losses, A, B = run(distance_type, quiet=False)
        
        # Visualize the point clouds with rotations
        visualize_point_clouds(A, B, rotations=intermediate_rotations, title=f"Visualization of {distance_type} distance")"""
