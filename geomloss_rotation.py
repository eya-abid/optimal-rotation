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

"""def create_gif(intermediate_rotations, A, B, filename='intermediate_rotations.gif'):
    temp_dir = "temp_gif_frames"
    os.makedirs(temp_dir, exist_ok=True)
    images = []

    for i, X_ in enumerate(intermediate_rotations):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        X_ = torch.from_numpy(X_).float()
        A_ = torch.from_numpy(A).float()
        B_ = torch.from_numpy(B).float()
        print(f"Frame {i}:")
        for a, b, x in zip(A_, B_, X_):
            print(f"a shape: {a.shape}, b shape: {b.shape}, x shape: {x.shape}")
            if a.dim() == 2 and a.shape[1] == 3 and b.dim() == 2 and b.shape[1] == 3:
                a_rotated = a @ x.T
                if a_rotated.dim() == 2 and a_rotated.shape[1] == 3:
                    ax.scatter(a_rotated[:, 0], a_rotated[:, 1], a_rotated[:, 2], c='r', marker='o')
                    ax.scatter(b[:, 0], b[:, 1], b[:, 2], c='g', marker='^')
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([-3, 3])
        ax.set_title(f'Intermediate Rotations - Frame {i+1}')
        plt.draw()

        # Save the current frame
        frame_filename = os.path.join(temp_dir, f"frame_{i:04d}.png")
        fig.savefig(frame_filename)
        plt.close(fig)
        images.append(frame_filename)

    # Create the GIF
    with imageio.get_writer(filename, mode='I', fps=5) as writer:
        for image_file in images:
            image = imageio.imread(image_file)
            writer.append_data(image)

    # Cleanup temporary files
    for image_file in images:
        os.remove(image_file)
    os.rmdir(temp_dir)"""
    

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
    num_points = 10
    dim = 3

    mean_A = np.zeros(dim)
    mean_B = np.ones(dim)
    cov_A = np.eye(dim) * 0.5
    cov_B = np.eye(dim) * 0.5

    A = np.array([generate_elliptical_cloud(mean_A, cov_A, num_points).numpy() for _ in range(num_points)])
    B = np.array([generate_elliptical_cloud(mean_B, cov_B, num_points).numpy() for _ in range(num_points)])

    print(f"A shape: {A.shape}, B shape: {B.shape}")

    intermediate_rotations = []
    intermediate_losses = []

    manifold = SpecialOrthogonalGroup(dim, k=num_points)
    cost, euclidean_gradient = create_cost_and_derivates(manifold, A, B, distance_type, intermediate_rotations, intermediate_losses)
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)

    optimizer = ConjugateGradient(verbosity=2 * int(not quiet))
    X = optimizer.run(problem).point

    print(f"X shape: {X.shape}")

    if not quiet:
        print("Optimized Rotation Matrix:\n", X)

    return X, intermediate_rotations, intermediate_losses, A, B

if __name__ == "__main__":
    for distance_type in ["energy", "sinkhorn", "gaussian"]:
        print(f"Running optimization for {distance_type} distance")
        X, intermediate_rotations, intermediate_losses, A, B = run(distance_type, quiet=False)
        print(f"Number of intermediate rotations saved: {len(intermediate_rotations)}")
        print(f"Number of intermediate losses saved: {len(intermediate_losses)}")
        plot_losses(intermediate_losses, filename=f'loss_plot_{distance_type}.png')
        create_gif(intermediate_rotations, A, B, filename=f'intermediate_rotations_{distance_type}.gif')

