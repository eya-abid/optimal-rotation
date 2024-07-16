import torch
import numpy as np
import pymanopt
import imageio
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from mpl_toolkits.mplot3d import Axes3D
from pymanopt.manifolds import SpecialOrthogonalGroup



def geom_energy_distance(x, y):
    loss = SamplesLoss("energy")
    return loss(x, y)


def geom_sinkhorn_distance(x, y, epsilon=0.01, p=2):
    loss = SamplesLoss("sinkhorn", blur=epsilon, p=p)
    return loss(x, y)


def geom_gaussian_distance(x, y, epsilon=0.01, p=2):
    loss = SamplesLoss("gaussian", blur=epsilon, p=p)
    return loss(x, y)


def create_cost_and_derivates(manifold, A, B, distance_type, intermediate_rotations, intermediate_losses):
    A_ = torch.from_numpy(A).float()
    B_ = torch.from_numpy(B).float()

    @pymanopt.function.pytorch(manifold)
    def cost(X):
        X_ = X.view(A_.shape[0], A_.shape[1], A_.shape[2]).float()
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



def run(distance_type="energy", quiet=True):
    num_points = 10
    dim = 3

    mean_A = np.zeros(dim)
    mean_B = np.ones(dim)
    cov_A = np.eye(dim) * 0.5
    cov_B = np.eye(dim) * 0.5

    A = np.array([generate_elliptical_cloud(mean_A, cov_A, dim).numpy() for _ in range(num_points)])
    B = np.array([generate_elliptical_cloud(mean_B, cov_B, dim).numpy() for _ in range(num_points)])

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
    distance_types = ["energy", "sinkhorn", "gaussian"]
    for distance_type in distance_types:
        print(f"Running optimization with {distance_type} distance")
        X, intermediate_rotations, intermediate_losses, A, B = run(distance_type=distance_type, quiet=False)
        print(f"Number of intermediate rotations saved: {len(intermediate_rotations)}")
        print(f"Number of intermediate losses saved: {len(intermediate_losses)}")
        plot_losses(intermediate_losses, filename=f'loss_plot_{distance_type}.png')
        create_gif(intermediate_rotations, A[0], B[0], filename=f'intermediate_rotations_{distance_type}.gif')
