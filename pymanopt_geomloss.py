import torch
import numpy as np
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from pymanopt.manifolds import SpecialOrthogonalGroup

def geom_energy_distance(x, y):
    loss = SamplesLoss("energy")
    return loss(x, y)

def geom_sinkhorn_distance(x, y, epsilon=0.01, p=2):
    loss = SamplesLoss("sinkhorn", blur=epsilon, p=p)
    return loss(x, y)

def geom_gaussian_distance(x, y, epsilon=0.01):
    loss = SamplesLoss("gaussian", blur=epsilon)
    return loss(x, y)

def generate_data():
    np.random.seed(0)
    x = torch.tensor(np.random.rand(100, 3), dtype=torch.float32)
    return x

def generate_rotations(manifold, angles):
    rotations = []
    for angle in angles:
        angle_radians = np.radians(angle)
        R = manifold.random_point()
        R[0, 0] = np.cos(angle_radians)
        R[0, 1] = -np.sin(angle_radians)
        R[1, 0] = np.sin(angle_radians)
        R[1, 1] = np.cos(angle_radians)
        R[2, 2] = 1.0
        rotations.append(torch.tensor(R, dtype=torch.float32))
    return rotations

def calculate_distances(x, rotations):
    energy_distances = []
    sinkhorn_distances = []
    gaussian_distances = []

    for R in rotations:
        x_rotated = x @ R.T

        energy_distances.append(geom_energy_distance(x, x_rotated).item())
        sinkhorn_distances.append(geom_sinkhorn_distance(x, x_rotated).item())
        gaussian_distances.append(geom_gaussian_distance(x, x_rotated).item())

    return energy_distances, sinkhorn_distances, gaussian_distances

def plot_distances(angles, energy_distances, sinkhorn_distances, gaussian_distances):
    plt.figure(figsize=(10, 6))
    plt.plot(angles, energy_distances, label='Energy Distance')
    plt.plot(angles, sinkhorn_distances, label='Sinkhorn Distance')
    plt.plot(angles, gaussian_distances, label='Gaussian Distance')
    plt.xlabel('Rotation Angle (degrees)')
    plt.ylabel('Distance')
    plt.title('Distances Between x and Rotated x')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    x = generate_data()
    angles = np.linspace(-90, 90, 180)
    manifold = SpecialOrthogonalGroup(3)
    rotations = generate_rotations(manifold, angles)
    energy_distances, sinkhorn_distances, gaussian_distances = calculate_distances(x, rotations)
    plot_distances(angles, energy_distances, sinkhorn_distances, gaussian_distances)

if __name__ == "__main__":
    main()
