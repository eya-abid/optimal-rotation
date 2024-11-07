import numpy as np
import torch
import matplotlib.pyplot as plt
from optimization_functions import weighted_energy_distance, energy_distance, rotation_matrix_z, optimize_rotation_multiple_angles, optimize_rotation_multiple_angles_lbfgs
from imports_and_utils import rotation_matrix_z, energy_distance, weighted_energy_distance, generate_elliptical_cloud



def plot_losses_for_angles(all_loss_values, initial_angles, filename):
    plt.figure(figsize=(10, 6))
    for angle, loss_values in all_loss_values:
        plt.plot(loss_values, label=f'Initial Angle {angle} degrees')
    plt.xlabel('Optimization Steps')
    plt.ylabel('Loss')
    plt.title('Loss Over Optimization Steps for Various Initial Rotation Angles')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def run_optimization():
    num_points = 100
    mean = [0, 0, 0]
    cov = [[1, 0.8, 0.5], [0.8, 1, 0.3], [0.5, 0.3, 1]]
    x = generate_elliptical_cloud(mean, cov, num_points)
    y = generate_elliptical_cloud(mean, cov, num_points)

    alpha = torch.rand(num_points)
    beta = torch.rand(num_points)
    alpha /= alpha.sum()
    beta /= beta.sum()

    initial_angles = [30, 45, 90, 135, 180]
    num_steps = 300
    lr = 0.01
    use_weighted = True

    #all_loss_values, all_rotations_and_predictions = optimize_rotation_multiple_angles(torch.optim.RMSprop, lr, num_steps, x, initial_angles, use_weighted=use_weighted, alpha=alpha, beta=beta)
    #all_loss_values, all_rotations_and_predictions = optimize_rotation_multiple_angles(torch.optim.Adam, lr, num_steps, x, initial_angles, use_weighted=use_weighted, alpha=alpha, beta=beta)
    #all_loss_values, all_rotations_and_predictions = optimize_rotation_multiple_angles(torch.optim.SGD, lr, num_steps, x, initial_angles, use_weighted=use_weighted, alpha=alpha, beta=beta)
    all_loss_values, all_rotations_and_predictions = optimize_rotation_multiple_angles_lbfgs(lr, num_steps, x, initial_angles, use_weighted=use_weighted, alpha=alpha, beta=beta)

    plot_losses_for_angles(all_loss_values, initial_angles, filename='loss_plot_many_angles_lbfgs.png')

if __name__ == "__main__":
    run_optimization()
