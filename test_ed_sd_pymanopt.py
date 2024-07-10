import torch
import pymanopt
from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.optimizers import SteepestDescent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def pairwise_distances(x, y):
    # Calculate pairwise Euclidean distances between points in x and y
    n, m = x.size(0), y.size(0)
    xx = torch.sum(x ** 2, dim=1, keepdim=True).expand(n, m)
    yy = torch.sum(y ** 2, dim=1, keepdim=True).expand(m, n).t()
    dist = xx + yy - 2 * torch.mm(x, y.t())
    dist = torch.clamp(dist, min=1e-12)  # Clamp to avoid negative values
    return torch.sqrt(dist)  # No need to add epsilon inside sqrt

def energy_distance(x, y):
    # Calculate energy distance between sets of points x and y
    n, m = x.size(0), y.size(0)
    d_xy = pairwise_distances(x, y)
    d_xx = pairwise_distances(x, x)
    d_yy = pairwise_distances(y, y)
    e_xy = torch.mean(d_xy)
    e_xx = torch.mean(d_xx)
    e_yy = torch.mean(d_yy)
    energy_dist = 2 * e_xy - e_xx - e_yy
    return energy_dist

def create_cost_and_derivates(manifold, A, B):
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
        return total_cost

    return cost, None

class CustomSteepestDescent(SteepestDescent):
    def __init__(self, intermediate_rotations, loss_values, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intermediate_rotations = intermediate_rotations
        self.loss_values = loss_values

    def step(self, *args, **kwargs):
        cost = super().step(*args, **kwargs)
        point = self._manifold.rand()
        point_reshaped = point.view(self._problem.manifold.k, self._problem.manifold.n, self._problem.manifold.n).detach().numpy()
        self.intermediate_rotations.append(point_reshaped)
        self.loss_values.append(cost)
        return cost

def run(quiet=True):
    # Define the dimensions of the matrices
    k, n, m = 10, 2, 2  # Using 2D for better visualization

    # Generate random 2D matrices A and B
    A = np.random.normal(size=(k, n, m))
    B = np.random.normal(size=(k, n, m))

    print(f"A shape: {A.shape}, B shape: {B.shape}")

    # Define the manifold for rotations
    manifold = SpecialOrthogonalGroup(n, k=k)  # n must match the dimension of the rotation matrices
    cost, euclidean_gradient = create_cost_and_derivates(manifold, A, B)
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)

    # Store intermediate rotations and losses for visualization
    intermediate_rotations = []
    loss_values = []

    # Initialize the custom optimizer
    optimizer = CustomSteepestDescent(intermediate_rotations, loss_values, verbosity=2 * int(not quiet))

    # Run the optimization
    X = optimizer.run(problem).point

    print(f"X shape: {X.shape}")

    if not quiet:
        print("Optimized Rotation Matrix:\n", X)

    # Plotting the intermediate rotations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    if len(loss_values) > 0:
        ax2.set_xlim(0, len(loss_values))
        ax2.set_ylim(0, max(loss_values))
    else:
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

    scat = ax1.scatter([], [], c='red', label='Rotated A')
    scat_b = ax1.scatter(B[:, :, 0], B[:, :, 1], c='blue', label='Target B')
    line, = ax2.plot([], [], 'b-', label='Loss')

    def update(frame):
        if frame < len(intermediate_rotations):
            X_rot = intermediate_rotations[frame]
            A_rotated = np.einsum('bij,bjk->bik', X_rot, A)
            scat.set_offsets(A_rotated.reshape(-1, 2))
            if len(loss_values) > 0:
                line.set_data(range(frame + 1), loss_values[:frame + 1])
        return scat, line

    if len(intermediate_rotations) > 0:
        ani = FuncAnimation(fig, update, frames=len(intermediate_rotations), blit=True)
    else:
        print("No frames to animate.")
    ax1.legend()
    ax2.legend()
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Evolution')
    plt.show()

if __name__ == "__main__":
    run(quiet=False)

