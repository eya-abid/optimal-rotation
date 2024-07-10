import torch
import pymanopt
from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.optimizers import SteepestDescent
import numpy as np
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d import Axes3D


def pairwise_distances(x, y):
    n, m = x.size(0), y.size(0)
    xx = torch.sum(x ** 2, dim=1, keepdim=True).expand(n, m)
    yy = torch.sum(y ** 2, dim=1, keepdim=True).expand(m, n).t()
    dist = xx + yy - 2 * torch.mm(x, y.t())
    dist = torch.clamp(dist, min=1e-12)
    return torch.sqrt(dist)

def weighted_energy_distance(x, y, alpha, beta):
    d_xy = pairwise_distances(x, y)
    d_xx = pairwise_distances(x, x)
    d_yy = pairwise_distances(y, y)

    alpha_x = alpha[:, None]
    beta_y = beta[None, :]

    e_xy = torch.sum(alpha_x[:d_xy.size(0), :] * d_xy * beta_y[:, :d_xy.size(1)])
    e_xx = torch.sum(alpha_x[:d_xx.size(0), :] * d_xx * alpha_x.t()[:, :d_xx.size(1)])
    e_yy = torch.sum(beta_y[:d_yy.size(0), :] * d_yy * beta_y.t()[:, :d_yy.size(1)])

    energy_dist = 2 * e_xy - e_xx - e_yy
    return energy_dist

def create_cost_and_derivates(manifold, A, B, alpha, beta, intermediate_rotations, intermediate_losses):
    A_ = torch.from_numpy(A).float()
    B_ = torch.from_numpy(B).float()
    alpha_ = torch.from_numpy(alpha).float()
    beta_ = torch.from_numpy(beta).float()

    @pymanopt.function.pytorch(manifold)
    def cost(X):
        X_ = X.view(A_.shape[0], A_.shape[1], A_.shape[2]).float()
        total_cost = 0.0
        for a, b, x in zip(A_, B_, X_):
            a_rotated = x @ a
            total_cost += weighted_energy_distance(a_rotated, b, alpha_, beta_)
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
    plt.savefig(filename)
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
        
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

    imageio.mimsave(filename, images, fps=5)

def run(quiet=True):
    x, y = 10, 3

    A = np.random.normal(size=(x, y, y))
    B = np.random.normal(size=(x, y, y))

    print(f"A shape: {A.shape}, B shape: {B.shape}")

    alpha = np.random.rand(x)
    alpha /= alpha.sum()

    beta = np.random.rand(y)
    beta /= beta.sum()

    print(f"Alpha: {alpha}, Sum: {alpha.sum()}")
    print(f"Beta: {beta}, Sum: {beta.sum()}")

    intermediate_rotations = []
    intermediate_losses = []

    manifold = SpecialOrthogonalGroup(y, k=x)
    cost, euclidean_gradient = create_cost_and_derivates(manifold, A, B, alpha, beta, intermediate_rotations, intermediate_losses)
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)

    optimizer = SteepestDescent(verbosity=2 * int(not quiet))
    result = optimizer.run(problem)

    X = result.point

    print(f"X shape: {X.shape}")

    if not quiet:
        print("Optimized Rotation Matrix:\n", X)

    return X, intermediate_rotations, intermediate_losses, A, B

if __name__ == "__main__":
    X, intermediate_rotations, intermediate_losses, A, B = run(quiet=False)
    print(f"Number of intermediate rotations saved: {len(intermediate_rotations)}")
    print(f"Number of intermediate losses saved: {len(intermediate_losses)}")
    plot_losses(intermediate_losses, filename='loss_plot.png')
    create_gif(intermediate_rotations, A, B)

