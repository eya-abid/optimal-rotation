import torch
import pymanopt
from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.optimizers import SteepestDescent
import numpy as np
import matplotlib.pyplot as plt

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

    print(f"d_xy shape: {d_xy.shape}")
    print(f"d_xx shape: {d_xx.shape}")
    print(f"d_yy shape: {d_yy.shape}")

    alpha_x = alpha[:, None]
    beta_y = beta[None, :]

    print(f"alpha_x shape: {alpha_x.shape}")
    print(f"beta_y shape: {beta_y.shape}")

    e_xy = torch.sum(alpha_x[:d_xy.size(0), :] * d_xy * beta_y[:, :d_xy.size(1)])
    e_xx = torch.sum(alpha_x[:d_xx.size(0), :] * d_xx * alpha_x.t()[:, :d_xx.size(1)])
    e_yy = torch.sum(beta_y[:d_yy.size(0), :] * d_yy * beta_y.t()[:, :d_yy.size(1)])

    print(f"e_xy: {e_xy.item()}")
    print(f"e_xx: {e_xx.item()}")
    print(f"e_yy: {e_yy.item()}")

    energy_dist = 2 * e_xy - e_xx - e_yy
    return energy_dist

def create_cost_and_derivates(manifold, A, B, alpha, beta):
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
        return total_cost

    return cost, None

def run(quiet=True):
    x, y = 10, 3  # Define the dimensions of the matrices

    # Generate random 2D matrices A and B
    A = np.random.normal(size=(x, y, y))
    B = np.random.normal(size=(x, y, y))

    print(f"A shape: {A.shape}, B shape: {B.shape}")

    alpha = np.random.rand(x)
    alpha /= alpha.sum()

    beta = np.random.rand(y)
    beta /= beta.sum()

    print(f"Alpha: {alpha}, Sum: {alpha.sum()}")
    print(f"Beta: {beta}, Sum: {beta.sum()}")

    manifold = SpecialOrthogonalGroup(y, k=x)
    cost, euclidean_gradient = create_cost_and_derivates(manifold, A, B, alpha, beta)
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)

    optimizer = SteepestDescent(verbosity=2 * int(not quiet))
    result = optimizer.run(problem)

    X = result.point
    intermediate_rotations = result.intermediate_points

    print(f"X shape: {X.shape}")

    if not quiet:
        print("Optimized Rotation Matrix:\n", X)

    return X, intermediate_rotations, A, B, alpha, beta

if __name__ == "__main__":
    X, intermediate_rotations, A, B, alpha, beta = run(quiet=False)
