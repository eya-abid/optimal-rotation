import torch
import numpy as np

from imports_and_utils import generate_elliptical_cloud

mean_x = [0, 0, 0]
cov_x = [[1, 0.5, 0.2], [0.5, 2, 0.3], [0.2, 0.3, 1]]
cov_y = [[1, 0.5, 0.2], [0.6, 1, 0.8], [0.2, 0.3, 1]]

num_points = 500
x = generate_elliptical_cloud(mean_x, cov_x, num_points)
alpha = torch.tensor(np.random.rand(num_points) * 1e-1)
beta = alpha

alpha_normalized = (alpha - alpha.min()) / (alpha.max() - alpha.min())
beta_normalized = (beta - beta.min()) / (beta.max() - beta.min())
