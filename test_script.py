import pytest
import torch
import numpy as np

from geomloss_rotation import (
    geom_energy_distance,
    geom_sinkhorn_distance,
    geom_gaussian_distance,
    generate_elliptical_cloud,
    run
)

def test_geom_energy_distance():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[1.0, 2.1], [2.9, 4.0]])
    result = geom_energy_distance(x, y)
    assert isinstance(result, torch.Tensor), "Result should be a torch tensor"
    assert result >= 0, "Distance should be non-negative"

def test_geom_sinkhorn_distance():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[1.0, 2.1], [2.9, 4.0]])
    result = geom_sinkhorn_distance(x, y)
    assert isinstance(result, torch.Tensor), "Result should be a torch tensor"
    assert result >= 0, "Distance should be non-negative"

def test_geom_gaussian_distance():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[1.0, 2.1], [2.9, 4.0]])
    result = geom_gaussian_distance(x, y)
    assert isinstance(result, torch.Tensor), "Result should be a torch tensor"
    assert result >= 0, "Distance should be non-negative"

def test_generate_elliptical_cloud():
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.eye(3)
    num_points = 10
    points = generate_elliptical_cloud(mean, cov, num_points)
    assert points.shape == (num_points, 3), "Generated cloud should have the correct shape"
    assert isinstance(points, torch.Tensor), "Generated cloud should be a torch tensor"

@pytest.mark.parametrize("distance_type", ["energy", "sinkhorn", "gaussian"])
def test_run(distance_type):
    X, intermediate_rotations, intermediate_losses, A, B = run(distance_type)
    assert isinstance(X, np.ndarray), "Optimized rotation should be a numpy array"
    assert X.shape == (50, 3, 3), "Optimized rotation should have the correct shape"
    assert len(intermediate_rotations) > 0, "There should be intermediate rotations saved"
    assert len(intermediate_losses) > 0, "There should be intermediate losses saved"

if __name__ == "__main__":
    pytest.main()

