import torch
import torch.optim as optim

from imports_and_utils import rotation_matrix_z, energy_distance, weighted_energy_distance, generate_elliptical_cloud
    
def optimize_rotation(optimizer_class, R, lr, num_steps, x, y,use_weighted=False, alpha=None, beta=None):
    optimizer = optimizer_class([R], lr=lr)
    loss_values = []
    intermediate_rotations = []

    for step in range(num_steps):
        optimizer.zero_grad()
        x_rotated = x @ R
        if use_weighted:
            loss = weighted_energy_distance(x_rotated, y, alpha, beta)
        else:
            loss = energy_distance(x_rotated, y)
        loss.backward()
        loss_values.append(loss.item())
        optimizer.step()
        with torch.no_grad():
            U, _, V = torch.svd(R)
            R.copy_(U @ V.t())
        intermediate_rotations.append(R.detach().clone())
    return R, loss_values, intermediate_rotations


   
def optimize_rotation_lbfgs(R, lr, num_steps, x, y, use_weighted=False, alpha=None, beta=None):
    optimizer = optim.LBFGS([R], lr=lr)
    loss_values = []
    intermediate_rotations = []

    def closure():
        optimizer.zero_grad()
        x_rotated = x @ R
        if use_weighted:
            loss = weighted_energy_distance(x_rotated, y, alpha, beta)
        else:
            loss = energy_distance(x_rotated, y)
        loss.backward()
        return loss

    for step in range(num_steps):
        optimizer.step(closure)
        with torch.no_grad():
            x_rotated = x @ R
            if use_weighted:
                loss = weighted_energy_distance(x_rotated, y, alpha, beta)
            else:
                loss = energy_distance(x_rotated, y)
            loss_values.append(loss.item())
            U, _, V = torch.svd(R)
            R.copy_(U @ V.t())
        intermediate_rotations.append(R.detach().clone())
    return R, loss_values, intermediate_rotations
    
    
def optimize_rotation_multiple_angles(optimizer_class, lr, num_steps, x, initial_angles, use_weighted=False, alpha=None, beta=None):
    x.requires_grad_(False)
    all_loss_values = []
    all_rotations_and_predictions = []

    for angle in initial_angles:
        R = rotation_matrix_z(angle)
        R.requires_grad_(True)
        optimizer = optimizer_class([R], lr=lr)
        loss_values = []
        rotations_and_predictions = []

        for step in range(num_steps):
            optimizer.zero_grad()
            x_rotated = x @ R
            if use_weighted:
                loss = weighted_energy_distance(x_rotated, x, alpha, beta)
            else:
                loss = energy_distance(x_rotated, x)
            loss.backward()
            loss_values.append(loss.item())
            if step % 20 == 0 or step == num_steps - 1:
                rotations_and_predictions.append((R.detach().clone().numpy(), x_rotated.detach().clone().numpy()))
            optimizer.step()
            with torch.no_grad():
                U, _, V = torch.svd(R)
                R.copy_(U @ V.t())
        all_loss_values.append((angle, loss_values))
        all_rotations_and_predictions.append((angle, rotations_and_predictions))
    return all_loss_values, all_rotations_and_predictions
    
    
def optimize_rotation_multiple_angles_lbfgs(lr, num_steps, x, initial_angles, use_weighted=False, alpha=None, beta=None):
    x.requires_grad_(False)
    all_loss_values = []
    all_rotations_and_predictions = []

    for angle in initial_angles:
        R = rotation_matrix_z(angle)
        R.requires_grad_(True)
        optimizer = optim.LBFGS([R], lr=lr)
        loss_values = []
        rotations_and_predictions = []

        def closure():
            optimizer.zero_grad()
            x_rotated = x @ R
            if use_weighted:
                loss = weighted_energy_distance(x_rotated, x, alpha, beta)
            else:
                loss = energy_distance(x_rotated, x)
            loss.backward()
            return loss

        for step in range(num_steps):
            optimizer.step(closure)
            with torch.no_grad():
                x_rotated = x @ R
                if use_weighted:
                    loss = weighted_energy_distance(x_rotated, x, alpha, beta)
                else:
                    loss = energy_distance(x_rotated, x)
                loss_values.append(loss.item())
                if step % 20 == 0 or step == num_steps - 1:
                    rotations_and_predictions.append((R.detach().clone().numpy(), x_rotated.detach().clone().numpy()))
                U, _, V = torch.svd(R)
                R.copy_(U @ V.t())
        all_loss_values.append((angle, loss_values))
        all_rotations_and_predictions.append((angle, rotations_and_predictions))
    return all_loss_values
