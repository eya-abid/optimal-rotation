import torch
import pymanopt
import vtk
import os
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import imageio
import nibabel as nib
import matplotlib.cm as cm
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from geomloss import SamplesLoss
from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.optimizers import ConjugateGradient
from pymanopt.optimizers import SteepestDescent
from pymanopt.optimizers import TrustRegions

pv.global_theme.allow_empty_mesh = True


def geom_energy_distance(x, y):
    loss = SamplesLoss("energy")
    return loss(x, y)

def geom_sinkhorn_distance(x, y, epsilon=0.5, p=2):
    loss = SamplesLoss("sinkhorn", blur=epsilon, p=p)
    return loss(x, y)

def geom_gaussian_distance(x, y, epsilon=0.5, p=2):
    loss = SamplesLoss("gaussian", blur=epsilon)
    return loss(x, y)



def create_cost_and_derivates(manifold, A, B, distance_type, intermediate_rotations, intermediate_losses, intermediate_rotation_vectors):
    A_ = torch.from_numpy(A).float()  # Shape (num_points, 3)
    B_ = torch.from_numpy(B).float()  # Shape (num_points, 3)
    
    @pymanopt.function.pytorch(manifold)
    def cost(X):
        # Print the shape of X during optimization
        #print(f"Shape of X: {X.shape}")
        
        if X.shape == (3, 3):  # Single 3x3 rotation matrix
            X_ = X.float()
        else:
            raise ValueError(f"Unexpected shape for X: {X.shape}")
        
        # Apply the rotation matrix
        A_rotated = A_ @ X_.T  # Rotate all points using the same rotation matrix

        # Compute the cost based on the chosen distance
        if distance_type == "energy":
            total_cost = geom_energy_distance(A_rotated, B_)
        elif distance_type == "sinkhorn":
            total_cost = geom_sinkhorn_distance(A_rotated, B_)
        elif distance_type == "gaussian":
            total_cost = geom_gaussian_distance(A_rotated, B_)
        else:
            raise ValueError(f"Unknown distance type: {distance_type}")

        # Append the rotation and loss for later tracking
        intermediate_rotations.append(X_.detach().clone().numpy())
        rotation_vector = rotation_matrix_to_rotation_vector(X_.detach().clone().numpy()) 
        intermediate_rotation_vectors.append(rotation_vector)
        intermediate_losses.append(total_cost.item())
        
        return total_cost
    
    return cost, None


def rotation_matrix_to_rotation_vector(R):
    """
    Converts a rotation matrix to a rotation vector.
    :param R: 3x3 rotation matrix
    :return: 3D rotation vector
    """
    theta = np.arccos((np.trace(R) - 1) / 2)
    if np.sin(theta) > 1e-3:
        a_x = (R[2, 1] - R[1, 2]) / (2 * np.sin(theta))
        a_y = (R[0, 2] - R[2, 0]) / (2 * np.sin(theta))
        a_z = (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))
    elif theta == 0:
        # Any axis will do for zero rotation
        a_x, a_y, a_z = 1, 0, 0
    else:  # theta == pi
        a_x = np.sqrt((R[0, 0] + 1) / 2)
        a_y = np.sqrt((R[1, 1] + 1) / 2)
        a_z = np.sqrt((R[2, 2] + 1) / 2)
        # Determine signs based on off-diagonal elements (example shown for a_x)
        if R[2, 1] < 0:
            a_x *= -1
    a = np.array([a_x, a_y, a_z])
    return theta * a


def rotation_vector_to_rotation_matrix(m):
    """
    Converts a rotation vector to a rotation matrix.
    :param m: 3D rotation vector
    :return: 3x3 rotation matrix
    """
    theta = np.linalg.norm(m)
    if theta != 0:
        a = m / theta
    else:
        # Any axis will do for zero rotation
        a = np.array([1, 0, 0])  

    a_hat = np.array([[0, -a[2], a[1]],
                      [a[2], 0, -a[0]],
                      [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(theta) * a_hat + (1 - np.cos(theta)) * np.dot(a_hat, a_hat)
    return R


def plot_losses(intermediate_losses, title, filename):
    plt.figure()
    plt.plot(intermediate_losses)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(filename)
    plt.close()

def generate_elliptical_cloud(mean, cov, num_points):
    points = np.random.multivariate_normal(mean, cov, num_points)
    norms = np.sqrt(np.sum(points**2, axis=1))
    points = points/(norms).reshape(num_points,1)
    return torch.tensor(points, dtype=torch.float32)


def create_ellipsoid_mesh(point_cloud):
    """
    Create a PyVista mesh from the generated point cloud.
    :param point_cloud: The generated point cloud as a torch.Tensor or NumPy array.
    :return: PyVista PolyData mesh.
    """
    # Ensure the point cloud is a NumPy array
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.numpy()

    # If the point cloud has an extra dimension, flatten it
    if point_cloud.ndim == 3:
        point_cloud = point_cloud.reshape(-1, 3)

    # Create and return the PyVista PolyData mesh
    return pv.PolyData(point_cloud)

    
def visualize_ellipsoid_mesh(mesh_A, mesh_B, title_A="Ellipsoid A", title_B="Ellipsoid B"):
    """
    Visualize the ellipsoid point clouds using PyVista.
    :param mesh_A: PyVista mesh for ellipsoid A.
    :param mesh_B: PyVista mesh for ellipsoid B.
    """
    plotter = pv.Plotter(shape=(1, 2))  # Create a 1x2 grid for side-by-side visualization

    # Visualize Ellipsoid A
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh_A, color="red", render_points_as_spheres=True, point_size=10)
    plotter.set_background('white')
    plotter.add_light(pv.Light(position=(5, 5, 5), intensity=0.8))
    plotter.add_text(title_A, font_size=12)
    plotter.view_isometric()

    # Visualize Ellipsoid B
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh_B, color="green", render_points_as_spheres=True, point_size=10)
    plotter.set_background('white')
    plotter.add_light(pv.Light(position=(5, 5, 5), intensity=0.8))
    plotter.add_text(title_B, font_size=12)
    plotter.view_isometric()

    # Show the plot
    plotter.show()
    
    
def create_vtk_files_with_rotations(intermediate_rotations, A, B, base_filename='ellipsoid_rotation', folder='vtk'):
    """
    Create VTK files showing the migration of A towards B using intermediate rotations.
    :param intermediate_rotations: List of rotation matrices applied to A.
    :param A: The original point cloud of A.
    :param B: The original point cloud of B.
    :param base_filename: The base name for the VTK files, e.g., 'ellipsoid_rotation'.
    :param folder: Folder where the VTK files will be saved.
    """
    # Create initial ellipsoids from points
    pts_A = A.copy()

    # Save the static B once
    save_vtk_files(B, rotation_index='B_static', base_filename=base_filename, folder=folder, is_static_B=True)

    # Loop over each rotation matrix and save a corresponding VTK file for rotated A only
    for i, rotation_matrix in enumerate(intermediate_rotations):
        # Convert the rotation matrix to NumPy if it's a tensor
        if isinstance(rotation_matrix, torch.Tensor):
            rotation_matrix = rotation_matrix.numpy()

        # Apply the rotation to ellipsoid A's points
        rotated_A = np.array([pts_A[j] @ rotation_matrix.T for j in range(len(pts_A))])

        # Save the rotated ellipsoid A as VTK file
        save_vtk_files(rotated_A, rotation_index=i + 1, base_filename=base_filename, folder=folder, is_static_B=False)



def save_vtk_files(A, rotation_index, base_filename='ellipsoid_rotation', folder='vtk', is_static_B=False):

    # Create the 'vtks' subfolder inside the main folder
    vtk_folder = os.path.join(folder, 'vtks')
    os.makedirs(vtk_folder, exist_ok=True)  # Ensure the 'vtks' subfolder exists

    # Create PyVista mesh for A
    mesh_A = create_ellipsoid_mesh(A)

    if is_static_B:
        vtk_file_B = os.path.join(vtk_folder, f"{base_filename}_B_static.vtk")
        mesh_A.save(vtk_file_B)
        print(f"Saved VTK file for static B: {vtk_file_B}")
    else:
        # Define file path for saving the VTK file for rotated A
        vtk_file_A = os.path.join(vtk_folder, f"{base_filename}_A_rotation_{rotation_index}.vtk")
        mesh_A.save(vtk_file_A)
        print(f"Saved VTK file for rotated A: {vtk_file_A}")



def create_gif(intermediate_rotations, A, B, filename='ellipsoid_rotation.gif'):
    # Create initial ellipsoids from points
    ellipsoid_A = create_ellipsoid_mesh(A)
    ellipsoid_B = create_ellipsoid_mesh(B)

    # Create a plotter object
    plotter = pv.Plotter(notebook=False, off_screen=True)

    # Add both ellipsoids to the plotter
    plotter.add_mesh(ellipsoid_A, color="red", render_points_as_spheres=True, point_size=10)
    plotter.add_mesh(ellipsoid_B, color="green", render_points_as_spheres=True, point_size=10)
    plotter.set_background('white')
    plotter.add_light(pv.Light(position=(5, 5, 5), intensity=0.8))
    #plotter.enable_shadows()

    # Open a GIF
    plotter.open_gif(filename)

    pts_A = ellipsoid_A.points.copy()

    # Update coordinates and write a frame for each rotation
    for rotation_matrix in intermediate_rotations:
        # Convert the rotation matrix to NumPy if it's a tensor
        if isinstance(rotation_matrix, torch.Tensor):
            rotation_matrix = rotation_matrix.numpy()

        # Apply the rotation to ellipsoid A's points
        rotated_A = pts_A @ rotation_matrix.T
        
        # Ensure rotated_A is a 2D array of shape (n_points, 3)
        if rotated_A.ndim == 3:
            rotated_A = rotated_A.reshape(-1, 3)

        # Update ellipsoid A's coordinates
        plotter.update_coordinates(rotated_A, render=False)

        # Write a frame. This triggers a render.
        plotter.write_frame()

    # Closes and finalizes the GIF
    plotter.close()


def run_optimization(distance_type, experience_index, optimizer_name, A, B, quiet=True, initial_point=None):
    """Run the optimization experiment with the given distance type, using the same A and B point clouds."""
    num_points = A.shape[0]  # Use the number of points from A
    dim = A.shape[1]  # Dimensionality should be 3
    folder = f"{optimizer_name}/{distance_type}/exp{experience_index}"
    os.makedirs(folder, exist_ok=True)

    mesh_A = create_ellipsoid_mesh(A)
    mesh_B = create_ellipsoid_mesh(B)
    visualize_ellipsoid_mesh(mesh_A, mesh_B)
    print(f"A.shape: {A.shape}, B.shape: {B.shape}")

    intermediate_rotations = []
    intermediate_rotation_vectors = []
    intermediate_losses = []

    manifold = SpecialOrthogonalGroup(dim, k=1)
    cost, euclidean_gradient = create_cost_and_derivates(manifold, A, B, distance_type, intermediate_rotations, intermediate_losses, intermediate_rotation_vectors)
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)

    # Dynamically select the optimizer
    if optimizer_name == "ConjugateGradient":
        optimizer = ConjugateGradient(verbosity=2 * int(not quiet))
    elif optimizer_name == "SteepestDescent":
        optimizer = SteepestDescent(verbosity=2 * int(not quiet))
    elif optimizer_name == "TrustRegions":
        optimizer = TrustRegions(verbosity=2 * int(not quiet))
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Use initial_point if provided
    if initial_point is not None:
        X = optimizer.run(problem, initial_point=initial_point).point
    else:
        X = optimizer.run(problem).point

    print(f"X shape: {X.shape}")
    return X, intermediate_rotations, intermediate_rotation_vectors, intermediate_losses



def generate_energy_landscape(A, B, distance_type, rotation_vectors):
    """Generates the energy landscape data for the given point clouds and distance type."""
    N = 150
    # Assume rotation_vectors have been properly generated and passed here
    directions = rotation_vectors / np.linalg.norm(rotation_vectors, axis=1, keepdims=True)
    theta = np.linalg.norm(rotation_vectors, axis=1)
    
    # Create a 3D meshgrid of points in the cube [-pi, pi] x [-pi, pi] x [-pi, pi]
    x = np.linspace(-np.pi, np.pi, N)
    y = np.linspace(-np.pi, np.pi, N)
    z = np.linspace(-np.pi, np.pi, N)
    X, Y, Z = np.meshgrid(x, y, z)
    mask = X**2 + Y**2 + Z**2 <= np.pi**2  # Mask the sphere
    
    img = np.zeros_like(X, dtype=float)  # Initialize array for loss values
    rot_vecs = directions * theta[:, None]  # Combine directions and angles
    
    # Calculate costs for the sampled rotation vectors
    costs = []
    manifold = SpecialOrthogonalGroup(3, k=1)
    for rot_vec in rot_vecs:
        R = rotation_vector_to_rotation_matrix(rot_vec)
        cost, _ = create_cost_and_derivates(manifold, A, B, distance_type, [], [], [])
        costs.append(cost(R).item())  # Evaluate the cost function
    
    # Interpolate the costs to the full grid using griddata
    img = griddata(rot_vecs, np.array(costs), (X, Y, Z), method='linear')
    img_sphere = np.where(mask, img, np.nan)  # Restrict to the sphere
    return img_sphere


def sample_directions(n_samples, random=False):
    if random:
        points = np.random.randn(n_samples, 3)
        points /= np.linalg.norm(points, axis=1)[:, None]
        return points
    else:
        # Fibonacci sphere
        points = []
        phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians
        i = np.arange(n_samples)
        y = 1 - (i / float(n_samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points = np.stack((x, y, z), axis=1)
        return points

def sample_angles(n_samples):
    theta = np.arccos(1 - 2 * np.random.rand(n_samples))  # Using the correct density
    return theta

def sample_vectors(n_samples, random=False):
    if random:
        angles = sample_angles(n_samples)
        directions = sample_directions(n_samples, random=random)
        vectors = (angles[:, None] * directions).reshape(-1, 3)
    else:
        n_angles = int(.5 * (n_samples ** (1/3)))
        n_directions = n_samples // n_angles
        angles = sample_angles(n_angles)
        directions = sample_directions(n_directions)
        angles = angles[:, None, None]
        directions = directions[None, :, :]
        vectors = (angles * directions).reshape(-1, 3)
    print(vectors.shape)
    return vectors


def visualize_energy_landscape_v2(img_sphere, intermediate_rotation_vectors, N):
    grid = pv.ImageData(
        dimensions=img_sphere.shape,
        origin=(-np.pi, -np.pi, -np.pi),  # Center the origin at the middle of the sphere
        spacing=(2 * np.pi / (N - 1), 2 * np.pi / (N - 1), 2 * np.pi / (N - 1)),
    )
    grid.point_data["img"] = img_sphere.flatten(order="F")
    pl = pv.Plotter()
    vmax = np.nanmax(img_sphere)
    vmin = np.nanmin(img_sphere)
    contours = grid.contour(
        isosurfaces=np.linspace(vmax, vmin, 11), scalars="img", method="flying_edges"
    )
    contours.compute_normals(inplace=True)
    sphere_surface = pv.Sphere(
        center=(0, 0, 0), radius=np.pi, theta_resolution=100, phi_resolution=100
    )
    pl.add_mesh(
        contours,
        opacity=0.1,  # Adjusted opacity for better contrast
        cmap="RdBu",
        ambient=0.2,
        diffuse=1,
        interpolation="gouraud",
        show_scalar_bar=True,
        scalar_bar_args=dict(vertical=True),
    )
    pl.add_mesh(
        sphere_surface,
        color="black",
        opacity=0.1,  # Adjusted opacity for better contrast
        culling="front",
        interpolation="pbr",
        roughness=1,
    )
    num_rotations = len(intermediate_rotation_vectors)
    colormap = plt.get_cmap("viridis")  # Use a high-contrast colormap

    # Create a VTK lookup table from the colormap
    lookup_table = vtk.vtkLookupTable()
    lookup_table.SetNumberOfTableValues(num_rotations)
    lookup_table.SetRange(1, num_rotations)  # Set the correct range from 1 to num_rotations
    lookup_table.Build()

    for i in range(num_rotations):
        color = colormap(i / (num_rotations - 1))[:3]
        lookup_table.SetTableValue(i, *color, 1.0)  # Add color to the lookup table

    # Create a grid for spheres with color data based on the optimization steps
    points = np.array(intermediate_rotation_vectors)
    scalars = np.arange(1, num_rotations + 1)  # Starting from 1

    # Add spheres with scalar colors
    for i, point in enumerate(points):
        scalar_value = scalars[i]
        color = colormap((scalar_value - 1) / (num_rotations - 1))[:3]  # Get RGB values from colormap
        sphere = pv.Sphere(radius=0.1, center=point)
        # Add the sphere with scalar value for correct coloring
        pl.add_mesh(sphere, color=color, opacity=1.0)
        
        if i < num_rotations - 1:
            next_point = points[i + 1]
            line = pv.Line(point, next_point)
            pl.add_mesh(line, color="black")

    # Add scalar bar
    scalar_bar = pl.add_scalar_bar(title="Optimization Step", vertical=True, n_labels=5)
    scalar_bar.SetLookupTable(lookup_table)

    pl.enable_ssao(radius=15, bias=0.5)
    pl.enable_anti_aliasing("ssaa")
    pl.camera.zoom(1.1)
    pl.save_graphic("energy_landscape.svg")
    pl.show()


"""if __name__ == "__main__":
    experience_index = 35
    optimizer_name = "ConjugateGradient"
    n_samples = 2000  # Adjust number of samples as needed
    rotation_vectors = sample_vectors(n_samples, random=True)
    dim = 3
    mean_A = np.zeros(dim)
    cov_A = np.diag([1.0, 1.0, 1.0])
    points_A = generate_elliptical_cloud(mean_A, cov_A, 50)
    points_A[:, 0] = points_A[:, 0] * 2
    A = points_A.numpy()
    B = A.copy()  # Ensure the optimum is centered
    
    manifold = SpecialOrthogonalGroup(dim, k=1)
    X_initial = manifold.random_point()
    print(f"Initial rotation matrix:\n{X_initial}")
    
    # Generate the energy landscape data
    distance_type = "gaussian"
    img_sphere = generate_energy_landscape(A, B, distance_type, rotation_vectors)
    
    # Run the optimization with X_initial as the initial point
    X, intermediate_rotations, intermediate_rotation_vectors, intermediate_losses = run_optimization(
        distance_type, experience_index, optimizer_name, A, B, quiet=False, initial_point=X_initial
    )
    print(intermediate_rotation_vectors)
    
    # Visualize the energy landscape with arrows
    visualize_energy_landscape_v2(img_sphere, intermediate_rotation_vectors, N=150)"""

if __name__ == "__main__":
    experience_index = 35
    optimizers = ["ConjugateGradient", "TrustRegions", "SteepestDescent"]  # Add more optimizers if needed
    distance_types = ["energy", "gaussian", "sinkhorn"]
    epsilon_values = [0.1, 0.5, 1.0]  # Different epsilon values to test
    n_samples = 2000  # Adjust number of samples as needed
    rotation_vectors = sample_vectors(n_samples, random=True)
    dim = 3
    mean_A = np.zeros(dim)
    cov_A = np.diag([1.0, 1.0, 1.0])
    points_A = generate_elliptical_cloud(mean_A, cov_A, 50)
    points_A[:, 0] = points_A[:, 0] * 2
    A = points_A.numpy()
    B = A.copy()  # Ensure the optimum is centered

    manifold = SpecialOrthogonalGroup(dim, k=1)
    X_initial = manifold.random_point()
    print(f"Initial rotation matrix:\n{X_initial}")

    def create_cost_and_derivates(manifold, A, B, distance_func, intermediate_rotations, intermediate_losses, intermediate_rotation_vectors):
        A_ = torch.from_numpy(A).float()
        B_ = torch.from_numpy(B).float()

        @pymanopt.function.pytorch(manifold)
        def cost(X):
            X_ = X.float() if X.shape == (3, 3) else ValueError(f"Unexpected shape for X: {X.shape}")

            A_rotated = A_ @ X_.T
            total_cost = distance_func(A_rotated, B_)

            intermediate_rotations.append(X_.detach().clone().numpy())
            rotation_vector = rotation_matrix_to_rotation_vector(X_.detach().clone().numpy())
            intermediate_rotation_vectors.append(rotation_vector)
            intermediate_losses.append(total_cost.item())

            return total_cost

        return cost, None

    for distance_type in distance_types:
        for epsilon in epsilon_values:
            for optimizer_name in optimizers:
                directory = f"results/{optimizer_name}/{distance_type}/epsilon_{epsilon}/"
                os.makedirs(directory, exist_ok=True)

                if distance_type == "energy":
                    distance_func = geom_energy_distance
                elif distance_type == "sinkhorn":
                    distance_func = lambda x, y: geom_sinkhorn_distance(x, y, epsilon=epsilon)
                elif distance_type == "gaussian":
                    distance_func = lambda x, y: geom_gaussian_distance(x, y, epsilon=epsilon)

                img_sphere = generate_energy_landscape(A, B, distance_func, rotation_vectors)
                
                X, intermediate_rotations, intermediate_rotation_vectors, intermediate_losses = run_optimization(
                    distance_func, experience_index, optimizer_name, A, B, quiet=False, initial_point=X_initial
                )
                print(intermediate_rotation_vectors)

                # Save the loss plot
                title = f'{distance_type.capitalize()} with {optimizer_name}, epsilon={epsilon}'
                loss_plot_filename = f'{directory}{distance_type}_{optimizer_name}_epsilon_{epsilon}_loss.png'
                plot_losses(intermediate_losses, title, loss_plot_filename)

                # Visualize and save the energy landscape with arrows
                visualize_energy_landscape_v2(img_sphere, intermediate_rotation_vectors, N=150)
                svg_filename = f'{directory}{distance_type}_{optimizer_name}_epsilon_{epsilon}_landscape.svg'
                plt.savefig(svg_filename)

