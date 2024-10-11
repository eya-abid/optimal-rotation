import torch
import pymanopt
import vtk
import os
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import imageio
import nibabel as nib
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from geomloss import SamplesLoss
from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.optimizers import ConjugateGradient
from pymanopt.optimizers import SteepestDescent
from pymanopt.optimizers import TrustRegions



def geom_energy_distance(x, y):
    loss = SamplesLoss("energy")
    return loss(x, y)

def geom_sinkhorn_distance(x, y, epsilon=0.5, p=2):
    loss = SamplesLoss("sinkhorn", blur=epsilon, p=p)
    return loss(x, y)

def geom_gaussian_distance(x, y, epsilon=0.5, p=2):
    loss = SamplesLoss("gaussian", blur=epsilon)
    return loss(x, y)



def create_cost_and_derivates(manifold, A, B, distance_type, intermediate_rotations, intermediate_losses):
    A_ = torch.from_numpy(A).float()  # Shape (num_points, 3)
    B_ = torch.from_numpy(B).float()  # Shape (num_points, 3)
    
    @pymanopt.function.pytorch(manifold)
    def cost(X):
        # Print the shape of X during optimization
        print(f"Shape of X: {X.shape}")
        
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

    if np.sin(theta) != 0:
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


def plot_losses(intermediate_losses, filename='loss_plot.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(intermediate_losses, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Intermediate Losses During Optimization')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()



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
    #plotter.enable_shadows()
    plotter.add_text(title_A, font_size=12)
    plotter.view_isometric()

    # Visualize Ellipsoid B
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh_B, color="green", render_points_as_spheres=True, point_size=10)
    plotter.set_background('white')
    plotter.add_light(pv.Light(position=(5, 5, 5), intensity=0.8))
    #plotter.enable_shadows()
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
    """
    Save the rotated point cloud A as VTK file, and B only once as static.
    :param A: Rotated point cloud for ellipsoid A.
    :param rotation_index: Index of the rotation step, used for naming the VTK files.
    :param base_filename: Base name to use for saving the files (default is 'ellipsoid_rotation').
    :param folder: Folder where the VTK files will be saved.
    :param is_static_B: Boolean flag to indicate if this is the static B ellipsoid.
    """
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

    

def run(distance_type, experience_index, optimizer_name, A, B, quiet=True):
    """
    Run the optimization experiment with the given distance type, using the same A and B point clouds.
    :param distance_type: Distance measure to use (e.g., "energy", "sinkhorn", "gaussian").
    :param experience_index: The experiment index.
    :param optimizer_name: The optimizer name (e.g., "ConjugateGradient").
    :param A: Point cloud for ellipsoid A.
    :param B: Point cloud for ellipsoid B.
    :param quiet: Whether to suppress optimizer output.
    """
    num_points = A.shape[0]  # Use the number of points from A
    dim = A.shape[1]  # Dimensionality should be 3

    # Correct folder structure: optimizer_name/distance_type/exp{experience_index}/vtks
    folder = f"{optimizer_name}/{distance_type}/exp{experience_index}"
    os.makedirs(folder, exist_ok=True)

    # Visualize ellipsoids
    mesh_A = create_ellipsoid_mesh(A)
    mesh_B = create_ellipsoid_mesh(B)
    visualize_ellipsoid_mesh(mesh_A, mesh_B)

    print(f"A.shape: {A.shape}, B.shape: {B.shape}")

    intermediate_rotations = []
    intermediate_losses = []

    # Define the SO(3) manifold
    manifold = SpecialOrthogonalGroup(dim, k=1)
    cost, euclidean_gradient = create_cost_and_derivates(manifold, A, B, distance_type, intermediate_rotations, intermediate_losses)
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)

    optimizer = ConjugateGradient(verbosity=2 * int(not quiet))
    X = optimizer.run(problem).point

    print(f"X shape: {X.shape}")

    # Save the loss plot in the correct folder
    plot_losses(intermediate_losses, filename=os.path.join(folder, f'loss_plot_{distance_type}.png'))

    # Save VTK files in the 'vtks' subfolder inside the experiment folder
    create_vtk_files_with_rotations(intermediate_rotations, A, B, base_filename=f'ellipsoid_{distance_type}', folder=folder)

    # Save the GIF in the correct folder
    create_gif(intermediate_rotations, A, B, filename=os.path.join(folder, f'migration_A_to_B_{distance_type}.gif'))

    return X, intermediate_rotations, intermediate_losses
    


def run_experiment_with_pymanopt_rotation(distance_type, experience_index, optimizer_name, A, B_rotated, X_initial, quiet=True):
    """
    Run the optimization experiment for a given distance type, using the same set of points A, 
    and the rotated version of A (B_rotated).
    """
    num_points = 25
    dim = 3

    # Create folder structure: optimizer_name/distance_type/expX/vtks
    folder = f"{optimizer_name}/{distance_type}/exp{experience_index}"
    os.makedirs(folder, exist_ok=True)

    # Visualize the original A and B_rotated (rotated A)
    mesh_A = create_ellipsoid_mesh(A)
    mesh_B_rotated = create_ellipsoid_mesh(B_rotated)
    visualize_ellipsoid_mesh(mesh_A, mesh_B_rotated)

    print(f"A.shape: {A.shape}, B_rotated.shape: {B_rotated.shape}")

    intermediate_rotations = []
    intermediate_losses = []

    # Use Pymanopt to optimize the rotation matrix that aligns A with B_rotated
    manifold = SpecialOrthogonalGroup(dim, k=1)
    cost, euclidean_gradient = create_cost_and_derivates(manifold, A, B_rotated, distance_type, intermediate_rotations, intermediate_losses)
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)

    optimizer = ConjugateGradient(verbosity=2 * int(not quiet))
    X_optimized = optimizer.run(problem).point  # Optimized rotation matrix
    print(f"Optimized rotation matrix for {distance_type}:\n{X_optimized}")

    # Save the loss plot in the folder
    plot_losses(intermediate_losses, filename=os.path.join(folder, f'loss_plot_{distance_type}.png'))

    # Save VTK files in the 'vtks' subfolder inside the folder
    create_vtk_files_with_rotations(intermediate_rotations, A, B_rotated, base_filename=f'ellipsoid_{distance_type}', folder=folder)

    # Save the GIF in the folder
    create_gif(intermediate_rotations, A, B_rotated, filename=os.path.join(folder, f'migration_A_to_B_{distance_type}.gif'))

    return X_optimized, intermediate_rotations, intermediate_losses


def run_optimization_experiment_phase_2(experience_index=10, optimizer_name="ConjugateGradient", num_points=50, dim=3):
    """
    Runs the optimization experiment for different distance types 
    using generated ellipsoid point clouds.
    """

    # Step 1: Generate the ellipsoid point clouds A and B 
    mean_A = np.zeros(dim)
    mean_B = np.zeros(dim)
    
    cov_A = np.diag([1.0, 1.0, 1.0]) 
    cov_B = np.diag([1.0, 1.0, 1.0])

    points_A = generate_elliptical_cloud(mean_A, cov_A, num_points)
    points_A[:, 0] = points_A[:, 0] * 2 
    points_B = generate_elliptical_cloud(mean_B, cov_B, num_points)
    points_B[:, 1] = points_B[:, 1] * 2

    A = points_A.numpy()
    B = points_B.numpy()

    # Step 2: Run the experiment for each distance measure
    for distance_type in ["energy", "sinkhorn", "gaussian"]:
        print(f"Running optimization for {distance_type} distance, experiment {experience_index}")
        X, intermediate_rotations, intermediate_losses = run(
            distance_type, experience_index, optimizer_name, A, B, quiet=False)
        print(f"Number of intermediate rotations saved: {len(intermediate_rotations)}")
        print(f"Number of intermediate losses saved: {len(intermediate_losses)}")

        
def run_pymanopt_rotation_experiment_phase_1(experience_index=20, optimizer_name="ConjugateGradient", num_points=50, dim=3):
    """
    Runs the Pymanopt rotation optimization experiment for different distance types
    using a generated ellipsoid point cloud and a random initial rotation.
    """

    # Step 1: Generate the ellipsoid point cloud A
    mean_A = np.zeros(dim)
    cov_A = np.diag([1.0, 1.0, 1.0])
    points_A = generate_elliptical_cloud(mean_A, cov_A, num_points)
    points_A[:, 0] = points_A[:, 0] * 2
    A = points_A.numpy()

    # Step 2: Generate a random initial rotation matrix
    manifold = SpecialOrthogonalGroup(dim, k=1)
    X_initial = manifold.random_point()
    print(f"Initial rotation matrix:\n{X_initial}")

    # Step 3: Apply the rotation to A to get B_rotated
    B_rotated = np.array([A[i] @ X_initial.T for i in range(A.shape[0])])

    # Step 4: Run the optimization experiment for each distance type
    for distance_type in ["energy", "sinkhorn", "gaussian"]:
        print(f"Running optimization for {distance_type} distance, experiment {experience_index}")
        X_optimized, intermediate_rotations, intermediate_losses = run_experiment_with_pymanopt_rotation(
            distance_type, experience_index, optimizer_name, A, B_rotated, X_initial, quiet=False)
        print(f"Number of intermediate rotations saved: {len(intermediate_rotations)}")
        print(f"Number of intermediate losses saved: {len(intermediate_losses)}")


def generate_energy_landscape(A, B, distance_type, num_samples=1001, N=100):
    """
    Generates the energy landscape data for the given point clouds and distance type.

    Args:
        A: Point cloud A.
        B: Point cloud B.
        distance_type: Type of distance metric ('energy', 'sinkhorn', 'gaussian').
        num_samples: Number of sample points for interpolation.
        N: Resolution of the 3D grid.

    Returns:
        img_sphere: 3D array containing the energy landscape data.
    """
    # Create a 3D meshgrid of points in the cube [-pi,pi] x [-pi,pi] x [-pi,pi]
    x = np.linspace(-np.pi, np.pi, N)
    y = np.linspace(-np.pi, np.pi, N)
    z = np.linspace(-np.pi, np.pi, N)
    X, Y, Z = np.meshgrid(x, y, z)

    # Mask the sphere centered at the origin with radius pi
    mask = X**2 + Y**2 + Z**2 <= np.pi**2

    # Initialize an array to store the loss values
    img = np.zeros_like(X, dtype=float)

    # Generate random rotation vectors within the sphere
    rot_vecs = np.random.rand(num_samples, 3) * 2 * np.pi - np.pi
    rot_vecs = rot_vecs[np.linalg.norm(rot_vecs, axis=1) <= np.pi]

    # Calculate costs for the sampled rotation vectors
    costs = []
    manifold = SpecialOrthogonalGroup(3, k=1)  # Define the manifold here
    for rot_vec in rot_vecs:
        R = rotation_vector_to_rotation_matrix(rot_vec)
        cost, _ = create_cost_and_derivates(manifold, A, B, distance_type, [], [])
        costs.append(cost(R).item())  # Evaluate the cost function

    # Interpolate the costs to the full grid using griddata
    img = griddata(rot_vecs, np.array(costs), (X, Y, Z), method='linear')

    # Restrict the img to the sphere, use NaN outside of it
    img_sphere = np.where(mask, img, np.nan)

    return img_sphere


def visualize_energy_landscape(img_sphere, N=100):
    """
    Visualizes the energy landscape data using PyVista.

    Args:
        img_sphere: 3D array containing the energy landscape data.
        N: Resolution of the 3D grid.
    """

    # Create a volumetric rendering of img
    grid = pv.ImageData(
        dimensions=img_sphere.shape,
        origin=(-np.pi, -np.pi, -np.pi),
        spacing=(2 * np.pi / (N - 1), 2 * np.pi / (N - 1), 2 * np.pi / (N - 1)),)
        
    grid.point_data["img"] = img_sphere.flatten(order="F")  # Flatten the image to a 1D array

    # Create a PyVista plotter
    pl = pv.Plotter()

    # Extract contours
    vmax = np.nanmax(img_sphere)
    vmin = np.nanmin(img_sphere)
    contours = grid.contour(
        isosurfaces=np.linspace(vmax, vmin, 11), scalars="img", method="flying_edges"
    )
    # Compute surface normals for better shading
    contours.compute_normals(inplace=True)

    # Create a sphere to intersect the contours
    sphere_surface = pv.Sphere(
        center=(0, 0, 0), radius=np.pi, theta_resolution=100, phi_resolution=100
    )
    # contours = contours.boolean_intersection(sphere_b)

    pl.add_mesh(
        contours,
        opacity=.3,
        cmap="RdBu",
        ambient=0.2,
        diffuse=1,
        interpolation="gouraud",
        show_scalar_bar=True,
        scalar_bar_args=dict(vertical=True),
    )
    if True:
        pl.add_mesh(
            sphere_surface,
            color="black",
            opacity=.1,
            culling="front",
            interpolation="pbr",
            roughness=1,
        )

    pl.enable_ssao(radius=15, bias=0.5)
    pl.enable_anti_aliasing("ssaa")
    pl.camera.zoom(1.1)
    pl.show()

if __name__ == "__main__":

    # Test the conversion functions
    test_matrix = np.array([[0.707, -0.707, 0],
                            [0.707, 0.707, 0],
                            [0, 0, 1]])  # Example rotation matrix

    # Convert rotation matrix to rotation vector
    test_vector = rotation_matrix_to_rotation_vector(test_matrix)
    print(f"Rotation vector: {test_vector}")

    # Convert rotation vector back to rotation matrix
    reconstructed_matrix = rotation_vector_to_rotation_matrix(test_vector)
    print(f"Reconstructed matrix:\n{reconstructed_matrix}")


if __name__ == "__main__":

    dim = 3
    mean_A = np.zeros(dim)
    cov_A = np.diag([1.0, 1.0, 1.0])  # Ellipsoid with different variances along each axis
    points_A = generate_elliptical_cloud(mean_A, cov_A, 100)
    points_A[:, 0] = points_A[:, 0] * 2
    A = points_A.numpy()  # Shape is (num_points, 3)

    manifold = SpecialOrthogonalGroup(dim, k=1)
    X_initial = manifold.random_point()  # Random initial 3x3 rotation matrix
    print(f"Initial rotation matrix:\n{X_initial}")

    # Step 3: Apply the rotation to A to get B_rotated
    B = np.array([A[i] @ X_initial.T for i in range(A.shape[0])])


    # Generate the energy landscape data
    distance_type = "energy"  # or "sinkhorn" or "energy"
    img_sphere = generate_energy_landscape(A, B, distance_type)

    # Visualize the energy landscape
    visualize_energy_landscape(img_sphere)
    


# If you want to save the sphere in a nii file format
"""if __name__ == "__main__":
    experience_index = 25
    optimizer_name = "ConjugateGradient" 
    num_points = 100
    dim = 3

    # Step 1: The existing code to generate A and B
    mean_A = np.zeros(dim)
    cov_A = np.diag([1.0, 1.0, 1.0])  # Ellipsoid with different variances along each axis
    points_A = generate_elliptical_cloud(mean_A, cov_A, num_points)
    points_A[:, 0] = points_A[:, 0] * 2
    A = points_A.numpy()  # Shape is (num_points, 3)
    
    # Step 2: Generate a random initial rotation matrix once
    manifold = SpecialOrthogonalGroup(dim, k=1)
    X_initial = manifold.random_point()  # Random initial 3x3 rotation matrix
    print(f"Initial rotation matrix:\n{X_initial}")

    # Step 3: Apply the rotation to A to get B_rotated
    B = np.array([A[i] @ X_initial.T for i in range(A.shape[0])])
    
    # Step 4: Energy Landscape Visualization ---

    # Create a 3D meshgrid of points in the cube [-pi,pi] x [-pi,pi] x [-pi,pi]
    N = 51  # Adjust for resolution
    x = np.linspace(-np.pi, np.pi, N)
    y = np.linspace(-np.pi, np.pi, N)
    z = np.linspace(-np.pi, np.pi, N)
    X, Y, Z = np.meshgrid(x, y, z)

    # Mask the sphere centered at the origin with radius pi
    mask = X**2 + Y**2 + Z**2 <= np.pi**2

    # Initialize an array to store the loss values
    img = np.zeros_like(X, dtype=float)

    # Choose your distance type
    distance_type = "gaussian"  # or "sinkhorn" or "energy"

    # Number of sample points for interpolation
    num_samples = 1000 

    # Generate random rotation vectors within the sphere
    rot_vecs = np.random.rand(num_samples, 3) * 2 * np.pi - np.pi
    rot_vecs = rot_vecs[np.linalg.norm(rot_vecs, axis=1) <= np.pi]

    # Calculate costs for the sampled rotation vectors
    costs = []
    for rot_vec in rot_vecs:
        R = rotation_vector_to_rotation_matrix(rot_vec)
        cost, _ = create_cost_and_derivates(manifold, A, B, distance_type, [], [])  
        costs.append(cost(R).item())  # Evaluate the cost function

    # Interpolate the costs to the full grid using griddata
    img = griddata(rot_vecs, np.array(costs), (X, Y, Z), method='linear') 

    # Restrict the img to the sphere, use NaN outside of it
    img_sphere = np.where(mask, img, np.nan)

    # Save the data to a .nii file
    affine_matrix = np.array(
        [
            [2 * np.pi / (N - 1), 0, 0, -np.pi],
            [0, 2 * np.pi / (N - 1), 0, -np.pi],
            [0, 0, 2 * np.pi / (N - 1), -np.pi],
            [0, 0, 0, 1],
        ]
    )
    nii = nib.Nifti1Image(img_sphere, affine_matrix)
    nib.save(nii, "gaussian_sphere_1000_100.nii")
    #nib.save(nii, "sinkhorn_sphere_1000.nii")
    #nib.save(nii, "energy_sphere_1000.nii")




    # Create a volumetric rendering of img
    grid = pv.ImageData(
        dimensions=img.shape,
        origin=(-np.pi, -np.pi, -np.pi),
        spacing=(2 * np.pi / (N - 1), 2 * np.pi / (N - 1), 2 * np.pi / (N - 1)),)
        
    grid.point_data["img"] = img_sphere.flatten(order="F")  # Flatten the image to a 1D array

    # Create a PyVista plotter
    pl = pv.Plotter()

    # Extract contours
    vmax = np.nanmax(img)
    vmin = np.nanmin(img)
    contours = grid.contour(
        isosurfaces=np.linspace(vmax, vmin, 11), scalars="img", method="flying_edges"
    )
    # Compute surface normals for better shading
    contours.compute_normals(inplace=True)

    # Create a sphere to intersect the contours
    sphere_surface = pv.Sphere(
        center=(0, 0, 0), radius=np.pi, theta_resolution=100, phi_resolution=100
    )
    # contours = contours.boolean_intersection(sphere_b)

    pl.add_mesh(
        contours,
        opacity=.3,
        cmap="RdBu",
        ambient=0.2,
        diffuse=1,
        interpolation="gouraud",
        show_scalar_bar=True,
        scalar_bar_args=dict(vertical=True),
    )
    if True:
        pl.add_mesh(
            sphere_surface,
            color="black",
            opacity=.1,
            culling="front",
            interpolation="pbr",
            roughness=1,
        )

    pl.enable_ssao(radius=15, bias=0.5)
    pl.enable_anti_aliasing("ssaa")
    pl.camera.zoom(1.1)
    pl.show(jupyter_backend="static")"""