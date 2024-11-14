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
import Bio.PDB

from scipy.interpolate import griddata, RBFInterpolator
from mpl_toolkits.mplot3d import Axes3D
from geomloss import SamplesLoss
from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.optimizers import ConjugateGradient
from pymanopt.optimizers import SteepestDescent
from pymanopt.optimizers import TrustRegions
from torch.profiler import profile, record_function, ProfilerActivity
from scipy.spatial.transform import Rotation as R


os.environ["QT_QPA_PLATFORM"] = "offscreen"
pv.global_theme.allow_empty_mesh = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


def geom_energy_distance(x, y):
    loss = SamplesLoss("energy")
    return loss(x, y)

def geom_sinkhorn_distance(x, y, epsilon=0.01, p=2):
    loss = SamplesLoss("sinkhorn", blur=epsilon, p=p)
    return loss(x, y)

def geom_gaussian_distance(x, y, epsilon=0.01, p=2):
    loss = SamplesLoss("gaussian", blur=epsilon)
    return loss(x, y)



def create_cost_and_derivates(
    manifold,
    A,
    B,
    distance_type,
    intermediate_rotations,
    intermediate_losses,
    intermediate_rotation_vectors,
    epsilon=0.01
):
    A_torch = torch.from_numpy(A).float()
    B_torch = torch.from_numpy(B).float()

    @pymanopt.function.pytorch(manifold)
    def cost(X):
        with record_function("cost_function"):
            X_torch = X.float()
            A_rotated = A_torch @ X_torch.T
            if distance_type == "energy":
                loss = geom_energy_distance(A_rotated, B_torch)
            elif distance_type == "sinkhorn":
                loss = geom_sinkhorn_distance(A_rotated, B_torch, epsilon=epsilon)
            elif distance_type == "gaussian":
                loss = geom_gaussian_distance(A_rotated, B_torch, epsilon=epsilon)
            else:
                raise ValueError(f"Unknown distance type: {distance_type}")

            # Record intermediate results
            intermediate_rotations.append(X_torch.detach().cpu().numpy())
            rotation_vector = rotation_matrix_to_rotation_vector(X_torch.detach().cpu().numpy())
            intermediate_rotation_vectors.append(rotation_vector)
            intermediate_losses.append(loss.item())

            return loss

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


def save_loss_plot(intermediate_losses, filename):
    """Saves a plot of the loss over iterations for a given optimizer."""
    # Create directory structure if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Plot the losses and save the figure
    plt.figure()
    plt.plot(intermediate_losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss over iterations")
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


def run_optimization_with_profiling(
    distance_type,
    experience_index,
    optimizer_name,
    A,
    B,
    quiet=True,
    initial_point=None,
    epsilon=0.01
):
    num_points = A.shape[0]
    dim = A.shape[1]
    results_dir = f"results/exp{experience_index}"
    os.makedirs(results_dir, exist_ok=True)

    intermediate_rotations = []
    intermediate_rotation_vectors = []
    intermediate_losses = []

    manifold = SpecialOrthogonalGroup(dim)
    cost, euclidean_gradient = create_cost_and_derivates(
        manifold,
        A,
        B,
        distance_type,
        intermediate_rotations,
        intermediate_losses,
        intermediate_rotation_vectors,
        epsilon=epsilon
    )
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)

    if optimizer_name == "ConjugateGradient":
        optimizer = ConjugateGradient(verbosity=2 * int(not quiet))
    elif optimizer_name == "SteepestDescent":
        optimizer = SteepestDescent(verbosity=2 * int(not quiet))
    elif optimizer_name == "TrustRegions":
        optimizer = TrustRegions(verbosity=2 * int(not quiet))
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")

    if initial_point is not None:
        x_initial = initial_point
    else:
        x_initial = manifold.random_point()

    profiler_trace_filename = f"{results_dir}/profiler_trace_{optimizer_name}_{distance_type}_epsilon{epsilon}_exp{experience_index}.json"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("optimization"):
            result = optimizer.run(problem, initial_point=x_initial)
            X_opt = result.point

    prof.export_chrome_trace(profiler_trace_filename)
    print(f"Profiler trace saved to {profiler_trace_filename}")

    # Return intermediate_rotation_vectors
    return X_opt, intermediate_rotations, intermediate_rotation_vectors, intermediate_losses



def generate_energy_landscape(A, B, distance_type, rotation_vectors):
    """Generates the energy landscape data for the given point clouds and distance type."""
    A_ = torch.tensor(A, dtype=torch.float32, device=device)
    B_ = torch.tensor(B, dtype=torch.float32, device=device)
    
    N = 150
    directions = rotation_vectors / np.linalg.norm(rotation_vectors, axis=1, keepdims=True)
    theta = np.linalg.norm(rotation_vectors, axis=1)
    
    # Create a 3D meshgrid of points in the cube [-pi, pi] x [-pi, pi] x [-pi, pi]
    x = np.linspace(-np.pi, np.pi, N)
    y = np.linspace(-np.pi, np.pi, N)
    z = np.linspace(-np.pi, np.pi, N)
    X, Y, Z = np.meshgrid(x, y, z)
    mask = X**2 + Y**2 + Z**2 <= np.pi**2  # Mask for spherical region
    
    img = np.zeros_like(X, dtype=float)  # Initialize array for loss values
    rot_vecs = directions * theta[:, None]  # Combine directions and angles
    
    # Calculate costs for the sampled rotation vectors
    costs = []
    manifold = SpecialOrthogonalGroup(3, k=1)
    
    for rot_vec in rot_vecs:
        R = torch.tensor(rotation_vector_to_rotation_matrix(rot_vec), dtype=torch.float32, device=device)
        A_rotated = A_ @ R.T
        if distance_type == "energy":
            cost = geom_energy_distance(A_rotated, B_)
        elif distance_type == "sinkhorn":
            cost = geom_sinkhorn_distance(A_rotated, B_)
        elif distance_type == "gaussian":
            cost = geom_gaussian_distance(A_rotated, B_)
        else:
            raise ValueError(f"Unknown distance type: {distance_type}")
        
        costs.append(cost.item())  # Append CPU float for interpolation
    
    # Interpolate the costs to the full grid
    #img = griddata(rot_vecs, np.array(costs), (X, Y, Z), method='linear')
    XYZ = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    img = RBFInterpolator(rot_vecs, np.array(costs), kernel="linear")(XYZ)
    img = img.reshape(N, N, N)
    img_sphere = np.where(mask, img, np.nan)  # Restrict to the spherical region

    return img_sphere  # Return as a NumPy array for PyVista


def rotation_vector_to_point(rot_vec):
    """
    Maps a rotation vector to a point on the unit sphere.

    Parameters:
    - rot_vec: numpy.ndarray of shape (3,)

    Returns:
    - point: numpy.ndarray of shape (3,)
    """
    # Normalize to ensure it's on the unit sphere
    point = rot_vec / np.linalg.norm(rot_vec)
    return point


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



def visualize_energy_landscape(img_sphere, intermediate_rotation_vectors, N, save_path):
    grid = pv.ImageData(
        dimensions=img_sphere.shape,
        origin=(-np.pi, -np.pi, -np.pi),  # Center the origin at the middle of the sphere
        spacing=(2 * np.pi / (N - 1), 2 * np.pi / (N - 1), 2 * np.pi / (N - 1)),
    )
    grid.point_data["img"] = img_sphere.flatten(order="F")
    pl = pv.Plotter(off_screen=True)  # Use off_screen=True to allow screenshots without opening a window
    
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

    # Save screenshot
    pl.screenshot(save_path)  # This will work with off_screen=True
    pl.show()  # Optional, only if you want to render onscreen if off_screen=False



def get_atom_coordinates(pdb_file):
    parser = Bio.PDB.PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    atoms = []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element != 'H':  # Exclude hydrogen atoms
                        atoms.append(atom.coord)
    
    return np.array(atoms)



def visualize_energy_landscape2(
    img_sphere, 
    intermediate_rotation_vectors, 
    N, 
    save_path, 
    create_video=True, 
    video_filename=None, 
    num_frames=180
):
    """
    Visualizes the energy landscape with optimization steps and optionally creates a rotating video.
    
    Parameters:
    - img_sphere: numpy.ndarray, shape (N, N, N), the energy landscape data.
    - intermediate_rotation_vectors: list or numpy.ndarray, rotation vectors from the optimization process.
    - N: int, resolution of the energy landscape grid.
    - save_path: str, path to save the snapshot image.
    - create_video: bool, whether to create a rotating video.
    - video_filename: str, path to save the video file (required if create_video=True).
    - num_frames: int, number of frames in the video (default: 180).
    """
    
    if create_video and not video_filename:
        raise ValueError("video_filename must be provided if create_video is True.")
    
    # Initialize Plotter with off_screen rendering
    pl = pv.Plotter(off_screen=True)
    
    # Create grid for the energy landscape
    grid = pv.ImageData(
        dimensions=img_sphere.shape,
        origin=(-np.pi, -np.pi, -np.pi),
        spacing=(2 * np.pi / (N - 1), 2 * np.pi / (N - 1), 2 * np.pi / (N - 1)),
    )
    grid.point_data["img"] = img_sphere.flatten(order="F")
    
    # Compute contours
    vmax = np.nanmax(img_sphere)
    vmin = np.nanmin(img_sphere)
    contours = grid.contour(
        isosurfaces=np.linspace(vmin, vmax, 11),
        scalars="img",
        method="flying_edges"
    )
    contours.compute_normals(inplace=True)
    
    # Create sphere surface
    sphere_surface = pv.Sphere(
        center=(0, 0, 0),
        radius=np.pi,
        theta_resolution=100,
        phi_resolution=100
    )
    
    # Add meshes to the plotter
    pl.add_mesh(
        contours,
        opacity=0.1,
        cmap="RdBu",
        ambient=0.2,
        diffuse=1,
        interpolation="gouraud",
        show_scalar_bar=False,
    )
    pl.add_mesh(
        sphere_surface,
        color="black",
        opacity=0.1,
        culling="front",
        interpolation="pbr",
        roughness=1,
    )
    
    # Plot optimization steps
    num_rotations = len(intermediate_rotation_vectors)
    colormap = plt.get_cmap("viridis")
    points = np.array(intermediate_rotation_vectors)
    scalars = np.arange(1, num_rotations + 1)
    
    for i, point in enumerate(points):
        color = colormap((i) / (num_rotations - 1))[:3]
        sphere = pv.Sphere(radius=0.1, center=point)
        pl.add_mesh(sphere, color=color, opacity=1.0)
        
        if i < num_rotations - 1:
            next_point = points[i + 1]
            line = pv.Line(point, next_point)
            pl.add_mesh(line, color="black")
    
    # Add scalar bar
    scalar_bar_args = {
        "title": "Optimization Step",
        "vertical": True,
        "position_x": 0.8,
        "position_y": 0.1,
        "height": 0.8,
        "width": 0.03,
        "fmt": "%.0f",
    }
    pl.add_scalar_bar(**scalar_bar_args)
    
    # Enhance visualization
    pl.enable_ssao(radius=15, bias=0.5)
    pl.enable_anti_aliasing("ssaa")
    pl.camera.zoom(1.1)
    pl.hide_axes()
    
    if create_video:
        # Open the movie file
        pl.open_movie(video_filename, framerate=30)
    
    # Render the scene
    pl.render()
    
    if create_video:
        # Rotate the camera and write frames
        for frame in range(num_frames):
            pl.camera.Azimuth(360.0 / num_frames)  # Rotate camera
            pl.render()
            pl.write_frame()
    
        # Close the movie
        pl.close()
        print(f"Video saved as {video_filename}")
    else:
        # Save the snapshot image
        pl.screenshot(save_path)
        pl.close()
        print(f"Snapshot saved as {save_path}")


def create_rotating_video(sphere_mesh, filename="energy_landscape.mp4", n_frames=180):
    plotter = pv.Plotter(off_screen=True)
    plotter.open_movie(filename)
    plotter.add_mesh(
        sphere_mesh,
        scalars="energies",
        cmap="viridis",
        clim=[np.nanmin(sphere_mesh["energies"]), np.nanmax(sphere_mesh["energies"])],
    )
    plotter.hide_axes()  # Hide the axes

    for _ in range(n_frames):
        plotter.camera.Azimuth(2)  # Rotate camera by 2 degrees
        plotter.render()
        plotter.write_frame()

    plotter.close()
    print(f"Video saved as {filename}")

"""if __name__ == "__main__":
    experience_index = 35
    n_samples = 2000
    rotation_vectors = sample_vectors(n_samples, random=True)
    dim = 3
    mean_A = np.zeros(dim)
    cov_A = np.diag([1.0, 1.0, 1.0])
    points_A = generate_elliptical_cloud(mean_A, cov_A, 50)
    points_A[:, 0] = points_A[:, 0] * 2

    #A = points_A.numpy()
    pdb_file = "./MDSPACE_tuto-Data/AK.pdb" 
    atoms = get_atom_coordinates(pdb_file)
    A = torch.tensor(atoms, dtype=torch.float32, device=device)
    B = A.clone()

    manifold = SpecialOrthogonalGroup(dim, k=1)
    X_initial = torch.tensor(manifold.random_point(), dtype=torch.float32, device=device)
    print(f"Initial rotation matrix:\n{X_initial.cpu()}")

    distance_type = "gaussian"
    img_sphere = generate_energy_landscape(A.cpu().numpy(), B.cpu().numpy(), distance_type, rotation_vectors)

    # Define the optimizers to use in the experiment
    optimizers = ["ConjugateGradient", "SteepestDescent", "TrustRegions"]

    # Run the optimization for each optimizer and save loss plots and snapshots
    for optimizer_name in optimizers:
        X, intermediate_rotations, intermediate_rotation_vectors, intermediate_losses = run_optimization(
            distance_type, experience_index, optimizer_name, A.cpu().numpy(), B.cpu().numpy(), quiet=False, initial_point=X_initial.cpu().numpy()
        )
        print(f"{optimizer_name} finished with final rotation:\n{X}")

        # Directory for saving results of this optimizer
        directory = f"results/{optimizer_name}/{distance_type}/exp{experience_index}"
        os.makedirs(directory, exist_ok=True)
        
        # Save the loss plot for the current optimizer
        save_loss_plot(intermediate_losses, optimizer_name, distance_type, experience_index)

        # Path to save the snapshot of the energy landscape
        snapshot_path = f"{directory}/energy_landscape.png"
        visualize_energy_landscape(img_sphere, intermediate_rotation_vectors, N=150, save_path=snapshot_path)
        """


def create_energy_landscape_video(img_sphere, intermediate_rotation_vectors, N, video_filename, num_frames=180):
    """
    Creates a rotating video of the energy landscape with optimization steps.

    Parameters:
    - img_sphere: numpy.ndarray, shape (N, N, N), energy landscape data.
    - intermediate_rotation_vectors: list or numpy.ndarray, list of rotation vectors from optimization.
    - N: int, resolution of the grid.
    - video_filename: str, path to save the output video.
    - num_frames: int, number of frames in the video.
    """
    # Create a temporary directory to store frames
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    # Initialize Plotter
    pl = pv.Plotter(off_screen=True)
    
    # Create the grid for the energy landscape
    grid = pv.ImageData(
        dimensions=img_sphere.shape,
        origin=(-np.pi, -np.pi, -np.pi),  # Center the origin at the middle of the sphere
        spacing=(2 * np.pi / (N - 1), 2 * np.pi / (N - 1), 2 * np.pi / (N - 1)),
    )
    grid.point_data["img"] = img_sphere.flatten(order="F")
    
    vmax = np.nanmax(img_sphere)
    vmin = np.nanmin(img_sphere)
    contours = grid.contour(
        isosurfaces=np.linspace(vmin, vmax, 11), scalars="img", method="flying_edges"
    )
    contours.compute_normals(inplace=True)
    
    # Create sphere surface
    sphere_surface = pv.Sphere(
        center=(0, 0, 0), radius=np.pi, theta_resolution=100, phi_resolution=100
    )
    
    # Add contour mesh
    pl.add_mesh(
        contours,
        opacity=0.1,  # Adjusted opacity for better contrast
        cmap="RdBu",
        ambient=0.2,
        diffuse=1,
        interpolation="gouraud",
        show_scalar_bar=False,  # We'll add a scalar bar at the end
    )
    
    # Add sphere surface mesh
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

    # Add optimization steps as spheres and lines
    points = np.array(intermediate_rotation_vectors)
    scalars = np.arange(1, num_rotations + 1)  # Starting from 1

    for i, point in enumerate(points):
        scalar_value = scalars[i]
        color = colormap((scalar_value - 1) / (num_rotations - 1))[:3]  # Get RGB values from colormap
        sphere = pv.Sphere(radius=0.1, center=point)
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
    pl.hide_axes()

    # Capture frames with rotating camera
    for frame in range(num_frames):
        # Rotate camera by a fixed angle
        pl.camera.Azimuth(360.0 / num_frames)
        
        # Render and capture frame
        frame_filename = os.path.join(temp_dir, f"frame_{frame:04d}.png")
        pl.screenshot(frame_filename)
        
    # Close the Plotter
    pl.close()

    # Compile frames into a video using OpenCV
    frame_files = [os.path.join(temp_dir, f) for f in sorted(os.listdir(temp_dir)) if f.endswith(".png")]

    if not frame_files:
        print("No frames captured. Video not created.")
        return

    # Read the first frame to get the frame dimensions
    frame = cv2.imread(frame_files[0])
    if frame is None:
        print(f"Error reading frame {frame_files[0]}. Video not created.")
        return
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame is not None:
            video.write(frame)
        else:
            print(f"Warning: Unable to read frame {frame_file}. Skipping.")

    video.release()

    # Clean up temporary frames
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(temp_dir)

    print(f"Video saved as {video_filename}")


if __name__ == "__main__":
    # Set experiment parameters
    experience_index = 44  # Update as needed
    distance_type = "sinkhorn"  # Options: 'energy', 'sinkhorn', 'gaussian'
    epsilon = 0.1  # Epsilon value for Sinkhorn or Gaussian distances
    optimizer_names = ["ConjugateGradient"]

    # Load or generate point clouds A and B
    pdb_file = "./MDSPACE_tuto-Data/AK.pdb"
    atoms = get_atom_coordinates(pdb_file)
    A = torch.tensor(atoms, dtype=torch.float32)
    B = A.clone()  # Use B as a rotated version of A if needed

    # Generate rotation vectors for the energy landscape
    n_samples = 2000
    rotation_vectors = sample_vectors(n_samples, random=True)

    # Generate energy landscape and create a rotating video
    img_sphere = generate_energy_landscape(
        A.numpy(), B.numpy(), distance_type, rotation_vectors
    )
    results_dir = f"results/exp{experience_index}"
    os.makedirs(results_dir, exist_ok=True)
    video_filename = f"{results_dir}/energy_landscape_{distance_type}_exp{experience_index}.mp4"

    # Run optimization with profiling for each optimizer
    for optimizer_name in optimizer_names:
        X_opt, rotations, rotation_vecs, losses = run_optimization_with_profiling(
            distance_type=distance_type,
            experience_index=experience_index,
            optimizer_name=optimizer_name,
            A=A.numpy(),
            B=B.numpy(),
            quiet=True,
            epsilon=epsilon
        )

        # Save results and plots
        loss_plot_filename = f"{results_dir}/loss_{optimizer_name}_{distance_type}_epsilon{epsilon}.png"
        save_loss_plot(losses, filename=loss_plot_filename)
        snapshot_filename = f"{results_dir}/snapshot_{optimizer_name}_{distance_type}_epsilon{epsilon}.png"
        visualize_energy_landscape(img_sphere, rotation_vecs, N=150, save_path=snapshot_filename)
        visualize_energy_landscape2(
            img_sphere=img_sphere,
            intermediate_rotation_vectors=rotation_vecs,
            N=150,
            save_path=snapshot_filename,
            video_filename=video_filename,
            num_frames=180  # Adjust the number of frames as needed
        )
    print("Optimization complete.")