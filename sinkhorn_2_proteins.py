import torch
import numpy as np
import pyvista as pv
from Bio.PDB import PDBParser
from geomloss import SamplesLoss


# Step 1: Load Raw 3D Coordinates from PDB
def load_coordinates_from_pdb(file_path):
    """
    Load protein atomic coordinates from a PDB file using Biopython.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', file_path)

    # Extract atomic coordinates
    coordinates = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coordinates.append(atom.get_coord())

    return np.array(coordinates)

# Step 2: Generate Point Clouds
def generate_point_cloud(coordinates):
    """
    Convert atomic coordinates to a point cloud format.
    Center the point cloud around the origin.
    """
    # Center the point cloud by subtracting the mean
    centered_coordinates = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    return centered_coordinates

# Step 3: Visualize the Point Clouds
def visualize_point_cloud(point_cloud, title="Protein Point Cloud"):
    """
    Visualize the generated point cloud using PyVista.
    """
    plotter = pv.Plotter()
    plotter.add_points(point_cloud, color="blue", point_size=10, render_points_as_spheres=True)
    plotter.add_axes()
    plotter.show_grid()
    plotter.view_isometric()
    plotter.show(title=title)
    
def compute_sinkhorn_distance(point_cloud1, point_cloud2, blur=0.01, scaling=0.9, p=2):
    """
    Compute the Sinkhorn distance between two point clouds.
    :param point_cloud1: First point cloud as a NumPy array.
    :param point_cloud2: Second point cloud as a NumPy array.
    :param blur: Regularization parameter (epsilon).
    :param scaling: Controls the scaling of the cost function.
    :param p: The power of the distance (p=2 corresponds to the Euclidean distance).
    :return: Sinkhorn distance.
    """
    # Convert to torch tensors
    x = torch.tensor(point_cloud1, dtype=torch.float32)
    y = torch.tensor(point_cloud2, dtype=torch.float32)
    
    # Sinkhorn distance computation
    loss = SamplesLoss("sinkhorn", blur=blur, scaling=scaling, p=p)
    sinkhorn_distance = loss(x, y)
    
    return sinkhorn_distance.item()
    
    
if __name__ == "__main__":
    # Load and generate point clouds for two PDB files
    file_path1 = "./MDSPACE_tuto-Data/AK.pdb"  # Replace with your actual PDB file path
    file_path2 = "./MDSPACE_tuto-Data/6rak.pdb"  # Replace with a second PDB file path
    
    coordinates1 = load_coordinates_from_pdb(file_path1)
    coordinates2 = load_coordinates_from_pdb(file_path2)
    
    point_cloud1 = generate_point_cloud(coordinates1)
    point_cloud2 = generate_point_cloud(coordinates2)
    
    # Compute Sinkhorn distance between the two point clouds
    distance = compute_sinkhorn_distance(point_cloud1, point_cloud2)
    print(f"Sinkhorn Distance between the two point clouds: {distance}")
    
    # Visualize the point clouds
    visualize_point_cloud(point_cloud1, title="AK Point Cloud")
    visualize_point_cloud(point_cloud2, title="6rak Point Cloud")
