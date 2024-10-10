import numpy as np
import pyvista as pv
from Bio.PDB import PDBParser

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

if __name__ == "__main__":
    # Example usage
    
    # Load raw 3D coordinates from a PDB file (replace with your actual file path)
    file_path = "./MDSPACE_tuto-Data/AK.pdb"  # Replace with the actual path to your PDB file
    coordinates = load_coordinates_from_pdb(file_path)
    
    # Generate point cloud from coordinates
    point_cloud = generate_point_cloud(coordinates)
    
    # Visualize the point cloud
    visualize_point_cloud(point_cloud, title="Protein Atomic Point Cloud")

