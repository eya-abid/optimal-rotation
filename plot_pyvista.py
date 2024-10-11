import nibabel as nib
import numpy as np
import pyvista as pv

nii_file = "sphere_eya.nii"  # Replace with the actual path to your .nii file
img_sphere = nib.load(nii_file).get_fdata()
N = img_sphere.shape[0]  # Assuming the image is a cube

# Create a volumetric rendering of img
grid = pv.ImageData(
    dimensions=img.shape,
    origin=(-np.pi, -np.pi, -np.pi),
    spacing=(2 * np.pi / (N - 1), 2 * np.pi / (N - 1), 2 * np.pi / (N - 1)),
)
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
pl.show(jupyter_backend="static")