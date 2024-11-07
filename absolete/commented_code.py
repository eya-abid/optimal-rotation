# If you want to save the sphere in a nii file format
if __name__ == "__main__":
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
    pl.show(jupyter_backend="static")