import importlib
import os
import subprocess
import sys
import numpy as np


def ensure_pip_is_current():
    """Upgrade pip if it is available and can be refreshed."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        pass


def ensure_module(module_name):
    """Install a missing Python package and import it."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        ensure_pip_is_current()
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", module_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return importlib.import_module(module_name)


trimesh = ensure_module("trimesh")
pv = ensure_module("pyvista")
ensure_module("matplotlib")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def trimesh_to_pyvista(mesh):
    """Convert a trimesh.Trimesh into a pyvista.PolyData for plotting."""
    faces = mesh.faces
    padded_faces = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
    )
    return pv.PolyData(mesh.vertices, padded_faces)


def show_mesh(mesh, title):
    """Pop up a pyvista window to visualize a mesh creation/transform step."""
    plotter = pv.Plotter()
    plotter.add_mesh(trimesh_to_pyvista(mesh), color="lightblue", show_edges=True)
    plotter.add_axes()
    plotter.view_xy()
    plotter.show(title=title)


def show_scene(mesh_specs, title):
    """Pop up a pyvista window showing multiple meshes, each with its own color/opacity.

    The fixture mesh (geometry/fixture.stl) is always included.
    """
    plotter = pv.Plotter()
    for mesh, color, opacity in mesh_specs:
        plotter.add_mesh(trimesh_to_pyvista(mesh), color=color, opacity=opacity, show_edges=True)
    if fixture_mesh is not None:
        plotter.add_mesh(
            trimesh_to_pyvista(fixture_mesh),
            color="tan",
            opacity=0.5,
            show_edges=True,
        )
    plotter.add_axes()
    plotter.view_xy()
    plotter.show(title=title)


def show_displacement(mesh, previous_vertices, title):
    """Pop up a wireframe colored by each vertex's displacement since the previous step."""
    displacement = np.linalg.norm(mesh.vertices - previous_vertices, axis=1)
    pv_mesh = trimesh_to_pyvista(mesh)
    pv_mesh["displacement"] = displacement
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="displacement", cmap="viridis", style="wireframe", line_width=2)
    plotter.add_axes()
    plotter.view_xy()
    plotter.show(title=title)


def plot_radius_distribution(meshes_by_label, title):
    """Histogram of each vertex's radial distance from the cylinder axis (0, 0), per method."""
    plt.figure()
    for label, mesh in meshes_by_label.items():
        xy = mesh.vertices[:, :2]
        radii = np.hypot(xy[:, 0], xy[:, 1])
        plt.hist(radii, bins=50, alpha=0.5, label=label)
    plt.xlabel("Radial distance from cylinder axis (mm)")
    plt.ylabel("Vertex count")
    plt.title(title)
    plt.legend()
    plt.show()


def align_thinnest_axis_to_z(mesh):
    """
    Rotate mesh so its smallest principal axis becomes the Z axis.
    This generally minimizes the height/thickness of the part.
    """

    vertices = mesh.vertices

    # Center vertices
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid

    # Covariance matrix
    cov = np.cov(centered.T)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort from smallest to largest variance
    order = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, order]

    # Smallest principal axis
    thinnest_axis = eigenvectors[:, 0]

    # Desired Z axis
    z_axis = np.array([0.0, 0.0, 1.0])

    # Compute rotation axis and angle
    rot_axis = np.cross(thinnest_axis, z_axis)
    axis_norm = np.linalg.norm(rot_axis)

    if axis_norm < 1e-8:
        return mesh.copy()

    rot_axis /= axis_norm

    angle = np.arccos(
        np.clip(np.dot(thinnest_axis, z_axis), -1.0, 1.0)
    )

    rot = Rotation.from_rotvec(rot_axis * angle)

    transformed = mesh.copy()

    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()

    transformed.apply_translation(-centroid)
    transformed.apply_transform(T)
    transformed.apply_translation(centroid)

    return transformed


def optimize_xy_placement(mesh):
    """
    Find the Z-axis rotation angle and XY translation that best center the
    mesh's vertices on the cylinder axis (0, 0), minimizing the largest
    radial distance from that axis (the smallest-enclosing-circle center
    under rotation), which fits an irregular part inside a cylinder better
    than a plain bounding-box center.
    """

    xy = mesh.vertices[:, :2]

    def max_radius(params):
        theta, dx, dy = params
        c, s = np.cos(theta), np.sin(theta)
        rotated_x = xy[:, 0] * c - xy[:, 1] * s
        rotated_y = xy[:, 0] * s + xy[:, 1] * c
        return np.max(np.hypot(rotated_x + dx, rotated_y + dy))

    bbox_center = (xy.min(axis=0) + xy.max(axis=0)) / 2.0
    initial_guess = [0.0, -bbox_center[0], -bbox_center[1]]
    result = minimize(max_radius, initial_guess, method="Nelder-Mead")

    return result.x  # theta (radians), offset_x, offset_y


def centroid_xy_offset(mesh):
    """Return the XY translation that moves the mesh's vertex centroid to (0, 0)."""
    xy_centroid = mesh.vertices[:, :2].mean(axis=0)
    return -xy_centroid[0], -xy_centroid[1]


if __name__ == "__main__":

    mesh = trimesh.load(os.path.join("input", "part.stl"))
    fixture_mesh = trimesh.load(os.path.join("geometry", "fixture.stl"))

    # Step 1: Move the part centroid to the global origin.
    original_mesh = mesh.copy()
    centroid = mesh.vertices.mean(axis=0)
    mesh.apply_translation(-centroid)
    show_scene(
        [
            (original_mesh, "lightgray", 0.4),
            (mesh, "lightblue", 1.0),
        ],
        "Step 1: Original (gray) and Centered at Origin (blue)",
    )

    # Step 2: Rotate the part so its thinnest principal axis is Z.
    centered_mesh = mesh.copy()
    rotated_mesh = align_thinnest_axis_to_z(mesh)

    show_scene(
        [
            (centered_mesh, "lightgray", 0.4),
            (rotated_mesh, "lightblue", 1.0),
        ],
        "Step 2: Before (gray) and After (blue) Z Height Optimization",
    )

    min_corner, max_corner = rotated_mesh.bounds
    bbox_size = max_corner - min_corner

    optimized_z_height = bbox_size[2]

    print(f"Bounding box dimensions: {bbox_size}")
    print(f"Optimized Z height: {optimized_z_height:.3f} mm")

    # Step 3: Create the smallest cylinder centered at the global origin.
    #
    # The part centroid is at (0, 0, 0), and the cylinder is also centered
    # at (0, 0, 0). Therefore, the cylinder height must cover the largest
    # absolute Z distance from the origin in both directions.

    radial_distances = np.hypot(
        rotated_mesh.vertices[:, 0],
        rotated_mesh.vertices[:, 1],
    )

    cylinder_radius = np.max(radial_distances)
    cylinder_diameter = int(np.ceil(2.0 * cylinder_radius))

    required_centered_height = 2.0 * np.max(
        np.abs(rotated_mesh.vertices[:, 2])
    )

    
    cylinder_height = int(np.ceil(required_centered_height))

    cylinder = trimesh.creation.cylinder(
        radius=cylinder_radius,
        height=cylinder_height,
        sections=128,
    )

    # trimesh creates the cylinder centered at the global origin by default.
    print(f"Enclosing cylinder diameter: {cylinder_diameter:.3f} mm")
    print(f"Enclosing cylinder height: {cylinder_height} mm")

    show_scene(
        [
            (cylinder, "lightgreen", 0.3),
            (rotated_mesh, "lightblue", 1.0),
        ],
        "Part Enclosed by Minimum-Diameter Cylinder",
    )

    # Export the part after Z optimization only.
    rotated_mesh.export(os.path.join("output", "part_min_z.stl"))

    # Export the enclosing cylinder.
    cylinder.export(os.path.join("output", "enclosing_cylinder.stl"))

