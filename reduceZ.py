import importlib
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
    plotter.show(title=title)


def show_scene(mesh_specs, title):
    """Pop up a pyvista window showing multiple meshes, each with its own color/opacity."""
    plotter = pv.Plotter()
    for mesh, color, opacity in mesh_specs:
        plotter.add_mesh(trimesh_to_pyvista(mesh), color=color, opacity=opacity, show_edges=True)
    plotter.add_axes()
    plotter.show(title=title)


def show_displacement(mesh, previous_vertices, title):
    """Pop up a wireframe colored by each vertex's displacement since the previous step."""
    displacement = np.linalg.norm(mesh.vertices - previous_vertices, axis=1)
    pv_mesh = trimesh_to_pyvista(mesh)
    pv_mesh["displacement"] = displacement
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="displacement", cmap="viridis", style="wireframe", line_width=2)
    plotter.add_axes()
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

    mesh = trimesh.load("part.stl")

    # Step 1: move the mesh centroid to the world origin
    centroid = mesh.vertices.mean(axis=0)
    mesh.apply_translation(-centroid)
    show_mesh(mesh, "Step 1: Centered Input Mesh")

    # Step 2: rotate so the thinnest axis points along Z
    centered_vertices = mesh.vertices.copy()
    rotated_mesh = align_thinnest_axis_to_z(mesh)
    show_displacement(rotated_mesh, centered_vertices, "Step 2: Vertex Displacement (Z Minimization)")

    min_corner, max_corner = rotated_mesh.bounds
    bbox_size = max_corner - min_corner

    print(f"Bounding box dimensions: {bbox_size}")
    print(f"Z thickness: {bbox_size[2]}")

    # Step 3: cylinder centered at the world origin (X, Y) and the part's Z center
    part_z_center = (min_corner[2] + max_corner[2]) / 2.0
    part_z_height = bbox_size[2]

    cylinder1 = trimesh.creation.cylinder(radius=51.0 / 2.0, height=part_z_height, sections=64)
    cylinder1.apply_translation([0.0, 0.0, part_z_center])
    #show_mesh(cylinder1, "Step 3: Cylinder 1 (51 mm diameter)")

    # Export 1: part after Z minimization only, before any XY adjustment
    rotated_mesh.export("part_min_z.stl")

    # Step 4a: XY optimization method 1 - minimize the largest radial distance
    # from the cylinder axis, allowing a Z-axis rotation as well
    method1_mesh = rotated_mesh.copy()
    theta, offset_x, offset_y = optimize_xy_placement(method1_mesh)
    z_rotation = np.eye(4)
    z_rotation[:3, :3] = Rotation.from_euler("z", theta).as_matrix()
    method1_mesh.apply_transform(z_rotation)
    method1_mesh.apply_translation([offset_x, offset_y, 0.0])
    show_displacement(
        method1_mesh, rotated_mesh.vertices, "Step 4a: Vertex Displacement (Min-Radius XY Optimization)"
    )
    show_scene(
        [
            (cylinder1, "lightgreen", 0.3),
            (method1_mesh, "lightblue", 1.0),
        ],
        "Step 4a: Part Re-centered in Cylinder 1 (Min-Radius Method)",
    )

    # Export 2: part after Z minimization plus the min-radius XY optimization
    method1_mesh.export("part_xy_optimized.stl")

    # Step 4b: XY optimization method 2 - simply move the XY centroid to the cylinder axis
    method2_mesh = rotated_mesh.copy()
    centroid_offset_x, centroid_offset_y = centroid_xy_offset(method2_mesh)
    method2_mesh.apply_translation([centroid_offset_x, centroid_offset_y, 0.0])
    show_displacement(
        method2_mesh, rotated_mesh.vertices, "Step 4b: Vertex Displacement (Centroid XY Method)"
    )
    show_scene(
        [
            (cylinder1, "lightgreen", 0.3),
            (method2_mesh, "lightblue", 1.0),
        ],
        "Step 4b: Part Re-centered in Cylinder 1 (Centroid Method)",
    )

    # Export 3: part after Z minimization plus the simple centroid XY method
    method2_mesh.export("part_xy_centroid.stl")

    plot_radius_distribution(
        {
            "Min-Radius Method": method1_mesh,
            "Centroid Method": method2_mesh,
        },
        "Vertex Radius Distribution Comparison",
    )