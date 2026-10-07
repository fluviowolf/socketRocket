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
from scipy.spatial import ConvexHull
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


def _circle_from_two_points(p1, p2):
    center = (p1 + p2) / 2.0
    radius = np.linalg.norm(p1 - p2) / 2.0
    return center, radius


def _circle_from_three_points(p1, p2, p3):
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        # Colinear points have no circumscribed circle; fall back to the
        # two-point circle spanning the farthest-apart pair.
        pts = [p1, p2, p3]
        farthest = max(
            ((i, j) for i in range(3) for j in range(i + 1, 3)),
            key=lambda ij: np.linalg.norm(pts[ij[0]] - pts[ij[1]]),
        )
        return _circle_from_two_points(pts[farthest[0]], pts[farthest[1]])
    ux = (
        (ax**2 + ay**2) * (by - cy)
        + (bx**2 + by**2) * (cy - ay)
        + (cx**2 + cy**2) * (ay - by)
    ) / d
    uy = (
        (ax**2 + ay**2) * (cx - bx)
        + (bx**2 + by**2) * (ax - cx)
        + (cx**2 + cy**2) * (bx - ax)
    ) / d
    center = np.array([ux, uy])
    radius = np.linalg.norm(center - p1)
    return center, radius


def _point_in_circle(point, center, radius, eps=1e-7):
    return np.linalg.norm(point - center) <= radius + eps


def minimum_enclosing_circle(points):
    """Welzl's algorithm: the true smallest circle enclosing all 2D points.

    Unlike optimize_xy_placement (which searches rotation/translation to
    minimize the max radius from a circle forced to be centered at the
    origin), this finds the globally smallest bounding circle regardless of
    where its center ends up, using only 2-3 of the outermost points to
    define it.
    """
    pts = np.asarray(points, dtype=float).copy()
    rng = np.random.default_rng(0)
    rng.shuffle(pts)

    center = pts[0]
    radius = 0.0
    for i in range(1, len(pts)):
        if _point_in_circle(pts[i], center, radius):
            continue
        center, radius = pts[i], 0.0
        for j in range(i):
            if _point_in_circle(pts[j], center, radius):
                continue
            center, radius = _circle_from_two_points(pts[i], pts[j])
            for k in range(j):
                if _point_in_circle(pts[k], center, radius):
                    continue
                center, radius = _circle_from_three_points(pts[i], pts[j], pts[k])

    return center, radius


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
    print(f"Old (origin-centered) cylinder diameter: {cylinder_diameter:.3f} mm")
    print(f"Old (origin-centered) cylinder height: {cylinder_height} mm")

    # Step 3b: True minimum-diameter cylinder from the smallest enclosing
    # circle of the part's outermost XY points (post rotation, Steps 1-2).
    # This circle need not be centered on the origin or the part centroid,
    # since only the height (already optimized in Step 2) is shared with it.
    xy_points = rotated_mesh.vertices[:, :2]
    hull_points = xy_points[ConvexHull(xy_points).vertices]
    mec_center, mec_radius = minimum_enclosing_circle(hull_points)

    # Snug the new puck's height to the mesh's actual Z extent instead of
    # the old symmetric-about-origin height, since it no longer shares the
    # part's center. Use the exact (unrounded) extent so the cylinder's
    # top/bottom faces land precisely on the mesh's highest/lowest points
    # rather than being padded evenly on both sides by integer rounding.
    z_min, z_max = rotated_mesh.vertices[:, 2].min(), rotated_mesh.vertices[:, 2].max()
    mec_height = z_max - z_min
    mec_z_center = (z_max + z_min) / 2.0

    # Add a 1 mm clearance offset in every direction (+1 mm diameter,
    # +1 mm height), then round the final puck dimensions up to the next
    # whole millimeter.
    puck_radius = mec_radius + 0.5
    puck_height = mec_height + 1.0
    puck_diameter = int(np.ceil(2.0 * puck_radius))
    puck_height = int(np.ceil(puck_height))

    optimized_cylinder = trimesh.creation.cylinder(
        radius=puck_diameter / 2.0,
        height=puck_height,
        sections=128,
    )
    optimized_cylinder.apply_translation([mec_center[0], mec_center[1], mec_z_center])

    print(f"New (min-enclosing-circle) puck diameter: {puck_diameter} mm")
    print(f"New (min-enclosing-circle) puck height: {puck_height} mm")
    print(
        f"New cylinder center offset from origin: "
        f"({mec_center[0]:.3f}, {mec_center[1]:.3f}, {mec_z_center:.3f}) mm"
    )

    show_scene(
        [
            (cylinder, "lightgreen", 0.3),
            (optimized_cylinder, "orange", 0.3),
            (rotated_mesh, "lightblue", 1.0),
        ],
        "Old Origin-Centered Cylinder (green) vs New Min-Enclosing Cylinder (orange)",
    )

    # Step 4: Recenter the new cylinder (and the part it encloses) so the
    # cylinder's own center lands on the scene origin.
    recenter_offset = [-mec_center[0], -mec_center[1], -mec_z_center]

    pre_recenter_cylinder = optimized_cylinder.copy()
    pre_recenter_mesh = rotated_mesh.copy()

    optimized_cylinder.apply_translation(recenter_offset)
    rotated_mesh.apply_translation(recenter_offset)

    show_scene(
        [
            (pre_recenter_cylinder, "lightgray", 0.3),
            (pre_recenter_mesh, "lightgray", 0.4),
            (optimized_cylinder, "orange", 0.3),
            (rotated_mesh, "lightblue", 1.0),
        ],
        "Step 4: Before (gray) and After (color) Recentering New Cylinder to Origin",
    )

    # Export the part after Z optimization only.
    rotated_mesh.export(os.path.join("output", "part_min_z.stl"))

    # Export both enclosing cylinders.
    cylinder.export(os.path.join("output", "enclosing_cylinder.stl"))
    optimized_cylinder.export(os.path.join("output", "enclosing_cylinder_optimized.stl"))

