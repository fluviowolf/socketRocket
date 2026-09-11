import numpy as np
import pyvista as pv
import trimesh
import os
from PreSlice import robust_boolean_intersection, show_meshes_overlay

def trimesh_to_pyvista(mesh):
    """Convert a trimesh mesh to a PyVista surface mesh."""
    faces = mesh.faces
    padded_faces = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
    )
    return pv.PolyData(mesh.vertices, padded_faces)

def extract_top_and_bottom_surfaces(mesh, tolerance_mm=0.0):
    """Extract mesh faces contained within the top and bottom Z bands."""
    if tolerance_mm < 0:
        raise ValueError("tolerance_mm must not be negative")

    z_values = mesh.vertices[:, 2]
    z_min = z_values.min()
    z_max = z_values.max()
    top_mask = np.all(mesh.vertices[mesh.faces, 2] >= z_max - tolerance_mm, axis=1)
    bottom_mask = np.all(mesh.vertices[mesh.faces, 2] <= z_min + tolerance_mm, axis=1)

    def build_surface(face_mask, name):
        faces = mesh.faces[face_mask]
        if len(faces) == 0:
            raise ValueError(f"No faces found for the {name} surface")
        surface = trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=faces,
            process=False,
        )
        surface.remove_unreferenced_vertices()
        return surface

    return build_surface(top_mask, "top"), build_surface(bottom_mask, "bottom")

def boundary_loop(surface):
    """Return the ordered boundary loop of a surface mesh."""
    face_edges = np.vstack([
        surface.faces[:, [0, 1]],
        surface.faces[:, [1, 2]],
        surface.faces[:, [2, 0]],
    ])
    undirected_edges = np.sort(face_edges, axis=1)
    edges, counts = np.unique(undirected_edges, axis=0, return_counts=True)
    boundary_edges = edges[counts == 1]
    if len(boundary_edges) < 3:
        raise ValueError("Surface does not contain a boundary loop")

    neighbors = {}
    for start, end in boundary_edges:
        neighbors.setdefault(int(start), []).append(int(end))
        neighbors.setdefault(int(end), []).append(int(start))
    loop = [int(boundary_edges[0, 0])]
    previous = None
    current = loop[0]
    while True:
        candidates = [index for index in neighbors[current] if index != previous]
        next_index = candidates[0]
        if next_index == loop[0]:
            break
        loop.append(next_index)
        previous, current = current, next_index
        if len(loop) > len(boundary_edges):
            raise ValueError("Could not order the surface boundary loop")
    return surface.vertices[loop]

def resample_closed_loop(points, sample_count):
    """Resample a closed 3D loop at equal XY perimeter intervals."""
    closed = np.vstack([points, points[0]])
    lengths = np.linalg.norm(np.diff(closed[:, :2], axis=0), axis=1)
    distances = np.concatenate([[0.0], np.cumsum(lengths)])
    samples = np.linspace(0.0, distances[-1], sample_count, endpoint=False)
    return np.column_stack([
        np.interp(samples, distances, closed[:, axis])
        for axis in range(3)
    ])

def loft_between_boundary_loops(top_surface, bottom_surface, sample_count=128):
    """Loft the actual top and bottom surface boundary loops."""
    top_loop = resample_closed_loop(boundary_loop(top_surface), sample_count)
    bottom_loop = resample_closed_loop(boundary_loop(bottom_surface), sample_count)
    vertices = np.vstack([top_loop, bottom_loop])
    faces = []
    for index in range(sample_count):
        next_index = (index + 1) % sample_count
        faces.extend([
            [index, sample_count + index, next_index],
            [next_index, sample_count + index, sample_count + next_index],
        ])
    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)

def cap_boundary_loft(loft, sample_count=128):
    """Close the loft with planar caps so it can be used as a boolean volume."""
    top_ring = loft.vertices[:sample_count]
    bottom_ring = loft.vertices[sample_count:2 * sample_count]
    top_center = top_ring.mean(axis=0)
    bottom_center = bottom_ring.mean(axis=0)
    top_center_index = len(loft.vertices)
    bottom_center_index = top_center_index + 1
    vertices = np.vstack([loft.vertices, top_center, bottom_center])
    faces = loft.faces.tolist()

    for index in range(sample_count):
        next_index = (index + 1) % sample_count
        faces.append([top_center_index, next_index, index])
        faces.append([
            bottom_center_index,
            sample_count + index,
            sample_count + next_index,
        ])

    capped = trimesh.Trimesh(
        vertices=vertices,
        faces=np.array(faces),
        process=True,
    )
    capped.remove_unreferenced_vertices()
    capped.fix_normals()
    return capped

def scale_boundary_loft_xy(mesh, offset=2.0):
    """Scale a boundary loft inward in X and Y by offset millimeters."""
    scaled = mesh.copy()
    vertices = scaled.vertices.copy()
    bounds = scaled.bounds
    extents_xy = bounds[1, :2] - bounds[0, :2]
    scale_xy = (extents_xy - 2.0 * offset) / extents_xy
    center_xy = (bounds[1, :2] + bounds[0, :2]) / 2.0
    vertices[:, :2] = (vertices[:, :2] - center_xy) * scale_xy + center_xy
    scaled.vertices = vertices
    return scaled

# The previous fusion and isotropic-remeshing experiment is intentionally disabled.
# def fuse_and_isotropic_remesh(meshes, target_length=0.25):
#     ...

def display_mesh(mesh, title, color):
    plotter = pv.Plotter()
    plotter.add_mesh(
        trimesh_to_pyvista(mesh),
        color=color,
        opacity=0.8,
        show_edges=True,
    )
    plotter.add_axes()
    plotter.show(title=title)

def display_boundary_loop_loft(surface_tolerance_mm=0.0, sample_count=128):
    """Extract the top/bottom surfaces, loft their boundaries, and display them."""
    input_path = os.path.join(os.path.dirname(__file__), "input", "envelop.stl")
    mesh = trimesh.load_mesh(input_path)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f"Unable to load a non-empty mesh from {input_path}")

    top_surface, bottom_surface = extract_top_and_bottom_surfaces(
        mesh,
        tolerance_mm=surface_tolerance_mm,
    )
    display_mesh(top_surface, "Top Surface of Envelop", "cornflowerblue")
    display_mesh(bottom_surface, "Bottom Surface of Envelop", "lightgreen")

    boundary_loft = loft_between_boundary_loops(
        top_surface,
        bottom_surface,
        sample_count=sample_count,
    )
    offset_text = input("Enter XY scale offset in mm [2.0]: ").strip()
    scale_offset_mm = 2.0 if not offset_text else float(offset_text)
    scaled_boundary_loft = scale_boundary_loft_xy(
        boundary_loft,
        offset=scale_offset_mm,
    )
    show_meshes_overlay(
        [
            (boundary_loft, "gold", 0.45),
            (scaled_boundary_loft, "darkorange", 0.55),
        ],
        f"Boundary Loop and {scale_offset_mm:g} mm XY-Scaled Boundary Loop",
    )

    # The previous fusion/remeshing experiment did not produce the expected result.
    # fused_mesh = trimesh.util.concatenate([
    #     top_surface,
    #     bottom_surface,
    #     boundary_loft,
    # ])
    # display_mesh(fused_mesh, "Fused Top, Bottom, and Boundary Surfaces", "steelblue")
    # remeshed_mesh = fuse_and_isotropic_remesh(
    #     [top_surface, bottom_surface, boundary_loft],
    #     target_length=0.25,
    # )
    # display_mesh(
    #     remeshed_mesh,
    #     "Fused Surface After Isotropic Remesh (0.25 mm)",
    #     "mediumseagreen",
    # )

    path_input_path = os.path.join(os.path.dirname(__file__), "input", "path.stl")
    path_mesh = trimesh.load_mesh(path_input_path)
    if not isinstance(path_mesh, trimesh.Trimesh) or len(path_mesh.vertices) == 0:
        raise ValueError(f"Unable to load a non-empty mesh from {path_input_path}")
    display_mesh(path_mesh, "Path Mesh", "cornflowerblue")

    scaled_boundary_volume = cap_boundary_loft(
        scaled_boundary_loft,
        sample_count=sample_count,
    )
    if not scaled_boundary_volume.is_volume:
        raise RuntimeError("Scaled boundary-loop loft is not a valid boolean volume")
    path_inside_scaled_boundary = robust_boolean_intersection(
        path_mesh,
        scaled_boundary_volume,
    )
    display_mesh(
        path_inside_scaled_boundary,
        f"Path Inside {scale_offset_mm:g} mm XY-Scaled Boundary Loop",
        "mediumseagreen",
    )

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    boundary_loft.export(os.path.join(output_dir, "loft.stl"))
    scaled_boundary_loft.export(os.path.join(output_dir, "loft_scaled_xy.stl"))
    path_inside_scaled_boundary.export(
        os.path.join(output_dir, "path_inside_scaled_boundary_loft.stl")
    )

    return (
        mesh,
        top_surface,
        bottom_surface,
        boundary_loft,
        scaled_boundary_loft,
        path_mesh,
        path_inside_scaled_boundary,
    )

if __name__ == "__main__":
    display_boundary_loop_loft()



