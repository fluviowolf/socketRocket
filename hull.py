import numpy as np
import pyvista as pv
import pymeshlab
import trimesh
import os

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

def fuse_and_isotropic_remesh(meshes, target_length=0.25):
    """Fuse component meshes and remesh the result at a uniform target length."""
    if target_length <= 0:
        raise ValueError("target_length must be greater than zero")

    fused_mesh = trimesh.util.concatenate(meshes)
    fused_mesh.merge_vertices()
    fused_mesh.remove_unreferenced_vertices()

    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=fused_mesh.vertices,
            face_matrix=fused_mesh.faces,
        )
    )
    mesh_set.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PureValue(target_length),
    )
    remeshed_data = mesh_set.current_mesh()
    return trimesh.Trimesh(
        vertices=remeshed_data.vertex_matrix(),
        faces=remeshed_data.face_matrix(),
        process=False,
    )

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
    display_mesh(
        boundary_loft,
        "Boundary-Loop Loft (preserves extracted XY outlines)",
        "gold",
    )

    fused_mesh = trimesh.util.concatenate([
        top_surface,
        bottom_surface,
        boundary_loft,
    ])
    display_mesh(fused_mesh, "Fused Top, Bottom, and Boundary Surfaces", "steelblue")

    remeshed_mesh = fuse_and_isotropic_remesh(
        [top_surface, bottom_surface, boundary_loft],
        target_length=0.25,
    )
    display_mesh(
        remeshed_mesh,
        "Fused Surface After Isotropic Remesh (0.25 mm)",
        "mediumseagreen",
    )
    output_path = os.path.join(os.path.dirname(__file__), "output", "loft.stl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    boundary_loft.export(output_path)
    remeshed_output_path = os.path.join(
        os.path.dirname(__file__),
        "output",
        "loft_remeshed.stl",
    )
    remeshed_mesh.export(remeshed_output_path)

    return mesh, top_surface, bottom_surface, boundary_loft, fused_mesh, remeshed_mesh

if __name__ == "__main__":
    display_boundary_loop_loft()



