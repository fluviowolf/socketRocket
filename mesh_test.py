import importlib
import subprocess
import sys

import numpy as np
from trimesh.smoothing import filter_laplacian


def ensure_pip_is_current():
	try:
		subprocess.check_call(
			[sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
			stdout=subprocess.DEVNULL,
			stderr=subprocess.DEVNULL,
		)
	except subprocess.CalledProcessError:
		pass


def ensure_module(module_name, package_name=None):
	try:
		return importlib.import_module(module_name)
	except ImportError:
		package_name = package_name or module_name
		ensure_pip_is_current()
		try:
			subprocess.check_call(
				[sys.executable, "-m", "pip", "install", package_name],
				stdout=subprocess.DEVNULL,
				stderr=subprocess.DEVNULL,
			)
		except subprocess.CalledProcessError as exc:
			raise RuntimeError(f"Unable to install {package_name}.") from exc
		return importlib.import_module(module_name)


trimesh = ensure_module("trimesh")
pv = ensure_module("pyvista")
ensure_module("manifold3d")
ensure_module("rtree")
pymeshlab = ensure_module("pymeshlab")
ndimage = ensure_module("scipy.ndimage", "scipy")


def trimesh_to_pyvista(mesh):
	faces = mesh.faces
	padded_faces = np.hstack(
		[np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
	)
	return pv.PolyData(mesh.vertices, padded_faces)


def show_mesh(mesh, title, color="lightblue", opacity=1.0):
	plotter = pv.Plotter()
	plotter.add_mesh(
		trimesh_to_pyvista(mesh),
		color=color,
		opacity=opacity,
		show_edges=True,
	)
	plotter.add_axes()
	plotter.show(title=title)


def show_meshes_overlay(mesh_specs, title):
	"""Display multiple meshes together in one plotter. mesh_specs is a list of (mesh, color, opacity)."""
	plotter = pv.Plotter()
	for mesh, color, opacity in mesh_specs:
		plotter.add_mesh(
			trimesh_to_pyvista(mesh),
			color=color,
			opacity=opacity,
			show_edges=True,
		)
	plotter.add_axes()
	plotter.show(title=title)


def offset_mesh(mesh, offset=1.0):
	"""Shift every vertex along its normal by offset (positive grows the mesh outward)."""
	offset_result = mesh.copy()
	vertices = offset_result.vertices.copy()
	vertices += offset_result.vertex_normals * offset
	offset_result.vertices = vertices
	return offset_result


def remesh_preserve_flats(mesh, target_len=0.25, flat_tol=None):
	"""Isotropically remesh, then snap top/bottom vertices back to an exactly flat z."""
	z_top = mesh.vertices[:, 2].max()
	z_bottom = mesh.vertices[:, 2].min()
	if flat_tol is None:
		flat_tol = max((z_top - z_bottom) * 1e-3, 1e-3)

	ms = pymeshlab.MeshSet()
	ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
	ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(target_len))
	remeshed_data = ms.current_mesh()
	remeshed = trimesh.Trimesh(
		vertices=remeshed_data.vertex_matrix(),
		faces=remeshed_data.face_matrix(),
		process=False,
	)

	vertices = remeshed.vertices.copy()
	vertices[np.abs(vertices[:, 2] - z_top) < flat_tol, 2] = z_top
	vertices[np.abs(vertices[:, 2] - z_bottom) < flat_tol, 2] = z_bottom
	remeshed.vertices = vertices
	return remeshed


def erode_hull_xy(mesh, offset=1.0):
	"""Shrink mesh inward in XY by offset for every vertex, leaving each vertex's Z unchanged."""
	eroded = mesh.copy()
	vertices = eroded.vertices.copy()

	normals = eroded.vertex_normals
	xy_normals = normals[:, :2]
	magnitudes = np.linalg.norm(xy_normals, axis=1)
	safe_magnitudes = np.where(magnitudes > 1e-9, magnitudes, 1.0)
	unit_xy = xy_normals / safe_magnitudes[:, None]
	# vertices with a purely vertical normal (magnitude ~0) get no XY shift
	unit_xy[magnitudes <= 1e-9] = 0.0

	vertices[:, :2] -= unit_xy * offset
	eroded.vertices = vertices
	return eroded


def scale_hull_xy(mesh, offset=1.0):
	"""Scale mesh about its bounding-box center in X and Y so each side moves inward by offset mm."""
	scaled = mesh.copy()
	vertices = scaled.vertices.copy()
	bounds = scaled.bounds
	extents_xy = bounds[1, :2] - bounds[0, :2]
	scale_xy = (extents_xy - 2.0 * offset) / extents_xy
	center_xy = (bounds[1, :2] + bounds[0, :2]) / 2.0
	vertices[:, :2] = (vertices[:, :2] - center_xy) * scale_xy + center_xy
	scaled.vertices = vertices
	print(scaled)
	return scaled


def keep_largest_component(mesh):
	"""Split a mesh into connected islands and return only the largest by volume/area."""
	parts = mesh.split(only_watertight=False)
	if len(parts) <= 1:
		return mesh

	def part_size(part):
		if hasattr(part, "volume") and np.isfinite(part.volume):
			return abs(part.volume)
		return part.area

	return max(parts, key=part_size)


def main():
    path_fine_path = "path_fine.stl"
    envelop_path_path = "envelop_path.stl"

    # Import the STL files
    path_fine = trimesh.load(path_fine_path)
    if isinstance(path_fine, trimesh.Scene):
        path_fine = path_fine.dump(concatenate=True)

    envelop_path = trimesh.load(envelop_path_path)
    if isinstance(envelop_path, trimesh.Scene):
        envelop_path = envelop_path.dump(concatenate=True)

    # Generate the convex hull of envelop_path
    hull = envelop_path.convex_hull

    # Densify the hull so the erosion has enough side-wall vertices to work with
    # remeshed_hull = remesh_preserve_flats(hull, target_len=0.25)

    # Erode the hull 1 mm inward in XY, preserving the top/bottom edges
    # eroded_hull = erode_hull_xy(remeshed_hull, offset=1.0)

    # Scale the hull so each side moves inward 1 mm in X-Y, preserving Z
    eroded_hull = scale_hull_xy(hull, offset=2.0)

    # Display the eroded hull
    show_mesh(
        eroded_hull,
        "envelop_path Convex Hull scaled -1mm XY",
        color="purple",
    )

    # Export the eroded hull
    eroded_hull_output_path = "hull_eroded_xy.stl"
    eroded_hull.export(eroded_hull_output_path)
    print(f"Eroded convex hull exported to: {eroded_hull_output_path}")

    # Boolean subtract the hull from path_fine
    difference_result = path_fine.difference(hull, engine="manifold")

    # Display the subtraction result
    show_mesh(
        difference_result,
        "path_fine minus envelop_path Convex Hull",
        color="darkgreen",
    )

    # Export the subtraction result
    difference_output_path = "path_fine_minus_hull.stl"
    difference_result.export(difference_output_path)
    print(f"Boolean difference exported to: {difference_output_path}")

    # Boolean intersect the hull with path_fine
    intersection_result = path_fine.intersection(hull, engine="manifold")

    # Remove isolated islands, keeping only the largest component
    intersection_result = keep_largest_component(intersection_result)

    # Display the intersection result
    show_mesh(
        intersection_result,
        "path_fine intersected with envelop_path Convex Hull",
        color="darkorange",
    )

    # Export the intersection result
    intersection_output_path = "path_fine_intersect_hull.stl"
    intersection_result.export(intersection_output_path)
    print(f"Boolean intersection exported to: {intersection_output_path}")

    # Boolean subtract the eroded hull from path_fine
    eroded_difference_result = path_fine.difference(eroded_hull, engine="manifold")

    # Display the eroded subtraction result
    show_mesh(
        eroded_difference_result,
        "path_fine minus eroded Convex Hull",
        color="darkgreen",
    )

    # Export the eroded subtraction result
    eroded_difference_output_path = "path_fine_minus_eroded_hull.stl"
    eroded_difference_result.export(eroded_difference_output_path)
    print(f"Eroded boolean difference exported to: {eroded_difference_output_path}")

    # Boolean intersect the eroded hull with path_fine
    eroded_intersection_result = path_fine.intersection(eroded_hull, engine="manifold")

    # Remove isolated islands, keeping only the largest component
    eroded_intersection_result = keep_largest_component(eroded_intersection_result)

    # Display the eroded intersection result
    show_mesh(
        eroded_intersection_result,
        "path_fine intersected with eroded Convex Hull",
        color="darkorange",
    )

    # Export the eroded intersection result
    eroded_intersection_output_path = "path_fine_intersect_eroded_hull.stl"
    eroded_intersection_result.export(eroded_intersection_output_path)
    print(f"Eroded boolean intersection exported to: {eroded_intersection_output_path}")

    # Offset the isolated eroded-hull intersection mesh 1 mm outward
    eroded_intersection_offset_result = offset_mesh(eroded_intersection_result, offset=1.0)

    # Display the offset mesh overlaid on the original path_fine mesh
    show_meshes_overlay(
        [
            (path_fine, "lightblue", 0.3),
            (eroded_intersection_offset_result, "darkorange", 1.0),
        ],
        "path_fine with 1mm-offset eroded-hull intersection overlay",
    )

    # Export the offset eroded-hull intersection mesh
    eroded_intersection_offset_output_path = "path_fine_intersect_eroded_hull_offset1mm.stl"
    eroded_intersection_offset_result.export(eroded_intersection_offset_output_path)
    print(f"Offset eroded intersection exported to: {eroded_intersection_offset_output_path}")


if __name__ == "__main__":
    main()
