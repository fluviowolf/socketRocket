import importlib
import os
import subprocess
import sys
from collections import Counter

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


def remesh_uniform(mesh, target_len=0.25):
	"""Isotropically remesh to a uniform target edge length."""
	ms = pymeshlab.MeshSet()
	ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
	ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(target_len))
	remeshed_data = ms.current_mesh()
	return trimesh.Trimesh(
		vertices=remeshed_data.vertex_matrix(),
		faces=remeshed_data.face_matrix(),
		process=False,
	)


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


def voxel_dilate_mesh(mesh, offset_mm=2.0, pitch_mm=0.20):
	"""Fill and dilate a mesh on a voxel grid by offset_mm, returning the voxel volume."""
	if offset_mm < 0:
		raise ValueError("offset_mm must not be negative.")
	if pitch_mm <= 0:
		raise ValueError("pitch_mm must be greater than zero.")

	voxel_grid = mesh.voxelized(pitch_mm).fill()
	padding = int(np.ceil(offset_mm / pitch_mm)) + 1
	voxel_matrix = np.pad(voxel_grid.matrix, padding, mode="constant")
	transform = voxel_grid.transform.copy()
	transform[:3, 3] -= transform[:3, :3] @ np.full(3, padding)

	if offset_mm > 0:
		distance = ndimage.distance_transform_edt(
			~voxel_matrix,
			sampling=pitch_mm,
		)
		voxel_matrix |= distance <= offset_mm

	return trimesh.voxel.VoxelGrid(voxel_matrix, transform=transform)


def voxel_to_mesh(voxel_grid):
	"""Reconstruct a surface mesh from a voxel volume via marching cubes."""
	# VoxelGrid.marching_cubes returns vertices in raw index units, so the
	# grid transform (pitch scale + origin) must be applied explicitly.
	surface_mesh = voxel_grid.marching_cubes
	surface_mesh.apply_transform(voxel_grid.transform)
	return surface_mesh


def plane_cut_from_bottom(mesh, height_mm=10.0):
	"""Cut the mesh with a horizontal plane height_mm above its lowest Z, keeping the upper portion."""
	z_min = mesh.bounds[0, 2]
	cut = mesh.slice_plane(
		plane_origin=[0.0, 0.0, z_min + height_mm],
		plane_normal=[0.0, 0.0, 1.0],
		cap=True,
	)
	if cut is None or len(cut.vertices) == 0:
		raise RuntimeError("Plane cut produced an empty result mesh.")
	return cleanup_mesh(cut)

def cleanup_mesh(mesh):
	mesh = mesh.copy()
	mesh.merge_vertices()
	if hasattr(mesh, "remove_degenerate_faces"):
		mesh.remove_degenerate_faces()
	if hasattr(mesh, "remove_duplicate_faces"):
		mesh.remove_duplicate_faces()
	if hasattr(mesh, "remove_unreferenced_vertices"):
		mesh.remove_unreferenced_vertices()
	return mesh

def isotropic_remesh(mesh, finemesh_path, targetlen=1.0):
	
	ms = pymeshlab.MeshSet()
	ms.add_mesh(
		pymeshlab.Mesh(
			vertex_matrix=mesh.vertices,
			face_matrix=mesh.faces,
		)
	)

	# Perform isotropic explicit remeshing with a target edge length of 0.25 mm
	ms.meshing_isotropic_explicit_remeshing(
		targetlen=pymeshlab.PureValue(targetlen)
	)

	# Save the remeshed result
	ms.save_current_mesh(finemesh_path)
	finemesh = ms.current_mesh()
	return trimesh.Trimesh(
			vertices=finemesh.vertex_matrix(),
			faces=finemesh.face_matrix(),
			process=False,
	)

def keep_largest_component(mesh):
	"""Split a mesh into connected islands and return the part that wins on at
	least 2 of 3 metrics: vertex count, face count, and volume/area."""
	parts = mesh.split(only_watertight=False)
	if len(parts) == 0:
		return mesh, 0
	if len(parts) == 1:
		return parts[0], 1

	def part_size(part):
		if hasattr(part, "volume") and np.isfinite(part.volume):
			return abs(part.volume)
		return part.area

	metrics = [(len(part.vertices), len(part.faces), part_size(part)) for part in parts]
	winners = [max(range(len(parts)), key=lambda i: metrics[i][m]) for m in range(3)]
	vote_counts = Counter(winners)
	majority_index, majority_votes = vote_counts.most_common(1)[0]
	# fall back to the volume/area winner if no metric agrees on a single part
	largest_index = majority_index if majority_votes >= 2 else winners[2]

	return parts[largest_index], len(parts)


def estimate_coplanar_normal(mesh):
	centered = mesh.vertices - mesh.vertices.mean(axis=0)
	cov = np.cov(centered.T)
	eigenvalues, eigenvectors = np.linalg.eigh(cov)
	return eigenvectors[:, np.argmin(eigenvalues)]


def robust_boolean_difference(blank_mesh, path_mesh):
	normal = estimate_coplanar_normal(blank_mesh)
	scale = max(np.linalg.norm(blank_mesh.extents), 1.0)
	eps = scale * 1e-6
	shift_families = {
		"positive": [0.0, eps, 5.0 * eps, 10.0 * eps],
		"negative": [0.0, -eps, -5.0 * eps, -10.0 * eps],
	}

	def score_candidate(candidate, component_count):
		volume = abs(candidate.volume) if np.isfinite(candidate.volume) else 0.0
		return (int(candidate.is_watertight), -component_count, volume, candidate.area)

	def best_for_family(shifts):
		best_mesh = None
		best_score = None
		best_shift = None
		for shift in shifts:
			cutter = path_mesh.copy()
			if shift != 0.0:
				cutter.apply_translation(normal * shift)

			try:
				candidate = blank_mesh.difference(cutter, engine="manifold")
			except BaseException:
				continue

			if candidate is None or len(candidate.vertices) == 0:
				continue

			candidate = cleanup_mesh(candidate)
			candidate, component_count = keep_largest_component(candidate)
			candidate = cleanup_mesh(candidate)

			if candidate is None or len(candidate.vertices) == 0:
				continue

			candidate_score = score_candidate(candidate, component_count)
			if best_score is None or candidate_score > best_score:
				best_mesh = candidate
				best_score = candidate_score
				best_shift = shift

		return best_mesh, best_shift

	best_pos, shift_pos = best_for_family(shift_families["positive"])
	best_neg, shift_neg = best_for_family(shift_families["negative"])

	if best_pos is None and best_neg is None:
		raise RuntimeError("Robust boolean difference failed for all coplanar offset attempts.")

	if best_pos is None:
		print(f"Using negative-shift candidate only: {shift_neg}")
		return best_neg

	if best_neg is None:
		print(f"Using positive-shift candidate only: {shift_pos}")
		return best_pos

	try:
		combined = best_pos.intersection(best_neg, engine="manifold")
	except BaseException:
		combined = None

	if combined is not None and len(combined.vertices) > 0:
		combined = cleanup_mesh(combined)
		combined, _ = keep_largest_component(combined)
		combined = cleanup_mesh(combined)
		if combined is not None and len(combined.vertices) > 0:
			print(f"Using mirrored-offset intersection: +{shift_pos} and {shift_neg}")
			return combined

	vol_pos = abs(best_pos.volume) if np.isfinite(best_pos.volume) else 0.0
	vol_neg = abs(best_neg.volume) if np.isfinite(best_neg.volume) else 0.0
	chosen = best_pos if vol_pos >= vol_neg else best_neg
	chosen_shift = shift_pos if vol_pos >= vol_neg else shift_neg
	print(f"Mirrored intersection unavailable, fallback shift: {chosen_shift}")
	return chosen


def robust_boolean_intersection(mesh_a, mesh_b):
	try:
		result = mesh_a.intersection(mesh_b, engine="manifold")
	except BaseException as exc:
		raise RuntimeError(f"Boolean intersection failed: {exc}") from exc

	if result is None or len(result.vertices) == 0:
		raise RuntimeError("Boolean intersection produced an empty result mesh.")

	result = cleanup_mesh(result)
	result, _ = keep_largest_component(result)
	result = cleanup_mesh(result)

	if result is None or len(result.vertices) == 0:
		raise RuntimeError("Boolean intersection cleanup produced an empty mesh.")

	return result

def main():

    # 1. Upload Path and Blank Mesh Files
    path_mesh = trimesh.load("path.stl")
    if isinstance(path_mesh, trimesh.Scene):
        path_mesh = path_mesh.dump(concatenate=True)
    blank_mesh = trimesh.load("blank.stl")
    if isinstance(blank_mesh, trimesh.Scene):
        blank_mesh = blank_mesh.dump(concatenate=True)

    # 2. Plane cut 10 mm from the bottom of the path and blank meshes
    plane_cut_height_mm = 10.0
    path_mesh = plane_cut_from_bottom(path_mesh, height_mm=plane_cut_height_mm)
    blank_mesh = plane_cut_from_bottom(blank_mesh, height_mm=plane_cut_height_mm)

    show_mesh(path_mesh, "Path Mesh (Plane Cut)", color="cornflowerblue")
    show_mesh(blank_mesh, "Blank Mesh (Plane Cut)", color="lightgray")

    # 3 - Remesh path and blank meshes
    path_fine = isotropic_remesh(path_mesh, "path_fine.stl", targetlen=1)
    blank_fine = isotropic_remesh(blank_mesh, "blank_fine.stl", targetlen=1)

    show_mesh(path_fine, "Path Mesh", color="cornflowerblue")
    show_mesh(blank_fine, "Blank Mesh", color="lightgray")

    # 4 - Boolean Difference (path - blank = envelop)
    envelop_mesh = robust_boolean_difference(blank_fine, path_fine)
    if envelop_mesh is None or len(envelop_mesh.vertices) == 0:
        raise RuntimeError("Boolean difference produced an empty result mesh.")
    envelop_mesh = cleanup_mesh(envelop_mesh)
    show_mesh(envelop_mesh, "Envelop Mesh (Path - Blank)", color="lightgreen")
    envelop_mesh.export("envelop.stl")

    # 7. Generate Convex Hull and Eroded Convex Hull
    hull = envelop_mesh.convex_hull
    eroded_hull = scale_hull_xy(hull, offset=2.0)

    show_meshes_overlay(
        [
            (path_fine, "lightblue", 0.3),
			(hull, "lightgreen", 0.3),
            (eroded_hull, "darkorange", 1.0),
        ],
        "Convex Hull and Eroded/Scaled Hull",
    )

    hull_output_path = "hull.stl"
    eroded_hull_output_path = "hull_eroded.stl"
    hull.export(hull_output_path)
    eroded_hull.export(eroded_hull_output_path)
    print(f"Eroded convex hull exported to: {eroded_hull_output_path}")

    # 8. Boolean Subtraction of Path - Eroded Hull
    eroded_difference_result = path_fine.difference(eroded_hull, engine="manifold")

    # Display the eroded subtraction result
    show_mesh(
        eroded_difference_result,
        "Path Minus Eroded/Scaled Hull",
        color="lightgreen",
    )

    # Export the eroded subtraction result
    eroded_difference_output_path = "path_fine_minus_eroded_hull.stl"
    eroded_difference_result.export(eroded_difference_output_path)
    print(f"Eroded boolean difference exported to: {eroded_difference_output_path}")

    # 9. Boolean Intersection of Path and Eroded Hull
    eroded_intersection_result = path_fine.intersection(eroded_hull, engine="manifold")

    # Remove isolated islands, keeping only the largest component
    eroded_intersection_result, _ = keep_largest_component(eroded_intersection_result)

    # Display the eroded intersection result
    show_mesh(
        eroded_intersection_result,
        "Implant Core",
        color="darkorange",
    )

    # Export the eroded intersection result
    eroded_intersection_output_path = "implant_core.stl"
    eroded_intersection_result.export(eroded_intersection_output_path)
    print(f"Eroded boolean intersection exported to: {eroded_intersection_output_path}")

    # 10. Isotropic Remesh of Implant Core
    remeshed_intersection_result = remesh_uniform(eroded_intersection_result, target_len=0.20)

    # Display the remeshed mesh overlaid on the original path_fine mesh
    show_meshes_overlay(
        [
            (path_fine, "lightblue", 0.3),
            (remeshed_intersection_result, "darkorange", 1.0),
        ],
        "Remeshed Implant Core over Path",
    )

    remeshed_intersection_output_path = "implant_core_remeshed.stl"
    remeshed_intersection_result.export(remeshed_intersection_output_path)

    # 10. Voxel Conversion of Implant Core
    voxel_dilated_result = voxel_dilate_mesh(remeshed_intersection_result, offset_mm=1.00)

    #voxel_dilated_output_path = "implant_core_dilated.stl"
    #voxel_dilated_result.export(voxel_dilated_output_path)
	
    # 11. Convert Voxelized Volume back to Mesh
    mesh_from_voxel_result = voxel_to_mesh(voxel_dilated_result)

    # Display the voxelized mesh overlaid on the original path_fine mesh
    show_meshes_overlay(
        [
            (path_fine, "lightblue", 0.3),
            (mesh_from_voxel_result, "darkorange", 1.0),
        ],
        "Mesh Implant Core over Path",
    )

    # 12. Smooth Mesh
    smooth_mesh_result = filter_laplacian(mesh_from_voxel_result, iterations=20)

    # Display the smoothed mesh overlaid on the original path_fine mesh
    show_meshes_overlay(
        [
            (path_fine, "lightblue", 0.3),
            (smooth_mesh_result, "darkorange", 1.0),
        ],
        "Smooth Implant Core over Path",
    )

    smooth_mesh_output_path = "implant_core_smooth.stl"
    smooth_mesh_result.export(smooth_mesh_output_path)

    outer_puck = blank_fine.difference(envelop_mesh, engine="manifold")
    outer_puck = outer_puck.difference(smooth_mesh_result, engine="manifold")
    inner_puck = envelop_mesh.difference(smooth_mesh_result, engine="manifold")

    show_meshes_overlay(
            [
                (outer_puck, "lightblue", 0.5),
                (inner_puck, "lightgreen", 0.2),
                (smooth_mesh_result, "darkorange", 1.0),
            ],
            "Final",
    )

    outer_puck_output_path = "outer_puck.stl"
    inner_puck_output_path = "inner_puck.stl"
    implant_envelop_output_path = "implant_envelop.stl"
    outer_puck.export(outer_puck_output_path)
    inner_puck.export(inner_puck_output_path)
    smooth_mesh_result.export(implant_envelop_output_path)


if __name__ == "__main__":
    main()
