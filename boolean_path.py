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


def subdivide_mesh(mesh, max_edge_length=1.0):
	"""Subdivide until the longest unique mesh edge is at most max_edge_length."""
	if max_edge_length <= 0:
		raise ValueError("max_edge_length must be greater than zero.")

	subdivided = mesh.copy()
	while len(subdivided.edges_unique_length) > 0:
		if subdivided.edges_unique_length.max() <= max_edge_length:
			break
		subdivided = subdivided.subdivide()
	return subdivided


def sampled_convex_hull(mesh, count=50000):
	"""Build a convex hull from evenly sampled points on the input mesh hull."""
	if count < 4:
		raise ValueError("count must be at least four.")

	hull = mesh.convex_hull
	points = trimesh.sample.sample_surface_even(hull, count=count)[0]
	return trimesh.convex.convex_hull(points)


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
	parts = mesh.split(only_watertight=False)
	if len(parts) == 0:
		return mesh, 0
	if len(parts) == 1:
		return parts[0], 1

	def part_size(part):
		if hasattr(part, "volume") and np.isfinite(part.volume):
			return abs(part.volume)
		return part.area

	largest = max(parts, key=part_size)
	return largest, len(parts)


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


def dilate_mesh(mesh, offset_mm=1.5):
	"""Expand a watertight mesh by offset_mm using its vertex normals."""
	dilated = mesh.copy()
	dilated.vertex_normals
	dilated.vertices += dilated.vertex_normals * offset_mm
	return cleanup_mesh(dilated)


def voxel_offset_mesh(mesh, offset_mm=2.0, pitch_mm=0.5):
	"""Fill and expand a mesh on a voxel grid, then reconstruct its surface."""
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

	return trimesh.voxel.VoxelGrid(
		voxel_matrix,
		transform=transform,
	).marching_cubes


def subtract_mesh_components(blank_mesh, cutter_mesh):
	"""Subtract a mesh and preserve each resulting disconnected surface."""
	try:
		result = blank_mesh.difference(cutter_mesh, engine="manifold")
	except BaseException as exc:
		raise RuntimeError(f"Boolean subtraction failed: {exc}") from exc

	if result is None or len(result.vertices) == 0:
		raise RuntimeError("Boolean subtraction produced an empty result mesh.")

	components = [cleanup_mesh(component) for component in result.split(only_watertight=False)]
	components = [component for component in components if len(component.vertices) > 0]
	components.sort(key=lambda component: component.area, reverse=True)

	if len(components) < 2:
		raise RuntimeError(
			f"Boolean subtraction produced {len(components)} surface(s); expected at least 2."
		)

	return components


if __name__ == "__main__":

	# 1 - Upload path and blank meshes
	path_mesh = trimesh.load("path.stl")
	if isinstance(path_mesh, trimesh.Scene):
		path_mesh = path_mesh.dump(concatenate=True)
	blank_mesh = trimesh.load("blank.stl")
	if isinstance(blank_mesh, trimesh.Scene):
		blank_mesh = blank_mesh.dump(concatenate=True)

	# 2 - Remesh path and blank meshes
	path_mesh = isotropic_remesh(path_mesh, "path_fine.stl", targetlen=1)
	blank_mesh = isotropic_remesh(blank_mesh, "blank_fine.stl", targetlen=1)

	show_mesh(path_mesh, "Path Mesh", color="cornflowerblue")
	show_mesh(blank_mesh, "Blank Mesh", color="lightgray")

	# 3 - Boolean Difference (path - blank = envelop)
	envelop_mesh = robust_boolean_difference(blank_mesh, path_mesh)
	if envelop_mesh is None or len(envelop_mesh.vertices) == 0:
		raise RuntimeError("Boolean difference produced an empty result mesh.")
	envelop_mesh = cleanup_mesh(envelop_mesh)

	show_mesh(envelop_mesh, "Envelop Mesh (Path - Blank)", color="lightgreen")
	envelop_mesh.export("envelop_path.stl")

	# 4 - Voxelization of envelop mesh
	voxel_pitch = 1.0
	voxel_grid = envelop_mesh.voxelized(pitch=voxel_pitch).fill()
	envelop_voxel_mesh = voxel_grid.as_boxes()

	# Preserve the result mesh dimensions after voxelization.
	result_extents = envelop_mesh.extents
	voxel_extents = envelop_voxel_mesh.extents
	scale_factors = np.divide(
		result_extents,
		voxel_extents,
		out=np.ones_like(result_extents),
		where=voxel_extents != 0,
	)
	envelop_voxel_mesh.apply_scale(scale_factors)
	result_center = envelop_mesh.bounds.mean(axis=0)
	voxel_center = envelop_voxel_mesh.bounds.mean(axis=0)
	envelop_voxel_mesh.apply_translation(result_center - voxel_center)

	show_mesh(
		envelop_voxel_mesh,
		"Voxelized Envelop",
		color="limegreen",
	)
	envelop_voxel_mesh.export("envelop_voxel_mesh.stl")

	# 5 - Dilate Voxelized Envelop
	dilation_mm = 1.0
	dilation_radius = int(np.ceil(dilation_mm / voxel_pitch))
	dilation_padding = dilation_radius + 1
	dilated_matrix = np.pad(
		voxel_grid.matrix,
		dilation_padding,
		mode="constant",
	)
	dilated_transform = voxel_grid.transform.copy()
	dilated_transform[:3, 3] -= (
		dilated_transform[:3, :3] @ np.full(3, dilation_padding)
	)
	distance = ndimage.distance_transform_edt(
		~dilated_matrix,
		sampling=voxel_pitch,
	)
	dilated_matrix |= distance <= dilation_mm
	dilated_voxel_grid = trimesh.voxel.VoxelGrid(
		dilated_matrix,
		transform=dilated_transform,
	)
	envelop_voxel_mesh_dilated = dilated_voxel_grid.as_boxes()

	show_mesh(
		envelop_voxel_mesh_dilated,
		"Dilated Voxelized Envelop",
		color="darkgreen",
	)
	envelop_voxel_mesh_dilated.export("envelop_voxel_mesh_dilated.stl")

	# 6 - Convert Dilated Envelop Voxel Mesh to Surface Mesh
	envelop_surface_mesh_dilated = cleanup_mesh(dilated_voxel_grid.marching_cubes)
	voxel_object_center = envelop_voxel_mesh_dilated.bounds.mean(axis=0)
	surface_center = envelop_surface_mesh_dilated.bounds.mean(axis=0)
	envelop_surface_mesh_dilated.apply_translation(voxel_object_center - surface_center)
	show_mesh(
		envelop_surface_mesh_dilated,
		"Surface Mesh from 1 mm Dilated Voxels",
		color="seagreen",
	)
	envelop_surface_mesh_dilated.export("envelop_surface_mesh_dilated.stl")

	# 7 - Smooth Envelop Surface Mesh
	envelop_smooth_mesh = envelop_surface_mesh_dilated.copy()
	filter_laplacian(envelop_smooth_mesh, iterations=20)
	envelop_smooth_mesh = cleanup_mesh(envelop_smooth_mesh)
	show_mesh(
		envelop_smooth_mesh,
		"Envelop Smooth",
		color="mediumseagreen",
	)
	envelop_smooth_mesh.export("envelop_smooth_mesh.stl")

	# 8 - Remeshed Envelop
	envelop_remeshed = isotropic_remesh(
		envelop_smooth_mesh,
		"envelop_remeshed.stl",
		targetlen=1.0,
	)
	envelop_remeshed = cleanup_mesh(
		envelop_remeshed
	)
	envelop_remeshed.export(
		"envelop_remeshed.stl"
	)
	show_mesh(
		envelop_remeshed,
		"Remeshed Envelop",
		color="teal",
	)

	# 9 - Envelop Intersection
	implant_core = robust_boolean_intersection(
		envelop_remeshed,
		path_mesh,
	)
	show_mesh(
		implant_core,
		"implant_core.stl",
		color="tomato",
	)
	implant_core.export("implant_core.stl")

	# 10 - Implant Core Remesh
	implant_core = isotropic_remesh(
		implant_core,
		"implant_core_remeshed.stl",
		targetlen=1.0,
	)
	implant_core = cleanup_mesh(implant_core)
	show_mesh(
		implant_core,
		"Remeshed Implant Core",
		color="coral",
	)
	implant_core.export("implant_core_remeshed.stl")

	# 11 - Dilate Implant Core
	implant_core_dilated = cleanup_mesh(
		dilate_mesh(implant_core, offset_mm=1.0)
	)
	show_mesh(
		implant_core_dilated,
		"Dilated Implant Core",
		color="crimson",
	)
	implant_core_dilated.export(
		"smoothed_remeshed_intersection_path_dilated_1mm.stl"
	)

	# 12 - Final Outer and Inner Blanks
	cut_result = blank_mesh.difference(implant_core_dilated, engine="manifold")
	if cut_result is None or len(cut_result.vertices) == 0:
		raise RuntimeError("Final boolean subtraction produced an empty result mesh.")
	cut_components = [
		cleanup_mesh(component)
		for component in cut_result.split(only_watertight=False)
	]
	cut_components = [component for component in cut_components if len(component.vertices) > 0]
	if len(cut_components) < 2:
		raise RuntimeError(
			f"Final boolean subtraction produced {len(cut_components)} blank part(s); "
			"expected an outer and inner blank."
		)

	cut_components.sort(key=lambda component: component.area, reverse=True)
	outer_blank = cut_components[0]
	inner_blank = cut_components[-1]

	show_mesh(outer_blank, "Outer Blank", color="lightblue")
	outer_blank.export("outer_blank.stl")
	show_mesh(inner_blank, "Inner Blank", color="lightgreen")
	inner_blank.export("inner_blank.stl")



