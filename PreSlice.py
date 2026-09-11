import importlib
import os
import subprocess
import sys
import threading
import time
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


def run_with_progress(operation, label, width=30):
	"""Run a blocking operation while showing an indeterminate terminal progress bar."""
	result = []
	error = []
	finished = threading.Event()

	def worker():
		try:
			result.append(operation())
		except BaseException as exc:
			error.append(exc)
		finally:
			finished.set()

	worker_thread = threading.Thread(target=worker)
	worker_thread.start()
	position = 0
	while not finished.wait(0.1):
		bar = ["-"] * width
		bar[position % width] = "="
		sys.stdout.write(f"\r{label} [{''.join(bar)}]")
		sys.stdout.flush()
		position += 1

	worker_thread.join()
	sys.stdout.write("\r" + (" " * (len(label) + width + 3)) + "\r")
	sys.stdout.flush()
	if error:
		raise error[0]
	return result[0]


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
	return scaled


def extract_top_and_bottom_surfaces(mesh, tolerance_mm=0.0):
	"""Extract faces lying exactly on the mesh's top and bottom Z planes."""
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
	"""Resample a closed loop at equal angular steps about its own XY centroid.

	Parameterizing by angle (instead of arc length) keeps corresponding
	indices of the top and bottom loops radially aligned and consistently
	wound, which is required to avoid twisted/self-intersecting side facets.
	"""
	center_xy = points[:, :2].mean(axis=0)
	angles = np.arctan2(points[:, 1] - center_xy[1], points[:, 0] - center_xy[0])
	order = np.argsort(angles)
	sorted_points = points[order]
	sorted_angles = angles[order]

	closed_points = np.vstack([sorted_points, sorted_points[0]])
	closed_angles = np.concatenate([sorted_angles, [sorted_angles[0] + 2.0 * np.pi]])
	sample_angles = np.linspace(
		closed_angles[0],
		closed_angles[0] + 2.0 * np.pi,
		sample_count,
		endpoint=False,
	)
	return np.column_stack([
		np.interp(sample_angles, closed_angles, closed_points[:, axis])
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
	"""Close a boundary loft with planar caps for boolean operations."""
	top_ring = loft.vertices[:sample_count]
	bottom_ring = loft.vertices[sample_count:2 * sample_count]
	vertices = np.vstack([
		loft.vertices,
		top_ring.mean(axis=0),
		bottom_ring.mean(axis=0),
	])
	top_center = len(loft.vertices)
	bottom_center = top_center + 1
	faces = loft.faces.tolist()
	for index in range(sample_count):
		next_index = (index + 1) % sample_count
		faces.extend([
			[top_center, next_index, index],
			[bottom_center, sample_count + index, sample_count + next_index],
		])
	capped = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
	capped.remove_unreferenced_vertices()
	capped.fix_normals()
	return capped


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
	# ms.save_current_mesh(finemesh_path)
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

if __name__ == "__main__":

	while True:
		part_type = input('Enter "A" for Asymmetric or "S" for Symmetric: ').strip().upper()
		if part_type in {"A", "S"}:
			break
		print('Invalid input. Please enter "A" or "S".')

    # 1. Upload Path and Blank Mesh Files
	input_dir = os.path.join(os.getcwd(), "input")
	output_dir = os.path.join(os.getcwd(), "output")
	os.makedirs(output_dir, exist_ok=True)
	path_mesh = trimesh.load(os.path.join(input_dir, "path.stl"))
	blank_mesh = trimesh.load(os.path.join(input_dir, "blank.stl"))
	print("[1] Imported path and blank files")

    # 2. Plane cut 10 mm from the bottom of the path and blank meshes
	if part_type == "A":
		plane_cut_height_mm = 10.0
		path_mesh = plane_cut_from_bottom(path_mesh, height_mm=plane_cut_height_mm)
		blank_mesh = plane_cut_from_bottom(blank_mesh, height_mm=plane_cut_height_mm)

		# show_mesh(path_mesh, "Path Mesh (Plane Cut)", color="cornflowerblue")
		# show_mesh(blank_mesh, "Blank Mesh (Plane Cut)", color="lightgray")
		print("[2] Reduced input path and blank by 10 mm")
	else:
		print("[2] Skipped 10 mm plane cut for symmetric part")

    # 3 - Remesh path and blank meshes
	path_fine = run_with_progress(
		lambda: isotropic_remesh(path_mesh, os.path.join(output_dir, "path_fine.stl"), 0.5),
		"Remeshing path",
	)
	blank_fine = run_with_progress(
		lambda: isotropic_remesh(blank_mesh, os.path.join(output_dir, "blank_fine.stl"), 1),
		"Remeshing blank",
	)

	show_mesh(path_fine, "Path Mesh", color="cornflowerblue")
	show_mesh(blank_fine, "Blank Mesh", color="lightgray")
	print("[3] Remeshed path and blank mesh files")

	# 4 - Boolean Difference (path - blank = envelop)
	envelop_mesh = robust_boolean_difference(blank_fine, path_fine)
	if envelop_mesh is None or len(envelop_mesh.vertices) == 0:
		raise RuntimeError("Boolean difference produced an empty result mesh.")
	envelop_mesh = cleanup_mesh(envelop_mesh)
	show_mesh(envelop_mesh, "Envelop Mesh (Path - Blank)", color="lightgreen")
	envelop_mesh.export(os.path.join(output_dir, "envelop.stl"))
	print("[4] Generated envelop mesh (path - blank = envelop)")

	# 5. Generate boundary loft and interactively select its scaled XY hull
	# tolerance accounts for remeshing/boolean noise so top/bottom faces aren't perfectly flat
	top_surface, bottom_surface = extract_top_and_bottom_surfaces(
		envelop_mesh,
		tolerance_mm=0.05,
	)
	show_meshes_overlay(
		[
			(top_surface, "cornflowerblue", 0.8),
			(bottom_surface, "lightgreen", 0.8),
		],
		"Envelop Top and Bottom Surfaces",
	)

	boundary_loft = loft_between_boundary_loops(
		top_surface,
		bottom_surface,
		sample_count=128,
	)
	show_mesh(boundary_loft, "Boundary Loop Surface", color="gold")

	boundary_volume = cap_boundary_loft(boundary_loft, sample_count=128)
	show_mesh(boundary_volume, "Capped Boundary Volume", color="darkgoldenrod")

	erosion_offset_mm = 2.0
	while True:
		eroded_hull = scale_hull_xy(boundary_volume, offset=erosion_offset_mm)
		show_meshes_overlay(
			[
				(path_fine, "lightblue", 0.30),
				(boundary_volume, "gold", 0.30),
				(eroded_hull, "darkorange", 0.45),
			],
			f"Boundary Hull and Eroded Hull ({erosion_offset_mm:.2f} mm)",
		)

		while True:
			update_offset = input(
				f"Erosion offset is {erosion_offset_mm:.2f} mm. Update? (Y/N): "
			).strip().upper()
			if update_offset in {"Y", "N"}:
				break
			print('Invalid input. Please enter "Y" or "N".')

		if update_offset == "N":
			break

		while True:
			try:
				new_offset = float(input("Enter a positive erosion offset in mm: ").strip())
			except ValueError:
				print("Invalid offset. Enter a positive number.")
				continue
			if new_offset > 0:
				erosion_offset_mm = new_offset
				break
			print("Invalid offset. Enter a positive number.")

	boundary_loft.export(os.path.join(output_dir, "boundary_loft.stl"))
	eroded_hull.export(os.path.join(output_dir, "hull_eroded.stl"))
	print("[5] Generated boundary hull and selected eroded hull")

	# 6. Boolean Subtraction of Path - Eroded Hull
	eroded_difference_result = robust_boolean_difference(path_fine, eroded_hull)
	show_mesh(
		eroded_difference_result,
		"Path Minus Boundary Eroded Hull",
		color="lightgreen",
	)
	eroded_difference_result.export(
		os.path.join(output_dir, "path_fine_minus_eroded_hull.stl")
	)
	print("[6] Subtracted boundary-based eroded hull from remeshed path file")

	# 7. Boolean Intersection of Path and Eroded Hull
	eroded_intersection_result = robust_boolean_intersection(
		path_fine,
		eroded_hull,
	)
	show_mesh(
		eroded_intersection_result,
		"Implant Core from Boundary Eroded Hull",
		color="darkorange",
	)
	eroded_intersection_result.export(
		os.path.join(output_dir, "implant_core.stl")
	)
	print("[7] Determined the intersection geometry using the boundary-based eroded hull")

	# 8. Isotropic Remesh of Implant Core
	remeshed_intersection_result = remesh_uniform(eroded_intersection_result, target_len=0.20)

	# Display the remeshed mesh overlaid on the original path_fine mesh
	show_meshes_overlay(
		[
			(path_fine, "lightblue", 0.3),
			(remeshed_intersection_result, "darkorange", 1.0),
		],
		"Remeshed Implant Core over Path",
	)

	remeshed_intersection_output_path = os.path.join(output_dir, "implant_core_remeshed.stl")
	# remeshed_intersection_result.export(remeshed_intersection_output_path)
	print("[8] Remeshed the intersection result or the implant/target mesh")

	# 9. Voxel Conversion of Implant Core
	voxel_dilated_result = voxel_dilate_mesh(remeshed_intersection_result, offset_mm=1.00)

	#voxel_dilated_output_path = "implant_core_dilated.stl"
	#voxel_dilated_result.export(voxel_dilated_output_path)
	print("[9] Voxelized and dilated implant/target mesh")

	# 10. Convert Voxelized Volume back to Mesh
	mesh_from_voxel_result = voxel_to_mesh(voxel_dilated_result)

	# Display the voxelized mesh overlaid on the original path_fine mesh
	#show_meshes_overlay(
	#	[
	#		(path_fine, "lightblue", 0.3),
	#		(mesh_from_voxel_result, "darkorange", 1.0),
	#	],
	#	"Mesh Implant Core over Path",
	#)

	print("[10] Converted dilated voxelized implant/target to mesh")

	# 11. Smooth Mesh
	smooth_mesh_result = filter_laplacian(mesh_from_voxel_result, iterations=20)

	# Display the smoothed mesh overlaid on the original path_fine mesh
	show_meshes_overlay(
		[
			(path_fine, "lightblue", 0.3),
			(smooth_mesh_result, "darkorange", 1.0),
		],
		"Smooth Implant Core over Path",
	)

	smooth_mesh_output_path = os.path.join(output_dir, "implant_core_smooth.stl")
	# smooth_mesh_result.export(smooth_mesh_output_path)
	print("[11] Laplacian smoothing of implant/target mesh")

	# 11b. Clip dilated implant envelop to the blank so it cannot exceed the blank's height
	implant_envelop_mesh = robust_boolean_intersection(smooth_mesh_result, blank_fine)

	# 12. Final Boolean
	outer_puck = blank_fine.difference(envelop_mesh, engine="manifold")
	outer_puck = outer_puck.difference(implant_envelop_mesh, engine="manifold")
	inner_puck = envelop_mesh.difference(implant_envelop_mesh, engine="manifold")

	show_meshes_overlay(
			[
				(outer_puck, "lightblue", 0.5),
				(inner_puck, "lightgreen", 0.2),
				(implant_envelop_mesh, "darkorange", 1.0),
			],
			"Final",
	)
	print("[12] Generated outer and inner puck mesh files from blank")

	outer_puck_output_path = os.path.join(output_dir, "outer_puck.stl")
	inner_puck_output_path = os.path.join(output_dir, "inner_puck.stl")
	implant_envelop_output_path = os.path.join(output_dir, "implant_envelop.stl")
	outer_puck.export(outer_puck_output_path)
	inner_puck.export(inner_puck_output_path)
	implant_envelop_mesh.export(implant_envelop_output_path)

