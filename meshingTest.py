import trimesh
import numpy as np
from trimesh.scene.cameras import Camera

# ---------------------------------------------------------------------------
# 1. Import blank.stl
# ---------------------------------------------------------------------------
original = trimesh.load("blank.stl")
if isinstance(original, trimesh.Scene):
    original = original.dump(concatenate=True)

print(f"Original mesh: {len(original.vertices)} vertices, {len(original.faces)} faces")
print(f"Original bounds: {original.bounds}")

# ---------------------------------------------------------------------------
# 2. Isotropic remesh with target edge length of 1.0 mm
# ---------------------------------------------------------------------------
try:
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=original.vertices, face_matrix=original.faces))
    ms.meshing_isotropic_explicit_remeshing(
        iterations=5,
        targetlen=pymeshlab.PureValue(1.0),
    )
    remeshed_mesh = ms.current_mesh()
    remeshed = trimesh.Trimesh(
        vertices=remeshed_mesh.vertex_matrix(),
        faces=remeshed_mesh.face_matrix(),
        process=False,
    )
except ImportError:
    print("pymeshlab not installed, falling back to trimesh subdivide_to_size")
    remeshed = original.subdivide_to_size(max_edge=1.0)

print(f"Remeshed mesh: {len(remeshed.vertices)} vertices, {len(remeshed.faces)} faces")
print(f"Remeshed bounds: {remeshed.bounds}")

# ---------------------------------------------------------------------------
# 3. Voxelize the remeshed object while preserving overall dimensions
# ---------------------------------------------------------------------------
voxel_pitch = 1.0  # mm, matches the remesh target length
voxel_grid = remeshed.voxelized(pitch=voxel_pitch)
voxel_grid = voxel_grid.fill()
voxelized_mesh = voxel_grid.as_boxes()

# Rescale the voxelized mesh so its bounding box matches the original exactly
orig_extents = original.extents
voxel_extents = voxelized_mesh.extents
scale_factors = orig_extents / voxel_extents
voxelized_mesh.apply_scale(scale_factors)

# Re-center voxelized mesh on the original's bounding box center
orig_center = original.bounds.mean(axis=0)
voxel_center = voxelized_mesh.bounds.mean(axis=0)
voxelized_mesh.apply_translation(orig_center - voxel_center)

print(f"Voxelized mesh: {len(voxelized_mesh.vertices)} vertices, {len(voxelized_mesh.faces)} faces")
print(f"Voxelized bounds: {voxelized_mesh.bounds}")

# Export each mesh in its original coordinate position.
original.export("blank_original.stl")
remeshed.export("blank_remeshed.stl")
voxelized_mesh.export("blank_voxelized.stl")





