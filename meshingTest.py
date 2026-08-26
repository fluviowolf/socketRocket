import pymeshlab
import trimesh


def meshset_from_trimesh(trimesh_mesh):
    """Create a PyMeshLab MeshSet from a trimesh.Trimesh object."""
    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=trimesh_mesh.vertices,
            face_matrix=trimesh_mesh.faces,
        )
    )
    return ms


def remesh(input_path, output_path, target_length=1):
    """Remesh a mesh and return the result as a trimesh.Trimesh object."""
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_path)

    ms.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PureValue(target_length)
    )

    ms.save_current_mesh(output_path)

    remeshed_mesh = ms.current_mesh()
    return trimesh.Trimesh(
        vertices=remeshed_mesh.vertex_matrix(),
        faces=remeshed_mesh.face_matrix(),
        process=False,
    )


if __name__ == "__main__":
    remesh("blank.stl", "blank_remeshed.stl")



