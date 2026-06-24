

import trimesh
import argparse
import open3d as o3d
import numpy as np


import os
import trimesh
import open3d as o3d
import numpy as np


def load_mesh(mesh_path):

    ext = os.path.splitext(
        mesh_path
    )[1].lower()

    # OBJ / PLY / STL
    if ext in [
        ".obj",
        ".ply",
        ".stl"
    ]:

        mesh = o3d.io.read_triangle_mesh(
            mesh_path
        )

        mesh.compute_vertex_normals()

        return mesh

    # GLB / GLTF
    elif ext in [
        ".glb",
        ".gltf"
    ]:

        tm = trimesh.load(
            mesh_path
        )

        if isinstance(
            tm,
            trimesh.Scene
        ):

            tm = tm.to_geometry()

        mesh = o3d.geometry.TriangleMesh()

        mesh.vertices = (
            o3d.utility.Vector3dVector(
                np.asarray(
                    tm.vertices
                )
            )
        )

        mesh.triangles = (
            o3d.utility.Vector3iVector(
                np.asarray(
                    tm.faces
                )
            )
        )

        mesh.compute_vertex_normals()

        return mesh

    else:

        raise ValueError(
            f"Unsupported format: {ext}"
        )



# ==========================================
# ARGUMENTS
# ==========================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mesh",
    required=True
)

parser.add_argument(
    "--output",
    required=True
)

parser.add_argument(
    "--points",
    type=int,
    default=10000
)

parser.add_argument(
    "--method",
    choices=["uniform", "poisson"],
    default="poisson"
)

args = parser.parse_args()

####################################################

#mesh = o3d.io.read_triangle_mesh(args.mesh)
mesh = load_mesh(
    args.mesh
)

mesh.compute_vertex_normals()

o3d.visualization.draw_geometries(
    [mesh],
    mesh_show_wireframe=True,
    mesh_show_back_face=True
)



# ==========================================
# LOAD MESH
# ==========================================

print("Loading mesh...")

# mesh = o3d.io.read_triangle_mesh(
#     args.mesh
# )
mesh = load_mesh(
    args.mesh
)

mesh.compute_vertex_normals()

print(
    "vertices:",
    np.asarray(mesh.vertices).shape
)

print(
    "triangles:",
    np.asarray(mesh.triangles).shape
)

# ==========================================
# SAMPLE POINTS
# ==========================================

print(
    f"Sampling {args.points} points..."
)

if args.method == "uniform":

    pcd = mesh.sample_points_uniformly(
        number_of_points=args.points
    )

else:

    pcd = mesh.sample_points_poisson_disk(
        number_of_points=args.points
    )

# ==========================================
# SAVE
# ==========================================

success = o3d.io.write_point_cloud(
    args.output,
    pcd
)

print(
    "saved:",
    success
)

print(
    "output:",
    args.output
)


# ==========================================
# SAVE NPY
# ==========================================

points = np.asarray(
    pcd.points,
    dtype=np.float32
)

npy_path = (
    args.output
    .replace(".ply", ".npy")
)

np.save(
    npy_path,
    points
)

print(
    "saved npy:",
    npy_path
)

# ==========================================
# INFO
# ==========================================

points = np.asarray(
    pcd.points
)

print(
    "point cloud:",
    points.shape
)

print(
    "min:",
    points.min(axis=0)
)

print(
    "max:",
    points.max(axis=0)
)

# ==========================================
# VISUALIZE
# ==========================================

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.2
)

o3d.visualization.draw_geometries(
    [pcd, axis]
)

