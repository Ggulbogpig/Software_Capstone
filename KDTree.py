# import os
# import torch
# import numpy as np
# import open3d as o3d
# from scipy.spatial import cKDTree

# # =====================================================
# # target object
# # =====================================================

# #target_id = "6f43bf97b213b888ca2ed12df13a916a" #병03593526
# #target_id ="4beb536ea2909f40713decb1a0563b12" #헤드셋03261776
# #target_id = "a232eec747955695609e2d916fa0da27" #소파
# #target_id = "f3a7f8198cc50c225f5e789acd4d1122" #컵 - 손잡이 달린 03797390
# #target_id = "7328b4c001a249ca39d3717288022c20" #손잡이 달린 캐비닛 - 02933112
# target_id = "7328b4c001a249ca39d3717288022c20" #손잡이 달린 캐비닛 - 02933112
# target_id = "236f75a784ed58d62b3e9b6eb52d35df" #손잡이 있는 의자 - 03001627
# target_id = "71ead7f072106c63ed13f430b2941481" #손잡이 있는 가방 - 02773838
# folder = "mask_outputs"

# mesh_path = "D:/Software_Capstone/archive (1)/ShapeNetCore.v2/ShapeNetCore.v2/02773838/" + target_id + "/models/model_normalized.ply"
# #mesh_path = "motor.glb"
# # =====================================================
# # affordance colors
# # =====================================================

# # affordance별 색
# affordance_colors = {
#     "grasp":       [1.0, 0.0, 0.0],   # red
#     "contain":     [0.0, 1.0, 0.0],   # green
#     "lift":        [0.0, 0.0, 1.0],   # blue
#     "openable":    [1.0, 1.0, 0.0],   # yellow
#     "layable":     [1.0, 0.0, 1.0],   # magenta
#     "sittable":    [0.0, 1.0, 1.0],   # cyan
#     "support":     [1.0, 0.5, 0.0],   # orange
#     "wrapGrasp":   [0.5, 0.0, 1.0],   # purple
#     "pourable":    [0.5, 1.0, 0.0],   # lime
#     "move":        [0.0, 0.5, 1.0],   # sky blue
#     "display":     [1.0, 0.0, 0.5],   # pink
#     "pushable":    [0.5, 0.5, 0.5],   # gray
#     "pull":        [0.3, 0.1, 0.0],   # brown
#     "listen":      [0.0, 0.0, 0.3],   # dark blue
#     "wear":        [1.0, 0.8, 0.6],   # skin tone
#     "press":       [0.2, 0.8, 0.8],   # teal
#     "cut":         [0.8, 0.0, 0.0],   # dark red
#     "stab":        [0.3, 0.0, 0.0],   # deep red
# }

# # =====================================================
# # mesh load
# # =====================================================

# mesh = o3d.io.read_triangle_mesh(mesh_path)

# mesh.compute_vertex_normals()
# ##############################################

# vertices = np.asarray(mesh.vertices)

# center = vertices.mean(axis=0)

# vertices = vertices - center

# scale = np.max(np.linalg.norm(vertices, axis=1))

# vertices = vertices / scale

# mesh.vertices = o3d.utility.Vector3dVector(vertices)
# # =====================================================
# # manual translation
# # =====================================================

# #vertices[:, 0] *= 0.99
# mesh.vertices = o3d.utility.Vector3dVector(vertices)
# R = mesh.get_rotation_matrix_from_xyz(
#     (0, np.pi, 0)
# )

# mesh.rotate(R, center=(0,0,0))

# # =====================================================
# # IMPORTANT
# # normalize mesh to point cloud coordinate
# # =====================================================

# vertices = np.asarray(mesh.vertices)

# center = vertices.mean(axis=0)

# vertices = vertices - center

# scale = np.max(np.linalg.norm(vertices, axis=1))

# vertices = vertices / scale

# mesh.vertices = o3d.utility.Vector3dVector(vertices)

# # subdivision optional
# mesh = mesh.subdivide_midpoint(number_of_iterations=1)

# mesh.compute_vertex_normals()

# vertices = np.asarray(mesh.vertices)

# print(mesh)

# # =====================================================
# # KDTree
# # =====================================================

# tree = cKDTree(vertices)

# # =====================================================
# # vertex color init
# # =====================================================

# vertex_colors = np.ones((len(vertices), 3)) * 0.7

# # =====================================================
# # process all affordance pt
# # =====================================================

# k = 15

# for file in os.listdir(folder):

#     if not file.endswith(".pt"):
#         continue

#     path = os.path.join(folder, file)

#     data = torch.load(path)

#     if data["shape_id"] != target_id:
#         continue

#     question = data["question"]

#     affordance_name = None

#     for key in affordance_colors.keys():

#         if key in question:
#             affordance_name = key
#             break

#     if affordance_name is None:
#         continue

#     print(file, "->", affordance_name)

#     current_color = affordance_colors[affordance_name]

#     points = data["points"].numpy()
#     #수정#######################
#     # pcd = o3d.io.read_point_cloud(
#     # "motor_affordance_colored.ply"
#     # )

#     # points = np.asarray(
#     #     pcd.points
#     # )

#     # colors = np.asarray(
#     #     pcd.colors
#     # )
# #######################3
#     labels = data["gt"].squeeze().numpy()

#     target_points = points[labels > 0]

#     # =====================================================
#     # point -> nearest mesh vertex
#     # =====================================================

#     for p in target_points:

#         _, neighbors = tree.query(p, k=k)

#         vertex_colors[neighbors] = current_color

# # =====================================================
# # apply colors
# # =====================================================

# mesh.vertex_colors = o3d.utility.Vector3dVector(
#     vertex_colors
# )

# # =====================================================
# # visualization
# # =====================================================
# mesh.compute_vertex_normals()
# mesh.compute_triangle_normals()

# lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
# lines.paint_uniform_color([0, 0, 0])  # black



# axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
#     size=0.2
# )





# o3d.visualization.draw_geometries(
#     [mesh, lines,axis],
#     mesh_show_back_face=True,
#     mesh_show_wireframe=True
# )

# print(np.asarray(mesh.vertex_colors).shape)
# o3d.io.write_triangle_mesh(
#     "colored_mesh_bag.ply",
#     mesh
# )

# print("colored mesh saved!")






import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import trimesh


affordance_colors = {
    "handle_grasp": [1.0, 0.0, 0.0],   # Red (빨강)

    "none":        [0.8, 0.8, 0.8],   # Light Gray (연회색)

    "grasp":       [1.0, 0.0, 0.0],   # Red (빨강)
    "contain":     [0.0, 1.0, 0.0],   # Green (초록)
    "lift":        [0.0, 0.0, 1.0],   # Blue (파랑)
    "openable":    [1.0, 1.0, 0.0],   # Yellow (노랑)

    "support":     [1.0, 0.5, 0.0],   # Orange (주황)
    "wrap_grasp":  [0.6, 0.0, 1.0],   # Purple (보라)

    "pourable":    [0.0, 1.0, 1.0],   # Cyan (시안 - 하늘)

    "pushable":    [0.4, 0.4, 0.4],   # Gray (회색)
    "pull":        [0.5, 0.25, 0.0],  # Brown (갈색)

    "listen":      [0.0, 0.0, 0.5],   # Navy (남색)

    "wear":        [1.0, 0.4, 0.7],   # Pink (분홍)

    "press":       [0.4, 0.8, 1.0],   # Sky Blue (하늘색)

    "cut":         [0.0, 0.0, 0.0],   # Black (검정)

    "stab":        [1.0, 1.0, 1.0],   # White (흰색)

    "layable":   [1.0, 0.0, 1.0],   # Magenta (자홍)
    "sittable":  [0.0, 0.8, 0.8],   # Teal Cyan (청록)

    "move":      [0.0, 0.6, 1.0],   # Deep Sky Blue (진한 하늘색)
    "display":   [1.0, 0.0, 0.5],   # Hot Pink (진분홍)
}

# =====================================================
# FILES
# =====================================================

# PLY_FILE = "motor_affordance_colored.ply"
# MESH_FILE = "motor.glb"

# OUTPUT_FILE = "motor_affordance_mesh.glb"

# PLY_FILE = "microwave3465_point_affordance_colored.ply"
# MESH_FILE = "D:/Software_Capstone/EmpartACD/empart/Microwave/Microwave/models/3465.obj"

# OUTPUT_FILE = "microwave3465_affordance_mesh.glb"

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--ply")
parser.add_argument("--mesh")
parser.add_argument("--glb")

parser.add_argument("--point_csv")
parser.add_argument("--face_csv")

args = parser.parse_args()

PLY_FILE = args.ply
MESH_FILE = args.mesh
OUTPUT_FILE = args.glb

# =====================================================
# LOAD POINT CLOUD
# =====================================================

print("Loading point cloud...")

pcd = o3d.io.read_point_cloud(
    PLY_FILE
)

points = np.asarray(
    pcd.points
)

colors = np.asarray(
    pcd.colors
)

print("points:", points.shape)
print("colors:", colors.shape)

# =====================================================
# LOAD MESH
# =====================================================

print("Loading mesh...")



mesh = o3d.io.read_triangle_mesh(
    MESH_FILE
)

mesh.compute_vertex_normals()

vertices = np.asarray(
    mesh.vertices
)

print("vertices:", vertices.shape)

# =====================================================
# NORMALIZE MESH
# (same as point cloud generation)
# =====================================================

# center = vertices.mean(axis=0)

# vertices = vertices - center

# scale = np.max(
#     np.linalg.norm(
#         vertices,
#         axis=1
#     )
# )

# vertices = vertices / scale

# mesh.vertices = o3d.utility.Vector3dVector(
#     vertices
# )

# vertices = np.asarray(
#     mesh.vertices
# )

# =====================================================
# KDTree
# =====================================================

print("Building KDTree...")

tree = cKDTree(
    points
)

# =====================================================
# VERTEX -> NEAREST POINT
# =====================================================

print("Matching vertices...")

dist, idx = tree.query(
    vertices,
    k=1
)

vertex_colors = colors[idx]




from collections import Counter
import pandas as pd
# =====================================================
# LOAD POINT AFFORDANCE CSV
# =====================================================

# POINT_AFF_CSV = "microwave3465_point_affordance.csv"
# FACE_OUTPUT_CSV = "microwave3465_face_affordance.csv"
POINT_AFF_CSV = args.point_csv
FACE_OUTPUT_CSV = args.face_csv

df = pd.read_csv(POINT_AFF_CSV)

point_affordance = df["affordance"].astype(str).values

# idx: mesh vertex -> nearest point index
vertex_affordance = point_affordance[idx]

# =====================================================
# FACE MAJORITY VOTING
# =====================================================

triangles = np.asarray(mesh.triangles)

face_rows = []

for face_id, tri in enumerate(triangles):

    affs = [
        vertex_affordance[tri[0]],
        vertex_affordance[tri[1]],
        vertex_affordance[tri[2]],
    ]

    # none이 2개 이상이면 none, 아니면 다수결
    majority_aff = Counter(affs).most_common(1)[0][0]

    face_rows.append([
        face_id,
        int(tri[0]),
        int(tri[1]),
        int(tri[2]),
        majority_aff
    ])

face_df = pd.DataFrame(
    face_rows,
    columns=[
        "face_id",
        "v1",
        "v2",
        "v3",
        "affordance"
    ]
)

face_df.to_csv(
    FACE_OUTPUT_CSV,
    index=False
)

# ==========================
# FACE COLOR VISUALIZATION
# ==========================

triangles = np.asarray(mesh.triangles)

face_colors = []

for aff in face_df["affordance"]:

    face_colors.append(
        affordance_colors[aff]
    )

face_colors = np.asarray(face_colors)

mesh.triangle_material_ids = o3d.utility.IntVector(
    np.zeros(len(triangles), dtype=np.int32)
)


# affordance별 face 개수 확인
unique_affordances = np.unique(
    face_df["affordance"]
)

for aff in unique_affordances:
    print(
        aff,
        np.sum(
            face_df["affordance"] == aff
        )
    )


print("saved:", FACE_OUTPUT_CSV)
print(face_df.head())



print(
    "Average distance:",
    dist.mean()
)

print(
    "Max distance:",
    dist.max()
)

# =====================================================
# APPLY COLORS
# =====================================================

mesh.vertex_colors = (
    o3d.utility.Vector3dVector(
        vertex_colors
    )
)

# =====================================================
# VISUALIZE
# =====================================================

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.2
)

o3d.visualization.draw_geometries(
    [mesh, axis],
    mesh_show_back_face=True
)

o3d.visualization.draw_geometries(
    [mesh],
    mesh_show_wireframe=True
)
# =====================================================
# SAVE
# =====================================================

success = o3d.io.write_triangle_mesh(
    OUTPUT_FILE,
    mesh,
    write_vertex_colors=True
)
print("POINT")
print(points.min(axis=0))
print(points.max(axis=0))

print("MESH")
print(vertices.min(axis=0))
print(vertices.max(axis=0))

print("points:", len(points))
print("vertices:", len(vertices))

print("save:", success)
print("saved to:", OUTPUT_FILE)

