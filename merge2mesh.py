# import open3d as o3d
# import numpy as np
# import torch
# from scipy.spatial import cKDTree

# # -----------------------------
# # 1. Mesh Load
# # -----------------------------
# mesh_path = "D:/Software_Capstone/archive (1)/ShapeNetCore.v2/ShapeNetCore.v2/03593526/6f43bf97b213b888ca2ed12df13a916a/models/model_normalized.ply"

# mesh = o3d.io.read_triangle_mesh(mesh_path)

# mesh.compute_vertex_normals()

# vertices = np.asarray(mesh.vertices)

# print(mesh)

# # -----------------------------
# # 2. ADLLM pt load
# # -----------------------------
# data = torch.load("mask_outputs/batch_50_sample_0.pt")

# points = data["points"].numpy()

# labels = data["gt"].squeeze().numpy()

# print("Points:", points.shape)
# print("GT:", labels.shape)

# # -----------------------------
# # 3. KDTree mapping
# # -----------------------------
# tree = cKDTree(vertices)

# _, idx = tree.query(points)

# # -----------------------------
# # affordance colors
# # -----------------------------
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

# # -----------------------------
# # vertex color init
# # -----------------------------
# vertex_colors = np.ones((len(vertices), 3)) * 0.7

# # # -----------------------------
# # # 4. Vertex Coloring
# # # -----------------------------
# # colors = np.zeros((len(vertices), 3))

# # # 기본 회색
# # colors[:] = [0.7, 0.7, 0.7]

# # # affordance 영역 빨강
# # for i in range(len(points)):
# #     if labels[i] > 0.5:
# #         colors[idx[i]] = [1, 0, 0]

# # k = 7  # k-NN의 k 값

# # for i in range(len(points)):

# #     if labels[i] > 0.5:

# #         _, neighbors = tree.query(points[i], k=k)

# #         colors[neighbors] = [1, 0, 0]

# # mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
# # -----------------------------
# # affordance 이름 찾기
# # -----------------------------
# question = data["question"]

# affordance_name = None

# for key in affordance_colors.keys():

#     if key in question:
#         affordance_name = key
#         break

# # affordance 색 가져오기
# if affordance_name is not None:
#     current_color = affordance_colors[affordance_name]
# else:
#     current_color = [1, 0, 0]

# print("Affordance:", affordance_name)

# # -----------------------------
# # KNN propagation
# # -----------------------------
# k = 7  # k-NN의 k 값

# for i in range(len(points)):

#     if labels[i] > 0.5:

#         _, neighbors = tree.query(points[i], k=k)

#         vertex_colors[neighbors] = current_color

# # -----------------------------
# # Apply colors
# # -----------------------------
# mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)




# # -----------------------------
# # 5. Visualization
# # -----------------------------
# axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

# o3d.visualization.draw_geometries(
#     [mesh, axis],
#     mesh_show_back_face=True
# )

# print(data["question"])
# print(data["shape_id"])



import os
import torch
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# =====================================================
# target object
# =====================================================

# 헤드셋
# target_id = "ue74f81c5-c851-4089-8dc0-e9944b0d4044"

# 병
#target_id = "6f43bf97b213b888ca2ed12df13a916a"

#소파
#target_id = "a232eec747955695609e2d916fa0da27"

#헤드셋
target_id = "4beb536ea2909f40713decb1a0563b12"

# =====================================================
# mesh load
# =====================================================

mesh_path = "D:/Software_Capstone/archive (1)/ShapeNetCore.v2/ShapeNetCore.v2/03261776/4beb536ea2909f40713decb1a0563b12/models/model_normalized.ply"

mesh = o3d.io.read_triangle_mesh(mesh_path)

mesh.compute_vertex_normals()

vertices = np.asarray(mesh.vertices)

print(mesh)

# =====================================================
# KDTree
# =====================================================

tree = cKDTree(vertices)

# =====================================================
# affordance files folder
# =====================================================

folder = "mask_outputs"

# =====================================================
# affordance colors
# =====================================================

affordance_colors = {
    "grasp":       [1.0, 0.0, 0.0],   # red
    "contain":     [0.0, 1.0, 0.0],   # green
    "lift":        [0.0, 0.0, 1.0],   # blue
    "openable":    [1.0, 1.0, 0.0],   # yellow
    "layable":     [1.0, 0.0, 1.0],   # magenta
    "sittable":    [0.0, 1.0, 1.0],   # cyan
    "support":     [1.0, 0.5, 0.0],   # orange
    "wrapGrasp":   [0.5, 0.0, 1.0],   # purple
    "pourable":    [0.5, 1.0, 0.0],   # lime
    "move":        [0.0, 0.5, 1.0],   # sky blue
    "display":     [1.0, 0.0, 0.5],   # pink
    "pushable":    [0.5, 0.5, 0.5],   # gray
    "pull":        [0.3, 0.1, 0.0],   # brown
    "listen":      [0.0, 0.0, 0.3],   # dark blue
    "wear":        [1.0, 0.8, 0.6],   # skin tone
    "press":       [0.2, 0.8, 0.8],   # teal
    "cut":         [0.8, 0.0, 0.0],   # dark red
    "stab":        [0.3, 0.0, 0.0],   # deep red
}

# =====================================================
# mesh vertex colors init
# =====================================================

vertex_colors = np.ones((len(vertices), 3)) * 0.7

# =====================================================
# KNN size
# =====================================================

k = 20

# =====================================================
# process all affordance pt files
# =====================================================

for file in os.listdir(folder):

    if not file.endswith(".pt"):
        continue

    path = os.path.join(folder, file)

    data = torch.load(path)

    # 다른 object는 스킵
    if data["shape_id"] != target_id:
        continue

    question = data["question"]

    # -------------------------------------------------
    # affordance 이름 찾기
    # -------------------------------------------------

    affordance_name = None

    for key in affordance_colors.keys():

        if key in question:
            affordance_name = key
            break

    if affordance_name is None:
        continue

    current_color = affordance_colors[affordance_name]

    print(file, "->", affordance_name)

    # -------------------------------------------------
    # point / gt load
    # -------------------------------------------------

    points = data["points"].numpy()

    gt = data["gt"].squeeze().numpy()

    mask_idx = gt > 0

    target_points = points[mask_idx]

    # -------------------------------------------------
    # mesh projection
    # -------------------------------------------------

    for p in target_points:

        _, neighbors = tree.query(p, k=k)


        # 이미 색칠된 vertex는 유지
        for v in np.atleast_1d(neighbors):

            if np.all(vertex_colors[v] == [0.7, 0.7, 0.7]):

                vertex_colors[v] = current_color

# =====================================================
# apply colors
# =====================================================

mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

# =====================================================
# visualization
# =====================================================

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

o3d.visualization.draw_geometries(
    [mesh, axis],
    mesh_show_back_face=True
)