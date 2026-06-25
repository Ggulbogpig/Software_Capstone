import os
import torch
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from collections import deque

target_id = "6f43bf97b213b888ca2ed12df13a916a"
folder = "mask_outputs"

mesh_path = "D:/Software_Capstone/archive (1)/ShapeNetCore.v2/ShapeNetCore.v2/03593526/6f43bf97b213b888ca2ed12df13a916a/models/model_normalized.ply"

affordance_colors = {
    "grasp":       [1.0, 0.0, 0.0],
    "contain":     [0.0, 1.0, 0.0],
    "lift":        [0.0, 0.0, 1.0],
    "openable":    [1.0, 1.0, 0.0],
    "layable":     [1.0, 0.0, 1.0],
    "sittable":    [0.0, 1.0, 1.0],
    "support":     [1.0, 0.5, 0.0],
    "wrapGrasp":   [0.5, 0.0, 1.0],
    "pourable":    [0.5, 1.0, 0.0],
    "move":        [0.0, 0.5, 1.0],
    "display":     [1.0, 0.0, 0.5],
    "pushable":    [0.5, 0.5, 0.5],
    "pull":        [0.3, 0.1, 0.0],
    "listen":      [0.0, 0.0, 0.3],
    "wear":        [1.0, 0.8, 0.6],
    "press":       [0.2, 0.8, 0.8],
    "cut":         [0.8, 0.0, 0.0],
    "stab":        [0.3, 0.0, 0.0],
}

# -----------------------------
# Mesh load
# -----------------------------
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()

# vertex 수 부족하면 1~2 조절
mesh = mesh.subdivide_midpoint(number_of_iterations=2)
mesh.compute_vertex_normals()

vertices = np.asarray(mesh.vertices)
tree = cKDTree(vertices)

mesh.compute_adjacency_list()
adj = mesh.adjacency_list

print(mesh)

# -----------------------------
# collect seeds
# -----------------------------
affordance_seeds = {}

for file in os.listdir(folder):
    if not file.endswith(".pt"):
        continue

    data = torch.load(os.path.join(folder, file))

    if data["shape_id"] != target_id:
        continue

    question = data["question"]

    affordance_name = None
    for key in affordance_colors:
        if key in question:
            affordance_name = key
            break

    if affordance_name is None:
        continue

    points = data["points"].numpy()
    gt = data["gt"].squeeze().numpy()

    positive_points = points[gt > 0]

    if len(positive_points) == 0:
        continue

    _, seeds = tree.query(positive_points, k=1)
    seeds = np.unique(seeds)

    if affordance_name not in affordance_seeds:
        affordance_seeds[affordance_name] = set()

    affordance_seeds[affordance_name].update(seeds.tolist())

    print(file, "->", affordance_name, "seed:", len(seeds))

# -----------------------------
# simultaneous multi-source BFS
# -----------------------------
vertex_owner = np.array([None] * len(vertices), dtype=object)
vertex_depth = np.full(len(vertices), np.inf)

q = deque()

# 모든 affordance seed를 동시에 queue에 넣음
for aff, seeds in affordance_seeds.items():
    for s in seeds:
        if vertex_owner[s] is None:
            vertex_owner[s] = aff
            vertex_depth[s] = 0
            q.append((s, aff, 0))
        else:
            # 같은 vertex에 여러 affordance seed가 걸린 경우는 그대로 유지
            # 필요하면 여기서 conflict list를 따로 저장 가능
            pass

# propagation 최대 깊이
max_depth = 15

while q:
    v, aff, depth = q.popleft()

    if depth >= max_depth:
        continue

    for nb in adj[v]:
        next_depth = depth + 1

        # 아직 아무 affordance도 차지하지 않은 vertex만 확장
        if vertex_owner[nb] is None:
            vertex_owner[nb] = aff
            vertex_depth[nb] = next_depth
            q.append((nb, aff, next_depth))

        # 같은 depth에서 동시에 도달한 경우는 덮어쓰지 않음
        # 즉 순서/우선순위 없음
        else:
            continue

# -----------------------------
# color assign
# -----------------------------
vertex_colors = np.ones((len(vertices), 3)) * 0.7

for aff, color in affordance_colors.items():
    mask = vertex_owner == aff
    vertex_colors[mask] = color

mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

# -----------------------------
# visualize
# -----------------------------
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

o3d.visualization.draw_geometries(
    [mesh, axis],
    mesh_show_back_face=True
)