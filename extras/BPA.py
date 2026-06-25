# import torch
# import numpy as np
# import open3d as o3d

# # pt 파일 로드
# data = torch.load("mask_outputs/batch_0_sample_0.pt")

# points = data["points"].numpy()

# # Open3D point cloud
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)

# # normal estimation (BPA 필수)
# pcd.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamHybrid(
#         radius=0.05,
#         max_nn=30
#     )
# )

# # BPA radius
# distances = pcd.compute_nearest_neighbor_distance()
# avg_dist = np.mean(distances)

# radius = avg_dist * 3

# # BPA mesh reconstruction
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     pcd,
#     o3d.utility.DoubleVector([radius, radius * 2])
# )

# # mesh normal
# mesh.compute_vertex_normals()

# # visualize
# o3d.visualization.draw_geometries([mesh])

# # 저장
# o3d.io.write_triangle_mesh("bpa_mesh.ply", mesh)

# print(mesh)
######################################################

import torch
import numpy as np
import open3d as o3d

# load
data = torch.load("mask_outputs/batch_50_sample_0.pt")
gt = data["gt"].squeeze().numpy()
points = data["points"].numpy()

# point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# normal estimation
pcd.estimate_normals()

# alpha shape
alpha = 0.09

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    pcd,
    alpha
)

mesh.compute_vertex_normals()

# visualize
o3d.visualization.draw_geometries([mesh])

# save
o3d.io.write_triangle_mesh("alpha_mesh.ply", mesh)

print(mesh)


# ####
mesh.compute_vertex_normals()

# =========================
# semantic mapping
# =========================

import numpy as np
from scipy.spatial import cKDTree

orig_points = points

gt_mask = gt > 0

vertices = np.asarray(mesh.vertices)

tree = cKDTree(orig_points)

dist, idx = tree.query(vertices)

vertex_labels = gt_mask[idx]

triangles = np.asarray(mesh.triangles)

face_labels = []

for tri in triangles:

    v_labels = vertex_labels[tri]

    label = int(np.sum(v_labels) >= 2)

    face_labels.append(label)

face_labels = np.array(face_labels)

print("semantic faces:", np.sum(face_labels))




vertex_colors = np.zeros((len(vertices), 3))

for i, label in enumerate(vertex_labels):

    if label:
        vertex_colors[i] = [1,0,0]
    else:
        vertex_colors[i] = [0.7,0.7,0.7]

mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

o3d.visualization.draw_geometries(
    [mesh],
    mesh_show_wireframe=True,
    mesh_show_back_face=True
)





