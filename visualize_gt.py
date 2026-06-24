import torch
import numpy as np
import open3d as o3d

# -----------------------------
# load data
# -----------------------------
data = torch.load("mask_outputs/batch_159_sample_0.pt")

points = data["points"].numpy()

gt = data["gt"].squeeze().numpy()

# -----------------------------
# color mapping
# -----------------------------
colors = np.zeros((points.shape[0], 3))

# GT affordance 영역 = 초록
colors[gt > 0.5] = [0, 1, 0]

# non-affordance = 회색
colors[gt <= 0.5] = [0.7, 0.7, 0.7]

# -----------------------------
# create point cloud
# -----------------------------
pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# -----------------------------
# visualize
# -----------------------------
o3d.visualization.draw_geometries([pcd])

print("GT positive points:", gt.sum())
print(data["question"])
print(data["shape_id"])



# import os
# import torch

# target_id = "6f43bf97b213b888ca2ed12df13a916a"

# files = os.listdir("mask_outputs")

# matched = []

# for f in files:

#     path = os.path.join("mask_outputs", f)

#     data = torch.load(path)

#     if data["shape_id"] == target_id:

#         matched.append(f)

#         print(f)
#         print("question:", data["question"])
#         print("------")