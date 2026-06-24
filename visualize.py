import torch
import open3d as o3d
import numpy as np

data = torch.load("mask_outputs/sample_0.pt")

xyz = data["points"].numpy()

pred = data["pred"].squeeze().numpy()

colors = np.zeros((xyz.shape[0], 3))

# affordance 영역 빨간색
colors[pred > 0.9] = [1, 0, 0]

# non-affordance 회색
colors[pred <= 0.5] = [0.7, 0.7, 0.7]

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(colors)

#o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("sample0_pred.ply", pcd)

print("saved sample0_pred.ply")