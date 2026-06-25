# import open3d as o3d

# pcd = o3d.io.read_point_cloud("sample0_pred.ply")

# o3d.visualization.draw_geometries([pcd])
import torch
import numpy as np

data = torch.load("mask_outputs/sample_1.pt")

pred = data["pred"].squeeze().numpy()

print(pred.min())
print(pred.max())
print(np.unique(pred[:50]))

gt = data["gt"].squeeze().numpy()

print(gt.min())
print(gt.max())
print(gt.sum())