import os
import torch
import numpy as np
import open3d as o3d

#target_id = "ue74f81c5-c851-4089-8dc0-e9944b0d4044" #헤드셋
#target_id = "6f43bf97b213b888ca2ed12df13a916a" #병
#target_id = "a232eec747955695609e2d916fa0da27" #소파
#target_id="4beb536ea2909f40713decb1a0563b12"
target_id = "7bce489081dd3a03379f47575c295bee"
folder = "mask_outputs"

# affordance별 색
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
all_points = None
all_colors = None

for file in os.listdir(folder):

    if not file.endswith(".pt"):
        continue

    path = os.path.join(folder, file)

    data = torch.load(path)

    if data["shape_id"] != target_id:
        continue

    question = data["question"]

    # affordance 이름 찾기
    affordance_name = None

    for key in affordance_colors.keys():
        if key in question:
            affordance_name = key
            break

    if affordance_name is None:
        continue

    color = affordance_colors[affordance_name]

    points = data["points"].numpy()

    gt = data["gt"].squeeze().numpy()

    if all_points is None:
        all_points = points
        all_colors = np.ones((points.shape[0], 3)) * 0.7  # gray

    mask_idx = gt > 0

    all_colors[mask_idx] = color

    print(file, affordance_name)

# visualize
pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(all_points)
pcd.colors = o3d.utility.Vector3dVector(all_colors)

o3d.visualization.draw_geometries([pcd])