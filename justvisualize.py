import open3d as o3d
import numpy as np
import torch
import os

# =====================================================
# target object
# =====================================================

#target_id = "6f43bf97b213b888ca2ed12df13a916a"
target_id ="4beb536ea2909f40713decb1a0563b12"
folder = "mask_outputs"

mesh_path = "D:/Software_Capstone/archive (1)/ShapeNetCore.v2/ShapeNetCore.v2/03261776/" + target_id + "/models/model_normalized.ply"

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
# mesh load
# =====================================================

mesh = o3d.io.read_triangle_mesh(mesh_path)
#mesh.compute_vertex_normals()

# # =====================================================
# # normalize mesh
# # =====================================================

vertices = np.asarray(mesh.vertices)

center = vertices.mean(axis=0)

vertices = vertices - center

scale = np.max(np.linalg.norm(vertices, axis=1))

vertices = vertices / scale

mesh.vertices = o3d.utility.Vector3dVector(vertices)
# =====================================================
# manual translation
# =====================================================

#vertices[:, 0] *= 0.99
mesh.vertices = o3d.utility.Vector3dVector(vertices)
R = mesh.get_rotation_matrix_from_xyz(
    (0, np.pi, 0)
)

mesh.rotate(R, center=(0,0,0))

# 회색 mesh
mesh.paint_uniform_color([0.8, 0.8, 0.8])

# =====================================================
# manual rotation only
# =====================================================


# =====================================================
# point clouds
# =====================================================

pcd_list = []

for file in os.listdir(folder):

    if not file.endswith(".pt"):
        continue

    path = os.path.join(folder, file)

    data = torch.load(path)

    if data["shape_id"] != target_id:
        continue

    question = data["question"]

    affordance_name = None

    for key in affordance_colors.keys():

        if key in question:
            affordance_name = key
            break

    if affordance_name is None:
        continue

    print(file, "->", affordance_name)

    color = affordance_colors[affordance_name]

    points = data["points"].numpy()
    gt = data["gt"].squeeze().numpy()

    # positive affordance points만
    aff_points = points[gt > 0]

    # point cloud 생성
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(aff_points)

    colors = np.tile(color, (len(aff_points), 1))

    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd_list.append(pcd)

# =====================================================
# axis
# =====================================================

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.2
)

# =====================================================
# visualization
# =====================================================

o3d.visualization.draw_geometries(
    [mesh, axis] + pcd_list,
    mesh_show_back_face=True
)