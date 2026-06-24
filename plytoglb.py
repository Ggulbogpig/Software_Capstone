import open3d as o3d
import trimesh
import numpy as np

# ply 읽기
mesh_o3d = o3d.io.read_triangle_mesh(
    #"colored_mesh_bag.ply"
    #"motor_affordance_colored.ply"
    #"cup_simple_affordance_colored.ply"
    "mug2_point_affordance_colored.ply"
)

# Open3D -> numpy
vertices = np.asarray(mesh_o3d.vertices)
faces = np.asarray(mesh_o3d.triangles)

# color 있으면 같이
if mesh_o3d.has_vertex_colors():
    colors = np.asarray(mesh_o3d.vertex_colors)
else:
    colors = None

# trimesh 생성
mesh_tm = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    vertex_colors=colors,
    process=False
)

# glb 저장
mesh_tm.export("mug2_affordance_colored.glb")

print("saved!")