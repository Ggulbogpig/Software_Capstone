import trimesh

# mesh_path = "object/knife5.glb"
mesh_path = "object/office_chair.glb"

mesh = trimesh.load(mesh_path, force='mesh')

print("vertices:", len(mesh.vertices))
print("faces:", len(mesh.faces))