# import numpy as np
# import matplotlib.pyplot as plt

# data = np.load("Knife_1be1aa66cd8674184e09ebaf49b0cb2f_grasp.npy")
# points = data[:, :3]

# fig = plt.figure(figsize=(15, 5))

# # 세 방향에서 보기
# for idx, (elev, azim, title) in enumerate([(30, 45, 'Perspective'), (90, 0, 'Top'), (0, 0, 'Front')]):
#     ax = fig.add_subplot(1, 3, idx+1, projection='3d')
#     ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='blue')
#     ax.view_init(elev=elev, azim=azim)
#     ax.set_title(title)

# plt.savefig('pointcloud_raw.png', dpi=150, bbox_inches='tight')
# print("저장됨!")
# print("data shape:", data.shape)
# print("columns:", data.shape[1])
import open3d as o3d

# 파일 경로 수정
point_path = "mug8848_point_affordance_colored.ply"
#point_path = "object/cup5.ply"
# 포인트 클라우드 읽기
pcd = o3d.io.read_point_cloud(point_path)

print(pcd)
print("points:", len(pcd.points))
print("has colors:", pcd.has_colors())

# 시각화
o3d.visualization.draw_geometries(
    [pcd],
    window_name="Mug8568 Point Cloud",
    width=1000,
    height=800
)