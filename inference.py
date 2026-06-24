# import torch
# import numpy as np
# from models import load_model_and_preprocess


# def pc_normalize(pc):
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
#     pc = pc / m
#     return pc


# if __name__ == "__main__":
#     print(torch.cuda.is_available())
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(device)
#     # load model
#     model = load_model_and_preprocess(
#         name="aff_phi",
#         model_type="phi",
#         is_eval=True,
#         device=device,
#     )
#     model.load_from_pretrained(
#         #"/workspace/project/Research_3D_Aff/Ckpts/Phi_Main/full_best.pth"
#         "dataset/ckpts/Phi_Main/full_best.pth"
#     )
#     # load data
#     file = "motor_points_20000.npy"
#     #file = "Knife_1be1aa66cd8674184e09ebaf49b0cb2f_grasp.npy"

#     #file = "D:\\Software_Capstone\\EmpartACD\\empart\\testing\\data\\models\\motor_points.npy"
#     data = np.load(file)
#     if data.shape[1] >= 3:
#         points = data[:, :3]
#     #instruction = "Please locate the areas of the Knife with function of grasp."
#     instruction = "Please locate the areas of the Motor with function of grasp. <SEG>"
    
#     input_points = torch.tensor(pc_normalize(points)).float().to(device)
#     #input_points = input_points.unsqueeze(0)
#     input = [input_points]

#     sample_affordance = {"question": instruction, "points": input}
#     output_aff = model.generate(sample_affordance, num_beams=1, max_length=30)

#     print(output_aff["text"])
#     print(output_aff["masks"])
#     print(output_aff)


#     print("masks_scores min:", output_aff["masks_scores"][0].min().item())
#     print("masks_scores max:", output_aff["masks_scores"][0].max().item())
#     print("masks_scores mean:", output_aff["masks_scores"][0].mean().item())
#     print("masks shape:", output_aff["masks"][0].shape)
#     print("masks sum (1의 개수):", output_aff["masks"][0].sum().item())



#     # # inference.py 마지막에 추가
#     # mask = output_aff["masks"][0].squeeze().cpu().numpy()
#     # scores = output_aff["masks_scores"][0].squeeze().cpu().float().numpy()

#     # np.save("all_points_with_mask_motor_10000.npy", 
#     #         np.concatenate([points, mask.reshape(-1,1), scores.reshape(-1,1)], axis=1))
#     # print("all_points_with_mask_motor_10000.npy 저장됨!")


#     # 마스크 첫 번째만 사용 (SEG 토큰이 여러 개면 첫 번째)
#     mask = output_aff["masks"][0][0].squeeze().cpu().numpy()    # [10000]
#     scores = output_aff["masks_scores"][0][0].squeeze().cpu().float().numpy()  # [10000]

#     print("mask shape:", mask.shape)
#     print("scores shape:", scores.shape)

#     np.save("all_points_with_mask_motor_20000.npy", 
#             np.concatenate([points, mask.reshape(-1,1), scores.reshape(-1,1)], axis=1))
#     print("all_points_with_mask_motor_20000.npy 저장됨!")






import torch
import numpy as np
import pandas as pd
import open3d as o3d

from collections import Counter
from models import load_model_and_preprocess

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

# =====================================================

# SETTINGS

# =====================================================

#POINT_FILE = "motor_points_10000.npy"
#POINT_FILE = "object/cup_points.npy"
#POINT_FILE = "Knife_1be1aa66cd8674184e09ebaf49b0cb2f_grasp.npy"
#POINT_FILE = "object/mug8555_points_shaped_2048.npy"
#POINT_FILE = "microwave3465_points_shaped_2048.npy"
#POINT_FILE = "object/mug8575.npy"
#POINT_FILE = "object/Table18196.npy"
#POINT_FILE = "object/Bag8532.npy"
#POINT_FILE = "object/Scissors11111.npy"

#POINT_FILE = "object/Storage39441.npy"
#POINT_FILE = "object/knife5.npy"
POINT_FILE = "object/gun.npy"
#POINT_FILE = "object/Bag9058.npy"
#POINT_FILE = "object/bag1.npy"
#POINT_FILE = "object/mug8848.npy"
#POINT_FILE = "object/office_chair.npy"
CHECKPOINT = (
"dataset/ckpts/Phi_Main/full_best.pth"
)

THRESHOLD = 0.5

# =====================================================

# AFFORDANCES

# =====================================================

# affordances = [
# "grasp",
# "contain",
# "lift",
# "openable",
# "support",
# "wrap_grasp",
# "pourable",
# "pushable",
# "pull",
# "listen",
# "wear",
# "press",
# "cut",
# "stab",
# ]
#affordances = ["grasp"]

affordances = [
"grasp",
"contain",
"lift",
"openable",
"layable",
"sittable",
"support",
"wrap_grasp",
"pourable",
"move",
"display",
"pushable",
"pull",
"listen",
"wear",
"press",
"cut",
"stab",
]

# =====================================================

# COLORS

# =====================================================

affordance_colors = {
    "handle_grasp": [1.0, 0.0, 0.0],   # Red (빨강)

    "none":        [0.8, 0.8, 0.8],   # Light Gray (연회색)

    "grasp":       [1.0, 0.0, 0.0],   # Red (빨강)
    "contain":     [0.0, 1.0, 0.0],   # Green (초록)
    "lift":        [0.0, 0.0, 1.0],   # Blue (파랑)
    "openable":    [1.0, 1.0, 0.0],   # Yellow (노랑)

    "support":     [1.0, 0.5, 0.0],   # Orange (주황)
    "wrap_grasp":  [0.6, 0.0, 1.0],   # Purple (보라)

    "pourable":    [0.0, 1.0, 1.0],   # Cyan (시안 - 하늘)

    "pushable":    [0.4, 0.4, 0.4],   # Gray (회색)
    "pull":        [0.5, 0.25, 0.0],  # Brown (갈색)

    "listen":      [0.0, 0.0, 0.5],   # Navy (남색)

    "wear":        [1.0, 0.4, 0.7],   # Pink (분홍)

    "press":       [0.4, 0.8, 1.0],   # Sky Blue (하늘색)

    "cut":         [0.0, 0.0, 0.0],   # Black (검정)

    "stab":        [1.0, 1.0, 1.0],   # White (흰색)

    "layable":   [1.0, 0.0, 1.0],   # Magenta (자홍)
    "sittable":  [0.0, 0.8, 0.8],   # Teal Cyan (청록)

    "move":      [0.0, 0.6, 1.0],   # Deep Sky Blue (진한 하늘색)
    "display":   [1.0, 0.0, 0.5],   # Hot Pink (진분홍)
}

# =====================================================

# MAIN

# =====================================================

if __name__ == "__main__":

    print(
        "CUDA:",
        torch.cuda.is_available()
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "DEVICE:",
        device
    )

# -------------------------------------------------
# MODEL
# -------------------------------------------------

model = load_model_and_preprocess(
    name="aff_phi",
    model_type="phi",
    is_eval=True,
    device=device,
)

model.load_from_pretrained(
    CHECKPOINT
)

print("MODEL LOADED")

# -------------------------------------------------
# POINT CLOUD
# -------------------------------------------------

data = np.load(
    POINT_FILE
)

points = data[:, :3]

print(
    "POINT SHAPE:",
    points.shape
)

input_points = torch.tensor(
    pc_normalize(points)
).float().to(device)

input_list = [input_points]

# -------------------------------------------------
# RUN ALL AFFORDANCES
# -------------------------------------------------

all_scores = []

for aff in affordances:

    print(
        "\n" + "=" * 60
    )

    print(
        "AFFORDANCE:",
        aff
    )

    instruction = (
        f"Please locate the areas "
        f"of the Chair with function "
        f"of {aff}. <SEG>"
        #"Please locate the handle of the object. <SEG>"
        #"I'm feeling a bit thirsty, could you pass me something I can grasp and lift up to drink water?"
    )

    sample_affordance = {
        "question": instruction,
        "points": input_list,
    }

    try:

        output_aff = model.generate(
            sample_affordance,
            num_beams=1,
            max_length=30,
        )

        print(
            output_aff["text"]
        )

        scores = (
            output_aff["masks_scores"][0][0]
            .squeeze()
            .cpu()
            .float()
            .numpy()
        )

        print(
            "min:",
            scores.min(),
            "max:",
            scores.max(),
            "mean:",
            scores.mean()
        )

    except Exception as e:

        print(
            "FAILED:",
            aff
        )

        print(e)

        scores = np.zeros(
            len(points),
            dtype=np.float32
        )

    all_scores.append(
        scores
    )

# -------------------------------------------------
# SCORE MATRIX
# -------------------------------------------------

all_scores = np.stack(
    all_scores,
    axis=1
)

print(
    "\nALL SCORE SHAPE:",
    all_scores.shape
)

print(
    "GLOBAL SCORE MIN:",
    np.min(all_scores)
)

print(
    "GLOBAL SCORE MAX:",
    np.max(all_scores)
)

print(
    "GLOBAL SCORE MEAN:",
    np.mean(all_scores)
)

# -------------------------------------------------
# ARGMAX
# -------------------------------------------------

best_aff_idx = np.argmax(
    all_scores,
    axis=1
)

best_score = np.max(
    all_scores,
    axis=1
)

best_aff_name = []

for idx, score in zip(
    best_aff_idx,
    best_score
):

    if score < THRESHOLD:

        best_aff_name.append(
            "none"
        )

    else:

        best_aff_name.append(
            affordances[idx]
        )

best_aff_name = np.array(
    best_aff_name
)

# -------------------------------------------------
# STATISTICS
# -------------------------------------------------

print(
    "\n===== AFFORDANCE COUNTS ====="
)

counter = Counter(
    best_aff_name
)

for k, v in sorted(
    counter.items()
):

    print(
        f"{k}: {v}"
    )

# -------------------------------------------------
# COLORS
# -------------------------------------------------

point_colors = np.zeros(
    (len(points), 3),
    dtype=np.float32
)

for i, aff in enumerate(
    best_aff_name
):

    point_colors[i] = (
        affordance_colors.get(
            aff,
            [1.0, 1.0, 1.0]
        )
    )

# -------------------------------------------------
# CSV
# -------------------------------------------------

df = pd.DataFrame({

    "x": points[:, 0],
    "y": points[:, 1],
    "z": points[:, 2],

    "affordance": best_aff_name,
    "score": best_score,

    "r": point_colors[:, 0],
    "g": point_colors[:, 1],
    "b": point_colors[:, 2],
})

df.to_csv(
    "office_gun_point_affordance.csv",
    index=False
)

print(
    "\nSaved:",
    "office_gun_point_affordance.csv"
)

# -------------------------------------------------
# PLY
# -------------------------------------------------

pcd = o3d.geometry.PointCloud()

pcd.points = (
    o3d.utility.Vector3dVector(
        points
    )
)

pcd.colors = (
    o3d.utility.Vector3dVector(
        point_colors
    )
)

o3d.io.write_point_cloud(
    "office_gun_point_affordance_colored.ply",
    pcd
)

print(
    "Saved:",
    "office_gun_point_affordance_colored.ply"
)

# -------------------------------------------------
# VISUALIZE
# -------------------------------------------------

o3d.visualization.draw_geometries(
    [pcd]
)






    # import matplotlib.pyplot as plt

    # # inference.py 맨 아래에 추가
    # mask = output_aff["masks"][0].squeeze().cpu().numpy()  # [2048]
    # scores = output_aff["masks_scores"][0].squeeze().cpu().float().numpy()  # [2048]

    # # 포인트 클라우드 시각화
    # fig = plt.figure(figsize=(12, 5))

    # # 왼쪽: 마스크 결과
    # ax1 = fig.add_subplot(121, projection='3d')
    # colors = np.where(mask == 1, 'red', 'blue')
    # ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
    # ax1.set_title(f'Mask Result (red={int(mask.sum())} points)')

    # # 오른쪽: score 히트맵
    # ax2 = fig.add_subplot(122, projection='3d')
    # sc = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=scores, cmap='hot', s=1)
    # plt.colorbar(sc, ax=ax2)
    # ax2.set_title('Score Heatmap')

    # plt.savefig('mask_result.png', dpi=150, bbox_inches='tight')
    # print("mask_result.png 저장됨!")
