import subprocess

# ==========================================
# INPUTS
# ==========================================

MESH_FILE = "object/office_chair.glb"

POINT_PLY = "office_chair_point_affordance_colored.ply"

POINT_CSV = "office_chair_point_affordance.csv"

IMAGE_FILE = "object/Mug8555.png"

RULE_FILE = "rule/chair.txt"

OUTPUT_DIR = (
    "D:/Software_Capstone/EmpartACD/empart/outputs"
    
)
# ==========================================
# STEP 1
# KDTree
# ==========================================

print("STEP 1 : KDTree")

subprocess.run(
    [
        "python",
        "KDTree.py",
        "--mesh",
        MESH_FILE,
        "--point_csv",
        POINT_CSV,
        "--ply",
        POINT_PLY,
        "--face_csv",
        f"{OUTPUT_DIR}/office_chair_point_face_affordance.csv",
        "--glb",
        "outputs/office_chair_point_affordance_mesh.glb"
    ],
    check=True
),
    

# ==========================================
# STEP 2
# Granularity
# ==========================================

print("STEP 2 : Granularity")

subprocess.run(
    [
        "python",
        "granularity.py",
        "--csv",
        f"{OUTPUT_DIR}/office_chair_point_face_affordance.csv",
        "--image",
        IMAGE_FILE,
        "--rule",
        RULE_FILE,
        "--object",
        "table",
        "--output",
        f"{OUTPUT_DIR}/face_affordance_granularity_office_chair.csv"
    ],
    check=True
)

# ==========================================
# STEP 3
# BBox
# ==========================================

print("STEP 3 : BBox")

subprocess.run(
    [
        "python",
        "bbox.py",
        "--mesh",
        MESH_FILE,
        "--face_csv",
        f"{OUTPUT_DIR}/face_affordance_granularity_office_chair.csv",
        "--output",
        f"{OUTPUT_DIR}/auto_regions_office_chair_bbox.csv",
        "--visualize"
    ],
    check=True
)

# ==========================================
# STEP 4
# Empart
# ==========================================

print("STEP 4 : Empart")

subprocess.run(
    [
        "wsl",
        "bash",
        "/mnt/d/Software_Capstone/EmpartACD/empart/run_empart.sh"
    ],
    check=True
)

print("Empart finished")

###############데모용 추가
# subprocess.run(
#     [
#         "wsl",
#         "bash",
#         "/mnt/d/Software_Capstone/EmpartACD/empart/run_empart.sh"
#     ],
#     check=True
# )


# print("STEP 5 : Visualize")

# subprocess.run(
#     [
#         "python",
#         "D:/Software_Capstone/EmpartACD/empart/visualizeCLI.py",
#         "--json",
#         f"D:/Software_Capstone/EmpartACD/empart/outputs/total_result_mug8848.json"
#     ],
#     check=True
# )