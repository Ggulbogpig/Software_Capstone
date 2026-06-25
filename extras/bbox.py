#1. affordance + adjacent merge
import pandas as pd
import networkx as nx
from collections import defaultdict

import open3d as o3d
import numpy as np
import pandas as pd
from collections import Counter

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mesh",
    required=True
)

parser.add_argument(
    "--face_csv",
    required=True
)

parser.add_argument(
    "--output",
    required=True
)

parser.add_argument(
    "--visualize",
    action="store_true"
)

args = parser.parse_args()

# =====================================================
# mesh load
# =====================================================
#걍 원본 메시를 넣으면 됨. bbox 생성은 csv 파일 기준으로 생성하는 것이기 때문. 
# mesh = o3d.io.read_triangle_mesh("colored_mesh_bag.ply")
#mesh = o3d.io.read_triangle_mesh("motor.glb")
#컵
#mesh = o3d.io.read_triangle_mesh("D:/Software_Capstone/archive (1)/ShapeNetCore.v2/ShapeNetCore.v2/03797390/1a97f3c83016abca21d0de04f408950f/models/model_normalized.ply")
#mug8555
#mesh = o3d.io.read_triangle_mesh("D:/Software_Capstone/EmpartACD/empart/Mug/Mug/models/8555.obj")
#microwave3465
#mesh = o3d.io.read_triangle_mesh("D:/Software_Capstone/EmpartACD/empart/Microwave/Microwave/models/3465.obj")
mesh = o3d.io.read_triangle_mesh(
    args.mesh
)

vertex_colors = np.asarray(mesh.vertex_colors)

triangles = np.asarray(mesh.triangles)

vertices = np.asarray(
    mesh.vertices
)

print(mesh)

def build_affordance_components(df):

    G = nx.Graph()

    edge_to_faces = defaultdict(list)

    # ------------------------
    # edge map 생성
    # ------------------------
    for _, row in df.iterrows():

        fid = row["face_id"]

        v1, v2, v3 = row["v1"], row["v2"], row["v3"]

        edges = [
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v3))),
            tuple(sorted((v3, v1)))
        ]

        for e in edges:
            edge_to_faces[e].append(fid)

        G.add_node(fid)

    # ------------------------
    # shared edge + same affordance
    # ------------------------
    face_lookup = df.set_index("face_id")

    for edge, faces in edge_to_faces.items():

        if len(faces) < 2:
            continue

        for i in range(len(faces)):
            for j in range(i+1, len(faces)):

                f1 = faces[i]
                f2 = faces[j]

                aff1 = face_lookup.loc[f1, "affordance"]
                aff2 = face_lookup.loc[f2, "affordance"]

                # 같은 affordance만 merge
                if aff1 == aff2:
                    G.add_edge(f1, f2)

    components = list(nx.connected_components(G))

    return components


#2. small component merge
def merge_small_components(
    df,
    components,
    vertices,
    min_faces=30
):

    face_lookup = df.set_index("face_id")

    merged = []
    large = []
    small = []

    # large / small 분리
    for comp in components:

        if len(comp) >= min_faces:
            large.append(comp)
        else:
            small.append(comp)

    # bbox center
    def comp_center(comp):

        comp_df = df[df.face_id.isin(comp)]

        vids = np.unique(
            comp_df[["v1","v2","v3"]].values
        )

        pts = vertices[vids]

        return pts.mean(axis=0)

    # small -> nearest large
    for s in small:

        s_aff = face_lookup.loc[
            next(iter(s)),
            "affordance"
        ]

        s_center = comp_center(s)

        best = None
        best_dist = 1e9

        for i, l in enumerate(large):

            l_aff = face_lookup.loc[
                next(iter(l)),
                "affordance"
            ]

            # 같은 affordance만 merge
            if s_aff != l_aff:
                continue

            l_center = comp_center(l)

            d = np.linalg.norm(
                s_center - l_center
            )

            if d < best_dist:
                best_dist = d
                best = i

        if best is not None:
            large[best] |= s
        else:
            large.append(s)

    return large


#components merge
def merge_nearby_components(
    df,
    components,
    vertices,
    distance_threshold=0.15
):

    face_lookup = df.set_index("face_id")

    # component center
    def comp_center(comp):

        comp_df = df[df.face_id.isin(comp)]

        vids = np.unique(
            comp_df[["v1","v2","v3"]].values
        )

        pts = vertices[vids]

        return pts.mean(axis=0)

    merged = []
    used = set()

    for i, c1 in enumerate(components):

        if i in used:
            continue

        new_comp = set(c1)

        aff1 = face_lookup.loc[
            next(iter(c1)),
            "affordance"
        ]

        center1 = comp_center(c1)

        for j, c2 in enumerate(components):

            if j <= i or j in used:
                continue

            aff2 = face_lookup.loc[
                next(iter(c2)),
                "affordance"
            ]

            if aff1 != aff2:
                continue

            center2 = comp_center(c2)

            d = np.linalg.norm(
                center1-center2
            )

            if d < distance_threshold:

                new_comp |= c2
                used.add(j)

        merged.append(new_comp)
        used.add(i)

    return merged

import numpy as np


def bbox_iou(a, b):

    xmin=max(a["xmin"], b["xmin"])
    ymin=max(a["ymin"], b["ymin"])
    zmin=max(a["zmin"], b["zmin"])

    xmax=min(a["xmax"], b["xmax"])
    ymax=min(a["ymax"], b["ymax"])
    zmax=min(a["zmax"], b["zmax"])

    inter=max(0,xmax-xmin) \
        *max(0,ymax-ymin) \
        *max(0,zmax-zmin)

    vol_a=(a["xmax"]-a["xmin"]) \
        *(a["ymax"]-a["ymin"]) \
        *(a["zmax"]-a["zmin"])

    vol_b=(b["xmax"]-b["xmin"]) \
        *(b["ymax"]-b["ymin"]) \
        *(b["zmax"]-b["zmin"])

    union=vol_a+vol_b-inter

    if union==0:
        return 0

    return inter/union

def merge_overlapping_regions(
    region_df,
    iou_thresh=0.3
):

    regions=region_df.to_dict("records")

    merged=[]
    used=set()

    for i,a in enumerate(regions):

        if i in used:
            continue

        cur=a.copy()

        for j,b in enumerate(regions):

            if j<=i or j in used:
                continue

            # none skip
            if b["affordance"]=="none":
                continue

            # affordance 다르면 유지
            if a["affordance"]!=b["affordance"]:
                continue

            iou=bbox_iou(cur,b)

            # 많이 겹치면 merge
            if iou>iou_thresh:

                cur["xmin"]=min(
                    cur["xmin"],b["xmin"]
                )
                cur["ymin"]=min(
                    cur["ymin"],b["ymin"]
                )
                cur["zmin"]=min(
                    cur["zmin"],b["zmin"]
                )

                cur["xmax"]=max(
                    cur["xmax"],b["xmax"]
                )
                cur["ymax"]=max(
                    cur["ymax"],b["ymax"]
                )
                cur["zmax"]=max(
                    cur["zmax"],b["zmax"]
                )

                used.add(j)

        merged.append(cur)
        used.add(i)

    return pd.DataFrame(merged)


#2. component → bbox + region table
def component_to_regions(df, components, vertices):


        # ==========================
    # overlap check
    # ==========================

    all_faces = set()

    for i, comp in enumerate(components):

        overlap = all_faces.intersection(comp)

        if len(overlap) > 0:
            print(
                "OVERLAP COMPONENT",
                i,
                "count:",
                len(overlap)
            )

        all_faces.update(comp)

    print("Assigned faces:", len(all_faces))
    print("Total faces:", len(df))

    faces_all = np.asarray(mesh.triangles)



    regions = []

    for idx, comp in enumerate(components):

        comp_df = df[df.face_id.isin(comp)]

        vids = np.unique(
            comp_df[["v1","v2","v3"]].values
        )

        pts = vertices[vids]

        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)

        affordance = comp_df["affordance"].mode()[0]
        granularity = comp_df["acd_granularity"].mean()

        regions.append({
            "region_id": idx,
            "affordance": affordance,
            "acd_granularity": granularity,
            "faces": list(comp),
            "xmin": mins[0],
            "ymin": mins[1],
            "zmin": mins[2],
            "xmax": maxs[0],
            "ymax": maxs[1],
            "zmax": maxs[2]
        })

    return pd.DataFrame(regions)


def granularity_to_max_hulls(g):

    if g >= 0.75:
        return 40

    elif g >= 0.6:
        return 20

    elif g >= 0.4:
        return 15

    else:
        return 10


# def granularity_to_threshold(g):

#     # high importance
#     # -> finer
#     return min(
#         0.002,
#         0.08 * (1-g)
#     )
def granularity_to_threshold(g):
    if g >= 0.85:
        return 0.01      # 매우 중요한 영역: cut, stab, blade edge 등
    elif g >= 0.70:
        return 0.02     # 중요한 영역: grasp, handle 등
    elif g >= 0.50:
        return 0.1      # 보통 영역
    else:
        return 0.1     # 덜 중요한 영역



#merged_df = pd.read_csv("face_affordance_granularity_microwave3465.csv")
merged_df = pd.read_csv(
    args.face_csv
)

components = build_affordance_components(merged_df)

components = merge_small_components(
    merged_df,
    components,
    vertices,
    min_faces=50
)

region_df = component_to_regions(
    merged_df,
    components,
    vertices
)
print(region_df)


region_df[
    "maxConvexHulls"
] = region_df[
    "acd_granularity"
].apply(
    granularity_to_max_hulls
)

region_df[
    "acdThreshold"
] = region_df[
    "acd_granularity"
].apply(
    granularity_to_threshold
)


# none 제거
region_df = region_df[
    region_df.affordance!="none"
]
# 겹치는 box suppression
region_df = merge_overlapping_regions(
    region_df,
    iou_thresh=0.25
)
region_df = region_df[
    region_df["acd_granularity"] >= 0.6
].reset_index(drop=True)



print(
    region_df[
        ["affordance",
         "acd_granularity"]
    ]
)

print(region_df)
print(
    region_df[
        [
            "affordance",
            "acd_granularity",
            "maxConvexHulls",
            "acdThreshold"
        ]
    ]
)

# components =merge_nearby_components(
#     merged_df,
#     components,
#     vertices,
#     distance_threshold=0.2
# )
# region_df = region_df.drop(
#     columns=["faces"],
#     errors="ignore"
# )

# region_df.to_csv(
#     "auto_regions_microwave3465_bbox.csv",
#     index=False
# )
region_df.to_csv(
    args.output,
    index=False
)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--face_csv")


print(region_df["acd_granularity"].tolist())


# # lift만
# lift_df = region_df[
#     region_df["affordance"] == "lift"
# ].copy()

# # 첫번째 lift box 하나만
# lift_df = lift_df.head(1)

# print(lift_df)

# lift_df.to_csv(
#     "lift_only.csv",
#     index=False
# )






#1. bbox → Open3D box 변환 함수
def create_bbox_lineset(xmin, ymin, zmin,
                        xmax, ymax, zmax):

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=[xmin, ymin, zmin],
        max_bound=[xmax, ymax, zmax]
    )

    lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        bbox
    )

    # 빨간색 box
    colors = [[1,0,0] for _ in range(
        len(lineset.lines)
    )]

    lineset.colors = o3d.utility.Vector3dVector(
        colors
    )

    return lineset


#2. mesh + box visualize
def visualize_regions(mesh, region_df):

    geometries = [mesh]

    for _, row in region_df.iterrows():

        box = create_bbox_lineset(
            row["xmin"],
            row["ymin"],
            row["zmin"],
            row["xmax"],
            row["ymax"],
            row["zmax"]
        )

        geometries.append(box)
        

    o3d.visualization.draw_geometries(
        geometries,
        mesh_show_back_face=True
    )

def create_bbox_lineset(
    xmin, ymin, zmin,
    xmax, ymax, zmax,
    color=[1,0,0]
):

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        [xmin, ymin, zmin],
        [xmax, ymax, zmax]
    )

    lineset = (
        o3d.geometry.LineSet
        .create_from_axis_aligned_bounding_box(
            bbox
        )
    )

    lineset.colors = (
        o3d.utility.Vector3dVector(
            [color] * len(lineset.lines)
        )
    )

    return lineset

def visualize_regions_with_text(mesh, region_df):

    app = gui.Application.instance
    app.initialize()

    w = app.create_window(
        "Region Weight Viewer",
        1200,
        900
    )

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(
        w.renderer
    )

    # mesh 추가
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"

    scene.scene.add_geometry(
        "mesh",
        mesh,
        mat
    )

    # 빨간색 box
    color = [1,0,0]

    # bbox + text
    line_mat = rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = 2

    for i, row in region_df.iterrows():

        g = row["acd_granularity"]


        box = create_bbox_lineset(
            row["xmin"],
            row["ymin"],
            row["zmin"],
            row["xmax"],
            row["ymax"],
            row["zmax"],
            color=[1,0,0]
        )

        scene.scene.add_geometry(
            f"box{i}",
            box,
            line_mat
        )

        # box center
        center = np.array([
            (row["xmin"]+row["xmax"])/2,
            (row["ymin"]+row["ymax"])/2,
            (row["zmin"]+row["zmax"])/2
        ])

        # text label
        txt = (
            f'{row["affordance"]}\n'
            f'{g:.2f}'
        )

        scene.add_3d_label(
            center,
            txt
        )

    bounds = mesh.get_axis_aligned_bounding_box()
    scene.setup_camera(
        60,
        bounds,
        bounds.get_center()
    )

    w.add_child(scene)
    app.run()



# visualize_regions(
#     mesh,
#     region_df
# )

# visualize_regions_with_text(
#     mesh,
#     region_df
# )
if args.visualize:

    visualize_regions(
        mesh,
        region_df
    )

    visualize_regions_with_text(
        mesh,
        region_df
    )
