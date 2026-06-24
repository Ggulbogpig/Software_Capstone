import open3d as o3d
import numpy as np
import pandas as pd
from collections import Counter

# =====================================================
# affordance colors
# =====================================================

affordance_colors = {
    "grasp":       [1.0, 0.0, 0.0],
    "contain":     [0.0, 1.0, 0.0],
    "lift":        [0.0, 0.0, 1.0],
    "openable":    [1.0, 1.0, 0.0],
    "layable":     [1.0, 0.0, 1.0],
    "sittable":    [0.0, 1.0, 1.0],
    "support":     [1.0, 0.5, 0.0],
    "wrapGrasp":   [0.5, 0.0, 1.0],
    "pourable":    [0.5, 1.0, 0.0],
    "move":        [0.0, 0.5, 1.0],
    "display":     [1.0, 0.0, 0.5],
    "pushable":    [0.5, 0.5, 0.5],
    "pull":        [0.3, 0.1, 0.0],
    "listen":      [0.0, 0.0, 0.3],
    "wear":        [1.0, 0.8, 0.6],
    "press":       [0.2, 0.8, 0.8],
    "cut":         [0.8, 0.0, 0.0],
    "stab":        [0.3, 0.0, 0.0],
}

# =====================================================
# mesh load
# =====================================================

mesh = o3d.io.read_triangle_mesh("colored_mesh.ply")

vertex_colors = np.asarray(mesh.vertex_colors)

triangles = np.asarray(mesh.triangles)

print(mesh)

# =====================================================
# vertex color -> affordance label
# =====================================================

vertex_labels = []

for c in vertex_colors:

    found = "none"

    for aff_name, aff_color in affordance_colors.items():

        if np.allclose(c, aff_color, atol=0.05):
            found = aff_name
            break

    vertex_labels.append(found)

# =====================================================
# face majority voting
# =====================================================

face_table = []

for face_id, face in enumerate(triangles):

    v1, v2, v3 = face

    labels = [
        vertex_labels[v1],
        vertex_labels[v2],
        vertex_labels[v3]
    ]

    face_label = Counter(labels).most_common(1)[0][0]

    face_table.append({
        "face_id": face_id,
        "v1": int(v1),
        "v2": int(v2),
        "v3": int(v3),
        "affordance": face_label
    })

# =====================================================
# dataframe
# =====================================================

df = pd.DataFrame(face_table)

print(df.head())

# =====================================================
# save
# =====================================================

df.to_csv("face_affordance_table_headset.csv", index=False)

print("saved!")