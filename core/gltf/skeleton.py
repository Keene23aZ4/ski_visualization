# core/gltf/skeleton.py

MEDIAPIPE_SKELETON = [
    ("pelvis", None),
    ("spine", "pelvis"),
    ("chest", "spine"),
    ("neck", "chest"),
    ("head", "neck"),

    ("left_shoulder", "chest"),
    ("left_elbow", "left_shoulder"),
    ("left_wrist", "left_elbow"),

    ("right_shoulder", "chest"),
    ("right_elbow", "right_shoulder"),
    ("right_wrist", "right_elbow"),

    ("left_hip", "pelvis"),
    ("left_knee", "left_hip"),
    ("left_ankle", "left_knee"),

    ("right_hip", "pelvis"),
    ("right_knee", "right_hip"),
    ("right_ankle", "right_knee"),
]
