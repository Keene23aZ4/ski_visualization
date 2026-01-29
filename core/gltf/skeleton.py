# MediaPipe landmark index
MP = {
    "pelvis": 23,     # LEFT_HIP
    "spine": 11,      # LEFT_SHOULDER
    "head": 0,        # NOSE
    "left_arm": 13,   # LEFT_ELBOW
    "right_arm": 14,  # RIGHT_ELBOW
    "left_leg": 25,   # LEFT_KNEE
    "right_leg": 26,  # RIGHT_KNEE
}

# 親子関係（glTF用）
BONES = {
    "pelvis": None,
    "spine": "pelvis",
    "head": "spine",
    "left_arm": "spine",
    "right_arm": "spine",
    "left_leg": "pelvis",
    "right_leg": "pelvis",
}
