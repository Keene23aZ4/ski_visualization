import numpy as np
from core.gltf.skeleton import MP

def extract_bone_positions(keypoints):
    """
    keypoints: [T, J, 3]
    return: dict[bone_name] -> [T, 3]
    """
    bones = {}
    for name, idx in MP.items():
        bones[name] = keypoints[:, idx, :]
    return bones
