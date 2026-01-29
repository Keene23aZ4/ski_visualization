# core/gltf/animation.py
import numpy as np

def normalize_keypoints(frames):
    """
    frames: List[Dict[str, (x, y, z)]]
    """
    all_coords = np.array(
        [[kp for kp in frame.values()] for frame in frames]
    )

    center = all_coords.mean(axis=(0, 1))
    scale = np.max(np.linalg.norm(all_coords - center, axis=-1))

    normalized = []
    for frame in frames:
        norm_frame = {}
        for name, coord in frame.items():
            norm_frame[name] = (coord - center) / scale
        normalized.append(norm_frame)

    return normalized
