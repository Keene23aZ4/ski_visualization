# core/pose/mediapipe_adapter.py

import numpy as np

def extract_frames_from_mediapipe(results):
    frames = []

    for res in results:
        if not res.pose_landmarks:
            continue

        landmarks = res.pose_landmarks[0]  # ← ここが決定的に重要

        frame = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks],
            dtype=np.float32
        )
        frames.append(frame)

    return frames


