# core/pose/mediapipe_adapter.py

import numpy as np

def extract_frames_from_mediapipe(results_list):
    """
    Parameters
    ----------
    results_list : list
        run_mediapipe() が返す MediaPipe results の list

    Returns
    -------
    frames : list[np.ndarray]
        shape: (33, 3)  # 33 pose landmarks, (x, y, z)
    """

    frames = []

    for res in results_list:
        if res.pose_landmarks is None:
            continue

        landmarks = res.pose_landmarks.landmark

        frame = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks],
            dtype=np.float32
        )

        frames.append(frame)

    return frames
