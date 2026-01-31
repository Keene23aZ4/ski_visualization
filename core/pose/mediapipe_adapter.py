# core/pose/mediapipe_adapter.py

import numpy as np

def extract_frames_from_mediapipe(results):
    """
    MediaPipe Pose の results から
    [frame][landmark][x,y,z,visibility] の numpy 配列を作る
    """

    all_frames = []

    for res in results:
        if res.pose_landmarks is None:
            all_frames.append(None)
            continue

        frame_landmarks = []
        for lm in res.pose_landmarks.landmark:
            frame_landmarks.append([
                lm.x,
                lm.y,
                lm.z,
                lm.visibility
            ])

        all_frames.append(np.array(frame_landmarks, dtype=float))

    return all_frames
