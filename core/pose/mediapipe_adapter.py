# core/pose/mediapipe_adapter.py

import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
PoseLandmark = mp_pose.PoseLandmark


def extract_frames_from_mediapipe(results):
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark

    frame = np.array([
        [
            landmarks[lm].x,
            landmarks[lm].y,
            landmarks[lm].z
        ]
        for lm in PoseLandmark
    ])

    return frame
