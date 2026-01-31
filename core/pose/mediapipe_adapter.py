# core/pose/mediapipe_adapter.py

import numpy as np
import mediapipe as mp

# 正しい使用方法
Pose = mp.solutions.pose.Pose
POSE_LANDMARKS = mp.solutions.pose.PoseLandmark  # 定数列挙



def _vec(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def extract_frames_from_mediapipe(results_list):
    frames = []

    for pose_landmarks in results_list:
        frame = {}
        for lm_enum in mp.solutions.pose.PoseLandmark:
            lm = pose_landmarks.landmark[lm_enum.value]
            frame[lm_enum.name] = np.array([lm.x, lm.y, lm.z])
        frames.append(frame)

    return frames
