# core/pose/mediapipe_adapter.py

"""
MediaPipe Pose の出力を
- フレーム単位
- ランドマーク名 → 3D座標
という構造に変換するアダプタ
"""

import numpy as np
import mediapipe as mp

# PoseLandmark enum（公式 Legacy API）
PoseLandmark = mp.solutions.pose.PoseLandmark


def extract_frames_from_mediapipe(pose_landmarks_list):
    """
    Parameters
    ----------
    pose_landmarks_list : list
        run_mediapipe() から返される
        results.pose_landmarks のリスト

    Returns
    -------
    frames : list[dict]
        [
          {
            "LEFT_HIP": np.array([x, y, z]),
            "RIGHT_HIP": np.array([x, y, z]),
            ...
          },
          ...
        ]
    """

    frames = []

    for pose_landmarks in pose_landmarks_list:
        frame = {}

        # 各ランドマークを名前付きで格納
        for lm_enum in PoseLandmark:
            lm = pose_landmarks.landmark[lm_enum.value]

            frame[lm_enum.name] = np.array(
                [lm.x, lm.y, lm.z],
                dtype=np.float32
            )

        frames.append(frame)

    return frames
