# core/pose/mediapipe_adapter.py
# core/pose/mediapipe_adapter.py

import numpy as np
import mediapipe as mp

PoseLandmark = mp.solutions.pose.PoseLandmark



def _vec(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def extract_frames_from_mediapipe(results_list):
    """
    results_list: List[mediapipe.solutions.pose.PoseLandmark]
                  各要素は 1フレーム分の landmarks
    return: List[Dict[str, np.ndarray]]
    """

    frames = []

    for landmarks in results_list:
        frame = {}

        # --- hips / pelvis ---
        left_hip = landmarks[PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[PoseLandmark.RIGHT_HIP.value]
        pelvis = (_vec(left_hip) + _vec(right_hip)) / 2.0
        frame["pelvis"] = pelvis

        # --- legs ---
        frame["left_hip"] = _vec(left_hip)
        frame["right_hip"] = _vec(right_hip)

        frame["left_knee"] = _vec(
            landmarks[PoseLandmark.LEFT_KNEE.value]
        )
        frame["right_knee"] = _vec(
            landmarks[PoseLandmark.RIGHT_KNEE.value]
        )

        frame["left_ankle"] = _vec(
            landmarks[PoseLandmark.LEFT_ANKLE.value]
        )
        frame["right_ankle"] = _vec(
            landmarks[PoseLandmark.RIGHT_ANKLE.value]
        )

        # --- shoulders ---
        left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER.value]

        frame["left_shoulder"] = _vec(left_shoulder)
        frame["right_shoulder"] = _vec(right_shoulder)

        # --- spine / chest ---
        chest = (_vec(left_shoulder) + _vec(right_shoulder)) / 2.0
        spine = (pelvis + chest) / 2.0

        frame["spine"] = spine
        frame["chest"] = chest

        # --- arms ---
        frame["left_elbow"] = _vec(
            landmarks[PoseLandmark.LEFT_ELBOW.value]
        )
        frame["right_elbow"] = _vec(
            landmarks[PoseLandmark.RIGHT_ELBOW.value]
        )

        frame["left_wrist"] = _vec(
            landmarks[PoseLandmark.LEFT_WRIST.value]
        )
        frame["right_wrist"] = _vec(
            landmarks[PoseLandmark.RIGHT_WRIST.value]
        )

        # --- neck / head ---
        mouth_left = landmarks[PoseLandmark.MOUTH_LEFT.value]
        mouth_right = landmarks[PoseLandmark.MOUTH_RIGHT.value]
        nose = landmarks[PoseLandmark.NOSE.value]

        head = (_vec(mouth_left) + _vec(mouth_right)) / 2.0
        frame["head"] = (head + _vec(nose)) / 2.0
        frame["neck"] = (frame["head"] + chest) / 2.0

        frames.append(frame)

    return frames

