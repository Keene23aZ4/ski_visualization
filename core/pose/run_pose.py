# core/pose/run_pose.py

import cv2
import mediapipe as mp


def run_mediapipe(video_path, static_image_mode=False):
    """
    Parameters
    ----------
    video_path : str
        動画ファイルのパス
    static_image_mode : bool
        False 推奨（動画用）

    Returns
    -------
    results_list : list
        各フレームの pose_landmarks.landmark
    """

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)

    results_list = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCVはBGR → MediaPipeはRGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            results_list.append(results.pose_landmarks.landmark)

        frame_idx += 1

    cap.release()
    pose.close()

    return results_list
