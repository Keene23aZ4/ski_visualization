# core/pose/run_pose.py

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def run_mediapipe(video_path: str):
    """
    MediaPipe Tasks API (PoseLandmarker) を使って
    動画から pose landmarks を抽出する
    """

    # ===== モデル準備 =====
    # MediaPipe 公式モデル
    model_path = "pose_landmarker_lite.task"

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
    )

    # ===== 動画読み込み =====
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp_ms = 0

    results_all = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # OpenCV → MediaPipe Image
            mp_image = mp.Image.create_from_array(
                frame[:, :, ::-1],  # BGR → RGB
                image_format=mp.ImageFormat.SRGB,
            )

            result = landmarker.detect_for_video(
                mp_image,
                int(timestamp_ms),
            )

            results_all.append(result)
            timestamp_ms += 1000 / fps

    cap.release()
    return results_all
