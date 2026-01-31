# core/pose/run_pose.py
import mediapipe as mp
print("MediaPipe Image:", mp.Image)
import cv2
from mediapipe.tasks.python import vision


def run_mediapipe(video_path: str):
    # ===== モデル =====
    model_path = "core/pose/pose_landmarker_lite.task"

    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    BaseOptions = mp.tasks.BaseOptions
    RunningMode = vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    timestamp_ms = 0

    results_all = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame[:, :, ::-1]
            )


            result = landmarker.detect_for_video(
                mp_image,
                int(timestamp_ms)
            )

            results_all.append(result)
            timestamp_ms += 1000 / fps

    cap.release()
    return results_all
