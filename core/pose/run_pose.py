# core/pose/run_pose.py

import cv2
import mediapipe as mp

def run_mediapipe(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose

    frames = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            frames.append(results)

    cap.release()
    return frames
