# core/pose/run_pose.py

import cv2
import mediapipe as mp


def run_mediapipe(video_path: str):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)

    cap = cv2.VideoCapture(video_path)

    results_all = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            results_all.append(results.pose_landmarks)

    cap.release()
    pose.close()

    return results_all
