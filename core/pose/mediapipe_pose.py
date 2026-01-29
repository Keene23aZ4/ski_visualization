import cv2
import numpy as np
import mediapipe as mp
from core.pose.base import PoseEstimator

class MediaPipePose(PoseEstimator):

    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=False)

    def estimate(self, video_path):
        cap = cv2.VideoCapture(video_path)
        keypoints = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            if results.pose_landmarks:
                frame_kp = []
                for lm in results.pose_landmarks.landmark:
                    frame_kp.append([lm.x, lm.y, lm.z])
                keypoints.append(frame_kp)

        cap.release()
        return np.array(keypoints)
