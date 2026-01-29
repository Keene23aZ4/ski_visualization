class PoseEstimator:
    def estimate(self, video_path):
        """
        Args:
            video_path (str)
        Returns:
            keypoints (np.ndarray): [T, J, 3]
        """
        raise NotImplementedError
