import cv2
import mediapipe as mp
import numpy as np

class PostureAnalyzer:
    """
    Analyzes body posture from an image frame to infer engagement.
    """
    def __init__(self):
        print("Loading posture analysis model...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        print("Posture analysis model loaded successfully.")

    def analyze_posture(self, frame):
        """
        Analyzes posture to generate an engagement score and an annotated frame.
        Returns:
            tuple: (posture_data, annotated_frame)
        """
        annotated_frame = frame.copy()
        posture_data = None
        
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                
                landmarks = results.pose_landmarks.landmark
                nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
                lean_factor = nose.y - shoulder_center_y
                engagement_score = np.clip(1 - (lean_factor * 5), 0, 1)

                if engagement_score > 0.75:
                    status = "Highly Engaged"
                elif engagement_score > 0.5:
                    status = "Engaged"
                else:
                    status = "Disengaged"

                posture_data = {
                    'status': status,
                    'engagement_score': round(float(engagement_score), 2)
                }
        except Exception:
            pass

        return posture_data, annotated_frame

    def close(self):
        """Releases the pose model resources."""
        self.pose.close()
