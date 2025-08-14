import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PostureAnalyzer:
    """
    Analyzes body posture from an image frame to infer engagement.
    """
    def __init__(self):
        """
        Initializes the MediaPipe Pose model.
        """
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("Posture analysis model loaded successfully.")

    def analyze_posture(self, frame):
        """
        Analyzes posture to generate an engagement score and an annotated frame.

        Args:
            frame (np.array): A video frame in BGR format from OpenCV.

        Returns:
            tuple: A tuple containing:
                   - posture_data (dict): Engagement status and score. None if no pose detected.
                   - annotated_frame (np.array): The frame with pose landmarks drawn on it.
        """
        # MediaPipe expects RGB images, but OpenCV provides BGR.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False # Improve performance
        results = self.pose.process(image_rgb)
        image_rgb.flags.writeable = True

        annotated_frame = frame.copy()
        posture_data = None

        if results.pose_landmarks:
            # Draw the pose annotation on the image.
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            landmarks = results.pose_landmarks.landmark
            
            # --- Simple Engagement Logic ---
            # Use the relative positions of the nose and shoulders to guess if the person is leaning forward.
            try:
                nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                # Calculate the center point of the shoulders
                shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
                
                # Calculate the vertical distance between nose and shoulder center
                # A smaller value means the head is lower or leaning forward.
                lean_factor = nose.y - shoulder_center_y
                
                # Normalize score to be between 0 and 1. This requires tuning.
                # We assume a 'neutral' lean_factor is around 0.1, and more negative is more engaged.
                engagement_score = np.clip(1 - (lean_factor * 5), 0, 1)

                status = "Disengaged"
                if engagement_score > 0.75:
                    status = "Highly Engaged"
                elif engagement_score > 0.5:
                    status = "Engaged"

                posture_data = {
                    'status': status,
                    'engagement_score': round(float(engagement_score), 2)
                }
            except Exception:
                # A landmark might not be visible in the frame.
                pass

        return posture_data, annotated_frame

    def close(self):
        """Releases the pose model resources."""
        self.pose.close()
