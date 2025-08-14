import cv2
from deepface import DeepFace
import numpy as np

class FacialAnalyzer:
    """
    Analyzes facial emotions from an image frame.
    This class is initialized once to load the model into memory.
    """
    def __init__(self, detector_backend='opencv'):
        self.detector_backend = detector_backend
        # Pre-load the model by analyzing a dummy frame.
        print("Loading facial analysis model...")
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            DeepFace.analyze(dummy_frame, actions=['emotion'], detector_backend=self.detector_backend, enforce_detection=False)
            print("Facial analysis model loaded successfully.")
        except Exception as e:
            print(f"Error loading facial model: {e}")

    def analyze_emotion(self, frame):
        """
        Analyzes the emotion of the most prominent face in a given frame.
        Returns:
            dict: A dictionary containing the dominant emotion and all emotion scores.
                  Returns None if no face is detected or an error occurs.
        """
        try:
            results = DeepFace.analyze(
                frame,
                actions=['emotion'],
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            if results and isinstance(results, list):
                main_face = results[0]
                return {
                    'dominant_emotion': main_face['dominant_emotion'],
                    'emotions': main_face['emotion']
                }
        except Exception:
            # This can happen if a frame is corrupted or if no face is detected.
            pass
        return None
