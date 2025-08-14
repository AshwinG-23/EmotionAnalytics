import cv2
from deepface import DeepFace
import numpy as np

class FacialAnalyzer:
    """
    Analyzes facial emotions from an image frame.
    This class is initialized once to load the model into memory.
    """
    def __init__(self, detector_backend='opencv'):
        """
        Initializes the facial analysis model.
        Args:
            detector_backend (str): The face detection backend to use ('opencv', 'ssd', 'dlib', 'mtcnn').
        """
        self.detector_backend = detector_backend
        print("Loading facial analysis model...")
        # Pre-load the model by analyzing a dummy frame.
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            DeepFace.analyze(dummy_frame, actions=['emotion'], detector_backend=self.detector_backend, enforce_detection=False)
            print("Facial analysis model loaded successfully.")
        except Exception as e:
            print(f"Error loading facial model: {e}")
            print("Please ensure all dependencies, especially TensorFlow/Keras, are installed correctly.")

    def analyze_emotion(self, frame):
        """
        Analyzes the emotion of the most prominent face in a given frame.

        Args:
            frame (np.array): A video frame in BGR format from OpenCV.

        Returns:
            dict: A dictionary containing the dominant emotion and a breakdown of all emotion scores.
                  Returns None if no face is detected or an error occurs.
        """
        try:
            # DeepFace.analyze can find and analyze multiple faces, but we'll focus on the first result.
            results = DeepFace.analyze(
                frame,
                actions=['emotion'],
                detector_backend=self.detector_backend,
                enforce_detection=False # Set to False to avoid errors when no face is found
            )
            # The result is a list of dictionaries, one for each face.
            if results and isinstance(results, list) and 'dominant_emotion' in results[0]:
                main_face = results[0]
                return {
                    'dominant_emotion': main_face['dominant_emotion'],
                    'emotions': main_face['emotion'] # Dictionary of all emotions and their scores
                }
        except Exception:
            # This can happen if a frame is corrupted or if no face is detected.
            pass
        return None
