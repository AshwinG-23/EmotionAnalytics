import cv2
import time
import json
import numpy as np
from threading import Thread, Lock
from collections import deque

# Import your custom analyzer classes
from facial_analyzer import FacialAnalyzer
from posture_analyzer import PostureAnalyzer
from audio_analyzer import AudioAnalyzer

# --- Custom JSON Encoder to handle NumPy data types ---
class NpEncoder(json.JSONEncoder):
    """ 
    Custom JSON encoder to handle NumPy types (like float32) that
    are not serializable by default.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# --- Shared State & Data Storage ---
DATA_LOCK = Lock()
ANALYSIS_RESULTS = {
    "facial": deque(maxlen=100),
    "posture": deque(maxlen=100),
    "audio": deque(maxlen=100),
}

# --- Worker Functions for Threading ---

def video_analysis_worker():
    """
    Connects to the local webcam, runs analysis, and saves the annotated frame as an image.
    """
    print("Starting video analysis worker...")
    facial_analyzer = FacialAnalyzer()
    posture_analyzer = PostureAnalyzer()
    
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from webcam.")
            break

        facial_data = facial_analyzer.analyze_emotion(frame)
        posture_data, annotated_frame = posture_analyzer.analyze_posture(frame)

        with DATA_LOCK:
            if facial_data:
                ANALYSIS_RESULTS["facial"].append(facial_data)
            if posture_data:
                ANALYSIS_RESULTS["posture"].append(posture_data)

        # Add annotations to the frame for display
        if posture_data:
            cv2.putText(annotated_frame, f"Engagement: {posture_data['status']} ({posture_data['engagement_score']})", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        if facial_data:
            cv2.putText(annotated_frame, f"Emotion: {facial_data['dominant_emotion']}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        # --- NEW: Save the annotated frame to a file for the dashboard ---
        cv2.imwrite("live_frame.jpg", annotated_frame)
        
        # The local display window is now optional but good for debugging
        # cv2.imshow('Real-time Analysis', annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
            
    cap.release()
    cv2.destroyAllWindows()
    posture_analyzer.close()
    print("Video analysis worker stopped.")


def audio_analysis_worker():
    """
    Runs the audio analyzer and continuously checks for new transcriptions.
    """
    print("Starting audio analysis worker...")
    audio_analyzer = AudioAnalyzer()
    audio_analyzer.start()

    try:
        while True:
            audio_data = audio_analyzer.get_latest_transcription()
            if audio_data:
                with DATA_LOCK:
                    ANALYSIS_RESULTS["audio"].append(audio_data)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        audio_analyzer.stop()
        print("Audio analysis worker stopped.")

def save_results_worker():
    """
    Periodically saves the analysis results to a JSON file using the custom encoder.
    """
    while True:
        with DATA_LOCK:
            results_to_save = {
                "facial": list(ANALYSIS_RESULTS["facial"]),
                "posture": list(ANALYSIS_RESULTS["posture"]),
                "audio": list(ANALYSIS_RESULTS["audio"])
            }
        
        try:
            with open("analysis_results.json", "w") as f:
                json.dump(results_to_save, f, cls=NpEncoder)
        except Exception as e:
            print(f"Error saving results to file: {e}")
        
        time.sleep(1)


if __name__ == "__main__":
    print("--- Starting Real-time Multimodal Analytics System (Windows Version) ---")
    print("Analysis is running. You can close this window to stop the program.")

    # Run all workers in the background as daemon threads
    video_thread = Thread(target=video_analysis_worker, daemon=True)
    audio_thread = Thread(target=audio_analysis_worker, daemon=True)
    saver_thread = Thread(target=save_results_worker, daemon=True)

    video_thread.start()
    audio_thread.start()
    saver_thread.start() 

    # Keep the main thread alive so daemon threads can run
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("--- System Shutting Down ---")

