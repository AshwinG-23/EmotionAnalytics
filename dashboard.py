import streamlit as st
import pandas as pd
import numpy as np
import cv2
import plotly.graph_objects as go
import time
import queue
import threading
from collections import deque

# Import your custom analyzer classes
from facial_analyzer import FacialAnalyzer
from posture_analyzer import PostureAnalyzer
from audio_analyzer import AudioAnalyzer

# --- Configuration ---
EMOTION_CONFIG = {
    'happy':    {'emoji': '😊', 'color': '#FFD700'}, 'sad':      {'emoji': '😢', 'color': '#4682B4'},
    'angry':    {'emoji': '😠', 'color': '#DC143C'}, 'fear':     {'emoji': '😨', 'color': '#9370DB'},
    'disgust':  {'emoji': '🤢', 'color': '#3CB371'}, 'surprise': {'emoji': '😮', 'color': '#FFA500'},
    'neutral':  {'emoji': '😐', 'color': '#D3D3D3'}
}

# --- Thread-safe Queues for Data Passing ---
# A queue for video-related data (frame, posture, facial)
video_data_queue = queue.Queue(maxsize=1)
# A queue for audio data (transcripts, sentiment)
audio_data_queue = queue.Queue(maxsize=10) 

# --- Analysis Thread Workers ---

def video_analysis_worker():
    """
    Captures video, runs facial/posture analysis, and puts results in the video queue.
    """
    facial_analyzer = FacialAnalyzer()
    posture_analyzer = PostureAnalyzer()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        facial_result = facial_analyzer.analyze_emotion(frame)
        posture_result, annotated_frame = posture_analyzer.analyze_posture(frame)
        
        data_packet = {"frame": annotated_frame, "facial": facial_result, "posture": posture_result}

        if video_data_queue.full():
            try: video_data_queue.get_nowait()
            except queue.Empty: pass
        video_data_queue.put(data_packet)
    cap.release()

def audio_analysis_worker():
    """
    Runs audio analysis and puts results in the audio queue.
    """
    # Pass the shared queue to the analyzer
    audio_analyzer = AudioAnalyzer(output_queue=audio_data_queue)
    audio_analyzer.start()
    # The analyzer runs its own loop, so this thread just needs to keep it alive.
    while True:
        time.sleep(1)

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Live Emotion AI", layout="wide")
st.title("🎭 Live Multimodal Emotion AI")

col1, col2 = st.columns([2, 1.5])

with col1:
    st.header("📷 Live Camera Feed")
    live_frame_placeholder = st.empty()
    live_frame_placeholder.image(np.zeros((480, 640, 3)), channels="BGR", use_container_width=True)

with col2:
    st.header("📊 Live Analytics")
    emotion_chart_placeholder = st.empty()
    posture_placeholder = st.empty()
    emotion_details_placeholder = st.empty()

st.markdown("---")
st.header("🎤 Live Transcript & Sentiment")
transcript_placeholder = st.empty()

# Use session_state to store the list of transcripts
if 'transcripts' not in st.session_state:
    st.session_state.transcripts = deque(maxlen=5) # Store the last 5 transcripts

# --- Start Analysis Threads ---
if 'analysis_threads_started' not in st.session_state:
    st.session_state.analysis_threads_started = True
    
    video_thread = threading.Thread(target=video_analysis_worker, daemon=True)
    video_thread.start()
    
    audio_thread = threading.Thread(target=audio_analysis_worker, daemon=True)
    audio_thread.start()
    
    print("--- Analysis threads started ---")

# --- Main UI Update Loop ---
while True:
    # --- Process Video Data ---
    try:
        video_data = video_data_queue.get(timeout=0.01)
        frame, facial_data, posture_data = video_data.values()

        if frame is not None:
            live_frame_placeholder.image(cv2.flip(frame, 1), channels="BGR", use_container_width=True)
        if posture_data:
            posture_placeholder.metric(
                "Engagement Status",
                posture_data.get('status', 'Unknown'),
                f"{posture_data.get('engagement_score', 0):.2f} Score"
            )
        if facial_data and facial_data.get('emotions'):
            emotions = facial_data['emotions']
            dominant_emotion = facial_data['dominant_emotion']
            df_emotions = pd.DataFrame.from_dict(emotions, orient='index', columns=['score']).reset_index()
            df_emotions['color'] = df_emotions['index'].map(lambda e: EMOTION_CONFIG.get(e, {}).get('color', '#808080'))
            
            fig_donut = go.Figure(
                data=[go.Pie(
                    labels=df_emotions['index'],
                    values=df_emotions['score'],
                    hole=.7,
                    marker_colors=df_emotions['color'],
                    textinfo='none'
                )]
            )
            fig_donut.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                annotations=[dict(
                    text=EMOTION_CONFIG.get(dominant_emotion, {}).get('emoji', '❓'),
                    x=0.5,
                    y=0.5,
                    font_size=80,
                    showarrow=False
                )]
            )

            # Unique key per loop iteration
            if "chart_counter" not in st.session_state:
                st.session_state.chart_counter = 0
            st.session_state.chart_counter += 1

            emotion_chart_placeholder.plotly_chart(
                fig_donut,
                use_container_width=True,
                key=f"emotion_chart_{st.session_state.chart_counter}"
            )

            with emotion_details_placeholder.container():
                st.subheader("Emotion Breakdown")
                for emotion, score in emotions.items():
                    st.progress(float(score / 100), text=f"{emotion.capitalize()}: {score:.1f}%")

    except queue.Empty:
        pass # It's okay if the queue is empty, we'll just try again.

    # --- Process Audio Data ---
    try:
        audio_data = audio_data_queue.get(timeout=0.01)
        st.session_state.transcripts.append(audio_data)
    except queue.Empty:
        pass # Also okay if this queue is empty.
    
    # --- Display Transcripts ---
    with transcript_placeholder.container():
        transcripts_html = ""
        for entry in reversed(st.session_state.transcripts):
            s_label = entry['sentiment']['label']
            s_score = entry['sentiment']['score']
            color = "green" if s_label == "POSITIVE" else "red" if s_label == "NEGATIVE" else "gray"
            transcripts_html += f"""
                <div style='border-left: 5px solid {color}; padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: #262730;'>
                    <p style='margin: 0; font-size: 1em;'><i>"{entry['text']}"</i></p>
                    <p style='margin: 0; font-size: 0.9em; text-align: right;'><b>Sentiment: <span style='color:{color};'>{s_label}</span> ({s_score:.2f})</b></p>
                </div>
            """
        st.markdown(transcripts_html, unsafe_allow_html=True)

    time.sleep(0.1)
