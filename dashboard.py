import streamlit as st
import pandas as pd
import json
import time
import plotly.graph_objects as go
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Engagement AI Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Style and Color Definitions ---
EMOTION_COLORS = {
    'happy': '#FFD700',   # Gold
    'sad': '#4682B4',     # SteelBlue
    'angry': '#DC143C',   # Crimson
    'fear': '#9370DB',    # MediumPurple
    'disgust': '#3CB371', # MediumSeaGreen
    'surprise': '#FFA500',# Orange
    'neutral': '#D3D3D3'  # LightGray
}

# --- Helper Functions ---
def load_data():
    """Loads analysis data from the JSON file."""
    try:
        with open("analysis_results.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"facial": [], "posture": [], "audio": []}

def calculate_overall_engagement(posture_score, facial_data):
    """Calculates a combined engagement score."""
    positive_emotions = ['happy', 'surprise']
    positivity_score = 0.5  # Default to neutral
    if facial_data and 'emotions' in facial_data:
        emotions = facial_data['emotions']
        pos_score = sum(emotions.get(e, 0) for e in positive_emotions)
        total_score = sum(emotions.values())
        if total_score > 0:
            positivity_score = pos_score / total_score
    
    overall_score = (posture_score * 0.7) + (positivity_score * 0.3)
    return min(1.0, overall_score) # Ensure score doesn't exceed 1.0

# --- UI Layout ---
st.title("🤖 Multimodal Engagement AI")

col1, col2 = st.columns([2, 1.5]) 

with col1:
    st.header("🎥 Live Camera Feed")
    live_frame_placeholder = st.empty()

with col2:
    st.header("📊 Live Analytics")
    engagement_gauge_placeholder = st.empty()
    
    st.subheader("Emotion")
    emotion_chart_placeholder = st.empty()
    
    st.subheader("Engagement Trend")
    engagement_chart_placeholder = st.empty()

st.markdown("---")
with st.expander("✍️ View Live Transcript & Sentiment Analysis", expanded=True):
    transcript_placeholder = st.empty()

# --- Main Loop for Data Visualization ---
while True:
    data = load_data()
    
    # --- 1. Update Live Video Feed ---
    try:
        image = Image.open("live_frame.jpg")
        live_frame_placeholder.image(image, use_container_width=True)
    except FileNotFoundError:
        live_frame_placeholder.warning("Waiting for camera feed to start...")

    # --- 2. Update Engagement Gauge ---
    if data["posture"]:
        latest_posture = data["posture"][-1]
        posture_score = latest_posture.get('engagement_score', 0)
        latest_facial = data["facial"][-1] if data["facial"] else None
        overall_score = calculate_overall_engagement(posture_score, latest_facial)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_score * 100,
            title={'text': "Overall Engagement Score", 'font': {'size': 20}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "royalblue"},
                   'steps': [
                       {'range': [0, 40], 'color': 'lightgray'},
                       {'range': [40, 70], 'color': 'gray'}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}
        ))
        fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
        engagement_gauge_placeholder.plotly_chart(fig_gauge, use_container_width=True, key="engagement_gauge")
        
        df_engagement = pd.DataFrame(data["posture"])
        # FIX: Removed the unsupported 'key' argument from line_chart
        engagement_chart_placeholder.line_chart(df_engagement[['engagement_score']])

    # --- 3. Update Emotion Donut Chart ---
    if data["facial"]:
        latest_facial = data["facial"][-1]
        emotions = latest_facial['emotions']
        
        df_emotions = pd.DataFrame.from_dict(emotions, orient='index', columns=['score']).reset_index()
        colors = df_emotions['index'].map(EMOTION_COLORS).fillna('grey').tolist()

        fig_donut = go.Figure(data=[go.Pie(
            labels=df_emotions['index'], 
            values=df_emotions['score'], 
            hole=.5,
            marker_colors=colors,
            textinfo='label+percent',
            insidetextorientation='radial'
        )])
        fig_donut.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        emotion_chart_placeholder.plotly_chart(fig_donut, use_container_width=True, key="emotion_donut")

    # --- 4. Update Transcript Section ---
    if data["audio"]:
        transcripts_html = ""
        for entry in reversed(data["audio"][-5:]):
            s_label = entry['sentiment']['label']
            s_score = entry['sentiment']['score']
            color = "green" if s_label == "POSITIVE" else "red" if s_label == "NEGATIVE" else "gray"
            transcripts_html += f"""
                <div style='border-left: 5px solid {color}; padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: #f9f9f9;'>
                    <p style='margin: 0; font-size: 1em;'><i>"{entry['text']}"</i></p>
                    <p style='margin: 0; font-size: 0.9em; text-align: right;'><b>Sentiment: <span style='color:{color};'>{s_label}</span> ({s_score:.2f})</b></p>
                </div>
            """
        transcript_placeholder.markdown(transcripts_html, unsafe_allow_html=True)

    time.sleep(0.1)
