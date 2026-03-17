import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 1. SETUP & DIRECTORIES
os.makedirs("sessions", exist_ok=True)

st.set_page_config(page_title="Guardian Gaze AI", layout="wide")

# --- CLEAN DARK MODE CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    [data-testid="stVerticalBlock"] > div:has(div.stButton) {
        background-color: #161B22; padding: 20px; border-radius: 8px; border: 1px solid #30363D;
    }
    h1, h2, h3 { color: #FFFFFF !important; }
    </style>
    """, unsafe_allow_html=True)

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

# --- WEBRTC VIDEO ENGINE ---
class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.counter = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = img.shape
                landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
                
                ear = (eye_aspect_ratio(landmarks[RIGHT_EYE]) + eye_aspect_ratio(landmarks[LEFT_EYE])) / 2.0
                
                cv2.putText(img, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if ear < 0.21:
                    self.counter += 1
                    if self.counter >= 15:
                        cv2.putText(img, "DROWSY ALERT!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    self.counter = 0
        return img

# 3. UI LAYOUT
st.title("Guardian Gaze AI")
col_cam, col_ctrl = st.columns([2, 1])

with col_ctrl:
    st.subheader("System Control")
    run = st.checkbox("Toggle Monitoring System", value=False)
    if run:
        st.success("Monitoring Active")
    else:
        st.info("System Standby")

with col_cam:
    if run:
        webrtc_streamer(
            key="drowsiness-detector",
            video_transformer_factory=DrowsinessTransformer,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )
    else:
        st.write("Turn on the toggle to start the camera.")
