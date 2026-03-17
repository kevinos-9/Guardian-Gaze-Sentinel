import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime
import pandas as pd

# 1. SETUP & DIRECTORIES
os.makedirs("sessions", exist_ok=True)
log_path = "drowsiness_logs.csv"

st.set_page_config(page_title="Guardian Gaze AI", layout="wide")

# --- CLEAN DARK MODE CSS ---
st.markdown(
    """
    <style>
    /* Standard Deep Dark Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Clean white borders for containers */
    [data-testid="stVerticalBlock"] > div:has(div.stButton) {
        background-color: #161B22;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #30363D;
    }

    /* Standard Bold Typography */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
    }

    /* Standard Streamlit Styled Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

def play_alarm():
    alarm_html = """
        <audio autoplay>
            <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock_short.ogg" type="audio/ogg">
        </audio>
    """
    st.components.v1.html(alarm_html, height=0)

# 2. SESSION STATE & AUTO-START (19:00 to 10:00)
if "run_monitor" not in st.session_state:
    st.session_state.run_monitor = False

now = datetime.now()
is_night_shift = (now.hour >= 19 or now.hour < 10)

if is_night_shift and not st.session_state.run_monitor:
    st.session_state.run_monitor = True
    st.rerun() 

# 3. UI LAYOUT
st.title("Guardian Gaze AI")
col_cam, col_ctrl = st.columns([2, 1])

with col_ctrl:
    st.subheader("System Control")
    if is_night_shift:
        st.info("🌙 Night Monitoring Enabled")
    else:
        st.write(f"System Standby (Time: {now.strftime('%H:%M')})")

    if st.button("Start Monitor"):
        st.session_state.run_monitor = True
        st.rerun()
        
    if st.button("Stop Monitor"):
        st.session_state.run_monitor = False
        st.rerun()

# 4. MONITORING ENGINE
with col_cam:
    image_placeholder = st.empty()
    alarm_placeholder = st.empty() 
    
    if st.session_state.run_monitor:
        cap = cv2.VideoCapture(0)
        width, height = 320, 240 
        fps = 10                  
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        file_name = f"sessions/session_{now.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
        out = cv2.VideoWriter(file_name, fourcc, fps, (width, height))

        with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
            counter = 0
            while st.session_state.run_monitor:
                if datetime.now().hour == 10 and is_night_shift:
                    st.session_state.run_monitor = False
                    break

                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.resize(frame, (width, height))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                # Simple Timestamp
                ts = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, ts, (width-70, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Draw standard landmarks (Green/Cyan)
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                        )

                        landmarks = np.array([(lm.x * width, lm.y * height) for lm in face_landmarks.landmark])
                        ear = (eye_aspect_ratio(landmarks[RIGHT_EYE]) + eye_aspect_ratio(landmarks[LEFT_EYE])) / 2.0
                        
                        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        if ear < 0.22:
                            counter += 1
                            if counter >= 10: 
                                cv2.putText(frame, "DROWSY!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                with alarm_placeholder:
                                    play_alarm()
                        else:
                            counter = 0
                            alarm_placeholder.empty()

                out.write(frame)
                image_placeholder.image(frame, channels="BGR")
                time.sleep(0.1) 

            cap.release()
            out.release()
            st.rerun()
    else:
        image_placeholder.info("System is currently idle.")

# 5. TABS
st.divider()
t1, t2 = st.tabs(["Logs", "Saved Videos"])
with t2:
    vids = [f for f in os.listdir("sessions") if f.endswith(".mp4")]
    if vids:
        s = st.selectbox("Select Video", sorted(vids, reverse=True))
        st.video(f"sessions/{s}")