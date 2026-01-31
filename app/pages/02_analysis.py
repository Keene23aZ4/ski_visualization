import sys
from pathlib import Path

# プロジェクトルートを Python Path に追加
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import streamlit as st
from core.pose.run_pose import run_mediapipe
from core.pose.mediapipe_adapter import extract_frames_from_mediapipe

st.header("② 姿勢推定")


video_path = st.session_state.get("video_path")

if video_path and st.button("姿勢推定を実行"):
    results = run_mediapipe(video_path)
    frames = extract_frames_from_mediapipe(results)
    
    st.write(len(frames))
    st.write(frames[0].shape)   # (33, 3)



