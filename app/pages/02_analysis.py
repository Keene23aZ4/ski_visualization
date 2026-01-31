import streamlit as st
from core.pose.run_pose import run_mediapipe
from core.pose.mediapipe_adapter import extract_frames_from_mediapipe

st.header("② 姿勢推定")

video_path = st.session_state.get("video_path")

if video_path and st.button("姿勢推定を実行"):
    results = run_mediapipe(video_path)

    pose_frames = [
        extract_frames_from_mediapipe(r)
        for r in results
        if extract_frames_from_mediapipe(r) is not None
    ]

    st.success(f"{len(pose_frames)} フレーム抽出完了")
