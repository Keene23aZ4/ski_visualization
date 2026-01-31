import streamlit as st
import numpy as np

from core.pose.run_pose import run_mediapipe
from core.pose.mediapipe_adapter import extract_frames_from_mediapipe

st.header("② 姿勢推定")

video_path = st.session_state.get("video_path")

if video_path and st.button("姿勢推定を実行"):
    # ① MediaPipe 実行（動画 → フレームごとの結果）
    results = run_mediapipe(video_path)

    # ② 結果 → (33,3) の配列リスト
    frames = extract_frames_from_mediapipe(results)

    st.write("num frames:", len(frames))
    st.write("frame shape:", frames[0].shape)

    # 🔽🔽🔽【ここに入れる】🔽🔽🔽
    pose_seq = np.stack(frames)   # (T, 33, 3)
    st.write("pose_seq shape:", pose_seq.shape)

    # （確認用）
    st.session_state["pose_seq"] = pose_seq
