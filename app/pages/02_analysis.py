import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import streamlit as st

from core.pose.run_pose import run_mediapipe
from core.pose.mediapipe_adapter import extract_frames_from_mediapipe

st.header("② 姿勢推定")

if "video_path" in st.session_state:
    if st.button("姿勢推定を実行"):
        results = run_mediapipe(st.session_state["video_path"])
        frames = extract_frames_from_mediapipe(results)
        st.success("姿勢推定完了")
else:
    st.warning("先に動画をアップロードしてください")
