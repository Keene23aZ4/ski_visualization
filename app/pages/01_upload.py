import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import streamlit as st
import uuid
from app.config import UPLOAD_DIR
from core.pose.run_pose import run_mediapipe
from core.pose.mediapipe_adapter import extract_frames_from_mediapipe

st.header("① 動画アップロード")

video = st.file_uploader("競技動画をアップロード", type=["mp4", "mov"])

if video:
    uid = str(uuid.uuid4())
    save_path = f"{UPLOAD_DIR}/{uid}.mp4"

    with open(save_path, "wb") as f:
        f.write(video.read())

    st.success("アップロード完了")
    st.session_state["video_path"] = save_path
    
if "video_path" in st.session_state:
    if st.button("② 解析開始"):
        results_list = run_mediapipe(st.session_state["video_path"])

        st.write("frames from mediapipe:", len(results_list))
