import streamlit as st
import uuid
from app.config import UPLOAD_DIR

st.header("① 動画アップロード")

video = st.file_uploader("競技動画をアップロード", type=["mp4", "mov"])

if video:
    uid = str(uuid.uuid4())
    save_path = f"{UPLOAD_DIR}/{uid}.mp4"

    with open(save_path, "wb") as f:
        f.write(video.read())

    st.success("アップロード完了")
    st.session_state["video_path"] = save_path
