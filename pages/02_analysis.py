import streamlit as st
from core.pose.mediapipe_pose import MediaPipePose

st.header("② 解析")

if "video_path" not in st.session_state:
    st.warning("先に動画をアップロードしてください")
    st.stop()

pose_model = st.selectbox("姿勢推定モデル", ["mediapipe"])

if st.button("解析開始"):
    estimator = MediaPipePose()
    keypoints = estimator.estimate(st.session_state["video_path"])
    st.session_state["keypoints"] = keypoints
    st.success("解析完了")
