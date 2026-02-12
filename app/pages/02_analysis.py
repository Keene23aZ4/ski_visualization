import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from mpl_toolkits.mplot3d import Axes3D

st.title("姿勢分析ページ")

# ----------------------------
# pose_seq 読み込み
# ----------------------------
if "pose_seq" not in st.session_state:
    st.warning("先に姿勢推定を実行してください")
    st.stop()

pose_seq = st.session_state["pose_seq"]
video_path = st.session_state.get("video_path", None)

total_frames = pose_seq.shape[0]

# ----------------------------
# FPS取得
# ----------------------------
fps = 30
if video_path:
    cap = cv2.VideoCapture(video_path)
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps_val > 0:
        fps = fps_val

video_duration = total_frames / fps

# ----------------------------
# 動画表示
# ----------------------------
if video_path:
    st.subheader("元動画")
    video_file = open(video_path, "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)

# ----------------------------
# 再生コントロール（1か所のみ）
# ----------------------------
st.subheader("再生コントロール")

col1, col2 = st.columns(2)

with col1:
    play = st.button("▶ 再生", key="play_main")

with col2:
    stop = st.button("■ 停止", key="stop_main")

if "playing" not in st.session_state:
    st.session_state.playing = False

if play:
    st.session_state.playing = True

if stop:
    st.session_state.playing = False

# ----------------------------
# 再生位置スライダー（1つだけ）
# ----------------------------
if "current_time" not in st.session_state:
    st.session_state.current_time = 0.0

if not st.session_state.playing:
    st.session_state.current_time = st.slider(
        "再生位置（秒）",
        0.0,
        float(video_duration),
        float(st.session_state.current_time),
        0.01,
        key="time_slider"
    )

current_frame = int(st.session_state.current_time * fps)
current_frame = min(current_frame, total_frames - 1)

# ----------------------------
# 自動再生処理
# ----------------------------
if st.session_state.playing:
    st.session_state.current_time += 1.0 / fps

    if st.session_state.current_time >= video_duration:
        st.session_state.current_time = video_duration
        st.session_state.playing = False

    st.rerun()


f = pose_seq[current_frame]

# ----------------------------
# 2Dスケルトン表示
# ----------------------------
st.subheader("2Dスケルトン")

fig2d, ax2d = plt.subplots(figsize=(5, 7))
ax2d.scatter(f[:, 0], -f[:, 1], s=20)

# MediaPipe接続（手動定義）
connections = [
    (11,13),(13,15),
    (12,14),(14,16),
    (11,12),
    (11,23),(12,24),
    (23,24),
    (23,25),(25,27),
    (24,26),(26,28)
]

for i, j in connections:
    ax2d.plot([f[i,0], f[j,0]],
              [-f[i,1], -f[j,1]])

ax2d.set_aspect("equal")
ax2d.axis("off")
st.pyplot(fig2d)

# ----------------------------
# 3Dスケルトン表示
# ----------------------------
st.subheader("3Dスケルトン")

fig3d = plt.figure(figsize=(6, 8))
ax3d = fig3d.add_subplot(111, projection="3d")

ax3d.scatter(f[:,0], f[:,1], f[:,2])

for i, j in connections:
    ax3d.plot([f[i,0], f[j,0]],
              [f[i,1], f[j,1]],
              [f[i,2], f[j,2]])

ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")
ax3d.view_init(elev=10, azim=-90)

st.pyplot(fig3d)

# ----------------------------
# ターン時間解析（簡易例）
# ----------------------------
st.subheader("ターン時間解析")

turn_segments = []
threshold = 0.02

for i in range(1, total_frames):
    diff = abs(pose_seq[i, 23, 0] - pose_seq[i-1, 23, 0])
    if diff > threshold:
        turn_segments.append(i)

turn_times = []

for idx in turn_segments:
    duration_sec = 1.0 / fps
    turn_times.append({
        "frame": idx,
        "time_sec": duration_sec
    })

if turn_times:
    import pandas as pd
    df_turn = pd.DataFrame(turn_times)
    st.dataframe(df_turn)

