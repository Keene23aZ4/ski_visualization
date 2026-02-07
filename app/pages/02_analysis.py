import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from core.pose.run_pose import run_mediapipe
from core.pose.mediapipe_adapter import extract_frames_from_mediapipe


# ======================================
# ヘッダ
# ======================================
st.header("② 姿勢推定・可視化")


# ======================================
# 動画パス取得
# ======================================
video_path = st.session_state.get("video_path")

if video_path is None:
    st.info("①で動画をアップロードしてください")
    st.stop()


# ======================================
# MediaPipe 実行（1回だけ）
# ======================================
if "pose_seq" not in st.session_state:

    if st.button("姿勢推定を実行"):

        with st.spinner("姿勢推定中..."):
            results = run_mediapipe(video_path)
            frames = extract_frames_from_mediapipe(results)

        # ---------- shape確認 ----------
        st.write("num frames:", len(frames))
        st.write("frame shape:", frames[0].shape)  # (33,3)

        # ---------- (T,33,3) ----------
        pose_seq = np.stack(frames)
        st.write("pose_seq shape:", pose_seq.shape)

        # session_state に保存
        st.session_state["pose_seq"] = pose_seq

        st.success("姿勢推定が完了しました")

        st.stop()


# ======================================
# ここからは「可視化・分析」
# ======================================
pose_seq = st.session_state["pose_seq"]
T = pose_seq.shape[0]


# ======================================
# フレーム指定（スライダー1個だけ）
# ======================================
frame_idx = st.slider(
    "フレーム指定表示",
    min_value=0,
    max_value=T - 1,
    value=0,
    key="frame_slider"
)


# ======================================
# 骨格描画（MediaPipe Tasks 用）
# ======================================
from mediapipe.tasks.python.vision import PoseLandmarksConnections

connections = PoseLandmarksConnections


# ======================================
# 可視化
# ======================================
f = pose_seq[frame_idx]  # (33,3)

plt.figure(figsize=(5, 7))

# 関節点
plt.scatter(
    f[:, 0],
    -f[:, 1],
    s=20
)

# 骨格ライン
for i, j in connections:
    x = [f[i, 0], f[j, 0]]
    y = [-f[i, 1], -f[j, 1]]
    plt.plot(x, y, linewidth=2)

plt.title(f"Frame {frame_idx}")
plt.axis("equal")
plt.axis("off")

st.pyplot(plt)


# ======================================
# 参考：全体データを保持
# ======================================
st.session_state["current_frame"] = frame_idx
