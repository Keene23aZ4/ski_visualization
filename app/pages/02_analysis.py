import streamlit as st
import numpy as np

from core.pose.run_pose import run_mediapipe
from core.pose.mediapipe_adapter import extract_frames_from_mediapipe

import matplotlib.pyplot as plt

st.header("② 姿勢推定")

# ----------------------------
# MediaPipe Pose connections（公式定義）
# ----------------------------
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),
    (11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (11,23),(12,24),
    (23,24),
    (23,25),(25,27),(27,29),(27,31),
    (24,26),(26,28),(28,30),(28,32),
]

# ----------------------------
# 動画パス取得
# ----------------------------
video_path = st.session_state.get("video_path")

# ----------------------------
# ① 姿勢推定（重い処理：ボタンで1回だけ）
# ----------------------------
if video_path and st.button("姿勢推定を実行"):
    with st.spinner("姿勢推定を実行中..."):
        results = run_mediapipe(video_path)
        frames = extract_frames_from_mediapipe(results)
        pose_seq = np.stack(frames)  # (T,33,3)

        st.session_state["pose_seq"] = pose_seq

    st.success("姿勢推定が完了しました")
    st.write("pose_seq shape:", pose_seq.shape)

# ----------------------------
# ② 表示UI（常に表示・再実行OK）
# ----------------------------
pose_seq = st.session_state.get("pose_seq")

if pose_seq is not None:
    st.subheader("フレーム指定表示（スライダー）")

    T = pose_seq.shape[0]

    frame_idx = st.slider(
        "Frame",
        min_value=0,
        max_value=T - 1,
        value=0,
        step=1
    )

    f = pose_seq[frame_idx]  # (33,3)

    fig, ax = plt.subplots(figsize=(4, 6))

    # 点
    ax.scatter(f[:, 0], -f[:, 1], s=20)

    # 骨格ライン
    for i, j in POSE_CONNECTIONS:
        ax.plot(
            [f[i, 0], f[j, 0]],
            [-f[i, 1], -f[j, 1]],
            linewidth=2
        )

    ax.set_title(f"Pose Skeleton (frame {frame_idx})")
    ax.axis("equal")
    ax.axis("off")

    st.pyplot(fig)
    plt.close(fig)
