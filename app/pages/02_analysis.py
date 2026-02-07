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

    import matplotlib.pyplot as plt
    # MediaPipe Pose connections（公式定義）
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

    import mediapipe as mp
    
    f = pose_seq[0]  # (33,3)

    plt.figure(figsize=(5, 7))
    
    plt.scatter(f[:, 0], -f[:, 1], s=20)
    
    for i, j in POSE_CONNECTIONS:
        x = [f[i, 0], f[j, 0]]
        y = [-f[i, 1], -f[j, 1]]
        plt.plot(x, y, linewidth=2)
    
    plt.title("Pose Skeleton (frame 0)")
    plt.axis("equal")
    plt.axis("off")
    
    st.pyplot(plt)
    import time
    import matplotlib.pyplot as plt
    
    # アニメーション表示用プレースホルダ
    placeholder = st.empty()
    
    T = pose_seq.shape[0]
    
    for t in range(T):
        f = pose_seq[t]
    
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
    
        ax.set_title(f"Frame {t}")
        ax.axis("equal")
        ax.axis("off")
    
        # Streamlitに描画
        placeholder.pyplot(fig)
    
        plt.close(fig)
        time.sleep(0.03)  # 約30fps
        st.subheader("フレーム指定表示（スライダー）")

    T = pose_seq.shape[0]
    
    # フレーム番号スライダー
    frame_idx = st.slider(
        "Frame",
        min_value=0,
        max_value=T - 1,
        value=0,
        step=1
    )
    
    f = pose_seq[frame_idx]
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(4, 6))
    
    plt.scatter(f[:, 0], -f[:, 1], s=20)
    
    for i, j in POSE_CONNECTIONS:
        plt.plot(
            [f[i, 0], f[j, 0]],
            [-f[i, 1], -f[j, 1]],
            linewidth=2
        )
    
    plt.title(f"Pose Skeleton (frame {frame_idx})")
    plt.axis("equal")
    plt.axis("off")
    
    st.pyplot(plt)





