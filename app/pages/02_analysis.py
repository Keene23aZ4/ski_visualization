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
    from mediapipe.tasks.python.vision import PoseLandmarksConnections
    
    # 1フレーム取り出し
    f = pose_seq[0]  # (33,3)
    
    plt.figure(figsize=(5, 7))
    
    # 点を描く
    plt.scatter(f[:, 0], -f[:, 1], s=20)
    
    # 骨格ラインを描く
    for i, j in PoseLandmarksConnections:
        x = [f[i, 0], f[j, 0]]
        y = [-f[i, 1], -f[j, 1]]
        plt.plot(x, y, linewidth=2)
    
    plt.title("Pose Skeleton (frame 0)")
    plt.axis("equal")
    plt.axis("off")
    
    st.pyplot(plt)
