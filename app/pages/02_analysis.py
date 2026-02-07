import streamlit as st
import numpy as np

from core.pose.run_pose import run_mediapipe
from core.pose.mediapipe_adapter import extract_frames_from_mediapipe

import matplotlib.pyplot as plt
def calc_angle(a, b, c):
    """
    a, b, c : np.array shape (2,) or (3,)
    b を頂点とする角度（deg）
    """
    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle


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
    # ----------------------------
    # 関節角度計算（2D: x,y）
    # ----------------------------
    LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
    RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28
    
    lh = f[LEFT_HIP][:2]
    lk = f[LEFT_KNEE][:2]
    la = f[LEFT_ANKLE][:2]
    
    rh = f[RIGHT_HIP][:2]
    rk = f[RIGHT_KNEE][:2]
    ra = f[RIGHT_ANKLE][:2]
    
    left_knee_angle = calc_angle(lh, lk, la)
    right_knee_angle = calc_angle(rh, rk, ra)
    
    left_hip_angle = calc_angle(lk, lh, f[11][:2])    # 膝-股-肩
    right_hip_angle = calc_angle(rk, rh, f[12][:2])   # 膝-股-肩
    
    # 表示
    st.subheader("関節角度（deg）")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("左膝", f"{left_knee_angle:.1f}")
        st.metric("左股", f"{left_hip_angle:.1f}")
    
    with col2:
        st.metric("右膝", f"{right_knee_angle:.1f}")
        st.metric("右股", f"{right_hip_angle:.1f}")
    # ----------------------------
    # 全フレームの角度を計算
    # ----------------------------
    T = pose_seq.shape[0]
    
    left_knee_angles = []
    right_knee_angles = []
    left_hip_angles = []
    right_hip_angles = []
    
    for t in range(T):
        f_t = pose_seq[t]
    
        lh = f_t[23][:2]
        lk = f_t[25][:2]
        la = f_t[27][:2]
    
        rh = f_t[24][:2]
        rk = f_t[26][:2]
        ra = f_t[28][:2]
    
        left_knee_angles.append(calc_angle(lh, lk, la))
        right_knee_angles.append(calc_angle(rh, rk, ra))
    
        left_hip_angles.append(calc_angle(lk, lh, f_t[11][:2]))
        right_hip_angles.append(calc_angle(rk, rh, f_t[12][:2]))
    
    left_knee_angles = np.array(left_knee_angles)
    right_knee_angles = np.array(right_knee_angles)
    left_hip_angles = np.array(left_hip_angles)
    right_hip_angles = np.array(right_hip_angles)

    import matplotlib.pyplot as plt

    st.subheader("関節角度の時系列")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(left_knee_angles, label="Left Knee")
    ax.plot(right_knee_angles, label="Right Knee")
    ax.plot(left_hip_angles, label="Left Hip", linestyle="--")
    ax.plot(right_hip_angles, label="Right Hip", linestyle="--")
    
    # スライダーフレーム位置を表示
    ax.axvline(frame_idx, color="k", linestyle=":", linewidth=2)
    
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle (deg)")
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)





