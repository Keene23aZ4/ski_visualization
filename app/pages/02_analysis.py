import streamlit as st
import numpy as np
import cv2
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
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
 

        # session_state に保存
        st.session_state["pose_seq"] = pose_seq
        st.session_state["left_knee_angles"] = np.array(left_knee_angles)
        st.session_state["right_knee_angles"] = np.array(right_knee_angles)
        st.session_state["left_hip_angles"] = np.array(left_hip_angles)
        st.session_state["right_hip_angles"] = np.array(right_hip_angles)
        # pose_seq が定義された直後
        frame_idx = st.slider(
            "表示フレーム",
            0,
            pose_seq.shape[0] - 1,
            0
        )

    st.success("姿勢推定が完了しました")
    st.write("pose_seq shape:", pose_seq.shape)

# ----------------------------
# ② 表示UI（常に表示・再実行OK）
# ----------------------------
pose_seq = st.session_state.get("pose_seq")

if pose_seq is not None:
    fps = st.session_state.get("fps", 30)
    T = pose_seq.shape[0]
    duration_sec = T / fps
    current_time = st.slider(
        "再生位置（秒）",
        min_value=0.0,
        max_value=float(duration_sec),
        value=0.0,
        step=1.0 / fps
    )
    frame_idx = int(current_time * fps)
    frame_idx = min(frame_idx, T - 1)
    



    st.subheader("元動画")

    video_path = st.session_state.get("video_path")
    if video_path:
        st.video(video_path, start_time=int(current_time))

    st.subheader("再生制御")
    play = st.button("▶ 元動画と姿勢を再生")
    import time

    fps = st.session_state.get("fps", 30)
    interval = 1.0 / fps
    
    placeholder = st.empty()
    
    if play:
        for t in range(pose_seq.shape[0]):
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection="3d")
    
            f = pose_seq[t]
            x, y, z = f[:, 0], -f[:, 1], -f[:, 2]
    
            ax.scatter(x, y, z, s=20)
            for i, j in POSE_CONNECTIONS:
                ax.plot([x[i], x[j]],
                        [y[i], y[j]],
                        [z[i], z[j]])
    
            ax.axis("off")
            ax.set_title(f"Frame {t}")
    
            placeholder.pyplot(fig)
            plt.close(fig)
    
            time.sleep(interval)



    st.subheader("フレーム指定表示（スライダー）")
    left_knee_angles = st.session_state["left_knee_angles"]
    right_knee_angles = st.session_state["right_knee_angles"]
    left_hip_angles = st.session_state["left_hip_angles"]
    right_hip_angles = st.session_state["right_hip_angles"]


    T = pose_seq.shape[0]


    f = pose_seq[frame_idx]  # (33,3)
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    
    x = f[:, 0]
    y = -f[:, 1]
    z = -f[:, 2]
    
    ax.scatter(x, y, z, s=20)
    ax.set_title(f"3D Pose (frame {frame_idx})")
    
    st.pyplot(fig)
    st.subheader("アニメーション再生")

    play = st.button("▶ 再生")

    if play:
        fps = st.session_state.get("fps", 30)
        interval = 1.0 / fps
    
        placeholder = st.empty()
    
        for t in range(T):
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="3d")
    
            f = pose_seq[t]
            x, y, z = f[:, 0], -f[:, 1], -f[:, 2]
    
            ax.scatter(x, y, z, s=20)
    
            for i, j in POSE_CONNECTIONS:
                ax.plot([x[i], x[j]],
                        [y[i], y[j]],
                        [z[i], z[j]])
    
            ax.set_title(f"Frame {t}")
            ax.axis("off")
    
            placeholder.pyplot(fig)
            plt.close(fig)
    
            time.sleep(interval)  # ★ 元動画fps同期

    
    placeholder = st.empty()
    
    if play:
        for t in range(0, T, 2):  # 2フレーム刻み
            f = pose_seq[t]
    
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="3d")
    
            x = f[:, 0]
            y = -f[:, 1]
            z = -f[:, 2]
    
            ax.scatter(x, y, z, s=20)
    
            for i, j in POSE_CONNECTIONS:
                ax.plot([x[i], x[j]],
                        [y[i], y[j]],
                        [z[i], z[j]])
    
            ax.set_title(f"Frame {t}")
            ax.axis("off")
    
            placeholder.pyplot(fig)
            plt.close(fig)

    POSE_CONNECTIONS = [
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 12),
        (23, 24),
        (11, 23), (12, 24),
        (23, 25), (25, 27),
        (24, 26), (26, 28),
    ]
    
    for i, j in POSE_CONNECTIONS:
        ax.plot(
            [x[i], x[j]],
            [y[i], y[j]],
            [z[i], z[j]],
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




    st.subheader("関節角度の時系列")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(left_knee_angles, label="Left Knee")
    ax.plot(right_knee_angles, label="Right Knee")
    ax.plot(left_hip_angles, label="Left Hip", linestyle="--")
    ax.plot(right_hip_angles, label="Right Hip", linestyle="--")
    # ----------------------------
    # 左右差（Left - Right）
    # ----------------------------
    knee_diff = left_knee_angles - right_knee_angles
        
    # 符号
    sign = np.sign(knee_diff)
    
    # 符号が変わったフレーム = ターン境界
    turn_change_idx = np.where(np.diff(sign) != 0)[0] + 1
    turn_segments = []
    
    
    for i in range(len(turn_change_idx) - 1):
        start = turn_change_idx[i]
        end = turn_change_idx[i + 1]
    
        mean_diff = knee_diff[start:end].mean()
        label = "Left Turn" if mean_diff > 0 else "Right Turn"
    
        turn_segments.append({
            "start": start,
            "end": end,
            "label": label,
            "mean_diff": mean_diff
        })
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if fps <= 0:
        fps = 30  # 保険
    
    st.session_state["fps"] = fps
    st.write(f"Video FPS: {fps}")
    # turn_segments ができたあと
    turn_times = []
    
    for turn in turn_segments:
        duration_frames = turn["end"] - turn["start"]
        duration_sec = duration_frames / fps
    
        turn_times.append({
            "label": turn["label"],
            "start": turn["start"],
            "end": turn["end"],
            "frames": duration_frames,
            "time_sec": duration_sec
        })
    df_turn = pd.DataFrame(turn_times)
    st.subheader("ターン時間一覧")
    st.dataframe(df_turn)
    
        
    hip_diff = left_hip_angles - right_hip_angles

    
    # ======================================
    # ターン可視化（④）
    # ======================================
    fig, ax = plt.subplots(figsize=(10, 3))
    
    ax.plot(knee_diff, color="black", label="Knee L-R")
    ax.axhline(0, linestyle="--", color="gray")
    
    for turn in turn_segments:
        color = "lightblue" if turn["label"] == "Left Turn" else "lightcoral"
        ax.axvspan(turn["start"], turn["end"], color=color, alpha=0.3)
    
    ax.axvline(frame_idx, linestyle=":", color="k")
    
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle difference (deg)")
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)

    st.subheader("左右差（Left − Right）")
    
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    
    ax2.plot(knee_diff, label="Knee (L-R)")
    ax2.plot(hip_diff, label="Hip (L-R)", linestyle="--")
    
    ax2.axhline(0, color="gray", linestyle=":")
    ax2.axvline(frame_idx, color="k", linestyle=":", linewidth=2)
    
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Angle difference (deg)")
    ax2.legend()
    ax2.grid(True)
    
    st.pyplot(fig2)










