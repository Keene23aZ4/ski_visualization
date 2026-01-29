import streamlit as st
import streamlit.components.v1 as components

st.header("③ 結果表示")

if "keypoints" not in st.session_state:
    st.warning("解析結果がありません")
    st.stop()

st.subheader("3D表示（360°回転）")

components.html(
    open("app/frontend/threejs/index.html").read(),
    height=600,
)
