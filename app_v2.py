"""AOI app v2: board localization + sliding-window CNN with background class (see pcb_common.py).
Run: streamlit run app_v2.py   (needs pcb_cnn_model_v2.keras next to it)"""
import cv2, numpy as np, streamlit as st
from pathlib import Path
from PIL import Image
from pcb_common import detect, localize_board
from gemini_review import review, find_defects

st.set_page_config(page_title="AOI Terminal | V2", layout="wide")
conf = st.sidebar.slider("Confidence threshold (%)", 30, 95, 50) / 100
use_gemini = st.sidebar.checkbox("Detailed defect review")
src = st.sidebar.radio("Input", ("Manual Upload", "Live Camera"))
st.title("Automated Optical Inspection (AOI) v2")
st.warning("Prototype. Detections are only searched inside the localized PCB; manual verification is required.")


@st.cache_resource
def load():
    import tensorflow as tf
    return tf.keras.models.load_model(str(Path(__file__).resolve().parent / "pcb_cnn_model_v2.keras"))


model = load()
f = st.file_uploader("Upload PCB image", type=["png", "jpg", "jpeg"]) if src == "Manual Upload" else st.camera_input("Scanner")
if f:
    bgr = cv2.cvtColor(np.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR)
    loc = localize_board(bgr)
    if loc is None:
        st.error("No PCB found in the image. Reframe so the board fills most of the view.")
    else:
        dets = detect(model, bgr, conf)
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
        x, y, w, h, _ = loc
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 3)
        for dx, dy, dw, dh, name, c in dets:
            cv2.rectangle(out, (dx, dy), (dx + dw, dy + dh), (255, 0, 0), 2)
            cv2.putText(out, f"{name} {c:.0%}", (dx, max(dy - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        gdets = []
        if use_gemini:
            with st.spinner("Scanning image for faults..."):
                try:
                    gdets = find_defects(bgr)
                except Exception:
                    st.error("Detailed scan could not be completed. Please try again.")
        for gx, gy, gw, gh, gn, gc, _ in gdets:
            cv2.rectangle(out, (gx, gy), (gx + gw, gy + gh), (0, 120, 255), 3)
            cv2.putText(out, f"{gn} {gc:.0%}", (gx, max(gy - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 2)
        c1, c2 = st.columns(2)
        c1.metric("Model detections (red, inside board)", len(dets))
        c2.metric("Detailed scan faults (orange)", len(gdets))
        st.image(out, use_container_width=True)
        if gdets:
            st.subheader("Detailed scan faults")
            st.dataframe([{"fault": g[4], "confidence": round(g[5] * 100, 1), "x": g[0], "y": g[1], "w": g[2], "h": g[3], "why": g[6]} for g in gdets])
        if use_gemini:
            with st.spinner("Analyzing image..."):
                try:
                    st.subheader("Detailed review")
                    st.write(review(cv2.cvtColor(out, cv2.COLOR_RGB2BGR), dets))
                except Exception as e:
                    st.error("Detailed review could not be completed. Please try again.")
        if dets:
            st.dataframe([{"class": d[4], "confidence": round(d[5] * 100, 1), "x": d[0], "y": d[1], "w": d[2], "h": d[3]} for d in dets])
