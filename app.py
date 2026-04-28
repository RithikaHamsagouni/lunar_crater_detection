import streamlit as st
import numpy as np
import cv2
from PIL import Image

from config import Config
from inference import load_model, run_inference_on_image
from probability_head import ProbabilityHead
from crater_classifier import CraterPipeline


# ─────────────────────────────────────────────
# Draw craters with colors
# ─────────────────────────────────────────────
def draw_craters(image_np, results):
    img = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    color_map = {
        "fresh":            (0, 255, 0),
        "degraded":         (0, 255, 255),
        "heavily_degraded": (0, 165, 255),
        "overlapping":      (0, 0, 255),
        "uncertain":        (80, 80, 255),
    }

    for i, (cls_result, inst) in enumerate(results):
        cx = int(inst["centroid_x"])
        cy = int(inst["centroid_y"])
        r  = int(inst["radius_px"])              # ← fixed: was inst.get("radius", 10)

        crater_type = cls_result.get("crater_type", "uncertain")
        color = color_map.get(crater_type, (255, 255, 255))
        p = inst["p_crater"]

        cv2.circle(img, (cx, cy), r, color, 2)
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(img, f"#{i+1} {crater_type[:3]} {p:.2f}",
                    (max(0, cx - r), max(12, cy - r - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return img


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.title("🌕 Crater Detection System")
st.write("Upload a lunar/Mars image and detect craters visually.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("L")
    image_np  = np.array(image_pil).astype(np.float32)

    st.image(image_pil, caption="Original Image", use_column_width=True)

    with st.spinner("Loading model..."):
        cfg = Config()
        model, schedule = load_model("checkpoints/best_model.pt", cfg)
        prob_head = ProbabilityHead(model, cfg).to(cfg.DEVICE)
        pipeline  = CraterPipeline(cfg, cnn_model=None)

    with st.spinner("Detecting craters..."):
        results, img_np = run_inference_on_image(
            image_np, model, schedule, prob_head, pipeline, cfg
        )

    if results:
        vis_img = draw_craters(img_np, results)
        st.image(vis_img, caption=f"Detected {len(results)} Crater(s)", use_column_width=True)
        st.success(f"✅ Detected {len(results)} craters")

        st.subheader("Crater Details")
        for i, (cls_result, inst) in enumerate(results):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"Crater #{i+1}", cls_result.get("crater_type", "?"))
            col2.metric("P(crater)",  f"{inst['p_crater']:.3f}")
            col3.metric("Radius (px)", f"{inst['radius_px']:.1f}")
            col4.metric("Age", cls_result.get("age_estimate", "?"))
    else:
        st.warning("⚠️ No craters detected in this image.")
        st.image(image_pil, caption="Input image (no detections)", use_column_width=True)