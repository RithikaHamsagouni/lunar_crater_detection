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
def draw_craters(image, results):
    img = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    color_map = {
        "fresh": (0, 255, 0),
        "degraded": (0, 255, 255),
        "heavily_degraded": (0, 165, 255),
        "overlapping": (0, 0, 255),
        "uncertain": (255, 0, 0),
    }

    for cls_result, inst in results:
        cx = int(inst["centroid_x"])
        cy = int(inst["centroid_y"])
        r = int(inst.get("radius", 10))

        crater_type = cls_result["crater_type"]
        color = color_map.get(crater_type, (255, 255, 255))

        cv2.circle(img, (cx, cy), r, color, 2)
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)

    return img


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.title("🌕 Crater Detection System")
st.write("Upload a lunar image and detect craters visually")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image).astype(np.float32)

    st.image(image, caption="Original Image", use_column_width=True)

    with st.spinner("Loading model..."):
        cfg = Config()
        model, schedule = load_model("checkpoints/best_model.pt", cfg)
        prob_head = ProbabilityHead(model, cfg).to(cfg.DEVICE)
        pipeline = CraterPipeline(cfg, cnn_model=None)

    with st.spinner("Detecting craters..."):
        results, img_np = run_inference_on_image(
            image_np, model, schedule, prob_head, pipeline, cfg
        )

        vis_img = draw_craters(img_np, results)

    st.image(vis_img, caption="Detected Craters", use_column_width=True)

    st.success(f"Detected {len(results)} craters")

    for cls_result, inst in results:
        st.write(f"{cls_result['crater_type']} crater at ({int(inst['centroid_x'])}, {int(inst['centroid_y'])})")