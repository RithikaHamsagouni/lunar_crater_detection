# app.py
"""
Streamlit crater detection app
Fixed for:
- Proper model loading
- Safe checkpoint handling
- Green crater boundary
- Red center dot
- Detailed crater stats
"""

import streamlit as st
import numpy as np
import cv2
import os

from PIL import Image

from config import Config
from inference import (
    load_model,
    run_inference_on_image,
    save_output_image,
    norm_uint8,
)

from probability_head import ProbabilityHead
from crater_classifier import CraterPipeline


# ─────────────────────────────────────────────
# Draw craters visually
# ─────────────────────────────────────────────
def draw_craters(image_np, results):
    img = cv2.cvtColor(
        norm_uint8(image_np),
        cv2.COLOR_GRAY2BGR
    )

    color_map = {
        "fresh": (0, 255, 0),
        "degraded": (0, 255, 255),
        "heavily_degraded": (0, 165, 255),
        "overlapping": (255, 0, 0),
        "uncertain": (180, 180, 180),
    }

    for i, (cls_result, inst) in enumerate(results):
        cx = int(inst["centroid_x"])
        cy = int(inst["centroid_y"])
        r = max(5, int(inst["radius_px"]))

        crater_type = cls_result.get(
            "crater_type",
            "uncertain"
        )

        color = color_map.get(
            crater_type,
            (255, 255, 255)
        )

        # Circle
        cv2.circle(
            img,
            (cx, cy),
            r,
            color,
            2
        )

        # Red center
        cv2.circle(
            img,
            (cx, cy),
            3,
            (0, 0, 255),
            -1
        )

        # Label
        cv2.putText(
            img,
            f"#{i+1}",
            (max(0, cx-r), max(15, cy-r-5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    return img


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Crater Detection",
    layout="wide"
)

st.title("🌕 Lunar / Mars Crater Detection System")

st.write(
    "Upload grayscale lunar or Mars imagery to detect craters."
)

uploaded_file = st.file_uploader(
    "Upload image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image_pil = Image.open(
        uploaded_file
    ).convert("L")

    image_np = np.array(
        image_pil
    ).astype(np.float32)

    st.image(
        image_pil,
        caption="Original Image",
        use_container_width=True
    )

    cfg = Config()

    checkpoint_path = os.path.join(
        cfg.CHECKPOINT_DIR,
        "best_model.pt"
    )

    if not os.path.exists(checkpoint_path):
        st.error(
            f"Checkpoint not found: {checkpoint_path}"
        )
        st.stop()

    # Load
    with st.spinner("Loading crater model..."):
        model, schedule = load_model(
            checkpoint_path,
            cfg
        )

        prob_head = ProbabilityHead(
            model,
            cfg
        ).to(cfg.DEVICE)

        pipeline = CraterPipeline(
            cfg,
            cnn_model=None
        )

    # Detect
    with st.spinner("Detecting craters..."):
        results, _ = run_inference_on_image(
            image_np,
            model,
            schedule,
            prob_head,
            pipeline,
            cfg
        )

    if results:
        vis_img = draw_craters(
            image_np,
            results
        )

        st.image(
            vis_img,
            caption=f"Detected {len(results)} Crater(s)",
            channels="BGR",
            use_container_width=True
        )

        st.success(
            f"✅ Detected {len(results)} crater(s)"
        )

        # Save outputs
        save_output_image(
            image_np,
            results,
            uploaded_file.name,
            output_dir=cfg.OUTPUT_DIR
        )

        st.subheader("Crater Details")

        for i, (cls_result, inst) in enumerate(results):
            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                f"Crater #{i+1}",
                cls_result.get("crater_type", "?")
            )

            c2.metric(
                "P(crater)",
                f"{inst['p_crater']:.3f}"
            )

            c3.metric(
                "Radius(px)",
                f"{inst['radius_px']:.1f}"
            )

            c4.metric(
                "Age",
                cls_result.get("age_estimate", "?")
            )

    else:
        st.warning("⚠️ No craters detected.")

        st.image(
            image_pil,
            caption="No detections",
            use_container_width=True
        )

        os.makedirs(
            cfg.OUTPUT_DIR,
            exist_ok=True
        )

        basename = os.path.splitext(
            uploaded_file.name
        )[0]

        out_path = os.path.join(
            cfg.OUTPUT_DIR,
            f"{basename}_no_detections.jpg"
        )

        Image.fromarray(
            norm_uint8(image_np)
        ).save(out_path)