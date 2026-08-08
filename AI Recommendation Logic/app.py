"""
🖼️ CYBER EYE – Image Recognition Module
YOLOv8 Object Detection + EasyOCR Text Extraction
"""

import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="CYBER EYE – Vision AI",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# CYBERPUNK CSS (same as main suite)
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
        .stApp {
            background: #0a0c10;
            background-image: 
                radial-gradient(circle at 20% 50%, rgba(0, 255, 225, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 80% 50%, rgba(255, 0, 204, 0.05) 0%, transparent 50%),
                repeating-linear-gradient(0deg, rgba(0,255,225,0.02) 0px, rgba(0,255,225,0.02) 1px, transparent 1px, transparent 3px);
            font-family: 'Share Tech Mono', monospace !important;
            color: #b0e0ff !important;
        }
        h1, h2, h3, .main-header {
            font-family: 'Orbitron', sans-serif !important;
            font-weight: 900 !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 1.6rem !important;
            background: linear-gradient(90deg, #00ffe1, #ff00cc, #00ffe1);
            background-size: 300% 100%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: neonFlow 4s ease-in-out infinite;
        }
        @keyframes neonFlow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .sub-header {
            font-family: 'Share Tech Mono', monospace;
            color: #00ffe1 !important;
            text-shadow: 0 0 20px #00ffe1, 0 0 40px #ff00cc;
            letter-spacing: 2px;
            font-size: 0.9rem !important;
            border-left: 3px solid #ff00cc;
            padding-left: 12px;
        }
        .stButton > button {
            font-family: 'Orbitron', sans-serif !important;
            background: transparent !important;
            border: 2px solid #00ffe1 !important;
            color: #00ffe1 !important;
            box-shadow: 0 0 20px rgba(0, 255, 225, 0.2);
            transition: all 0.3s ease !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            width: 100%;
            min-height: 44px !important;
        }
        .stButton > button:hover {
            background: #00ffe1 !important;
            color: #0a0c10 !important;
            box-shadow: 0 0 60px #00ffe1;
        }
        .stFileUploader > div {
            border: 2px dashed #00ffe1 !important;
            background: rgba(0, 255, 225, 0.05) !important;
            border-radius: 12px;
        }
        .footer {
            text-align: center;
            font-family: 'Share Tech Mono', monospace;
            color: #ff00cc;
            font-size: 0.7rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px dashed #00ffe1;
            animation: glitch 2s infinite;
        }
        @keyframes glitch {
            0% { opacity: 1; }
            95% { opacity: 1; }
            96% { opacity: 0; }
            97% { opacity: 1; }
            98% { opacity: 0; }
            100% { opacity: 1; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 8px 0;">
            <span style="font-family: 'Orbitron', sans-serif; font-size: 1.5rem; color: #ff00cc; text-shadow: 0 0 30px #ff00cc;">
                ⚡ CYBER EYE ⚡
            </span>
            <div style="font-family: 'Share Tech Mono', monospace; color: #00ffe1; font-size: 0.7rem; border-bottom: 1px solid #00ffe1; padding-bottom: 8px;">
                [ VISION MODULE v1.0 ]
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("🛡️ FIREWALL: ACTIVE")
    st.caption("🔍 SCANNER: READY")
    st.caption("🎮 PLAYER: SANKALP")

# ----------------------------------------------------------------------
# MAIN TITLE
# ----------------------------------------------------------------------
st.markdown('<div class="main-header">🖼️ CYBER EYE (VISION)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">// YOLOv8 OBJECT DETECTION + EASYOCR TEXT EXTRACTION</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# LAZY LOAD HEAVY LIBRARIES
# ----------------------------------------------------------------------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

if not YOLO_AVAILABLE:
    st.warning("⚠️ YOLO not installed. Run: pip install ultralytics")
if not EASYOCR_AVAILABLE:
    st.warning("⚠️ EasyOCR not installed. Run: pip install easyocr")

# ----------------------------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------------------------
uploaded_file = st.file_uploader("📂 UPLOAD IMAGE", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        st.error("❌ OpenCV (cv2) not installed. Run: pip install opencv-python-headless")
        st.stop()

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="ORIGINAL", use_container_width=True)

    col_det, col_ocr = st.columns(2)

    # ---------- OBJECT DETECTION ----------
    with col_det:
        st.subheader("🔍 OBJECT DETECTION")
        if YOLO_AVAILABLE:
            with st.spinner("SCANNING..."):
                model = YOLO("yolov8n.pt")  # downloads automatically
                results = model(image)
                annotated = results[0].plot()
                st.image(annotated, caption="DETECTED OBJECTS", use_container_width=True)

                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    st.caption("**Detections:**")
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = model.names[cls]
                        st.write(f"- {label} (confidence: {conf:.2f})")
                else:
                    st.info("No objects detected.")
        else:
            st.warning("YOLO offline – install ultralytics")

    # ---------- TEXT EXTRACTION ----------
    with col_ocr:
        st.subheader("📝 TEXT EXTRACTION")
        if EASYOCR_AVAILABLE:
            with st.spinner("DECRYPTING..."):
                reader = easyocr.Reader(["en"])
                img_np = np.array(image)
                ocr_result = reader.readtext(img_np, detail=0)
                if ocr_result:
                    st.success(f"EXTRACTED {len(ocr_result)} TEXT BLOCK(S):")
                    for text in ocr_result:
                        st.write(f"- {text}")
                else:
                    st.info("No text found in image.")
        else:
            st.warning("OCR offline – install easyocr")

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        ⚡ CYBER EYE v1.0 || PRACTICAL TRAINING - II || SANKALP SHARMA ⚡
    </div>
    """,
    unsafe_allow_html=True,
)
