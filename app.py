"""
CYBERPUNK AI SUITE – Neon Dream v1.1 (Lazy Load)
Projects: Chatbot · Classification · Recommendation · Vision
"""

import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="CYBERPUNK AI SUITE",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# CYBERPUNK NEON CSS
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
            letter-spacing: 2px;
            background: linear-gradient(90deg, #00ffe1, #ff00cc, #00ffe1);
            background-size: 300% 100%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: neonFlow 4s ease-in-out infinite;
            text-shadow: 0 0 30px rgba(0, 255, 225, 0.3);
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
            letter-spacing: 3px;
            font-size: 1.2rem !important;
            border-left: 4px solid #ff00cc;
            padding-left: 15px;
        }
        
        .css-1d391kg, .css-1adrfps, .stSidebar {
            background: rgba(10, 14, 20, 0.85) !important;
            backdrop-filter: blur(12px);
            border-right: 2px solid rgba(0, 255, 225, 0.3) !important;
            box-shadow: 10px 0 50px rgba(0, 255, 225, 0.1);
        }
        .stSidebar .stRadio label {
            font-family: 'Orbitron', sans-serif !important;
            color: #b0e0ff !important;
            font-size: 0.9rem !important;
            padding: 8px 12px !important;
            border: 1px solid transparent;
            transition: all 0.3s ease;
        }
        .stSidebar .stRadio label:hover {
            border-color: #00ffe1;
            box-shadow: 0 0 25px rgba(0, 255, 225, 0.2);
            background: rgba(0, 255, 225, 0.05);
        }
        .stButton > button {
            font-family: 'Orbitron', sans-serif !important;
            background: transparent !important;
            border: 2px solid #00ffe1 !important;
            color: #00ffe1 !important;
            box-shadow: 0 0 20px rgba(0, 255, 225, 0.2);
            transition: all 0.3s ease !important;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .stButton > button:hover {
            background: #00ffe1 !important;
            color: #0a0c10 !important;
            box-shadow: 0 0 60px #00ffe1, inset 0 0 20px #0a0c10;
            transform: scale(1.03);
            border-color: #00ffe1 !important;
        }
        .stChatMessage {
            background: rgba(20, 30, 50, 0.6) !important;
            backdrop-filter: blur(8px);
            border: 1px solid rgba(0, 255, 225, 0.2) !important;
            border-radius: 16px !important;
            box-shadow: 0 0 30px rgba(0, 255, 225, 0.05);
            padding: 12px 18px !important;
            margin: 8px 0 !important;
        }
        .stChatMessage.user {
            border-color: #ff00cc !important;
            box-shadow: 0 0 30px rgba(255, 0, 204, 0.15);
        }
        .stChatMessage.assistant {
            border-color: #00ffe1 !important;
        }
        
        .stMetric {
            background: rgba(10, 14, 20, 0.7) !important;
            backdrop-filter: blur(4px);
            border: 1px solid rgba(0, 255, 225, 0.2);
            border-radius: 12px;
            padding: 10px 16px;
            transition: all 0.4s ease;
            box-shadow: 0 0 20px rgba(0, 255, 225, 0.05);
        }
        .stMetric:hover {
            border-color: #ff00cc;
            box-shadow: 0 0 50px rgba(255, 0, 204, 0.2);
            transform: translateY(-4px);
        }
        .stMetric label {
            font-family: 'Orbitron', sans-serif !important;
            color: #ff00cc !important;
            font-size: 0.8rem !important;
            letter-spacing: 1px;
        }
        .stMetric [data-testid="stMetricValue"] {
            font-family: 'Orbitron', sans-serif !important;
            color: #00ffe1 !important;
            font-size: 2rem !important;
            text-shadow: 0 0 30px rgba(0, 255, 225, 0.4);
        }
        
        .footer {
            text-align: center;
            font-family: 'Share Tech Mono', monospace;
            color: #ff00cc;
            font-size: 0.8rem;
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px dashed #00ffe1;
            text-shadow: 0 0 10px #ff00cc;
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
        .stFileUploader > div {
            border: 2px dashed #00ffe1 !important;
            background: rgba(0, 255, 225, 0.05) !important;
            border-radius: 12px;
        }
        .stFileUploader > div:hover {
            border-color: #ff00cc !important;
            box-shadow: 0 0 40px rgba(255, 0, 204, 0.2);
        }
        .stSlider [data-baseweb="slider"] {
            accent-color: #ff00cc;
        }
        ::-webkit-scrollbar {
            width: 8px;
            background: #0a0c10;
        }
        ::-webkit-scrollbar-thumb {
            background: #00ffe1;
            border-radius: 10px;
            box-shadow: 0 0 20px #00ffe1;
        }
        ::-webkit-scrollbar-track {
            background: #0a0c10;
            border-left: 1px solid #ff00cc;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# SIDEBAR – GAMIFIED MENU
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 10px 0;">
            <span style="font-family: 'Orbitron', sans-serif; font-size: 1.5rem; color: #ff00cc; text-shadow: 0 0 30px #ff00cc;">
                ⚡ NEON AI ⚡
            </span>
            <div style="font-family: 'Share Tech Mono', monospace; color: #00ffe1; font-size: 0.7rem; border-bottom: 1px solid #00ffe1; padding-bottom: 10px;">
                [ SYSTEM v1.1 ]
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    app_mode = st.radio(
        "SELECT MISSION",
        [
            "💬 CHATBOT",
            "📊 CLASSIFY",
            "💼 RECOMMEND",
            "🖼️ VISION",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-family: 'Share Tech Mono'; color: #b0e0ff; font-size: 0.7rem; text-align: center; border-top: 1px dashed #ff00cc; padding-top: 10px;">
            🛡️ FIREWALL: ACTIVE<br>
            🔥 CPU: 98%<br>
            🎮 PLAYER: SANKALP
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# PROJECT 1: CHATBOT
# ----------------------------------------------------------------------
def project_chatbot():
    st.markdown('<div class="main-header">💬 NEURAL CHAT INTERFACE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">// DETERMINISTIC RULE ENGINE // WHITE-BOX IPO</div>', unsafe_allow_html=True)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_count" not in st.session_state:
        st.session_state.chat_count = 0
    if "chat_closed" not in st.session_state:
        st.session_state.chat_closed = False

    with st.expander("🛡️ READ PROTOCOL", expanded=False):
        st.markdown("""
        **Input:** Sanitize user text  
        **Process:** Explicit `if-elif-else` decision tree  
        **Output:** Traceable response  
        **Intents:** `hello`, `help`, `time`, `about`, `ipo`, `guardrail`, `exit`
        """)

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "intent" in msg and msg["intent"]:
                st.caption(f"🧬 INTENT: {msg['intent']}  |  PATH: {msg.get('path', 'n/a')}")

    user_input = st.chat_input("> ENTER COMMAND...")

    if user_input:
        normalized = user_input.lower().strip()
        if normalized in {"hello", "hi", "hey"}:
            intent, response, path = "greeting", "Hello, hacker. Rule engine engaged.", "greeting"
        elif normalized in {"quit", "exit", "bye"}:
            intent, response, path = "exit", "System shutdown. Session terminated.", "exit"
            st.session_state.chat_closed = True
        elif normalized == "help":
            intent, response, path = "help", "COMMANDS: hello, help, time, about, ipo, guardrail, exit", "help"
        elif normalized == "time":
            intent, response, path = "system_time", f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "time"
        elif normalized == "about":
            intent, response, path = "about", "CYBERPUNK AI v1.0 – Rule-based core with neon UI.", "about"
        elif normalized in {"ipo", "ipo explanation"}:
            intent, response, path = "ipo", "IPO = Input → Process → Output. Deterministic flow.", "ipo"
        elif normalized in {"guardrail", "guardrails"}:
            intent, response, path = "guardrail", "🛡️ Guardrails active: deterministic safety layer engaged.", "guardrail"
        else:
            intent, response, path = "fallback", "⚠️ UNKNOWN INPUT. Try 'help'.", "fallback"

        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        st.session_state.chat_messages.append({"role": "assistant", "content": response, "intent": intent, "path": path})
        st.session_state.chat_count += 1
        st.rerun()

    if st.session_state.chat_closed:
        st.warning("⛔ EXIT TRIGGERED. Session closed.")

    col1, col2, col3 = st.columns(3)
    col1.metric("🧬 XP GAINED", st.session_state.chat_count)
    col2.metric("⚡ MODEL", "RULE-BASED")
    col3.metric("🔥 ARCH", "WHITE-BOX")

# ----------------------------------------------------------------------
# PROJECT 2: CLASSIFICATION (KNN)
# ----------------------------------------------------------------------
@st.cache_data
def load_iris_data():
    data = load_iris()
    return data.data, data.target, data.feature_names, data.target_names

def project_classification():
    st.markdown('<div class="main-header">📊 NEURAL CLASSIFIER (KNN)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">// IRIS DATASET // K-NEAREST NEIGHBORS</div>', unsafe_allow_html=True)

    X, y, feature_names, target_names = load_iris_data()

    with st.expander("📡 DATASET SCAN", expanded=False):
        st.dataframe(pd.DataFrame(X, columns=feature_names).head(5))

    col_left, col_right = st.columns([1, 2])
    with col_left:
        k_value = st.slider("🛠️ K-VALUE (NEIGHBORS)", 1, 15, 5, step=2)
        test_size = st.slider("📊 TEST SPLIT %", 10, 40, 20, step=5) / 100
        run_btn = st.button("⚡ DEPLOY CLASSIFICATION", type="primary")

    with col_right:
        if run_btn:
            with st.spinner("🔮 CALCULATING DISTANCES..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                knn = KNeighborsClassifier(n_neighbors=k_value)
                knn.fit(X_train_scaled, y_train)
                y_pred = knn.predict(X_test_scaled)
                acc = knn.score(X_test_scaled, y_test)
                cm = confusion_matrix(y_test, y_pred)

                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("🎯 ACCURACY", f"{acc:.2%}")
                col_m2.metric("🧠 TRAIN SAMPLES", len(y_train))
                col_m3.metric("💾 TEST SAMPLES", len(y_test))

                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", 
                            xticklabels=target_names, yticklabels=target_names, 
                            ax=ax, cbar_kws={'label': 'Count'})
                ax.set_title(f"Confusion Matrix (k={k_value})", color='#00ffe1', fontsize=14)
                ax.set_ylabel("Actual", color='#ff00cc')
                ax.set_xlabel("Predicted", color='#ff00cc')
                ax.tick_params(colors='#b0e0ff')
                st.pyplot(fig)

                st.subheader("📜 CLASSIFICATION REPORT")
                report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=target_names, output_dict=True)).transpose()
                st.dataframe(report_df.style.background_gradient(cmap="coolwarm", subset=["precision", "recall", "f1-score"]))
        else:
            st.info("👈 ADJUST PARAMETERS & DEPLOY.")

# ----------------------------------------------------------------------
# PROJECT 3: RECOMMENDATION
# ----------------------------------------------------------------------
@st.cache_data
def get_job_data():
    return pd.DataFrame({
        "Job_Title": ["Data Scientist", "ML Engineer", "Cloud Architect", "Web Dev", "DevOps", 
                      "Data Analyst", "Full Stack", "Python Dev", "Frontend", "Backend", 
                      "Security", "Mobile"],
        "Category": ["Data", "AI", "Cloud", "Web", "Ops", "Data", "Web", "Dev", "Web", "Dev", "Sec", "Mobile"],
        "Skills": [
            "Python SQL ML Stats Deep Learning Pandas",
            "Python TensorFlow PyTorch Docker Kubernetes MLflow",
            "AWS Azure GCP Terraform Kubernetes Python DevOps",
            "HTML CSS JavaScript React Node.js MongoDB Express",
            "Linux Docker Kubernetes AWS Jenkins CI/CD Python Shell",
            "SQL Excel Tableau Python Pandas Stats Power BI",
            "JavaScript React Node.js MongoDB Express HTML CSS",
            "Python Django Flask REST API SQL Git",
            "HTML CSS JavaScript React TypeScript Webpack",
            "Node.js Python Java Spring Boot SQL REST API",
            "Network Security Python Firewalls SIEM Linux AWS",
            "iOS Android Swift Kotlin React Native Java",
        ],
    })

def project_recommendation():
    st.markdown('<div class="main-header">💼 SKILLS SYNAPSE (RECOMMENDER)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">// CONTENT-BASED FILTERING // TF-IDF + COSINE</div>', unsafe_allow_html=True)

    jobs = get_job_data()

    with st.expander("🗃️ JOB DATABASE", expanded=False):
        st.dataframe(jobs)

    col_left, col_right = st.columns([1, 2])
    with col_left:
        skills_input = st.text_area("🛠️ INPUT SKILLS", "Python, Machine Learning, SQL", placeholder="e.g., Python, AWS, Docker")
        top_n = st.slider("🎯 TOP RESULTS", 1, 5, 3)
        recommend_btn = st.button("🔍 SCAN MATCHES", type="primary")

    with col_right:
        if recommend_btn:
            user_skills = [s.strip() for s in skills_input.split(",") if s.strip()]
            if not user_skills:
                st.warning("⚠️ ENTER SKILLS.")
                return
            
            alias = {"cloud computing": "cloud", "ci/cd": "devops", "machine learning": "ml"}
            normalized = [alias.get(s.lower(), s.lower()) for s in user_skills]
            user_profile = " ".join(normalized)

            documents = jobs["Skills"].tolist() + [user_profile]
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            tfidf = vectorizer.fit_transform(documents)
            similarities = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
            jobs["Similarity"] = similarities
            top_results = jobs.sort_values("Similarity", ascending=False).head(top_n)

            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(top_results["Job_Title"], top_results["Similarity"] * 100, 
                           color=['#00ffe1', '#ff00cc', '#ffcc00'][:top_n])
            ax.set_xlabel("SIMILARITY %", color='#00ffe1')
            ax.set_title("TOP MATCHES", color='#ff00cc', fontsize=14)
            ax.tick_params(colors='#b0e0ff')
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va='center', color='#00ffe1')
            st.pyplot(fig)

            for idx, row in top_results.iterrows():
                with st.container():
                    st.markdown(f"**🔥 {row['Job_Title']}**  ·  *{row['Category']}*")
                    st.write(f"📊 MATCH: **{row['Similarity']:.1%}**")
                    st.write(f"⚙️ REQUIREMENTS: {row['Skills']}")
                    match = set(normalized).intersection(set(row["Skills"].lower().split()))
                    if match:
                        st.success(f"✅ SYNAPSE MATCH: {', '.join(match)}")
                    st.divider()
        else:
            st.info("👈 INPUT SKILLS & SCAN.")

# ----------------------------------------------------------------------
# PROJECT 4: VISION – LAZY LOAD ALL HEAVY STUFF
# ----------------------------------------------------------------------
def project_vision():
    st.markdown('<div class="main-header">🖼️ CYBER EYE (VISION)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">// YOLOv8 + OCR // REAL-TIME ANALYSIS</div>', unsafe_allow_html=True)

    # Try to import YOLO, EasyOCR, and cv2 inside the function
    try:
        from ultralytics import YOLO
        YOLO_AVAILABLE = True
    except ImportError as e:
        YOLO_AVAILABLE = False
        yolo_error = str(e)

    try:
        import easyocr
        EASYOCR_AVAILABLE = True
    except ImportError as e:
        EASYOCR_AVAILABLE = False
        easyocr_error = str(e)

    if not YOLO_AVAILABLE:
        st.error(f"⚠️ YOLO not available: {yolo_error}\nPlease install `ultralytics` and `opencv-python-headless`.")
    if not EASYOCR_AVAILABLE:
        st.warning(f"⚠️ EasyOCR not available: {easyocr_error}\nOCR disabled.")

    uploaded_file = st.file_uploader("📂 UPLOAD IMAGE", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            import cv2
            import numpy as np
        except ImportError:
            st.error("❌ OpenCV (cv2) is not installed. Please add `opencv-python-headless` to requirements.")
            return

        # Load image with PIL and convert to numpy for CV2
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="ORIGINAL", width=400)

        col_det, col_ocr = st.columns(2)

        # ---- Object Detection ----
        with col_det:
            st.subheader("🔍 OBJECT DETECTION")
            if YOLO_AVAILABLE:
                with st.spinner("SCANNING..."):
                    model = YOLO("yolov8n.pt")  # downloads automatically
                    results = model(image)
                    # results[0].plot() returns numpy array with boxes
                    annotated = results[0].plot()
                    st.image(annotated, caption="DETECTED", use_container_width=True)
                    detections = results[0].boxes
                    if detections is not None and len(detections) > 0:
                        for box in detections:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            st.caption(f"- {model.names[cls]} (conf: {conf:.2f})")
                    else:
                        st.info("No objects detected.")
            else:
                st.warning("YOLO offline – install ultralytics and opencv-python-headless.")

        # ---- OCR ----
        with col_ocr:
            st.subheader("📝 TEXT EXTRACTION")
            if EASYOCR_AVAILABLE:
                with st.spinner("DECRYPTING..."):
                    reader = easyocr.Reader(["en"])
                    img_np = np.array(image)
                    ocr_result = reader.readtext(img_np, detail=0)
                    if ocr_result:
                        st.success(f"EXTRACTED {len(ocr_result)} BLOCKS:")
                        for t in ocr_result:
                            st.write(f"- {t}")
                    else:
                        st.info("No text found.")
            else:
                st.warning("OCR offline – install easyocr and torch.")

# ----------------------------------------------------------------------
# ROUTER
# ----------------------------------------------------------------------
if app_mode == "💬 CHATBOT":
    project_chatbot()
elif app_mode == "📊 CLASSIFY":
    project_classification()
elif app_mode == "💼 RECOMMEND":
    project_recommendation()
elif app_mode == "🖼️ VISION":
    project_vision()

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        ⚡ CYBERPUNK AI SUITE v1.1 || PRACTICAL TRAINING - II || SANKALP SHARMA ⚡
    </div>
    """,
    unsafe_allow_html=True,
    )
