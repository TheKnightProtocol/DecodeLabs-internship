"""
Unified AI Applications – Streamlit Dashboard
Projects: Chatbot · Classification · Recommendation · Recognition
"""

import logging
import tempfile
from datetime import datetime
from pathlib import Path

import easyocr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import yaml
from PIL import Image
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Applications Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# CUSTOM CSS FOR PROFESSIONAL LOOK
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp { background-color: #f8fafc; }
        .main-header {
            font-size: 2.5rem;
            font-weight: 600;
            color: #0f172a;
            margin-bottom: 0;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #475569;
            margin-top: -0.2rem;
            margin-bottom: 1.2rem;
        }
        .project-card {
            background: white;
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            border: 1px solid #e9edf2;
            margin-bottom: 1.5rem;
        }
        .stChatMessage {
            border-radius: 14px !important;
            padding: 10px 16px !important;
            margin: 6px 0 !important;
            background-color: #ffffff !important;
            border: 1px solid #e9edf2 !important;
        }
        .stChatMessage.user {
            background-color: #e9f0fe !important;
            border-color: #c7d8f5 !important;
        }
        .stChatMessage.assistant {
            background-color: #f1f4f9 !important;
        }
        .stMetric {
            background-color: white;
            border-radius: 8px;
            padding: 8px 12px;
            border: 1px solid #e9edf2;
        }
        .stSidebar {
            background-color: #ffffff;
            border-right: 1px solid #e9edf2;
        }
        .footer {
            text-align: center;
            color: #94a3b8;
            font-size: 0.8rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e9edf2;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
st.sidebar.image(
    "https://img.icons8.com/fluency/96/000000/artificial-intelligence.png",
    width=80,
)
st.sidebar.markdown("## AI Application Suite")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Select Project",
    [
        "💬 Rule-Based Chatbot",
        "📊 Data Classification (KNN)",
        "💼 Career Recommendation",
        "🖼️ Image Recognition (OCR + Detection)",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · Python · AI/ML")
st.sidebar.caption("© 2026 Sankalp Sharma")

# ----------------------------------------------------------------------
# PROJECT 1: RULE-BASED CHATBOT
# ----------------------------------------------------------------------
def project_chatbot():
    st.markdown('<div class="main-header">💬 Rule-Based Chatbot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Deterministic · White‑Box · IPO‑Driven</div>',
        unsafe_allow_html=True,
    )

    # Initialize session state for chatbot
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_count" not in st.session_state:
        st.session_state.chat_count = 0
    if "chat_closed" not in st.session_state:
        st.session_state.chat_closed = False

    # Sidebar info for chatbot
    with st.expander("📖 How it works", expanded=False):
        st.markdown("""
        **IPO Model:**  
        - **Input:** Sanitize user text (`lower()`, `strip()`)  
        - **Process:** Explicit `if-elif-else` decision tree  
        - **Output:** Traceable response + audit log  

        **Supported intents:**  
        `hello`, `help`, `time`, `about`, `ipo`, `guardrail`, `exit`, `fallback`
        """)

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "intent" in msg and msg["intent"]:
                st.caption(f"Intent: {msg['intent']}  |  Path: {msg.get('path', 'n/a')}")

    # Input
    user_input = st.chat_input("Type a message (e.g., hello, help, time, ipo, exit)")

    if user_input:
        normalized = user_input.lower().strip()

        # ---- Rule Engine ----
        if normalized in {"hello", "hi", "hey"}:
            intent, response, path = (
                "greeting",
                "Hello! I am a deterministic rule‑based chatbot.",
                "greeting rule",
            )
        elif normalized in {"quit", "exit", "bye"}:
            intent, response, path = (
                "exit",
                "Goodbye. Session complete.",
                "exit rule",
            )
            st.session_state.chat_closed = True
        elif normalized == "help":
            intent, response, path = (
                "help",
                "Available commands: hello, help, time, about, ipo, guardrail, exit.",
                "help rule",
            )
        elif normalized == "time":
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            intent, response, path = ("system_time", f"Local time: {now}", "time rule")
        elif normalized == "about":
            intent, response, path = (
                "about_project",
                "This is a white‑box chatbot demonstrating deterministic logic and guardrail concepts.",
                "about rule",
            )
        elif normalized in {"ipo", "ipo explanation", "explain ipo"}:
            intent, response, path = (
                "ipo_explanation",
                "IPO = Input → Process → Output. Sanitize → Apply rules → Respond.",
                "ipo rule",
            )
        elif normalized in {"guardrail", "guardrails", "control layer"}:
            intent, response, path = (
                "guardrail_explanation",
                "A guardrail is deterministic control logic that constrains AI responses for safety.",
                "guardrail rule",
            )
        else:
            intent, response, path = (
                "fallback",
                "Fallback: I don't understand that. Try 'help' for options.",
                "fallback rule",
            )

        # Store messages
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": response, "intent": intent, "path": path}
        )
        st.session_state.chat_count += 1

        # Simulated audit log
        logging.info(f"Input: {user_input} → Intent: {intent}")

        st.rerun()

    if st.session_state.chat_closed:
        st.info("ℹ️ Exit rule triggered. You can continue chatting or refresh.")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Interactions", st.session_state.chat_count)
    col2.metric("Model Type", "Rule‑Based")
    col3.metric("Architecture", "White‑Box")


# ----------------------------------------------------------------------
# PROJECT 2: DATA CLASSIFICATION (KNN)
# ----------------------------------------------------------------------
@st.cache_data
def load_iris_data():
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    return X, y, feature_names, target_names


def project_classification():
    st.markdown('<div class="main-header">📊 Data Classification (KNN)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Iris Dataset · K‑Nearest Neighbors</div>',
        unsafe_allow_html=True,
    )

    X, y, feature_names, target_names = load_iris_data()

    with st.expander("📖 About the Dataset", expanded=False):
        st.write(
            f"**Samples:** {X.shape[0]}  |  **Features:** {X.shape[1]}  |  **Classes:** {len(target_names)}"
        )
        st.dataframe(pd.DataFrame(X, columns=feature_names).head(5))

    col_left, col_right = st.columns([1, 2])

    with col_left:
        k_value = st.slider("Select k (number of neighbours)", 1, 15, 5, step=2)
        test_size = st.slider("Test size (%)", 10, 40, 20, step=5) / 100
        run_btn = st.button("🚀 Run Classification", type="primary")

    with col_right:
        if run_btn:
            with st.spinner("Training KNN..."):
                # Split & scale
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train
                knn = KNeighborsClassifier(n_neighbors=k_value)
                knn.fit(X_train_scaled, y_train)
                y_pred = knn.predict(X_test_scaled)

                # Metrics
                acc = knn.score(X_test_scaled, y_test)
                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

                # Display metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Test Accuracy", f"{acc:.2%}")
                col_m2.metric("Training Samples", len(y_train))
                col_m3.metric("Test Samples", len(y_test))

                # Confusion Matrix
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=target_names,
                    yticklabels=target_names,
                    ax=ax,
                )
                ax.set_title(f"Confusion Matrix (k={k_value})")
                ax.set_ylabel("Actual")
                ax.set_xlabel("Predicted")
                st.pyplot(fig)

                # Classification Report
                st.subheader("Classification Report")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]))

        else:
            st.info("👈 Adjust parameters and click **Run Classification** to see results.")


# ----------------------------------------------------------------------
# PROJECT 3: CAREER RECOMMENDATION
# ----------------------------------------------------------------------
@st.cache_data
def get_job_data():
    return pd.DataFrame({
        "Job_Title": [
            "Data Scientist",
            "Machine Learning Engineer",
            "Cloud Architect",
            "Web Developer",
            "DevOps Engineer",
            "Data Analyst",
            "Full Stack Developer",
            "Python Developer",
            "Frontend Developer",
            "Backend Developer",
            "Security Engineer",
            "Mobile Developer",
        ],
        "Category": [
            "Data Science",
            "ML/AI",
            "Cloud",
            "Web",
            "DevOps",
            "Data Science",
            "Web",
            "Development",
            "Web",
            "Development",
            "Security",
            "Mobile",
        ],
        "Skills": [
            "Python SQL Machine Learning Statistics Deep Learning Pandas",
            "Python TensorFlow PyTorch Docker Kubernetes MLflow",
            "AWS Azure GCP Terraform Kubernetes Python DevOps",
            "HTML CSS JavaScript React Node.js MongoDB Express",
            "Linux Docker Kubernetes AWS Jenkins CI/CD Python Shell",
            "SQL Excel Tableau Python Pandas Statistics Power BI",
            "JavaScript React Node.js MongoDB Express HTML CSS",
            "Python Django Flask REST API SQL Git",
            "HTML CSS JavaScript React TypeScript Webpack",
            "Node.js Python Java Spring Boot SQL REST API",
            "Network Security Python Firewalls SIEM Linux AWS",
            "iOS Android Swift Kotlin React Native Java",
        ],
    })


def project_recommendation():
    st.markdown('<div class="main-header">💼 Career Recommendation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Content‑Based Filtering · TF‑IDF + Cosine Similarity</div>',
        unsafe_allow_html=True,
    )

    jobs = get_job_data()

    with st.expander("📖 Job Roles Dataset", expanded=False):
        st.dataframe(jobs)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        skills_input = st.text_area(
            "Enter your skills (comma‑separated)",
            placeholder="e.g., Python, Machine Learning, SQL",
            value="Python, Machine Learning, SQL",
        )
        top_n = st.slider("Number of recommendations", 1, 5, 3)
        recommend_btn = st.button("🔍 Get Recommendations", type="primary")

    with col_right:
        if recommend_btn:
            user_skills = [s.strip() for s in skills_input.split(",") if s.strip()]
            if not user_skills:
                st.warning("Please enter at least one skill.")
                return

            # Alias mapping (simplified)
            alias = {
                "cloud computing": "cloud",
                "ci/cd": "devops",
                "continuous integration": "devops",
                "machine learning": "ml",
                "deep learning": "dl",
            }
            normalized = []
            for s in user_skills:
                s_low = s.lower()
                normalized.append(alias.get(s_low, s_low))

            user_profile = " ".join(normalized)

            # TF-IDF
            documents = jobs["Skills"].tolist() + [user_profile]
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            tfidf = vectorizer.fit_transform(documents)

            job_vectors = tfidf[:-1]
            user_vector = tfidf[-1]
            similarities = cosine_similarity(user_vector, job_vectors).flatten()

            jobs["Similarity"] = similarities
            top_results = jobs.sort_values("Similarity", ascending=False).head(top_n)

            # Display results
            st.subheader(f"Top {top_n} Recommendations")

            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(
                top_results["Job_Title"],
                top_results["Similarity"] * 100,
                color=["#2e86de", "#00b894", "#fdcb6e"][:top_n],
            )
            ax.set_xlabel("Similarity Score (%)")
            ax.set_title("Matching Job Roles")
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center")
            st.pyplot(fig)

            # Detailed cards
            for idx, row in top_results.iterrows():
                with st.container():
                    st.markdown(f"**{row['Job_Title']}**  ·  *{row['Category']}*")
                    st.write(f"📌 Similarity: **{row['Similarity']:.1%}**")
                    st.write(f"🔧 Required Skills: {row['Skills']}")

                    # Highlight matching skills
                    user_set = set(normalized)
                    job_set = set(row["Skills"].lower().split())
                    match = user_set.intersection(job_set)
                    if match:
                        st.success(f"✅ Matching: {', '.join(match)}")
                    st.divider()
        else:
            st.info("👈 Enter your skills and click **Get Recommendations**.")


# ----------------------------------------------------------------------
# PROJECT 4: IMAGE RECOGNITION (OCR + OBJECT DETECTION)
# ----------------------------------------------------------------------
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")  # downloads automatically


@st.cache_resource
def load_easyocr():
    return easyocr.Reader(["en"])


def project_vision():
    st.markdown('<div class="main-header">🖼️ Image Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">YOLOv8 Object Detection + EasyOCR Text Extraction</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload an image (JPG, PNG, JPEG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", width=400)

        col_det, col_ocr = st.columns(2)

        with col_det:
            st.subheader("🔍 Object Detection (YOLOv8)")
            with st.spinner("Running YOLO..."):
                model = load_yolo()
                results = model(image, stream=False)

                # Plot results
                fig = results[0].plot()  # returns numpy array
                st.image(fig, caption="Detected Objects", use_container_width=True)

                # Annotations
                detections = results[0].boxes
                if detections is not None and len(detections) > 0:
                    for box in detections:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = model.names[cls]
                        st.caption(f"- {label} (confidence: {conf:.2f})")
                else:
                    st.info("No objects detected.")

        with col_ocr:
            st.subheader("📝 Text Extraction (EasyOCR)")
            with st.spinner("Running OCR..."):
                reader = load_easyocr()
                # Convert PIL to numpy
                import numpy as np

                img_np = np.array(image)
                ocr_result = reader.readtext(img_np, detail=0)

                if ocr_result:
                    st.success(f"Extracted {len(ocr_result)} text block(s):")
                    for i, text in enumerate(ocr_result, 1):
                        st.write(f"{i}. {text}")
                else:
                    st.info("No text found in the image.")

    else:
        st.info("👆 Upload an image to start analysis.")


# ----------------------------------------------------------------------
# ROUTING
# ----------------------------------------------------------------------
if app_mode == "💬 Rule-Based Chatbot":
    project_chatbot()
elif app_mode == "📊 Data Classification (KNN)":
    project_classification()
elif app_mode == "💼 Career Recommendation":
    project_recommendation()
elif app_mode == "🖼️ Image Recognition (OCR + Detection)":
    project_vision()

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Practical Training – II Report · AI Applications · June – August 2026
    </div>
    """,
    unsafe_allow_html=True,
)
