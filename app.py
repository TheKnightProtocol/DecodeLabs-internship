"""
CYBERPUNK AI SUITE – Neon Dream v2.1 (Final Showcase Edition)
Projects: Dashboard · Chatbot · Classification · Recommendation · Vision · System Status
Author: Sankalp Sharma | Academic Final Project
"""

import logging
from datetime import datetime
from io import BytesIO

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
# LOGGING & PAGE CONFIG
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeonAI")

st.set_page_config(
    page_title="CYBERPUNK AI SUITE | Neon Dream v2.1",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# ADVANCED CYBERPUNK CSS
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

:root {
    --neon-cyan: #00ffe1;
    --neon-pink: #ff00cc;
    --neon-yellow: #ffcc00;
    --bg-dark: #07090d;
}

.stApp {
    background: var(--bg-dark);
    background-image: 
        radial-gradient(ellipse at 15% 40%, rgba(0, 255, 225, 0.07) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 60%, rgba(255, 0, 204, 0.06) 0%, transparent 55%),
        repeating-linear-gradient(0deg, rgba(0,255,225,0.015) 0px, rgba(0,255,225,0.015) 1px, transparent 1px, transparent 4px);
    font-family: 'Share Tech Mono', monospace !important;
    color: #b8e0ff !important;
}

.main-header {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 900 !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 1.75rem !important;
    background: linear-gradient(90deg, #00ffe1, #ff00cc, #00ffe1);
    background-size: 250% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: neonFlow 5s ease-in-out infinite;
    text-shadow: 0 0 40px rgba(0, 255, 225, 0.35);
    margin-bottom: 0.3rem;
}
@keyframes neonFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.sub-header {
    font-family: 'Share Tech Mono', monospace;
    color: var(--neon-cyan) !important;
    text-shadow: 0 0 18px var(--neon-cyan);
    letter-spacing: 2px;
    font-size: 0.85rem !important;
    border-left: 3px solid var(--neon-pink);
    padding-left: 12px;
    margin-bottom: 1.2rem;
}

section[data-testid="stSidebar"] {
    background: rgba(8, 12, 18, 0.95) !important;
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(0, 255, 225, 0.25) !important;
    box-shadow: 8px 0 40px rgba(0, 255, 225, 0.08);
}
.stSidebar .stRadio label {
    font-family: 'Orbitron', sans-serif !important;
    color: #b0e0ff !important;
    font-size: 0.78rem !important;
    padding: 8px 12px !important;
    border-radius: 8px;
    transition: all 0.25s ease;
    border: 1px solid transparent;
}
.stSidebar .stRadio label:hover {
    border-color: var(--neon-cyan);
    background: rgba(0, 255, 225, 0.07);
    box-shadow: 0 0 20px rgba(0, 255, 225, 0.15);
}

.stButton > button {
    font-family: 'Orbitron', sans-serif !important;
    background: transparent !important;
    border: 2px solid var(--neon-cyan) !important;
    color: var(--neon-cyan) !important;
    box-shadow: 0 0 18px rgba(0, 255, 225, 0.25);
    transition: all 0.3s cubic-bezier(0.23, 1, 0.32, 1) !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    width: 100%;
    min-height: 46px !important;
    font-size: 0.78rem !important;
    border-radius: 10px !important;
}
.stButton > button:hover {
    background: var(--neon-cyan) !important;
    color: #07090d !important;
    box-shadow: 0 0 45px var(--neon-cyan), 0 0 15px inset #07090d;
    transform: translateY(-2px) scale(1.01);
}
.stButton > button:active {
    transform: scale(0.97);
}

.stMetric {
    background: linear-gradient(145deg, rgba(12, 18, 28, 0.85), rgba(8, 12, 20, 0.9)) !important;
    border: 1px solid rgba(0, 255, 225, 0.22);
    border-radius: 14px !important;
    padding: 12px 14px !important;
    transition: all 0.35s ease;
    box-shadow: 0 0 25px rgba(0, 255, 225, 0.06);
    position: relative;
    overflow: hidden;
}
.stMetric::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent);
}
.stMetric:hover {
    border-color: var(--neon-pink);
    box-shadow: 0 0 40px rgba(255, 0, 204, 0.18);
    transform: translateY(-4px);
}
.stMetric label {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--neon-pink) !important;
    font-size: 0.62rem !important;
    letter-spacing: 1.2px;
}
.stMetric [data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--neon-cyan) !important;
    font-size: 1.45rem !important;
    text-shadow: 0 0 25px rgba(0, 255, 225, 0.45);
}

.neon-card {
    background: rgba(12, 18, 28, 0.65);
    border: 1px solid rgba(0, 255, 225, 0.2);
    border-radius: 16px;
    padding: 1.25rem;
    margin: 0.6rem 0;
    box-shadow: 0 0 30px rgba(0, 255, 225, 0.05);
    transition: all 0.3s ease;
    position: relative;
}
.neon-card:hover {
    border-color: rgba(255, 0, 204, 0.45);
    box-shadow: 0 0 45px rgba(255, 0, 204, 0.12);
    transform: translateY(-3px);
}

.stChatMessage {
    background: rgba(15, 25, 40, 0.7) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(0, 255, 225, 0.18) !important;
    border-radius: 18px !important;
    box-shadow: 0 0 25px rgba(0, 255, 225, 0.04);
    padding: 12px 16px !important;
    margin: 8px 0 !important;
}

.stFileUploader > div {
    border: 2px dashed rgba(0, 255, 225, 0.45) !important;
    background: rgba(0, 255, 225, 0.04) !important;
    border-radius: 14px !important;
    padding: 18px !important;
    transition: all 0.3s ease;
}
.stFileUploader > div:hover {
    border-color: var(--neon-pink) !important;
    box-shadow: 0 0 35px rgba(255, 0, 204, 0.15);
}

.streamlit-expanderHeader {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--neon-cyan) !important;
    letter-spacing: 1px;
}

.footer {
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
    color: var(--neon-pink);
    font-size: 0.72rem;
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    border-top: 1px dashed rgba(0, 255, 225, 0.4);
    text-shadow: 0 0 12px var(--neon-pink);
    animation: glitch 2.8s infinite;
}
@keyframes glitch {
    0%, 94%, 100% { opacity: 1; }
    95%, 97% { opacity: 0.3; }
}

@media (max-width: 768px) {
    .main-header { font-size: 1.25rem !important; letter-spacing: 1px; }
    .sub-header { font-size: 0.72rem !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.15rem !important; }
}

::-webkit-scrollbar { width: 7px; background: #07090d; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(var(--neon-cyan), var(--neon-pink));
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
if "total_xp" not in st.session_state:
    st.session_state.total_xp = 0
if "last_activity" not in st.session_state:
    st.session_state.last_activity = datetime.now().strftime("%H:%M:%S")
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_count" not in st.session_state:
    st.session_state.chat_count = 0
if "chat_closed" not in st.session_state:
    st.session_state.chat_closed = False

def bump_xp(amount=1):
    st.session_state.total_xp += amount
    st.session_state.last_activity = datetime.now().strftime("%H:%M:%S")

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 6px 0;">
        <div style="font-family:'Orbitron';font-size:1.45rem;color:#ff00cc;text-shadow:0 0 28px #ff00cc;letter-spacing:2px;">
            ⚡ NEON AI
        </div>
        <div style="font-family:'Share Tech Mono';color:#00ffe1;font-size:0.68rem;border-bottom:1px solid rgba(0,255,225,0.4);padding-bottom:8px;margin-top:2px;">
            [ SYSTEM v2.1 – SHOWCASE ]
        </div>
    </div>
    """, unsafe_allow_html=True)

    app_mode = st.radio(
        "SELECT MISSION",
        [
            "🏠 DASHBOARD",
            "💬 CHATBOT",
            "📊 CLASSIFY",
            "💼 RECOMMEND",
            "🖼️ VISION",
            "🛰️ SYSTEM STATUS",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="font-family:'Share Tech Mono';color:#b0e0ff;font-size:0.68rem;text-align:center;line-height:1.7;">
        🛡️ FIREWALL <span style="color:#00ffe1">ACTIVE</span><br>
        🔥 SESSION XP <span style="color:#ff00cc">{st.session_state.total_xp}</span><br>
        ⏱️ LAST PULSE <span style="color:#00ffe1">{st.session_state.last_activity}</span><br>
        🎮 OPERATOR <span style="color:#ffcc00">SANKALP</span>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# DASHBOARD
# ----------------------------------------------------------------------
def project_dashboard():
    st.markdown('<div class="main-header">🏠 NEON COMMAND CENTER</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">// MULTI-MODULE AI PLATFORM // FINAL PROJECT SHOWCASE</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="neon-card">
        <p style="margin:0;font-size:0.95rem;line-height:1.6;">
            Welcome, Operator. This suite demonstrates four core AI capabilities under a unified cyberpunk interface.
            Each module is designed for transparency, reproducibility, and visual impact.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="neon-card">
            <h4 style="color:#00ffe1;font-family:Orbitron;margin:0 0 8px 0;">💬 CHATBOT</h4>
            <p style="margin:0;font-size:0.82rem;opacity:0.9;">Deterministic rule engine with full IPO traceability. White-box decision paths.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="neon-card">
            <h4 style="color:#00ffe1;font-family:Orbitron;margin:0 0 8px 0;">📊 CLASSIFIER</h4>
            <p style="margin:0;font-size:0.82rem;opacity:0.9;">K-Nearest Neighbors on Iris. Live prediction form + full metrics + confusion matrix.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="neon-card">
            <h4 style="color:#ff00cc;font-family:Orbitron;margin:0 0 8px 0;">💼 RECOMMENDER</h4>
            <p style="margin:0;font-size:0.82rem;opacity:0.9;">Content-based job matching via TF-IDF + Cosine Similarity. Skill synapse highlighting.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="neon-card">
            <h4 style="color:#ff00cc;font-family:Orbitron;margin:0 0 8px 0;">🖼️ VISION</h4>
            <p style="margin:0;font-size:0.82rem;opacity:0.9;">YOLOv8 object detection + EasyOCR text extraction. Adjustable confidence + download.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📈 SESSION HUD")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🧬 TOTAL XP", st.session_state.total_xp)
    m2.metric("💬 CHAT TURNS", st.session_state.chat_count)
    m3.metric("⚡ CORE", "RULE + ML + CV")
    m4.metric("🛡️ STATUS", "NOMINAL")

# ----------------------------------------------------------------------
# CHATBOT
# ----------------------------------------------------------------------
def project_chatbot():
    st.markdown('<div class="main-header">💬 NEURAL CHAT INTERFACE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">// DETERMINISTIC RULE ENGINE // WHITE-BOX IPO // FULLY TRACEABLE</div>', unsafe_allow_html=True)

    with st.expander("🛡️ PROTOCOL (IPO Architecture)", expanded=False):
        st.markdown("""
        **Input** → Normalize & sanitize  
        **Process** → Explicit if-elif decision tree (no black-box)  
        **Output** → Response + Intent label + Decision path  

        **Intents:** hello, help, time, about, ipo, guardrail, features, status, clear, exit
        """)

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("intent"):
                st.caption(f"🧬 INTENT: `{msg['intent']}`  ·  PATH: `{msg.get('path', 'n/a')}`")

    user_input = st.chat_input("> ENTER COMMAND...")

    if user_input and not st.session_state.chat_closed:
        normalized = user_input.lower().strip()

        if normalized in {"hello", "hi", "hey", "greetings"}:
            intent, response, path = "greeting", "⚡ Hello, Operator. Rule engine locked and loaded.", "greeting"
        elif normalized in {"quit", "exit", "bye", "shutdown"}:
            intent, response, path = "exit", "⛔ Shutdown sequence complete. Session terminated.", "exit"
            st.session_state.chat_closed = True
        elif normalized == "help":
            intent, response, path = "help", "COMMANDS → hello · help · time · about · ipo · guardrail · features · status · clear · exit", "help"
        elif normalized == "time":
            intent, response, path = "system_time", f"⏰ SYSTEM CLOCK: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "time"
        elif normalized == "about":
            intent, response, path = "about", "CYBERPUNK AI SUITE v2.1 – Transparent white-box core wrapped in neon UI. Built for academic excellence.", "about"
        elif normalized in {"ipo", "ipo explanation"}:
            intent, response, path = "ipo", "IPO = Input → Process → Output. Every decision is explicit, logged, and auditable.", "ipo"
        elif normalized in {"guardrail", "guardrails"}:
            intent, response, path = "guardrail", "🛡️ Guardrails online: deterministic isolation + intent sandboxing active.", "guardrail"
        elif normalized in {"features", "what can you do"}:
            intent, response, path = "features", "Modules online: Chatbot · KNN Classifier · Skills Recommender · YOLO+OCR Vision · System Status.", "features"
        elif normalized == "status":
            intent, response, path = "status", f"SYSTEM NOMINAL · XP: {st.session_state.total_xp} · Chat turns: {st.session_state.chat_count}", "status"
        elif normalized in {"clear", "reset"}:
            st.session_state.chat_messages = []
            st.session_state.chat_count = 0
            bump_xp(1)
            st.rerun()
        else:
            intent, response, path = "fallback", "⚠️ UNKNOWN INPUT. Type `help` for command list.", "fallback"

        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        st.session_state.chat_messages.append({
            "role": "assistant", "content": response, "intent": intent, "path": path
        })
        st.session_state.chat_count += 1
        bump_xp(2)
        st.rerun()

    if st.session_state.chat_closed:
        st.warning("⛔ EXIT TRIGGERED. Refresh the page to restart the neural link.")

    col1, col2, col3 = st.columns(3)
    col1.metric("🧬 XP THIS MODULE", st.session_state.chat_count * 2)
    col2.metric("⚡ MODEL", "RULE-BASED")
    col3.metric("🔥 ARCHITECTURE", "WHITE-BOX")

    if st.session_state.chat_messages:
        log_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_messages])
        st.download_button(
            "📥 EXPORT CHAT LOG",
            data=log_text,
            file_name=f"neon_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# ----------------------------------------------------------------------
# CLASSIFICATION
# ----------------------------------------------------------------------
@st.cache_data
def load_iris_data():
    data = load_iris()
    return data.data, data.target, data.feature_names, data.target_names

def project_classification():
    st.markdown('<div class="main-header">📊 NEURAL CLASSIFIER (KNN)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">// IRIS DATASET // K-NEAREST NEIGHBORS // LIVE PREDICTION</div>', unsafe_allow_html=True)

    X, y, feature_names, target_names = load_iris_data()

    with st.expander("📡 DATASET SCAN", expanded=False):
        df = pd.DataFrame(X, columns=feature_names)
        df["Species"] = [target_names[i] for i in y]
        st.dataframe(df.head(8), use_container_width=True)
        st.caption(f"Shape: {X.shape} · Classes: {list(target_names)}")

    tab1, tab2 = st.tabs(["⚡ TRAIN & EVALUATE", "🎯 LIVE PREDICT"])

    with tab1:
        col_left, col_right = st.columns([1, 2])
        with col_left:
            k_value = st.slider("🛠️ K (Neighbors)", 1, 15, 5, step=2)
            test_size = st.slider("📊 Test Split %", 10, 40, 20, step=5) / 100
            run_btn = st.button("⚡ DEPLOY MODEL", type="primary", key="train_btn")

        with col_right:
            if run_btn:
                with st.spinner("🔮 Computing distances..."):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)

                    knn = KNeighborsClassifier(n_neighbors=k_value)
                    knn.fit(X_train_s, y_train)
                    y_pred = knn.predict(X_test_s)
                    acc = knn.score(X_test_s, y_test)
                    cm = confusion_matrix(y_test, y_pred)

                    bump_xp(5)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("🎯 ACCURACY", f"{acc:.2%}")
                    m2.metric("🧠 TRAIN", len(y_train))
                    m3.metric("💾 TEST", len(y_test))

                    plt.style.use("dark_background")
                    fig, ax = plt.subplots(figsize=(6, 4.2))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
                                xticklabels=target_names, yticklabels=target_names,
                                ax=ax, cbar_kws={"label": "Count"})
                    ax.set_title(f"Confusion Matrix · k={k_value}", color="#00ffe1", fontsize=13)
                    ax.set_ylabel("Actual", color="#ff00cc")
                    ax.set_xlabel("Predicted", color="#ff00cc")
                    ax.tick_params(colors="#b0e0ff")
                    st.pyplot(fig)

                    st.subheader("📜 Classification Report")
                    report = pd.DataFrame(
                        classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
                    ).transpose()
                    st.dataframe(
                        report.style.background_gradient(cmap="coolwarm", subset=["precision", "recall", "f1-score"]),
                        use_container_width=True
                    )
            else:
                st.info("👈 Set parameters and deploy the model.")

    with tab2:
        st.markdown("Enter flower measurements to get an instant species prediction.")
        c1, c2 = st.columns(2)
        with c1:
            sepal_l = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
            sepal_w = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
        with c2:
            petal_l = st.number_input("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
            petal_w = st.number_input("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)

        if st.button("🔮 PREDICT SPECIES", type="primary"):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_scaled, y)
            sample = scaler.transform([[sepal_l, sepal_w, petal_l, petal_w]])
            pred = knn.predict(sample)[0]
            proba = knn.predict_proba(sample)[0]
            species = target_names[pred]

            bump_xp(3)
            st.success(f"**Predicted Species → {species.upper()}**")
            st.write("Class probabilities:")
            for name, p in zip(target_names, proba):
                st.progress(float(p), text=f"{name}: {p:.1%}")

# ----------------------------------------------------------------------
# RECOMMENDATION
# ----------------------------------------------------------------------
@st.cache_data
def get_job_data():
    return pd.DataFrame({
        "Job_Title": [
            "Data Scientist", "ML Engineer", "Cloud Architect", "Web Developer",
            "DevOps Engineer", "Data Analyst", "Full Stack Developer", "Python Developer",
            "Frontend Developer", "Backend Developer", "Security Engineer", "Mobile Developer",
            "AI Research Intern", "Business Intelligence Analyst"
        ],
        "Category": [
            "Data", "AI", "Cloud", "Web", "Ops", "Data", "Web", "Dev",
            "Web", "Dev", "Sec", "Mobile", "AI", "Data"
        ],
        "Skills": [
            "Python SQL ML Stats Deep Learning Pandas Scikit-learn",
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
            "Python Research Papers PyTorch NLP Computer Vision",
            "SQL Power BI Tableau Excel Python Storytelling"
        ],
    })

def project_recommendation():
    st.markdown('<div class="main-header">💼 SKILLS SYNAPSE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">// CONTENT-BASED FILTERING // TF-IDF + COSINE SIMILARITY</div>', unsafe_allow_html=True)

    jobs = get_job_data()

    with st.expander("🗃️ JOB DATABASE", expanded=False):
        st.dataframe(jobs, use_container_width=True)

    col_left, col_right = st.columns([1, 2])
    with col_left:
        skills_input = st.text_area(
            "🛠️ YOUR SKILLS (comma separated)",
            "Python, Machine Learning, SQL, Docker",
            height=100
        )
        top_n = st.slider("🎯 TOP RESULTS", 1, 7, 4)
        recommend_btn = st.button("🔍 SCAN MATCHES", type="primary")

    with col_right:
        if recommend_btn:
            user_skills = [s.strip() for s in skills_input.split(",") if s.strip()]
            if not user_skills:
                st.warning("⚠️ Enter at least one skill.")
                return

            alias = {
                "cloud computing": "cloud", "ci/cd": "devops", "machine learning": "ml",
                "deep learning": "ml", "react.js": "react", "node.js": "node"
            }
            normalized = [alias.get(s.lower(), s.lower()) for s in user_skills]
            user_profile = " ".join(normalized)

            documents = jobs["Skills"].tolist() + [user_profile]
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            tfidf = vectorizer.fit_transform(documents)
            similarities = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()

            jobs = jobs.copy()
            jobs["Similarity"] = similarities
            top_results = jobs.sort_values("Similarity", ascending=False).head(top_n)

            bump_xp(4)

            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            colors = ["#00ffe1", "#ff00cc", "#ffcc00", "#00ccff", "#ff6600", "#cc00ff", "#66ff99"]
            bars = ax.barh(top_results["Job_Title"], top_results["Similarity"] * 100,
                           color=colors[:len(top_results)])
            ax.set_xlabel("SIMILARITY %", color="#00ffe1")
            ax.set_title("TOP JOB MATCHES", color="#ff00cc", fontsize=13)
            ax.tick_params(colors="#b0e0ff")
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%",
                        va="center", color="#00ffe1", fontsize=9)
            st.pyplot(fig)

            for _, row in top_results.iterrows():
                with st.container():
                    st.markdown(f"**🔥 {row['Job_Title']}**  ·  *{row['Category']}*")
                    st.write(f"📊 Match: **{row['Similarity']:.1%}**")
                    st.caption(f"Required: {row['Skills']}")
                    match = set(normalized) & set(row["Skills"].lower().split())
                    if match:
                        st.success(f"✅ Synapse hits: {', '.join(sorted(match))}")
                    st.divider()
        else:
            st.info("👈 Enter skills and scan the job matrix.")

# ----------------------------------------------------------------------
# VISION
# ----------------------------------------------------------------------
def project_vision():
    st.markdown('<div class="main-header">🖼️ CYBER EYE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">// YOLOv8 OBJECT DETECTION + EASYOCR // REAL-TIME ANALYSIS</div>', unsafe_allow_html=True)

    YOLO_AVAILABLE = False
    EASYOCR_AVAILABLE = False
    yolo_error = easyocr_error = ""

    try:
        from ultralytics import YOLO
        YOLO_AVAILABLE = True
    except Exception as e:
        yolo_error = str(e)

    try:
        import easyocr
        EASYOCR_AVAILABLE = True
    except Exception as e:
        easyocr_error = str(e)

    if not YOLO_AVAILABLE or not EASYOCR_AVAILABLE:
        st.error("⚠️ Vision modules failed to load")

        with st.expander("🔧 How to fix this (NumPy conflict) – IMPORTANT", expanded=True):
            st.markdown("""
**Most common cause:** NumPy 2.x is installed, but YOLO / EasyOCR need NumPy 1.x.

**Recommended fix (copy-paste these commands one by one):**

```bash
pip uninstall -y numpy ultralytics easyocr opencv-python opencv-python-headless
pip install "numpy==1.26.4"
pip install ultralytics easyocr opencv-python-headless
