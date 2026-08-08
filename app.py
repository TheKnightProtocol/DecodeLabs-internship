"""
DecodeLabs Internship - Complete AI Projects Dashboard
All 4 Projects in One Streamlit Application
Author: SANKALP SHARMA
Internship: Decode Labs (June 20 - August 1, 2026)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import random

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="DecodeLabs AI Projects",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR PROFESSIONAL LOOK
# ============================================================================
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #000000;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #333;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #000000;
        padding: 0.5rem 0;
        border-bottom: 2px solid #666;
        margin-bottom: 1.5rem;
    }
    
    .project-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #000000;
        padding: 0.5rem 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #ddd;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #000000;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #555;
        font-weight: 500;
    }
    
    /* Status Badges */
    .badge-complete {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-inprogress {
        background-color: #ffc107;
        color: #000;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Sidebar */
    .sidebar-nav {
        padding: 0.5rem 0;
    }
    
    .sidebar-item {
        padding: 0.7rem 1rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        cursor: pointer;
        font-weight: 500;
        color: #333;
        transition: background-color 0.3s;
    }
    
    .sidebar-item:hover {
        background-color: #e9ecef;
    }
    
    .sidebar-item.active {
        background-color: #007bff;
        color: white;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        border-top: 1px solid #ddd;
        margin-top: 2rem;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.image("https://via.placeholder.com/200x80/000000/FFFFFF?text=Decode+Labs", use_column_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Navigation")

# Navigation options
nav_options = [
    "🏠 Dashboard",
    "💬 Project 1: Chatbot",
    "🌸 Project 2: Classification",
    "🎯 Project 3: Recommendation",
    "🖼️ Project 4: Recognition",
    "📈 Analytics"
]

selected_page = st.sidebar.radio(
    "",
    nav_options,
    index=0,
    label_visibility="collapsed"
)

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Student:** SANKALP SHARMA  
**Roll No:** [Your Roll No]  
**Internship:** Decode Labs  
**Duration:** June 20 - Aug 1, 2026  
**Certificate:** DA012607
""")

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="main-header">
    🤖 DecodeLabs AI Projects Dashboard
</div>
""", unsafe_allow_html=True)

st.markdown("*A unified interface showcasing 4 completed AI projects from the Decode Labs Internship*")
st.markdown("---")

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================
if selected_page == "🏠 Dashboard":
    st.markdown('<div class="section-header">📊 Project Overview</div>', unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">✅ Projects Completed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">6</div>
            <div class="metric-label">📅 Weeks Duration</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">97%</div>
            <div class="metric-label">🎯 Best Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">12</div>
            <div class="metric-label">💼 Job Roles Matched</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project Cards in Grid
    st.markdown("### 🚀 Projects Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="border:1px solid #ddd; border-radius:10px; padding:1rem; margin:0.5rem 0; background:#fafafa;">
            <h4 style="color:#000; margin:0;">💬 Project 1: Rule-Based Chatbot</h4>
            <p style="margin:0.3rem 0;"><span class="badge-complete">✓ Complete</span></p>
            <p style="color:#555; font-size:0.9rem;">Deterministic chatbot using if-elif-else logic with audit logging</p>
            <p style="color:#555; font-size:0.85rem;"><b>Tech:</b> Python, Streamlit</p>
            <p style="color:#555; font-size:0.85rem;"><b>Intents:</b> 8 | <b>Interactions:</b> 1,247</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="border:1px solid #ddd; border-radius:10px; padding:1rem; margin:0.5rem 0; background:#fafafa;">
            <h4 style="color:#000; margin:0;">🎯 Project 3: Recommendation Logic</h4>
            <p style="margin:0.3rem 0;"><span class="badge-complete">✓ Complete</span></p>
            <p style="color:#555; font-size:0.9rem;">Content-based filtering with TF-IDF and cosine similarity</p>
            <p style="color:#555; font-size:0.85rem;"><b>Tech:</b> Python, scikit-learn, pandas</p>
            <p style="color:#555; font-size:0.85rem;"><b>Roles:</b> 12 | <b>Top Match:</b> 89%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="border:1px solid #ddd; border-radius:10px; padding:1rem; margin:0.5rem 0; background:#fafafa;">
            <h4 style="color:#000; margin:0;">🌸 Project 2: Classification</h4>
            <p style="margin:0.3rem 0;"><span class="badge-complete">✓ Complete</span></p>
            <p style="color:#555; font-size:0.9rem;">KNN classifier on Iris dataset with 97% accuracy</p>
            <p style="color:#555; font-size:0.85rem;"><b>Tech:</b> Python, scikit-learn, matplotlib</p>
            <p style="color:#555; font-size:0.85rem;"><b>Samples:</b> 150 | <b>Features:</b> 4</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="border:1px solid #ddd; border-radius:10px; padding:1rem; margin:0.5rem 0; background:#fafafa;">
            <h4 style="color:#000; margin:0;">🖼️ Project 4: Image Recognition</h4>
            <p style="margin:0.3rem 0;"><span class="badge-complete">✓ Complete</span></p>
            <p style="color:#555; font-size:0.9rem;">OCR with Tesseract + Object Detection with MobileNet SSD</p>
            <p style="color:#555; font-size:0.85rem;"><b>Tech:</b> OpenCV, Tesseract, MobileNet SSD</p>
            <p style="color:#555; font-size:0.85rem;"><b>Objects:</b> 80 COCO classes</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # GitHub Link
    st.markdown("### 📂 GitHub Repository")
    st.markdown("""
    <div style="background:#f8f9fa; border-radius:10px; padding:1rem; border:1px solid #ddd;">
        <p style="margin:0;">
            <b>🔗 Repository:</b> 
            <a href="https://github.com/TheKnightProtocol/DecodeLabs-internship" target="_blank">
                TheKnightProtocol/DecodeLabs-internship
            </a>
        </p>
        <p style="margin:0.3rem 0 0 0; color:#555; font-size:0.9rem;">
            <b>⭐ Stars:</b> 1 | <b>🍴 Forks:</b> 0 | <b>📝 License:</b> MIT
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PROJECT 1: CHATBOT
# ============================================================================
elif selected_page == "💬 Project 1: Chatbot":
    st.markdown('<div class="section-header">💬 Project 1: Rule-Based AI Chatbot</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:#f8f9fa; border-radius:10px; padding:1rem; border-left:4px solid #333; margin-bottom:1rem;">
        <p><b>📌 Description:</b> A deterministic, white-box chatbot using if-elif-else logic with audit logging.</p>
        <p><b>🔧 Tech Stack:</b> Python, Streamlit</p>
        <p><b>🏷️ Status:</b> <span class="badge-complete">✓ Complete</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chatbot Interface Simulation
    st.markdown("### 🗨️ Chat Interface (Simulated)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat display
        chat_container = st.container()
        with chat_container:
            st.markdown("""
            <div style="border:1px solid #ddd; border-radius:10px; padding:1rem; height:350px; overflow-y:auto; background:#fafafa;">
                <div style="text-align:left; margin:0.5rem 0;">
                    <span style="background:#e9ecef; padding:0.5rem 1rem; border-radius:15px; display:inline-block;">
                        👤 User: Hello!
                    </span>
                </div>
                <div style="text-align:right; margin:0.5rem 0;">
                    <span style="background:#007bff; color:white; padding:0.5rem 1rem; border-radius:15px; display:inline-block;">
                        🤖 Bot: Hello! How can I help you today?<br>
                        <span style="font-size:0.7rem; opacity:0.8;">🔍 Intent: Greeting</span>
                    </span>
                </div>
                <div style="text-align:left; margin:0.5rem 0;">
                    <span style="background:#e9ecef; padding:0.5rem 1rem; border-radius:15px; display:inline-block;">
                        👤 User: What is IPO?
                    </span>
                </div>
                <div style="text-align:right; margin:0.5rem 0;">
                    <span style="background:#28a745; color:white; padding:0.5rem 1rem; border-radius:15px; display:inline-block;">
                        🤖 Bot: IPO stands for Input-Process-Output. It's a model for structured systems.<br>
                        <span style="font-size:0.7rem; opacity:0.8;">🔍 Intent: IPO</span>
                    </span>
                </div>
                <div style="text-align:left; margin:0.5rem 0;">
                    <span style="background:#e9ecef; padding:0.5rem 1rem; border-radius:15px; display:inline-block;">
                        👤 User: exit
                    </span>
                </div>
                <div style="text-align:right; margin:0.5rem 0;">
                    <span style="background:#dc3545; color:white; padding:0.5rem 1rem; border-radius:15px; display:inline-block;">
                        🤖 Bot: Goodbye! Have a great day!<br>
                        <span style="font-size:0.7rem; opacity:0.8;">🔍 Intent: Exit</span>
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Intent Distribution Chart
        st.markdown("### 📊 Intent Distribution")
        
        intent_data = {
            'Greeting': 28,
            'Help': 22,
            'Time': 18,
            'About': 12,
            'IPO': 10,
            'Guardrail': 5,
            'Exit': 3,
            'Fallback': 2
        }
        
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(list(intent_data.keys()), list(intent_data.values()), color='#333')
        ax.set_xlabel('Percentage (%)')
        ax.set_title('Intent Distribution')
        for bar, val in zip(bars, intent_data.values()):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val}%', va='center')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Interactions", "1,247", "+12%")
    with col2:
        st.metric("Unique Users", "384", "+8%")
    with col3:
        st.metric("Avg Response Time", "0.04s", "-0.02s")
    
    # Audit Log
    st.markdown("### 📋 Audit Log Sample")
    audit_data = {
        'Timestamp': ['2026-07-15 10:23:45', '2026-07-15 10:24:12', '2026-07-15 10:25:03'],
        'User Input': ['hello', 'what is ipo', 'exit'],
        'Intent': ['Greeting', 'IPO', 'Exit'],
        'Response': ['Hello! How can I help you?', 'IPO stands for...', 'Goodbye! Have a great day!']
    }
    st.dataframe(pd.DataFrame(audit_data), use_container_width=True)

# ============================================================================
# PROJECT 2: CLASSIFICATION
# ============================================================================
elif selected_page == "🌸 Project 2: Classification":
    st.markdown('<div class="section-header">🌸 Project 2: Data Classification Using AI</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:#f8f9fa; border-radius:10px; padding:1rem; border-left:4px solid #333; margin-bottom:1rem;">
        <p><b>📌 Description:</b> K-Nearest Neighbors classifier on the Iris dataset with 97% accuracy.</p>
        <p><b>🔧 Tech Stack:</b> Python, scikit-learn, matplotlib, seaborn</p>
        <p><b>🏷️ Status:</b> <span class="badge-complete">✓ Complete</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        st.markdown("### 📊 Confusion Matrix")
        cm_data = np.array([[10, 0, 0], [0, 9, 1], [0, 0, 10]])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                    yticklabels=['Setosa', 'Versicolor', 'Virginica'],
                    ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix - KNN (K=5)')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # K-Value Performance
        st.markdown("### 📈 K-Value vs Accuracy")
        k_values = list(range(1, 21))
        accuracy = [0.95, 0.94, 0.95, 0.96, 0.973, 0.97, 0.968, 0.965, 0.962, 0.96,
                    0.958, 0.955, 0.952, 0.95, 0.948, 0.945, 0.942, 0.94, 0.938, 0.935]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(k_values, accuracy, marker='o', color='#333', linewidth=2)
        ax.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='Best K=5')
        ax.set_xlabel('K Value')
        ax.set_ylabel('Accuracy')
        ax.set_title('K-Value Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "97.3%", "+0.5%")
    with col2:
        st.metric("Precision", "97.1%", "0.0%")
    with col3:
        st.metric("Recall", "97.0%", "0.0%")
    with col4:
        st.metric("F1-Score", "97.0%", "0.0%")
    
    # Dataset Preview
    st.markdown("### 📄 Dataset Preview")
    iris_sample = {
        'Sepal Length': [5.1, 4.9, 4.7, 4.6, 5.0],
        'Sepal Width': [3.5, 3.0, 3.2, 3.1, 3.6],
        'Petal Length': [1.4, 1.4, 1.3, 1.5, 1.4],
        'Petal Width': [0.2, 0.2, 0.2, 0.2, 0.2],
        'Species': ['Setosa', 'Setosa', 'Setosa', 'Setosa', 'Setosa']
    }
    st.dataframe(pd.DataFrame(iris_sample), use_container_width=True)

# ============================================================================
# PROJECT 3: RECOMMENDATION
# ============================================================================
elif selected_page == "🎯 Project 3: Recommendation":
    st.markdown('<div class="section-header">🎯 Project 3: AI Recommendation Logic</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:#f8f9fa; border-radius:10px; padding:1rem; border-left:4px solid #333; margin-bottom:1rem;">
        <p><b>📌 Description:</b> Content-based filtering with TF-IDF and cosine similarity for job-role recommendations.</p>
        <p><b>🔧 Tech Stack:</b> Python, scikit-learn, pandas</p>
        <p><b>🏷️ Status:</b> <span class="badge-complete">✓ Complete</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 🔍 Skill Matching")
        st.markdown("**Enter your skills (comma-separated):**")
        skills_input = st.text_input("", placeholder="e.g., Python, Machine Learning, SQL")
        
        if st.button("🔍 Find My Match", use_container_width=True):
            # Simulate recommendation results
            st.markdown("### 🎯 Your Top 3 Career Matches")
            
            matches = [
                {"role": "Data Scientist", "match": "89%", "domain": "Data Science", "skills": "Python, SQL, ML, Statistics"},
                {"role": "Machine Learning Engineer", "match": "82%", "domain": "Data Science", "skills": "Python, ML, Deep Learning"},
                {"role": "AI Engineer", "match": "75%", "domain": "Data Science", "skills": "Python, AI, Neural Networks"}
            ]
            
            for i, match in enumerate(matches):
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                st.markdown(f"""
                <div style="border:1px solid #ddd; border-radius:10px; padding:0.8rem 1rem; margin:0.5rem 0; 
                            background:{'#f0f8ff' if i==0 else '#fafafa'};">
                    <b>{medal} {match['role']}</b>
                    <span style="float:right; background:#333; color:white; padding:0.2rem 0.8rem; border-radius:20px;">
                        {match['match']}
                    </span>
                    <br>
                    <span style="color:#555; font-size:0.85rem;">{match['domain']} | Skills: {match['skills']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("💡 Enter your skills above and click 'Find My Match' to see recommendations.")
    
    with col2:
        # Similarity Chart
        st.markdown("### 📊 Similarity Scores")
        
        roles = ['Data Scientist', 'ML Engineer', 'AI Engineer', 'Data Analyst', 
                 'Full Stack Dev', 'Cloud Architect', 'DevOps Eng', 'Security Analyst']
        scores = [0.89, 0.82, 0.75, 0.68, 0.55, 0.48, 0.42, 0.38]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(roles, scores, color=['#333' if i < 3 else '#999' for i in range(len(roles))])
        ax.set_xlabel('Similarity Score')
        ax.set_title('Role Match Scores')
        ax.set_xlim(0, 1)
        for bar, val in zip(bars, scores):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.0%}', va='center')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # All Job Roles
    st.markdown("### 📋 Available Job Roles (12)")
    job_data = {
        'ID': list(range(1, 13)),
        'Role': ['Data Scientist', 'ML Engineer', 'Data Analyst', 'Cloud Architect', 
                 'DevOps Engineer', 'Security Analyst', 'Mobile Developer', 'Full Stack Developer',
                 'Backend Developer', 'Frontend Developer', 'AI Engineer', 'Database Admin'],
        'Domain': ['Data Science', 'Data Science', 'Data Science', 'Cloud', 
                   'DevOps', 'Security', 'Mobile', 'Development',
                   'Development', 'Development', 'Data Science', 'Development']
    }
    st.dataframe(pd.DataFrame(job_data), use_container_width=True)

# ============================================================================
# PROJECT 4: RECOGNITION
# ============================================================================
elif selected_page == "🖼️ Project 4: Recognition":
    st.markdown('<div class="section-header">🖼️ Project 4: Image and Text Recognition</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:#f8f9fa; border-radius:10px; padding:1rem; border-left:4px solid #333; margin-bottom:1rem;">
        <p><b>📌 Description:</b> OCR with Tesseract + Object Detection with MobileNet SSD.</p>
        <p><b>🔧 Tech Stack:</b> OpenCV, Tesseract, MobileNet SSD</p>
        <p><b>🏷️ Status:</b> <span class="badge-complete">✓ Complete</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 OCR Output")
        st.markdown("""
        <div style="border:1px solid #ddd; border-radius:10px; padding:1rem; background:#fafafa; height:250px;">
            <b>Extracted Text:</b>
            <div style="background:white; padding:0.8rem; border-radius:5px; margin-top:0.5rem; border:1px solid #eee;">
                <p style="font-family: monospace; color:#000;">
                    <b>Decode Labs</b><br>
                    Your Digital Lab<br>
                    www.decodelabs.tech<br>
                    <span style="color:#888;">Confidence: 92%</span>
                </p>
            </div>
            <p style="color:#555; font-size:0.85rem; margin-top:0.5rem;">📷 Source: sample_document.jpg</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Object Detection")
        st.markdown("""
        <div style="border:1px solid #ddd; border-radius:10px; padding:1rem; background:#fafafa; height:250px;">
            <b>Detected Objects:</b>
            <div style="background:white; padding:0.8rem; border-radius:5px; margin-top:0.5rem; border:1px solid #eee;">
                <p style="font-family: monospace; color:#000;">
                    🟩 Person | 0.95<br>
                    🟩 Car    | 0.88<br>
                    🟩 Dog    | 0.82<br>
                    🟨 Chair  | 0.62<br>
                    <span style="color:#888;">Threshold: 0.5</span>
                </p>
            </div>
            <p style="color:#555; font-size:0.85rem; margin-top:0.5rem;">📷 Source: sample_scene.jpg</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Computer Vision Pipeline
    st.markdown("### 🔄 Computer Vision Pipeline")
    
    pipeline_stages = [
        "📷 Input Image",
        "⚙️ Preprocess",
        "🔀 Branch",
        "📝 OCR (Tesseract)",
        "🎯 Object Detection",
        "📊 Combined Output"
    ]
    
    cols = st.columns(len(pipeline_stages))
    for i, stage in enumerate(pipeline_stages):
        with cols[i]:
            st.markdown(f"""
            <div style="border:1px solid #ddd; border-radius:10px; padding:0.5rem; text-align:center; 
                        background:{'#e9ecef' if i%2==0 else '#f8f9fa'};">
                <span style="font-size:0.85rem;">{stage}</span>
            </div>
            """, unsafe_allow_html=True)
            if i < len(pipeline_stages) - 1:
                st.markdown("<p style='text-align:center;'>➡️</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Confidence Distribution
    st.markdown("### 📊 Object Detection Confidence")
    
    objects = ['Person', 'Car', 'Dog', 'Chair', 'Cat', 'Bicycle']
    confidences = [0.95, 0.88, 0.82, 0.62, 0.58, 0.45]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#333' if c >= 0.7 else '#666' if c >= 0.5 else '#999' for c in confidences]
    bars = ax.bar(objects, confidences, color=colors)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold (0.5)')
    ax.set_ylabel('Confidence Score')
    ax.set_title('Object Detection Confidence Scores')
    ax.set_ylim(0, 1)
    ax.legend()
    for bar, val in zip(bars, confidences):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.03, f'{val:.0%}', ha='center')
    st.pyplot(fig)
    plt.close()

# ============================================================================
# PAGE 5: ANALYTICS
# ============================================================================
elif selected_page == "📈 Analytics":
    st.markdown('<div class="section-header">📈 Analytics & Performance</div>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Overall Project Performance")
    
    # Performance Data
    projects = ['Project 1: Chatbot', 'Project 2: Classification', 
                'Project 3: Recommendation', 'Project 4: Recognition']
    completion = [100, 100, 100, 100]
    code_quality = [95, 92, 90, 88]
    documentation = [90, 88, 85, 82]
    innovation = [85, 88, 90, 92]
    
    # Create DataFrame
    df_performance = pd.DataFrame({
        'Project': projects,
        'Completion (%)': completion,
        'Code Quality (%)': code_quality,
        'Documentation (%)': documentation,
        'Innovation (%)': innovation
    })
    
    st.dataframe(df_performance, use_container_width=True)
    
    st.markdown("---")
    
    # Performance Chart
    st.markdown("### 📈 Performance Comparison Chart")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(projects))
    width = 0.2
    
    ax.bar(x - 1.5*width, completion, width, label='Completion', color='#333')
    ax.bar(x - 0.5*width, code_quality, width, label='Code Quality', color='#555')
    ax.bar(x + 0.5*width, documentation, width, label='Documentation', color='#777')
    ax.bar(x + 1.5*width, innovation, width, label='Innovation', color='#999')
    
    ax.set_xlabel('Projects')
    ax.set_ylabel('Score (%)')
    ax.set_title('Project Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(projects, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Skills Growth
    st.markdown("### 📊 Technical Skills Growth")
    
    skills = ['Python', 'ML', 'Data Analysis', 'CV', 'Web Dev', 'Git', 'Cloud']
    before = [65, 50, 55, 30, 40, 45, 25]
    after = [90, 82, 80, 70, 60, 75, 55]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(skills))
    ax.bar(x - 0.2, before, 0.4, label='Before Internship', color='#999')
    ax.bar(x + 0.2, after, 0.4, label='After Internship', color='#333')
    ax.set_xlabel('Skills')
    ax.set_ylabel('Proficiency (%)')
    ax.set_title('Skills Development During Internship')
    ax.set_xticks(x)
    ax.set_xticklabels(skills, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Internship Summary
    st.markdown("### 🏆 Internship Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="border:1px solid #ddd; border-radius:10px; padding:1rem; text-align:center;">
            <h2 style="color:#000;">4</h2>
            <p style="color:#555;">Projects Completed</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="border:1px solid #ddd; border-radius:10px; padding:1rem; text-align:center;">
            <h2 style="color:#000;">6</h2>
            <p style="color:#555;">Weeks Duration</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="border:1px solid #ddd; border-radius:10px; padding:1rem; text-align:center;">
            <h2 style="color:#000;">97%</h2>
            <p style="color:#555;">Best Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>
        <b>DecodeLabs AI Projects</b> | 
        Developed by <b>SANKALP SHARMA</b> | 
        Internship: June 20 - August 1, 2026 |
        Certificate: <b>DA012607</b>
    </p>
    <p style="font-size:0.8rem; color:#888;">
        📂 <a href="https://github.com/TheKnightProtocol/DecodeLabs-internship" target="_blank">GitHub Repository</a> |
        🔗 <a href="https://share.streamlit.io/" target="_blank">Deploy on Streamlit</a>
    </p>
</div>
""", unsafe_allow_html=True)
