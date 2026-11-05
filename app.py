import io
import importlib
from pathlib import Path
import sys
import uuid
import joblib
import pandas as pd
import streamlit as st

# ==============================================================================
# BASE DIRECTORY & DYNAMIC SERVICE IMPORTS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import services.bulk_processor
import services.feature_extractor
import services.resume_parser

importlib.reload(services.resume_parser)
importlib.reload(services.feature_extractor)
importlib.reload(services.bulk_processor)

from services.bulk_processor import process_bulk_resumes
from services.feature_extractor import DatasetBenchmark

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="ResumeAI Pro | Intelligent Screening",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# LOAD PRE-TRAINED ML MODEL
# ==============================================================================
MODEL_PATH = BASE_DIR / "models" / "resume_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model from {MODEL_PATH}: {e}")
    model = None

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "candidate_name" not in st.session_state:
    st.session_state["candidate_name"] = "Sample Candidate"
if "candidate_email" not in st.session_state:
    st.session_state["candidate_email"] = ""
if "candidate_phone" not in st.session_state:
    st.session_state["candidate_phone"] = ""
if "candidate_exp" not in st.session_state:
    st.session_state["candidate_exp"] = 2
if "candidate_skills" not in st.session_state:
    st.session_state["candidate_skills"] = 70.0
if "candidate_edu" not in st.session_state:
    st.session_state["candidate_edu"] = "Bachelors"
if "candidate_proj" not in st.session_state:
    st.session_state["candidate_proj"] = 5
if "candidate_len" not in st.session_state:
    st.session_state["candidate_len"] = 500
if "candidate_gh" not in st.session_state:
    st.session_state["candidate_gh"] = 200
if "candidate_detected_skills" not in st.session_state:
    st.session_state["candidate_detected_skills"] = []
if "candidate_imputed" not in st.session_state:
    st.session_state["candidate_imputed"] = {}
if "candidate_source" not in st.session_state:
    st.session_state["candidate_source"] = None
if "candidate_version" not in st.session_state:
    st.session_state["candidate_version"] = "v0"
if "bulk_results" not in st.session_state:
    st.session_state["bulk_results"] = None
if "selected_candidate_idx" not in st.session_state:
    st.session_state["selected_candidate_idx"] = 0

# ==============================================================================
# MODERN DARK GLASSMORPHISM STYLESHEET
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

* {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

.stApp {
    background:
        radial-gradient(circle at 10% 8%, rgba(139, 92, 246, 0.22), transparent 30%),
        radial-gradient(circle at 90% 12%, rgba(6, 182, 212, 0.18), transparent 30%),
        radial-gradient(circle at 50% 95%, rgba(99, 102, 241, 0.14), transparent 45%),
        linear-gradient(180deg, #070b14 0%, #0a0f1d 50%, #060911 100%);
    color: #f1f5f9;
}

.block-container {
    max-width: 1140px;
    padding-top: 36px !important;
    padding-bottom: 60px !important;
}

/* Header & Brand */
.brand-strip {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}

.brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(139, 92, 246, 0.22);
    border: 1px solid rgba(167, 139, 250, 0.55);
    padding: 7px 16px;
    border-radius: 9999px;
    font-size: 12.5px;
    font-weight: 800;
    letter-spacing: 0.8px;
    color: #ddd6fe;
    box-shadow: 0 0 18px rgba(139, 92, 246, 0.3);
}

.live-dot {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 10px #10b981;
    animation: pulseDot 2s infinite;
}

@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
}

.hero-heading {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    line-height: 1.15;
    margin: 8px 0 12px 0;
    color: #ffffff;
}

.gradient-title {
    background: linear-gradient(135deg, #c4b5fd 0%, #a78bfa 40%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    text-align: center;
    color: #e2e8f0 !important;
    -webkit-text-fill-color: #e2e8f0 !important;
    font-size: 16.5px !important;
    font-weight: 500 !important;
    line-height: 1.65;
    max-width: 780px;
    margin: 0 auto 30px auto;
}

/* Headings and Paragraphs */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 800 !important;
}

p, span, label {
    color: #f1f5f9;
}

.stCaption, .stCaption p {
    color: #cbd5e1 !important;
    -webkit-text-fill-color: #cbd5e1 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px !important;
    background: rgba(17, 27, 49, 0.95) !important;
    padding: 8px !important;
    border-radius: 14px !important;
    border: 1px solid rgba(139, 92, 246, 0.4) !important;
    margin-bottom: 24px !important;
}

.stTabs [data-baseweb="tab"] {
    height: 48px !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    -webkit-text-fill-color: #e2e8f0 !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 0 24px !important;
    transition: all 0.25s ease !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    background: rgba(255, 255, 255, 0.05) !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background: rgba(139, 92, 246, 0.25) !important;
    border-color: rgba(167, 139, 250, 0.5) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid #c4b5fd !important;
    box-shadow: 0 4px 18px rgba(124, 58, 237, 0.45) !important;
}

.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span,
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span {
    color: inherit !important;
    -webkit-text-fill-color: inherit !important;
    font-weight: 700 !important;
}

/* Containers & Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(13, 20, 36, 0.9) !important;
    border: 1px solid rgba(71, 85, 105, 0.7) !important;
    border-radius: 18px !important;
    padding: 22px 24px !important;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45) !important;
    backdrop-filter: blur(14px);
}

/* Metrics Cards */
div[data-testid="stMetric"] {
    background: rgba(17, 27, 49, 0.95) !important;
    border: 1px solid rgba(139, 92, 246, 0.4) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4) !important;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricLabel"] span,
div[data-testid="stMetricLabel"] div {
    color: #e2e8f0 !important;
    -webkit-text-fill-color: #e2e8f0 !important;
    font-size: 14.5px !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div,
div[data-testid="stMetricValue"] p {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}

/* Radio Buttons & Filters */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span,
div[data-testid="stRadio"] [data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}

div[data-testid="stRadio"] [role="radiogroup"] label {
    background: rgba(22, 33, 58, 0.85) !important;
    padding: 7px 16px !important;
    border-radius: 10px !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    margin-right: 10px !important;
}

div[data-testid="stRadio"] [role="radiogroup"] label p {
    color: #f1f5f9 !important;
    -webkit-text-fill-color: #f1f5f9 !important;
    font-weight: 700 !important;
}

/* Badges & Pills */
.skill-pill {
    display: inline-block;
    background: rgba(139, 92, 246, 0.25);
    color: #ede9fe;
    border: 1px solid rgba(167, 139, 250, 0.5);
    border-radius: 14px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 700;
    margin: 2px 4px 2px 0;
}

.verdict-shortlisted {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.5);
    border-radius: 8px;
    padding: 5px 12px;
    font-weight: 800;
    font-size: 13px;
}

.verdict-rejected {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(244, 63, 94, 0.2);
    color: #fb7185;
    border: 1px solid rgba(251, 113, 133, 0.5);
    border-radius: 8px;
    padding: 5px 12px;
    font-weight: 800;
    font-size: 13px;
}

/* File Uploader Outer Box */
[data-testid="stFileUploader"],
section[data-testid="stFileUploader"],
div[data-testid="stFileUploader"],
.stFileUploader {
    background: rgba(13, 20, 36, 0.95) !important;
    border: 1px solid rgba(71, 85, 105, 0.6) !important;
    border-radius: 16px !important;
    padding: 18px 22px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label p,
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    margin-bottom: 10px !important;
    letter-spacing: 0.2px !important;
}

/* File Uploader Dropzone - Dark glassmorphism with vivid violet dashed border */
[data-testid="stFileUploaderDropzone"],
section[data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploaderDropzone"],
[data-baseweb="file-uploader"],
section[data-testid="stFileUploaderDropzone"] > div {
    background: #0d1527 !important;
    background-color: #0d1527 !important;
    border: 2px dashed #8b5cf6 !important;
    border-radius: 12px !important;
    padding: 22px 26px !important;
    box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.6) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploaderDropzone"]:hover,
section[data-testid="stFileUploaderDropzone"]:hover,
div[data-testid="stFileUploaderDropzone"]:hover {
    background: #121c33 !important;
    background-color: #121c33 !important;
    border-color: #c4b5fd !important;
    box-shadow: inset 0 2px 14px rgba(139, 92, 246, 0.3), 0 0 15px rgba(139, 92, 246, 0.25) !important;
}

/* Dropzone Upload Button */
[data-testid="stFileUploaderDropzone"] button,
section[data-testid="stFileUploaderDropzone"] button,
div[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] [data-testid="baseButton-secondary"],
[data-testid="stFileUploaderDropzone"] [data-baseweb="button"] {
    width: auto !important;
    min-width: 110px !important;
    min-height: 42px !important;
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
    background-color: #7c3aed !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid #c4b5fd !important;
    border-radius: 10px !important;
    padding: 8px 24px !important;
    font-size: 14.5px !important;
    font-weight: 800 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 16px rgba(124, 58, 237, 0.5) !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
}

[data-testid="stFileUploaderDropzone"] button:hover,
section[data-testid="stFileUploaderDropzone"] button:hover,
div[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploader"] [data-testid="baseButton-secondary"]:hover {
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
    background-color: #8b5cf6 !important;
    border-color: #ffffff !important;
    box-shadow: 0 6px 22px rgba(139, 92, 246, 0.7) !important;
    transform: translateY(-2px) !important;
}

[data-testid="stFileUploaderDropzone"] button *,
section[data-testid="stFileUploaderDropzone"] button *,
div[data-testid="stFileUploaderDropzone"] button *,
[data-testid="stFileUploader"] [data-testid="baseButton-secondary"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    fill: #ffffff !important;
    font-weight: 800 !important;
}

/* Dropzone Instructions Text */
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] *,
section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] *,
section[data-testid="stFileUploaderDropzone"] small,
section[data-testid="stFileUploaderDropzone"] span,
section[data-testid="stFileUploaderDropzone"] div:not(button *) {
    color: #e2e8f0 !important;
    -webkit-text-fill-color: #e2e8f0 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
}

/* Uploaded Files Item in Dropzone */
[data-testid="stFileUploaderFile"],
div[data-testid="stFileUploaderFile"] {
    background: rgba(22, 33, 58, 0.9) !important;
    border: 1px solid rgba(139, 92, 246, 0.45) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-top: 10px !important;
}

[data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFile"] small {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploaderFile"] button,
[data-testid="stFileUploaderDeleteBtn"] {
    color: #f87171 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    min-height: auto !important;
    width: auto !important;
    padding: 4px 8px !important;
}

/* Action Buttons (Primary, Secondary, Download) */
div[data-testid="stButton"] button,
div[data-testid="stDownloadButton"] button,
.stButton > button {
    width: 100% !important;
    min-height: 48px !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid #c4b5fd !important;
    font-size: 15px !important;
    font-weight: 800 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.45) !important;
    transition: all 0.25s ease !important;
}

div[data-testid="stButton"] button:hover,
div[data-testid="stDownloadButton"] button:hover,
.stButton > button:hover {
    background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-color: #ffffff !important;
    box-shadow: 0 10px 28px rgba(124, 58, 237, 0.6) !important;
    transform: translateY(-2px) !important;
}

div[data-testid="stButton"] button p,
div[data-testid="stButton"] button span,
div[data-testid="stDownloadButton"] button p,
div[data-testid="stDownloadButton"] button span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 800 !important;
}

/* Sliders */
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] label p {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}

div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"],
div[data-testid="stSlider"] [data-baseweb="slider"] div {
    color: #e2e8f0 !important;
    -webkit-text-fill-color: #e2e8f0 !important;
    font-weight: 600 !important;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: #172033 !important;
    border: 1px solid #64748b !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 600 !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(139, 92, 246, 0.4) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Text Area */
div[data-testid="stTextArea"] textarea {
    background-color: #0b1120 !important;
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    border: 1px solid #475569 !important;
    border-radius: 10px !important;
    font-size: 14px !important;
}

div[data-testid="stTextArea"] label,
div[data-testid="stTextArea"] label p {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 14.5px !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: rgba(17, 27, 49, 0.9) !important;
    border: 1px solid rgba(139, 92, 246, 0.4) !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
}

div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] summary span {
    font-weight: 700 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 15px !important;
}

div[data-testid="stExpander"] summary:hover {
    color: #c4b5fd !important;
    -webkit-text-fill-color: #c4b5fd !important;
}

/* Footer */
.app-footer {
    text-align: center;
    color: #cbd5e1 !important;
    -webkit-text-fill-color: #cbd5e1 !important;
    font-size: 14px;
    font-weight: 500;
    margin-top: 50px;
    padding-top: 22px;
    border-top: 1px solid rgba(71, 85, 105, 0.6);
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# HEADER HERO SECTION
# ==============================================================================
st.markdown(
    '''
    <div class="brand-strip">
        <span class="brand-badge">
            <span class="live-dot"></span>
            RESUMEAI PRO · NEURAL SCREENING ENGINE
        </span>
    </div>
    <div class="hero-heading">
        Intelligent Candidate <span class="gradient-title">Screening & Profiling</span>
    </div>
    <div class="hero-subtitle">
        Automated parsing, feature extraction, and ML shortlisting powered by Gradient Boosting.
        Upload raw resumes to screen batches instantly, or simulate candidates with interactive controls.
    </div>
    ''',
    unsafe_allow_html=True
)

# ==============================================================================
# MAIN TABS ARCHITECTURE
# ==============================================================================
tab_screening, tab_simulator = st.tabs([
    "📂 Automated Resume Screening",
    "👤 Candidate Simulator & Tuning"
])

# ==============================================================================
# TAB 1: AUTOMATED RESUME SCREENING (BATCH & SINGLE)
# ==============================================================================
with tab_screening:
    st.markdown("### 📤 Upload Resumes for Automated Evaluation")
    st.write("Upload candidate resumes in **PDF, DOCX, DOC, TXT**, or **Image** format. Text, contact details, experience, skills, and projects are extracted automatically.")

    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded_files = st.file_uploader(
            "Select resumes to screen",
            type=["pdf", "docx", "doc", "txt", "png", "jpg", "jpeg", "tiff", "bmp"],
            accept_multiple_files=True,
            help="Upload one or multiple resumes to screen simultaneously."
        )

    with col_sample:
        st.markdown("**Quick Test Sample**")
        sample_path = BASE_DIR / "uploads" / "HansikaGoel.pdf"
        if sample_path.exists():
            st.caption("Included test resume: `HansikaGoel.pdf`")
            if st.button("📄 Screen Sample Resume"):
                with st.spinner("Processing sample resume..."):
                    sample_results = process_bulk_resumes(str(BASE_DIR / "uploads"), model=model)
                st.session_state["bulk_results"] = sample_results
                if sample_results:
                    first = sample_results[0]
                    feats = first["features"]
                    st.session_state["selected_candidate_idx"] = 0
                    st.session_state["candidate_name"] = feats.get("name", "Candidate")
                    st.session_state["candidate_email"] = feats.get("email", "")
                    st.session_state["candidate_phone"] = feats.get("phone", "")
                    st.session_state["candidate_exp"] = feats.get("years_experience", 0)
                    st.session_state["candidate_skills"] = feats.get("skills_match_score", 70.0)
                    st.session_state["candidate_edu"] = feats.get("education_level", "Bachelors")
                    st.session_state["candidate_proj"] = feats.get("project_count", 3)
                    st.session_state["candidate_len"] = feats.get("resume_length", 300)
                    st.session_state["candidate_gh"] = feats.get("github_activity", 200)
                    st.session_state["candidate_detected_skills"] = feats.get("skills_detected", [])
                    st.session_state["candidate_imputed"] = feats.get("imputed_fields", {})
                    st.session_state["candidate_source"] = first["clean_filename"]
                    st.session_state["candidate_version"] = str(uuid.uuid4())
                st.rerun()

    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) ready for processing.")
        btn_col1, btn_col2 = st.columns([2, 1])
        with btn_col1:
            process_clicked = st.button("⚡ PROCESS & SCREEN ALL RESUMES", use_container_width=True)
        with btn_col2:
            if st.session_state.get("bulk_results"):
                if st.button("🗑️ Clear Results", use_container_width=True):
                    st.session_state["bulk_results"] = None
                    st.rerun()

        if process_clicked:
            upload_batch_dir = BASE_DIR / "uploads" / str(uuid.uuid4())
            upload_batch_dir.mkdir(parents=True, exist_ok=True)

            for idx, file in enumerate(uploaded_files):
                fname = Path(file.name).name
                save_dest = upload_batch_dir / f"{idx}_{fname}"
                with open(save_dest, "wb") as f:
                    f.write(file.getbuffer())

            with st.spinner("Extracting features, evaluating skills, and querying ML model..."):
                results = process_bulk_resumes(str(upload_batch_dir), model=model)

            st.session_state["bulk_results"] = results

            # Select first candidate into simulator
            successes = [r for r in results if r["status"] == "Success"]
            if successes:
                first = successes[0]
                feats = first["features"]
                st.session_state["selected_candidate_idx"] = 0
                st.session_state["candidate_name"] = feats.get("name", "Candidate")
                st.session_state["candidate_email"] = feats.get("email", "")
                st.session_state["candidate_phone"] = feats.get("phone", "")
                st.session_state["candidate_exp"] = feats.get("years_experience", 0)
                st.session_state["candidate_skills"] = feats.get("skills_match_score", 70.0)
                st.session_state["candidate_edu"] = feats.get("education_level", "Bachelors")
                st.session_state["candidate_proj"] = feats.get("project_count", 3)
                st.session_state["candidate_len"] = feats.get("resume_length", 300)
                st.session_state["candidate_gh"] = feats.get("github_activity", 200)
                st.session_state["candidate_detected_skills"] = feats.get("skills_detected", [])
                st.session_state["candidate_imputed"] = feats.get("imputed_fields", {})
                st.session_state["candidate_source"] = first["clean_filename"]
                st.session_state["candidate_version"] = str(uuid.uuid4())

            st.success(f"✓ Screening complete! Processed {len(results)} resumes.")
            st.rerun()

    # RENDER BATCH SCREENING RESULTS
    if st.session_state.get("bulk_results"):
        results = st.session_state["bulk_results"]
        success_list = [r for r in results if r["status"] == "Success"]
        total_count = len(results)
        success_count = len(success_list)
        shortlisted_count = sum(r.get("shortlisted") == "Yes" for r in success_list)
        pass_rate = (shortlisted_count / success_count * 100) if success_count > 0 else 0

        st.markdown("---")
        st.markdown("### 🏆 Screening Leaderboard & Overview")

        # Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Resumes", total_count)
        m2.metric("Parsed Successfully", success_count)
        m3.metric("AI Shortlisted", f"{shortlisted_count} ({pass_rate:.1f}%)")
        m4.metric("Not Shortlisted", total_count - shortlisted_count)

        # Build Interactive Table Data
        table_records = []
        for r in results:
            feats = r.get("features", {})
            prob = r.get("probability")
            table_records.append({
                "Filename": r.get("clean_filename", r.get("filename")),
                "Candidate Name": feats.get("name", "N/A"),
                "Education": feats.get("education_level", "N/A"),
                "Experience": f"{feats.get('years_experience', 0)} yrs",
                "Skills Score": f"{feats.get('skills_match_score', 0):.1f}%",
                "Projects": feats.get("project_count", 0),
                "GitHub Act.": feats.get("github_activity", 0),
                "Decision": "✓ Shortlisted" if r.get("shortlisted") == "Yes" else "✕ Not Shortlisted",
                "Match Confidence": f"{prob*100:.1f}%" if prob is not None else "N/A"
            })

        df_display = pd.DataFrame(table_records)

        # Filter bar
        filter_col1, filter_col2 = st.columns([3, 1])
        with filter_col1:
            status_filter = st.radio(
                "Filter candidates by decision:",
                ["All Candidates", "Shortlisted Only", "Not Shortlisted"],
                horizontal=True
            )
        with filter_col2:
            # Download CSV report button
            csv_buffer = io.StringIO()
            df_display.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Export Report (CSV)",
                data=csv_buffer.getvalue(),
                file_name="resume_screening_report.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Apply filter
        if status_filter == "Shortlisted Only":
            df_filtered = df_display[df_display["Decision"] == "✓ Shortlisted"]
        elif status_filter == "Not Shortlisted":
            df_filtered = df_display[df_display["Decision"] == "✕ Not Shortlisted"]
        else:
            df_filtered = df_display

        st.dataframe(df_filtered, use_container_width=True, hide_index=True)

        # Candidate Detailed Profiles Expanders
        st.markdown("#### 🔍 Candidate Dossiers & Extraction Details")
        for i, res in enumerate(results):
            c_name = res.get("features", {}).get("name", "Candidate")
            c_file = res.get("clean_filename", res.get("filename"))
            c_decision = res.get("shortlisted", "N/A")
            tag = "🟢 SHORTLISTED" if c_decision == "Yes" else "🔴 NOT SHORTLISTED"

            with st.expander(f"{tag} — {c_name} ({c_file})"):
                if res["status"] == "Success":
                    f = res["features"]
                    cols_info = st.columns([3, 1])
                    with cols_info[0]:
                        st.markdown(f"**Contact:** 📧 `{f.get('email', 'N/A')}` | 📞 `{f.get('phone', 'N/A')}`")
                        st.markdown(f"**Education:** `{f.get('education_level')}` | **Experience:** `{f.get('years_experience')} years` | **Projects:** `{f.get('project_count')}`")

                        skills = f.get("skills_detected", [])
                        if skills:
                            pills_html = "".join([f'<span class="skill-pill">{s}</span>' for s in skills[:18]])
                            st.markdown(f"**Skills Detected:** {pills_html}", unsafe_allow_html=True)

                        # Imputation notice if any
                        imputed = f.get("imputed_fields", {})
                        if imputed:
                            imputed_lines = "".join([f"<li><code>{k}</code>: {v}</li>" for k, v in imputed.items()])
                            st.markdown(
                                f"""
                                <div style="background: rgba(14, 165, 233, 0.1); border-left: 3px solid #38bdf8; padding: 6px 12px; border-radius: 6px; margin: 8px 0; font-size: 12px; color: #cbd5e1;">
                                    <strong>Dataset Imputations Applied:</strong>
                                    <ul style="margin: 2px 0 0 16px; padding: 0;">{imputed_lines}</ul>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    with cols_info[1]:
                        if st.button("➡️ Load into Simulator", key=f"load_sim_{i}"):
                            st.session_state["candidate_name"] = f.get("name", "Candidate")
                            st.session_state["candidate_email"] = f.get("email", "")
                            st.session_state["candidate_phone"] = f.get("phone", "")
                            st.session_state["candidate_exp"] = f.get("years_experience", 0)
                            st.session_state["candidate_skills"] = f.get("skills_match_score", 70.0)
                            st.session_state["candidate_edu"] = f.get("education_level", "Bachelors")
                            st.session_state["candidate_proj"] = f.get("project_count", 3)
                            st.session_state["candidate_len"] = f.get("resume_length", 300)
                            st.session_state["candidate_gh"] = f.get("github_activity", 200)
                            st.session_state["candidate_detected_skills"] = f.get("skills_detected", [])
                            st.session_state["candidate_imputed"] = f.get("imputed_fields", {})
                            st.session_state["candidate_source"] = c_file
                            st.session_state["candidate_version"] = str(uuid.uuid4())
                            st.toast(f"Loaded {c_name} into Simulator tab!", icon="✓")

                    st.text_area(
                        "Extracted & Cleaned Resume Text",
                        res["extracted_text"],
                        height=180,
                        key=f"text_area_{i}"
                    )
                else:
                    st.error(f"Failed to process: {res.get('error')}")

# ==============================================================================
# TAB 2: CANDIDATE SIMULATOR & WHAT-IF STUDIO
# ==============================================================================
with tab_simulator:
    st.markdown("### 👤 Candidate Profile Simulator & What-If Studio")
    st.write("Tune candidate parameters or test simulated profiles to observe how experience, degree, projects, and skills impact ML shortlisting.")

    # Preset Profiles
    st.markdown("**Load Preset Role Archetype:**")
    preset_cols = st.columns(4)
    with preset_cols[0]:
        if st.button("🎓 College Fresher"):
            st.session_state["candidate_name"] = "Alex Kumar (Fresher)"
            st.session_state["candidate_exp"] = 0
            st.session_state["candidate_skills"] = 82.0
            st.session_state["candidate_edu"] = "Bachelors"
            st.session_state["candidate_proj"] = 4
            st.session_state["candidate_len"] = 320
            st.session_state["candidate_gh"] = 280
            st.session_state["candidate_source"] = "Preset: College Fresher"
            st.session_state["candidate_imputed"] = {}
            st.session_state["candidate_version"] = str(uuid.uuid4())
            st.rerun()

    with preset_cols[1]:
        if st.button("💻 Mid-Level Engineer"):
            st.session_state["candidate_name"] = "Jordan Rivera (Mid-Level)"
            st.session_state["candidate_exp"] = 4
            st.session_state["candidate_skills"] = 78.0
            st.session_state["candidate_edu"] = "Bachelors"
            st.session_state["candidate_proj"] = 8
            st.session_state["candidate_len"] = 520
            st.session_state["candidate_gh"] = 320
            st.session_state["candidate_source"] = "Preset: Mid-Level Engineer"
            st.session_state["candidate_imputed"] = {}
            st.session_state["candidate_version"] = str(uuid.uuid4())
            st.rerun()

    with preset_cols[2]:
        if st.button("🚀 Senior Tech Lead"):
            st.session_state["candidate_name"] = "Morgan Chen (Tech Lead)"
            st.session_state["candidate_exp"] = 10
            st.session_state["candidate_skills"] = 92.0
            st.session_state["candidate_edu"] = "Masters"
            st.session_state["candidate_proj"] = 14
            st.session_state["candidate_len"] = 680
            st.session_state["candidate_gh"] = 500
            st.session_state["candidate_source"] = "Preset: Senior Tech Lead"
            st.session_state["candidate_imputed"] = {}
            st.session_state["candidate_version"] = str(uuid.uuid4())
            st.rerun()

    with preset_cols[3]:
        if st.button("🔬 AI Research Scientist"):
            st.session_state["candidate_name"] = "Dr. Elena Rostova (PhD)"
            st.session_state["candidate_exp"] = 6
            st.session_state["candidate_skills"] = 96.0
            st.session_state["candidate_edu"] = "PhD"
            st.session_state["candidate_proj"] = 12
            st.session_state["candidate_len"] = 750
            st.session_state["candidate_gh"] = 580
            st.session_state["candidate_source"] = "Preset: AI Research Scientist"
            st.session_state["candidate_imputed"] = {}
            st.session_state["candidate_version"] = str(uuid.uuid4())
            st.rerun()

    # Active Candidate Source Notification
    if st.session_state.get("candidate_source"):
        c_src = st.session_state["candidate_source"]
        c_nm = st.session_state.get("candidate_name", "Candidate")
        st.info(f"✦ Active profile loaded from: **{c_nm}** ({c_src})")

    # Interactive Form Sliders with Dynamic Keys
    cand_v = st.session_state.get("candidate_version", "default")
    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        years_exp = st.slider(
            "Years of Professional Experience",
            min_value=0,
            max_value=30,
            value=int(st.session_state.get("candidate_exp", 2)),
            key=f"sim_exp_{cand_v}"
        )

        skills_score = st.slider(
            "Skills Match Alignment Score (%)",
            min_value=0,
            max_value=100,
            value=int(round(st.session_state.get("candidate_skills", 70.0))),
            key=f"sim_skills_{cand_v}"
        )

        edu_opts = ["High School", "Bachelors", "Masters", "PhD"]
        current_edu = st.session_state.get("candidate_edu", "Bachelors")
        edu_idx = edu_opts.index(current_edu) if current_edu in edu_opts else 1

        edu_level = st.selectbox(
            "Highest Education Degree",
            edu_opts,
            index=edu_idx,
            key=f"sim_edu_{cand_v}"
        )

    with sim_col2:
        proj_count = st.slider(
            "Number of Completed Projects",
            min_value=0,
            max_value=30,
            value=int(st.session_state.get("candidate_proj", 5)),
            key=f"sim_proj_{cand_v}"
        )

        resume_len = st.slider(
            "Resume Word Count",
            min_value=50,
            max_value=1500,
            value=int(st.session_state.get("candidate_len", 500)),
            key=f"sim_len_{cand_v}"
        )

        gh_act = st.slider(
            "GitHub & Open Source Activity Score",
            min_value=0,
            max_value=1000,
            value=int(st.session_state.get("candidate_gh", 200)),
            key=f"sim_gh_{cand_v}"
        )

    # RUN INFERENCE
    candidate_payload = pd.DataFrame([{
        "years_experience": years_exp,
        "skills_match_score": skills_score,
        "education_level": edu_level,
        "project_count": proj_count,
        "resume_length": resume_len,
        "github_activity": gh_act
    }])

    if model:
        pred = int(model.predict(candidate_payload)[0])
        prob = float(model.predict_proba(candidate_payload)[0][1])
        prob_pct = prob * 100
    else:
        pred = 0
        prob = 0.0
        prob_pct = 0.0

    st.markdown("---")
    st.markdown("### ✦ AI Evaluation Verdict")

    # Decision Banner
    verdict_col1, verdict_col2 = st.columns([2, 1])
    with verdict_col1:
        if pred == 1:
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.15); border: 2px solid #10b981; border-radius: 14px; padding: 20px; box-shadow: 0 0 25px rgba(16, 185, 129, 0.2);">
                    <div style="font-size: 22px; font-weight: 800; color: #34d399; margin-bottom: 4px;">
                        ✓ CANDIDATE SHORTLISTED FOR INTERVIEW
                    </div>
                    <div style="color: #cbd5e1; font-size: 14px;">
                        Candidate demonstrates a high probability of successful role alignment based on experience, skill parity, and execution metrics.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background: rgba(244, 63, 94, 0.15); border: 2px solid #f43f5e; border-radius: 14px; padding: 20px; box-shadow: 0 0 25px rgba(244, 63, 94, 0.2);">
                    <div style="font-size: 22px; font-weight: 800; color: #fb7185; margin-bottom: 4px;">
                        ✕ CANDIDATE NOT SHORTLISTED
                    </div>
                    <div style="color: #cbd5e1; font-size: 14px;">
                        Candidate does not meet the current probability threshold for automatic shortlisting. Review recommendations below.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with verdict_col2:
        st.metric("Shortlisting Probability", f"{prob_pct:.1f}%")
        st.progress(prob, text=f"Match Confidence: {prob_pct:.1f}%")

    # Diagnostic Recommendations
    st.markdown("#### 💡 Diagnostic Candidate Takeaways")
    diag_c1, diag_c2 = st.columns(2)
    with diag_c1:
        st.markdown("**Candidate Strengths:**")
        strengths = []
        if skills_score >= 80:
            strengths.append("High skills match score indicating strong alignment with required tech stack.")
        if proj_count >= 8:
            strengths.append(f"Substantial hands-on portfolio with {proj_count} demonstrated projects.")
        if years_exp >= 5:
            strengths.append(f"Solid industry track record with {years_exp} years of domain experience.")
        if gh_act >= 350:
            strengths.append(f"High technical engagement and open-source contributions ({gh_act} score).")
        if edu_level in ["Masters", "PhD"]:
            strengths.append(f"Advanced degree credential ({edu_level}) supporting complex problem solving.")

        if not strengths:
            strengths.append("Emerging profile with baseline qualifications ready for initial screening.")

        for s in strengths:
            st.markdown(f"- {s}")

    with diag_c2:
        st.markdown("**Areas for Evaluation / Improvement:**")
        improvements = []
        if skills_score < 70:
            improvements.append("Skills alignment score is below the 70% benchmark; explore relevant technical upskilling.")
        if years_exp < 2:
            improvements.append("Limited formal experience; emphasize internships, open-source work, and project impact.")
        if proj_count < 4:
            improvements.append("Project count is modest; recommend building 2-3 production-grade full-stack apps.")
        if gh_act < 150:
            improvements.append("Low open source / public activity score; encourage sharing code repositories and solving algorithmic challenges.")

        if not improvements:
            improvements.append("Profile demonstrates well-rounded metrics across all evaluated dimensions.")

        for imp in improvements:
            st.markdown(f"- {imp}")

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown(
    '''
    <div class="app-footer">
        ResumeAI Pro · Autonomous Candidate Screening & Intelligent Talent Analytics<br>
        Engineered with Python · Scikit-Learn · Pandas · PyMuPDF · Streamlit
    </div>
    ''',
    unsafe_allow_html=True
)
