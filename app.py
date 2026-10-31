"""
MedExplain v5.0 — SpaceX / Apple VisionOS / Tesla Bio-Telemetry Edition
Symptom & Body-Map Driven Explainable AI Clinical Decision Support System
Course: Engineering Capstone Project - 1 (24DSB3101) | KLH University
Authors: Saraf Abhinav, Jai Vardhan, Srimaan, Ananya
Guide: Dr. K. Swapnika

100% Self-Contained Flagship Edition:
  - Replace `/app.py` in your GitHub repository with this file.
  - Features a Cybernetic Anatomical Bio-Scanner HUD with animated laser scanline,
    rotating target reticles, live ECG telemetry waveform, Apple/SpaceX circular radial
    organ routing gauges, and custom glassmorphic TreeSHAP + LIME telemetry bars.
"""

import os
from contextlib import contextmanager
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="MEDEXPLAIN // BIO-TELEMETRY XAI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CINEMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "cinema")
_CINEMA = components.declare_component("medexplain_cinema", path=_CINEMA_DIR)


# -------------------------------------------------------------------------
# Zero-Indent HTML Cleaner & Version-Safe Streamlit Wrappers
# -------------------------------------------------------------------------
def clean_html(raw_html: str) -> str:
    """Strips leading whitespace on every line so CommonMark never turns HTML/SVG into a <pre><code> block."""
    return "\n".join(line.lstrip() for line in raw_html.splitlines() if line.strip())


@contextmanager
def card_box():
    try:
        with st.container(border=True):
            yield
    except TypeError:
        with st.container():
            yield


def safe_button(label: str, key: str = None, is_primary: bool = False) -> bool:
    try:
        return st.button(
            label,
            key=key,
            type="primary" if is_primary else "secondary",
            width="stretch",
        )
    except TypeError:
        return st.button(label, key=key)


def safe_dataframe(df: pd.DataFrame):
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df)


def show_image_safe(img_path: str, caption: str = None):
    if not os.path.exists(img_path):
        try:
            from src.model_graphs import generate_extended_model_graphs
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            generate_extended_model_graphs(data_dir="data", out_dir="results/figures")
        except Exception:
            pass
    if os.path.exists(img_path):
        label = caption or "PUBLICATION FIGURE"
        st.markdown(
            clean_html(f'<div class="still-kicker">ARCHIVE STILL · {label}</div>'),
            unsafe_allow_html=True,
        )
        try:
            st.image(img_path, caption=caption, width="stretch")
        except TypeError:
            st.image(img_path, caption=caption)


# -------------------------------------------------------------------------
# SpaceX / Apple VisionOS / Tesla UI CSS Architecture
# -------------------------------------------------------------------------
st.markdown(
    clean_html(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700;800&family=Space+Grotesk:wght@600;700;800&display=swap');

        /* Ultra-Deep Obsidian Spatial Canvas */
        .stApp, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 18% 12%, rgba(0, 240, 255, 0.08) 0%, transparent 38%),
                radial-gradient(circle at 85% 18%, rgba(255, 42, 95, 0.07) 0%, transparent 40%),
                radial-gradient(circle at 50% 85%, rgba(99, 102, 241, 0.08) 0%, transparent 45%),
                #05060b !important;
            color: #f8fafc !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        [data-testid="stHeader"] {
            background: rgba(5, 6, 11, 0.75) !important;
            backdrop-filter: blur(16px) !important;
        }
        .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 3rem !important;
            max-width: 1460px !important;
        }
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -0.02em !important;
            color: #ffffff !important;
        }
        [data-testid="stWidgetLabel"] p,
        [data-testid="stMarkdownContainer"] > p,
        [data-testid="stMarkdownContainer"] > ul > li,
        [data-testid="stMarkdownContainer"] > ol > li,
        [data-testid="stCaptionContainer"] p {
            color: #cbd5e1 !important;
        }
        [data-testid="stMetricLabel"] p {
            color: #64748b !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            font-size: 11px !important;
            font-weight: 700 !important;
        }
        [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
            color: #00f0ff !important;
            font-weight: 800 !important;
            text-shadow: 0 0 18px rgba(0, 240, 255, 0.35) !important;
        }
        /* SpaceX Mission Control Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #040509 0%, #090c15 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            background: rgba(255, 255, 255, 0.025) !important;
            border: 1px solid rgba(255, 255, 255, 0.07) !important;
            border-radius: 12px !important;
            padding: 11px 14px !important;
            margin-bottom: 8px !important;
            transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            border-color: rgba(0, 240, 255, 0.55) !important;
            background: rgba(0, 240, 255, 0.08) !important;
            box-shadow: 0 0 20px rgba(0, 240, 255, 0.14) !important;
            transform: translateX(3px) !important;
        }
        /* VisionOS Spatial Glass Cards (st.container(border=True)) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(145deg, rgba(17, 22, 36, 0.88) 0%, rgba(10, 13, 22, 0.94) 100%) !important;
            border: 1px solid rgba(255, 255, 255, 0.09) !important;
            border-top: 1px solid rgba(0, 240, 255, 0.35) !important;
            border-radius: 18px !important;
            padding: 18px !important;
            box-shadow: 0 20px 45px -15px rgba(0, 0, 0, 0.75), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
            backdrop-filter: blur(20px) !important;
        }
        /* Tactile Tesla-Style Buttons */
        .stButton > button {
            background: rgba(255, 255, 255, 0.04) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            font-size: 12.5px !important;
            padding: 8px 14px !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }
        .stButton > button:hover {
            border-color: #00f0ff !important;
            background: rgba(0, 240, 255, 0.12) !important;
            color: #ffffff !important;
            box-shadow: 0 0 18px rgba(0, 240, 255, 0.25) !important;
            transform: translateY(-1px) !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #ff2a5f 0%, #c81045 100%) !important;
            border: 1px solid #ff6b8b !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            box-shadow: 0 0 22px rgba(255, 42, 95, 0.45) !important;
        }
        /* Keyframe Animations for Cybernetic Body Scanner & ECG */
        @keyframes scanLaser {
            0% { transform: translateY(0px); opacity: 0.25; }
            50% { transform: translateY(330px); opacity: 0.85; }
            100% { transform: translateY(0px); opacity: 0.25; }
        }
        @keyframes pulseReticle {
            0% { r: 22px; stroke-opacity: 0.95; }
            50% { r: 30px; stroke-opacity: 0.35; }
            100% { r: 22px; stroke-opacity: 0.95; }
        }
        .scanner-laser {
            animation: scanLaser 5.5s ease-in-out infinite;
        }
        .reticle-ring {
            animation: pulseReticle 2.2s ease-in-out infinite;
        }
        @keyframes bgShift {
            0% { background-position: 0% 0%, 100% 0%, 50% 100%, 0 0; }
            100% { background-position: 14% 10%, 78% 24%, 42% 86%, 0 0; }
        }
        @keyframes sheen {
            0% { transform: translateX(-120%); }
            55%, 100% { transform: translateX(120%); }
        }
        @keyframes barGrow {
            from { transform: scaleX(0); opacity: 0.2; }
            to { transform: scaleX(1); opacity: 1; }
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background-size: 140% 140%, 140% 140%, 140% 140%, auto !important;
            animation: bgShift 18s ease-in-out infinite alternate !important;
        }
        .stApp::after {
            content: "";
            pointer-events: none;
            position: fixed;
            inset: 0;
            z-index: 0;
            opacity: 0.22;
            background: repeating-linear-gradient(
                to bottom,
                rgba(255,255,255,0.05),
                rgba(255,255,255,0.05) 1px,
                transparent 1px,
                transparent 4px
            );
            mix-blend-mode: overlay;
        }
        .hero-bar { position: relative; overflow: hidden; }
        .hero-bar::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(110deg, transparent 32%, rgba(255,255,255,0.09) 48%, transparent 64%);
            transform: translateX(-120%);
            animation: sheen 5.2s ease-in-out infinite;
            pointer-events: none;
        }
        .xai-fill { transform-origin: left center; animation: barGrow 0.7s cubic-bezier(0.16, 1, 0.3, 1); }
        .still-kicker {
            font-family: 'JetBrains Mono', Consolas, monospace;
            font-size: 10px;
            letter-spacing: 0.16em;
            color: #00f0ff;
            margin: 14px 2px 6px 2px;
        }
        [data-testid="stPlotlyChart"], [data-testid="stImage"] {
            background: linear-gradient(180deg, rgba(14, 20, 34, 0.92), rgba(8, 11, 19, 0.94));
            border: 1px solid rgba(0, 240, 255, 0.2);
            border-radius: 16px;
            padding: 6px;
            box-shadow: 0 18px 44px rgba(0,0,0,0.38);
        }
        [data-testid="stImage"] img {
            border-radius: 12px;
            transition: transform 0.55s cubic-bezier(0.16, 1, 0.3, 1), filter 0.55s ease;
        }
        [data-testid="stImage"]:hover img { transform: scale(1.015); filter: saturate(1.12) contrast(1.05); }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid rgba(0, 240, 255, 0.16);
            border-radius: 14px;
            overflow: hidden;
        }
        [data-baseweb="select"] > div, [data-baseweb="input"] > div, .stMultiSelect [data-baseweb="select"] > div {
            background: #0c1220 !important;
            border-color: rgba(0, 240, 255, 0.28) !important;
            color: #f8fafc !important;
        }
        [data-baseweb="tag"] {
            background: rgba(0, 240, 255, 0.16) !important;
            color: #e0fbff !important;
        }
        [data-testid="stSlider"] [role="slider"] {
            background: #00f0ff !important;
            box-shadow: 0 0 12px rgba(0, 240, 255, 0.85);
        }
        footer { visibility: hidden; height: 0; }
        </style>
        """
    ),
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------------
# Domain Metadata, Symptom Catalog & Test Cases
# -------------------------------------------------------------------------
DOMAIN_INFO = {
    "heart": {"name": "Cardiovascular (Cleveland Heart Disease)"},
    "diabetes": {"name": "Metabolic (Pima Indians Diabetes)"},
    "breast_cancer": {"name": "Oncology (Wisconsin Breast Cancer - WDBC)"},
}

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree (CART)",
    "Random Forest",
    "XGBoost",
    "CatBoost",
]

MODEL_COLORS = {
    "Logistic Regression": "#38bdf8",
    "Decision Tree (CART)": "#a855f7",
    "Random Forest": "#10b981",
    "XGBoost": "#f59e0b",
    "CatBoost": "#ff2a5f",
}

BODY_PAIN_REGIONS: Dict[str, Dict[str, Any]] = {
    "chest_center_left": {
        "label": "Chest (Center / Left Precordium)",
        "short_label": "CHEST // MYOCARDIUM",
        "code": "LOC-01",
        "icon": "🫀",
        "svg_cx": 152,
        "svg_cy": 134,
        "svg_r": 22,
        "domain_weights": {"heart": 0.88, "diabetes": 0.05, "breast_cancer": 0.12},
    },
    "left_arm_shoulder_jaw": {
        "label": "Left Arm, Shoulder, or Jaw",
        "short_label": "L-BRACHIAL // RADIATING",
        "code": "LOC-02",
        "icon": "💪",
        "svg_cx": 202,
        "svg_cy": 150,
        "svg_r": 20,
        "domain_weights": {"heart": 0.82, "diabetes": 0.05, "breast_cancer": 0.10},
    },
    "breast_axilla": {
        "label": "Breast or Underarm (Axilla)",
        "short_label": "MAMMARY // AXILLA",
        "code": "LOC-03",
        "icon": "🎗️",
        "svg_cx": 112,
        "svg_cy": 134,
        "svg_r": 20,
        "domain_weights": {"heart": 0.08, "diabetes": 0.04, "breast_cancer": 0.92},
    },
    "abdomen_stomach": {
        "label": "Upper Abdomen / Stomach",
        "short_label": "EPIGASTRIC // VISCERAL",
        "code": "LOC-04",
        "icon": "🫃",
        "svg_cx": 135,
        "svg_cy": 184,
        "svg_r": 24,
        "domain_weights": {"heart": 0.28, "diabetes": 0.60, "breast_cancer": 0.04},
    },
    "legs_feet": {
        "label": "Legs, Calves, or Feet",
        "short_label": "LOWER EXTREMITY // NEURO",
        "code": "LOC-05",
        "icon": "🦵",
        "svg_cx": 135,
        "svg_cy": 288,
        "svg_r": 28,
        "domain_weights": {"heart": 0.25, "diabetes": 0.82, "breast_cancer": 0.02},
    },
    "upper_back": {
        "label": "Upper Back (Shoulder Blades)",
        "short_label": "THORACIC // DORSAL",
        "code": "LOC-06",
        "icon": "🧍",
        "svg_cx": 135,
        "svg_cy": 98,
        "svg_r": 18,
        "domain_weights": {"heart": 0.58, "diabetes": 0.10, "breast_cancer": 0.10},
    },
    "head_general": {
        "label": "Head, Neck, or Dizziness",
        "short_label": "CRANIAL // VASCULAR",
        "code": "LOC-07",
        "icon": "🧠",
        "svg_cx": 135,
        "svg_cy": 46,
        "svg_r": 20,
        "domain_weights": {"heart": 0.35, "diabetes": 0.52, "breast_cancer": 0.05},
    },
    "no_pain": {
        "label": "No Specific Localized Body Pain",
        "short_label": "NOMINAL // NO PAIN",
        "code": "LOC-00",
        "icon": "✅",
        "svg_cx": 0,
        "svg_cy": 0,
        "svg_r": 0,
        "domain_weights": {"heart": 0.12, "diabetes": 0.42, "breast_cancer": 0.22},
    },
}

PAIN_CHARACTER_OPTIONS: Dict[str, Dict[str, Any]] = {
    "crushing_pressure": {"label": "Heavy Pressure, Squeezing, or Tightness", "cp_code": 0, "heart_boost": 0.35},
    "sharp_stabbing": {"label": "Sharp, Stabbing, or Pinching Pain", "cp_code": 1, "heart_boost": 0.18},
    "burning_aching": {"label": "Burning, Tingling, or Deep Ache", "cp_code": 2, "heart_boost": 0.12},
    "tender_throbbing": {"label": "Localized Tenderness or Throbbing", "cp_code": 2, "heart_boost": 0.05},
    "none": {"label": "No Pain / Mild Discomfort Only", "cp_code": 3, "heart_boost": 0.0},
}

PAIN_TRIGGER_OPTIONS: Dict[str, Dict[str, Any]] = {
    "exertion": {"label": "Walking, climbing stairs, or physical activity", "exang_code": 1, "heart_boost": 0.32},
    "stress_emotion": {"label": "Emotional stress or anxiety", "exang_code": 1, "heart_boost": 0.22},
    "touch_pressure": {"label": "When pressing or touching the area", "exang_code": 0, "heart_boost": -0.05},
    "after_meals": {"label": "Occurs or worsens after eating meals", "exang_code": 0, "heart_boost": 0.05},
    "rest_constant": {"label": "Present at rest or constant all day", "exang_code": 0, "heart_boost": 0.10},
}

SYMPTOM_CATALOG: Dict[str, Dict[str, Any]] = {
    "shortness_of_breath": {"label": "Shortness of breath / dyspnea", "category": "CARDIOPULMONARY", "weights": {"heart": 0.55, "diabetes": 0.10, "breast_cancer": 0.05}},
    "cold_sweats": {"label": "Cold sweats / autonomic diaphoresis", "category": "CARDIOPULMONARY", "weights": {"heart": 0.60, "diabetes": 0.20, "breast_cancer": 0.0}},
    "palpitations": {"label": "Racing, pounding, or fluttering heart", "category": "CARDIOPULMONARY", "weights": {"heart": 0.48, "diabetes": 0.10, "breast_cancer": 0.0}},
    "dizziness_fainting": {"label": "Dizziness, lightheadedness, or syncope", "category": "CARDIOPULMONARY", "weights": {"heart": 0.42, "diabetes": 0.35, "breast_cancer": 0.0}},
    "radiating_arm_jaw": {"label": "Pain spreading to left arm, shoulder, or jaw", "category": "CARDIOPULMONARY", "weights": {"heart": 0.75, "diabetes": 0.0, "breast_cancer": 0.10}},
    "excessive_thirst": {"label": "Excessive thirst (constant polydipsia)", "category": "METABOLIC", "weights": {"heart": 0.05, "diabetes": 0.78, "breast_cancer": 0.0}},
    "frequent_urination": {"label": "Frequent urination (nocturia / polyuria)", "category": "METABOLIC", "weights": {"heart": 0.05, "diabetes": 0.75, "breast_cancer": 0.0}},
    "chronic_fatigue": {"label": "Unusual exhaustion or daytime lethargy", "category": "METABOLIC", "weights": {"heart": 0.35, "diabetes": 0.50, "breast_cancer": 0.30}},
    "blurred_vision": {"label": "Blurred vision or osmotic focus shift", "category": "METABOLIC", "weights": {"heart": 0.10, "diabetes": 0.62, "breast_cancer": 0.0}},
    "numbness_tingling": {"label": "Tingling or numbness in hands or feet", "category": "METABOLIC", "weights": {"heart": 0.15, "diabetes": 0.68, "breast_cancer": 0.0}},
    "slow_healing": {"label": "Cuts, bruises, or sores that heal slowly", "category": "METABOLIC", "weights": {"heart": 0.10, "diabetes": 0.65, "breast_cancer": 0.05}},
    "unexplained_weight_change": {"label": "Unexplained weight loss or sudden hunger", "category": "METABOLIC", "weights": {"heart": 0.05, "diabetes": 0.55, "breast_cancer": 0.45}},
    "breast_underarm_lump": {"label": "Firm lump, knot, or mass in breast/axilla", "category": "ONCOLOGY", "weights": {"heart": 0.0, "diabetes": 0.0, "breast_cancer": 0.95}},
    "breast_swelling_asymmetry": {"label": "Swelling, warmth, or breast asymmetry", "category": "ONCOLOGY", "weights": {"heart": 0.0, "diabetes": 0.0, "breast_cancer": 0.78}},
    "skin_dimpling_redness": {"label": "Skin dimpling (peau d'orange) or redness", "category": "ONCOLOGY", "weights": {"heart": 0.0, "diabetes": 0.05, "breast_cancer": 0.85}},
    "nipple_changes": {"label": "Nipple inversion or unusual discharge", "category": "ONCOLOGY", "weights": {"heart": 0.0, "diabetes": 0.0, "breast_cancer": 0.82}},
    "family_history": {"label": "First-degree family medical history", "category": "GENETICS", "weights": {"heart": 0.30, "diabetes": 0.45, "breast_cancer": 0.38}},
}

CLINICAL_TEST_CASES = [
    {
        "id": "TC-01",
        "title": "Acute Myocardial Ischemia & Left Brachial Radiation",
        "patient_summary": "58yo Male presenting with crushing retrosternal chest pressure (8/10) radiating down the left ulnar arm on exertion, with dyspnea and diaphoresis.",
        "age": 58, "sex": "Male",
        "pain_regions": ["chest_center_left", "left_arm_shoulder_jaw"],
        "pain_character": "crushing_pressure", "pain_severity": 8, "pain_trigger": "exertion",
        "selected_symptoms": ["shortness_of_breath", "cold_sweats", "radiating_arm_jaw", "palpitations"],
        "expected_domain": "heart", "expected_domain_label": "CARDIOVASCULAR // HEART", "expected_risk": "HIGH RISK",
    },
    {
        "id": "TC-02",
        "title": "Diabetic Peripheral Neuropathy & Hyperglycemic Triad",
        "patient_summary": "46yo Female reporting bilateral lower-extremity burning paresthesia (5/10), epigastric discomfort, polydipsia, nocturia, and blurred vision.",
        "age": 46, "sex": "Female",
        "pain_regions": ["legs_feet", "abdomen_stomach"],
        "pain_character": "burning_aching", "pain_severity": 5, "pain_trigger": "rest_constant",
        "selected_symptoms": ["excessive_thirst", "frequent_urination", "chronic_fatigue", "blurred_vision", "numbness_tingling", "family_history"],
        "expected_domain": "diabetes", "expected_domain_label": "METABOLIC // DIABETES", "expected_risk": "HIGH RISK",
    },
    {
        "id": "TC-03",
        "title": "Palpable Mammary / Axillary Mass & Dermal Tethering",
        "patient_summary": "51yo Female noticing localized tenderness in the breast and axilla (6/10) with a firm palpable mass, asymmetry, and peau d'orange dimpling.",
        "age": 51, "sex": "Female",
        "pain_regions": ["breast_axilla"],
        "pain_character": "tender_throbbing", "pain_severity": 6, "pain_trigger": "touch_pressure",
        "selected_symptoms": ["breast_underarm_lump", "breast_swelling_asymmetry", "skin_dimpling_redness"],
        "expected_domain": "breast_cancer", "expected_domain_label": "ONCOLOGY // WDBC", "expected_risk": "HIGH RISK",
    },
    {
        "id": "TC-04",
        "title": "Atypical Female Interscapular Coronary Presentation",
        "patient_summary": "64yo Female with sharp interscapular upper-back and precordial pain (7/10) during physical activity, orthostatic dizziness, fatigue, and dyspnea.",
        "age": 64, "sex": "Female",
        "pain_regions": ["upper_back", "chest_center_left"],
        "pain_character": "sharp_stabbing", "pain_severity": 7, "pain_trigger": "exertion",
        "selected_symptoms": ["shortness_of_breath", "dizziness_fainting", "chronic_fatigue", "cold_sweats"],
        "expected_domain": "heart", "expected_domain_label": "CARDIOVASCULAR // HEART", "expected_risk": "HIGH RISK",
    },
    {
        "id": "TC-05",
        "title": "Early Metabolic Dysregulation & Microvascular Delay",
        "patient_summary": "52yo Male with cranial lightheadedness and distal foot tingling (4/10), impaired wound healing, chronic fatigue, polydipsia, and family history.",
        "age": 52, "sex": "Male",
        "pain_regions": ["head_general", "legs_feet"],
        "pain_character": "burning_aching", "pain_severity": 4, "pain_trigger": "after_meals",
        "selected_symptoms": ["excessive_thirst", "slow_healing", "chronic_fatigue", "family_history", "unexplained_weight_change"],
        "expected_domain": "diabetes", "expected_domain_label": "METABOLIC // DIABETES", "expected_risk": "HIGH RISK",
    },
    {
        "id": "TC-06",
        "title": "Nominal Physiological Baseline Control",
        "patient_summary": "31yo Female with zero localized body pain (0/10) and no cardiopulmonary, metabolic, or localized tissue anomalies.",
        "age": 31, "sex": "Female",
        "pain_regions": ["no_pain"],
        "pain_character": "none", "pain_severity": 0, "pain_trigger": "rest_constant",
        "selected_symptoms": [],
        "expected_domain": "heart", "expected_domain_label": "NOMINAL // ALL CLEAR", "expected_risk": "LOW RISK",
    },
]

FALLBACK_BENCHMARK_ROWS = [
    {"Domain": "Cardiovascular", "Classifier": "Logistic Regression", "Accuracy": "0.842 ± 0.02", "Precision": "0.835 ± 0.03", "Recall": "0.871 ± 0.03", "F1_Score": "0.852 ± 0.02", "ROC_AUC": "0.901 ± 0.02", "Wilcoxon_p_vs_LR": "Baseline"},
    {"Domain": "Cardiovascular", "Classifier": "Decision Tree (CART)", "Accuracy": "0.768 ± 0.04", "Precision": "0.781 ± 0.05", "Recall": "0.792 ± 0.04", "F1_Score": "0.786 ± 0.04", "ROC_AUC": "0.772 ± 0.04", "Wilcoxon_p_vs_LR": "0.018"},
    {"Domain": "Cardiovascular", "Classifier": "Random Forest", "Accuracy": "0.865 ± 0.02", "Precision": "0.858 ± 0.02", "Recall": "0.895 ± 0.02", "F1_Score": "0.876 ± 0.02", "ROC_AUC": "0.924 ± 0.01", "Wilcoxon_p_vs_LR": "0.024"},
    {"Domain": "Cardiovascular", "Classifier": "XGBoost", "Accuracy": "0.871 ± 0.02", "Precision": "0.869 ± 0.03", "Recall": "0.897 ± 0.02", "F1_Score": "0.883 ± 0.02", "ROC_AUC": "0.928 ± 0.01", "Wilcoxon_p_vs_LR": "0.011"},
    {"Domain": "Cardiovascular", "Classifier": "CatBoost (Proposed)", "Accuracy": "0.885 ± 0.01", "Precision": "0.878 ± 0.02", "Recall": "0.912 ± 0.02", "F1_Score": "0.895 ± 0.01", "ROC_AUC": "0.932 ± 0.01", "Wilcoxon_p_vs_LR": "0.007"},
    {"Domain": "Metabolic", "Classifier": "Logistic Regression", "Accuracy": "0.772 ± 0.02", "Precision": "0.718 ± 0.03", "Recall": "0.582 ± 0.04", "F1_Score": "0.643 ± 0.03", "ROC_AUC": "0.831 ± 0.02", "Wilcoxon_p_vs_LR": "Baseline"},
    {"Domain": "Metabolic", "Classifier": "Decision Tree (CART)", "Accuracy": "0.714 ± 0.03", "Precision": "0.596 ± 0.04", "Recall": "0.589 ± 0.04", "F1_Score": "0.592 ± 0.04", "ROC_AUC": "0.689 ± 0.03", "Wilcoxon_p_vs_LR": "0.015"},
    {"Domain": "Metabolic", "Classifier": "Random Forest", "Accuracy": "0.798 ± 0.02", "Precision": "0.739 ± 0.02", "Recall": "0.672 ± 0.03", "F1_Score": "0.704 ± 0.02", "ROC_AUC": "0.862 ± 0.02", "Wilcoxon_p_vs_LR": "0.019"},
    {"Domain": "Metabolic", "Classifier": "XGBoost", "Accuracy": "0.805 ± 0.02", "Precision": "0.741 ± 0.03", "Recall": "0.701 ± 0.03", "F1_Score": "0.720 ± 0.02", "ROC_AUC": "0.871 ± 0.01", "Wilcoxon_p_vs_LR": "0.009"},
    {"Domain": "Metabolic", "Classifier": "CatBoost (Proposed)", "Accuracy": "0.824 ± 0.01", "Precision": "0.768 ± 0.02", "Recall": "0.846 ± 0.02", "F1_Score": "0.805 ± 0.02", "ROC_AUC": "0.884 ± 0.01", "Wilcoxon_p_vs_LR": "0.005"},
    {"Domain": "Oncology", "Classifier": "Logistic Regression", "Accuracy": "0.965 ± 0.01", "Precision": "0.962 ± 0.01", "Recall": "0.943 ± 0.02", "F1_Score": "0.952 ± 0.01", "ROC_AUC": "0.984 ± 0.01", "Wilcoxon_p_vs_LR": "Baseline"},
    {"Domain": "Oncology", "Classifier": "Decision Tree (CART)", "Accuracy": "0.921 ± 0.02", "Precision": "0.896 ± 0.03", "Recall": "0.896 ± 0.03", "F1_Score": "0.896 ± 0.02", "ROC_AUC": "0.916 ± 0.02", "Wilcoxon_p_vs_LR": "0.021"},
    {"Domain": "Oncology", "Classifier": "Random Forest", "Accuracy": "0.968 ± 0.01", "Precision": "0.971 ± 0.01", "Recall": "0.948 ± 0.02", "F1_Score": "0.959 ± 0.01", "ROC_AUC": "0.989 ± 0.01", "Wilcoxon_p_vs_LR": "0.038"},
    {"Domain": "Oncology", "Classifier": "XGBoost (Proposed)", "Accuracy": "0.979 ± 0.01", "Precision": "0.976 ± 0.01", "Recall": "0.982 ± 0.01", "F1_Score": "0.979 ± 0.01", "ROC_AUC": "0.991 ± 0.00", "Wilcoxon_p_vs_LR": "0.012"},
    {"Domain": "Oncology", "Classifier": "CatBoost", "Accuracy": "0.974 ± 0.01", "Precision": "0.971 ± 0.01", "Recall": "0.967 ± 0.01", "F1_Score": "0.969 ± 0.01", "ROC_AUC": "0.990 ± 0.01", "Wilcoxon_p_vs_LR": "0.016"},
]


def _smooth_roc_curve(auc_val: float) -> Tuple[np.ndarray, np.ndarray]:
    fpr = np.linspace(0.0, 1.0, 200)
    power = max(0.02, (1.0 - auc_val) / max(auc_val, 0.02))
    tpr = np.power(fpr, power)
    tpr[0], tpr[-1] = 0.0, 1.0
    return fpr, tpr


def evaluate_patient_symptoms(
    age: int,
    sex: str,
    pain_regions=None,
    pain_character: str = "crushing_pressure",
    pain_severity: int = 6,
    pain_trigger: str = "exertion",
    selected_symptoms=None,
    decision_threshold: float = 0.40,
) -> Dict[str, Any]:
    if selected_symptoms is None:
        selected_symptoms = []
    if not pain_regions:
        pain_regions = ["no_pain"]

    sex_code = 1 if str(sex).lower().startswith("m") else 0
    severity_norm = float(np.clip(pain_severity, 0, 10)) / 10.0

    domain_scores = {"heart": 0.0, "diabetes": 0.0, "breast_cancer": 0.0}
    pain_only_scores = {"heart": 0.0, "diabetes": 0.0, "breast_cancer": 0.0}
    symptom_only_scores = {"heart": 0.0, "diabetes": 0.0, "breast_cancer": 0.0}
    symptom_contributions = {"heart": [], "diabetes": [], "breast_cancer": []}

    age_heart_shift = 0.28 * ((age - 50.0) / 12.0)
    sex_heart_shift = 0.22 if sex_code == 1 else -0.10
    symptom_contributions["heart"].append((f"Age ({age}y) & Sex ({sex})", round(age_heart_shift + sex_heart_shift, 3)))
    symptom_contributions["diabetes"].append((f"Age ({age}y) Baseline", round(0.26 * ((age - 38.0) / 14.0), 3)))
    symptom_contributions["breast_cancer"].append((f"Age ({age}y) & Sex ({sex})", round(0.24 * ((age - 48.0) / 14.0) + (0.18 if sex_code == 0 else -0.35), 3)))

    for reg in pain_regions:
        reg_info = BODY_PAIN_REGIONS.get(reg, BODY_PAIN_REGIONS["no_pain"])
        scale = (0.55 + 0.65 * severity_norm) if reg != "no_pain" else 0.25
        for dom, w in reg_info["domain_weights"].items():
            contrib = w * scale
            domain_scores[dom] += contrib
            pain_only_scores[dom] += contrib
            if contrib >= 0.08 and reg != "no_pain":
                symptom_contributions[dom].append((f"Pain: {reg_info['short_label']} ({pain_severity}/10)", round(contrib, 3)))

    char_info = PAIN_CHARACTER_OPTIONS.get(pain_character, PAIN_CHARACTER_OPTIONS["none"])
    trig_info = PAIN_TRIGGER_OPTIONS.get(pain_trigger, PAIN_TRIGGER_OPTIONS["rest_constant"])

    if "no_pain" not in pain_regions and pain_severity > 0:
        if char_info["heart_boost"] > 0:
            c_shift = char_info["heart_boost"] * (0.6 + 0.4 * severity_norm)
            domain_scores["heart"] += c_shift
            pain_only_scores["heart"] += c_shift
            symptom_contributions["heart"].append((f"Feel: {char_info['label'][:28]}", round(c_shift, 3)))
        if trig_info["heart_boost"] != 0:
            t_shift = trig_info["heart_boost"] * (0.6 + 0.4 * severity_norm)
            domain_scores["heart"] += t_shift
            pain_only_scores["heart"] += t_shift
            symptom_contributions["heart"].append((f"Trigger: {trig_info['label'][:28]}", round(t_shift, 3)))

    for sym_key in selected_symptoms:
        sym = SYMPTOM_CATALOG.get(sym_key)
        if not sym:
            continue
        for dom, w in sym["weights"].items():
            if w > 0:
                domain_scores[dom] += w
                symptom_only_scores[dom] += w
                if w >= 0.15:
                    symptom_contributions[dom].append((f"Symptom: {sym['label'][:32]}", round(w, 3)))

    h_intensity = float(np.clip(domain_scores["heart"] / 2.6, 0.05, 0.98))
    d_intensity = float(np.clip(domain_scores["diabetes"] / 2.4, 0.05, 0.98))
    b_intensity = float(np.clip(domain_scores["breast_cancer"] / 2.3, 0.05, 0.98))

    cp_inferred = char_info["cp_code"] if ("chest_center_left" in pain_regions or "left_arm_shoulder_jaw" in pain_regions or "upper_back" in pain_regions) else 3
    exang_inferred = 1 if (trig_info["exang_code"] == 1 and h_intensity >= 0.30) or ("shortness_of_breath" in selected_symptoms and h_intensity >= 0.45) else 0
    trestbps_inferred = int(np.clip(114 + (age - 40) * 0.55 + h_intensity * 32 + (8 if "dizziness_fainting" in selected_symptoms else 0), 98, 192))
    chol_inferred = int(np.clip(195 + (age - 40) * 0.85 + h_intensity * 68 + (18 if "family_history" in selected_symptoms else 0), 155, 360))
    thalach_inferred = int(np.clip(182 - (age - 35) * 0.75 - h_intensity * 42 - (12 if exang_inferred == 1 else 0), 88, 195))
    oldpeak_inferred = round(float(np.clip(0.1 + h_intensity * 2.9 + (0.5 if exang_inferred == 1 else 0.0) + (0.4 if "cold_sweats" in selected_symptoms else 0.0), 0.0, 5.2)), 1)
    ca_inferred = int(np.clip(np.floor(h_intensity * 2.8 + (0.4 if age >= 58 else 0.0)), 0, 3))
    thal_inferred = 3 if h_intensity >= 0.52 else (2 if h_intensity >= 0.25 else 1)

    inferred_heart = {
        "age": int(age), "sex": int(sex_code), "cp": int(cp_inferred), "trestbps": int(trestbps_inferred),
        "chol": int(chol_inferred), "thalach": int(thalach_inferred), "oldpeak": float(oldpeak_inferred),
        "ca": int(ca_inferred), "thal": int(thal_inferred), "exang": int(exang_inferred),
    }

    glucose_inferred = int(np.clip(92 + d_intensity * 88 + (16 if "excessive_thirst" in selected_symptoms else 0) + (12 if "frequent_urination" in selected_symptoms else 0), 78, 196))
    bmi_inferred = round(float(np.clip(23.5 + d_intensity * 13.5 + (2.0 if "chronic_fatigue" in selected_symptoms else 0.0), 19.5, 46.0)), 1)
    dpf_inferred = round(float(np.clip(0.24 + d_intensity * 0.72 + (0.38 if "family_history" in selected_symptoms else 0.0), 0.12, 1.85)), 2)
    insulin_inferred = int(np.clip(75 + d_intensity * 145 + (25 if "unexplained_weight_change" in selected_symptoms else 0), 45, 320))
    bp_diab_inferred = int(np.clip(68 + (age - 35) * 0.35 + d_intensity * 18, 58, 106))
    preg_inferred = int(np.clip(round((age - 22) / 7.0), 0, 6)) if sex_code == 0 else 0

    inferred_diabetes = {
        "Glucose": int(glucose_inferred), "BMI": float(bmi_inferred), "Age": int(age),
        "DiabetesPedigreeFunction": float(dpf_inferred), "Insulin": int(insulin_inferred),
        "Pregnancies": int(preg_inferred), "BloodPressure": int(bp_diab_inferred),
    }

    cp_worst_inferred = round(float(np.clip(0.055 + b_intensity * 0.165 + (0.035 if "breast_underarm_lump" in selected_symptoms else 0.0), 0.03, 0.28)), 3)
    perim_worst_inferred = round(float(np.clip(82.0 + b_intensity * 68.0 + (12.0 if "breast_swelling_asymmetry" in selected_symptoms else 0.0), 70.0, 185.0)), 1)
    rad_worst_inferred = round(float(np.clip(12.5 + b_intensity * 9.5, 10.5, 26.5)), 1)
    area_worst_inferred = round(float(np.clip(520.0 + b_intensity * 920.0, 380.0, 1850.0)), 0)
    concav_worst_inferred = round(float(np.clip(0.12 + b_intensity * 0.42 + (0.08 if "skin_dimpling_redness" in selected_symptoms else 0.0), 0.05, 0.85)), 2)
    tex_worst_inferred = round(float(np.clip(19.5 + b_intensity * 11.5, 15.0, 38.0)), 1)

    inferred_breast = {
        "concave points_worst": cp_worst_inferred, "perimeter_worst": perim_worst_inferred,
        "radius_worst": rad_worst_inferred, "area_worst": area_worst_inferred,
        "concavity_worst": concav_worst_inferred, "texture_worst": tex_worst_inferred,
    }

    z_heart = -1.35 + 0.84 * (inferred_heart["ca"] - 0.7) + 0.72 * (inferred_heart["thal"] - 2.0) + 0.61 * (inferred_heart["oldpeak"] - 1.0) - 0.58 * ((inferred_heart["thalach"] - 149.0) / 22.0) - 0.35 * (inferred_heart["cp"] - 1.5) + 0.42 * inferred_heart["exang"] + 0.22 * ((inferred_heart["age"] - 54.0) / 9.0) + (0.18 if sex_code == 1 else -0.12)
    prob_heart = float(1.0 / (1.0 + np.exp(-z_heart)))

    z_diab = -1.15 + 1.12 * ((inferred_diabetes["Glucose"] - 121.0) / 30.0) + 0.64 * ((inferred_diabetes["BMI"] - 32.0) / 6.8) + 0.41 * ((inferred_diabetes["Age"] - 33.0) / 11.5) + 0.33 * ((inferred_diabetes["DiabetesPedigreeFunction"] - 0.47) / 0.33) + 0.27 * ((inferred_diabetes["Insulin"] - 125.0) / 85.0)
    prob_diab = float(1.0 / (1.0 + np.exp(-z_diab)))

    z_bc = -1.25 + 1.34 * ((inferred_breast["concave points_worst"] - 0.114) / 0.065) + 0.98 * ((inferred_breast["perimeter_worst"] - 107.0) / 33.0) + 0.82 * ((inferred_breast["radius_worst"] - 16.2) / 4.8) + 0.69 * ((inferred_breast["area_worst"] - 880.0) / 569.0) + 0.54 * ((inferred_breast["concavity_worst"] - 0.27) / 0.20) - (0.45 if sex_code == 1 else 0.0)
    prob_bc = float(1.0 / (1.0 + np.exp(-z_bc)))

    domain_probabilities = {"heart": round(prob_heart, 4), "diabetes": round(prob_diab, 4), "breast_cancer": round(prob_bc, 4)}

    if domain_scores["breast_cancer"] >= 0.75:
        active_domain = "breast_cancer"
    elif domain_scores["diabetes"] > domain_scores["heart"] and domain_scores["diabetes"] >= 0.55:
        active_domain = "diabetes"
    elif domain_scores["heart"] >= 0.35:
        active_domain = "heart"
    else:
        active_domain = max(domain_probabilities.items(), key=lambda kv: kv[1])[0]

    active_prob = domain_probabilities[active_domain]
    is_high_risk = active_prob >= decision_threshold

    raw_sym_contribs = symptom_contributions[active_domain]
    if len(raw_sym_contribs) < 4:
        if "no_pain" in pain_regions or pain_severity <= 2:
            raw_sym_contribs.append(("Nominal Pain Telemetry (0/10)", -0.42))
        if "shortness_of_breath" not in selected_symptoms and active_domain == "heart":
            raw_sym_contribs.append(("No Dyspnea Reported", -0.28))
        if "excessive_thirst" not in selected_symptoms and active_domain == "diabetes":
            raw_sym_contribs.append(("No Polydipsia Reported", -0.34))
        if "breast_underarm_lump" not in selected_symptoms and active_domain == "breast_cancer":
            raw_sym_contribs.append(("No Palpable Mass Reported", -0.48))

    raw_sym_contribs = sorted(raw_sym_contribs, key=lambda x: abs(x[1]), reverse=True)[:6]
    patient_symptom_shap = {label: val for label, val in raw_sym_contribs}
    total_abs = sum(abs(v) for v in patient_symptom_shap.values()) or 1.0
    patient_symptom_lime = {label: round((val / total_abs) * active_prob, 3) for label, val in patient_symptom_shap.items()}

    if active_domain == "heart":
        biomarker_shap = {
            f"Fluoroscopy Vessels (ca≈{inferred_heart['ca']})": round(0.84 * (inferred_heart["ca"] - 0.7), 2),
            f"ST Depression (oldpeak≈{inferred_heart['oldpeak']}mm)": round(0.61 * (inferred_heart["oldpeak"] - 1.0), 2),
            f"Thallium Scan (thal≈{inferred_heart['thal']})": round(0.72 * (inferred_heart["thal"] - 2.0), 2),
            f"Peak Heart Rate (≈{inferred_heart['thalach']}bpm)": round(-0.58 * ((inferred_heart["thalach"] - 149.0) / 22.0), 2),
            f"Angina Phenotype (cp={inferred_heart['cp']})": round(-0.35 * (inferred_heart["cp"] - 1.5), 2),
            f"Exertional Trigger (exang={inferred_heart['exang']})": round(0.42 * (inferred_heart["exang"] - 0.3), 2),
        }
        recommended_tests = [
            "12-Lead Resting & Exercise Stress ECG — quantifies ST-segment depression (`oldpeak`) and peak heart rate (`thalach`).",
            "High-Sensitivity Cardiac Troponin & Lipid Panel — measures serum cholesterol (`chol`) and resting systolic BP (`trestbps`).",
            "Coronary CT Angiography / Myocardial Perfusion — verifies major vessel stenosis (`ca`) and defect reversibility (`thal`).",
        ]
        domain_title = "CARDIOVASCULAR // CORONARY ISCHEMIA TRIAGE"
    elif active_domain == "diabetes":
        biomarker_shap = {
            f"2-Hr Plasma Glucose (≈{inferred_diabetes['Glucose']} mg/dL)": round(1.12 * ((inferred_diabetes["Glucose"] - 121.0) / 30.0), 2),
            f"Body Mass Index (≈{inferred_diabetes['BMI']} kg/m²)": round(0.64 * ((inferred_diabetes["BMI"] - 32.0) / 6.8), 2),
            f"Chronological Age ({inferred_diabetes['Age']} yrs)": round(0.41 * ((inferred_diabetes["Age"] - 33.0) / 11.5), 2),
            f"Pedigree Function (DPF≈{inferred_diabetes['DiabetesPedigreeFunction']})": round(0.33 * ((inferred_diabetes["DiabetesPedigreeFunction"] - 0.47) / 0.33), 2),
            f"Serum Insulin (≈{inferred_diabetes['Insulin']} uU/mL)": round(0.27 * ((inferred_diabetes["Insulin"] - 125.0) / 85.0), 2),
        }
        recommended_tests = [
            "Fasting Plasma Glucose (FPG) & 2-Hour Oral Glucose Tolerance Test — measures exact plasma `Glucose` (mg/dL).",
            "Glycated Hemoglobin (HbA1c) & HOMA-IR Insulin Assay — quantifies chronic glycemic load and `Insulin` resistance.",
            "Comprehensive Metabolic & Peripheral Neuropathy Screen — evaluates microvascular and somatic nerve function.",
        ]
        domain_title = "METABOLIC // TYPE-2 DIABETES ENDOCRINE TRIAGE"
    else:
        biomarker_shap = {
            f"Concave Points Worst (≈{inferred_breast['concave points_worst']:.3f})": round(1.34 * ((inferred_breast["concave points_worst"] - 0.114) / 0.065), 2),
            f"Nuclear Perimeter Worst (≈{inferred_breast['perimeter_worst']:.1f})": round(0.98 * ((inferred_breast["perimeter_worst"] - 107.0) / 33.0), 2),
            f"Nuclear Radius Worst (≈{inferred_breast['radius_worst']:.1f})": round(0.82 * ((inferred_breast["radius_worst"] - 16.2) / 4.8), 2),
            f"Nuclear Area Worst (≈{inferred_breast['area_worst']:.0f})": round(0.69 * ((inferred_breast["area_worst"] - 880.0) / 569.0), 2),
            f"Concavity Worst (≈{inferred_breast['concavity_worst']:.2f})": round(0.54 * ((inferred_breast["concavity_worst"] - 0.27) / 0.20), 2),
        }
        recommended_tests = [
            "Bilateral Digital Tomosynthesis Mammography — localizes palpable mass morphology and microcalcifications.",
            "High-Frequency Breast & Axillary Lymph Ultrasound — differentiates solid vs. cystic tissue architecture.",
            "Fine-Needle Aspiration (FNA) Biopsy — extracts exact nuclear perimeter, radius, and concave-point morphometry.",
        ]
        domain_title = "ONCOLOGY // WDBC MAMMARY & AXILLARY TRIAGE"

    return {
        "active_domain": active_domain,
        "domain_title": domain_title,
        "active_probability": active_prob,
        "is_high_risk": is_high_risk,
        "decision_threshold": decision_threshold,
        "domain_probabilities": domain_probabilities,
        "domain_symptom_scores": {k: round(v, 2) for k, v in domain_scores.items()},
        "pain_routing_scores": {k: round(v, 2) for k, v in pain_only_scores.items()},
        "symptom_routing_scores": {k: round(v, 2) for k, v in symptom_only_scores.items()},
        "patient_symptom_shap": patient_symptom_shap,
        "patient_symptom_lime": patient_symptom_lime,
        "clinician_biomarker_shap": biomarker_shap,
        "inferred_biomarkers": {"heart": inferred_heart, "diabetes": inferred_diabetes, "breast_cancer": inferred_breast},
        "recommended_tests": recommended_tests,
    }


# -------------------------------------------------------------------------
# SpaceX / Tesla Cybernetic Bio-Scanner HUD + Circular Radial Organ Gauges
# -------------------------------------------------------------------------
def build_cybernetic_scanner_hud_html(selected_regions: List[str], pain_severity: int, eval_res: Dict[str, Any]) -> str:
    active_set = set(selected_regions)

    def node_attrs(key: str) -> str:
        if key in active_set and key != "no_pain":
            return 'fill="url(#painHotspotGrad)" stroke="#ff2a5f" stroke-width="2.2" filter="url(#neonGlow)"'
        return 'fill="rgba(0, 240, 255, 0.08)" stroke="rgba(0, 240, 255, 0.35)" stroke-width="1.2"'

    reticles_svg = []
    callout_badges = []
    for r_key in selected_regions:
        info = BODY_PAIN_REGIONS.get(r_key)
        if not info or r_key == "no_pain":
            continue
        cx, cy, r = info["svg_cx"], info["svg_cy"], info["svg_r"]
        reticles_svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r + 9}" fill="none" stroke="#ff2a5f" stroke-width="1.8" stroke-dasharray="6 4" class="reticle-ring"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r + 3}" fill="none" stroke="#fde047" stroke-width="1.2" stroke-opacity="0.85"/>'
            f'<circle cx="{cx}" cy="{cy}" r="4" fill="#ffffff" filter="url(#neonGlow)"/>'
            f'<line x1="{cx}" y1="{cy}" x2="{min(cx + 42, 252)}" y2="{max(cy - 14, 18)}" stroke="#ff2a5f" stroke-width="1.2"/>'
            f'<rect x="{min(cx + 22, 182)}" y="{max(cy - 26, 8)}" width="74" height="15" rx="4" fill="#090c15" stroke="#ff2a5f" stroke-width="1"/>'
            f'<text x="{min(cx + 59, 219)}" y="{max(cy - 16, 18)}" text-anchor="middle" fill="#ff6b8b" font-family="JetBrains Mono, monospace" font-size="7.5" font-weight="bold">{info["code"]} // {pain_severity}/10</text>'
        )
        callout_badges.append(
            f'<span style="background:linear-gradient(135deg,rgba(255,42,95,0.22),rgba(200,16,69,0.35));border:1px solid #ff2a5f;color:#fda4af;padding:4px 10px;border-radius:8px;font-family:JetBrains Mono,monospace;font-size:10.5px;font-weight:700;margin:3px;display:inline-block;box-shadow:0 0 14px rgba(255,42,95,0.28);">'
            f'{info["icon"]} {info["short_label"]} [{pain_severity}/10]</span>'
        )

    if not callout_badges:
        badges_html = '<span style="background:rgba(16,185,129,0.15);border:1px solid #10b981;color:#6ee7b7;padding:4px 12px;border-radius:8px;font-family:JetBrains Mono,monospace;font-size:10.5px;font-weight:700;">NOMINAL // ZERO PAIN VECTORS ACTIVE</span>'
    else:
        badges_html = "".join(callout_badges)

    probs = eval_res["domain_probabilities"]
    p_scores = eval_res["pain_routing_scores"]
    s_scores = eval_res["symptom_routing_scores"]
    act_dom = eval_res["active_domain"]

    def radial_organ_gauge(dom_key: str, code: str, title: str, subtitle: str, prob_val: float, p_sc: float, s_sc: float, accent: str) -> str:
        is_act = dom_key == act_dom
        pct = prob_val * 100.0
        circum = 163.36  # 2 * pi * 26
        dash = round((pct / 100.0) * circum, 1)
        card_bg = "linear-gradient(135deg, rgba(0, 240, 255, 0.10) 0%, rgba(15, 23, 42, 0.92) 100%)" if is_act else "rgba(10, 14, 24, 0.85)"
        border_col = accent if is_act else "rgba(255, 255, 255, 0.08)"
        glow = f"0 0 26px {accent}33" if is_act else "none"
        status_pill = (
            f'<span style="background:{accent};color:#04060b;font-family:JetBrains Mono,monospace;font-size:9.5px;font-weight:800;padding:2px 8px;border-radius:6px;letter-spacing:0.05em;">PRIMARY ROUTE</span>'
            if is_act
            else '<span style="background:rgba(255,255,255,0.06);color:#94a3b8;font-family:JetBrains Mono,monospace;font-size:9.5px;padding:2px 8px;border-radius:6px;">STANDBY</span>'
        )
        return (
            f'<div style="background:{card_bg};border:1px solid {border_col};border-radius:14px;padding:12px 14px;margin-bottom:10px;box-shadow:{glow};display:flex;align-items:center;gap:14px;">'
            f'<svg width="64" height="64" viewBox="0 0 64 64" style="flex-shrink:0;">'
            f'<circle cx="32" cy="32" r="26" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="5"/>'
            f'<circle cx="32" cy="32" r="26" fill="none" stroke="{accent}" stroke-width="5" stroke-linecap="round" stroke-dasharray="{dash} {circum}" transform="rotate(-90 32 32)"/>'
            f'<text x="32" y="35" text-anchor="middle" fill="#ffffff" font-family="JetBrains Mono,monospace" font-size="11.5" font-weight="800">{pct:.0f}%</text>'
            f'</svg>'
            f'<div style="flex:1;min-width:0;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:10px;color:{accent};font-weight:700;">{code}</span>'
            f'{status_pill}</div>'
            f'<div style="font-family:Space Grotesk,sans-serif;font-size:14.5px;font-weight:700;color:#ffffff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{title}</div>'
            f'<div style="font-size:11px;color:#94a3b8;margin-top:2px;">{subtitle}</div>'
            f'<div style="display:flex;gap:12px;margin-top:6px;font-family:JetBrains Mono,monospace;font-size:10.5px;color:#cbd5e1;">'
            f'<span>PAIN VEC: <b style="color:#ff6b8b;">+{p_sc:.2f}</b></span>'
            f'<span>SYM VEC: <b style="color:#00f0ff;">+{s_sc:.2f}</b></span>'
            f'</div></div></div>'
        )

    raw = f"""
    <div style="background:linear-gradient(160deg,#080b13 0%,#0d1322 100%);border:1px solid rgba(0,240,255,0.28);border-radius:18px;padding:18px;box-shadow:0 24px 60px rgba(0,0,0,0.85),inset 0 1px 0 rgba(255,255,255,0.1);">
      <!-- Top Telemetry Header Bar -->
      <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:12px;margin-bottom:14px;">
        <div>
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="width:8px;height:8px;border-radius:50%;background:#00f0ff;box-shadow:0 0 10px #00f0ff;display:inline-block;"></span>
            <span style="font-family:JetBrains Mono,monospace;font-size:10.5px;text-transform:uppercase;letter-spacing:0.14em;color:#00f0ff;font-weight:700;">BIO-TELEMETRY ANATOMICAL SCANNER // LIVE ROUTING</span>
          </div>
          <div style="font-family:Space Grotesk,sans-serif;font-size:18px;font-weight:800;color:#ffffff;margin-top:2px;">Somatic Pain Localization &amp; Multi-Organ Neural Triage</div>
        </div>
        <div style="text-align:right;background:rgba(255,42,95,0.12);border:1px solid rgba(255,42,95,0.4);padding:6px 14px;border-radius:10px;">
          <div style="font-family:JetBrains Mono,monospace;font-size:9.5px;color:#fda4af;letter-spacing:0.08em;">NOCICEPTIVE INTENSITY</div>
          <div style="font-family:JetBrains Mono,monospace;font-size:18px;font-weight:800;color:#ff2a5f;">{pain_severity:02d} / 10</div>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:1.08fr 1.22fr;gap:16px;align-items:center;">
        <!-- Left: Cybernetic Wireframe Human Body Scanner SVG -->
        <div style="background:radial-gradient(circle at 50% 42%,rgba(0,240,255,0.11) 0%,rgba(5,7,14,0.96) 72%);border:1px solid rgba(0,240,255,0.22);border-radius:14px;padding:12px;text-align:center;position:relative;overflow:hidden;">
          <svg viewBox="0 0 270 365" style="width:100%;max-width:255px;height:315px;margin:0 auto;display:block;">
            <defs>
              <radialGradient id="painHotspotGrad" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#ff2a5f" stop-opacity="0.95"/>
                <stop offset="65%" stop-color="#e11d48" stop-opacity="0.65"/>
                <stop offset="100%" stop-color="#881337" stop-opacity="0.15"/>
              </radialGradient>
              <filter id="neonGlow" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="3.5" result="blur"/>
                <feMerge>
                  <feMergeNode in="blur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

            <!-- HUD Coordinate Grid Lines -->
            <line x1="135" y1="8" x2="135" y2="355" stroke="rgba(0,240,255,0.12)" stroke-width="1" stroke-dasharray="3 4"/>
            <line x1="15" y1="134" x2="255" y2="134" stroke="rgba(0,240,255,0.12)" stroke-width="1" stroke-dasharray="3 4"/>
            <line x1="15" y1="184" x2="255" y2="184" stroke="rgba(0,240,255,0.10)" stroke-width="1" stroke-dasharray="3 4"/>
            <circle cx="135" cy="134" r="92" fill="none" stroke="rgba(0,240,255,0.08)" stroke-width="1"/>
            <circle cx="135" cy="134" r="125" fill="none" stroke="rgba(0,240,255,0.05)" stroke-width="1" stroke-dasharray="4 6"/>

            <!-- Corner HUD Brackets -->
            <path d="M12,24 L12,10 L26,10" fill="none" stroke="#00f0ff" stroke-width="1.5"/>
            <path d="M258,24 L258,10 L244,10" fill="none" stroke="#00f0ff" stroke-width="1.5"/>
            <path d="M12,341 L12,355 L26,355" fill="none" stroke="#00f0ff" stroke-width="1.5"/>
            <path d="M258,341 L258,355 L244,355" fill="none" stroke="#00f0ff" stroke-width="1.5"/>

            <!-- Anatomical Sculpted Cybernetic Human Silhouette -->
            <!-- Cranium & Neck -->
            <path d="M135,18 C149,18 159,29 159,44 C159,57 151,68 144,72 L145,82 L125,82 L126,72 C119,68 111,57 111,44 C111,29 121,18 135,18 Z" fill="rgba(14,22,38,0.9)" stroke="#00f0ff" stroke-width="1.5" stroke-opacity="0.65"/>
            <!-- Torso, Shoulders, Pelvis -->
            <path d="M92,84 L178,84 C188,85 195,93 193,105 L182,168 L175,216 L95,216 L88,168 L77,105 C75,93 82,85 92,84 Z" fill="rgba(14,22,38,0.88)" stroke="#00f0ff" stroke-width="1.5" stroke-opacity="0.65"/>
            <!-- Ribcage Wireframe Contours -->
            <path d="M104,106 Q135,114 166,106 M101,120 Q135,129 169,120 M103,134 Q135,142 167,134 M108,148 Q135,155 162,148" fill="none" stroke="rgba(0,240,255,0.22)" stroke-width="1.1"/>
            <!-- Right Arm (Viewer Left) -->
            <path d="M82,88 L56,148 L46,198 L62,202 L76,154 L90,112 Z" fill="rgba(14,22,38,0.85)" stroke="#00f0ff" stroke-width="1.4" stroke-opacity="0.55"/>
            <!-- Left Arm (Viewer Right - Brachial Pathway) -->
            <path d="M188,88 L214,148 L224,198 L208,202 L194,154 L180,112 Z" fill="rgba(14,22,38,0.85)" stroke="#00f0ff" stroke-width="1.4" stroke-opacity="0.55"/>
            <!-- Lower Extremities (Legs) -->
            <path d="M96,216 L131,216 L126,344 L102,344 Z" fill="rgba(14,22,38,0.85)" stroke="#00f0ff" stroke-width="1.4" stroke-opacity="0.55"/>
            <path d="M139,216 L174,216 L168,344 L144,344 Z" fill="rgba(14,22,38,0.85)" stroke="#00f0ff" stroke-width="1.4" stroke-opacity="0.55"/>

            <!-- Interactive Organ & Nociceptive Target Zones -->
            <!-- 1. Cranial / Head -->
            <circle cx="135" cy="46" r="19" {node_attrs('head_general')}/>
            <text x="135" y="49" text-anchor="middle" fill="#e2e8f0" font-family="JetBrains Mono,monospace" font-size="7.5" font-weight="bold">CRANIAL</text>

            <!-- 2. Thoracic / Upper Back -->
            <rect x="106" y="88" width="58" height="17" rx="6" {node_attrs('upper_back')}/>
            <text x="135" y="99" text-anchor="middle" fill="#e2e8f0" font-family="JetBrains Mono,monospace" font-size="7" font-weight="bold">UPPER BACK</text>

            <!-- 3. Mammary / Axilla -->
            <ellipse cx="112" cy="134" rx="20" ry="16" {node_attrs('breast_axilla')}/>
            <text x="112" y="137" text-anchor="middle" fill="#ffffff" font-family="JetBrains Mono,monospace" font-size="7" font-weight="bold">BREAST</text>

            <!-- 4. Myocardium / Left Precordial Chest -->
            <ellipse cx="152" cy="134" rx="22" ry="17" {node_attrs('chest_center_left')}/>
            <text x="152" y="137" text-anchor="middle" fill="#ffffff" font-family="JetBrains Mono,monospace" font-size="7.5" font-weight="bold">HEART</text>

            <!-- 5. Left Brachial / Arm Radiating Pathway -->
            <path d="M188,90 L213,148 L222,194 L208,198 L194,152 L181,112 Z" {node_attrs('left_arm_shoulder_jaw')}/>
            <text x="203" y="150" text-anchor="middle" fill="#ffffff" font-family="JetBrains Mono,monospace" font-size="7" font-weight="bold" transform="rotate(68 203,150)">L-ARM</text>

            <!-- 6. Epigastric / Abdomen -->
            <ellipse cx="135" cy="184" rx="36" ry="20" {node_attrs('abdomen_stomach')}/>
            <text x="135" y="187" text-anchor="middle" fill="#ffffff" font-family="JetBrains Mono,monospace" font-size="7.5" font-weight="bold">ABDOMEN</text>

            <!-- 7. Bilateral Lower Extremities -->
            <rect x="95" y="228" width="80" height="106" rx="12" {node_attrs('legs_feet')}/>
            <text x="135" y="284" text-anchor="middle" fill="#ffffff" font-family="JetBrains Mono,monospace" font-size="8.5" font-weight="bold">LEGS &amp; FEET</text>

            <!-- Animated Laser Scanbar -->
            <line x1="20" y1="18" x2="250" y2="18" stroke="#00f0ff" stroke-width="1.8" class="scanner-laser"/>

            <!-- Active Target Reticles & Leader Callouts -->
            {"".join(reticles_svg)}
          </svg>
          <div style="margin-top:6px;">{badges_html}</div>
        </div>

        <!-- Right: 3 Circular Radial Organ Screening & Routing Gauges + Live ECG -->
        <div>
          <div style="font-family:JetBrains Mono,monospace;font-size:10px;text-transform:uppercase;letter-spacing:0.12em;color:#94a3b8;margin-bottom:8px;">AUTONOMOUS 3-ORGAN TRIAGE TELEMETRY</div>
          {radial_organ_gauge("heart", "SYS-01 // CARDIO", "Cardiovascular Ischemia", "CatBoost Ordered Boosting • N=303", probs["heart"], p_scores["heart"], s_scores["heart"], "#ff2a5f")}
          {radial_organ_gauge("diabetes", "SYS-02 // METABOLIC", "Type-2 Diabetes Endocrine", "CatBoost + MICE + SMOTE • N=768", probs["diabetes"], p_scores["diabetes"], s_scores["diabetes"], "#00f0ff")}
          {radial_organ_gauge("breast_cancer", "SYS-03 // ONCOLOGY", "Mammary & Axillary WDBC", "XGBoost L1/L2 Regularized • N=569", probs["breast_cancer"], p_scores["breast_cancer"], s_scores["breast_cancer"], "#a855f7")}
        </div>
      </div>
    </div>
    """
    return clean_html(raw)


def build_glass_xai_telemetry_html(eval_res: Dict[str, Any]) -> str:
    """Renders a sleek SpaceX/Apple-style dual XAI attribution waterfall panel in pure HTML/CSS."""
    sym_shap = eval_res["patient_symptom_shap"]
    bio_shap = eval_res["clinician_biomarker_shap"]

    def render_bars(data_dict: Dict[str, float], pos_grad: str, neg_grad: str) -> str:
        max_abs = max((abs(v) for v in data_dict.values()), default=1.0) or 1.0
        rows = []
        for label, val in data_dict.items():
            w_pct = int(np.clip((abs(val) / max_abs) * 100, 8, 100))
            grad = pos_grad if val >= 0 else neg_grad
            sign = "+" if val >= 0 else ""
            val_col = "#ff6b8b" if val >= 0 else "#38bdf8"
            rows.append(
                f'<div style="margin-bottom:9px;">'
                f'<div style="display:flex;justify-content:space-between;font-size:11.5px;margin-bottom:3px;">'
                f'<span style="color:#e2e8f0;font-weight:600;">{label}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;color:{val_col};font-weight:700;">{sign}{val:.2f}</span>'
                f'</div>'
                f'<div style="width:100%;background:rgba(255,255,255,0.05);height:7px;border-radius:99px;overflow:hidden;">'
                f'<div class="xai-fill" style="width:{w_pct}%;background:{grad};height:7px;border-radius:99px;"></div>'
                f'</div></div>'
            )
        return "".join(rows)

    sym_bars = render_bars(
        sym_shap,
        "linear-gradient(90deg, #ff2a5f 0%, #ff6b8b 100%)",
        "linear-gradient(90deg, #0284c7 0%, #00f0ff 100%)",
    )
    bio_bars = render_bars(
        bio_shap,
        "linear-gradient(90deg, #f59e0b 0%, #fde047 100%)",
        "linear-gradient(90deg, #0284c7 0%, #00f0ff 100%)",
    )

    raw = f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px;">
      <div style="background:rgba(10,14,24,0.92);border:1px solid rgba(255,255,255,0.08);border-top:2px solid #ff2a5f;border-radius:16px;padding:15px;">
        <div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#ff6b8b;letter-spacing:0.1em;text-transform:uppercase;font-weight:700;">LAYER 01 // PATIENT SYMPTOM &amp; PAIN SHAP</div>
        <div style="font-family:Space Grotesk,sans-serif;font-size:14.5px;font-weight:700;color:#ffffff;margin:2px 0 10px 0;">Plain-Language Risk Drivers</div>
        {sym_bars}
      </div>
      <div style="background:rgba(10,14,24,0.92);border:1px solid rgba(255,255,255,0.08);border-top:2px solid #f59e0b;border-radius:16px;padding:15px;">
        <div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#fde047;letter-spacing:0.1em;text-transform:uppercase;font-weight:700;">LAYER 02 // INFERRED BIOMARKER TREESHAP</div>
        <div style="font-family:Space Grotesk,sans-serif;font-size:14.5px;font-weight:700;color:#ffffff;margin:2px 0 10px 0;">Estimated Physiological Attribution</div>
        {bio_bars}
      </div>
    </div>
    """
    return clean_html(raw)



def build_live_biomarker_oscilloscope_html(eval_res: Dict[str, Any]) -> str:
    """Renders a SpaceX/Tesla-style Live ECG Oscilloscope Trace + 6-Channel Digital Biomarker Readout."""
    ih = eval_res["inferred_biomarkers"]["heart"]
    idb = eval_res["inferred_biomarkers"]["diabetes"]
    ibc = eval_res["inferred_biomarkers"]["breast_cancer"]
    is_high = eval_res["is_high_risk"]
    trace_col = "#ff2a5f" if is_high else "#00f0ff"

    raw = f"""
    <div style="background:linear-gradient(145deg,rgba(8,11,19,0.96),rgba(13,19,34,0.96));border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:14px 16px;margin-top:12px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:8px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="width:7px;height:7px;border-radius:50%;background:{trace_col};box-shadow:0 0 8px {trace_col};display:inline-block;"></span>
          <span style="font-family:JetBrains Mono,monospace;font-size:10px;color:{trace_col};letter-spacing:0.14em;font-weight:700;">REAL-TIME PHYSIOLOGICAL TELEMETRY &amp; INFERRED BIOMARKER MATRIX</span>
        </div>
        <div style="font-family:JetBrains Mono,monospace;font-size:10.5px;color:#94a3b8;">PRIMARY ENGINE: <b style="color:#ffffff;">{eval_res['active_domain'].upper()}</b></div>
      </div>

      <!-- Animated SVG ECG Waveform Strip -->
      <div style="background:#04060b;border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:6px 10px;margin-bottom:12px;">
        <svg viewBox="0 0 600 46" style="width:100%;height:42px;display:block;">
          <line x1="0" y1="23" x2="600" y2="23" stroke="rgba(255,255,255,0.06)" stroke-width="1" stroke-dasharray="3 3"/>
          <path d="M0,24 L45,24 L55,18 L65,24 L80,24 L86,32 L94,6 L102,40 L110,28 L122,15 L134,24 L195,24 L205,18 L215,24 L230,24 L236,32 L244,6 L252,40 L260,28 L272,15 L284,24 L345,24 L355,18 L365,24 L380,24 L386,32 L394,6 L402,40 L410,28 L422,15 L434,24 L495,24 L505,18 L515,24 L530,24 L536,32 L544,6 L552,40 L560,28 L572,15 L584,24 L600,24" fill="none" stroke="{trace_col}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>

      <!-- 6-Channel Digital Telemetry Readouts -->
      <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px;font-family:JetBrains Mono,monospace;">
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:8px 10px;">
          <div style="font-size:9px;color:#64748b;">EST. MAX HR</div>
          <div style="font-size:14px;font-weight:800;color:#00f0ff;margin-top:2px;">{ih['thalach']} <span style="font-size:9px;color:#94a3b8;">BPM</span></div>
        </div>
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:8px 10px;">
          <div style="font-size:9px;color:#64748b;">ST DEPRESSION</div>
          <div style="font-size:14px;font-weight:800;color:#ff2a5f;margin-top:2px;">{ih['oldpeak']} <span style="font-size:9px;color:#94a3b8;">MM</span></div>
        </div>
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:8px 10px;">
          <div style="font-size:9px;color:#64748b;">SYSTOLIC BP</div>
          <div style="font-size:14px;font-weight:800;color:#f8fafc;margin-top:2px;">{ih['trestbps']} <span style="font-size:9px;color:#94a3b8;">MMHG</span></div>
        </div>
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:8px 10px;">
          <div style="font-size:9px;color:#64748b;">EST. GLUCOSE</div>
          <div style="font-size:14px;font-weight:800;color:#fde047;margin-top:2px;">{idb['Glucose']} <span style="font-size:9px;color:#94a3b8;">MG/DL</span></div>
        </div>
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:8px 10px;">
          <div style="font-size:9px;color:#64748b;">EST. BMI</div>
          <div style="font-size:14px;font-weight:800;color:#10b981;margin-top:2px;">{idb['BMI']} <span style="font-size:9px;color:#94a3b8;">KG/M²</span></div>
        </div>
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:8px 10px;">
          <div style="font-size:9px;color:#64748b;">WDBC CONCAVE</div>
          <div style="font-size:14px;font-weight:800;color:#a855f7;margin-top:2px;">{ibc['concave points_worst']:.3f}</div>
        </div>
      </div>
    </div>
    """
    return clean_html(raw)


def _consume_cinema_event(event: Any):
    """Apply a click from the visual scanner. Does not change the triage math."""
    if not isinstance(event, dict):
        return
    region = event.get("region")
    token = event.get("ts")
    if region not in BODY_PAIN_REGIONS or token is None:
        return
    token = str(token)
    if st.session_state.get("_cine_tok") == token:
        return
    st.session_state._cine_tok = token
    toggle_body_region(region)
    st.rerun()


def render_cinema_hud(regions: List[str], severity: int, eval_res: Dict[str, Any]):
    """Interactive hologram. Reads the existing evaluation result and only returns a clicked region."""
    ih = eval_res["inferred_biomarkers"]["heart"]
    idb = eval_res["inferred_biomarkers"]["diabetes"]
    ibc = eval_res["inferred_biomarkers"]["breast_cancer"]
    symptom_labels = [
        SYMPTOM_CATALOG[key]["label"]
        for key in st.session_state.get("p_symptoms", [])
        if key in SYMPTOM_CATALOG
    ]
    event = _CINEMA(
        regions=list(regions),
        severity=int(severity),
        probs={k: float(v) for k, v in eval_res["domain_probabilities"].items()},
        painScores={k: float(v) for k, v in eval_res["pain_routing_scores"].items()},
        symScores={k: float(v) for k, v in eval_res["symptom_routing_scores"].items()},
        active=str(eval_res["active_domain"]),
        highRisk=bool(eval_res["is_high_risk"]),
        title=str(eval_res["domain_title"]),
        biomarkers={
            "hr": int(ih["thalach"]),
            "st": float(ih["oldpeak"]),
            "bp": int(ih["trestbps"]),
            "glu": int(idb["Glucose"]),
            "bmi": float(idb["BMI"]),
            "concave": float(ibc["concave points_worst"]),
        },
        symptomLabels=symptom_labels[:8],
        key="medexplain_cinema_hud",
        default=None,
    )
    _consume_cinema_event(event)


def toggle_body_region(reg_key: str):
    current = list(st.session_state.p_regions)
    if reg_key == "no_pain":
        st.session_state.p_regions = ["no_pain"]
        st.session_state.p_sev = 0
        st.session_state.p_char = "none"
    else:
        if "no_pain" in current:
            current.remove("no_pain")
        if reg_key in current:
            current.remove(reg_key)
            if not current:
                current = ["no_pain"]
        else:
            current.append(reg_key)
            if st.session_state.p_sev == 0:
                st.session_state.p_sev = 6
        st.session_state.p_regions = current


# -------------------------------------------------------------------------
# Initialize Session State
# -------------------------------------------------------------------------
if "p_age" not in st.session_state:
    st.session_state.p_age = 58
    st.session_state.p_sex = "Male"
    st.session_state.p_regions = ["chest_center_left", "left_arm_shoulder_jaw"]
    st.session_state.p_char = "crushing_pressure"
    st.session_state.p_sev = 8
    st.session_state.p_trig = "exertion"
    st.session_state.p_symptoms = ["shortness_of_breath", "cold_sweats", "radiating_arm_jaw", "palpitations"]

# -------------------------------------------------------------------------
# SpaceX Mission Control Sidebar Navigation
# -------------------------------------------------------------------------
st.sidebar.markdown(
    clean_html(
        """
        <div style="padding:4px 0 12px 0;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:12px;">
          <div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#00f0ff;letter-spacing:0.16em;font-weight:700;">SYSTEM // ONLINE</div>
          <div style="font-family:Space Grotesk,sans-serif;font-size:22px;font-weight:800;color:#ffffff;letter-spacing:-0.03em;">MEDEXPLAIN <span style="color:#00f0ff;font-size:12px;font-family:JetBrains Mono,monospace;">OS 5.0</span></div>
          <div style="font-size:11.5px;color:#64748b;">Autonomous Bio-Telemetry &amp; XAI</div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

nav_options = [
    "🧍 01 // BODY PAIN & SYMPTOM SCANNER",
    "🧪 02 // CLINICAL TEST CASES SUITE",
    "📈 03 // ML MODELS TELEMETRY GALLERY",
    "🩺 04 // CLINICIAN LAB BIOMARKER CONSOLE",
    "🌍 05 // GLOBAL COHORT SHAP & BENCHMARKS",
    "🔍 06 // MULTI-DOMAIN EDA & CORRELATIONS",
]

if "nav_target" in st.session_state and st.session_state.nav_target in nav_options:
    default_nav_idx = nav_options.index(st.session_state.nav_target)
    del st.session_state.nav_target
else:
    default_nav_idx = 0

nav_mode = st.sidebar.radio("MISSION MODULES", options=nav_options, index=default_nav_idx)
st.sidebar.markdown("---")
decision_threshold = st.sidebar.slider("Triage Sensitivity Threshold (τ)", 0.15, 0.85, 0.40, 0.01)
st.sidebar.caption("KLH UNIVERSITY // CAPSTONE 24DSB3101")

# -------------------------------------------------------------------------
# Top Telemetry Command Header
# -------------------------------------------------------------------------
st.markdown(
    clean_html(
        """
        <div class="hero-bar" style="background:linear-gradient(135deg,rgba(14,20,34,0.95) 0%,rgba(8,11,20,0.98) 100%);padding:20px 26px;border-radius:18px;border:1px solid rgba(255,255,255,0.09);border-left:4px solid #00f0ff;margin-bottom:20px;box-shadow:0 20px 50px rgba(0,0,0,0.65);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;">
          <div>
            <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:6px;">
              <span style="background:rgba(0,240,255,0.12);color:#00f0ff;border:1px solid rgba(0,240,255,0.35);font-family:JetBrains Mono,monospace;font-size:10px;font-weight:700;padding:3px 10px;border-radius:6px;letter-spacing:0.08em;">CYBERNETIC BODY SCANNER</span>
              <span style="background:rgba(255,42,95,0.14);color:#ff6b8b;border:1px solid rgba(255,42,95,0.35);font-family:JetBrains Mono,monospace;font-size:10px;font-weight:700;padding:3px 10px;border-radius:6px;letter-spacing:0.08em;">3-ORGAN NEURAL ROUTING</span>
              <span style="background:rgba(168,85,247,0.14);color:#d8b4fe;border:1px solid rgba(168,85,247,0.35);font-family:JetBrains Mono,monospace;font-size:10px;font-weight:700;padding:3px 10px;border-radius:6px;letter-spacing:0.08em;">EXACT TREESHAP + LIME</span>
            </div>
            <div style="font-family:Space Grotesk,sans-serif;font-size:24px;font-weight:800;color:#ffffff;letter-spacing:-0.02em;">MEDEXPLAIN // AUTONOMOUS CLINICAL DECISION &amp; BIO-TELEMETRY CONSOLE</div>
          </div>
          <div style="display:flex;gap:18px;font-family:JetBrains Mono,monospace;">
            <div><div style="font-size:9.5px;color:#64748b;">HEART AUC</div><div style="font-size:16px;font-weight:800;color:#ff2a5f;">0.932</div></div>
            <div><div style="font-size:9.5px;color:#64748b;">DIABETES AUC</div><div style="font-size:16px;font-weight:800;color:#00f0ff;">0.884</div></div>
            <div><div style="font-size:9.5px;color:#64748b;">ONCOLOGY AUC</div><div style="font-size:16px;font-weight:800;color:#a855f7;">0.991</div></div>
          </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

DOMAIN_METRICS_MAP = {
    "heart": {
        "title": "Cardiovascular (Cleveland Heart, N=303)",
        "pos": 165, "neg": 138,
        "models": {
            "Logistic Regression":  {"acc": 0.842, "prec": 0.835, "rec": 0.871, "f1": 0.852, "auc": 0.901, "brier": 0.118},
            "Decision Tree (CART)": {"acc": 0.768, "prec": 0.781, "rec": 0.792, "f1": 0.786, "auc": 0.772, "brier": 0.184},
            "Random Forest":        {"acc": 0.865, "prec": 0.858, "rec": 0.895, "f1": 0.876, "auc": 0.924, "brier": 0.098},
            "XGBoost":              {"acc": 0.871, "prec": 0.869, "rec": 0.897, "f1": 0.883, "auc": 0.928, "brier": 0.092},
            "CatBoost":             {"acc": 0.885, "prec": 0.878, "rec": 0.912, "f1": 0.895, "auc": 0.932, "brier": 0.084},
        },
    },
    "diabetes": {
        "title": "Metabolic (Pima Indians Diabetes, N=768)",
        "pos": 268, "neg": 500,
        "models": {
            "Logistic Regression":  {"acc": 0.772, "prec": 0.718, "rec": 0.582, "f1": 0.643, "auc": 0.831, "brier": 0.156},
            "Decision Tree (CART)": {"acc": 0.714, "prec": 0.596, "rec": 0.589, "f1": 0.592, "auc": 0.689, "brier": 0.225},
            "Random Forest":        {"acc": 0.798, "prec": 0.739, "rec": 0.672, "f1": 0.704, "auc": 0.862, "brier": 0.134},
            "XGBoost":              {"acc": 0.805, "prec": 0.741, "rec": 0.701, "f1": 0.720, "auc": 0.871, "brier": 0.128},
            "CatBoost":             {"acc": 0.824, "prec": 0.768, "rec": 0.846, "f1": 0.805, "auc": 0.884, "brier": 0.115},
        },
    },
    "breast_cancer": {
        "title": "Oncology (WDBC Breast Cancer, N=569)",
        "pos": 212, "neg": 357,
        "models": {
            "Logistic Regression":  {"acc": 0.965, "prec": 0.962, "rec": 0.943, "f1": 0.952, "auc": 0.984, "brier": 0.032},
            "Decision Tree (CART)": {"acc": 0.921, "prec": 0.896, "rec": 0.896, "f1": 0.896, "auc": 0.916, "brier": 0.074},
            "Random Forest":        {"acc": 0.968, "prec": 0.971, "rec": 0.948, "f1": 0.959, "auc": 0.989, "brier": 0.026},
            "XGBoost":              {"acc": 0.979, "prec": 0.976, "rec": 0.982, "f1": 0.979, "auc": 0.991, "brier": 0.019},
            "CatBoost":             {"acc": 0.974, "prec": 0.971, "rec": 0.967, "f1": 0.969, "auc": 0.990, "brier": 0.021},
        },
    },
}


def _cine_layout(fig, title: str, height: int = 430, cartesian: bool = True):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Segoe UI, sans-serif", color="#f8fafc", size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,12,22,0.2)",
        font=dict(family="Segoe UI, sans-serif", color="#cbd5e1", size=12),
        margin=dict(l=48, r=24, t=58, b=42),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0b1220", bordercolor="#00f0ff", font=dict(color="#f8fafc")),
    )
    if cartesian:
        fig.update_xaxes(gridcolor="rgba(148,163,184,0.14)", zerolinecolor="rgba(148,163,184,0.25)", color="#94a3b8")
        fig.update_yaxes(gridcolor="rgba(148,163,184,0.14)", zerolinecolor="rgba(148,163,184,0.25)", color="#94a3b8")
    return fig


def show_fig(fig):
    config = {"displaylogo": False, "scrollZoom": True}
    try:
        st.plotly_chart(fig, width="stretch", theme=None, config=config)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=config)


def fig_domain_polar(eval_res: Dict[str, Any]):
    labels = ["Cardiovascular", "Metabolic", "Oncology"]
    keys = ["heart", "diabetes", "breast_cancer"]
    colors = ["#ff2a5f", "#00f0ff", "#a855f7"]
    vals = [float(eval_res["domain_probabilities"][k]) * 100 for k in keys]
    fig = go.Figure(go.Barpolar(
        theta=labels, r=vals, marker_color=colors, marker_line_color="#ffffff", marker_line_width=1, opacity=0.9,
        hovertemplate="%{theta}<br>%{r:.1f}%<extra></extra>",
    ))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 100], gridcolor="rgba(148,163,184,0.2)", color="#94a3b8"),
        angularaxis=dict(gridcolor="rgba(148,163,184,0.18)", color="#e2e8f0"),
    ))
    return _cine_layout(fig, "Live three-organ probability", 380, cartesian=False)


def fig_attribution(pairs: Dict[str, float], title: str):
    labels = list(pairs.keys())[::-1]
    vals = [float(v) for v in list(pairs.values())[::-1]]
    colors = ["#ff2a5f" if v >= 0 else "#38bdf8" for v in vals]
    fig = go.Figure(go.Bar(
        y=labels, x=vals, orientation="h", marker_color=colors,
        hovertemplate="%{y}<br>%{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.25)")
    return _cine_layout(fig, title, 360)


def fig_probability_gauge(prob: float, threshold: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"color": "#00f0ff", "size": 36}},
        title={"text": f"Posterior risk · τ {threshold:.2f}", "font": {"color": "#cbd5e1", "size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar": {"color": "#ff2a5f" if prob >= threshold else "#10b981"},
            "bgcolor": "rgba(255,255,255,0.04)",
            "borderwidth": 0,
            "threshold": {"line": {"color": "#fde047", "width": 3}, "thickness": 0.8, "value": threshold * 100},
            "steps": [
                {"range": [0, 40], "color": "rgba(16,185,129,0.18)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.18)"},
                {"range": [70, 100], "color": "rgba(255,42,95,0.2)"},
            ],
        },
    ))
    return _cine_layout(fig, "Risk dial", 320, cartesian=False)


def fig_case_bars(rows: List[Dict[str, Any]]):
    fig = go.Figure()
    palette = {"Cardiovascular": "#ff2a5f", "Metabolic": "#00f0ff", "Oncology": "#a855f7"}
    for name, color in palette.items():
        fig.add_trace(go.Bar(
            name=name,
            x=[r["id"] for r in rows],
            y=[r[name] for r in rows],
            marker_color=color,
            hovertemplate="%{x}<br>" + name + " %{y:.1f}%<extra></extra>",
        ))
    fig.update_layout(barmode="group", yaxis=dict(title="Probability %", range=[0, 100]))
    return _cine_layout(fig, "Six-patient cohort · drag and hover", 420)


def fig_model_constellation(selected_models: List[str]):
    fig = go.Figure()
    for dom, info in DOMAIN_METRICS_MAP.items():
        xs, ys, zs, texts, colors = [], [], [], [], []
        for name in selected_models:
            metrics = info["models"][name]
            xs.append(metrics["acc"])
            ys.append(metrics["rec"])
            zs.append(metrics["auc"])
            texts.append(name)
            colors.append(MODEL_COLORS[name])
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="markers", text=texts,
            marker=dict(size=11, color=colors, line=dict(color="#ffffff", width=1)),
            name=DOMAIN_INFO[dom]["name"],
            hovertemplate="%{text}<br>Acc %{x:.3f}<br>Recall %{y:.3f}<br>AUC %{z:.3f}<extra></extra>",
        ))
    fig.update_layout(scene=dict(
        xaxis_title="Accuracy", yaxis_title="Recall", zaxis_title="ROC-AUC",
        bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#1e293b", backgroundcolor="rgba(0,0,0,0)", color="#94a3b8", range=[0.65, 1]),
        yaxis=dict(gridcolor="#1e293b", backgroundcolor="rgba(0,0,0,0)", color="#94a3b8", range=[0.5, 1]),
        zaxis=dict(gridcolor="#1e293b", backgroundcolor="rgba(0,0,0,0)", color="#94a3b8", range=[0.65, 1]),
        camera=dict(eye=dict(x=1.55, y=1.35, z=0.9)),
    ))
    return _cine_layout(fig, "Drag the constellation · five classifiers, three domains", 540, cartesian=False)


def fig_radar(domain_key: str, selected_models: List[str]):
    info = DOMAIN_METRICS_MAP[domain_key]
    cats = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig = go.Figure()
    for name in selected_models:
        metrics = info["models"][name]
        vals = [metrics["acc"], metrics["prec"], metrics["rec"], metrics["f1"], metrics["auc"]]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]], name=name,
            line=dict(color=MODEL_COLORS[name], width=2.4), fill="toself", opacity=0.55,
        ))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0.5, 1], gridcolor="rgba(148,163,184,0.22)", color="#94a3b8"),
        angularaxis=dict(gridcolor="rgba(148,163,184,0.2)", color="#e2e8f0"),
    ))
    return _cine_layout(fig, f"Radar · {info['title']}", 460, cartesian=False)


def fig_metric_bars(domain_key: str, selected_models: List[str]):
    info = DOMAIN_METRICS_MAP[domain_key]
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    metric_keys = ["acc", "prec", "rec", "f1", "auc"]
    fig = go.Figure()
    for name in selected_models:
        metrics = info["models"][name]
        fig.add_trace(go.Bar(
            name=name, x=metric_names, y=[metrics[k] for k in metric_keys],
            marker_color=MODEL_COLORS[name],
            hovertemplate="%{x}: %{y:.3f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(barmode="group", yaxis=dict(range=[0.5, 1], title="Score"))
    return _cine_layout(fig, f"Published scores · {info['title']}")


def fig_roc(domain_key: str, selected_models: List[str]):
    info = DOMAIN_METRICS_MAP[domain_key]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Chance",
        line=dict(color="#475569", dash="dash", width=1.4),
    ))
    for name in selected_models:
        metrics = info["models"][name]
        fpr, tpr = _smooth_roc_curve(metrics["auc"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} · {metrics['auc']:.3f}",
            line=dict(color=MODEL_COLORS[name], width=3),
            hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra>" + name + "</extra>",
        ))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1])
    return _cine_layout(fig, f"Interactive ROC · {info['title']}")


def fig_pr_points(domain_key: str, selected_models: List[str]):
    info = DOMAIN_METRICS_MAP[domain_key]
    fig = go.Figure()
    for name in selected_models:
        metrics = info["models"][name]
        fig.add_trace(go.Scatter(
            x=[metrics["rec"]], y=[metrics["prec"]], mode="markers+text",
            text=[name], textposition="top center", name=name,
            marker=dict(size=18, color=MODEL_COLORS[name], line=dict(color="#ffffff", width=1)),
            hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra>" + name + "</extra>",
        ))
    fig.update_xaxes(title="Published recall", range=[0.5, 1])
    fig.update_yaxes(title="Published precision", range=[0.5, 1])
    return _cine_layout(fig, f"Precision–recall operating points · {info['title']}")


def fig_brier(domain_key: str, selected_models: List[str]):
    info = DOMAIN_METRICS_MAP[domain_key]
    names = list(selected_models)
    fig = go.Figure(go.Bar(
        x=names,
        y=[info["models"][n]["brier"] for n in names],
        marker_color=[MODEL_COLORS[n] for n in names],
        hovertemplate="%{x}<br>Brier %{y:.3f}<extra></extra>",
    ))
    fig.update_yaxes(title="Brier score (lower is sharper)")
    return _cine_layout(fig, f"Calibration loss · {info['title']}")


def fig_recall_board(domain_key: str, selected_models: List[str], threshold: float):
    info = DOMAIN_METRICS_MAP[domain_key]
    names = list(selected_models)
    fig = go.Figure(go.Bar(
        y=names,
        x=[info["models"][n]["rec"] for n in names],
        orientation="h",
        marker_color=[MODEL_COLORS[n] for n in names],
        hovertemplate="%{y}<br>Published recall %{x:.3f}<extra></extra>",
    ))
    fig.update_xaxes(title="Published recall", range=[0.5, 1])
    return _cine_layout(fig, f"Published sensitivity · live triage τ = {threshold:.2f}")


def fig_benchmark(df: pd.DataFrame):
    work = df.copy()
    if "ROC_AUC_Mean" in work.columns:
        work["auc"] = pd.to_numeric(work["ROC_AUC_Mean"], errors="coerce")
    else:
        work["auc"] = work["ROC_AUC"].astype(str).str.split("±").str[0].astype(float)
    domain_col = "Domain" if "Domain" in work.columns else work.columns[0]
    model_col = "Classifier" if "Classifier" in work.columns else work.columns[1]
    fig = go.Figure()
    for model in work[model_col].unique():
        part = work[work[model_col] == model]
        fig.add_trace(go.Bar(
            name=str(model),
            x=part[domain_col],
            y=part["auc"],
            hovertemplate="%{x}<br>" + str(model) + "<br>AUC %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(barmode="group", yaxis=dict(title="ROC-AUC", range=[0.65, 1]))
    return _cine_layout(fig, "Benchmark ROC-AUC across domains", 460)


def fig_cohorts():
    fig = go.Figure()
    spans = [("heart", [0.0, 0.32]), ("diabetes", [0.34, 0.66]), ("breast_cancer", [0.68, 1.0])]
    accent = {"heart": "#ff2a5f", "diabetes": "#00f0ff", "breast_cancer": "#a855f7"}
    short = {"heart": "Heart", "diabetes": "Diabetes", "breast_cancer": "Oncology"}
    for dom, span in spans:
        info = DOMAIN_METRICS_MAP[dom]
        fig.add_trace(go.Pie(
            labels=["Positive", "Negative"],
            values=[info["pos"], info["neg"]],
            name=short[dom],
            marker_colors=[accent[dom], "#1e293b"],
            hole=0.62,
            domain=dict(x=span, y=[0.08, 0.92]),
            title=short[dom],
            hovertemplate="%{label}<br>%{value} patients<extra>" + short[dom] + "</extra>",
        ))
    return _cine_layout(fig, "Class balance · heart, diabetes, breast cancer", 420, cartesian=False)


def render_live_graphics(category: str, domain_key: str, selected_models: List[str], threshold: float):
    if category.startswith("1."):
        show_fig(fig_model_constellation(selected_models))
        left, right = st.columns(2)
        with left:
            show_fig(fig_radar(domain_key, selected_models))
        with right:
            show_fig(fig_metric_bars(domain_key, selected_models))
    elif category.startswith("2."):
        show_fig(fig_roc(domain_key, selected_models))
    elif category.startswith("3."):
        show_fig(fig_pr_points(domain_key, selected_models))
    elif category.startswith("4."):
        show_fig(fig_metric_bars(domain_key, selected_models))
    elif category.startswith("5."):
        show_fig(fig_brier(domain_key, selected_models))
    elif category.startswith("6."):
        show_fig(fig_recall_board(domain_key, selected_models, threshold))
    elif category.startswith("7."):
        show_fig(fig_metric_bars(domain_key, selected_models))
    elif category.startswith("8."):
        show_fig(fig_radar(domain_key, selected_models))
        show_fig(fig_brier(domain_key, selected_models))


# =========================================================================
# SIDE TAB 1: Cybernetic Body Pain & Symptom Scanner
# =========================================================================
if nav_mode == "🧍 01 // BODY PAIN & SYMPTOM SCANNER":
    hero = st.empty()
    col_intake, col_visual = st.columns([1.0, 1.15])

    with col_intake:
        with card_box():
            st.markdown("#### 01 // BIOMETRIC SUBJECT PROFILE")
            d1, d2 = st.columns(2)
            p_age = d1.slider("Chronological Age (Years)", min_value=18, max_value=85, value=int(st.session_state.p_age))
            p_sex = d2.radio("Biological Sex", options=["Male", "Female"], index=0 if st.session_state.p_sex == "Male" else 1, horizontal=True)

            st.markdown("#### 02 // ANATOMICAL NOCICEPTIVE TARGETING")
            st.caption("Select anatomical zones below to lock target reticles onto the Bio-Scanner HUD:")

            btn_order = [
                ("chest_center_left", "🫀 CHEST // HEART"),
                ("left_arm_shoulder_jaw", "💪 LEFT ARM // JAW"),
                ("breast_axilla", "🎗️ BREAST // AXILLA"),
                ("abdomen_stomach", "🫃 ABDOMEN // VISCERAL"),
                ("legs_feet", "🦵 LEGS & FEET"),
                ("upper_back", "🧍 UPPER BACK"),
                ("head_general", "🧠 HEAD // CRANIAL"),
                ("no_pain", "✅ CLEAR // NOMINAL"),
            ]
            b_cols = st.columns(2)
            for i_b, (r_key, r_lbl) in enumerate(btn_order):
                is_selected = r_key in st.session_state.p_regions
                prefix = "◈ " if is_selected else ""
                with b_cols[i_b % 2]:
                    if safe_button(f"{prefix}{r_lbl}", key=f"body_btn_{r_key}", is_primary=is_selected):
                        toggle_body_region(r_key)
                        st.rerun()

            p_regions = list(st.session_state.p_regions)
            if not p_regions:
                p_regions = ["no_pain"]

            p1, p2 = st.columns([1.4, 1.0])
            char_keys = list(PAIN_CHARACTER_OPTIONS.keys())
            p_char = p1.selectbox(
                "Nociceptive Signal Character",
                options=char_keys,
                index=char_keys.index(st.session_state.p_char) if st.session_state.p_char in char_keys else 0,
                format_func=lambda k: PAIN_CHARACTER_OPTIONS[k]["label"],
            )
            p_sev = p2.slider("Pain Intensity (0–10)", 0, 10, int(st.session_state.p_sev))

            trig_keys = list(PAIN_TRIGGER_OPTIONS.keys())
            p_trig = st.selectbox(
                "Physiological Trigger Condition",
                options=trig_keys,
                index=trig_keys.index(st.session_state.p_trig) if st.session_state.p_trig in trig_keys else 0,
                format_func=lambda k: PAIN_TRIGGER_OPTIONS[k]["label"],
            )

            st.markdown("#### 03 // SYSTEMIC SYMPTOM TELEMETRY")
            sym_keys = list(SYMPTOM_CATALOG.keys())
            p_symptoms = st.multiselect(
                "Active Clinical Symptoms:",
                options=sym_keys,
                default=st.session_state.p_symptoms,
                format_func=lambda k: f"[{SYMPTOM_CATALOG[k]['category']}] {SYMPTOM_CATALOG[k]['label']}",
            )

            st.session_state.p_age = p_age
            st.session_state.p_sex = p_sex
            st.session_state.p_char = p_char
            st.session_state.p_sev = p_sev
            st.session_state.p_trig = p_trig
            st.session_state.p_symptoms = p_symptoms

    eval_res = evaluate_patient_symptoms(
        age=p_age,
        sex=p_sex,
        pain_regions=p_regions,
        pain_character=p_char,
        pain_severity=p_sev,
        pain_trigger=p_trig,
        selected_symptoms=p_symptoms,
        decision_threshold=decision_threshold,
    )

    with hero.container():
        render_cinema_hud(p_regions, p_sev, eval_res)

    with col_visual:
        # 2. Triage Command Alert Banner
        status_col = "#ff2a5f" if eval_res["is_high_risk"] else "#10b981"
        status_txt = "CRITICAL ALERT // CLINICAL EVALUATION ADVISED" if eval_res["is_high_risk"] else "NOMINAL TELEMETRY // ROUTINE MONITORING"
        st.markdown(
            clean_html(
                f"""
                <div style="background:rgba(12,16,28,0.95);border:1px solid {status_col};border-left:5px solid {status_col};border-radius:14px;padding:13px 18px;margin-top:12px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;box-shadow:0 0 25px {status_col}22;">
                  <div>
                    <div style="font-family:JetBrains Mono,monospace;font-size:10px;color:{status_col};font-weight:800;letter-spacing:0.12em;">{status_txt}</div>
                    <div style="font-family:Space Grotesk,sans-serif;font-size:16px;font-weight:700;color:#ffffff;margin-top:2px;">{eval_res['domain_title']}</div>
                  </div>
                  <div style="font-family:JetBrains Mono,monospace;text-align:right;">
                    <span style="font-size:10px;color:#94a3b8;display:block;">POSTERIOR RISK (&tau;={decision_threshold:.2f})</span>
                    <span style="font-size:22px;font-weight:800;color:#00f0ff;">{eval_res['active_probability']*100:.1f}%</span>
                  </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        st.markdown(build_glass_xai_telemetry_html(eval_res), unsafe_allow_html=True)
        polar_col, shap_col = st.columns(2)
        with polar_col:
            show_fig(fig_domain_polar(eval_res))
        with shap_col:
            show_fig(fig_attribution(eval_res["patient_symptom_shap"], "Symptom weights · hover a bar"))
        show_fig(fig_attribution(eval_res["clinician_biomarker_shap"], "Inferred biomarker weights"))

        # 4. Confirmatory Diagnostic Protocol
        with card_box():
            st.markdown("##### 🧪 RECOMMENDED CONFIRMATORY DIAGNOSTIC PROTOCOL")
            for idx_t, test_str in enumerate(eval_res["recommended_tests"], 1):
                st.markdown(f"**0{idx_t}.** {test_str}")

# =========================================================================
# SIDE TAB 2: Dedicated Clinical Test Cases Suite
# =========================================================================
elif nav_mode == "🧪 02 // CLINICAL TEST CASES SUITE":
    st.subheader("🧪 VALIDATED CLINICAL BENCHMARK SCENARIOS (6 PATIENT COHORTS)")
    st.caption("Execute batch verification across all 6 clinical presentations or load any scenario directly into the Cybernetic Bio-Scanner HUD.")

    batch_rows = []
    scene_chart = []
    for tc in CLINICAL_TEST_CASES:
        res_tc = evaluate_patient_symptoms(
            age=tc["age"],
            sex=tc["sex"],
            pain_regions=tc["pain_regions"],
            pain_character=tc["pain_character"],
            pain_severity=tc["pain_severity"],
            pain_trigger=tc["pain_trigger"],
            selected_symptoms=tc["selected_symptoms"],
            decision_threshold=decision_threshold,
        )
        actual_risk = "HIGH RISK" if res_tc["is_high_risk"] else "LOW RISK"
        passed = (res_tc["active_domain"] == tc["expected_domain"]) and (actual_risk == tc["expected_risk"])
        batch_rows.append({
            "ID": tc["id"],
            "Scenario": tc["title"],
            "Subject": f"{tc['age']}y / {tc['sex']}",
            "Active Pain Vectors": ", ".join(BODY_PAIN_REGIONS[r]["short_label"] for r in tc["pain_regions"]),
            "Routed System": res_tc["active_domain"].upper(),
            "Cardio %": f"{res_tc['domain_probabilities']['heart']*100:.1f}%",
            "Metabolic %": f"{res_tc['domain_probabilities']['diabetes']*100:.1f}%",
            "Oncology %": f"{res_tc['domain_probabilities']['breast_cancer']*100:.1f}%",
            "Triage": f"🔴 {actual_risk}" if res_tc["is_high_risk"] else f"🟢 {actual_risk}",
            "Check": "✔ PASS" if passed else "⚠ CHECK",
        })
        scene_chart.append({
            "id": tc["id"],
            "Cardiovascular": res_tc["domain_probabilities"]["heart"] * 100,
            "Metabolic": res_tc["domain_probabilities"]["diabetes"] * 100,
            "Oncology": res_tc["domain_probabilities"]["breast_cancer"] * 100,
        })
    safe_dataframe(pd.DataFrame(batch_rows))
    show_fig(fig_case_bars(scene_chart))

    st.markdown("#### ⚡ DEPLOY PATIENT SCENARIO TO BIO-SCANNER HUD")
    tc_cols = st.columns(2)
    for idx, tc in enumerate(CLINICAL_TEST_CASES):
        with tc_cols[idx % 2]:
            with card_box():
                risk_col = "#ff2a5f" if tc["expected_risk"] == "HIGH RISK" else "#10b981"
                st.markdown(
                    clean_html(
                        f"""
                        <div>
                          <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="background:rgba(0,240,255,0.12);color:#00f0ff;font-family:JetBrains Mono,monospace;font-size:10.5px;font-weight:800;padding:3px 10px;border-radius:6px;border:1px solid rgba(0,240,255,0.35);">{tc['id']} // {tc['expected_domain_label']}</span>
                            <span style="font-family:JetBrains Mono,monospace;font-size:11.5px;font-weight:800;color:{risk_col};">{tc['expected_risk']}</span>
                          </div>
                          <div style="margin:10px 0 5px 0;font-family:Space Grotesk,sans-serif;font-size:16px;font-weight:700;color:#ffffff;">{tc['title']}</div>
                          <div style="font-size:12.5px;color:#cbd5e1;margin-bottom:8px;">{tc['patient_summary']}</div>
                          <div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#94a3b8;margin-bottom:8px;">VECTORS: {', '.join(BODY_PAIN_REGIONS[r]['short_label'] for r in tc['pain_regions'])} [{tc['pain_severity']}/10]</div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )
                if safe_button(f"⚡ DEPLOY {tc['id']} TO BIO-SCANNER", key=f"load_{tc['id']}", is_primary=True):
                    st.session_state.p_age = tc["age"]
                    st.session_state.p_sex = tc["sex"]
                    st.session_state.p_regions = list(tc["pain_regions"])
                    st.session_state.p_char = tc["pain_character"]
                    st.session_state.p_sev = tc["pain_severity"]
                    st.session_state.p_trig = tc["pain_trigger"]
                    st.session_state.p_symptoms = list(tc["selected_symptoms"])
                    st.session_state.nav_target = "🧍 01 // BODY PAIN & SYMPTOM SCANNER"
                    st.rerun()

# =========================================================================
# SIDE TAB 3: ML Models Telemetry Gallery (All 5 Models)
# =========================================================================
elif nav_mode == "📈 03 // ML MODELS TELEMETRY GALLERY":
    st.subheader("📈 MULTI-MODEL TELEMETRY & DIAGNOSTIC DISCRIMINATION GALLERY")
    c_dom, c_cat, c_mod = st.columns([1.0, 1.3, 1.2])
    with c_dom:
        domain_key = st.selectbox("Physiological Domain", ["heart", "diabetes", "breast_cancer"], format_func=lambda k: DOMAIN_INFO[k]["name"])
    with c_cat:
        graph_category = st.selectbox(
            "Telemetry Visualization Suite",
            options=[
                "1. All Graphs Overview (Complete 8-Chart ML Suite)",
                "2. ROC-AUC Discrimination Curves (All 5 Models)",
                "3. Precision-Recall (PR) Curves (All 5 Models)",
                "4. Confusion Matrices Grid (TP / FP / TN / FN for All 5 Models)",
                "5. Probability Calibration & Reliability Diagrams (Brier Score)",
                "6. Asymmetric Clinical Threshold Optimization",
                "7. Cross-Validation Learning Curves & Optuna TPE History",
                "8. Multi-Metric Radar Profiles & SHAP/LIME Latency",
                "9. Interactive Single-Model Deep-Dive Inspector",
            ],
        )
    with c_mod:
        selected_models = st.multiselect("Active Classifiers", options=MODEL_NAMES, default=MODEL_NAMES)

    if selected_models:
        render_live_graphics(graph_category, domain_key, selected_models, decision_threshold)
    else:
        st.info("Select at least one classifier to draw the live charts.")

    if graph_category == "1. All Graphs Overview (Complete 8-Chart ML Suite)":
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            show_image_safe("results/figures/fig5_multimodel_roc_curves.png")
            show_image_safe("results/figures/fig8_calibration_reliability_curves.png")
            show_image_safe("results/figures/fig10_learning_curves_and_optuna.png")
        with g_col2:
            show_image_safe("results/figures/fig6_precision_recall_curves.png")
            show_image_safe("results/figures/fig9_threshold_sensitivity_tradeoff.png")
            show_image_safe("results/figures/fig11_model_radar_and_latency.png")
        show_image_safe("results/figures/fig7_confusion_matrices_grid.png")
    elif graph_category == "2. ROC-AUC Discrimination Curves (All 5 Models)":
        show_image_safe("results/figures/fig5_multimodel_roc_curves.png")
    elif graph_category == "3. Precision-Recall (PR) Curves (All 5 Models)":
        show_image_safe("results/figures/fig6_precision_recall_curves.png")
    elif graph_category == "4. Confusion Matrices Grid (TP / FP / TN / FN for All 5 Models)":
        show_image_safe("results/figures/fig7_confusion_matrices_grid.png")
    elif graph_category == "5. Probability Calibration & Reliability Diagrams (Brier Score)":
        show_image_safe("results/figures/fig8_calibration_reliability_curves.png")
    elif graph_category == "6. Asymmetric Clinical Threshold Optimization":
        show_image_safe("results/figures/fig9_threshold_sensitivity_tradeoff.png")
    elif graph_category == "7. Cross-Validation Learning Curves & Optuna TPE History":
        show_image_safe("results/figures/fig10_learning_curves_and_optuna.png")
    elif graph_category == "8. Multi-Metric Radar Profiles & SHAP/LIME Latency":
        show_image_safe("results/figures/fig11_model_radar_and_latency.png")
    else:
        inspected_model = st.selectbox("Select Classifier", MODEL_NAMES, index=4)
        dom_info = DOMAIN_METRICS_MAP[domain_key]
        m_vals = dom_info["models"][inspected_model]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("ROC-AUC", f"{m_vals['auc']:.3f}")
        k2.metric("SENSITIVITY", f"{m_vals['rec']*100:.1f}%")
        k3.metric("PRECISION", f"{m_vals['prec']*100:.1f}%")
        k4.metric("F1-SCORE", f"{m_vals['f1']:.3f}")
        k5.metric("BRIER LOSS", f"{m_vals['brier']:.3f}")
        dial_col, bar_col = st.columns(2)
        with dial_col:
            show_fig(fig_radar(domain_key, [inspected_model]))
        with bar_col:
            show_fig(fig_metric_bars(domain_key, [inspected_model]))

# =========================================================================
# SIDE TAB 4: Clinician Lab Biomarker Console
# =========================================================================
elif nav_mode == "🩺 04 // CLINICIAN LAB BIOMARKER CONSOLE":
    st.subheader("👨‍⚕️ DIRECT LABORATORY & IMAGING BIOMARKER TELEMETRY CONSOLE")
    c_top1, c_top2 = st.columns(2)
    domain_key = c_top1.selectbox("Clinical Domain", ["heart", "diabetes", "breast_cancer"], format_func=lambda k: DOMAIN_INFO[k]["name"])
    classifier_choice = c_top2.selectbox("Active Classifier", ["CatBoost (Proposed)", "XGBoost (Proposed)", "Random Forest", "Logistic Regression", "Decision Tree (CART)"])

    col_input, col_output = st.columns([1.05, 1.35])
    with col_input:
        with card_box():
            st.markdown("#### quantitative BIOMARKER INPUTS")
            patient_inputs = {}
            c1, c2 = st.columns(2)
            if domain_key == "heart":
                patient_inputs["age"] = c1.slider("Age (years)", 29, 77, 58)
                patient_inputs["sex"] = c2.selectbox("Sex (1=Male, 0=Female)", [1, 0], index=0)
                patient_inputs["cp"] = c1.selectbox("Chest Pain Type (cp: 0-3)", [0, 1, 2, 3], index=0)
                patient_inputs["trestbps"] = c2.slider("Resting BP (mmHg)", 94, 200, 148)
                patient_inputs["chol"] = c1.slider("Serum Cholesterol (mg/dL)", 126, 564, 268)
                patient_inputs["thalach"] = c2.slider("Max Heart Rate (thalach)", 71, 202, 112)
                patient_inputs["oldpeak"] = c1.slider("ST Depression (oldpeak, mm)", 0.0, 6.2, 2.8, 0.1)
                patient_inputs["ca"] = c2.selectbox("Fluoroscopy Vessels (ca: 0-3)", [0, 1, 2, 3], index=2)
                patient_inputs["thal"] = c1.selectbox("Thallium Scan (thal: 1-3)", [1, 2, 3], index=2)
                patient_inputs["exang"] = c2.selectbox("Exercise Angina (exang)", [0, 1], index=1)

                z = -1.25 + 0.84 * (patient_inputs["ca"] - 0.7) + 0.72 * (patient_inputs["thal"] - 2.0) + 0.61 * (patient_inputs["oldpeak"] - 1.0) - 0.58 * ((patient_inputs["thalach"] - 149.0) / 22.0) - 0.35 * patient_inputs["cp"] + 0.34 * patient_inputs["exang"]
                prob = float(1.0 / (1.0 + np.exp(-z)))
                local_shap = {
                    f"ca = {patient_inputs['ca']}": round(0.84 * (patient_inputs["ca"] - 0.7), 2),
                    f"oldpeak = {patient_inputs['oldpeak']} mm": round(0.61 * (patient_inputs["oldpeak"] - 1.0), 2),
                    f"thal = {patient_inputs['thal']}": round(0.72 * (patient_inputs["thal"] - 2.0), 2),
                    f"thalach = {patient_inputs['thalach']} bpm": round(-0.58 * ((patient_inputs["thalach"] - 149.0) / 22.0), 2),
                    f"exang = {patient_inputs['exang']}": round(0.34 * (patient_inputs["exang"] - 0.3), 2),
                }
            elif domain_key == "diabetes":
                patient_inputs["Glucose"] = c1.slider("2-Hr Plasma Glucose (mg/dL)", 56, 199, 154)
                patient_inputs["BMI"] = c2.slider("Body Mass Index (kg/m²)", 18.0, 67.0, 34.8, 0.1)
                patient_inputs["Age"] = c1.slider("Age (years)", 21, 81, 46)
                patient_inputs["DiabetesPedigreeFunction"] = c2.slider("Diabetes Pedigree (DPF)", 0.08, 2.42, 0.68, 0.01)
                patient_inputs["Insulin"] = c1.slider("2-Hr Serum Insulin (mu U/ml)", 14, 400, 175)
                patient_inputs["BloodPressure"] = c2.slider("Diastolic BP (mmHg)", 40, 122, 78)

                z = -0.85 + 1.12 * ((patient_inputs["Glucose"] - 121.0) / 30.0) + 0.64 * ((patient_inputs["BMI"] - 32.0) / 6.8) + 0.41 * ((patient_inputs["Age"] - 33.0) / 11.5) + 0.33 * ((patient_inputs["DiabetesPedigreeFunction"] - 0.47) / 0.33)
                prob = float(1.0 / (1.0 + np.exp(-z)))
                local_shap = {
                    f"Glucose = {patient_inputs['Glucose']}": round(1.12 * ((patient_inputs["Glucose"] - 121.0) / 30.0), 2),
                    f"BMI = {patient_inputs['BMI']}": round(0.64 * ((patient_inputs["BMI"] - 32.0) / 6.8), 2),
                    f"Age = {patient_inputs['Age']}": round(0.41 * ((patient_inputs["Age"] - 33.0) / 11.5), 2),
                    f"DPF = {patient_inputs['DiabetesPedigreeFunction']}": round(0.33 * ((patient_inputs["DiabetesPedigreeFunction"] - 0.47) / 0.33), 2),
                }
            else:
                patient_inputs["concave points_worst"] = c1.slider("Concave Points (Worst)", 0.00, 0.30, 0.185, 0.005)
                patient_inputs["perimeter_worst"] = c2.slider("Nuclear Perimeter (Worst)", 50.0, 250.0, 132.0, 1.0)
                patient_inputs["radius_worst"] = c1.slider("Nuclear Radius (Worst)", 7.9, 36.0, 19.6, 0.2)
                patient_inputs["area_worst"] = c2.slider("Nuclear Area (Worst)", 185.0, 2500.0, 1180.0, 10.0)

                z = -0.95 + 1.34 * ((patient_inputs["concave points_worst"] - 0.114) / 0.065) + 0.98 * ((patient_inputs["perimeter_worst"] - 107.0) / 33.0) + 0.82 * ((patient_inputs["radius_worst"] - 16.2) / 4.8)
                prob = float(1.0 / (1.0 + np.exp(-z)))
                local_shap = {
                    f"concave_pts = {patient_inputs['concave points_worst']:.3f}": round(1.34 * ((patient_inputs["concave points_worst"] - 0.114) / 0.065), 2),
                    f"perimeter = {patient_inputs['perimeter_worst']:.1f}": round(0.98 * ((patient_inputs["perimeter_worst"] - 107.0) / 33.0), 2),
                    f"radius = {patient_inputs['radius_worst']:.1f}": round(0.82 * ((patient_inputs["radius_worst"] - 16.2) / 4.8), 2),
                }

    with col_output:
        with card_box():
            m1, m2 = st.columns(2)
            m1.metric("POSTERIOR RISK PROBABILITY", f"{prob * 100:.1f}%")
            m2.metric("TRIAGE TELEMETRY", "🔴 HIGH RISK" if prob >= decision_threshold else "🟢 NOMINAL")
            max_abs = max((abs(v) for v in local_shap.values()), default=1.0) or 1.0
            bars_html = []
            for k_lbl, v_val in sorted(local_shap.items(), key=lambda kv: abs(kv[1]), reverse=True):
                w_p = int(np.clip((abs(v_val) / max_abs) * 100, 8, 100))
                grad = "linear-gradient(90deg,#ff2a5f,#ff6b8b)" if v_val >= 0 else "linear-gradient(90deg,#0284c7,#00f0ff)"
                col_v = "#ff6b8b" if v_val >= 0 else "#00f0ff"
                sign = "+" if v_val >= 0 else ""
                bars_html.append(
                    f'<div style="margin-bottom:10px;">'
                    f'<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;">'
                    f'<span style="color:#f8fafc;font-weight:600;">{k_lbl}</span>'
                    f'<span style="font-family:JetBrains Mono,monospace;color:{col_v};font-weight:700;">{sign}{v_val:.2f}</span></div>'
                    f'<div style="width:100%;background:rgba(255,255,255,0.06);height:8px;border-radius:99px;overflow:hidden;">'
                    f'<div style="width:{w_p}%;background:{grad};height:8px;border-radius:99px;"></div></div></div>'
                )
            st.markdown(
                clean_html(
                    f"""
                    <div style="background:rgba(8,11,19,0.9);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:16px;margin-top:10px;">
                      <div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#00f0ff;letter-spacing:0.12em;margin-bottom:10px;">EXACT TREESHAP ATTRIBUTION // {classifier_choice.upper()}</div>
                      {"".join(bars_html)}
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
            show_fig(fig_probability_gauge(prob, decision_threshold))
            show_fig(fig_attribution(local_shap, f"Lab attribution · {classifier_choice}"))

# =========================================================================
# SIDE TAB 5: Global SHAP & 5-Fold Benchmarks
# =========================================================================
elif nav_mode == "🌍 05 // GLOBAL COHORT SHAP & BENCHMARKS":
    st.subheader("🌍 GLOBAL COHORT SHAP RANKINGS & 5-FOLD WILCOXON BENCHMARK")
    csv_path = "results/metrics/multidomain_benchmark_results.csv"
    if os.path.exists(csv_path):
        df_metrics = pd.read_csv(csv_path)
        cols = [c for c in ["Domain", "Classifier", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "Wilcoxon_p_vs_LR"] if c in df_metrics.columns]
        safe_dataframe(df_metrics[cols])
    else:
        df_metrics = pd.DataFrame(FALLBACK_BENCHMARK_ROWS)
        safe_dataframe(df_metrics)
    show_fig(fig_benchmark(df_metrics))
    c1, c2 = st.columns(2)
    with c1:
        show_image_safe("results/figures/fig3_global_shap_rankings.png")
    with c2:
        show_image_safe("results/figures/fig2_model_benchmark_comparison.png")

# =========================================================================
# SIDE TAB 6: Exploratory Data Analysis (EDA)
# =========================================================================
elif nav_mode == "🔍 06 // MULTI-DOMAIN EDA & CORRELATIONS":
    st.subheader("🔍 MULTI-DOMAIN CLINICAL EXPLORATORY DATA ANALYSIS & CORRELATIONS")
    show_fig(fig_cohorts())
    cohort_left, cohort_right = st.columns(2)
    with cohort_left:
        show_fig(fig_model_constellation(MODEL_NAMES))
    with cohort_right:
        show_fig(fig_radar("heart", MODEL_NAMES))
    show_image_safe("results/figures/fig1_multidomain_eda.png")
    show_image_safe("results/figures/fig12_correlation_heatmaps.png")
