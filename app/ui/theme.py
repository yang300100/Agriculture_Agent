"""
Anthropic/Claude Design System — Tokens, CSS & Top Navigation Bar
Implements the cream-canvas + coral-primary + dark-navy design language from design.md.
"""

import streamlit as st
import streamlit.components.v1 as components

# =========================
# Design Tokens
# =========================
DESIGN_TOKENS = {
    "colors": {
        "canvas": "#faf9f5",
        "primary": "#cc785c",
        "primary_active": "#a9583e",
        "primary_disabled": "#e6dfd8",
        "accent_teal": "#5db8a6",
        "accent_amber": "#e8a55a",
        "surface_soft": "#f5f0e8",
        "surface_card": "#efe9de",
        "surface_cream_strong": "#e8e0d2",
        "surface_dark": "#181715",
        "surface_dark_elevated": "#252320",
        "surface_dark_soft": "#1f1e1b",
        "hairline": "#e6dfd8",
        "hairline_soft": "#ebe6df",
        "ink": "#141413",
        "body_strong": "#252523",
        "body": "#3d3d3a",
        "muted": "#6c6a64",
        "muted_soft": "#8e8b82",
        "on_primary": "#ffffff",
        "on_dark": "#faf9f5",
        "on_dark_soft": "#a09d96",
        "success": "#5db872",
        "warning": "#d4a017",
        "error": "#c64545",
    },
    "typography": {
        "display_family": (
            "'Cormorant Garamond', 'Tiempos Headline', Garamond, "
            "'Times New Roman', serif"
        ),
        "body_family": (
            "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', "
            "Roboto, sans-serif"
        ),
        "code_family": "'JetBrains Mono', 'Fira Code', monospace",
    },
    "spacing": {
        "xxs": "4px", "xs": "8px", "sm": "12px", "md": "16px",
        "lg": "24px", "xl": "32px", "xxl": "48px", "section": "96px",
    },
    "radius": {
        "xs": "4px", "sm": "6px", "md": "8px", "lg": "12px",
        "xl": "16px", "pill": "9999px",
    },
}

# Navigation items
NAV_ITEMS = [
    {"id": "dashboard", "label": "概览", "icon": "📊"},
    {"id": "chat", "label": "对话", "icon": "💬"},
    {"id": "profile", "label": "基本信息", "icon": "👤"},
    {"id": "fields", "label": "地块管理", "icon": "📍"},
    {"id": "finance", "label": "财务管理", "icon": "💰"},
    {"id": "calendar", "label": "农事日历", "icon": "📅"},

    {"id": "encyclopedia", "label": "作物百科", "icon": "📖"},
    {"id": "calculator", "label": "农资计算", "icon": "🧮"},
    {"id": "wizard", "label": "种植向导", "icon": "🪄"},
    {"id": "devices", "label": "设备仪表盘", "icon": "🤖"},
    {"id": "rules", "label": "规则管理", "icon": "📋"},
]


def apply_theme():
    """Inject the Anthropic design system CSS + Google Fonts into the Streamlit app."""

    css = """
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">

    <style>
    /* ===== ROOT / APP BACKGROUND ===== */
    .stApp {
        background-color: #faf9f5;
    }
    .main .block-container {
        padding-top: 1rem;
    }

    /* ===== TYPOGRAPHY ===== */
    h1, .stMarkdown h1 {
        font-family: 'Cormorant Garamond', 'Tiempos Headline', Garamond, serif !important;
        font-weight: 400 !important;
        letter-spacing: -1px !important;
        color: #141413 !important;
    }
    h2, .stMarkdown h2 {
        font-family: 'Cormorant Garamond', 'Tiempos Headline', Garamond, serif !important;
        font-weight: 400 !important;
        letter-spacing: -0.5px !important;
        color: #141413 !important;
    }
    h3, .stMarkdown h3 {
        font-family: 'Cormorant Garamond', 'Tiempos Headline', Garamond, serif !important;
        font-weight: 400 !important;
        letter-spacing: -0.3px !important;
        color: #141413 !important;
    }
    p, span, div, label, .stMarkdown {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #3d3d3a;
    }

    /* ===== TOP NAVIGATION BAR (st.radio styled as nav pills) ===== */
    /* Radio group container */
    [data-testid="stHorizontalBlock"] div[role="radiogroup"] {
        display: flex;
        gap: 4px;
        padding: 0;
        background: #faf9f5;
        border-bottom: 1px solid #e6dfd8;
        padding-bottom: 12px;
        margin-bottom: 16px;
    }
    /* Individual radio labels (nav items) */
    div[role="radiogroup"] label {
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #6c6a64 !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        background: transparent !important;
        border: none !important;
        margin: 0 !important;
        transition: background 0.15s, color 0.15s;
    }
    div[role="radiogroup"] label:hover {
        background: #efe9de !important;
        color: #141413 !important;
    }
    /* Active/selected nav item */
    div[role="radiogroup"] label[data-selected="true"],
    div[role="radiogroup"] label[data-checked="true"],
    div[role="radiogroup"] label[aria-checked="true"] {
        background: #cc785c !important;
        color: #ffffff !important;
    }
    /* Hide the radio circle */
    div[role="radiogroup"] label input[type="radio"] {
        display: none !important;
    }
    /* Radio group container spacing fix */
    div[role="radiogroup"] {
        flex-wrap: wrap;
    }

    /* ===== PRIMARY BUTTONS (CORAL) ===== */
    .stButton > button[kind="primary"], button[kind="primary"] {
        background-color: #cc785c !important;
        border: none !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        border-radius: 8px !important;
        height: 40px !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #a9583e !important;
        border: none !important;
        color: #ffffff !important;
    }

    /* ===== SECONDARY BUTTONS ===== */
    .stButton > button[kind="secondary"], button[kind="secondary"] {
        background-color: #faf9f5 !important;
        border: 1px solid #e6dfd8 !important;
        color: #141413 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        border-radius: 8px !important;
        height: 40px !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #efe9de !important;
        border-color: #cc785c !important;
    }

    /* ===== CHAT MESSAGES ===== */
    [data-testid="stChatMessage"] {
        background-color: #efe9de !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-bottom: 8px !important;
    }

    /* ===== CHAT INPUT ===== */
    [data-testid="stChatInput"] {
        flex-direction: row !important;
        align-items: center !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #faf9f5 !important;
        border: 1px solid #e6dfd8 !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 16px !important;
        color: #141413 !important;
        flex: 1 !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #cc785c !important;
        box-shadow: 0 0 0 3px rgba(204, 120, 92, 0.15) !important;
    }
    [data-testid="stChatInput"] button {
        flex-shrink: 0 !important;
        margin-left: 4px !important;
    }

    /* ===== TEXT INPUTS ===== */
    input[type="text"], input[type="number"], .stTextInput input, .stNumberInput input {
        background-color: #faf9f5 !important;
        border: 1px solid #e6dfd8 !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        color: #141413 !important;
    }
    input:focus, .stTextInput input:focus {
        border-color: #cc785c !important;
        box-shadow: 0 0 0 3px rgba(204, 120, 92, 0.15) !important;
    }

    /* ===== SELECT BOXES ===== */
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #faf9f5 !important;
        border: 1px solid #e6dfd8 !important;
        border-radius: 8px !important;
    }

    /* ===== EXPANDERS ===== */
    [data-testid="stExpander"] {
        background-color: #efe9de !important;
        border: 1px solid #e6dfd8 !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: #141413 !important;
    }

    /* ===== PROGRESS BARS ===== */
    .stProgress > div > div {
        background-color: #cc785c !important;
    }
    .stProgress > div {
        background-color: #e6dfd8 !important;
        border-radius: 8px !important;
    }

    /* ===== FORMS ===== */
    [data-testid="stForm"] {
        background-color: #efe9de !important;
        border: 1px solid #e6dfd8 !important;
        border-radius: 12px !important;
        padding: 24px !important;
    }

    /* ===== DIVIDERS ===== */
    hr, .stDivider {
        border-color: #e6dfd8 !important;
    }

    /* ===== CODE BLOCKS ===== */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 14px !important;
        background-color: #181715 !important;
        color: #faf9f5 !important;
        border-radius: 12px !important;
        padding: 24px !important;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background-color: #efe9de;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-family: 'Cormorant Garamond', serif !important;
        font-weight: 400 !important;
        letter-spacing: -0.3px !important;
        color: #141413 !important;
    }

    /* ===== SIDEBAR CONTAINERS (cards) ===== */
    [data-testid="stSidebar"] .stContainer {
        background-color: #faf9f5 !important;
        border: 1px solid #e6dfd8 !important;
        border-radius: 12px !important;
        padding: 12px !important;
        margin-bottom: 8px !important;
    }

    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] section {
        background-color: #faf9f5 !important;
        border: 1px dashed #e6dfd8 !important;
        border-radius: 12px !important;
    }

    /* ===== INFO / SUCCESS / ERROR ===== */
    .stAlert {
        border-radius: 8px !important;
    }

    /* ===== CARD STYLE ===== */
    .agri-card {
        background: #ffffff;
        border: 1px solid #e6dfd8;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }

    /* ===== PAGE CONTENT ===== */
    .agri-page-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 2rem;
        font-weight: 400;
        letter-spacing: -0.5px;
        color: #141413;
        margin-bottom: 8px;
    }

    /* ===== RESPONSIVE: 手机端 (width < 768px) ===== */
    @media screen and (max-width: 767px) {
        h1 { font-size: 20px !important; }
        h2 { font-size: 17px !important; }
        h3 { font-size: 15px !important; }

        div[role="radiogroup"] {
            flex-wrap: nowrap !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch;
        }
        div[role="radiogroup"] label {
            font-size: 12px !important;
            padding: 6px 10px !important;
            white-space: nowrap;
            flex-shrink: 0;
        }

        .stButton > button { width: 100% !important; }
        .main .block-container { padding: 0.5rem !important; }

        [data-testid="stSidebar"] {
            min-width: 280px !important;
            max-width: 90vw !important;
        }

        [data-testid="stExpander"] { padding: 8px !important; }
        [data-testid="stForm"] { padding: 12px !important; }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_nav_bar():
    """Render the Anthropic-style top navigation bar using st.radio styled as nav pills.

    Uses a horizontal radio button group styled with CSS to match the
    design.md top-nav component. The brand label is rendered above.
    """
    # Initialize default page
    if "current_page" not in st.session_state:
        st.session_state.current_page = "dashboard"

    is_mobile = st.session_state.get("is_mobile", False)

    # Brand header row — 手机端由 test1.py 的标题替代，避免重复
    if not is_mobile:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
            '<span style="font-family:\'Cormorant Garamond\',serif;font-size:18px;'
            'font-weight:400;color:#141413;letter-spacing:-0.3px">'
            '🌾 智能种植规划助手</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    options = [f"{item['icon']} {item['label']}" for item in NAV_ITEMS]
    page_ids = [item["id"] for item in NAV_ITEMS]
    current_id = st.session_state.get("current_page", "chat")
    default_idx = page_ids.index(current_id) if current_id in page_ids else 0

    if is_mobile:
        st.html("<style>[data-baseweb='select'] input {pointer-events: none !important;}</style>")
        selected_label = st.selectbox(
            "导航", options, index=default_idx,
            label_visibility="collapsed", key="nav_select",
        )
    else:
        selected_label = st.radio(
            "导航", options, index=default_idx,
            horizontal=True, label_visibility="collapsed", key="nav_radio",
        )

    try:
        new_idx = options.index(selected_label)
        new_id = page_ids[new_idx]
    except (ValueError, IndexError):
        new_id = current_id

    if new_id != current_id:
        st.session_state.current_page = new_id
        st.rerun()
