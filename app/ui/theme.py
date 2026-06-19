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
    {"id": "policy", "label": "政策补贴", "icon": "📜"},
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

    /* ===== HORIZONTAL RADIO PILLS（仅在 st.columns 等水平容器中使用）===== */
    /* Radio group container — 限定在 stHorizontalBlock 内 */
    [data-testid="stHorizontalBlock"] div[role="radiogroup"] {
        display: flex;
        gap: 4px;
        padding: 0;
        background: #faf9f5;
        border-bottom: 1px solid #e6dfd8;
        padding-bottom: 12px;
        margin-bottom: 16px;
        flex-wrap: wrap;
    }
    /* Individual radio labels — 限定在 stHorizontalBlock 内，避免误伤右侧导航 */
    [data-testid="stHorizontalBlock"] div[role="radiogroup"] label {
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
    [data-testid="stHorizontalBlock"] div[role="radiogroup"] label:hover {
        background: #efe9de !important;
        color: #141413 !important;
    }
    /* Active/selected — 限定在 stHorizontalBlock 内 */
    [data-testid="stHorizontalBlock"] div[role="radiogroup"] label[data-selected="true"],
    [data-testid="stHorizontalBlock"] div[role="radiogroup"] label[data-checked="true"],
    [data-testid="stHorizontalBlock"] div[role="radiogroup"] label[aria-checked="true"] {
        background: #cc785c !important;
        color: #ffffff !important;
    }
    /* Hide the radio circle — 全局生效（不影响布局，只隐藏圆点） */
    div[role="radiogroup"] label input[type="radio"] {
        display: none !important;
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
    """右侧可折叠纵向导航栏：默认仅显示图标，鼠标悬停展开图标+文字

    桌面端：fixed 定位在页面右侧，52px 宽（仅图标）→ 悬停 160px 宽（图标+文字）
    手机端：沿用下拉 selectbox

    核心思路：
    - 容器 overflow:hidden 裁剪掉文字部分，只露出 emoji
    - emoji 通过 flex-start + 精确 padding 在 52px 容器中居中
    - 悬停时容器展宽 + 字号缩小，完整文字自然显示
    - 右侧加一层渐变遮罩，让裁剪边缘更柔和
    """
    if "current_page" not in st.session_state:
        st.session_state.current_page = "dashboard"

    is_mobile = st.session_state.get("is_mobile", False)
    current_id = st.session_state.get("current_page", "dashboard")
    options = [f"{item['icon']} {item['label']}" for item in NAV_ITEMS]
    page_ids = [item["id"] for item in NAV_ITEMS]
    default_idx = page_ids.index(current_id) if current_id in page_ids else 0

    # ── 手机端：下拉导航 ──────────────────────────
    if is_mobile:
        st.html("<style>[data-baseweb='select'] input {pointer-events: none !important;}</style>")
        selected_label = st.selectbox(
            "导航", options, index=default_idx,
            label_visibility="collapsed", key="nav_select",
        )
        try:
            new_idx = options.index(selected_label)
            new_id = page_ids[new_idx]
        except (ValueError, IndexError):
            new_id = current_id
        if new_id != current_id:
            st.session_state.current_page = new_id
            st.rerun()
        return

    # ── 桌面端：右侧可折叠导航（纯 st.radio + CSS，不依赖 JS）──

    st.markdown("""<style>
/* ============================================
   右侧可折叠导航栏
   折叠态：52px 宽，仅显示 emoji 图标
   悬停态：160px 宽，显示图标 + 中文标签

   定位策略：用 :has() 匹配包含 10+ 个 label 的 radio group
   （导航有 12 项，其他 radio 最多 3 项），完全不依赖 JS。
   同时兼容 JS 添加的 .right-nav-radio class。
   ============================================ */

/* 目标容器：导航 radio group（12个选项） */
div[role="radiogroup"]:has(> label:nth-child(10)),
.right-nav-radio {
    position: fixed !important;
    right: 0 !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    z-index: 9999 !important;

    /* 视觉风格 */
    background: #efe9de !important;
    border: 1px solid #e6dfd8 !important;
    border-right: none !important;
    border-radius: 12px 0 0 12px !important;
    box-shadow: -2px 0 20px rgba(20,20,19,0.10) !important;

    /* 布局 */
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
    padding: 10px 5px !important;

    /* 折叠态尺寸 + 裁剪 */
    width: 52px !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;

    /* 平滑过渡 */
    transition: width 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;

    /* 防止过长时溢出屏幕 */
    max-height: 82vh !important;
}

/* 悬停展开 */
div[role="radiogroup"]:has(> label:nth-child(10)):hover,
.right-nav-radio:hover {
    width: 160px !important;
}

/* ── 每个导航项（label） ── */
div[role="radiogroup"]:has(> label:nth-child(10)) label,
.right-nav-radio label {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;

    /* 内边距：上下紧凑，左侧精确控制 emoji 在 52px 容器中的居中 */
    padding: 8px 8px 8px 9px !important;
    border-radius: 8px !important;
    cursor: pointer !important;

    /* 折叠态：大号 emoji（24px），文字自然溢出被父容器 overflow:hidden 裁剪 */
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 24px !important;
    font-weight: 500 !important;
    color: #6c6a64 !important;
    white-space: nowrap !important;
    min-height: 42px !important;
    margin: 0 !important;
    background: transparent !important;
    border: none !important;
    overflow: hidden !important;

    /* 过渡 */
    transition:
        background 0.18s ease,
        color 0.18s ease,
        font-size 0.25s ease,
        padding 0.25s ease !important;
}

/* 悬停展开后：缩小字号、文字完全可见 */
div[role="radiogroup"]:has(> label:nth-child(10)):hover label,
.right-nav-radio:hover label {
    font-size: 13.5px !important;
    padding: 8px 12px !important;
}

/* 单项 hover 高亮 */
div[role="radiogroup"]:has(> label:nth-child(10)) label:hover,
.right-nav-radio label:hover {
    background: #cc785c !important;
    color: #ffffff !important;
}

/* 隐藏 Streamlit radio 原生圆点 */
div[role="radiogroup"]:has(> label:nth-child(10)) label > div:first-child,
.right-nav-radio label > div:first-child {
    display: none !important;
}

/* 当前选中项 — 浅棕色背景，一眼辨识
   兼容多种 Streamlit 版本的 checked 标记方式：
   - data-checked / aria-checked（新版 Streamlit）
   - input:checked（通用，最可靠）
*/
div[role="radiogroup"]:has(> label:nth-child(10)) label[data-checked="true"],
div[role="radiogroup"]:has(> label:nth-child(10)) label[aria-checked="true"],
div[role="radiogroup"]:has(> label:nth-child(10)) label:has(input:checked),
.right-nav-radio label[data-checked="true"],
.right-nav-radio label[aria-checked="true"],
.right-nav-radio label:has(input:checked) {
    background: #c8a27a !important;
    color: #141413 !important;
}

/* ── 右侧渐变遮罩（折叠态让文字裁剪边缘更柔和） ── */
div[role="radiogroup"]:has(> label:nth-child(10))::after,
.right-nav-radio::after {
    content: "" !important;
    position: absolute !important;
    top: 0 !important;
    right: 0 !important;
    width: 18px !important;
    height: 100% !important;
    pointer-events: none !important;
    background: linear-gradient(to right, transparent, #efe9de 85%) !important;
    border-radius: 0 0 0 0 !important;
    transition: opacity 0.25s ease !important;
    opacity: 1 !important;
}
div[role="radiogroup"]:has(> label:nth-child(10)):hover::after,
.right-nav-radio:hover::after {
    opacity: 0 !important;
}

/* ── 主内容区右侧留白，防止被导航遮挡 ── */
.main .block-container {
    padding-right: 68px !important;
}

/* ── 手机端：隐藏右侧导航，恢复正常留白 ── */
@media screen and (max-width: 767px) {
    div[role="radiogroup"]:has(> label:nth-child(10)),
    .right-nav-radio {
        display: none !important;
    }
    .main .block-container {
        padding-right: 1rem !important;
    }
}
</style>""", unsafe_allow_html=True)

    # JS: 辅助给导航 radio group 加上 class（用于不支持 :has() 的旧浏览器降级）
    st.html("""<script>
(function() {
    if (typeof MutationObserver === 'undefined') return;
    var tries = 0, maxTries = 10;
    function findAndTag() {
        var groups = document.querySelectorAll('div[role="radiogroup"]');
        var maxCount = 0, navGroup = null;
        groups.forEach(function(rg) {
            var n = rg.querySelectorAll('label').length;
            if (n > maxCount) { maxCount = n; navGroup = rg; }
        });
        if (navGroup && maxCount >= 10) {
            navGroup.classList.add('right-nav-radio');
            navGroup.setAttribute('data-nav', 'right');
            return true;
        }
        return false;
    }
    if (!findAndTag() && tries < maxTries) {
        var obs = new MutationObserver(function() {
            tries++;
            if (findAndTag()) obs.disconnect();
            else if (tries >= maxTries) obs.disconnect();
        });
        obs.observe(document.body, { childList: true, subtree: true });
    }
})();
</script>""")

    selected_label = st.radio(
        "导航", options, index=default_idx,
        label_visibility="collapsed", key="right_nav_radio",
    )

    try:
        new_idx = options.index(selected_label)
        new_id = page_ids[new_idx]
    except (ValueError, IndexError):
        new_id = current_id
    if new_id != current_id:
        st.session_state.current_page = new_id
        st.rerun()
