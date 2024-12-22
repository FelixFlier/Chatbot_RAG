import streamlit as st

def apply_enhanced_custom_css():
    current_theme = st.session_state.get("theme", "light")
    
    colors = {
        "light": {
            "bg_primary": "#e0f2f7",
            "bg_secondary": "#b3e5ee",
            "text_primary": "#1a1a1a",
            "text_secondary": "#1a1a1a",
            "border": "rgba(0, 0, 0, 0.1)",
            "shadow": "rgba(0, 0, 0, 0.1)",
            "input_bg": "#b3e5ee",
            "input_border": "#e0e0e0",
            "title_gradient": "linear-gradient(120deg, #06b6d4, #0891b2, #06b6d4)",
            "typical_question_bg": "#b3e5ee",
            "typical_question_text": "#1a1a1a",
            "header_bg": "#b3e5ee",
            "footer_bg": "#b3e5ee",
            "button_bg": "#b3e5ee",
            "button_text": "#1a1a1a",
            "input_text": "#000000",
            "input_placeholder": "rgba(0, 0, 0, 0.6)",
            "chat_message_bg": "rgba(179, 229, 238, 0.7)",
            "button_hover_bg": "#0891b2",
            "button_hover_text": "#FFFFFF"
        },
        "dark": {
            "bg_primary": "#04768a",
            "bg_secondary": "#035d6d",
            "text_primary": "#ffffff",
            "text_secondary": "#ffffff",
            "border": "rgba(255, 255, 255, 0.2)",
            "shadow": "rgba(0, 0, 0, 0.3)",
            "input_bg": "#035d6d",
            "input_border": "#04768a",
            "title_gradient": "linear-gradient(120deg, #00ffff, #00ccff, #00ffff)",
            "typical_question_bg": "#024857",
            "typical_question_text": "#ffffff",
            "header_bg": "#035d6d",
            "footer_bg": "#035d6d",
            "button_bg": "#024857",
            "button_text": "#ffffff",
            "input_text": "#ffffff",
            "input_placeholder": "rgba(255, 255, 255, 0.6)",
            "chat_message_bg": "rgba(3, 93, 109, 0.7)",
            "button_hover_bg": "#035d6d",
            "button_hover_text": "#ffffff"
        }
    }
    
    theme_colors = colors[current_theme]
    
    css = f"""
    <style>
    /* Basis-Styles */
    h1 {{
        background: {theme_colors["title_gradient"]};
        background-size: 200% auto;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
        font-weight: bold;
        margin-bottom: 2rem;
        filter: brightness(1.2);
        text-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }}

    @keyframes gradient {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    .stApp {{
        background-color: {theme_colors["bg_primary"]};
        transition: background-color 0.5s ease-in-out;
    }}

    /* Button Styles - Überarbeitete Version */
    .stButton > button {{
        background-color: {theme_colors["button_bg"]} !important;
        color: {theme_colors["button_text"]} !important;
        border: 1px solid {theme_colors["border"]} !important;
        transition: all 0.3s ease !important;
    }}

    [data-theme="dark"] .stButton > button {{
        background-color: rgba(2, 72, 87, 0.8) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }}

    [data-theme="dark"] .stButton > button:hover {{
        background-color: rgba(3, 93, 109, 0.9) !important;
        color: white !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}

    [data-theme="dark"] .stButton > button:hover * {{
        color: white !important;
    }}

    /* Typische Fragen Buttons im Dark Mode */
    [data-theme="dark"] div[data-testid="stHorizontalBlock"] .stButton > button {{
        background-color: rgba(2, 72, 87, 0.8) !important;
    }}

    [data-theme="dark"] div[data-testid="stHorizontalBlock"] .stButton > button:hover {{
        background-color: rgba(3, 93, 109, 0.9) !important;
    }}

    /* Feedback und Copy Buttons im Dark Mode */
    [data-theme="dark"] div[data-testid="column"] .stButton > button {{
        background-color: rgba(2, 72, 87, 0.8) !important;
    }}

    [data-theme="dark"] div[data-testid="column"] .stButton > button:hover {{
        background-color: rgba(3, 93, 109, 0.9) !important;
    }}

    /* Chat Messages & Text */
    .stChatMessage,
    .stChatMessage p,
    .stMarkdown p {{
        color: {theme_colors["text_primary"]} !important;
        transition: color 0.3s ease-in-out;
    }}

    div.stChatMessage {{
        background-color: {theme_colors["chat_message_bg"]} !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 2px 4px {theme_colors["shadow"]} !important;
    }}

    .stChatMessage ul li::marker,
    .stChatMessage ul li {{
        color: {theme_colors["text_primary"]} !important;
    }}

    /* Input Field Styling */
    .stChatInput {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100vw !important;
        padding: 1rem;
        background-color: {theme_colors["bg_secondary"]};
        border-top: 1px solid {theme_colors["border"]};
        box-shadow: 0 -2px 10px {theme_colors["shadow"]};
        z-index: 1000;
    }}

    .stChatInput > div {{
        width: 65% !important;
        margin-left: 35.5% !important;
        margin-right: 0% !important;
        background-color: transparent !important;
    }}

    .stChatInput input,
    .stChatInput textarea {{
        background-color: {theme_colors["input_bg"]} !important;
        color: {theme_colors["input_text"]} !important;
        border: 1px solid {theme_colors["input_border"]} !important;
        border-radius: 20px !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        caret-color: {theme_colors["input_text"]} !important;
    }}

    .stChatInput input:focus {{
        color: {theme_colors["input_text"]} !important;
        border-color: {theme_colors["button_hover_bg"]} !important;
        box-shadow: 0 2px 8px rgba(0, 204, 255, 0.2) !important;
        outline: none !important;
    }}

    /* Details/Expander Styling */
    [data-theme="dark"] .streamlit-expanderHeader {{
        border: 1.5px solid rgba(255, 255, 255, 0.3) !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 5px !important;
        margin: 0.5rem 0 !important;
        padding: 0.5rem !important;
    }}

    [data-theme="dark"] .streamlit-expanderContent {{
        border: 1.5px solid rgba(255, 255, 255, 0.2) !important;
        border-top: none !important;
        background-color: rgba(255, 255, 255, 0.02) !important;
        border-radius: 0 0 5px 5px !important;
        padding: 0.5rem !important;
    }}

    [data-theme="dark"] .streamlit-expanderHeader div,
    [data-theme="dark"] .streamlit-expanderContent div {{
        color: {theme_colors["text_primary"]} !important;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] > div {{
        background-color: {theme_colors["bg_secondary"]} !important;
        border-right: 1.5px solid {theme_colors["border"]};
    }}

    section[data-testid="stSidebar"] * {{
        color: {theme_colors["text_secondary"]} !important;
    }}

    [data-theme="dark"] .stSidebar hr {{
        border: none !important;
        height: 2px !important;
        background: linear-gradient(
            to right,
            rgba(255, 255, 255, 0.1),
            rgba(255, 255, 255, 0.3),
            rgba(255, 255, 255, 0.1)
        ) !important;
        margin: 20px 0 !important;
    }}

    .stChatInput input::placeholder {{
        color: {theme_colors["input_placeholder"]} !important;
        opacity: 1 !important;
    }}
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)
