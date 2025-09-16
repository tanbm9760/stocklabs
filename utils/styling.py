"""
Common styling utilities for VN Stock AI Platform
"""
import streamlit as st
import os

def load_css():
    """Load common CSS styles for the application"""
    css_path = os.path.join(os.path.dirname(__file__), "..", "styles", "common.css")
    streamlit_css_path = os.path.join(os.path.dirname(__file__), "..", ".streamlit", "style.css")
    
    # Enhanced fallback CSS with Streamlit customizations
    fallback_css = """
    <style>
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --success-color: #28a745;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
            --light-color: #f8f9fa;
            --border-radius: 8px;
            --box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            --transition: all 0.3s ease;
        }
        
        .main-header {
            background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white !important;
            text-align: center;
            box-shadow: var(--box-shadow);
        }
        
        .main-header h1, .main-header p {
            color: white !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
            margin: 0 !important;
        }
        
        .feature-card, .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: var(--box-shadow);
            border-left: 4px solid var(--primary-color);
            margin-bottom: 1rem;
            transition: var(--transition);
        }
        
        .feature-card:hover, .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        
        .section-header {
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
            color: white !important;
            padding: 1rem;
            border-radius: var(--border-radius) var(--border-radius) 0 0;
            margin-bottom: 0;
            font-weight: 600;
        }
        
        .section-header h1, .section-header h2, .section-header h3, .section-header h4, .section-header h5 {
            color: white !important;
            text-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
            margin: 0 !important;
        }
        
        /* Override any inherited colors on gradient backgrounds */
        .main-header *, .section-header *, .forecast-header * {
            color: white !important;
        }
        
        .control-panel {
            background: var(--light-color);
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #e9ecef;
            margin-bottom: 1rem;
        }
        
        .metric-box {
            background: white;
            padding: 1rem;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            text-align: center;
            border-left: 4px solid var(--primary-color);
            transition: var(--transition);
        }
        
        /* Enhanced Streamlit button styling */
        .stButton > button {
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color)) !important;
            color: white !important;
            border: none !important;
            border-radius: 25px !important;
            font-weight: 600 !important;
            transition: var(--transition) !important;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.5) !important;
        }
        
        /* Input field enhancements */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select,
        .stNumberInput > div > div > input {
            border-radius: 8px !important;
            border: 2px solid #e9ecef !important;
            transition: border-color 0.3s ease !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div > select:focus,
        .stNumberInput > div > div > input:focus {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)) !important;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--light-color);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--primary-color);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--secondary-color);
        }
    </style>
    """
    
    try:
        css_content = ""
        
        # Load common CSS
        if os.path.exists(css_path):
            with open(css_path, 'r', encoding='utf-8') as f:
                css_content += f.read()
        
        # Load Streamlit-specific CSS
        if os.path.exists(streamlit_css_path):
            with open(streamlit_css_path, 'r', encoding='utf-8') as f:
                css_content += f.read()
        
        if css_content:
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        else:
            st.markdown(fallback_css, unsafe_allow_html=True)
            
    except Exception:
        # If there's any error, use the fallback CSS
        st.markdown(fallback_css, unsafe_allow_html=True)

def create_metric_card(title: str, value: str, subtitle: str = "", color: str = "#2a5298"):
    """Create a styled metric card"""
    return f"""
    <div class="metric-card">
        <h4>{title}</h4>
        <h2 style="color: {color};">{value}</h2>
        <small>{subtitle}</small>
    </div>
    """

def create_section_header(title: str):
    """Create a styled section header"""
    return f"""
    <div class="section-header">
        <h3>{title}</h3>
    </div>
    """

def create_status_message(message: str, status: str = "info"):
    """Create a styled status message"""
    status_classes = {
        "success": "status-success",
        "warning": "status-training", 
        "error": "status-error",
        "info": "status-success"
    }
    
    icons = {
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "info": "ℹ️"
    }
    
    status_class = status_classes.get(status, "status-success")
    icon = icons.get(status, "ℹ️")
    
    return f"""
    <div class="model-status {status_class}">
        {icon} <strong>{message}</strong>
    </div>
    """