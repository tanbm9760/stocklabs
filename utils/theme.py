"""
Enhanced styling injector for Streamlit app
This module ensures consistent styling across all pages
"""
import streamlit as st
import os

def inject_custom_css():
    """Inject custom CSS to override Streamlit defaults"""
    
    # Advanced CSS for professional appearance
    custom_css = """
    <style>
        /* ========== STREAMLIT OVERRIDES ========== */
        
        /* Hide Streamlit branding and menu */
        #MainMenu {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        header {visibility: hidden !important;}
        .stDeployButton {visibility: hidden !important;}
        
        /* Remove padding from main container */
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
        
        /* Sidebar enhancements */
        .css-1d391kg, .css-1cypcdb {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%) !important;
        }
        
        /* ========== CUSTOM COMPONENTS ========== */
        
        /* Professional buttons */
        .stButton > button {
            background: linear-gradient(45deg, #667eea, #764ba2) !important;
            color: white !important;
            border: none !important;
            border-radius: 25px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
            min-height: 2.5rem !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.5) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0) !important;
        }
        
        /* Input fields styling */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select,
        .stNumberInput > div > div > input,
        .stDateInput > div > div > input,
        .stTimeInput > div > div > input {
            border-radius: 8px !important;
            border: 2px solid #e9ecef !important;
            transition: all 0.3s ease !important;
            font-size: 14px !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div > select:focus,
        .stNumberInput > div > div > input:focus,
        .stDateInput > div > div > input:focus,
        .stTimeInput > div > div > input:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
            outline: none !important;
        }
        
        /* Slider styling */
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #667eea, #764ba2) !important;
        }
        
        .stSlider > div > div > div > div > div {
            background: white !important;
            border: 2px solid #667eea !important;
        }
        
        /* Selectbox dropdown */
        .stSelectbox > div > div > div {
            border-radius: 8px !important;
            border: 2px solid #e9ecef !important;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #667eea, #764ba2) !important;
            border-radius: 10px !important;
        }
        
        /* Success/Warning/Error messages */
        .stSuccess {
            background: #d4edda !important;
            border: 1px solid #c3e6cb !important;
            border-radius: 8px !important;
            padding: 1rem !important;
            border-left: 4px solid #28a745 !important;
        }
        
        .stWarning {
            background: #fff3cd !important;
            border: 1px solid #ffeaa7 !important;
            border-radius: 8px !important;
            padding: 1rem !important;
            border-left: 4px solid #ffc107 !important;
        }
        
        .stError {
            background: #f8d7da !important;
            border: 1px solid #f5c6cb !important;
            border-radius: 8px !important;
            padding: 1rem !important;
            border-left: 4px solid #dc3545 !important;
        }
        
        .stInfo {
            background: #d1ecf1 !important;
            border: 1px solid #bee5eb !important;
            border-radius: 8px !important;
            padding: 1rem !important;
            border-left: 4px solid #17a2b8 !important;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border-radius: 8px !important;
            overflow: hidden !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        }
        
        /* Chart containers */
        .js-plotly-plot {
            border-radius: 8px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
            overflow: hidden !important;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0 !important;
            background: #f8f9fa !important;
            border: 1px solid #e9ecef !important;
            padding: 0.5rem 1rem !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(45deg, #667eea, #764ba2) !important;
            color: white !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: #f8f9fa !important;
            border-radius: 8px !important;
            border: 1px solid #e9ecef !important;
            transition: all 0.3s ease !important;
        }
        
        .streamlit-expanderHeader:hover {
            background: #e9ecef !important;
        }
        
        /* Metric styling */
        [data-testid="metric-container"] {
            background: white !important;
            border: 1px solid #e9ecef !important;
            padding: 1rem !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
            border-left: 4px solid #667eea !important;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px !important;
            height: 8px !important;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1 !important;
            border-radius: 4px !important;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #667eea, #764ba2) !important;
            border-radius: 4px !important;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #5a6fd8 !important;
        }
        
        /* Checkbox and radio styling */
        .stCheckbox > label > div {
            border-radius: 4px !important;
            border: 2px solid #e9ecef !important;
        }
        
        .stCheckbox > label > div[data-checked="true"] {
            background: #667eea !important;
            border-color: #667eea !important;
        }
        
        .stRadio > label > div {
            border-radius: 50% !important;
            border: 2px solid #e9ecef !important;
        }
        
        .stRadio > label > div[data-checked="true"] {
            background: #667eea !important;
            border-color: #667eea !important;
        }
        
        /* File uploader styling */
        .stFileUploader > div > div {
            border-radius: 8px !important;
            border: 2px dashed #e9ecef !important;
            transition: all 0.3s ease !important;
        }
        
        .stFileUploader > div > div:hover {
            border-color: #667eea !important;
            background: #f8f9fa !important;
        }
        
        /* Loading spinner */
        .stSpinner > div {
            border-top-color: #667eea !important;
            border-left-color: #667eea !important;
        }
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .main .block-container {
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
            
            .stButton > button {
                width: 100% !important;
                margin-bottom: 0.5rem !important;
            }
        }
        
        /* Custom animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease-out !important;
        }
        
        /* Typography improvements */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #2c3e50 !important;
        }
        
        .stMarkdown p {
            color: #495057 !important;
            line-height: 1.6 !important;
        }
        
        /* Fix text color in gradient headers */
        .main-header h1, .main-header p,
        .main-title h1, .main-title p,
        .forecast-header h1, .forecast-header p,
        .section-header h3, .section-header h4 {
            color: white !important;
            text-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
        }
        
        /* Ensure good contrast for all gradient backgrounds */
        .main-header *, .main-title *, .forecast-header *, .section-header * {
            color: white !important;
        }
        
        /* Text selection */
        ::selection {
            background: #667eea !important;
            color: white !important;
        }
    </style>
    """
    
    st.markdown(custom_css, unsafe_allow_html=True)

def set_page_config(title="VN Stock AI", icon="📈"):
    """Set optimized page configuration"""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': None
        }
    )

def apply_theme():
    """Apply complete theme including CSS injection"""
    inject_custom_css()