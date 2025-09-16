# app.py
from __future__ import annotations
import streamlit as st
import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(__file__))

try:
    from utils.styling import load_css, create_metric_card, create_section_header
    from utils.theme import set_page_config, apply_theme
except ImportError:
    # Fallback if utils not available
    def load_css():
        pass
    def create_metric_card(title, value, subtitle="", color="#2a5298"):
        return f"<div><h4>{title}</h4><h2>{value}</h2><small>{subtitle}</small></div>"
    def create_section_header(title):
        return f"<h3>{title}</h3>"
    def set_page_config(title="VN Stock AI", icon="📈"):
        st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    def apply_theme():
        pass

# Set page config with optimized settings
set_page_config("VN Stock AI Platform", "📈")

# Apply theme and load CSS
apply_theme()
load_css()

# Load common CSS
load_css()

# Hero Section
st.markdown("""
<div class="main-header">
    <h1 style="color: white !important; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🚀 VN Stock AI Platform</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem; color: white !important; text-shadow: 0 1px 3px rgba(0,0,0,0.3);">
        Nền tảng phân tích và dự báo chứng khoán Việt Nam thông minh
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced styling
with st.sidebar:
    st.markdown(create_section_header("📊 Điều hướng nhanh"), unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-section">
        <p>Chọn công cụ phân tích phù hợp với nhu cầu của bạn</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(create_section_header("💡 Tính năng nổi bật"), unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-section">
        <ul>
            <li>Phân tích kỹ thuật chuyên sâu</li>
            <li>Mô hình AI dự báo giá</li>
            <li>Chấm điểm & xếp hạng cổ phiếu</li>
            <li>Giao diện thân thiện, dễ sử dụng</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main content with feature cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Stock Screener</h3>
        <p><strong>Công cụ sàng lọc & chấm điểm cổ phiếu</strong></p>
        <ul>
            <li>🎯 Chấm điểm đa tiêu chí (kỹ thuật + cơ bản)</li>
            <li>📈 Biểu đồ giá & khối lượng tương tác</li>
            <li>📋 Bảng lịch sử giá chi tiết</li>
            <li>🔍 Lọc theo ngành, vốn hóa</li>
            <li>📊 Phân tích so sánh</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🔮 LSTM Forecast</h3>
        <p><strong>Dự báo giá cổ phiếu bằng AI</strong></p>
        <ul>
            <li>🤖 Mô hình LSTM tiên tiến</li>
            <li>📈 Dự báo xu hướng giá</li>
            <li>🎛️ Tùy chỉnh tham số huấn luyện</li>
            <li>📊 Đánh giá độ chính xác</li>
            <li>💾 Lưu & tái sử dụng mô hình</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Quick navigation buttons
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🎯 Truy cập nhanh")

if hasattr(st, "switch_page"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("� Mở Stock Screener", key="screener_btn", use_container_width=True):
            st.switch_page("pages/1_Screener.py")
    with col2:
        if st.button("� Mở LSTM Forecast", key="forecast_btn", use_container_width=True):
            st.switch_page("pages/2_LSTM_Forecast.py")
else:
    st.info(
        "💡 **Hướng dẫn sử dụng:** Bạn có thể chuyển trang bằng menu điều hướng ở bên trái hoặc sử dụng các nút bên dưới."
    )

# Additional info section
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(create_metric_card("⚡ Hiệu suất", "Cao", "Xử lý dữ liệu nhanh chóng với cache thông minh"), unsafe_allow_html=True)

with col2:
    st.markdown(create_metric_card("🔒 Bảo mật", "Đảm bảo", "Dữ liệu được mã hóa và bảo vệ an toàn"), unsafe_allow_html=True)

with col3:
    st.markdown(create_metric_card("📱 Responsive", "Tối ưu", "Tối ưu cho mọi thiết bị và kích thước màn hình"), unsafe_allow_html=True)

# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🇻🇳 Phát triển cho thị trường chứng khoán Việt Nam</p>
    <p><small>Cập nhật: """ + "2025" + """ | Phiên bản 2.0</small></p>
</div>
""", unsafe_allow_html=True)
