# pages/2_LSTM_Forecast.py
from __future__ import annotations
from datetime import datetime, timedelta
from typing import List
import sys
import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pick_best_by_symbols import VnAdapter

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.theme import set_page_config, apply_theme
    from utils.styling import load_css, create_section_header, create_metric_card, create_status_message
except ImportError:
    def set_page_config(title, icon):
        st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    def apply_theme():
        pass
    def load_css():
        pass
    def create_section_header(title):
        return f"<h3>{title}</h3>"
    def create_metric_card(title, value, subtitle="", color="#2a5298"):
        return f"<div><h4>{title}</h4><h2>{value}</h2><small>{subtitle}</small></div>"
    def create_status_message(message, status="info"):
        return f"<div>{message}</div>"

# ----- TF/Keras -----
try:
    import tensorflow as tf
    from tensorflow.keras import Input
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dropout, Dense
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
except Exception:
    tf = None

# Page config with enhanced styling
set_page_config("🔮 LSTM Forecast", "🔮")
apply_theme()
load_css()

st.markdown("""
<div class="forecast-header">
    <h1 style="color: white !important; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🔮 LSTM Forecast</h1>
    <p style="color: white !important; text-shadow: 0 1px 3px rgba(0,0,0,0.3);">Dự báo giá cổ phiếu bằng mô hình Deep Learning LSTM</p>
</div>
""", unsafe_allow_html=True)

# --------- helpers ----------
def _prev_weekday(dt: datetime) -> datetime:
    while dt.weekday() >= 5: dt -= timedelta(days=1)
    return dt

@st.cache_data(show_spinner=False, ttl=60*30)
def _load_close_series_cached(symbol: str, days: int, end_date: str, sources: List[str]) -> pd.Series:
    adapter = VnAdapter(preferred_sources=sources, end_date=end_date, verbose=False)
    px = adapter.get_quote_history(symbol, days=days)
    if px.empty or "close" not in px.columns:
        return pd.Series(dtype=float)
    s = pd.Series(px["close"].values, index=pd.to_datetime(px["date"]))
    return s.sort_index()

def _make_sequences_1d(arr_scaled: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(arr_scaled)):
        X.append(arr_scaled[i-window:i, 0])
        y.append(arr_scaled[i, 0])
    return np.array(X).reshape(-1, window, 1), np.array(y).reshape(-1, 1)

def build_model(window: int):
    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(128, return_sequences=True),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def _multistep_forecast_scaled(model, scaler: MinMaxScaler, vals: np.ndarray, window: int, horizon: int) -> np.ndarray:
    last_window_vals = vals[-window:]
    last_window_sc   = scaler.transform(last_window_vals)
    x = last_window_sc.reshape(1, window, 1)
    preds_sc = []
    for _ in range(int(horizon)):
        p_sc = model.predict(x, verbose=0)[0, 0]
        preds_sc.append(p_sc)
        x = np.concatenate([x[:, 1:, :], np.array(p_sc).reshape(1, 1, 1)], axis=1)
    preds_sc = np.array(preds_sc).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_sc).ravel()
    return preds

def lstm_forecast_from_close_series(
    close_series: pd.Series,
    *,
    window: int = 50,
    epochs: int = 100,
    batch_size: int = 50,
    train_ratio: float = 0.8,
    horizon: int = 1,
    checkpoint_path: str = "best_model.keras",
    use_earlystop: bool = True
):
    s = close_series.dropna().astype(float).copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    vals = s.values.reshape(-1, 1)
    n = len(vals)

    n_train = max(int(n * train_ratio), window + 1)
    train_vals = vals[:n_train]
    test_vals  = vals[n_train - window:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_vals)
    train_scaled = scaler.transform(train_vals)
    test_scaled  = scaler.transform(test_vals)

    X_train, y_train = _make_sequences_1d(train_scaled, window)
    X_test,  y_test  = _make_sequences_1d(test_scaled,  window)

    model = build_model(window)
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor="loss", save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=0)
    ]
    if use_earlystop:
        callbacks.insert(1, EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0))

    model.fit(
        X_train, y_train,
        epochs=int(epochs),
        batch_size=int(batch_size),
        verbose=0,
        validation_split=0.2,
        callbacks=callbacks
    )

    try:
        model = load_model(checkpoint_path)
    except Exception:
        pass

    y_train_pred = model.predict(X_train, verbose=0)
    y_test_pred  = model.predict(X_test,  verbose=0)

    y_train_inv       = scaler.inverse_transform(y_train)
    y_test_inv        = scaler.inverse_transform(y_test)
    y_train_pred_inv  = scaler.inverse_transform(y_train_pred)
    y_test_pred_inv   = scaler.inverse_transform(y_test_pred)

    train_dates = s.index[window:n_train]
    test_dates  = s.index[n_train:]

    train_pred_df = pd.DataFrame({"date": train_dates, "y_true": y_train_inv.ravel(), "y_pred": y_train_pred_inv.ravel()})
    test_pred_df  = pd.DataFrame({"date": test_dates, "y_true": y_test_inv.ravel(), "y_pred": y_test_pred_inv.ravel()})

    metrics = {
        "train": {
            "R2":   float(r2_score(train_pred_df["y_true"], train_pred_df["y_pred"])) if len(train_pred_df) > 1 else np.nan,
            "MAE":  float(mean_absolute_error(train_pred_df["y_true"], train_pred_df["y_pred"])) if len(train_pred_df) > 0 else np.nan,
            "MAPE": float(mean_absolute_percentage_error(train_pred_df["y_true"], train_pred_df["y_pred"])) if len(train_pred_df) > 0 else np.nan,
        },
        "test": {
            "R2":   float(r2_score(test_pred_df["y_true"], test_pred_df["y_pred"])) if len(test_pred_df) > 1 else np.nan,
            "MAE":  float(mean_absolute_error(test_pred_df["y_true"], test_pred_df["y_pred"])) if len(test_pred_df) > 0 else np.nan,
            "MAPE": float(mean_absolute_percentage_error(test_pred_df["y_true"], test_pred_df["y_pred"])) if len(test_pred_df) > 0 else np.nan,
        }
    }

    fut_vals = _multistep_forecast_scaled(model, scaler, vals, window, int(horizon))
    fut_dates = pd.bdate_range(s.index[-1] + pd.Timedelta(days=1), periods=int(horizon))
    future_df = pd.DataFrame({"date": fut_dates, "pred_raw": fut_vals})

    next_pred = {
        "next_date": fut_dates[0],
        "pred": float(fut_vals[0]),
        "last_close": float(s.iloc[-1])
    }
    return {
        "train_pred_df": train_pred_df, "test_pred_df": test_pred_df,
        "metrics": metrics, "next_pred": next_pred, "future_df": future_df
    }

# ==========================================
# Enhanced Sidebar with Professional Styling
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="section-header">
        <h3>⚙️ Cấu hình LSTM Model</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Stock Selection
    st.markdown('<div class="parameter-card">', unsafe_allow_html=True)
    st.markdown("**📈 Lựa chọn cổ phiếu**")
    symbol = st.text_input(
        "Mã cổ phiếu", 
        "VIX", 
        help="Nhập mã cổ phiếu Việt Nam (VD: VNM, FPT, VIX...)",
        placeholder="VD: VIX"
    ).upper()
    
    col1, col2 = st.columns(2)
    with col1:
        days = st.number_input(
            "📅 Số ngày lịch sử", 
            200, 2000, 800, 10, 
            help="Số ngày dữ liệu để huấn luyện mô hình"
        )
    with col2:
        tminus = st.number_input(
            "⏰ Lùi ngày T-n", 
            0, 30, 0, 1, 
            help="0 = đến ngày gần nhất"
        )
    
    sources_str = st.text_input(
        "🔌 Nguồn dữ liệu", 
        value="TCBS, VCI, MSN", 
        help="Thứ tự ưu tiên nguồn lấy dữ liệu"
    )
    
    end_date = st.text_input(
        "📅 Ngày kết thúc", 
        value="", 
        placeholder="YYYY-MM-DD hoặc để trống",
        help="Để trống để sử dụng ngày hôm nay"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Model Parameters
    st.markdown('<div class="parameter-card">', unsafe_allow_html=True)
    st.markdown("**🧠 Tham số mô hình**")
    
    window = st.slider(
        "🔍 Lookback window", 
        20, 120, 50, 5, 
        help="Số ngày dữ liệu đưa vào LSTM để dự báo"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider(
            "🔄 Epochs", 
            10, 300, 120, 10, 
            help="Số vòng lặp huấn luyện"
        )
    with col2:
        batch_size = st.slider(
            "📦 Batch size", 
            16, 256, 64, 16, 
            help="Kích thước lô dữ liệu"
        )
    
    train_ratio = st.slider(
        "📊 Tỷ lệ train (%)", 
        60, 95, 80, 1, 
        help="Tỷ lệ dữ liệu dành cho huấn luyện"
    ).__trunc__()/100.0
    
    horizon = st.slider(
        "🔮 Số phiên dự báo", 
        1, 20, 5, 1, 
        help="Số phiên giao dịch cần dự báo"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Training Status
    st.markdown('<div class="parameter-card">', unsafe_allow_html=True)
    st.markdown("**📊 Thông tin huấn luyện**")
    
    total_samples = max(0, days - window) if days > window else 0
    train_samples = int(total_samples * train_ratio)
    test_samples = total_samples - train_samples
    
    st.markdown(f"""
    <div class="metric-box">
        <strong>Tổng mẫu:</strong> {total_samples}<br>
        <strong>Train:</strong> {train_samples} | <strong>Test:</strong> {test_samples}
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button(
        "🚀 Bắt đầu huấn luyện & dự báo", 
        width="stretch",
        type="primary",
        help="Nhấn để bắt đầu quá trình huấn luyện mô hình LSTM"
    )

if tf is None:
    st.markdown("""
    <div class="model-status status-error">
        ❌ <strong>Thiếu thư viện TensorFlow!</strong><br>
        Vui lòng cài đặt: <code>pip install tensorflow scikit-learn</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ==========================================
# Main Processing Logic
# ==========================================
if tminus and tminus > 0:
    ed = _prev_weekday(datetime.today() - timedelta(days=int(tminus)))
else:
    ed = _prev_weekday(datetime.today())
    if end_date.strip():
        try:
            ed = _prev_weekday(pd.to_datetime(end_date))
        except Exception:
            st.warning("⚠️ Ngày kết thúc không hợp lệ, sử dụng ngày làm việc gần nhất.")

ed_str = ed.strftime("%Y-%m-%d")
sources = [s.strip().upper() for s in sources_str.split(",") if s.strip()]

# Display current configuration
if not run_btn:
    st.markdown("""
    <div class="forecast-section">
        <h4>📋 Cấu hình hiện tại</h4>
        <div class="metric-grid">
            <div class="metric-box">
                <h5>📈 Cổ phiếu</h5>
                <strong>{symbol}</strong>
            </div>
            <div class="metric-box">
                <h5>📅 Dữ liệu</h5>
                <strong>{days} ngày</strong><br>
                <small>đến {ed_str}</small>
            </div>
            <div class="metric-box">
                <h5>🧠 Window</h5>
                <strong>{window} ngày</strong>
            </div>
            <div class="metric-box">
                <h5>🔮 Dự báo</h5>
                <strong>{horizon} phiên</strong>
            </div>
        </div>
        <div style="text-align: center; margin-top: 1rem;">
            <p>👆 Điều chỉnh các tham số trong sidebar và nhấn <strong>🚀 Bắt đầu huấn luyện</strong> để bắt đầu</p>
        </div>
    </div>
    """.format(symbol=symbol, days=days, ed_str=ed_str, window=window, horizon=horizon), unsafe_allow_html=True)

if run_btn:
    # Progress tracking
    progress_container = st.container()
    
    with progress_container:
        st.markdown("""
        <div class="model-status status-training">
            ⏳ <strong>Đang xử lý...</strong> Vui lòng đợi trong khi mô hình được huấn luyện
        </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load data
            status_text.text("📥 Đang tải dữ liệu...")
            progress_bar.progress(20)
            
            s_close = _load_close_series_cached(symbol, int(days), ed_str, sources)
            if s_close.empty:
                st.error("❌ Không thể lấy dữ liệu giá cho mã " + symbol)
                st.stop()

            # Step 2: Prepare data
            status_text.text("🔄 Đang chuẩn bị dữ liệu...")
            progress_bar.progress(40)

            # Step 3: Train model
            status_text.text("🤖 Đang huấn luyện mô hình LSTM...")
            progress_bar.progress(60)

            res = lstm_forecast_from_close_series(
                s_close,
                window=int(window), epochs=int(epochs), batch_size=int(batch_size),
                train_ratio=float(train_ratio), horizon=int(horizon),
                checkpoint_path=f"{symbol}_best.keras", use_earlystop=True
            )

            # Step 4: Generate results
            status_text.text("📊 Đang tạo kết quả...")
            progress_bar.progress(90)

            # Clear progress
            progress_bar.progress(100)
            status_text.text("✅ Hoàn thành!")
            
            # Clear the progress container after a short delay
            import time
            time.sleep(1)
            progress_container.empty()

            # ==========================================
            # Enhanced Results Display
            # ==========================================
            st.markdown("""
            <div class="model-status status-success">
                ✅ <strong>Huấn luyện hoàn thành!</strong> Mô hình đã được lưu thành công
            </div>
            """, unsafe_allow_html=True)

            # Model Performance Metrics
            st.markdown("""
            <div class="forecast-section">
                <h4>📊 Đánh giá hiệu suất mô hình</h4>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🏃 Hiệu suất trên tập Train**")
                train_metrics = res["metrics"]["train"]
                st.markdown(f"""
                <div class="metric-box">
                    <strong>R² Score:</strong> {train_metrics.get('r2', 'N/A'):.4f}<br>
                    <strong>MAE:</strong> {train_metrics.get('mae', 'N/A'):.2f}<br>
                    <strong>MAPE:</strong> {train_metrics.get('mape', 'N/A'):.2f}%
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**🎯 Hiệu suất trên tập Test**")
                test_metrics = res["metrics"]["test"]
                st.markdown(f"""
                <div class="metric-box">
                    <strong>R² Score:</strong> {test_metrics.get('r2', 'N/A'):.4f}<br>
                    <strong>MAE:</strong> {test_metrics.get('mae', 'N/A'):.2f}<br>
                    <strong>MAPE:</strong> {test_metrics.get('mape', 'N/A'):.2f}%
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

            # Enhanced Chart Display
            st.markdown("""
            <div class="forecast-section">
                <h4>📈 Biểu đồ Train/Test Performance</h4>
            """, unsafe_allow_html=True)
            
            fig1, ax1 = plt.subplots(figsize=(12, 6), dpi=150)
            ax1.plot(res["train_pred_df"]["date"], res["train_pred_df"]["y_true"], label="Train Actual", linewidth=1.5, alpha=0.8)
            ax1.plot(res["train_pred_df"]["date"], res["train_pred_df"]["y_pred"], label="Train Predicted", linewidth=1.5, alpha=0.8)
            ax1.plot(res["test_pred_df"]["date"], res["test_pred_df"]["y_true"], label="Test Actual", linewidth=2, color='green')
            ax1.plot(res["test_pred_df"]["date"], res["test_pred_df"]["y_pred"], label="Test Predicted", linewidth=2, color='red', linestyle='--')
            ax1.set_title(f"📊 {symbol} - Mô hình LSTM Performance", fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1, clear_figure=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            fig1, ax1 = plt.subplots(figsize=(12, 6), dpi=150)
            ax1.plot(res["train_pred_df"]["date"], res["train_pred_df"]["y_true"], label="Train Actual", linewidth=1.5, alpha=0.8)
            ax1.plot(res["train_pred_df"]["date"], res["train_pred_df"]["y_pred"], label="Train Predicted", linewidth=1.5, alpha=0.8)
            ax1.plot(res["test_pred_df"]["date"], res["test_pred_df"]["y_true"], label="Test Actual", linewidth=2, color='green')
            ax1.plot(res["test_pred_df"]["date"], res["test_pred_df"]["y_pred"], label="Test Predicted", linewidth=2, color='red', linestyle='--')
            ax1.set_title(f"📊 {symbol} - Mô hình LSTM Performance", fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1, clear_figure=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Historical + Forecast Chart
            st.markdown("""
            <div class="forecast-section">
                <h4>🔮 Dự báo tương lai</h4>
            """, unsafe_allow_html=True)
            
            fig2, ax2 = plt.subplots(figsize=(12, 6), dpi=150)
            ax2.plot(s_close.index, s_close.values, label="Lịch sử giá đóng cửa", linewidth=1.5, color='blue')
            ax2.plot(res["future_df"]["date"], res["future_df"]["pred_raw"], label=f"Dự báo {horizon} phiên", 
                    linestyle="--", linewidth=3, color='red', marker='o', markersize=4)
            ax2.axhline(float(res["next_pred"]["last_close"]), color="gray", linewidth=1, linestyle=":", 
                       label="Giá đóng cửa gần nhất")
            ax2.set_title(f"📈 {symbol} - Dự báo LSTM ({int(horizon)} phiên)", fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2, clear_figure=True)

            # Zoom forecast chart
            fut_df = res["future_df"].copy()
            fut_df["date"] = pd.to_datetime(fut_df["date"])
            forecast_start = fut_df["date"].min()
            forecast_end = fut_df["date"].max()
            zoom_start = forecast_start - pd.Timedelta(days=60)

            hist_zoom = s_close[s_close.index >= zoom_start]

            fig3, ax3 = plt.subplots(figsize=(12, 6), dpi=150)
            ax3.plot(hist_zoom.index, hist_zoom.values, label="Lịch sử (2 tháng gần nhất)", linewidth=2, color='blue')
            
            if "pred_adj" in fut_df.columns:
                ax3.plot(fut_df["date"], fut_df["pred_adj"], label="Dự báo (adjusted)", 
                        linestyle="-.", linewidth=2, color='orange')
            
            ax3.plot(fut_df["date"], fut_df["pred_raw"], label="Dự báo (raw)", 
                    linestyle="--", linewidth=3, color='red', marker='o', markersize=5)
            
            ax3.axvline(forecast_start, color="gray", linestyle=":", linewidth=2, alpha=0.7, 
                       label="Điểm bắt đầu dự báo")
            ax3.set_xlim(zoom_start, forecast_end)
            ax3.set_title(f"🔍 {symbol} - Chi tiết dự báo (cửa sổ 2 tháng)", fontsize=14, fontweight='bold')
            ax3.set_xlabel("Ngày")
            ax3.set_ylabel("Giá (VND)")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3, clear_figure=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Next Session Prediction
            st.markdown("""
            <div class="forecast-section">
                <h4>🔔 Thông tin dự báo chi tiết</h4>
            """, unsafe_allow_html=True)
            
            next_price = float(res["future_df"]["pred_raw"].iloc[0])
            last_price = float(res["next_pred"]["last_close"])
            price_change = ((next_price - last_price) / last_price) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <h5>📅 Ngày dự báo đầu tiên</h5>
                    <strong>{str(res["next_pred"]["next_date"].date())}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <h5>💰 Giá dự báo</h5>
                    <strong style="color: {'green' if price_change > 0 else 'red'};">
                        {next_price:,.0f} VND
                    </strong>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-box">
                    <h5>📈 Thay đổi dự kiến</h5>
                    <strong style="color: {'green' if price_change > 0 else 'red'};">
                        {price_change:+.2f}%
                    </strong>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional forecast details
            st.markdown("**📋 Chi tiết tất cả phiên dự báo:**")
            forecast_details = res["future_df"][["date", "pred_raw"]].copy()
            forecast_details.columns = ["Ngày", "Giá dự báo (VND)"]
            forecast_details["Giá dự báo (VND)"] = forecast_details["Giá dự báo (VND)"].apply(lambda x: f"{x:,.0f}")
            st.dataframe(forecast_details, width="stretch")
            
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.markdown("""
            <div class="model-status status-error">
                ❌ <strong>Lỗi xảy ra trong quá trình huấn luyện!</strong><br>
                Vui lòng kiểm tra lại cấu hình và thử lại.
            </div>
            """, unsafe_allow_html=True)
            st.exception(e)
