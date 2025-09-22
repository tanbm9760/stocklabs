# pages/1_Screener.py
from __future__ import annotations

import io
import time
from datetime import datetime, timedelta
from typing import Dict, List
import json
import sys
import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# dùng module adapter mới (singular)
from pick_best_by_symbols import VnAdapter, ScoreWeights, score_symbols, _calc_adtv_vnd

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.theme import set_page_config, apply_theme
    from utils.styling import load_css, create_section_header, create_metric_card
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

# ===== Optional deps for LLM & news =====
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    OpenAI = None
    _OPENAI_OK = False

try:
    import requests
except Exception:
    requests = None

# =========================
# Page config & Styling
# =========================
set_page_config("FinStock", "📊")
apply_theme()
load_css()

st.markdown("""
<div class="main-title">
    <h1 style="color: white !important; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">📊 FinStock</h1>
    <p style="color: white !important; text-shadow: 0 1px 3px rgba(0,0,0,0.3);">Công cụ sàng lọc và chấm điểm cổ phiếu</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def _prev_weekday(dt: datetime) -> datetime:
    while dt.weekday() >= 5:
        dt -= timedelta(days=1)
    return dt

def _parse_symbols_input(raw: str) -> List[str]:
    seen, out = set(), []
    for p in [x.strip().upper() for x in raw.split(",") if x.strip()]:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def _filter_company_tickers_only(symbols: List[str]) -> List[str]:
    """Chỉ giữ mã 3 ký tự chữ cái (VD: FPT, VNM, HPG). Loại ETF/CW/CCQ."""
    return [s for s in (symbols or []) if len(s) == 3 and s.isalpha()]

@st.cache_data(show_spinner=False, ttl=60*45)
def _get_quote_history_cached(symbol: str, days: int, end_date: str, sources: List[str]) -> pd.DataFrame:
    adapter = VnAdapter(preferred_sources=sources, end_date=end_date, verbose=False)
    return adapter.get_quote_history(symbol, days=days)

@st.cache_data(show_spinner=False, ttl=60*60)
def _get_screener_snapshot_cached(source: str = "TCBS") -> pd.DataFrame:
    """
    Dùng vnstock.Screener (nếu có) chỉ để lấy sector phục vụ sector-neutral scoring.
    Không dùng các số liệu PE/PB/ROE… ở đây cho kết quả chính (đã lấy qua adapter.get_fundamentals).
    """
    try:
        adapter = VnAdapter(preferred_sources=[source], end_date=None, verbose=False)
        v = adapter.v
        Scr = getattr(v, "Screener", None)
        if Scr is None:
            return pd.DataFrame()
        scr = Scr(source=source)
        for meth in ("snapshot", "overview"):
            if hasattr(scr, meth):
                df = getattr(scr, meth)()
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df.columns = [str(c).strip() for c in df.columns]
                    return df
    except Exception:
        pass
    return pd.DataFrame()

def _extract_sector_map_from_snapshot(snapshot_df: pd.DataFrame) -> Dict[str, str]:
    """Trả về dict { 'FPT': 'Technology', ... } nếu snapshot có cột ngành."""
    if snapshot_df is None or snapshot_df.empty:
        return {}
    low = {c.lower(): c for c in snapshot_df.columns}
    sym_col = None
    for k in ("ticker", "symbol", "code"):
        if k in low:
            sym_col = low[k]; break
    if sym_col is None:
        return {}
    sector_col = None
    for k in ("sector", "industry", "icb_name", "industry_name", "sector_name"):
        if k in low:
            sector_col = low[k]; break
    if sector_col is None:
        return {}
    out = {}
    for _, r in snapshot_df.iterrows():
        sym = str(r[sym_col]).strip().upper()
        sec = str(r.get(sector_col, "")).strip()
        if sym and sec:
            out[sym] = sec
    return out

@st.cache_data(show_spinner=False, ttl=60*60)
def _get_fundamentals_precise_cached(symbol: str, source: str = "TCBS", period: str = "quarter") -> Dict[str, float]:
    """
    Lấy fundamentals qua adapter.get_fundamentals:
      - PE/PB/ROE/... từ finance.ratio(period)
      - rev_yoy từ **TCBS API** (yearRevenueGrowth, quý mới nhất)
      - Các trường dạng % đã được chuẩn hoá thành đơn vị phần trăm trong adapter.
    """
    try:
        adapter = VnAdapter(preferred_sources=[source], end_date=None, verbose=False)
        return adapter.get_fundamentals(symbol, period=period)
    except Exception:
        return {}

def _annualized_vol(returns: pd.Series):
    d = returns.dropna()
    return d.std(ddof=0) * (252 ** 0.5) if len(d) > 1 else np.nan

def _safe_pct(a, b):
    try:
        if b == 0 or pd.isna(b): return np.nan
        return (float(a) - float(b)) / abs(float(b))
    except Exception:
        return np.nan

def compute_features_local_from_px(px: pd.DataFrame, fund: Dict) -> Dict:
    """Tính m1/m3/m6, ADTV(20) từ dữ liệu giá; ghép thêm fundamentals đã chuẩn hoá từ adapter."""
    if px is None or px.empty:
        base = {"m1": np.nan, "m3": np.nan, "m6": np.nan, "adtv": np.nan, "vol": np.nan}
        base.update(fund or {})
        return base

    p = px.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values("date")

    def pct_change(days_back):
        if len(p) < days_back + 1: return np.nan
        return _safe_pct(p["close"].iloc[-1], p["close"].iloc[-(days_back + 1)])

    m1 = pct_change(21)
    m3 = pct_change(63)
    m6 = pct_change(126)

    # Tính ADTV trực tiếp từ dữ liệu lịch sử (tốt hơn)
    adtv = _calc_adtv_from_history(p, n=20)
    vol = _annualized_vol(p["close"].pct_change())

    out = {"m1": m1, "m3": m3, "m6": m6, "adtv": adtv, "vol": vol}
    out.update(fund or {})
    return out

def _calc_adtv_from_history(px: pd.DataFrame, n: int = 20) -> float:
    """Tính ADTV từ dữ liệu lịch sử giá cải tiến."""
    if px is None or px.empty:
        return np.nan
    
    # Đảm bảo có đủ cột cần thiết
    available_cols = set(px.columns)
    required_cols = {"close", "volume"}
    
    if not required_cols.issubset(available_cols):
        return np.nan
    
    # Lấy n phiên gần nhất và làm sạch dữ liệu
    data = px.tail(n).copy()
    data["close_clean"] = pd.to_numeric(data["close"], errors="coerce")
    data["volume_clean"] = pd.to_numeric(data["volume"], errors="coerce")
    
    # Loại bỏ các ngày có volume = 0 hoặc NaN (không giao dịch)
    valid_data = data.dropna(subset=["close_clean", "volume_clean"])
    valid_data = valid_data[valid_data["volume_clean"] > 0]
    
    if len(valid_data) < 5:  # Cần ít nhất 5 phiên có giao dịch
        return np.nan
    
    # Tính giá trị giao dịch trung bình
    # Giá từ vnstock: 41.10 → 41,100 VND (nhân 1000)
    # Volume: đã đúng đơn vị (số cổ phiếu)
    daily_value = valid_data["close_clean"] * 1000 * valid_data["volume_clean"]
    adtv = float(daily_value.mean())
    
    return adtv if not np.isnan(adtv) and adtv > 0 else np.nan

def _missing_bdays_to_break(dates: pd.Series) -> List[pd.Timestamp]:
    if dates.empty:
        return []
    d0 = pd.to_datetime(dates).sort_values()
    start = d0.min().normalize()
    end = d0.max().normalize()
    all_bdays = pd.bdate_range(start, end, freq="B")
    present = pd.to_datetime(d0.dt.normalize().unique())
    present_set = set(present)
    return [d for d in all_bdays if d not in present_set]

def _company_name_from_snapshot(snapshot_df: pd.DataFrame, symbol: str) -> str:
    if snapshot_df is None or snapshot_df.empty:
        return "—"
    low = {c.lower(): c for c in snapshot_df.columns}
    sym_col = None
    for k in ("ticker", "symbol", "code"):
        if k in low:
            sym_col = low[k]; break
    if sym_col is None:
        return "—"
    row = snapshot_df[sym_col].astype(str).str.upper() == symbol.upper()
    hit = snapshot_df[row]
    if hit.empty:
        return "—"
    name_candidates = [
        "organ_short_name", "organ_short_nm", "organ_name", "organname",
        "company_name", "companyName", "name", "org_name", "short_name"
    ]
    for cand in name_candidates:
        col = low.get(cand.lower())
        if col and col in hit.columns:
            val = str(hit.iloc[0][col]).strip()
            if val:
                return val
    return "—"

def make_ohlcv_figure(
    px: pd.DataFrame,
    title: str,
    *,
    default_months_view: int = 3,
    right_pad_months: int = 2,
    height: int = 700,
    vol_frac: float = 0.18,
    gap: float = 0.02,
    show_ma9: bool = True,
    show_ma20: bool = True,
    show_ma50: bool = False,
    show_ma200: bool = False,
    show_bollinger: bool = True,
) -> go.Figure:
    df = px.copy()
    df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date")
    missing_vals = _missing_bdays_to_break(df["date"])

    UP, DOWN = "#26a69a", "#ef5350"
    UP_A, DOWN_A = "rgba(38,166,154,0.85)", "rgba(239,83,80,0.85)"

    # Tính toán SMA
    if show_ma9:
        df["ma9"] = df["close"].rolling(9, min_periods=1).mean()
    if show_ma20:
        df["ma20"] = df["close"].rolling(20, min_periods=1).mean()
    if show_ma50:
        df["ma50"] = df["close"].rolling(50, min_periods=1).mean()
    if show_ma200:
        df["ma200"] = df["close"].rolling(200, min_periods=1).mean()
        
    # Tính toán Bollinger Bands
    if show_bollinger:
        bb_period = 20
        bb_multiplier = 2
        df["bb_middle"] = df["close"].rolling(bb_period, min_periods=1).mean()
        bb_std = df["close"].rolling(bb_period, min_periods=1).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * bb_multiplier)
        df["bb_lower"] = df["bb_middle"] - (bb_std * bb_multiplier)

    fig = go.Figure()
    
    # Thêm Bollinger Bands trước để hiển thị ở phía sau
    if show_bollinger:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["bb_upper"], 
            mode="lines", name="BB Upper",
            line=dict(color="rgba(156, 163, 175, 0.3)", width=1, dash="dash"),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["bb_lower"], 
            mode="lines", name="BB Lower",
            line=dict(color="rgba(156, 163, 175, 0.3)", width=1, dash="dash"),
            fill="tonexty", fillcolor="rgba(156, 163, 175, 0.1)",
            showlegend=False
        ))
    
    # Thêm candlestick chart
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Giá",
        increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
        decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN),
        whiskerwidth=0,
        showlegend=True
    ))
    
    # Thêm các đường SMA
    if show_ma9:
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma9"], mode="lines", name="SMA9",
                                 line=dict(color="#f97316", width=1.2), showlegend=True))
    if show_ma20:
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"], mode="lines", name="SMA20",
                                 line=dict(color="#f59e0b", width=1.4), showlegend=True))
    if show_ma50:
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma50"], mode="lines", name="SMA50",
                                 line=dict(color="#60a5fa", width=1.4), showlegend=True))
    if show_ma200:
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma200"], mode="lines", name="SMA200",
                                 line=dict(color="#a78bfa", width=1.6), showlegend=True))
    
    # Thêm đường Bollinger Middle (chính là SMA20)
    if show_bollinger:
        fig.add_trace(go.Scatter(x=df["date"], y=df["bb_middle"], mode="lines", name="BB Middle",
                                 line=dict(color="#9ca3af", width=1.0, dash="dot"), showlegend=True))

    up = (df["close"] >= df["open"]).fillna(False)
    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"].fillna(0), name="Khối lượng",
        marker_color=np.where(up, UP_A, DOWN_A), marker_line_width=0, opacity=0.95,
        yaxis="y2", showlegend=True
    ))

    x_end = df["date"].max()
    x_max_limit = x_end + pd.DateOffset(months=1)  # Giới hạn tối đa +1 tháng từ ngày cuối
    x_pad = x_end + pd.DateOffset(months=int(right_pad_months))
    x_start = x_end - pd.DateOffset(months=int(default_months_view))
    
    # Tính toán các mốc thời gian từ ngày cuối dữ liệu
    x_1m_start = x_end - pd.DateOffset(months=1)
    x_3m_start = x_end - pd.DateOffset(months=3)
    x_6m_start = x_end - pd.DateOffset(months=6)

    v = pd.to_numeric(df["volume"], errors="coerce").dropna()
    y2_max = None
    if len(v) > 0:
        q995 = float(np.percentile(v, 99.5)); med = float(np.median(v))
        y2_max = max(q995, med*3.0) * 1.10

    grid, axis, font = "#1f2937", "#374151", "#e5e7eb"
    paper = "#0b1220"; plot = "#0b1220"

    fig.update_layout(
        xaxis=dict(
            range=[x_start, x_end],  # Set range to end at last data date
            rangebreaks=[dict(bounds=["sat","mon"]), dict(values=missing_vals)],
            rangeslider=dict(visible=True, 
                             range=[df["date"].min(), x_max_limit],  # Giới hạn tối đa +1 tháng
                             thickness=0.12, bgcolor="#0f172a", bordercolor="#334155"),
            rangeselector=dict(
                y=0, yanchor="bottom", x=0.5, xanchor="center",
                bgcolor="#0b1220", activecolor="#1f2a44",
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ],
            ),
            ticklabelposition="outside",
            showline=True, linewidth=1, linecolor=axis, gridcolor=grid, zeroline=False
        ),
        yaxis=dict(
            domain=[vol_frac + gap, 1.0],
            autorange=True,  # Tự động scale theo dữ liệu hiển thị
            fixedrange=False,  # Cho phép zoom/pan trục Y
            side="right",  # Đặt trục giá bên phải
            showline=True, linewidth=1, linecolor=axis, gridcolor=grid, zeroline=False
        ),
        yaxis2=dict(
            domain=[0.0, vol_frac],
            rangemode="tozero", 
            autorange=True,  # Tự động scale cho trục volume
            fixedrange=False,  # Cho phép zoom/pan trục Y volume 
            side="right",  # Đặt trục khối lượng bên phải
            tickformat="~s",
            showline=True, linewidth=1, linecolor=axis, gridcolor=grid, zeroline=False
        ),
        legend=dict(
            x=0.005, y=0.995, xanchor="left", yanchor="top",
            bgcolor="rgba(15,23,42,0.65)", bordercolor="#334155", borderwidth=1,
            font=dict(color=font, size=11),
            orientation="v",
            itemsizing="constant",
        ),
        title=dict(text=title, x=0.5, xanchor="center", font=dict(color="#ffffff", size=16)),
        hovermode="x unified",
        bargap=0.12, bargroupgap=0.0,
        margin=dict(l=6, r=6, t=52, b=10),
        paper_bgcolor=paper, plot_bgcolor=plot,
        font=dict(color=font),
        hoverlabel=dict(bgcolor="#0f172a", font_color=font, bordercolor=axis),
        height=height,
        # Thêm cấu hình để tối ưu hóa tương tác
        dragmode="pan",  # Mặc định là pan (kéo để di chuyển biểu đồ)
        selectdirection="h",  # Chỉ zoom theo chiều ngang (h = horizontal)
        # Thêm scroll zoom để có thể zoom bằng cuộn chuột
        xaxis_fixedrange=False,  # Cho phép zoom trục X
        yaxis_fixedrange=False   # Cho phép zoom trục Y
    )
    
    return fig


# ======== TCBS Activities News helpers ========
@st.cache_data(show_spinner=False, ttl=600)
def fetch_activity_news_raw(symbol: str, page: int = 0, size: int = 100) -> Dict:
    """Gọi API TCBS activities để lấy danh sách công bố/hoạt động theo mã."""
    if requests is None:
        return {}
    url = "https://apipubaws.tcbs.com.vn/tcanalysis/v1/news/activities"
    params = {"fData": symbol, "fType": "tickers", "page": int(page), "size": int(size)}
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StockApp/1.0)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def filter_recent_activity_news(payload: Dict, recent_days: int = 7) -> List[Dict]:
    """Lọc danh sách news, chỉ lấy trong vòng `recent_days` gần nhất."""
    items = (payload or {}).get("listActivityNews", []) or []
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=int(recent_days))
    out: List[Dict] = []
    for it in items:
        try:
            ts = pd.to_datetime(it.get("publishDate"), errors="coerce")
        except Exception:
            ts = pd.NaT
        if pd.isna(ts):
            continue
        if ts >= cutoff:
            out.append({
                "ticker": it.get("ticker"),
                "title": (it.get("title") or "").strip(),
                "source": it.get("source"),
                "published_at": ts,
            })
    out.sort(key=lambda x: x["published_at"], reverse=True)
    return out


# ======== Indicators / stats for FORM ========
def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    s = series.astype(float)
    delta = s.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _atr_df(df: pd.DataFrame, n: int = 14) -> pd.Series:
    d = df.copy()
    d["prev_close"] = d["close"].shift(1)
    tr = pd.concat([
        (d["high"] - d["low"]).abs(),
        (d["high"] - d["prev_close"]).abs(),
        (d["low"]  - d["prev_close"]).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _slope(arr: pd.Series, lookback: int = 5) -> float:
    y = pd.to_numeric(arr.tail(lookback), errors="coerce").dropna()
    if len(y) < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)
    A = np.vstack([x, np.ones(len(x))]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)

def _pivot_levels(df: pd.DataFrame, win_high: int = 20, win_low: int = 20):
    d = df.copy().tail(max(win_high, win_low) + 5)
    res = float(d["high"].rolling(win_high, min_periods=1).max().iloc[-2])
    sup = float(d["low"].rolling(win_low, min_periods=1).min().iloc[-2])
    return res, sup

def build_structured_stats(px: pd.DataFrame) -> dict:
    p = px.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values("date").reset_index(drop=True)

    last_close = float(p["close"].iloc[-1])
    last_date  = p["date"].iloc[-1].strftime("%Y-%m-%d")

    # Chỉ MA20/50/200
    for w in (20, 50, 200):
        p[f"ma{w}"] = p["close"].rolling(w).mean()

    def rr(days):
        return float((p["close"].iloc[-1] / p["close"].iloc[max(0, len(p)-1-days)] - 1) * 100) if len(p) > days else np.nan
    r1m, r3m, r6m = rr(21), rr(63), rr(126)

    rsi14 = float(_rsi(p["close"], 14).iloc[-1])
    atr14 = _atr_df(p, 14).iloc[-1]
    atr_pct = float((atr14 / last_close) * 100) if pd.notna(atr14) else np.nan
    adtv_vnd = float((p["close"].tail(20) * 1000 * p["volume"].tail(20)).mean()) if "volume" in p else np.nan

    win = min(252, len(p))
    hi_52w = float(p["high"].tail(win).max())
    lo_52w = float(p["low"].tail(win).min())

    slope_ma20  = _slope(p["ma20"], 5)
    slope_ma50  = _slope(p["ma50"], 5)
    slope_ma200 = _slope(p["ma200"], 10)

    vol_mean_20 = float(p["volume"].tail(20).mean())
    vol_mean_60 = float(p["volume"].tail(60).mean())
    vol_trend = "tăng" if vol_mean_20 > vol_mean_60 * 1.1 else ("giảm" if vol_mean_20 < vol_mean_60 * 0.9 else "đi ngang")

    res_piv, sup_piv = _pivot_levels(p, 20, 20)
    levels = {
        "kháng_cự_gần": res_piv,
        "hỗ_trợ_gần": sup_piv,
        "ma20": float(p["ma20"].iloc[-1]) if not pd.isna(p["ma20"].iloc[-1]) else np.nan,
        "ma50": float(p["ma50"].iloc[-1]) if not pd.isna(p["ma50"].iloc[-1]) else np.nan,
        "ma200": float(p["ma200"].iloc[-1]) if not pd.isna(p["ma200"].iloc[-1]) else np.nan,
        "hi_52w": hi_52w, "lo_52w": lo_52w
    }

    return {
        "last_date": last_date, "last_close": last_close,
        "ret_1m_%": r1m, "ret_3m_%": r3m, "ret_6m_%": r6m,
        "rsi14": rsi14, "atr14_%": atr_pct,
        "adtv_20_vnd": adtv_vnd,
        "slopes": {"ma20": slope_ma20, "ma50": slope_ma50, "ma200": slope_ma200},
        "levels": levels,
        "vol_trend": vol_trend
    }

def auto_generate_report_if_possible(symbol: str, tech_stats: dict, llm_model: str, company_name: str = None) -> bool:
    """
    Tự động tạo báo cáo nếu có API key và chưa có báo cáo cho symbol này.
    Returns True nếu đã tạo báo cáo thành công.
    """
    key = st.session_state.get("openai_api_key", "") or ""
    if not key or not _OPENAI_OK:
        return False
    
    # Kiểm tra nếu đã có báo cáo cho symbol này rồi
    existing_report = (st.session_state.get("form_cache") or {}).get(symbol)
    if existing_report:
        return True
    
    try:
        model = llm_model or "gpt-4o-mini"
        template = st.session_state.get("analysis_template", "")
        prompt = st.session_state.get("analysis_prompt", "")
        system_prompt = st.session_state.get("system_prompt", "")
        
        with st.spinner(f"🤖 Đang tự động phân tích {symbol}..."):
            report = call_llm_structured_report(
                key, model, symbol, tech_stats,
                template=template, prompt=prompt, system_prompt=system_prompt, company_name=company_name
            )
            
        st.session_state.setdefault("form_cache", {})[symbol] = report
        return True
    except Exception as e:
        st.error(f"❌ Lỗi tự động tạo báo cáo cho {symbol}: {e}")
        return False

def call_llm_structured_report(api_key: str, model: str, symbol: str, tech_stats: dict, 
                              template: str = None, prompt: str = None, system_prompt: str = None, company_name: str = None) -> str:
    if not _OPENAI_OK or not api_key:
        return "⛔ Chưa cấu hình OpenAI API key."

    # Sử dụng template từ tham số hoặc mặc định
    default_template = (
        "📊 PHÂN TÍCH KỸ THUẬT - {symbol}\n\n"
        "PHÂN TÍCH KỸ THUẬT\n\n"
        "1. Xu hướng giá:\n- ...\n- ...\n- ...\n\n"
        "2. Đường MA (20/50/200):\n- MA20: ...\n- MA50: ...\n- MA200: ...\n\n"
        "3. Khối lượng:\n- ...\n- ...\n\n"
        "4. Hỗ trợ & Kháng cự:\n- Kháng cự: ...\n- Hỗ trợ gần: ...\n- Hỗ trợ sâu: ...\n\n"
        "NHẬN ĐỊNH NHANH & CHIẾN LƯỢC\n\n"
        "- Ngắn hạn: ...\n- Trung hạn: ...\n\n"
        "Chiến lược:\n- Lướt sóng: ...\n- Trung hạn: ..."
    )
    
    used_template = template or default_template
    
    # Format template với symbol và company name
    display_company_name = company_name or "—"
    formatted_template = used_template.format(symbol=symbol, company_name=display_company_name)
    
    # Sử dụng prompt từ tham số hoặc mặc định
    default_prompt = (
        "Bạn là chuyên gia PTKT cổ phiếu Việt Nam. Dựa **duy nhất** vào dữ liệu cung cấp, "
        "hãy viết báo cáo đúng **form mẫu** (tiếng Việt, ngắn gọn). Chỉ đánh giá MA20/MA50/MA200.\n\n"
        "TEMPLATE:\n{template}\n\n"
        "HƯỚNG DẪN:\n"
        "- Bắt đầu báo cáo bằng header có mã cổ phiếu và tên công ty như trong template.\n"
        "- 'Đường MA' nêu hướng (lên/xuống/đi ngang) + vai trò (hỗ trợ/kháng cự) theo độ dốc & vị trí giá.\n"
        "- 'Khối lượng' so sánh trung bình 20 vs 60 phiên.\n"
        "- 'Hỗ trợ & Kháng cự' dựa pivot gần nhất, MA và 52W.\n"
        "- 'Lướt sóng/Trung hạn' có vùng mua tham khảo, stoploss, mục tiêu theo kháng cự/đỉnh cũ.\n"
        "- Định dạng số có **dấu phẩy** (vd 31,000). Không coi đây là khuyến nghị đầu tư."
    )
    
    used_prompt = prompt or default_prompt
    guidance = used_prompt.format(template=formatted_template)
    
    # Sử dụng system prompt từ tham số hoặc mặc định
    default_system_prompt = "Bạn là chuyên gia phân tích kỹ thuật cổ phiếu Việt Nam, viết báo cáo theo template được cung cấp."
    used_system_prompt = system_prompt or default_system_prompt

    payload = {"symbol": symbol, **tech_stats}
    try:
        client = OpenAI(api_key=api_key)
        msgs = [
            {"role": "system", "content": used_system_prompt},
            {"role": "user", "content": guidance + "\n\nDỮ LIỆU:\n" + json.dumps(payload, ensure_ascii=False)}
        ]
        out = client.chat.completions.create(model=model, messages=msgs, temperature=0.15, max_tokens=1000)
        text = out.choices[0].message.content if out and out.choices else ""
        return text or "_Không có nội dung_"
    except Exception as e:
        return f"❌ Lỗi gọi LLM: {e}"

# =========================
# Sidebar — Watchlist Manager
# =========================
# Enhanced Sidebar - Simplified
# =========================
with st.sidebar:
    # --- Khởi tạo state ---
    if "watchlists" not in st.session_state:
        # Danh sách mẫu theo ngành với các cổ phiếu vốn hóa lớn nhất
        st.session_state.watchlists = {
            "My Picks": ["FPT", "VNM", "HPG", "MWG", "SSI", "VCB"],
            
            # Ngân hàng (Top 10 theo vốn hóa)
            "🏦 Ngân hàng": ["VCB", "BID", "CTG", "TCB", "VPB", "MBB", "ACB", "TPB", "STB", "SHB"],
            
            # Chứng khoán (Top 10)
            "📈 Chứng khoán": ["SSI", "VND", "HCM", "VCI", "MBS", "CTS", "VIX", "FTS", "BSI", "AGR"],
            
            # Bất động sản (Top 10)
            "🏢 Bất động sản": ["VIC", "VHM", "VRE", "NVL", "PDR", "KDH", "DXG", "BCM", "NLG", "HDG"],
            
            # Dầu khí (Top 10)
            "⛽ Dầu khí": ["GAS", "PLX", "PVD", "PVC", "PVS", "PVB", "PSH", "PVT", "BSR", "OIL"],
            
            # Thép (Top 10)
            "🔩 Thép": ["HPG", "HSG", "NKG", "SMC", "TLH", "VGS", "TVN", "KSS", "VCA", "DTL"],
            
            # Công nghệ (Top 10)
            "💻 Công nghệ": ["FPT", "CMG", "CTR", "ELC", "VGI", "ITD", "DST", "VCS", "ICT", "IDC"],
            
            # Bán lẻ (Top 10)
            "🛒 Bán lẻ": ["MWG", "PNJ", "FRT", "MBB", "CTF", "VRE", "DGW", "AST", "SBT", "VGC"],
            
            # Thực phẩm & Đồ uống (Top 10)
            "🍺 Thực phẩm": ["VNM", "SAB", "MSN", "MCH", "KDC", "VHC", "CII", "QNS", "LSS", "TAC"],
            
            # Điện (Top 10)
            "⚡ Điện": ["POW", "GEG", "PC1", "NT2", "REE", "VSH", "SBA", "HND", "EVE", "BWE"],
            
            # Hàng không & Vận tải (Top 10)
            "✈️ Vận tải": ["HVN", "VJC", "GMD", "VOS", "STG", "MVN", "PVT", "TCO", "VSC", "VIP"],
            
            # Bluechips tổng hợp
            "⭐ Bluechips": ["VIC", "VHM", "VCB", "BID", "VNM", "SAB", "GAS", "PLX", "FPT", "HPG", "MWG", "SSI"],
        }
    if "current_watchlist" not in st.session_state:
        st.session_state.current_watchlist = "My Picks"

    # Hiển thị số lượng danh sách
    total_lists = len(st.session_state.watchlists)
    total_symbols = sum(len(symbols) for symbols in st.session_state.watchlists.values())
    
    st.markdown(f"""
    <div class="metric-card">
        <strong>📋 {total_lists}</strong> danh sách | <strong>📈 {total_symbols}</strong> mã cổ phiếu
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # PHẦN 1: CHỌN DANH SÁCH PHÂN TÍCH
    # =========================
    st.markdown("""
    <div class="section-header">
        <h4>🎯 Chọn danh sách phân tích</h4>
    </div>
    """, unsafe_allow_html=True)

    # Callback function cho selectbox
    def update_current_watchlist():
        """Callback function khi selectbox thay đổi"""
        new_selection = st.session_state.active_watchlist_picker
        if new_selection != st.session_state.current_watchlist:
            st.session_state.current_watchlist = new_selection
            # Clear cached AI reports khi thay đổi watchlist
            if 'ai_reports' in st.session_state:
                st.session_state.ai_reports = {}

    # Tính toán options một lần và đảm bảo consistency
    available_watchlists = sorted(st.session_state.watchlists.keys())
    
    # Đảm bảo current_watchlist hợp lệ
    if st.session_state.current_watchlist not in available_watchlists:
        st.session_state.current_watchlist = available_watchlists[0] if available_watchlists else "My Picks"
    
    # Tính index an toàn
    try:
        default_index = available_watchlists.index(st.session_state.current_watchlist)
    except ValueError:
        default_index = 0
        st.session_state.current_watchlist = available_watchlists[0] if available_watchlists else "My Picks"
    
    # Selectbox with callback - không cần gán trực tiếp
    st.selectbox(
        "Danh sách để phân tích:",
        options=available_watchlists,
        index=default_index,
        key="active_watchlist_picker",
        help="Danh sách này sẽ được sử dụng để phân tích khi nhấn nút Phân tích",
        on_change=update_current_watchlist
    )

    # Hiển thị các mã trong danh sách được chọn
    current_symbols = st.session_state.watchlists.get(st.session_state.current_watchlist, [])
    if current_symbols:
        st.info(f"📊 **{len(current_symbols)} mã cổ phiếu**: {', '.join(current_symbols)}")
    else:
        st.warning("⚠️ Danh sách trống")

    # =========================
    # PHẦN 2: CẤU HÌNH DANH SÁCH
    # =========================
    with st.expander("⚙️ Cấu hình danh sách", expanded=False):
        st.markdown("""
        <div class="control-panel" style="margin-top: 0;">
        """, unsafe_allow_html=True)

        # Selectbox cho cấu hình - không liên quan đến current_watchlist
        selected_wl = st.selectbox(
            "Chọn danh sách để chỉnh sửa:",
            options=available_watchlists,  # Sử dụng cùng list đã sort
            key="config_watchlist_picker",
            help="Chọn danh sách muốn chỉnh sửa (độc lập với danh sách phân tích)"
        )

        new_wl_name = st.text_input(
            "Tên danh sách mới:",
            value="",
            placeholder="Nhập tên để tạo danh sách mới",
            key="new_wl_name_input",
            help="Nhập tên để tạo danh sách mới"
        )

        # Ô text chỉnh mã cho watchlist đang chọn
        current_symbols_str = ", ".join(st.session_state.watchlists.get(selected_wl, []))
        edited_symbols_str = st.text_area(
            f"Danh sách mã cổ phiếu cho '{selected_wl}':",
            value=current_symbols_str,
            key=f"wl_text_{selected_wl}",
            help="Ví dụ: FPT, VNM, HPG, SSI (chỉ chấp nhận mã 3 ký tự)",
            height=100
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Action buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn1:
            if st.button("💾 Lưu", use_container_width=True, help="Lưu thay đổi danh sách", key="save_btn", type="primary"):
                cleaned = _filter_company_tickers_only(_parse_symbols_input(edited_symbols_str))
                st.session_state.watchlists[selected_wl] = cleaned
                st.success(f"✅ Đã lưu '{selected_wl}' ({len(cleaned)} mã)")
        with col_btn2:
            if st.button("➕ Tạo mới", use_container_width=True, help="Tạo danh sách mới", key="create_btn", type="secondary"):
                name = (new_wl_name or "").strip()
                if not name:
                    st.warning("⚠️ Vui lòng nhập tên danh sách")
                elif name in st.session_state.watchlists:
                    st.warning("⚠️ Tên đã tồn tại")
                else:
                    cleaned = _filter_company_tickers_only(_parse_symbols_input(edited_symbols_str))
                    st.session_state.watchlists[name] = cleaned
                    st.session_state.current_watchlist = name
                    st.success(f"✅ Đã tạo '{name}' ({len(cleaned)} mã)")
                    st.rerun()
        with col_btn3:
            if st.button("🗑️ Xóa", use_container_width=True, help="Xóa danh sách", key="delete_btn", type="secondary"):
                if selected_wl in st.session_state.watchlists:
                    if len(st.session_state.watchlists) <= 1:
                        st.warning("⚠️ Cần ít nhất 1 danh sách")
                    else:
                        del st.session_state.watchlists[selected_wl]
                        st.session_state.current_watchlist = next(iter(st.session_state.watchlists.keys()))
                        st.success(f"✅ Đã xóa '{selected_wl}'")
                        st.rerun()
    
    # =========================
    # AI Reports Management
    # =========================
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <h3>🤖 Báo cáo AI</h3>
    </div>
    """, unsafe_allow_html=True)
    
    cached_reports = st.session_state.get("form_cache", {})
    if cached_reports:
        st.info(f"📝 Có {len(cached_reports)} báo cáo AI trong cache")
        
        # Hiển thị danh sách báo cáo
        for symbol in sorted(cached_reports.keys()):
            with st.expander(f"📄 {symbol}", expanded=False):
                report_text = cached_reports[symbol]
                st.markdown(report_text)
                
                # Nút download cho từng báo cáo
                st.download_button(
                    label="⬇️ Tải báo cáo",
                    data="\ufeff" + report_text,
                    file_name=f"{symbol}_AI_Report.txt",
                    mime="text/plain; charset=utf-8",
                    key=f"download_{symbol}"
                )
        
        # Nút xóa tất cả báo cáo
        if st.button("🗑️ Xóa tất cả báo cáo AI", key="clear_all_reports", 
                    help="Xóa tất cả báo cáo AI đã cached", type="secondary"):
            st.session_state["form_cache"] = {}
            st.success("✅ Đã xóa tất cả báo cáo AI")
            st.rerun()
            
        # Nút download tất cả báo cáo
        if len(cached_reports) > 1:
            all_reports = ""
            for symbol, report in cached_reports.items():
                all_reports += f"{'='*50}\n"
                all_reports += f"BÁO CÁO PHÂN TÍCH: {symbol}\n"
                all_reports += f"{'='*50}\n\n"
                all_reports += report + "\n\n"
            
            st.download_button(
                label="📦 Tải tất cả báo cáo",
                data="\ufeff" + all_reports,
                file_name="All_AI_Reports.txt",
                mime="text/plain; charset=utf-8",
                key="download_all_reports"
            )
    else:
        st.info("📝 Chưa có báo cáo AI nào. Nhập API key và phân tích để tạo báo cáo tự động.")

# =========================
# MAIN CONTENT - Quick Analysis Section
# =========================

# Quick Analysis Panel (ở đầu)
col_main1, col_main2 = st.columns([2, 1])

with col_main1:
    # Hiển thị danh sách đang chọn
    active_wl_name = st.session_state.current_watchlist
    current_symbols = st.session_state.watchlists.get(active_wl_name, [])
    
    st.markdown(f"""
    <div class="metric-card" style="padding: 1rem; margin-bottom: 1rem;">
        <h4>📂 Danh sách: {active_wl_name}</h4>
        <p><strong>{len(current_symbols)}</strong> mã cổ phiếu: {', '.join(current_symbols[:10])}{' ...' if len(current_symbols) > 10 else ''}</p>
    </div>
    """, unsafe_allow_html=True)

with col_main2:
    # OpenAI API Key input
    st.markdown("**🔑 OpenAI API Key**")
    api_key_default = st.session_state.get("openai_api_key", "")
    api_key_input = st.text_input(
        "Nhập API key để sử dụng AI:", 
        value=api_key_default, 
        type="password",
        placeholder="sk-...",
        help="API key để tạo báo cáo phân tích bằng AI",
        label_visibility="collapsed"
    )
    if api_key_input:
        st.session_state["openai_api_key"] = api_key_input
    
    # Mặc định sử dụng gpt-4o-mini
    st.session_state["llm_model"] = "gpt-4o-mini"
    
# Nút phân tích chính
analyze_btn = st.button(
    "🚀 Bắt đầu phân tích", 
    use_container_width=True,
    type="primary",
    help="Nhấn để bắt đầu phân tích các cổ phiếu trong danh sách"
)

# Advanced Configuration (có thể ẩn/hiện)
with st.expander("⚙️ Cấu hình nâng cao", expanded=False):
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        st.markdown("**📊 Tham số phân tích**")
        days = st.number_input(
            "📅 Số ngày lịch sử", 
            60, 1500, 360, 30, 
            help="Số ngày dữ liệu giá để phân tích (≥252 cho đủ 52 tuần)"
        )
        
        tminus = st.number_input(
            "⏰ Lùi ngày (T-n)", 
            0, 30, 0, 1,
            help="Lùi ngày kết thúc để phân tích dữ liệu trước đó"
        )
        
        end_date = st.text_input(
            "📅 Ngày kết thúc", 
            value="",
            placeholder="YYYY-MM-DD hoặc để trống",
            help="Ngày kết thúc phân tích (mặc định: hôm nay)"
        )

    with col_config2:
        st.markdown("**🤖 Cấu hình AI**")
        
        polite_delay_ms = st.slider(
            "⏱️ Độ trễ API (ms)", 
            0, 1000, 300, 50,
            help="Thời gian chờ giữa các lệnh gọi API"
        )

# =========================
# Cấu hình Prompt và Template (Riêng biệt)
# =========================
with st.expander("📝 Cấu hình Prompt & Template AI", expanded=False):
    st.markdown("### 🛠️ Tùy chỉnh cách AI tạo báo cáo phân tích")
    st.info("💡 **Hướng dẫn:** Bạn có thể tùy chỉnh template báo cáo và prompt hướng dẫn để AI tạo ra báo cáo theo ý muốn.")
    
    col_template, col_prompt = st.columns(2)
    
    with col_template:
        st.markdown("**📋 Template báo cáo**")
        # Template configuration
        default_template = (
            "PHÂN TÍCH KỸ THUẬT\n\n"
            "1. Xu hướng giá:\n- ...\n- ...\n- ...\n\n"
            "2. Đường MA (20/50/200):\n- MA20: ...\n- MA50: ...\n- MA200: ...\n\n"
            "3. Khối lượng:\n- ...\n- ...\n\n"
            "4. Hỗ trợ & Kháng cự:\n- Kháng cự: ...\n- Hỗ trợ gần: ...\n- Hỗ trợ sâu: ...\n\n"
            "NHẬN ĐỊNH NHANH & CHIẾN LƯỢC\n\n"
            "- Ngắn hạn: ...\n- Trung hạn: ...\n\n"
            "Chiến lược:\n- Lướt sóng: ...\n- Trung hạn: ..."
        )
        
        analysis_template = st.text_area(
            "Template mẫu báo cáo",
            value=st.session_state.get("analysis_template", default_template),
            height=300,
            help="Định dạng template cho báo cáo phân tích. Sử dụng ... để AI điền nội dung.",
            key="template_area"
        )
        st.session_state["analysis_template"] = analysis_template
        
        if st.button("🔄 Reset Template", help="Khôi phục template mặc định", key="reset_template"):
            st.session_state["analysis_template"] = default_template
            st.rerun()
    
    with col_prompt:
        st.markdown("**💬 Prompt hướng dẫn**")
        # Prompt configuration
        default_prompt = (
            "Bạn là chuyên gia PTKT cổ phiếu Việt Nam. Dựa **duy nhất** vào dữ liệu cung cấp, "
            "hãy viết báo cáo đúng **form mẫu** (tiếng Việt, ngắn gọn). Chỉ đánh giá MA20/MA50/MA200.\n\n"
            "TEMPLATE:\n{template}\n\n"
            "HƯỚNG DẪN:\n"
            "- 'Đường MA' nêu hướng (lên/xuống/đi ngang) + vai trò (hỗ trợ/kháng cự) theo độ dốc & vị trí giá.\n"
            "- 'Khối lượng' so sánh trung bình 20 vs 60 phiên.\n"
            "- 'Hỗ trợ & Kháng cự' dựa pivot gần nhất, MA và 52W.\n"
            "- 'Lướt sóng/Trung hạn' có vùng mua tham khảo, stoploss, mục tiêu theo kháng cự/đỉnh cũ.\n"
            "- Định dạng số có **dấu phẩy** (vd 31,000). Không coi đây là khuyến nghị đầu tư."
        )
        
        analysis_prompt = st.text_area(
            "Hướng dẫn chi tiết cho AI",
            value=st.session_state.get("analysis_prompt", default_prompt),
            height=200,
            help="Hướng dẫn cho AI về cách phân tích. Sử dụng {template} để chèn template.",
            key="prompt_area"
        )
        st.session_state["analysis_prompt"] = analysis_prompt
        
        # System prompt configuration
        default_system_prompt = "Bạn là chuyên gia phân tích kỹ thuật cổ phiếu Việt Nam, viết báo cáo theo template được cung cấp."
        
        system_prompt = st.text_area(
            "System Prompt (Vai trò AI)",
            value=st.session_state.get("system_prompt", default_system_prompt),
            height=80,
            help="Vai trò và ngữ cảnh cho AI.",
            key="system_prompt_area"
        )
        st.session_state["system_prompt"] = system_prompt
        
        if st.button("🔄 Reset Prompts", help="Khôi phục prompts mặc định", key="reset_prompts"):
            st.session_state["analysis_prompt"] = default_prompt
            st.session_state["system_prompt"] = default_system_prompt
            st.rerun()
    
    # Preview section
    st.markdown("---")
    st.markdown("**👁️ Xem trước cấu hình hiện tại:**")
    
    col_preview1, col_preview2 = st.columns(2)
    with col_preview1:
        with st.container(border=True):
            st.markdown("**Template sẽ sử dụng:**")
            st.code(st.session_state.get("analysis_template", default_template)[:200] + "...", language="text")
    
    with col_preview2:
        with st.container(border=True):
            st.markdown("**System Prompt:**")
            st.code(st.session_state.get("system_prompt", default_system_prompt), language="text")
    
    # Template presets
    st.markdown("---")
    st.markdown("**🎨 Template có sẵn:**")
    
    col_preset1, col_preset2, col_preset3 = st.columns(3)
    
    with col_preset1:
        if st.button("📊 Template Cơ bản", help="Template phân tích cơ bản", key="preset_basic"):
            st.session_state["analysis_template"] = default_template
            st.rerun()
    
    with col_preset2:
        if st.button("📈 Template Chi tiết", help="Template phân tích chi tiết hơn", key="preset_detailed"):
            detailed_template = (
                "PHÂN TÍCH KỸ THUẬT CHI TIẾT\n\n"
                "1. Tổng quan thị trường:\n- Xu hướng tổng thể: ...\n- Vị trí trong chu kỳ: ...\n\n"
                "2. Phân tích giá:\n- Xu hướng ngắn hạn (1-5 ngày): ...\n- Xu hướng trung hạn (1-4 tuần): ...\n- Xu hướng dài hạn (1-3 tháng): ...\n\n"
                "3. Đường trung bình động:\n- SMA9: ...\n- SMA20: ...\n- SMA50: ...\n- SMA200: ...\n\n"
                "4. Khối lượng giao dịch:\n- Khối lượng hiện tại vs TB20: ...\n- Khối lượng hiện tại vs TB60: ...\n- Đánh giá thanh khoản: ...\n\n"
                "5. Hỗ trợ & Kháng cự:\n- Kháng cự gần nhất: ...\n- Kháng cự mạnh: ...\n- Hỗ trợ gần nhất: ...\n- Hỗ trợ mạnh: ...\n\n"
                "6. Chỉ báo kỹ thuật:\n- RSI: ...\n- ATR: ...\n- Đỉnh/đáy 52 tuần: ...\n\n"
                "CHIẾN LƯỢC ĐẦU TƯ\n\n"
                "• Ngắn hạn (1-2 tuần):\n- Xu hướng: ...\n- Vùng mua: ...\n- Stop-loss: ...\n- Take-profit: ...\n\n"
                "• Trung hạn (1-3 tháng):\n- Xu hướng: ...\n- Vùng tích lũy: ...\n- Mục tiêu: ...\n\n"
                "• Rủi ro cần lưu ý: ..."
            )
            st.session_state["analysis_template"] = detailed_template
            st.rerun()
    
    with col_preset3:
        if st.button("⚡ Template Nhanh", help="Template báo cáo ngắn gọn", key="preset_quick"):
            quick_template = (
                "PHÂN TÍCH NHANH\n\n"
                "📈 Xu hướng: ...\n"
                "📊 MA20/50/200: ... / ... / ...\n"
                "📦 Khối lượng: ...\n"
                "🔺 Kháng cự: ...\n"
                "🔻 Hỗ trợ: ...\n\n"
                "⚡ CHIẾN LƯỢC:\n"
                "• Ngắn hạn: ...\n"
                "• Vùng mua: ...\n"
                "• Stop-loss: ...\n"
                "• Mục tiêu: ..."
            )
            st.session_state["analysis_template"] = quick_template
            st.rerun()
    
    # Import/Export configuration
    st.markdown("---")
    st.markdown("**💾 Sao lưu & Khôi phục cấu hình:**")
    
    col_export, col_import = st.columns(2)
    
    with col_export:
        if st.button("📤 Export cấu hình", help="Xuất cấu hình hiện tại", key="export_config"):
            config_data = {
                "analysis_template": st.session_state.get("analysis_template", default_template),
                "analysis_prompt": st.session_state.get("analysis_prompt", default_prompt),
                "system_prompt": st.session_state.get("system_prompt", default_system_prompt)
            }
            config_json = json.dumps(config_data, ensure_ascii=False, indent=2)
            st.download_button(
                label="⬇️ Tải file cấu hình",
                data=config_json,
                file_name="ai_analysis_config.json",
                mime="application/json"
            )
    
    with col_import:
        uploaded_config = st.file_uploader(
            "📤 Import cấu hình", 
            type=["json"], 
            help="Tải lên file cấu hình đã xuất trước đó",
            key="import_config"
        )
        
        if uploaded_config is not None:
            try:
                config_data = json.load(uploaded_config)
                st.session_state["analysis_template"] = config_data.get("analysis_template", default_template)
                st.session_state["analysis_prompt"] = config_data.get("analysis_prompt", default_prompt)
                st.session_state["system_prompt"] = config_data.get("system_prompt", default_system_prompt)
                st.success("✅ Đã import cấu hình thành công!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Lỗi khi import cấu hình: {e}")

# =========================
# Resolve dates & symbols with default values
# =========================
# Set default values for removed variables
max_symbols = 200  # Không giới hạn số mã
min_adtv = 0  # Không lọc thanh khoản
sources_str = "TCBS"  # Nguồn mặc định
show_charts = True  # Luôn hiển thị biểu đồ
show_ma9 = True  # SMA9 mặc định hiển thị
show_ma20 = True  # SMA20 mặc định hiển thị
show_ma50 = True  # SMA50 mặc định hiển thị
show_ma200 = True  # SMA200 mặc định hiển thị
show_bollinger = True  # Bollinger Bands mặc định hiển thị

# Lấy giá trị từ config hoặc dùng mặc định
if 'days' not in locals():
    days = 360
if 'tminus' not in locals():
    tminus = 0
if 'end_date' not in locals():
    end_date = ""
if 'polite_delay_ms' not in locals():
    polite_delay_ms = 300

if tminus and tminus > 0:
    ed = _prev_weekday(datetime.today() - timedelta(days=int(tminus)))
else:
    ed = _prev_weekday(datetime.today())
    if end_date.strip():
        try:
            ed = _prev_weekday(pd.to_datetime(end_date))
        except Exception:
            st.warning("Ngày kết thúc không hợp lệ, dùng ngày làm việc gần nhất.")
ed_str = ed.strftime("%Y-%m-%d")
sources = [s.strip().upper() for s in (sources_str or "TCBS").split(",") if s.strip()]

# Lấy mã từ watchlist đang chọn
active_wl_name = st.session_state.current_watchlist
symbols_all = _filter_company_tickers_only(st.session_state.watchlists.get(active_wl_name, []))
symbols = symbols_all[:max_symbols]
if len(symbols_all) > len(symbols):
    st.info(f"Đã giới hạn {len(symbols)}/{len(symbols_all)} mã để tránh rate-limit.")

# =========================
# Main: run analysis once & store
# =========================
def run_analysis_and_store():
    if not symbols:
        st.error("Danh sách hiện trống hoặc không có mã hợp lệ (3 ký tự chữ cái).")
        return

    st.info(f"Đang phân tích **{len(symbols)}** mã từ watchlist **{active_wl_name}** · days={days} · end={ed_str} · sources={sources}")

    primary_source_for_fund = sources[0] if sources else "TCBS"
    snap_df = _get_screener_snapshot_cached(primary_source_for_fund)  # chỉ để lấy sector nếu có
    sector_map = _extract_sector_map_from_snapshot(snap_df)  # có thể rỗng nếu không lấy được

    rows_feat: List[Dict] = []
    px_map: Dict[str, pd.DataFrame] = {}

    try:
        prog = st.progress(0, text="Chưa bắt đầu")
        prog_text = None
    except TypeError:
        prog = st.progress(0)
        prog_text = st.empty()

    price_sources = [sources[0]] if sources else ["TCBS"]

    for i, sym in enumerate(symbols, 1):
        label = f"Đang tải dữ liệu {i}/{len(symbols)} - {sym}"
        try:
            prog.progress(i/len(symbols), text=label)
        except TypeError:
            prog.progress(i/len(symbols))
            if prog_text: prog_text.markdown(f"**{label}**")

        # Giá
        try:
            px = _get_quote_history_cached(sym, int(days), ed_str, price_sources)
        except Exception:
            px = pd.DataFrame()
        px_map[sym] = px

        # Fundamentals (rev_yoy từ TCBS API – xử lý trong adapter)
        fund_precise = _get_fundamentals_precise_cached(sym, source=primary_source_for_fund, period="quarter")

        # Tính features từ px + fund
        feat = {"symbol": sym}
        feat.update(compute_features_local_from_px(px, fund_precise))
        rows_feat.append(feat)

        if polite_delay_ms > 0:
            time.sleep(polite_delay_ms / 1000.0)

    df_feat = pd.DataFrame(rows_feat)

    # Thêm sector (nếu có)
    if not df_feat.empty:
        df_feat["sector"] = df_feat["symbol"].map(lambda s: sector_map.get(str(s).upper(), ""))

    # Bảo đảm các cột có tồn tại
    for col in ["pe","pb","roe","net_margin","de","rev_yoy","eps_yoy","m1","m3","m6","adtv","vol"]:
        if col not in df_feat.columns:
            df_feat[col] = np.nan
        if df_feat[col].notna().sum() == 0:
            df_feat[col] = 0.0

    ranked = score_symbols(df_feat, ScoreWeights(), min_adtv=float(min_adtv))

    st.session_state["screener_store"] = {
        "ranked": ranked, "df_feat": df_feat, "px_map": px_map,
        "ed_str": ed_str, "sources": sources, "params": dict(days=days, min_adtv=min_adtv),
        "snapshot_df": snap_df,
        "active_watchlist": active_wl_name,
    }
    if "form_cache" not in st.session_state:
        st.session_state["form_cache"] = {}

def auto_analyze_all_symbols_in_background():
    """Tự động phân tích tất cả các mã trong nền nếu có API key"""
    api_key = st.session_state.get("openai_api_key", "") or ""
    if not api_key or not _OPENAI_OK:
        return
    
    store = st.session_state.get("screener_store", {})
    if not store:
        return
        
    ranked = store.get("ranked", pd.DataFrame())
    if ranked.empty:
        return
        
    # Lấy top symbols
    top_syms = ranked.head(20)["symbol"].tolist()
    px_map = store.get("px_map", {})
    snapshot_df = store.get("snapshot_df", pd.DataFrame())
    
    # Đếm số mã chưa có báo cáo
    cached_reports = st.session_state.get("form_cache", {})
    symbols_to_analyze = [sym for sym in top_syms if sym not in cached_reports]
    
    if not symbols_to_analyze:
        st.success("✅ Tất cả các mã đã có báo cáo AI!")
        return
    
    # Hiển thị progress
    st.info(f"🤖 Đang tự động tạo báo cáo AI cho {len(symbols_to_analyze)} mã trong nền...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Phân tích từng mã
    for i, symbol in enumerate(symbols_to_analyze):
        status_text.text(f"Đang phân tích {symbol} ({i+1}/{len(symbols_to_analyze)})...")
        
        # Lấy dữ liệu kỹ thuật
        px_sel = px_map.get(symbol, pd.DataFrame())
        if px_sel.empty:
            continue
            
        tech_stats = build_structured_stats(px_sel)
        company_name = _company_name_from_snapshot(snapshot_df, symbol)
        
        # Tạo báo cáo
        try:
            model = st.session_state.get("llm_model", "gpt-4o-mini")
            template = st.session_state.get("analysis_template", "")
            prompt = st.session_state.get("analysis_prompt", "")
            system_prompt = st.session_state.get("system_prompt", "")
            
            report = call_llm_structured_report(
                api_key, model, symbol, tech_stats,
                template=template, prompt=prompt, system_prompt=system_prompt, company_name=company_name
            )
            st.session_state.setdefault("form_cache", {})[symbol] = report
        except Exception as e:
            st.warning(f"⚠️ Lỗi tạo báo cáo cho {symbol}: {e}")
        
        # Cập nhật progress
        progress_bar.progress((i + 1) / len(symbols_to_analyze))
    
    status_text.text("Hoàn thành!")
    progress_bar.progress(1.0)
    st.success(f"🎉 Đã tạo xong báo cáo AI cho {len(symbols_to_analyze)} mã!")

if analyze_btn:
    with st.spinner("🔄 Đang phân tích dữ liệu..."):
        run_analysis_and_store()
    
    # Tự động phân tích AI trong nền nếu có API key
    auto_analyze_all_symbols_in_background()

# =========================
# Enhanced Results Display
# =========================
store = st.session_state.get("screener_store")

if store is None:
    st.markdown("""
    <div class="metric-card" style="text-align: center; padding: 2rem;">
        <h4>👋 Chào mừng đến với FinStock </h4>
        <p>Vui lòng chọn danh sách cổ phiếu trong sidebar và nhấn nút <strong>🚀 Bắt đầu phân tích</strong> để bắt đầu.</p>
        <br>
        <div class="status-indicator status-warning"></div>
        <small>Chưa có dữ liệu để hiển thị</small>
    </div>
    """, unsafe_allow_html=True)
else:
    ranked = store["ranked"]; px_map = store["px_map"]
    
    # Summary metrics
    total_analyzed = len(ranked)
    top_performer = ranked.iloc[0]['symbol'] if len(ranked) > 0 else "N/A"
    
    st.markdown(f"""
    <div class="section-header">
        <h3>🏆 Kết quả phân tích - {store.get('active_watchlist','Danh sách')}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Kết quả phân tích gom gọn trong 1 dòng
    analyzed_date = store.get('ed_str', 'N/A')
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%); padding: 0.8rem 1.2rem; border-radius: 8px; border-left: 4px solid #2a5298; margin-bottom: 1rem;">
        <p style="margin: 0; font-size: 1rem; color: #495057;">
            <strong>📊 {total_analyzed}</strong> mã cổ phiếu đã phân tích • 
            <strong style="color: #28a745;">🥇 {top_performer}</strong> dẫn đầu • 
            <strong>📅 {analyzed_date}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Enhanced ranking table
    st.markdown("""
    <div class="section-header">
        <h4>📋 Bảng xếp hạng chi tiết</h4>
    </div>
    """, unsafe_allow_html=True)
    
    view = ranked.copy()

    # ---- format helpers ----
    def pct_return(x):  # m1/m3/m6 là tỉ lệ (0.123)
        return "" if pd.isna(x) else f"{x*100.0:.1f}%"

    def pct_ready(x):   # đã là % (vd 29.7)
        return "" if pd.isna(x) else f"{x:.1f}%"

    # Score & component scores
    for c in ("score","Value","Quality","Growth","Momentum","Liquidity","RiskAdj","pe","pb"):
        if c in view.columns:
            view[c] = view[c].apply(lambda x: "" if pd.isna(x) else (f"{x:.3f}" if c=="score" else f"{x:.2f}"))

    # Returns: m1/m3/m6 (tỉ lệ → *100 khi hiển thị)
    for c in ("m1","m3","m6"):
        if c in view.columns:
            view[c] = view[c].apply(pct_return)

    # % metrics đã chuẩn hoá sẵn trong adapter: rev_yoy, roe, eps_yoy, net_margin
    for c in ("rev_yoy","roe","eps_yoy","net_margin"):
        if c in view.columns:
            view[c] = view[c].apply(pct_ready)

    # ADTV - Debug thông tin
    if "adtv" in view.columns:
        # Đếm số mã có ADTV hợp lệ - dùng adtv_raw nếu có
        adtv_col = "adtv_raw" if "adtv_raw" in view.columns else "adtv"
        valid_adtv = view[adtv_col].apply(lambda x: not pd.isna(x) and x > 0).sum()
        total_stocks = len(view)
        
        # Kiểm tra khoảng giá trị để quyết định đơn vị hiển thị
        max_adtv = view[adtv_col].max() if not view[adtv_col].empty else 0
        
        if max_adtv > 1e9:  # Lớn hơn 1 tỷ → hiển thị theo tỷ
            view["adtv"] = view[adtv_col].apply(lambda x: "N/A" if pd.isna(x) or x <= 0 else f"{x/1e9:.1f} tỷ")
        elif max_adtv > 1e6:  # Lớn hơn 1 triệu → hiển thị theo triệu
            view["adtv"] = view[adtv_col].apply(lambda x: "N/A" if pd.isna(x) or x <= 0 else f"{x/1e6:.1f} tr")
        elif max_adtv > 1e3:  # Lớn hơn 1 nghìn → hiển thị theo nghìn
            view["adtv"] = view[adtv_col].apply(lambda x: "N/A" if pd.isna(x) or x <= 0 else f"{x/1e3:.1f} k")
        else:  # Nhỏ hơn → hiển thị nguyên giá trị
            view["adtv"] = view[adtv_col].apply(lambda x: "N/A" if pd.isna(x) or x <= 0 else f"{x:.2f}")

    # Thêm cột STT mới (đánh số từ 1)
    view.reset_index(drop=True, inplace=True)
    view["stt"] = range(1, len(view) + 1)

    cols = [c for c in [
        "stt","symbol","score","Value","Quality","Growth","Momentum","Liquidity","RiskAdj",
        "m1","m3","m6","pe","pb","roe","rev_yoy","eps_yoy","net_margin","adtv"
    ] if c in view.columns]

    # ---- Column help / tooltips ----
    col_help = {
        "stt": "Số thứ tự xếp hạng dựa trên điểm tổng.",
        "symbol": "Mã cổ phiếu.",
        "score": "Điểm tổng hợp theo trọng số: Value(0.22), Quality(0.22), Growth(0.20), Momentum(0.20), Liquidity(0.10), Risk(0.06). Điểm cao = cân bằng hấp dẫn.",
        "Value": "Định giá tương đối (P/E, P/B). Cao → rẻ tương đối sau khi so sánh với nhóm ngành/toàn thị trường",
        "Quality": "Chất lượng tài chính (ROE↑, biên ròng↑, D/E↓). Cao → chất lượng tài chính tốt & đòn bẩy hợp lý.",
        "Growth": "Tổng hợp Rev YoY, EPS YoY. Cao tốt.",
        "Momentum": "Xu hướng giá (1–3–6 tháng). Dương/tốt → điểm cao.",
        "Liquidity": "Thanh khoản (ADTV). Cao dễ giao dịch.",
        "RiskAdj": "Điểm rủi ro điều chỉnh theo biến động (vol thấp được cộng điểm).",
        "m1": "Hiệu suất ~1 tháng (%). Dương → tăng. Âm → giảm",
        "m3": "Hiệu suất ~3 tháng (%). Dương → tăng. Âm → giảm",
        "m6": "Hiệu suất ~6 tháng (%). Dương → tăng. Âm → giảm",
        "pe": "P/E — thấp thường rẻ hơn tương đối (xem kèm Quality/Growth).",
        "pb": "P/B — thấp có thể rẻ hơn tài sản ròng (tùy ngành).",
        "roe": "ROE (%) — Tỷ suất sinh lời trên vốn CSH (đã chuẩn hoá thành %). Cao → chất lượng lợi nhuận tốt.",
        "rev_yoy": "Doanh thu YoY quý mới nhất (%) — Cao → tăng trưởng doanh thu tốt.",
        "eps_yoy": "Tăng trưởng EPS theo năm (%) — Cao → lợi nhuận/cổ phiếu tăng.",
        "net_margin": "Biên lợi nhuận ròng (%). Cao → doanh nghiệp giữ lại nhiều lợi nhuận từ doanh thu.",
        "adtv": "Giá trị giao dịch TB 20 phiên (VND). Cao → thanh khoản tốt.",
        "sector": "Ngành (dùng để chuẩn hoá theo ngành).",
    }
    from streamlit import column_config as cc
    column_config = {
        "stt": cc.NumberColumn("STT", help=col_help["stt"], width="small", format="%d"),
        "symbol": cc.TextColumn("Mã CP", help=col_help["symbol"]),
        "score": cc.TextColumn("Điểm tổng", help=col_help["score"]),
        "Value": cc.TextColumn("Định giá", help=col_help["Value"]),
        "Quality": cc.TextColumn("Chất lượng", help=col_help["Quality"]),
        "Growth": cc.TextColumn("Tăng trưởng", help=col_help["Growth"]),
        "Momentum": cc.TextColumn("Xu hướng", help=col_help["Momentum"]),
        "Liquidity": cc.TextColumn("Thanh khoản", help=col_help["Liquidity"]),
        "RiskAdj": cc.TextColumn("Rủi ro", help=col_help["RiskAdj"]),
        "m1": cc.TextColumn("1 tháng (%)", help=col_help["m1"]),
        "m3": cc.TextColumn("3 tháng (%)", help=col_help["m3"]),
        "m6": cc.TextColumn("6 tháng (%)", help=col_help["m6"]),
        "pe": cc.TextColumn("P/E", help=col_help["pe"]),
        "pb": cc.TextColumn("P/B", help=col_help["pb"]),
        "roe": cc.TextColumn("ROE (%)", help=col_help["roe"]),
        "rev_yoy": cc.TextColumn("DT YoY (%)", help=col_help["rev_yoy"]),
        "eps_yoy": cc.TextColumn("EPS YoY (%)", help=col_help["eps_yoy"]),
        "net_margin": cc.TextColumn("Biên LN (%)", help=col_help["net_margin"]),
        "adtv": cc.TextColumn("GTGD TB", help=col_help["adtv"]),
        "sector": cc.TextColumn("Ngành", help=col_help["sector"]),
    }

    st.dataframe(
        view[cols],
        use_container_width=True,
        height=380,
        column_config=column_config,
        hide_index=True
    )
    st.caption("Mẹo: Di chuột vào biểu tượng ⓘ cạnh tiêu đề cột để xem chú thích nhanh.")

    # Ghi chú nhanh về Z-Score
    with st.expander("📚 Phương pháp Z-Score"):
        st.markdown(
            """
- **Z-Score là gì?**: Phương pháp "chấm điểm" để so sánh các chỉ số khác nhau của cổ phiếu một cách công bằng.
- **Tại sao cần Z-Score?**: VD: Làm sao so sánh P/E=15 lần vs Tăng trưởng=20% vs ADTV=10 tỷ? Z-Score đưa tất cả về cùng thang điểm.
- **Cách tính**: Z = (Giá trị cổ phiếu - Trung bình nhóm) / Độ lệch chuẩn. Kết quả: Z=0 (trung bình), Z=+1 (tốt hơn 68% nhóm), Z=+2 (tốt hơn 95% nhóm).
- **Điểm tổng hợp**: Score = Trung bình có trọng số của 6 nhóm: Value(22%), Quality(22%), Growth(20%), Momentum(20%), Liquidity(10%), Risk(6%).
  - Các chỉ tiêu “**càng thấp càng tốt**” (P/E, P/B, D/E) được **đảo dấu** để điểm cao = tốt.
- **Giải thích điểm**:
  - Z ≈ **0**: ngang trung vị nhóm so sánh; **+1**: tốt hơn đáng kể; **−1**: kém hơn đáng kể.
  - **score** = tổng hợp có trọng số: **Value(0.22)**, **Quality(0.22)**, **Growth(0.20)**, **Momentum(0.20)**, **Liquidity(0.10)**, **Risk(0.06)**.
- **🎯 Kết luận**: Cổ phiếu có điểm Z-Score cao nhất = Tổng hợp tốt nhất trên tất cả tiêu chí!
            """
        )

    # ---- Kết luận ----
    st.subheader("✅ Kết luận (Top pick)")
    if ranked.empty:
        st.warning("Không có mã vượt ngưỡng thanh khoản hoặc đủ dữ liệu.")
    else:
        best = ranked.iloc[0]
        reasons = []
        for comp in ("Value","Quality","Growth","Momentum","Liquidity","RiskAdj"):
            val = best.get(comp, np.nan)
            if not pd.isna(val):
                reasons.append(f"{comp}={float(val):.2f}")
        sec = best.get("sector", "") or ""
        sec_txt = f" · Sector: {sec}" if sec else ""
        st.success(f"**{best['symbol']}** là mã phù hợp nhất{sec_txt}. Lý do: " + "; ".join(reasons))

    if show_charts and not ranked.empty:
        st.subheader("📊 Biểu đồ & bảng lịch sử (chọn mã)")

        top_syms = list(ranked["symbol"].head(min(10, len(ranked))))
        # Lưu top_syms để sử dụng trong auto-advance
        st.session_state["top_syms"] = top_syms
        if "selected_symbol" not in st.session_state:
            st.session_state["selected_symbol"] = top_syms[0]

        selected_from_top = st.radio(
            "Top 10 theo điểm:",
            options=top_syms,
            index=top_syms.index(st.session_state["selected_symbol"]) if st.session_state["selected_symbol"] in top_syms else 0,
            horizontal=True,
            key="sym_radio",
            help="Chọn nhanh một mã trong Top 10 theo điểm."
        )
        st.session_state["selected_symbol"] = selected_from_top

        def _on_enter_symbol():
            s = st.session_state.get("manual_symbol_input", "").strip().upper()
            if s:
                st.session_state["selected_symbol"] = s

        st.text_input("Hoặc nhập mã khác rồi nhấn Enter:", value="", placeholder="Ví dụ: FPT",
                      key="manual_symbol_input", on_change=_on_enter_symbol)

        selected_symbol = st.session_state["selected_symbol"]
        px_sel = px_map.get(selected_symbol)
        if (px_sel is None) or px_sel.empty:
            try:
                price_sources = [store["sources"][0]] if store.get("sources") else ["TCBS"]
                px_sel = _get_quote_history_cached(selected_symbol, int(store["params"]["days"]), store["ed_str"], price_sources)
                st.session_state["screener_store"]["px_map"][selected_symbol] = px_sel
            except Exception:
                px_sel = pd.DataFrame()

        chart_title = f"{selected_symbol}"

        if px_sel is None or px_sel.empty:
            st.info("Không có dữ liệu giá để vẽ.")
        else:
            # Thông tin hướng dẫn sử dụng biểu đồ
            with st.expander("💡 Hướng dẫn tương tác với biểu đồ", expanded=False):
                st.markdown("""
                **🎯 Tính năng Auto-scaling trục Y:**
                - ✅ Trục Y sẽ **tự động co giãn** khi bạn thay đổi khoảng thời gian
                - 🔍 Sử dụng các nút **1M, 3M, 6M, All** để thay đổi range nhanh
                - 📊 Trục Y tự động tối ưu hiển thị theo giá cao nhất/thấp nhất trong khoảng đã chọn
                - 🖱️ **Double-click** trên biểu đồ để reset về trạng thái ban đầu
                
                **🔍 Tính năng Zoom nâng cao:**
                - 🖱️ **Cuộn chuột** để zoom in/out trực tiếp trên biểu đồ
                - ⚡ Kéo thả để zoom vùng cụ thể
                - 📏 Sử dụng range slider ở dưới để điều hướng nhanh
                - 🎛️ Có thể nhập range trực tiếp vào các ô input
                
                **📈 Tương tác khác:**
                - Hover để xem thông tin chi tiết tại từng điểm
                - Click vào legend để ẩn/hiện các đường kỹ thuật
                - Kéo trục để pan (di chuyển) biểu đồ
                """)

            # Tạo layout 2 cột: biểu đồ và báo cáo AI
            chart_col, ai_col = st.columns([2, 1])
            
            # Cột bên trái: Biểu đồ
            with chart_col:
                fig = make_ohlcv_figure(
                    px_sel, chart_title,
                    default_months_view=3, right_pad_months=2, height=850,
                    show_ma9=show_ma9, show_ma20=show_ma20, show_ma50=show_ma50, 
                    show_ma200=show_ma200, show_bollinger=show_bollinger
                )
                
                # Cấu hình plotly để có thể tương tác tốt hơn
                plotly_config = {
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
                    "modeBarButtonsToAdd": ["resetScale2d"],
                    "showTips": True,
                    "displayModeBar": True,
                    "responsive": True,
                    "doubleClick": "reset+autosize",  # Double click để reset về auto-scale
                    "scrollZoom": True,  # Bật zoom bằng cuộn chuột
                    "showAxisDragHandles": True,  # Hiển thị handles để kéo trục
                    "showAxisRangeEntryBoxes": True  # Hiển thị box để nhập range trực tiếp
                }
                
                st.plotly_chart(fig, use_container_width=True, config=plotly_config)
            
            # Cột bên phải: Báo cáo AI
            with ai_col:
                # Kiểm tra trạng thái báo cáo
                current_analyzing_symbol = st.session_state.get("selected_symbol", "")
                api_key = st.session_state.get("openai_api_key", "") or ""
                cached_reports = st.session_state.get("form_cache", {})
                has_report = current_analyzing_symbol in cached_reports
                
                # Auto-generate report if conditions are met
                api_key = st.session_state.get("openai_api_key", "") or ""
                if api_key and not has_report:
                        
                        # Lấy tech_stats cho symbol hiện tại
                        px_sel_current = st.session_state.get("screener_store", {}).get("px_map", {}).get(current_analyzing_symbol)
                        if px_sel_current is not None and not px_sel_current.empty:
                            tech_stats_current = build_structured_stats(px_sel_current)
                            snapshot_df = store.get("snapshot_df", pd.DataFrame())
                            company_name = _company_name_from_snapshot(snapshot_df, current_analyzing_symbol)
                            
                            key = st.session_state.get("openai_api_key", "")
                            model = "gpt-4o-mini"
                            template = st.session_state.get("analysis_template", "")
                            prompt = st.session_state.get("analysis_prompt", "")
                            system_prompt = st.session_state.get("system_prompt", "")
                            
                            with st.spinner("🤖 Đang tạo..."):
                                report = call_llm_structured_report(
                                    key, model, current_analyzing_symbol, tech_stats_current,
                                    template=template, prompt=prompt, system_prompt=system_prompt, company_name=company_name
                                )
                                st.session_state.setdefault("form_cache", {})[current_analyzing_symbol] = report
                            st.rerun()
                
                # Hiển thị báo cáo AI
                form_text = cached_reports.get(current_analyzing_symbol)
                if form_text:
                    # Hiển thị báo cáo trong container với scroll
                    with st.container(height=780):
                        st.markdown(form_text)
                    
                    # Download button compact
                    st.download_button(
                        label="⬇️ Tải báo cáo",
                        data="\ufeff" + form_text,
                        file_name=f"{current_analyzing_symbol}_PTKT.txt",
                        mime="text/plain; charset=utf-8",
                        use_container_width=True
                    )
                elif not api_key:
                    st.info("💡 Nhập OpenAI API Key để sử dụng báo cáo AI")

            # ====== 📜 Lịch sử giá (có thể ẩn/hiện) ======
            st.markdown("---")
            
            # Tạo expander cho lịch sử giá với tính năng ẩn/hiện
            with st.expander("📜 Lịch sử giá chi tiết", expanded=False):
                # Tùy chọn số phiên hiển thị
                col_option1, col_option2 = st.columns([1, 3])
                
                with col_option1:
                    num_sessions = st.number_input(
                        "Số phiên:",
                        min_value=10,
                        max_value=500,
                        value=50,
                        step=10,
                        key="history_sessions_input",
                        help="Nhập số phiên muốn hiển thị (10-500)"
                    )
                
                with col_option2:
                    st.markdown(f"**Dữ liệu giá {num_sessions} phiên gần nhất**")
                
                # Format dữ liệu để hiển thị đẹp (auto hiển thị tất cả các cột)
                px_display = px_sel.sort_values("date", ascending=False).head(num_sessions).copy()
                
                # Format ngày theo kiểu dd/mm/yyyy
                if "date" in px_display.columns:
                    px_display["date"] = pd.to_datetime(px_display["date"]).dt.strftime("%d/%m/%Y")
                
                # Format các cột giá với 2 chữ số thập phân
                price_cols = ["open", "high", "low", "close"]
                for col in price_cols:
                    if col in px_display.columns:
                        px_display[col] = px_display[col].apply(
                            lambda x: f"{float(x):,.2f}" if pd.notna(x) else "N/A"
                        )
                
                # Format volume và value nếu có
                if "volume" in px_display.columns:
                    px_display["volume"] = px_display["volume"].apply(
                        lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "0"
                    )
                
                if "value" in px_display.columns:
                    px_display["value"] = px_display["value"].apply(
                        lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "0"
                    )
                
                # Column config tự động cho tất cả các cột
                auto_column_config = {
                    "date": cc.TextColumn("Ngày", width="small"),
                    "open": cc.TextColumn("Mở cửa", width="small"),
                    "high": cc.TextColumn("Cao nhất", width="small"),
                    "low": cc.TextColumn("Thấp nhất", width="small"), 
                    "close": cc.TextColumn("Đóng cửa", width="small"),
                    "volume": cc.TextColumn("Khối lượng", width="medium"),
                    "value": cc.TextColumn("Giá trị GD", width="medium")
                }
                
                # Chỉ giữ config cho các cột thực sự có trong data
                final_column_config = {k: v for k, v in auto_column_config.items() if k in px_display.columns}
                
                # Tự động điều chỉnh chiều cao theo số phiên
                table_height = min(500, max(200, num_sessions * 8 + 50))
                
                # Hiển thị bảng
                st.dataframe(
                    px_display, 
                    use_container_width=True, 
                    height=table_height, 
                    column_config=final_column_config,
                    hide_index=True
                )
                
                # Thống kê tóm tắt (thêm metric giá đóng cửa gần nhất)
                st.markdown(f"**📊 Thống kê {num_sessions} phiên gần nhất:**")
                col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
                
                if len(px_sel) > 0:
                    # Lấy dữ liệu theo số phiên đã chọn, sắp xếp theo thời gian tăng dần để tính đúng
                    period_data = px_sel.tail(num_sessions).sort_values("date", ascending=True)
                    
                    with col_stat1:
                        # Giá đóng cửa gần nhất (phiên mới nhất)
                        latest_close = period_data["close"].iloc[-1]
                        st.metric("Giá hiện tại", f"{latest_close:,.2f}")
                    
                    with col_stat2:
                        highest = period_data["high"].max()
                        st.metric("Cao nhất", f"{highest:,.2f}")
                    
                    with col_stat3:
                        lowest = period_data["low"].min()
                        st.metric("Thấp nhất", f"{lowest:,.2f}")
                    
                    with col_stat4:
                        if len(period_data) >= 2:
                            # Giá đầu kỳ (cũ nhất) và cuối kỳ (mới nhất) 
                            first_close = period_data["close"].iloc[0]  # Phiên cũ nhất
                            last_close = period_data["close"].iloc[-1]  # Phiên mới nhất
                            change_pct = ((last_close - first_close) / first_close) * 100
                            st.metric("Biến động", f"{change_pct:+.2f}%")
                        else:
                            st.metric("Biến động", "N/A")
                    
                    with col_stat5:
                        if "volume" in period_data.columns:
                            avg_volume = period_data["volume"].mean()
                            st.metric("KL TB", f"{avg_volume:,.0f}")
                        else:
                            st.metric("KL TB", "N/A")

            # ====== Tin tức trong 7 ngày gần đây (TCBS) ======
            st.markdown("### Tin tức trong 7 ngày gần đây (TCBS)")
            raw = fetch_activity_news_raw(st.session_state.get("selected_symbol", ""), size=100)
            recent_items = filter_recent_activity_news(raw, recent_days=7)
            if not recent_items:
                st.markdown("_Không thấy công bố trong 7 ngày gần đây_")
            else:
                for it in recent_items:
                    ts_str = it["published_at"].strftime("%Y-%m-%d %H:%M")
                    src = it.get("source") or ""
                    title = it["title"]
                    st.markdown(f"- {ts_str} · {src} — {title}")

# ===== FOOTER =====
st.caption("(*) Công cụ sàng lọc định lượng mang tính tham khảo. Không phải khuyến nghị đầu tư. Kết hợp thêm phân tích ngành, catalyst và quản trị rủi ro.")
