# pick_best_by_symbol.py
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Optional: requests để gọi TCBS API trực tiếp cho rev_yoy
try:
    import requests  # type: ignore
except Exception:
    requests = None  # fallback: nếu thiếu requests sẽ bỏ qua TCBS API

# =========================
# Public exports
# =========================
__all__ = [
    "VnAdapter", "VnstockAdapterError",
    "ScoreWeights", "score_symbols", "score_symbols_v2",
    "compute_features",
    "_calc_adtv_vnd",
]

# =========================
# Common helpers
# =========================
def _prev_weekday(dt: datetime) -> datetime:
    while dt.weekday() >= 5:  # 5=Sat 6=Sun
        dt -= timedelta(days=1)
    return dt

def _safe_pct(a, b):
    try:
        if b in (0, None) or pd.isna(b): return np.nan
        return (float(a) - float(b)) / abs(float(b))
    except Exception:
        return np.nan

def _annualized_vol(returns: pd.Series):
    d = returns.dropna()
    return d.std(ddof=0) * math.sqrt(252) if len(d) > 1 else np.nan

def _winsorize(s: pd.Series, p=0.02):
    if s is None or len(s) == 0: return s
    try:
        lo, hi = s.quantile(p), s.quantile(1 - p)
        return s.clip(lower=lo, upper=hi)
    except Exception:
        return s

def _zscore(s: pd.Series):
    s = s.astype(float)
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0: return s * 0.0
    return (s - s.mean()) / sd

def _mad(s: pd.Series):
    med = s.median()
    return (s - med).abs().median()

def _robust_zscore(s: pd.Series):
    s = s.astype(float)
    med = s.median()
    mad = _mad(s)
    if pd.isna(mad) or mad == 0:
        return _zscore(s)  # fallback
    # 1.4826 ≈ Phi^-1(0.75)
    return (s - med) / (1.4826 * mad)

def _groupwise_apply(s: pd.Series, groups: pd.Series, fn):
    g = pd.Series(groups, index=s.index)
    out = []
    for _, idx in g.groupby(g).groups.items():
        out.append(fn(s.loc[idx]))
    return pd.concat(out).sort_index()

def _impute_by_group(df: pd.DataFrame, col: str, groupcol: Optional[str]):
    if col not in df.columns:
        df[col] = np.nan
    if groupcol and groupcol in df.columns:
        df[col] = df.groupby(groupcol)[col].transform(lambda x: x.fillna(x.median()))
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
    return df

def _norm_pct(x):
    """Quy về phần trăm: nếu giá trị là tỷ lệ thập phân (-1..1) => x*100."""
    try:
        x = float(x)
    except Exception:
        return np.nan
    return x * 100.0

def _calc_adtv_vnd(px: pd.DataFrame, n: int = 20) -> float:
    """ADTV(n) theo VND = mean(close × volume) trên n phiên gần nhất."""
    if px is None or px.empty or not {"close", "volume"}.issubset(px.columns):
        return np.nan
    d = px.tail(n).copy()
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce")
    # Nhân 1000 để convert từ format vnstock (41.10) sang VND thực (41,100)
    tv = (d["close"] * 1000 * d["volume"]).dropna()
    return float(tv.mean()) if len(tv) else np.nan

# ——— TCBS API: lấy Rev YoY (quý mới nhất) ———
def _fetch_rev_yoy_tcbs(symbol: str, timeout: int = 10) -> float:
    """
    Gọi TCBS API:
      https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/{symbol.lower()}/incomestatement?yearly=0
    Chọn quý mới nhất theo (year, quarter). Trả về yearRevenueGrowth đã chuẩn hoá về %.
    """
    if requests is None:  # không có requests
        return np.nan
    try:
        url = f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/{symbol.strip().lower()}/incomestatement"
        params = {"yearly": 0}
        headers = {"User-Agent": "Mozilla/5.0 (compatible; StockApp/1.0)"}
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return np.nan
        # chọn quý mới nhất
        def key(x): 
            return (int(x.get("year", -9_999)), int(x.get("quarter", -9_999)))
        latest = max(data, key=key)
        val = latest.get("yearRevenueGrowth", np.nan)
        return _norm_pct(val)
    except Exception:
        return np.nan

# =========================
# vnstock adapter (prices + ratios)
# =========================
def _import_vnstock():
    try:
        import vnstock  # noqa: F401
        return __import__("vnstock")
    except Exception:
        raise RuntimeError("[ERR] Không thể import vnstock. Cài: pip install -U vnstock")

class VnstockAdapterError(RuntimeError):
    pass

class VnAdapter:
    """
    - Giá: Vnstock().stock(symbol, source).quote.* (start, end, interval="1D")
    - Chỉ số: Vnstock().stock(symbol, source).finance.ratio(period=..., dropna=True)
    - rev_yoy: LẤY TRỰC TIẾP từ TCBS API (quý mới nhất, yearRevenueGrowth)
    """
    def __init__(
        self,
        preferred_sources: Optional[List[str]] = None,  # ví dụ: ["TCBS"]
        end_date: Optional[str] = None,                 # YYYY-MM-DD
        verbose: bool = False,
        max_retries: int = 2,
        rate_limit_wait: int = 60
    ):
        v = _import_vnstock()
        self.v = v
        self.verbose = bool(verbose)
        self.Vnstock = getattr(v, "Vnstock", None)
        self.preferred_sources = [s.strip().upper() for s in (preferred_sources or ["TCBS"])]
        self.end_date = end_date
        self.max_retries = int(max_retries)
        self.rate_limit_wait = int(rate_limit_wait)

    # ---------- PRICE ----------
    def get_quote_history(self, symbol: str, days: int = 270) -> pd.DataFrame:
        end_dt = pd.to_datetime(self.end_date or datetime.today().strftime("%Y-%m-%d"))
        end_dt = _prev_weekday(end_dt)
        start_dt = end_dt - timedelta(days=int(days * 1.5))
        start = start_dt.strftime("%Y-%m-%d")
        end = end_dt.strftime("%Y-%m-%d")

        # Thử lùi tối đa 10 ngày nếu ngày cuối thiếu dữ liệu
        for _ in range(11):
            try:
                df = self._fetch_price(symbol, start, end, interval="1D")
                if isinstance(df, pd.DataFrame) and not df.empty:
                    last_day = pd.to_datetime(df["date"].max())
                    if (end_dt - last_day).days <= 3:
                        return df
            except Exception:
                pass
            end_dt = _prev_weekday(end_dt - timedelta(days=1))
            start_dt = end_dt - timedelta(days=int(days * 1.5))
            start, end = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
        raise VnstockAdapterError(f"Không lấy được lịch sử giá cho {symbol}.")

    def _norm_price_df(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()

        def lc(s): return str(s).strip().lower()
        # cột ngày
        date_col = None
        for c in d.columns:
            if lc(c) in ("date","time","datetime","tradingdate","ngay","timestamp"):
                date_col = c; break
        if date_col is None:
            if isinstance(d.index, pd.DatetimeIndex) or lc(d.index.name or "") in ("date","datetime"):
                d = d.reset_index()
                date_col = d.columns[0]
            else:
                date_col = d.columns[0]
        d = d.rename(columns={date_col: "date"})

        # alias OHLCV
        alias = {
            "open":   ["open","o","openprice","open price","gia mo cua"],
            "high":   ["high","h","highprice","gia cao nhat","highest"],
            "low":    ["low","l","lowprice","gia thap nhat","lowest"],
            "close":  ["close","c","closeprice","adj close","price","last","match_price","gia dong cua"],
            "volume": [
                "volume","vol","match_volume","totalvolume","khoi luong",
                "accumulatedvolume","accumulated_vol","accumulatedvol","nmvolume","nm_volume"
            ],
        }
        rename = {}
        norm_cols = {lc(c): c for c in d.columns}
        for std, cands in alias.items():
            for cand in cands:
                if cand in norm_cols and std not in d.columns:
                    rename[norm_cols[cand]] = std; break
        d = d.rename(columns=rename)

        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"]).sort_values("date")
        for c in ("open","high","low","close","volume"):
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        if "close" not in d.columns:
            raise VnstockAdapterError("Thiếu cột close trong dữ liệu giá.")

        # bù OHLC thiếu (an toàn)
        if "open" not in d.columns:  d["open"]  = d["close"].shift(1).fillna(d["close"])
        if "high" not in d.columns:  d["high"]  = np.maximum(d["open"], d["close"])
        if "low"  not in d.columns:  d["low"]   = np.minimum(d["open"], d["close"])
        if "volume" not in d.columns: d["volume"] = np.nan

        # gộp theo ngày nếu dữ liệu intra
        d["_day"] = d["date"].dt.normalize()
        d = (d.groupby("_day", as_index=False)
               .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}))
        d = d.rename(columns={"_day": "date"})
        return d[["date","open","high","low","close","volume"]]

    def _fetch_price(self, symbol: str, start: str, end: str, interval: str = "1D") -> pd.DataFrame:
        if self.Vnstock is None:
            raise VnstockAdapterError("Không có lớp Vnstock trong thư viện vnstock.")
        for src in self.preferred_sources:
            try:
                stock = self.Vnstock().stock(symbol=symbol, source=src)
                q = getattr(stock, "quote", None)
                if q is None:
                    continue
                for meth in ("history","historical","price_series","prices"):
                    if hasattr(q, meth):
                        try:
                            raw = getattr(q, meth)(start=start, end=end, interval=interval)
                            if isinstance(raw, pd.DataFrame) and not raw.empty:
                                return self._norm_price_df(raw)
                        except TypeError:
                            try:
                                raw = getattr(q, meth)(start=start, end=end)
                                if isinstance(raw, pd.DataFrame) and not raw.empty:
                                    return self._norm_price_df(raw)
                            except Exception:
                                pass
                        except Exception:
                            pass
            except Exception:
                continue
        raise VnstockAdapterError("Không lấy được dữ liệu giá qua Vnstock().stock(...).quote")

    # ---------- FUNDAMENTALS ----------
    def get_fundamentals(self, symbol: str, *, period: str = "year") -> Dict[str, float]:
        """
        PE/PB/ROE/...: từ finance.ratio(period=..., dropna=True), kỳ mới nhất.
        rev_yoy: LẤY TRỰC TIẾP từ TCBS API (quý mới nhất, yearRevenueGrowth).
        Tất cả trường dạng % sẽ được chuẩn hoá về %.
        """
        out: Dict[str, float] = {}
        if self.Vnstock is not None:
            for src in self.preferred_sources:
                try:
                    stock = self.Vnstock().stock(symbol=symbol, source=src)
                    fin = getattr(stock, "finance", None)
                    if fin is None or not hasattr(fin, "ratio"):
                        continue
                    df = fin.ratio(period=period, dropna=True)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        out = self._extract_ratios(df)
                        break
                except Exception:
                    continue

        # rev_yoy từ TCBS API (override)
        rev = _fetch_rev_yoy_tcbs(symbol)
        if not pd.isna(rev):
            out["rev_yoy"] = rev

        return out

    def _latest_row(self, df: pd.DataFrame) -> Dict:
        """Chọn dòng mới nhất theo period/report_date/date/year (nếu có)."""
        cand = df.copy()
        try:
            if "period" not in cand.columns and str(cand.index.name or "").lower() == "period":
                cand = cand.reset_index()
        except Exception:
            pass

        for key in ("period", "report_date", "date", "year"):
            if key in cand.columns:
                try:
                    if key == "year":
                        cand["_t"] = pd.to_datetime(cand[key].astype(str) + "-12-31", errors="coerce")
                    else:
                        cand["_t"] = pd.to_datetime(cand[key], errors="coerce")
                    cand = cand.sort_values("_t")
                except Exception:
                    pass
                break
        if not cand.empty:
            try:
                return cand.tail(1).iloc[0].to_dict()
            except Exception:
                pass
        # fallback
        return df.tail(1).iloc[0].to_dict()

    def _extract_ratios(self, df: pd.DataFrame) -> Dict[str, float]:
        row = self._latest_row(df)
        rl = {str(k).strip().lower(): v for k, v in row.items()}

        def pick(*keys, default=np.nan):
            for k in keys:
                if k in row and not pd.isna(row[k]): return row[k]
                lk = str(k).strip().lower()
                if lk in rl and not pd.isna(rl[lk]): return rl[lk]
            return default

        out = {
            "pe": pick("price_to_earning","pe","pe_ttm","peTrailing","priceToEarnings"),
            "pb": pick("price_to_book","pb","priceToBook"),
            "roe": pick("roe","returnOnEquity"),
            "de": pick("debt_on_equity","debtToEquity","de","de_ratio"),
            "net_margin": pick("post_tax_margin","net_profit_margin","profitMargin"),
            "eps_yoy": pick("eps_change","eps yoy","eps_growth_yoy","year_eps_growth"),
            "market_cap": pick("market_cap","marketcap","mc"),
        }

        # Ép float
        for k in list(out.keys()):
            try: out[k] = float(out[k])
            except Exception: out[k] = np.nan

        # Chuẩn hoá % (rev_yoy xử lý riêng bằng TCBS API)
        for k in ("roe","net_margin","eps_yoy"):
            if not pd.isna(out.get(k)):
                out[k] = _norm_pct(out[k])

        return out

# =========================
# Scoring
# =========================
@dataclass
class ScoreWeights:
    value: float = 0.22
    quality: float = 0.22
    growth: float = 0.20
    momentum: float = 0.20
    liquidity: float = 0.10
    risk: float = 0.06

def compute_features(
    adapter: VnAdapter,
    symbol: str,
    price_days: int = 270,
    verbose: bool = False,
    *,
    finance_period: str = "year"  # 'year' | 'quarter'
) -> Dict:
    """Tính các đặc trưng cơ bản từ dữ liệu giá + chỉ số (rev_yoy từ TCBS API)."""
    try:
        px = adapter.get_quote_history(symbol, days=price_days)
    except Exception as e:
        if verbose:
            print(f"  [!] {symbol}: lỗi lấy giá — {e}")
        return {"symbol": symbol, "m1": np.nan, "m3": np.nan, "m6": np.nan, "adtv": np.nan, "vol": np.nan}

    def pct_change(days_back: int):
        try:
            if len(px) < days_back + 1: return np.nan
            return _safe_pct(px["close"].iloc[-1], px["close"].iloc[-(days_back + 1)])
        except Exception:
            return np.nan

    m1 = pct_change(21)
    m3 = pct_change(63)
    m6 = pct_change(126)

    # ADTV(20)
    adtv = _calc_adtv_vnd(px, n=20)
    vol = _annualized_vol(px["close"].pct_change())

    # Fundamentals
    f = adapter.get_fundamentals(symbol, period=finance_period)

    if verbose:
        pe, pb, roe = f.get("pe", np.nan), f.get("pb", np.nan), f.get("roe", np.nan)
        def fmt_pct(x):
            try: return f"{float(x):.1f}%"
            except Exception: return ""
        print(
            f"    · {symbol}: 3M {m3:.2%} | 1M {m1:.2%} | "
            f"P/E {'' if pd.isna(pe) else f'{pe:.1f}'} | "
            f"P/B {'' if pd.isna(pb) else f'{pb:.2f}'} | "
            f"ROE {'' if pd.isna(roe) else fmt_pct(roe)} | "
            f"ADTV {adtv/1e9:.1f} tỷ"
        )

    return {
        "symbol": symbol,
        "m1": m1, "m3": m3, "m6": m6,
        "adtv": adtv, "vol": vol,
        **f,
    }

def score_symbols_v2(
    df: pd.DataFrame,
    weights: ScoreWeights,
    min_adtv: float,
    *,
    sector_col: str = "sector",
    robust: bool = True,
    winsor_p: float = 0.02,
    use_m6_in_momentum: bool = True,
    log_transform_cols: List[str] = ("adtv", "vol"),
) -> pd.DataFrame:
    d = df.copy()

    # 0) Lọc thanh khoản
    if "adtv" not in d.columns:
        d["adtv"] = np.nan
    d = d[d["adtv"].fillna(0) >= float(min_adtv)]

    # 0.5) Lưu raw ADTV để hiển thị (trước khi log transform)
    if "adtv" in d.columns:
        d["adtv_raw"] = d["adtv"].copy()

    # 1) log1p cho các thước đo lệch phải
    for c in log_transform_cols:
        if c in d.columns:
            d[c] = np.log1p(pd.to_numeric(d[c], errors="coerce"))

    # 2) Impute theo ngành rồi toàn thị trường
    groupcol = sector_col if sector_col in d.columns else None
    for col in ["pe","pb","roe","net_margin","de","rev_yoy","eps_yoy","m1","m3","m6","adtv","vol"]:
        d = _impute_by_group(d, col, groupcol)

    # 3) Winsorize & scale (sector-neutral nếu có ngành)
    def _scale_series(s: pd.Series) -> pd.Series:
        s2 = _winsorize(s.astype(float), p=winsor_p) if winsor_p else s.astype(float)
        return _robust_zscore(s2) if robust else _zscore(s2)

    def _neutralize(s: pd.Series) -> pd.Series:
        if groupcol:
            return _groupwise_apply(s, d[groupcol], _scale_series)
        return _scale_series(s)

    # Value (thấp tốt)
    d["z_pe"] = -_neutralize(d.get("pe", pd.Series(np.nan, index=d.index)))
    d["z_pb"] = -_neutralize(d.get("pb", pd.Series(np.nan, index=d.index)))
    value = (d["z_pe"] + d["z_pb"]) / 2

    # Quality
    d["z_roe"]    =  _neutralize(d.get("roe", pd.Series(np.nan, index=d.index)))
    d["z_margin"] =  _neutralize(d.get("net_margin", pd.Series(np.nan, index=d.index)))
    d["z_de"]     = -_neutralize(d.get("de", pd.Series(np.nan, index=d.index)))
    quality = (d["z_roe"] + d["z_margin"] + d["z_de"]) / 3

    # Growth
    d["z_rev"] = _neutralize(d.get("rev_yoy", pd.Series(np.nan, index=d.index)))
    d["z_eps"] = _neutralize(d.get("eps_yoy", pd.Series(np.nan, index=d.index)))
    growth = (d["z_rev"] + d["z_eps"]) / 2

    # Momentum (1M/3M/(6M))
    d["z_m1"] = _neutralize(d.get("m1", pd.Series(np.nan, index=d.index)))
    d["z_m3"] = _neutralize(d.get("m3", pd.Series(np.nan, index=d.index)))
    if use_m6_in_momentum:
        d["z_m6"] = _neutralize(d.get("m6", pd.Series(np.nan, index=d.index)))
        momentum = (d["z_m1"] + d["z_m3"] + d["z_m6"]) / 3
    else:
        momentum = (d["z_m1"] + d["z_m3"]) / 2

    # Liquidity
    d["z_adtv"] = _neutralize(d.get("adtv", pd.Series(np.nan, index=d.index)))
    liquidity = d["z_adtv"]

    # Risk (biến động thấp tốt)
    d["z_vol"] = -_neutralize(d.get("vol", pd.Series(np.nan, index=d.index)))
    risk_pen = d["z_vol"]

    # 4) Hợp nhất điểm
    d["Value"]     = value
    d["Quality"]   = quality
    d["Growth"]    = growth
    d["Momentum"]  = momentum
    d["Liquidity"] = liquidity
    d["RiskAdj"]   = risk_pen

    d["score"] = (
        weights.value   * d["Value"]   +
        weights.quality * d["Quality"] +
        weights.growth  * d["Growth"]  +
        weights.momentum* d["Momentum"]+
        weights.liquidity*d["Liquidity"]+
        weights.risk    * d["RiskAdj"]
    )
    return d.sort_values("score", ascending=False)

# Giữ API cũ
def score_symbols(df: pd.DataFrame, weights: ScoreWeights, min_adtv: float) -> pd.DataFrame:
    return score_symbols_v2(
        df, weights, min_adtv,
        sector_col="sector",
        robust=True,
        winsor_p=0.02,
        use_m6_in_momentum=True,
        log_transform_cols=("adtv","vol"),
    )
