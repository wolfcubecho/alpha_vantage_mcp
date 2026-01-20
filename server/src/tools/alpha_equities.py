"""Alpha Equities tools: macro snapshot, symbol snapshot, and discovery.

Mirrors functionality from the Node MCP implementation with conservative
request patterns, Alpha Vantage as primary data source, and Yahoo Finance
fallbacks when AV is unavailable or rate-limited.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import httpx
try:
    import yfinance as yf  # optional, improves Yahoo reliability
    _HAS_YF = True
except Exception:
    _HAS_YF = False

from src.tools.registry import tool
from src.common import _make_api_request


# Yahoo Finance helpers (no key required)
def _yf_quote(symbol: str) -> Optional[Dict[str, Optional[float]]]:
    """Get Yahoo Finance quote for a symbol.

    Returns:
        Dict with price, pctChange, volume or None on error.
    """
    try:
        url = "https://query1.finance.yahoo.com/v7/finance/quote"
        params = {"symbols": symbol}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
        }
        with httpx.Client(timeout=10, follow_redirects=True, headers=headers) as client:
            for _ in range(3):
                res = client.get(url, params=params)
                if res.status_code == 200:
                    try:
                        json = res.json()
                        arr = (json.get("quoteResponse", {}) or {}).get("result", [])
                        q = arr[0] if arr else None
                        if not q:
                            raise ValueError("empty quote result")
                        price = q.get("regularMarketPrice")
                        pct = q.get("regularMarketChangePercent")
                        vol = q.get("regularMarketVolume")
                        return {
                            "price": float(price) if isinstance(price, (int, float)) else None,
                            "pctChange": float(pct) if isinstance(pct, (int, float)) else None,
                            "volume": float(vol) if isinstance(vol, (int, float)) else None,
                        }
                    except Exception:
                        pass
                time.sleep(0.3)
        return None
    except Exception:
        return None


def _yf_daily(symbol: str) -> Optional[Dict[str, List[float]]]:
    """Get Yahoo Finance daily candles for symbol (2y range, 1d interval)."""
    try:
        base = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"range": "2y", "interval": "1d"}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
        }
        with httpx.Client(timeout=10, follow_redirects=True, headers=headers) as client:
            for _ in range(3):
                res = client.get(base, params=params)
                if res.status_code == 200:
                    try:
                        data = res.json()
                        result = ((data or {}).get("chart", {}) or {}).get("result", [])
                        if not result:
                            raise ValueError("empty chart result")
                        quote = (result[0].get("indicators", {}) or {}).get("quote", [])
                        q0 = quote[0] if quote else {}
                        closes = [float(v) for v in (q0.get("close") or []) if isinstance(v, (int, float))]
                        highs = [float(v) for v in (q0.get("high") or []) if isinstance(v, (int, float))]
                        lows = [float(v) for v in (q0.get("low") or []) if isinstance(v, (int, float))]
                        if not closes:
                            raise ValueError("no closes")
                        return {"closes": closes, "highs": highs, "lows": lows}
                    except Exception:
                        pass
                time.sleep(0.3)
        return None
    except Exception:
        return None


def _yfi_quote(symbol: str) -> Optional[Dict[str, Optional[float]]]:
    """Quote via yfinance (robust Yahoo wrapper). Computes pctChange from history.

    Returns price, pctChange, volume when available.
    """
    if not _HAS_YF:
        return None
    try:
        df = yf.download(symbol, period="5d", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        last = df.tail(1)
        prev = df.tail(2).head(1)
        price = float(last["Close"].to_numpy()[0]) if "Close" in last.columns else None
        volume = float(last["Volume"].to_numpy()[0]) if "Volume" in last.columns else None
        pct = None
        if price is not None and not prev.empty and "Close" in prev.columns:
            prev_close = float(prev["Close"].to_numpy()[0])
            if prev_close:
                pct = (price - prev_close) / prev_close * 100.0
        return {"price": price, "pctChange": pct, "volume": volume}
    except Exception:
        return None


def _yfi_daily(symbol: str) -> Optional[Dict[str, List[float]]]:
    """Daily candles via yfinance (2y, 1d)."""
    if not _HAS_YF:
        return None
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        closes = [float(x) for x in df["Close"].tolist()] if "Close" in df.columns else []
        highs = [float(x) for x in df["High"].tolist()] if "High" in df.columns else []
        lows = [float(x) for x in df["Low"].tolist()] if "Low" in df.columns else []
        if not closes:
            return None
        return {"closes": closes, "highs": highs, "lows": lows}
    except Exception:
        return None


def _ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def _rma(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    s = sum(values[:period])
    v = s / period
    a = 1 / period
    for x in values[period:]:
        v = a * x + (1 - a) * v
    return v


@tool
def macro_snapshot(
    symbols: Optional[str] = None,
    includeSector: Optional[str] = None,
    includeMovers: Optional[str] = None,
    noNulls: Optional[str] = None,
) -> Dict[str, Any]:
    """Efficient macro snapshot with optional sector and movers.

    Args:
        symbols: Comma-separated macro tickers; default SPY,VIX,UUP (limit ~3)
        includeSector: Include sector performance ("true"/"false")
        includeMovers: Include top gainers/losers ("true"/"false")
    Returns:
        Dict containing sector, movers, and per-symbol snapshot (price, pctChange, volume).
    """
    syms = (symbols or "SPY,^VIX,UUP").split(",")
    syms = [s.strip() for s in syms if s.strip()][:3]
    want_sector = (includeSector or "false").lower() == "true"
    want_movers = (includeMovers or "false").lower() == "true"

    sector = _make_api_request("SECTOR", {"datatype": "json"}) if want_sector else None
    movers = _make_api_request("TOP_GAINERS_LOSERS", {"datatype": "json"}) if want_movers else None

    out: List[Dict[str, Optional[float]]] = []
    for sym in syms:
        try:
            vix_like = sym.upper() in ("VIX", "^VIX")
            price = None
            pct = None
            vol = None
            # Alpha Vantage only (use VIXY proxy for VIX)
            cand = ["VIXY"] if vix_like else [sym]
            quote_json = None
            for s in cand:
                q = _make_api_request("GLOBAL_QUOTE", {"symbol": s, "datatype": "json"})
                qq = (q or {}).get("Global Quote") if isinstance(q, dict) else None
                ptmp = float(qq.get("05. price")) if qq and qq.get("05. price") else None
                if ptmp and not math.isnan(ptmp):
                    quote_json = qq
                    break
            if quote_json:
                try:
                    price = float(quote_json.get("05. price"))
                except Exception:
                    price = None
                try:
                    pct_str = (quote_json.get("10. change percent") or "0").replace("%", "")
                    pct = float(pct_str)
                except Exception:
                    pct = None
                try:
                    vol = float(quote_json.get("06. volume"))
                except Exception:
                    vol = None

            out.append({"symbol": sym, "price": price, "pctChange": pct, "volume": vol})
        except Exception:
            out.append({"symbol": sym, "price": None, "pctChange": None, "volume": None})

    # Coerce nulls if requested (default true)
    if (noNulls or "true").lower() == "true":
        for i in range(len(out)):
            row = out[i]
            row["price"] = float(row["price"]) if isinstance(row.get("price"), (int, float)) else 0.0
            row["pctChange"] = float(row["pctChange"]) if isinstance(row.get("pctChange"), (int, float)) else 0.0
            row["volume"] = float(row["volume"]) if isinstance(row.get("volume"), (int, float)) else 0.0
    return {"sector": sector, "movers": movers, "etfs": out}


@tool
def symbol_snapshot(
    symbol: str,
    outputsize: Optional[str] = None,
    atrPctMax: Optional[str] = None,
    trendFilter: Optional[str] = None,
    noNulls: Optional[str] = None,
) -> Dict[str, Any]:
    """Per-symbol snapshot: EMA(50/200), ATR% (14), latest quote.

    Args:
        symbol: Equity/ETF symbol, e.g., AAPL or SPY
        outputsize: Alpha Vantage output size: compact|full (default compact)
        atrPctMax: Max ATR% filter hint
        trendFilter: Trend filter hint: up|down
    Returns:
        Dict with symbol metrics and filter hints.
    """
    outsize = outputsize or "compact"

    # Prefer Yahoo/yfinance for all symbols; AV as last resort (skip AV for VIX)
    vix_like = symbol.upper() in ("VIX", "^VIX")
    closes: List[float] = []
    highs: List[float] = []
    lows: List[float] = []

    # direct Yahoo first
    for fs in (["^VIX", "VIXY"] if vix_like else [symbol]):
        yf = _yf_daily(fs)
        if yf:
            closes = yf["closes"]
            highs = yf["highs"]
            lows = yf["lows"]
            break

    # yfinance next
    if not closes:
        yfd = (_yfi_daily("^VIX") or _yfi_daily("VIXY")) if vix_like else _yfi_daily(symbol)
        if yfd:
            closes = yfd["closes"]
            highs = yfd["highs"]
            lows = yfd["lows"]

    # Alpha Vantage last (non-VIX only)
    if not closes and not vix_like:
        daily = _make_api_request(
            "TIME_SERIES_DAILY_ADJUSTED", {"symbol": symbol, "outputsize": outsize, "datatype": "json"}
        )
        series = daily.get("Time Series (Daily)") if isinstance(daily, dict) else {}
        dates = sorted(series.keys()) if series else []
        if dates:
            closes = [float(series[d]["5. adjusted close"]) for d in dates]
            highs = [float(series[d]["2. high"]) for d in dates]
            lows = [float(series[d]["3. low"]) for d in dates]

    # If still no candles, use latest Yahoo/yfinance quote for last price
    if not closes:
        yq = (_yfi_quote("^VIX") or _yf_quote("^VIX") or _yfi_quote("VIXY") or _yf_quote("VIXY")) if vix_like else (_yfi_quote(symbol) or _yf_quote(symbol))
        if yq and isinstance(yq.get("price"), (int, float)):
            closes = [float(yq.get("price"))]

    last_close = closes[-1] if closes else None
    tr: List[float] = []
    for i in range(len(closes)):
        hl = (highs[i] if i < len(highs) else 0.0) - (lows[i] if i < len(lows) else 0.0)
        hc = abs((highs[i] if i < len(highs) else 0.0) - (closes[i - 1] if i > 0 else 0.0)) if i > 0 else 0.0
        lc = abs((lows[i] if i < len(lows) else 0.0) - (closes[i - 1] if i > 0 else 0.0)) if i > 0 else 0.0
        tr.append(max(hl, hc, lc))

    atr = _rma(tr, 14)
    atr_pct = (atr / last_close * 100.0) if (atr is not None and last_close) else None
    ema50 = _ema(closes, 50)
    ema200 = _ema(closes, 200)
    trend: Optional[str] = None
    if ema50 is not None and ema200 is not None:
        trend = "up" if ema50 > ema200 else "down"

    atr_cap = float(atrPctMax) if atrPctMax else None
    trend_pref = (trendFilter or "").lower() or None
    hint = {
        "atrPctUnderCap": (atr_cap is not None and atr_pct is not None and atr_pct <= atr_cap) or None,
        "trendMatches": (trend_pref == trend) if trend_pref else None,
    }

    pct_change: Optional[float] = None
    volume: Optional[float] = None
    # Yahoo/yfinance first
    for yf_sym in (["^VIX", "VIXY"] if vix_like else [symbol]):
        yq = _yfi_quote(yf_sym) or _yf_quote(yf_sym)
        if yq:
            pct_change = yq.get("pctChange")
            volume = yq.get("volume")
            break
    # Alpha Vantage fallback for non-VIX
    if pct_change is None and not vix_like:
        quote = _make_api_request("GLOBAL_QUOTE", {"symbol": symbol, "datatype": "json"})
        q = quote.get("Global Quote") if isinstance(quote, dict) else None
        if q:
            try:
                pct_str = (q.get("10. change percent") or "0").replace("%", "")
                pct_change = float(pct_str)
            except Exception:
                pct_change = None
            try:
                volume = float(q.get("06. volume"))
            except Exception:
                volume = None

    result = {
        "symbol": symbol,
        "price": last_close,
        "pctChange": pct_change,
        "volume": volume,
        "atrPct": atr_pct,
        "trend": trend,
        "ema50": ema50,
        "ema200": ema200,
        "hint": hint,
    }

    # Coerce nulls if requested (default true)
    if (noNulls or "true").lower() == "true":
        result["price"] = float(result["price"]) if isinstance(result.get("price"), (int, float)) else 0.0
        result["pctChange"] = float(result["pctChange"]) if isinstance(result.get("pctChange"), (int, float)) else 0.0
        result["volume"] = float(result["volume"]) if isinstance(result.get("volume"), (int, float)) else 0.0
        result["atrPct"] = float(result["atrPct"]) if isinstance(result.get("atrPct"), (int, float)) else 0.0
        result["ema50"] = float(result["ema50"]) if isinstance(result.get("ema50"), (int, float)) else 0.0
        result["ema200"] = float(result["ema200"]) if isinstance(result.get("ema200"), (int, float)) else 0.0
        result["trend"] = result["trend"] if isinstance(result.get("trend"), str) and result["trend"] else "unknown"
    return result


@tool
def discover_equities(
    source: Optional[str] = None,
    topN: Optional[str] = None,
    resultCount: Optional[str] = None,
    maxSymbols: Optional[str] = None,
    computeVolatility: Optional[str] = None,
    outputsize: Optional[str] = None,
    noNulls: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Discover top equities from Alpha Vantage movers with ranking.

    Args:
        source: combined|gainers|losers|most_active (default: combined)
        topN: Number of candidates from source lists (default: 10)
        resultCount: Number of results to return (default: 5)
        maxSymbols: Max symbols to quote (default: 4)
        computeVolatility: Compute ATR% for finalists only: true|false (default: false)
        outputsize: Daily series size if computing volatility: compact|full (default compact)
    Returns:
        Ranked list of dicts with symbol, price, pctChange, volume, and optional atrPct.
    """
    src = (source or "combined").lower()
    n_top = int(topN or "10")
    n_out = int(resultCount or "5")
    max_sym = int(maxSymbols or "4")
    want_vol = ((computeVolatility or "false").lower() == "true")
    outsize = outputsize or "compact"

    movers = _make_api_request("TOP_GAINERS_LOSERS", {"datatype": "json"})
    bucket = {
        "gainers": movers.get("top_gainers", []) if isinstance(movers, dict) else [],
        "losers": movers.get("top_losers", []) if isinstance(movers, dict) else [],
        "most_active": movers.get("most_actively_traded", []) if isinstance(movers, dict) else [],
    }

    if src == "gainers":
        tickers = [x.get("ticker") for x in bucket["gainers"]]
    elif src == "losers":
        tickers = [x.get("ticker") for x in bucket["losers"]]
    elif src == "most_active":
        tickers = [x.get("ticker") for x in bucket["most_active"]]
    else:
        tickers = [x.get("ticker") for x in (bucket["gainers"] + bucket["most_active"])]

    uniq = []
    seen = set()
    for t in tickers:
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            uniq.append(t)
        if len(uniq) >= n_top:
            break

    # Quote up to max_sym using Alpha Vantage only
    to_quote = uniq[: max(1, max_sym)]
    quotes: List[Dict[str, Optional[float]]] = []
    for sym in to_quote:
        price = None
        pct = None
        vol = None
        qres = _make_api_request("GLOBAL_QUOTE", {"symbol": sym, "datatype": "json"})
        q = qres.get("Global Quote") if isinstance(qres, dict) else None
        if q:
            try:
                price = float(q.get("05. price"))
            except Exception:
                price = None
            try:
                pct_str = (q.get("10. change percent") or "0").replace("%", "")
                pct = float(pct_str)
            except Exception:
                pct = None
            try:
                vol = float(q.get("06. volume"))
            except Exception:
                vol = None
        quotes.append({"symbol": sym, "price": price, "pctChange": pct, "volume": vol})

    # Rank by abs pctChange then volume
    def _rank_key(x: Dict[str, Optional[float]]):
        ma = abs(x.get("pctChange") or 0.0)
        va = x.get("volume") or 0.0
        return (-ma, -va)

    quotes.sort(key=_rank_key)
    out = [{**x, "atrPct": None} for x in quotes[:n_out]]

    if want_vol:
        for i in range(len(out)):
            sym = out[i]["symbol"]
            try:
                daily = _make_api_request(
                    "TIME_SERIES_DAILY_ADJUSTED",
                    {"symbol": sym, "outputsize": outsize, "datatype": "json"},
                )
                series = daily.get("Time Series (Daily)") if isinstance(daily, dict) else {}
                dates = sorted(series.keys()) if series else []
                closes = [float(series[d]["5. adjusted close"]) for d in dates] if dates else []
                highs = [float(series[d]["2. high"]) for d in dates] if dates else []
                lows = [float(series[d]["3. low"]) for d in dates] if dates else []
                if not closes:
                    yf = _yf_daily(sym)
                    if yf:
                        closes = yf["closes"]
                        highs = yf["highs"]
                        lows = yf["lows"]
                last_close = closes[-1] if closes else None
                tr: List[float] = []
                for j in range(len(closes)):
                    hl = (highs[j] if j < len(highs) else 0.0) - (lows[j] if j < len(lows) else 0.0)
                    hc = abs((highs[j] if j < len(highs) else 0.0) - (closes[j - 1] if j > 0 else 0.0)) if j > 0 else 0.0
                    lc = abs((lows[j] if j < len(lows) else 0.0) - (closes[j - 1] if j > 0 else 0.0)) if j > 0 else 0.0
                    tr.append(max(hl, hc, lc))
                atr = _rma(tr, 14)
                out[i]["atrPct"] = (atr / last_close * 100.0) if (atr is not None and last_close) else None
            except Exception:
                out[i]["atrPct"] = None

        # After pctChange/volume, prefer lower ATR%
        def _sort_key(x: Dict[str, Any]):
            atr = x.get("atrPct")
            atr_key = atr if isinstance(atr, (int, float)) else float("inf")
            ma = abs(x.get("pctChange") or 0.0)
            va = x.get("volume") or 0.0
            return (atr_key, -ma, -va)

        out.sort(key=_sort_key)

    # Coerce nulls if requested (default true)
    if (noNulls or "true").lower() == "true":
        for i in range(len(out)):
            row = out[i]
            row["price"] = float(row["price"]) if isinstance(row.get("price"), (int, float)) else 0.0
            row["pctChange"] = float(row["pctChange"]) if isinstance(row.get("pctChange"), (int, float)) else 0.0
            row["volume"] = float(row["volume"]) if isinstance(row.get("volume"), (int, float)) else 0.0
            row["atrPct"] = float(row["atrPct"]) if isinstance(row.get("atrPct"), (int, float)) else 0.0
    return out
