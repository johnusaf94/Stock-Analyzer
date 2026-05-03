"""
druckenmiller_analyzer.py
==========================
Stanley Druckenmiller's five-pattern macro framework.

1. Pig Philosophy      — Triple Alignment score (macro + momentum + technicals)
2. Rate of Change      — Second derivative of earnings/revenue growth
3. Liquidity           — M2 + Fed Balance Sheet regime signal
4. Mental Flexibility  — Stop-loss / trend-break signal (is the story still intact?)
5. Technical Verify    — 200-day MA regime filter + price confirmation

All scores are deterministic Python. No LLM involvement.
Requires: pip install yfinance requests pandas numpy
"""

import yfinance as yf
import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta


# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────
ROC_ACCELERATION_MIN    =  0.0    # growth rate must be accelerating (positive 2nd deriv)
ROC_DECEL_FLAG          = -0.05   # growth decelerating > 5% = red flag
MA_200_BUFFER           =  0.02   # must be > 2% above 200d MA to confirm uptrend
STOP_LOSS_MA50_BUFFER   = -0.05   # if price > 5% below 50d MA = story broken
TRIPLE_ALIGN_THRESHOLD  =  0.65   # must score 65%+ across all three pillars for "pig" sizing
LIQUIDITY_LOOKBACK_DAYS = 90      # compare M2/fed balance to 90 days ago


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class PigPhilosophy:
    """Triple alignment: macro + momentum + technicals all pointing same direction."""
    macro_score:        Optional[float] = None   # 0-1 from liquidity
    momentum_score:     Optional[float] = None   # 0-1 from rate of change
    technical_score:    Optional[float] = None   # 0-1 from chart confirmation
    triple_align_score: Optional[float] = None   # weighted average
    aligned:            bool = False             # all three above threshold?
    conviction_level:   str = "LOW"             # LOW / MODERATE / HIGH / MAX
    signal:             str = "WAIT"            # WAIT / POSITION / SIZE UP
    reasoning:          str = ""


@dataclass
class RateOfChange:
    """Second derivative of growth — is the acceleration positive or negative?"""
    revenue_growth_now:    Optional[float] = None   # most recent YoY
    revenue_growth_prior:  Optional[float] = None   # prior period YoY
    earnings_growth_now:   Optional[float] = None
    earnings_growth_prior: Optional[float] = None
    rev_acceleration:      Optional[float] = None   # 2nd derivative: now - prior
    earn_acceleration:     Optional[float] = None
    composite_roc:         Optional[float] = None   # blended acceleration
    # Druckenmiller: "invest where earnings WILL BE in 18 months, not where they are"
    forward_eps:           Optional[float] = None   # analyst forward EPS estimate
    trailing_eps:          Optional[float] = None   # current TTM EPS
    eps_revision_direction:str = "UNKNOWN"          # UP / FLAT / DOWN (analyst revisions)
    signal:                str = "UNKNOWN"           # ACCELERATING / FLAT / DECELERATING
    score:                 Optional[float] = None    # 0-1
    druckenmiller_note:    str = ""


@dataclass
class LiquidityRegime:
    """Fed balance sheet + M2 money supply regime."""
    fed_balance_sheet_now:   Optional[float] = None
    fed_balance_sheet_prior: Optional[float] = None
    fed_balance_change_pct:  Optional[float] = None

    m2_now:     Optional[float] = None
    m2_prior:   Optional[float] = None
    m2_change_pct: Optional[float] = None

    regime:     str = "UNKNOWN"   # EXPANDING / NEUTRAL / TIGHTENING
    score:      float = 0.5
    note:       str = ""
    data_source: str = ""


@dataclass
class MentalFlexibility:
    """Stop-loss and trend-break signals — is the original thesis still intact?"""
    current_price:      Optional[float] = None
    ma_50d:             Optional[float] = None
    ma_200d:            Optional[float] = None
    pct_vs_ma50:        Optional[float] = None
    pct_vs_ma200:       Optional[float] = None

    # Beta / momentum regime
    beta:               Optional[float] = None
    momentum_3m:        Optional[float] = None   # 3-month price return
    momentum_6m:        Optional[float] = None   # 6-month price return
    momentum_12m:       Optional[float] = None   # 12-month price return

    # Stop signals
    below_ma50_flag:    bool = False   # price broken below 50d MA
    below_ma200_flag:   bool = False   # price broken below 200d MA
    thesis_intact:      bool = True
    exit_signal:        str = "HOLD"  # HOLD / CAUTION / EXIT
    stop_note:          str = ""
    score:              float = 0.5


# Sector ETF proxies for relative strength calculation
SECTOR_ETFS = {
    "Technology":              "XLK",
    "Healthcare":              "XLV",
    "Financial Services":      "XLF",
    "Consumer Cyclical":       "XLY",
    "Consumer Defensive":      "XLP",
    "Energy":                  "XLE",
    "Industrials":             "XLI",
    "Basic Materials":         "XLB",
    "Real Estate":             "XLRE",
    "Utilities":               "XLU",
    "Communication Services":  "XLC",
    "Semiconductor":           "SOXX",
}


@dataclass
class TechnicalVerify:
    """200-day MA regime filter + RSI + relative strength + chart confirmation."""
    current_price:      Optional[float] = None
    ma_200d:            Optional[float] = None
    pct_above_ma200:    Optional[float] = None
    above_ma200:        bool = False

    rsi_14:             Optional[float] = None
    rsi_signal:         str = ""

    # 52-week positioning
    pct_from_52wk_high: Optional[float] = None
    pct_from_52wk_low:  Optional[float] = None

    # New highs / new lows check
    near_52wk_high:     bool = False   # within 10% of 52wk high = price confirming
    near_52wk_low:      bool = False   # within 10% of 52wk low = charts against thesis

    # Relative strength vs sector — Druckenmiller buys sector leaders
    sector_etf:              str = ""
    stock_return_3m:         Optional[float] = None
    sector_return_3m:        Optional[float] = None
    relative_strength_3m:   Optional[float] = None   # stock - sector (positive = outperforming)
    rs_signal:               str = "UNKNOWN"          # LEADING / INLINE / LAGGING

    regime:             str = "UNKNOWN"   # BULLISH / NEUTRAL / BEARISH
    vote:               str = "NO VOTE"   # CONFIRMS / NEUTRAL / AGAINST
    score:              float = 0.5
    note:               str = ""


@dataclass
class DruckenmillerAnalysis:
    ticker:             str = ""
    company_name:       str = ""
    sector:             str = ""
    analysis_date:      str = ""

    pig_philosophy:     PigPhilosophy = field(default_factory=PigPhilosophy)
    rate_of_change:     RateOfChange = field(default_factory=RateOfChange)
    liquidity:          LiquidityRegime = field(default_factory=LiquidityRegime)
    mental_flexibility: MentalFlexibility = field(default_factory=MentalFlexibility)
    technical_verify:   TechnicalVerify = field(default_factory=TechnicalVerify)

    overall_signal:     str = "WAIT"        # WAIT / WATCH / POSITION / SIZE_UP
    overall_score:      float = 0.0         # 0-100
    conviction:         str = "LOW"
    errors:             list = field(default_factory=list)


# ─────────────────────────────────────────────
# PILLAR 1: RATE OF CHANGE (2nd Derivative)
# ─────────────────────────────────────────────

def analyze_rate_of_change(ticker: str, info: dict) -> RateOfChange:
    roc = RateOfChange()

    try:
        t = yf.Ticker(ticker)
        income = t.income_stmt

        if income is not None and not income.empty and len(income.columns) >= 3:
            # Get 3 periods to compute 2nd derivative
            rev_key = next((k for k in ["Total Revenue","Revenue"] if k in income.index), None)
            earn_key = next((k for k in ["Net Income","Net Income Common Stockholders"] if k in income.index), None)

            if rev_key:
                rev = income.loc[rev_key].dropna()
                if len(rev) >= 3:
                    vals = list(rev.values)
                    # YoY growth rates (most recent = index 0)
                    g0 = (vals[0] - vals[1]) / abs(vals[1]) if vals[1] != 0 else None
                    g1 = (vals[1] - vals[2]) / abs(vals[2]) if vals[2] != 0 else None
                    roc.revenue_growth_now   = float(g0) if g0 is not None else None
                    roc.revenue_growth_prior = float(g1) if g1 is not None else None
                    if g0 is not None and g1 is not None:
                        roc.rev_acceleration = g0 - g1   # positive = accelerating

            if earn_key:
                earn = income.loc[earn_key].dropna()
                if len(earn) >= 3:
                    vals = list(earn.values)
                    g0 = (vals[0] - vals[1]) / abs(vals[1]) if vals[1] != 0 else None
                    g1 = (vals[1] - vals[2]) / abs(vals[2]) if vals[2] != 0 else None
                    roc.earnings_growth_now   = float(g0) if g0 is not None else None
                    roc.earnings_growth_prior = float(g1) if g1 is not None else None
                    if g0 is not None and g1 is not None:
                        roc.earn_acceleration = g0 - g1

        # Composite 2nd derivative — blend rev and earnings accelerations
        accels = [x for x in [roc.rev_acceleration, roc.earn_acceleration] if x is not None]
        if accels:
            roc.composite_roc = float(np.mean(accels))

            if roc.composite_roc > 0.05:
                roc.signal = "ACCELERATING"
                roc.score  = min(1.0, 0.6 + roc.composite_roc * 2)
                roc.druckenmiller_note = (
                    f"Growth is accelerating — 2nd derivative is positive ({roc.composite_roc:+.1%}). "
                    f"Druckenmiller sees increasing momentum; this is investable."
                )
            elif roc.composite_roc > ROC_DECEL_FLAG:
                roc.signal = "FLAT"
                roc.score  = 0.45
                roc.druckenmiller_note = (
                    f"Growth rate roughly flat ({roc.composite_roc:+.1%} change). "
                    f"Druckenmiller would wait for clearer directional momentum."
                )
            else:
                roc.signal = "DECELERATING"
                roc.score  = max(0.0, 0.4 + roc.composite_roc)
                roc.druckenmiller_note = (
                    f"Growth is DECELERATING — 2nd derivative negative ({roc.composite_roc:+.1%}). "
                    f"Druckenmiller would likely be a seller regardless of current earnings level."
                )
        else:
            # Fallback: use yfinance single-period metrics
            rev_growth = info.get("revenueGrowth")
            earn_growth = info.get("earningsGrowth")
            if rev_growth is not None:
                roc.revenue_growth_now = float(rev_growth)
                roc.score  = 0.65 if rev_growth > 0.10 else (0.45 if rev_growth > 0 else 0.25)
                roc.signal = "GROWING" if rev_growth > 0 else "DECLINING"
                roc.druckenmiller_note = (
                    f"Single-period revenue growth {rev_growth:.1%} — "
                    f"2nd derivative unavailable (need 3+ years of data)."
                )
            else:
                roc.score  = 0.5
                roc.signal = "UNKNOWN"
                roc.druckenmiller_note = "Insufficient data for rate-of-change analysis."

        # ── FORWARD EPS — Druckenmiller's "18 months into the future" signal ──
        # He invests where earnings WILL BE, not where they are today.
        # If forward EPS > trailing EPS, analysts expect growth — favourable.
        # If forward EPS < trailing EPS, analysts see earnings declining — bearish.
        roc.trailing_eps = info.get("trailingEps")
        roc.forward_eps  = info.get("forwardEps")

        if roc.trailing_eps and roc.forward_eps and roc.trailing_eps != 0:
            eps_delta_pct = (roc.forward_eps - roc.trailing_eps) / abs(roc.trailing_eps)
            if eps_delta_pct > 0.10:
                roc.eps_revision_direction = "UP"
                # Boost score — analysts see earnings expansion ahead
                roc.score = min(1.0, (roc.score or 0.5) + 0.10)
                roc.druckenmiller_note += (
                    f" | Forward EPS ${roc.forward_eps:.2f} vs trailing ${roc.trailing_eps:.2f} "
                    f"({eps_delta_pct:+.1%}) — analysts see earnings EXPANDING. "
                    f"Druckenmiller: the future looks better than today."
                )
            elif eps_delta_pct < -0.10:
                roc.eps_revision_direction = "DOWN"
                # Penalise — analysts see earnings contraction
                roc.score = max(0.0, (roc.score or 0.5) - 0.15)
                roc.druckenmiller_note += (
                    f" | Forward EPS ${roc.forward_eps:.2f} vs trailing ${roc.trailing_eps:.2f} "
                    f"({eps_delta_pct:+.1%}) — analysts see earnings CONTRACTING. "
                    f"Druckenmiller: current earnings are already priced in and future looks worse."
                )
            else:
                roc.eps_revision_direction = "FLAT"
                roc.druckenmiller_note += (
                    f" | Forward EPS ~flat vs trailing ({eps_delta_pct:+.1%})."
                )

    except Exception as e:
        roc.score  = 0.5
        roc.signal = "ERROR"
        roc.druckenmiller_note = str(e)

    return roc


# ─────────────────────────────────────────────
# PILLAR 2: LIQUIDITY REGIME
# ─────────────────────────────────────────────

def fetch_fred_series(series_id: str, api_key: str = "DEMO") -> Optional[pd.Series]:
    """
    Fetch FRED data series. Uses public FRED API.
    series_id examples: 'M2SL' (M2 money supply), 'WALCL' (Fed balance sheet)
    """
    try:
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={api_key}&file_type=json"
            f"&observation_start={(datetime.now()-timedelta(days=180)).strftime('%Y-%m-%d')}"
            f"&sort_order=desc&limit=10"
        )
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            obs = data.get("observations", [])
            vals = {o["date"]: float(o["value"]) for o in obs if o["value"] != "."}
            if vals:
                return pd.Series(vals)
    except Exception:
        pass
    return None


def analyze_liquidity() -> LiquidityRegime:
    lr = LiquidityRegime()

    # Try FRED API (works without a key for basic data — rate limited but functional)
    m2_series = fetch_fred_series("M2SL")
    fed_series = fetch_fred_series("WALCL")

    signals = []

    if m2_series is not None and len(m2_series) >= 2:
        m2_sorted = m2_series.sort_index(ascending=False)
        lr.m2_now   = float(m2_sorted.iloc[0])
        lr.m2_prior = float(m2_sorted.iloc[-1])
        if lr.m2_prior > 0:
            lr.m2_change_pct = (lr.m2_now - lr.m2_prior) / lr.m2_prior
            signals.append(lr.m2_change_pct)
        lr.data_source = "FRED API"

    if fed_series is not None and len(fed_series) >= 2:
        fed_sorted = fed_series.sort_index(ascending=False)
        lr.fed_balance_sheet_now   = float(fed_sorted.iloc[0])
        lr.fed_balance_sheet_prior = float(fed_sorted.iloc[-1])
        if lr.fed_balance_sheet_prior > 0:
            lr.fed_balance_change_pct = (
                (lr.fed_balance_sheet_now - lr.fed_balance_sheet_prior)
                / lr.fed_balance_sheet_prior
            )
            signals.append(lr.fed_balance_change_pct)

    if signals:
        avg_change = float(np.mean(signals))
        if avg_change > 0.01:
            lr.regime = "EXPANDING"
            lr.score  = min(1.0, 0.65 + avg_change * 5)
            lr.note   = (
                f"Liquidity is EXPANDING — M2/Fed balance growing {avg_change:+.1%}. "
                f"Druckenmiller: go aggressive. The Fed is your friend."
            )
        elif avg_change > -0.01:
            lr.regime = "NEUTRAL"
            lr.score  = 0.50
            lr.note   = (
                f"Liquidity roughly flat ({avg_change:+.1%}). "
                f"Druckenmiller: selective positioning, not full conviction."
            )
        else:
            lr.regime = "TIGHTENING"
            lr.score  = max(0.0, 0.40 + avg_change * 3)
            lr.note   = (
                f"Liquidity is TIGHTENING — M2/Fed balance shrinking {avg_change:+.1%}. "
                f"Druckenmiller: be defensive regardless of how good individual stocks look. "
                f"Central banks move markets, not earnings."
            )
    else:
        # Fallback: estimate from current rate environment
        # Use 10yr treasury level as proxy — rising rates = tightening
        try:
            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="6mo", interval="1mo")
            if not hist.empty:
                rate_now   = float(hist["Close"].iloc[-1]) / 100
                rate_prior = float(hist["Close"].iloc[0]) / 100
                rate_delta = rate_now - rate_prior
                if rate_delta > 0.005:
                    lr.regime = "TIGHTENING"
                    lr.score  = 0.30
                    lr.note   = (
                        f"10yr yield rising {rate_delta*100:.1f}bps — "
                        f"tightening proxy. FRED data unavailable. "
                        f"Druckenmiller would reduce risk."
                    )
                elif rate_delta < -0.005:
                    lr.regime = "EASING"
                    lr.score  = 0.70
                    lr.note   = (
                        f"10yr yield falling {abs(rate_delta)*100:.1f}bps — "
                        f"easing proxy. FRED data unavailable. "
                        f"Druckenmiller would increase exposure."
                    )
                else:
                    lr.regime = "NEUTRAL"
                    lr.score  = 0.50
                    lr.note   = "Rate environment stable — neutral liquidity proxy."
                lr.data_source = "10yr Treasury proxy (FRED unavailable)"
        except Exception:
            lr.regime = "UNKNOWN"
            lr.score  = 0.50
            lr.note   = "Liquidity data unavailable — defaulting to neutral."
            lr.data_source = "None"

    return lr


# ─────────────────────────────────────────────
# PILLAR 3: MENTAL FLEXIBILITY (Stop-Loss / Thesis Check)
# ─────────────────────────────────────────────

def analyze_mental_flexibility(ticker: str, info: dict) -> MentalFlexibility:
    mf = MentalFlexibility()

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="2y", interval="1d")

        if hist.empty:
            mf.exit_signal = "NO DATA"
            return mf

        prices = hist["Close"]
        mf.current_price = float(prices.iloc[-1])
        mf.beta = info.get("beta")

        # Moving averages
        if len(prices) >= 50:
            mf.ma_50d  = float(prices.rolling(50).mean().iloc[-1])
            mf.pct_vs_ma50 = (mf.current_price - mf.ma_50d) / mf.ma_50d
        if len(prices) >= 200:
            mf.ma_200d = float(prices.rolling(200).mean().iloc[-1])
            mf.pct_vs_ma200 = (mf.current_price - mf.ma_200d) / mf.ma_200d

        # Momentum (price returns)
        if len(prices) >= 65:
            mf.momentum_3m  = float((prices.iloc[-1] / prices.iloc[-65]) - 1)
        if len(prices) >= 130:
            mf.momentum_6m  = float((prices.iloc[-1] / prices.iloc[-130]) - 1)
        if len(prices) >= 252:
            mf.momentum_12m = float((prices.iloc[-1] / prices.iloc[-252]) - 1)

        # Stop-loss signals
        if mf.pct_vs_ma50 is not None and mf.pct_vs_ma50 < STOP_LOSS_MA50_BUFFER:
            mf.below_ma50_flag = True
        if mf.pct_vs_ma200 is not None and mf.pct_vs_ma200 < 0:
            mf.below_ma200_flag = True

        # Score and signal
        score = 0.5
        notes = []

        if not mf.below_ma200_flag:
            score += 0.15
        else:
            score -= 0.20
            notes.append("Price below 200d MA — Druckenmiller exits. Thesis broken.")

        if not mf.below_ma50_flag:
            score += 0.10
        else:
            score -= 0.10
            notes.append(f"Price {mf.pct_vs_ma50:.1%} below 50d MA — stop-loss level breached.")

        if mf.momentum_6m is not None:
            if mf.momentum_6m > 0.15:
                score += 0.15
                notes.append(f"Strong 6m momentum: {mf.momentum_6m:+.1%}")
            elif mf.momentum_6m > 0:
                score += 0.05
            elif mf.momentum_6m < -0.20:
                score -= 0.20
                notes.append(f"Deeply negative 6m momentum: {mf.momentum_6m:+.1%}. Story has changed.")
            else:
                score -= 0.05

        mf.score = max(0.0, min(1.0, score))
        mf.thesis_intact = not (mf.below_ma200_flag or mf.below_ma50_flag)

        if mf.score >= 0.65 and mf.thesis_intact:
            mf.exit_signal = "HOLD"
            mf.stop_note = "Price action confirms thesis. No stop triggered."
        elif mf.score >= 0.45:
            mf.exit_signal = "CAUTION"
            mf.stop_note = "Mixed signals — monitor closely. " + " | ".join(notes)
        else:
            mf.exit_signal = "EXIT"
            mf.thesis_intact = False
            mf.stop_note = "STOP TRIGGERED. " + " | ".join(notes)

    except Exception as e:
        mf.exit_signal = "ERROR"
        mf.stop_note = str(e)
        mf.score = 0.5

    return mf


# ─────────────────────────────────────────────
# PILLAR 4: TECHNICAL VERIFICATION
# ─────────────────────────────────────────────

def _calc_rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    rsi   = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else None


def analyze_technical_verify(ticker: str, info: dict) -> TechnicalVerify:
    tv = TechnicalVerify()

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="2y", interval="1d")

        if hist.empty:
            tv.regime = "NO DATA"
            tv.vote   = "NO VOTE"
            return tv

        prices = hist["Close"]
        tv.current_price = float(prices.iloc[-1])

        # 200d MA — the primary Druckenmiller regime filter
        if len(prices) >= 200:
            tv.ma_200d = float(prices.rolling(200).mean().iloc[-1])
            tv.pct_above_ma200 = (tv.current_price - tv.ma_200d) / tv.ma_200d
            tv.above_ma200 = tv.pct_above_ma200 > MA_200_BUFFER

        # RSI
        tv.rsi_14 = _calc_rsi(prices)
        if tv.rsi_14:
            if tv.rsi_14 > 70:
                tv.rsi_signal = "OVERBOUGHT"
            elif tv.rsi_14 < 30:
                tv.rsi_signal = "OVERSOLD"
            else:
                tv.rsi_signal = "NEUTRAL"

        # 52-week positioning
        tv.pct_from_52wk_high = info.get("52WeekChange")
        high_52 = info.get("fiftyTwoWeekHigh")
        low_52  = info.get("fiftyTwoWeekLow")
        if high_52 and tv.current_price:
            tv.pct_from_52wk_high = (tv.current_price - high_52) / high_52
            tv.near_52wk_high = tv.pct_from_52wk_high > -0.10
        if low_52 and tv.current_price:
            tv.pct_from_52wk_low = (tv.current_price - low_52) / low_52
            tv.near_52wk_low  = tv.pct_from_52wk_low < 0.10

        # ── RELATIVE STRENGTH vs SECTOR ─────────────────────────────────────
        # Druckenmiller: "I look at hundreds of charts every night."
        # He buys stocks that are LEADING their sector, not lagging.
        # A bullish macro idea with a lagging chart = no entry.
        sector = info.get("sector", "")
        sector_etf = SECTOR_ETFS.get(sector, "SPY")
        tv.sector_etf = sector_etf

        try:
            s_hist = yf.Ticker(sector_etf).history(period="4mo", interval="1d")
            if not s_hist.empty and len(prices) >= 65:
                # Align to same date range
                stock_65  = float(prices.iloc[-65])
                stock_now = float(prices.iloc[-1])
                tv.stock_return_3m = (stock_now / stock_65) - 1

                sector_prices = s_hist["Close"]
                if len(sector_prices) >= 65:
                    tv.sector_return_3m = float(
                        (sector_prices.iloc[-1] / sector_prices.iloc[-65]) - 1
                    )
                    tv.relative_strength_3m = tv.stock_return_3m - tv.sector_return_3m

                    if tv.relative_strength_3m > 0.05:
                        tv.rs_signal = "LEADING"
                    elif tv.relative_strength_3m > -0.05:
                        tv.rs_signal = "INLINE"
                    else:
                        tv.rs_signal = "LAGGING"
        except Exception:
            tv.rs_signal = "UNAVAILABLE"

        # Score and regime
        score = 0.5
        notes = []

        if tv.above_ma200:
            score += 0.25
            notes.append(f"Above 200d MA by {tv.pct_above_ma200:.1%} — Druckenmiller regime: LONG OK")
        else:
            score -= 0.30
            notes.append(f"Below 200d MA by {abs(tv.pct_above_ma200):.1%} — Druckenmiller would NOT buy long")

        if tv.near_52wk_high:
            score += 0.15
            notes.append("Near 52wk high — market voting in favor of thesis")
        elif tv.near_52wk_low:
            score -= 0.15
            notes.append("Near 52wk low — charts making new lows, Druckenmiller won't buy")

        if tv.rsi_signal == "OVERSOLD":
            score += 0.05
            notes.append(f"RSI {tv.rsi_14:.0f} — oversold, potential entry")
        elif tv.rsi_signal == "OVERBOUGHT":
            score -= 0.05
            notes.append(f"RSI {tv.rsi_14:.0f} — overbought")

        # Relative strength adjustment
        if tv.rs_signal == "LEADING":
            score += 0.15
            rs_str = f"{tv.relative_strength_3m:+.1%}" if tv.relative_strength_3m else ""
            notes.append(f"Outperforming {tv.sector_etf} by {rs_str} — market leader, Druckenmiller buys leaders")
        elif tv.rs_signal == "LAGGING":
            score -= 0.15
            rs_str = f"{tv.relative_strength_3m:+.1%}" if tv.relative_strength_3m else ""
            notes.append(f"Underperforming {tv.sector_etf} by {rs_str} — laggard, Druckenmiller avoids laggards")
        elif tv.rs_signal == "INLINE":
            notes.append(f"In line with {tv.sector_etf} — no relative strength edge")

        tv.score = max(0.0, min(1.0, score))

        if tv.score >= 0.70:
            tv.regime = "BULLISH"
            tv.vote   = "CONFIRMS"
        elif tv.score >= 0.45:
            tv.regime = "NEUTRAL"
            tv.vote   = "NEUTRAL"
        else:
            tv.regime = "BEARISH"
            tv.vote   = "AGAINST"

        tv.note = " | ".join(notes)

    except Exception as e:
        tv.regime = "ERROR"
        tv.vote   = "NO VOTE"
        tv.note   = str(e)
        tv.score  = 0.5

    return tv


# ─────────────────────────────────────────────
# PIG PHILOSOPHY AGGREGATOR
# ─────────────────────────────────────────────

def build_pig_philosophy(
    roc: RateOfChange,
    liquidity: LiquidityRegime,
    mf: MentalFlexibility,
    tv: TechnicalVerify,
) -> PigPhilosophy:
    pp = PigPhilosophy()

    pp.macro_score     = liquidity.score
    # Boost momentum score if relative strength confirms the thesis
    base_momentum = roc.score if roc.score else 0.5
    if tv.rs_signal == "LEADING":
        pp.momentum_score = min(1.0, base_momentum + 0.08)
    elif tv.rs_signal == "LAGGING":
        pp.momentum_score = max(0.0, base_momentum - 0.10)
    else:
        pp.momentum_score = base_momentum
    pp.technical_score = tv.score

    # Triple alignment: weighted average
    # Liquidity is most important (Druckenmiller: "The Fed moves markets, not earnings")
    pp.triple_align_score = (
        pp.macro_score     * 0.40 +
        pp.momentum_score  * 0.35 +
        pp.technical_score * 0.25
    )

    pp.aligned = (
        pp.triple_align_score >= TRIPLE_ALIGN_THRESHOLD
        and liquidity.regime in ("EXPANDING", "EASING", "NEUTRAL")
        and tv.vote != "AGAINST"
        and mf.exit_signal not in ("EXIT",)
    )

    if pp.triple_align_score >= 0.80 and pp.aligned:
        pp.conviction_level = "MAX"
        pp.signal           = "SIZE UP"
        pp.reasoning = (
            f"Triple alignment at {pp.triple_align_score:.0%} — macro, momentum, and technicals all green. "
            f"Druckenmiller: this is an 'unequivocal certainty' moment. Concentrate the bet."
        )
    elif pp.triple_align_score >= 0.65 and pp.aligned:
        pp.conviction_level = "HIGH"
        pp.signal           = "POSITION"
        pp.reasoning = (
            f"Triple alignment at {pp.triple_align_score:.0%} — strong but not maximum conviction. "
            f"Druckenmiller: take a meaningful position but not bet-the-farm sizing."
        )
    elif pp.triple_align_score >= 0.50:
        pp.conviction_level = "MODERATE"
        pp.signal           = "WATCH"
        pp.reasoning = (
            f"Partial alignment at {pp.triple_align_score:.0%} — one or more pillars not confirming. "
            f"Druckenmiller: watch but don't act. Wait for all three to align."
        )
    else:
        pp.conviction_level = "LOW"
        pp.signal           = "WAIT"
        pp.reasoning = (
            f"Triple alignment failing at {pp.triple_align_score:.0%}. "
            f"Druckenmiller: do nothing. There's no edge here."
        )

    return pp


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_druckenmiller_analysis(ticker: str) -> DruckenmillerAnalysis:
    analysis = DruckenmillerAnalysis(
        ticker=ticker.upper(),
        analysis_date=datetime.now().strftime("%Y-%m-%d")
    )

    try:
        t = yf.Ticker(ticker)
        info = t.info
        analysis.company_name = info.get("longName", ticker)
        analysis.sector       = info.get("sector", "Unknown")
    except Exception as e:
        analysis.errors.append(f"Info fetch: {e}")
        info = {}

    analysis.rate_of_change     = analyze_rate_of_change(ticker, info)
    analysis.liquidity           = analyze_liquidity()
    analysis.mental_flexibility  = analyze_mental_flexibility(ticker, info)
    analysis.technical_verify    = analyze_technical_verify(ticker, info)

    analysis.pig_philosophy = build_pig_philosophy(
        analysis.rate_of_change,
        analysis.liquidity,
        analysis.mental_flexibility,
        analysis.technical_verify,
    )

    # Overall score (0-100)
    weights = {
        "pig":        0.25,
        "roc":        0.20,
        "liquidity":  0.25,
        "mental":     0.15,
        "technical":  0.15,
    }
    analysis.overall_score = (
        (analysis.pig_philosophy.triple_align_score or 0.5) * weights["pig"] +
        (analysis.rate_of_change.score or 0.5)              * weights["roc"] +
        analysis.liquidity.score                            * weights["liquidity"] +
        analysis.mental_flexibility.score                   * weights["mental"] +
        analysis.technical_verify.score                     * weights["technical"]
    ) * 100

    analysis.overall_signal = analysis.pig_philosophy.signal
    analysis.conviction     = analysis.pig_philosophy.conviction_level

    return analysis


# ─────────────────────────────────────────────
# DISPLAY FORMATTERS
# ─────────────────────────────────────────────

def format_druckenmiller_display(analysis: DruckenmillerAnalysis) -> str:
    pp = analysis.pig_philosophy
    roc = analysis.rate_of_change
    liq = analysis.liquidity
    mf  = analysis.mental_flexibility
    tv  = analysis.technical_verify

    def p(val, d=1):
        return f"{val:+.{d}%}" if val is not None else "N/A"

    def n(val, d=2):
        return f"{val:.{d}f}" if val is not None else "N/A"

    lines = [
        "",
        f"  ── TRIPLE ALIGNMENT (Pig Philosophy) ─────────────────",
        f"  Macro (Liquidity):   {pp.macro_score:.0%}    {liq.regime}",
        f"  Momentum (RoC):      {pp.momentum_score:.0%}    {roc.signal}",
        f"  Technical:           {pp.technical_score:.0%}    {tv.vote}",
        f"  ─────────────────────────────────────────────────────",
        f"  ALIGNMENT SCORE:     {pp.triple_align_score:.0%}",
        f"  CONVICTION:          {pp.conviction_level}",
        f"  SIGNAL:              {pp.signal}",
        "",
        f"  ── RATE OF CHANGE (2nd Derivative) ───────────────────",
        f"  Revenue growth now:  {p(roc.revenue_growth_now)}",
        f"  Revenue growth prior:{p(roc.revenue_growth_prior)}",
        f"  Acceleration:        {p(roc.rev_acceleration)}   (positive = Druck buys)",
        f"  Composite RoC:       {roc.signal}",
        "",
        f"  ── LIQUIDITY REGIME ──────────────────────────────────",
        f"  Regime:              {liq.regime}",
        f"  M2 Change:           {p(liq.m2_change_pct)}",
        f"  Fed Balance:         {p(liq.fed_balance_change_pct)}",
        f"  Source:              {liq.data_source}",
        "",
        f"  ── MENTAL FLEXIBILITY (Stop-Loss Check) ──────────────",
        f"  vs 50d MA:           {p(mf.pct_vs_ma50)}    {'🔴 STOP' if mf.below_ma50_flag else '✅ OK'}",
        f"  vs 200d MA:          {p(mf.pct_vs_ma200)}    {'🔴 STOP' if mf.below_ma200_flag else '✅ OK'}",
        f"  6m Momentum:         {p(mf.momentum_6m)}",
        f"  Exit Signal:         {mf.exit_signal}",
        "",
        f"  ── TECHNICAL VERIFICATION ────────────────────────────",
        f"  Above 200d MA:       {'YES ✅' if tv.above_ma200 else 'NO ❌'}   ({p(tv.pct_above_ma200)})",
        f"  RSI (14):            {n(tv.rsi_14, 0)}   {tv.rsi_signal}",
        f"  Near 52wk High:      {'YES' if tv.near_52wk_high else 'NO'}",
        f"  Rel. Strength (3m):  {p(tv.relative_strength_3m)} vs {tv.sector_etf}   {tv.rs_signal}",
        f"  Chart Vote:          {tv.vote}   — {tv.regime}",
        "",
        f"  OVERALL SCORE:       {analysis.overall_score:.1f}/100   {analysis.overall_signal}",
    ]
    return "\n".join(lines)


def format_druckenmiller_for_composite(analysis: DruckenmillerAnalysis) -> float:
    """Return normalized 0-1 score for composite integration."""
    return analysis.overall_score / 100.0


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    print(f"\nRunning Druckenmiller analysis on {ticker}...\n")
    result = run_druckenmiller_analysis(ticker)
    print(format_druckenmiller_display(result))
