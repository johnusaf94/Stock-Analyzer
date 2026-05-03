"""
squeeze_analyzers.py
====================
Two short squeeze hunters with distinct philosophies.

KEITH GILL (DeepFuckingValue):
  Bottom-up, retail-powered. Finds fundamentally sound companies
  with extreme short interest that Wall Street has crowded against.
  His edge: patience + conviction + the math of forced covering.

CHAMATH PALIHAPITIYA:
  Top-down, narrative-driven. Looks for macro catalysts, insider
  positioning, and retail momentum that can trigger institutional
  short covering cascades.

Data sources: yfinance (short interest, float, volume)
Note: Cost to Borrow (CTB) is not available in free yfinance.
      We approximate it using short interest % of float + fee proxy.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


# ─────────────────────────────────────────────
# THRESHOLDS — Gill's specific criteria
# ─────────────────────────────────────────────
GILL_SHORT_INTEREST_MIN    = 0.20   # > 20% of float shorted
GILL_DTC_MIN               = 5.0    # > 5 days to cover
GILL_CTB_PROXY_MIN         = 10.0   # > 10% implied cost to borrow
GILL_PE_MAX                = 30.0   # not insanely overvalued
GILL_REVENUE_GROWTH_MIN    = 0.0    # at least flat revenue

CHAMATH_SHORT_INTEREST_MIN = 0.15   # slightly lower threshold
CHAMATH_MOMENTUM_MIN       = 0.05   # needs price momentum
CHAMATH_INSIDER_THRESHOLD  = 0.05   # >5% insider ownership = skin in game


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class SqueezeMetrics:
    """Raw short squeeze data points."""
    ticker:                  str = ""
    company_name:            str = ""
    sector:                  str = ""

    # Core squeeze metrics
    short_interest_pct:      Optional[float] = None   # % of float short
    shares_short:            Optional[int]   = None   # total shares shorted
    float_shares:            Optional[int]   = None   # shares available to trade
    avg_daily_volume:        Optional[float] = None   # 10-day avg volume
    days_to_cover:           Optional[float] = None   # shares_short / avg_volume
    ctb_proxy:               Optional[float] = None   # estimated cost to borrow %

    # Price action
    current_price:           Optional[float] = None
    price_change_1m:         Optional[float] = None   # 1-month return
    price_change_3m:         Optional[float] = None
    rsi_14:                  Optional[float] = None
    avg_volume_10d:          Optional[float] = None
    volume_surge:            Optional[float] = None   # recent vol / avg vol

    # Fundamentals (Gill cares about these)
    pe_ratio:                Optional[float] = None
    revenue_growth:          Optional[float] = None
    free_cash_flow:          Optional[float] = None
    debt_to_equity:          Optional[float] = None
    market_cap:              Optional[float] = None

    # Ownership (Chamath cares about these)
    insider_ownership:       Optional[float] = None
    institutional_ownership: Optional[float] = None
    short_change_pct:        Optional[float] = None   # change in short interest

    # Data quality
    fetch_errors:            list = field(default_factory=list)


@dataclass
class GillAnalysis:
    ticker:               str = ""
    metrics:              SqueezeMetrics = field(default_factory=SqueezeMetrics)

    # Pillar scores
    squeeze_setup_score:  float = 0.0   # 0-40: short interest + DTC + CTB
    fundamental_score:    float = 0.0   # 0-35: not a zombie company
    catalyst_score:       float = 0.0   # 0-25: volume surge, momentum, options activity

    total_score:          float = 0.0   # 0-100
    verdict:              str = ""      # SQUEEZE CANDIDATE / WATCH / PASS
    conviction:           str = ""      # YOLO / HIGH / MODERATE / LOW
    thesis:               str = ""      # one-paragraph Gill-style thesis
    red_flags:            list = field(default_factory=list)
    green_flags:          list = field(default_factory=list)


@dataclass
class ChamathAnalysis:
    ticker:                  str = ""
    metrics:                 SqueezeMetrics = field(default_factory=SqueezeMetrics)

    # Chamath-specific scores
    macro_setup_score:       float = 0.0   # 0-30: top-down narrative fit
    squeeze_pressure_score:  float = 0.0   # 0-35: short metrics
    catalyst_momentum_score: float = 0.0   # 0-35: price action + insider signal

    total_score:             float = 0.0
    verdict:                 str = ""
    narrative:               str = ""   # what's the macro story?
    thesis:                  str = ""
    red_flags:               list = field(default_factory=list)
    green_flags:             list = field(default_factory=list)


# ─────────────────────────────────────────────
# SHARED DATA FETCHER
# ─────────────────────────────────────────────

def fetch_squeeze_metrics(ticker: str) -> SqueezeMetrics:
    """
    Pull all short squeeze relevant data from yfinance.
    Note: CTB (cost to borrow) isn't free — we proxy it from
    short interest %, float scarcity, and borrow rate proxies.
    """
    m = SqueezeMetrics(ticker=ticker.upper())

    try:
        t = yf.Ticker(ticker)
        info = t.info

        m.company_name = info.get("longName", ticker)
        m.sector       = info.get("sector", "Unknown")
        m.current_price       = info.get("currentPrice") or info.get("regularMarketPrice")
        m.market_cap          = info.get("marketCap")
        m.pe_ratio            = info.get("trailingPE")
        m.revenue_growth      = info.get("revenueGrowth")
        m.free_cash_flow      = info.get("freeCashflow")
        m.debt_to_equity      = info.get("debtToEquity")
        m.insider_ownership   = info.get("heldPercentInsiders")
        m.institutional_ownership = info.get("heldPercentInstitutions")

        # ── Short interest data ──
        m.shares_short     = info.get("sharesShort")
        m.float_shares     = info.get("floatShares")
        m.avg_daily_volume = info.get("averageVolume10days") or info.get("averageVolume")

        # Short % of float
        short_pct = info.get("shortPercentOfFloat")
        if short_pct:
            m.short_interest_pct = float(short_pct)
            # yfinance returns this as decimal (0.20) or percent (20) inconsistently
            if m.short_interest_pct > 1.0:
                m.short_interest_pct /= 100.0

        # Days to Cover
        dtc = info.get("shortRatio")   # yfinance has this directly
        if dtc:
            m.days_to_cover = float(dtc)
        elif m.shares_short and m.avg_daily_volume and m.avg_daily_volume > 0:
            m.days_to_cover = m.shares_short / m.avg_daily_volume

        # Short change (prior vs current)
        prior_short = info.get("sharesShortPriorMonth")
        if prior_short and m.shares_short and prior_short > 0:
            m.short_change_pct = (m.shares_short - prior_short) / prior_short

        # ── Cost to Borrow Proxy ──
        # CTB isn't free but we can estimate it:
        # High short % of float + low float = scarce borrow supply = higher CTB
        # Formula: CTB_proxy = short_pct * (1 / float_ratio) * base_rate
        if m.short_interest_pct and m.float_shares and m.market_cap and m.current_price:
            float_ratio = (m.float_shares * m.current_price) / m.market_cap
            scarcity_factor = m.short_interest_pct * (1.0 / max(float_ratio, 0.1))
            m.ctb_proxy = min(scarcity_factor * 50, 200)  # cap at 200%

        # ── Price action ──
        hist = t.history(period="4mo", interval="1d")
        if not hist.empty:
            prices = hist["Close"]
            vols   = hist["Volume"]

            if len(prices) >= 20:
                m.price_change_1m = float((prices.iloc[-1] / prices.iloc[-21]) - 1)
            if len(prices) >= 65:
                m.price_change_3m = float((prices.iloc[-1] / prices.iloc[-65]) - 1)

            # RSI-14
            delta = prices.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss
            rsi   = 100 - (100 / (1 + rs))
            m.rsi_14 = float(rsi.iloc[-1]) if not rsi.empty else None

            # Volume surge: recent 5d avg vs 20d avg
            if len(vols) >= 20:
                recent_vol = float(vols.iloc[-5:].mean())
                avg_vol    = float(vols.iloc[-20:].mean())
                m.volume_surge = recent_vol / avg_vol if avg_vol > 0 else None

    except Exception as e:
        m.fetch_errors.append(str(e))

    return m


# ─────────────────────────────────────────────
# KEITH GILL ANALYZER
# ─────────────────────────────────────────────

def run_gill_analysis(ticker: str) -> GillAnalysis:
    """
    Keith Gill's squeeze framework:
    1. Squeeze Setup: Is the short thesis mathematically unsustainable?
    2. Fundamental Quality: Is this actually a real company, not a zombie?
    3. Catalyst Signal: Is something about to force short covering?

    Gill's famous quote: "The stock is not adequately valued."
    He needed BOTH: extreme short pressure AND an undervalued business.
    """
    g = GillAnalysis(ticker=ticker.upper())
    g.metrics = fetch_squeeze_metrics(ticker)
    m = g.metrics

    green = []
    red   = []

    # ── PILLAR 1: Squeeze Setup Score (40 pts) ──
    squeeze = 0.0

    # Short Interest % of Float (max 18 pts)
    si = m.short_interest_pct
    if si is not None:
        if si >= 0.50:
            squeeze += 18
            green.append(f"Short interest {si:.0%} — EXTREME. Wall Street is ALL IN on the short.")
        elif si >= 0.30:
            squeeze += 14
            green.append(f"Short interest {si:.0%} — very high. Significant forced-covering risk.")
        elif si >= GILL_SHORT_INTEREST_MIN:
            squeeze += 9
            green.append(f"Short interest {si:.0%} — above Gill's 20% threshold.")
        elif si >= 0.10:
            squeeze += 4
            red.append(f"Short interest {si:.0%} — below 20% threshold. Mild pressure only.")
        else:
            red.append(f"Short interest {si:.0%} — very low. No meaningful squeeze setup.")
    else:
        red.append("Short interest data unavailable.")

    # Days to Cover (max 14 pts)
    dtc = m.days_to_cover
    if dtc is not None:
        if dtc >= 20:
            squeeze += 14
            green.append(f"DTC {dtc:.1f} days — CRITICAL. Shorts are completely trapped.")
        elif dtc >= 10:
            squeeze += 11
            green.append(f"DTC {dtc:.1f} days — very dangerous for shorts.")
        elif dtc >= GILL_DTC_MIN:
            squeeze += 7
            green.append(f"DTC {dtc:.1f} days — above 5-day threshold. Exit door is narrow.")
        elif dtc >= 2:
            squeeze += 3
            red.append(f"DTC {dtc:.1f} days — shorts can exit relatively easily.")
        else:
            red.append(f"DTC {dtc:.1f} days — shorts can cover in under 2 days. No trap.")
    else:
        red.append("Days to cover unavailable.")

    # Cost to Borrow Proxy (max 8 pts)
    ctb = m.ctb_proxy
    if ctb is not None:
        if ctb >= 50:
            squeeze += 8
            green.append(f"CTB proxy {ctb:.0f}% — borrow is extremely scarce.")
        elif ctb >= GILL_CTB_PROXY_MIN:
            squeeze += 5
            green.append(f"CTB proxy {ctb:.0f}% — meaningful borrow cost.")
        else:
            squeeze += 2
            red.append(f"CTB proxy {ctb:.0f}% — borrow appears readily available.")
    else:
        red.append("Cost to borrow proxy unavailable.")

    g.squeeze_setup_score = min(squeeze, 40)

    # ── PILLAR 2: Fundamental Quality Score (35 pts) ──
    # Gill's core insight: GME wasn't just a squeeze, it was undervalued.
    # He needed the business to have REAL value, not be a zombie.
    fund = 0.0

    # Not insanely overvalued (max 10 pts)
    pe = m.pe_ratio
    if pe is not None and pe > 0:
        if pe <= 10:
            fund += 10
            green.append(f"P/E {pe:.1f}x — deeply cheap. Shorts are wrong on valuation.")
        elif pe <= 20:
            fund += 7
            green.append(f"P/E {pe:.1f}x — reasonable valuation.")
        elif pe <= GILL_PE_MAX:
            fund += 4
            red.append(f"P/E {pe:.1f}x — fair to slightly elevated.")
        else:
            red.append(f"P/E {pe:.1f}x — expensive. Shorts may have valuation right.")
    elif pe is None:
        fund += 3   # benefit of doubt for non-earnings companies
        red.append("P/E unavailable — may be unprofitable.")

    # Revenue growth (max 10 pts)
    rev = m.revenue_growth
    if rev is not None:
        if rev >= 0.20:
            fund += 10
            green.append(f"Revenue growing {rev:.0%} — company is growing into the story.")
        elif rev >= GILL_REVENUE_GROWTH_MIN:
            fund += 6
            green.append(f"Revenue growth {rev:.0%} — positive trajectory.")
        else:
            fund += 1
            red.append(f"Revenue declining {rev:.0%} — shorts may have fundamental thesis right.")
    else:
        red.append("Revenue growth data unavailable.")

    # Free cash flow (max 8 pts)
    fcf = m.free_cash_flow
    if fcf is not None:
        if fcf > 0:
            fund += 8
            green.append(f"Positive FCF ${fcf/1e6:.0f}M — real business, not burning cash.")
        else:
            fund += 1
            red.append(f"Negative FCF ${fcf/1e6:.0f}M — cash burn is a concern.")

    # Debt (max 7 pts)
    dte = m.debt_to_equity
    if dte is not None:
        if dte < 0.5:
            fund += 7
            green.append(f"Debt/equity {dte:.2f} — clean balance sheet. Squeeze has time to play out.")
        elif dte < 1.5:
            fund += 4
            green.append(f"Debt/equity {dte:.2f} — manageable.")
        else:
            red.append(f"Debt/equity {dte:.2f} — heavy debt limits runway.")

    g.fundamental_score = min(fund, 35)

    # ── PILLAR 3: Catalyst Score (25 pts) ──
    cat = 0.0

    # Volume surge (max 12 pts)
    vs = m.volume_surge
    if vs is not None:
        if vs >= 3.0:
            cat += 12
            green.append(f"Volume surge {vs:.1f}x normal — someone is loading up. Retail awakening?")
        elif vs >= 2.0:
            cat += 8
            green.append(f"Volume surge {vs:.1f}x normal — elevated interest.")
        elif vs >= 1.3:
            cat += 4
            green.append(f"Volume slightly elevated ({vs:.1f}x) — early signal.")
        else:
            red.append(f"Volume normal ({vs:.1f}x) — no crowd forming yet.")

    # Short interest INCREASING (shorts doubling down = bigger squeeze when it pops)
    sc = m.short_change_pct
    if sc is not None:
        if sc > 0.10:
            cat += 7
            green.append(f"Short interest grew {sc:.0%} last month — shorts adding conviction. FUEL for squeeze.")
        elif sc < -0.10:
            cat += 2
            red.append(f"Short interest decreased {sc:.0%} — shorts already covering. Squeeze partially played out.")
        else:
            cat += 4
            green.append("Short interest stable — no capitulation yet.")

    # Price not already squeezed (RSI check)
    rsi = m.rsi_14
    if rsi is not None:
        if 30 <= rsi <= 60:
            cat += 6
            green.append(f"RSI {rsi:.0f} — not overbought. Squeeze hasn't started yet.")
        elif rsi < 30:
            cat += 6
            green.append(f"RSI {rsi:.0f} — OVERSOLD. Shorts are winning but the setup is building.")
        elif rsi <= 75:
            cat += 3
            red.append(f"RSI {rsi:.0f} — momentum building but watch for overextension.")
        else:
            red.append(f"RSI {rsi:.0f} — OVERBOUGHT. Squeeze may have already begun or nearly over.")

    g.catalyst_score = min(cat, 25)

    # ── Total & Verdict ──
    g.total_score = g.squeeze_setup_score + g.fundamental_score + g.catalyst_score
    g.green_flags = green
    g.red_flags   = red

    if g.total_score >= 75:
        g.verdict    = "SQUEEZE CANDIDATE"
        g.conviction = "YOLO"
    elif g.total_score >= 55:
        g.verdict    = "SQUEEZE CANDIDATE"
        g.conviction = "HIGH"
    elif g.total_score >= 38:
        g.verdict    = "WATCH"
        g.conviction = "MODERATE"
    else:
        g.verdict    = "PASS"
        g.conviction = "LOW"

    # Build Gill-style thesis
    si_str  = f"{si:.0%}" if si else "unknown"
    dtc_str = f"{dtc:.1f}" if dtc else "unknown"
    g.thesis = (
        f"{ticker.upper()} has {si_str} of float short with {dtc_str} days to cover. "
        f"Squeeze setup score: {g.squeeze_setup_score:.0f}/40 | "
        f"Fundamental score: {g.fundamental_score:.0f}/35 | "
        f"Catalyst score: {g.catalyst_score:.0f}/25. "
        f"Verdict: {g.verdict} ({g.conviction} conviction)."
    )

    return g


# ─────────────────────────────────────────────
# CHAMATH PALIHAPITIYA ANALYZER
# ─────────────────────────────────────────────

def run_chamath_analysis(ticker: str) -> ChamathAnalysis:
    """
    Chamath's squeeze framework:
    1. Macro Setup: Is there a narrative shift that invalidates the short thesis?
    2. Squeeze Pressure: Are the short metrics building toward an explosion?
    3. Catalyst Momentum: Insider conviction + retail momentum + options flow?

    Chamath's approach: find narrative-driven inflection points where
    institutional short sellers are caught on the wrong side of history.
    """
    c = ChamathAnalysis(ticker=ticker.upper())
    c.metrics = fetch_squeeze_metrics(ticker)
    m = c.metrics

    green = []
    red   = []

    # ── PILLAR 1: Macro / Narrative Setup (30 pts) ──
    # Chamath looks for structural shifts — when a sector is being re-rated
    macro = 0.0

    # Market cap sweet spot — not mega cap (hard to squeeze) not micro cap (too risky)
    mc = m.market_cap
    if mc is not None:
        if 500e6 <= mc <= 10e9:
            macro += 12
            green.append(f"Market cap ${mc/1e9:.1f}B — ideal squeeze size. Big enough to matter, small enough to move.")
        elif 10e9 < mc <= 50e9:
            macro += 7
            green.append(f"Market cap ${mc/1e9:.1f}B — larger cap squeeze needs bigger catalyst.")
        elif mc < 500e6:
            macro += 4
            red.append(f"Market cap ${mc/1e6:.0f}M — micro cap. Chamath prefers more liquid names.")
        else:
            red.append(f"Market cap ${mc/1e9:.0f}B — mega cap. Squeeze mathematically harder.")

    # Institutional + insider ownership (skin in the game)
    ins = m.insider_ownership
    inst = m.institutional_ownership
    if ins is not None:
        if ins >= CHAMATH_INSIDER_THRESHOLD:
            macro += 10
            green.append(f"Insider ownership {ins:.0%} — management has real skin in the game. Squeeze helps them too.")
        elif ins >= 0.02:
            macro += 5
            green.append(f"Insider ownership {ins:.0%} — some alignment.")
        else:
            red.append(f"Insider ownership {ins:.0%} — management not aligned. Chamath cautious.")

    if inst is not None:
        if 0.40 <= inst <= 0.80:
            macro += 8
            green.append(f"Institutional ownership {inst:.0%} — institutional consensus can flip rapidly.")
        elif inst > 0.80:
            macro += 4
            red.append(f"Institutional ownership {inst:.0%} — heavily owned. Institutions ARE the short.")
        else:
            macro += 5
            green.append(f"Institutional ownership {inst:.0%} — low institutional, retail can drive narrative.")

    c.macro_setup_score = min(macro, 30)

    # ── PILLAR 2: Squeeze Pressure Score (35 pts) ──
    pressure = 0.0

    si = m.short_interest_pct
    dtc = m.days_to_cover

    # Short interest (max 18 pts)
    if si is not None:
        if si >= 0.40:
            pressure += 18
            green.append(f"Short interest {si:.0%} — maximum squeeze pressure. Chamath's dream setup.")
        elif si >= CHAMATH_SHORT_INTEREST_MIN:
            pressure += 12
            green.append(f"Short interest {si:.0%} — meaningful. Narrative flip could cascade.")
        elif si >= 0.08:
            pressure += 5
            red.append(f"Short interest {si:.0%} — modest. Needs strong catalyst to squeeze.")
        else:
            red.append(f"Short interest {si:.0%} — insufficient short pressure.")

    # Days to Cover (max 12 pts)
    if dtc is not None:
        if dtc >= 15:
            pressure += 12
            green.append(f"DTC {dtc:.1f} — catastrophically trapped. Cascade covering certain if triggered.")
        elif dtc >= GILL_DTC_MIN:
            pressure += 8
            green.append(f"DTC {dtc:.1f} — tight exit. Chamath: 'when they run, they all run at once'.")
        elif dtc >= 2:
            pressure += 3
            red.append(f"DTC {dtc:.1f} — shorts can exit. Squeeze needs to be fast.")

    # CTB proxy (max 5 pts)
    ctb = m.ctb_proxy
    if ctb is not None and ctb >= GILL_CTB_PROXY_MIN:
        pressure += 5
        green.append(f"CTB proxy {ctb:.0f}% — expensive to short. New shorts deterred.")

    c.squeeze_pressure_score = min(pressure, 35)

    # ── PILLAR 3: Catalyst Momentum Score (35 pts) ──
    catalyst = 0.0

    # Price momentum — Chamath needs momentum, unlike Gill who can wait
    p1m = m.price_change_1m
    p3m = m.price_change_3m
    if p1m is not None:
        if p1m >= 0.30:
            catalyst += 12
            green.append(f"1-month return {p1m:.0%} — MOMENTUM IGNITED. Chamath's catalyst may be here.")
        elif p1m >= CHAMATH_MOMENTUM_MIN:
            catalyst += 7
            green.append(f"1-month return {p1m:.0%} — positive momentum building.")
        elif p1m >= -0.10:
            catalyst += 3
            red.append(f"1-month return {p1m:.0%} — flat. Waiting for catalyst.")
        else:
            catalyst += 0
            red.append(f"1-month return {p1m:.0%} — declining. Shorts still winning.")

    if p3m is not None:
        if p3m >= 0.50:
            catalyst += 8
            green.append(f"3-month return {p3m:.0%} — strong sustained momentum. Chamath loves this.")
        elif p3m >= 0.15:
            catalyst += 5
            green.append(f"3-month return {p3m:.0%} — trend is your friend.")
        elif p3m < -0.20:
            red.append(f"3-month return {p3m:.0%} — weak trend. Chamath waits for reversal signal.")

    # Volume and short change confirmation
    vs = m.volume_surge
    if vs is not None and vs >= 2.0:
        catalyst += 8
        green.append(f"Volume surge {vs:.1f}x — crowd forming. Retail + institutional momentum.")

    sc = m.short_change_pct
    if sc is not None:
        if sc >= 0.20:
            catalyst += 7
            green.append(f"Short interest surged {sc:.0%} — shorts doubling down = more fuel when catalyst hits.")
        elif sc <= -0.15:
            red.append(f"Short interest dropped {sc:.0%} — early covering. Best squeeze entry may have passed.")
        else:
            catalyst += 3

    c.catalyst_momentum_score = min(catalyst, 35)

    # ── Total & Verdict ──
    c.total_score = (c.macro_setup_score +
                     c.squeeze_pressure_score +
                     c.catalyst_momentum_score)
    c.green_flags = green
    c.red_flags   = red

    if c.total_score >= 70:
        c.verdict = "SQUEEZE CANDIDATE"
    elif c.total_score >= 50:
        c.verdict = "WATCH — Building Setup"
    else:
        c.verdict = "PASS"

    si_str  = f"{si:.0%}" if si else "N/A"
    dtc_str = f"{dtc:.1f}d" if dtc else "N/A"
    mc_str  = f"${mc/1e9:.1f}B" if mc else "N/A"

    c.narrative = (
        f"Macro: {c.macro_setup_score:.0f}/30 | "
        f"Squeeze pressure: {c.squeeze_pressure_score:.0f}/35 | "
        f"Catalyst: {c.catalyst_momentum_score:.0f}/35"
    )
    c.thesis = (
        f"{ticker.upper()} ({mc_str} mkt cap): SI {si_str}, DTC {dtc_str}. "
        f"Total: {c.total_score:.0f}/100 — {c.verdict}."
    )

    return c


# ─────────────────────────────────────────────
# DISPLAY FORMATTERS
# ─────────────────────────────────────────────

def format_gill_display(g: GillAnalysis) -> str:
    m = g.metrics

    def pct(v): return f"{v:.1%}" if v is not None else "N/A"
    def n(v, d=2): return f"{v:.{d}f}" if v is not None else "N/A"

    lines = [
        "",
        f"  ── SQUEEZE SETUP ({g.squeeze_setup_score:.0f}/40) ────────────────────────────────",
        f"  Short Interest % Float:  {pct(m.short_interest_pct):<12} threshold > 20%",
        f"  Days to Cover (DTC):     {n(m.days_to_cover, 1):<12} threshold > 5 days",
        f"  Cost-to-Borrow Proxy:    {n(m.ctb_proxy, 1)}%        threshold > 10%",
        f"  Short Change (1mo):      {pct(m.short_change_pct):<12} (+ = shorts adding)",
        "",
        f"  ── FUNDAMENTAL QUALITY ({g.fundamental_score:.0f}/35) ─────────────────────────────",
        f"  P/E Ratio:               {n(m.pe_ratio, 1):<12} threshold < 30x",
        f"  Revenue Growth:          {pct(m.revenue_growth):<12} threshold > 0%",
        f"  Free Cash Flow:          {'${:,.0f}M'.format(m.free_cash_flow/1e6) if m.free_cash_flow else 'N/A':<12}",
        f"  Debt / Equity:           {n(m.debt_to_equity):<12}",
        "",
        f"  ── CATALYST SIGNALS ({g.catalyst_score:.0f}/25) ───────────────────────────────────",
        f"  Volume Surge:            {n(m.volume_surge, 1)+'x':<12} (vs 20d avg)",
        f"  RSI (14):                {n(m.rsi_14, 0):<12}",
        f"  1-Month Return:          {pct(m.price_change_1m):<12}",
        "",
        f"  ── GILL VERDICT ───────────────────────────────────────────────",
        f"  Total Score:   {g.total_score:.0f}/100",
        f"  Verdict:       {g.verdict}",
        f"  Conviction:    {g.conviction}",
        "",
        f"  GREEN FLAGS:",
    ]
    for flag in g.green_flags:
        lines.append(f"    ✅ {flag}")
    lines.append(f"  RED FLAGS:")
    for flag in g.red_flags:
        lines.append(f"    ❌ {flag}")
    lines.append("")
    return "\n".join(lines)


def format_chamath_display(c: ChamathAnalysis) -> str:
    m = c.metrics

    def pct(v): return f"{v:.1%}" if v is not None else "N/A"
    def n(v, d=2): return f"{v:.{d}f}" if v is not None else "N/A"

    lines = [
        "",
        f"  ── MACRO SETUP ({c.macro_setup_score:.0f}/30) ──────────────────────────────────────",
        f"  Market Cap:              {'${:,.0f}B'.format(m.market_cap/1e9) if m.market_cap else 'N/A':<12}",
        f"  Insider Ownership:       {pct(m.insider_ownership):<12} threshold > 5%",
        f"  Institutional Own:       {pct(m.institutional_ownership):<12}",
        "",
        f"  ── SQUEEZE PRESSURE ({c.squeeze_pressure_score:.0f}/35) ──────────────────────────────",
        f"  Short Interest % Float:  {pct(m.short_interest_pct):<12} threshold > 15%",
        f"  Days to Cover (DTC):     {n(m.days_to_cover, 1):<12} threshold > 5 days",
        f"  Cost-to-Borrow Proxy:    {n(m.ctb_proxy, 1)}%",
        f"  Short Change (1mo):      {pct(m.short_change_pct):<12}",
        "",
        f"  ── CATALYST MOMENTUM ({c.catalyst_momentum_score:.0f}/35) ─────────────────────────────",
        f"  1-Month Return:          {pct(m.price_change_1m):<12}",
        f"  3-Month Return:          {pct(m.price_change_3m):<12}",
        f"  Volume Surge:            {n(m.volume_surge, 1)+'x':<12}",
        f"  RSI (14):                {n(m.rsi_14, 0):<12}",
        "",
        f"  ── CHAMATH VERDICT ─────────────────────────────────────────────",
        f"  {c.narrative}",
        f"  Total Score:   {c.total_score:.0f}/100",
        f"  Verdict:       {c.verdict}",
        "",
        f"  GREEN FLAGS:",
    ]
    for flag in c.green_flags:
        lines.append(f"    ✅ {flag}")
    lines.append(f"  RED FLAGS:")
    for flag in c.red_flags:
        lines.append(f"    ❌ {flag}")
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "GME"
    print(f"\n{'='*60}")
    print(f"KEITH GILL ANALYSIS — {ticker.upper()}")
    print('='*60)
    gill = run_gill_analysis(ticker)
    print(format_gill_display(gill))

    print(f"\n{'='*60}")
    print(f"CHAMATH ANALYSIS — {ticker.upper()}")
    print('='*60)
    chamath = run_chamath_analysis(ticker)
    print(format_chamath_display(chamath))
