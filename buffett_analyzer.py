"""
buffett_analyzer.py
====================
Concrete Buffett financial analysis pipeline.
Pulls real data via yfinance, calculates metrics deterministically,
then passes a structured fact sheet to the LLM for interpretation.

Requires: pip install yfinance requests
"""

import yfinance as yf
import requests
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

# ─────────────────────────────────────────────
# BUFFETT THRESHOLDS (source: documented philosophy)
# ─────────────────────────────────────────────
ROIC_MIN            = 0.15   # > 15%
GROSS_MARGIN_MIN    = 0.40   # > 40%
DEBT_TO_EQUITY_MAX  = 0.50   # < 0.5
MARGIN_OF_SAFETY    = 0.25   # wants 25% discount to intrinsic value
DCF_GROWTH_YEARS    = 10     # project FCF for 10 years
TERMINAL_GROWTH     = 0.03   # 3% perpetual growth after year 10
DISCOUNT_RATE       = 0.09   # Buffett's ~9% hurdle rate

# Buffett Indicator thresholds (total market cap / GDP)
BUFFETT_IND_FAIR       = 1.00   # 100% = fairly valued
BUFFETT_IND_OVERVALUED = 1.20   # 120% = overvalued
BUFFETT_IND_EXTREME    = 1.50   # 150% = extreme bubble territory


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────
@dataclass
class MoatMetrics:
    roic:               Optional[float] = None   # Return on Invested Capital
    gross_margin:       Optional[float] = None   # Gross profit margin
    debt_to_equity:     Optional[float] = None   # D/E ratio
    net_income:         Optional[float] = None   # TTM net income
    free_cash_flow:     Optional[float] = None   # TTM FCF
    fcf_to_net_income:  Optional[float] = None   # FCF quality ratio
    roe:                Optional[float] = None   # Return on equity (supplemental)
    operating_margin:   Optional[float] = None   # Operating margin

@dataclass
class ValuationMetrics:
    current_price:      Optional[float] = None
    eps_ttm:            Optional[float] = None
    earnings_yield:     Optional[float] = None   # 1 / P/E
    treasury_10yr:      Optional[float] = None   # current 10yr yield
    margin_vs_treasury: Optional[float] = None   # earnings yield - treasury yield
    fcf_per_share:      Optional[float] = None
    shares_outstanding: Optional[float] = None
    dcf_intrinsic_value:Optional[float] = None   # 10yr DCF result
    dcf_upside_pct:     Optional[float] = None   # % upside/downside to intrinsic
    dcf_growth_assumed: Optional[float] = None   # growth rate used in DCF
    dcf_growth_source:  str = ""                   # where the growth rate came from
    pe_ratio:           Optional[float] = None
    forward_pe:         Optional[float] = None
    market_cap:         Optional[float] = None

@dataclass
class BuffettIndicator:
    total_market_cap_usd:   Optional[float] = None   # Wilshire 5000 proxy
    gdp_usd:                Optional[float] = None   # US GDP (annualized)
    ratio:                  Optional[float] = None   # market cap / GDP
    signal:                 str = "UNKNOWN"          # FAIR / CAUTION / OVERVALUED / EXTREME

@dataclass
class MoatScore:
    roic_pass:          bool = False
    gross_margin_pass:  bool = False
    debt_pass:          bool = False
    fcf_quality_pass:   bool = False
    score:              int = 0          # 0-4 metrics passing
    rating:             str = "WEAK"    # WEAK / MODERATE / STRONG
    flags:              list = field(default_factory=list)

@dataclass
class BuffettAnalysis:
    ticker:             str = ""
    company_name:       str = ""
    sector:             str = ""
    industry:           str = ""
    analysis_date:      str = ""
    moat:               MoatMetrics = field(default_factory=MoatMetrics)
    valuation:          ValuationMetrics = field(default_factory=ValuationMetrics)
    buffett_indicator:  BuffettIndicator = field(default_factory=BuffettIndicator)
    moat_score:         MoatScore = field(default_factory=MoatScore)
    errors:             list = field(default_factory=list)


# ─────────────────────────────────────────────
# DATA FETCHERS
# ─────────────────────────────────────────────
def fetch_ticker_data(ticker: str) -> dict:
    """Pull all relevant yfinance data for a ticker."""
    t = yf.Ticker(ticker)
    info = t.info

    # Cash flow statement — annual, most recent year
    cf = t.cashflow
    fcf = None
    try:
        # FCF = Operating Cash Flow - Capital Expenditures
        op_cf = cf.loc["Operating Cash Flow"].iloc[0] if "Operating Cash Flow" in cf.index else None
        capex = cf.loc["Capital Expenditure"].iloc[0] if "Capital Expenditure" in cf.index else 0
        if op_cf is not None:
            fcf = float(op_cf) + float(capex)   # capex is negative in yfinance
    except Exception:
        fcf = info.get("freeCashflow")

    # ROIC calculation: Net Income / (Total Equity + Total Debt)
    roic = None
    try:
        bs = t.balance_sheet
        net_income = info.get("netIncomeToCommon") or info.get("netIncome")
        total_equity = bs.loc["Stockholders Equity"].iloc[0] if "Stockholders Equity" in bs.index else None
        total_debt_raw = bs.loc["Total Debt"].iloc[0] if "Total Debt" in bs.index else None
        if net_income and total_equity and total_debt_raw:
            invested_capital = float(total_equity) + float(total_debt_raw)
            if invested_capital > 0:
                roic = float(net_income) / invested_capital
    except Exception:
        pass

    return {
        "info":   info,
        "fcf":    fcf,
        "roic":   roic,
    }


def fetch_treasury_yield() -> Optional[float]:
    """
    Fetch current 10-year US Treasury yield.
    Uses ^TNX ticker from yfinance (CBOE 10yr Treasury Note Yield Index).
    """
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1]) / 100.0   # convert % to decimal
    except Exception:
        pass
    return None


def fetch_buffett_indicator() -> BuffettIndicator:
    """
    Buffett Indicator = Total US Market Cap / US GDP
    Market cap proxy: Wilshire 5000 Total Market Index (^W5000) — represents total US market cap
    GDP: pulled from World Bank API (quarterly, annualized)
    """
    bi = BuffettIndicator()

    # Market cap via Wilshire 5000 (price * shares is embedded; index level IS the market cap proxy in trillions)
    try:
        w5000 = yf.Ticker("^W5000")
        hist = w5000.history(period="5d")
        if not hist.empty:
            # Wilshire 5000 index level ≈ total US market cap in billions (historical calibration)
            # The index was set so that 1 point ≈ $1 billion of market cap
            level = float(hist["Close"].iloc[-1])
            bi.total_market_cap_usd = level * 1e9   # convert to dollars
    except Exception:
        pass

    # GDP from World Bank (most recent annual US GDP in current USD)
    try:
        url = "https://api.worldbank.org/v2/country/US/indicator/NY.GDP.MKTP.CD?format=json&mrv=2"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data and len(data) > 1 and data[1]:
            for entry in data[1]:
                if entry.get("value"):
                    bi.gdp_usd = float(entry["value"])
                    break
    except Exception:
        # Fallback: use approximate current US GDP
        bi.gdp_usd = 29.0e12   # ~$29 trillion (2024 estimate)

    if bi.total_market_cap_usd and bi.gdp_usd:
        bi.ratio = bi.total_market_cap_usd / bi.gdp_usd
        if bi.ratio < BUFFETT_IND_FAIR:
            bi.signal = "FAIR VALUE"
        elif bi.ratio < BUFFETT_IND_OVERVALUED:
            bi.signal = "CAUTION"
        elif bi.ratio < BUFFETT_IND_EXTREME:
            bi.signal = "OVERVALUED"
        else:
            bi.signal = "EXTREME — BUBBLE TERRITORY"

    return bi


def calculate_dcf(fcf: float, shares: float, growth_rate: float) -> float:
    """
    Simple 2-stage DCF:
    Stage 1: Project FCF for DCF_GROWTH_YEARS at given growth_rate
    Stage 2: Terminal value at TERMINAL_GROWTH in perpetuity
    Discount everything back at DISCOUNT_RATE
    Returns: intrinsic value per share
    """
    if not fcf or not shares or shares == 0:
        return None

    pv_fcfs = 0.0
    current_fcf = fcf
    for year in range(1, DCF_GROWTH_YEARS + 1):
        current_fcf *= (1 + growth_rate)
        pv_fcfs += current_fcf / ((1 + DISCOUNT_RATE) ** year)

    # Terminal value
    terminal_fcf = current_fcf * (1 + TERMINAL_GROWTH)
    terminal_value = terminal_fcf / (DISCOUNT_RATE - TERMINAL_GROWTH)
    pv_terminal = terminal_value / ((1 + DISCOUNT_RATE) ** DCF_GROWTH_YEARS)

    total_pv = pv_fcfs + pv_terminal
    return total_pv / shares


def estimate_growth_rate(info: dict) -> tuple:
    """
    Estimate FCF growth rate for DCF. Returns (rate, source, warning).

    Priority order:
    1. Analyst 5yr EPS growth estimate (forwardEpsGrowth / earningsQuarterlyGrowth) — forward-looking
    2. Revenue growth — more stable than earnings for cyclical companies
    3. Conservative default of 5% — Buffett's baseline for quality companies

    Rules:
    - NEVER use a negative growth rate in a DCF — produces meaningless results
    - If all available data is negative, use 0% (flat) and flag it
    - Hard cap at 25% — Buffett is skeptical of high-growth assumptions
    - Always return what source was used so the display is transparent
    """
    candidates = []

    # 1. Best source: analyst forward EPS growth (5yr estimate)
    fwd = info.get("earningsQuarterlyGrowth")  # QoQ growth — proxy
    if fwd and float(fwd) > 0:
        candidates.append((float(fwd), "analyst quarterly EPS growth"))

    # 2. Revenue growth — more reliable for cyclicals than earnings
    rev = info.get("revenueGrowth")
    if rev and float(rev) > 0:
        candidates.append((float(rev), "revenue growth (YoY)"))

    # 3. Earnings growth — last resort, noisy for cyclicals
    earn = info.get("earningsGrowth")
    if earn and float(earn) > 0:
        candidates.append((float(earn), "earnings growth (YoY)"))

    if candidates:
        # Pick the most conservative positive estimate
        rate, source = min(candidates, key=lambda x: x[0])
        rate = min(rate, 0.25)  # cap at 25%
        warning = None
        if rate > 0.15:
            warning = f"High assumed growth ({rate:.1%}) — Buffett is skeptical; treat DCF as optimistic scenario"
        return rate, source, warning

    # All available data is negative or missing
    # Check if we have any data at all to explain why
    all_data = [info.get("earningsGrowth"), info.get("revenueGrowth"), info.get("earningsQuarterlyGrowth")]
    has_negative = any(v is not None and float(v) < 0 for v in all_data if v is not None)

    if has_negative:
        return 0.0, "0% (all available growth metrics are negative — cyclical or declining earnings)",                "⚠️  Growth data is negative — DCF uses 0% growth floor. Value likely understated for cyclical companies. Do not rely on this DCF for energy/commodity stocks."
    else:
        return 0.05, "5% (conservative default — no growth data available)",                "No growth data from yfinance — using 5% conservative default"


# ─────────────────────────────────────────────
# MOAT SCORER
# ─────────────────────────────────────────────
def score_moat(moat: MoatMetrics) -> MoatScore:
    ms = MoatScore()
    flags = []

    # ROIC
    if moat.roic is not None:
        if moat.roic >= ROIC_MIN:
            ms.roic_pass = True
        else:
            flags.append(f"ROIC {moat.roic:.1%} is below Buffett's 15% threshold")

    # Gross Margin
    if moat.gross_margin is not None:
        if moat.gross_margin >= GROSS_MARGIN_MIN:
            ms.gross_margin_pass = True
        else:
            flags.append(f"Gross margin {moat.gross_margin:.1%} below 40% — possible commodity business")

    # Debt
    if moat.debt_to_equity is not None:
        if moat.debt_to_equity <= DEBT_TO_EQUITY_MAX:
            ms.debt_pass = True
        else:
            flags.append(f"Debt/Equity {moat.debt_to_equity:.2f} exceeds 0.5 — Buffett would flag this")

    # FCF Quality (FCF should be ≥ 70% of net income — "creative accounting" flag otherwise)
    if moat.fcf_to_net_income is not None:
        if moat.fcf_to_net_income >= 0.70:
            ms.fcf_quality_pass = True
        else:
            flags.append(
                f"FCF is only {moat.fcf_to_net_income:.0%} of net income — "
                f"Buffett suspects 'creative accounting' when FCF lags earnings"
            )
    elif moat.free_cash_flow is not None:
        # FCF exists but we couldn't compute ratio — still pass
        ms.fcf_quality_pass = True

    ms.score = sum([ms.roic_pass, ms.gross_margin_pass, ms.debt_pass, ms.fcf_quality_pass])
    ms.flags = flags

    if ms.score >= 4:
        ms.rating = "STRONG"
    elif ms.score >= 2:
        ms.rating = "MODERATE"
    else:
        ms.rating = "WEAK"

    return ms


# ─────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────
def run_buffett_analysis(ticker: str) -> BuffettAnalysis:
    """
    Full pipeline: fetch data → calculate metrics → score moat → return structured fact sheet.
    """
    analysis = BuffettAnalysis(
        ticker=ticker.upper(),
        analysis_date=datetime.now().strftime("%Y-%m-%d")
    )

    # ── FETCH ──
    try:
        raw = fetch_ticker_data(ticker)
        info = raw["info"]
        fcf_raw = raw["fcf"]
        roic_raw = raw["roic"]
    except Exception as e:
        analysis.errors.append(f"Data fetch failed: {e}")
        return analysis

    analysis.company_name = info.get("longName", ticker)
    analysis.sector       = info.get("sector", "Unknown")
    analysis.industry     = info.get("industry", "Unknown")

    # ── DATA VALIDATION GATE ──
    from data_validator import validate, format_validation_header, cannot_conclude_prompt
    dq = validate(ticker, info, "buffett")
    analysis.errors.append(f"VALIDATION:{dq.confidence}:{dq.can_analyze}:{dq.asset_type}")

    if not dq.can_analyze:
        # Store gate info so the caller can use cannot_conclude_prompt
        analysis.errors.append(f"GATE_REASON:{dq.gate_reason}")
        analysis.errors.append(f"DQ_OBJECT:blocked")
        # Still return partial analysis with what we have
        return analysis

    if dq.warnings:
        for w in dq.warnings:
            analysis.errors.append(f"WARNING:{w}")

    # ── MOAT METRICS ──
    m = analysis.moat
    m.roic          = roic_raw
    def safe_pct(val):
        if val is None: return None
        v = float(val)
        return v / 100.0 if v > 1.0 else v

    m.gross_margin     = safe_pct(info.get("grossMargins"))
    m.debt_to_equity   = info.get("debtToEquity")
    if m.debt_to_equity:
        m.debt_to_equity = m.debt_to_equity / 100.0
    m.free_cash_flow   = fcf_raw
    m.net_income       = info.get("netIncomeToCommon") or info.get("netIncome")
    m.roe              = safe_pct(info.get("returnOnEquity"))
    m.operating_margin = safe_pct(info.get("operatingMargins"))

    if m.free_cash_flow and m.net_income and m.net_income != 0:
        m.fcf_to_net_income = m.free_cash_flow / m.net_income

    # ── VALUATION METRICS ──
    v = analysis.valuation
    v.current_price     = info.get("currentPrice") or info.get("regularMarketPrice")
    v.eps_ttm           = info.get("trailingEps")
    v.pe_ratio          = info.get("trailingPE")
    v.forward_pe        = info.get("forwardPE")
    v.market_cap        = info.get("marketCap")
    v.shares_outstanding= info.get("sharesOutstanding")

    if v.pe_ratio and v.pe_ratio > 0:
        v.earnings_yield = 1.0 / v.pe_ratio

    # 10yr Treasury
    v.treasury_10yr = fetch_treasury_yield()

    if v.earnings_yield and v.treasury_10yr:
        v.margin_vs_treasury = v.earnings_yield - v.treasury_10yr

    # FCF per share
    if m.free_cash_flow and v.shares_outstanding and v.shares_outstanding > 0:
        v.fcf_per_share = m.free_cash_flow / v.shares_outstanding

    # DCF Intrinsic Value
    if m.free_cash_flow and v.shares_outstanding:
        growth, growth_source, growth_warning = estimate_growth_rate(info)
        v.dcf_growth_assumed = growth
        v.dcf_growth_source = growth_source if hasattr(v, "dcf_growth_source") else growth_source
        if growth_warning:
            analysis.errors.append(f"DCF_WARNING:{growth_warning}")
        v.dcf_intrinsic_value = calculate_dcf(m.free_cash_flow, v.shares_outstanding, growth)
        if v.dcf_intrinsic_value and v.current_price:
            v.dcf_upside_pct = (v.dcf_intrinsic_value - v.current_price) / v.current_price

    # NOTE: Buffett Indicator is now fetched once at app startup as macro context.
    # It is displayed in the session header, not per stock.

    # ── MOAT SCORE ──
    analysis.moat_score = score_moat(analysis.moat)

    return analysis


# ─────────────────────────────────────────────
# FORMAT FOR LLM — this is what the LLM receives
# ─────────────────────────────────────────────
def format_for_llm(analysis: BuffettAnalysis, portfolio_context: str = "") -> str:
    """
    Convert the analysis into a structured fact sheet for the LLM.
    The LLM's ONLY job is to interpret these numbers through Buffett's lens.
    """
    a = analysis
    m = a.moat
    v = a.valuation
    bi = a.buffett_indicator
    ms = a.moat_score

    def fmt_pct(val, decimals=1):
        if val is None: return "N/A"
        return f"{val:.{decimals}%}"

    def fmt_dollar(val):
        if val is None: return "N/A"
        if abs(val) >= 1e12: return f"${val/1e12:.2f}T"
        if abs(val) >= 1e9:  return f"${val/1e9:.2f}B"
        if abs(val) >= 1e6:  return f"${val/1e6:.2f}M"
        return f"${val:.2f}"

    def fmt_num(val, decimals=2):
        if val is None: return "N/A"
        return f"{val:.{decimals}f}"

    def pass_fail(passed: bool, val_str: str, threshold_str: str):
        symbol = "✅ PASS" if passed else "❌ FAIL"
        return f"{symbol}  |  {val_str}  (threshold: {threshold_str})"

    block = f"""
================================================================================
BUFFETT ANALYSIS FACT SHEET — {a.ticker} ({a.company_name})
Sector: {a.sector} | Industry: {a.industry} | Date: {a.analysis_date}
================================================================================

YOUR ROLE: You are Warren Buffett. Interpret the metrics below through your
documented investment philosophy. Reference the specific numbers. Be direct
and opinionated. Do NOT invent data — only reference what is provided here.
Maximum 250 words. No generic commentary — address the actual numbers.

────────────────────────────────────────────────────────────────────────────────
SECTION 1 — MOAT METRICS (Competitive Advantage Assessment)
────────────────────────────────────────────────────────────────────────────────

ROIC (Return on Invested Capital):
  {pass_fail(ms.roic_pass, fmt_pct(m.roic), "> 15%")}
  Interpretation: Every dollar invested generates {fmt_pct(m.roic)} in profit.

Gross Margin:
  {pass_fail(ms.gross_margin_pass, fmt_pct(m.gross_margin), "> 40%")}
  Interpretation: {"Pricing power evident — commodity businesses can't sustain this." if ms.gross_margin_pass else "Thin margins suggest competitive pressure or commodity-like pricing."}

Debt/Equity Ratio:
  {pass_fail(ms.debt_pass, fmt_num(m.debt_to_equity, 2), "< 0.50")}
  Interpretation: {"Conservative balance sheet — growing on own cash." if ms.debt_pass else "Relies on debt to grow — Buffett would demand explanation."}

FCF Quality (FCF as % of Net Income):
  {pass_fail(ms.fcf_quality_pass, fmt_pct(m.fcf_to_net_income) if m.fcf_to_net_income else "FCF: " + fmt_dollar(m.free_cash_flow), ">= 70%")}
  Net Income: {fmt_dollar(m.net_income)} | Free Cash Flow: {fmt_dollar(m.free_cash_flow)}
  {"FCF closely tracks earnings — accounting appears clean." if ms.fcf_quality_pass else "⚠️  FCF lags net income significantly — Buffett suspects 'creative accounting'."}

Supplemental:
  ROE: {fmt_pct(m.roe)} | Operating Margin: {fmt_pct(m.operating_margin)}

MOAT SCORE: {ms.score}/4 — {ms.rating}
{"FLAGS: " + " | ".join(ms.flags) if ms.flags else "No flags."}

────────────────────────────────────────────────────────────────────────────────
SECTION 2 — FAIR PRICE / INTRINSIC VALUE
────────────────────────────────────────────────────────────────────────────────

Current Price:       ${v.current_price:.2f} (if available)
P/E (Trailing):      {fmt_num(v.pe_ratio, 1)}x
P/E (Forward):       {fmt_num(v.forward_pe, 1)}x
Earnings Yield:      {fmt_pct(v.earnings_yield)}  (inverse of P/E — what the stock "pays" you)
10yr Treasury Yield: {fmt_pct(v.treasury_10yr)}  (the risk-free alternative)
Earnings Yield vs Treasury: {fmt_pct(v.margin_vs_treasury)} spread
  {"✅ Stock yields MORE than treasuries — some compensation for equity risk." if (v.margin_vs_treasury or 0) > 0 else "❌ Stock yields LESS than treasuries — Buffett would not buy; risk not compensated."}

DCF Intrinsic Value Estimate:
  FCF/Share: {fmt_num(v.fcf_per_share, 2)}
  Growth rate assumed: {fmt_pct(v.dcf_growth_assumed)} — Source: {v.dcf_growth_source}
  Discount rate: {DISCOUNT_RATE:.0%} | Terminal growth: {TERMINAL_GROWTH:.0%}
  DCF Intrinsic Value: {fmt_dollar(v.dcf_intrinsic_value) if v.dcf_intrinsic_value else "N/A (insufficient FCF data)"}
  Upside / (Downside) to intrinsic value: {fmt_pct(v.dcf_upside_pct) if v.dcf_upside_pct else "N/A"}
  {"✅ Trading at discount — margin of safety present." if (v.dcf_upside_pct or 0) > MARGIN_OF_SAFETY else ("⚠️  Trading near intrinsic value — thin margin of safety." if (v.dcf_upside_pct or 0) > 0 else "❌ Trading above intrinsic value — no margin of safety.")}
  {"⚠️  DCF RELIABILITY NOTE: " + [e.replace("DCF_WARNING:","") for e in a.errors if "DCF_WARNING:" in e][0] if any("DCF_WARNING:" in e for e in a.errors) else ""}

{"────────────────────────────────────────────────────────────────────────────────" if portfolio_context else ""}
{"SECTION 3 — PORTFOLIO IMPACT CONTEXT" if portfolio_context else ""}
{"────────────────────────────────────────────────────────────────────────────────" if portfolio_context else ""}
{portfolio_context if portfolio_context else ""}
NOTE: The Buffett Indicator (market cap/GDP) is shown in the session header above — use it as macro context for your margin of safety assessment.

================================================================================
INSTRUCTION — Warren Buffett, answer ALL of these specifically:

1. MOAT VERDICT: Is the moat real or an illusion? Name the specific metric that proves it.
   If ROIC < 15%, say what that means in plain English for a business owner.

2. PRICE CHECK: Is this a fat pitch or a foul tip? Compare earnings yield to the 10yr treasury
   explicitly. If DCF shows negative upside, say how much overpayment that represents in dollars
   per share, not just a percentage.

3. vs JOHNATHAN'S EXISTING HOLDINGS: Compare this directly to something he already owns.
   Would you swap 10 shares of this for any of his current positions? Which one and why?

4. ALTERNATIVE SUGGESTION: If you would NOT buy this, name one specific stock you would
   prefer instead that fits a similar thesis — with one sentence on why.

5. ONE-LINE VERDICT: Buy at [price], Hold above [price], or Avoid entirely.
================================================================================
"""
    return block


# ─────────────────────────────────────────────
# DISPLAY — human-readable summary for the GUI
# ─────────────────────────────────────────────
def format_display_summary(analysis: BuffettAnalysis) -> str:
    """Compact display string for the chat window header before LLM response."""
    a = analysis
    m = a.moat
    v = a.valuation
    ms = a.moat_score
    bi = a.buffett_indicator

    def p(val, fmt=".1%"):
        return f"{val:{fmt}}" if val is not None else "N/A"

    def d(val):
        if val is None: return "N/A"
        if abs(val) >= 1e9: return f"${val/1e9:.1f}B"
        if abs(val) >= 1e6: return f"${val/1e6:.1f}M"
        return f"${val:.2f}"

    lines = [
        f"  {'Metric':<30} {'Value':>12}  {'Threshold':>12}  {'Pass?':>6}",
        f"  {'─'*65}",
        f"  {'ROIC':<30} {p(m.roic):>12}  {'> 15%':>12}  {'✅' if ms.roic_pass else '❌':>6}",
        f"  {'Gross Margin':<30} {p(m.gross_margin):>12}  {'> 40%':>12}  {'✅' if ms.gross_margin_pass else '❌':>6}",
        f"  {'Debt / Equity':<30} {p(m.debt_to_equity, '.2f') if m.debt_to_equity else 'N/A':>12}  {'< 0.50':>12}  {'✅' if ms.debt_pass else '❌':>6}",
        f"  {'FCF Quality (FCF/NI)':<30} {p(m.fcf_to_net_income) if m.fcf_to_net_income else 'see below':>12}  {'> 70%':>12}  {'✅' if ms.fcf_quality_pass else '❌':>6}",
        f"  {'─'*65}",
        f"  {'MOAT SCORE':<30} {ms.score}/4 — {ms.rating}",
        f"",
        f"  {'Current Price':<30} {'$'+str(round(v.current_price,2)) if v.current_price else 'N/A':>12}",
        f"  {'Earnings Yield':<30} {p(v.earnings_yield):>12}",
        f"  {'10yr Treasury Yield':<30} {p(v.treasury_10yr):>12}",
        f"  {'Yield Spread':<30} {p(v.margin_vs_treasury):>12}",
        f"  {'DCF Intrinsic Value':<30} {d(v.dcf_intrinsic_value):>12}",
        f"  {'Upside to Intrinsic':<30} {p(v.dcf_upside_pct):>12}",
        f"  {'DCF Growth Assumed':<30} {p(v.dcf_growth_assumed, fmt='.1%') if v.dcf_growth_assumed is not None else 'N/A':>12}",
        f"  {'Growth Rate Source':<30} {v.dcf_growth_source[:28] if v.dcf_growth_source else 'N/A':>12}",
    ]
    # Append DCF warning if present
    dcf_warnings = [e.replace("DCF_WARNING:", "") for e in (a.errors if hasattr(a, "errors") else []) if "DCF_WARNING:" in e]
    if dcf_warnings:
        lines.append(f"")
        lines.append(f"  ⚠️  DCF NOTE: {dcf_warnings[0][:80]}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    print(f"\nRunning Buffett analysis on {ticker}...\n")
    result = run_buffett_analysis(ticker)
    print(format_display_summary(result))
    print()
    print("─" * 80)
    print("LLM PROMPT THAT WOULD BE SENT:")
    print("─" * 80)
    print(format_for_llm(result))
