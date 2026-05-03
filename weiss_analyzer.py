"""
weiss_analyzer.py
==================
Geraldine Weiss's Investment Quality Trends methodology.

Two pillars:
1. Yield vs 10-year average yield — primary BUY/SELL signal
2. Seven Blue Chip Quality Criteria — scored 0-7, determines if stock qualifies

Thresholds sourced directly from Weiss's documented IQT methodology.

Requires: pip install yfinance requests
"""

import yfinance as yf
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# WEISS THRESHOLDS (from Investment Quality Trends)
# ─────────────────────────────────────────────
YIELD_HISTORY_YEARS     = 10     # Weiss used 10yr yield history
YIELD_BUY_THRESHOLD     = 0.95   # within 5% of 10yr high yield = BUY zone
YIELD_SELL_THRESHOLD    = 1.05   # within 5% of 10yr low yield = SELL zone

# Seven Criteria thresholds
DIV_GROWTH_YEARS        = 12     # must have 12 years of dividend history
DIV_GROWTH_RATE_MIN     = 0.10   # 10% compound annual growth
PE_MAX                  = 20     # P/E must be 20x or less
PAYOUT_MAX              = 0.50   # payout ratio 50% or less
DEBT_CAP_MAX            = 0.50   # debt as % of total capitalization, 50% or less
PRICE_TO_BOOK_MAX       = 2.0    # price/book 2x or less
EARNINGS_GROWTH_YEARS   = 12     # must evaluate 12 years
EARNINGS_GROWTH_MIN_YRS = 7      # earnings must improve in at least 7 of 12 years
SP_RATING_PASS          = {"A+", "A", "A-"}   # must be A-rated or better


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class YieldSignal:
    """Primary Weiss signal: current yield vs 10yr historical yield range."""
    current_yield:          Optional[float] = None
    avg_yield_10yr:         Optional[float] = None
    high_yield_10yr:        Optional[float] = None   # historically HIGH = undervalued
    low_yield_10yr:         Optional[float] = None   # historically LOW = overvalued
    yield_vs_avg:           Optional[float] = None   # % above/below 10yr average
    yield_percentile:       Optional[float] = None   # where current sits in 10yr range

    signal:                 str = "UNKNOWN"          # BUY / WATCH_BUY / HOLD / WATCH_SELL / SELL
    signal_strength:        str = "UNKNOWN"          # STRONG / MODERATE / WEAK
    reasoning:              str = ""
    years_of_data:          int = 0


@dataclass
class CriterionResult:
    name:           str = ""
    passed:         bool = False
    actual_value:   str = ""
    threshold:      str = ""
    note:           str = ""


@dataclass
class BlueChipScore:
    """Seven Weiss blue chip quality criteria."""
    criteria:               list = field(default_factory=list)   # list of CriterionResult
    score:                  int = 0       # 0-7 passing
    max_score:              int = 7
    rating:                 str = "UNKNOWN"   # BLUE CHIP / NEAR BLUE CHIP / SPECULATIVE
    qualifies:              bool = False
    sp_rating_note:         str = ""    # note about S&P rating data availability


@dataclass
class WeissAnalysis:
    ticker:             str = ""
    company_name:       str = ""
    sector:             str = ""
    analysis_date:      str = ""
    yield_signal:       YieldSignal = field(default_factory=YieldSignal)
    blue_chip:          BlueChipScore = field(default_factory=BlueChipScore)
    errors:             list = field(default_factory=list)


# ─────────────────────────────────────────────
# PILLAR 1: YIELD vs 10-YEAR AVERAGE
# ─────────────────────────────────────────────

def analyze_yield_signal(ticker: str, info: dict) -> YieldSignal:
    ys = YieldSignal()

    try:
        t = yf.Ticker(ticker)

        current_div_rate = info.get("dividendRate") or 0.0   # $0 = no dividend, not missing data
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")

        if not current_price or current_price == 0:
            ys.signal = "NO PRICE DATA"
            ys.reasoning = "Cannot calculate yield without a current price."
            return ys

        if current_div_rate == 0:
            ys.signal = "NON-DIVIDEND STOCK"
            ys.reasoning = (
                "This stock pays no dividend. Weiss's yield signal does not apply. "
                "The 7 Blue Chip Quality Criteria will still be evaluated where data is available. "
                "Weiss would generally not include non-dividend stocks in her portfolio."
            )
            return ys

        ys.current_yield = current_div_rate / current_price

        # Pull 10 years of monthly price history
        hist = t.history(period="10y", interval="1mo")
        if hist.empty:
            ys.signal = "INSUFFICIENT DATA"
            ys.reasoning = "Less than 10 years of price history available."
            return ys

        ys.years_of_data = len(hist) // 12

        # Calculate yield at each monthly close using CURRENT dividend rate
        # Note: ideally we'd use the dividend rate at each point in history,
        # but yfinance doesn't give per-month historical dividend rates cleanly.
        # Using current rate is a reasonable proxy — Weiss herself often did this
        # for stable dividend payers where the rate changes slowly.
        monthly_yields = []
        for _, row in hist.iterrows():
            price = row["Close"]
            if price and price > 0:
                monthly_yields.append(current_div_rate / price)

        if not monthly_yields:
            ys.signal = "INSUFFICIENT DATA"
            return ys

        yields_arr = np.array(monthly_yields)
        ys.avg_yield_10yr   = float(np.mean(yields_arr))
        ys.high_yield_10yr  = float(np.percentile(yields_arr, 90))   # 90th pct = historically high
        ys.low_yield_10yr   = float(np.percentile(yields_arr, 10))   # 10th pct = historically low

        # Where does current yield sit?
        yield_range = ys.high_yield_10yr - ys.low_yield_10yr
        if yield_range > 0:
            ys.yield_percentile = (ys.current_yield - ys.low_yield_10yr) / yield_range
        else:
            ys.yield_percentile = 0.5

        # vs 10yr average
        if ys.avg_yield_10yr > 0:
            ys.yield_vs_avg = (ys.current_yield - ys.avg_yield_10yr) / ys.avg_yield_10yr

        # ── WEISS SIGNAL ─────────────────────────────────────────────────────
        # Weiss: buy when yield is near its HISTORIC HIGH (stock is undervalued)
        #        sell when yield is near its HISTORIC LOW (stock is overvalued)
        # "Near" = within 5% of the threshold

        at_high = ys.current_yield >= ys.high_yield_10yr * YIELD_BUY_THRESHOLD
        near_high = ys.yield_percentile >= 0.75
        at_low = ys.current_yield <= ys.low_yield_10yr * YIELD_SELL_THRESHOLD
        near_low = ys.yield_percentile <= 0.25

        if at_high:
            ys.signal = "BUY"
            ys.signal_strength = "STRONG"
            ys.reasoning = (
                f"Current yield {ys.current_yield:.2%} is at or above the 10yr high yield zone "
                f"({ys.high_yield_10yr:.2%}). Weiss: stock is undervalued — this is a strong buy."
            )
        elif near_high:
            ys.signal = "WATCH — BUY ZONE"
            ys.signal_strength = "MODERATE"
            ys.reasoning = (
                f"Current yield {ys.current_yield:.2%} is in the upper quartile of its 10yr range "
                f"(75th percentile). Approaching but not yet at Weiss buy zone ({ys.high_yield_10yr:.2%})."
            )
        elif at_low:
            ys.signal = "SELL"
            ys.signal_strength = "STRONG"
            ys.reasoning = (
                f"Current yield {ys.current_yield:.2%} is at or below the 10yr low yield zone "
                f"({ys.low_yield_10yr:.2%}). Weiss: stock is overvalued — avoid or sell."
            )
        elif near_low:
            ys.signal = "WATCH — SELL ZONE"
            ys.signal_strength = "MODERATE"
            ys.reasoning = (
                f"Current yield {ys.current_yield:.2%} is in the lower quartile of its 10yr range "
                f"(25th percentile). Near Weiss sell zone ({ys.low_yield_10yr:.2%}) — caution."
            )
        else:
            ys.signal = "HOLD"
            ys.signal_strength = "NEUTRAL"
            ys.reasoning = (
                f"Current yield {ys.current_yield:.2%} is in the middle of its 10yr range "
                f"(avg: {ys.avg_yield_10yr:.2%}). Neither buy nor sell signal. "
                f"Percentile rank: {ys.yield_percentile:.0%}."
            )

    except Exception as e:
        ys.signal = "ERROR"
        ys.reasoning = str(e)

    return ys


# ─────────────────────────────────────────────
# PILLAR 2: SEVEN BLUE CHIP CRITERIA
# ─────────────────────────────────────────────

def _get_div_growth_cagr(ticker: str, years: int = 12) -> Optional[float]:
    """Calculate dividend CAGR over N years from dividend history."""
    try:
        t = yf.Ticker(ticker)
        divs = t.dividends
        if divs.empty:
            return None

        # Annual dividend sums
        divs.index = divs.index.tz_localize(None) if divs.index.tz else divs.index
        cutoff = datetime.now() - timedelta(days=years * 365)
        divs = divs[divs.index >= cutoff]

        if len(divs) < 4:
            return None

        annual = divs.resample("YE").sum()
        if len(annual) < 2:
            return None

        start = float(annual.iloc[0])
        end = float(annual.iloc[-1])
        n = len(annual) - 1

        if start <= 0 or end <= 0 or n <= 0:
            return None

        return (end / start) ** (1 / n) - 1

    except Exception:
        return None


def _get_earnings_improvement_count(ticker: str, years: int = 12) -> Optional[int]:
    """
    Count how many of the last N years showed EPS improvement vs prior year.
    Returns count of improving years out of (years-1) possible comparisons.
    """
    try:
        t = yf.Ticker(ticker)
        income = t.income_stmt

        if income is None or income.empty:
            return None

        if "Net Income" in income.index:
            earnings = income.loc["Net Income"]
        elif "Basic EPS" in income.index:
            earnings = income.loc["Basic EPS"]
        else:
            return None

        earnings = earnings.dropna().sort_index()
        if len(earnings) < 2:
            return None

        improvements = 0
        vals = list(earnings.values)
        for i in range(1, len(vals)):
            if vals[i] > vals[i - 1]:
                improvements += 1

        return improvements

    except Exception:
        return None


def score_blue_chip_criteria(ticker: str, info: dict) -> BlueChipScore:
    bc = BlueChipScore()
    criteria = []

    def safe_pct(val):
        """Normalize yfinance percentage fields — may come as 0.034 or 3.4 depending on ticker."""
        if val is None: return None
        v = float(val)
        return v / 100.0 if v > 1.0 else v

    # ── CRITERION 1: Dividend Growth ≥ 10% CAGR over 12 years ──────────────
    div_cagr = _get_div_growth_cagr(ticker, DIV_GROWTH_YEARS)
    c1 = CriterionResult(
        name="Dividend Growth (12yr CAGR ≥ 10%)",
        threshold=f"≥ {DIV_GROWTH_RATE_MIN:.0%}",
    )
    if div_cagr is not None:
        c1.actual_value = f"{div_cagr:.2%}"
        c1.passed = div_cagr >= DIV_GROWTH_RATE_MIN
        c1.note = "Strong dividend growth confirms company reinvests in shareholders." if c1.passed else \
                  f"Growth of {div_cagr:.2%} below Weiss's 10% threshold — dividend not growing fast enough."
    else:
        c1.actual_value = "N/A"
        c1.passed = False
        c1.note = "Insufficient dividend history — Weiss requires 12 years of consecutive dividends."
    criteria.append(c1)

    # ── CRITERION 2: P/E ≤ 20 ───────────────────────────────────────────────
    pe = info.get("trailingPE")
    c2 = CriterionResult(
        name="P/E Ratio (≤ 20x)",
        threshold="≤ 20x",
        actual_value=f"{pe:.1f}x" if pe else "N/A",
        passed=bool(pe and pe <= PE_MAX),
    )
    c2.note = "Reasonably priced relative to earnings." if c2.passed else \
              f"P/E of {pe:.1f}x exceeds 20x — Weiss says don't overpay for earnings." if pe else \
              "P/E unavailable."
    criteria.append(c2)

    # ── CRITERION 3: Payout Ratio ≤ 50% ────────────────────────────────────
    payout = safe_pct(info.get("payoutRatio"))
    c3 = CriterionResult(
        name="Payout Ratio (≤ 50%)",
        threshold="≤ 50%",
        actual_value=f"{payout:.1%}" if payout else "N/A",
        passed=bool(payout and payout <= PAYOUT_MAX),
    )
    c3.note = "Dividend is well-covered — room to grow." if c3.passed else \
              f"Payout of {payout:.1%} is elevated — dividend safety is a concern." if payout else \
              "Payout ratio unavailable."
    criteria.append(c3)

    # ── CRITERION 4: Debt ≤ 50% of Total Capitalization ───────────────────
    total_debt = info.get("totalDebt")
    market_cap = info.get("marketCap")
    c4 = CriterionResult(
        name="Debt (≤ 50% of total capitalization)",
        threshold="≤ 50%",
    )
    if total_debt and market_cap:
        total_cap = total_debt + market_cap
        debt_pct = total_debt / total_cap
        c4.actual_value = f"{debt_pct:.1%}"
        c4.passed = debt_pct <= DEBT_CAP_MAX
        c4.note = "Conservative leverage — low bankruptcy risk." if c4.passed else \
                  f"Debt is {debt_pct:.1%} of total cap — above Weiss's 50% threshold."
    else:
        # Fallback to D/E ratio
        de = info.get("debtToEquity")
        if de:
            de_decimal = de / 100
            # Convert D/E to debt as % of total cap: D/(D+E) = (D/E)/((D/E)+1)
            debt_pct_approx = de_decimal / (de_decimal + 1)
            c4.actual_value = f"~{debt_pct_approx:.1%} (from D/E {de_decimal:.2f})"
            c4.passed = debt_pct_approx <= DEBT_CAP_MAX
            c4.note = "Conservative leverage." if c4.passed else \
                      f"Estimated debt at {debt_pct_approx:.1%} of capital — borderline or elevated."
        else:
            c4.actual_value = "N/A"
            c4.passed = False
            c4.note = "Debt data unavailable."
    criteria.append(c4)

    # ── CRITERION 5: Price-to-Book ≤ 2x ────────────────────────────────────
    pb = info.get("priceToBook")
    c5 = CriterionResult(
        name="Price-to-Book (≤ 2x)",
        threshold="≤ 2.0x",
        actual_value=f"{pb:.2f}x" if pb else "N/A",
        passed=bool(pb and pb <= PRICE_TO_BOOK_MAX),
    )
    c5.note = "Trading near or below book value — classic Weiss value signal." if c5.passed else \
              f"P/B of {pb:.2f}x exceeds 2x — market is pricing in significant intangibles/goodwill." if pb else \
              "Book value data unavailable."
    criteria.append(c5)

    # ── CRITERION 6: Earnings Improved in ≥ 7 of 12 Years ─────────────────
    improving_yrs = _get_earnings_improvement_count(ticker, EARNINGS_GROWTH_YEARS)
    c6 = CriterionResult(
        name=f"Earnings Growth (≥ 7 of last 12 years improving)",
        threshold=f"≥ {EARNINGS_GROWTH_MIN_YRS} improving years",
    )
    if improving_yrs is not None:
        c6.actual_value = f"{improving_yrs} of {EARNINGS_GROWTH_YEARS - 1} years improved"
        c6.passed = improving_yrs >= EARNINGS_GROWTH_MIN_YRS
        c6.note = "Consistently profitable — confirms business quality." if c6.passed else \
                  f"Only {improving_yrs} improving years — Weiss requires steady earnings progression."
    else:
        c6.actual_value = "N/A (limited history)"
        c6.passed = False
        c6.note = "Insufficient earnings history to evaluate."
    criteria.append(c6)

    # ── CRITERION 7: S&P Quality Ranking A- or better ──────────────────────
    # yfinance does not provide S&P Quality Rankings directly.
    # We use a composite proxy: strong ROE + low debt + long dividend history
    roe = safe_pct(info.get("returnOnEquity"))
    div_yield = safe_pct(info.get("dividendYield"))
    years_public = info.get("sharesOutstanding")  # proxy for established company
    de_ratio = info.get("debtToEquity")

    c7 = CriterionResult(
        name="S&P Quality Rating (A- or better)",
        threshold="A / A- or higher",
    )

    # Proxy scoring: ROE > 15%, D/E < 1.0, has dividend, payout ratio reasonable
    proxy_points = 0
    proxy_reasons = []
    if roe and roe >= 0.15:
        proxy_points += 1
        proxy_reasons.append(f"ROE {roe:.1%}")
    if de_ratio and de_ratio < 100:   # D/E < 1.0
        proxy_points += 1
        proxy_reasons.append(f"D/E {de_ratio/100:.2f}")
    if div_yield and div_yield > 0:
        proxy_points += 1
        proxy_reasons.append("pays dividend")
    if payout and payout <= 0.60:
        proxy_points += 1
        proxy_reasons.append(f"payout {payout:.0%}")

    # If we have 3+ proxy signals, likely A-rated
    c7.passed = proxy_points >= 3
    if c7.passed:
        c7.actual_value = f"Likely A-range (proxy score {proxy_points}/4)"
        c7.note = f"Proxy indicators suggest investment-grade quality: {', '.join(proxy_reasons)}. Verify actual S&P rating independently."
    else:
        c7.actual_value = f"Below A-range proxy (score {proxy_points}/4)"
        c7.note = "Proxy indicators do not strongly suggest A-rated quality. Verify S&P rating — yfinance does not provide this directly."

    bc.sp_rating_note = "⚠️  S&P Quality Rankings not available via yfinance. Criterion 7 uses a proxy (ROE + D/E + dividend consistency). Verify at standardandpoors.com or your brokerage."
    criteria.append(c7)

    # ── SCORE AND RATING ────────────────────────────────────────────────────
    bc.criteria = criteria
    bc.score = sum(1 for c in criteria if c.passed)

    if bc.score == 7:
        bc.rating = "BLUE CHIP — Full Weiss qualification"
        bc.qualifies = True
    elif bc.score >= 5:
        bc.rating = "NEAR BLUE CHIP — strong but not full qualification"
        bc.qualifies = False
    elif bc.score >= 3:
        bc.rating = "SPECULATIVE — marginal Weiss quality"
        bc.qualifies = False
    else:
        bc.rating = "DOES NOT QUALIFY — fails majority of criteria"
        bc.qualifies = False

    return bc


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_weiss_analysis(ticker: str) -> WeissAnalysis:
    analysis = WeissAnalysis(
        ticker=ticker.upper(),
        analysis_date=datetime.now().strftime("%Y-%m-%d")
    )

    try:
        t = yf.Ticker(ticker)
        info = t.info
        analysis.company_name = info.get("longName", ticker)
        analysis.sector = info.get("sector", "Unknown")
    except Exception as e:
        analysis.errors.append(f"Data fetch failed: {e}")
        info = {}

    # ── DATA VALIDATION GATE ──
    from data_validator import validate, cannot_conclude_prompt
    dq = validate(ticker, info, "weiss")
    analysis.errors.append(f"VALIDATION:{dq.confidence}:{dq.can_analyze}:{dq.asset_type}")

    if not dq.can_analyze:
        analysis.errors.append(f"GATE_REASON:{dq.gate_reason}")
        return analysis

    if dq.warnings:
        for w in dq.warnings:
            analysis.errors.append(f"WARNING:{w}")

    analysis.yield_signal = analyze_yield_signal(ticker, info)

    # Blue chip criteria apply to stocks and REITs regardless of dividend status
    # (a non-dividend stock will fail criteria 1 & 7 but the rest still evaluate)
    if dq.asset_type in ("STOCK", "REIT"):
        analysis.blue_chip = score_blue_chip_criteria(ticker, info)
    else:
        analysis.blue_chip.rating = f"Not applicable — {dq.asset_type} asset"
        analysis.blue_chip.sp_rating_note = "Blue chip criteria require individual company financials."

    return analysis


# ─────────────────────────────────────────────
# FORMAT FOR LLM
# ─────────────────────────────────────────────

def format_weiss_for_llm(analysis: WeissAnalysis) -> str:
    ys = analysis.yield_signal
    bc = analysis.blue_chip

    def p(val, d=2):
        return f"{val:.{d}%}" if val is not None else "N/A"

    criteria_block = ""
    for i, c in enumerate(bc.criteria, 1):
        symbol = "✅" if c.passed else "❌"
        criteria_block += f"  {symbol} {i}. {c.name}\n"
        criteria_block += f"        Value: {c.actual_value}  |  Threshold: {c.threshold}\n"
        criteria_block += f"        {c.note}\n\n"

    # Yield chart (ASCII bar showing where current yield sits in 10yr range)
    yield_bar = ""
    if ys.yield_percentile is not None:
        filled = int(ys.yield_percentile * 20)
        bar = "█" * filled + "░" * (20 - filled)
        yield_bar = f"  10yr Yield Range: [{bar}] {ys.yield_percentile:.0%}\n"
        yield_bar += f"  Low: {p(ys.low_yield_10yr)}  ←  Current: {p(ys.current_yield)}  →  High: {p(ys.high_yield_10yr)}\n"

    signal_emoji = {
        "BUY": "✅", "WATCH — BUY ZONE": "👀", "HOLD": "⚪",
        "WATCH — SELL ZONE": "⚠️", "SELL": "🔴",
        "NON-DIVIDEND STOCK": "🚫", "NO DIVIDEND": "🚫", "NO PRICE DATA": "❌"
    }.get(ys.signal, "❓")

    block = f"""
================================================================================
WEISS ANALYSIS FACT SHEET — {analysis.ticker} ({analysis.company_name})
Sector: {analysis.sector} | Date: {analysis.analysis_date}
================================================================================

YOUR ROLE: You are Geraldine Weiss. Interpret the metrics below through your
Investment Quality Trends methodology. Reference specific numbers.
Do NOT invent data. Maximum 150 words. No pleasantries. Open with your verdict.

────────────────────────────────────────────────────────────────────────────────
PRIMARY SIGNAL — YIELD vs 10-YEAR AVERAGE
────────────────────────────────────────────────────────────────────────────────
{yield_bar}
Current Yield:           {p(ys.current_yield)}
10yr Average Yield:      {p(ys.avg_yield_10yr)}
10yr High Yield (buy):   {p(ys.high_yield_10yr)}     ← undervalued zone
10yr Low Yield (sell):   {p(ys.low_yield_10yr)}     ← overvalued zone
Yield vs 10yr Avg:       {p(ys.yield_vs_avg)} above/below average
Yield Percentile (10yr): {f"{ys.yield_percentile:.0%}" if ys.yield_percentile is not None else "N/A"}
Years of data used:      {ys.years_of_data}

{signal_emoji} WEISS SIGNAL: {ys.signal} ({ys.signal_strength})
{ys.reasoning}

────────────────────────────────────────────────────────────────────────────────
SEVEN BLUE CHIP QUALITY CRITERIA  [{bc.score}/7 passing]
────────────────────────────────────────────────────────────────────────────────
{criteria_block}
BLUE CHIP RATING: {bc.rating}
{bc.sp_rating_note}

================================================================================
INSTRUCTION — Geraldine Weiss, answer ALL of these specifically:

1. YIELD SIGNAL CONVICTION: Is the yield signal strong enough to act on, or is the
   10yr data range too narrow to trust? State the exact yield spread between current
   and the buy zone threshold in basis points.

2. BLUE CHIP VERDICT: Score this brutally — which of the 7 criteria does it fail and
   why does each failure matter? Don't just list passes and fails, explain the consequence.

3. INCOME IMPACT ON JOHNATHAN'S PORTFOLIO: His current annual dividend income is approximately
   $X from his holdings. If he bought [X shares] of this stock with $9,000 of his available cash,
   what would his new annual income be? Calculate the actual dollar increase.

4. DIVIDEND SAFETY STRESS TEST: Given the payout ratio shown, how much could earnings fall
   before this dividend gets cut? Give a percentage.

5. ALTERNATIVE BLUE CHIP: If this stock fails your criteria, name one specific stock that
   DOES pass all 7 and has a similar yield profile. Give the ticker and current yield.
================================================================================
"""
    return block


def format_weiss_display(analysis: WeissAnalysis) -> str:
    """Compact display for the chat window."""
    ys = analysis.yield_signal
    bc = analysis.blue_chip

    def p(val, d=2):
        return f"{val:.{d}%}" if val is not None else "N/A"

    signal_bar = {
        "BUY":                  "████████████ BUY",
        "WATCH — BUY ZONE":     "█████████░░░ WATCH BUY",
        "HOLD":                 "██████░░░░░░ HOLD",
        "WATCH — SELL ZONE":    "███░░░░░░░░░ WATCH SELL",
        "SELL":                 "░░░░░░░░░░░░ SELL",
        "NON-DIVIDEND STOCK":   "🚫           N/A — no dividend",
        "NO DIVIDEND":          "🚫           N/A — no dividend",
    }.get(ys.signal, ys.signal)

    # Criteria summary
    crit_line = ""
    for c in bc.criteria:
        crit_line += "✅" if c.passed else "❌"

    # Yield bar
    if ys.yield_percentile is not None:
        filled = int(ys.yield_percentile * 20)
        bar = "█" * filled + "░" * (20 - filled)
        yield_viz = f"  [{bar}] {ys.yield_percentile:.0%} of 10yr range"
    else:
        yield_viz = "  [yield range unavailable]"

    lines = [
        f"",
        f"  ── YIELD SIGNAL ───────────────────────────────────────────",
        f"  Current Yield:  {p(ys.current_yield)}   10yr Avg: {p(ys.avg_yield_10yr)}",
        f"  10yr High:      {p(ys.high_yield_10yr)}   10yr Low: {p(ys.low_yield_10yr)}",
        yield_viz,
        f"  Signal:         {signal_bar}",
        f"",
        f"  ── BLUE CHIP CRITERIA [{bc.score}/7] ──────────────────────────",
    ]
    for c in bc.criteria:
        sym = "✅" if c.passed else "❌"
        lines.append(f"  {sym} {c.name:<42} {c.actual_value}")

    lines += [
        f"",
        f"  Rating:         {bc.rating}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "O"
    print(f"\nRunning Weiss analysis on {ticker}...\n")
    result = run_weiss_analysis(ticker)
    print(format_weiss_display(result))
    print()
    print(format_weiss_for_llm(result))
