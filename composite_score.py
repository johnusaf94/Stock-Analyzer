"""
composite_score.py
==================
Pure Python composite scoring engine.
Aggregates all framework scores into a single 0-100 score.
Zero LLM involvement — entirely deterministic and reproducible.
"""

from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# WEIGHTS (must sum to 100)
# ─────────────────────────────────────────────
WEIGHTS = {
    "buffett_moat":           17,
    "buffett_valuation":      13,
    "weiss_yield":             7,
    "weiss_quality":           7,
    "bogle_timing":            7,
    "bogle_diversification":   7,
    "dalio_debt":              7,
    "dalio_bubble":            7,
    "lynch_peg":               6,
    "druckenmiller":          13,   # triple alignment: macro+momentum+technicals
    "gill_squeeze":            5,   # short squeeze setup — only relevant if SI > 15%
    "chamath_squeeze":         4,   # narrative squeeze + catalyst momentum
}
assert sum(WEIGHTS.values()) == 100, "Weights must sum to 100"

# Score thresholds
STRONG_BUY  = 75
BUY         = 60
WATCHLIST   = 45
# below 45 = AVOID

# Account fit thresholds
ROTH_MIN_SCORE    = 55   # higher risk acceptable in Roth
TAXABLE_MIN_SCORE = 60   # need more stability in taxable


# ─────────────────────────────────────────────
# INDIVIDUAL COMPONENT SCORERS
# Each returns a (raw_score 0-1, display, detail) tuple
# ─────────────────────────────────────────────

def score_buffett_moat(moat_score) -> tuple:
    """
    Buffett moat: 0-4 criteria passing.
    0→0%, 1→25%, 2→50%, 3→75%, 4→100%
    """
    if moat_score is None:
        return 0.0, "N/A", "No data"
    score = moat_score.score / 4.0
    rating = moat_score.rating
    flags = " | ".join(moat_score.flags) if moat_score.flags else "No flags"
    return score, f"{moat_score.score}/4 — {rating}", flags


def score_buffett_valuation(valuation) -> tuple:
    """
    Buffett valuation: earnings yield vs treasury + DCF margin of safety.
    Both passing = 100%, one = 50%, neither = 0%
    Also penalizes extreme overvaluation.
    """
    if valuation is None:
        return 0.0, "N/A", "No data"

    points = 0
    details = []

    # Earnings yield > treasury yield
    if valuation.margin_vs_treasury is not None:
        if valuation.margin_vs_treasury > 0.02:   # 2%+ spread
            points += 1
            details.append(f"Earnings yield {valuation.margin_vs_treasury:+.1%} above treasury ✅")
        elif valuation.margin_vs_treasury > 0:
            points += 0.5
            details.append(f"Thin yield spread: {valuation.margin_vs_treasury:+.1%} ⚠️")
        else:
            details.append(f"Stock yields less than treasury: {valuation.margin_vs_treasury:+.1%} ❌")

    # DCF margin of safety
    if valuation.dcf_upside_pct is not None:
        if valuation.dcf_upside_pct > 0.25:
            points += 1
            details.append(f"DCF upside {valuation.dcf_upside_pct:.0%} — strong margin of safety ✅")
        elif valuation.dcf_upside_pct > 0:
            points += 0.5
            details.append(f"DCF upside {valuation.dcf_upside_pct:.0%} — thin margin of safety ⚠️")
        elif valuation.dcf_upside_pct > -0.30:
            points += 0.25
            details.append(f"DCF downside {valuation.dcf_upside_pct:.0%} — moderately overvalued ⚠️")
        else:
            details.append(f"DCF downside {valuation.dcf_upside_pct:.0%} — significantly overvalued ❌")
    else:
        # No DCF data — use P/E as fallback
        if valuation.pe_ratio:
            if valuation.pe_ratio < 15:
                points += 0.75
                details.append(f"P/E {valuation.pe_ratio:.1f}x — cheap ✅")
            elif valuation.pe_ratio < 25:
                points += 0.4
                details.append(f"P/E {valuation.pe_ratio:.1f}x — fair value ⚠️")
            else:
                details.append(f"P/E {valuation.pe_ratio:.1f}x — expensive ❌")

    score = min(points / 2.0, 1.0)
    return score, f"{score:.0%} valuation score", " | ".join(details)


def score_weiss_yield(yield_signal) -> tuple:
    """
    Weiss yield: BUY=100%, WATCH_BUY=75%, HOLD=50%, WATCH_SELL=25%, SELL/NO_DIV=0%
    Non-dividend stocks get 50% (neutral — not penalized)
    """
    if yield_signal is None:
        return 0.5, "N/A", "No yield data"

    signal_map = {
        "BUY":               (1.00, "✅ In buy zone"),
        "WATCH — BUY ZONE":  (0.75, "👀 Approaching buy zone"),
        "HOLD":              (0.50, "⚪ Mid-range — neutral"),
        "WATCH — SELL ZONE": (0.25, "⚠️ Approaching sell zone"),
        "SELL":              (0.00, "🔴 In sell zone — overvalued by yield"),
        "NON-DIVIDEND STOCK":(0.50, "— No dividend (neutral for growth stock)"),
        "NO DIVIDEND":       (0.50, "— No dividend (neutral for growth stock)"),
        "NO PRICE DATA":     (0.50, "No price data"),
        "INSUFFICIENT DATA": (0.50, "Insufficient history"),
        "ERROR":             (0.50, "Data error"),
    }
    score, label = signal_map.get(yield_signal.signal, (0.50, yield_signal.signal))
    detail = yield_signal.reasoning[:100] if yield_signal.reasoning else ""
    return score, label, detail


def score_weiss_quality(blue_chip) -> tuple:
    """
    Weiss 7 blue chip criteria: score/7 directly.
    """
    if blue_chip is None or not blue_chip.criteria:
        return 0.5, "N/A", "No quality data"
    score = blue_chip.score / 7.0
    return score, f"{blue_chip.score}/7 — {blue_chip.rating}", ""


def score_bogle_timing(reversion) -> tuple:
    """
    Bogle reversion timing: 0-10 score, normalized to 0-1.
    """
    if reversion is None or reversion.timing_score is None:
        return 0.5, "N/A", "No timing data"
    score = reversion.timing_score / 10.0
    return score, f"{reversion.timing_score}/10 — {reversion.timing_signal}", reversion.timing_reasoning[:80]


def score_bogle_diversification(diversification) -> tuple:
    """
    Bogle diversification impact:
    IMPROVES=100%, NEUTRAL=60%, HURTS=20%, HURTS_SIGNIFICANTLY=0%
    """
    if diversification is None:
        return 0.5, "N/A", "No portfolio data"
    impact_map = {
        "IMPROVES":             (1.00, "✅ Reduces concentration"),
        "NEUTRAL":              (0.60, "⚪ No meaningful diversification change"),
        "HURTS":                (0.20, "⚠️ Increases concentration"),
        "HURTS_SIGNIFICANTLY":  (0.00, "❌ Significantly increases concentration"),
    }
    score, label = impact_map.get(diversification.diversification_impact, (0.5, "Unknown"))
    return score, label, diversification.impact_reasoning[:80]


def score_dalio_debt(debt_cycle) -> tuple:
    """
    Dalio debt: pass/fail with gradient.
    Below 3x=100%, 3-5x=40%, above 5x=0%
    """
    if debt_cycle is None or debt_cycle.debt_to_ebitda is None:
        return 0.5, "N/A", "No debt data"
    d = debt_cycle.debt_to_ebitda
    if d == 0:
        return 1.0, "✅ No debt", "Zero leverage"
    elif d < 1.5:
        return 1.0, f"✅ {d:.1f}x — very low", "Conservative balance sheet"
    elif d < 3.0:
        return 0.75, f"✅ {d:.1f}x — manageable", "Within Dalio threshold"
    elif d < 5.0:
        return 0.35, f"⚠️ {d:.1f}x — elevated", "Above 3x threshold"
    else:
        return 0.0, f"❌ {d:.1f}x — dangerous", "High debt in late cycle"


def score_dalio_bubble(bubble) -> tuple:
    """
    Dalio bubble: 3 checks, each worth 1/3.
    """
    if bubble is None:
        return 0.5, "N/A", "No bubble data"
    score = bubble.checks_passed / 3.0
    return score, f"{bubble.checks_passed}/3 checks — {bubble.result.note[:50]}", ""


def score_lynch_peg(live_data) -> tuple:
    """
    Lynch PEG ratio:
    <0.5=100% (screaming bargain), 0.5-1.0=85%, 1.0-1.5=65%,
    1.5-2.0=40%, 2.0-2.5=20%, >2.5=0%
    No dividend/no PEG = 50% neutral
    """
    if live_data is None or live_data.peg_ratio is None:
        return 0.5, "N/A (no PEG data)", "Using neutral score"
    peg = live_data.peg_ratio
    if peg <= 0:
        return 0.5, f"PEG {peg:.2f} (negative — unprofitable)", "Negative earnings, PEG unreliable"
    elif peg < 0.5:
        return 1.00, f"PEG {peg:.2f} — screaming bargain", "Lynch loves this"
    elif peg < 1.0:
        return 0.85, f"PEG {peg:.2f} — attractive", "Below Lynch's target of 1.0"
    elif peg < 1.5:
        return 0.65, f"PEG {peg:.2f} — fair value", "Reasonable but not cheap"
    elif peg < 2.0:
        return 0.40, f"PEG {peg:.2f} — stretched", "Paying up for growth"
    elif peg < 2.5:
        return 0.20, f"PEG {peg:.2f} — expensive", "Lynch would pass"
    else:
        return 0.00, f"PEG {peg:.2f} — overpriced", "Market pricing in perfection"


# ─────────────────────────────────────────────
# ACCOUNT FIT
# ─────────────────────────────────────────────

def determine_account_fit(composite: float, live_data, bogle_div) -> tuple:
    """
    Given composite score and stock characteristics, recommend account placement.
    Returns (account, reasoning)
    """
    is_dividend = live_data and live_data.dividend_rate and live_data.dividend_rate > 0
    dividend_yield = live_data.dividend_yield if live_data else None
    beta = live_data.beta if live_data else None

    high_yield = dividend_yield and dividend_yield > 0.04   # >4% yield
    high_beta  = beta and beta > 1.4

    if composite >= STRONG_BUY:
        if high_beta and not high_yield:
            return "Roth 401k", "High growth, higher volatility — tax-free compounding maximizes return"
        elif high_yield:
            return "Either", "Strong score with meaningful yield — works in both accounts"
        else:
            return "Roth 401k", "Strong compounder — let it grow tax-free over 30 years"
    elif composite >= BUY:
        if high_yield and not high_beta:
            return "Taxable Brokerage", "Solid yield, lower volatility — fits income + stability goal"
        else:
            return "Roth 401k", "Growth profile — better in tax-free account"
    elif composite >= WATCHLIST:
        if high_yield:
            return "Watchlist — Taxable", "Watch for better entry; yield supports taxable placement if bought"
        else:
            return "Watchlist — Roth", "Not a buy yet; monitor for improvement in valuation/timing"
    else:
        return "Avoid", "Score too low across multiple frameworks"


# ─────────────────────────────────────────────
# MAIN COMPOSITE SCORER
# ─────────────────────────────────────────────

@dataclass
class ComponentScore:
    name:         str = ""
    weight:       int = 0
    raw:          float = 0.0    # 0.0 to 1.0
    weighted:     float = 0.0    # raw * weight
    display:      str = ""
    detail:       str = ""
    bar_filled:   int = 0        # 0-20 blocks for display


@dataclass
class CompositeResult:
    ticker:           str = ""
    company_name:     str = ""
    date:             str = ""
    components:       list = field(default_factory=list)
    total_score:      float = 0.0    # 0-100
    signal:           str = ""       # STRONG BUY / BUY / WATCHLIST / AVOID
    account_fit:      str = ""
    account_reason:   str = ""
    market_context:   str = ""       # Buffett Indicator reading
    data_quality:     str = ""       # HIGH / MEDIUM / LOW
    missing_data:     list = field(default_factory=list)
    skipped:          list = field(default_factory=list)


def score_gill(gill_analysis) -> tuple:
    """
    Keith Gill squeeze score: 0-1 normalized from 0-100.
    Returns neutral 0.5 if short interest is below threshold
    (squeeze analysis only meaningful for heavily shorted stocks).
    """
    if gill_analysis is None:
        return 0.5, "N/A", "No Gill analysis"
    g = gill_analysis
    m = g.metrics

    # If barely shorted, this metric is neutral — don't penalize normal stocks
    if m.short_interest_pct is None or m.short_interest_pct < 0.10:
        return 0.5, f"SI {m.short_interest_pct:.0%} — below squeeze threshold (neutral)" if m.short_interest_pct else "No short data (neutral)", ""

    score = g.total_score / 100.0
    display = f"{g.verdict} | Score {g.total_score:.0f}/100 | SI {m.short_interest_pct:.0%} | DTC {m.days_to_cover:.1f}d" if m.days_to_cover else f"{g.verdict} | {g.total_score:.0f}/100"
    return score, display, g.thesis[:80]


def score_chamath(chamath_analysis) -> tuple:
    """
    Chamath squeeze score: 0-1 normalized.
    Also neutral for low short-interest stocks.
    """
    if chamath_analysis is None:
        return 0.5, "N/A", "No Chamath analysis"
    c = chamath_analysis
    m = c.metrics

    if m.short_interest_pct is None or m.short_interest_pct < 0.08:
        return 0.5, f"SI {m.short_interest_pct:.0%} — below squeeze threshold (neutral)" if m.short_interest_pct else "No short data (neutral)", ""

    score = c.total_score / 100.0
    display = f"{c.verdict} | Score {c.total_score:.0f}/100 | {c.narrative}"
    return score, display, c.thesis[:80]


def score_druckenmiller(druck_analysis) -> tuple:
    """
    Druckenmiller triple alignment: pig philosophy score normalized to 0-1.
    MAX conviction + SIZE UP signal = 100%
    EXIT signal = 0% regardless of other scores
    """
    if druck_analysis is None:
        return 0.5, "N/A", "No Druckenmiller data"

    # Hard exit overrides everything
    mf = druck_analysis.mental_flexibility
    if mf and mf.exit_signal == "EXIT":
        return 0.0, f"❌ EXIT — Stop triggered ({mf.stop_note[:60]})", "Druckenmiller exits immediately when price breaks key MA"

    pp = druck_analysis.pig_philosophy
    if pp.triple_align_score is None:
        return 0.5, "N/A", "No alignment data"

    score = pp.triple_align_score
    display = f"{score:.0%} alignment — {pp.conviction_level} conviction — {pp.signal}"
    detail = pp.reasoning[:90]
    return score, display, detail


def build_composite(
    ticker: str,
    company_name: str,
    buffett_analysis=None,
    weiss_analysis=None,
    bogle_analysis=None,
    dalio_analysis=None,
    druckenmiller_analysis=None,
    gill_analysis=None,
    chamath_analysis=None,
    live_data=None,
    market_context_str: str = "",
    skipped: set = None,
) -> CompositeResult:
    """
    Build the composite score from all framework analyses.

    Key behaviour:
    - skipped: set of analyzer names that were intentionally excluded for this
               asset class (e.g. {"buffett", "weiss"} for ETFs).
               Skipped components are REMOVED and their weights redistributed
               proportionally across remaining components so the total = 100.
    - None analysis (data error): component included at 0% and flagged as missing.
    """
    from datetime import datetime
    if skipped is None:
        skipped = set()

    result = CompositeResult(
        ticker=ticker.upper(),
        company_name=company_name,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        market_context=market_context_str,
    )

    # ── Also auto-skip Weiss yield if stock pays no dividend ──
    if live_data and (not live_data.dividend_rate or live_data.dividend_rate == 0):
        skipped = skipped | {"weiss_yield"}

    components_raw = []   # (key, name, base_weight, raw, display, detail)
    missing = []

    # Helper: don't add component if its framework key is in skipped
    def maybe_add(key, name, base_weight, raw_fn, *args):
        # Check if this component's framework is skipped
        framework = key.split("_")[0]   # "buffett_moat" -> "buffett"
        if key in skipped or framework in skipped:
            return   # exclude entirely — weight will be redistributed
        raw, display, detail = raw_fn(*args)
        components_raw.append((key, name, base_weight, raw, display, detail))

    def flag_missing(key, label, analysis_obj):
        if analysis_obj is None:
            missing.append(label)

    # ── Build candidate components ──
    flag_missing("buffett", "Buffett Moat",        buffett_analysis)
    flag_missing("buffett", "Buffett Valuation",   buffett_analysis)
    flag_missing("weiss",   "Weiss Yield",         weiss_analysis)
    flag_missing("weiss",   "Weiss Quality",       weiss_analysis)
    flag_missing("bogle",   "Bogle Timing",        bogle_analysis)
    flag_missing("bogle",   "Bogle Diversif.",     bogle_analysis)
    flag_missing("dalio",   "Dalio Debt",          dalio_analysis)
    flag_missing("dalio",   "Dalio Bubble",        dalio_analysis)
    flag_missing("druck",   "Druckenmiller",       druckenmiller_analysis)
    flag_missing("gill",    "Gill Squeeze",        gill_analysis)
    flag_missing("chamath", "Chamath Squeeze",     chamath_analysis)

    moat_obj = buffett_analysis.moat_score if buffett_analysis else None
    val_obj  = buffett_analysis.valuation  if buffett_analysis else None
    ys_obj   = weiss_analysis.yield_signal  if weiss_analysis  else None
    bc_obj   = weiss_analysis.blue_chip     if weiss_analysis  else None
    rev_obj  = bogle_analysis.reversion     if bogle_analysis  else None
    div_obj  = bogle_analysis.diversification if bogle_analysis else None
    debt_obj = dalio_analysis.debt_cycle    if dalio_analysis  else None
    bub_obj  = dalio_analysis.bubble        if dalio_analysis  else None

    maybe_add("buffett_moat",          "Buffett — Moat Quality",      WEIGHTS["buffett_moat"],          score_buffett_moat,      moat_obj)
    maybe_add("buffett_valuation",     "Buffett — Valuation/DCF",     WEIGHTS["buffett_valuation"],     score_buffett_valuation, val_obj)
    maybe_add("weiss_yield",           "Weiss — Yield Signal",        WEIGHTS["weiss_yield"],           score_weiss_yield,       ys_obj)
    maybe_add("weiss_quality",         "Weiss — Blue Chip Quality",   WEIGHTS["weiss_quality"],         score_weiss_quality,     bc_obj)
    maybe_add("bogle_timing",          "Bogle — Buy Timing",          WEIGHTS["bogle_timing"],          score_bogle_timing,      rev_obj)
    maybe_add("bogle_diversification", "Bogle — Diversification",     WEIGHTS["bogle_diversification"], score_bogle_diversification, div_obj)
    maybe_add("dalio_debt",            "Dalio — Debt Cycle",          WEIGHTS["dalio_debt"],            score_dalio_debt,        debt_obj)
    maybe_add("dalio_bubble",          "Dalio — Bubble Risk",         WEIGHTS["dalio_bubble"],          score_dalio_bubble,      bub_obj)
    maybe_add("lynch_peg",             "Lynch — PEG Ratio",           WEIGHTS["lynch_peg"],             score_lynch_peg,         live_data)
    maybe_add("druckenmiller",         "Druckenmiller — Triple Align",WEIGHTS["druckenmiller"],         score_druckenmiller,     druckenmiller_analysis)

    # ── Redistribute weights so active components always sum to 100 ──
    total_base_weight = sum(row[2] for row in components_raw)
    components = []
    for key, name, base_w, raw, display, detail in components_raw:
        if total_base_weight > 0:
            adjusted_w = round(base_w / total_base_weight * 100, 1)
        else:
            adjusted_w = 0
        components.append(ComponentScore(
            name=name,
            weight=adjusted_w,
            raw=raw,
            weighted=raw * adjusted_w,
            display=display,
            detail=detail,
        ))

    # ── Fill bar blocks (0-20) ──
    for c in components:
        c.bar_filled = round(c.raw * 20)

    result.components = components
    result.total_score = sum(c.weighted for c in components)
    result.missing_data = missing
    result.skipped = list(skipped)

    # ── Signal ──
    if result.total_score >= STRONG_BUY:
        result.signal = "STRONG BUY"
    elif result.total_score >= BUY:
        result.signal = "BUY"
    elif result.total_score >= WATCHLIST:
        result.signal = "WATCHLIST"
    else:
        result.signal = "AVOID"

    # ── Data quality ──
    if len(missing) == 0:
        result.data_quality = "HIGH"
    elif len(missing) <= 3:
        result.data_quality = "MEDIUM"
    else:
        result.data_quality = "LOW"

    # ── Account fit ──
    result.account_fit, result.account_reason = determine_account_fit(
        result.total_score, live_data, div_obj
    )

    return result


# ─────────────────────────────────────────────
# DISPLAY FORMATTERS
# ─────────────────────────────────────────────

SIGNAL_COLORS = {
    "STRONG BUY": "\033[92m",   # bright green
    "BUY":        "\033[32m",   # green
    "WATCHLIST":  "\033[93m",   # yellow
    "AVOID":      "\033[91m",   # red
}
RESET = "\033[0m"


def format_composite_terminal(result: CompositeResult) -> str:
    """Plain text table for terminal display (used in tkinter chat window)."""
    lines = []
    lines.append("")
    lines.append(f"  ╔══════════════════════════════════════════════════════════╗")
    lines.append(f"  ║  COMPOSITE SCORE — {result.ticker:<10} {result.company_name[:28]:<28}  ║")
    lines.append(f"  ╠══════════════════════════════════════════════════════════╣")
    lines.append(f"  ║  {'Framework':<28} {'Score':>6}  {'Contribution':>5}  Bar               ║")
    lines.append(f"  ╠══════════════════════════════════════════════════════════╣")

    for c in result.components:
        bar = "█" * c.bar_filled + "░" * (20 - c.bar_filled)
        pct = f"{c.raw:.0%}"
        contrib = f"+{c.weighted:.1f}"
        lines.append(f"  ║  {c.name:<28} {pct:>6}  {contrib:>5}  {bar}  ║")

    lines.append(f"  ╠══════════════════════════════════════════════════════════╣")

    # Score bar
    score_bar_filled = round(result.total_score / 5)
    score_bar = "█" * score_bar_filled + "░" * (20 - score_bar_filled)
    lines.append(f"  ║  {'TOTAL SCORE':<28} {result.total_score:>5.1f}  {'':>5}  {score_bar}  ║")
    lines.append(f"  ╠══════════════════════════════════════════════════════════╣")
    lines.append(f"  ║  Signal:      {result.signal:<44}  ║")
    lines.append(f"  ║  Account fit: {result.account_fit:<44}  ║")
    lines.append(f"  ║  Data quality:{result.data_quality:<44}  ║")
    if result.market_context:
        mc = result.market_context[:44]
        lines.append(f"  ║  Market:      {mc:<44}  ║")
    lines.append(f"  ╚══════════════════════════════════════════════════════════╝")

    # Component details
    lines.append("")
    lines.append("  BREAKDOWN:")
    for c in result.components:
        if c.display and c.display != "N/A":
            lines.append(f"  • {c.name}: {c.display}")
            if c.detail:
                lines.append(f"    {c.detail[:90]}")

    if result.missing_data:
        lines.append(f"\n  ⚠️  Missing data: {', '.join(result.missing_data)}")

    lines.append(f"\n  Account reasoning: {result.account_reason}")
    lines.append("")
    return "\n".join(lines)


def format_composite_for_claude(result: CompositeResult) -> str:
    """Structured fact sheet for Claude API Q&A — contains all scores as context."""
    lines = [
        f"COMPOSITE ANALYSIS — {result.ticker} ({result.company_name})",
        f"Date: {result.date}",
        f"Overall Score: {result.total_score:.1f}/100 — {result.signal}",
        f"Account Fit: {result.account_fit}",
        f"Data Quality: {result.data_quality}",
        f"Market Context: {result.market_context}",
        "",
        "COMPONENT SCORES:",
    ]
    for c in result.components:
        lines.append(f"  {c.name} (weight {c.weight}%): {c.raw:.0%} → +{c.weighted:.1f}pts | {c.display}")
        if c.detail:
            lines.append(f"    Detail: {c.detail}")
    if result.missing_data:
        lines.append(f"\nMissing data (used neutral 50%): {', '.join(result.missing_data)}")
    lines.append(f"\nAccount recommendation: {result.account_fit} — {result.account_reason}")
    return "\n".join(lines)
