"""
data_validator.py
==================
Shared validation layer for all investor analyzers.

Responsibilities:
1. Detect asset type (stock, ETF, mutual fund, REIT, index) 
2. Define which metrics are REQUIRED vs OPTIONAL per analyzer
3. Validate that required fields are present and non-garbage
4. Return a clear DataQuality report so analyzers can gate on it
5. Generate a "cannot conclude" LLM prompt when data is insufficient

The rule: if required data is missing, the LLM is told explicitly
it cannot reach a conclusion. No hallucination allowed.
"""

from dataclasses import dataclass, field
from typing import Optional
import yfinance as yf


# ─────────────────────────────────────────────
# ASSET TYPE CLASSIFICATION
# ─────────────────────────────────────────────

ASSET_TYPE_STOCK        = "STOCK"
ASSET_TYPE_ETF          = "ETF"
ASSET_TYPE_MUTUAL_FUND  = "MUTUAL_FUND"
ASSET_TYPE_REIT         = "REIT"
ASSET_TYPE_INDEX        = "INDEX"
ASSET_TYPE_UNKNOWN      = "UNKNOWN"

# What each analyzer can and cannot do per asset type
ANALYZER_COMPATIBILITY = {
    "buffett": {
        ASSET_TYPE_STOCK:       "FULL",       # full moat + DCF analysis
        ASSET_TYPE_REIT:        "PARTIAL",    # moat limited, DCF modified
        ASSET_TYPE_ETF:         "NOT_APPLICABLE",
        ASSET_TYPE_MUTUAL_FUND: "NOT_APPLICABLE",
        ASSET_TYPE_INDEX:       "NOT_APPLICABLE",
        ASSET_TYPE_UNKNOWN:     "LIMITED",
    },
    "weiss": {
        ASSET_TYPE_STOCK:       "FULL",
        ASSET_TYPE_REIT:        "FULL",       # REITs are Weiss's bread and butter
        ASSET_TYPE_ETF:         "PARTIAL",    # yield signal only, no blue chip criteria
        ASSET_TYPE_MUTUAL_FUND: "PARTIAL",
        ASSET_TYPE_INDEX:       "NOT_APPLICABLE",
        ASSET_TYPE_UNKNOWN:     "LIMITED",
    },
    "bogle": {
        ASSET_TYPE_STOCK:       "FULL",
        ASSET_TYPE_REIT:        "FULL",
        ASSET_TYPE_ETF:         "FULL",       # Bogle LOVES ETFs — full analysis
        ASSET_TYPE_MUTUAL_FUND: "FULL",
        ASSET_TYPE_INDEX:       "FULL",
        ASSET_TYPE_UNKNOWN:     "LIMITED",
    },
    "lynch": {
        ASSET_TYPE_STOCK:       "FULL",
        ASSET_TYPE_REIT:        "PARTIAL",
        ASSET_TYPE_ETF:         "NOT_APPLICABLE",
        ASSET_TYPE_MUTUAL_FUND: "NOT_APPLICABLE",
        ASSET_TYPE_INDEX:       "NOT_APPLICABLE",
        ASSET_TYPE_UNKNOWN:     "LIMITED",
    },
}

# Required fields per analyzer (must be non-None and non-zero)
REQUIRED_FIELDS = {
    "buffett": [
        ("grossMargins",        "Gross Margin",         "Moat analysis requires gross margin data"),
        ("debtToEquity",        "Debt/Equity",          "Moat analysis requires debt data"),
        ("freeCashflow",        "Free Cash Flow",       "DCF requires free cash flow — cannot value without it"),
        ("trailingEps",         "Earnings Per Share",   "Valuation requires EPS data"),
        ("currentPrice",        "Current Price",        "Cannot calculate earnings yield without price"),
    ],
    "weiss": [
        # Note: dividendRate is NOT required — a missing/zero dividend just means
        # the yield signal portion returns N/A. The 7 blue chip criteria still run.
        ("currentPrice",        "Current Price",        "Cannot calculate metrics without price"),
    ],
    "bogle": [
        ("currentPrice",        "Current Price",        "Cannot run analysis without price data"),
    ],
    "lynch": [
        ("trailingPE",          "P/E Ratio",            "PEG ratio requires P/E — Lynch cannot evaluate without earnings"),
        ("earningsGrowth",      "Earnings Growth",      "PEG ratio requires growth rate"),
        ("currentPrice",        "Current Price",        "Cannot calculate metrics without price"),
    ],
}

# Optional fields — present = better analysis, absent = note it but don't block
OPTIONAL_FIELDS = {
    "buffett": ["returnOnEquity", "operatingMargins", "netIncomeToCommon", "marketCap"],
    "weiss":   ["payoutRatio", "debtToEquity", "priceToBook", "trailingPE"],
    "bogle":   ["beta", "dividendYield", "fiftyTwoWeekHigh"],
    "lynch":   ["pegRatio", "revenueGrowth", "marketCap"],
}


# ─────────────────────────────────────────────
# DATA QUALITY REPORT
# ─────────────────────────────────────────────

@dataclass
class FieldStatus:
    field_name:     str = ""
    display_name:   str = ""
    present:        bool = False
    value:          str = ""
    required:       bool = False
    reason:         str = ""    # why it's required


@dataclass
class DataQuality:
    ticker:             str = ""
    asset_type:         str = ASSET_TYPE_UNKNOWN
    asset_type_note:    str = ""

    # Per-analyzer compatibility
    analyzer:           str = ""
    compatibility:      str = "UNKNOWN"   # FULL / PARTIAL / NOT_APPLICABLE / LIMITED

    # Field validation
    required_fields:    list = field(default_factory=list)   # list of FieldStatus
    optional_fields:    list = field(default_factory=list)
    missing_required:   list = field(default_factory=list)   # names of missing required fields
    missing_optional:   list = field(default_factory=list)

    # Final gate
    can_analyze:        bool = False
    confidence:         str = "NONE"     # HIGH / MEDIUM / LOW / NONE
    gate_reason:        str = ""         # human-readable reason if blocked
    warnings:           list = field(default_factory=list)


# ─────────────────────────────────────────────
# ASSET TYPE DETECTION
# ─────────────────────────────────────────────

def detect_asset_type(info: dict):
    """
    Returns (asset_type, note) based on yfinance info dict.
    yfinance provides 'quoteType' field: EQUITY, ETF, MUTUALFUND, INDEX, etc.
    """
    quote_type = (info.get("quoteType") or "").upper()
    fund_family = info.get("fundFamily")
    sector = info.get("sector") or ""
    industry = (info.get("industry") or "").upper()
    long_name = (info.get("longName") or "").upper()

    # Direct quoteType mapping
    if quote_type == "ETF":
        return ASSET_TYPE_ETF, f"Exchange-traded fund ({info.get('longName', '')})"

    if quote_type == "MUTUALFUND":
        return ASSET_TYPE_MUTUAL_FUND, f"Mutual fund ({info.get('longName', '')})"

    if quote_type == "INDEX":
        return ASSET_TYPE_INDEX, f"Market index — cannot be purchased directly"

    if quote_type == "EQUITY":
        # Check if it's a REIT
        if (sector == "Real Estate" or
                "REIT" in long_name or
                "REAL ESTATE INVESTMENT" in long_name or
                "REIT" in industry):
            return ASSET_TYPE_REIT, f"Real Estate Investment Trust — special tax/payout rules apply"
        return ASSET_TYPE_STOCK, f"Common equity — {info.get('sector', 'Unknown sector')}"

    # Fallback heuristics
    if fund_family:
        return ASSET_TYPE_ETF, f"Fund product ({fund_family})"
    if sector == "Real Estate":
        return ASSET_TYPE_REIT, "Real Estate sector — likely a REIT"
    if quote_type:
        return ASSET_TYPE_UNKNOWN, f"Unrecognized quote type: {quote_type}"

    return ASSET_TYPE_UNKNOWN, "Asset type could not be determined"


# ─────────────────────────────────────────────
# FIELD VALIDATOR
# ─────────────────────────────────────────────

def _field_value_ok(info: dict, field_key: str):
    """
    Check if a field is present and has a meaningful value.
    Returns (ok, display_value).
    """
    val = info.get(field_key)

    if val is None:
        return False, "N/A"

    # Reject clearly garbage values
    if isinstance(val, float):
        import math
        if math.isnan(val) or math.isinf(val):
            return False, "N/A (invalid)"

    # Reject zero for fields that can't meaningfully be zero
    non_zero_fields = {"freeCashflow", "currentPrice", "trailingEps",
                       "marketCap", "sharesOutstanding"}
    if field_key in non_zero_fields and val == 0:
        return False, "0 (invalid)"

    # Format display value cleanly
    dollar_fields = {"freeCashflow", "marketCap", "netIncomeToCommon", "totalDebt",
                     "netIncome", "sharesOutstanding", "enterpriseValue"}
    pct_fields    = {"grossMargins", "operatingMargins", "returnOnEquity",
                     "returnOnAssets", "dividendYield", "payoutRatio",
                     "earningsGrowth", "revenueGrowth"}

    if isinstance(val, (int, float)):
        if field_key in dollar_fields:
            av = abs(val)
            if av >= 1e12: display = f"${val/1e12:.2f}T"
            elif av >= 1e9: display = f"${val/1e9:.1f}B"
            elif av >= 1e6: display = f"${val/1e6:.1f}M"
            else: display = f"${val:,.0f}"
        elif field_key in pct_fields:
            display = f"{val:.1%}"
        elif field_key == "trailingPE" or field_key == "forwardPE":
            display = f"{val:.1f}x"
        elif field_key == "debtToEquity":
            display = f"{val/100:.2f}"   # yfinance returns as 47.3 not 0.473
        elif abs(val) < 10:
            display = f"{val:.2f}"
        else:
            display = f"{val:,.2f}"
    else:
        display = str(val)

    return True, display


# ─────────────────────────────────────────────
# MAIN VALIDATION FUNCTION
# ─────────────────────────────────────────────

def validate(ticker: str, info: dict, analyzer: str) -> DataQuality:
    """
    Full validation pipeline for a given ticker and analyzer.
    Call this at the start of every analyzer before doing any calculation.
    """
    dq = DataQuality(ticker=ticker.upper(), analyzer=analyzer)

    # 1. Detect asset type
    dq.asset_type, dq.asset_type_note = detect_asset_type(info)

    # 2. Check analyzer compatibility
    compat_map = ANALYZER_COMPATIBILITY.get(analyzer, {})
    dq.compatibility = compat_map.get(dq.asset_type, "LIMITED")

    # 3. Hard block on NOT_APPLICABLE
    if dq.compatibility == "NOT_APPLICABLE":
        dq.can_analyze = False
        dq.confidence = "NONE"
        dq.gate_reason = (
            f"{analyzer.title()} methodology does not apply to {dq.asset_type} assets. "
            f"{_not_applicable_note(analyzer, dq.asset_type)}"
        )
        return dq

    # 4. Validate required fields
    req_fields_def = REQUIRED_FIELDS.get(analyzer, [])
    for field_key, display_name, reason in req_fields_def:
        ok, display_val = _field_value_ok(info, field_key)
        fs = FieldStatus(
            field_name=field_key,
            display_name=display_name,
            present=ok,
            value=display_val,
            required=True,
            reason=reason,
        )
        dq.required_fields.append(fs)
        if not ok:
            dq.missing_required.append(display_name)

    # 5. Validate optional fields
    opt_fields_def = OPTIONAL_FIELDS.get(analyzer, [])
    for field_key in opt_fields_def:
        ok, display_val = _field_value_ok(info, field_key)
        fs = FieldStatus(
            field_name=field_key,
            display_name=field_key,
            present=ok,
            value=display_val,
            required=False,
        )
        dq.optional_fields.append(fs)
        if not ok:
            dq.missing_optional.append(field_key)

    # 6. PARTIAL compatibility — relax some required fields
    #    e.g. Weiss on an ETF only needs yield, not all 7 criteria fields
    if dq.compatibility == "PARTIAL":
        _relax_requirements(dq, analyzer)

    # 7. Determine final gate
    if not dq.missing_required:
        dq.can_analyze = True
        missing_opt = len(dq.missing_optional)
        if missing_opt == 0:
            dq.confidence = "HIGH"
        elif missing_opt <= 2:
            dq.confidence = "MEDIUM"
            dq.warnings.append(f"Optional fields missing: {', '.join(dq.missing_optional)}")
        else:
            dq.confidence = "LOW"
            dq.warnings.append(f"Several optional fields missing — analysis is less precise: {', '.join(dq.missing_optional)}")
    else:
        dq.can_analyze = False
        dq.confidence = "NONE"
        dq.gate_reason = (
            f"Cannot complete {analyzer.title()} analysis — required data is missing from yfinance:\n"
            + "\n".join(f"  • {name}: {reason}"
                        for name, reason in
                        [(f.display_name, f.reason) for f in dq.required_fields if not f.present])
        )

    return dq


def _relax_requirements(dq: DataQuality, analyzer: str):
    """
    For PARTIAL compatibility, remove fields that don't apply to this asset type.
    E.g. Weiss on an ETF: blue chip criteria don't apply, only the yield signal.
    """
    if analyzer == "weiss" and dq.asset_type == ASSET_TYPE_ETF:
        # Only require dividend data for ETFs — skip all the equity-only criteria
        dq.missing_required = [
            m for m in dq.missing_required
            if m in ("Annual Dividend", "Current Price")
        ]
        dq.warnings.append(
            "ETF detected — Blue Chip Criteria (P/E, P/B, debt, earnings) do not apply. "
            "Weiss analysis limited to yield signal only."
        )

    if analyzer == "buffett" and dq.asset_type == ASSET_TYPE_REIT:
        # REITs use FFO not FCF — relax FCF requirement
        dq.missing_required = [
            m for m in dq.missing_required
            if m != "Free Cash Flow"
        ]
        dq.warnings.append(
            "REIT detected — FCF analysis modified. REITs use FFO (Funds from Operations). "
            "DCF valuation less reliable for REITs."
        )


def _not_applicable_note(analyzer: str, asset_type: str) -> str:
    notes = {
        ("buffett", ASSET_TYPE_ETF): (
            "Buffett's moat and DCF analysis requires individual company financials — "
            "an ETF holds hundreds of companies and has no single moat, FCF, or earnings to evaluate. "
            "For ETF analysis, John Bogle is the appropriate evaluator."
        ),
        ("buffett", ASSET_TYPE_MUTUAL_FUND): (
            "Mutual funds are pools of securities — Buffett's framework requires a single company's financials. "
            "Bogle's cost and index analysis is appropriate here."
        ),
        ("lynch", ASSET_TYPE_ETF): (
            "Lynch's PEG ratio and business classification require a single operating company. "
            "ETFs have no P/E or earnings growth in the traditional sense."
        ),
    }
    return notes.get((analyzer, asset_type), "")


# ─────────────────────────────────────────────
# CANNOT CONCLUDE PROMPT
# ─────────────────────────────────────────────

def cannot_conclude_prompt(dq: DataQuality, investor_name: str) -> str:
    """
    Returns the LLM prompt to use when data is insufficient.
    Forces the investor to say they cannot conclude rather than hallucinate.
    """
    if dq.compatibility == "NOT_APPLICABLE":
        return f"""
IMPORTANT INSTRUCTION: {investor_name}, you have been asked to evaluate {dq.ticker} ({dq.asset_type_note}).

Your methodology ({dq.analyzer}) does NOT apply to this asset type ({dq.asset_type}).
{dq.gate_reason}

You must:
1. State clearly that your methodology cannot evaluate this type of asset
2. Briefly explain why (one sentence)
3. Recommend which of your fellow advisors IS appropriate for this asset
4. Do NOT attempt to evaluate it anyway
Maximum 60 words.
"""
    else:
        missing_list = "\n".join(f"  • {name}" for name in dq.missing_required)
        return f"""
IMPORTANT INSTRUCTION: {investor_name}, you have been asked to evaluate {dq.ticker}.

The data pipeline could not retrieve sufficient data to run your analysis.
Missing required data:
{missing_list}

{dq.gate_reason}

You must:
1. State clearly that you cannot reach a conclusion due to missing data
2. List specifically what data is missing
3. Do NOT invent or estimate the missing figures
4. Do NOT give a BUY/HOLD/AVOID verdict without the data
Maximum 80 words.
"""


# ─────────────────────────────────────────────
# DISPLAY HELPER
# ─────────────────────────────────────────────

def format_validation_header(dq: DataQuality) -> str:
    """
    Single-line status shown before each analyzer output.
    Keeps the chat clean — detailed metrics are in the analyzer's own table.
    """
    if dq.compatibility == "NOT_APPLICABLE":
        return (
            f"\n  ⛔ {dq.asset_type} — {dq.analyzer.title()} analysis not applicable\n"
        )

    if not dq.can_analyze:
        missing = ", ".join(dq.missing_required)
        return (
            f"\n  ⚠️  INSUFFICIENT DATA — cannot analyze ({missing})\n"
        )

    # Single status line — confidence + asset type + any warnings
    status = f"  ✅ {dq.asset_type} | Data confidence: {dq.confidence}"
    lines = ["", status]
    for w in dq.warnings:
        lines.append(f"  ⚠️  {w}")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SCHD"
    analyzer = sys.argv[2] if len(sys.argv) > 2 else "buffett"

    print(f"\nValidating {ticker} for {analyzer}...\n")
    t = yf.Ticker(ticker)
    info = t.info
    dq = validate(ticker, info, analyzer)

    print(f"Asset Type:    {dq.asset_type} — {dq.asset_type_note}")
    print(f"Compatibility: {dq.compatibility}")
    print(f"Can Analyze:   {dq.can_analyze}")
    print(f"Confidence:    {dq.confidence}")
    if dq.gate_reason:
        print(f"Gate Reason:   {dq.gate_reason}")
    print()
    print(format_validation_header(dq))
    if not dq.can_analyze:
        print(cannot_conclude_prompt(dq, analyzer.title()))
