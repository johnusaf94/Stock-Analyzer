"""
ticker_resolver.py
==================
Step 1: Resolve any fuzzy user input to a confirmed ticker + company name.
Step 2: Pull live fundamental data that ALL investors share — no hallucination.

Every investor's prompt gets the VERIFIED_DATA block prepended.
The LLM's job is interpretation only — never data generation.

Requires: pip install yfinance requests
"""

import yfinance as yf
import requests
import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# TICKER RESOLUTION
# ─────────────────────────────────────────────

# Common misspellings / company-name-to-ticker map
# Extend this as you encounter bad inputs
KNOWN_ALIASES = {
    "pfizer":       "PFE",
    "pfitzer":      "PFE",
    "microsoft":    "MSFT",
    "apple":        "AAPL",
    "amazon":       "AMZN",
    "google":       "GOOGL",
    "alphabet":     "GOOGL",
    "tesla":        "TSLA",
    "nvidia":       "NVDA",
    "realty income":"O",
    "johnson":      "JNJ",
    "jnj":          "JNJ",
    "verizon":      "VZ",
    "coca cola":    "KO",
    "cocacola":     "KO",
    "coca-cola":    "KO",
    "procter":      "PG",
    "procter gamble":"PG",
    "pg":           "PG",
    "schwab":       "SCHW",
    "schd":         "SCHD",
    "voo":          "VOO",
    "spy":          "SPY",
    "qqq":          "QQQ",
}

def resolve_ticker(user_input: str):
    """
    Convert fuzzy user input to (ticker, company_name, success).

    Strategy:
    1. Clean and check known alias map
    2. Extract anything that looks like a ticker (2-5 uppercase letters)
    3. Try the extracted/cleaned ticker against yfinance to confirm it's real
    4. If confirmation fails, return failure so GUI can prompt user

    Returns: (ticker, company_name, success)
    """
    raw = user_input.strip()

    # Step 1: Check alias map (case-insensitive, try full string and each word)
    candidates = [raw.lower()] + [w.lower() for w in raw.split()]
    for candidate in candidates:
        if candidate in KNOWN_ALIASES:
            ticker = KNOWN_ALIASES[candidate]
            name = _verify_ticker(ticker)
            if name:
                return ticker, name, True

    # Step 2: Extract ticker-looking tokens (2-5 uppercase or easily uppercased letters)
    tokens = raw.upper().split()
    ticker_tokens = [t for t in tokens if re.match(r'^[A-Z]{1,6}$', t)]

    # Try each token as a ticker, longest match first (avoids matching "A" accidentally)
    ticker_tokens.sort(key=len, reverse=True)
    for token in ticker_tokens:
        name = _verify_ticker(token)
        if name:
            return token, name, True

    # Step 3: Nothing worked
    return raw.upper(), "", False


def _verify_ticker(ticker: str) -> str:
    """
    Confirm a ticker is real by fetching its shortName from yfinance.
    Returns company name if valid, empty string if not.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        name = info.get("longName") or info.get("shortName") or ""
        # yfinance returns a mostly-empty dict for invalid tickers
        if name and info.get("regularMarketPrice") is not None:
            return name
        # Some valid tickers don't have regularMarketPrice but do have shortName
        if name and len(info) > 5:
            return name
        return ""
    except Exception:
        return ""


# ─────────────────────────────────────────────
# SHARED LIVE DATA FETCH
# ─────────────────────────────────────────────

@dataclass
class LiveTickerData:
    ticker:             str = ""
    company_name:       str = ""
    sector:             str = ""
    industry:           str = ""
    current_price:      Optional[float] = None
    market_cap:         Optional[float] = None

    # Dividend data (critical for Weiss)
    dividend_yield:     Optional[float] = None   # current TTM yield
    dividend_rate:      Optional[float] = None   # annual dividend $ per share
    payout_ratio:       Optional[float] = None
    five_yr_div_growth: Optional[float] = None   # 5yr dividend CAGR
    ex_dividend_date:   Optional[str] = None
    dividend_history:   list = field(default_factory=list)  # last 5 years of annual divs
    yield_5yr_high:     Optional[float] = None   # calculated from price/div history
    yield_5yr_low:      Optional[float] = None   # calculated from price/div history

    # Valuation (Lynch, Buffett)
    pe_ratio:           Optional[float] = None
    forward_pe:         Optional[float] = None
    peg_ratio:          Optional[float] = None
    eps_ttm:            Optional[float] = None
    earnings_growth:    Optional[float] = None
    revenue_growth:     Optional[float] = None

    # Quality (Buffett)
    gross_margin:       Optional[float] = None
    operating_margin:   Optional[float] = None
    roe:                Optional[float] = None
    debt_to_equity:     Optional[float] = None
    free_cash_flow:     Optional[float] = None
    net_income:         Optional[float] = None

    # Risk
    beta:               Optional[float] = None
    fifty_two_wk_high:  Optional[float] = None
    fifty_two_wk_low:   Optional[float] = None
    pct_from_52wk_high: Optional[float] = None

    fetch_errors:       list = field(default_factory=list)


def fetch_live_data(ticker: str) -> LiveTickerData:
    """
    Pull all shared fundamental data for a confirmed ticker.
    This runs ONCE and is shared across all investors.
    """
    data = LiveTickerData(ticker=ticker.upper())

    try:
        t = yf.Ticker(ticker)
        info = t.info

        data.company_name   = info.get("longName", ticker)
        data.sector         = info.get("sector", "N/A")
        data.industry       = info.get("industry", "N/A")
        data.current_price  = info.get("currentPrice") or info.get("regularMarketPrice")
        data.market_cap     = info.get("marketCap")

        # ── Dividend data ──
        # ── Percentage field normalization ──
        # yfinance is inconsistent — dividendYield sometimes returns 0.0281 (decimal)
        # and sometimes 2.81 (already multiplied by 100). Normalize everything to decimal.
        def safe_pct(val):
            """Ensure a percentage value is in decimal form (0.0281 not 2.81)."""
            if val is None:
                return None
            val = float(val)
            # If value > 1.0 it's almost certainly already in percentage form
            # (a 100%+ yield would be extraordinary and we'd still want decimal)
            if val > 1.0:
                return val / 100.0
            return val

        data.dividend_yield     = safe_pct(info.get("dividendYield"))
        data.dividend_rate      = info.get("dividendRate") or 0.0  # None = no dividend ($0), not missing data
        data.payout_ratio       = safe_pct(info.get("payoutRatio"))
        data.five_yr_div_growth = info.get("fiveYearAvgDividendYield")  # NOTE: this is yield avg not growth
        data.ex_dividend_date   = str(info.get("exDividendDate", "N/A"))

        # ── Yield range from 5yr price + dividend history ──
        try:
            hist = t.history(period="5y", interval="1mo")
            divs = t.dividends
            if not hist.empty and data.dividend_rate:
                # Estimate historical yield = annual dividend / monthly closing price
                yields = []
                for date, row in hist.iterrows():
                    price = row["Close"]
                    if price and price > 0 and data.dividend_rate:
                        yields.append(data.dividend_rate / price)
                if yields:
                    data.yield_5yr_high = max(yields)
                    data.yield_5yr_low  = min(yields)
        except Exception as e:
            data.fetch_errors.append(f"Yield range: {e}")

        # ── Valuation ──
        data.pe_ratio       = info.get("trailingPE")
        data.forward_pe     = info.get("forwardPE")
        data.peg_ratio      = info.get("pegRatio")
        data.eps_ttm        = info.get("trailingEps")
        data.earnings_growth= info.get("earningsGrowth")
        data.revenue_growth = info.get("revenueGrowth")

        # ── Quality ──
        data.gross_margin   = info.get("grossMargins")
        data.operating_margin = info.get("operatingMargins")
        data.roe            = info.get("returnOnEquity")
        data.free_cash_flow = info.get("freeCashflow")
        data.net_income     = info.get("netIncomeToCommon")

        raw_de = info.get("debtToEquity")
        data.debt_to_equity = raw_de / 100.0 if raw_de else None

        # ── Risk ──
        data.beta               = info.get("beta")
        data.fifty_two_wk_high  = info.get("fiftyTwoWeekHigh")
        data.fifty_two_wk_low   = info.get("fiftyTwoWeekLow")
        if data.current_price and data.fifty_two_wk_high:
            data.pct_from_52wk_high = (data.current_price - data.fifty_two_wk_high) / data.fifty_two_wk_high

    except Exception as e:
        data.fetch_errors.append(f"Main fetch: {e}")

    return data


# ─────────────────────────────────────────────
# FORMAT VERIFIED DATA BLOCK FOR LLM
# ─────────────────────────────────────────────

def format_verified_data_block(data: LiveTickerData) -> str:
    """
    Produces the VERIFIED DATA section injected into every investor's prompt.
    The LLM is told explicitly: use only these numbers, do not invent data.
    """
    def safe_pct_display(val):
        """Normalize and format a percentage value for display.
        yfinance inconsistently returns dividendYield as 0.034 OR 3.4 depending on ticker.
        Always normalize to decimal before formatting."""
        if val is None: return "N/A"
        v = float(val)
        if v > 1.0:
            v = v / 100.0   # was already multiplied by 100 — convert back
        return f"{v:.2%}"

    def p(val, fmt=".1%"):
        if val is None: return "N/A"
        try:
            v = float(val)
            # Auto-detect pre-multiplied percentages (anything > 1.0 for a rate field)
            if fmt.endswith('%') and v > 1.0:
                v = v / 100.0
            return f"{v:{fmt}}"
        except: return str(val)

    def d(val):
        if val is None: return "N/A"
        try:
            if abs(val) >= 1e12: return f"${val/1e12:.2f}T"
            if abs(val) >= 1e9:  return f"${val/1e9:.1f}B"
            if abs(val) >= 1e6:  return f"${val/1e6:.1f}M"
            return f"${val:.2f}"
        except: return str(val)

    def n(val, fmt=".2f"):
        if val is None: return "N/A"
        try: return f"{val:{fmt}}"
        except: return str(val)

    # NOTE: No pre-interpreted signals here — each investor gets raw numbers only.
    # Weiss interprets yield in her own analyzer. Lynch interprets PEG in his prompt.
    # Cross-methodology signals in shared data cause investors to reference each other's
    # conclusions before those investors have run.

    block = f"""
================================================================================
⚠️  VERIFIED LIVE DATA — USE ONLY THESE NUMBERS. DO NOT INVENT OR ESTIMATE.
If a value shows N/A, say so — do not substitute your own estimate.
================================================================================

COMPANY:  {data.ticker} — {data.company_name}
SECTOR:   {data.sector} | {data.industry}
PRICE:    ${n(data.current_price)} | Market Cap: {d(data.market_cap)}
52wk:     High ${n(data.fifty_two_wk_high)} / Low ${n(data.fifty_two_wk_low)} | Currently {p(data.pct_from_52wk_high)} from 52wk high
Beta:     {n(data.beta)}

── DIVIDEND & YIELD DATA ────────────────────────────────────────────────────────
Pays Dividend:        {"YES" if data.dividend_rate and data.dividend_rate > 0 else "NO"}
Current Yield:        {p(data.dividend_yield) if data.dividend_rate else "N/A"}
Annual Dividend/sh:   {"${:.4f}".format(data.dividend_rate) if data.dividend_rate else "$0.00"}
Payout Ratio:         {p(data.payout_ratio) if data.payout_ratio else "N/A"}
5yr Yield High:       {p(data.yield_5yr_high) if data.yield_5yr_high else "N/A"}
5yr Yield Low:        {p(data.yield_5yr_low) if data.yield_5yr_low else "N/A"}

── VALUATION ────────────────────────────────────────────────────────────────────
P/E (Trailing):       {n(data.pe_ratio, '.1f')}x
P/E (Forward):        {n(data.forward_pe, '.1f')}x
PEG Ratio:            {n(data.peg_ratio)}
EPS (TTM):            ${n(data.eps_ttm)}
Earnings Growth:      {p(data.earnings_growth)}
Revenue Growth:       {p(data.revenue_growth)}

── QUALITY / MOAT (Buffett) ────────────────────────────────────────────────────
Gross Margin:         {p(data.gross_margin)}     (Buffett threshold: > 40%)
Operating Margin:     {p(data.operating_margin)}
ROE:                  {p(data.roe)}
Debt / Equity:        {n(data.debt_to_equity)}   (Buffett threshold: < 0.5)
Free Cash Flow:       {d(data.free_cash_flow)}
Net Income:           {d(data.net_income)}
================================================================================
"""
    if data.fetch_errors:
        block += f"Data warnings: {'; '.join(data.fetch_errors)}\n"

    return block


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    test_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "pfitzer pfe"
    print(f"Resolving: '{test_input}'")
    ticker, name, ok = resolve_ticker(test_input)
    print(f"Result: {ticker} — {name} (success={ok})")
    if ok:
        print("\nFetching live data...")
        data = fetch_live_data(ticker)
        print(format_verified_data_block(data))
