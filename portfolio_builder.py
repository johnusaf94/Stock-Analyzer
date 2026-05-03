"""
portfolio_builder.py
====================
Portfolio Builder — deterministic scoring engine.

Michael's job:
- Receives scored stock candidates from the composite engine
- Selects/rejects based on Sharpe ratio, expected return, and diversification
- Maintains a live portfolio targeting 15%+ annual return with max market exposure
- Outputs a ranked portfolio with allocation percentages

All scoring is deterministic Python. LM Studio used only for Michael's commentary.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime, timedelta


# ─────────────────────────────────────────────
# LIVE MARKET CAP FETCHER
# ─────────────────────────────────────────────

# Map our sector names to yfinance/finviz screener sector codes
SECTOR_SCREENER_MAP = {
    "Technology":             "Technology",
    "Healthcare":             "Healthcare",
    "Financials":             "Financial Services",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples":       "Consumer Defensive",
    "Energy":                 "Energy",
    "Industrials":            "Industrials",
    "Real Estate":            "Real Estate",
    "Communication Services": "Communication Services",
    "Utilities":              "Utilities",
    "Materials":              "Basic Materials",
}


def fetch_sector_tickers_by_marketcap(sector: str, n: int = 20,
                                       progress_callback=None) -> List[str]:
    """
    Fetch the top N tickers in a sector ranked by market cap.

    Strategy (tries each in order, uses first that works):
    1. yfinance Screener API (yfinance >= 0.2.40) — live, sorted by market cap
    2. Sector ETF top holdings via yfinance funds_data — live, market-cap weighted
    3. S&P 500 Wikipedia list filtered by sector — live, broad coverage
    4. Hardcoded SECTOR_UNIVERSE fallback — always works, ~20 stocks per sector

    On your machine all four will be attempted. The hardcoded list is the guaranteed
    floor so you always get something even if all APIs are unavailable.
    """

    # ── Method 1: yfinance Screener ──
    try:
        screener_sector = SECTOR_SCREENER_MAP.get(sector, sector)
        eq_screener = yf.Screener()
        body = {
            "offset": 0,
            "size": min(n, 100),
            "sortField": "intradaymarketcap",
            "sortType": "DESC",
            "quoteType": "EQUITY",
            "query": {
                "operator": "and",
                "operands": [
                    {"operator": "eq",  "operands": ["sector", screener_sector]},
                    {"operator": "gt",  "operands": ["intradaymarketcap", 1_000_000_000]},
                    {"operator": "eq",  "operands": ["exchange", "NMS"]},
                ]
            },
            "userId": "",
            "userIdType": "guest"
        }
        eq_screener.set_body(body)
        result = eq_screener.response
        quotes = result.get("quotes", [])
        tickers = [q["symbol"] for q in quotes if q.get("symbol")]
        if len(tickers) >= min(n, 5):
            if progress_callback:
                progress_callback("", f"  📡 Screener: {len(tickers)} {sector} tickers")
            return tickers[:n]
    except Exception:
        pass

    # ── Method 2: Sector ETF top holdings ──
    sector_etf_map = {
        "Technology":             "XLK",
        "Healthcare":             "XLV",
        "Financials":             "XLF",
        "Consumer Discretionary": "XLY",
        "Consumer Staples":       "XLP",
        "Energy":                 "XLE",
        "Industrials":            "XLI",
        "Real Estate":            "XLRE",
        "Communication Services": "XLC",
        "Utilities":              "XLU",
        "Materials":              "XLB",
    }
    etf_ticker = sector_etf_map.get(sector)
    etf_partial = []   # save partial results to merge later if needed
    if etf_ticker:
        try:
            etf = yf.Ticker(etf_ticker)
            fd = etf.funds_data
            if fd is not None and hasattr(fd, "top_holdings"):
                top = fd.top_holdings
                if top is not None and len(top) >= 3:
                    tickers = [str(t) for t in list(top.index)]
                    if tickers:
                        if len(tickers) >= n:
                            # ETF returned enough — use it
                            if progress_callback:
                                progress_callback("", f"  📡 ETF holdings ({etf_ticker}): {len(tickers)} {sector} tickers")
                            return tickers[:n]
                        else:
                            # ETF only has partial results (e.g. 10) — save and fall through
                            etf_partial = tickers
                            if progress_callback:
                                progress_callback("", f"  📡 ETF partial ({etf_ticker}): {len(tickers)} tickers, need {n} — supplementing...")
        except Exception:
            pass

    # ── Method 3: S&P 500 Wikipedia list filtered by GICS sector ──
    gics_map = {
        "Technology":             "Information Technology",
        "Healthcare":             "Health Care",
        "Financials":             "Financials",
        "Consumer Discretionary": "Consumer Discretionary",
        "Consumer Staples":       "Consumer Staples",
        "Energy":                 "Energy",
        "Industrials":            "Industrials",
        "Real Estate":            "Real Estate",
        "Communication Services": "Communication Services",
        "Utilities":              "Utilities",
        "Materials":              "Materials",
    }
    try:
        import pandas as pd
        gics_sector = gics_map.get(sector, sector)
        sp500 = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"}
        )[0]
        col = "GICS Sector" if "GICS Sector" in sp500.columns else sp500.columns[3]
        filtered = sp500[sp500[col] == gics_sector]
        sym_col = "Symbol" if "Symbol" in sp500.columns else sp500.columns[0]
        tickers = filtered[sym_col].str.replace(".", "-", regex=False).tolist()
        if len(tickers) >= 3:
            # Merge with ETF partial results — ETF order is market-cap weighted
            # so ETF tickers first, then Wikipedia tickers not already in the list
            combined = list(etf_partial)
            seen = set(combined)
            for t in tickers:
                if t not in seen:
                    combined.append(t)
                    seen.add(t)
            if progress_callback:
                progress_callback("", f"  📡 S&P500 list: {len(combined)} {sector} tickers (ETF top {len(etf_partial)} + {len(combined)-len(etf_partial)} from S&P500)")
            return combined[:n]
    except Exception:
        pass

    # ── Method 4: Hardcoded fallback — merge with any ETF partial results ──
    fallback = SECTOR_UNIVERSE.get(sector, [])
    combined = list(etf_partial)
    seen = set(combined)
    for t in fallback:
        if t not in seen:
            combined.append(t)
            seen.add(t)
    if n > len(combined):
        if progress_callback:
            progress_callback("", (
                f"  ⚠️  {sector}: Only {len(combined)} tickers available "
                f"(requested {n}). All known sources exhausted."
            ))
    return combined[:n]


# ─────────────────────────────────────────────
# SECTOR UNIVERSE — tickers to scan per sector
# ─────────────────────────────────────────────
SECTOR_UNIVERSE = {
    "Technology": [
        "NVDA","MSFT","AAPL","AMD","AVGO","META","GOOGL","ORCL","CRM","ADBE",
        "QCOM","TXN","MU","AMAT","LRCX","KLAC","MRVL","PANW","SNOW","PLTR"
    ],
    "Healthcare": [
        "LLY","UNH","JNJ","ABBV","MRK","ABT","TMO","DHR","ISRG","REGN",
        "VRTX","AMGN","BSX","EW","DXCM","IDXX","MRNA","BIO","BIIB","HUM"
    ],
    "Financials": [
        "BRK-B","JPM","V","MA","BAC","GS","MS","BLK","SCHW","AXP",
        "SPGI","ICE","CME","COF","USB","TFC","FITB","WFC","C","PNC"
    ],
    "Consumer Discretionary": [
        "AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","TGT","BKNG","CMG",
        "DHI","LEN","PHM","GM","F","RIVN","ABNB","LYFT","UBER","DASH"
    ],
    "Consumer Staples": [
        "PG","KO","PEP","COST","WMT","PM","MO","MDLZ","CL","KMB",
        "GIS","K","CPB","HRL","SJM","CHD","CLX","HSY","MKC","CAG"
    ],
    "Energy": [
        "XOM","CVX","COP","EOG","SLB","PXD","MPC","VLO","PSX","OXY",
        "DVN","FANG","HES","APA","HAL","BKR","NOV","WMB","KMI","OKE"
    ],
    "Industrials": [
        "GE","CAT","HON","UPS","RTX","LMT","BA","DE","MMM","EMR",
        "ETN","PH","ROK","IR","AME","FTV","GD","NOC","HII","TDG"
    ],
    "Real Estate": [
        "AMT","PLD","EQIX","CCI","SPG","O","VICI","PSA","EXR","WELL",
        "ARE","BXP","KIM","REG","FRT","NNN","WPC","STAG","COLD","ELS"
    ],
    "Communication Services": [
        "META","GOOGL","NFLX","DIS","CMCSA","T","VZ","TMUS","CHTR","PARA",
        "WBD","FOXA","OMC","IPG","ZM","SNAP","PINS","SPOT","TTD","MTCH"
    ],
    "Utilities": [
        "NEE","SO","DUK","SRE","AEP","XEL","PCG","EXC","D","ES",
        "AWK","PPL","CMS","NI","ATO","WEC","ETR","LNT","EVRG","OGE"
    ],
    "Materials": [
        "LIN","APD","SHW","ECL","NEM","FCX","NUE","VMC","MLM","ALB",
        "MOS","CF","FMC","IFF","EMN","CE","HUN","RPM","SEE","GRA"
    ],
}

# Sector ETF map for benchmarking
SECTOR_ETFS = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP",
    "Energy": "XLE", "Industrials": "XLI", "Real Estate": "XLRE",
    "Communication Services": "XLC", "Utilities": "XLU", "Materials": "XLB",
}


# ─────────────────────────────────────────────
# PORTFOLIO CONSTRAINTS (Portfolio Builder mandate)
# ─────────────────────────────────────────────
TARGET_ANNUAL_RETURN    = 0.15    # 15% annual return target
MIN_SHARPE              = 0.8     # minimum acceptable Sharpe ratio
MAX_POSITION_PCT        = 0.20    # no single stock > 20% of portfolio
MAX_SECTOR_PCT          = 0.35    # no single sector > 35%
MIN_POSITIONS           = 8       # minimum stocks for diversification
MAX_POSITIONS           = 20      # maximum stocks Michael will hold
MIN_COMPOSITE_SCORE     = 50      # only consider stocks scoring 50+


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class StockCandidate:
    ticker:             str = ""
    company_name:       str = ""
    sector:             str = ""
    composite_score:    float = 0.0
    signal:             str = ""

    # Return/risk metrics (calculated from 3yr history)
    annual_return:      Optional[float] = None
    annual_volatility:  Optional[float] = None
    sharpe_ratio:       Optional[float] = None
    max_drawdown:       Optional[float] = None
    beta:               Optional[float] = None

    # Current price data
    current_price:      Optional[float] = None
    market_cap:         Optional[float] = None
    pe_ratio:           Optional[float] = None
    dividend_yield:     Optional[float] = None

    # Michael's evaluation
    pb_score:      float = 0.0    # 0-100 Michael's own scoring
    pb_verdict:    str = ""       # INCLUDE / WATCH / REJECT
    pb_reason:     str = ""
    fetch_error:        str = ""


@dataclass
class PortfolioPosition:
    ticker:         str = ""
    company_name:   str = ""
    sector:         str = ""
    shares:         float = 0.0
    cost_basis:     float = 0.0       # price when allocated
    current_price:  float = 0.0
    allocation_pct: float = 0.0       # % of total portfolio
    value:          float = 0.0
    annual_return:  Optional[float] = None
    sharpe_ratio:   Optional[float] = None
    composite_score: float = 0.0


@dataclass
class PortfolioResult:
    positions:          List[PortfolioPosition] = field(default_factory=list)
    total_value:        float = 0.0
    cash_remaining:     float = 0.0
    sector_weights:     Dict[str, float] = field(default_factory=dict)
    expected_return:    Optional[float] = None
    expected_volatility: Optional[float] = None
    expected_sharpe:    Optional[float] = None
    meets_target:       bool = False
    pb_summary:    str = ""
    candidates_scanned: int = 0
    candidates_passed:  int = 0
    build_date:         str = ""


# ─────────────────────────────────────────────
# RETURN/RISK CALCULATOR
# ─────────────────────────────────────────────

def fetch_return_metrics(ticker: str) -> dict:
    """
    Calculate CAGR, volatility, Sharpe, and max drawdown.

    Uses 10-year history (or max available) for return calculation to avoid
    cherry-picking recent bull runs. Falls back to shorter periods if needed.
    The CAGR method (start price to end price) is used instead of mean-daily
    compounding which inflates returns in volatile periods.
    """
    result = {
        "annual_return": None, "annual_volatility": None,
        "sharpe_ratio": None, "max_drawdown": None, "beta": None,
        "years_of_data": None,
    }
    try:
        t = yf.Ticker(ticker)

        # Try 10y first, fall back to 5y, then 3y
        hist = None
        for period, min_days in [("10y", 1800), ("5y", 900), ("3y", 500)]:
            h = t.history(period=period, interval="1d")
            if not h.empty and len(h) >= min_days:
                hist = h
                result["years_of_data"] = len(h) / 252
                break

        if hist is None or hist.empty:
            return result

        prices = hist["Close"]
        returns = prices.pct_change().dropna()
        n_years = len(prices) / 252

        # ── CAGR (start-to-end, not mean-daily compounding) ──
        # This is the honest 10-year average — not inflated by mean-daily method
        start_price = float(prices.iloc[0])
        end_price   = float(prices.iloc[-1])
        if start_price > 0 and n_years > 0:
            cagr = (end_price / start_price) ** (1 / n_years) - 1
            result["annual_return"] = cagr

        # ── Annualized volatility (daily std * sqrt(252)) ──
        ann_vol = float(returns.std() * np.sqrt(252))
        result["annual_volatility"] = ann_vol

        # ── Sharpe ratio (4.5% risk-free rate) ──
        if result["annual_return"] is not None and ann_vol > 0:
            result["sharpe_ratio"] = (result["annual_return"] - 0.045) / ann_vol

        # ── Max drawdown ──
        cum = (1 + returns).cumprod()
        roll_max = cum.cummax()
        drawdown = (cum - roll_max) / roll_max
        result["max_drawdown"] = float(drawdown.min())

        # ── Beta vs SPY (same lookback period) ──
        try:
            spy_hist = yf.Ticker("SPY").history(period="10y", interval="1d")
            spy_ret = spy_hist["Close"].pct_change().dropna()
            common = returns.index.intersection(spy_ret.index)
            if len(common) > 200:
                r_align = returns[common]
                s_align = spy_ret[common]
                result["beta"] = float(r_align.cov(s_align) / s_align.var())
        except Exception:
            pass

    except Exception as e:
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────────
# PORTFOLIO BUILDER SCORING LOGIC
# ─────────────────────────────────────────────

def pb_score_candidate(candidate: StockCandidate, target_return: float = None) -> StockCandidate:
    """
    Score every candidate on a continuous 0-100 scale — no binary gate.

    All stocks are scored and ranked. Allocation weight, not a yes/no verdict,
    determines what gets into the portfolio. This ensures every selected sector
    always contributes something, even if its best stock is mediocre.

    Weights:
      Composite score (framework quality):  35 pts
      Sharpe ratio (risk-adjusted return):  35 pts
      10yr CAGR vs target:                 20 pts
      Drawdown resilience:                 10 pts
    """
    hurdle = target_return if target_return is not None else TARGET_ANNUAL_RETURN
    reasons = []
    score = 0.0

    # ── 1. Composite score — business quality (35 pts) ──
    cs = candidate.composite_score
    if cs > 0:
        # Normalize 0-100 composite to 0-35 pts
        score += (cs / 100.0) * 35
        reasons.append(f"Composite {cs:.0f}/100")
    else:
        reasons.append("No composite score — data incomplete")

    # ── 2. Sharpe ratio — risk-adjusted return (35 pts) ──
    sharpe = candidate.sharpe_ratio
    if sharpe is not None:
        # Sharpe 0→0pts, 0.5→12pts, 1.0→24pts, 1.5→35pts, 2.0→35pts (capped)
        sharpe_score = min(sharpe / 1.5, 1.0) * 35
        score += max(0, sharpe_score)
        reasons.append(f"Sharpe {sharpe:.2f}")
    else:
        reasons.append("Sharpe unavailable")

    # ── 3. 10yr CAGR vs target (20 pts) ──
    ret = candidate.annual_return
    if ret is not None:
        if ret >= hurdle * 1.5:
            score += 20
            reasons.append(f"Return {ret:.1%} — well above {hurdle:.0%} target")
        elif ret >= hurdle:
            # Scales from 12 at target to 20 at 1.5x target
            score += 12 + ((ret - hurdle) / (hurdle * 0.5)) * 8
            reasons.append(f"Return {ret:.1%} — meets {hurdle:.0%} target")
        elif ret >= 0:
            # Partial credit — still positive return
            score += max(0, (ret / hurdle) * 12)
            reasons.append(f"Return {ret:.1%} — below {hurdle:.0%} target")
        else:
            reasons.append(f"Return {ret:.1%} — negative (loses points)")
            score -= 5   # active penalty for negative CAGR
    else:
        reasons.append("No return history")

    # ── 4. Drawdown resilience (10 pts) ──
    dd = candidate.max_drawdown
    if dd is not None:
        if dd > -0.15:
            score += 10
            reasons.append(f"Drawdown {dd:.0%} — very resilient")
        elif dd > -0.25:
            score += 8
            reasons.append(f"Drawdown {dd:.0%} — resilient")
        elif dd > -0.35:
            score += 5
            reasons.append(f"Drawdown {dd:.0%} — moderate")
        elif dd > -0.50:
            score += 2
            reasons.append(f"Drawdown {dd:.0%} — high risk")
        else:
            score += 0
            reasons.append(f"Drawdown {dd:.0%} — severe crash risk")
    else:
        score += 3   # neutral if unknown
        reasons.append("Drawdown unavailable — neutral")

    candidate.pb_score = round(max(0.0, min(score, 100.0)), 1)
    candidate.pb_reason = " | ".join(reasons)

    # Soft verdict labels — purely informational, NOT used as a gate
    # Allocation weight decides inclusion, not this label
    if candidate.pb_score >= 65:
        candidate.pb_verdict = "STRONG"
    elif candidate.pb_score >= 45:
        candidate.pb_verdict = "GOOD"
    elif candidate.pb_score >= 25:
        candidate.pb_verdict = "FAIR"
    else:
        candidate.pb_verdict = "WEAK"

    return candidate


# ─────────────────────────────────────────────
# PORTFOLIO ALLOCATION
# ─────────────────────────────────────────────

def build_allocation(
    candidates: List[StockCandidate],
    starting_cash: float,
    max_sector_override: float = None,
    max_position_override: float = None,
    target_return: float = None,
) -> PortfolioResult:
    """
    Allocate capital across ALL scored candidates — no binary gate.

    Logic:
    1. Rank all candidates by pb_score (continuous 0-100)
    2. Pick best N candidates ensuring every scanned sector
       gets at least its best stock represented (sector floor)
    3. Within each sector, prefer lower volatility when scores are close
    4. Raw weight = pb_score / sum(scores) — proportional to quality
    5. Apply per-position cap (user slider)
    6. Apply per-sector cap (user slider) — scale down proportionally
    7. Volatility minimization pass: if two stocks within 5 pts of each
       other in score, prefer the lower-volatility one
    8. Normalize to 97% invested, 3% cash buffer
    """
    max_sec = max_sector_override  if max_sector_override  is not None else MAX_SECTOR_PCT
    max_pos = max_position_override if max_position_override is not None else MAX_POSITION_PCT
    hurdle  = target_return        if target_return         is not None else TARGET_ANNUAL_RETURN

    result = PortfolioResult(build_date=datetime.now().strftime("%Y-%m-%d %H:%M"))
    result.candidates_scanned = len(candidates)

    # Filter to stocks with actual data only (not failed fetches)
    valid = [c for c in candidates if c.pb_score > 0 and not c.fetch_error]
    result.candidates_passed = len(valid)

    if not valid:
        result.pb_summary = "Portfolio Builder: No valid candidates with data. Holding cash."
        result.cash_remaining = starting_cash
        result.total_value = starting_cash
        return result

    # ── Step 1: Sector-aware selection ──
    # Guarantee each scanned sector contributes its best stock first,
    # then fill remaining slots with global top scorers.
    sectors_seen = {}
    for c in sorted(valid, key=lambda x: x.pb_score, reverse=True):
        if c.sector not in sectors_seen:
            sectors_seen[c.sector] = c   # best per sector

    # Start with sector representatives, then top overall
    selected = list(sectors_seen.values())
    selected_tickers = {c.ticker for c in selected}

    remaining_slots = MAX_POSITIONS - len(selected)
    if remaining_slots > 0:
        for c in sorted(valid, key=lambda x: x.pb_score, reverse=True):
            if c.ticker not in selected_tickers:
                selected.append(c)
                selected_tickers.add(c.ticker)
                remaining_slots -= 1
                if remaining_slots == 0:
                    break

    # ── Step 2: Volatility tiebreaker ──
    # Within each sector, if two stocks score within 5pts of each other,
    # prefer the lower-volatility one. This minimizes portfolio vol.
    sector_groups: Dict[str, List] = {}
    for c in selected:
        sector_groups.setdefault(c.sector, []).append(c)

    volatility_adjusted = []
    for sec, group in sector_groups.items():
        if len(group) == 1:
            volatility_adjusted.extend(group)
            continue
        # Sort within sector: primary = score desc, secondary = volatility asc
        # For stocks within 5pts of each other, lower vol wins
        group_sorted = sorted(group, key=lambda x: x.pb_score, reverse=True)
        top_score = group_sorted[0].pb_score
        result_group = []
        for c in group_sorted:
            # If within 5pts of top scorer, apply volatility preference
            if (top_score - c.pb_score) <= 5 and c.annual_volatility is not None:
                c._vol_adjusted_score = c.pb_score - (c.annual_volatility * 10)
            else:
                c._vol_adjusted_score = c.pb_score
        group_sorted.sort(key=lambda x: x._vol_adjusted_score, reverse=True)
        volatility_adjusted.extend(group_sorted)

    selected = volatility_adjusted

    # ── Step 3: Proportional weights by pb_score ──
    total_score = sum(c.pb_score for c in selected)
    if total_score == 0:
        total_score = 1
    raw_allocs = {c.ticker: c.pb_score / total_score for c in selected}

    # ── Step 4: Apply per-position cap ──
    final_allocs: Dict[str, float] = {}
    sector_totals: Dict[str, float] = {}
    for c in selected:
        alloc = min(raw_allocs[c.ticker], max_pos)
        final_allocs[c.ticker] = alloc
        sector_totals[c.sector] = sector_totals.get(c.sector, 0) + alloc

    # ── Step 5: Apply per-sector cap, scale down excess proportionally ──
    for sec, sec_total in sector_totals.items():
        if sec_total > max_sec:
            scale = max_sec / sec_total
            for c in selected:
                if c.sector == sec:
                    final_allocs[c.ticker] *= scale

    # ── Step 6: Normalize to 97% invested ──
    total_alloc = sum(final_allocs.values())
    if total_alloc > 0:
        norm = 0.97 / total_alloc
        final_allocs = {t: v * norm for t, v in final_allocs.items()}

    # ── Step 7: Build positions ──
    positions = []
    invested = 0.0
    all_candidates_map = {c.ticker: c for c in valid}

    for c in selected:
        alloc_pct = final_allocs.get(c.ticker, 0)
        if alloc_pct < 0.001:   # skip dust allocations < 0.1%
            continue
        dollar_val = starting_cash * alloc_pct
        price = c.current_price or 1.0
        shares = dollar_val / price if price > 0 else 0

        pos = PortfolioPosition(
            ticker=c.ticker,
            company_name=c.company_name,
            sector=c.sector,
            shares=round(shares, 4),
            cost_basis=price,
            current_price=price,
            allocation_pct=alloc_pct * 100,
            value=dollar_val,
            annual_return=c.annual_return,
            sharpe_ratio=c.sharpe_ratio,
            composite_score=c.composite_score,
        )
        positions.append(pos)
        invested += dollar_val

    result.positions = sorted(positions, key=lambda x: x.allocation_pct, reverse=True)
    result.cash_remaining = starting_cash - invested
    result.total_value = starting_cash

    # ── Step 8: Portfolio-level metrics ──
    sector_weights: Dict[str, float] = {}
    weighted_return = 0.0
    weighted_vol    = 0.0

    for pos in result.positions:
        w   = pos.allocation_pct / 100
        sec = pos.sector
        sector_weights[sec] = sector_weights.get(sec, 0) + pos.allocation_pct
        if pos.annual_return:
            weighted_return += w * pos.annual_return
        cand = all_candidates_map.get(pos.ticker)
        if cand and cand.annual_volatility:
            weighted_vol += w * cand.annual_volatility

    result.sector_weights     = sector_weights
    result.expected_return    = weighted_return
    result.expected_volatility = weighted_vol if weighted_vol > 0 else None
    result.expected_sharpe    = (
        (weighted_return - 0.045) / weighted_vol
        if weighted_vol > 0 else None
    )
    result.meets_target = weighted_return >= hurdle

    n         = len(result.positions)
    n_sectors = len(sector_weights)
    ret_str    = f"{result.expected_return:.1%}" if result.expected_return else "N/A"
    vol_str    = f"{result.expected_volatility:.1%}" if result.expected_volatility else "N/A"
    sharpe_str = f"{result.expected_sharpe:.2f}" if result.expected_sharpe else "N/A"
    target_str = "✅ MEETS TARGET" if result.meets_target else "⚠️ BELOW TARGET"

    result.pb_summary = (
        f"Portfolio Builder: {n} positions across {n_sectors} sectors. "
        f"Target: {hurdle:.0%}  |  Return: {ret_str} ({target_str}). "
        f"Vol: {vol_str}  |  Sharpe: {sharpe_str}  |  "
        f"Caps: pos {max_pos:.0%} / sector {max_sec:.0%}. "
        f"Cash: ${result.cash_remaining:,.0f}."
    )

    return result


# ─────────────────────────────────────────────
# MAIN DISCOVERY LOOP
# ─────────────────────────────────────────────

def run_portfolio_builder(
    starting_cash: float,
    years: int,
    monthly_contrib: float,
    selected_sectors: List[str],
    selected_agents: List[str],
    tickers_per_sector: int = 6,
    max_sector_pct: float = 0.35,
    max_position_pct: float = 0.20,
    target_annual_return: float = 0.15,
    include_etf: bool = False,
    progress_callback=None,
) -> tuple:
    """
    Full portfolio discovery and construction run.

    max_sector_pct: hard cap per sector (0-1.0)
    target_annual_return: passed to scorer as the target hurdle rate
    include_etf: if True, include sector ETFs ranked by market cap
    Returns: (PortfolioResult, List[StockCandidate] all_candidates)
    """
    from ticker_resolver import fetch_live_data
    from composite_score import build_composite

    # Import analyzers based on selected agents
    analyzer_map = {}
    if "buffett" in selected_agents:
        from buffett_analyzer import run_buffett_analysis
        analyzer_map["buffett"] = run_buffett_analysis
    if "weiss" in selected_agents:
        from weiss_analyzer import run_weiss_analysis
        analyzer_map["weiss"] = run_weiss_analysis
    if "bogle" in selected_agents:
        from bogle_analyzer import run_bogle_analysis
        analyzer_map["bogle"] = run_bogle_analysis
    if "dalio" in selected_agents:
        from dalio_analyzer import run_dalio_analysis
        analyzer_map["dalio"] = run_dalio_analysis
    if "druckenmiller" in selected_agents:
        from druckenmiller_analyzer import run_druckenmiller_analysis
        analyzer_map["druckenmiller"] = run_druckenmiller_analysis

    all_candidates: List[StockCandidate] = []
    seen_tickers: set = set()   # global dedup — prevents same ticker across sectors

    # ETF universe — Bogle's contribution, one ETF per sector for low-cost exposure
    SECTOR_ETFS_UNIVERSE = {
        "Technology":             ["QQQ", "XLK", "VGT", "SMH", "SOXX"],
        "Healthcare":             ["XLV", "VHT", "IBB", "XBI"],
        "Financials":             ["XLF", "VFH", "KBE", "KRE"],
        "Consumer Discretionary": ["XLY", "VCR", "IBUY"],
        "Consumer Staples":       ["XLP", "VDC", "KXI"],
        "Energy":                 ["XLE", "VDE", "OIH"],
        "Industrials":            ["XLI", "VIS", "ITA"],
        "Real Estate":            ["XLRE", "VNQ", "IYR"],
        "Communication Services": ["XLC", "VOX", "FCOM"],
        "Utilities":              ["XLU", "VPU", "IDU"],
        "Materials":              ["XLB", "VAW", "PDBC"],
    }

    for sector in selected_sectors:
        # ── Fetch tickers by market cap (live) with hardcoded fallback ──
        if progress_callback:
            progress_callback("", f"Fetching top {tickers_per_sector} {sector} tickers by market cap...")

        stock_universe = fetch_sector_tickers_by_marketcap(
            sector, n=tickers_per_sector,
            progress_callback=progress_callback
        )

        # ETFs: ONLY include if user explicitly toggled "Include Sector ETFs"
        etf_universe = []
        if include_etf:
            etf_universe = SECTOR_ETFS_UNIVERSE.get(sector, [])[:3]

        universe = stock_universe + etf_universe

        for ticker in universe:
            # ── DEDUP: skip if already scanned in any sector ──
            if ticker in seen_tickers:
                if progress_callback:
                    progress_callback(ticker, f"⏭  {ticker} already scanned — skipping duplicate")
                continue
            seen_tickers.add(ticker)

            if progress_callback:
                progress_callback(ticker, f"Scanning {ticker} ({sector})")

            candidate = StockCandidate(ticker=ticker, sector=sector)

            # Fetch live data
            try:
                live = fetch_live_data(ticker)
                candidate.company_name  = live.company_name or ticker
                candidate.current_price = live.current_price
                candidate.market_cap    = live.market_cap
                candidate.pe_ratio      = live.pe_ratio
                candidate.dividend_yield= live.dividend_yield
                # Use yfinance's actual sector label — overrides loop variable
                # This prevents META showing as "Technology" when scanned in that loop
                if live.sector and live.sector != "Unknown":
                    candidate.sector = live.sector
            except Exception as e:
                candidate.fetch_error = str(e)
                candidate.pb_verdict = "REJECT"
                candidate.pb_reason = f"Data fetch failed: {e}"
                all_candidates.append(candidate)
                continue

            # Run return/risk metrics
            metrics = fetch_return_metrics(ticker)
            candidate.annual_return     = metrics.get("annual_return")
            candidate.annual_volatility = metrics.get("annual_volatility")
            candidate.sharpe_ratio      = metrics.get("sharpe_ratio")
            candidate.max_drawdown      = metrics.get("max_drawdown")
            candidate.beta              = metrics.get("beta")

            # Run composite score with selected agents
            try:
                results_map = {}
                for agent, fn in analyzer_map.items():
                    try:
                        if agent in ("bogle", "dalio"):
                            results_map[agent] = fn(ticker, "portfolio.xlsx")
                        else:
                            results_map[agent] = fn(ticker)
                    except Exception:
                        results_map[agent] = None

                comp = build_composite(
                    ticker=ticker,
                    company_name=candidate.company_name,
                    buffett_analysis=results_map.get("buffett"),
                    weiss_analysis=results_map.get("weiss"),
                    bogle_analysis=results_map.get("bogle"),
                    dalio_analysis=results_map.get("dalio"),
                    druckenmiller_analysis=results_map.get("druckenmiller"),
                    live_data=live,
                    skipped=set(a for a in ["buffett","weiss","bogle","dalio","druckenmiller"]
                                if a not in selected_agents),
                )
                candidate.composite_score = comp.total_score
                candidate.signal          = comp.signal
            except Exception as e:
                candidate.composite_score = 0
                candidate.signal = "ERROR"

            # Portfolio Builder evaluates
            candidate = pb_score_candidate(candidate, target_return=target_annual_return)

            all_candidates.append(candidate)

            if progress_callback:
                verdict_icon = {"INCLUDE": "✅", "WATCH": "👀", "REJECT": "❌"}.get(
                    candidate.pb_verdict, "?"
                )
                progress_callback(
                    ticker,
                    f"{verdict_icon} {ticker}: Michael={candidate.pb_score:.0f} "
                    f"Composite={candidate.composite_score:.0f} "
                    f"Return={candidate.annual_return:.1%} "
                    f"Sharpe={candidate.sharpe_ratio:.2f}"
                    if candidate.annual_return and candidate.sharpe_ratio
                    else f"{verdict_icon} {ticker}: {candidate.pb_verdict}"
                )

    # Portfolio Builder allocates capital with all user constraints
    portfolio = build_allocation(all_candidates, starting_cash,
                                 max_sector_override=max_sector_pct,
                                 max_position_override=max_position_pct,
                                 target_return=target_annual_return)

    # Project future value — with realistic cap and scenario analysis
    portfolio.projected_value_at_retirement = None
    portfolio.projection_scenarios = {}

    if portfolio.expected_return and years > 0:
        raw_return = portfolio.expected_return

        # Hard cap: portfolio CAGR is historical, future rarely replicates exactly.
        # Cap at 20% to prevent quadrillion projections from inflated short-period data.
        # Historical SP500 long-run = ~10%, great stock picker = ~15-20%
        capped_return = min(raw_return, 0.20)
        portfolio.return_was_capped = raw_return > 0.20
        portfolio.capped_return = capped_return

        def fv(annual_rate, start, monthly, n_years):
            if annual_rate <= 0:
                return start + monthly * 12 * n_years
            r = annual_rate / 12
            n = n_years * 12
            return start * ((1 + r) ** n) + monthly * (((1 + r) ** n - 1) / r)

        # Three scenarios
        portfolio.projected_value_at_retirement = fv(capped_return, starting_cash, monthly_contrib, years)
        portfolio.projection_scenarios = {
            f"Conservative (7% / S&P avg)":     fv(0.07,          starting_cash, monthly_contrib, years),
            f"Moderate (10% / hist market)":    fv(0.10,          starting_cash, monthly_contrib, years),
            f"Optimistic ({capped_return:.0%} / portfolio)": fv(capped_return, starting_cash, monthly_contrib, years),
        }

    return portfolio, all_candidates
