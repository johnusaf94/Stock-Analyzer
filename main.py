import tkinter as tk
import customtkinter as ctk
import yfinance as yf
import requests
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

# ─────────────────────────────────────────────
# 1. BUFFETT THRESHOLDS & LOGIC
# ─────────────────────────────────────────────
ROIC_MIN = 0.15
GROSS_MARGIN_MIN = 0.40
DEBT_TO_EQUITY_MAX = 0.50
MARGIN_OF_SAFETY = 0.25
DCF_GROWTH_YEARS = 10
TERMINAL_GROWTH = 0.03
DISCOUNT_RATE = 0.09

BUFFETT_IND_FAIR = 1.00
BUFFETT_IND_OVERVALUED = 1.20
BUFFETT_IND_EXTREME = 1.50


@dataclass
class MoatMetrics:
    roic: Optional[float] = None
    gross_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    net_income: Optional[float] = None
    free_cash_flow: Optional[float] = None
    fcf_to_net_income: Optional[float] = None
    roe: Optional[float] = None
    operating_margin: Optional[float] = None


@dataclass
class ValuationMetrics:
    current_price: Optional[float] = None
    eps_ttm: Optional[float] = None
    earnings_yield: Optional[float] = None
    treasury_10yr: Optional[float] = None
    margin_vs_treasury: Optional[float] = None
    fcf_per_share: Optional[float] = None
    shares_outstanding: Optional[float] = None
    dcf_intrinsic_value: Optional[float] = None
    dcf_upside_pct: Optional[float] = None
    dcf_growth_assumed: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    market_cap: Optional[float] = None


@dataclass
class BuffettIndicator:
    total_market_cap_usd: Optional[float] = None
    gdp_usd: Optional[float] = None
    ratio: Optional[float] = None
    signal: str = "UNKNOWN"


@dataclass
class MoatScore:
    roic_pass: bool = False
    gross_margin_pass: bool = False
    debt_pass: bool = False
    fcf_quality_pass: bool = False
    score: int = 0
    rating: str = "WEAK"
    flags: list = field(default_factory=list)


@dataclass
class BuffettAnalysis:
    ticker: str = ""
    company_name: str = ""
    sector: str = ""
    industry: str = ""
    analysis_date: str = ""
    moat: MoatMetrics = field(default_factory=MoatMetrics)
    valuation: ValuationMetrics = field(default_factory=ValuationMetrics)
    buffett_indicator: BuffettIndicator = field(default_factory=BuffettIndicator)
    moat_score: MoatScore = field(default_factory=MoatScore)
    errors: list = field(default_factory=list)


# ─────────────────────────────────────────────
# 2. CALCULATION ENGINE
# ─────────────────────────────────────────────
def fetch_ticker_data(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info
    cf = t.cashflow
    fcf = None
    try:
        op_cf = cf.loc["Operating Cash Flow"].iloc[0] if "Operating Cash Flow" in cf.index else None
        capex = cf.loc["Capital Expenditure"].iloc[0] if "Capital Expenditure" in cf.index else 0
        if op_cf is not None:
            fcf = float(op_cf) + float(capex)
    except:
        fcf = info.get("freeCashflow")

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
    except:
        pass

    return {"info": info, "fcf": fcf, "roic": roic}


def calculate_dcf(fcf: float, shares: float, growth_rate: float) -> float:
    if not fcf or not shares or shares == 0: return None
    pv_fcfs = 0.0
    current_fcf = fcf
    for year in range(1, DCF_GROWTH_YEARS + 1):
        current_fcf *= (1 + growth_rate)
        pv_fcfs += current_fcf / ((1 + DISCOUNT_RATE) ** year)
    terminal_value = (current_fcf * (1 + TERMINAL_GROWTH)) / (DISCOUNT_RATE - TERMINAL_GROWTH)
    pv_terminal = terminal_value / ((1 + DISCOUNT_RATE) ** DCF_GROWTH_YEARS)
    return (pv_fcfs + pv_terminal) / shares


def score_moat(moat: MoatMetrics) -> MoatScore:
    ms = MoatScore()
    if moat.roic and moat.roic >= ROIC_MIN: ms.roic_pass = True
    if moat.gross_margin and moat.gross_margin >= GROSS_MARGIN_MIN: ms.gross_margin_pass = True
    if moat.debt_to_equity and moat.debt_to_equity <= DEBT_TO_EQUITY_MAX: ms.debt_pass = True
    if moat.fcf_to_net_income and moat.fcf_to_net_income >= 0.70: ms.fcf_quality_pass = True
    ms.score = sum([ms.roic_pass, ms.gross_margin_pass, ms.debt_pass, ms.fcf_quality_pass])
    ms.rating = "STRONG" if ms.score >= 4 else "MODERATE" if ms.score >= 2 else "WEAK"
    return ms


def run_buffett_analysis(ticker: str) -> BuffettAnalysis:
    analysis = BuffettAnalysis(ticker=ticker.upper(), analysis_date=datetime.now().strftime("%Y-%m-%d"))
    try:
        raw = fetch_ticker_data(ticker)
        info = raw["info"]
        analysis.company_name = info.get("longName", ticker)

        m = analysis.moat
        m.roic = raw["roic"]
        m.gross_margin = info.get("grossMargins")
        m.debt_to_equity = (info.get("debtToEquity") / 100.0) if info.get("debtToEquity") else None
        m.free_cash_flow = raw["fcf"]
        m.net_income = info.get("netIncomeToCommon")
        if m.free_cash_flow and m.net_income:
            m.fcf_to_net_income = m.free_cash_flow / m.net_income

        analysis.moat_score = score_moat(m)

        v = analysis.valuation
        v.current_price = info.get("currentPrice")
        v.pe_ratio = info.get("trailingPE")
        v.shares_outstanding = info.get("sharesOutstanding")
        if v.pe_ratio: v.earnings_yield = 1.0 / v.pe_ratio

        growth = min(float(info.get("earningsGrowth", 0.05)), 0.25)
        v.dcf_intrinsic_value = calculate_dcf(m.free_cash_flow, v.shares_outstanding, growth)
        if v.dcf_intrinsic_value and v.current_price:
            v.dcf_upside_pct = (v.dcf_intrinsic_value - v.current_price) / v.current_price

    except Exception as e:
        analysis.errors.append(str(e))
    return analysis


# ─────────────────────────────────────────────
# 3. GUI INTERFACE (Combined Roundtable GUI)
# ─────────────────────────────────────────────
class BuffettApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Investor Roundtable - Buffett Analyzer")
        self.geometry("800x600")
        ctk.set_appearance_mode("dark")

        # Sidebar / Input
        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.label = ctk.CTkLabel(self.sidebar, text="Enter Ticker", font=("Helvetica", 16, "bold"))
        self.label.pack(pady=10)

        self.ticker_input = ctk.CTkEntry(self.sidebar, placeholder_text="e.g. MSFT")
        self.ticker_input.pack(pady=10, padx=10)

        self.analyze_btn = ctk.CTkButton(self.sidebar, text="Run Analysis", command=self.start_analysis)
        self.analyze_btn.pack(pady=20)

        # Main Display
        self.results_area = ctk.CTkTextbox(self, font=("Courier New", 13))
        self.results_area.pack(side="right", expand=True, fill="both", padx=10, pady=10)
        self.results_area.insert("0.0", "Enter a ticker and click 'Run Analysis' to see the Buffett breakdown...")

    def start_analysis(self):
        ticker = self.ticker_input.get().strip()
        if not ticker: return

        self.results_area.delete("0.0", "end")
        self.results_area.insert("0.0", f"Analyzing {ticker}... Please wait.")
        self.update()

        result = run_buffett_analysis(ticker)
        self.display_results(result)

    def display_results(self, a: BuffettAnalysis):
        self.results_area.delete("0.0", "end")

        if a.errors:
            self.results_area.insert("end", f"ERRORS FOUND:\n{a.errors}")
            return

        summary = f"""
================================================================================
BUFFETT ANALYSIS: {a.ticker} ({a.company_name})
Date: {a.analysis_date}
================================================================================

MOAT SCORE: {a.moat_score.score}/4 ({a.moat_score.rating})

- ROIC: {a.moat.roic * 100:.1f}% {'✅' if a.moat_score.roic_pass else '❌'}
- Gross Margin: {a.moat.gross_margin * 100:.1f}% {'✅' if a.moat_score.gross_margin_pass else '❌'}
- Debt/Equity: {a.moat.debt_to_equity:.2f} {'✅' if a.moat_score.debt_pass else '❌'}
- FCF Quality: {a.moat.fcf_to_net_income * 100:.1f}% {'✅' if a.moat_score.fcf_quality_pass else '❌'}

VALUATION BREAKDOWN:
- Current Price: ${a.valuation.current_price:.2f}
- DCF Intrinsic Value: ${a.valuation.dcf_intrinsic_value:.2f}
- Upside Potential: {a.valuation.dcf_upside_pct * 100:.1f}%

VERDICT:
{"Highly attractive margin of safety." if (a.valuation.dcf_upside_pct or 0) > 0.25 else "Fairly valued or overpriced."}
================================================================================
        """
        self.results_area.insert("0.0", summary)


if __name__ == "__main__":
    app = BuffettApp()
    app.mainloop()