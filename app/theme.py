"""Plate-system styling for the Streamlit workbench (same palette and type roles as the website).

Palette: paper #efedea, ink #0a0a0a, cobalt #1e2edc, sky #7fb2e8, signal #e0492a, surround #e5e5e5.
Type: Instrument Serif (titles), Geist Mono (labels, captions, numbers), Inter (body).
"""
import streamlit as st

CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root { --paper:#efedea; --ink:#0a0a0a; --cobalt:#1e2edc; --sky:#7fb2e8; --signal:#e0492a; --surround:#e5e5e5;
  --serif:"Instrument Serif", Georgia, serif; --mono:"Geist Mono", ui-monospace, Menlo, monospace;
  --sans:"Inter", -apple-system, BlinkMacSystemFont, sans-serif; }
html, body, [data-testid="stAppViewContainer"] { background: var(--surround); font-family: var(--sans); color: var(--ink);
  font-variant-numeric: tabular-nums; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMain"] .block-container { background: var(--paper); margin-top: 0.9rem; padding: 2rem clamp(1rem, 3vw, 3rem);
  box-shadow: 0 1px 2px rgb(0 0 0 / .06), 0 12px 34px rgb(0 0 0 / .09); max-width: 1180px; }
h1, h2, h3 { font-family: var(--serif) !important; font-weight: 400 !important; letter-spacing: -0.015em; }
h1 { font-size: clamp(1.9rem, 4vw, 2.7rem) !important; line-height: 1.04 !important; }
[data-testid="stCaptionContainer"], [data-testid="stMetricLabel"], [data-testid="stTab"] p,
[data-testid="stWidgetLabel"] p { font-family: var(--mono) !important; letter-spacing: 0.06em; }
[data-testid="stMetricLabel"] p, [data-testid="stTab"] p, [data-testid="stWidgetLabel"] p {
  font-size: 0.7rem !important; text-transform: uppercase; }
[data-testid="stMetricValue"] { font-family: var(--serif); font-weight: 400; }
[data-testid="stTab"][aria-selected="true"] p { color: var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--cobalt); }
code { font-family: var(--mono); color: var(--cobalt); background: transparent; }
[data-testid="stSidebar"] { background: var(--ink); }
[data-testid="stSidebar"] * { color: var(--paper); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-family: var(--serif) !important; }
.stButton > button[kind="primary"], .stDownloadButton > button { background: var(--ink); color: var(--paper); border: 0; border-radius: 0;
  font-family: var(--mono); font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; }
.stButton > button { border-radius: 0; font-family: var(--mono); font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; }
.genqai-band { background: var(--ink); color: var(--paper); padding: .9rem 1.4rem; display: flex; justify-content: space-between;
  align-items: center; gap: 1.5rem; flex-wrap: wrap; box-shadow: 0 1px 2px rgb(0 0 0 / .06), 0 12px 34px rgb(0 0 0 / .09); }
.genqai-band .wordmark { font-family: var(--serif); font-size: 1.2rem; letter-spacing: -0.02em; }
.genqai-band a, .genqai-band span.meta { color: var(--paper); text-decoration: none; font-family: var(--mono); font-size: .68rem;
  letter-spacing: .08em; text-transform: uppercase; opacity: .6; margin-left: 1.4rem; }
.genqai-band a:hover { opacity: 1; }
.genqai-topline { display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap; font-family: var(--mono);
  font-size: .6875rem; letter-spacing: .08em; text-transform: uppercase; opacity: .55; margin: 1.2rem 0 .2rem; }
</style>
"""

BAND = """
<div class="genqai-band">
  <span class="wordmark">GenQAI</span>
  <nav>
    <a href="https://udsy19.github.io/genqai/" target="_blank" rel="noopener">Paper</a>
    <a href="https://qollab.xyz/u/udsy/genqai-one-shot-quantum-chemis" target="_blank" rel="noopener">Run on Qollab</a>
    <a href="https://github.com/udsy19/genqai" target="_blank" rel="noopener">Open source</a>
  </nav>
</div>
<div class="genqai-topline"><span>Workbench · local</span><span>Every number from a live computation or a committed result file</span></div>
"""


def apply() -> None:
    st.markdown(CSS + BAND, unsafe_allow_html=True)
