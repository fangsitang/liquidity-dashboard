"""
Macro Liquidity / Derivative Dashboard — Streamlit version.

Run locally:    streamlit run streamlit_app.py
Deploy free:    push this file + requirements.txt to a GitHub repo, then
                deploy on https://share.streamlit.io (Streamlit Community Cloud).

FRED_API_KEY is read from Streamlit secrets (st.secrets) or the
FRED_API_KEY environment variable — never hardcode it in this file.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from fredapi import Fred

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Macro Liquidity Dashboard", layout="wide")

# Colors match .streamlit/config.toml's [theme] block — keep these two in sync
# if you ever change the theme colors there.
DASHBOARD_BG = "#0e1117"
DASHBOARD_TEXT = "#fafafa"

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Source Sans Pro', 'Helvetica Neue', 'Arial', 'sans-serif'],
    'figure.facecolor': DASHBOARD_BG,
    'axes.facecolor': DASHBOARD_BG,
    'savefig.facecolor': DASHBOARD_BG,
    'text.color': DASHBOARD_TEXT,
    'axes.labelcolor': DASHBOARD_TEXT,
    'axes.titlecolor': DASHBOARD_TEXT,
    'xtick.color': DASHBOARD_TEXT,
    'ytick.color': DASHBOARD_TEXT,
    'axes.edgecolor': '#444654',
    'grid.color': '#333333',
    'lines.linewidth': 0.9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
})

SERIES = {
    "M2SL": "M2 Money Supply",
    "WALCL": "Fed Total Assets (Balance Sheet)",
    "WTREGEN": "Treasury General Account",
    "RRPONTSYD": "Reverse Repo (ON RRP)",
    "TOTBKCR": "Total Bank Credit",
    "BUSLOANS": "C&I Loans",
    "BAMLH0A0HYM2": "High Yield OAS (Credit Spread)",
    "BAMLC0A0CM": "Investment Grade OAS",
    "CP": "Corporate Profits (GDP-based)",
}

METRIC_INFO = {
    "M2SL": "Captures broad money in the economy (cash + deposits). Rising growth --> more liquidity available to chase assets; decelerating growth --> less fuel for risk-taking.",
    "WALCL": "Captures the size of the Fed's balance sheet (QE/QT). Expansion --> direct liquidity injection into markets; contraction --> liquidity drain and tighter financial conditions.",
    "WTREGEN": "Captures Treasury's cash balance held at the Fed. A rising TGA --> cash pulled out of the banking system (liquidity drain); a falling TGA --> cash released back in (liquidity boost).",
    "RRPONTSYD": "Captures cash parked overnight at the Fed by money funds/banks. Rising RRP --> liquidity being pulled out of the system; falling RRP --> liquidity being freed up for other uses.",
    "NET_LIQUIDITY": "Captures usable liquidity actually available to markets (Fed BS - TGA - RRP). Rising --> broadly supportive for risk assets; falling/decelerating --> a headwind that often precedes equity weakness.",
    "TOTBKCR": "Captures aggregate bank lending. Accelerating growth --> credit expansion, typically late-cycle bullish; decelerating/contracting --> tightening credit conditions, a recession precursor.",
    "BUSLOANS": "Captures C&I loan growth, a cleaner read on business borrowing/investment appetite. Rising --> businesses expanding/investing; falling --> retrenchment and reduced capex.",
    "BAMLH0A0HYM2": "Captures the spread junk bonds pay over Treasuries. Widening --> credit stress/risk-off building; tightening --> risk appetite improving. Moves early relative to equities.",
    "BAMLC0A0CM": "Captures the same credit-stress signal for investment-grade issuers. Widening --> stress spreading to higher-quality credit (more serious); tightening --> broad-based risk-on.",
    "CP": "Captures aggregate corporate profit growth (the lagging fundamental). Accelerating --> confirms an already-priced-in expansion; decelerating --> confirms a slowdown the liquidity/credit metrics likely signaled months earlier.",
}

# ---------------------------------------------------------------------------
# Data functions (cached — this is what makes the "live refresh" work)
# ---------------------------------------------------------------------------

def get_fred_client():
    key = st.secrets.get("FRED_API_KEY", os.environ.get("FRED_API_KEY", ""))
    if not key:
        st.error("No FRED_API_KEY found. Add it to .streamlit/secrets.toml or as an env var.")
        st.stop()
    return Fred(api_key=key)


@st.cache_data(ttl=3600)  # re-fetch from FRED at most once per hour
def fetch_all(series_dict, start="2010-01-01"):
    fred = get_fred_client()
    frames = {}
    for code in series_dict:
        try:
            frames[code] = fred.get_series(code, observation_start=start)
        except Exception as e:
            st.warning(f"Failed to fetch {code}: {e}")
    df = pd.DataFrame(frames)
    df = df.asfreq('D').ffill()
    walcl, tga, rrp = "WALCL", "WTREGEN", "RRPONTSYD"
    if all(c in df.columns for c in (walcl, tga, rrp)):
        df["NET_LIQUIDITY"] = df[walcl] - df[tga] - df[rrp]
    return df


def derivatives(series, window_1st='ME', window_2nd=3, smooth=3):
    resampled = series.resample(window_1st).last()
    first_raw = resampled.pct_change() * 100
    first = first_raw.rolling(smooth, min_periods=1).mean()
    second_raw = first.diff(window_2nd)
    second = second_raw.rolling(smooth, min_periods=1).mean()
    return first, second


def inflection_points(second_deriv):
    sign = second_deriv.dropna().apply(lambda x: 1 if x > 0 else -1)
    flips = sign[sign.diff() != 0]
    return flips.index[1:]


def render_chart(series, label, smooth):
    first, second = derivatives(series, smooth=smooth)
    flips = inflection_points(second)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig.patch.set_facecolor(DASHBOARD_BG)

    axes[0].set_facecolor(DASHBOARD_BG)
    axes[0].plot(series.index, series.values, color=DASHBOARD_TEXT, linewidth=0.9)
    axes[0].set_title(f"{label} — Level")

    axes[1].set_facecolor(DASHBOARD_BG)
    axes[1].plot(first.index, first.values, color='#5DA9E9', linewidth=0.9)
    axes[1].axhline(0, color='#888888', linewidth=0.6)
    axes[1].set_title(f"1st Derivative (% change, {smooth}-period smoothed)")

    axes[2].set_facecolor(DASHBOARD_BG)
    axes[2].plot(second.index, second.values, color='#F4A64B', linewidth=0.9)
    axes[2].axhline(0, color='#888888', linewidth=0.6)
    for d in flips:
        axes[2].axvline(d, color='#FF5C5C', linestyle='--', linewidth=0.7, alpha=0.6)
    axes[2].set_title(f"2nd Derivative (acceleration/deceleration, {smooth}-period smoothed) — dashed = inflection")

    for ax in axes:
        ax.tick_params(axis='x', labelbottom=True, labelsize=8, colors=DASHBOARD_TEXT)
        ax.tick_params(axis='y', colors=DASHBOARD_TEXT)
        for tick_label in ax.get_xticklabels():
            tick_label.set_rotation(45)
    plt.tight_layout()
    return fig, flips


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("Macro Liquidity / Derivative Dashboard")
st.caption("Fed liquidity, credit, and earnings series with 1st/2nd derivatives — refreshed from FRED.")

with st.sidebar:
    st.header("Settings")
    smooth = st.slider("Smoothing window (months)", 1, 6, 3)
    auto_refresh_min = st.number_input("Auto-refresh every N minutes (0 = off)", 0, 240, 0)
    if st.button("Refresh data now"):
        fetch_all.clear()  # bust the cache
    all_labels = {**SERIES, "NET_LIQUIDITY": "Net Liquidity (Fed BS - TGA - RRP)"}
    chosen = st.multiselect("Series to show", list(all_labels.keys()),
                             default=list(all_labels.keys()),
                             format_func=lambda c: all_labels[c])

if HAS_AUTOREFRESH and auto_refresh_min > 0:
    st_autorefresh(interval=auto_refresh_min * 60 * 1000, key="datarefresh")
elif auto_refresh_min > 0:
    st.sidebar.info("Install `streamlit-autorefresh` to enable timed auto-refresh.")

raw = fetch_all(SERIES)
st.caption(f"Data as of: {raw.index.max().date()} (cached up to 1 hour)")

for code in chosen:
    label = all_labels[code]
    if code not in raw.columns:
        continue
    st.subheader(f"{label} ({code})")
    if code in METRIC_INFO:
        st.write(METRIC_INFO[code])
    fig, flips = render_chart(raw[code].dropna(), label, smooth)
    st.pyplot(fig)
    plt.close(fig)
    if len(flips):
        st.caption(f"Most recent inflection points: {', '.join(str(d.date()) for d in flips[-3:])}")
