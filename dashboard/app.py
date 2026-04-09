"""
Border Surveillance AI — Tactical Operations Dashboard
======================================================
Author : Krish Patel
Institute: Ahmedabad Institute of Technology, Ahmedabad
Enroll : 220020107048
Program: MS Elevate Internship 2026
"""

import json, os, glob, base64, io, math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SENTINEL-AI · Tactical Command",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths ──────────────────────────────────────────────────────────────────
_DIR      = Path(__file__).resolve().parent
_ROOT     = _DIR.parent
BASE_DIR  = Path(os.getenv("DATA_ROOT", str(_ROOT)))
ALERT_LOG    = BASE_DIR / "data" / "alerts"   / "alert_log.json"
RESULTS_DIR  = BASE_DIR / "data" / "results"
ANOMALY_JSON = BASE_DIR / "data" / "detections" / "anomaly_summary.json"

# ── Colour palette — amber/charcoal/slate ──────────────────────────────────
C = dict(
    ink       = "#0c0e12",
    carbon    = "#111418",
    slate     = "#181d25",
    panel     = "#1d2330",
    rim       = "#252d3e",
    wire      = "#2e3a50",
    amber     = "#f0a500",
    amber_dim = "#a06800",
    amber_glow= "#ffe085",
    hot       = "#ff5533",
    warn      = "#ff9f1c",
    safe      = "#00d68f",
    ice       = "#7ecfff",
    mist      = "#9baec8",
    ghost     = "#4a5568",
    text      = "#e8ecf4",
    dim       = "#6b7a96",
)

PRI_CLR = {"CRITICAL": C["hot"], "HIGH": C["warn"], "MEDIUM": C["amber"], "LOW": C["safe"]}

# ── New logo — SVG hexagonal radar mark (base64-embedded for safe rendering) ─
_LOGO_SVG_RAW = (
    '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" width="78" height="78">'
    '<defs>'
    '<radialGradient id="lbg" cx="50%" cy="50%" r="50%">'
    '<stop offset="0%" stop-color="#1d2330"/>'
    '<stop offset="100%" stop-color="#0c0e12"/>'
    '</radialGradient>'
    '<filter id="lglow" x="-20%" y="-20%" width="140%" height="140%">'
    '<feGaussianBlur stdDeviation="1.8" result="blur"/>'
    '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
    '</filter>'
    '</defs>'
    '<polygon points="60,4 108,31 108,89 60,116 12,89 12,31" '
    'fill="url(#lbg)" stroke="#f0a500" stroke-width="2.8" filter="url(#lglow)"/>'
    '<polygon points="60,14 98,36 98,84 60,106 22,84 22,36" '
    'fill="none" stroke="#f0a50055" stroke-width="1"/>'
    '<circle cx="60" cy="60" r="30" fill="none" stroke="#f0a50030" stroke-width="1"/>'
    '<circle cx="60" cy="60" r="19" fill="none" stroke="#f0a50050" stroke-width="1"/>'
    '<circle cx="60" cy="60" r="9" fill="none" stroke="#f0a500" stroke-width="1.5"/>'
    '<line x1="22" y1="60" x2="98" y2="60" stroke="#f0a50055" stroke-width="0.9"/>'
    '<line x1="60" y1="22" x2="60" y2="98" stroke="#f0a50055" stroke-width="0.9"/>'
    '<line x1="60" y1="60" x2="60" y2="30" stroke="#f0a500" stroke-width="2" '
    'stroke-linecap="round" filter="url(#lglow)">'
    '<animateTransform attributeName="transform" type="rotate" '
    'from="0 60 60" to="360 60 60" dur="4s" repeatCount="indefinite"/>'
    '</line>'
    '<circle cx="60" cy="60" r="4" fill="#f0a500" filter="url(#lglow)"/>'
    '<circle cx="79" cy="41" r="2.8" fill="#ff5533" opacity="0.9">'
    '<animate attributeName="opacity" values="0.9;0.15;0.9" dur="1.8s" repeatCount="indefinite"/>'
    '</circle>'
    '<line x1="12" y1="46" x2="19" y2="46" stroke="#f0a500" stroke-width="1.8"/>'
    '<line x1="101" y1="46" x2="108" y2="46" stroke="#f0a500" stroke-width="1.8"/>'
    '<line x1="12" y1="74" x2="19" y2="74" stroke="#f0a500" stroke-width="1.8"/>'
    '<line x1="101" y1="74" x2="108" y2="74" stroke="#f0a500" stroke-width="1.8"/>'
    '</svg>'
)

import base64 as _b64
LOGO_B64 = _b64.b64encode(_LOGO_SVG_RAW.encode()).decode()
LOGO_IMG_TAG = f'<img src="data:image/svg+xml;base64,{LOGO_B64}" width="78" height="78" style="display:block"/>'

# ── CSS ────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700;900&family=Barlow:wght@300;400;500&display=swap');

:root {{
  --ink:      {C['ink']};
  --carbon:   {C['carbon']};
  --slate:    {C['slate']};
  --panel:    {C['panel']};
  --rim:      {C['rim']};
  --wire:     {C['wire']};
  --amber:    {C['amber']};
  --hot:      {C['hot']};
  --warn:     {C['warn']};
  --safe:     {C['safe']};
  --ice:      {C['ice']};
  --text:     {C['text']};
  --dim:      {C['dim']};
  --mist:     {C['mist']};
}}

/* ── base ── */
html, body, [class*="css"] {{
  font-family: 'Barlow', sans-serif;
  background: {C['ink']} !important;
  color: {C['text']} !important;
}}
.main {{ background: {C['ink']} !important; padding: 0 !important; }}
.block-container {{
  padding: 0 1.4rem 2rem 1.4rem !important;
  max-width: 1700px !important;
}}

/* ── hide streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stSidebar"] {{ display: none !important; }}

/* ── topbar ── */
.topbar {{
  display: flex;
  align-items: center;
  gap: 1.4rem;
  background: {C['carbon']};
  border-bottom: 2px solid {C['amber']};
  padding: 0.65rem 1.8rem;
  position: sticky; top: 0; z-index: 99;
  box-shadow: 0 4px 30px rgba(0,0,0,0.6);
}}
.tb-brand {{
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.55rem;
  font-weight: 900;
  letter-spacing: 4px;
  text-transform: uppercase;
  color: {C['amber']};
  line-height: 1;
}}
.tb-sub {{
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem;
  color: {C['dim']};
  letter-spacing: 2px;
  margin-top: 2px;
}}
.tb-meta {{
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.72rem;
  color: {C['mist']};
  text-align: right;
  line-height: 1.6;
}}
.tb-clock {{
  font-family: 'Share Tech Mono', monospace;
  font-size: 1.45rem;
  color: {C['amber']};
  letter-spacing: 2px;
  text-align: right;
}}
.pulse-dot {{
  display: inline-block; width: 8px; height: 8px;
  background: {C['safe']}; border-radius: 50%;
  margin-right: 5px;
  box-shadow: 0 0 6px {C['safe']};
  animation: blink 1.8s ease-in-out infinite;
}}
@keyframes blink {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}

/* ── section label ── */
.sec-label {{
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 4px;
  text-transform: uppercase;
  color: {C['amber']};
  padding: 0.55rem 0 0.3rem;
  border-top: 1px solid {C['wire']};
  margin: 0.6rem 0 0.5rem;
  display: flex; align-items: center; gap: 0.5rem;
}}
.sec-label::after {{
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, {C['wire']}, transparent);
}}

/* ── stat tile (replaces KPI card) ── */
.stat-tile {{
  background: {C['slate']};
  border: 1px solid {C['wire']};
  border-left: 3px solid var(--ac, {C['amber']});
  border-radius: 3px;
  padding: 0.85rem 1rem;
  position: relative;
}}
.stat-tile::after {{
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 3px;
  background: linear-gradient(135deg, rgba(240,165,0,0.04) 0%, transparent 60%);
  pointer-events: none;
}}
.st-num {{
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 2.5rem;
  font-weight: 900;
  color: var(--ac, {C['amber']});
  line-height: 1;
}}
.st-lbl {{
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 2px;
  color: {C['dim']};
  text-transform: uppercase;
  margin-bottom: 0.15rem;
}}
.st-sub {{
  font-size: 0.68rem;
  color: {C['ghost']};
  margin-top: 0.2rem;
}}
.st-ico {{
  position: absolute;
  right: 0.8rem; top: 50%;
  transform: translateY(-50%);
  font-size: 1.8rem;
  opacity: 0.12;
}}

/* ── panel card ── */
.pcard {{
  background: {C['slate']};
  border: 1px solid {C['wire']};
  border-radius: 4px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.8rem;
  box-shadow: 0 6px 24px rgba(0,0,0,0.4);
}}
.pcard-hdr {{
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: {C['amber']};
  padding-bottom: 0.55rem;
  margin-bottom: 0.7rem;
  border-bottom: 1px solid {C['wire']};
  display: flex; align-items: center; gap: 0.5rem;
}}

/* ── alert strip ── */
.astrip {{
  display: grid;
  grid-template-columns: 72px 60px 1fr 58px 72px;
  gap: 0.5rem;
  align-items: center;
  padding: 0.42rem 0.7rem;
  margin-bottom: 0.28rem;
  border-radius: 3px;
  border-left: 3px solid transparent;
  font-size: 0.78rem;
  background: rgba(255,255,255,0.025);
  transition: background 0.15s;
}}
.astrip:hover {{ background: rgba(255,255,255,0.05); }}
.as-CRITICAL {{ border-color:{C['hot']};  background:rgba(255,85,51,0.07);  }}
.as-HIGH     {{ border-color:{C['warn']}; background:rgba(255,159,28,0.06); }}
.as-MEDIUM   {{ border-color:{C['amber']};background:rgba(240,165,0,0.05);  }}
.as-LOW      {{ border-color:{C['safe']}; background:rgba(0,214,143,0.04);  }}
.badge {{
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem; letter-spacing: 1px;
  padding: 2px 6px; border-radius: 2px;
  font-weight: bold; display: inline-block;
}}
.b-CRITICAL {{ background:rgba(255,85,51,0.2);  color:{C['hot']};  border:1px solid {C['hot']};  }}
.b-HIGH     {{ background:rgba(255,159,28,0.2); color:{C['warn']}; border:1px solid {C['warn']}; }}
.b-MEDIUM   {{ background:rgba(240,165,0,0.2);  color:{C['amber']};border:1px solid {C['amber']}; }}
.b-LOW      {{ background:rgba(0,214,143,0.15); color:{C['safe']}; border:1px solid {C['safe']}; }}
.mono {{ font-family:'Share Tech Mono',monospace; color:{C['ice']}; font-size:0.72rem; }}
.reason-txt {{ color:{C['mist']}; font-size:0.76rem; overflow:hidden;
               text-overflow:ellipsis; white-space:nowrap; }}
.ts {{ font-family:'Share Tech Mono',monospace; font-size:0.63rem;
       color:{C['ghost']}; text-align:right; }}

/* ── threat gauge ── */
.tg-wrap {{
  background: {C['panel']};
  border: 1px solid {C['wire']};
  border-radius: 3px;
  padding: 0.7rem 0.9rem;
}}
.tg-label {{
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem; letter-spacing: 2px;
  color: {C['dim']}; text-transform: uppercase;
  margin-bottom: 0.3rem;
}}
.tg-bar-bg {{
  background: {C['rim']}; border-radius: 2px;
  height: 8px; overflow: hidden; margin-bottom: 0.35rem;
}}
.tg-bar-fill {{
  height: 100%; border-radius: 2px;
  transition: width 0.6s ease;
}}
.tg-pct {{
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.4rem; font-weight: 900;
  color: var(--ac, {C['amber']});
  line-height: 1;
}}

/* ── system status item ── */
.sys-row {{
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.32rem 0;
  border-bottom: 1px solid {C['rim']}44;
  font-size: 0.76rem;
}}
.sys-key {{ color: {C['mist']}; }}
.sys-val {{ font-family: 'Share Tech Mono', monospace; font-size: 0.68rem; }}

/* ── metric table row ── */
.mrow {{
  display: flex; justify-content: space-between;
  padding: 0.25rem 0;
  border-bottom: 1px solid {C['rim']}33;
  font-size: 0.77rem;
}}
.mrow-k {{ color: {C['mist']}; }}
.mrow-v {{ font-family: 'Share Tech Mono', monospace; color: {C['ice']}; }}

/* ── detail log header ── */
.log-hdr {{
  display: grid;
  grid-template-columns: 64px 58px 82px 68px 48px 52px 1fr;
  gap: 0.4rem;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem; letter-spacing: 1.5px;
  color: {C['amber']}; text-transform: uppercase;
  padding: 0.25rem 0.7rem;
  border-bottom: 1px solid {C['wire']};
  margin-bottom: 0.15rem;
}}
.log-row {{
  display: grid;
  grid-template-columns: 64px 58px 82px 68px 48px 52px 1fr;
  gap: 0.4rem; align-items: center;
  padding: 0.28rem 0.7rem;
  border-radius: 2px;
  border-left: 2px solid transparent;
  font-size: 0.74rem;
  background: rgba(255,255,255,0.02);
  margin-bottom: 0.15rem;
}}
.log-row:hover {{ background: rgba(255,255,255,0.045); }}
.lr-CRITICAL {{ border-color: {C['hot']};  }}
.lr-HIGH     {{ border-color: {C['warn']}; }}
.lr-MEDIUM   {{ border-color: {C['amber']}; }}
.lr-LOW      {{ border-color: {C['safe']}; }}

/* ── about block ── */
.about-block {{
  background: {C['panel']};
  border: 1px solid {C['wire']};
  border-top: 2px solid {C['amber']};
  border-radius: 3px;
  padding: 1rem 1.2rem;
}}
.ab-name {{
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.3rem; font-weight: 700;
  color: {C['text']}; letter-spacing: 1px;
}}
.ab-role {{
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.63rem; color: {C['amber']};
  letter-spacing: 2px; margin-bottom: 0.6rem;
}}
.ab-line {{
  font-size: 0.75rem; color: {C['mist']};
  line-height: 1.85;
}}

/* ── plotly container ── */
.js-plotly-plot .plotly {{ border-radius: 3px !important; }}

/* ── scrollbar ── */
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: {C['ink']}; }}
::-webkit-scrollbar-thumb {{ background: {C['wire']}; border-radius: 2px; }}

/* ── streamlit override ── */
.stSlider > div > div {{ background: {C['wire']} !important; }}
.stMultiSelect > div {{ background: {C['panel']} !important;
                         border: 1px solid {C['wire']} !important; }}
.stCheckbox label {{ color: {C['text']} !important; }}
div[data-testid="stMetric"] {{ background: transparent !important; }}
[data-testid="stSelectbox"] > div {{
  background: {C['panel']} !important;
  border: 1px solid {C['wire']} !important;
  border-radius: 3px !important;
}}
button[kind="primary"] {{
  background: {C['amber']} !important;
  color: {C['ink']} !important;
  border: none !important;
  font-family: 'Barlow Condensed', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 1px !important;
}}
</style>
""", unsafe_allow_html=True)


# ── Data loaders ────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def load_alerts() -> pd.DataFrame:
    if not ALERT_LOG.exists():
        return _demo_alerts()
    try:
        with open(ALERT_LOG) as f:
            data = json.load(f)
        if not data:
            return _demo_alerts()
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["time_str"]  = df["timestamp"].dt.strftime("%H:%M:%S")
        return df
    except Exception:
        return _demo_alerts()

@st.cache_data(ttl=10)
def load_sessions() -> list:
    sessions = []
    for p in sorted(RESULTS_DIR.glob("session_*.json")):
        try:
            with open(p) as f:
                sessions.append(json.load(f))
        except Exception:
            pass
    return sessions if sessions else [_demo_session()]

@st.cache_data(ttl=10)
def load_anomaly() -> dict:
    if ANOMALY_JSON.exists():
        try:
            with open(ANOMALY_JSON) as f:
                return json.load(f)
        except Exception:
            pass
    return _demo_anomaly()

# ── Demo data ───────────────────────────────────────────────────────────────
def _demo_alerts() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    now = datetime.now()
    pool_r = ["military_vehicle detected","suspicious_object detected",
              "crowd gathering detected","aircraft in surveillance zone",
              "high motion activity (score=14.2)","anomaly IF score=-0.21",
              "multiple threat classes: aircraft, crowd","crowd + high motion"]
    classes = ["vehicle","person","crowd","aircraft","military_vehicle","ship","suspicious_object"]
    rows = []
    for i, pri in enumerate(["CRITICAL"]*5+["HIGH"]*18+["MEDIUM"]*12+["LOW"]*5):
        ts = now - timedelta(minutes=int(rng.integers(1,480)))
        rows.append(dict(
            alert_id        = f"alert_{i:04d}",
            frame_id        = int(rng.integers(30,950)),
            timestamp       = ts,
            time_str        = ts.strftime("%H:%M:%S"),
            priority        = pri,
            anomaly_score   = round(float(rng.uniform(-0.35,0.05)),4),
            anomaly_prob    = round(float(rng.uniform(0.4,0.98)),4),
            reasons         = [str(rng.choice(pool_r))],
            detection_count = int(rng.integers(1,20)),
            motion_score    = round(float(rng.uniform(3,18)),2),
            class_name      = str(rng.choice(classes)),
        ))
    return pd.DataFrame(rows).sort_values("timestamp", ascending=False)

def _demo_session() -> dict:
    return dict(source="data/test_videos/dota_aerial_test.mp4",
                elapsed_seconds=1420, effective_fps=4.2, total_frames=188,
                baseline_frames=30, frames_scored=158, model_fitted=True,
                total_detections=2351, normal_frames=120, high_alert_frames=33,
                critical_frames=5, alerts_raised=38, alert_rate=0.241,
                avg_inference_ms=86.24, avg_anomaly_ms=4.2, avg_preprocess_ms=12.1)

def _demo_anomaly() -> dict:
    return dict(total_frames=158, normal_frames=120, high_alert_frames=33,
                critical_frames=5, alert_rate=0.241, avg_anomaly_score=-0.063,
                min_anomaly_score=-0.287, avg_anomaly_prob=0.412, model_fitted=True)

def _demo_class_counts() -> dict:
    return dict(vehicle=2121, aircraft=222, suspicious_object=6,
                crowd=2, person=0, ship=0, military_vehicle=0)

# ── Chart theme helpers ──────────────────────────────────────────────────────
_BG    = "rgba(0,0,0,0)"
_GRID  = "rgba(46,58,80,0.6)"
_FONT  = "Barlow Condensed, Barlow, sans-serif"

def _layout(**kw):
    base = dict(paper_bgcolor=_BG, plot_bgcolor=_BG,
                font=dict(family=_FONT, color=C["mist"], size=11),
                margin=dict(l=8,r=8,t=28,b=8), showlegend=False)
    base.update(kw); return base

# ── Chart: Anomaly score – area sparkline ───────────────────────────────────
def chart_anomaly(df):
    if df.empty or "anomaly_score" not in df.columns: return None
    d = df.sort_values("frame_id" if "frame_id" in df.columns else "timestamp")
    xc = "frame_id" if "frame_id" in d.columns else "timestamp"
    fig = go.Figure()
    fig.add_hrect(y0=-1,y1=-0.15, fillcolor="rgba(255,85,51,0.07)",   line_width=0)
    fig.add_hrect(y0=-0.15,y1=-0.05, fillcolor="rgba(255,159,28,0.05)", line_width=0)
    fig.add_hline(y=-0.05, line=dict(color=C["warn"], dash="dash", width=1))
    fig.add_hline(y=-0.15, line=dict(color=C["hot"],  dash="dash", width=1))
    fig.add_trace(go.Scatter(
        x=d[xc], y=d["anomaly_score"], mode="lines",
        line=dict(color=C["amber"], width=1.8),
        fill="tozeroy", fillcolor="rgba(240,165,0,0.08)",
        hovertemplate="Frame %{x}<br>Score: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(**_layout(
        xaxis=dict(gridcolor=_GRID, tickfont=dict(size=9)),
        yaxis=dict(gridcolor=_GRID, tickfont=dict(size=9), title="Score"),
        height=190,
    ))
    return fig

# ── Chart: Priority ring ────────────────────────────────────────────────────
def chart_ring(df):
    if df.empty or "priority" not in df.columns: return None
    counts = df["priority"].value_counts().reset_index()
    counts.columns = ["priority","count"]
    colors = [PRI_CLR.get(p, C["amber"]) for p in counts["priority"]]
    fig = go.Figure(go.Pie(
        labels=counts["priority"], values=counts["count"],
        hole=0.7, marker=dict(colors=colors, line=dict(color=C["ink"], width=2)),
        textfont=dict(family=_FONT, size=10),
        hovertemplate="<b>%{label}</b><br>%{value} alerts — %{percent}<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{len(df)}</b>", x=0.5, y=0.55,
                       font=dict(size=24,color=C["text"],family=_FONT), showarrow=False)
    fig.add_annotation(text="ALERTS", x=0.5, y=0.38,
                       font=dict(size=9,color=C["dim"],family="Share Tech Mono, monospace"), showarrow=False)
    fig.update_layout(**_layout(height=220, margin=dict(l=4,r=4,t=6,b=6)))
    return fig

# ── Chart: Hourly stacked bar ───────────────────────────────────────────────
def chart_hourly(df):
    if df.empty or "timestamp" not in df.columns: return None
    d = df.copy(); d["hour"] = d["timestamp"].dt.floor("h")
    hourly = d.groupby(["hour","priority"]).size().reset_index(name="n")
    fig = go.Figure()
    for pri in ["CRITICAL","HIGH","MEDIUM","LOW"]:
        sub = hourly[hourly["priority"]==pri]
        if sub.empty: continue
        fig.add_trace(go.Bar(x=sub["hour"], y=sub["n"], name=pri,
                             marker_color=PRI_CLR.get(pri, C["amber"]), opacity=0.88,
                             hovertemplate=f"<b>{pri}</b><br>%{{x|%H:%M}}<br>n=%{{y}}<extra></extra>"))
    fig.update_layout(**_layout(
        barmode="stack", showlegend=True,
        legend=dict(orientation="h",x=0,y=1.12,
                    font=dict(size=9,color=C["mist"]),bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=_GRID,tickfont=dict(size=9),tickformat="%H:%M"),
        yaxis=dict(gridcolor=_GRID,tickfont=dict(size=9)),
        height=200,
    ))
    return fig

# ── Chart: Class distribution – vertical bars ───────────────────────────────
def chart_classes(counts: dict):
    if not counts: return None
    d = {k:v for k,v in counts.items() if v>0}
    if not d: return None
    cls_colors = {
        "vehicle": C["warn"], "aircraft": C["amber"], "crowd": C["hot"],
        "military_vehicle": "#ff2255", "person": C["safe"],
        "ship": C["ice"], "suspicious_object": "#c77dff",
    }
    labels, vals, colors = zip(*[(k,v,cls_colors.get(k,C["amber"])) for k,v in d.items()])
    fig = go.Figure(go.Bar(
        x=list(labels), y=list(vals),
        marker=dict(color=list(colors), opacity=0.85,
                    line=dict(color="rgba(255,255,255,0.06)",width=1)),
        text=list(vals), textposition="outside",
        textfont=dict(size=9,color=C["ghost"],family=_FONT),
        hovertemplate="<b>%{x}</b><br>%{y:,}<extra></extra>",
    ))
    fig.update_layout(**_layout(
        xaxis=dict(tickfont=dict(size=9,family=_FONT)),
        yaxis=dict(gridcolor=_GRID,tickfont=dict(size=9)),
        height=195, bargap=0.28,
    ))
    return fig

# ── Chart: Motion sparkline ─────────────────────────────────────────────────
def chart_motion(df):
    if df.empty or "motion_score" not in df.columns: return None
    d = df.dropna(subset=["motion_score"]).sort_values(
        "frame_id" if "frame_id" in df.columns else "timestamp")
    xc = "frame_id" if "frame_id" in d.columns else "timestamp"
    fig = go.Figure(go.Scatter(
        x=d[xc], y=d["motion_score"], mode="lines",
        line=dict(color=C["ice"], width=1.6),
        fill="tozeroy", fillcolor="rgba(126,207,255,0.06)",
        hovertemplate="F%{x}<br>Motion: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=8.0, line=dict(color=C["warn"], dash="dot", width=1))
    fig.update_layout(**_layout(
        xaxis=dict(gridcolor=_GRID,tickfont=dict(size=9)),
        yaxis=dict(gridcolor=_GRID,tickfont=dict(size=9),title="Motion"),
        height=180,
    ))
    return fig

# ── Chart: Heatmap contour ──────────────────────────────────────────────────
def chart_heatmap(df):
    rng = np.random.default_rng(int(len(df)) if not df.empty else 99)
    n = max(len(df)*8, 200)
    x = rng.normal(0.5, 0.22, n).clip(0,1)
    y = rng.normal(0.5, 0.20, n).clip(0,1)
    fig = go.Figure(go.Histogram2dContour(
        x=x, y=y, ncontours=14, showscale=False, line=dict(width=0),
        colorscale=[[0,"rgba(240,165,0,0)"],[0.35,"rgba(240,165,0,0.25)"],
                    [0.7,"rgba(255,159,28,0.65)"],[1,"rgba(255,85,51,0.95)"]],
    ))
    for zx,zy,zr,clr in [(0.1,0.5,0.08,"rgba(255,85,51,0.45)"),
                          (0.9,0.5,0.08,"rgba(255,85,51,0.45)"),
                          (0.5,0.1,0.07,"rgba(255,159,28,0.35)")]:
        fig.add_shape(type="circle",x0=zx-zr,x1=zx+zr,y0=zy-zr,y1=zy+zr,
                      line=dict(color=clr,width=1.5,dash="dot"))
    fig.update_layout(**_layout(
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,range=[0,1]),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,range=[0,1]),
        height=215, margin=dict(l=4,r=4,t=6,b=4),
    ))
    return fig

# ── HTML helpers ─────────────────────────────────────────────────────────────
def _stat(num, label, sub="", ac=None):
    ac = ac or C["amber"]
    return f"""
    <div class="stat-tile" style="--ac:{ac}">
      <div class="st-lbl">{label}</div>
      <div class="st-num">{num}</div>
      {'<div class="st-sub">'+sub+'</div>' if sub else ''}
    </div>"""

def _sec(txt, icon="▸"):
    st.markdown(f'<div class="sec-label">{icon} {txt}</div>', unsafe_allow_html=True)

def _pcard_open(title, icon="◆"):
    st.markdown(f'<div class="pcard"><div class="pcard-hdr">{icon} {title}</div>', unsafe_allow_html=True)

def _pcard_close():
    st.markdown('</div>', unsafe_allow_html=True)

def _astrip(a: dict) -> str:
    pri   = a.get("priority","LOW")
    frame = a.get("frame_id","—")
    rsns  = a.get("reasons",[])
    rsn   = rsns[0] if rsns else "—"
    ts    = a.get("time_str","")
    sc    = a.get("anomaly_score",0)
    return (f'<div class="astrip as-{pri}">'
            f'<span class="badge b-{pri}">{pri}</span>'
            f'<span class="mono">F{frame:04d}</span>'
            f'<span class="reason-txt" title="{rsn}">{rsn}</span>'
            f'<span class="mono">{sc:.3f}</span>'
            f'<span class="ts">{ts}</span>'
            f'</div>')


# ── TOPBAR ──────────────────────────────────────────────────────────────────
def render_topbar(session: dict):
    now    = datetime.now()
    src    = Path(session.get("source", "—")).name
    fps    = session.get("effective_fps", 0)
    fitted = session.get("model_fitted", False)
    fitted_clr = C["safe"] if fitted else C["warn"]
    fitted_txt = "FITTED" if fitted else "PENDING"

    # Topbar wrapper open
    st.markdown(f'<div class="topbar">', unsafe_allow_html=True)

    # Use a single HTML block — logo now via base64 img so it won't be escaped
    st.markdown(f"""
      <div style="display:flex;align-items:center;gap:0.9rem;flex:0 0 auto">
        {LOGO_IMG_TAG}
        <div>
          <div class="tb-brand">SENTINEL&#8209;AI</div>
          <div class="tb-sub">BORDER SURVEILLANCE COMMAND &nbsp;·&nbsp; TACTICAL DISPLAY</div>
        </div>
      </div>

      <div style="display:flex;gap:2.2rem;align-items:center;padding-left:1.8rem;flex:1">
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem">
          <div style="color:{C['dim']};letter-spacing:1px;margin-bottom:2px">SOURCE</div>
          <div style="color:{C['text']};font-size:0.75rem">{src}</div>
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem">
          <div style="color:{C['dim']};letter-spacing:1px;margin-bottom:2px">EFF. FPS</div>
          <div style="color:{C['amber']};font-size:0.9rem;font-weight:bold">{fps:.1f}</div>
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem">
          <div style="color:{C['dim']};letter-spacing:1px;margin-bottom:2px">IF MODEL</div>
          <div style="color:{fitted_clr};font-size:0.82rem">{fitted_txt}</div>
        </div>
      </div>

      <div style="text-align:right;font-family:'Share Tech Mono',monospace;
                  font-size:0.7rem;color:{C['mist']};flex:0 0 auto;line-height:1.8">
        <div><span class="pulse-dot"></span>OPERATIONAL</div>
        <div style="color:{C['dim']}">{now.strftime('%d %b %Y  ·  %a')}</div>
        <div class="tb-clock">{now.strftime('%H:%M:%S')}</div>
      </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── ROW A: Five stat tiles ───────────────────────────────────────────────────
def render_stats(df: pd.DataFrame, session: dict, anomaly: dict):
    _sec("LIVE METRICS", "◉")
    total   = len(df)
    crit    = len(df[df["priority"]=="CRITICAL"]) if not df.empty else 0
    high    = len(df[df["priority"]=="HIGH"])     if not df.empty else 0
    dets    = session.get("total_detections",0)
    rate    = anomaly.get("alert_rate", session.get("alert_rate",0))
    inf_ms  = session.get("avg_inference_ms",0)
    frames  = session.get("frames_scored",0)
    elapsed = session.get("elapsed_seconds",0)
    m,s     = divmod(int(elapsed),60)

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    with c1: st.markdown(_stat(f"{total:,}",  "TOTAL ALERTS",  "this session",       C["amber"]), unsafe_allow_html=True)
    with c2: st.markdown(_stat(str(crit),     "CRITICAL",      "immediate action",   C["hot"]),   unsafe_allow_html=True)
    with c3: st.markdown(_stat(str(high),     "HIGH PRIORITY", "requires attention", C["warn"]),  unsafe_allow_html=True)
    with c4: st.markdown(_stat(f"{dets:,}",   "DETECTIONS",    "objects identified", C["safe"]),  unsafe_allow_html=True)
    with c5: st.markdown(_stat(f"{rate:.1%}", "ALERT RATE",    "of scored frames",   C["ice"]),   unsafe_allow_html=True)
    with c6: st.markdown(_stat(f"{inf_ms:.0f}ms","INFERENCE",  "avg per frame",      C["mist"]),  unsafe_allow_html=True)
    with c7: st.markdown(_stat(f"{m}m{s:02d}s","DURATION",     f"{frames:,} frames scored", C["ghost"]), unsafe_allow_html=True)


# ── ROW B: Alert feed (wide) + right column (ring + heatmap) ────────────────
def render_feed_and_analysis(df: pd.DataFrame, n_alerts: int, pf: list):
    _sec("THREAT FEED & SPATIAL ANALYSIS", "◈")
    col_feed, col_right = st.columns([2.4, 1.0])

    # Feed
    with col_feed:
        _pcard_open("LIVE ALERT STREAM", "📡")
        filt = df[df["priority"].isin(pf)].head(n_alerts) if not df.empty else df
        if filt.empty:
            st.markdown(f'<div style="color:{C["ghost"]};padding:1.5rem;text-align:center;'
                        f'font-size:0.82rem">No alerts match filters — system monitoring…</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown("".join(_astrip(r) for r in filt.to_dict("records")),
                        unsafe_allow_html=True)
        _pcard_close()

    # Right: ring + heatmap stacked
    with col_right:
        _pcard_open("PRIORITY DISTRIBUTION", "◎")
        fig = chart_ring(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        # mini breakdown
        for pri,clr in [("CRITICAL",C["hot"]),("HIGH",C["warn"]),("MEDIUM",C["amber"]),("LOW",C["safe"])]:
            n = len(df[df["priority"]==pri]) if not df.empty else 0
            pct = n/max(len(df),1)*100
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;font-size:0.7rem;
                        color:{C['ghost']};padding:2px 0">
              <span style="color:{clr}">{pri}</span>
              <span style="color:{C['mist']}">{n} · {pct:.0f}%</span>
            </div>""", unsafe_allow_html=True)
        _pcard_close()

        _pcard_open("DETECTION HEATMAP", "🗺")
        fig = chart_heatmap(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown(f'<div style="font-size:0.62rem;color:{C["ghost"]};text-align:center;'
                    f'margin-top:-6px">spatial distribution of border events</div>',
                    unsafe_allow_html=True)
        _pcard_close()


# ── ROW C: Timelines ─────────────────────────────────────────────────────────
def render_timelines(df: pd.DataFrame):
    _sec("TEMPORAL ANALYSIS", "◉")
    ca, cb = st.columns([3, 2])

    with ca:
        _pcard_open("ANOMALY SCORE TIMELINE", "📈")
        fig = chart_anomaly(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.markdown(f'<div style="color:{C["ghost"]};padding:1.5rem;text-align:center;'
                        f'font-size:0.8rem">Run pipeline to populate anomaly data</div>',
                        unsafe_allow_html=True)
        _pcard_close()

    with cb:
        _pcard_open("ALERTS / HOUR", "📅")
        fig = chart_hourly(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        _pcard_close()


# ── ROW D: Class bars + Motion + Session metrics ─────────────────────────────
def render_detail_row(df: pd.DataFrame, session: dict):
    _sec("DETECTION BREAKDOWN & PERFORMANCE", "◆")
    cd, cm, cs = st.columns([1.4, 1.3, 1.0])

    with cd:
        _pcard_open("OBJECT CLASS DISTRIBUTION", "🔍")
        cc = {}
        sess = load_sessions()
        if sess:
            for a in sess[-1].get("alerts",[]):
                cn = a.get("class_name","")
                if cn: cc[cn] = cc.get(cn,0)+1
        if not cc: cc = _demo_class_counts()
        fig = chart_classes(cc)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        _pcard_close()

    with cm:
        _pcard_open("OPTICAL FLOW · MOTION INDEX", "〰")
        fig = chart_motion(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.markdown(f'<div style="color:{C["ghost"]};padding:1.5rem;text-align:center;'
                        f'font-size:0.8rem">Run pipeline to populate motion data</div>',
                        unsafe_allow_html=True)
        _pcard_close()

    with cs:
        _pcard_open("SESSION PERFORMANCE", "⚙")
        metrics = [
            ("Total frames",    session.get("total_frames",0)),
            ("Baseline frames", session.get("baseline_frames",0)),
            ("Frames scored",   session.get("frames_scored",0)),
            ("Avg inference",   f"{session.get('avg_inference_ms',0):.1f} ms"),
            ("Avg anomaly",     f"{session.get('avg_anomaly_ms',0):.1f} ms"),
            ("Avg preprocess",  f"{session.get('avg_preprocess_ms',0):.1f} ms"),
            ("Alert rate",      f"{session.get('alert_rate',0):.1%}"),
            ("Alerts raised",   session.get("alerts_raised",0)),
            ("Normal frames",   session.get("normal_frames",0)),
        ]
        for k,v in metrics:
            st.markdown(f'<div class="mrow"><span class="mrow-k">{k}</span>'
                        f'<span class="mrow-v">{v}</span></div>', unsafe_allow_html=True)
        _pcard_close()


# ── ROW E: Threat gauges ──────────────────────────────────────────────────────
def render_gauges(df: pd.DataFrame):
    _sec("THREAT ASSESSMENT MATRIX", "▲")
    total = max(len(df),1)
    gauges = [
        ("CRITICAL THREAT",  len(df[df["priority"]=="CRITICAL"])/total*100, C["hot"]),
        ("HIGH THREAT",      len(df[df["priority"]=="HIGH"])/total*100,     C["warn"]),
        ("MEDIUM WATCH",     len(df[df["priority"]=="MEDIUM"])/total*100,   C["amber"]),
        ("LOW / ROUTINE",    len(df[df["priority"]=="LOW"])/total*100,      C["safe"]),
    ]
    cols = st.columns(4)
    for col,(label,pct,clr) in zip(cols,gauges):
        with col:
            st.markdown(f"""
            <div class="tg-wrap" style="--ac:{clr}">
              <div class="tg-label">{label}</div>
              <div class="tg-pct" style="color:{clr}">{pct:.1f}%</div>
              <div class="tg-bar-bg" style="margin-top:0.4rem">
                <div class="tg-bar-fill" style="width:{min(pct,100):.1f}%;background:{clr}"></div>
              </div>
            </div>""", unsafe_allow_html=True)


# ── ROW F: Detailed log ───────────────────────────────────────────────────────
def render_log(df: pd.DataFrame, pf: list):
    _sec("FULL ALERT LOG", "☰")
    if df.empty: return

    show = [c for c in ["time_str","frame_id","priority","anomaly_score",
                          "anomaly_prob","detection_count","motion_score","reasons"]
            if c in df.columns]
    display = df[df["priority"].isin(pf)][show].head(60).copy()
    if "reasons" in display.columns:
        display["reason"] = display["reasons"].apply(
            lambda r: r[0] if isinstance(r,list) and r else str(r))

    st.markdown(f"""
    <div class="pcard">
      <div class="log-hdr">
        <span>TIME</span><span>FRAME</span><span>PRIORITY</span>
        <span>SCORE</span><span>DETS</span><span>MOTION</span><span>REASON</span>
      </div>""", unsafe_allow_html=True)

    for _,row in display.iterrows():
        pri   = row.get("priority","LOW")
        clr   = PRI_CLR.get(pri, C["amber"])
        fid   = row.get("frame_id",0)
        sc    = row.get("anomaly_score",0)
        dets  = row.get("detection_count","—")
        mot   = row.get("motion_score","—")
        mot   = f"{mot:.1f}" if isinstance(mot,float) else "—"
        rsn   = row.get("reason","—")
        ts    = row.get("time_str","—")
        st.markdown(
            f'<div class="log-row lr-{pri}">'
            f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:0.68rem;color:{C["ghost"]}">{ts}</span>'
            f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:0.68rem;color:{C["ice"]}">F{int(fid):04d}</span>'
            f'<span><span class="badge b-{pri}">{pri}</span></span>'
            f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:0.68rem;color:{clr}">{sc:.3f}</span>'
            f'<span style="font-size:0.72rem;color:{C["mist"]}">{dets}</span>'
            f'<span style="font-size:0.72rem;color:{C["mist"]}">{mot}</span>'
            f'<span style="font-size:0.72rem;color:{C["mist"]};overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{rsn}</span>'
            f'</div>',
            unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── CONTROLS ROW (inline — no sidebar) ───────────────────────────────────────
def render_controls(df: pd.DataFrame):
    _sec("DISPLAY CONTROLS", "⚙")
    c1,c2,c3,c4 = st.columns([1.4,2.2,1.0,1.0])
    with c1:
        n_alerts = st.slider("Feed rows", 5, 60, 20, label_visibility="visible")
    with c2:
        pf = st.multiselect("Priority filter",
                            ["CRITICAL","HIGH","MEDIUM","LOW"],
                            default=["CRITICAL","HIGH","MEDIUM","LOW"])
    with c3:
        auto = st.checkbox("Auto-refresh 5s", value=False)
        if auto:
            st_autorefresh(interval=5000, key="ar")
            st.caption("🔄 live")
    with c4:
        if st.button("⟳  Refresh now", type="primary"):
            st.cache_data.clear()
            st.rerun()
    return n_alerts, pf


# ── ABOUT + SYSTEM STATUS side-by-side ───────────────────────────────────────
def render_about_and_status(session: dict):
    _sec("OPERATOR INFO & SYSTEM HEALTH", "◉")
    ca, cb, cc = st.columns([1.5,1.5,2.0])

    with ca:
        st.markdown(f"""
        <div class="about-block">
          <div class="ab-name">Krish Patel</div>
          <div class="ab-role">AZURE SPECIALIST · AI ENGINEER</div>
          <div class="ab-line">
            🏛 Ahmedabad Institute of Technology<br>
            &nbsp;&nbsp;&nbsp;Ahmedabad, Gujarat<br>
            🎓 B.E. ICT · Class of 2026<br>
            📋 Enrollment: 220020107048<br>
            🏆 MS Elevate Internship 2026
          </div>
        </div>""", unsafe_allow_html=True)

    with cb:
        _pcard_open("SYSTEM STATUS", "◆")
        items = [
            ("Detection Pipeline", "ONLINE",      C["safe"]),
            ("YOLOv8 Model",       "LOADED",      C["safe"]),
            ("Isolation Forest",
             "FITTED" if session.get("model_fitted") else "PENDING",
             C["safe"] if session.get("model_fitted") else C["warn"]),
            ("Alert Manager",      "ACTIVE",      C["safe"]),
            ("Data Feed",
             "LIVE" if ALERT_LOG.exists() else "DEMO",
             C["safe"] if ALERT_LOG.exists() else C["amber"]),
        ]
        for nm,st_,clr in items:
            st.markdown(
                f'<div class="sys-row">'
                f'<span class="sys-key">{nm}</span>'
                f'<span class="sys-val" style="color:{clr}">{st_}</span>'
                f'</div>', unsafe_allow_html=True)
        _pcard_close()

    with cc:
        _pcard_open("PROJECT IDENTITY", "🛡")
        st.markdown(f"""
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
                    color:{C['mist']};line-height:2.0">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.05rem;
                      font-weight:700;color:{C['text']};letter-spacing:2px;
                      margin-bottom:0.5rem">
            BORDER DEFENCE AI · SENTINEL PLATFORM
          </div>
          <div>⚡ YOLOv8 real-time object detection</div>
          <div>🧠 Isolation Forest anomaly scoring</div>
          <div>📡 Multi-class threat classification</div>
          <div>🎯 Optical flow motion analysis</div>
          <div>🗺 Spatial heatmap visualisation</div>
          <div style="margin-top:0.5rem;color:{C['ghost']}">
            Stack: Python · Streamlit · Plotly · OpenCV · Ultralytics
          </div>
        </div>""", unsafe_allow_html=True)
        _pcard_close()


# ── FOOTER ───────────────────────────────────────────────────────────────────
def render_footer():
    st.markdown(f"""
    <div style="text-align:center;padding:1.2rem 0 0.6rem;margin-top:0.8rem;
                border-top:1px solid {C['wire']};
                font-family:'Share Tech Mono',monospace;
                font-size:0.62rem;color:{C['ghost']};letter-spacing:1px">
      SENTINEL-AI · BORDER DEFENCE COMMAND · MS ELEVATE INTERNSHIP 2026 ·
      AHMEDABAD INSTITUTE OF TECHNOLOGY · KRISH PATEL · 220020107048 ·
      <span style="color:{C['amber']}">YOLOv8 + ISOLATION FOREST + STREAMLIT</span>
    </div>""", unsafe_allow_html=True)


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    inject_css()

    df       = load_alerts()
    sessions = load_sessions()
    session  = sessions[-1] if sessions else _demo_session()
    anomaly  = load_anomaly()

    render_topbar(session)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    # controls first (inline, compact)
    n_alerts, pf = render_controls(df)

    render_stats(df, session, anomaly)
    render_feed_and_analysis(df, n_alerts, pf)
    render_timelines(df)
    render_detail_row(df, session)
    render_gauges(df)
    render_log(df, pf)
    render_about_and_status(session)
    render_footer()


if __name__ == "__main__":
    main()
