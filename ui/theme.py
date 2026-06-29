"""Visual system for the SciUQ Streamlit application."""

import html

import streamlit as st


CSS = """
<style>
:root {
  --navy: #0b132b;
  --navy-2: #14213d;
  --blue: #2563eb;
  --cyan: #06b6d4;
  --ink: #172033;
  --muted: #64748b;
  --line: #e2e8f0;
  --surface: #ffffff;
  --canvas: #f4f7fb;
}

html, body, [class*="css"] { font-family: Inter, "PingFang SC", "Microsoft YaHei", sans-serif; }
.stApp { background: radial-gradient(circle at 85% -10%, #e0f2fe 0, transparent 25%), var(--canvas); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 1rem; }
.block-container { max-width: 1420px; padding-top: 1.6rem; padding-bottom: 3rem; }

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b132b 0%, #111d3a 58%, #172554 100%);
  border-right: 0;
}
[data-testid="stSidebar"] * { color: #dbeafe; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  border-radius: 10px; padding: .42rem .65rem; margin: .12rem 0;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.08); }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12); }
.sidebar-brand { padding: .45rem .25rem 1.1rem; }
.sidebar-brand__mark {
  display: inline-grid; place-items: center; width: 38px; height: 38px;
  border-radius: 12px; background: linear-gradient(135deg,#38bdf8,#2563eb);
  font-weight: 800; color: white; box-shadow: 0 8px 22px rgba(37,99,235,.35);
}
.sidebar-brand__name { font-weight: 800; font-size: 1.16rem; margin-left: .65rem; color: white; }
.sidebar-brand__sub { color: #93c5fd; font-size: .76rem; margin: .55rem 0 0 .1rem; letter-spacing: .04em; }
.user-card { padding: .8rem; border: 1px solid rgba(255,255,255,.12); border-radius: 13px; background: rgba(255,255,255,.06); margin-bottom: .8rem; }
.user-card__name { color: white; font-weight: 700; }
.user-card__meta { color: #93c5fd; font-size: .78rem; margin-top: .18rem; }

h1, h2, h3 { color: var(--ink); letter-spacing: -.025em; }
.page-kicker { color: var(--blue); font-weight: 800; font-size: .74rem; letter-spacing: .12em; text-transform: uppercase; margin-bottom: .32rem; }
.page-title { color: var(--ink); font-size: 2.05rem; line-height: 1.16; font-weight: 850; letter-spacing: -.035em; margin: 0; }
.page-description { color: var(--muted); font-size: .96rem; margin: .62rem 0 1.45rem; max-width: 850px; line-height: 1.72; }
.section-label { color: var(--ink); font-size: 1.12rem; font-weight: 800; margin: 1.25rem 0 .7rem; }

.hero {
  position: relative; overflow: hidden; padding: 2.2rem 2.35rem; border-radius: 22px;
  background: linear-gradient(120deg,#0b132b 0%,#172554 54%,#164e63 100%);
  box-shadow: 0 18px 45px rgba(15,23,42,.18); margin-bottom: 1.25rem;
}
.hero:after { content:""; position:absolute; width:330px; height:330px; border-radius:50%; right:-100px; top:-160px; background:rgba(56,189,248,.18); }
.hero__eyebrow { color:#67e8f9; font-size:.78rem; font-weight:800; letter-spacing:.13em; }
.hero__title { color:white; font-size:2rem; font-weight:850; max-width:820px; line-height:1.25; margin:.45rem 0 .8rem; }
.hero__copy { color:#bfdbfe; max-width:760px; line-height:1.75; font-size:.96rem; }
.hero__pill { display:inline-block; margin-top:1rem; padding:.35rem .7rem; color:#cffafe; background:rgba(6,182,212,.16); border:1px solid rgba(103,232,249,.28); border-radius:999px; font-size:.76rem; }

.feature-card { min-height: 175px; padding: 1.25rem 1.3rem; border: 1px solid var(--line); border-radius: 16px; background: rgba(255,255,255,.92); box-shadow: 0 8px 24px rgba(15,23,42,.055); }
.feature-card__icon { font-size: 1.45rem; }
.feature-card__title { color:var(--ink); font-weight:800; font-size:1.02rem; margin:.6rem 0 .35rem; }
.feature-card__copy { color:var(--muted); font-size:.86rem; line-height:1.62; }
.feature-card__tag { color:var(--blue); font-weight:700; font-size:.73rem; margin-top:.7rem; }

[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); padding: 1rem 1.05rem; border-radius: 14px; box-shadow: 0 5px 17px rgba(15,23,42,.04); }
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stMetricValue"] { color: var(--ink); font-weight: 800; }
[data-testid="stFileUploader"] { background:white; border:1px solid var(--line); border-radius:14px; padding:.65rem; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
[data-testid="stForm"] { background:rgba(255,255,255,.92); border:1px solid var(--line); border-radius:16px; padding:1rem 1.15rem; }

.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius: 10px; font-weight: 750; min-height: 2.65rem; border-color:#cbd5e1;
}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] {
  background: linear-gradient(90deg,#2563eb,#0891b2); border:0; box-shadow:0 7px 18px rgba(37,99,235,.22);
}
.stTabs [data-baseweb="tab-list"] { gap:.4rem; background:#eaf0f8; border-radius:12px; padding:.3rem; }
.stTabs [data-baseweb="tab"] { border-radius:9px; padding:.55rem 1rem; }
.stTabs [aria-selected="true"] { background:white; box-shadow:0 2px 9px rgba(15,23,42,.08); }

.auth-shell { padding-top: 5vh; }
.auth-hero { min-height: 560px; padding: 3rem; border-radius: 24px; background: linear-gradient(145deg,#091128,#172554 58%,#0e7490); color:white; box-shadow:0 24px 60px rgba(15,23,42,.22); }
.auth-hero__logo { display:inline-grid; place-items:center; width:50px; height:50px; border-radius:15px; background:linear-gradient(135deg,#67e8f9,#2563eb); font-size:1.25rem; font-weight:900; }
.auth-hero__title { font-size:2.35rem; font-weight:850; line-height:1.2; margin:2.3rem 0 1rem; max-width:540px; }
.auth-hero__copy { color:#bfdbfe; line-height:1.8; max-width:520px; }
.auth-list { margin-top:2.2rem; display:grid; gap:.9rem; color:#dbeafe; font-size:.9rem; }
.auth-list span { color:#67e8f9; margin-right:.55rem; }
.auth-panel-title { font-size:1.65rem; font-weight:850; color:var(--ink); margin:.4rem 0 .25rem; }
.auth-panel-copy { color:var(--muted); margin-bottom:1.2rem; }
.status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#22c55e; margin-right:.4rem; }

@media (max-width: 900px) {
  .block-container { padding: 1rem; }
  .auth-hero { min-height:auto; padding:1.6rem; }
  .auth-hero__title { font-size:1.7rem; margin-top:1.2rem; }
  .page-title { font-size:1.65rem; }
}
</style>
"""


def inject_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="page-kicker">{html.escape(kicker)}</div>'
        f'<h1 class="page-title">{html.escape(title)}</h1>'
        f'<p class="page-description">{html.escape(description)}</p>',
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{html.escape(text)}</div>', unsafe_allow_html=True)

