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
[data-testid="stHeader"] { background: transparent; height: 2.6rem; }
[data-testid="stToolbar"] { right: 1rem; }
.block-container { max-width: 1420px; padding-top: 1.15rem; padding-bottom: 3rem; }

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b132b 0%, #111d3a 58%, #172554 100%);
  border-right: 0;
}
[data-testid="stSidebar"] * { color: #dbeafe; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  border-radius: 11px; padding: .58rem .72rem; margin: .14rem 0; transition: all .18s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.08); }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background: linear-gradient(90deg,rgba(37,99,235,.48),rgba(6,182,212,.17));
  box-shadow: inset 3px 0 0 #38bdf8, 0 6px 18px rgba(0,0,0,.12);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child { display:none; }
[data-testid="stSidebar"] [data-testid="stRadio"] label p { font-weight:650; font-size:.88rem; }
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
.page-head { display:flex; align-items:flex-start; justify-content:space-between; gap:1.2rem; padding:1.15rem 1.3rem; margin:0 0 1.15rem; border:1px solid rgba(203,213,225,.82); border-radius:18px; background:rgba(255,255,255,.78); box-shadow:0 8px 26px rgba(15,23,42,.045); backdrop-filter:blur(12px); }
.page-head__main { display:flex; gap:1rem; align-items:flex-start; }
.page-head__icon { display:grid; place-items:center; flex:0 0 46px; width:46px; height:46px; border-radius:14px; color:white; font-weight:850; font-size:1.1rem; background:linear-gradient(135deg,#2563eb,#06b6d4); box-shadow:0 9px 22px rgba(37,99,235,.24); }
.page-head__state { white-space:nowrap; margin-top:.25rem; color:#0369a1; background:#e0f2fe; border:1px solid #bae6fd; padding:.38rem .65rem; border-radius:999px; font-size:.72rem; font-weight:750; }
.page-kicker { color: var(--blue); font-weight: 800; font-size: .7rem; letter-spacing: .12em; text-transform: uppercase; margin-bottom: .22rem; }
.page-title { color: var(--ink); font-size: 1.72rem; line-height: 1.16; font-weight: 850; letter-spacing: -.03em; margin: 0; }
.page-description { color: var(--muted); font-size: .9rem; margin: .42rem 0 0; max-width: 850px; line-height: 1.62; }
.section-label { color: var(--ink); font-size: 1.12rem; font-weight: 800; margin: 1.25rem 0 .7rem; }

.workflow { display:grid; grid-template-columns:repeat(4,1fr); gap:.6rem; margin:.35rem 0 1rem; }
.workflow__step { position:relative; display:flex; align-items:center; gap:.65rem; padding:.7rem .8rem; border:1px solid var(--line); border-radius:12px; background:white; color:#64748b; font-size:.78rem; font-weight:650; }
.workflow__step b { display:grid; place-items:center; width:25px; height:25px; border-radius:8px; background:#eef2ff; color:#4f46e5; font-size:.7rem; }
.workflow__step--active { border-color:#93c5fd; color:#1e40af; background:linear-gradient(135deg,#eff6ff,#ecfeff); }
.workflow__step--active b { color:white; background:linear-gradient(135deg,#2563eb,#06b6d4); }

.summary-band { display:flex; justify-content:space-between; align-items:center; gap:1rem; padding:.78rem 1rem; border-radius:13px; margin:.4rem 0 1rem; background:linear-gradient(90deg,#eff6ff,#ecfeff); border:1px solid #bfdbfe; }
.summary-band__title { color:#1e3a8a; font-weight:800; }
.summary-band__copy { color:#475569; font-size:.8rem; margin-top:.15rem; }
.summary-band__badge { color:#0e7490; background:white; padding:.38rem .65rem; border-radius:999px; border:1px solid #a5f3fc; font-size:.72rem; font-weight:800; }

.legend-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.65rem; margin:.45rem 0 1rem; }
.legend-card { padding:.78rem .9rem; background:white; border:1px solid var(--line); border-radius:12px; }
.legend-card__bar { width:28px; height:4px; border-radius:9px; margin-bottom:.55rem; }
.legend-card__name { color:var(--ink); font-weight:800; font-size:.82rem; }
.legend-card__range { color:var(--muted); font-size:.72rem; margin-top:.18rem; }

.project-card { min-height:165px; padding:1.05rem 1.15rem; border:1px solid var(--line); border-radius:15px; background:white; box-shadow:0 6px 20px rgba(15,23,42,.045); margin-bottom:.75rem; }
.project-card__top { display:flex; justify-content:space-between; gap:.6rem; align-items:center; }
.project-card__code { color:#2563eb; font-size:.71rem; font-weight:850; letter-spacing:.06em; }
.project-card__status { color:#047857; background:#d1fae5; border-radius:999px; padding:.22rem .48rem; font-size:.68rem; font-weight:750; }
.project-card__title { color:var(--ink); font-size:1rem; font-weight:820; margin:.65rem 0 .32rem; }
.project-card__copy { color:var(--muted); font-size:.78rem; line-height:1.52; min-height:38px; }
.project-card__meta { margin-top:.7rem; padding-top:.58rem; border-top:1px solid #eef2f7; color:#64748b; font-size:.7rem; }

.model-card { min-height:245px; padding:1.35rem; border-radius:18px; border:1px solid var(--line); background:linear-gradient(145deg,#fff,#f8fafc); box-shadow:0 8px 28px rgba(15,23,42,.055); }
.model-card__head { display:flex; align-items:center; justify-content:space-between; }
.model-card__logo { display:grid; place-items:center; width:44px; height:44px; border-radius:13px; color:white; font-size:.76rem; font-weight:900; background:linear-gradient(135deg,#1d4ed8,#0891b2); }
.model-card__state { color:#a16207; background:#fef3c7; border:1px solid #fde68a; border-radius:999px; padding:.28rem .55rem; font-size:.69rem; font-weight:800; }
.model-card__title { color:var(--ink); font-size:1.18rem; font-weight:850; margin:.9rem 0 .25rem; }
.model-card__task { color:#64748b; font-size:.8rem; }
.model-card__path { color:#334155; font-family:ui-monospace,SFMono-Regular,monospace; background:#f1f5f9; border-radius:8px; padding:.5rem .6rem; font-size:.7rem; margin:.8rem 0; word-break:break-all; }
.model-card__output { color:#0e7490; font-size:.76rem; font-weight:700; }
.model-card__progress { height:6px; border-radius:10px; background:#e2e8f0; overflow:hidden; margin-top:1rem; }
.model-card__progress span { display:block; height:100%; width:18%; background:linear-gradient(90deg,#2563eb,#06b6d4); }

.profile-card { min-height:255px; padding:1.5rem; color:white; border-radius:20px; background:linear-gradient(145deg,#0b132b,#172554 58%,#155e75); box-shadow:0 16px 36px rgba(15,23,42,.18); }
.profile-card__avatar { display:grid; place-items:center; width:58px; height:58px; border-radius:18px; background:linear-gradient(135deg,#38bdf8,#2563eb); font-weight:900; font-size:1.1rem; }
.profile-card__name { font-size:1.35rem; font-weight:850; margin:1rem 0 .2rem; }
.profile-card__user { color:#93c5fd; font-size:.8rem; }
.profile-card__row { display:flex; justify-content:space-between; border-top:1px solid rgba(255,255,255,.12); padding-top:.72rem; margin-top:.72rem; color:#dbeafe; font-size:.78rem; }

.empty-state { text-align:center; padding:2.4rem 1rem; border:1px dashed #cbd5e1; border-radius:15px; background:rgba(255,255,255,.65); }
.empty-state__icon { font-size:1.75rem; color:#38bdf8; }
.empty-state__title { color:var(--ink); font-weight:800; margin:.55rem 0 .25rem; }
.empty-state__copy { color:var(--muted); font-size:.82rem; }

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

[data-testid="stMetric"] { background: linear-gradient(145deg,#fff,#fbfdff); border: 1px solid var(--line); padding: .9rem .85rem; border-radius: 14px; box-shadow: 0 5px 17px rgba(15,23,42,.04); }
[data-testid="stMetricLabel"] { color: var(--muted); font-size: .72rem; }
[data-testid="stMetricValue"] { color: var(--ink); font-weight: 800; font-size: 1.55rem; white-space: nowrap; }
[data-testid="stFileUploader"] { background:linear-gradient(145deg,#fff,#f8fbff); border:1px dashed #93c5fd; border-radius:14px; padding:.65rem; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
[data-testid="stForm"] { background:rgba(255,255,255,.92); border:1px solid var(--line); border-radius:16px; padding:1rem 1.15rem; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:#dbe4ef !important; border-radius:16px !important; background:rgba(255,255,255,.82); box-shadow:0 7px 22px rgba(15,23,42,.035); }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:13px; background:rgba(255,255,255,.75); overflow:hidden; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div, textarea { border-radius:10px !important; }

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
  .workflow, .legend-grid { grid-template-columns:1fr 1fr; }
  .page-head__state { display:none; }
}
</style>
"""


def inject_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(kicker: str, title: str, description: str, icon: str = "UQ", state: str = "工作区在线") -> None:
    st.markdown(
        '<div class="page-head"><div class="page-head__main">'
        f'<div class="page-head__icon">{html.escape(icon)}</div><div>'
        f'<div class="page-kicker">{html.escape(kicker)}</div>'
        f'<h1 class="page-title">{html.escape(title)}</h1>'
        f'<p class="page-description">{html.escape(description)}</p></div></div>'
        f'<div class="page-head__state">● {html.escape(state)}</div></div>',
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def workflow_steps(items: list[str], active: int = 1) -> None:
    blocks = []
    for index, item in enumerate(items, 1):
        modifier = " workflow__step--active" if index == active else ""
        blocks.append(
            f'<div class="workflow__step{modifier}"><b>{index:02d}</b>{html.escape(item)}</div>'
        )
    st.markdown(f'<div class="workflow">{"".join(blocks)}</div>', unsafe_allow_html=True)


def empty_state(title: str, copy: str, icon: str = "◇") -> None:
    st.markdown(
        f'<div class="empty-state"><div class="empty-state__icon">{html.escape(icon)}</div>'
        f'<div class="empty-state__title">{html.escape(title)}</div>'
        f'<div class="empty-state__copy">{html.escape(copy)}</div></div>',
        unsafe_allow_html=True,
    )
