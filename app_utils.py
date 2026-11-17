"""Utilitas dashboard SER IndoWaveSentiment (v2 - tampilan dipercantik)."""
import os, sys, json, math, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
FONT = "Plus Jakarta Sans, Inter, system-ui, sans-serif"
PRIMARY = "#6366f1"
PALETTE = ["#6366f1", "#ec4899", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6"]

# label (Inggris, dari model) -> (nama Indonesia, emoji, warna)
EMO = {
    "neutral": ("Netral", "😐", "#94a3b8"),
    "happy": ("Senang", "😄", "#f59e0b"),
    "surprise": ("Terkejut", "😲", "#8b5cf6"),
    "disgust": ("Jijik", "🤢", "#10b981"),
    "disappointed": ("Kecewa", "😞", "#3b82f6"),
}
EMO_DESC = {
    "neutral": "Nada relatif datar dan stabil, tanpa luapan perasaan yang menonjol.",
    "happy": "Umumnya bernada lebih tinggi, bertenaga, dan terdengar ceria.",
    "surprise": "Sering ditandai lonjakan nada yang tiba-tiba dan tempo yang cepat.",
    "disgust": "Biasanya terdengar menekan, tegas, dan penuh penolakan.",
    "disappointed": "Cenderung turun nadanya, lebih lambat, dan terdengar lesu.",
}
SENT_MAP = {"neutral": "netral", "happy": "positif", "surprise": "positif", "disgust": "negatif", "disappointed": "negatif"}
SENT_COLOR = {"positif": "#22c55e", "netral": "#94a3b8", "negatif": "#ef4444"}
SENT_EMOJI = {"positif": "🙂", "netral": "😐", "negatif": "🙁"}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea, .stApp td, .stApp th {font-family: 'Inter', 'Plus Jakarta Sans', system-ui, sans-serif;}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .hero h1, .sec .t, .kpi .v, .result .name, .brand .t, .card h4 {font-family: 'Plus Jakarta Sans', 'Inter', system-ui, sans-serif !important;}
.stApp {background-image: radial-gradient(900px 420px at 8% -5%, rgba(99,102,241,.13), transparent 60%), radial-gradient(800px 380px at 100% 0%, rgba(236,72,153,.10), transparent 60%);}
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1280px;}
footer {visibility: hidden;}
@keyframes flow {0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;}}
@keyframes eq {from {transform: scaleY(.22);} to {transform: scaleY(1);}}
@keyframes rise {from {opacity: 0; transform: translateY(12px);} to {opacity: 1; transform: translateY(0);}}
.hero {position: relative; overflow: hidden; background: linear-gradient(120deg, #4f46e5, #7c3aed, #db2777, #4f46e5); background-size: 300% 300%; animation: flow 14s ease infinite; border-radius: 24px; padding: 2.2rem 2.4rem; margin-bottom: 1.4rem; box-shadow: 0 14px 40px rgba(79,70,229,.32);}
.hero:before {content: ""; position: absolute; width: 320px; height: 320px; right: -90px; top: -120px; border-radius: 50%; background: rgba(255,255,255,.12);}
.hero:after {content: ""; position: absolute; width: 200px; height: 200px; left: 38%; bottom: -130px; border-radius: 50%; background: rgba(255,255,255,.08);}
.hero h1 {position: relative; z-index: 1; font-size: 2.3rem; font-weight: 800; letter-spacing: -.02em; margin: 0 0 .4rem 0; padding: 0; color: #fff !important;}
.hero p {position: relative; z-index: 1; margin: 0; max-width: 760px; font-size: 1.04rem; line-height: 1.55; opacity: .94; color: #fff !important;}
.hero .chips {position: relative; z-index: 1; margin-top: 1rem;}
.chip {display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.32); padding: .28rem .8rem; border-radius: 999px; font-size: .82rem; font-weight: 600; margin: 0 .45rem .35rem 0; color: #fff;}
.eq {position: absolute; right: 32px; bottom: 22px; display: flex; align-items: flex-end; gap: 4px; z-index: 1; opacity: .85;}
.eq span {display: block; width: 5px; border-radius: 3px; background: rgba(255,255,255,.75); transform-origin: bottom; animation: eq 1.15s ease-in-out infinite alternate;}
@media (max-width: 760px) {.eq {display: none;} .hero h1 {font-size: 1.7rem;} .hero {padding: 1.6rem 1.4rem;}}
.sec {display: flex; align-items: center; gap: .6rem; margin: 1.7rem 0 .85rem 0;}
.sec:before {content: ""; width: 6px; height: 26px; border-radius: 4px; background: linear-gradient(180deg, #6366f1, #ec4899);}
.sec .t {font-size: 1.25rem; font-weight: 800;}
.sec .s {font-size: .86rem; opacity: .65; margin-left: .3rem;}
.card {position: relative; overflow: hidden; background: rgba(127,127,127,.07); border: 1px solid rgba(127,127,127,.18); border-radius: 18px; padding: 1.15rem 1.25rem; height: 100%; transition: transform .2s, box-shadow .2s; animation: rise .5s ease both;}
.card:hover {transform: translateY(-3px); box-shadow: 0 10px 26px rgba(99,102,241,.2);}
.card.acc {border-top: 4px solid var(--a);}
.card .ic {font-size: 1.8rem; margin-bottom: .25rem;}
.card h4 {margin: 0 0 .4rem 0; font-size: 1.05rem; font-weight: 700; padding: 0;}
.card .bd {font-size: .92rem; line-height: 1.6; opacity: .88;}
.card.center {text-align: center;}
.kpi {background: linear-gradient(135deg, rgba(99,102,241,.15), rgba(236,72,153,.08)); border: 1px solid rgba(99,102,241,.28); border-radius: 18px; padding: 1rem 1.15rem; height: 100%; animation: rise .5s ease both;}
.kpi .i {float: right; font-size: 1.5rem;}
.kpi .l {font-size: .74rem; opacity: .72; text-transform: uppercase; letter-spacing: .07em; font-weight: 600;}
.kpi .v {font-size: 1.7rem; font-weight: 800; margin-top: .15rem; line-height: 1.15;}
.kpi .s {font-size: .8rem; opacity: .7; margin-top: .1rem;}
.flow {display: flex; gap: .8rem; flex-wrap: wrap;}
.step {flex: 1 1 160px; position: relative; background: rgba(127,127,127,.07); border: 1px solid rgba(127,127,127,.18); border-radius: 16px; padding: 1.1rem .9rem .9rem .9rem; text-align: center; transition: transform .2s;}
.step:hover {transform: translateY(-3px);}
.step .n {position: absolute; top: -10px; left: -6px; width: 26px; height: 26px; border-radius: 50%; background: #6366f1; color: #fff; font-size: .8rem; font-weight: 700; display: flex; align-items: center; justify-content: center; box-shadow: 0 3px 8px rgba(99,102,241,.45);}
.step .e {font-size: 1.9rem;}
.step .t {font-weight: 700; margin-top: .2rem;}
.step .d {font-size: .8rem; opacity: .75; margin-top: .15rem; line-height: 1.45;}
.result {border-radius: 22px; padding: 1.7rem 1.4rem; text-align: center; color: #fff; box-shadow: 0 10px 28px rgba(0,0,0,.2); animation: rise .5s ease both;}
.result .emo {font-size: 3.8rem; line-height: 1.1;}
.result .name {font-size: 2rem; font-weight: 800; margin-top: .2rem; color: #fff;}
.result .conf {font-size: 1rem; opacity: .95;}
.pill {display: inline-block; padding: .28rem .85rem; border-radius: 999px; color: #fff; font-weight: 700; font-size: .85rem; margin-top: .7rem;}
.brand {display: flex; align-items: center; gap: .75rem; margin: .2rem 0 1rem 0;}
.brand .logo {width: 46px; height: 46px; border-radius: 14px; background: linear-gradient(135deg, #6366f1, #ec4899); display: flex; align-items: center; justify-content: center; font-size: 1.45rem; box-shadow: 0 6px 16px rgba(99,102,241,.4);}
.brand .t {font-weight: 800; font-size: 1.2rem; line-height: 1.1;}
.brand .s {font-size: .75rem; opacity: .7;}
.side {background: rgba(99,102,241,.10); border: 1px solid rgba(99,102,241,.25); border-radius: 16px; padding: .9rem 1rem;}
.side .l {font-size: .72rem; opacity: .7; text-transform: uppercase; letter-spacing: .06em; font-weight: 600;}
.side .m {font-weight: 700; margin: .15rem 0;}
.side .v {font-size: 1.9rem; font-weight: 800; line-height: 1.1;}
.badge {display: inline-block; padding: .2rem .65rem; border-radius: 999px; background: rgba(34,197,94,.16); color: #16a34a; font-size: .74rem; font-weight: 700;}
.foot {text-align: center; opacity: .62; font-size: .82rem; margin-top: 2.6rem; padding-top: 1rem; border-top: 1px solid rgba(127,127,127,.2);}
section[data-testid="stSidebar"] {background-image: linear-gradient(180deg, rgba(99,102,241,.13), rgba(236,72,153,.07)); border-right: 1px solid rgba(127,127,127,.2);}
section[data-testid="stSidebar"] [role="radiogroup"] label {padding: .5rem .8rem; border-radius: 12px; margin-bottom: .15rem; transition: background .15s;}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {background: rgba(99,102,241,.13);}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background: linear-gradient(135deg, rgba(99,102,241,.25), rgba(236,72,153,.15)); font-weight: 700;}
.stTabs [data-baseweb="tab-list"] {gap: .4rem; border-bottom: none;}
.stTabs [data-baseweb="tab"] {border-radius: 999px; padding: .45rem 1.05rem; background: rgba(127,127,127,.09); font-weight: 600; height: auto;}
.stTabs [aria-selected="true"] {background: linear-gradient(135deg, #6366f1, #8b5cf6);}
.stTabs [aria-selected="true"] p {color: #fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}
.stButton > button {border-radius: 12px; border: none; background: linear-gradient(135deg, #6366f1, #8b5cf6); font-weight: 600; padding: .5rem 1.3rem; box-shadow: 0 4px 12px rgba(99,102,241,.35);}
.stButton > button p {color: #fff !important;}
[data-testid="stFileUploaderDropzone"] {border: 2px dashed rgba(99,102,241,.55); border-radius: 16px; background: rgba(99,102,241,.06);}
[data-testid="stPlotlyChart"] {background: rgba(127,127,127,.05); border: 1px solid rgba(127,127,127,.15); border-radius: 18px; padding: .4rem;}
"""


# ----------------------------------------------------------------- komponen HTML
def inject_css():
    st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


def hero(title, subtitle, chips=None):
    eq = "".join(f'<span style="height:{int(18 + 46 * abs(math.sin(i * 0.75)))}px;animation-delay:{i * 0.07:.2f}s"></span>' for i in range(26))
    ch = "".join(f'<span class="chip">{c}</span>' for c in (chips or []))
    html = f'<div class="hero"><div class="eq">{eq}</div><h1>{title}</h1><p>{subtitle}</p>'
    if ch: html += f'<div class="chips">{ch}</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


def section(title, sub=""):
    s = f'<span class="s">{sub}</span>' if sub else ""
    st.markdown(f'<div class="sec"><span class="t">{title}</span>{s}</div>', unsafe_allow_html=True)


def kpi(label, value, sub="", icon=""):
    i = f'<span class="i">{icon}</span>' if icon else ""
    st.markdown(f'<div class="kpi">{i}<div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


def card(title, body, icon="", accent=None, center=False):
    cls = "card" + (" acc" if accent else "") + (" center" if center else "")
    style = f' style="--a:{accent}"' if accent else ""
    ic = f'<div class="ic">{icon}</div>' if icon else ""
    st.markdown(f'<div class="{cls}"{style}>{ic}<h4>{title}</h4><div class="bd">{body}</div></div>', unsafe_allow_html=True)


def steps(items):
    html = '<div class="flow">' + "".join(
        f'<div class="step"><div class="n">{i}</div><div class="e">{e}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
        for i, (e, t, d) in enumerate(items, 1)) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def brand():
    st.markdown('<div class="brand"><div class="logo">🎙️</div><div><div class="t">SER Indo</div><div class="s">Emosi dari suara berbahasa Indonesia</div></div></div>', unsafe_allow_html=True)


def side_model(name, detail, f1, std):
    st.markdown(f'<div class="side"><div class="l">Model aktif</div><div class="m">{name}</div><div class="s" style="font-size:.8rem;opacity:.75">{detail}</div>'
                f'<div class="l" style="margin-top:.7rem">Macro-F1 (5 fold)</div><div class="v">{f1:.3f}</div>'
                f'<div style="font-size:.8rem;opacity:.7">± {std:.3f}</div><div style="margin-top:.6rem"><span class="badge">● Model siap</span></div></div>', unsafe_allow_html=True)


def footer():
    st.markdown('<div class="foot">🎙️ SER IndoWaveSentiment · validasi 5-fold berbasis aktor · dibuat dengan Streamlit &amp; Plotly</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------- tampilan kompatibel antar versi Streamlit
def show(fig, height=None):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, size=13),
                      margin=dict(l=12, r=12, t=52, b=12), colorway=PALETTE, title_font=dict(size=16, family=FONT),
                      hoverlabel=dict(font_family=FONT))
    try: fig.update_traces(marker_cornerradius=8, selector=dict(type="bar"))
    except Exception: pass
    if height: fig.update_layout(height=height)
    try: st.plotly_chart(fig, width="stretch")
    except TypeError: st.plotly_chart(fig, use_container_width=True)


def table(df, **kw):
    try: st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError: st.dataframe(df, use_container_width=True, hide_index=True, **kw)


# ----------------------------------------------------------------- aset & model
@st.cache_data(show_spinner=False)
def load_assets():
    return {
        "res": pd.read_csv(ASSETS / "results_all.csv"),
        "lead": pd.read_csv(ASSETS / "leaderboard.csv"),
        "meta": pd.read_csv(ASSETS / "dataset_meta.csv"),
        "oof": pd.read_csv(ASSETS / "oof_all.csv"),
        "selection": json.load(open(ASSETS / "selection.json")),
    }


@st.cache_resource(show_spinner="Memuat model…")
def load_model():
    src = str(ROOT / "src")
    if src not in sys.path: sys.path.insert(0, src)
    import infer
    return infer.load(str(ROOT / "model"), src)


def analyze(pre, data: bytes, suffix=".wav"):
    """Simpan audio ke berkas sementara, prediksi dengan infer.predict, kembalikan (labels, probs, sinyal)."""
    import infer, librosa
    fd, tmp = tempfile.mkstemp(suffix=suffix); os.close(fd)
    try:
        with open(tmp, "wb") as f: f.write(data)
        labels, p = infer.predict(pre, tmp)
        y = librosa.load(tmp, sr=16000, mono=True)[0]
    finally:
        try: os.remove(tmp)
        except OSError: pass
    return list(labels), np.asarray(p, dtype=float), y


def sentiment_probs(labels, p):
    out = {"positif": 0.0, "netral": 0.0, "negatif": 0.0}
    for l, v in zip(labels, p): out[SENT_MAP[l]] += float(v)
    return out


# ----------------------------------------------------------------- grafik audio & probabilitas
def fig_waveform(y, sr=16000):
    step = max(1, len(y) // 4000)
    t = np.arange(len(y))[::step] / sr
    fig = go.Figure(go.Scatter(x=t, y=y[::step], mode="lines", line=dict(color=PRIMARY, width=1),
                               fill="tozeroy", fillcolor="rgba(99,102,241,.25)"))
    fig.update_layout(title="Gelombang suara", xaxis_title="detik", yaxis_title="amplitudo", showlegend=False)
    return fig


def fig_mel(y, sr=16000):
    import librosa
    M = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128), ref=np.max, top_db=80)
    t = np.arange(M.shape[1]) * 256 / sr
    fig = px.imshow(M, x=t, origin="lower", aspect="auto", color_continuous_scale="Magma", labels=dict(x="detik", y="band mel", color="dB"))
    fig.update_layout(title="Spektrogram Log-Mel", coloraxis_showscale=False)
    return fig


def fig_probs(labels, p):
    order = np.argsort(p)
    names = [f"{EMO[labels[i]][1]} {EMO[labels[i]][0]}" for i in order]
    cols = [EMO[labels[i]][2] for i in order]
    fig = go.Figure(go.Bar(x=[float(p[i]) for i in order], y=names, orientation="h", marker_color=cols,
                           text=[f"{p[i] * 100:.1f}%" for i in order], textposition="outside", cliponaxis=False))
    fig.update_layout(title="Probabilitas tiap emosi", xaxis=dict(range=[0, 1.15], tickformat=".0%"), showlegend=False)
    return fig


def fig_sentiment(sp):
    keys = ["negatif", "netral", "positif"]
    fig = go.Figure(go.Bar(x=[f"{SENT_EMOJI[k]} {k}" for k in keys], y=[sp[k] for k in keys], marker_color=[SENT_COLOR[k] for k in keys],
                           text=[f"{sp[k] * 100:.1f}%" for k in keys], textposition="outside", cliponaxis=False))
    fig.update_layout(title="Sentimen 3 kelas", yaxis=dict(range=[0, 1.2], tickformat=".0%"), showlegend=False)
    return fig


def fig_gauge(value, color):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=float(value) * 100, number=dict(suffix="%", font=dict(size=38)),
                                 gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=color, thickness=0.3), bgcolor="rgba(127,127,127,.15)", borderwidth=0,
                                            steps=[dict(range=[0, 50], color="rgba(239,68,68,.12)"), dict(range=[50, 80], color="rgba(245,158,11,.12)"),
                                                   dict(range=[80, 100], color="rgba(34,197,94,.14)")])))
    fig.update_layout(title="Tingkat keyakinan")
    return fig


def fig_radar(labels, p):
    names = [f"{EMO[l][1]} {EMO[l][0]}" for l in labels]
    r = [float(v) for v in p]
    fig = go.Figure(go.Scatterpolar(r=r + r[:1], theta=names + names[:1], fill="toself", line=dict(color=PRIMARY, width=2),
                                    fillcolor="rgba(99,102,241,.28)"))
    fig.update_layout(title="Profil emosi (radar)", showlegend=False, polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 1], tickformat=".0%", showline=False)))
    return fig
