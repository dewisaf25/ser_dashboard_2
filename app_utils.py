"""Utilitas dashboard SER IndoWaveSentiment: muat model, aset, prediksi, tema, dan grafik."""
import os, sys, json, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

# label (Inggris, dari model) -> (nama Indonesia, emoji, warna)
EMO = {
    "neutral": ("Netral", "😐", "#94a3b8"),
    "happy": ("Senang", "😄", "#f59e0b"),
    "surprise": ("Terkejut", "😲", "#8b5cf6"),
    "disgust": ("Jijik", "🤢", "#10b981"),
    "disappointed": ("Kecewa", "😞", "#3b82f6"),
}
SENT_MAP = {"neutral": "netral", "happy": "positif", "surprise": "positif", "disgust": "negatif", "disappointed": "negatif"}
SENT_COLOR = {"positif": "#22c55e", "netral": "#94a3b8", "negatif": "#ef4444"}
SENT_EMOJI = {"positif": "🙂", "netral": "😐", "negatif": "🙁"}
PRIMARY = "#6366f1"


# ----------------------------------------------------------------- tema
def inject_css():
    st.markdown(
        """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
       border-radius: 22px; padding: 2.0rem 2.2rem; color: white; margin-bottom: 1.4rem;
       box-shadow: 0 10px 30px rgba(79,70,229,.30);}
.hero h1 {margin: 0 0 .35rem 0; font-size: 2.1rem; color: white;}
.hero p {margin: 0; opacity: .92; font-size: 1.02rem;}
.card {background: rgba(127,127,127,.08); border: 1px solid rgba(127,127,127,.18);
       border-radius: 16px; padding: 1.1rem 1.2rem; height: 100%;}
.card h4 {margin: 0 0 .4rem 0;}
.kpi {background: rgba(99,102,241,.10); border: 1px solid rgba(99,102,241,.30);
      border-radius: 16px; padding: 1rem 1.1rem;}
.kpi .l {font-size: .80rem; opacity: .75; text-transform: uppercase; letter-spacing: .04em;}
.kpi .v {font-size: 1.55rem; font-weight: 700; margin-top: .1rem;}
.kpi .s {font-size: .80rem; opacity: .7;}
.result {border-radius: 20px; padding: 1.6rem 1.4rem; text-align: center; color: white;
         box-shadow: 0 8px 24px rgba(0,0,0,.18);}
.result .emo {font-size: 3.6rem; line-height: 1.1;}
.result .name {font-size: 1.9rem; font-weight: 800; margin-top: .2rem;}
.result .conf {font-size: 1.0rem; opacity: .95;}
.pill {display: inline-block; padding: .25rem .8rem; border-radius: 999px; color: white;
       font-weight: 600; font-size: .85rem; margin-top: .6rem;}
section[data-testid="stSidebar"] {border-right: 1px solid rgba(127,127,127,.2);}
</style>
""",
        unsafe_allow_html=True,
    )


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(label, value, sub=""):
    st.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>',
                unsafe_allow_html=True)


def card(title, body):
    st.markdown(f'<div class="card"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------- tampilan kompatibel antar versi Streamlit
def show(fig, height=None):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=45, b=10), font=dict(size=13))
    if height: fig.update_layout(height=height)
    try: st.plotly_chart(fig, width="stretch")
    except TypeError: st.plotly_chart(fig, use_container_width=True)


def table(df, **kw):
    try: st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError: st.dataframe(df, use_container_width=True, hide_index=True, **kw)


# ----------------------------------------------------------------- aset & model
@st.cache_data(show_spinner=False)
def load_assets():
    a = {
        "res": pd.read_csv(ASSETS / "results_all.csv"),
        "abl": pd.read_csv(ASSETS / "ablation_all.csv"),
        "lead": pd.read_csv(ASSETS / "leaderboard.csv"),
        "meta": pd.read_csv(ASSETS / "dataset_meta.csv"),
        "oof": pd.read_csv(ASSETS / "oof_all.csv"),
        "selection": json.load(open(ASSETS / "selection.json")),
    }
    return a


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
    fig = go.Figure(go.Scatter(x=t, y=y[::step], mode="lines", line=dict(color=PRIMARY, width=1)))
    fig.update_layout(title="Gelombang suara", xaxis_title="detik", yaxis_title="amplitudo", showlegend=False)
    return fig


def fig_mel(y, sr=16000):
    import librosa
    M = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128), ref=np.max, top_db=80)
    fig = px.imshow(M, origin="lower", aspect="auto", color_continuous_scale="Magma",
                    labels=dict(x="frame", y="band mel", color="dB"))
    fig.update_layout(title="Spektrogram Log-Mel", coloraxis_showscale=False)
    return fig


def fig_probs(labels, p):
    order = np.argsort(p)
    names = [f"{EMO[labels[i]][1]} {EMO[labels[i]][0]}" for i in order]
    cols = [EMO[labels[i]][2] for i in order]
    fig = go.Figure(go.Bar(x=[float(p[i]) for i in order], y=names, orientation="h", marker_color=cols,
                           text=[f"{p[i]*100:.1f}%" for i in order], textposition="outside"))
    fig.update_layout(title="Probabilitas tiap emosi", xaxis=dict(range=[0, 1.12], tickformat=".0%"), showlegend=False)
    return fig


def fig_sentiment(sp):
    keys = ["negatif", "netral", "positif"]
    fig = go.Figure(go.Bar(x=keys, y=[sp[k] for k in keys], marker_color=[SENT_COLOR[k] for k in keys],
                           text=[f"{sp[k]*100:.1f}%" for k in keys], textposition="outside"))
    fig.update_layout(title="Sentimen 3 kelas (gabungan probabilitas)", yaxis=dict(range=[0, 1.15], tickformat=".0%"), showlegend=False)
    return fig
