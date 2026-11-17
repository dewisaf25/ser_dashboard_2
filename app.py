"""Dashboard Streamlit - SER IndoWaveSentiment (v2, tampilan dipercantik).
Jalankan:  streamlit run app.py   (dari dalam folder dashboard_package/)"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import app_utils as U

st.set_page_config(page_title="SER IndoWaveSentiment", page_icon="🎙️", layout="wide")
U.inject_css()

try:
    A = U.load_assets()
except FileNotFoundError as e:
    st.error(f"Aset tidak ditemukan: {e}. Pastikan folder `assets/` dari notebook ada di samping app.py.")
    st.stop()

SEL, RES, LEAD, META, OOF = A["selection"], A["res"], A["lead"], A["meta"], A["oof"]
MN = SEL["method_names"]
SHORT = {k: v.split(" (")[0] for k, v in MN.items()}
LABELS = sorted(U.EMO)
BASE = RES[RES.varian == "base"]
LEAD = LEAD.sort_values("f1", ascending=False).reset_index(drop=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    U.brand()
    page = st.radio("Menu", ["🏠 Beranda", "🎯 Prediksi", "📊 Performa Model", "🗂️ Dataset"], label_visibility="collapsed")
    st.write("")
    U.side_model(SHORT[SEL["metode"]], f"{SEL['lapis']} · {SEL['model']}", SEL["f1"], SEL["f1_std"])
    st.caption("Skor adalah hasil validasi pada 10 aktor; pada suara baru hasilnya bisa berbeda.")


# ================================================================== BERANDA
if page == "🏠 Beranda":
    U.hero("Speech Emotion Recognition",
           "Mengenali emosi dari suara berbahasa Indonesia (dataset IndoWaveSentiment) dengan membandingkan fitur MFCC, Log-Mel, dan Prosodik "
           "pada ML klasik, MLP, dan deep learning.",
           chips=[f"🎙️ {len(META)} rekaman", f"😊 {META.emotion.nunique()} emosi", f"👥 {META.actor.nunique()} aktor", "🧪 5-fold berbasis aktor"])
    c = st.columns(4)
    with c[0]: U.kpi("Fitur terbaik", SHORT[SEL["best_method"]], "rata-rata Lapis 1 + 2", "🏅")
    with c[1]: U.kpi("Model terpilih", SEL["model"], f"{SEL['lapis']} · {SHORT[SEL['metode']]}", "🤖")
    with c[2]: U.kpi("Macro-F1", f"{SEL['f1']:.3f}", f"± {SEL['f1_std']:.3f} (5 fold)", "🎯")
    with c[3]: U.kpi("Akurasi", f"{SEL['acc']:.3f}", "rata-rata 5 fold", "✅")

    U.section("Cara kerja", "dari suara mentah hingga label emosi")
    U.steps([("🎤", "Rekaman suara", "Audio .wav diubah ke 16 kHz mono."),
             ("✂️", "Pra-pemrosesan", "Senyap dipotong, lalu dipad atau dipotong menjadi 3 detik."),
             ("🧬", "Ekstraksi fitur", "MFCC, Log-Mel, atau fitur prosodik."),
             ("🧠", "Model klasifikasi", f"{SEL['model']} memberi probabilitas tiap emosi."),
             ("🎯", "Hasil", "Emosi, keyakinan, dan sentimen.")])

    U.section("Lima emosi yang dikenali", "gambaran umum ciri suara")
    c = st.columns(5)
    for col, l in zip(c, LABELS):
        n, e, colr = U.EMO[l]
        with col: U.card(n, U.EMO_DESC[l], icon=e, accent=colr, center=True)

    U.section("Tiga jenis fitur yang dibandingkan")
    c = st.columns(3)
    with c[0]: U.card("MFCC", "40 koefisien + delta + delta-delta, diringkas mean/std/min/max menjadi <b>480 fitur</b>.", icon="🎼", accent="#6366f1")
    with c[1]: U.card("Log-Mel", "Spektrogram Mel 128 band: vektor mean &amp; std (256 fitur) atau citra 128×188 untuk CNN 2D / MobileNetV2.", icon="🌈", accent="#ec4899")
    with c[2]: U.card("Prosodik", "ZCR, RMS, centroid, rolloff, contrast, chroma, dan F0 (pYIN) menjadi <b>63 fitur</b>.", icon="🎚️", accent="#f59e0b")

    U.section("Tiga model teratas", "berdasarkan macro-F1 rata-rata 5 fold")
    c = st.columns(3)
    medals, tones = ["🥇", "🥈", "🥉"], ["#f59e0b", "#94a3b8", "#b45309"]
    for k, (col, (_, r)) in enumerate(zip(c, LEAD.head(3).iterrows())):
        with col:
            U.card(f"{medals[k]} {r.model}", f"{SHORT[r.metode]} · {r.lapis}<br><b style='font-size:1.4rem'>F1 {r.f1:.3f}</b> ± {r.f1_std:.3f}<br>"
                   f"akurasi {r.acc:.3f} · inferensi {r.ms:.1f} ms", accent=tones[k])

    U.section("Perbandingan metode fitur", "rata-rata macro-F1 Lapis 1 + 2")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        g = BASE[BASE.lapis.isin(["ML", "MLP"])].groupby("metode").macro_f1.mean().reindex(list(MN)).dropna()
        fig = go.Figure(go.Bar(x=[SHORT[m] for m in g.index], y=g.values, marker_color=U.PALETTE[:len(g)], text=[f"{v:.3f}" for v in g.values],
                               textposition="outside", cliponaxis=False))
        fig.update_layout(title="Macro-F1 rata-rata per metode", yaxis=dict(range=[0, max(0.5, float(g.max()) * 1.25)]), showlegend=False)
        U.show(fig, 340)
    with c2:
        U.card("🧪 Protokol evaluasi",
               "• 300 rekaman, 5 emosi × 60 file, 10 aktor<br>• <b>5-fold berbasis aktor</b>: aktor uji tidak muncul di data latih<br>"
               "• Tuning hyperparameter hanya di data latih (GroupKFold)<br>• Metrik utama: <b>macro-F1</b>")
        st.write("")
        U.card("⚠️ Catatan penggunaan",
               "Model dilatih pada aktor yang terbatas dan rekaman studio pendek (~3 detik). Pada suara baru, dialek lain, atau lingkungan bising, "
               "hasilnya bisa lebih rendah dari skor validasi. Gunakan sebagai demonstrasi riset.")


# ================================================================== PREDIKSI
elif page == "🎯 Prediksi":
    U.hero("Prediksi Emosi", "Unggah, rekam, atau pilih contoh audio, lalu model akan menebak emosi dan sentimennya.",
           chips=["📁 Unggah file", "🎤 Rekam suara", "🎧 Contoh dataset"])
    try:
        pre = U.load_model()
    except Exception as e:
        st.error(f"Model gagal dimuat: {type(e).__name__}: {e}")
        st.info("Biasanya karena versi paket berbeda dari saat pelatihan. Pastikan Python 3.11 dan `requirements.txt` terpasang.")
        st.stop()

    t_up, t_rec, t_smp = st.tabs(["📁 Unggah file", "🎤 Rekam suara", "🎧 Contoh dataset"])
    data, suffix = None, ".wav"
    with t_up:
        f = st.file_uploader("Pilih file audio", type=["wav", "flac", "ogg"])
        if f is not None: data, suffix = f.getvalue(), "." + f.name.split(".")[-1].lower()
    with t_rec:
        if hasattr(st, "audio_input"):
            r = st.audio_input("Tekan untuk merekam (±3 detik, ucapkan dengan emosi)")
            if r is not None: data, suffix = r.getvalue(), ".wav"
        else:
            st.info("Versi Streamlit ini belum mendukung perekaman. Gunakan tab Unggah file.")
    with t_smp:
        samples = sorted((U.ASSETS / "samples").glob("*.wav"))
        if samples:
            pick = st.selectbox("Contoh", samples, format_func=lambda p: f"{U.EMO.get(p.stem, (p.stem,))[0]} ({p.stem})")
            if st.button("Gunakan contoh ini"): st.session_state["sample"] = str(pick)
            sp = st.session_state.get("sample")
            if sp and data is None: data = open(sp, "rb").read()
        else:
            st.caption("Tidak ada contoh audio di assets/samples.")

    if data is None:
        U.section("Cara memakai")
        U.steps([("🎤", "Siapkan suara", "Ucapkan satu kalimat pendek dengan emosi yang jelas."),
                 ("⬆️", "Unggah atau rekam", "Gunakan tab di atas."),
                 ("🧠", "Model menganalisis", "Fitur diekstrak lalu diklasifikasikan."),
                 ("🎯", "Lihat hasil", "Emosi, keyakinan, dan grafik suara.")])
        U.section("Tips agar hasil lebih akurat")
        c = st.columns(3)
        with c[0]: U.card("Ruangan tenang", "Kurangi bising latar seperti kipas, lalu lintas, atau musik.", icon="🔇")
        with c[1]: U.card("Durasi 2–4 detik", "Model dilatih pada klip sekitar 3 detik; klip terlalu panjang dipotong.", icon="⏱️")
        with c[2]: U.card("Ekspresikan emosi", "Emosi yang jelas dan natural lebih mudah dikenali daripada yang samar.", icon="🎭")
    else:
        st.audio(data)
        t0 = time.perf_counter()
        with st.spinner("Menganalisis suara…"):
            try:
                labels, p, y = U.analyze(pre, data, suffix)
            except Exception as e:
                st.error(f"Gagal memproses audio: {type(e).__name__}: {e}")
                st.stop()
        ms = (time.perf_counter() - t0) * 1000
        i = int(np.argmax(p)); lab = labels[i]; name, emo, color = U.EMO[lab]
        sp_ = U.sentiment_probs(labels, p); sent = max(sp_, key=sp_.get)
        U.section("Hasil prediksi")
        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.markdown(f'<div class="result" style="background:linear-gradient(135deg,{color},#1e1b4b)">'
                        f'<div class="emo">{emo}</div><div class="name">{name}</div>'
                        f'<div class="conf">keyakinan {p[i] * 100:.1f}%</div>'
                        f'<span class="pill" style="background:{U.SENT_COLOR[sent]}">{U.SENT_EMOJI[sent]} sentimen {sent}</span></div>', unsafe_allow_html=True)
            st.write("")
            U.card("Gambaran umum", U.EMO_DESC[lab], icon="💡")
            if p[i] < 0.5: st.warning("Keyakinan rendah (< 50%). Hasil ini kurang dapat diandalkan.")
            if len(y) / 16000 < 1.0: st.warning("Audio sangat pendek; hasil mungkin tidak stabil.")
        with c2: U.show(U.fig_probs(labels, p), 400)

        dur = len(y) / 16000; rms = 20 * np.log10(float(np.sqrt(np.mean(y ** 2))) + 1e-9); peak = float(np.max(np.abs(y)))
        c = st.columns(4)
        with c[0]: U.kpi("Durasi", f"{dur:.1f} dtk", "sebelum dipotong", "⏱️")
        with c[1]: U.kpi("Level suara", f"{rms:.0f} dB", "RMS rata-rata", "🔊")
        with c[2]: U.kpi("Puncak amplitudo", f"{peak:.2f}", "maks. 1.00", "📈")
        with c[3]: U.kpi("Waktu proses", f"{ms:.0f} ms", "ekstraksi + prediksi", "⚡")

        U.section("Analisis sinyal")
        c3, c4 = st.columns(2)
        with c3: U.show(U.fig_waveform(y), 300)
        with c4: U.show(U.fig_mel(y), 300)
        U.section("Rincian keyakinan")
        c5, c6, c7 = st.columns(3)
        with c5: U.show(U.fig_gauge(p[i], color), 300)
        with c6: U.show(U.fig_radar(labels, p), 300)
        with c7: U.show(U.fig_sentiment(sp_), 300)
        st.caption("Audio dipotong senyap (top_db=30) lalu dipad/dipotong 3 detik, sama persis dengan saat pelatihan.")


# ================================================================== PERFORMA
elif page == "📊 Performa Model":
    U.hero("Performa Model", "Hasil validasi silang 5 fold berbasis aktor untuk seluruh metode dan model.", chips=["📊 macro-F1", "🧪 tanpa kebocoran aktor"])
    fast = LEAD.sort_values("ms").iloc[0]
    c = st.columns(4)
    with c[0]: U.kpi("Kombinasi dievaluasi", len(LEAD), "metode × model", "🧮")
    with c[1]: U.kpi("F1 tertinggi", f"{LEAD.f1.max():.3f}", f"{LEAD.iloc[0].model} · {SHORT[LEAD.iloc[0].metode]}", "🏆")
    with c[2]: U.kpi("F1 terendah", f"{LEAD.f1.min():.3f}", "batas bawah", "📉")
    with c[3]: U.kpi("Inferensi tercepat", f"{fast.ms:.1f} ms", f"{fast.model} · {SHORT[fast.metode]}", "⚡")
    st.write("")
    t1, t2, t3, t4, t5 = st.tabs(["🏆 Peringkat", "🔥 Fitur × Model", "🎯 Model terpilih", "📈 Per fold", "⚡ Efisiensi"])

    with t1:
        d = LEAD.copy(); d["nama"] = d.metode.map(SHORT) + " · " + d.model; d = d.sort_values("f1")
        sel = [(m == SEL["metode"] and mo == SEL["model"] and la == SEL["lapis"]) for m, mo, la in zip(d.metode, d.model, d.lapis)]
        fig = go.Figure(go.Bar(x=d.f1, y=d.nama, orientation="h", error_x=dict(type="data", array=d.f1_std, color="rgba(127,127,127,.8)"),
                               marker_color=["#4f46e5" if s else "#a5b4fc" for s in sel], text=[f"{v:.3f}" for v in d.f1],
                               textposition="inside", insidetextanchor="start", textfont=dict(color="#1e1b4b")))
        fig.update_layout(title="Macro-F1 (rata-rata ± std 5 fold); model terpilih berwarna tua", xaxis=dict(range=[0, 1.0]))
        U.show(fig, max(400, 30 * len(d) + 120))
        U.section("Tabel lengkap")
        U.table(LEAD.assign(metode=LEAD.metode.map(SHORT)).rename(columns={"f1": "macro-F1", "f1_std": "std", "acc": "akurasi", "ms": "inferensi (ms)"}).round(3))

    with t2:
        g = BASE[BASE.lapis.isin(["ML", "MLP"])].groupby(["model", "metode"]).macro_f1.mean().reset_index()
        pv = g.pivot(index="model", columns="metode", values="macro_f1")
        pv = pv.reindex([m for m in ["SVM", "RF", "XGBoost", "MLP"] if m in pv.index])[[m for m in MN if m in pv.columns]]
        c1, c2 = st.columns(2)
        with c1:
            fig = px.imshow(pv.values, x=[SHORT[c_] for c_ in pv.columns], y=list(pv.index), text_auto=".2f", color_continuous_scale="Viridis", zmin=0, zmax=1, aspect="auto")
            fig.update_layout(title="Heatmap Lapis 1 & 2: perbandingan fitur paling bersih", coloraxis_showscale=False)
            U.show(fig, 380)
        with c2:
            fig = go.Figure()
            for k, m in enumerate(pv.columns):
                fig.add_bar(name=SHORT[m], x=list(pv.index), y=pv[m].values, marker_color=U.PALETTE[k])
            fig.update_layout(barmode="group", title="Macro-F1 per model, dikelompokkan menurut metode", yaxis=dict(range=[0, 1]))
            U.show(fig, 380)
        l3 = BASE[BASE.lapis == "DL-khusus"].groupby(["metode", "model"]).agg(
            akurasi=("accuracy", "mean"), macro_F1=("macro_f1", "mean"), std=("macro_f1", "std"), inferensi_ms=("infer_time_ms", "mean")).round(3).reset_index()
        if len(l3):
            U.section("Lapis 3: deep learning khusus", "kombinasi fitur + arsitektur, bukan murni perbandingan fitur")
            l3["metode"] = l3.metode.map(SHORT); U.table(l3)

    with t3:
        key = SEL["key"]
        if key not in OOF.columns:
            st.warning(f"Kolom prediksi `{key}` tidak ada di oof_all.csv.")
        else:
            o = OOF[["filename", "y_true", key]].merge(META, on="filename")
            yt, yp = o.y_true.values, o[key].values
            sm = lambda s: s.map(U.SENT_MAP)
            c = st.columns(3)
            with c[0]: U.kpi("Akurasi (gabungan OOF)", f"{accuracy_score(yt, yp):.3f}", icon="✅")
            with c[1]: U.kpi("Macro-F1 (gabungan OOF)", f"{f1_score(yt, yp, average='macro'):.3f}", icon="🎯")
            with c[2]: U.kpi("Akurasi sentimen 3 kelas", f"{accuracy_score(sm(pd.Series(yt)), sm(pd.Series(yp))):.3f}", "pelengkap", "🙂")
            st.write("")
            c1, c2 = st.columns(2)
            nm = [U.EMO[l][0] for l in LABELS]
            with c1:
                cm = confusion_matrix(yt, yp, labels=LABELS)
                fig = px.imshow(cm, x=nm, y=nm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Prediksi", y="Sebenarnya"))
                fig.update_layout(title="Confusion matrix", coloraxis_showscale=False)
                U.show(fig, 430)
            with c2:
                f1c = f1_score(yt, yp, labels=LABELS, average=None)
                fig = go.Figure(go.Bar(x=nm, y=f1c, marker_color=[U.EMO[l][2] for l in LABELS], text=[f"{v:.2f}" for v in f1c], textposition="outside", cliponaxis=False))
                fig.update_layout(title="F1 per kelas", yaxis=dict(range=[0, 1.15]))
                U.show(fig, 430)
            rows = []
            for col in ("gender", "intensity"):
                for gv in o[col].unique():
                    m = (o[col] == gv).values
                    rows.append({"kelompok": col, "nilai": gv, "akurasi": round(accuracy_score(yt[m], yp[m]), 3), "n": int(m.sum())})
            U.section("Akurasi per kelompok"); U.table(pd.DataFrame(rows))
            off = sorted([(nm[i], nm[j], int(cm[i, j])) for i in range(5) for j in range(5) if i != j], key=lambda t: -t[2])[:3]
            st.info("Kesalahan terbanyak: " + "; ".join(f"{a} → diprediksi {b} ({n}×)" for a, b, n in off))

    with t4:
        c = st.columns(3)
        mk = c[0].selectbox("Metode", list(MN), index=list(MN).index(SEL["metode"]), format_func=lambda k: SHORT[k])
        sub = BASE[BASE.metode == mk]
        lp = c[1].selectbox("Lapis", sorted(sub.lapis.unique()))
        sub = sub[sub.lapis == lp]
        md = c[2].selectbox("Model", sorted(sub.model.unique()))
        s = sub[sub.model == md].sort_values("fold")
        fig = go.Figure(go.Bar(x=[f"Fold {k}" for k in s.fold], y=s.macro_f1, marker_color=U.PRIMARY, text=[f"{v:.2f}" for v in s.macro_f1], textposition="outside", cliponaxis=False))
        fig.add_hline(y=s.macro_f1.mean(), line_dash="dash", line_color="#ec4899", annotation_text=f"rata-rata {s.macro_f1.mean():.3f}")
        fig.update_layout(title="Macro-F1 per fold (tiap fold = sepasang aktor uji)", yaxis=dict(range=[0, 1.15]))
        U.show(fig, 380)
        U.section("Sebaran antar fold untuk semua model", "kotak sempit = stabil")
        b = BASE.copy(); b["nama"] = b.metode.map(SHORT) + " · " + b.model
        fig = px.box(b, x="nama", y="macro_f1", color="metode", points="all", color_discrete_sequence=U.PALETTE, labels=dict(nama="", macro_f1="macro-F1"))
        fig.update_layout(showlegend=False, xaxis_tickangle=-35, title="Sebaran macro-F1 per fold")
        U.show(fig, 460)
        st.caption("Selisih antar model yang lebih kecil dari 1 std tidak bermakna dengan hanya 5 fold.")

    with t5:
        d = LEAD.copy(); d["nama"] = d.metode.map(SHORT) + " · " + d.model; d["Metode"] = d.metode.map(SHORT)
        fig = px.scatter(d, x="ms", y="f1", color="Metode", hover_name="nama", log_x=True, color_discrete_sequence=U.PALETTE, error_y="f1_std",
                         labels=dict(ms="waktu inferensi per sampel (ms, skala log)", f1="macro-F1"))
        fig.update_traces(marker=dict(size=13, line=dict(width=1, color="white")))
        fig.update_layout(title="Akurasi vs kecepatan: kiri-atas = akurat dan cepat")
        U.show(fig, 470)
        U.card("Cara membaca", "Model di pojok kiri-atas paling ideal: macro-F1 tinggi dengan waktu inferensi rendah. "
               "Jika dua model hampir sama skornya, pilih yang lebih ringan dan lebih stabil.", icon="🧭")


# ================================================================== DATASET
else:
    U.hero("Dataset IndoWaveSentiment", "300 rekaman, 5 emosi, 10 aktor (5 laki-laki, 5 perempuan).", chips=["🎧 .wav 16 kHz", "⏱️ ±3 detik", "🧑‍🤝‍🧑 5 L + 5 P"])
    c = st.columns(4)
    with c[0]: U.kpi("Rekaman", len(META), icon="🎙️")
    with c[1]: U.kpi("Emosi", META.emotion.nunique(), icon="😊")
    with c[2]: U.kpi("Aktor", META.actor.nunique(), icon="👥")
    with c[3]: U.kpi("Fold", META.fold.nunique(), "pasangan 1 L + 1 P", "🧩")
    U.section("Sebaran data")
    c1, c2, c3 = st.columns(3)
    with c1:
        d = META.emotion.value_counts().reindex(LABELS).reset_index(); d.columns = ["emosi", "jumlah"]; d["nama"] = d.emosi.map(lambda e: U.EMO[e][0])
        fig = px.bar(d, x="nama", y="jumlah", color="emosi", color_discrete_map={e: U.EMO[e][2] for e in LABELS}, text="jumlah")
        fig.update_layout(title="Rekaman per emosi", showlegend=False, xaxis_title="", yaxis_title="")
        U.show(fig, 340)
    with c2:
        d = META.gender.map({"L": "Laki-laki", "P": "Perempuan"}).value_counts().reset_index(); d.columns = ["gender", "n"]
        fig = px.pie(d, names="gender", values="n", hole=0.55, color_discrete_sequence=["#6366f1", "#ec4899"])
        fig.update_layout(title="Gender"); fig.update_traces(textinfo="percent+label")
        U.show(fig, 340)
    with c3:
        d = META.intensity.value_counts().reset_index(); d.columns = ["intensitas", "n"]
        fig = px.pie(d, names="intensitas", values="n", hole=0.55, color_discrete_sequence=["#f59e0b", "#10b981"])
        fig.update_layout(title="Intensitas"); fig.update_traces(textinfo="percent+label")
        U.show(fig, 340)
    c4, c5 = st.columns(2)
    with c4:
        d = META.groupby(["fold", "gender"]).size().reset_index(name="n"); d["fold"] = "Fold " + d.fold.astype(str)
        fig = px.bar(d, x="fold", y="n", color="gender", barmode="group", title="Komposisi gender per fold", color_discrete_map={"L": "#6366f1", "P": "#ec4899"})
        U.show(fig, 330)
    with c5:
        d = META.groupby(["emotion", "intensity"]).size().reset_index(name="n"); d["emosi"] = d.emotion.map(lambda e: U.EMO[e][0])
        fig = px.bar(d, x="emosi", y="n", color="intensity", barmode="stack", title="Intensitas per emosi", color_discrete_sequence=["#f59e0b", "#10b981"])
        U.show(fig, 330)
    U.section("Contoh audio", "satu rekaman untuk tiap emosi")
    samples = {p.stem: p for p in sorted((U.ASSETS / "samples").glob("*.wav"))}
    if samples:
        cols = st.columns(len(samples))
        for col, (k, p) in zip(cols, samples.items()):
            with col:
                st.markdown(f"**{U.EMO.get(k, ('', ''))[1]} {U.EMO.get(k, (k,))[0]}**"); st.audio(str(p))
    with st.expander("Lihat metadata & pembagian fold"):
        U.table(META)

U.footer()
