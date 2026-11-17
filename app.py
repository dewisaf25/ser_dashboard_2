"""Dashboard Streamlit - SER IndoWaveSentiment.
Jalankan:  streamlit run app.py   (dari dalam folder dashboard_package/)"""
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

SEL, RES, ABL, LEAD, META, OOF = A["selection"], A["res"], A["abl"], A["lead"], A["meta"], A["oof"]
MN = SEL["method_names"]
SHORT = {k: v.split(" (")[0] for k, v in MN.items()}
LABELS = sorted(U.EMO)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🎙️ SER Indo")
    st.caption("Klasifikasi emosi suara berbahasa Indonesia")
    page = st.radio("Menu", ["🏠 Beranda", "🎯 Prediksi", "📊 Performa Model", "🧪 Ablasi", "🗂️ Dataset"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Model aktif**")
    st.markdown(f"{SHORT[SEL['metode']]}  \n`{SEL['lapis']} · {SEL['model']}`")
    st.metric("Macro-F1 (5 fold)", f"{SEL['f1']:.3f}", f"± {SEL['f1_std']:.3f}", delta_color="off")


# ================================================================== BERANDA
if page == "🏠 Beranda":
    U.hero("Speech Emotion Recognition", "Mengenali emosi dari suara berbahasa Indonesia (dataset IndoWaveSentiment) — "
           "membandingkan fitur MFCC, Log-Mel, dan Prosodik dengan ML klasik, MLP, dan deep learning.")
    c = st.columns(4)
    with c[0]: U.kpi("Fitur terbaik", SHORT[SEL["best_method"]], "rata-rata Lapis 1 + 2")
    with c[1]: U.kpi("Model terpilih", SEL["model"], f"{SEL['lapis']} · {SHORT[SEL['metode']]}")
    with c[2]: U.kpi("Macro-F1", f"{SEL['f1']:.3f}", f"± {SEL['f1_std']:.3f} (5 fold)")
    with c[3]: U.kpi("Akurasi", f"{SEL['acc']:.3f}", "rata-rata 5 fold")
    st.write("")
    c = st.columns(3)
    with c[0]: U.card("🎼 MFCC", "40 koefisien + delta + delta-delta, diringkas mean/std/min/max → <b>480 fitur</b>.")
    with c[1]: U.card("🌈 Log-Mel", "Spektrogram Mel 128 band; vektor mean &amp; std (256 fitur) atau gambar 128×188 untuk CNN 2D / MobileNetV2.")
    with c[2]: U.card("🎚️ Prosodik", "ZCR, RMS, centroid, rolloff, contrast, chroma, dan F0 (pYIN) → <b>63 fitur</b>.")
    st.write("")
    c = st.columns(2)
    with c[0]:
        U.card("🧪 Protokol evaluasi",
               "• 300 rekaman, 5 emosi × 60 file, 10 aktor<br>• <b>5-fold berbasis aktor</b>: aktor uji tidak pernah muncul di data latih (tanpa kebocoran)<br>"
               "• Tuning hyperparameter hanya di data latih (GroupKFold)<br>• Metrik utama: <b>macro-F1</b>")
    with c[1]:
        U.card("⚠️ Catatan penggunaan",
               "Model dilatih pada <b>aktor terbatas</b> dan rekaman studio pendek (~3 detik). Hasil pada suara baru, "
               "dialek lain, atau lingkungan bising bisa jauh lebih rendah daripada skor validasi. Gunakan sebagai demonstrasi riset.")


# ================================================================== PREDIKSI
elif page == "🎯 Prediksi":
    U.hero("Prediksi Emosi", "Unggah, rekam, atau pilih contoh audio — model akan menebak emosinya.")
    try:
        pre = U.load_model()
    except Exception as e:
        st.error(f"Model gagal dimuat: {type(e).__name__}: {e}")
        st.info("Biasanya karena versi paket berbeda dari saat pelatihan. Pasang ulang dengan `pip install -r requirements.txt`.")
        st.stop()

    t_up, t_rec, t_smp = st.tabs(["📁 Unggah file", "🎤 Rekam suara", "🎧 Contoh dataset"])
    data, suffix, src_name = None, ".wav", ""
    with t_up:
        f = st.file_uploader("Pilih file audio", type=["wav", "flac", "ogg"])
        if f is not None:
            data, suffix, src_name = f.getvalue(), "." + f.name.split(".")[-1].lower(), f.name
    with t_rec:
        if hasattr(st, "audio_input"):
            r = st.audio_input("Tekan untuk merekam (±3 detik, ucapkan dengan emosi)")
            if r is not None: data, suffix, src_name = r.getvalue(), ".wav", "rekaman"
        else:
            st.info("Versi Streamlit ini belum mendukung perekaman. Perbarui: `pip install -U streamlit`.")
    with t_smp:
        samples = sorted((U.ASSETS / "samples").glob("*.wav"))
        if samples:
            pick = st.selectbox("Contoh", samples, format_func=lambda p: f"{U.EMO.get(p.stem, (p.stem,))[0]} ({p.stem})")
            if st.button("Gunakan contoh ini"):
                st.session_state["sample"] = str(pick)
            sp = st.session_state.get("sample")
            if sp and data is None:
                data, suffix, src_name = open(sp, "rb").read(), ".wav", f"contoh: {sp.split('/')[-1].split(chr(92))[-1]}"
        else:
            st.caption("Tidak ada contoh audio di assets/samples.")

    if data is None:
        st.info("Pilih salah satu sumber audio di atas untuk memulai.")
    else:
        st.audio(data)
        with st.spinner("Menganalisis suara…"):
            try:
                labels, p, y = U.analyze(pre, data, suffix)
            except Exception as e:
                st.error(f"Gagal memproses audio: {type(e).__name__}: {e}")
                st.stop()
        i = int(np.argmax(p)); lab = labels[i]; name, emo, color = U.EMO[lab]
        sp_ = U.sentiment_probs(labels, p); sent = max(sp_, key=sp_.get)
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.markdown(f'<div class="result" style="background:linear-gradient(135deg,{color},#1e1b4b)">'
                        f'<div class="emo">{emo}</div><div class="name">{name}</div>'
                        f'<div class="conf">keyakinan {p[i]*100:.1f}%</div>'
                        f'<span class="pill" style="background:{U.SENT_COLOR[sent]}">{U.SENT_EMOJI[sent]} sentimen {sent}</span></div>',
                        unsafe_allow_html=True)
            if p[i] < 0.5: st.warning("Keyakinan rendah (< 50%). Hasil ini kurang dapat diandalkan.")
            if len(y) / 16000 < 1.0: st.warning("Audio sangat pendek; hasil mungkin tidak stabil.")
        with c2: U.show(U.fig_probs(labels, p), 330)
        c3, c4 = st.columns(2)
        with c3: U.show(U.fig_waveform(y), 280)
        with c4: U.show(U.fig_mel(y), 280)
        U.show(U.fig_sentiment(sp_), 280)
        st.caption("Audio dipotong senyap (top_db=30) lalu dipad/dipotong 3 detik, sama persis dengan saat pelatihan.")


# ================================================================== PERFORMA
elif page == "📊 Performa Model":
    U.hero("Performa Model", "Hasil validasi silang 5 fold berbasis aktor untuk seluruh metode dan model.")
    base = RES[RES.varian == "base"]
    t1, t2, t3, t4 = st.tabs(["🏆 Peringkat", "🔥 Fitur × Model", "🎯 Model terpilih", "📈 Per fold"])

    with t1:
        d = LEAD.copy()
        d["nama"] = d.metode.map(SHORT) + " · " + d.model
        d = d.sort_values("f1")
        fig = go.Figure(go.Bar(x=d.f1, y=d.nama, orientation="h", error_x=dict(type="data", array=d.f1_std),
                               marker_color=[U.PRIMARY if (m == SEL["metode"] and mo == SEL["model"] and la == SEL["lapis"]) else "#a5b4fc"
                                             for m, mo, la in zip(d.metode, d.model, d.lapis)],
                               text=[f"{v:.3f}" for v in d.f1], textposition="outside"))
        fig.update_layout(title="Macro-F1 (rata-rata ± std 5 fold) — model terpilih berwarna tua", xaxis=dict(range=[0, 1.05]))
        U.show(fig, max(380, 28 * len(d) + 120))
        show = LEAD.assign(metode=LEAD.metode.map(SHORT)).rename(columns={"f1": "macro-F1", "f1_std": "std", "acc": "akurasi", "ms": "inferensi (ms)"})
        U.table(show.round(3))

    with t2:
        g = base[base.lapis.isin(["ML", "MLP"])].groupby(["model", "metode"]).macro_f1.mean().reset_index()
        pv = g.pivot(index="model", columns="metode", values="macro_f1")
        pv = pv.reindex([m for m in ["SVM", "RF", "XGBoost", "MLP"] if m in pv.index])[[m for m in MN if m in pv.columns]]
        fig = px.imshow(pv.values, x=[SHORT[c] for c in pv.columns], y=list(pv.index), text_auto=".2f",
                        color_continuous_scale="Viridis", zmin=0, zmax=1, aspect="auto")
        fig.update_layout(title="Macro-F1 Lapis 1 & 2 — perbandingan fitur yang paling bersih")
        U.show(fig, 360)
        l3 = base[base.lapis == "DL-khusus"].groupby(["metode", "model"]).agg(
            akurasi=("accuracy", "mean"), macro_F1=("macro_f1", "mean"), std=("macro_f1", "std"), inferensi_ms=("infer_time_ms", "mean")).round(3).reset_index()
        if len(l3):
            st.markdown("**Lapis 3 — deep learning khusus** (gabungan fitur + arsitektur, bukan murni perbandingan fitur)")
            l3["metode"] = l3.metode.map(SHORT); U.table(l3)

    with t3:
        key = SEL["key"]
        if key not in OOF.columns:
            st.warning(f"Kolom prediksi `{key}` tidak ada di oof_all.csv.")
        else:
            o = OOF[["filename", "y_true", key]].merge(META, on="filename")
            yt, yp = o.y_true.values, o[key].values
            c = st.columns(3)
            with c[0]: U.kpi("Akurasi (gabungan OOF)", f"{accuracy_score(yt, yp):.3f}")
            with c[1]: U.kpi("Macro-F1 (gabungan OOF)", f"{f1_score(yt, yp, average='macro'):.3f}")
            sm = lambda s: s.map(U.SENT_MAP)
            with c[2]: U.kpi("Akurasi sentimen 3 kelas", f"{accuracy_score(sm(pd.Series(yt)), sm(pd.Series(yp))):.3f}", "pelengkap")
            st.write("")
            c1, c2 = st.columns(2)
            with c1:
                cm = confusion_matrix(yt, yp, labels=LABELS)
                nm = [U.EMO[l][0] for l in LABELS]
                fig = px.imshow(cm, x=nm, y=nm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Prediksi", y="Sebenarnya"))
                fig.update_layout(title="Confusion matrix", coloraxis_showscale=False)
                U.show(fig, 420)
            with c2:
                f1c = f1_score(yt, yp, labels=LABELS, average=None)
                fig = go.Figure(go.Bar(x=nm, y=f1c, marker_color=[U.EMO[l][2] for l in LABELS], text=[f"{v:.2f}" for v in f1c], textposition="outside"))
                fig.update_layout(title="F1 per kelas", yaxis=dict(range=[0, 1.1]))
                U.show(fig, 420)
            rows = []
            for col in ("gender", "intensity"):
                for gv in o[col].unique():
                    m = (o[col] == gv).values
                    rows.append({"kelompok": col, "nilai": gv, "akurasi": round(accuracy_score(yt[m], yp[m]), 3), "n": int(m.sum())})
            st.markdown("**Akurasi per kelompok**"); U.table(pd.DataFrame(rows))
            off = sorted([(nm[i], nm[j], int(cm[i, j])) for i in range(5) for j in range(5) if i != j], key=lambda t: -t[2])[:3]
            st.info("Kesalahan terbanyak: " + "; ".join(f"{a} → diprediksi {b} ({n}×)" for a, b, n in off))

    with t4:
        c = st.columns(3)
        mk = c[0].selectbox("Metode", list(MN), index=list(MN).index(SEL["metode"]), format_func=lambda k: SHORT[k])
        sub = base[base.metode == mk]
        lp = c[1].selectbox("Lapis", sorted(sub.lapis.unique()))
        sub = sub[sub.lapis == lp]
        md = c[2].selectbox("Model", sorted(sub.model.unique()))
        s = sub[sub.model == md].sort_values("fold")
        fig = go.Figure(go.Bar(x=[f"Fold {k}" for k in s.fold], y=s.macro_f1, marker_color=U.PRIMARY, text=[f"{v:.2f}" for v in s.macro_f1], textposition="outside"))
        fig.add_hline(y=s.macro_f1.mean(), line_dash="dash", annotation_text=f"rata-rata {s.macro_f1.mean():.3f}")
        fig.update_layout(title="Macro-F1 per fold (tiap fold = sepasang aktor uji)", yaxis=dict(range=[0, 1.1]))
        U.show(fig, 380)
        st.caption("Selisih antar model yang lebih kecil dari 1 std tidak bermakna dengan hanya 5 fold.")


# ================================================================== ABLASI
elif page == "🧪 Ablasi":
    U.hero("Ablasi", "Apakah denoising (MFCC/Log-Mel) atau augmentasi (Prosodik) benar-benar membantu?")
    rows = []
    for m in MN:
        a = ABL[ABL.metode == m]
        for (lap, mod), gg in a.groupby(["lapis", "model"]):
            b = gg[gg.varian == "base"].set_index("fold").macro_f1
            for v in [x for x in gg.varian.unique() if x != "base"]:
                vv = gg[gg.varian == v].set_index("fold").macro_f1; d = vv - b
                rows.append(dict(metode=SHORT[m], model=mod, varian=v, f1_base=round(b.mean(), 3), f1_varian=round(vv.mean(), 3),
                                 selisih=round(d.mean(), 3), std_base=round(b.std(), 3), fold_membaik=f"{int((d > 0).sum())}/{len(d)}"))
    ab = pd.DataFrame(rows)
    if ab.empty:
        st.info("Belum ada hasil ablasi.")
    else:
        ab["label"] = ab.metode + " · " + ab.model + " (" + ab.varian + ")"
        fig = go.Figure()
        fig.add_bar(name="base", x=ab.label, y=ab.f1_base, marker_color="#a5b4fc")
        fig.add_bar(name="varian", x=ab.label, y=ab.f1_varian, marker_color=U.PRIMARY)
        fig.update_layout(barmode="group", title="Macro-F1: base vs varian", yaxis=dict(range=[0, 1]), xaxis_tickangle=-25)
        U.show(fig, 420)
        U.table(ab.drop(columns="label"))
        sig = ab[ab.selisih.abs() >= ab.std_base]
        if len(sig): st.success("Perubahan yang melebihi 1 std: " + ", ".join(sig.label))
        else: st.warning("Tidak ada selisih yang melebihi 1 std fold — jangan mengklaim 'lebih baik' hanya dari satu angka.")


# ================================================================== DATASET
else:
    U.hero("Dataset IndoWaveSentiment", "300 rekaman, 5 emosi, 10 aktor (5 laki-laki, 5 perempuan).")
    c = st.columns(4)
    with c[0]: U.kpi("Rekaman", len(META))
    with c[1]: U.kpi("Emosi", META.emotion.nunique())
    with c[2]: U.kpi("Aktor", META.actor.nunique())
    with c[3]: U.kpi("Fold", META.fold.nunique(), "pasangan 1 L + 1 P")
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        d = META.emotion.value_counts().reindex(LABELS).reset_index(); d.columns = ["emosi", "jumlah"]
        d["nama"] = d.emosi.map(lambda e: U.EMO[e][0])
        fig = px.bar(d, x="nama", y="jumlah", color="emosi", color_discrete_map={e: U.EMO[e][2] for e in LABELS}, text="jumlah")
        fig.update_layout(title="Jumlah rekaman per emosi", showlegend=False)
        U.show(fig, 340)
    with c2:
        d = META.groupby(["fold", "gender"]).size().reset_index(name="n")
        d["fold"] = "Fold " + d.fold.astype(str)
        fig = px.bar(d, x="fold", y="n", color="gender", barmode="group", title="Komposisi gender per fold",
                     color_discrete_map={"L": "#6366f1", "P": "#ec4899"})
        U.show(fig, 340)
    d = META.groupby(["emotion", "intensity"]).size().reset_index(name="n")
    d["emosi"] = d.emotion.map(lambda e: U.EMO[e][0])
    fig = px.bar(d, x="emosi", y="n", color="intensity", barmode="stack", title="Intensitas per emosi")
    U.show(fig, 320)
    st.markdown("**Contoh audio**")
    samples = {p.stem: p for p in sorted((U.ASSETS / "samples").glob("*.wav"))}
    if samples:
        cols = st.columns(len(samples))
        for col, (k, p) in zip(cols, samples.items()):
            with col:
                st.markdown(f"{U.EMO.get(k, ('', ''))[1]} **{U.EMO.get(k, (k,))[0]}**"); st.audio(str(p))
    with st.expander("Lihat metadata & pembagian fold"):
        U.table(META)
