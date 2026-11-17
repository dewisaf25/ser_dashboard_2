"""Inferensi bersama (notebook & dashboard): memuat model final + modul fitur yang sama dengan saat pelatihan."""
import os, sys, importlib, joblib, numpy as np

def load(model_dir="models", src_dir="src"):
    if os.path.abspath(src_dir) not in sys.path: sys.path.insert(0, os.path.abspath(src_dir))
    pk = os.path.join(model_dir, "preprocessor.joblib")
    if os.path.exists(pk):
        pre = joblib.load(pk)
        from tensorflow import keras
        pre["model"] = keras.models.load_model(os.path.join(model_dir, "model_final.keras"))
    else:
        pre = joblib.load(os.path.join(model_dir, "model_final.joblib"))
    pre["mod"] = importlib.import_module(pre["module"]); return pre

def predict(pre, path):
    mod, t = pre["mod"], pre["type"]
    vec, seq = mod.features_from_signal(mod.load_base(path, denoise=pre.get("denoise", False)))
    if t == "sklearn": p = pre["model"].predict_proba(pre["scaler"].transform(vec[None]))[0]
    elif t == "keras_vec": p = pre["model"].predict(pre["scaler"].transform(vec[None]), verbose=0)[0]
    elif t == "keras_seq": p = pre["model"].predict((seq[None] - pre["mean"]) / pre["std"], verbose=0)[0]
    elif t == "keras_img": p = pre["model"].predict(((seq - pre["mean"]) / pre["std"])[None, ..., None], verbose=0)[0]
    else: raise ValueError(t)
    return pre["labels"], np.asarray(p, dtype=float)
