"""Fitur MFCC. Dipakai saat pelatihan DAN di dashboard (hindari training-serving skew)."""
import numpy as np, librosa
SR, DUR, TOP_DB = 16000, 3.0, 30
N_MFCC, NFFT, HOP = 40, 1024, 512

def pad_crop(y):
    n = int(SR * DUR)
    return np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32)

def base_signal(y):                       # LANGKAH DASAR (identik): trim silence top_db=30, pad/crop 3 detik
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)
    return pad_crop(y)

def load_base(path, denoise=False):
    y = librosa.load(path, sr=SR, mono=True)[0]                       # 16 kHz, mono
    if denoise:                           # ablasi denoising (butuh: pip install noisereduce)
        import noisereduce as nr
        y = nr.reduce_noise(y=y, sr=SR)
    return base_signal(y)

def vec_seq_from_path(path, denoise=False):
    return features_from_signal(load_base(path, denoise))

def extract(wav_path, denoise=False):     # KONTRAK: extract(wav_path) -> np.ndarray (belum di-scale)
    return vec_seq_from_path(wav_path, denoise)[0]

def features_from_signal(y):
    mf = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=NFFT, hop_length=HOP)
    A = np.vstack([mf, librosa.feature.delta(mf), librosa.feature.delta(mf, order=2)])      # (120, T)
    vec = np.concatenate([A.mean(1), A.std(1), A.min(1), A.max(1)]).astype(np.float32)      # mean, std, min, max -> 480
    return vec, A.T.astype(np.float32)                                                      # seq: (T, 120)

def vec_names():
    rows = [f"mfcc{i}" for i in range(40)] + [f"d{i}" for i in range(40)] + [f"dd{i}" for i in range(40)]
    return [f"{r}_{s}" for s in ("mean", "std", "min", "max") for r in rows]
