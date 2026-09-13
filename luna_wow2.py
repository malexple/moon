#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wow&flutter v2: матрица когерентности ЧИСТЫХ тонов + исправленный спектр модуляции."""
import argparse
import numpy as np
from scipy.signal import stft, butter, sosfilt, hilbert, resample_poly
from scipy.io import wavfile
from math import gcd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 16000
ENV_SR = 250
CLEAN = [796.9, 996.1, 1093.8]      # чистые линии пьедестала
SUSPECT = 2550.0

def load_wav(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64) / max(1.0, np.abs(x).max())
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g)
    return x

def freq_track(x, center, bw=12):
    """Узкая полоса + нормировка аналитического сигнала (устойчиво к AM)."""
    sos = butter(4, [(center-bw)/(SR/2), (center+bw)/(SR/2)],
                 btype="band", output="sos")
    y = sosfilt(sos, x)
    z = hilbert(y)
    z = z / (np.abs(z) + 1e-12)                 # убираем влияние AM на фазу
    f = np.diff(np.unwrap(np.angle(z))) / (2*np.pi) * SR
    k = SR // 100
    tr = f[::k]
    tr = np.clip(tr, center*0.99, center*1.01)  # чистим выбросы жёстче
    return (tr - center) / center

def env_track(x, center, bw=150):
    sos = butter(4, [(center-bw)/(SR/2), (center+bw)/(SR/2)],
                 btype="band", output="sos")
    y = sosfilt(sos, x)
    f, t, S = stft(y, fs=SR, nperseg=512, noverlap=448)
    return np.abs(S).sum(axis=0)

def mod_spectrum(env, t0min, t1min):
    e = env[int(t0min*60*ENV_SR):int(t1min*60*ENV_SR)]   # <-- ИСПРАВЛЕНО ×60
    e = e - e.mean()
    nperseg = min(len(e) - 1, 8192)                       # 32.8 с, разр. 0.03 Гц
    fe, te, Se = stft(e, fs=ENV_SR, nperseg=nperseg, noverlap=nperseg*3//4)
    Se = np.abs(Se).mean(axis=1)
    b = (fe > 0.03) & (fe < 3)
    idx = np.argsort(Se[b])[::-1][:5]
    return [(round(float(v), 3), round(1/float(v), 1)) for v in fe[b][idx]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--t0", type=float, default=43.0)
    ap.add_argument("--t1", type=float, default=44.0)
    ap.add_argument("--out", default="wow2")
    a = ap.parse_args()
    x = load_wav(a.wav)
    seg = x[int(a.t0*60*SR):int(a.t1*60*SR)]

    tracks = {c: freq_track(seg, c) for c in CLEAN + [SUSPECT]}
    names = CLEAN + [SUSPECT]
    print("[матрица корреляций Δf/f] (окно %.0f-%.0f мин):" % (a.t0, a.t1))
    for i in range(len(names)):
        row = []
        for j in range(len(names)):
            n = min(len(tracks[names[i]]), len(tracks[names[j]]))
            row.append(np.corrcoef(tracks[names[i]][:n],
                                   tracks[names[j]][:n])[0, 1])
        print("  " + "  ".join(f"{v:+.2f}" for v in row))
    print("  порядок:", names)
    cc = np.corrcoef(tracks[CLEAN[0]], tracks[CLEAN[1]])[0, 1]
    cs = np.corrcoef(tracks[CLEAN[0]], tracks[SUSPECT])[0, 1]
    print(f"\nчистые тона между собой: {cc:.2f} | чистый тон vs 2550: {cs:.2f}")
    if cc > 0.7 and cs < 0.3:
        print("-> wow&flutter ЕСТЬ на чистых тонах, но у 2550 СВОЁ дыхание:")
        print("   дыхание свиста = динамика ИСТОЧНИКА (биения передатчиков)")
    elif cc > 0.7 and cs > 0.5:
        print("-> всё дрожит когерентно: дыхание = механика ленты")
    else:
        print("-> когерентности нет даже у чистых тонов: дрожание электрическое/оценочное")

    env_full = env_track(x, SUSPECT)
    print("\n[спектр модуляции 2550 Гц, честно] топ-5 (Гц, период с):")
    for w in [(20, 24), (43, 47), (80, 84)]:
        print(f"  окно {w[0]}-{w[1]} мин:", mod_spectrum(env_full, *w))

if __name__ == "__main__":
    main()