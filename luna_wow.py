#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тест wow&flutter: когерентно ли дрожание частоты разных тонов записи?"""
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

def load_wav(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64) / max(1.0, np.abs(x).max())
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g)
    return x

def freq_track(x, center, bw=25, out_sr=100):
    """Относительное отклонение частоты линии (Δf/f) во времени."""
    sos = butter(4, [(center-bw)/(SR/2), (center+bw)/(SR/2)],
                 btype="band", output="sos")
    y = sosfilt(sos, x)
    ph = np.unwrap(np.angle(hilbert(y)))
    f = np.diff(ph) / (2*np.pi) * SR
    k = SR // out_sr
    tr = f[::k]
    tr = np.clip(tr, center*0.97, center*1.03)
    return (tr - center) / center

def env_track(x, center, bw=150):
    sos = butter(4, [(center-bw)/(SR/2), (center+bw)/(SR/2)],
                 btype="band", output="sos")
    y = sosfilt(sos, x)
    f, t, S = stft(y, fs=SR, nperseg=512, noverlap=448)
    return np.abs(S).sum(axis=0)

def mod_top(env, t0, t1, n=3):
    """Топ-частоты модуляции в окне [t0, t1] минут (адаптивный nperseg)."""
    e = env[int(t0*ENV_SR):int(t1*ENV_SR)]
    if len(e) < 256:
        return "слишком короткий сегмент"
    e_centered = e - e.mean()
    nperseg = min(len(e_centered) - 1, 1024)
    noverlap = nperseg * 3 // 4
    fe, te, Se = stft(e_centered, fs=ENV_SR, nperseg=nperseg, noverlap=noverlap)
    Se = np.abs(Se).mean(axis=1)
    b = (fe > 0.05) & (fe < 6)
    if not b.any():
        return "нет частот в диапазоне"
    idx = np.argsort(Se[b])[::-1][:n]
    return [(round(float(v), 2), round(1/float(v), 1)) for v in fe[b][idx]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--t0", type=float, default=43.0)
    ap.add_argument("--t1", type=float, default=44.0)
    ap.add_argument("--out", default="wow")
    a = ap.parse_args()
    x = load_wav(a.wav)
    i0, i1 = int(a.t0*60*SR), int(a.t1*60*SR)
    seg = x[i0:i1]

    d999 = freq_track(seg, 999.0)
    d2550 = freq_track(seg, 2550.0)
    n = min(len(d999), len(d2550))
    d999, d2550 = d999[:n], d2550[:n]
    r = float(np.corrcoef(d999, d2550)[0, 1])
    print(f"[wow&flutter] корреляция Δf/f линий 999 и 2550 Гц: {r:.3f}")
    print(f"  std(999)={d999.std():.2e}, std(2550)={d2550.std():.2e}, "
          f"отношение std={d2550.std()/d999.std():.2f}")
    if r > 0.7:
        print("  -> КОГЕРЕНТНО: дрожание общее для всех тонов = механика ленты")
    elif r < 0.3:
        print("  -> НЕкогерентно: у линии 2550 Гц собственное дыхание")
    else:
        print("  -> промежуточный результат: смешанный вклад")

    fig, ax = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    tt = np.arange(n) / 100.0
    ax[0].plot(tt, d999*1e4, lw=1, label="999 Гц")
    ax[0].plot(tt, d2550*1e4, lw=1, label="2550 Гц")
    ax[0].set_ylabel("Δf/f, 1e-4"); ax[0].legend(); ax[0].grid(alpha=.3)
    e = env_track(seg, 2550.0)
    ax[1].plot(np.arange(len(e))/ENV_SR, e, lw=0.8, color="r")
    ax[1].set_ylabel("огибающая 2550 Гц"); ax[1].set_xlabel("секунды окна")
    ax[1].grid(alpha=.3)
    fig.suptitle(f"Дрожание частот тонов и огибающая свиста, окно {a.t0}-{a.t1} мин")
    fig.tight_layout(); fig.savefig(a.out + ".png", dpi=150); plt.close(fig)
    print("[график]", a.out + ".png")

    env_full = env_track(x, 2550.0)
    print("\n[спектр модуляции огибающей 2550 Гц] топ-частоты (Гц, период с):")
    for w in [(20, 24), (43, 47), (80, 84)]:
        result = mod_top(env_full, *w)
        print(f"  окно {w[0]}-{w[1]} мин:", result)
    print("  если топ-частоты одинаковы во всех окнах -> машинный ритм;"
          " если дрейфуют -> живая динамика источника")

if __name__ == "__main__":
    main()