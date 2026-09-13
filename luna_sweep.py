#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Охота на транзиентный свист (H2) и тест маскирования (H1)."""
import argparse
import numpy as np
from scipy.signal import stft, iirnotch, filtfilt, resample_poly
from scipy.io import wavfile
from math import gcd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 16000
ART = [330.0, 421.9, 699.2, 800.3, 845.3, 898.1, 998.6, 1097.9, 1195.3, 1296.9]

def load_wav(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64) / max(1.0, np.abs(x).max())
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g)
    return x

def clean(x):
    y = x
    for f0 in ART:
        b, a = iirnotch(f0 / (SR / 2), Q=30)
        y = filtfilt(b, a, y)
    return y

def zoom_spectro(x, t0, t1, out):
    seg = clean(x[int(t0*SR):int(t1*SR)])
    f, t, S = stft(seg, fs=SR, nperseg=4096, noverlap=3584)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.pcolormesh(t0 + t/60.0, f, 20*np.log10(np.abs(S)+1e-12),
                  shading="auto", cmap="magma")
    ax.set_ylim(0, 4000); ax.set_ylabel("Гц"); ax.set_xlabel("минуты трека")
    ax.set_title(f"Зум {t0}-{t1} мин (пьедестал вырезан)")
    fig.savefig(out + "_zoom.png", dpi=150); plt.close(fig)
    print("[зум]", out + "_zoom.png")

def tonal_events(x, min_dur=4.0, tonal_thr=8.0):
    """H2: ищем тональные линии с дрейфом частоты (чирпы)."""
    y = clean(x)
    f, t, S = stft(y, fs=SR, nperseg=1024, noverlap=768)
    P = np.abs(S)
    m = (f >= 300) & (f <= 3000)
    Pm, fm = P[m], f[m]
    ton = Pm.max(axis=0) / (Pm.mean(axis=0) + 1e-12)
    fp = fm[np.argmax(Pm, axis=0)]
    good = ton > tonal_thr
    dt = t[1] - t[0]
    ev, i, n = [], 0, len(t)
    while i < n:
        if good[i]:
            j = i
            while j+1 < n and good[j+1] and abs(fp[j+1]-fp[j]) < 40:
                j += 1
            if (j-i)*dt >= min_dur:
                ev.append((round(t[i]/60, 2), round(t[j]/60, 2),
                           round(float(fp[i:j+1].mean()), 1),
                           round(float(fp[j]-fp[i]), 1)))
            i = j+1
        else:
            i += 1
    return ev

def masking_test(x):
    """H1: доля пьедестала в полной энергии в тихих и громких окнах."""
    f, t, S = stft(x, fs=SR, nperseg=1024, noverlap=768)
    P = np.abs(S)
    i = (f >= 300) & (f <= 3000)
    E = P[i].sum(axis=0)
    df = f[1] - f[0]
    Pb = np.zeros_like(E)
    for a in ART:
        k = int(round(a/df))
        Pb += P[max(0, k-1):k+2].sum(axis=0)
    r = Pb / (E + 1e-12)
    quiet = E < np.percentile(E, 20)
    loud = E > np.percentile(E, 80)
    return float(r[quiet].mean()), float(r[loud].mean()), t/60.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--t0", type=float, default=40.0)
    ap.add_argument("--t1", type=float, default=50.0)
    ap.add_argument("--out", default="sweep")
    a = ap.parse_args()
    x = load_wav(a.wav)

    zoom_spectro(x, a.t0, a.t1, a.out)

    ev = tonal_events(x)
    print("\n[H2] тональные транзиенты (нач.мин, кон.мин, ср.Гц, дрейф.Гц):")
    for e in ev[:25]:
        star = "  <<< окно 'музыки'" if e[0] <= 44 <= e[1] else ""
        print("  ", e, star)
    if not ev:
        print("   не найдено")

    rq, rl, tmin = masking_test(x)
    print(f"\n[H1] доля пьедестала: тихие окна {rq:.3f}, громкие {rl:.3f}, "
          f"отношение {rq/rl:.2f}x")
    if rq/rl > 1.5:
        print("-> маскирование ПОДТВЕРЖДЕНО: в тишине пьедестал слышен в разы сильнее")
    else:
        print("-> маскирование не выражено")

if __name__ == "__main__":
    main()