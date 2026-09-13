#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тонкая структура компоненты 2550 Гц: дублет? боковые? гребёнка?"""
import argparse
import numpy as np
from scipy.signal import stft, resample_poly
from scipy.io import wavfile
from math import gcd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 16000

def load_wav(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64) / max(1.0, np.abs(x).max())
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--center", type=float, default=2550.0)
    ap.add_argument("--span", type=float, default=150.0)
    ap.add_argument("--t0", type=float, default=43.0)
    ap.add_argument("--dur", type=float, default=60.0)
    ap.add_argument("--out", default="fine")
    a = ap.parse_args()
    x = load_wav(a.wav)
    seg = x[int(a.t0*60*SR):int((a.t0+a.dur)*60*SR)]

    f, t, S = stft(seg, fs=SR, nperseg=16384, noverlap=15360)  # ~1 Гц / ~64 мс
    P = np.abs(S)
    m = (f > a.center-a.span) & (f < a.center+a.span)
    fm, Pm = f[m], P[m]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.pcolormesh(t, fm, 20*np.log10(Pm+1e-12), shading="auto", cmap="magma")
    ax.set_ylabel("Гц"); ax.set_xlabel("секунды окна")
    ax.set_title(f"Тонкая структура вокруг {a.center:.0f} Гц")
    fig.savefig(a.out + ".png", dpi=150); plt.close(fig)
    print("[зум]", a.out + ".png")

    pm = Pm.mean(axis=1)
    thr = np.median(pm) * 4
    peaks = []
    for i in range(1, len(pm)-1):
        if pm[i] > thr and pm[i] >= pm[i-1] and pm[i] > pm[i+1]:
            peaks.append(round(float(fm[i]), 1))
    print("пики среднего спектра, Гц:", peaks)
    if len(peaks) > 1:
        sp = [round(peaks[i+1]-peaks[i], 1) for i in range(len(peaks)-1)]
        print("зазоры между пиками, Гц:", sp)
        print("  один пик + симметричные боковые -> АМ с ритмом = зазор")
        print("  два близких пика               -> биения пары несущих")
        print("  ровная гребёнка                -> набор продуктов интермодуляции")

if __name__ == "__main__":
    main()