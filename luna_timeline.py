#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Временна́я карта свиста по всему треку (v2 — исправлен SR)."""
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
        print(f"Ресемплинг {sr} -> {SR} Гц (трек был {sr} Гц)")
    return x

def metrics(seg):
    f, t, S = stft(seg, fs=SR, nperseg=2048, noverlap=1536)
    P = np.abs(S)
    df = f[1] - f[0]
    def band(lo, hi):
        i = (f >= lo) & (f <= hi)
        return P[i].sum(axis=0)
    i999 = int(round(999 / df))
    line999 = np.mean(P[i999-1:i999+2] / (np.median(P, axis=0) + 1e-12))
    tonal = np.mean(P.max(axis=0) / (P.mean(axis=0) + 1e-12))
    ratio = band(850, 1150).mean() / (band(300, 3000).mean() + 1e-12)
    # PTT-детектор: резкие ступеньки энергии = начало передачи
    e = band(300, 3000)
    ptt = np.mean(np.abs(np.diff(e))) / (e.mean() + 1e-12)
    return line999, tonal, ratio, ptt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--win", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=30.0)
    ap.add_argument("--out", default="timeline")
    a = ap.parse_args()

    x = load_wav(a.wav)
    total_min = len(x) / SR / 60.0
    print(f"Длительность трека: {total_min:.1f} мин")

    W, S = int(a.win * SR), int(a.step * SR)
    T, L, Tn, R, Ptt = [], [], [], [], []
    for t0 in range(0, len(x) - W, S):
        l, tn, r, p = metrics(x[t0:t0+W])
        T.append(t0 / SR / 60.0)
        L.append(l); Tn.append(tn); R.append(r); Ptt.append(p)

    fig, ax = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    ax[0].plot(T, L, lw=1); ax[0].set_ylabel("линия 999 Гц\n(пик/медиана)")
    ax[1].plot(T, Tn, lw=1, color="g"); ax[1].set_ylabel("тональность")
    ax[2].plot(T, R, lw=1, color="r"); ax[2].set_ylabel("доля энергии\n850-1150 Гц")
    ax[3].plot(T, Ptt, lw=1, color="m"); ax[3].set_ylabel("PTT-активность\n(ступеньки энергии)")
    ax[3].set_xlabel("минуты от начала трека")
    for a_ in ax: a_.grid(alpha=.3)
    # отметим 44-ю минуту (разговор о "музыке")
    for a_ in ax:
        a_.axvline(44.0, color="orange", ls="--", alpha=.6, label="44 мин — 'музыка'")
    ax[0].legend(loc="upper right")
    fig.suptitle("Временна́я карта трека 5-OF-6 (Apollo 10)", fontsize=13)
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=150)
    print("Сохранено:", a.out + ".png")

    # поиск эпизодов по линии 999 Гц
    L = np.array(L)
    thr = np.median(L) + 1.5 * (np.percentile(L, 84) - np.median(L))
    on = L > thr
    edges = np.where(np.diff(on.astype(int)) != 0)[0]
    print(f"\nПорог линии 999 Гц: {thr:.2f}")
    print("Эпизоды повышенного свиста (реальные минуты трека):")
    for i in range(0, len(edges) - 1, 2):
        a, b = T[edges[i]], T[edges[i+1]]
        flag = " ★" if a <= 44.0 <= b else ""
        print(f"  {a:.1f} .. {b:.1f}  (длительность {b-a:.1f} мин){flag}")
    print("★ — эпизод, в который попал момент 'музыки' на 44 мин")

    # корреляция PTT vs свист
    r_ptt = float(np.corrcoef(Ptt, L)[0, 1])
    print(f"\nКорреляция PTT-активность ↔ линия 999 Гц: {r_ptt:.3f}")
    if abs(r_ptt) > 0.3:
        print("-> сильная: свист живёт, когда включена межкорабельная связь")
    elif abs(r_ptt) < 0.1:
        print("-> слабая: свист не связан с PTT, ищем другую причину")

if __name__ == "__main__":
    main()