#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Изоляция подозрительной линии ~2.5 кГц (с исправленным багом stft)."""
import argparse
import numpy as np
from scipy.signal import stft, butter, sosfilt, resample_poly
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

def zoom(x, t0min, t1min, out):
    seg = x[int(t0min*60*SR):int(t1min*60*SR)]
    f, t, S = stft(seg, fs=SR, nperseg=4096, noverlap=3584)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.pcolormesh(t0min + t/60.0, f, 20*np.log10(np.abs(S)+1e-12),
                  shading="auto", cmap="magma")
    ax.set_ylim(0, 4000); ax.set_ylabel("Гц"); ax.set_xlabel("минуты трека")
    ax.set_title(f"Зум {t0min}-{t1min} мин (сырой, без notch)")
    fig.savefig(out + "_zoom.png", dpi=150); plt.close(fig)
    print("[зум]", out + "_zoom.png")

def isolate_audio(x, t0min, t1min, center, out, tag=""):
    """Полосовой фильтр вокруг центра + WAV для ушей + огибающая."""
    lo, hi = center - 150, center + 150
    seg = x[int(t0min*60*SR):int(t1min*60*SR)]
    sos = butter(4, [lo/(SR/2), hi/(SR/2)], btype="band", output="sos")
    y = sosfilt(sos, seg)
    y = 0.9 * y / (np.abs(y).max() + 1e-12)
    wavfile.write(out + f"_isolated_{tag}.wav", SR, (y*32767).astype(np.int16))
    print(f"[аудио] {out}_isolated_{tag}.wav  (полоса {lo}-{hi} Гц)")

    # спектр огибающей (биения)
    f, t, S = stft(y, fs=SR, nperseg=512, noverlap=448)
    env = np.abs(S).sum(axis=0)
    fe, te, Se = stft(env, fs=SR, nperseg=1024, noverlap=896)  # <-- ИСПРАВЛЕНО
    Se = np.abs(Se).mean(axis=1)
    band = (fe > 0.05) & (fe < 5)
    if band.any():
        pk = float(fe[band][np.argmax(Se[band])])
        print(f"[биения огибающей] пик на {pk:.2f} Гц (период {1/pk:.1f} с)")
    else:
        print("[биения] пик не найден")

def harmonic_series(x):
    """Проверяем гипотезу о гармоническом ряде ~100 Гц."""
    f, t, S = stft(x, fs=SR, nperseg=2048, noverlap=1536)
    P = np.abs(S).mean(axis=1)
    # ищем линии с шагом ~100 Гц
    df = f[1] - f[0]
    found = []
    for base in [99, 100, 101, 102, 98]:
        for k in range(4, 30):
            target = base * k
            i = int(round(target / df))
            if 0 <= i < len(f) and P[i] > np.percentile(P, 95):
                found.append((target, round(float(f[i]), 1), round(float(P[i]), 1)))
    return found

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", default="iso")
    a = ap.parse_args()
    x = load_wav(a.wav)

    print("\n[гипотеза] проверка гармонического ряда с базой ~100 Гц:")
    hs = harmonic_series(x)
    for target, actual, power in hs:
        print(f"  {target} Гц (гармоника) -> реально {actual} Гц, мощность {power}")

    zoom(x, 43.0, 47.0, a.out)

    # Изолируем два кандидата:
    # 1. Линию ~999 Гц (главный подозреваемый из предыдущих прогонов)
    # 2. Линию ~2550 Гц (видна на зум-спектрограмме)
    isolate_audio(x, 43.0, 47.0, 999.0, a.out, tag="1kHz")
    isolate_audio(x, 43.0, 47.0, 2550.0, a.out, tag="2k5Hz")

    print("\nПослушай оба файла в наушниках:")
    print(f"  {a.out}_isolated_1kHz.wav  — основной подозреваемый")
    print(f"  {a.out}_isolated_2k5Hz.wav — линия с зум-спектрограммы")

if __name__ == "__main__":
    main()