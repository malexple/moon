#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Паспорт свиста 2.5 кГц: спектр модуляции, глубина модуляции по треку,
привязка активных окон к фазам миссии (расстыковка/стыковка LM-CM)."""
import argparse
import numpy as np
from scipy.signal import stft, butter, sosfilt, resample_poly
from scipy.io import wavfile
from math import gcd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 16000
ENV_SR = 250          # частота семплирования огибающей (SR / hop)
LO, HI = 2400, 2700   # полоса подозреваемого

def load_wav(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64) / max(1.0, np.abs(x).max())
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g)
    return x

def whistle_env(x):
    sos = butter(4, [LO/(SR/2), HI/(SR/2)], btype="band", output="sos")
    y = sosfilt(sos, x)
    f, t, S = stft(y, fs=SR, nperseg=512, noverlap=448)
    return t, np.abs(S).sum(axis=0)          # t в секундах, шаг 1/250 с

def mod_spectrum(env, t0, t1):
    e = env[int(t0*ENV_SR):int(t1*ENV_SR)]
    fe, te, Se = stft(e - e.mean(), fs=ENV_SR, nperseg=2048, noverlap=1536)
    Se = np.abs(Se).mean(axis=1)
    band = (fe > 0.02) & (fe < 10)
    return fe[band], Se[band]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", default="whistle")
    a = ap.parse_args()
    x = load_wav(a.wav)
    t, env = whistle_env(x)
    tmin = t / 60.0

    # глубина модуляции скользящим окном 60 с
    W = 60 * ENV_SR
    depth, T = [], []
    for i in range(0, len(env) - W, W // 2):
        w = env[i:i+W]
        depth.append(float(w.std() / (w.mean() + 1e-12)))
        T.append(i / ENV_SR / 60.0)
    depth = np.array(depth); T = np.array(T)

    fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    ax[0].plot(tmin, env, lw=0.6)
    ax[0].set_ylabel("огибающая 2.5 кГц")
    ax[1].plot(T, depth, lw=1, color="r")
    ax[1].set_ylabel("глубина модуляции\n(std/mean, окно 60 с)")
    ax[1].set_xlabel("минуты трека")
    for a_ in ax:
        a_.axvline(44.0, color="orange", ls="--", alpha=.7)
        a_.grid(alpha=.3)
    ax[0].set_title("Свист 2.5 кГц: огибающая и глубина модуляции "
                    "(оранжевая линия — 44 мин, разговор о 'музыке')")
    fig.tight_layout()
    fig.savefig(a.out + "_env.png", dpi=150); plt.close(fig)
    print("[график]", a.out + "_env.png")

    # активные окна свиста
    thr = np.median(depth) + 1.5 * (np.percentile(depth, 84) - np.median(depth))
    on = depth > thr
    edges = np.where(np.diff(on.astype(int)) != 0)[0]
    print(f"\nПорог глубины модуляции: {thr:.3f}")
    print("Активные окна свиста (минуты трека):")
    for i in range(0, len(edges) - 1, 2):
        s, e = T[edges[i]], T[edges[i+1]]
        star = "  <<< момент 'музыки'" if s <= 44 <= e else ""
        print(f"  {s:.1f} .. {e:.1f}  ({e-s:.0f} мин){star}")

    # спектр модуляции внутри окна с 44-й минутой
    i44 = int(np.argmin(np.abs(T - 44.0)))
    if on[i44]:
        s, e = T[edges[max(0, np.searchsorted(edges, i44) - 1)]], \
               T[edges[np.searchsorted(edges, i44)]]
        fe, Se = mod_spectrum(env, s, e)
        top = fe[np.argsort(Se)[::-1][:5]]
        print(f"\nСпектр модуляции в активном окне {s:.1f}-{e:.1f} мин:")
        print("  топ-частоты модуляции, Гц:", [round(float(v), 2) for v in top])
        print("  (периоды, с:", [round(1/float(v), 1) for v in top], ")")
    else:
        print("\n44-я минута НЕ в активном окне — смотри график!")

    # длинный whistle-only wav для прослушивания
    seg = x[int(40*60*SR):int(50*60*SR)]
    sos = butter(4, [LO/(SR/2), HI/(SR/2)], btype="band", output="sos")
    y = sosfilt(sos, seg)
    y = 0.9 * y / (np.abs(y).max() + 1e-12)
    wavfile.write(a.out + "_full.wav", SR, (y*32767).astype(np.int16))
    print("\n[аудио]", a.out + "_full.wav — 10 минут свиста без примесей")

if __name__ == "__main__":
    main()