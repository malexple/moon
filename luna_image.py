#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LUNA-IMAGE v0.1: перевод свиста в образы.
Шаг 0: вычитаем технический пьедестал (моды, найденные в прошлом анализе).
Шаг 1: пять кодировок "звук -> картинка".
Сравнивай run1 (44 мин, свист) и control (30 мин): уникальное в run1 = кандидат в "образ".
Зависимости: numpy scipy matplotlib
"""
import argparse
import numpy as np
from scipy.signal import stft, iirnotch, filtfilt, butter, sosfilt, hilbert
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 16000
# пьедестал, измеренный на прошлом шаге (общий для всего архива)
ARTIFACTS = [421.9, 699.2, 800.3, 845.3, 898.1, 998.6, 1097.9, 1195.3, 1296.9]


def load_wav(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    return x / max(1.0, np.abs(x).max())


def clean(x):
    """Вырезаем технический пьедестал notch-фильтрами."""
    y = x
    for f0 in ARTIFACTS:
        b, a = iirnotch(f0 / (SR / 2), Q=30)
        y = filtfilt(b, a, y)
    return y


# ---------- A: очищенная спектрограмма ----------
def img_spectro(x, out):
    y = clean(x)
    f, t, S = stft(y, fs=SR, nperseg=2048, noverlap=1792)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.pcolormesh(t, f, 20 * np.log10(np.abs(S) + 1e-12),
                  shading="auto", cmap="magma")
    ax.set_yscale("log")
    ax.set_ylim(200, 4000)
    ax.set_ylabel("Гц (лог)")
    fig.savefig(out + "_A_spectro.png", dpi=150)
    plt.close(fig)
    print("[A] очищенная спектрограмма:", out + "_A_spectro.png")


# ---------- B: растрированная мелодия-карта ----------
def img_raster(x, out, nrows=256, ncols=600):
    """Белим спектр (убираем остаточный пьедестал) и рисуем
    траекторию тональных линий во времени = "рисунок свиста"."""
    y = clean(x)
    f, t, S = stft(y, fs=SR, nperseg=1024, noverlap=896)
    P = np.abs(S)
    med = np.median(P, axis=1, keepdims=True) + 1e-12
    W = P / med                                   # whitening по частоте
    idx = (f >= 300) & (f <= 3000)
    f2, W2 = f[idx], W[idx, :ncols]
    rows = np.geomspace(300, 3000, nrows)
    img = np.zeros((nrows, W2.shape[1]))
    lf = np.log(f2)
    for c in range(W2.shape[1]):
        img[:, c] = np.interp(np.log(rows), lf, W2[:, c])
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(img, aspect="auto", cmap="inferno", origin="lower")
    ax.set_title("Мелодия-карта (whitened)")
    fig.savefig(out + "_B_raster.png", dpi=150)
    plt.close(fig)
    print("[B] мелодия-карта:", out + "_B_raster.png")


# ---------- C: SSTV-попытка ----------
def img_sstv(x, out):
    """Классика SSTV: 1200 Гц = sync, 1500..2300 Гц = чёрное..белое.
    1) FM-демодуляция в яркость; 2) ищем sync-импульсы; 3) растры."""
    ana = hilbert(x)
    ph = np.unwrap(np.angle(ana))
    ifreq = np.diff(ph) / (2 * np.pi) * SR
    g = np.clip((ifreq - 1500.0) / 800.0, 0, 1)   # яркость 0..1

    sos = butter(3, [1100 / (SR / 2), 1300 / (SR / 2)], btype="band", output="sos")
    env = np.abs(hilbert(sosfilt(sos, x)))
    thr = env.mean() + 3 * env.std()
    pulses = np.where(env > thr)[0]
    sync = None
    if len(pulses) > 4:
        d = np.diff(pulses)
        d = d[(d > SR * 0.005) & (d < SR * 2.0)]
        if len(d):
            sync = float(np.median(d)) / SR
            print(f"[C] найден sync-период: {sync*1000:.1f} мс")
    for line_px in (120, 128, 160, 256, 320):
        n = (len(g) // line_px) * line_px
        if n // line_px < 8:
            continue
        im = g[:n].reshape(-1, line_px)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(im, cmap="gray", aspect="auto")
        ax.set_title(f"SSTV-растр line={line_px}px")
        fig.savefig(f"{out}_C_sstv_{line_px}.png", dpi=120)
        plt.close(fig)
    print("[C] растры сохранены:", out + "_C_sstv_*.png")


# ---------- D: фазовый портрет ----------
def img_phase(x, out, tau=8):
    y = clean(x)
    h, _, _ = np.histogram2d(y[:-tau], y[tau:], bins=400,
                             range=[[-1, 1], [-1, 1]])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.log1p(h.T), cmap="magma", origin="lower")
    ax.set_title("Фазовый портрет (аттрактор формы)")
    fig.savefig(out + "_D_phase.png", dpi=150)
    plt.close(fig)
    print("[D] фазовый портрет:", out + "_D_phase.png")


# ---------- E: рекуррентный плот ----------
def img_recurrence(x, out):
    y = clean(x)
    f, t, S = stft(y, fs=SR, nperseg=512, noverlap=448)
    env = np.abs(S).mean(axis=0)
    env = (env - env.mean()) / (env.std() + 1e-12)
    step = max(1, len(env) // 800)
    e = env[::step]
    D = np.abs(e[:, None] - e[None, :])
    Z = (D < 0.5).astype(float)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(Z, cmap="gray_r")
    ax.set_title("Рекуррентный плот (повторы во времени)")
    fig.savefig(out + "_E_recurrence.png", dpi=150)
    plt.close(fig)
    print("[E] рекуррентный плот:", out + "_E_recurrence.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--start", type=float, default=2640.0)
    ap.add_argument("--dur", type=float, default=300.0)
    ap.add_argument("--out", default="img1")
    a = ap.parse_args()
    x = load_wav(a.wav)
    seg = x[int(a.start * SR):int((a.start + a.dur) * SR)]
    print(f"Сегмент {a.start:.0f}..{a.start+a.dur:.0f} c")
    img_spectro(seg, a.out)
    img_raster(seg, a.out)
    img_sstv(seg, a.out)
    img_phase(seg, a.out)
    img_recurrence(seg, a.out)


if __name__ == "__main__":
    main()