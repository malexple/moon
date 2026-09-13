#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LUNA-DECODE v0.1
Анализатор "пения птиц" Apollo-10 (трек 10-030702_5-OF-6, от ~44:00).

Гипотеза: сигнал может быть не сообщением (последовательностью символов),
а следом присутствия стационарной плазменной формы -> ищем ГЕОМЕТРИЮ:
устойчивые спектральные моды, их топологию и инварианты, плюс контрольный
тест официальной версии (интермодуляция VHF, модулированная голосами).

Зависимости: pip install numpy scipy matplotlib
"""
import argparse
import numpy as np
from scipy.signal import stft, butter, sosfilt, hilbert, correlate
from scipy.signal import resample_poly
from scipy.io import wavfile
from math import gcd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 16000  # рабочий sample rate (голосовой канал уже, выше не нужно)


# ---------- загрузка ----------
def load_wav(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    x /= max(1.0, np.abs(x).max())
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g)
    return x


# ---------- блок A: спектрограммы на нескольких разрешениях ----------
def export_spectrograms(x, out, windows=(256, 1024, 4096)):
    """Смотрим "почерк" поля на разной детализации.
    Инвариантная фигура при смене окна = признак геометрии, а не шума."""
    fig, axes = plt.subplots(len(windows), 1, figsize=(16, 4 * len(windows)))
    for ax, n in zip(np.atleast_1d(axes), windows):
        f, t, S = stft(x, fs=SR, nperseg=n, noverlap=n * 7 // 8)
        ax.pcolormesh(t, f, 20 * np.log10(np.abs(S) + 1e-12),
                      shading="auto", cmap="magma")
        ax.set_ylabel("Гц")
        ax.set_title(f"Окно FFT n={n}")
    fig.tight_layout()
    fig.savefig(out + "_spectro.png", dpi=150)
    plt.close(fig)
    print(f"[A] спектрограммы: {out}_spectro.png")


# ---------- блок B: устойчивые моды и их топология ----------
def persistent_modes(x, occupancy=0.5, nperseg=2048):
    """Линии, живущие дольше половины времени анализа = стационарные моды.
    Мелодия/речь таких не дают: они дают мазки."""
    f, t, S = stft(x, fs=SR, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    P = np.abs(S)
    thr = np.percentile(P, 97, axis=0, keepdims=True)  # топ-3% кадра
    occ = (P >= thr).mean(axis=1)                      # занятость бина по времени
    idx = np.where(occ > occupancy)[0]
    lines = []
    if len(idx):
        groups = np.split(idx, np.where(np.diff(idx) > 2)[0] + 1)
        for g in groups:
            if len(g) == 0:
                continue
            fc = np.average(f[g], weights=occ[g])
            lines.append((round(fc, 2), round(float(occ[g].max()), 3)))
    return lines


def scan_relations(lines):
    """Топология мод: отношения частот. Ищем малые целые (гармоники/аккорд)
    и золотое сечение. Последовательный код так не выглядит."""
    freqs = [l[0] for l in lines]
    hits = []
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            r = freqs[j] / freqs[i]
            for n in range(1, 9):
                for m in range(1, 9):
                    if abs(r - n / m) < 0.01 * n / m:
                        hits.append((freqs[i], freqs[j], f"{n}:{m}"))
            if abs(r - 1.6180339887) < 0.016:
                hits.append((freqs[i], freqs[j], "golden"))
    return hits


# ---------- блок C: тест интермодуляции (голос <-> свист) ----------
def intermod_test(x, nperseg=1024):
    """Офиц. версия: свист = биения VHF-несущих, модулированных голосами.
    Тогда тональность свиста должна коррелировать с речевой активностью.
    Низкая корреляция = свист автономен от голосов."""
    f, t, S = stft(x, fs=SR, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    P = np.abs(S)
    tonality = P.max(axis=0) / (P.mean(axis=0) + 1e-12)  # пик/среднее = тональность
    sos = butter(4, [300 / (SR / 2), 3000 / (SR / 2)], btype="band", output="sos")
    env = np.abs(hilbert(sosfilt(sos, x)))               # речевая огибающая
    tt = np.arange(len(env)) / SR
    voice_at_frames = np.interp(t, tt, env)
    r0 = float(np.corrcoef(tonality, voice_at_frames)[0, 1])
    xc = correlate(tonality - tonality.mean(),
                   voice_at_frames - voice_at_frames.mean(), mode="full")
    dt = t[1] - t[0]
    lags = (np.arange(len(xc)) - (len(voice_at_frames) - 1)) * dt
    sel = np.abs(lags) <= 15
    k = int(np.argmax(np.abs(xc[sel])))
    return {"pearson_r": round(r0, 3),
            "max_xcorr_lag_sec": round(float(lags[sel][k]), 2),
            "xcorr_norm": round(float(np.abs(xc[sel][k]) /
                                      (np.linalg.norm(tonality) *
                                       np.linalg.norm(voice_at_frames))), 3)}


# ---------- блок D: геометрические инварианты ----------
def box_dim(Z, sizes=(2, 4, 8, 16, 32, 64)):
    counts = []
    for s in sizes:
        h, w = Z.shape[0] // s * s, Z.shape[1] // s * s
        Zc = Z[:h, :w].reshape(h // s, s, w // s, s)
        counts.append(Zc.any(axis=(1, 3)).sum())
    return float(np.polyfit(np.log(1 / np.array(sizes)), np.log(counts), 1)[0])


def structure_metrics(x, nperseg=1024):
    """Энтропия ( шум/порядок ), фрактальная размерность и симметрия
    бинаризованной спектрограммы = геометрический портрет поля."""
    f, t, S = stft(x, fs=SR, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    P = np.abs(S)
    p = P / (P.sum(axis=0, keepdims=True) + 1e-12)
    H = -(p * np.log2(p + 1e-12)).sum(axis=0) / np.log2(len(f))
    img = 20 * np.log10(P + 1e-12)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    Z = img > 0.7
    z = Z.astype(float)
    sym_t = float(np.corrcoef(z.ravel(), z[::-1, :].ravel())[0, 1])  # реверс времени
    sym_f = float(np.corrcoef(z.ravel(), z[:, ::-1].ravel())[0, 1])  # зеркало частот
    return {"entropy_mean": round(float(H.mean()), 3),
            "entropy_std": round(float(H.std()), 3),
            "fractal_dim": round(box_dim(Z), 3),
            "sym_time": round(sym_t, 3),
            "sym_freq": round(sym_f, 3)}


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--start", type=float, default=2640.0, help="сек (44 мин = 2640)")
    ap.add_argument("--dur", type=float, default=300.0, help="длина сегмента, сек")
    ap.add_argument("--out", default="run1")
    a = ap.parse_args()

    x = load_wav(a.wav)
    i0, i1 = int(a.start * SR), int((a.start + a.dur) * SR)
    seg = x[i0:min(i1, len(x))]
    print(f"Сегмент {a.start:.0f}..{a.start + a.dur:.0f} c, {len(seg)/SR:.1f} c аудио")

    export_spectrograms(seg, a.out)

    lines = persistent_modes(seg)
    print(f"[B] устойчивые моды (Гц, занятость): {lines}")
    rel = scan_relations(lines)
    print(f"[B] отношения частот: {rel if rel else 'не найдено'}")

    im = intermod_test(seg)
    print(f"[C] тест интермодуляции: {im}")
    print("    интерпретация: |r|<0.2 и xcorr_norm<0.2 => свист автономен от голосов")

    sm = structure_metrics(seg)
    print(f"[D] геометрия поля: {sm}")
    print("    интерпретация: fractal_dim 1.2-1.8 + низкая entropy_std =")
    print("    стационарная структура; entropy_mean~1 = белый шум; ~0 = чистый тон")


if __name__ == "__main__":
    main()