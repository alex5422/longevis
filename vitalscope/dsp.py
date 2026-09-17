"""Traitement du signal : détendance, filtrage, spectre, qualité."""

from __future__ import annotations
import numpy as np
from scipy import signal as sps
from scipy.sparse import spdiags, eye as speye
from scipy.sparse.linalg import spsolve


def detrend_smoothness(x: np.ndarray, lam: float = 100.0) -> np.ndarray:
    """Détendance à priori de régularité (Tarvainen et al., 2002).

    Retire les dérives lentes (éclairage, thermique) sans déphaser le signal,
    contrairement à un filtre passe-haut d'ordre élevé.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 5:
        return x - x.mean()
    ident = speye(n, format="csc")
    data = np.array([np.ones(n), -2 * np.ones(n), np.ones(n)])
    d2 = spdiags(data, [0, 1, 2], n - 2, n).tocsc()
    z = x - spsolve((ident + lam ** 2 * (d2.T @ d2)).tocsc(), x)
    return np.asarray(z).ravel()


def bandpass(x: np.ndarray, fps: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Butterworth à phase nulle. Repli sur détendance si la bande est invalide."""
    nyq = fps / 2.0
    lo_n, hi_n = lo / nyq, min(hi / nyq, 0.99)
    if not (0 < lo_n < hi_n < 1) or x.size < 3 * order * 3:
        return x - np.mean(x)
    b, a = sps.butter(order, [lo_n, hi_n], btype="band")
    padlen = min(3 * max(len(a), len(b)), x.size - 1)
    return sps.filtfilt(b, a, x, padlen=padlen)


def welch_psd(x: np.ndarray, fps: float, nperseg_s: float = 8.0):
    """Densité spectrale de puissance (Welch), fenêtre exprimée en secondes."""
    nper = int(min(len(x), max(64, nperseg_s * fps)))
    f, pxx = sps.welch(x, fs=fps, nperseg=nper, noverlap=nper // 2,
                       detrend="constant", scaling="density")
    return f, pxx


def band_power(f: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f <= hi)
    if not m.any():
        return 0.0
    return float(np.trapezoid(pxx[m], f[m]))


def dominant_frequency(x: np.ndarray, fps: float, band: tuple[float, float],
                       nperseg_s: float = 8.0) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fréquence dominante dans une bande + rapport de puissance associé."""
    f, pxx = welch_psd(x, fps, nperseg_s)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any() or pxx[m].sum() == 0:
        return float("nan"), 0.0, f, pxx
    f_band, p_band = f[m], pxx[m]
    i0 = int(np.argmax(p_band))
    f0 = float(f_band[i0])
    # Affinement parabolique : la résolution de Welch est grossière (~1/T),
    # l'interpolation du sommet évite de quantifier la fréquence cardiaque.
    if 0 < i0 < len(p_band) - 1:
        l, c, r = p_band[i0 - 1], p_band[i0], p_band[i0 + 1]
        den = l - 2 * c + r
        if abs(den) > 1e-15:
            delta = float(np.clip(0.5 * (l - r) / den, -0.5, 0.5))
            f0 += delta * float(f_band[1] - f_band[0]) if len(f_band) > 1 else 0.0
    df = float(f[1] - f[0]) if len(f) > 1 else 0.1
    half = max(0.12, 1.5 * df)          # au moins trois raies autour du sommet
    total = band_power(f, pxx, band[0], band[1])
    peak = band_power(f, pxx, f0 - half, f0 + half)
    ratio = float(np.clip(peak / (total + 1e-12), 0.0, 1.0))
    return f0, ratio, f, pxx


def snr_db(x: np.ndarray, fps: float, f0: float, band: tuple[float, float],
           half_width: float = 0.12) -> float:
    """SNR spectral : puissance autour de f0 et de son harmonique 2f0
    rapportée au reste de la bande physiologique (de Haan & van Leest, 2014)."""
    if not np.isfinite(f0) or f0 <= 0:
        return float("-inf")
    f, pxx = welch_psd(x, fps)
    m_band = (f >= band[0]) & (f <= band[1])
    if not m_band.any():
        return float("-inf")
    sig_mask = (np.abs(f - f0) <= half_width) | (np.abs(f - 2 * f0) <= half_width)
    sig = float(np.trapezoid(pxx[m_band & sig_mask], f[m_band & sig_mask])) if (m_band & sig_mask).any() else 0.0
    noise_mask = m_band & ~sig_mask
    noise = float(np.trapezoid(pxx[noise_mask], f[noise_mask])) if noise_mask.any() else 0.0
    if noise <= 0:
        return float("inf") if sig > 0 else float("-inf")
    return float(10.0 * np.log10((sig + 1e-15) / noise))


def find_pulse_peaks(x: np.ndarray, fps: float, f0: float) -> np.ndarray:
    """Pics systoliques, avec distance minimale dérivée de la fréquence cardiaque."""
    if not np.isfinite(f0) or f0 <= 0:
        return np.array([], dtype=int)
    min_dist = max(1, int(0.62 * fps / f0))
    prom = 0.35 * np.std(x)
    peaks, _ = sps.find_peaks(x, distance=min_dist, prominence=prom)
    return peaks


def refine_peaks(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Position sous-image des pics par ajustement parabolique.

    À 30 i/s, un pic localisé à l'image près quantifie les intervalles à 33 ms,
    soit l'ordre de grandeur du RMSSD lui-même : sans cet affinement, la
    variabilité mesurée serait dominée par l'erreur d'échantillonnage.
    """
    out = []
    for p in peaks:
        if 0 < p < len(x) - 1:
            l, c, r = float(x[p - 1]), float(x[p]), float(x[p + 1])
            den = l - 2 * c + r
            delta = 0.5 * (l - r) / den if abs(den) > 1e-12 else 0.0
            out.append(p + float(np.clip(delta, -0.5, 0.5)))
        else:
            out.append(float(p))
    return np.asarray(out, dtype=float)


def robust_z(v: float, mean: float, std: float, sense: int) -> float:
    """z-score orienté : positif = favorable."""
    if not np.isfinite(v) or std <= 0:
        return float("nan")
    return float(sense * (v - mean) / std)


def z_to_score(z: float, spread: float = 2.2) -> float:
    """Compression d'un z-score vers une échelle 0–100 (logistique douce)."""
    if not np.isfinite(z):
        return float("nan")
    return float(100.0 / (1.0 + np.exp(-z / spread * 1.7)))


def interp_nan(x: np.ndarray) -> np.ndarray:
    """Interpolation linéaire des trous (frames sans peau détectée)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    idx = np.arange(n)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.zeros(n)
    return np.interp(idx, idx[ok], x[ok])
