"""Extraction du signal photopléthysmographique à distance (rPPG).

Trois méthodes implémentées, puis fusion pondérée par le SNR spectral :
  * GREEN  — Verkruysse et al., Optics Express 2008
  * CHROM  — de Haan & Jeanne, IEEE TBME 2013
  * POS    — Wang et al., IEEE TBME 2017 (plane-orthogonal-to-skin)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import numpy as np

from . import dsp
from .config import PULSE_BAND, ProcessingConfig, DEFAULT


@dataclass
class PulseSignal:
    method: str
    signal: np.ndarray
    fps: float
    f0_hz: float
    snr_db: float
    spectral_purity: float

    @property
    def hr_bpm(self) -> float:
        return float(self.f0_hz * 60.0)


def _prep(rgb: np.ndarray) -> np.ndarray:
    """Interpole les trous et retire la dérive lente de chaque canal."""
    out = np.zeros_like(rgb, dtype=float)
    for c in range(3):
        out[:, c] = dsp.interp_nan(rgb[:, c])
    return out


def green(rgb: np.ndarray, fps: float) -> np.ndarray:
    g = _prep(rgb)[:, 1]
    return -dsp.detrend_smoothness(g / (np.mean(g) + 1e-9))


def chrom(rgb: np.ndarray, fps: float, cfg: ProcessingConfig = DEFAULT) -> np.ndarray:
    """CHROM avec fenêtrage glissant et recombinaison par addition-recouvrement."""
    c = _prep(rgb)
    n = len(c)
    win = int(round(1.6 * fps))
    if n < win * 2:
        win = max(8, n // 2)
    step = max(1, win // 2)
    out = np.zeros(n)
    wsum = np.zeros(n)
    hann = np.hanning(win)

    for s in range(0, n - win + 1, step):
        seg = c[s:s + win]
        mu = seg.mean(axis=0) + 1e-9
        cn = seg / mu
        xs = 3 * cn[:, 0] - 2 * cn[:, 1]
        ys = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
        xf = dsp.bandpass(xs, fps, *PULSE_BAND)
        yf = dsp.bandpass(ys, fps, *PULSE_BAND)
        alpha = np.std(xf) / (np.std(yf) + 1e-9)
        sig = xf - alpha * yf
        sig = (sig - sig.mean()) / (np.std(sig) + 1e-9)
        out[s:s + win] += sig * hann
        wsum[s:s + win] += hann

    wsum[wsum == 0] = 1.0
    return out / wsum


def pos(rgb: np.ndarray, fps: float, cfg: ProcessingConfig = DEFAULT) -> np.ndarray:
    """POS : projection orthogonale au plan de la peau."""
    c = _prep(rgb)
    n = len(c)
    l = max(8, int(round(cfg.pos_window_s * fps)))
    if n <= l:
        l = max(8, n // 2)
    h = np.zeros(n)
    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])

    for t in range(0, n - l + 1):
        seg = c[t:t + l]
        mu = seg.mean(axis=0) + 1e-9
        cn = seg / mu
        s = proj @ cn.T                       # (2, l)
        alpha = np.std(s[0]) / (np.std(s[1]) + 1e-9)
        hh = s[0] + alpha * s[1]
        h[t:t + l] += hh - hh.mean()

    return -h if np.mean(h) < 0 else h


def _evaluate(sig: np.ndarray, fps: float, name: str) -> PulseSignal:
    filt = dsp.bandpass(sig, fps, *PULSE_BAND)
    f0, purity, _, _ = dsp.dominant_frequency(filt, fps, PULSE_BAND)
    snr = dsp.snr_db(filt, fps, f0, PULSE_BAND)
    filt = (filt - filt.mean()) / (np.std(filt) + 1e-12)
    return PulseSignal(name, filt, fps, f0, snr, purity)


def extract_pulse(rgb_by_roi: Dict[str, np.ndarray], fps: float,
                  cfg: ProcessingConfig = DEFAULT) -> Dict[str, PulseSignal]:
    """Applique les 3 méthodes sur la meilleure combinaison de ROI.

    Les ROI riches en capillaires (front, joues) sont moyennées ; le visage
    entier sert de repli quand les sous-ROI sont trop bruitées.
    """
    prefer = [r for r in ("front", "joue_gauche", "joue_droite") if r in rgb_by_roi]
    stack = [rgb_by_roi[r] for r in prefer] or [rgb_by_roi["visage"]]
    rgb = np.nanmean(np.stack(stack, axis=0), axis=0)

    candidates = {
        "POS": pos(rgb, fps, cfg),
        "CHROM": chrom(rgb, fps, cfg),
        "GREEN": green(rgb, fps),
    }
    results = {k: _evaluate(v, fps, k) for k, v in candidates.items()}

    # Fusion : moyenne pondérée par le SNR (poids nuls si SNR négatif)
    weights, aligned = [], []
    ref = max(results.values(), key=lambda p: p.snr_db)
    for p in results.values():
        w = max(0.0, p.snr_db)
        if w <= 0:
            continue
        s = p.signal
        # alignement de phase sur la méthode de référence
        if np.dot(s, ref.signal) < 0:
            s = -s
        aligned.append(s)
        weights.append(w)
    if aligned:
        fused = np.average(np.stack(aligned), axis=0, weights=weights)
    else:
        fused = ref.signal
    results["FUSION"] = _evaluate(fused, fps, "FUSION")
    return results


def best_pulse(results: Dict[str, PulseSignal]) -> PulseSignal:
    return max(results.values(), key=lambda p: (p.snr_db, p.spectral_purity))
