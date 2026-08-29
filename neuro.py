"""Biomarqueurs neuromoteurs, oculomoteurs et respiratoires.

Le mouvement de tête, mesuré par flux optique, porte trois informations
superposées et séparables en fréquence :
  0.13–0.50 Hz  respiration (balancement céphalique induit, Balakrishnan 2013)
  0.10–2.0  Hz  oscillation posturale / stabilité de maintien
  4–12      Hz  tremblement physiologique
"""

from __future__ import annotations
from typing import Dict

import numpy as np
from scipy import signal as sps

from . import dsp
from .config import RESP_BAND, TREMOR_BAND, SWAY_BAND


def _normalize_by_face(head_xy: np.ndarray, face_size: np.ndarray) -> np.ndarray:
    """Convertit les pixels en unités « diagonale de visage » (indépendant de la distance)."""
    scale = np.median(face_size[np.isfinite(face_size)]) if face_size.size else 1.0
    scale = scale if scale and np.isfinite(scale) and scale > 1 else 1.0
    return head_xy / scale * 100.0     # unité : % de diagonale faciale


def respiration(head_xy: np.ndarray, fps: float, pulse_signal: np.ndarray | None = None
                ) -> Dict[str, float]:
    """Fréquence respiratoire par mouvement céphalique vertical, avec repli
    sur la modulation d'amplitude du pouls (RSA optique)."""
    vy = dsp.detrend_smoothness(head_xy[:, 1], lam=200.0)
    f0, purity, _, _ = dsp.dominant_frequency(
        dsp.bandpass(vy, fps, *RESP_BAND), fps, RESP_BAND, nperseg_s=20.0)
    source = "mouvement_cephalique"

    if (not np.isfinite(f0) or purity < 0.15) and pulse_signal is not None:
        env = np.abs(sps.hilbert(pulse_signal))
        env = dsp.detrend_smoothness(env, lam=200.0)
        f0b, purity_b, _, _ = dsp.dominant_frequency(
            dsp.bandpass(env, fps, *RESP_BAND), fps, RESP_BAND, nperseg_s=20.0)
        if np.isfinite(f0b) and purity_b > purity:
            f0, purity, source = f0b, purity_b, "modulation_du_pouls"

    return {
        "resp_rate_cpm": float(f0 * 60.0) if np.isfinite(f0) else float("nan"),
        "resp_purity": float(purity),
        "resp_source": source,
    }


def neuromotor(head_xy: np.ndarray, face_size: np.ndarray, fps: float) -> Dict[str, float]:
    """Tremblement physiologique et stabilité posturale."""
    norm = _normalize_by_face(head_xy, face_size)
    out: Dict[str, float] = {}

    # Vitesse instantanée (dérivée du déplacement cumulé)
    vel = np.diff(norm, axis=0) * fps
    if vel.shape[0] < int(2 * fps):
        return {"tremor_power_norm": float("nan"), "tremor_peak_hz": float("nan"),
                "sway_rms_px": float("nan"), "motion_artifact_px": float("nan")}

    mag = np.hypot(vel[:, 0], vel[:, 1])
    out["motion_artifact_px"] = float(np.median(np.abs(np.diff(head_xy, axis=0)).sum(axis=1)))

    f, pxx = dsp.welch_psd(mag - mag.mean(), fps, nperseg_s=8.0)
    total = dsp.band_power(f, pxx, 0.05, min(fps / 2 - 0.1, 15.0))
    tremor = dsp.band_power(f, pxx, *TREMOR_BAND) if fps > 2 * TREMOR_BAND[1] else float("nan")
    out["tremor_power_norm"] = float(tremor / total) if total > 0 and np.isfinite(tremor) else float("nan")

    if np.isfinite(out["tremor_power_norm"]):
        m = (f >= TREMOR_BAND[0]) & (f <= TREMOR_BAND[1])
        out["tremor_peak_hz"] = float(f[m][np.argmax(pxx[m])]) if m.any() else float("nan")
    else:
        out["tremor_peak_hz"] = float("nan")

    sway = np.stack([dsp.bandpass(norm[:, 0], fps, *SWAY_BAND),
                     dsp.bandpass(norm[:, 1], fps, *SWAY_BAND)], axis=1)
    out["sway_rms_px"] = float(np.sqrt(np.mean(sway ** 2)))
    out["sway_path_length"] = float(np.sum(np.hypot(*np.diff(sway, axis=0).T)) / (len(sway) / fps))
    return out


def oculomotor(eye_signal: np.ndarray, fps: float, duration_s: float) -> Dict[str, float]:
    """Clignements détectés comme transitoires d'intensité de la région oculaire.

    Proxy : la paupière fermée réfléchit davantage que l'iris/la sclère.
    Fiable en éclairage stable et cadrage fixe ; à recouper avec un modèle de
    points de repère palpébraux pour un usage clinique.
    """
    x = dsp.interp_nan(eye_signal)
    if x.size < int(3 * fps) or duration_s < 5:
        return {"blink_rate_min": float("nan"), "blink_mean_dur_ms": float("nan"),
                "blink_count": 0.0}

    x = dsp.detrend_smoothness(x, lam=150.0)
    x = dsp.bandpass(x, fps, 0.8, min(9.0, fps / 2 - 0.5), order=3)
    sd = np.std(x) + 1e-9
    peaks, props = sps.find_peaks(np.abs(x) / sd, height=1.8, distance=max(1, int(0.18 * fps)),
                                  width=(max(1, int(0.03 * fps)), max(2, int(0.6 * fps))))
    widths = props.get("widths", np.array([]))
    return {
        "blink_count": float(peaks.size),
        "blink_rate_min": float(peaks.size / (duration_s / 60.0)),
        "blink_mean_dur_ms": float(np.mean(widths) / fps * 1000.0) if widths.size else float("nan"),
    }
