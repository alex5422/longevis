"""Biomarqueurs cardiovasculaires dérivés du signal rPPG.

Fréquence cardiaque, variabilité (domaines temporel et fréquentiel),
et morphologie de l'onde de pouls (proxy de compliance artérielle).
"""

from __future__ import annotations
from typing import Dict

import numpy as np
from scipy import interpolate

from . import dsp
from .config import LF_BAND, HF_BAND, PULSE_BAND, QUALITY
from .rppg import PulseSignal


def _ibi_series(pulse: PulseSignal) -> tuple[np.ndarray, np.ndarray]:
    """Intervalles inter-battements (ms) et instants associés (s)."""
    peaks = dsp.find_pulse_peaks(pulse.signal, pulse.fps, pulse.f0_hz)
    if peaks.size < 3:
        return np.array([]), np.array([])
    t = dsp.refine_peaks(pulse.signal, peaks) / pulse.fps
    ibi = np.diff(t) * 1000.0
    # Filtrage des ectopies / faux pics : écart > 25 % de la médiane glissante
    med = np.median(ibi)
    keep = (ibi > 0.70 * med) & (ibi < 1.35 * med) & (ibi > 280) & (ibi < 1600)
    return ibi[keep], t[1:][keep]


def _frequency_domain(ibi: np.ndarray, t: np.ndarray) -> Dict[str, float]:
    """Analyse spectrale de la VFC après ré-échantillonnage à 4 Hz."""
    if ibi.size < 20 or t.size < 20 or (t[-1] - t[0]) < 20.0:
        return {"lf_power": float("nan"), "hf_power": float("nan"),
                "lf_hf_ratio": float("nan")}
    fs = 4.0
    tt = np.arange(t[0], t[-1], 1.0 / fs)
    f_interp = interpolate.interp1d(t, ibi, kind="cubic", bounds_error=False,
                                    fill_value=(ibi[0], ibi[-1]))
    series = f_interp(tt)
    series = series - series.mean()
    f, pxx = dsp.welch_psd(series, fs, nperseg_s=min(60.0, max(20.0, (tt[-1] - tt[0]) / 3)))
    lf = dsp.band_power(f, pxx, *LF_BAND)
    hf = dsp.band_power(f, pxx, *HF_BAND)
    return {"lf_power": lf, "hf_power": hf,
            "lf_hf_ratio": float(lf / hf) if hf > 0 else float("nan")}


def _morphology(pulse: PulseSignal) -> Dict[str, float]:
    """Morphologie de l'onde : temps de montée relatif et régularité d'amplitude.

    Un temps de montée relatif court est associé à une rigidité artérielle plus
    élevée dans la littérature PPG au doigt ; ici il s'agit d'un *proxy* optique.
    """
    s = pulse.signal
    peaks = dsp.find_pulse_peaks(s, pulse.fps, pulse.f0_hz)
    if peaks.size < 4:
        return {"pulse_rise_ratio": float("nan"), "pulse_amp_cv": float("nan"),
                "pulse_amp": float("nan")}
    # Cycle délimité par deux creux successifs : le temps de montée relatif est
    # alors comparable d'un sujet à l'autre malgré des fréquences différentes.
    troughs = []
    for i in range(1, len(peaks)):
        a, b = peaks[i - 1], peaks[i]
        if b - a >= 4:
            troughs.append(a + int(np.argmin(s[a:b])))
    rises, amps = [], []
    for i in range(1, len(troughs)):
        t0, t1 = troughs[i - 1], troughs[i]
        seg = s[t0:t1]
        if seg.size < 4:
            continue
        peak_rel = int(np.argmax(seg))
        cycle = t1 - t0
        if 0 < peak_rel < cycle:
            rises.append(peak_rel / cycle)
            amps.append(float(seg[peak_rel] - seg[0]))
    if not rises:
        return {"pulse_rise_ratio": float("nan"), "pulse_amp_cv": float("nan"),
                "pulse_amp": float("nan")}
    amps = np.asarray(amps)
    return {
        "pulse_rise_ratio": float(np.median(rises)),
        "pulse_amp_cv": float(np.std(amps) / (np.mean(amps) + 1e-9)),
        "pulse_amp": float(np.mean(amps)),
    }


def perfusion_index(rgb_by_roi: Dict[str, np.ndarray], pulse: PulseSignal) -> float:
    """Indice de perfusion : composante pulsatile / composante continue (%).

    Calculé sur le canal vert de la ROI la plus vascularisée disponible.
    """
    key = "front" if "front" in rgb_by_roi else "visage"
    g = dsp.interp_nan(rgb_by_roi[key][:, 1])
    dc = float(np.mean(g))
    if dc <= 0:
        return float("nan")
    ac = dsp.bandpass(g - dc, pulse.fps, *PULSE_BAND)
    if ac.size < 10:
        return float("nan")
    amp = float(np.percentile(ac, 97.5) - np.percentile(ac, 2.5))
    return float(100.0 * amp / dc)


def cardiovascular_features(pulse: PulseSignal,
                            rgb_by_roi: Dict[str, np.ndarray],
                            duration_s: float) -> Dict[str, float]:
    ibi, t = _ibi_series(pulse)
    feats: Dict[str, float] = {
        "hr_bpm": pulse.hr_bpm,
        "pulse_snr_db": pulse.snr_db,
        "pulse_spectral_purity": pulse.spectral_purity,
        "n_beats": float(ibi.size + 1),
    }

    if ibi.size >= 10:
        diff = np.diff(ibi)
        feats.update({
            "hrv_sdnn_ms": float(np.std(ibi, ddof=1)),
            "hrv_rmssd_ms": float(np.sqrt(np.mean(diff ** 2))),
            "hrv_pnn50_pct": float(100.0 * np.mean(np.abs(diff) > 50.0)),
            "hr_from_ibi_bpm": float(60000.0 / np.mean(ibi)),
        })
    else:
        feats.update({"hrv_sdnn_ms": float("nan"), "hrv_rmssd_ms": float("nan"),
                      "hrv_pnn50_pct": float("nan"), "hr_from_ibi_bpm": float("nan")})

    feats.update(_frequency_domain(ibi, t))
    feats.update(_morphology(pulse))
    feats["perfusion_index"] = perfusion_index(rgb_by_roi, pulse)

    # Fiabilité de la VFC : durée + nombre de battements + SNR
    reliable = (duration_s >= 60.0 and ibi.size >= QUALITY["hrv_min_beats"]
                and pulse.snr_db >= QUALITY["snr_db_usable"])
    feats["hrv_reliable"] = float(bool(reliable))
    return feats
