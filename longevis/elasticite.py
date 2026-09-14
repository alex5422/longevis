"""Élasticité — amplitude de flexion filmée de face.

Test bêta : pas encore intégré aux scores Kinexa, en attente de calibration
sur cohorte. Sans repérage articulaire, ce module ne mesure pas un angle
mais la réduction relative de la hauteur de silhouette pendant une flexion
avant ou un étirement, par rapport à la position debout de référence — une
mesure grossière de l'amplitude, pas une goniométrie.
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np

from . import dsp
from .body import BodyTraces


def amplitude(height_px: np.ndarray, valid: np.ndarray, fps: float,
              reference_s: float = 1.5) -> Dict[str, float]:
    n = len(valid)
    vide = {"flexion_amplitude_pct": float("nan"),
            "flexion_maintien_s": float("nan")}
    if n < int(3 * fps):
        return vide

    h = dsp.interp_nan(np.where(valid, height_px, np.nan))
    n_ref = max(3, int(reference_s * fps))
    reference = float(np.nanmedian(h[:n_ref]))
    if not np.isfinite(reference) or reference <= 0:
        return vide

    minimum = float(np.nanpercentile(h, 2))
    amp = float(np.clip(100.0 * (reference - minimum) / reference, 0.0, 100.0))

    # tenue : temps passé près du point le plus loin (plateau tenu), plutôt
    # qu'un simple aller-retour rapide qui atteindrait la même amplitude
    seuil = reference - 0.9 * (reference - minimum)
    tenue = float(np.sum(h < seuil) / fps)

    return {"flexion_amplitude_pct": round(amp, 1),
            "flexion_maintien_s": round(tenue, 1)}


def analyze_elasticite(b: BodyTraces, px_per_m: Optional[float] = None) -> Dict[str, object]:
    # `px_per_m` n'est pas utilisé ici (amplitude en % et tenue en secondes
    # sont déjà sans unité physique) — accepté pour un appel uniforme depuis
    # `pipeline._analyze_beta` avec les deux autres tests bêta.
    f = amplitude(b.height_px, b.valid, b.fps)
    return {"task": "elasticite", "features": f, "segments": {},
            "signals": {"height_px": b.height_px, "fps": b.fps}}
