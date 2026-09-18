"""Élasticité — amplitude de flexion filmée de face.

Hors score Kinexa : mesure indépendante, non pondérée dans les biomarqueurs
agrégés. Sans repérage articulaire, ce module ne mesure pas un angle
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
            "flexion_maintien_s": float("nan"), "flexion_score": float("nan")}
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

    score = score_souplesse(amp, tenue)
    return {"flexion_amplitude_pct": round(amp, 1),
            "flexion_maintien_s": round(tenue, 1),
            "flexion_score": score}


def score_souplesse(amplitude_pct: float, maintien_s: float) -> float:
    """Score Souplesse /100 — amplitude atteinte, avec un bonus de tenue.

    `amplitude_pct` est déjà un proxy 0-100 auto-normalisé sur la hauteur
    debout du sujet. La tenue au maximum ajoute jusqu'à 10 points (2 points
    par seconde tenue au-delà du point le plus loin, plafonné) : un
    aller-retour rapide n'a pas la même valeur qu'une position tenue.
    """
    if not np.isfinite(amplitude_pct):
        return float("nan")
    bonus = min(10.0, 2.0 * maintien_s) if np.isfinite(maintien_s) else 0.0
    return round(min(100.0, amplitude_pct + bonus), 1)


def analyze_elasticite(b: BodyTraces, px_per_m: Optional[float] = None) -> Dict[str, object]:
    # `px_per_m` n'est pas utilisé ici (amplitude en % et tenue en secondes
    # sont déjà sans unité physique) — accepté pour un appel uniforme depuis
    # `pipeline._analyze_geste` avec les autres tests complémentaires.
    f = amplitude(b.height_px, b.valid, b.fps)
    return {"task": "elasticite", "features": f, "segments": {},
            "signals": {"height_px": b.height_px, "cx": b.centroid[:, 0],
                        "cy": b.centroid[:, 1], "fps": b.fps}}
