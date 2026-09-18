"""Sollicitation anti-ostéoporotique — petits sauts talon filmés.

Mesure indépendante, hors score Kinexa. Approxime la charge imposée à l'os
par la vitesse de descente juste avant l'impact au sol — un indicateur du
stimulus ostéogénique de l'impact (Frost, "mechanostat", 1987).

Le saut talon est une voie de sollicitation osseuse parmi d'autres
(résistance à charge élevée, marche, transferts assis-debout — déjà couverts
par ailleurs dans l'outil) : un score bas ici ne dit rien des autres.
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np
from scipy import signal as sps

from . import dsp
from .body import BodyTraces


def impacts(foot_y: np.ndarray, valid: np.ndarray, fps: float,
            prominence_px: float = 8.0,
            px_per_m: Optional[float] = None) -> Dict[str, float]:
    """`px_per_m` : échelle de conversion (pixels par mètre), fournie par
    l'appelant — taille du sujet déclarée ou distance connue dans le champ.
    Sans elle, la vitesse reste en pixels/s, une unité qui ne dit rien à un
    médecin : impossible de comparer deux vidéos filmées à des distances
    différentes.
    """
    n = len(valid)
    vide = {"impact_nombre": 0.0, "impact_taux_par_min": float("nan"),
            "impact_vitesse_descente": float("nan"),
            "impact_regularite": float("nan"), "sollicitation_score": float("nan")}
    if n < int(3 * fps):
        return vide

    y = dsp.interp_nan(np.where(valid, foot_y, np.nan))
    # y croît vers le bas de l'image : un impact est un maximum local (le
    # pied est à son point le plus bas de la trajectoire à cet instant).
    pics, _ = sps.find_peaks(y, prominence=prominence_px,
                              distance=max(1, int(0.25 * fps)))
    if len(pics) == 0:
        return {**vide, "impact_taux_par_min": 0.0}

    duree = n / fps
    taux = 60.0 * len(pics) / duree

    fen = max(1, int(0.12 * fps))          # vitesse sur les 120 ms avant l'impact
    vitesses = []
    for p in pics:
        a = max(0, p - fen)
        if p - a >= 2:
            v = (y[p] - y[a]) / ((p - a) / fps)   # px/s, positif = descente
            if v > 0:
                vitesses.append(v)
    v_moy_px_s = float(np.mean(vitesses)) if vitesses else float("nan")
    if px_per_m and np.isfinite(v_moy_px_s):
        v_moy = v_moy_px_s / px_per_m           # m/s
    else:
        v_moy = float("nan")

    regularite = (float(np.std(vitesses) / np.mean(vitesses))
                  if len(vitesses) >= 2 and np.mean(vitesses) > 0 else float("nan"))
    score = score_sollicitation(regularite, len(vitesses))

    return {"impact_nombre": float(len(pics)),
            "impact_taux_par_min": round(taux, 1),
            "impact_vitesse_descente": round(v_moy, 2) if np.isfinite(v_moy) else float("nan"),
            "impact_regularite": round(100.0 * (1.0 - min(1.0, regularite)), 1)
                                  if np.isfinite(regularite) else float("nan"),
            "sollicitation_score": score}


def score_sollicitation(cv_vitesse: float, n_impacts: int) -> float:
    """Score Sollicitation /100 — régularité du stimulus d'un saut à l'autre,
    pas son intensité.

    Coefficient de variation de la vitesse de descente talon, inversé en
    score 0-100 : un stimulus reproductible d'un saut à l'autre est plus
    interprétable qu'une série erratique. En dessous de 5 impacts exploités,
    le coefficient de variation n'est pas assez stable pour être noté.
    """
    if not np.isfinite(cv_vitesse) or n_impacts < 5:
        return float("nan")
    return round(100.0 * (1.0 - min(1.0, cv_vitesse)), 1)


def analyze_sollicitation(b: BodyTraces, px_per_m: Optional[float] = None) -> Dict[str, object]:
    f = impacts(b.foot_y, b.valid, b.fps, px_per_m=px_per_m)
    return {"task": "sollicitation", "features": f, "segments": {},
            "signals": {"foot_y": b.foot_y, "cx": b.centroid[:, 0],
                        "cy": b.centroid[:, 1], "fps": b.fps}}
