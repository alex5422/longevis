"""Tonus postural — gainage chronométré filmé de profil.

Test bêta : pas encore intégré aux scores Kinexa, en attente de calibration
sur cohorte. Sans repérage articulaire (il n'y a que le contour détecté de
la silhouette, pas un squelette), ce module n'estime pas un angle de tronc —
il regarde si la position tenue reste immobile et si la forme du contour
(rapport largeur/hauteur) reste constante pendant l'appui, ce qui est un
indicateur indirect de stabilité, pas une goniométrie.
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np
from scipy import signal as sps

from . import dsp
from .body import BodyTraces


def _smooth(x: np.ndarray, fps: float, cutoff: float = 3.0) -> np.ndarray:
    nyq = fps / 2
    if cutoff >= nyq or len(x) < 9:
        return x
    b, a = sps.butter(2, cutoff / nyq)
    return sps.filtfilt(b, a, x)


def hold(centroid: np.ndarray, trunk_y: np.ndarray, bbox: np.ndarray,
         valid: np.ndarray, fps: float,
         mouvement_max_px: float = 6.0) -> Dict[str, float]:
    """Repère le plus long segment immobile de la vidéo et évalue sa stabilité.

    `mouvement_max_px` : déplacement image-à-image toléré avant de considérer
    que la position tenue est rompue. Dépend de la résolution et de la
    distance caméra-sujet — un seuil à ajuster empiriquement plutôt qu'une
    constante universelle.
    """
    n = len(valid)
    vide = {"gainage_duree_s": 0.0, "gainage_stabilite": float("nan"),
            "gainage_alignement": float("nan")}
    if n < int(2 * fps):
        return vide

    cx = dsp.interp_nan(np.where(valid, centroid[:, 0], np.nan))
    cy = dsp.interp_nan(np.where(valid, centroid[:, 1], np.nan))
    depl = np.hypot(np.diff(cx), np.diff(cy))
    immobile = np.concatenate([[True], depl < mouvement_max_px])

    meilleur = (0, 0)
    debut = None
    for i, ok in enumerate(immobile):
        if ok and debut is None:
            debut = i
        elif not ok and debut is not None:
            if i - debut > meilleur[1] - meilleur[0]:
                meilleur = (debut, i)
            debut = None
    if debut is not None and n - debut > meilleur[1] - meilleur[0]:
        meilleur = (debut, n)

    a, b = meilleur
    duree = (b - a) / fps
    if duree < 2.0:
        return {**vide, "gainage_duree_s": round(duree, 1)}

    # segment tenu, mais silhouette jamais détectée dedans (sujet trop
    # immobile pour la soustraction de fond, mauvais cadrage…) : pas de
    # calcul possible, on le dit plutôt que de produire un chiffre creux
    if int(np.sum(valid[a:b])) < max(3, int(0.5 * fps)):
        return {**vide, "gainage_duree_s": round(duree, 1)}

    ty = _smooth(dsp.interp_nan(np.where(valid, trunk_y, np.nan))[a:b], fps)
    largeur = dsp.interp_nan(np.where(valid, bbox[:, 2], np.nan))[a:b]
    hauteur = dsp.interp_nan(np.where(valid, bbox[:, 3], np.nan))[a:b]

    hauteur_ref = float(np.median(hauteur)) or 1.0
    disp = float(np.std(ty)) / hauteur_ref
    stabilite = float(np.clip(100.0 * (1.0 - disp / 0.04), 0.0, 100.0))

    ratio = np.divide(largeur, hauteur, out=np.full_like(largeur, np.nan),
                       where=hauteur > 0)
    moy_ratio = float(np.mean(ratio)) or 1.0
    var_ratio = float(np.std(ratio)) / moy_ratio
    alignement = float(np.clip(100.0 * (1.0 - var_ratio / 0.08), 0.0, 100.0))

    return {"gainage_duree_s": round(duree, 1),
            "gainage_stabilite": round(stabilite, 1),
            "gainage_alignement": round(alignement, 1)}


def analyze_tonus(b: BodyTraces, px_per_m: Optional[float] = None) -> Dict[str, object]:
    # `px_per_m` n'est pas utilisé ici (durée, stabilité et alignement sont
    # déjà sans unité physique) — accepté pour un appel uniforme depuis
    # `pipeline._analyze_beta` avec les deux autres tests bêta.
    f = hold(b.centroid, b.trunk_y, b.bbox, b.valid, b.fps)
    return {"task": "gainage", "features": f, "segments": {},
            "signals": {"trunk_y": b.trunk_y, "fps": b.fps}}
