"""Agrégation des biomarqueurs en indices lisibles.

Deux modes :
  * heuristique  — z-scores contre `REFERENCE_NORMS`, pondérés par domaine.
                   Sert de lecture relative, sans valeur diagnostique.
  * calibré      — modèle linéaire régularisé appris sur une cohorte annotée
                   (voir `calibrate.py`). Seul ce mode autorise une lecture
                   quantitative, dans les limites de la cohorte d'entraînement.
"""

from __future__ import annotations
import json
from typing import Dict, Optional

import numpy as np

from . import dsp
from .config import (REFERENCE_NORMS, DOMAINS, DOMAIN_WEIGHTS, QUALITY,
                     METHOD_NOISE_FLOOR, is_resolvable)


def z_scores(features: Dict[str, float], strict: bool = True) -> Dict[str, float]:
    """z-scores orientés.

    En mode strict (défaut), les marqueurs non résolus par la méthode sont
    écartés : leur écart à la norme serait indiscernable de leur propre bruit.

    En mode non strict — destiné aux démonstrations — tous les marqueurs mesurés
    sont notés. Les chiffres restent exacts au sens où ils sortent bien de la
    mesure, mais ceux dont le plancher de bruit dépasse la norme ne permettent
    pas de comparer deux sujets. À réserver à la présentation.
    """
    out = {}
    for k, (mu, sd, sense) in REFERENCE_NORMS.items():
        v = features.get(k, float("nan"))
        keep = True if not strict else is_resolvable(k)
        out[k] = dsp.robust_z(v, mu, sd, sense) if keep else float("nan")
    return out


def unresolved(features: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Marqueurs mesurés mais dont le plancher de bruit dépasse la norme."""
    out = {}
    for k, floor in METHOD_NOISE_FLOOR.items():
        v = features.get(k, float("nan"))
        norm = REFERENCE_NORMS.get(k)
        if norm and np.isfinite(v) and not is_resolvable(k):
            out[k] = {"value": float(v), "noise_floor": floor, "norm_sd": norm[1]}
    return out


def domain_scores(z: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    res = {}
    for domain, weights in DOMAINS.items():
        vals, ws = [], []
        for k, w in weights.items():
            if np.isfinite(z.get(k, float("nan"))):
                vals.append(z[k]); ws.append(w)
        if not vals:
            res[domain] = {"z": float("nan"), "score": float("nan"), "coverage": 0.0}
            continue
        zz = float(np.average(vals, weights=ws))
        res[domain] = {
            "z": zz,
            "score": dsp.z_to_score(zz),
            "coverage": float(sum(ws) / sum(weights.values())),
        }
    return res


def quality_report(features: Dict[str, float], meta: Dict[str, float]) -> Dict[str, object]:
    analyses = meta.get("analyses", [])
    dur = meta.get("duration_s", 0.0)
    flags = []

    if "mouvement" in analyses:
        if meta.get("camera_motion_px", 0.0) > QUALITY["camera_motion_px"]:
            flags.append("Caméra mobile détectée : le mouvement mesuré mêle sujet et caméra.")
        if meta.get("body_mode") == "suivi":
            flags.append("Sujet non segmenté (immobile dès la première image) : "
                         "analyse par suivi d'imagette, silhouette indisponible.")
        elif meta.get("body_detection_rate", 1.0) < QUALITY["body_detection_min"]:
            flags.append("Silhouette détectée par intermittence : cadrer le corps entier "
                         "sur fond contrasté.")
        if meta.get("task") == "marche":
            if features.get("n_steps", 0) < QUALITY["gait_min_steps"]:
                flags.append("Moins de 12 pas exploitables : vitesse indicative seulement.")
            if features.get("n_passes", 0) < QUALITY["gait_min_passes"]:
                flags.append("Un seul trajet rectiligne : allonger l'enregistrement.")
        if not np.isfinite(features.get("px_per_m", float("nan"))):
            flags.append("Aucune échelle fournie : les distances restent en pixels.")

    if "visage" not in analyses:
        grade_src = None
        if flags:
            grade = "C" if len(flags) > 1 else "B"
        else:
            grade = "A"
        return {"grade": grade, "snr_db": float("nan"), "flags": flags}

    snr = features.get("pulse_snr_db", float("-inf"))
    if meta.get("fallback_roi"):
        flags.append("Aucun visage détecté : région centrale utilisée par défaut.")
    if meta.get("detection_rate", 1.0) < 0.6:
        flags.append("Visage détecté par intermittence : recadrer et stabiliser la caméra.")
    if snr < QUALITY["snr_db_usable"]:
        flags.append("Signal de pouls trop bruité pour une lecture fiable.")
    elif snr < QUALITY["snr_db_good"]:
        flags.append("Signal de pouls exploitable mais modéré : améliorer l'éclairage.")
    if dur < 60:
        flags.append("Durée < 60 s : variabilité cardiaque non fiable.")
    if not features.get("hrv_reliable", 0.0):
        flags.append("Critères de fiabilité VFC non réunis (durée, battements ou SNR).")
    if features.get("motion_artifact_px", 0.0) > QUALITY["motion_reject_px"]:
        flags.append("Mouvements importants : indices neuromoteurs peu interprétables.")
    if meta.get("fps", 30) < 24:
        flags.append("Cadence < 24 i/s : bande de tremblement partiellement inaccessible.")

    if snr >= QUALITY["snr_db_good"] and not flags:
        grade = "A"
    elif snr >= QUALITY["snr_db_good"]:
        grade = "B"
    elif snr >= QUALITY["snr_db_usable"]:
        grade = "C"
    else:
        grade = "D"
    return {"grade": grade, "snr_db": snr, "flags": flags}


def load_model(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def apply_model(features: Dict[str, float], model: dict) -> Dict[str, float]:
    """Applique un modèle linéaire calibré (coefficients standardisés)."""
    names = model["features"]
    mu = np.asarray(model["mean"], dtype=float)
    sd = np.asarray(model["std"], dtype=float)
    coef = np.asarray(model["coef"], dtype=float)
    x = np.asarray([features.get(n, np.nan) for n in names], dtype=float)
    filled = np.where(np.isfinite(x), x, mu)
    xs = (filled - mu) / np.where(sd > 0, sd, 1.0)
    pred = float(coef @ xs + model["intercept"])
    return {
        "prediction": pred,
        "target": model.get("target", "inconnu"),
        "missing": [n for n, v in zip(names, x) if not np.isfinite(v)],
        "cv_mae": model.get("cv_mae"),
        "cv_r2": model.get("cv_r2"),
        "n_train": model.get("n_train"),
    }


def build_indices(features: Dict[str, float], meta: Dict[str, float],
                  model: Optional[dict] = None, strict: bool = True) -> Dict[str, object]:
    z = z_scores(features, strict=strict)
    doms = domain_scores(z)

    # Le composite est renormalisé sur les seuls domaines mesurés : un
    # enregistrement de marche ne doit pas être pénalisé pour l'absence de
    # marqueurs dermiques qu'il ne pouvait pas contenir.
    vals, ws = [], []
    for d, w in DOMAIN_WEIGHTS.items():
        sc = doms.get(d, {}).get("score", float("nan"))
        if np.isfinite(sc):
            vals.append(sc); ws.append(w * doms[d]["coverage"])
    composite = float(np.average(vals, weights=ws)) if vals else float("nan")

    out: Dict[str, object] = {
        "z_scores": z,
        "unresolved": unresolved(features) if strict else {},
        "strict": strict,
        "domains": doms,
        "composite_score": composite,
        "composite_coverage": float(sum(ws) / sum(DOMAIN_WEIGHTS.values())) if ws else 0.0,
        "quality": quality_report(features, meta),
        "mode": "heuristique",
    }
    if model:
        out["calibrated"] = apply_model(features, model)
        out["mode"] = "calibré"
    return out
