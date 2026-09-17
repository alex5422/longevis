"""Chaîne complète : vidéo → traces → biomarqueurs → indices.

Deux modes d'analyse, combinables :
  * `visage`   — rPPG, respiration, oculomoteur, dermique (plan rapproché)
  * `mouvement` — marche, équilibre, transferts (plan large)

Le mode `auto` essaie le mouvement d'abord ; si aucun corps entier n'est
détecté, il bascule sur l'analyse faciale.
"""

from __future__ import annotations
import time
from typing import Dict, Optional

import numpy as np

from . import body, cardio, derma, elasticite, gait, index, neuro, rppg, sollicitation, tonus, video
from .config import DEFAULT, ProcessingConfig


def analyze_face(path: str, cfg: ProcessingConfig = DEFAULT) -> Dict[str, object]:
    tr = video.extract_traces(path, cfg)
    pulses = rppg.extract_pulse(tr.rgb, tr.fps, cfg)
    best = rppg.best_pulse(pulses)

    features: Dict[str, float] = {}
    features.update(cardio.cardiovascular_features(best, tr.rgb, tr.duration_s))
    features.update(neuro.respiration(tr.head_xy, tr.fps, best.signal))
    features.update(neuro.neuromotor(tr.head_xy, tr.face_size, tr.fps))
    features.update(neuro.oculomotor(tr.eye_signal, tr.fps, tr.duration_s))
    features.update(derma.dermal_features(tr.skin_patches))
    features["skin_fraction_mean"] = float(np.nanmean(tr.skin_fraction))

    meta = {"fps": tr.fps, "n_frames": tr.n_frames, "duration_s": tr.duration_s,
            "detection_rate": tr.detection_rate, "fallback_roi": tr.fallback_roi,
            "best_method": best.method}
    signals = {"pulse": best.signal, "fps": tr.fps, "head": tr.head_xy}
    methods = {k: {"hr_bpm": p.hr_bpm, "snr_db": p.snr_db,
                   "spectral_purity": p.spectral_purity} for k, p in pulses.items()}
    return {"features": features, "meta": meta, "signals": signals, "methods": methods}


def analyze_body(path: str, task: str = "auto",
                 subject_height_m: Optional[float] = None,
                 px_per_m: Optional[float] = None) -> Dict[str, object]:
    b = body.extract_body(path)
    res = gait.analyze_motion(b, task=task, subject_height_m=subject_height_m,
                              px_per_m=px_per_m)
    meta = {"fps": b.fps, "n_frames": b.n_frames, "duration_s": b.duration_s,
            "body_detection_rate": b.detection_rate, "body_mode": b.mode,
            "camera_motion_px": b.camera_motion_px, "task": res["task"],
            "frame_size": tuple(b.frame_size)}
    return {"features": res["features"], "meta": meta,
            "signals": res["signals"], "segments": res["segments"], "traces": b}


def _analyze_geste(path: str, analyze_fn,
                   subject_height_m: Optional[float] = None,
                   px_per_m: Optional[float] = None) -> Dict[str, object]:
    """Squelette commun aux tests tonus/sollicitation/élasticité : même
    silhouette extraite (`body.extract_body`), même échelle pixels-par-mètre
    qu'`analyze_body` (distance connue si fournie, sinon stature déclarée),
    analyse spécifique déléguée à `analyze_fn(b, px_per_m)`, mêmes clés de
    retour qu'`analyze_body` pour un affichage uniforme."""
    b = body.extract_body(path)
    scale = px_per_m
    if scale is None and subject_height_m:
        hh = b.height_px[np.isfinite(b.height_px)]
        if hh.size:
            stature_px = float(np.percentile(hh, 97))
            scale = stature_px / subject_height_m
    res = analyze_fn(b, scale)
    meta = {"fps": b.fps, "n_frames": b.n_frames, "duration_s": b.duration_s,
            "body_detection_rate": b.detection_rate, "body_mode": b.mode,
            "camera_motion_px": b.camera_motion_px, "task": res["task"],
            "frame_size": tuple(b.frame_size), "px_per_m": float(scale) if scale else float("nan")}
    return {"features": res["features"], "meta": meta,
            "signals": res["signals"], "segments": res["segments"], "traces": b}


def analyze_tonus(path: str, subject_height_m: Optional[float] = None,
                  px_per_m: Optional[float] = None) -> Dict[str, object]:
    """Gainage chronométré filmé de profil — hors score Kinexa."""
    return _analyze_geste(path, tonus.analyze_tonus, subject_height_m, px_per_m)


def analyze_sollicitation(path: str, subject_height_m: Optional[float] = None,
                          px_per_m: Optional[float] = None) -> Dict[str, object]:
    """Petits sauts talon filmés — hors score Kinexa."""
    return _analyze_geste(path, sollicitation.analyze_sollicitation, subject_height_m, px_per_m)


def analyze_elasticite(path: str, subject_height_m: Optional[float] = None,
                       px_per_m: Optional[float] = None) -> Dict[str, object]:
    """Flexion/étirement filmé de face — hors score Kinexa."""
    return _analyze_geste(path, elasticite.analyze_elasticite, subject_height_m, px_per_m)


def analyze_equilibre(path: str, subject_height_m: Optional[float] = None,
                      px_per_m: Optional[float] = None) -> Dict[str, object]:
    """Appui unipodal chronométré, filmé de face — hors score Kinexa.

    Réutilise directement `analyze_body` avec la tâche « posture » forcée :
    le moteur de mesure de l'oscillation posturale existe déjà dans
    `gait.postural_sway`, seule l'auto-détection de tâche l'empêchait
    d'être sollicité de façon fiable depuis une vidéo dédiée."""
    return analyze_body(path, task="posture", subject_height_m=subject_height_m,
                        px_per_m=px_per_m)


def analyze_transfert(path: str, subject_height_m: Optional[float] = None,
                      px_per_m: Optional[float] = None) -> Dict[str, object]:
    """Levers de chaise chronométrés, filmés de profil — hors score Kinexa."""
    return analyze_body(path, task="leve", subject_height_m=subject_height_m,
                        px_per_m=px_per_m)


def analyze_mouvement_libre(path: str, subject_height_m: Optional[float] = None,
                            px_per_m: Optional[float] = None) -> Dict[str, object]:
    """Geste répété au choix (gymnastique, tai-chi, rééducation), filmé de
    face ou de profil — hors score Kinexa."""
    return analyze_body(path, task="mouvement", subject_height_m=subject_height_m,
                        px_per_m=px_per_m)


def analyze(path: str, mode: str = "auto", cfg: ProcessingConfig = DEFAULT,
            model: Optional[dict] = None, task: str = "auto",
            subject_height_m: Optional[float] = None,
            px_per_m: Optional[float] = None,
            strict: bool = True) -> Dict[str, object]:
    """`strict=False` note tous les marqueurs mesurés, y compris ceux dont le
    bruit de méthode dépasse la dispersion entre individus. Mode démonstration."""
    t0 = time.time()
    features: Dict[str, object] = {}
    meta: Dict[str, object] = {"file": path, "mode": mode}
    signals: Dict[str, object] = {}
    methods: Dict[str, object] = {}
    segments = None
    parts = []

    want_body = mode in ("mouvement", "auto", "complet")
    want_face = mode in ("visage", "complet") or mode == "auto"

    if want_body:
        try:
            rb = analyze_body(path, task, subject_height_m, px_per_m)
            body_ok = (rb["meta"]["body_detection_rate"] > 0.2
                       or rb["meta"]["body_mode"] == "suivi")
            if body_ok:
                features.update(rb["features"])
                meta.update(rb["meta"])
                signals.update({f"body_{k}": v for k, v in rb["signals"].items()})
                segments = rb["segments"]
                parts.append("mouvement")
                if mode == "auto":
                    want_face = False        # plan large : le visage est trop petit
        except (IOError, ValueError):
            pass

    if want_face:
        rf = analyze_face(path, cfg)
        features.update(rf["features"])
        meta.update({k: v for k, v in rf["meta"].items() if k not in meta})
        signals.update(rf["signals"])
        methods = rf["methods"]
        parts.append("visage")

    meta["analyses"] = parts
    meta["processing_s"] = round(time.time() - t0, 2)
    meta.setdefault("best_method", "—")

    numeric = {k: v for k, v in features.items() if isinstance(v, (int, float))}
    indices = index.build_indices(numeric, meta, model, strict=strict)

    return {"meta": meta, "features": features, "indices": indices,
            "methods": methods, "segments": segments, "_signals": signals}
