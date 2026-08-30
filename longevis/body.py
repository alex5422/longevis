"""Extraction des trajectoires corporelles par silhouette.

Aucun modèle de pose n'est requis : la segmentation avant-plan suffit à obtenir
les signaux qui portent l'information de mouvement (centre de masse, hauteur,
écartement des jambes). C'est un choix délibéré — la chaîne tourne hors ligne,
sans téléchargement de poids, et reste vérifiable ligne à ligne.

Contrepartie assumée : la caméra doit être fixe. Un mouvement de caméra est
détecté et signalé plutôt que silencieusement confondu avec du mouvement du
sujet.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .video import TemplateTracker


@dataclass
class BodyTraces:
    fps: float
    n_frames: int
    duration_s: float
    centroid: np.ndarray        # (N,2) centre de masse de la silhouette, px
    bbox: np.ndarray            # (N,4) x, y, largeur, hauteur
    height_px: np.ndarray       # (N,) hauteur de la silhouette
    leg_spread: np.ndarray      # (N,) largeur du quart inférieur (écartement des pieds)
    trunk_y: np.ndarray         # (N,) ordonnée du centre de masse du tronc seul
    area: np.ndarray            # (N,) surface de la silhouette
    foot_y: np.ndarray          # (N,) ordonnée du point le plus bas
    valid: np.ndarray           # (N,) bool : silhouette exploitable
    track_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    track_confidence: float = 0.0
    camera_motion_px: float = 0.0
    detection_rate: float = 0.0
    mode: str = "silhouette"
    frame_size: Tuple[int, int] = (0, 0)
    preview: Optional[np.ndarray] = None


def _largest_person(mask: np.ndarray, min_area: int) -> Optional[np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < min_area:
        return None
    return (labels == idx).astype(np.uint8) * 255


def _camera_motion(prev_gray: np.ndarray, gray: np.ndarray, fg: np.ndarray) -> float:
    """Déplacement du fond, hors zone du sujet : trahit une caméra mobile."""
    bg = cv2.bitwise_not(cv2.dilate(fg, np.ones((15, 15), np.uint8)))
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=60, qualityLevel=0.06,
                                  minDistance=12, mask=bg)
    if pts is None or len(pts) < 6:
        return 0.0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21))
    if nxt is None or st is None or st.sum() < 4:
        return 0.0
    d = nxt[st.ravel() == 1].reshape(-1, 2) - pts[st.ravel() == 1].reshape(-1, 2)
    return float(np.median(np.hypot(d[:, 0], d[:, 1])))


def _hog_person(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Détecteur de piéton HOG livré avec OpenCV.

    Indispensable pour un sujet immobile : la soustraction de fond l'apprend
    comme décor et ne le voit jamais. Le HOG, lui, reconnaît une silhouette
    debout sans rien télécharger.
    """
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    scale = 640.0 / max(frame.shape[1], 1)
    small = cv2.resize(frame, None, fx=min(1.0, scale), fy=min(1.0, scale)) \
        if scale < 1.0 else frame
    rects, weights = hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8),
                                          scale=1.05)
    if len(rects) == 0:
        return None
    i = int(np.argmax(weights))
    x, y, w, h = rects[i]
    f = 1.0 / min(1.0, scale)
    # Le HOG cadre large : on resserre de 15 % latéralement.
    return (int((x + 0.15 * w) * f), int(y * f), int(0.70 * w * f), int(h * f))


def extract_body(path: str, max_frames: int = 12000,
                 warmup: int = 25) -> BodyTraces:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Vidéo illisible : {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1.0 < fps < 240.0):
        fps = 30.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:
        # Le modèle de fond a besoin de quelques images, mais sur une vidéo
        # courte un amorçage fixe de 25 images en gâche une part notable.
        warmup = int(min(warmup, max(4, total // 12)))
    sub = cv2.createBackgroundSubtractorMOG2(history=350, varThreshold=28,
                                             detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cent, bbox, hgt, spread, area, foot, valid = [], [], [], [], [], [], []
    trunk = []
    track: List[Tuple[float, float]] = []
    torso = TemplateTracker(margin=24)
    torso_box: Optional[Tuple[int, int, int, int]] = None
    hog_box: Optional[Tuple[int, int, int, int]] = None
    mode = "silhouette"
    cam_moves: List[float] = []
    prev_gray = None
    preview = None
    idx = 0
    H = W = 0

    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if H == 0:
            H, W = frame.shape[:2]
            min_area = int(0.004 * H * W)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Après amorçage, le fond n'est presque plus mis à jour : sans cela un
        # sujet immobile (test d'équilibre) serait absorbé par le modèle en
        # une dizaine de secondes et disparaîtrait de la segmentation.
        fg = sub.apply(frame, learningRate=-1 if idx < warmup else 0.0006)
        fg[fg < 200] = 0                       # écarte les ombres (valeur 127)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=3)

        person = _largest_person(fg, min_area) if idx >= warmup else None

        # Sujet jamais segmenté après l'amorçage : on tente le détecteur HOG
        # une fois, et l'analyse bascule sur le suivi d'imagette.
        if (person is None and torso_box is None and mode == "silhouette"
                and idx == warmup + 15):
            hb = _hog_person(frame)
            if hb is not None:
                torso_box = hb
                mode = "suivi"
                hog_box = hb

        if person is None:
            cent.append((np.nan, np.nan)); bbox.append((np.nan,) * 4); trunk.append(np.nan)
            hgt.append(np.nan); spread.append(np.nan); area.append(np.nan)
            foot.append(np.nan); valid.append(False)
        else:
            ys, xs = np.nonzero(person)
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
            h = y1 - y0 + 1
            cent.append((float(xs.mean()), float(ys.mean())))
            bbox.append((float(x0), float(y0), float(x1 - x0 + 1), float(h)))
            hgt.append(float(h))
            # Écartement des jambes : largeur du quart inférieur de la silhouette.
            # Les bras perturbent la largeur globale, pas celle-ci.
            # Tronc : moitié supérieure de la silhouette. Le centre de masse du
            # corps entier est contaminé par le balancement des jambes, qui y
            # réinjecte le rythme du pas — mesurer le couplage tronc/jambes sur
            # ce signal reviendrait à corréler les jambes avec elles-mêmes.
            haut = ys <= (y0 + 0.45 * h)
            trunk.append(float(ys[haut].mean()) if haut.any() else float(ys.mean()))
            band = ys >= (y1 - 0.25 * h)
            spread.append(float(xs[band].max() - xs[band].min() + 1) if band.any() else np.nan)
            area.append(float(person.sum() / 255.0))
            foot.append(float(y1))
            valid.append(True)
            if preview is None and idx > warmup + 5:
                vue = frame.copy()
                cv2.rectangle(vue, (x0, y0), (x1, y1), (40, 220, 60), 2)
                bande = int(y1 - 0.25 * h)
                cv2.line(vue, (x0, bande), (x1, bande), (60, 160, 255), 1)
                cv2.putText(vue, "silhouette detectee", (x0, max(16, y0 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 60), 1,
                            cv2.LINE_AA)
                preview = vue
            if torso_box is None and idx > warmup + 5 and h > 0.2 * H:
                torso_box = (x0, y0, x1 - x0 + 1, h)

        # Suivi du torse au sous-pixel : indispensable pour l'équilibre, où
        # l'amplitude à mesurer est inférieure au pixel de la silhouette.
        if torso_box is not None:
            track.append(torso.update(gray, torso_box))
        else:
            track.append((np.nan, np.nan))

        if prev_gray is not None and idx % 5 == 0:
            cam_moves.append(_camera_motion(prev_gray, gray, fg))
        prev_gray = gray
        idx += 1

    cap.release()
    if idx == 0:
        raise IOError("Aucune image décodée.")

    v = np.asarray(valid, dtype=bool)
    hgt_a = np.asarray(hgt, dtype=float)
    if hog_box is not None and not np.isfinite(hgt_a).any():
        hgt_a = np.full(idx, float(hog_box[3]))      # stature issue du HOG
    return BodyTraces(
        fps=float(fps), n_frames=idx, duration_s=idx / float(fps),
        centroid=np.asarray(cent, dtype=float), bbox=np.asarray(bbox, dtype=float),
        height_px=hgt_a, leg_spread=np.asarray(spread, dtype=float),
        area=np.asarray(area, dtype=float), foot_y=np.asarray(foot, dtype=float),
        trunk_y=np.asarray(trunk, dtype=float),
        valid=v,
        camera_motion_px=float(np.median(cam_moves)) if cam_moves else 0.0,
        detection_rate=float(v.mean()), mode=mode,
        track_xy=np.asarray(track, dtype=float),
        track_confidence=float(np.median(torso.confidence)) if torso.confidence else 0.0,
        frame_size=(W, H), preview=preview,
    )
