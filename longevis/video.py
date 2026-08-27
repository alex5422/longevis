"""Extraction des traces brutes à partir de la vidéo.

Sorties : traces RGB par région d'intérêt, signal de mouvement de tête,
signal oculaire, imagettes de peau pour l'analyse de texture.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import ProcessingConfig, DEFAULT


@dataclass
class Traces:
    fps: float
    n_frames: int
    duration_s: float
    rgb: Dict[str, np.ndarray]            # ROI -> (N,3)
    skin_fraction: np.ndarray             # (N,) fraction de pixels peau du visage
    head_xy: np.ndarray                   # (N,2) déplacement cumulé de la tête (px)
    eye_signal: np.ndarray                # (N,) intensité moyenne région oculaire
    face_size: np.ndarray                 # (N,) diagonale de la boîte visage
    skin_patches: List[np.ndarray] = field(default_factory=list)
    detection_rate: float = 0.0
    fallback_roi: bool = False
    track_confidence: float = 0.0
    track_sharpness: float = 0.0


def _skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Masque de peau YCrCb + HSV (intersection), robuste aux phototypes clairs à foncés."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m1 = cv2.inRange(ycrcb, (0, 133, 77), (255, 180, 132))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m2 = cv2.inRange(hsv, (0, 20, 40), (25, 255, 255))
    m3 = cv2.inRange(hsv, (160, 20, 40), (180, 255, 255))
    mask = cv2.bitwise_and(m1, cv2.bitwise_or(m2, m3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def _sub_rois(x: int, y: int, w: int, h: int) -> Dict[str, Tuple[int, int, int, int]]:
    """Régions anatomiques dérivées de la boîte visage (proportions canoniques)."""
    return {
        "front":       (x + int(0.30 * w), y + int(0.10 * h), int(0.40 * w), int(0.15 * h)),
        "joue_gauche": (x + int(0.10 * w), y + int(0.52 * h), int(0.22 * w), int(0.22 * h)),
        "joue_droite": (x + int(0.68 * w), y + int(0.52 * h), int(0.22 * w), int(0.22 * h)),
        "visage":      (x, y, w, h),
    }


def _eye_box(x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    return (x + int(0.20 * w), y + int(0.28 * h), int(0.60 * w), int(0.16 * h))


def _crop(img: np.ndarray, box) -> np.ndarray:
    bx, by, bw, bh = box
    H, W = img.shape[:2]
    bx, by = max(0, bx), max(0, by)
    bw, bh = max(1, min(bw, W - bx)), max(1, min(bh, H - by))
    return img[by:by + bh, bx:bx + bw]


class FaceTracker:
    """Détection Haar périodique + lissage exponentiel de la boîte.

    Si aucun visage n'est jamais détecté (vidéo synthétique, cadrage serré),
    bascule sur une région centrale fixe et le signale via `fallback`.
    """

    def __init__(self, cfg: ProcessingConfig = DEFAULT):
        self.cfg = cfg
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(path)
        self.box: Optional[Tuple[int, int, int, int]] = None
        self.hits = 0
        self.calls = 0
        self.fallback = False

    def update(self, gray: np.ndarray, frame_idx: int) -> Tuple[int, int, int, int]:
        H, W = gray.shape[:2]
        if frame_idx % self.cfg.detect_every_n_frames == 0 or self.box is None:
            self.calls += 1
            faces = self.cascade.detectMultiScale(
                cv2.equalizeHist(gray), scaleFactor=1.15, minNeighbors=5,
                minSize=(int(0.12 * min(H, W)), int(0.12 * min(H, W))))
            if len(faces) > 0:
                self.hits += 1
                fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                new = (float(fx), float(fy), float(fw), float(fh))
                if self.box is None:
                    self.box = new
                else:
                    a = self.cfg.bbox_smoothing
                    self.box = tuple(a * o + (1 - a) * n for o, n in zip(self.box, new))
        if self.box is None:
            self.fallback = True
            side = int(0.55 * min(H, W))
            self.box = (float((W - side) // 2), float((H - side) // 2), float(side), float(side))
        return tuple(int(round(v)) for v in self.box)


class TemplateTracker:
    """Déplacement absolu de la tête par appariement de gabarit.

    Un gabarit prélevé sur la première image est recherché dans une fenêtre
    étroite des images suivantes. Contrairement à l'intégration du flux optique,
    la mesure ne dérive pas : chaque image est comparée à la même référence.
    Le maximum de corrélation est affiné au sous-pixel par ajustement
    parabolique, ce qui donne accès aux micro-mouvements de l'ordre de 0,05 px.
    """

    def __init__(self, margin: int = 20):
        self.margin = margin
        self.template: Optional[np.ndarray] = None
        self.anchor: Optional[Tuple[int, int]] = None   # coin haut-gauche du gabarit
        self.offset = np.zeros(2, dtype=float)          # cumul des ré-ancrages
        self.side = 0
        self.texture_energy = 0.0
        self.confidence: List[float] = []
        self.sharpness: List[float] = []
        self.last_pos = np.zeros(2, dtype=float)

    @staticmethod
    def _subpixel(res: np.ndarray, loc: Tuple[int, int]) -> Tuple[float, float]:
        cx, cy = loc
        dx = dy = 0.0
        if 0 < cx < res.shape[1] - 1:
            l, c, r = res[cy, cx - 1], res[cy, cx], res[cy, cx + 1]
            den = l - 2 * c + r
            if abs(den) > 1e-9:
                dx = 0.5 * (l - r) / den
        if 0 < cy < res.shape[0] - 1:
            u, c, d = res[cy - 1, cx], res[cy, cx], res[cy + 1, cx]
            den = u - 2 * c + d
            if abs(den) > 1e-9:
                dy = 0.5 * (u - d) / den
        return float(np.clip(dx, -1, 1)), float(np.clip(dy, -1, 1))

    def update(self, gray: np.ndarray, face_box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        H, W = gray.shape[:2]
        x, y, w, h = face_box
        if self.template is None:
            self.side = int(max(40, min(0.62 * w, 0.62 * h)))
            # On écarte la bande oculaire (0.25–0.48 h) : les clignements y
            # modifient le gabarit et feraient sauter l'appariement.
            candidates = []
            for fy in (0.44, 0.52, 0.60):
                for fx in (0.10, 0.50, 0.90):
                    cx = x + fx * w - self.side / 2
                    cy = y + fy * h - self.side / 2
                    ax = int(np.clip(cx, self.margin, max(self.margin, W - self.side - self.margin)))
                    ay = int(np.clip(cy, self.margin, max(self.margin, H - self.side - self.margin)))
                    patch = gray[ay:ay + self.side, ax:ax + self.side]
                    if patch.shape[0] < self.side or patch.shape[1] < self.side:
                        continue
                    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
                    candidates.append((float(np.mean(np.hypot(gx, gy))), ax, ay))
            if not candidates:
                return (0.0, 0.0)
            self.texture_energy, ax, ay = max(candidates)
            self.anchor = (ax, ay)
            self.template = gray[ay:ay + self.side, ax:ax + self.side].copy()
            self.confidence.append(1.0)
            return (0.0, 0.0)

        ax, ay = self.anchor
        m = self.margin
        x0, y0 = max(0, ax - m), max(0, ay - m)
        x1, y1 = min(W, ax + self.side + m), min(H, ay + self.side + m)
        window = gray[y0:y1, x0:x1]
        if window.shape[0] <= self.side or window.shape[1] <= self.side:
            return tuple(self.offset)

        res = cv2.matchTemplate(window, self.template, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        # Netteté du pic : un plateau de corrélation signale une texture trop
        # faible pour mesurer un micro-mouvement — on le signale au lieu de
        # produire un chiffre qui ne serait que du bruit.
        sharp = float((maxv - res.mean()) / (res.std() + 1e-9))
        self.sharpness.append(sharp)
        if maxv < 0.35 or sharp < 2.0:
            self.confidence.append(float(maxv))
            return (float(self.last_pos[0]), float(self.last_pos[1]))
        sx, sy = self._subpixel(res, maxloc)
        dx = (x0 + maxloc[0] + sx) - ax
        dy = (y0 + maxloc[1] + sy) - ay
        self.confidence.append(float(maxv))

        pos = self.offset + np.array([dx, dy])
        # Ré-ancrage si la tête s'approche du bord de la fenêtre de recherche
        if max(abs(dx), abs(dy)) > 0.75 * m:
            nax = int(np.clip(ax + round(dx), 0, W - self.side))
            nay = int(np.clip(ay + round(dy), 0, H - self.side))
            self.template = gray[nay:nay + self.side, nax:nax + self.side].copy()
            self.offset = pos.copy()
            self.anchor = (nax, nay)
        self.last_pos = pos
        return (float(pos[0]), float(pos[1]))


def extract_traces(path: str, cfg: ProcessingConfig = DEFAULT,
                   progress: bool = False) -> Traces:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Vidéo illisible : {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.target_fps
    if not (1.0 < fps < 240.0):
        fps = cfg.target_fps

    tracker = FaceTracker(cfg)
    roi_names = ["front", "joue_gauche", "joue_droite", "visage"]
    rgb = {k: [] for k in roi_names}
    skin_frac, eye_sig, face_sz = [], [], []
    head_xy = [(0.0, 0.0)]
    patches: List[np.ndarray] = []

    tracker_t = TemplateTracker()
    idx = 0

    while idx < cfg.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = tracker.update(gray, idx)
        face_sz.append(float(np.hypot(w, h)))
        rois = _sub_rois(x, y, w, h)

        # --- traces RGB sur pixels de peau ---
        total_skin, total_px = 0, 0
        for name in roi_names:
            patch = _crop(frame, rois[name])
            if patch.size == 0:
                rgb[name].append((np.nan,) * 3)
                continue
            mask = _skin_mask(patch)
            n_skin = int(np.count_nonzero(mask))
            if name == "visage":
                total_skin, total_px = n_skin, mask.size
            if n_skin >= cfg.min_skin_pixels:
                mean = cv2.mean(patch, mask=mask)[:3]          # BGR
            elif patch.shape[0] * patch.shape[1] >= 25:
                mean = cv2.mean(patch)[:3]                      # repli : ROI entière
            else:
                mean = (np.nan, np.nan, np.nan)
            rgb[name].append((mean[2], mean[1], mean[0]))       # -> RGB
        skin_frac.append(total_skin / max(1, total_px))

        # --- signal oculaire ---
        eye = _crop(gray, _eye_box(x, y, w, h))
        eye_sig.append(float(eye.mean()) if eye.size else np.nan)

        # --- mouvement de tête (appariement de gabarit, sous-pixel) ---
        head_xy.append(tracker_t.update(gray, (x, y, w, h)))

        # --- imagettes de peau pour la texture ---
        if len(patches) < cfg.texture_samples and idx % max(1, int(fps)) == 0:
            cheek = _crop(frame, rois["joue_droite"])
            if cheek.shape[0] >= 24 and cheek.shape[1] >= 24:
                patches.append(cheek.copy())

        idx += 1

    cap.release()
    if idx == 0:
        raise IOError("Aucune image décodée.")

    head = np.asarray(head_xy[:idx], dtype=float)
    head = head - np.median(head, axis=0)
    return Traces(
        fps=float(fps), n_frames=idx, duration_s=idx / float(fps),
        rgb={k: np.asarray(v, dtype=float) for k, v in rgb.items()},
        skin_fraction=np.asarray(skin_frac, dtype=float),
        head_xy=head, eye_signal=np.asarray(eye_sig, dtype=float),
        face_size=np.asarray(face_sz, dtype=float),
        skin_patches=patches,
        detection_rate=(tracker.hits / tracker.calls) if tracker.calls else 0.0,
        fallback_roi=tracker.fallback,
        track_confidence=float(np.median(tracker_t.confidence)) if tracker_t.confidence else 0.0,
        track_sharpness=float(np.median(tracker_t.sharpness)) if tracker_t.sharpness else 0.0,
    )
