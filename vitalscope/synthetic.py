"""Génération d'une vidéo synthétique avec vérité terrain connue.

Sert de test de non-régression : on injecte une fréquence cardiaque, une
variabilité, une respiration, un tremblement et un rythme de clignement connus,
puis on vérifie que le pipeline les retrouve. C'est le seul moyen honnête de
valider la chaîne de traitement sans cohorte humaine.

Le visage rendu porte une texture stable (sourcils, arête nasale, bouche, grain
de peau) car un aplat lisse ne représenterait pas le domaine d'entrée réel :
le suivi de micro-mouvement a besoin de structure pour s'accrocher.
"""

from __future__ import annotations
from typing import Dict, Tuple

import cv2
import numpy as np


def _build_face(size: int, rng: np.random.Generator):
    """Construit l'image de visage de référence, son masque de peau et la bande oculaire."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = size / 2.0
    ell = ((xx - cx) / (0.30 * size)) ** 2 + ((yy - cy) / (0.40 * size)) ** 2
    mask = np.clip(1.6 - ell, 0.0, 1.0)

    base = np.array([118.0, 150.0, 196.0], dtype=np.float32)      # BGR peau moyenne
    shading = 0.86 + 0.14 * np.exp(-ell * 1.1)
    img = base[None, None, :] * shading[..., None]

    def band(x0, x1, y0, y1, factor):
        img[int(y0 * size):int(y1 * size), int(x0 * size):int(x1 * size)] *= factor

    band(0.34, 0.46, 0.310, 0.335, 0.45)     # sourcil gauche
    band(0.54, 0.66, 0.310, 0.335, 0.45)     # sourcil droit
    band(0.47, 0.53, 0.380, 0.560, 0.90)     # arête nasale
    band(0.44, 0.56, 0.560, 0.585, 0.72)     # base du nez
    band(0.42, 0.58, 0.680, 0.705, 0.60)     # bouche
    band(0.28, 0.36, 0.600, 0.660, 1.05)     # pommette gauche
    band(0.64, 0.72, 0.600, 0.660, 1.05)     # pommette droite

    grain = rng.normal(0.0, 3.0, (size, size)).astype(np.float32)
    img = img + cv2.GaussianBlur(grain, (0, 0), 1.1)[..., None]

    img = img * mask[..., None] + 34.0 * (1.0 - mask[..., None])
    eye_box = (int(0.32 * size), int(0.345 * size), int(0.36 * size), int(0.06 * size))
    return img.astype(np.float32), mask.astype(np.float32), eye_box


def make_video(path: str, duration_s: float = 60.0, fps: float = 30.0,
               hr_bpm: float = 66.0, hrv_sd_ms: float = 45.0,
               resp_cpm: float = 15.0, tremor_hz: float = 8.0,
               blink_per_min: float = 15.0,
               pulse_amplitude: float = 0.014, noise: float = 0.003,
               size: int = 320, seed: int = 7) -> Dict[str, float]:
    """Écrit une vidéo et retourne les paramètres de vérité terrain."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fps)
    t = np.arange(n) / fps

    # --- battements avec arythmie sinusale respiratoire + bruit ---
    ibi = 60.0 / hr_bpm
    beats, cur = [0.0], 0.0
    while cur < duration_s + 2 * ibi:
        rsa = 0.6 * hrv_sd_ms / 1000.0 * np.sin(2 * np.pi * resp_cpm / 60.0 * cur)
        cur += float(np.clip(ibi + rsa + rng.normal(0, hrv_sd_ms / 1000.0 * 0.6), 0.35, 2.0))
        beats.append(cur)
    beats = np.asarray(beats)
    phase = np.interp(t, beats, np.arange(len(beats)))
    frac = phase - np.floor(phase)

    # Onde de pouls : pic systolique puis onde dicrote
    wave = (np.exp(-((frac - 0.22) ** 2) / 0.010)
            + 0.35 * np.exp(-((frac - 0.48) ** 2) / 0.020))
    wave = (wave - wave.mean()) / (wave.std() + 1e-9)

    resp = np.sin(2 * np.pi * resp_cpm / 60.0 * t)
    tremor = np.sin(2 * np.pi * tremor_hz * t + rng.uniform(0, 6.28))

    face, mask, (ex, ey, ew, eh) = _build_face(size, rng)
    mask3 = mask[..., None]

    blink_period = fps * 60.0 / blink_per_min
    blink_frames, k = set(), 0
    while k * blink_period < n:
        start = int(k * blink_period + rng.integers(-8, 9))
        for f in range(max(0, start), min(n, start + max(2, int(0.13 * fps)))):
            blink_frames.add(f)
        k += 1

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    if not writer.isOpened():
        raise IOError("Encodeur vidéo indisponible.")

    for i in range(n):
        dx = 0.9 * resp[i] + 0.35 * tremor[i] + rng.normal(0, 0.04)
        dy = 1.6 * resp[i] + 0.30 * tremor[i] + rng.normal(0, 0.04)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(face, m, (size, size), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
        skin = cv2.warpAffine(mask3, m, (size, size), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        skin = skin[..., None] if skin.ndim == 2 else skin

        amp = pulse_amplitude * wave[i]
        gains = np.array([1 + 0.25 * amp, 1 + 1.00 * amp, 1 + 0.45 * amp], dtype=np.float32)
        drift = 1.0 + 0.02 * np.sin(2 * np.pi * 0.01 * t[i])       # dérive d'éclairage lente
        frame = frame * (1.0 + (gains[None, None, :] * drift - 1.0) * skin)

        y0, y1 = ey + int(dy), ey + eh + int(dy)
        x0, x1 = ex + int(dx), ex + ew + int(dx)
        frame[max(0, y0):y1, max(0, x0):x1] *= 1.45 if i in blink_frames else 0.62

        frame = frame + rng.normal(0, noise * 255, (size, size, 3))
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))

    writer.release()
    d = np.diff(beats)
    return {
        "path": path, "duration_s": duration_s, "fps": fps,
        "true_hr_bpm": float(60.0 / np.mean(d)),
        "true_hrv_sdnn_ms": float(np.std(d, ddof=1) * 1000.0),
        "true_hrv_rmssd_ms": float(np.sqrt(np.mean(np.diff(d) ** 2)) * 1000.0),
        "true_resp_cpm": float(resp_cpm),
        "true_tremor_hz": float(tremor_hz),
        "true_blink_rate_min": float(blink_per_min),
        "n_beats": int(len(beats)),
    }
