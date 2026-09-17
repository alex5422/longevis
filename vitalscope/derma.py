"""Biomarqueurs dermiques mesurés sur les imagettes de joue.

Séparation des chromophores par l'approche log-opposant (Tsumura et al., 1999,
version simplifiée sans calibration spectrale) : les indices mélanine et
hémoglobine obtenus sont *relatifs*, non des concentrations absolues.
"""

from __future__ import annotations
from typing import Dict, List

import cv2
import numpy as np


def _standardize(patch: np.ndarray, size: int = 128) -> np.ndarray:
    p = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return p.astype(np.float32) + 1.0     # évite log(0)


def _wrinkle_index(gray: np.ndarray) -> float:
    """Énergie des hautes fréquences spatiales / énergie totale.

    Les rides et le grain de peau concentrent l'énergie dans les échelles fines
    de la pyramide laplacienne ; l'éclairage et le modelé du visage occupent les
    échelles grossières, qui sont donc écartées du numérateur.
    """
    g = gray / (gray.mean() + 1e-6)
    cur = g.copy()
    energies = []
    for _ in range(3):
        down = cv2.pyrDown(cur)
        up = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
        energies.append(float(np.mean((cur - up) ** 2)))
        cur = down
    total = sum(energies) + float(np.var(cur)) + 1e-9
    return float((energies[0] + energies[1]) / total)


def _directional_anisotropy(gray: np.ndarray) -> float:
    """Anisotropie du gradient : les rides installées sont orientées,
    le grain de peau jeune est isotrope."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    m = mag > np.percentile(mag, 75)
    if m.sum() < 50:
        return float("nan")
    ang = np.arctan2(gy[m], gx[m]) * 2.0          # orientation modulo pi
    r = np.hypot(np.mean(np.cos(ang)), np.mean(np.sin(ang)))
    return float(r)                                # 0 = isotrope, 1 = aligné


def _chromophores(bgr: np.ndarray) -> Dict[str, float]:
    b, g, r = cv2.split(bgr)
    log_r, log_g = np.log(r), np.log(g)
    # Densités optiques : le rouge est peu absorbé par l'hémoglobine, il porte
    # donc surtout la mélanine ; l'écart vert-rouge isole la composante sanguine.
    melanin = np.log(255.0) - log_r
    hemoglobin = (np.log(255.0) - log_g) - melanin
    erythema = float(np.mean(hemoglobin))
    return {
        "melanin_index": float(np.mean(melanin)),
        "erythema_index": float(np.clip(erythema, -5, 5)),
        "melanin_heterogeneity": float(np.std(melanin)),
    }


def dermal_features(patches: List[np.ndarray]) -> Dict[str, float]:
    if not patches:
        return {k: float("nan") for k in
                ("wrinkle_index", "texture_anisotropy", "tone_evenness",
                 "melanin_index", "erythema_index", "n_skin_patches")}

    wr, an, ev, mel, ery, het = [], [], [], [], [], []
    for p in patches:
        std = _standardize(p)
        gray = cv2.cvtColor(std, cv2.COLOR_BGR2GRAY)
        wr.append(_wrinkle_index(gray))
        an.append(_directional_anisotropy(gray))
        ch = _chromophores(std)
        mel.append(ch["melanin_index"])
        ery.append(ch["erythema_index"])
        het.append(ch["melanin_heterogeneity"])
        # Homogénéité du teint : 1 - dispersion relative de la luminance basse fréquence
        low = cv2.GaussianBlur(gray, (0, 0), 6)
        ev.append(float(1.0 - np.clip(np.std(low) / (np.mean(low) + 1e-6) * 4.0, 0, 1)))

    def _m(v):
        v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
        return float(np.median(v)) if v.size else float("nan")

    return {
        "wrinkle_index": _m(wr),
        "texture_anisotropy": _m(an),
        "tone_evenness": _m(ev),
        "melanin_index": _m(mel),
        "melanin_heterogeneity": _m(het),
        "erythema_index": _m(ery),
        "n_skin_patches": float(len(patches)),
    }
