"""Figures de l'écran Kinexa — SVG écrits à la main, sans dépendance.

Trois lectures visuelles, dans l'esprit du reste de l'interface : verre sombre,
chiffres serrés, une couleur d'accent par famille.

  courbe_age   la courbe de vitesse par décennie, et le sujet posé dessus
  radar        cinq axes de vitalité, du centre (bas) au bord (haut)
  frise        la séquence : trajets, demi-tours, arrêts, dans le temps

Aucune de ces figures n'invente de donnée : ce qui manque n'est pas dessiné.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ACCENT = "#F97316"
BLEU = "#7AA2FF"
TURQ = "#14D6C4"
VIOLET = "#C86BFF"
GRIS = "#6C6C7A"
CLAIR = "#F5F5F7"
DIM = "#9A9AA8"


def _txt(x, y, s, taille=11, couleur=DIM, ancre="start", graisse=400, espace=".02em"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{couleur}" font-size="{taille}" '
            f'font-weight="{graisse}" text-anchor="{ancre}" letter-spacing="{espace}" '
            f'font-family="Inter Tight,-apple-system,Segoe UI,sans-serif">{s}</text>')


def _cadre(w: int, h: int, contenu: str) -> str:
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" '
            f'style="display:block;background:transparent" '
            f'xmlns="http://www.w3.org/2000/svg" role="img">{contenu}</svg>')


# ─────────────────────────────────────────────────────────────────────────
#  1 · La courbe d'âge locomoteur
# ─────────────────────────────────────────────────────────────────────────
def courbe_age(courbe: Sequence[Tuple[float, float]], vitesse: float,
               age_lu: Optional[float] = None, age_declare: Optional[float] = None,
               w: int = 900, h: int = 300) -> str:
    """La vitesse attendue par décennie, avec le sujet posé sur la courbe."""
    ages = np.array([a for a, _ in courbe], dtype=float)
    vits = np.array([v for _, v in courbe], dtype=float)
    ax0, ax1 = 62, w - 26
    ay0, ay1 = 26, h - 44
    a_min, a_max = float(ages.min()), float(ages.max())
    v_min, v_max = 0.55, 1.55

    def X(a):
        return ax0 + (ax1 - ax0) * (float(a) - a_min) / max(1e-6, a_max - a_min)

    def Y(v):
        return ay1 - (ay1 - ay0) * (float(v) - v_min) / max(1e-6, v_max - v_min)

    out = []
    # grille horizontale
    for v in (0.6, 0.8, 1.0, 1.2, 1.4):
        y = Y(v)
        out.append(f'<line x1="{ax0}" y1="{y:.1f}" x2="{ax1}" y2="{y:.1f}" '
                   f'stroke="rgba(255,255,255,.07)"/>')
        out.append(_txt(ax0 - 10, y + 4, f"{v:.1f}", 10, GRIS, "end"))
    out.append(_txt(ax0 - 10, ay0 + 4, "m/s", 9, GRIS, "end"))
    for a in ages[::2]:
        out.append(_txt(X(a), ay1 + 18, f"{a:.0f}", 10, GRIS, "middle"))
    out.append(_txt((ax0 + ax1) / 2, h - 8, "âge", 10, GRIS, "middle"))

    # bande de dispersion (± 0,15 m/s, ordre de grandeur d'une population)
    haut = " ".join(f"{X(a):.1f},{Y(v + 0.15):.1f}" for a, v in zip(ages, vits))
    bas = " ".join(f"{X(a):.1f},{Y(v - 0.15):.1f}" for a, v in zip(ages[::-1], vits[::-1]))
    out.append(f'<polygon points="{haut} {bas}" fill="rgba(122,162,255,.10)"/>')
    # la courbe
    pts = " ".join(f"{X(a):.1f},{Y(v):.1f}" for a, v in zip(ages, vits))
    out.append(f'<polyline points="{pts}" fill="none" stroke="{BLEU}" stroke-width="2" '
               f'stroke-linejoin="round"/>')

    # le sujet
    if np.isfinite(vitesse) and age_declare and np.isfinite(age_declare):
        x, y = X(np.clip(age_declare, a_min, a_max)), Y(np.clip(vitesse, v_min, v_max))
        out.append(f'<line x1="{x:.1f}" y1="{ay1}" x2="{x:.1f}" y2="{y:.1f}" '
                   f'stroke="rgba(249,115,22,.45)" stroke-dasharray="3 3"/>')
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="none" '
                   f'stroke="{ACCENT}" stroke-width="1.5" opacity=".55"/>')
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{ACCENT}"/>')
        out.append(_txt(x + 12, y - 8, f"{vitesse:.2f} m/s", 12, CLAIR, "start", 500))
    # l'âge lu, reporté sur l'axe
    if age_lu and np.isfinite(age_lu):
        xa = X(np.clip(age_lu, a_min, a_max))
        out.append(f'<line x1="{xa:.1f}" y1="{ay0}" x2="{xa:.1f}" y2="{ay1}" '
                   f'stroke="{ACCENT}" stroke-width="1" opacity=".5"/>')
        out.append(_txt(xa, ay0 - 8, f"âge locomoteur {age_lu:.0f}", 10, ACCENT, "middle", 500))
    return _cadre(w, h, "".join(out))


# ─────────────────────────────────────────────────────────────────────────
#  2 · Le radar de vitalité
# ─────────────────────────────────────────────────────────────────────────
def radar(axes: Dict[str, float], w: int = 420, h: int = 340) -> str:
    """Cinq axes notés sur 100 ; ce qui n'est pas mesuré n'est pas tracé."""
    dispo = [(k, v) for k, v in axes.items()
             if isinstance(v, (int, float)) and np.isfinite(v)]
    if len(dispo) < 3:
        return ""
    cx, cy, R = w / 2, h / 2 + 6, min(w, h) * 0.34
    n = len(dispo)
    ang = [(-np.pi / 2 + 2 * np.pi * i / n) for i in range(n)]
    out = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(f"{cx + R * frac * np.cos(a):.1f},{cy + R * frac * np.sin(a):.1f}"
                       for a in ang)
        out.append(f'<polygon points="{pts}" fill="none" '
                   f'stroke="rgba(255,255,255,{.10 if frac < 1 else .18})"/>')
    for a in ang:
        out.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx + R * np.cos(a):.1f}" '
                   f'y2="{cy + R * np.sin(a):.1f}" stroke="rgba(255,255,255,.10)"/>')
    pts = " ".join(f"{cx + R * (v / 100) * np.cos(a):.1f},{cy + R * (v / 100) * np.sin(a):.1f}"
                   for (k, v), a in zip(dispo, ang))
    out.append(f'<polygon points="{pts}" fill="rgba(20,214,196,.18)" stroke="{TURQ}" '
               f'stroke-width="2" stroke-linejoin="round"/>')
    for (k, v), a in zip(dispo, ang):
        x, y = cx + R * (v / 100) * np.cos(a), cy + R * (v / 100) * np.sin(a)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{TURQ}"/>')
        lx, ly = cx + (R + 26) * np.cos(a), cy + (R + 26) * np.sin(a)
        ancre = "middle" if abs(np.cos(a)) < 0.3 else ("start" if np.cos(a) > 0 else "end")
        out.append(_txt(lx, ly, k, 10, DIM, ancre, 400, ".12em"))
        out.append(_txt(lx, ly + 13, f"{v:.0f}", 12, CLAIR, ancre, 500))
    return _cadre(w, h, "".join(out))


# ─────────────────────────────────────────────────────────────────────────
#  3 · La frise de la séquence
# ─────────────────────────────────────────────────────────────────────────
def frise(passes: List[Tuple[int, int]], turns: List[Tuple[int, int]],
          n_frames: int, fps: float, w: int = 900, h: int = 92) -> str:
    """Trajets, demi-tours et arrêts, à l'échelle du temps."""
    if not n_frames or fps <= 0:
        return ""
    x0, x1 = 16, w - 16
    y = 30
    ht = 26

    def X(i):
        return x0 + (x1 - x0) * float(i) / max(1, n_frames - 1)

    out = [f'<rect x="{x0}" y="{y}" width="{x1 - x0}" height="{ht}" rx="7" '
           f'fill="rgba(255,255,255,.05)"/>']
    for a, b in passes:
        out.append(f'<rect x="{X(a):.1f}" y="{y}" width="{max(2, X(b) - X(a)):.1f}" '
                   f'height="{ht}" rx="6" fill="rgba(122,162,255,.55)"/>')
    for a, b in turns:
        out.append(f'<rect x="{X(a):.1f}" y="{y}" width="{max(2, X(b) - X(a)):.1f}" '
                   f'height="{ht}" rx="6" fill="rgba(200,107,255,.55)"/>')
    duree = n_frames / fps
    for s in range(0, int(duree) + 1, max(1, int(duree // 8) or 1)):
        x = X(s * fps)
        out.append(f'<line x1="{x:.1f}" y1="{y + ht}" x2="{x:.1f}" y2="{y + ht + 5}" '
                   f'stroke="rgba(255,255,255,.2)"/>')
        out.append(_txt(x, y + ht + 17, f"{s}s", 9, GRIS, "middle"))
    out.append(_txt(x0, 18, "trajets", 10, BLEU, "start", 500, ".12em"))
    out.append(_txt(x0 + 74, 18, "demi-tours", 10, VIOLET, "start", 500, ".12em"))
    out.append(_txt(x1, 18, f"{duree:.0f} s analysées", 10, GRIS, "end"))
    return _cadre(w, h, "".join(out))


# ─────────────────────────────────────────────────────────────────────────
#  Assemblage des axes du radar à partir des mesures
# ─────────────────────────────────────────────────────────────────────────
def axes_vitalite(features: Dict[str, float], biomarqueurs: Dict[str, object]
                  ) -> Dict[str, float]:
    """Cinq familles, notées sur 100 à partir de ce qui est disponible."""
    from . import kinexa

    def n(cle):
        return kinexa._note(cle, kinexa._valeur(features, cle))

    def moy(*notes):
        v = [x for x in notes if isinstance(x, float) and np.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")

    bm = (biomarqueurs.get("bio_mobility") or {}).get("score", float("nan"))
    npx = (biomarqueurs.get("neuroplasticity") or {}).get("score", float("nan"))
    return {
        "MOBILITÉ": bm,
        "CONTRÔLE": npx,
        "RYTHME": moy(n("cadence_spm"), n("step_time_cv_pct")),
        "AMPLITUDE": moy(n("step_length_stature"), n("com_bob_stature_pct")),
        "SYMÉTRIE": moy(n("step_asymmetry_pct"), n("harmonic_ratio")),
    }
