"""Biomarqueurs de mouvement.

C'est le domaine le mieux étayé du logiciel. La vitesse de marche confortable est
associée à la survie dans des cohortes de grande taille (Studenski et al., JAMA
2011, 34 485 sujets) ; la variabilité du temps de pas et l'asymétrie sont liées au
risque de chute (Hausdorff, 2007) ; le rapport harmonique quantifie la régularité
du tronc (Menz et al., 2003) ; le SPARC mesure la fluidité d'un geste
(Balasubramanian et al., IEEE TBME 2015).

Ces associations sont populationnelles. Elles décrivent des risques moyens dans
des cohortes, jamais la trajectoire d'un individu.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sps

from . import composites, dsp
from .body import BodyTraces

# Seuils cliniques usuels de vitesse de marche confortable (m/s)
SPEED_THRESHOLDS = [
    (0.60, "Très ralentie — seuil usuel de limitation fonctionnelle"),
    (0.80, "Ralentie — seuil fréquemment retenu pour la fragilité"),
    (1.00, "Intermédiaire"),
    (1.20, "Conservée"),
    (float("inf"), "Rapide"),
]


def _clean(sig: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return dsp.interp_nan(np.where(valid, sig, np.nan))


def _smooth(x: np.ndarray, fps: float, cutoff: float = 6.0) -> np.ndarray:
    nyq = fps / 2
    if cutoff >= nyq:
        return x
    b, a = sps.butter(3, cutoff / nyq, btype="low")
    return sps.filtfilt(b, a, x, padlen=min(30, len(x) - 1))


# --------------------------------------------------------------------------- #
# Segmentation de la trajectoire
# --------------------------------------------------------------------------- #
def segment_passes(cx: np.ndarray, fps: float, height_px: float
                   ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Sépare les trajets rectilignes des demi-tours.

    Le demi-tour est une manœuvre à part : sa durée est un marqueur de mobilité
    (composante la plus discriminante du Timed Up and Go), et l'inclure dans le
    calcul tirerait la vitesse de marche vers le bas.
    """
    v = _smooth(np.gradient(cx) * fps, fps, 1.5)

    # Le seuil de « vitesse soutenue » est d'abord fixé à 25 % de la stature par
    # seconde. Sur une marche lente, un cadrage large ou un couloir court, aucun
    # trajet ne le franchit et toutes les mesures deviennent indisponibles — d'où
    # des résultats entièrement vides. On le relâche donc par paliers plutôt que
    # de ne rien renvoyer.
    moving = np.abs(v) > 0.25 * height_px
    for facteur in (0.25, 0.12, 0.06, 0.03):
        candidat = np.abs(v) > facteur * height_px
        if candidat.sum() >= max(int(0.8 * fps), 8):
            moving = candidat
            break

    def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
        out, start = [], None
        for i, m in enumerate(mask):
            if m and start is None:
                start = i
            elif not m and start is not None:
                out.append((start, i)); start = None
        if start is not None:
            out.append((start, len(mask)))
        return out

    passes = []
    for a, b in _runs(moving):
        if b - a < int(0.6 * fps):
            continue
        # Un trajet peut contenir un changement de sens si le seuil n'a pas été
        # franchi : on le découpe au passage par zéro de la vitesse.
        sign = np.sign(v[a:b])
        cuts = [a] + [a + i for i in range(1, b - a) if sign[i] != sign[i - 1] and sign[i] != 0] + [b]
        for s, e in zip(cuts[:-1], cuts[1:]):
            if e - s >= int(0.6 * fps):
                passes.append((s, e))

    turns = []
    for a, b in _runs(~moving):
        if b - a < int(0.15 * fps):
            continue
        before = [p for p in passes if p[1] <= a + 2]
        after = [p for p in passes if p[0] >= b - 2]
        if not before or not after:
            continue                                # arrêt en début ou fin de séquence
        d0 = np.sign(np.median(v[before[-1][0]:before[-1][1]]))
        d1 = np.sign(np.median(v[after[0][0]:after[0][1]]))
        if d0 != 0 and d1 != 0 and d0 != d1:
            # La manœuvre ne se réduit pas à la fenêtre sous le seuil : on
            # l'étend à la décélération qui la précède et à la réaccélération
            # qui la suit, jusqu'à 80 % de la vitesse de croisière de chaque
            # trajet. C'est la définition retenue en clinique pour le demi-tour.
            v0 = 0.8 * abs(np.median(v[before[-1][0]:before[-1][1]]))
            v1 = 0.8 * abs(np.median(v[after[0][0]:after[0][1]]))
            s = a
            while s > before[-1][0] and abs(v[s - 1]) < v0:
                s -= 1
            e = b
            while e < after[0][1] - 1 and abs(v[e]) < v1:
                e += 1
            turns.append((s, e))
    return passes, turns


# --------------------------------------------------------------------------- #
# Détection des pas
# --------------------------------------------------------------------------- #
def step_events(spread: np.ndarray, fps: float,
                window: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, float]:
    """Instants de pas à partir de l'écartement des jambes.

    L'écartement est maximal au double appui : un maximum = un pas. Mais ce
    maximum est plat, donc mal localisé — le pointer directement laisse une
    gigue de l'ordre de 10 % du temps de pas, ce qui noie la variabilité
    recherchée (2–5 % chez un sujet sain).

    Deux estimateurs plus fins ont été écartés :
      * l'ajustement parabolique sur trois points reste dominé par le bruit,
        la courbure au sommet étant faible ;
      * la phase du signal analytique (Hilbert) est très précise mais impose
        une bande étroite, qui lisse justement l'alternance gauche/droite que
        l'on veut mesurer — testé, il ramenait l'asymétrie détectée à zéro.

    Retenu : le passage du flanc montant à mi-hauteur du cycle. Le flanc est
    raide, donc bien localisé, et cet estimateur n'impose aucune régularité.
    """
    seg = spread if window is None else spread[window[0]:window[1]]
    if seg.size < int(1.5 * fps):
        return np.array([]), float("nan")
    s = dsp.detrend_smoothness(seg, lam=80.0)
    s = _smooth(s, fps, 6.0)
    f0, _, _, _ = dsp.dominant_frequency(s, fps, (0.5, 3.5), nperseg_s=6.0)
    if not np.isfinite(f0) or f0 <= 0:
        return np.array([]), float("nan")

    dist = max(1, int(0.62 * fps / f0))
    peaks, _ = sps.find_peaks(s, distance=dist, prominence=0.30 * np.std(s))
    if peaks.size < 3:
        return np.array([]), float("nan")

    half_cycle = 0.5 * fps / f0
    times = []
    for p in peaks:
        a = max(0, int(p - half_cycle))
        b = min(len(s), int(p + half_cycle) + 1)
        base = float(np.min(s[a:b]))
        half = base + 0.5 * (float(s[p]) - base)
        if s[p] <= half:
            continue
        # Seul le flanc montant sert de repère. Moyenner les deux flancs
        # ramènerait l'événement au centre du cycle et lisserait l'alternance
        # gauche/droite — vérifié : l'asymétrie détectée chutait des deux tiers.
        left = _cross(s, p, half, -1)
        if left is None:
            continue
        times.append(left / fps)
    if len(times) < 3:
        return np.array([]), float("nan")
    return np.asarray(times), float(f0)


def _cross(s: np.ndarray, start: int, level: float, direction: int) -> Optional[float]:
    """Position sous-image du passage par `level` en partant de `start`."""
    i = start
    while 0 <= i < len(s) and s[i] > level:
        i += direction
        if i < 0 or i >= len(s):
            return None
    j = i - direction
    if s[j] == s[i]:
        return float(i)
    frac = (s[j] - level) / (s[j] - s[i])
    return float(j + direction * frac)


def gait_timing(spread: np.ndarray, fps: float,
                passes: List[Tuple[int, int]]) -> Dict[str, float]:
    """Temps de pas mesurés trajet par trajet.

    Point décisif : les demi-tours ne sont pas de la marche. Dater des pas à
    travers un demi-tour fabrique des intervalles qui n'existent pas et gonfle
    la variabilité — c'est ce qui rendait la mesure inutilisable au départ.
    """
    all_iv: List[np.ndarray] = []
    asyms: List[float] = []
    freqs: List[float] = []
    n_steps = 0

    for a, b in passes:
        times, f0 = step_events(spread, fps, (a, b))
        if times.size < 4:
            continue
        iv = np.diff(times)
        med = np.median(iv)
        iv = iv[(iv > 0.6 * med) & (iv < 1.7 * med)]
        if iv.size < 3:
            continue
        n_steps += times.size
        if np.isfinite(f0):
            freqs.append(f0)
        all_iv.append(iv)
        # L'alternance gauche/droite est cohérente à l'intérieur d'un trajet,
        # pas d'un trajet à l'autre : l'asymétrie se calcule donc localement.
        left, right = iv[0::2], iv[1::2]
        k = min(len(left), len(right))
        if k >= 2:
            asyms.append(abs(left[:k].mean() - right[:k].mean()) / iv.mean() * 100.0)

    if not all_iv:
        return {k: float("nan") for k in
                ("step_time_s", "step_time_cv_pct", "stride_time_cv_pct",
                 "step_asymmetry_pct", "step_asymmetry_null_pct",
                 "step_asymmetry_net_pct", "n_steps", "step_hz")} | {"n_steps": 0.0}

    iv = np.concatenate(all_iv)
    strides = np.concatenate([v[:2 * (len(v) // 2)].reshape(-1, 2).sum(axis=1)
                              for v in all_iv if len(v) >= 2]) \
        if any(len(v) >= 2 for v in all_iv) else np.array([])

    # Plancher de détection : avec peu de pas par trajet, l'indice d'asymétrie
    # est une valeur absolue et reste donc positif même quand la marche est
    # parfaitement symétrique. On estime ce plancher en rebattant l'ordre des
    # intervalles à l'intérieur de chaque trajet — ce qui détruit l'alternance
    # gauche/droite sans toucher à la distribution des durées.
    rng = np.random.default_rng(0)
    null = []
    for _ in range(300):
        vals = []
        for v in all_iv:
            if len(v) < 4:
                continue
            p = rng.permutation(v)
            left, right = p[0::2], p[1::2]
            k = min(len(left), len(right))
            if k >= 2:
                vals.append(abs(left[:k].mean() - right[:k].mean()) / p.mean() * 100.0)
        if vals:
            null.append(np.mean(vals))
    null_level = float(np.mean(null)) if null else float("nan")
    observed = float(np.mean(asyms)) if asyms else float("nan")
    p_value = (float(np.mean([x >= observed for x in null])) if null and np.isfinite(observed)
               else float("nan"))

    return {
        "step_time_s": float(iv.mean()),
        "step_time_cv_pct": float(100.0 * iv.std(ddof=1) / iv.mean()) if iv.size > 2 else float("nan"),
        "stride_time_cv_pct": float(100.0 * strides.std(ddof=1) / strides.mean())
        if strides.size >= 3 else float("nan"),
        "step_asymmetry_pct": observed,
        "step_asymmetry_null_pct": null_level,
        "step_asymmetry_net_pct": float(max(0.0, observed - null_level))
        if np.isfinite(observed) and np.isfinite(null_level) else float("nan"),
        "step_asymmetry_p": p_value,
        "n_steps": float(n_steps),
        "step_hz": float(np.median(freqs)) if freqs else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Régularité et fluidité
# --------------------------------------------------------------------------- #
def harmonic_ratio(vertical: np.ndarray, fps: float, stride_hz: float) -> float:
    """Rapport harmonique de l'accélération verticale du tronc.

    Une foulée régulière concentre l'énergie sur les harmoniques paires de la
    fréquence de foulée ; l'irrégularité alimente les impaires. Valeur haute =
    marche régulière.
    """
    if not np.isfinite(stride_hz) or stride_hz <= 0 or len(vertical) < int(4 * fps):
        return float("nan")
    acc = np.gradient(np.gradient(vertical)) * fps ** 2
    acc = acc - acc.mean()
    n_strides = int((len(acc) / fps) * stride_hz)
    if n_strides < 4:
        return float("nan")
    n = int(n_strides / stride_hz * fps)            # nombre entier de foulées
    seg = acc[:n] * np.hanning(n)
    spec = np.abs(np.fft.rfft(seg))
    even = spec[n_strides * 2::n_strides * 2][:10]
    odd = spec[n_strides::n_strides * 2][:10]
    if odd.sum() <= 0:
        return float("nan")
    return float(even.sum() / odd.sum())


def sparc(speed: np.ndarray, fps: float, fc: float = 10.0,
          amp_th: float = 0.05) -> float:
    """Spectral arc length : fluidité d'un profil de vitesse.

    Valeur proche de -1,5 = geste très fluide ; plus négative = geste haché.
    """
    if len(speed) < int(0.5 * fps):
        return float("nan")
    n = int(2 ** np.ceil(np.log2(len(speed)) + 2))
    mag = np.abs(np.fft.rfft(speed - speed.mean(), n=n))
    if mag.max() <= 0:
        return float("nan")
    mag = mag / mag.max()
    freq = np.fft.rfftfreq(n, 1.0 / fps)
    keep = freq <= fc
    mag, freq = mag[keep], freq[keep]
    above = np.nonzero(mag >= amp_th)[0]
    if above.size < 3:
        return float("nan")
    mag, freq = mag[: above[-1] + 1], freq[: above[-1] + 1]
    df = np.diff(freq) / (freq[-1] - freq[0] + 1e-12)
    dm = np.diff(mag)
    return float(-np.sum(np.sqrt(df ** 2 + dm ** 2)))


def normalized_jerk(pos: np.ndarray, fps: float) -> float:
    """Secousse normalisée (sans dimension) : coût de fluidité d'un déplacement."""
    if len(pos) < int(1.0 * fps):
        return float("nan")
    jerk = np.gradient(np.gradient(np.gradient(pos))) * fps ** 3
    dur = len(pos) / fps
    amp = np.ptp(pos)
    if amp <= 0:
        return float("nan")
    return float(np.sqrt(0.5 * np.sum(jerk ** 2) / fps * dur ** 5 / amp ** 2))


# --------------------------------------------------------------------------- #
# Équilibre debout
# --------------------------------------------------------------------------- #
def postural_sway(cx: np.ndarray, cy: np.ndarray, fps: float,
                  px_per_m: Optional[float]) -> Dict[str, float]:
    """Oscillation posturale : dispersion, aire et vitesse du centre de masse."""
    # Bande large en bas du spectre : l'oscillation posturale contient une
    # composante très lente (dérive du centre de pression) qui compte autant
    # que les corrections rapides. Un passe-haut trop agressif l'effacerait.
    ap = dsp.detrend_smoothness(cy, lam=1500.0)
    ml = dsp.detrend_smoothness(cx, lam=1500.0)
    ap = dsp.bandpass(ap, fps, 0.02, min(3.0, fps / 2 - 0.5), order=3)
    ml = dsp.bandpass(ml, fps, 0.02, min(3.0, fps / 2 - 0.5), order=3)

    rms_ap, rms_ml = float(np.sqrt(np.mean(ap ** 2))), float(np.sqrt(np.mean(ml ** 2)))
    path = float(np.sum(np.hypot(np.diff(ml), np.diff(ap))) / (len(ap) / fps))
    cov = np.cov(np.stack([ml, ap]))
    ev = np.linalg.eigvalsh(cov)
    area = float(np.pi * 5.99 * np.sqrt(max(ev[0], 0) * max(ev[1], 0)))   # ellipse à 95 %

    f, pxx = dsp.welch_psd(ap, fps, nperseg_s=min(20.0, len(ap) / fps / 3))
    m = (f >= 0.05) & (f <= 2.0)
    cum = np.cumsum(pxx[m]) / (np.sum(pxx[m]) + 1e-12)
    idx95 = int(min(np.searchsorted(cum, 0.95), cum.size - 1))
    f95 = float(f[m][idx95]) if m.any() and cum.size else float("nan")

    out = {
        "sway_rms_ap_px": rms_ap, "sway_rms_ml_px": rms_ml,
        "sway_path_px_s": path, "sway_area_px2": area, "sway_f95_hz": f95,
    }
    if px_per_m:
        out.update({
            "sway_rms_ap_mm": rms_ap / px_per_m * 1000.0,
            "sway_rms_ml_mm": rms_ml / px_per_m * 1000.0,
            "sway_path_mm_s": path / px_per_m * 1000.0,
            "sway_area_mm2": area / (px_per_m ** 2) * 1e6,
        })
    return out


# --------------------------------------------------------------------------- #
# Transferts assis-debout
# --------------------------------------------------------------------------- #
def sit_to_stand(height_px: np.ndarray, fps: float) -> Dict[str, float]:
    """Transferts détectés sur la hauteur de silhouette."""
    h = _smooth(dsp.interp_nan(height_px), fps, 2.0)
    if len(h) < int(3 * fps):
        return {"sts_count": 0.0, "sts_mean_dur_s": float("nan"),
                "sts_rise_speed": float("nan")}
    rng = np.percentile(h, 95) - np.percentile(h, 5)
    if rng < 0.18 * np.median(h):                  # amplitude insuffisante
        return {"sts_count": 0.0, "sts_mean_dur_s": float("nan"),
                "sts_rise_speed": float("nan")}
    lo, hi = np.percentile(h, 10), np.percentile(h, 90)
    state = h > (lo + hi) / 2
    trans = np.nonzero(np.diff(state.astype(int)) != 0)[0]
    rises = [t for t in trans if state[t + 1]]
    durs = []
    for t in rises:
        a = t
        while a > 0 and h[a] > lo + 0.1 * (hi - lo):
            a -= 1
        b = t
        while b < len(h) - 1 and h[b] < lo + 0.9 * (hi - lo):
            b += 1
        if 0.2 < (b - a) / fps < 6.0:
            durs.append((b - a) / fps)
    return {
        "sts_count": float(len(durs)),
        "sts_mean_dur_s": float(np.mean(durs)) if durs else float("nan"),
        "sts_rise_speed": float(rng / np.mean(durs) / np.median(h)) if durs else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Chaîne complète
# --------------------------------------------------------------------------- #
def mouvement_libre(cx: np.ndarray, cy: np.ndarray, spread: np.ndarray,
                    hh: np.ndarray, fps: float, med_h: float) -> Dict[str, float]:
    """Ce qu'on peut mesurer d'un mouvement quelconque.

    Gymnastique, danse, tai-chi, exercice de rééducation, gestes assis : rien
    de tout cela n'a de pas ni de lever de chaise, mais tout a une amplitude,
    un rythme, une fluidité et une régularité. Ces quatre-là ne supposent
    aucune tâche particulière — seulement un corps qui bouge.
    """
    out: Dict[str, float] = {}
    n = int(min(len(cx), len(cy)))
    if n < int(2 * fps) or med_h <= 0:
        return out

    candidats = {"horizontal": np.asarray(cx, float), "vertical": np.asarray(cy, float)}
    if len(spread) == n:
        candidats["membres"] = np.asarray(spread, float)
    meilleur, etendue = None, 0.0
    for nom, sig in candidats.items():
        e = float(np.percentile(sig, 97) - np.percentile(sig, 3))
        if e > etendue:
            meilleur, etendue = nom, e
    if meilleur is None:
        return out
    s = candidats[meilleur]

    out["move_source"] = meilleur
    out["move_amplitude_stature"] = float(etendue / med_h)

    v = np.abs(np.gradient(s) * fps)
    out["move_peak_speed_stature"] = float(np.percentile(v, 95) / med_h)
    out["move_mean_speed_stature"] = float(np.mean(v) / med_h)
    out["move_sparc"] = sparc(v, fps)
    out["move_jerk_norm"] = normalized_jerk(s, fps)

    c = dsp.detrend_smoothness(s, lam=80.0)
    f0, _, _, _ = dsp.dominant_frequency(c, fps, (0.15, 3.5), nperseg_s=6.0)
    if np.isfinite(f0) and f0 > 0:
        out["move_rate_hz"] = float(f0)
        out["move_rate_cpm"] = float(f0 * 60.0)
        niveau = float(np.median(c))
        croise = np.flatnonzero((c[:-1] <= niveau) & (c[1:] > niveau)) / fps
        if croise.size >= 3:
            d = np.diff(croise)
            if d.size and np.mean(d) > 0:
                out["move_cycle_cv_pct"] = float(100.0 * np.std(d) / np.mean(d))
                out["move_n_cycles"] = float(croise.size - 1)
    seuil = max(0.02 * med_h, 0.25 * float(np.median(v)))
    out["move_active_pct"] = float(100.0 * np.mean(v > seuil))
    return out



def detect_task(b: BodyTraces) -> str:
    if b.mode == "suivi" or b.detection_rate < 0.15:
        return "posture"                             # sujet immobile, suivi d'imagette
    cx = _clean(b.centroid[:, 0], b.valid)
    h = _clean(b.height_px, b.valid)
    med_h = float(np.nanmedian(h)) or 1.0
    span = float(np.ptp(cx))
    h_var = float(np.percentile(h, 95) - np.percentile(h, 5)) / med_h
    # Le déplacement horizontal prime : un sujet qui traverse le cadre marche,
    # même si sa silhouette change de hauteur (boiterie, port de charge).
    if span > 1.2 * med_h:
        return "marche"
    # Un couloir court, un cadrage serré ou une personne âgée qui avance de
    # deux mètres ne franchissent pas ce seuil, et l'enregistrement était
    # jusqu'ici déclaré « posture » : plus aucune mesure de marche. On cherche
    # donc l'alternance des jambes avant de conclure à l'immobilité.
    try:
        sp = _clean(b.leg_spread, b.valid)
        # Un squat, un balancement du tronc ou un geste des bras ne font pas
        # varier l'écartement des jambes. Sans cette modulation, les « pas »
        # détectés ne sont que du bruit mis en forme.
        modulation = float(np.percentile(sp, 97) - np.percentile(sp, 3)) / max(1e-6, med_h)
        pas, f0 = step_events(sp, b.fps)
        if modulation >= 0.04 and pas.size >= 4 and np.isfinite(f0) and 0.5 <= f0 <= 3.5:
            return "marche"
    except (ValueError, IndexError):
        pass
    if h_var > 0.20:
        return "leve"
    # Un corps qui bouge franchement sans pas ni lever : gymnastique, danse,
    # exercice, gestes assis. Il ne s'agit pas d'une posture immobile, et le
    # traiter comme telle ne mesurerait qu'un ballant qui n'existe pas.
    try:
        cy = _clean(b.centroid[:, 1], b.valid)
        v = np.abs(np.gradient(cx) * b.fps) + np.abs(np.gradient(cy) * b.fps)
        if float(np.percentile(v, 90)) > 0.06 * med_h:
            return "mouvement"
    except (ValueError, IndexError):
        pass
    return "posture"


def analyze_motion(b: BodyTraces, task: str = "auto",
                   subject_height_m: Optional[float] = None,
                   px_per_m: Optional[float] = None) -> Dict[str, object]:
    fps = b.fps
    cx = _clean(b.centroid[:, 0], b.valid)
    cy = _clean(b.centroid[:, 1], b.valid)
    hh = dsp.interp_nan(b.height_px)
    spread = _clean(b.leg_spread, b.valid)
    ty = _clean(b.trunk_y, b.valid)
    med_h = float(np.nanmedian(hh))
    if not np.isfinite(med_h) or med_h <= 0:
        med_h = 1.0

    if task == "auto":
        task = detect_task(b)

    # Échelle : distance connue si fournie, sinon stature via la silhouette
    scale = px_per_m
    scale_source = "échelle fournie"
    if scale is None and subject_height_m:
        stature_px = float(np.percentile(hh[np.isfinite(hh)], 97)) if np.isfinite(hh).any() else med_h
        scale = stature_px / subject_height_m
        scale_source = "stature déclarée"
    if scale is None:
        scale_source = "aucune — mesures en pixels"

    feats: Dict[str, float] = {
        "task_detected": task,
        "silhouette_height_px": med_h,
        "px_per_m": float(scale) if scale else float("nan"),
        "body_detection_rate": b.detection_rate,
        "camera_motion_px": b.camera_motion_px,
    }

    passes, turns = segment_passes(cx, fps, med_h)
    feats["n_passes"] = float(len(passes))
    feats["n_turns"] = float(len(turns))
    if turns:
        feats["turn_mean_dur_s"] = float(np.mean([(b1 - a) / fps for a, b1 in turns]))
        feats["turn_max_dur_s"] = float(np.max([(b1 - a) / fps for a, b1 in turns]))
    else:
        feats["turn_mean_dur_s"] = feats["turn_max_dur_s"] = float("nan")

    if task == "marche" and \
            float(np.percentile(spread, 97) - np.percentile(spread, 3)) < 0.04 * med_h \
            and not passes:
        task = "mouvement"                    # ni pas ni trajet : geste libre
        feats["task_detected"] = task

    if task == "marche":
        # Sans trajet franc — marche sur place, couloir court — on analyse
        # l'enregistrement entier : cadence, régularité, symétrie et harmonie
        # ne dépendent pas de la distance parcourue. Seules la vitesse et la
        # longueur de pas en dépendent, et elles restent alors indisponibles.
        fenetres = passes if passes else [(0, int(cx.size))]
        feats["marche_sur_place"] = 0.0 if passes else 1.0
        # Vitesse : trajets rectilignes uniquement, bords rognés pour écarter
        # les phases d'accélération et de freinage autour des demi-tours.
        speeds, seg_sparc = [], []
        margin = int(0.5 * fps)
        for a, b1 in passes:
            if b1 - a < int(1.2 * fps):
                continue
            seg = cx[a + margin:b1 - margin]
            if seg.size < int(0.6 * fps):
                continue
            v = np.abs(np.gradient(seg) * fps)
            speeds.append(np.median(v))
            seg_sparc.append(sparc(v, fps))
        v_px = float(np.median(speeds)) if speeds else float("nan")
        feats["gait_speed_px_s"] = v_px
        feats["gait_speed_m_s"] = float(v_px / scale) if (scale and np.isfinite(v_px)) else float("nan")
        feats["gait_speed_stature_s"] = float(v_px / med_h) if np.isfinite(v_px) else float("nan")
        feats["gait_sparc"] = float(np.median(seg_sparc)) if seg_sparc else float("nan")

        timing = gait_timing(spread, fps, fenetres)
        step_hz = timing.pop("step_hz", float("nan"))
        feats["cadence_spm"] = float(step_hz * 60.0) if np.isfinite(step_hz) else float("nan")
        feats.update(timing)
        stride_hz = step_hz / 2.0 if np.isfinite(step_hz) else float("nan")
        feats["harmonic_ratio"] = harmonic_ratio(cy, fps, stride_hz)

        # ── longueur de pas mesurée directement sur la silhouette ──────────
        # De profil, l'écartement des pieds au double appui EST la longueur du
        # pas : on peut donc la mesurer sans que le sujet traverse le champ, et
        # en déduire une vitesse là où le déplacement ne donne rien (couloir
        # court, marche sur place, plan serré). Quand les deux sont disponibles,
        # la mesure par déplacement reste prioritaire : elle ne suppose pas que
        # la caméra soit de profil.
        # Amplitude mesurée sur le signal brut : le lissage arrondit le sommet
        # du cycle et rabotait la longueur de pas de dix pour cent. Les
        # percentiles extrêmes suffisent à écarter le bruit.
        sp = spread
        if sp.size > int(fps):
            # p97 − p3 plutôt que p92 − p8 : les percentiles resserrés
            # amputaient le sommet du cycle et sous-estimaient la longueur de
            # pas d'environ 15 %, donc la vitesse d'autant.
            amp_px = float(np.percentile(sp, 97) - np.percentile(sp, 3))
            if np.isfinite(amp_px) and amp_px > 0.02 * med_h:
                feats["step_length_spread_px"] = amp_px
                feats["step_length_spread_stature"] = float(amp_px / med_h)
                if not np.isfinite(feats.get("gait_speed_px_s", float("nan"))) \
                        and np.isfinite(step_hz) and step_hz > 0:
                    v_est = amp_px * step_hz
                    feats["gait_speed_px_s"] = float(v_est)
                    feats["gait_speed_stature_s"] = float(v_est / med_h)
                    if scale:
                        feats["gait_speed_m_s"] = float(v_est / scale)
                    feats["speed_source"] = "amplitude des pas (vue de profil)"
                    v_px = float(v_est)
                else:
                    feats.setdefault("speed_source", "déplacement dans le champ")

        if np.isfinite(feats.get("cadence_spm", np.nan)) and np.isfinite(v_px) and step_hz > 0:
            step_len_px = v_px / step_hz
            feats["step_length_px"] = float(step_len_px)
            feats["step_length_stature"] = float(step_len_px / med_h)
            if scale:
                feats["step_length_m"] = float(step_len_px / scale)

        bob = dsp.bandpass(cy, fps, 0.6, min(4.0, fps / 2 - 0.5), order=3)
        feats["com_bob_px"] = float(np.sqrt(np.mean(bob ** 2)) * np.sqrt(2))
        feats["com_bob_stature_pct"] = float(feats["com_bob_px"] / med_h * 100.0)
        # Secousse normalisée par trajet : elle croît en durée^5, une valeur
        # calculée sur l'enregistrement entier ne serait comparable à rien.
        jerks = [normalized_jerk(cy[a:b1], fps) for a, b1 in fenetres if b1 - a > int(1.2 * fps)]
        jerks = [j for j in jerks if np.isfinite(j)]
        feats["gait_jerk_norm"] = float(np.median(jerks)) if jerks else float("nan")

        if np.isfinite(feats.get("gait_speed_m_s", np.nan)):
            v = feats["gait_speed_m_s"]
            feats["gait_speed_band"] = next(lab for th, lab in SPEED_THRESHOLDS if v < th)

    elif task == "mouvement":
        feats.update(mouvement_libre(cx, cy, spread, hh, fps, med_h))

    elif task == "leve":
        feats.update(sit_to_stand(hh, fps))
        feats["sts_sparc"] = sparc(np.abs(np.gradient(hh) * fps), fps)
        feats["sts_jerk_norm"] = normalized_jerk(hh, fps)

    else:                                            # posture
        # Le suivi de torse est préféré au centroïde de silhouette : la
        # quantification au pixel de la silhouette masquerait une oscillation
        # de deux ou trois pixels.
        tx, ty = cx, cy
        source = "centroïde de silhouette"
        if b.track_xy.size and np.isfinite(b.track_xy).all(axis=1).mean() > 0.8 \
                and b.track_confidence > 0.5:
            tx = dsp.interp_nan(b.track_xy[:, 0])
            ty = dsp.interp_nan(b.track_xy[:, 1])
            source = "suivi de torse au sous-pixel"
        feats.update(postural_sway(tx, ty, fps, scale))
        feats["sway_source"] = source
        feats["sway_jerk_norm"] = normalized_jerk(ty, fps)

    if task in ("marche", "leve"):
        for k, v in mouvement_libre(cx, cy, spread, hh, fps, med_h).items():
            feats.setdefault(k, v)

    feats["scale_source"] = scale_source

    # Indices composites : rapports sans dimension entre grandeurs déjà mesurées
    sig = {"cx": cx, "cy": cy, "spread": spread, "height": hh,
           "trunk_y": ty, "fps": fps}
    feats.update(composites.composite_indices(feats, sig, passes, turns))

    return {"task": task, "features": feats,
            "segments": {"passes": passes, "turns": turns},
            "signals": sig}
