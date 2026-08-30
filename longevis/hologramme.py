"""Rejeu de la vidéo avec les chiffres incrustés, façon hologramme.

Le lecteur est construit en HTML : la vidéo est rendue par le navigateur et les
chiffres sont posés par-dessus, synchronisés sur son horloge. Rien n'est
ré-encodé côté serveur — un ré-encodage coûterait plusieurs minutes et
échouerait sur la plupart des hébergements sans ffmpeg.

Ce qui s'anime pendant la lecture :
  · les quatre biomarqueurs, qui montent de zéro à leur valeur en deux secondes
  · la vitesse instantanée, lue image par image sur la trajectoire mesurée
  · le compteur de pas, qui s'incrémente à chaque pas détecté
  · une trame de balayage et une ligne d'horizon, pour la matière
"""

from __future__ import annotations
import base64
import json
import os
from typing import Dict, List, Optional

import numpy as np


MAX_MO = 28.0          # au-delà, l'encodage en base64 alourdit trop la page


def _serie(sig: Dict[str, object], fps: float, duree: float,
           taille_image=(0, 0), pas_par_s: float = 10.0) -> List[Dict[str, float]]:
    """Vitesse, pas, et position du corps dans l'image — pour poser les
    incrustations sur le sujet plutôt que dans un coin."""
    cx = np.asarray(sig.get("body_cx", []), dtype=float)
    cy = np.asarray(sig.get("body_cy", []), dtype=float)
    haut = np.asarray(sig.get("body_height", []), dtype=float)
    spread = np.asarray(sig.get("body_spread", []), dtype=float)
    if cx.size < 4:
        return []
    n = cx.size
    t = np.arange(n) / max(1e-6, fps)
    v = np.abs(np.gradient(cx) * fps)                    # px/s
    if v.size > 8:                                       # lissage court
        k = max(3, int(fps / 3) | 1)
        noyau = np.ones(k) / k
        v = np.convolve(v, noyau, mode="same")

    # pas : passages du flanc montant de l'écartement à mi-hauteur
    pas_t: List[float] = []
    if spread.size == n and np.isfinite(spread).any():
        s = spread - np.nanmedian(spread)
        seuil = 0.25 * (np.nanpercentile(s, 95) - np.nanpercentile(s, 5))
        montant = (s[:-1] <= seuil) & (s[1:] > seuil)
        pas_t = list(t[1:][montant])

    lw = float(taille_image[0] or 0) or float(np.nanmax(cx) * 1.25 or 1)
    lh = float(taille_image[1] or 0) or (float(np.nanmax(cy) * 1.6) if cy.size else 1)

    sortie = []
    for i in range(int(duree * pas_par_s) + 1):
        ti = i / pas_par_s
        j = int(min(n - 1, ti * fps))
        p = {"t": round(ti, 2),
             "v": round(float(v[j]), 2),
             "n": int(sum(1 for x in pas_t if x <= ti))}
        if lw > 0 and np.isfinite(cx[j]):
            p["x"] = round(float(np.clip(cx[j] / lw, 0, 1)), 4)
        if cy.size == n and lh > 0 and np.isfinite(cy[j]):
            p["y"] = round(float(np.clip(cy[j] / lh, 0, 1)), 4)
        if haut.size == n and lh > 0 and np.isfinite(haut[j]):
            p["h"] = round(float(np.clip(haut[j] / lh, 0.05, 1.4)), 4)
        sortie.append(p)
    return sortie


def video_base64(chemin: str) -> Optional[str]:
    """Encode la vidéo pour l'incruster dans la page, si elle n'est pas trop lourde."""
    try:
        mo = os.path.getsize(chemin) / 1e6
        if mo > MAX_MO:
            return None
        with open(chemin, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    except OSError:
        return None


def hauteur_composant(meta: Dict[str, object], largeur: int = 900) -> int:
    """Hauteur à réserver : l'image plus le bandeau de chiffres.

    Une vidéo de téléphone est le plus souvent verticale ; réserver du 16/9
    coupait alors le bas du lecteur, donc les quatre cartes.
    """
    taille = meta.get("frame_size") or (16, 9)
    try:
        w, h = float(taille[0]), float(taille[1])
        rapport = h / w if w > 0 else 9 / 16
    except (TypeError, ValueError, IndexError):
        rapport = 9 / 16
    rapport = min(max(rapport, 0.4), 1.9)          # ni panoramique ni colonne
    return int(largeur * rapport) + 170




# ─────────────────────────────────────────────────────────────────────────
#  La musique du mouvement
#
#  Le corps écrit la partition. Rien n'est plaqué par-dessus : le tempo naît
#  de la cadence mesurée, le mode de la qualité du contrôle, les accords des
#  accents et des repos du geste, et chaque note tombe sur un pas.
#
#  Le résultat doit sonner harmonieux quand le mouvement est ample et
#  régulier, et se ternir quand il est heurté — c'est là le retour.
# ─────────────────────────────────────────────────────────────────────────

#  Trois couleurs, de la plus lumineuse à la plus grave.
MODES = {
    "lydien":  [0, 2, 4, 6, 7, 9, 11],
    "ionien":  [0, 2, 4, 5, 7, 9, 11],
    "dorien":  [0, 2, 3, 5, 7, 9, 10],
    "eolien":  [0, 2, 3, 5, 7, 8, 10],
}

#  Degrés de la marche harmonique : I, V, vi, IV — l'enchaînement le plus
#  simple qui sonne juste dans tous les modes ci-dessus.
DEGRES = {"I": 0, "V": 4, "vi": 5, "IV": 3}


def _mode_de(score: float) -> str:
    """Le contrôle du geste choisit la couleur de l'harmonie."""
    if not isinstance(score, (int, float)) or not np.isfinite(score):
        return "ionien"
    if score >= 72:
        return "lydien"
    if score >= 56:
        return "ionien"
    if score >= 40:
        return "dorien"
    return "eolien"


def _tempo_de(features: Dict[str, float]) -> float:
    """Le pouls de la musique est celui du corps."""
    for cle in ("cadence_spm", "move_rate_cpm"):
        v = features.get(cle, float("nan"))
        if isinstance(v, (int, float)) and np.isfinite(v) and v > 0:
            while v > 132:                    # une cadence rapide se lit à la moitié
                v /= 2.0
            while v < 46:
                v *= 2.0
            return float(np.clip(v, 46.0, 132.0))
    return 76.0


def _hauteur(midi: float) -> float:
    return 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))


def musique(serie: List[Dict[str, float]], features: Dict[str, float],
            biomarqueurs: Dict[str, object], duree: float) -> Dict[str, object]:
    """Partition complète, en secondes de vidéo."""
    if not serie:
        return {}
    t = np.array([p["t"] for p in serie], dtype=float)
    v = np.array([p["v"] for p in serie], dtype=float)
    n = np.array([p.get("n", 0) for p in serie], dtype=float)
    if t.size < 4:
        return {}

    npx = (biomarqueurs.get("neuroplasticity") or {}).get("score", float("nan"))
    mob = (biomarqueurs.get("bio_mobility") or {}).get("score", float("nan"))
    mode = _mode_de(npx)
    gamme = MODES[mode]
    tempo = _tempo_de(features)
    temps = 60.0 / tempo

    #  La tonique monte avec la mobilité : un corps allant sonne plus clair.
    base = 50 if not np.isfinite(mob) else int(np.clip(45 + mob / 9.0, 45, 57))

    #  ── lecture du geste : accents, renversements, repos ──────────────
    vmax = float(np.nanmax(v)) or 1.0
    dv = np.gradient(v)
    accent = (v > 0.68 * vmax) & (np.abs(dv) < 0.18 * vmax)
    repos = v < 0.22 * vmax
    renv = np.zeros_like(v, dtype=bool)
    renv[1:] = (np.sign(dv[:-1]) != np.sign(dv[1:])) & (np.abs(dv[1:]) > 0.06 * vmax)

    #  ── les accords : un pas dans la marche harmonique par événement ──
    accords, tc, i, suite, depuis = [], 0.0, 0, "I", 0
    while tc < duree and len(accords) < 400:
        i = int(min(len(t) - 1, np.searchsorted(t, tc)))
        if repos[i]:
            suite = "IV" if suite != "IV" else "I"
        elif accent[i]:
            suite = "V"
        elif renv[i]:
            suite = "vi"
        else:
            suite = suite
        depuis += 1
        if depuis >= 8:                       # la phrase revient chez elle
            suite, depuis = "I", 0
        deg = DEGRES[suite]
        fond = [gamme[deg % 7] + 12 * (deg // 7),
                gamme[(deg + 2) % 7] + 12 * ((deg + 2) // 7),
                gamme[(deg + 4) % 7] + 12 * ((deg + 4) // 7)]
        #  la durée suit l'agitation : geste vif, accords brefs
        mesures = 2 if v[i] > 0.5 * vmax else 3
        accords.append({"t": round(tc, 3), "d": round(mesures * temps, 3),
                        "b": base + fond[0] - 12,
                        "n": [base + x for x in fond]})
        tc += mesures * temps

    #  ── la mélodie : une note par pas mesuré ──────────────────────────
    notes = []
    for k in range(1, n.size):
        if n[k] <= n[k - 1]:
            continue
        ti = float(t[k])
        acc = None
        for a in accords:
            if a["t"] <= ti < a["t"] + a["d"]:
                acc = a
                break
        if acc is None:
            continue
        #  la vigueur de l'instant choisit le degré de l'accord et l'octave
        r = float(np.clip(v[k] / vmax, 0, 1))
        choix = acc["n"][int(round(r * 2))]
        octave = 12 if r > 0.72 else 0
        notes.append({"t": round(ti, 3), "p": choix + 12 + octave,
                      "d": round(0.9 * temps, 3), "g": round(0.30 + 0.35 * r, 3)})

    #  ── la justesse : un geste irrégulier désaccorde légèrement ───────
    #  En deçà de 2,5 % de variabilité le pas est régulier : la musique reste
    #  juste. Au-delà, l'harmonie se ternit, jusqu'à un tiers de demi-ton.
    cv = float("nan")
    for cle in ("step_time_cv_pct", "move_cycle_cv_pct", "stride_time_cv_pct"):
        x = features.get(cle, float("nan"))
        if isinstance(x, (int, float)) and np.isfinite(x):
            cv = float(x)
            break
    if not np.isfinite(cv):
        cv = 4.0
    detune = float(np.clip((cv - 2.5) * 3.4, 0.0, 32.0))     # en cents

    return {"tempo": round(tempo, 1), "mode": mode, "tonique": base,
            "accords": accords, "notes": notes, "detune": round(detune, 1),
            "hz_tonique": round(_hauteur(base), 2)}



def rejeu(chemin: str, biomarqueurs: Dict[str, object], features: Dict[str, float],
          signaux: Dict[str, object], meta: Dict[str, object]) -> Optional[str]:
    """Retourne le lecteur complet, ou None si la vidéo ne peut pas être incrustée."""
    b64 = video_base64(chemin)
    if not b64:
        return None

    fps = float(meta.get("fps") or signaux.get("body_fps") or 25.0)
    duree = float(meta.get("duration_s") or 0.0) or 1.0
    echelle = float(features.get("px_per_m") or 0.0)
    serie = _serie(signaux, fps, duree, meta.get("frame_size") or (0, 0))
    for p in serie:                                      # px/s → m/s si possible
        p["v"] = round(p["v"] / echelle, 2) if echelle > 0 else round(p["v"] / 100.0, 2)

    def val(bloc: str, cle: str, defaut: float = float("nan")) -> float:
        d = biomarqueurs.get(bloc) or {}
        v = d.get(cle, defaut)
        return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else float("nan")

    cartes = [
        {"nom": "Bio-Mobility", "unite": "/100", "val": val("bio_mobility", "score"),
         "dec": 0, "teinte": "#7AA2FF"},
        {"nom": "Neuroplasticity", "unite": "/100", "val": val("neuroplasticity", "score"),
         "dec": 0, "teinte": "#14D6C4"},
        {"nom": "Kinetic Age", "unite": "ans", "val": val("kinetic_age", "age"),
         "dec": 0, "teinte": "#F97316"},
        {"nom": "Vitality Margin", "unite": "%", "val": val("vitality_margin", "marge_pct"),
         "dec": 0, "teinte": "#C86BFF"},
    ]
    cartes = [c for c in cartes if np.isfinite(c["val"])]

    #  Tout ce qui a été mesuré, prêt à être dévoilé par le bouton « métriques ».
    LIB = [
        ("gait_speed_m_s", "Vitesse de marche", "m/s", 2),
        ("cadence_spm", "Cadence", "pas/min", 0),
        ("step_length_m", "Longueur de pas", "m", 2),
        ("step_length_stature", "Pas / stature", "", 2),
        ("step_time_cv_pct", "Variabilité du pas", "%", 1),
        ("stride_time_cv_pct", "Variabilité du cycle", "%", 1),
        ("step_asymmetry_pct", "Asymétrie", "%", 1),
        ("harmonic_ratio", "Rapport harmonique", "", 2),
        ("gait_sparc", "Fluidité (SPARC)", "", 2),
        ("com_bob_stature_pct", "Oscillation verticale", "%", 1),
        ("n_steps", "Pas analysés", "", 0),
        ("n_turns", "Demi-tours", "", 0),
        ("turn_mean_dur_s", "Durée de demi-tour", "s", 2),
        ("ird_reserve_dynamique", "Réserve dynamique", "", 2),
        ("scf_signature_foulee", "Signature de foulée", "", 3),
        ("move_amplitude_stature", "Amplitude du geste", "×stature", 2),
        ("move_peak_speed_stature", "Vitesse de pointe", "stature/s", 2),
        ("move_rate_cpm", "Rythme du geste", "cycles/min", 0),
        ("move_cycle_cv_pct", "Variabilité des cycles", "%", 1),
        ("move_sparc", "Fluidité du geste", "", 2),
        ("move_active_pct", "Temps en mouvement", "%", 0),
        ("sway_rms_ap_mm", "Ballant avant-arrière", "mm", 1),
        ("sway_rms_ml_mm", "Ballant latéral", "mm", 1),
        ("sts_mean_dur_s", "Lever de chaise", "s", 2),
        ("px_per_m", "Échelle", "px/m", 0),
        ("body_detection_rate", "Détection du corps", "", 2),
    ]
    from . import kinexa as _kx
    metriques = []
    for cle, nom, unite, dec in LIB:
        v = features.get(cle, float("nan"))
        if isinstance(v, (int, float)) and np.isfinite(v) and _kx.plausible(cle, v):
            metriques.append({"nom": nom, "val": round(float(v), dec), "unite": unite})
    vmax = max([p["v"] for p in serie] or [1.0]) or 1.0
    partition = musique(serie, features, biomarqueurs, duree)
    donnees = json.dumps({"cartes": cartes, "serie": serie, "metriques": metriques,
                          "vmax": round(float(vmax), 3), "musique": partition,
                          "duree": round(duree, 2),
                          "unite_v": "m/s" if echelle > 0 else "u/s"})

    return _GABARIT.replace("__B64__", b64).replace("__DATA__", donnees)


_GABARIT = """
<style>
.kx-scene{position:relative;border-radius:18px;overflow:hidden;background:#07070C;
 font-family:'Inter Tight',-apple-system,'Segoe UI',sans-serif;
 box-shadow:0 24px 70px rgba(0,0,0,.6)}
.kx-scene video{display:block;width:100%;filter:saturate(.85) contrast(1.05)}
.kx-veil{position:absolute;inset:0;pointer-events:none;
 background:radial-gradient(120% 90% at 50% 40%,transparent 35%,rgba(4,6,14,.72) 100%)}
.kx-scan{position:absolute;inset:0;pointer-events:none;opacity:.20;
 background:repeating-linear-gradient(180deg,rgba(180,230,255,.16) 0 1px,transparent 1px 4px)}
.kx-hud{position:absolute;left:0;right:0;bottom:0;padding:14px 16px 16px;
 display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end;pointer-events:none}
.kx-card{flex:1 1 168px;min-width:150px;border-radius:14px;padding:11px 14px 13px;
 background:rgba(8,12,22,.42);backdrop-filter:blur(12px) saturate(140%);
 -webkit-backdrop-filter:blur(12px) saturate(140%);
 border:1px solid rgba(255,255,255,.14);border-top-width:2px}
.kx-lab{font-size:clamp(12px,1.45vw,22px);letter-spacing:.22em;text-transform:uppercase;
 color:#B6C8E4;font-weight:500;text-shadow:0 0 14px rgba(120,180,255,.28)}
.kx-val{font-size:clamp(38px,5vw,86px);font-weight:200;letter-spacing:-.045em;color:#fff;
 line-height:.95;font-variant-numeric:tabular-nums;margin-top:3px;
 text-shadow:0 0 10px rgba(255,255,255,.30),0 0 34px rgba(140,200,255,.45),
             0 0 72px rgba(90,170,255,.22)}
.kx-val small{font-size:clamp(12px,1.35vw,22px);color:#9BB0D0;margin-left:5px;
 letter-spacing:0;text-shadow:none}
.kx-live{position:absolute;top:14px;right:16px;text-align:right;pointer-events:none}
.kx-live div{font-size:clamp(11px,1.25vw,19px);letter-spacing:.2em;text-transform:uppercase;
 color:#9BB0D0}
.kx-live b{display:block;font-size:clamp(46px,6.2vw,108px);font-weight:200;color:#fff;
 letter-spacing:-.05em;line-height:.92;font-variant-numeric:tabular-nums;
 text-shadow:0 0 12px rgba(255,255,255,.32),0 0 40px rgba(120,220,255,.52),
             0 0 84px rgba(60,190,255,.24)}
.kx-live b.petit{font-size:clamp(30px,3.9vw,68px)}
.kx-inst{position:absolute;top:14px;left:16px;font-size:clamp(10px,1.1vw,16px);
 letter-spacing:.24em;
 text-transform:uppercase;color:#F97316;border:1px solid rgba(249,115,22,.5);
 border-radius:99px;padding:4px 12px;pointer-events:none}
.kx-line{position:absolute;left:0;right:0;height:1px;pointer-events:none;
 background:linear-gradient(90deg,transparent,rgba(140,220,255,.55),transparent)}
.kx-time{position:absolute;top:46px;left:16px;font-size:clamp(14px,1.6vw,24px);
 letter-spacing:.22em;
 color:#9BB0D0;font-variant-numeric:tabular-nums;pointer-events:none}
.kx-bar{position:absolute;left:16px;right:16px;bottom:0;height:2px;
 background:rgba(255,255,255,.10);pointer-events:none}
.kx-bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,#14D6C4,#7AA2FF)}
.kx-corps{position:absolute;pointer-events:none;transition:left .08s linear,top .08s linear,
 width .12s linear,height .12s linear}
.kx-corps i{position:absolute;width:16px;height:16px;border:2px solid rgba(140,220,255,.9);
 filter:drop-shadow(0 0 8px rgba(120,220,255,.7))}
.kx-corps i:nth-child(1){left:0;top:0;border-right:0;border-bottom:0}
.kx-corps i:nth-child(2){right:0;top:0;border-left:0;border-bottom:0}
.kx-corps i:nth-child(3){left:0;bottom:0;border-right:0;border-top:0}
.kx-corps i:nth-child(4){right:0;bottom:0;border-left:0;border-top:0}
.kx-axe{position:absolute;left:50%;top:0;bottom:0;width:1px;
 background:linear-gradient(180deg,transparent,rgba(140,220,255,.5),transparent)}
.kx-halo{position:absolute;left:50%;top:50%;width:56px;height:56px;margin:-28px 0 0 -28px;
 border-radius:50%;border:1px solid rgba(20,214,196,.55);
 box-shadow:0 0 22px rgba(20,214,196,.35) inset,0 0 18px rgba(20,214,196,.25);
 animation:kxpulse 2.4s ease-in-out infinite}
@keyframes kxpulse{0%,100%{transform:scale(.92);opacity:.55}50%{transform:scale(1.12);opacity:1}}
.kx-etiq{position:absolute;left:calc(100% + 10px);top:8px;white-space:nowrap;
 background:rgba(8,12,22,.5);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
 border:1px solid rgba(255,255,255,.16);border-left:2px solid #14D6C4;border-radius:8px;
 padding:7px 12px;font-size:clamp(11px,1.2vw,18px);letter-spacing:.17em;
 text-transform:uppercase;color:#CFE4FF}
.kx-etiq b{display:block;font-size:clamp(24px,3vw,52px);font-weight:200;line-height:1;
 letter-spacing:-.035em;color:#fff;
 text-shadow:0 0 10px rgba(255,255,255,.28),0 0 30px rgba(140,220,255,.45);
 font-variant-numeric:tabular-nums;text-transform:none}
.kx-trace{position:absolute;width:6px;height:6px;margin:-3px 0 0 -3px;border-radius:50%;
 background:#14D6C4;pointer-events:none}
.kx-btn{position:absolute;top:14px;right:16px;z-index:6;display:flex;gap:6px}
.kx-btn button{background:rgba(8,12,22,.5);backdrop-filter:blur(10px);
 -webkit-backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,.18);
 color:#CFE4FF;border-radius:9px;padding:5px 10px;font-size:11px;letter-spacing:.12em;
 text-transform:uppercase;cursor:pointer;font-family:inherit}
.kx-btn button:hover{border-color:rgba(20,214,196,.7);color:#fff}
.kx-btn button.actif{border-color:rgba(20,214,196,.9);color:#14D6C4;
 box-shadow:0 0 18px rgba(20,214,196,.35)}
.kx-scene.plein{position:fixed;inset:0;z-index:99999;border-radius:0;
 display:flex;align-items:center;justify-content:center;background:#04060C}
.kx-scene.plein video{max-height:100vh;max-width:100vw;width:auto;height:100vh}
.kx-scene:fullscreen{display:flex;align-items:center;justify-content:center;background:#04060C}
.kx-scene:fullscreen video{max-height:100vh;width:auto;height:100vh}
/* ---- grille et coins d'instrument ---- */
.kx-grille{position:absolute;inset:0;pointer-events:none;opacity:.16;
 background:linear-gradient(90deg,rgba(140,220,255,.5) 1px,transparent 1px) 0 0/72px 100%,
            linear-gradient(180deg,rgba(140,220,255,.5) 1px,transparent 1px) 0 0/100% 72px}
.kx-coin{position:absolute;width:26px;height:26px;border:2px solid rgba(140,220,255,.55);
 pointer-events:none}
.kx-coin.h g,.kx-coin.tl{left:14px;top:44px;border-right:0;border-bottom:0}
.kx-coin.tr{right:14px;top:44px;border-left:0;border-bottom:0}
.kx-coin.bl{left:14px;bottom:96px;border-right:0;border-top:0}
.kx-coin.br{right:14px;bottom:96px;border-left:0;border-top:0}
/* ---- balayage périodique ---- */
.kx-sweep{position:absolute;left:0;right:0;height:120px;pointer-events:none;
 background:linear-gradient(180deg,transparent,rgba(20,214,196,.16),transparent);
 animation:kxsweep 6s linear infinite}
@keyframes kxsweep{0%{top:-120px}100%{top:100%}}
/* ---- oscillogramme de vitesse ---- */
.kx-osc{position:absolute;left:16px;right:16px;bottom:96px;height:46px;pointer-events:none}
.kx-osc svg{width:100%;height:100%;overflow:visible}
.kx-mets{position:absolute;top:56px;right:16px;bottom:110px;width:min(430px,52%);
 overflow:auto;padding:12px 14px;border-radius:14px;z-index:5;
 background:rgba(8,12,22,.62);backdrop-filter:blur(16px) saturate(150%);
 -webkit-backdrop-filter:blur(16px) saturate(150%);
 border:1px solid rgba(255,255,255,.16);
 transform:translateX(112%);transition:transform .28s cubic-bezier(.22,1,.36,1);
 scrollbar-width:thin}
.kx-mets.ouvert{transform:translateX(0)}
.kx-mets h4{margin:0 0 10px;font-size:clamp(11px,1.2vw,17px);letter-spacing:.22em;
 text-transform:uppercase;
 color:#F97316;font-weight:500}
.kx-met{display:flex;justify-content:space-between;gap:10px;padding:5px 0;
 border-bottom:1px solid rgba(255,255,255,.06);font-size:clamp(13px,1.4vw,21px);
 color:#9BB0D0}
.kx-met b{color:#fff;font-weight:300;font-variant-numeric:tabular-nums;white-space:nowrap;
 text-shadow:0 0 16px rgba(140,200,255,.35)}
.kx-met b i{font-style:normal;color:#9BB0D0;font-size:.62em;margin-left:4px}
</style>

<div class="kx-scene" id="kxs">
  <video id="kxv" controls playsinline preload="auto" autoplay muted loop
         src="data:video/mp4;base64,__B64__"></video>
  <div class="kx-veil"></div><div class="kx-scan"></div><div class="kx-grille"></div>
  <div class="kx-sweep"></div>
  <span class="kx-coin tl"></span><span class="kx-coin tr"></span>
  <span class="kx-coin bl"></span><span class="kx-coin br"></span>
  <div class="kx-osc"><svg viewBox="0 0 600 46" preserveAspectRatio="none">
    <polyline id="kxosc" fill="none" stroke="#14D6C4" stroke-width="1.6"
      stroke-linejoin="round" points=""></polyline>
    <polyline id="kxoscf" fill="rgba(20,214,196,.12)" stroke="none" points=""></polyline>
  </svg></div>
  <div class="kx-btn">
    <button id="kxson" type="button">♪ son</button>
    <button id="kxleg" type="button">?</button>
    <button id="kxmet" type="button">métriques</button>
    <button id="kxplein" type="button">⛶ agrandir</button>
  </div>
  <div class="kx-mets" id="kxpan"><h4>mesures disponibles</h4><div id="kxlist"></div></div>
  <div class="kx-mets" id="kxlegp" style="width:min(300px,42%)"><h4>lecture de l'écran</h4>
    <div class="kx-met"><span>équerres</span><b>suivi du corps</b></div>
    <div class="kx-met"><span>couleur</span><b>vigueur du geste</b></div>
    <div class="kx-met"><span>halo</span><b>centre de masse</b></div>
    <div class="kx-met"><span>points</span><b>trajet parcouru</b></div>
    <div class="kx-met"><span>courbe du bas</span><b>vitesse, 6 s</b></div>
    <div class="kx-met"><span>cartes</span><b>biomarqueurs</b></div>
  </div>
  <div class="kx-line" id="kxline" style="top:50%"></div>
  <div class="kx-corps" id="kxcorps" style="display:none">
    <i></i><i></i><i></i><i></i>
    <div class="kx-axe"></div><div class="kx-halo"></div>
    <div class="kx-etiq"><span id="kxetl">vitesse</span><b id="kxetv">0.00</b></div>
  </div>
  <div class="kx-inst">Kinexa · Metrology of Vitality</div>
  <div class="kx-time" id="kxt">00:00</div>
  <div class="kx-live"><div id="kxvu">vitesse</div><b id="kxv1">0.00</b>
    <div id="kxpu">pas</div><b id="kxp1" class="petit">0</b></div>
  <div class="kx-hud" id="kxhud"></div>
  <div class="kx-bar"><i id="kxbar"></i></div>
</div>

<script>
(function(){
  var D = __DATA__;
  var hud = document.getElementById('kxhud');
  D.cartes.forEach(function(c,i){
    var d = document.createElement('div');
    d.className = 'kx-card';
    d.style.borderTopColor = c.teinte;
    d.innerHTML = '<div class="kx-lab">'+c.nom+'</div>'+
                  '<div class="kx-val"><span id="kxc'+i+'">0</span>'+
                  '<small>'+c.unite+'</small></div>';
    hud.appendChild(d);
  });
  document.getElementById('kxvu').textContent = 'vitesse ' + D.unite_v;

  var v = document.getElementById('kxv');
  var monte = null;

  function anime(){                       /* les biomarqueurs montent de zéro */
    var t0 = performance.now();
    cancelAnimationFrame(monte);
    (function pas(){
      var k = Math.min(1, (performance.now()-t0)/1800);
      var e = 1-Math.pow(1-k,3);
      D.cartes.forEach(function(c,i){
        var el = document.getElementById('kxc'+i);
        if(el) el.textContent = (c.val*e).toFixed(c.dec);
      });
      if(k<1) monte = requestAnimationFrame(pas);
    })();
  }

  var traces = [];                        /* la traînée du corps */

  function poseCorps(p){
    var box = document.getElementById('kxcorps');
    if(!box) return;
    if(p.x === undefined){ box.style.display = 'none'; return; }   /* corps non suivi */
    var h = (p.h || 0.5), w = h * 0.42;   /* une silhouette est deux fois plus haute que large */
    var cx = p.x, cy = (p.y === undefined ? 0.5 : p.y);
    box.style.display = 'block';
    box.style.left   = (100*(cx - w/2)).toFixed(2)+'%';
    box.style.top    = (100*(cy - h/2)).toFixed(2)+'%';
    box.style.width  = (100*w).toFixed(2)+'%';
    box.style.height = (100*h).toFixed(2)+'%';
    /* la couleur suit la vigueur : bleu au repos, orange sur l'élan */
    var vmax = D.vmax || 1, r = Math.max(0, Math.min(1, p.v / vmax));
    var teinte = r < 0.33 ? '122,162,255' : (r < 0.66 ? '20,214,196' : '249,115,22');
    box.style.setProperty('--kxc', teinte);
    var eq = box.querySelectorAll('i');
    for(var q=0;q<eq.length;q++) eq[q].style.borderColor = 'rgba('+teinte+',.95)';
    var halo = box.querySelector('.kx-halo');
    if(halo){ halo.style.borderColor = 'rgba('+teinte+',.6)';
              halo.style.boxShadow = '0 0 22px rgba('+teinte+',.35) inset,0 0 18px rgba('+teinte+',.28)'; }
    var sc = document.getElementById('kxs');
    var pt = document.createElement('div');
    pt.className = 'kx-trace';
    pt.style.left = (100*cx).toFixed(2)+'%';
    pt.style.top  = (100*cy).toFixed(2)+'%';
    sc.appendChild(pt);
    traces.push(pt);
    if(traces.length > 26){ var vieux = traces.shift(); if(vieux.parentNode) vieux.parentNode.removeChild(vieux); }
    traces.forEach(function(el,k){ el.style.opacity = (0.05 + 0.5*k/traces.length).toFixed(2); });
  }

  function suit(){                        /* la vitesse et les pas suivent l'image */
    var t = v.currentTime, s = D.serie, i = Math.min(s.length-1,
              Math.max(0, Math.round(t*(s.length-1)/Math.max(0.01,D.duree))));
    if(s.length){
      document.getElementById('kxv1').textContent = s[i].v.toFixed(2);
      document.getElementById('kxp1').textContent = s[i].n;
      poseCorps(s[i]);
      osc(i);
      var et = document.getElementById('kxetv');
      if(et) et.textContent = s[i].v.toFixed(2) + ' ' + D.unite_v;
      var el = document.getElementById('kxetl');
      if(el) el.textContent = s[i].n + ' pas';
    }
    var l = document.getElementById('kxline');
    if(l) l.style.top = (28 + 44*Math.abs(Math.sin(t*1.1))).toFixed(1)+'%';
    var m = Math.floor(t/60), sec = Math.floor(t%60);
    var tc = document.getElementById('kxt');
    if(tc) tc.textContent = (m<10?'0':'')+m+':'+(sec<10?'0':'')+sec;
    var bar = document.getElementById('kxbar');
    if(bar && v.duration) bar.style.width = (100*t/v.duration).toFixed(1)+'%';
  }

  /* oscillogramme : la vitesse des dernières secondes */
  function osc(i){
    var s = D.serie, deb = Math.max(0, i-60), pts = [], remp = [];
    var vs = s.slice(deb, i+1).map(function(p){return p.v;});
    if(!vs.length) return;
    var mx = Math.max.apply(null, vs) || 1;
    vs.forEach(function(val, k){
      var x = 600*k/Math.max(1, vs.length-1), y = 44 - 40*(val/mx);
      pts.push(x.toFixed(1)+','+y.toFixed(1));
    });
    var el = document.getElementById('kxosc');
    if(el) el.setAttribute('points', pts.join(' '));
    var f = document.getElementById('kxoscf');
    if(f) f.setAttribute('points', '0,46 ' + pts.join(' ') + ' 600,46');
  }

  /* les métriques : cachées par défaut, dévoilées par le bouton */
  (function(){
    var liste = document.getElementById('kxlist');
    if(!liste) return;
    (D.metriques || []).forEach(function(m){
      var d = document.createElement('div');
      d.className = 'kx-met';
      d.innerHTML = '<span>'+m.nom+'</span><b>'+m.val+
                    (m.unite ? '<i>'+m.unite+'</i>' : '')+'</b>';
      liste.appendChild(d);
    });
    var bm = document.getElementById('kxmet'), pan = document.getElementById('kxpan');
    var bl = document.getElementById('kxleg'), legp = document.getElementById('kxlegp');
    if(bm && pan) bm.addEventListener('click', function(){
      if(legp) legp.classList.remove('ouvert');
      var ouvert = pan.classList.toggle('ouvert');
      bm.textContent = ouvert ? 'masquer' : 'métriques';
    });
    if(bl && legp) bl.addEventListener('click', function(){
      if(pan){ pan.classList.remove('ouvert'); if(bm) bm.textContent = 'métriques'; }
      legp.classList.toggle('ouvert');
    });
  })();

  /* ─────────────────────────────────────────────────────────────────
     La musique du mouvement.

     La partition est calculée côté serveur à partir des mesures ; ici on
     ne fait que la jouer, calée sur l'horloge de la vidéo. Le son ne peut
     démarrer que sur un geste de l'utilisateur — c'est la règle de tous
     les navigateurs — d'où le bouton, et le silence par défaut.
     ───────────────────────────────────────────────────────────────── */
  (function(){
    var M = D.musique;
    var bs = document.getElementById('kxson');
    if(!bs) return;
    if(!M || !M.accords || !M.accords.length){ bs.style.display = 'none'; return; }

    var ctx = null, maitre = null, echo = null, joue = false, minuteur = null;
    var faitA = {}, faitN = {};              /* ce qui a déjà sonné, ce tour-ci */
    var dernier = 0;

    function hz(midi, cents){
      return 440 * Math.pow(2, (midi - 69) / 12) * Math.pow(2, (cents || 0) / 1200);
    }

    function ouvre(){
      var AC = window.AudioContext || window.webkitAudioContext;
      if(!AC) return false;
      ctx = new AC();
      maitre = ctx.createGain();
      maitre.gain.value = 0.0001;
      var filtre = ctx.createBiquadFilter();
      filtre.type = 'lowpass';
      filtre.frequency.value = 4200;
      /* un écho court donne l'espace, sans réverbération coûteuse */
      echo = ctx.createDelay(1.0);
      echo.delayTime.value = 0.28;
      var retour = ctx.createGain();
      retour.gain.value = 0.26;
      var envoi = ctx.createGain();
      envoi.gain.value = 0.30;
      maitre.connect(filtre); filtre.connect(ctx.destination);
      maitre.connect(envoi); envoi.connect(echo);
      echo.connect(retour); retour.connect(echo); echo.connect(ctx.destination);
      maitre.gain.linearRampToValueAtTime(0.16, ctx.currentTime + 1.2);
      return true;
    }

    /* une voix : deux ondes légèrement écartées, enveloppe douce */
    function voix(freq, quand, duree, gain, timbre, cents){
      var g = ctx.createGain();
      g.gain.setValueAtTime(0.0001, quand);
      var attaque = timbre === 'nappe' ? 0.35 : 0.012;
      g.gain.exponentialRampToValueAtTime(Math.max(0.0002, gain), quand + attaque);
      g.gain.exponentialRampToValueAtTime(0.0001, quand + duree);
      g.connect(maitre);
      [0, 1].forEach(function(k){
        var o = ctx.createOscillator();
        o.type = timbre === 'basse' ? 'sine' : 'triangle';
        var ec = (k ? (cents || 0) + 4 : -(cents || 0) - 4);
        o.frequency.setValueAtTime(freq * Math.pow(2, ec / 1200), quand);
        o.connect(g);
        o.start(quand);
        o.stop(quand + duree + 0.06);
      });
    }

    function rendu(){
      if(!ctx || !joue || v.paused) return;
      var t = v.currentTime;
      if(t < dernier - 0.4){ faitA = {}; faitN = {}; }    /* la vidéo a rebouclé */
      dernier = t;
      var horizon = t + 0.22, maintenant = ctx.currentTime;

      M.accords.forEach(function(a, i){
        if(faitA[i] || a.t > horizon || a.t < t - 0.25) return;
        faitA[i] = 1;
        var quand = maintenant + Math.max(0, a.t - t);
        a.n.forEach(function(p, k){
          voix(hz(p, (k - 1) * M.detune * 0.5), quand, a.d * 0.96,
               0.055 - 0.008 * k, 'nappe', M.detune);
        });
        voix(hz(a.b), quand, Math.min(a.d, 1.6), 0.075, 'basse', 0);
      });

      M.notes.forEach(function(nt, i){
        if(faitN[i] || nt.t > horizon || nt.t < t - 0.25) return;
        faitN[i] = 1;
        voix(hz(nt.p, M.detune), maintenant + Math.max(0, nt.t - t),
             nt.d, 0.085 * nt.g, 'cloche', M.detune);
      });
    }

    bs.addEventListener('click', function(){
      if(!ctx && !ouvre()){ bs.style.display = 'none'; return; }
      if(ctx.state === 'suspended') ctx.resume();
      joue = !joue;
      bs.classList.toggle('actif', joue);
      bs.textContent = joue ? '♪ ' + M.mode : '♪ son';
      if(joue){
        faitA = {}; faitN = {};
        maitre.gain.cancelScheduledValues(ctx.currentTime);
        maitre.gain.linearRampToValueAtTime(0.16, ctx.currentTime + 0.8);
        if(!minuteur) minuteur = setInterval(rendu, 60);
      }else{
        maitre.gain.cancelScheduledValues(ctx.currentTime);
        maitre.gain.linearRampToValueAtTime(0.0001, ctx.currentTime + 0.5);
      }
    });
  })();

  /* agrandissement : vrai plein écran si le navigateur l'autorise, sinon
     la scène occupe toute la fenêtre — le repli marche dans tous les cas. */
  var btn = document.getElementById('kxplein');
  var scene = document.getElementById('kxs');
  function bascule(){
    var plein = scene.classList.contains('plein') || document.fullscreenElement;
    if(plein){
      if(document.exitFullscreen && document.fullscreenElement) document.exitFullscreen();
      scene.classList.remove('plein');
      btn.textContent = '⛶ agrandir';
    }else{
      try{
        if(scene.requestFullscreen) scene.requestFullscreen().catch(function(){
          scene.classList.add('plein');
        });
        else scene.classList.add('plein');
      }catch(e){ scene.classList.add('plein'); }
      scene.classList.add('plein');
      btn.textContent = '⛶ réduire';
    }
  }
  if(btn) btn.addEventListener('click', bascule);
  document.addEventListener('keydown', function(e){
    if(e.key === 'Escape'){ scene.classList.remove('plein');
      if(btn) btn.textContent = '⛶ agrandir'; }
  });

  v.addEventListener('play', anime);
  v.addEventListener('seeked', function(){
    if(v.currentTime < 0.2){
      anime();
      traces.forEach(function(el){ if(el.parentNode) el.parentNode.removeChild(el); });
      traces = [];
    }
  });
  v.addEventListener('timeupdate', suit);
  v.addEventListener('loadeddata', function(){ anime(); suit(); });
})();
</script>
"""
