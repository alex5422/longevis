"""Les quatre biomarqueurs de l'Institut Kinexa.

Quatre lectures, calculées à partir des marqueurs déjà mesurés par le reste du
programme. Aucune n'est un diagnostic : ce sont des agrégats lisibles, dont on
donne toujours la couverture — c'est-à-dire la part de mesures réellement
disponibles derrière le chiffre.

  Bio-Mobility Score      /100   ce que le corps sait encore faire : vitesse,
                                 longueur de pas, équilibre, transfert
  Neuroplasticity Index   /100   la qualité du contrôle : régularité, fluidité,
                                 harmonie, absence de tremblement
  Kinetic Ageing Profile  ± ans  l'âge locomoteur lu dans la vitesse de marche,
                                 avec sa marge d'incertitude
  Vitality Margin         %      l'écart de vitesse à sa propre classe d'âge

Le dépistage ICOPE reste disponible plus bas (`icope`), mais il n'appartient
pas à cette façade : sa place est du côté clinique, dans OrthoScope.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np

from . import dsp
from .config import REFERENCE_NORMS


# ─────────────────────────────────────────────────────────────────────────
#  Outils communs
# ─────────────────────────────────────────────────────────────────────────
def _note_mouvement(cle: str, valeur: float) -> float:
    """Même échelle 0–100, mais sur les normes des mesures génériques."""
    norme = NORMES_MOUVEMENT.get(cle)
    if norme is None or valeur is None or not np.isfinite(valeur):
        return float("nan")
    mu, sd, sens = norme
    z = dsp.robust_z(valeur, mu, sd, sens)
    if not np.isfinite(z):
        return float("nan")
    return float(np.clip(50 + 20 * z, 0, 100))


def _note(cle: str, valeur: float, attendu: Optional[float] = None) -> float:
    """0 à 100 à partir de l'écart à la référence, dans le bon sens.

    Un z de 0 vaut 50 ; ± 2,5 écarts-types couvrent toute l'échelle. Si un âge
    est déclaré, `attendu` remplace la moyenne générale par celle de la classe
    d'âge : un marcheur de 80 ans n'est plus noté contre un adulte de 40.
    """
    norme = REFERENCE_NORMS.get(cle)
    if norme is None or valeur is None or not np.isfinite(valeur):
        return float("nan")
    mu, sd, sens = norme
    if attendu is not None and np.isfinite(attendu):
        mu = attendu
    z = dsp.robust_z(valeur, mu, sd, sens)
    if not np.isfinite(z):
        return float("nan")
    return float(np.clip(50 + 20 * z, 0, 100))


def vitesse_attendue(age: Optional[float]) -> float:
    """Vitesse de marche confortable attendue à cet âge (m/s)."""
    if age is None or not np.isfinite(age):
        return float("nan")
    ages = np.array([a for a, _ in COURBE_AGE], dtype=float)
    vits = np.array([v for _, v in COURBE_AGE], dtype=float)
    return float(np.interp(age, ages, vits))


#  Certaines grandeurs ont un équivalent mesurable autrement : de profil, la
#  longueur de pas se lit sur l'écartement des pieds plutôt que sur la vitesse.
EQUIVALENTS = {"step_length_stature": "step_length_spread_stature"}

#  Tout mouvement n'a pas de pas. Quand la marche n'a rien donné, on lit les
#  mêmes qualités sur les grandeurs génériques : amplitude, rythme, fluidité,
#  régularité. Les échelles diffèrent, d'où une norme propre à chacune.
#  Même convention que REFERENCE_NORMS : (moyenne, écart-type, sens),
#  le sens valant +1 quand « plus grand » est favorable, −1 sinon.
#  Repères calés sur des gestes de référence : balancement lent, flexions,
#  gestes des bras, mouvement heurté. Ils valent pour la démonstration et
#  seront réétalonnés sur des sujets réels.
NORMES_MOUVEMENT = {
    "move_amplitude_stature":    (0.55, 0.25, +1),
    "move_peak_speed_stature":   (1.30, 0.50, +1),
    "move_rate_cpm":             (55.0, 30.0, +1),
    "move_cycle_cv_pct":         (9.0, 7.0, -1),
    "move_sparc":                (-7.0, 3.5, +1),
    "move_active_pct":           (62.0, 18.0, +1),
}


#  Un signal bruité peut produire une valeur absurde — un rapport harmonique à
#  1944, une cadence à 400 pas par minute. Ces valeurs sont écartées du calcul
#  et de l'affichage : mieux vaut une case vide qu'un chiffre qui décrédibilise.
PLAGES_PLAUSIBLES = {
    "gait_speed_m_s": (0.15, 2.5),
    "cadence_spm": (30.0, 200.0),
    "step_length_m": (0.10, 1.20),
    "step_length_stature": (0.05, 0.95),
    "step_time_cv_pct": (0.0, 60.0),
    "stride_time_cv_pct": (0.0, 60.0),
    "step_asymmetry_pct": (0.0, 60.0),
    "harmonic_ratio": (0.2, 12.0),
    "gait_sparc": (-30.0, 0.0),
    "com_bob_stature_pct": (0.0, 25.0),
    "turn_mean_dur_s": (0.2, 15.0),
    "sway_rms_ap_mm": (0.5, 120.0),
    "sway_rms_ml_mm": (0.5, 120.0),
    "sts_mean_dur_s": (0.4, 12.0),
    "move_amplitude_stature": (0.01, 3.0),
    "move_peak_speed_stature": (0.02, 8.0),
    "move_rate_cpm": (5.0, 250.0),
    "move_cycle_cv_pct": (0.0, 90.0),
    "move_sparc": (-40.0, 0.0),
    "move_active_pct": (0.0, 100.0),
}


def plausible(cle: str, valeur) -> bool:
    """La valeur est-elle dans une plage physiquement acceptable ?"""
    if not isinstance(valeur, (int, float)) or not np.isfinite(valeur):
        return False
    bornes = PLAGES_PLAUSIBLES.get(cle)
    if bornes is None:
        return True
    return bool(bornes[0] <= float(valeur) <= bornes[1])


def _valeur(f: Dict[str, float], cle: str) -> float:
    v = f.get(cle, float("nan"))
    if not plausible(cle, v):
        v = float("nan")
    if not np.isfinite(v) and cle in EQUIVALENTS:
        remplacant = f.get(EQUIVALENTS[cle], float("nan"))
        v = remplacant if plausible(EQUIVALENTS[cle], remplacant) else float("nan")
    return v


def _agrege(f: Dict[str, float], poids: Dict[str, float],
            age: Optional[float] = None) -> Tuple[float, float, Dict[str, float]]:
    """Moyenne pondérée des notes disponibles, avec la couverture obtenue."""
    notes, ws, detail = [], [], {}
    attendus = {"gait_speed_m_s": vitesse_attendue(age)} if age else {}
    for cle, w in poids.items():
        n = _note(cle, _valeur(f, cle), attendus.get(cle))
        if np.isfinite(n):
            notes.append(n)
            ws.append(w)
            detail[cle] = round(n, 1)
    if not notes:
        return float("nan"), 0.0, {}
    score = float(np.average(notes, weights=ws))
    couverture = float(sum(ws) / sum(poids.values()))
    return score, couverture, detail


# ─────────────────────────────────────────────────────────────────────────
#  1 · Bio-Mobility Score — la capacité
# ─────────────────────────────────────────────────────────────────────────
POIDS_MOBILITE = {
    "gait_speed_m_s": 0.34,        # le marqueur le mieux établi de la fonction
    "step_length_stature": 0.16,
    "cadence_spm": 0.10,
    "sts_mean_dur_s": 0.16,        # transfert assis-debout
    "sway_rms_ap_mm": 0.12,        # équilibre debout
    "sway_rms_ml_mm": 0.12,
}


POIDS_MOBILITE_LIBRE = {
    "move_amplitude_stature": 0.40,
    "move_peak_speed_stature": 0.35,
    "move_active_pct": 0.25,
}


def _agrege_libre(f: Dict[str, float], poids: Dict[str, float]):
    notes, ws, detail = [], [], {}
    for cle, w in poids.items():
        brut = f.get(cle, float("nan"))
        n = _note_mouvement(cle, brut if plausible(cle, brut) else float("nan"))
        if np.isfinite(n):
            notes.append(n)
            ws.append(w)
            detail[cle] = round(n, 1)
    if not notes:
        return float("nan"), 0.0, {}
    return (float(np.average(notes, weights=ws)),
            float(sum(ws) / sum(poids.values())), detail)


def bio_mobility(f: Dict[str, float], age: Optional[float] = None) -> Dict[str, object]:
    score, couv, detail = _agrege(f, POIDS_MOBILITE, age)
    libre = False
    if not np.isfinite(score) or couv < 0.25:      # pas de marche exploitable
        s2, c2, d2 = _agrege_libre(f, POIDS_MOBILITE_LIBRE)
        if np.isfinite(s2):
            score, couv, detail, libre = s2, c2, d2, True
    return {"score": score, "couverture": couv, "detail": detail, "libre": libre,
            "compare_age": bool(age) and not libre,
            "unite": "/100", "nom": "Bio-Mobility Score"}


# ─────────────────────────────────────────────────────────────────────────
#  2 · Neuroplasticity Index — la qualité du contrôle
# ─────────────────────────────────────────────────────────────────────────
POIDS_CONTROLE = {
    "step_time_cv_pct": 0.26,      # régularité du pas
    "stride_time_cv_pct": 0.14,
    "harmonic_ratio": 0.20,        # tenue du tronc
    "gait_sparc": 0.18,            # douceur du mouvement
    "step_asymmetry_pct": 0.12,
    "tremor_power_norm": 0.10,     # présent seulement en analyse faciale
}


POIDS_CONTROLE_LIBRE = {
    "move_cycle_cv_pct": 0.40,
    "move_sparc": 0.35,
    "move_rate_cpm": 0.25,
}


def neuroplasticity(f: Dict[str, float], age: Optional[float] = None) -> Dict[str, object]:
    score, couv, detail = _agrege(f, POIDS_CONTROLE, age)
    libre = False
    if not np.isfinite(score) or couv < 0.25:
        s2, c2, d2 = _agrege_libre(f, POIDS_CONTROLE_LIBRE)
        if np.isfinite(s2):
            score, couv, detail, libre = s2, c2, d2, True
    return {"score": score, "couverture": couv, "detail": detail, "libre": libre,
            "unite": "/100", "nom": "Neuroplasticity Index"}


# ─────────────────────────────────────────────────────────────────────────
#  3 · Kinetic Ageing Profile — l'âge locomoteur
# ─────────────────────────────────────────────────────────────────────────
#  Vitesse de marche confortable par décennie, hommes et femmes confondus,
#  d'après la méta-analyse de Bohannon & Williams Andrews (2011). La courbe
#  est plate avant 60 ans : en deçà, la vitesse ne date pas une personne, et
#  la marge d'erreur retournée le dit.
COURBE_AGE = [(25, 1.39), (35, 1.43), (45, 1.43), (55, 1.39),
              (65, 1.34), (75, 1.26), (85, 0.97), (95, 0.72)]


def _vitesse_equivalente(f: Dict[str, float]) -> float:
    """Vitesse « équivalente » d'un mouvement sans marche.

    La vitesse de pointe du geste, rapportée à la stature, est convertie en
    l'échelle de la vitesse de marche. C'est une correspondance grossière,
    destinée à la démonstration : elle place le sujet sur la courbe sans
    prétendre à la précision d'une mesure de marche.
    """
    p = f.get("move_peak_speed_stature", float("nan"))
    if not np.isfinite(p):
        return float("nan")
    return float(np.clip(0.62 * p + 0.28, 0.45, 1.60))


def kinetic_age(f: Dict[str, float], age_declare: Optional[float] = None) -> Dict[str, object]:
    v = f.get("gait_speed_m_s", float("nan"))
    approx = False
    if not np.isfinite(v):
        v = _vitesse_equivalente(f)
        approx = np.isfinite(v)
    if not np.isfinite(v):
        return {"age": float("nan"), "marge": float("nan"), "ecart": float("nan"),
                "unite": "ans", "nom": "Kinetic Ageing Profile", "fiable": False}

    ages = np.array([a for a, _ in COURBE_AGE], dtype=float)
    vits = np.array([s for _, s in COURBE_AGE], dtype=float)
    # la courbe décroît après 45 ans : on interpole sur la partie décroissante
    dec_a, dec_v = ages[3:], vits[3:]
    if v >= dec_v[0]:
        age = float(dec_a[0])
        plateau = True
    elif v <= dec_v[-1]:
        age = float(dec_a[-1])
        plateau = False
    else:
        age = float(np.interp(v, dec_v[::-1], dec_a[::-1]))
        plateau = False

    # marge : 6 % d'erreur sur l'échelle (taille déclarée) plus le bruit de la
    # mesure, traduits en années par la pente locale de la courbe.
    pente = abs(np.gradient(dec_v, dec_a).mean()) or 0.004     # m/s par an
    marge = float(np.clip(0.06 * v / pente, 4, 20))
    if plateau:
        marge = 20.0                                  # la vitesse ne date plus

    ecart = float(age - age_declare) if (age_declare and np.isfinite(age_declare)) else float("nan")
    if approx:
        marge = max(marge, 12.0)                  # lecture indirecte : marge élargie
    return {"age": age, "marge": marge, "ecart": ecart, "plateau": plateau,
            "approx": approx, "unite": "ans", "nom": "Kinetic Ageing Profile",
            "fiable": bool(np.isfinite(v) and not plateau and not approx)}


# ─────────────────────────────────────────────────────────────────────────
#  4 · Vitality Margin — l'écart à sa propre classe d'âge
# ─────────────────────────────────────────────────────────────────────────
def vitality_margin(f: Dict[str, float], age: Optional[float] = None) -> Dict[str, object]:
    """De combien la vitesse de marche dépasse — ou manque — celle attendue.

    Sans âge déclaré, la référence est la moyenne adulte : la lecture reste
    juste, mais elle compare à la population entière et non à ses pairs.
    """
    v = f.get("gait_speed_m_s", float("nan"))
    if not np.isfinite(v):
        v = _vitesse_equivalente(f)
    attendu = vitesse_attendue(age)
    if not np.isfinite(attendu):
        norme = REFERENCE_NORMS.get("gait_speed_m_s")
        attendu = norme[0] if norme else float("nan")
        compare_age = False
    else:
        compare_age = True
    if not np.isfinite(v) or not np.isfinite(attendu) or attendu <= 0:
        return {"marge_pct": float("nan"), "attendu": attendu, "mesure": v,
                "compare_age": compare_age, "unite": "%", "nom": "Vitality Margin"}
    marge = 100.0 * (v - attendu) / attendu
    return {"marge_pct": float(marge), "attendu": float(attendu), "mesure": float(v),
            "compare_age": compare_age, "unite": "%", "nom": "Vitality Margin"}


# ─────────────────────────────────────────────────────────────────────────
#  Annexe · ICOPE Screening — conservé pour l'usage clinique (OrthoScope) — les six domaines de l'OMS
# ─────────────────────────────────────────────────────────────────────────
#  Une vidéo de marche renseigne la mobilité, et rien d'autre. Les cinq autres
#  domaines sont affichés vides plutôt que devinés : c'est le seul usage
#  honnête d'un dépistage.
ICOPE_DOMAINES = ["Mobilité", "Nutrition", "Vision", "Audition",
                  "Cognition", "Humeur"]


def icope(f: Dict[str, float], meta: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    meta = meta or {}
    res = {d: {"etat": "non évalué", "valeur": None, "source": None}
           for d in ICOPE_DOMAINES}

    # Mobilité : critère ICOPE — lever de chaise, ou vitesse de marche.
    sts = f.get("sts_mean_dur_s", float("nan"))
    v = f.get("gait_speed_m_s", float("nan"))
    if np.isfinite(sts):
        alerte = sts * 5 > 14.0          # cinq levers en plus de 14 s
        res["Mobilité"] = {"etat": "à revoir" if alerte else "sans signal",
                           "valeur": round(float(sts) * 5, 1),
                           "source": "cinq levers de chaise (s)"}
    elif np.isfinite(v):
        alerte = v < 0.8                 # seuil usuel de fragilité locomotrice
        res["Mobilité"] = {"etat": "à revoir" if alerte else "sans signal",
                           "valeur": round(float(v), 2),
                           "source": "vitesse de marche (m/s)"}

    evalues = sum(1 for d in ICOPE_DOMAINES if res[d]["etat"] != "non évalué")
    return {"domaines": res, "evalues": evalues, "total": len(ICOPE_DOMAINES),
            "nom": "ICOPE Screening"}


# ─────────────────────────────────────────────────────────────────────────
#  Les quatre d'un coup
# ─────────────────────────────────────────────────────────────────────────
def biomarqueurs(features: Dict[str, float], meta: Optional[Dict[str, object]] = None,
                 age_declare: Optional[float] = None) -> Dict[str, object]:
    f = {k: v for k, v in features.items() if isinstance(v, (int, float))}
    return {
        "bio_mobility": bio_mobility(f, age_declare),
        "neuroplasticity": neuroplasticity(f, age_declare),
        "kinetic_age": kinetic_age(f, age_declare),
        "vitality_margin": vitality_margin(f, age_declare),
        "icope": icope(f, meta),          # calculé, mais pas affiché côté Longévité
    }
