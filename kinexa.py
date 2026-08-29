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


def _agrege(f: Dict[str, float], poids: Dict[str, float],
            age: Optional[float] = None) -> Tuple[float, float, Dict[str, float]]:
    """Moyenne pondérée des notes disponibles, avec la couverture obtenue."""
    notes, ws, detail = [], [], {}
    attendus = {"gait_speed_m_s": vitesse_attendue(age)} if age else {}
    for cle, w in poids.items():
        n = _note(cle, f.get(cle, float("nan")), attendus.get(cle))
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


def bio_mobility(f: Dict[str, float], age: Optional[float] = None) -> Dict[str, object]:
    score, couv, detail = _agrege(f, POIDS_MOBILITE, age)
    return {"score": score, "couverture": couv, "detail": detail,
            "compare_age": bool(age), "unite": "/100", "nom": "Bio-Mobility Score"}


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


def neuroplasticity(f: Dict[str, float], age: Optional[float] = None) -> Dict[str, object]:
    score, couv, detail = _agrege(f, POIDS_CONTROLE, age)
    return {"score": score, "couverture": couv, "detail": detail,
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


def kinetic_age(f: Dict[str, float], age_declare: Optional[float] = None) -> Dict[str, object]:
    v = f.get("gait_speed_m_s", float("nan"))
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
    return {"age": age, "marge": marge, "ecart": ecart, "plateau": plateau,
            "unite": "ans", "nom": "Kinetic Ageing Profile",
            "fiable": bool(np.isfinite(v) and not plateau)}


# ─────────────────────────────────────────────────────────────────────────
#  4 · Vitality Margin — l'écart à sa propre classe d'âge
# ─────────────────────────────────────────────────────────────────────────
def vitality_margin(f: Dict[str, float], age: Optional[float] = None) -> Dict[str, object]:
    """De combien la vitesse de marche dépasse — ou manque — celle attendue.

    Sans âge déclaré, la référence est la moyenne adulte : la lecture reste
    juste, mais elle compare à la population entière et non à ses pairs.
    """
    v = f.get("gait_speed_m_s", float("nan"))
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
