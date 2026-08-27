"""Indices composites du mouvement.

Trois indices construits par *combinaison* de mesures élémentaires, et non par
moyenne pondérée. Chacun est un rapport entre deux grandeurs mesurées dans le
même enregistrement, ce qui lui donne trois propriétés utiles :

  * **sans dimension** — aucune calibration d'échelle nécessaire, donc aucune
    dépendance à la stature déclarée ni à la distance de la caméra ;
  * **auto-normalisé** — l'éclairage, la résolution et l'appareil s'annulent au
    numérateur et au dénominateur ;
  * **intra-sujet** — comparable d'un enregistrement à l'autre chez la même
    personne, ce qui est le cas d'usage réaliste d'un outil de suivi.

Statut scientifique : ces trois indices sont des **hypothèses de recherche**.
Ils combinent des grandeurs dont l'association au vieillissement est documentée,
mais les indices eux-mêmes n'ont pas été validés sur cohorte humaine. Ce qui est
mesuré ici, c'est leur plancher de bruit et leur réponse à une dégradation
injectée — c'est-à-dire s'ils *peuvent* mesurer quelque chose, pas ce qu'ils
mesurent chez un patient.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sps

from . import dsp


# --------------------------------------------------------------------------- #
# 1. IRD — Indice de Réserve Dynamique
# --------------------------------------------------------------------------- #
def indice_reserve_dynamique(turn_dur_s: float, step_time_s: float) -> float:
    """IRD — nombre de pas « perdus » à chaque changement de direction.

    Le demi-tour est le moment où la marche cesse d'être automatique : il faut
    freiner, se réorienter, relancer. C'est là que la réserve motrice se voit
    en premier, bien avant que la vitesse en ligne droite ne baisse.

    En exprimant la durée du demi-tour en *pas* plutôt qu'en secondes, l'indice
    devient indépendant de la cadence propre du sujet : un marcheur rapide et un
    marcheur lent qui paient le même prix relatif obtiennent le même score.

    Lecture : 2 = demi-tour fluide, coûtant l'équivalent de deux pas.
              6 et plus = manœuvre laborieuse.
    """
    if not (np.isfinite(turn_dur_s) and np.isfinite(step_time_s)) or step_time_s <= 0:
        return float("nan")
    return float(turn_dur_s / step_time_s)


# --------------------------------------------------------------------------- #
# 2. SCF — Signature Cinétique de Foulée
# --------------------------------------------------------------------------- #
def signature_cinetique_foulee(step_length_stature: float, cadence_spm: float) -> float:
    """SCF — rapport entre l'amplitude du pas et la fréquence choisie.

    À vitesse donnée, on peut faire des pas longs et lents ou courts et rapides.
    Le rapport entre les deux est remarquablement stable chez l'adulte sain et
    se déplace vers les pas courts et rapides avec l'âge, la peur de tomber et
    la perte de force : c'est un choix de stratégie, pas de performance.

    D'où l'intérêt : la vitesse peut rester normale alors que la stratégie a
    déjà basculé. L'indice voit ce basculement avant que la vitesse ne bouge.

    Normalisé à la stature, il ne demande aucune calibration métrique.
    Lecture : valeur haute = pas amples et posés ; valeur basse = pas hachés.
    """
    if not (np.isfinite(step_length_stature) and np.isfinite(cadence_spm)) or cadence_spm <= 0:
        return float("nan")
    return float(step_length_stature / (cadence_spm / 60.0))


# --------------------------------------------------------------------------- #
# 3. CAX — Cohérence Axiale
# --------------------------------------------------------------------------- #
def coherence_axiale(spread: np.ndarray, com_vertical: np.ndarray,
                           fps: float, passes: List[Tuple[int, int]],
                           step_hz: float) -> Dict[str, float]:
    """CAX — verrouillage entre le rythme des jambes et l'oscillation du tronc.

    À chaque pas, le centre de masse monte et redescend. Chez un marcheur
    coordonné, ce mouvement du tronc est asservi au rythme des jambes : les deux
    signaux sont verrouillés en phase. Quand la coordination se dégrade, le
    tronc se met à osciller pour son propre compte.

    C'est ce que la vidéo permet et qu'un capteur unique ne permet pas : les
    deux signaux viennent de la même image, donc ils sont synchronisés par
    construction, sans horloge commune à établir.

    ⚠️ INDICE NON UTILISABLE EN L'ÉTAT — ne pas interpréter, ne pas présenter
    comme une mesure. Voir README, section « Diagnostic du CAX ». Trois variantes
    d'estimation ont donné trois sens de variation différents face au même
    découplage injecté : l'indice n'est pas stable, et l'amplitude du signal de
    tronc (~2 px) est du même ordre que le bruit de segmentation. Il est calculé
    et exposé pour permettre la poursuite du diagnostic, rien de plus.

    Mesuré par cohérence spectrale à la fréquence du pas, sur les trajets
    rectilignes. Lecture visée : 1 = verrouillage parfait, 0 = indépendance.
    """
    out = {"cax_coherence": float("nan"), "cax_n_segments": 0.0}
    if not np.isfinite(step_hz) or step_hz <= 0 or not passes:
        return out

    vals = []
    for a, b in passes:
        if b - a < int(2.5 * fps):
            continue
        x = dsp.detrend_smoothness(spread[a:b], lam=80.0)
        y = dsp.detrend_smoothness(com_vertical[a:b], lam=80.0)
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            continue
        nper = int(min(len(x), max(32, 2.5 * fps / step_hz)))
        f, cxy = sps.coherence(x, y, fs=fps, nperseg=nper, noverlap=nper // 2)
        # Cohérence dans une fenêtre étroite autour de la fréquence du pas
        # On lit la cohérence à la fréquence du pas, sans chercher le maximum
        # dans une fenêtre : prendre le maximum biaise l'indice vers le haut
        # d'autant plus que le spectre est bruité — c'est-à-dire exactement
        # quand la marche est irrégulière. Le défaut inversait le sens de
        # variation de l'indice.
        i0 = int(np.argmin(np.abs(f - step_hz)))
        if 0 <= i0 < len(cxy):
            vals.append(float(cxy[i0]))

    if vals:
        out["cax_coherence"] = float(np.median(vals))
        out["cax_n_segments"] = float(len(vals))
    return out


# --------------------------------------------------------------------------- #
# 4. ICR — Indice de Cinétique de Relance
# --------------------------------------------------------------------------- #
def indice_cinetique_relance(cx: np.ndarray, fps: float,
                             turns: List[Tuple[int, int]],
                             cruise_px_s: float) -> Dict[str, float]:
    """ICR — rapport entre le temps de relance et le temps de freinage.

    Freiner et relancer sollicitent deux capacités différentes : l'une est
    largement excentrique et passive, l'autre demande de produire de la
    puissance. Elles se dégradent à des rythmes différents, la production de
    puissance en premier.

    Le rapport des deux temps est ce qui rend l'indice utile : un demi-tour
    globalement lent allonge le numérateur *et* le dénominateur, donc n'affecte
    pas le rapport. L'ICR ne mesure pas la lenteur du demi-tour — c'est le rôle
    de l'IRD — mais l'asymétrie entre arrêter et repartir. Les deux indices sont
    construits pour être indépendants.

    Lecture : 1,0 = freinage et relance équilibrés.
              1,5 et plus = la relance coûte nettement plus que l'arrêt.
    """
    out = {"icr_relance": float("nan"), "icr_t_freinage_s": float("nan"),
           "icr_t_relance_s": float("nan"), "icr_n_virages": 0.0}
    if not turns or not np.isfinite(cruise_px_s) or cruise_px_s <= 0:
        return out

    v = np.abs(_lowpass(np.gradient(cx) * fps, fps, 2.0))
    seuil = 0.90 * cruise_px_s
    ratios, tds, tas = [], [], []

    for a, b in turns:
        marge = int(2.5 * fps)
        i0, i1 = max(0, a - marge), min(len(v), b + marge)
        if i1 - i0 < int(1.0 * fps):
            continue
        creux = i0 + int(np.argmin(v[i0:i1]))

        # Freinage : dernier instant au-dessus du seuil avant le creux
        j = creux
        while j > i0 and v[j] < seuil:
            j -= 1
        # Relance : premier instant au-dessus du seuil après le creux
        k = creux
        while k < i1 - 1 and v[k] < seuil:
            k += 1
        td, ta = (creux - j) / fps, (k - creux) / fps
        if 0.05 < td < 5.0 and 0.05 < ta < 5.0:
            ratios.append(ta / td); tds.append(td); tas.append(ta)

    if ratios:
        out.update({"icr_relance": float(np.median(ratios)),
                    "icr_t_freinage_s": float(np.median(tds)),
                    "icr_t_relance_s": float(np.median(tas)),
                    "icr_n_virages": float(len(ratios))})
    return out


def _lowpass(x: np.ndarray, fps: float, fc: float) -> np.ndarray:
    nyq = fps / 2
    if fc >= nyq:
        return x
    b, a = sps.butter(3, fc / nyq, "low")
    return sps.filtfilt(b, a, x, padlen=min(30, len(x) - 1))


# --------------------------------------------------------------------------- #
# Assemblage
# --------------------------------------------------------------------------- #
def composite_indices(features: Dict[str, float],
                      signals: Optional[Dict[str, object]] = None,
                      passes: Optional[List[Tuple[int, int]]] = None,
                      turns: Optional[List[Tuple[int, int]]] = None
                      ) -> Dict[str, float]:
    """Calcule les trois indices à partir des mesures déjà extraites."""
    g = lambda k: features.get(k, float("nan"))
    out: Dict[str, float] = {
        "ird_reserve_dynamique": indice_reserve_dynamique(g("turn_mean_dur_s"), g("step_time_s")),
        "scf_signature_foulee": signature_cinetique_foulee(g("step_length_stature"),
                                                  g("cadence_spm")),
    }
    if signals is not None and turns is not None:
        cx = np.asarray(signals.get("cx", []), dtype=float)
        if cx.size:
            out.update(indice_cinetique_relance(
                cx, float(signals.get("fps", 30.0)), turns,
                features.get("gait_speed_px_s", float("nan"))))

    if signals is not None and passes:
        spread = np.asarray(signals.get("spread", []), dtype=float)
        # Signal de tronc si disponible, centre de masse global sinon
        cy = np.asarray(signals.get("trunk_y", signals.get("cy", [])), dtype=float)
        step_hz = g("cadence_spm") / 60.0 if np.isfinite(g("cadence_spm")) else float("nan")
        if spread.size and cy.size:
            out.update(coherence_axiale(spread, cy, float(signals.get("fps", 30.0)),
                                              passes, step_hz))
    out.setdefault("cax_coherence", float("nan"))
    out.setdefault("icr_relance", float("nan"))
    return out
