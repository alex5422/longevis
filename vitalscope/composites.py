"""Indices composites du mouvement.

Quatre indices construits par *combinaison* de mesures élémentaires, et non par
moyenne pondérée. Les trois premiers (IRD, SCF, ICR — le CAX est actuellement
invalidé, voir plus bas) sont chacun un rapport entre deux grandeurs mesurées
dans le même enregistrement, ce qui leur donne trois propriétés utiles :

  * **sans dimension** — aucune calibration d'échelle nécessaire, donc aucune
    dépendance à la stature déclarée ni à la distance de la caméra ;
  * **auto-normalisé** — l'éclairage, la résolution et l'appareil s'annulent au
    numérateur et au dénominateur ;
  * **intra-sujet** — comparable d'un enregistrement à l'autre chez la même
    personne, ce qui est le cas d'usage réaliste d'un outil de suivi.

Le cinquième, l'ISPT, combine deux signaux d'un geste différent (transfert
assis-debout) et n'est pas un rapport sans dimension — voir sa docstring pour
pourquoi. Son principe (la vitesse de récupération après une perturbation
posturale, plutôt que le déséquilibre initial, sépare les profils à risque) et
la faisabilité d'une mesure vidéo du centre de masse pendant ce geste
s'appuient sur : Rabuffetti et al. (Gait & Posture, 2022) sur le temps de
stabilisation après un transfert postural chez la personne âgée ; Kimura et
al. (J. Phys. Ther. Sci., 2024) sur la décroissance du balancement après une
perturbation volontaire ; Lee et al. (Gait & Posture, 2025) sur l'estimation
vidéo du centre de masse pendant un lever de chaise. Les citations complètes
sont dans la docstring de `stabilite_post_transfert`.

Statut scientifique : ces cinq indices sont des **hypothèses de recherche**.
Ils combinent des grandeurs dont l'association au vieillissement est documentée
dans la littérature ci-dessus, mais les indices eux-mêmes n'ont pas été validés
sur cohorte humaine. Ce qui est mesuré ici, c'est leur plancher de bruit et leur
réponse à une dégradation injectée — c'est-à-dire s'ils *peuvent* mesurer
quelque chose, pas ce qu'ils mesurent chez un patient.
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
# 5. ISPT — Indice de Stabilité Post-Transfert
# --------------------------------------------------------------------------- #
def stabilite_post_transfert(cx: np.ndarray, cy: np.ndarray, fps: float,
                             rises: List[Tuple[int, int]],
                             px_per_m: Optional[float] = None,
                             delai_s: float = 0.3, duree_s: float = 2.2
                             ) -> Dict[str, float]:
    """ISPT — instabilité posturale résiduelle après un transfert assis-debout.

    Le lever de chaise est lui-même une perturbation posturale auto-infligée :
    le passage du siège à l'appui bipodal déséquilibre tout le monde un peu,
    y compris un sujet en pleine forme. Ce qui distingue une bonne réserve
    d'équilibre d'une réserve entamée, ce n'est pas ce déséquilibre initial —
    universel — mais la vitesse à laquelle il se résorbe : chez des sujets âgés,
    ce temps de stabilisation après un transfert postural est plus du double
    en institution que chez des sujets autonomes du même âge, à transfert
    identique (Rabuffetti et al., Gait & Posture 2022) — c'est la vitesse de
    récupération, pas le déséquilibre initial, qui sépare les deux groupes.
    Kimura et al. (J. Phys. Ther. Sci. 2024) montrent le même écart sur une
    perturbation volontaire (un pas descendant, plutôt qu'une chaise) : le
    balancement résiduel décroît plus vite chez les sujets jeunes que chez les
    âgés, avec un profil temporel mesurable seconde par seconde — c'est
    exactement la décroissance que cette fenêtre cherche à capturer.

    Sans rien ajouter au protocole : le lever de chaise déjà filmé pour compter
    les transferts EST la perturbation, et sa propre vidéo contient déjà la
    récupération. L'estimation du centre de masse par vidéo pendant un
    transfert assis-debout, y compris depuis une caméra de smartphone, est
    elle-même une approche déjà publiée (Lee et al., Gait & Posture 2025), ce
    qui ancre la faisabilité de la mesure — pas la validité de l'indice
    lui-même, qui reste à établir sur cohorte (voir Statut ci-dessous).

    L'indice isole une fenêtre qui commence `delai_s` après que la silhouette
    a atteint sa hauteur debout (le temps que le premier à-coup s'amortisse)
    et mesure le trajet du centre de masse qui subsiste pendant les
    `duree_s` secondes suivantes, une fois que le sujet est censé être
    stabilisé. Une valeur élevée dit : l'équilibre n'est pas encore acquis
    alors que le corps a fini de se redresser.

    Statut : hypothèse de recherche, comme les indices ci-dessus — la fenêtre
    (0,3 s puis 2,2 s) est choisie par raisonnement physiologique, pas
    calibrée sur cohorte. Contrairement à IRD/SCF/ICR, ce n'est pas un
    rapport sans dimension : il n'y a pas de dénominateur naturel pour un
    résidu de balancement (en forcer un serait arbitraire), donc la valeur
    reste en pixels/seconde — en mm/s si une échelle est fournie — et se lit
    en suivi longitudinal chez un même sujet, pas en comparaison absolue
    entre personnes.
    """
    out: Dict[str, float] = {"ispt_residuel_px_s": float("nan"), "ispt_n_leves": 0.0}
    if px_per_m:
        out["ispt_residuel_mm_s"] = float("nan")
    if not rises or not np.isfinite(fps) or fps <= 0:
        return out

    n = int(min(len(cx), len(cy)))
    fc = min(3.0, fps / 2 - 0.5)
    if fc <= 0.05:
        return out

    vals: List[float] = []
    for (_, b) in rises:
        i0 = int(b + delai_s * fps)
        i1 = min(n, i0 + int(duree_s * fps))
        if i0 >= n or i1 - i0 < int(0.8 * fps):
            continue
        x = dsp.detrend_smoothness(cx[i0:i1], lam=300.0)
        y = dsp.detrend_smoothness(cy[i0:i1], lam=300.0)
        x = dsp.bandpass(x, fps, 0.05, fc, order=2)
        y = dsp.bandpass(y, fps, 0.05, fc, order=2)
        path = float(np.sum(np.hypot(np.diff(x), np.diff(y))) / (len(x) / fps))
        if np.isfinite(path):
            vals.append(path)

    if vals:
        out["ispt_residuel_px_s"] = float(np.median(vals))
        out["ispt_n_leves"] = float(len(vals))
        if px_per_m:
            out["ispt_residuel_mm_s"] = float(out["ispt_residuel_px_s"] / px_per_m * 1000.0)
    return out


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
