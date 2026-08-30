"""Paramètres de traitement et normes de référence.

IMPORTANT — statut scientifique des normes
------------------------------------------
Les valeurs `REFERENCE_NORMS` ci-dessous sont des *ordres de grandeur* issus de la
littérature (HRV : Task Force 1996, Nunan 2010 ; clignement : Bentivoglio 1997 ;
tremblement physiologique : Elble 2003). Elles servent à produire des z-scores
lisibles, PAS à poser un diagnostic. Tant que `calibrate.py` n'a pas été exécuté
sur une cohorte réelle, tout score composite doit être lu comme un indicateur
relatif de qualité de signal et de tendance, non comme une mesure de longévité.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple

# --- Bandes fréquentielles (Hz) ---------------------------------------------
PULSE_BAND: Tuple[float, float] = (0.70, 3.50)      # 42 – 210 bpm
RESP_BAND: Tuple[float, float] = (0.13, 0.50)       # 8 – 30 cycles/min
TREMOR_BAND: Tuple[float, float] = (4.0, 12.0)      # tremblement physiologique
SWAY_BAND: Tuple[float, float] = (0.10, 2.00)       # oscillation posturale
LF_BAND: Tuple[float, float] = (0.04, 0.15)         # HRV basse fréquence
HF_BAND: Tuple[float, float] = (0.15, 0.40)         # HRV haute fréquence

# --- Traitement --------------------------------------------------------------
@dataclass
class ProcessingConfig:
    target_fps: float = 30.0
    min_duration_s: float = 20.0        # en-deçà : HRV non calculable
    recommended_duration_s: float = 90.0
    detect_every_n_frames: int = 15
    bbox_smoothing: float = 0.85        # lissage exponentiel de la boîte visage
    detrend_lambda: float = 100.0       # détendance Tarvainen (smoothness priors)
    pos_window_s: float = 1.6           # fenêtre POS (Wang et al. 2017)
    texture_samples: int = 24           # nb d'imagettes de joue analysées
    min_skin_pixels: int = 150
    max_frames: int = 9000              # garde-fou mémoire (~5 min @30fps)


# --- Normes de référence : (moyenne, écart-type, sens) -----------------------
# sens = +1  → une valeur haute est *favorable*
# sens = -1  → une valeur haute est *défavorable*
REFERENCE_NORMS: Dict[str, Tuple[float, float, int]] = {
    # Cardiaque / autonome
    "hr_bpm":            (68.0, 9.0, -1),
    "hrv_sdnn_ms":       (50.0, 18.0, +1),
    "hrv_rmssd_ms":      (42.0, 20.0, +1),
    "hrv_pnn50_pct":     (18.0, 12.0, +1),
    "lf_hf_ratio":       (2.0, 1.1, -1),
    # Vasculaire / perfusion
    "perfusion_index":   (1.20, 0.55, +1),
    "pulse_rise_ratio":  (0.36, 0.07, +1),   # temps de montée / durée du cycle
    "pulse_amp_cv":      (0.28, 0.12, -1),   # variabilité d'amplitude
    # Respiratoire
    "resp_rate_cpm":     (14.0, 3.5, -1),
    # Neuromoteur
    "tremor_power_norm": (0.045, 0.030, -1),
    "sway_rms_px":       (0.55, 0.35, -1),
    "blink_rate_min":    (17.0, 7.0, -1),
    # Mouvement — marche
    # Vitesse : Bohannon & Williams Andrews (2011), méta-analyse 41 études.
    # CV du temps de pas et asymétrie : Hausdorff (2007), marqueurs de chute.
    # Rapport harmonique : Menz et al. (2003).
    "gait_speed_m_s":       (1.25, 0.20, +1),
    "cadence_spm":          (112.0, 9.0, +1),
    "step_time_cv_pct":     (3.0, 1.6, -1),
    "stride_time_cv_pct":   (2.4, 1.4, -1),
    "step_asymmetry_pct":   (2.5, 2.0, -1),
    "step_length_stature":  (0.41, 0.05, +1),
    "harmonic_ratio":       (2.80, 0.70, +1),
    "gait_sparc":           (-6.5, 1.5, +1),
    "com_bob_stature_pct":  (2.6, 0.9, -1),
    "turn_mean_dur_s":      (1.10, 0.40, -1),
    # Mouvement — équilibre debout
    "sway_rms_ap_mm":       (7.0, 3.0, -1),
    "sway_rms_ml_mm":       (5.0, 2.5, -1),
    "sway_path_mm_s":       (18.0, 8.0, -1),
    # Mouvement — transfert assis-debout
    "sts_mean_dur_s":       (1.60, 0.55, -1),
    # Dermique
    "wrinkle_index":     (0.185, 0.070, -1),
    "tone_evenness":     (0.72, 0.14, +1),
    "erythema_index":    (0.42, 0.15, -1),
}

# --- Domaines et pondérations internes ---------------------------------------
DOMAINS: Dict[str, Dict[str, float]] = {
    "marche": {
        "gait_speed_m_s": 0.34, "step_time_cv_pct": 0.16,
        "step_asymmetry_pct": 0.14, "harmonic_ratio": 0.12,
        "step_length_stature": 0.10, "cadence_spm": 0.08, "gait_sparc": 0.06,
    },
    "equilibre": {
        "sway_rms_ap_mm": 0.40, "sway_rms_ml_mm": 0.30, "sway_path_mm_s": 0.30,
    },
    "transfert": {
        "sts_mean_dur_s": 1.00,
    },
    "cardio_autonome": {
        "hrv_rmssd_ms": 0.30, "hrv_sdnn_ms": 0.25,
        "hrv_pnn50_pct": 0.15, "hr_bpm": 0.20, "lf_hf_ratio": 0.10,
    },
    "vasculaire": {
        "perfusion_index": 0.40, "pulse_rise_ratio": 0.35, "pulse_amp_cv": 0.25,
    },
    "respiratoire": {
        "resp_rate_cpm": 1.00,
    },
    "neuromoteur": {
        "tremor_power_norm": 0.40, "sway_rms_px": 0.35, "blink_rate_min": 0.25,
    },
    "dermique": {
        "wrinkle_index": 0.45, "tone_evenness": 0.35, "erythema_index": 0.20,
    },
}

# La marche pèse le plus : c'est le domaine dont l'association à la survie est
# la mieux établie dans la littérature. Les poids sont renormalisés sur les
# seuls domaines effectivement mesurés dans l'enregistrement analysé.
DOMAIN_WEIGHTS: Dict[str, float] = {
    "marche": 0.30,
    "equilibre": 0.12,
    "transfert": 0.08,
    "cardio_autonome": 0.22,
    "vasculaire": 0.12,
    "respiratoire": 0.04,
    "neuromoteur": 0.08,
    "dermique": 0.04,
}

# --- Plancher de bruit mesuré de la méthode -----------------------------------
# Obtenu sur vidéos synthétiques où la grandeur injectée vaut zéro : ce qui est
# mesuré alors est entièrement du bruit de méthode. Un marqueur dont le plancher
# dépasse l'écart-type de sa norme ne peut pas discriminer deux sujets — il est
# alors automatiquement écarté du score et signalé comme non résolu, plutôt
# qu'affiché comme s'il portait une information.
METHOD_NOISE_FLOOR: Dict[str, float] = {
    # --- Vrais planchers : grandeurs dont la valeur injectée était nulle ------
    # Couloir large (960 px), 40 s, caméra fixe, 2 tirages.
    "step_time_cv_pct": 3.4,        # norme 3.0 ± 1.6  → NON RÉSOLU à 30 i/s
    "stride_time_cv_pct": 2.1,      # norme 2.4 ± 1.4  → NON RÉSOLU
    "step_asymmetry_pct": 2.7,      # norme 2.5 ± 2.0  → NON RÉSOLU
    # --- Erreurs absolues moyennes : grandeurs à valeur injectée non nulle ----
    # Même critère de lecture : une erreur supérieure à la dispersion entre
    # individus rend la comparaison de deux sujets impossible.
    "gait_speed_m_s": 0.02,         # norme 1.25 ± 0.20 → résolu
    "cadence_spm": 1.0,             # norme 112 ± 9     → résolu
    "turn_mean_dur_s": 0.31,        # norme 1.10 ± 0.40 → résolu
    "hrv_sdnn_ms": 7.1,             # mesuré dans tests/validation.py
    "hrv_rmssd_ms": 13.4,
}

# Le rapport harmonique, le SPARC et l'oscillation du tronc n'ont pas de valeur
# nulle de référence : mesurés sur une marche parfaitement régulière, ils valent
# respectivement ~1,3, ~-8,9 et ~6 %. Ces valeurs décrivent la marche
# synthétique, pas un bruit de méthode — les inscrire ici serait un abus.


def is_resolvable(name: str) -> bool:
    """Le marqueur discrimine-t-il mieux que son propre bruit de mesure ?"""
    floor = METHOD_NOISE_FLOOR.get(name)
    norm = REFERENCE_NORMS.get(name)
    if floor is None or norm is None:
        return True
    return floor <= norm[1]


# --- Seuils de qualité -------------------------------------------------------
QUALITY = {
    "snr_db_good": 4.0,
    "snr_db_usable": 0.5,
    "hrv_min_beats": 40,
    "motion_reject_px": 12.0,   # mouvement médian/frame au-delà duquel on alerte
    "camera_motion_px": 0.8,    # déplacement du fond signalant une caméra mobile
    "body_detection_min": 0.55, # taux de silhouette en deçà duquel on alerte
    "gait_min_steps": 12,       # pas nécessaires à une variabilité fiable
    "gait_min_passes": 2,
}

DEFAULT = ProcessingConfig()


def config_dict() -> dict:
    return {
        "processing": asdict(DEFAULT),
        "bands": {
            "pulse": PULSE_BAND, "resp": RESP_BAND, "tremor": TREMOR_BAND,
            "sway": SWAY_BAND, "lf": LF_BAND, "hf": HF_BAND,
        },
        "quality": QUALITY,
    }


# --- Indices composites du mouvement -----------------------------------------
# IRD, SCF et CAX sont des rapports sans dimension entre grandeurs mesurées dans
# le même enregistrement. Aucune norme de population publiée n'existe pour eux :
# les valeurs ci-dessous sont des repères provisoires issus de la marche
# synthétique de référence, à remplacer par des normes de cohorte avant tout
# usage comparatif entre individus. Ils ne sont volontairement PAS versés dans
# les scores par domaine tant que cette calibration n'a pas eu lieu.
COMPOSITE_REFERENCE: Dict[str, Tuple[float, float, int]] = {
    "ird_reserve_dynamique": (1.60, 0.35, -1),   # pas perdus par demi-tour
    "scf_signature_foulee":  (0.228, 0.030, +1), # amplitude / fréquence
    "cax_coherence":         (0.86, 0.08, +1),   # verrouillage pas–tronc
}

# Plancher de reproductibilité mesuré sur trois enregistrements identiques
# (tests/validation_composites.py) : écart-type entre prises successives.
COMPOSITE_NOISE_FLOOR: Dict[str, float] = {
    "ird_reserve_dynamique": 0.008,
    "scf_signature_foulee": 0.0003,
    "cax_coherence": 0.025,
}
