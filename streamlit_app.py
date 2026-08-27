"""LongeVis — application web (Streamlit Community Cloud).

Les vidéos reçues ne sont jamais conservées : elles sont analysées puis
supprimées. Une vidéo de démarche est une donnée biométrique au sens du RGPD
(article 9) ; la stocker sur un serveur public engagerait la responsabilité de
l'exploitant du site.
"""

import os
import tempfile
import traceback

import numpy as np
import streamlit as st

from longevis import pipeline
from longevis.config import METHOD_NOISE_FLOOR, REFERENCE_NORMS
from longevis.report import LABELS, UNITS

STATUTS = {
    "valide": "✔ fiable",
    "recherche": "~ recherche",
    "non_resolu": "⚠ non résolu",
    "instable": "⚠ non fiable",
    "brut": "· technique",
}
INSTABLES = {"cax_coherence", "cax_n_segments", "icr_relance", "icr_t_freinage_s",
             "icr_t_relance_s", "icr_n_virages"}
RECHERCHE = {"ird_reserve_dynamique", "scf_signature_foulee"}
TECHNIQUE = {"silhouette_height_px", "px_per_m", "body_detection_rate",
             "camera_motion_px", "n_passes", "gait_speed_px_s",
             "step_asymmetry_null_pct", "step_asymmetry_net_pct", "step_asymmetry_p"}

GROUPES = [
    ("Vitesse et rythme", ["gait_speed_m_s", "gait_speed_px_s", "gait_speed_stature_s",
                           "cadence_spm", "step_time_s", "step_length_m",
                           "step_length_stature", "n_steps"]),
    ("Régularité", ["step_time_cv_pct", "stride_time_cv_pct", "step_asymmetry_pct",
                    "step_asymmetry_null_pct", "step_asymmetry_net_pct",
                    "step_asymmetry_p"]),
    ("Tenue et fluidité", ["harmonic_ratio", "gait_sparc", "gait_jerk_norm",
                           "com_bob_stature_pct"]),
    ("Demi-tours", ["n_turns", "n_passes", "turn_mean_dur_s", "turn_max_dur_s"]),
    ("Indices composites", ["ird_reserve_dynamique", "scf_signature_foulee",
                            "cax_coherence", "icr_relance", "icr_t_freinage_s",
                            "icr_t_relance_s", "icr_n_virages"]),
    ("Équilibre debout", ["sway_rms_ap_mm", "sway_rms_ml_mm", "sway_path_mm_s",
                          "sway_area_mm2", "sway_f95_hz"]),
    ("Transfert assis-debout", ["sts_count", "sts_mean_dur_s"]),
    ("Acquisition", ["silhouette_height_px", "px_per_m", "body_detection_rate",
                     "camera_motion_px"]),
]


def statut(cle):
    if cle in INSTABLES:
        return STATUTS["instable"]
    if cle in RECHERCHE:
        return STATUTS["recherche"]
    floor, norm = METHOD_NOISE_FLOOR.get(cle), REFERENCE_NORMS.get(cle)
    if floor is not None and norm is not None and floor > norm[1]:
        return STATUTS["non_resolu"]
    if cle in TECHNIQUE:
        return STATUTS["brut"]
    return STATUTS["valide"]


def tableau_complet(f):
    lignes = []
    for titre, cles in GROUPES:
        for k in cles:
            v = f.get(k)
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                continue
            norm = REFERENCE_NORMS.get(k)
            dec = 3 if abs(v) < 1 else (0 if abs(v) > 100 else 2)
            lignes.append({
                "Groupe": titre,
                "Mesure": LABELS.get(k, k),
                "Valeur": f"{v:.{dec}f}",
                "Unité": UNITS.get(k, ""),
                "Référence": f"{norm[0]:g} ± {norm[1]:g}" if norm else "—",
                "Statut": statut(k),
            })
    return lignes


st.set_page_config(page_title="LongeVis", page_icon="🚶", layout="wide")
st.title("LongeVis")
st.write("Filmez une personne qui marche, l'analyse renvoie ses indices de mouvement.")

with st.expander("Comment filmer", expanded=False):
    st.markdown(
        "- Téléphone **posé**, jamais tenu à la main\n"
        "- Corps entier dans l'image, de la tête aux pieds\n"
        "- Des **allers-retours** dans le champ, 30 secondes au moins\n"
        "- Fond dégagé, lumière stable")

col_g, col_d = st.columns([1, 2])

with col_g:
    fichier = st.file_uploader("Vidéo de marche",
                               type=["mp4", "mov", "avi", "mkv", "webm", "m4v"])
    taille = st.number_input("Taille du sujet (mètres)", value=1.72,
                             min_value=0.5, max_value=2.5, step=0.01,
                             help="Introduit environ 6 % d'erreur sur la vitesse.")
    echelle = st.number_input("Échelle (pixels par mètre) — optionnel", value=0.0,
                              min_value=0.0, step=1.0,
                              help="Plus fiable que la taille. Laissez à 0 pour "
                                   "utiliser la taille déclarée.")
    lancer = st.button("Analyser", type="primary", disabled=fichier is None)

with col_d:
    if lancer and fichier is not None:
        chemin = None
        try:
            suffixe = os.path.splitext(fichier.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffixe) as tmp:
                tmp.write(fichier.getbuffer())
                chemin = tmp.name

            with st.spinner("Analyse en cours — comptez une à trois minutes…"):
                res = pipeline.analyze(
                    chemin, mode="mouvement",
                    subject_height_m=None if echelle > 0 else taille,
                    px_per_m=echelle if echelle > 0 else None,
                    strict=False)

            f, meta = res["features"], res["meta"]
            g = lambda k: f.get(k, float("nan"))

            st.subheader("Résultats")
            c1, c2, c3 = st.columns(3)
            c1.metric("Vitesse de marche", f"{g('gait_speed_m_s'):.2f} m/s")
            c2.metric("Cadence", f"{g('cadence_spm'):.0f} pas/min")
            c3.metric("Longueur de pas", f"{g('step_length_m'):.2f} m")
            c1.metric("Pas analysés", f"{g('n_steps'):.0f}")
            c2.metric("Demi-tours", f"{g('n_turns'):.0f}")
            c3.metric("Durée analysée", f"{meta['duration_s']:.0f} s")

            alertes = []
            if meta.get("camera_motion_px", 0) > 0.5:
                alertes.append("La caméra bouge. Posez-la sur un support stable.")
            if meta.get("body_detection_rate", 1) < 0.6:
                alertes.append("Corps mal détecté. Reculez, dégagez le fond.")
            if meta.get("task") != "marche":
                alertes.append(f"Activité reconnue : « {meta.get('task')} ». "
                               "Pour la marche, filmez des allers-retours.")
            for a in alertes:
                st.warning(a)

            st.subheader("Indices composites")
            i1, i2 = st.columns(2)
            i1.metric("IRD — Réserve Dynamique", f"{g('ird_reserve_dynamique'):.2f}",
                      help="Pas dépensés par demi-tour.")
            i2.metric("SCF — Signature de Foulée", f"{g('scf_signature_foulee'):.3f}",
                      help="Haut = pas amples ; bas = pas hachés.")
            st.caption("Indices sans dimension, indépendants de la calibration. "
                       "Hypothèses de recherche non validées sur cohorte humaine : "
                       "à lire en comparant une personne à elle-même dans le temps.")

            with st.expander(f"Toutes les mesures"):
                st.dataframe(tableau_complet(f), use_container_width=True,
                             hide_index=True)
                st.caption("« Non résolu » : le bruit de la méthode dépasse l'écart "
                           "attendu entre deux personnes — la valeur est exacte mais "
                           "ne les distingue pas. « Non fiable » : indice n'ayant pas "
                           "passé sa validation, à ne pas interpréter.")

        except Exception:
            st.error("L'analyse a échoué. Vérifiez que le fichier est bien une vidéo.")
            st.code(traceback.format_exc(limit=2))
        finally:
            if chemin and os.path.exists(chemin):
                os.remove(chemin)          # aucune conservation

st.divider()
st.caption(
    "**Outil de recherche — pas un dispositif médical.** Aucun diagnostic, aucune "
    "prédiction d'espérance de vie. Les repères cités proviennent d'études de "
    "population : ils décrivent des moyennes dans de grands groupes, jamais la "
    "trajectoire d'une personne. Pour toute question de santé, consultez un médecin. "
    "**Vos vidéos ne sont pas conservées** : chaque fichier est supprimé dès "
    "l'analyse terminée.")
