"""LongeVis — interface web.

Déployable tel quel sur Hugging Face Spaces (offre gratuite).

Note de conception : les vidéos reçues ne sont jamais conservées. Elles sont
analysées puis supprimées, et seuls les indices numériques sont renvoyés. Une
vidéo de visage ou de démarche est une donnée biométrique au sens du RGPD
(article 9) : la conserver sur un serveur public engagerait la responsabilité
de l'exploitant du site.
"""

from __future__ import annotations
import os
import tempfile
import traceback

import gradio as gr

import numpy as np

from longevis import pipeline
from longevis.config import METHOD_NOISE_FLOOR, REFERENCE_NORMS
from longevis.report import LABELS, UNITS

# Statut affiché à côté de chaque mesure. Il ne s'agit pas d'un habillage :
# une mesure « non résolu » a un bruit de méthode supérieur à la dispersion
# entre individus, elle ne peut donc pas servir à comparer deux personnes.
STATUTS = {
    "valide": "✔ mesure fiable",
    "recherche": "~ indice de recherche",
    "non_resolu": "⚠ non résolu — bruit > dispersion entre sujets",
    "instable": "⚠ non fiable — ne pas interpréter",
    "brut": "· donnée technique",
}
INSTABLES = {"cax_coherence", "cax_n_segments", "icr_relance", "icr_t_freinage_s",
             "icr_t_relance_s", "icr_n_virages"}
RECHERCHE = {"ird_reserve_dynamique", "scf_signature_foulee"}
TECHNIQUE = {"silhouette_height_px", "px_per_m", "body_detection_rate",
             "camera_motion_px", "n_passes", "gait_speed_px_s",
             "step_asymmetry_null_pct", "step_asymmetry_net_pct", "step_asymmetry_p"}


def _statut(cle: str) -> str:
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


def _tableau_complet(f: dict) -> str:
    txt = "## Toutes les mesures\n\n"
    for titre, cles in GROUPES:
        lignes = []
        for k in cles:
            v = f.get(k)
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                continue
            norm = REFERENCE_NORMS.get(k)
            ref = f"{norm[0]:g} ± {norm[1]:g}" if norm else "—"
            dec = 3 if abs(v) < 1 else (0 if abs(v) > 100 else 2)
            lignes.append(f"| {LABELS.get(k, k)} | {v:.{dec}f} {UNITS.get(k, '')} "
                          f"| {ref} | {_statut(k)} |")
        if lignes:
            txt += f"### {titre}\n\n| Mesure | Valeur | Référence | Statut |\n"
            txt += "|---|---|---|---|\n" + "\n".join(lignes) + "\n\n"
    txt += ("*« Non résolu » signifie que le bruit propre de la méthode dépasse "
            "l'écart attendu entre deux personnes : la valeur est exacte, mais "
            "elle ne permet pas de les distinguer. « Non fiable » signifie que "
            "l'indice n'a pas passé sa validation et ne doit pas être "
            "interprété.*\n")
    return txt

DESCRIPTION = """
Filmez une personne qui marche, l'analyse renvoie ses indices de mouvement.

**Comment filmer** — téléphone **posé** (jamais tenu à la main), corps entier
dans l'image, des allers-retours pendant 30 secondes au moins, fond dégagé.
"""

AVERTISSEMENT = """
**Outil de recherche — pas un dispositif médical.** Aucun diagnostic, aucune
prédiction d'espérance de vie. Les repères cités proviennent d'études de
population : ils décrivent des moyennes dans de grands groupes, jamais la
trajectoire d'une personne. Pour toute question de santé, consultez un médecin.

**Vos vidéos ne sont pas conservées.** Chaque fichier est supprimé dès
l'analyse terminée.
"""


def _ligne(nom: str, valeur, unite: str = "", decimales: int = 2) -> str:
    if valeur is None or valeur != valeur:      # NaN
        return f"| {nom} | — | |\n"
    return f"| {nom} | {valeur:.{decimales}f} | {unite} |\n"


def analyser(video, taille_m, echelle_px_m):
    """Analyse une vidéo et renvoie (résumé markdown, indices composites)."""
    if not video:
        return "Envoyez d'abord une vidéo.", "", ""

    chemin = video if isinstance(video, str) else getattr(video, "name", None)
    if not chemin or not os.path.exists(chemin):
        return "Fichier introuvable.", "", ""

    try:
        echelle = float(echelle_px_m) if echelle_px_m and float(echelle_px_m) > 0 else None
        taille = float(taille_m) if taille_m and float(taille_m) > 0 else None
        res = pipeline.analyze(chemin, mode="mouvement",
                               subject_height_m=None if echelle else taille,
                               px_per_m=echelle, strict=False)
    except Exception:                                    # noqa: BLE001
        return ("L'analyse a échoué. Vérifiez que le fichier est bien une vidéo.\n\n"
                f"```\n{traceback.format_exc(limit=1)}\n```"), "", ""
    finally:
        try:
            os.remove(chemin)                            # aucune conservation
        except OSError:
            pass

    f, meta = res["features"], res["meta"]
    g = lambda k: f.get(k, float("nan"))

    alertes = []
    if meta.get("camera_motion_px", 0) > 0.5:
        alertes.append("La caméra bouge. Posez-la sur un support stable.")
    if meta.get("body_detection_rate", 1) < 0.6:
        alertes.append("Le corps est mal détecté. Reculez, dégagez le fond.")
    if meta.get("task") != "marche":
        alertes.append(f"Activité reconnue : « {meta.get('task')} ». "
                       "Pour la marche, filmez des allers-retours.")

    txt = f"## Résultats\n\n"
    txt += f"Durée analysée : {meta['duration_s']:.0f} s — "
    txt += f"corps détecté sur {meta.get('body_detection_rate', 0) * 100:.0f} % des images\n\n"
    txt += "| Mesure | Valeur | Unité |\n|---|---|---|\n"
    txt += _ligne("Vitesse de marche", g("gait_speed_m_s"), "m/s")
    txt += _ligne("Cadence", g("cadence_spm"), "pas/min", 0)
    txt += _ligne("Longueur de pas", g("step_length_m"), "m")
    txt += _ligne("Pas analysés", g("n_steps"), "", 0)
    txt += _ligne("Demi-tours", g("n_turns"), "", 0)
    if alertes:
        txt += "\n**Réserves sur cet enregistrement**\n\n"
        txt += "".join(f"- {a}\n" for a in alertes)

    ind = "## Indices composites\n\n"
    ind += "| Indice | Valeur | Lecture |\n|---|---|---|\n"
    ird, scf = g("ird_reserve_dynamique"), g("scf_signature_foulee")
    ind += (f"| **IRD** — Réserve Dynamique | {ird:.2f} | "
            "pas dépensés par demi-tour |\n" if ird == ird else "| **IRD** | — | |\n")
    ind += (f"| **SCF** — Signature de Foulée | {scf:.3f} | "
            "haut = pas amples ; bas = pas hachés |\n" if scf == scf else "| **SCF** | — | |\n")
    ind += ("\n*Indices sans dimension, indépendants de la calibration. "
            "Hypothèses de recherche non validées sur cohorte humaine : "
            "à lire en comparant une personne à elle-même dans le temps.*\n")
    return txt, ind, _tableau_complet(f)


with gr.Blocks(title="LongeVis — indices de mouvement") as demo:
    gr.Markdown("# LongeVis")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            video = gr.Video(label="Vidéo de marche", sources=["upload"])
            taille = gr.Number(label="Taille du sujet (mètres)", value=1.72,
                               info="Utilisée si aucune échelle n'est fournie. "
                                    "Introduit environ 6 % d'erreur sur la vitesse.")
            echelle = gr.Number(
                label="Échelle (pixels par mètre) — optionnel", value=None,
                info="Plus fiable que la taille. Mesurez un objet de longueur "
                     "connue dans l'axe de marche.")
            lancer = gr.Button("Analyser", variant="primary")
        with gr.Column():
            sortie = gr.Markdown()
            indices = gr.Markdown()
            with gr.Accordion("Toutes les mesures", open=False):
                complet = gr.Markdown()

    lancer.click(analyser, inputs=[video, taille, echelle],
                 outputs=[sortie, indices, complet])
    gr.Markdown("---")
    gr.Markdown(AVERTISSEMENT)


if __name__ == "__main__":
    demo.launch()
