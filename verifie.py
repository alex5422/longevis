"""Batterie de vérification — à lancer avant chaque dépôt sur GitHub.

    python verifie.py

Elle n'a besoin ni de vidéo ni de navigateur : les traces de silhouette sont
fabriquées, streamlit est simulé. Tout ce qui casserait la démonstration est
vérifié ici plutôt que découvert en réunion.
"""

from __future__ import annotations
import json
import os
import re
import sys
import types
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

OK, KO = 0, 0


def t(nom, fn):
    global OK, KO
    try:
        fn()
        print("  ✔ " + nom)
        OK += 1
    except Exception as e:                                   # noqa: BLE001
        print("  ✘ " + nom + " → " + str(e))
        KO += 1


# ─────────────────────────────────────────────────────────────────────────
#  Traces de synthèse : une marche vue de profil, sans déplacement imposé
# ─────────────────────────────────────────────────────────────────────────
from longevis.body import BodyTraces                          # noqa: E402
from longevis import gait, kinexa, hologramme, vue            # noqa: E402


def traces(cadence=100., amp_pas_m=0.55, cv=3., duree=12., fps=30.,
           h_px=240., taille=1.62, deplacement_px_s=0., bruit=0.004):
    n = int(duree * fps)
    tt = np.arange(n) / fps
    px_m = h_px / taille
    phase = 2 * np.pi * (cadence / 60.) * tt
    r = np.random.default_rng(7)
    if cv:
        phase = phase + np.cumsum(r.normal(0, cv / 100., n)) * 0.3
    spread = amp_pas_m * px_m * np.abs(np.sin(phase / 2)) + r.normal(0, bruit * px_m, n)
    cx = 200 + deplacement_px_s * tt
    cy = 0.5 * h_px + 2.5 * np.sin(2 * phase)
    hh = np.full(n, h_px) + 1.5 * np.sin(2 * phase)
    return BodyTraces(fps=fps, n_frames=n, duration_s=duree, centroid=np.c_[cx, cy],
                      bbox=np.zeros((n, 4)), height_px=hh, leg_spread=spread, trunk_y=cy,
                      area=np.full(n, 1e4), foot_y=cy + h_px / 2,
                      valid=np.ones(n, bool), detection_rate=0.95,
                      mode="silhouette", frame_size=(640, 360))


def mesures(**kw):
    return gait.analyze_motion(traces(**kw), task="auto", subject_height_m=1.62)["features"]


print("── la marche est reconnue dans tous les cadrages ──")


def _reconnue(nom, **kw):
    def f():
        r = gait.analyze_motion(traces(**kw), task="auto", subject_height_m=1.62)
        assert r["task"] == "marche", "tâche lue : " + r["task"]
        assert np.isfinite(r["features"].get("cadence_spm", np.nan)), "cadence absente"
    t(nom, f)


_reconnue("traversée franche du champ", deplacement_px_s=60)
_reconnue("deux mètres de recul", deplacement_px_s=12, duree=10)
_reconnue("marche sur place, de profil", deplacement_px_s=0)
_reconnue("pas courts et irréguliers", amp_pas_m=0.34, cadence=88, cv=11)


print("── la vitesse reste juste sans déplacement ──")


def _vitesse(nom, cadence, amp, tol=0.15):
    def f():
        v = mesures(cadence=cadence, amp_pas_m=amp).get("gait_speed_m_s", np.nan)
        vraie = amp * cadence / 60.
        assert np.isfinite(v), "vitesse absente"
        ecart = abs(v - vraie) / vraie
        assert ecart < tol, "écart de %.0f %% (%.2f contre %.2f)" % (100 * ecart, v, vraie)
    t(nom + " (%.2f m/s attendus)" % (amp * cadence / 60.), f)


_vitesse("marche lente", 88, 0.34)
_vitesse("marche ordinaire", 100, 0.52)
_vitesse("marche allante", 112, 0.66)


print("── les quatre biomarqueurs sortent toujours un chiffre ──")


def _quatre(nom, age, **kw):
    def f():
        b = kinexa.biomarqueurs(mesures(**kw), None, age)
        for cle, champ in [("bio_mobility", "score"), ("neuroplasticity", "score"),
                           ("kinetic_age", "age"), ("vitality_margin", "marge_pct")]:
            v = b[cle][champ]
            assert isinstance(v, float) and np.isfinite(v), cle + " vide"
            if champ == "score":
                assert 0 <= v <= 100, cle + " hors bornes : %.1f" % v
        assert 20 <= b["kinetic_age"]["age"] <= 100, "âge aberrant"
    t(nom, f)


_quatre("82 ans, sur place", 82, cadence=100, amp_pas_m=0.52, cv=4)
_quatre("82 ans, pas courts", 82, cadence=88, amp_pas_m=0.34, cv=11)
_quatre("70 ans, allante", 70, cadence=112, amp_pas_m=0.66, cv=2)
_quatre("45 ans", 45, cadence=115, amp_pas_m=0.74, cv=2)
_quatre("sans âge déclaré", None, cadence=100, amp_pas_m=0.52)


def _comparaison_age():
    f80 = mesures(cadence=96, amp_pas_m=0.46)
    sans = kinexa.biomarqueurs(f80, None, None)["bio_mobility"]["score"]
    avec = kinexa.biomarqueurs(f80, None, 82)["bio_mobility"]["score"]
    assert avec > sans + 5, "la classe d'âge ne change rien (%.0f contre %.0f)" % (avec, sans)


t("la comparaison à la classe d'âge relève le score d'un sujet âgé", _comparaison_age)


def _icope_absent():
    src = open(os.path.join("longevis", "kinexa.py"), encoding="utf-8").read()
    assert "def icope" in src, "la fonction ICOPE a disparu"
    app = open("streamlit_app.py", encoding="utf-8").read()
    assert "ICOPE" not in app, "l'ICOPE est revenu dans la façade Longévité"


t("l'ICOPE reste disponible mais hors de l'écran Longévité", _icope_absent)


print("── le rejeu incrusté ──")

import tempfile
FAUSSE = os.path.join(tempfile.gettempdir(), "_kinexa_faux.mp4")


def _rejeu_donnees():
    open(FAUSSE, "wb").write(os.urandom(40000))
    n = 360
    sig = {"body_cx": (200 + np.linspace(0, 300, n)).tolist(),
           "body_cy": (180 + 6 * np.sin(np.linspace(0, 30, n))).tolist(),
           "body_height": (220 + 4 * np.sin(np.linspace(0, 30, n))).tolist(),
           "body_spread": (20 + 18 * np.abs(np.sin(np.linspace(0, 20, n)))).tolist(),
           "body_fps": 30.}
    f = mesures()
    f["px_per_m"] = 140.
    b = kinexa.biomarqueurs(f, None, 72)
    html = hologramme.rejeu(FAUSSE, b, f, sig, {"fps": 30., "duration_s": 12.,
                                                "frame_size": (640, 360)})
    assert html and "kx-scene" in html, "lecteur non produit"
    d = json.loads(re.search(r"var D = (\{.*?\});", html, re.S).group(1))
    assert len(d["cartes"]) == 4, "%d cartes" % len(d["cartes"])
    assert len(d["serie"]) > 100, "série trop courte"
    assert d["serie"][-1]["n"] >= 4, "aucun pas compté"
    assert "autoplay" in html and "loop" in html, "le rejeu ne démarre pas seul"
    assert "kxbar" in html and "kxt" in html, "barre ou minutage absents"
    assert "kx-corps" in html and "kx-halo" in html, "incrustations du corps absentes"
    for attendu in ["kxplein", "requestFullscreen", "kx-grille", "kx-sweep",
                    "kx-osc", "kx-coin"]:
        assert attendu in html, attendu + " absent du rejeu"
    assert "classList.add('plein')" in html, "pas de repli si le plein écran est refusé"
    assert "kxmet" in html and "kx-mets" in html, "bouton ou panneau de métriques absent"
    assert "kxleg" in html and "kxlegp" in html, "légende absente"
    assert "vmax" in d, "échelle de vigueur absente"
    assert "249,115,22" in html, "la teinte de vigueur ne monte pas jusqu'à l'orange"
    assert "box.style.display = 'none'" in html, "le réticule reste affiché sans suivi"
    #  les chiffres doivent être grands, et grandir avec l'écran
    import re as _re
    for sel, mini in [(".kx-val{font-size:clamp(", 38), (".kx-live b{display:block;font-size:clamp(", 46)]:
        i = html.find(sel)
        assert i > 0, "taille de " + sel + " introuvable"
        m = _re.search(r"clamp\((\d+)px,([\d.]+)vw,(\d+)px\)", html[i:i + 160])
        assert m, "taille non lisible pour " + sel
        assert int(m.group(1)) >= mini, "%s trop petit : %spx" % (sel, m.group(1))
        assert int(m.group(3)) >= 80, "%s ne grandit pas assez : %spx" % (sel, m.group(3))
        assert float(m.group(2)) > 0, "%s ne suit pas la largeur" % sel
    assert html.count("text-shadow:0 0 1") >= 2, "les chiffres n'ont plus de halo"
    m2 = _re.search(r"\.kx-lab\{font-size:clamp\((\d+)px,", html)
    assert m2 and int(m2.group(1)) >= 12, "les intitulés des index sont restés petits"
    assert 'class="petit"' in html, "le compteur de pas n'a plus sa taille propre"
    assert "font-size:22px" not in html, "une taille fixe subsiste dans la balise"
    assert "ouvert" in html, "le panneau ne se dévoile pas"
    assert len(d["metriques"]) >= 8, "%d métriques seulement" % len(d["metriques"])
    noms = [m["nom"] for m in d["metriques"]]
    for attendu in ["Cadence", "Vitesse de marche"]:
        assert attendu in noms, attendu + " absent des métriques"
    assert any("Fluidité" in x for x in noms), "aucune mesure de fluidité"
    assert any("Amplitude" in x or "Longueur" in x for x in noms), "aucune amplitude"
    for m in d["metriques"]:
        assert isinstance(m["val"], (int, float)), "métrique non numérique : " + m["nom"]
    for p in d["serie"]:
        assert 0 <= p.get("x", 0) <= 1 and 0 <= p.get("y", 0) <= 1, "position hors image"
    assert d["serie"][0].get("h", 0) > 0, "hauteur de silhouette absente"
    xs = [p["x"] for p in d["serie"] if "x" in p]
    assert max(xs) - min(xs) > 0.1, "le repère ne suit pas le sujet"


def _rejeu_lourd():
    open(FAUSSE, "wb").write(os.urandom(30_000_000))
    assert hologramme.rejeu(FAUSSE, {}, {}, {}, {}) is None, "vidéo de 30 Mo acceptée"


def _hauteur():
    p = hologramme.hauteur_composant({"frame_size": (1080, 1920)})   # téléphone vertical
    l = hologramme.hauteur_composant({"frame_size": (1920, 1080)})   # paysage
    assert p > l + 200, "la vidéo verticale n'a pas plus de hauteur (%d / %d)" % (p, l)
    assert 300 < l < 900, "hauteur paysage inattendue : %d" % l


t("le lecteur porte les quatre cartes, la série et les pas", _rejeu_donnees)
t("une vidéo trop lourde est refusée proprement", _rejeu_lourd)
t("la hauteur s'adapte au format vertical", _hauteur)
try:
    os.remove(FAUSSE)
except OSError:
    pass


print("── le répertoire ──")


def _partition(**kw):
    age = kw.pop("age", 70)
    f = mesures(**kw)
    f["px_per_m"] = 140.
    b = kinexa.biomarqueurs(f, None, age)
    n = 360
    sig = {"body_cx": (200 + np.linspace(0, 300, n)).tolist(),
           "body_spread": (20 + 18 * np.abs(np.sin(np.linspace(0, 20, n)))).tolist(),
           "body_fps": 30.}
    serie = hologramme._serie(sig, 30., 12., (640, 360))
    return hologramme.musique(serie, f, b, 12.), f, b


def _repertoire_libre():
    """Cinq œuvres, cinq auteurs morts depuis plus d'un siècle et demi."""
    assert len(hologramme.REPERTOIRE) == 5, "le répertoire a changé de taille"
    auteurs = {p["auteur"] for p in hologramme.REPERTOIRE.values()}
    assert auteurs == {"J.-S. Bach", "Beethoven", "Pachelbel"}, auteurs
    for cle, p in hologramme.REPERTOIRE.items():
        chant, accomp, longueur = p["bati"]()
        assert len(chant) >= 8, cle + " : mélodie trop courte (%d notes)" % len(chant)
        assert accomp, cle + " : aucun accompagnement"
        assert longueur > 0, cle + " : longueur nulle"
        for (b, n, d) in chant:
            assert 0 <= b < longueur, cle + " : note hors de la pièce"
            assert 36 <= n <= 96, cle + " : note hors du clavier utile : %s" % n
            assert d > 0, cle + " : durée nulle"
        assert p["tempo"][0] < p["tempo"][1], cle + " : plage de tempo absurde"


def _oeuvres_fideles():
    """Quelques repères sur les œuvres, pour détecter une transcription abîmée."""
    chant, accomp, lg = hologramme._bach_prelude()
    assert len(chant) == 128, "le prélude de Bach doit faire 128 doubles-croches"
    assert chant[0][1] == 48 and chant[2][1] == 55, "le prélude ne commence pas par do–mi–sol"
    chant, accomp, lg = hologramme._bach_menuet()
    assert chant[0][1] == 74, "le menuet ne commence pas sur le ré"
    chant, accomp, lg = hologramme._beethoven_elise()
    debut = [n for (_, n, _) in chant[:8]]
    assert debut == [76, 75, 76, 75, 76, 71, 74, 72], "Élise : ouverture inexacte"
    chant, accomp, lg = hologramme._beethoven_joie()
    assert [n for (_, n, _) in chant[:4]] == [64, 64, 65, 67], "l'Hymne à la joie est faux"
    chant, accomp, lg = hologramme._pachelbel_canon()
    assert [a[1][0] for a in accomp] == [50, 45, 47, 42, 43, 50, 43, 45], \
        "la basse du Canon n'est pas la bonne"


def _caractere_choisit():
    cas = [({"move_active_pct": 12}, "repos"),
           ({"cadence_spm": 100, "move_active_pct": 70}, "marche"),
           ({"cadence_spm": 118, "move_active_pct": 80}, "elan"),
           ({"move_active_pct": 60, "move_amplitude_stature": .6,
             "move_rate_cpm": 22}, "souffle"),
           ({"move_active_pct": 65, "move_amplitude_stature": .3,
             "move_rate_cpm": 52}, "appui")]
    for f, attendu in cas:
        lu = hologramme.caractere(f)
        assert lu == attendu, "%s attendait %s, a lu %s" % (f, attendu, lu)
    for c, cle in hologramme.POUR_PIECE.items():
        assert cle in hologramme.REPERTOIRE, c + " renvoie à une pièce inexistante"


def _partition_complete():
    M, f, b = _partition()
    assert M, "aucune partition"
    for cle in ("titre", "auteur", "tempo", "accords", "notes", "detune", "liste"):
        assert cle in M, cle + " absent de la partition"
    assert len(M["liste"]) == 5, "les cinq pièces ne sont pas toutes prêtes"
    assert M["liste"][0]["titre"] == M["titre"], "la pièce de tête n'est pas la bonne"
    for a in M["liste"]:
        assert a["notes"], a["titre"] + " : aucune note"
        assert a["accords"], a["titre"] + " : aucun accompagnement"
        assert 40 <= a["tempo"] <= 150, a["titre"] + " : tempo %s" % a["tempo"]


def _partition_tient_dans_la_video():
    M, f, b = _partition()
    for a in M["liste"]:
        assert a["notes"][0]["t"] == 0.0, a["titre"] + " ne commence pas avec la vidéo"
        assert max(n["t"] for n in a["notes"]) < 12.0, a["titre"] + " déborde de la vidéo"
        assert max(n["t"] for n in a["notes"]) > 7.0, a["titre"] + " s'arrête trop tôt"
        for n in a["notes"]:
            assert 30 <= n["p"] <= 100, a["titre"] + " : note hors registre"
            assert 0.55 <= n["g"] <= 1.0, a["titre"] + " : nuance hors bornes"


def _nuances_suivent_le_corps():
    M, f, b = _partition()
    g = [n["g"] for n in M["liste"][0]["notes"]]
    assert max(g) - min(g) > 0.05, "les nuances ne suivent pas le mouvement"


def _juste_quand_regulier():
    reg, _, _ = _partition(cv=1.5)
    heurte, _, _ = _partition(cv=22)
    assert reg["detune"] < heurte["detune"], \
        "un geste heurté devrait sonner moins juste (%.1f contre %.1f)" % (
            reg["detune"], heurte["detune"])
    assert reg["detune"] <= 6 and heurte["detune"] <= 32, "désaccord hors bornes"


def _tempo_suit_la_cadence():
    lent, _, _ = _partition(cadence=64, amp_pas_m=0.5)
    vif, _, _ = _partition(cadence=112, amp_pas_m=0.5)
    assert vif["tempo"] >= lent["tempo"], "le pouls ne suit pas la cadence (%.0f / %.0f)" % (
        lent["tempo"], vif["tempo"])


def _hauteur_suit_la_mobilite():
    M, f, b = _partition()
    assert -4 <= M["transposition"] <= 4, "transposition excessive : %s" % M["transposition"]


def _musique_dans_le_lecteur():
    open(FAUSSE, "wb").write(os.urandom(30000))
    f = mesures()
    f["px_per_m"] = 140.
    n = 360
    sig = {"body_cx": (200 + np.linspace(0, 300, n)).tolist(),
           "body_spread": (20 + 18 * np.abs(np.sin(np.linspace(0, 20, n)))).tolist(),
           "body_fps": 30.}
    html = hologramme.rejeu(FAUSSE, kinexa.biomarqueurs(f, None, 70), f, sig,
                            {"fps": 30., "duration_s": 12., "frame_size": (640, 360)})
    d = json.loads(re.search(r"var D = (\{.*?\});", html, re.S).group(1))
    assert len(d.get("musique", {}).get("liste", [])) == 5, "le répertoire n'est pas embarqué"
    for attendu in ["kxson", "kxair", "AudioContext", "createDelay", "rebouclé"]:
        assert attendu in html, attendu + " absent du lecteur"
    assert "joue = false" in html, "le son n'est pas silencieux au départ"
    assert "addEventListener('click'" in html, "le son ne démarre pas sur un geste"
    for auteur in ("Bach", "Beethoven", "Pachelbel"):
        assert auteur in html, auteur + " absent du lecteur"
    try:
        os.remove(FAUSSE)
    except OSError:
        pass


t("cinq œuvres, toutes du domaine public", _repertoire_libre)
t("les transcriptions sont fidèles aux œuvres", _oeuvres_fideles)
t("le caractère du mouvement choisit la pièce", _caractere_choisit)
t("la partition est complète et les cinq pièces prêtes", _partition_complete)
t("chaque pièce couvre toute la vidéo sans déborder", _partition_tient_dans_la_video)
t("les nuances suivent la vigueur du corps", _nuances_suivent_le_corps)
t("un geste régulier sonne plus juste qu'un geste heurté", _juste_quand_regulier)
t("le pouls de la musique suit la cadence", _tempo_suit_la_cadence)
t("la hauteur suit la mobilité sans dénaturer l'œuvre", _hauteur_suit_la_mobilite)
t("le lecteur embarque le répertoire, muet par défaut", _musique_dans_le_lecteur)


def _son_dans_le_navigateur():
    """Le lecteur est exécuté par Node, avec un faux DOM et une fausse carte son."""
    import shutil
    import subprocess
    node = shutil.which("node") or shutil.which("nodejs")
    if node is None:
        raise AssertionError("Node absent : contrôle du lecteur non exécuté")
    open(FAUSSE, "wb").write(os.urandom(30000))
    f = mesures()
    f["px_per_m"] = 140.
    n = 360
    sig = {"body_cx": (200 + np.linspace(0, 300, n)).tolist(),
           "body_cy": (180 + 6 * np.sin(np.linspace(0, 30, n))).tolist(),
           "body_height": (220 + 4 * np.sin(np.linspace(0, 30, n))).tolist(),
           "body_spread": (20 + 18 * np.abs(np.sin(np.linspace(0, 20, n)))).tolist(),
           "body_fps": 30.}
    html = hologramme.rejeu(FAUSSE, kinexa.biomarqueurs(f, None, 70), f, sig,
                            {"fps": 30., "duration_s": 12., "frame_size": (640, 360)})
    chemin = os.path.join(tempfile.gettempdir(), "_kinexa_lecteur.html")
    with open(chemin, "w", encoding="utf-8") as fh:
        fh.write(html)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verifie_son.js")
    r = subprocess.run([node, script, chemin], capture_output=True, text=True)
    for ligne in r.stdout.splitlines():
        if ligne.strip().startswith(("\u2714", "\u2718")):
            print("   " + ligne.strip())
    if r.returncode:
        raise AssertionError("le lecteur ne se comporte pas comme prévu")
    for tmp in (FAUSSE, chemin):
        try:
            os.remove(tmp)
        except OSError:
            pass


t("le lecteur joue vraiment, boutons compris", _son_dans_le_navigateur)


print("── les valeurs aberrantes sont écartées ──")


def _plausible():
    assert kinexa.plausible("cadence_spm", 104), "cadence normale refusée"
    assert not kinexa.plausible("cadence_spm", 420), "cadence de 420 acceptée"
    assert not kinexa.plausible("harmonic_ratio", 1944.9), "rapport harmonique absurde accepté"
    assert not kinexa.plausible("gait_speed_m_s", 9.4), "vitesse de 9,4 m/s acceptée"
    assert kinexa.plausible("gait_speed_m_s", 1.2), "vitesse normale refusée"
    assert not kinexa.plausible("move_rate_cpm", float("nan")), "valeur absente acceptée"
    assert kinexa.plausible("clé_inconnue", 12.0), "une mesure sans plage doit passer"


def _aberrante_ecartee_du_score():
    f = mesures()
    f["harmonic_ratio"] = 1944.9                 # artefact
    d = kinexa.biomarqueurs(f, None, 70)["neuroplasticity"]["detail"]
    assert "harmonic_ratio" not in d, "l'aberration entre dans le score"


def _aberrante_ecartee_du_panneau():
    open(FAUSSE, "wb").write(os.urandom(30000))
    f = mesures()
    f["harmonic_ratio"] = 1944.9
    f["px_per_m"] = 140.
    b = kinexa.biomarqueurs(f, None, 70)
    html = hologramme.rejeu(FAUSSE, b, f, {"body_cx": list(range(50)), "body_fps": 30.},
                            {"fps": 30., "duration_s": 5., "frame_size": (640, 360)})
    d = json.loads(re.search(r"var D = (\{.*?\});", html, re.S).group(1))
    for m in d["metriques"]:
        assert m["val"] < 500, "valeur aberrante affichée : %s %s" % (m["nom"], m["val"])
    try:
        os.remove(FAUSSE)
    except OSError:
        pass


t("les plages de plausibilité filtrent l'absurde", _plausible)
t("une mesure aberrante n'entre pas dans les scores", _aberrante_ecartee_du_score)
t("une mesure aberrante n'est pas affichée", _aberrante_ecartee_du_panneau)


print("── tous les mouvements, pas seulement la marche ──")


def geste(nom_axe="vertical", freq=0.6, amplitude=0.35, cv=6., duree=14., fps=30.,
          h_px=240., derive=0.):
    """Un mouvement quelconque : gymnastique, danse, exercice assis.

    `cv` fait varier la durée de chaque cycle, comme un geste irrégulier.
    """
    n = int(duree * fps)
    tt = np.arange(n) / fps
    r = np.random.default_rng(11)
    periodes, acc = [], 0.
    while acc < duree + 2:
        p = (1.0 / freq) * (1 + r.normal(0, cv / 100.))
        periodes.append(max(0.15, p))
        acc += periodes[-1]
    bornes = np.cumsum([0.] + periodes)
    phase = np.interp(tt, bornes, 2 * np.pi * np.arange(len(bornes)))
    onde = amplitude * h_px * np.sin(phase)
    cx = np.full(n, 300.) + (onde if nom_axe == "horizontal" else 0) + derive * tt
    cy = np.full(n, 0.5 * h_px) + (onde if nom_axe == "vertical" else 0)
    spread = np.full(n, 0.10 * h_px) + (onde * 0.3 if nom_axe == "membres" else 0)
    hh = np.full(n, h_px) + (onde * 0.15 if nom_axe == "vertical" else 0)
    return BodyTraces(fps=fps, n_frames=n, duration_s=duree, centroid=np.c_[cx, cy],
                      bbox=np.zeros((n, 4)), height_px=hh, leg_spread=spread, trunk_y=cy,
                      area=np.full(n, 1e4), foot_y=cy + h_px / 2,
                      valid=np.ones(n, bool), detection_rate=0.95,
                      mode="silhouette", frame_size=(640, 360))


def _mouvement(nom, **kw):
    def f():
        r = gait.analyze_motion(geste(**kw), task="auto", subject_height_m=1.62)
        assert r["task"] in ("mouvement", "marche", "leve"), "classé « %s »" % r["task"]
        fe = r["features"]
        for cle in ("move_amplitude_stature", "move_peak_speed_stature", "move_sparc"):
            assert np.isfinite(fe.get(cle, np.nan)), cle + " absent"
    t(nom, f)


_mouvement("balancement du tronc (tai-chi)", nom_axe="horizontal", freq=0.4, amplitude=0.25)
_mouvement("flexions verticales (squats)", nom_axe="vertical", freq=0.7, amplitude=0.30)
_mouvement("gestes des bras assis", nom_axe="membres", freq=1.1, amplitude=0.22)
_mouvement("mouvement lent et ample", nom_axe="vertical", freq=0.25, amplitude=0.45)


def _cadence_generique():
    fe = gait.analyze_motion(geste(freq=0.8, amplitude=0.3), task="auto",
                             subject_height_m=1.62)["features"]
    lu = fe.get("move_rate_hz", np.nan)
    assert np.isfinite(lu), "rythme non lu"
    assert abs(lu - 0.8) < 0.2, "rythme lu %.2f Hz au lieu de 0,80" % lu


def _quatre_libre():
    fe = gait.analyze_motion(geste(freq=0.8, amplitude=0.3), task="auto",
                             subject_height_m=1.62)["features"]
    b = kinexa.biomarqueurs(fe, None, 68)
    for cle, champ in [("bio_mobility", "score"), ("neuroplasticity", "score"),
                       ("kinetic_age", "age"), ("vitality_margin", "marge_pct")]:
        v = b[cle][champ]
        assert isinstance(v, float) and np.isfinite(v), cle + " vide sur un geste libre"
    assert b["bio_mobility"]["libre"], "le repli générique n'a pas servi"
    assert b["kinetic_age"]["approx"], "l'âge devrait être marqué approché"
    assert b["kinetic_age"]["marge"] >= 12, "marge non élargie sur lecture indirecte"


def _regularite_lue():
    net = gait.analyze_motion(geste(freq=0.8, cv=2.), task="auto",
                              subject_height_m=1.62)["features"]
    heurte = gait.analyze_motion(geste(freq=0.8, cv=35.), task="auto",
                                 subject_height_m=1.62)["features"]
    a = kinexa.biomarqueurs(net, None, 68)["neuroplasticity"]["score"]
    b = kinexa.biomarqueurs(heurte, None, 68)["neuroplasticity"]["score"]
    assert a > b + 5, "geste net %.0f contre geste heurté %.0f" % (a, b)


t("le rythme d'un geste quelconque est mesuré", _cadence_generique)
t("les quatre lectures sortent aussi sur un geste sans marche", _quatre_libre)
t("un geste régulier est mieux noté qu'un geste heurté", _regularite_lue)


print("── les figures de vitalité ──")


def _courbe():
    svg = vue.courbe_age(kinexa.COURBE_AGE, 0.92, 88., 75.)
    assert svg.startswith("<svg") and svg.endswith("</svg>"), "SVG mal formé"
    assert svg.count("<polyline") >= 1 and "polygon" in svg, "courbe ou bande absente"
    assert "0.92 m/s" in svg, "le sujet n'est pas placé"
    assert "âge locomoteur 88" in svg, "l'âge lu n'est pas reporté"
    vide = vue.courbe_age(kinexa.COURBE_AGE, float("nan"), None, None)
    assert "m/s" in vide and "circle" not in vide, "point tracé sans mesure"


def _radar():
    axes = vue.axes_vitalite(mesures(), kinexa.biomarqueurs(mesures(), None, 70))
    assert len(axes) == 5, "%d axes" % len(axes)
    svg = vue.radar(axes)
    assert svg.startswith("<svg"), "radar non produit"
    for nom in axes:
        assert nom in svg, nom + " absent du radar"
    assert vue.radar({"A": 50, "B": float("nan")}) == "", "radar tracé avec deux axes"


def _frise():
    svg = vue.frise([(0, 90), (150, 260)], [(90, 150)], 300, 30.)
    assert svg.startswith("<svg"), "frise non produite"
    assert svg.count("<rect") >= 4, "segments manquants"
    assert "10 s analysées" in svg, "durée absente"
    assert vue.frise([], [], 0, 30.) == "", "frise tracée sans images"


t("la courbe d'âge place le sujet et son âge locomoteur", _courbe)
t("le radar porte cinq axes et refuse d'en tracer deux", _radar)
t("la frise montre trajets et demi-tours à l'échelle du temps", _frise)


print("── la page se construit ──")


def _page():
    J = []

    class Faux:
        def __enter__(s):
            return s

        def __exit__(s, *a):
            return False

        def __getattr__(s, n):
            return lambda *a, **k: Faux()

    st = types.ModuleType("streamlit")
    for n in ["set_page_config", "markdown", "title", "write", "divider", "caption",
              "warning", "error", "code", "dataframe", "line_chart", "video",
              "subheader", "metric", "image", "html", "set_option"]:
        setattr(st, n, (lambda nom: (lambda *a, **k: J.append(nom)))(n))
    st.columns = lambda *a, **k: [Faux(), Faux(), Faux()]
    st.expander = lambda *a, **k: Faux()
    st.spinner = lambda *a, **k: Faux()
    st.sidebar = Faux()
    st.container = lambda *a, **k: Faux()
    st.tabs = lambda x: [Faux() for _ in x]
    st.file_uploader = lambda *a, **k: None
    st.number_input = lambda *a, **k: k.get("value", 0)
    st.select_slider = lambda *a, **k: k.get("value", None)
    st.download_button = lambda *a, **k: J.append("download")
    st.button = lambda *a, **k: False
    comp = types.ModuleType("streamlit.components.v1")
    comp.html = lambda *a, **k: J.append("components.html")
    st.components = types.ModuleType("streamlit.components")
    st.components.v1 = comp
    sys.modules["streamlit"] = st
    sys.modules["streamlit.components"] = st.components
    sys.modules["streamlit.components.v1"] = comp
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", "streamlit_app.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert J, "la page n'a rien produit"


def _contenus():
    s = open("streamlit_app.py", encoding="utf-8").read()
    assert "Mouvement reconnu" not in s, "la démo commente encore ce qu'elle reconnaît"
    for attendu in ["Bio-Mobility Score", "Neuroplasticity Index", "Kinetic Ageing Profile",
                    "Vitality Margin", "Âge du sujet", "Rejeu incrusté",
                    "Profil de vitalité", "Taille du rejeu", "Détail des scores",
                    "mesures (CSV)", "tout (JSON)",
                    "Longevity Institute · Metrology of Vitality"]:
        assert attendu in s, "« %s » absent de la page" % attendu
    assert "except ImportError" in s, "les imports fragiles ne sont pas protégés"


t("la page s'exécute sans navigateur", _page)
t("elle contient les quatre lectures, l'âge et le rejeu", _contenus)

print()
print(("✔ %d contrôles passés, 0 échec" % OK) if not KO else ("✘ %d échec(s)" % KO))
sys.exit(1 if KO else 0)
