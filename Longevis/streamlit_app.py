"""LongeVis — interface.

Direction visuelle : lumière et matière, pas de fil de fer. La démarche est
rendue comme un ruban continu dont la hauteur suit le déplacement et
l'épaisseur le rythme des jambes, posé sur une nappe de lumière diffuse. Les
surfaces sont du verre sombre, les chiffres très grands et serrés.

Le ruban n'est pas une décoration : son axe vertical est la position réelle du
sujet dans le champ, son épaisseur l'écartement mesuré des jambes. Deux
signaux, une seule forme — on lit la donnée, pas une illustration.

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

from longevis import body, pipeline
from longevis.config import METHOD_NOISE_FLOOR, REFERENCE_NORMS
from longevis.report import LABELS, UNITS

st.set_page_config(page_title="LongeVis", page_icon="◗", layout="wide",
                   initial_sidebar_state="expanded")

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@300;400;500;600&display=swap');
:root{
--bg:#0A0A0F; --bg-2:#101018;
--surface:rgba(255,255,255,.045); --surface-2:rgba(255,255,255,.07);
--edge:rgba(255,255,255,.09); --edge-2:rgba(255,255,255,.16);
--text:#F5F5F7; --dim:#9A9AA8; --faint:#6C6C7A;
--a1:#5B8CFF; --a2:#14D6C4; --a3:#C86BFF;
--ok:#3DDC97; --warn:#FFB84D; --bad:#FF6B6B;
--f:'Inter Tight',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}

.stApp{background:
 radial-gradient(900px 520px at 12% -8%, rgba(91,140,255,.16), transparent 62%),
 radial-gradient(760px 460px at 88% 4%, rgba(200,107,255,.13), transparent 60%),
 radial-gradient(680px 420px at 60% 108%, rgba(20,214,196,.10), transparent 58%),
 var(--bg); color:var(--text);}
html,body,[class*="css"]{font-family:var(--f);color:var(--text);
 -webkit-font-smoothing:antialiased}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:3rem;max-width:1180px}

/* ---------- En-tête ---------- */
.iv-tag{display:inline-flex;align-items:center;gap:9px;font-size:12px;
 letter-spacing:.01em;color:var(--dim);background:var(--surface);
 border:1px solid var(--edge);border-radius:100px;padding:6px 14px;margin:0 0 22px}
.iv-tag i{width:6px;height:6px;border-radius:50%;background:var(--a2);
 box-shadow:0 0 10px var(--a2);font-style:normal}
.iv-title{font-size:clamp(40px,6.4vw,74px);font-weight:600;letter-spacing:-.045em;
 line-height:1.02;margin:0;
 background:linear-gradient(120deg,#FFFFFF 18%,#C9D4FF 52%,#9FE9E0 88%);
 -webkit-background-clip:text;background-clip:text;color:transparent}
.iv-lede{color:var(--dim);font-size:17px;font-weight:300;letter-spacing:-.01em;
 margin:16px 0 0;max-width:52ch;line-height:1.55}

/* ---------- Scène ---------- */
.iv-stage{position:relative;margin:34px 0 0;border:1px solid var(--edge);
 border-radius:20px;overflow:hidden;background:
 linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.015));
 box-shadow:0 24px 70px rgba(0,0,0,.55),
  inset 0 1px 0 rgba(255,255,255,.10)}
.iv-stage svg{display:block;width:100%;height:auto}

.ribbon-line{fill:none;stroke:url(#grad);stroke-width:2.4;stroke-linecap:round;
 stroke-linejoin:round}
.ribbon-body{stroke:none}
.turnband{fill:url(#turngrad)}
.stepdot{fill:#FFFFFF;opacity:.85}
.lbl{font-size:11px;fill:var(--faint);letter-spacing:.01em;
 font-family:var(--f);font-weight:400}
.lbl-t{font-size:10.5px;fill:var(--faint);text-anchor:middle;font-family:var(--f)}

@media (prefers-reduced-motion:no-preference){
 .iv-reveal{clip-path:inset(0 100% 0 0);animation:wipe 1.5s cubic-bezier(.22,1,.36,1) forwards}
 @keyframes wipe{to{clip-path:inset(0 0 0 0)}}
 .iv-breathe{animation:breathe 5s ease-in-out infinite}
 @keyframes breathe{0%,100%{opacity:.55}50%{opacity:1}}
}

.iv-key{display:flex;flex-wrap:wrap;gap:22px;margin:16px 4px 0;
 font-size:12.5px;color:var(--faint)}
.iv-key i{display:inline-block;width:22px;height:3px;border-radius:2px;
 margin-right:8px;vertical-align:3px;font-style:normal}
.k1{background:linear-gradient(90deg,var(--a2),var(--a1))}
.k2{background:rgba(255,255,255,.8)}
.k3{background:rgba(200,107,255,.45)}

/* ---------- Cartes ---------- */
.iv-grid{display:grid;gap:14px;
 grid-template-columns:repeat(auto-fit,minmax(190px,1fr));margin:4px 0}
.iv-card{position:relative;background:var(--surface);border:1px solid var(--edge);
 border-radius:16px;padding:22px 22px 20px;overflow:hidden;
 box-shadow:inset 0 1px 0 rgba(255,255,255,.07)}
.iv-card--hero{background:
 linear-gradient(150deg,rgba(91,140,255,.16),rgba(20,214,196,.07) 60%,transparent),
 var(--surface);border-color:var(--edge-2)}
.iv-lab{font-size:12px;font-weight:400;color:var(--dim);margin:0 0 14px;
 letter-spacing:-.005em}
.iv-num{font-size:46px;font-weight:500;letter-spacing:-.05em;line-height:.94;
 margin:0;font-variant-numeric:tabular-nums;color:#FFF}
.iv-num small{font-size:15px;font-weight:400;color:var(--dim);margin-left:7px;
 letter-spacing:-.01em}
.iv-cap{font-size:12px;color:var(--faint);margin:12px 0 0;font-weight:300}

/* ---------- Sections ---------- */
.iv-h{font-size:13px;font-weight:500;color:var(--dim);letter-spacing:-.005em;
 margin:46px 0 16px;display:flex;align-items:center;gap:14px}
.iv-h::after{content:"";flex:1;height:1px;background:var(--edge)}

/* ---------- Tableau ---------- */
.iv-tbl{width:100%;border-collapse:collapse;font-size:14px;font-weight:300}
.iv-tbl th{font-size:11.5px;font-weight:400;color:var(--faint);text-align:left;
 padding:10px 14px 10px 0;border-bottom:1px solid var(--edge)}
.iv-tbl td{padding:11px 14px 11px 0;border-bottom:1px solid rgba(255,255,255,.05)}
.iv-tbl tr:hover td{background:rgba(255,255,255,.028)}
.iv-v{font-variant-numeric:tabular-nums;white-space:nowrap;color:#FFF;
 font-weight:400}
.iv-r{font-size:12.5px;color:var(--faint);white-space:nowrap}
.iv-g{font-size:11.5px;color:var(--dim);padding-top:26px;font-weight:500;
 letter-spacing:-.005em}

/* ---------- Puces ---------- */
.tag{font-size:11px;font-weight:400;padding:3px 10px;border-radius:100px;
 white-space:nowrap;border:1px solid transparent}
.t-ok{color:var(--ok);background:rgba(61,220,151,.11);border-color:rgba(61,220,151,.24)}
.t-re{color:var(--a3);background:rgba(200,107,255,.11);border-color:rgba(200,107,255,.24)}
.t-nr{color:var(--warn);background:rgba(255,184,77,.11);border-color:rgba(255,184,77,.24)}
.t-ko{color:var(--bad);background:rgba(255,107,107,.11);border-color:rgba(255,107,107,.24)}
.t-te{color:var(--faint);background:rgba(255,255,255,.05);border-color:var(--edge)}

/* ---------- Messages ---------- */
.iv-msg{background:var(--surface);border:1px solid var(--edge);border-radius:14px;
 padding:18px 20px;font-size:14px;color:var(--dim);margin:16px 0;font-weight:300;
 line-height:1.6}
.iv-msg--warn{border-color:rgba(255,184,77,.3);background:rgba(255,184,77,.06)}
.iv-msg--stop{border-color:rgba(255,107,107,.32);background:rgba(255,107,107,.06);
 color:var(--text)}
.iv-msg b{color:var(--text);font-weight:500}
.iv-msg ol{margin:12px 0 0;padding-left:20px}
.iv-msg li{margin-bottom:7px}

/* ---------- Latéral ---------- */
section[data-testid="stSidebar"]{background:rgba(12,12,18,.9);
 border-right:1px solid var(--edge)}
section[data-testid="stSidebar"] *{color:var(--text)}
.stButton>button{font-family:var(--f);font-size:14px;font-weight:500;
 letter-spacing:-.01em;border-radius:12px;border:1px solid rgba(255,255,255,.14);
 background:linear-gradient(135deg,rgba(91,140,255,.9),rgba(20,214,196,.75));
 color:#08080C;padding:12px 20px}
.stButton>button:hover{filter:brightness(1.12);color:#08080C;
 border-color:rgba(255,255,255,.25)}
.stButton>button:disabled{background:var(--surface);color:var(--faint);
 border-color:var(--edge)}
:focus-visible{outline:2px solid var(--a1)!important;outline-offset:3px}
.iv-foot{font-size:12px;color:var(--faint);line-height:1.8;margin-top:26px;
 font-weight:300}
.iv-foot b{color:var(--dim);font-weight:500}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

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


def tag(cle):
    if cle in INSTABLES:
        return '<span class="tag t-ko">non fiable</span>'
    if cle in RECHERCHE:
        return '<span class="tag t-re">recherche</span>'
    floor, norm = METHOD_NOISE_FLOOR.get(cle), REFERENCE_NORMS.get(cle)
    if floor is not None and norm is not None and floor > norm[1]:
        return '<span class="tag t-nr">non résolu</span>'
    if cle in TECHNIQUE:
        return '<span class="tag t-te">technique</span>'
    return '<span class="tag t-ok">fiable</span>'


def carte(label, valeur, unite="", note="", hero=False, dec=2):
    aff = "—" if (valeur is None or (isinstance(valeur, float)
                                     and not np.isfinite(valeur))) else f"{valeur:.{dec}f}"
    n = f'<p class="iv-cap">{note}</p>' if note else ""
    return (f'<div class="iv-card{" iv-card--hero" if hero else ""}">'
            f'<p class="iv-lab">{label}</p>'
            f'<p class="iv-num">{aff}<small>{unite}</small></p>{n}</div>')


LEGENDE = ('<div class="iv-key">'
           '<span><i class="k1"></i>déplacement et rythme du sujet</span>'
           '<span><i class="k2"></i>pas détectés</span>'
           '<span><i class="k3"></i>demi-tours</span></div>')

W, H = 1000, 340


def _defs():
    return '''<defs>
 <linearGradient id="grad" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="#14D6C4"/><stop offset=".45" stop-color="#5B8CFF"/>
  <stop offset="1" stop-color="#C86BFF"/></linearGradient>
 <linearGradient id="fill" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="#14D6C4" stop-opacity=".55"/>
  <stop offset=".45" stop-color="#5B8CFF" stop-opacity=".5"/>
  <stop offset="1" stop-color="#C86BFF" stop-opacity=".45"/></linearGradient>
 <linearGradient id="turngrad" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0" stop-color="#C86BFF" stop-opacity="0"/>
  <stop offset=".5" stop-color="#C86BFF" stop-opacity=".22"/>
  <stop offset="1" stop-color="#C86BFF" stop-opacity="0"/></linearGradient>
 <radialGradient id="bloom1"><stop offset="0" stop-color="#5B8CFF" stop-opacity=".5"/>
  <stop offset="1" stop-color="#5B8CFF" stop-opacity="0"/></radialGradient>
 <radialGradient id="bloom2"><stop offset="0" stop-color="#14D6C4" stop-opacity=".4"/>
  <stop offset="1" stop-color="#14D6C4" stop-opacity="0"/></radialGradient>
 <filter id="soft" x="-30%" y="-60%" width="160%" height="220%">
  <feGaussianBlur stdDeviation="16"/></filter>
</defs>'''


def _nappe():
    return (f'<rect width="{W}" height="{H}" fill="#0B0B12"/>'
            f'<ellipse class="iv-breathe" cx="230" cy="{H * .72:.0f}" rx="330" ry="150" '
            f'fill="url(#bloom1)"/>'
            f'<ellipse class="iv-breathe" cx="790" cy="{H * .34:.0f}" rx="300" ry="140" '
            f'fill="url(#bloom2)" style="animation-delay:-2.2s"/>')


def scene_repos():
    y = H * 0.58
    return f'''<div class="iv-stage"><svg viewBox="0 0 {W} {H}" role="img"
 aria-label="Scène en attente d'une séquence">{_defs()}{_nappe()}
 <path d="M 60,{y:.0f} L {W - 60},{y:.0f}" stroke="url(#grad)" stroke-width="2"
   opacity=".30" stroke-linecap="round"/>
 <path class="iv-breathe" d="M 60,{y:.0f} L {W - 60},{y:.0f}" stroke="url(#grad)"
   stroke-width="10" opacity=".18" filter="url(#soft)"/>
 <text class="lbl-t" x="{W / 2:.0f}" y="{y + 40:.0f}">
   Déposez une séquence pour lancer l'analyse</text>
</svg></div>'''


def scene_analyse(cx, spread, fps, passes, turns):
    """Ruban : hauteur = déplacement du sujet, épaisseur = écartement des jambes."""
    n = len(cx)
    if n < 10:
        return scene_repos()

    idx = np.linspace(0, n - 1, min(n, 700)).astype(int)
    xs = 54 + np.linspace(0, 1, len(idx)) * (W - 108)

    v = cx[idx].astype(float)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    ys = (H - 74) - (v - lo) / ((hi - lo) or 1.0) * (H - 158)

    sp = spread[idx].astype(float)
    ok = np.isfinite(sp)
    if ok.sum() > 5:
        sp = np.interp(np.arange(len(sp)), np.flatnonzero(ok), sp[ok])
        lo2, hi2 = np.percentile(sp, 4), np.percentile(sp, 96)
        sp = np.clip((sp - lo2) / ((hi2 - lo2) or 1.0), 0, 1)
    else:
        sp = np.zeros(len(sp))
    ep = 2.5 + sp * 15.0                       # demi-épaisseur du ruban

    haut = " L ".join(f"{a:.1f},{b - e:.1f}" for a, b, e in zip(xs, ys, ep))
    bas = " L ".join(f"{a:.1f},{b + e:.1f}" for a, b, e in
                     zip(xs[::-1], ys[::-1], ep[::-1]))
    corps = f'<path class="ribbon-body" d="M {haut} L {bas} Z" fill="url(#fill)"/>'
    lueur = (f'<path d="M {" L ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))}" '
             f'stroke="url(#grad)" stroke-width="22" opacity=".22" fill="none" '
             f'filter="url(#soft)"/>')
    ligne = (f'<path class="ribbon-line" d="M '
             f'{" L ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))}"/>')

    bandes = []
    for a, b in turns:
        x0 = 54 + (a / max(1, n - 1)) * (W - 108)
        x1 = 54 + (b / max(1, n - 1)) * (W - 108)
        bandes.append(f'<rect class="turnband" x="{x0:.1f}" y="28" '
                      f'width="{max(2.5, x1 - x0):.1f}" height="{H - 76}" rx="3"/>')

    from longevis import gait as _g
    pts = []
    for a, b in passes:
        times, _f0 = _g.step_events(spread, fps, (a, b))
        for s in times:
            i = int(a + s * fps)
            if 0 <= i < n:
                j = int(np.searchsorted(idx, i))
                if 0 <= j < len(xs):
                    pts.append(f'<circle class="stepdot" cx="{xs[j]:.1f}" '
                               f'cy="{ys[j]:.1f}" r="2.1"/>')

    duree = n / fps
    ticks = "".join(
        f'<text class="lbl-t" x="{54 + f * (W - 108):.0f}" y="{H - 18}">'
        f'{f * duree:.0f}s</text>' for f in (0, .25, .5, .75, 1))

    return f'''<div class="iv-stage"><svg viewBox="0 0 {W} {H}" role="img"
 aria-label="Ruban de marche : déplacement et rythme des jambes">
 {_defs()}{_nappe()}{"".join(bandes)}
 <g class="iv-reveal">{lueur}{corps}{ligne}{"".join(pts)}</g>
 <text class="lbl" x="54" y="26">Déplacement du sujet dans le champ</text>
 {ticks}</svg></div>'''


with st.sidebar:
    st.markdown('<p class="iv-lab" style="margin-bottom:10px">Séquence</p>',
                unsafe_allow_html=True)
    fichier = st.file_uploader("Vidéo de marche",
                               type=["mp4", "mov", "avi", "mkv", "webm", "m4v"])
    taille = st.number_input("Taille du sujet (m)", value=1.72, min_value=0.5,
                             max_value=2.5, step=0.01,
                             help="Environ 6 % d'erreur sur la vitesse.")
    echelle = st.number_input("Échelle (pixels par mètre)", value=0.0,
                              min_value=0.0, step=1.0,
                              help="Plus fiable que la taille. 0 = utiliser la taille.")
    lancer = st.button("Analyser", type="primary", disabled=fichier is None,
                       use_container_width=True)
    st.markdown('<div class="iv-msg" style="font-size:13px;margin-top:26px">'
                '<b>Comment filmer</b><br>Caméra posée, jamais tenue à la main. '
                'Corps entier dans le champ. Le sujet traverse l\'image en '
                'allers-retours, 30 secondes au moins. Fond dégagé.</div>',
                unsafe_allow_html=True)

st.markdown('<span class="iv-tag"><i></i>Analyse vidéo de la marche</span>'
            '<h1 class="iv-title">LongeVis</h1>'
            '<p class="iv-lede">Une vidéo de quelqu\'un qui marche suffit à mesurer '
            'sa vitesse, sa cadence, l\'amplitude de ses pas et ce que lui coûtent '
            'ses demi-tours.</p>', unsafe_allow_html=True)

if not (lancer and fichier is not None):
    st.markdown(scene_repos(), unsafe_allow_html=True)
    st.markdown(LEGENDE, unsafe_allow_html=True)
else:
    chemin = None
    try:
        suffixe = os.path.splitext(fichier.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffixe) as tmp:
            tmp.write(fichier.getbuffer())
            chemin = tmp.name

        with st.spinner("Analyse en cours — une à trois minutes…"):
            traces = body.extract_body(chemin)
            res = pipeline.analyze(
                chemin, mode="mouvement",
                subject_height_m=None if echelle > 0 else taille,
                px_per_m=echelle if echelle > 0 else None, strict=False)

        f, meta = res["features"], res["meta"]
        segs = res.get("segments") or {"passes": [], "turns": []}
        sig = res.get("_signals", {})
        g = lambda k: f.get(k, float("nan"))

        cx = np.asarray(sig.get("body_cx", traces.centroid[:, 0]), dtype=float)
        spread = np.asarray(sig.get("body_spread", traces.leg_spread), dtype=float)
        fps_p = float(sig.get("body_fps", traces.fps))
        if np.isfinite(cx).sum() > 10:
            st.markdown(scene_analyse(cx, spread, fps_p, segs.get("passes", []),
                                      segs.get("turns", [])), unsafe_allow_html=True)
            st.markdown('<div class="iv-key">'
                        '<span><i class="k1"></i>déplacement et rythme du sujet</span>'
                        f'<span><i class="k2"></i>{g("n_steps"):.0f} pas détectés</span>'
                        f'<span><i class="k3"></i>{g("n_turns"):.0f} demi-tours</span>'
                        '</div>', unsafe_allow_html=True)
        else:
            st.markdown(scene_repos(), unsafe_allow_html=True)

        if not np.isfinite(g("gait_speed_m_s")):
            st.markdown('<div class="iv-msg iv-msg--stop"><b>Aucune marche '
                        'mesurable dans cette vidéo.</b> Les causes, par fréquence :'
                        '<ol><li>Le sujet ne <b>traverse</b> pas l\'image. Il faut '
                        'des allers-retours latéraux, pas du surplace ni de la '
                        'marche vers la caméra.</li>'
                        '<li>La caméra bouge. Posez-la sur un support.</li>'
                        '<li>Le sujet est trop près : cadrez le corps entier avec '
                        'de la marge.</li>'
                        '<li>Vidéo trop courte : 30 secondes au moins.</li></ol>'
                        '</div>', unsafe_allow_html=True)

        st.markdown('<p class="iv-h">Mesures principales</p>', unsafe_allow_html=True)
        st.markdown('<div class="iv-grid">'
                    + carte("Vitesse de marche", g("gait_speed_m_s"), "m/s",
                            f.get("gait_speed_band", ""), hero=True)
                    + carte("Cadence", g("cadence_spm"), "pas/min", dec=0)
                    + carte("Amplitude du pas", g("step_length_m"), "m")
                    + carte("Pas analysés", g("n_steps"), "", dec=0)
                    + carte("Demi-tours", g("n_turns"), "", dec=0)
                    + carte("Durée", meta.get("duration_s"), "s", dec=0)
                    + '</div>', unsafe_allow_html=True)

        st.markdown('<p class="iv-h">Indices composites</p>', unsafe_allow_html=True)
        st.markdown('<div class="iv-grid">'
                    + carte("Réserve dynamique · IRD", g("ird_reserve_dynamique"),
                            "pas", "Pas dépensés à chaque demi-tour", hero=True)
                    + carte("Signature de foulée · SCF", g("scf_signature_foulee"),
                            "", "Élevé : pas amples. Bas : pas hachés.",
                            hero=True, dec=3)
                    + '</div>'
                    '<div class="iv-msg">Ces deux indices sont des rapports sans '
                    'dimension : ils ne dépendent ni de la calibration, ni de la '
                    'distance de la caméra. Ce sont des <b>hypothèses de '
                    'recherche</b>, non validées sur cohorte humaine — à lire en '
                    'comparant une personne à elle-même dans le temps.</div>',
                    unsafe_allow_html=True)

        st.markdown('<p class="iv-h">Contrôle de l\'acquisition</p>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns([3, 2])
        with c1:
            if traces.preview is not None:
                st.image(traces.preview[:, :, ::-1],
                         caption="Cadre vert : silhouette détectée. "
                                 "Trait orange : zone des jambes.",
                         use_container_width=True)
            else:
                st.markdown('<div class="iv-msg iv-msg--stop">Aucune silhouette '
                            'isolée. La caméra bouge, le fond est chargé, ou le '
                            'sujet occupe presque tout le champ.</div>',
                            unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="iv-grid" style="grid-template-columns:1fr 1fr">'
                        + carte("Corps détecté", traces.detection_rate * 100, "%",
                                dec=0)
                        + carte("Caméra", traces.camera_motion_px, "px/img")
                        + carte("Trajets", g("n_passes"), "", dec=0)
                        + carte("Échelle", g("px_per_m"), "px/m", dec=0)
                        + '</div>', unsafe_allow_html=True)
            for a in ([] if traces.camera_motion_px <= 0.5 else
                      ["La caméra bouge. Posez-la sur un support stable."]) + \
                     ([] if traces.detection_rate >= 0.6 else
                      ["Corps mal détecté. Reculez, dégagez le fond."]) + \
                     ([] if meta.get("task") == "marche" else
                      [f"Activité reconnue : « {meta.get('task')} ». "
                       "Pour la marche, filmez des allers-retours."]):
                st.markdown(f'<div class="iv-msg iv-msg--warn">{a}</div>',
                            unsafe_allow_html=True)

        with st.expander("Toutes les mesures"):
            L = ['<table class="iv-tbl"><thead><tr><th>Mesure</th><th>Valeur</th>'
                 '<th>Référence</th><th>Statut</th></tr></thead><tbody>']
            for titre, cles in GROUPES:
                dispo = [k for k in cles if isinstance(f.get(k), (int, float))
                         and np.isfinite(f.get(k))]
                if not dispo:
                    continue
                L.append(f'<tr><td class="iv-g" colspan="4">{titre}</td></tr>')
                for k in dispo:
                    v = f[k]
                    norm = REFERENCE_NORMS.get(k)
                    ref = f"{norm[0]:g} ± {norm[1]:g}" if norm else "—"
                    dec = 3 if abs(v) < 1 else (0 if abs(v) > 100 else 2)
                    L.append(f'<tr><td>{LABELS.get(k, k)}</td>'
                             f'<td class="iv-v">{v:.{dec}f} '
                             f'<span style="color:var(--faint);font-size:11.5px">'
                             f'{UNITS.get(k, "")}</span></td>'
                             f'<td class="iv-r">{ref}</td><td>{tag(k)}</td></tr>')
            L.append('</tbody></table>')
            st.markdown("".join(L), unsafe_allow_html=True)
            st.markdown('<div class="iv-msg"><b>Non résolu</b> : le bruit propre de '
                        'la méthode dépasse l\'écart attendu entre deux personnes. '
                        'La valeur est exacte, mais elle ne permet pas de les '
                        'distinguer.<br><b>Non fiable</b> : indice n\'ayant pas '
                        'passé sa validation, à ne pas interpréter.</div>',
                        unsafe_allow_html=True)

    except Exception:
        st.markdown('<div class="iv-msg iv-msg--stop"><b>L\'analyse a échoué.</b> '
                    'Vérifiez que le fichier est une vidéo lisible.</div>',
                    unsafe_allow_html=True)
        st.code(traceback.format_exc(limit=2))
    finally:
        if chemin and os.path.exists(chemin):
            os.remove(chemin)

st.markdown('<p class="iv-foot"><b>Outil de recherche — pas un dispositif '
            'médical.</b> Aucun diagnostic, aucune prédiction d\'espérance de vie. '
            'Les repères proviennent d\'études de population : ils décrivent des '
            'moyennes dans de grands groupes, jamais la trajectoire d\'une '
            'personne.<br><b>Vos vidéos ne sont pas conservées</b> : chaque fichier '
            'est supprimé dès l\'analyse terminée.</p>', unsafe_allow_html=True)
