"""Rapport HTML autonome (aucune ressource externe, lisible hors ligne)."""

from __future__ import annotations
import datetime as _dt
import html
import json
from typing import Dict, List, Optional

import numpy as np

from .config import REFERENCE_NORMS, DOMAINS

UNITS = {
    "hr_bpm": "bpm", "hr_from_ibi_bpm": "bpm", "hrv_sdnn_ms": "ms",
    "hrv_rmssd_ms": "ms", "hrv_pnn50_pct": "%", "lf_hf_ratio": "",
    "perfusion_index": "%", "pulse_rise_ratio": "", "pulse_amp_cv": "",
    "resp_rate_cpm": "cycles/min", "tremor_power_norm": "", "tremor_peak_hz": "Hz",
    "sway_rms_px": "% diag.", "blink_rate_min": "/min", "blink_mean_dur_ms": "ms",
    "wrinkle_index": "", "tone_evenness": "", "erythema_index": "",
    "melanin_index": "", "texture_anisotropy": "", "pulse_snr_db": "dB",
}

LABELS = {
    "hr_bpm": "Fréquence cardiaque (spectre)",
    "hr_from_ibi_bpm": "Fréquence cardiaque (battements)",
    "hrv_sdnn_ms": "SDNN — variabilité globale",
    "hrv_rmssd_ms": "RMSSD — modulation vagale",
    "hrv_pnn50_pct": "pNN50",
    "lf_hf_ratio": "Rapport BF/HF",
    "perfusion_index": "Indice de perfusion",
    "pulse_rise_ratio": "Temps de montée relatif",
    "pulse_amp_cv": "Variabilité d'amplitude",
    "resp_rate_cpm": "Fréquence respiratoire",
    "tremor_power_norm": "Puissance de tremblement (4–12 Hz)",
    "tremor_peak_hz": "Pic de tremblement",
    "sway_rms_px": "Oscillation posturale",
    "blink_rate_min": "Taux de clignement",
    "blink_mean_dur_ms": "Durée de clignement",
    "wrinkle_index": "Indice de rides",
    "tone_evenness": "Homogénéité du teint",
    "erythema_index": "Indice d'érythème",
    "melanin_index": "Indice de mélanine",
    "texture_anisotropy": "Anisotropie de texture",
    "ird_reserve_dynamique": "IRD — Indice de Réserve Dynamique",
    "scf_signature_foulee": "SCF — Signature Cinétique de Foulée",
    "cax_coherence": "CAX — Cohérence Axiale",
    "cax_n_segments": "CAX — trajets exploités",
    "icr_relance": "ICR — Indice de Cinétique de Relance",
    "icr_t_freinage_s": "ICR — temps de freinage",
    "icr_t_relance_s": "ICR — temps de relance",
    "icr_n_virages": "ICR — demi-tours exploités",
    "gait_speed_m_s": "Vitesse de marche",
    "gait_speed_px_s": "Vitesse (unités image)",
    "gait_speed_stature_s": "Vitesse en statures/s",
    "cadence_spm": "Cadence",
    "step_time_s": "Temps de pas",
    "step_time_cv_pct": "Variabilité du temps de pas",
    "stride_time_cv_pct": "Variabilité du temps de foulée",
    "step_asymmetry_pct": "Asymétrie gauche/droite (brute)",
    "step_asymmetry_null_pct": "Plancher d'asymétrie (permutation)",
    "step_asymmetry_net_pct": "Asymétrie nette",
    "step_asymmetry_p": "p (asymétrie ≠ hasard)",
    "step_length_m": "Longueur de pas",
    "step_length_stature": "Longueur de pas relative",
    "harmonic_ratio": "Rapport harmonique du tronc",
    "gait_sparc": "Fluidité (SPARC)",
    "gait_jerk_norm": "Secousse normalisée",
    "com_bob_stature_pct": "Oscillation verticale du tronc",
    "turn_mean_dur_s": "Durée moyenne de demi-tour",
    "turn_max_dur_s": "Durée maximale de demi-tour",
    "n_steps": "Pas analysés", "n_passes": "Trajets", "n_turns": "Demi-tours",
    "sway_rms_ap_mm": "Oscillation antéro-postérieure",
    "sway_rms_ml_mm": "Oscillation médio-latérale",
    "sway_path_mm_s": "Trajet du centre de masse",
    "sway_area_mm2": "Aire d'oscillation (95 %)",
    "sway_f95_hz": "Fréquence à 95 % de puissance",
    "sts_count": "Transferts détectés",
    "sts_mean_dur_s": "Durée moyenne de lever",
    "silhouette_height_px": "Hauteur de silhouette",
    "px_per_m": "Échelle",
    "body_detection_rate": "Taux de détection du corps",
    "camera_motion_px": "Mouvement de caméra",
    "pulse_snr_db": "Rapport signal/bruit du pouls",
}

DOMAIN_LABELS = {
    "marche": "Marche",
    "equilibre": "Équilibre debout",
    "transfert": "Transfert assis-debout",
    "cardio_autonome": "Cardiaque et autonome",
    "vasculaire": "Vasculaire",
    "respiratoire": "Respiratoire",
    "neuromoteur": "Neuromoteur",
    "dermique": "Dermique",
}

GRADE_TEXT = {
    "A": "Signal net. Les biomarqueurs peuvent être lus tels quels.",
    "B": "Signal exploitable. Lire les écarts avec prudence.",
    "C": "Signal faible. Seules les tendances larges ont du sens.",
    "D": "Signal insuffisant. Refaire l'enregistrement avant toute lecture.",
}


def _fmt(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{v:.0f}"
    return html.escape(str(v))


def _waveform_svg(sig: np.ndarray, fps: float, window_s: float = 12.0,
                  width: int = 1000, height: int = 190) -> str:
    """Tracé du pouls sur grille d'enregistrement, avec battements marqués."""
    from . import dsp
    n_win = int(min(len(sig), window_s * fps))
    if n_win < 10:
        return '<p class="empty">Signal trop court pour être tracé.</p>'
    # Fenêtre la plus propre : variance locale la plus proche de la médiane
    best_start, best_cost = 0, None
    step = max(1, int(fps))
    for s in range(0, len(sig) - n_win + 1, step):
        seg = sig[s:s + n_win]
        cost = abs(np.std(seg) - np.std(sig))
        if best_cost is None or cost < best_cost:
            best_start, best_cost = s, cost
    seg = sig[best_start:best_start + n_win]
    t = np.arange(n_win) / fps

    lo, hi = float(np.min(seg)), float(np.max(seg))
    rng = (hi - lo) or 1.0
    pad = 24
    xs = pad + (t / (t[-1] or 1)) * (width - 2 * pad)
    ys = height - pad - ((np.clip(seg, lo, hi) - lo) / rng) * (height - 2 * pad)
    path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    f0, _, _, _ = dsp.dominant_frequency(seg, fps, (0.7, 3.5))
    peaks = dsp.find_pulse_peaks(seg, fps, f0)
    marks = "".join(
        f'<circle cx="{xs[p]:.1f}" cy="{ys[p]:.1f}" r="3.2" class="beat"/>'
        for p in peaks if p < len(xs))

    secs = "".join(
        f'<line class="tick" x1="{pad + (s / window_s) * (width - 2 * pad):.1f}" y1="{height - pad + 4}"'
        f' x2="{pad + (s / window_s) * (width - 2 * pad):.1f}" y2="{height - pad + 9}"/>'
        f'<text class="tickval" x="{pad + (s / window_s) * (width - 2 * pad):.1f}"'
        f' y="{height - 2}">{s}s</text>'
        for s in range(0, int(window_s) + 1, 2))

    return f'''<svg class="strip" viewBox="0 0 {width} {height + 4}" role="img"
     aria-label="Tracé du pouls extrait, {len(peaks)} battements marqués">
  <defs>
    <pattern id="mm" width="8" height="8" patternUnits="userSpaceOnUse">
      <path d="M8 0H0V8" class="gfine"/>
    </pattern>
    <pattern id="cm" width="40" height="40" patternUnits="userSpaceOnUse">
      <rect width="40" height="40" fill="url(#mm)"/>
      <path d="M40 0H0V40" class="gbold"/>
    </pattern>
  </defs>
  <rect width="{width}" height="{height + 4}" fill="url(#cm)"/>
  <path class="trace" d="{path}"/>{marks}{secs}
</svg>'''


def _gait_svg(cx: np.ndarray, spread: np.ndarray, fps: float,
              passes, turns, width: int = 1000, height: int = 210) -> str:
    """Trajectoire horizontale et écartement des jambes sur la même échelle.

    Les trajets rectilignes sont laissés clairs, les demi-tours ombrés : c'est
    la séparation sur laquelle repose tout le calcul de vitesse et de cadence,
    autant la rendre vérifiable à l'œil.
    """
    n = len(cx)
    if n < 10:
        return '<p class="empty">Trajectoire trop courte pour être tracée.</p>'
    pad = 22
    top_h, gap = 104, 8
    xs = pad + np.linspace(0, 1, n) * (width - 2 * pad)

    def _scale(v, y0, y1):
        lo, hi = float(np.min(v)), float(np.max(v))
        rng = (hi - lo) or 1.0
        return y1 - (v - lo) / rng * (y1 - y0)

    ys_pos = _scale(cx, pad, top_h)
    ys_spr = _scale(spread, top_h + gap + 10, height - pad)

    shade = "".join(
        f'<rect class="turn" x="{xs[a]:.1f}" y="{pad - 6:.1f}" '
        f'width="{max(1.0, xs[min(b, n - 1)] - xs[a]):.1f}" height="{height - pad - 10:.1f}"/>'
        for a, b in turns)

    p1 = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys_pos))
    p2 = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys_spr))

    from . import gait as _g
    marks = []
    for a, b in passes:
        times, _f0 = _g.step_events(spread, fps, (a, b))
        for t in times:
            i = int(a + t * fps)
            if 0 <= i < n:
                marks.append(f'<circle cx="{xs[i]:.1f}" cy="{ys_spr[i]:.1f}" r="2.8" class="beat"/>')

    dur = n / fps
    ticks = "".join(
        f'<text class="tickval" x="{pad + (s / dur) * (width - 2 * pad):.1f}"'
        f' y="{height - 4}">{s}s</text>'
        for s in range(0, int(dur) + 1, max(2, int(dur // 8))))

    return f'''<svg class="strip" viewBox="0 0 {width} {height}" role="img"
     aria-label="Trajectoire horizontale et écartement des jambes,
     {len(passes)} trajets et {len(turns)} demi-tours">
  <defs>
    <pattern id="mm2" width="8" height="8" patternUnits="userSpaceOnUse">
      <path d="M8 0H0V8" class="gfine"/>
    </pattern>
    <pattern id="cm2" width="40" height="40" patternUnits="userSpaceOnUse">
      <rect width="40" height="40" fill="url(#mm2)"/>
      <path d="M40 0H0V40" class="gbold"/>
    </pattern>
  </defs>
  <rect width="{width}" height="{height}" fill="url(#cm2)"/>
  {shade}
  <path class="trace trace--motion" d="{p1}"/>
  <path class="trace" d="{p2}"/>{"".join(marks)}
  <text class="slab" x="{pad}" y="{pad - 8}">position horizontale</text>
  <text class="slab" x="{pad}" y="{top_h + gap + 6}">écartement des jambes</text>
  {ticks}
</svg>'''


def _motion_svg(head: np.ndarray, fps: float, window_s: float = 12.0,
                width: int = 1000, height: int = 78) -> str:
    if head is None or len(head) < 10:
        return ""
    n = int(min(len(head), window_s * fps))
    v = head[:n, 1] - np.median(head[:n, 1])
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    rng = (hi - lo) or 1.0
    pad = 12
    xs = pad + np.linspace(0, 1, n) * (width - 2 * pad)
    ys = height - pad - ((np.clip(v, lo, hi) - lo) / rng) * (height - 2 * pad)
    path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return f'''<svg class="strip strip--motion" viewBox="0 0 {width} {height}" role="img"
     aria-label="Déplacement vertical de la tête sur la même fenêtre">
  <rect width="{width}" height="{height}" fill="url(#cm)"/>
  <path class="trace trace--motion" d="{path}"/>
</svg>'''


def _z_axis(z: float) -> str:
    if not np.isfinite(z):
        return '<span class="zbar zbar--empty">non mesuré</span>'
    pos = 50 + max(-3, min(3, z)) / 3 * 46
    cls = "hi" if z > 0.8 else ("lo" if z < -0.8 else "mid")
    return (f'<span class="zbar"><i class="zaxis"></i>'
            f'<i class="zmark {cls}" style="left:{pos:.1f}%"></i></span>')


def _feature_rows(features: Dict[str, float], zs: Dict[str, float],
                  keys: List[str]) -> str:
    rows = []
    for k in keys:
        if k not in features:
            continue
        v = features[k]
        if not isinstance(v, (int, float)):
            continue
        norm = REFERENCE_NORMS.get(k)
        ref = f"{_fmt(norm[0])} ± {_fmt(norm[1])}" if norm else "—"
        digits = 3 if abs(v) < 1 and np.isfinite(v) else 1
        rows.append(f'''<tr>
  <th scope="row">{html.escape(LABELS.get(k, k))}</th>
  <td class="num">{_fmt(v, digits)}<span class="unit">{html.escape(UNITS.get(k, ""))}</span></td>
  <td class="ref">{ref}</td>
  <td class="zcell">{_z_axis(zs.get(k, float("nan")))}</td>
</tr>''')
    return "\n".join(rows)


def render(result: Dict[str, object], out_path: str,
           ground_truth: Optional[Dict[str, float]] = None,
           demo: bool = False) -> str:
    """Rapport HTML.

    `demo=True` affiche et note tous les marqueurs mesurés, sans le bloc de
    réserve sur ceux que la méthode résout mal. Réservé à la présentation :
    les chiffres sortent bien de la mesure, mais certains ne permettent pas de
    comparer deux sujets. Le mode par défaut reste le mode strict.

    """
    meta = result["meta"]
    feats = result["features"]
    ind = result["indices"]
    q = ind["quality"]
    zs = ind["z_scores"]
    sig = result.get("_signals", {})
    analyses = meta.get("analyses", ["visage"])
    has_body = "mouvement" in analyses
    has_face = "visage" in analyses

    recorder, legend = "", ""
    if has_body:
        segs = result.get("segments") or {"passes": [], "turns": []}
        recorder = _gait_svg(np.asarray(sig.get("body_cx", [])),
                             np.asarray(sig.get("body_spread", [])),
                             sig.get("body_fps", 30.0),
                             segs.get("passes", []), segs.get("turns", []))
        legend = (f'<span><b>Tracé bleu</b> position horizontale</span>'
                  f'<span><b>Tracé rouge</b> écartement des jambes</span>'
                  f'<span><b>Points</b> pas datés ({feats.get("n_steps", 0):.0f})</span>'
                  f'<span><b>Zones ombrées</b> demi-tours ({feats.get("n_turns", 0):.0f})</span>')
        v = feats.get("gait_speed_m_s")
        if v and np.isfinite(v):
            legend += f'<span><b>{v:.2f} m/s</b></span>'
    if has_face:
        strip = _waveform_svg(np.asarray(sig.get("pulse", [])), sig.get("fps", 30.0))
        motion = _motion_svg(np.asarray(sig.get("head", [])), sig.get("fps", 30.0))
        recorder += strip + motion
        hr = feats.get("hr_from_ibi_bpm") or feats.get("hr_bpm")
        legend += (f'<span><b>Tracé rouge</b> pouls ({html.escape(str(meta.get("best_method", "—")))})</span>'
                   f'<span><b>{_fmt(hr, 1)} bpm</b></span>')

    domain_cards = []
    for d, w in DOMAINS.items():
        s = ind["domains"].get(d, {})
        score = s.get("score", float("nan"))
        cov = s.get("coverage", 0.0)
        # Un domaine sans aucun marqueur mesuré n'a rien à dire : l'afficher
        # vide donnerait l'impression d'un résultat manquant plutôt que d'une
        # analyse hors périmètre de l'enregistrement.
        if cov <= 0 or not np.isfinite(score):
            continue
        domain_cards.append(f'''<article class="dom">
  <p class="dom__name">{html.escape(DOMAIN_LABELS.get(d, d))}</p>
  <p class="dom__score">{_fmt(score, 0) if np.isfinite(score) else "—"}</p>
  <p class="dom__cov">{_fmt(cov * 100, 0)}&#8239;% des marqueurs mesurés</p>
  {_z_axis(s.get("z", float("nan")))}
</article>''')

    groups = [
        ("Indices composites", ["ird_reserve_dynamique", "scf_signature_foulee",
                                "cax_coherence", "cax_n_segments",
                                "icr_relance", "icr_t_freinage_s",
                                "icr_t_relance_s", "icr_n_virages"]),
        ("Marche", ["gait_speed_m_s", "gait_speed_stature_s", "cadence_spm",
                    "step_time_s", "step_length_m", "step_length_stature",
                    "step_time_cv_pct", "stride_time_cv_pct",
                    "step_asymmetry_pct", "step_asymmetry_null_pct",
                    "step_asymmetry_net_pct", "step_asymmetry_p",
                    "harmonic_ratio", "gait_sparc", "com_bob_stature_pct",
                    "turn_mean_dur_s", "turn_max_dur_s",
                    "n_steps", "n_passes", "n_turns"]),
        ("Équilibre debout", ["sway_rms_ap_mm", "sway_rms_ml_mm", "sway_path_mm_s",
                              "sway_area_mm2", "sway_f95_hz"]),
        ("Transfert assis-debout", ["sts_count", "sts_mean_dur_s"]),
        ("Acquisition", ["silhouette_height_px", "px_per_m",
                         "body_detection_rate", "camera_motion_px"]),
        ("Cardiaque et autonome", ["hr_bpm", "hr_from_ibi_bpm", "hrv_sdnn_ms",
                                   "hrv_rmssd_ms", "hrv_pnn50_pct", "lf_hf_ratio"]),
        ("Vasculaire", ["perfusion_index", "pulse_rise_ratio", "pulse_amp_cv"]),
        ("Respiratoire", ["resp_rate_cpm"]),
        ("Neuromoteur", ["tremor_power_norm", "tremor_peak_hz", "sway_rms_px",
                         "blink_rate_min", "blink_mean_dur_ms"]),
        ("Dermique", ["wrinkle_index", "texture_anisotropy", "tone_evenness",
                      "melanin_index", "erythema_index"]),
    ]
    groups = [(nm, ks) for nm, ks in groups
              if any(isinstance(feats.get(k), (int, float)) for k in ks)]
    tables = "".join(
        f'''<section class="tblock">
  <h3>{html.escape(name)}</h3>
  <table class="feat">
    <thead><tr><th scope="col">Marqueur</th><th scope="col">Valeur</th>
      <th scope="col">Référence</th><th scope="col">Écart</th></tr></thead>
    <tbody>{_feature_rows(feats, zs, keys)}</tbody>
  </table>
</section>''' for name, keys in groups)

    methods = "".join(
        f'<tr><th scope="row">{html.escape(k)}</th><td class="num">{_fmt(v["hr_bpm"], 1)}</td>'
        f'<td class="num">{_fmt(v["snr_db"], 1)}</td>'
        f'<td class="num">{_fmt(v["spectral_purity"], 2)}</td></tr>'
        for k, v in result["methods"].items())

    flags = "".join(f"<li>{html.escape(f)}</li>" for f in q["flags"]) or \
        "<li>Aucune réserve technique sur cet enregistrement.</li>"

    gt_block = ""
    if ground_truth:
        pairs = [("Fréquence cardiaque", "true_hr_bpm", "hr_from_ibi_bpm", "bpm"),
                 ("SDNN", "true_hrv_sdnn_ms", "hrv_sdnn_ms", "ms"),
                 ("RMSSD", "true_hrv_rmssd_ms", "hrv_rmssd_ms", "ms"),
                 ("Fréquence respiratoire", "true_resp_cpm", "resp_rate_cpm", "cycles/min"),
                 ("Tremblement", "true_tremor_hz", "tremor_peak_hz", "Hz"),
                 ("Clignement", "true_blink_rate_min", "blink_rate_min", "/min")]
        rows = []
        for label, gk, fk, unit in pairs:
            if gk not in ground_truth:
                continue
            tv, mv = ground_truth[gk], feats.get(fk, float("nan"))
            err = mv - tv if np.isfinite(mv) else float("nan")
            pct = 100 * err / tv if tv else float("nan")
            rows.append(f'<tr><th scope="row">{label}</th><td class="num">{_fmt(tv,2)}'
                        f'<span class="unit">{unit}</span></td><td class="num">{_fmt(mv,2)}</td>'
                        f'<td class="num">{_fmt(err,2)}</td><td class="num">{_fmt(pct,1)}<span class="unit">%</span></td></tr>')
        gt_block = f'''<section class="tblock">
  <h3>Contrôle sur vérité terrain</h3>
  <p class="lede">Enregistrement synthétique : les valeurs injectées sont connues,
  l'écart mesure donc l'erreur propre de la chaîne de traitement.</p>
  <table class="feat">
    <thead><tr><th scope="col">Grandeur</th><th scope="col">Injecté</th>
    <th scope="col">Mesuré</th><th scope="col">Écart</th><th scope="col">Écart relatif</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</section>'''

    calib = ind.get("calibrated")
    calib_block = ""
    if calib:
        calib_block = f'''<section class="tblock">
  <h3>Modèle calibré</h3>
  <table class="feat"><tbody>
    <tr><th scope="row">Cible apprise</th><td class="num">{html.escape(str(calib["target"]))}</td></tr>
    <tr><th scope="row">Prédiction</th><td class="num">{_fmt(calib["prediction"], 2)}</td></tr>
    <tr><th scope="row">Erreur absolue en validation croisée</th><td class="num">{_fmt(calib.get("cv_mae"), 2)}</td></tr>
    <tr><th scope="row">R² en validation croisée</th><td class="num">{_fmt(calib.get("cv_r2"), 3)}</td></tr>
    <tr><th scope="row">Enregistrements d'entraînement</th><td class="num">{_fmt(calib.get("n_train"), 0)}</td></tr>
  </tbody></table>
</section>'''

    bits = []
    if has_body:
        bits.append(f'corps détecté sur {_fmt(meta.get("body_detection_rate", 0) * 100, 0)}&#8239;% des images')
        cm = meta.get("camera_motion_px", 0.0)
        bits.append(f'mouvement de caméra {_fmt(cm, 2)}&#8239;px/image'
                    + (" — caméra non fixe, mesures à interpréter avec réserve" if cm > 0.5 else ""))
    if has_face:
        bits.append(f'rapport signal/bruit du pouls {_fmt(q["snr_db"], 1)}&#8239;dB')
        bits.append(f'visage détecté sur {_fmt(meta.get("detection_rate", 0) * 100, 0)}&#8239;% des tentatives')
    quality_line = " — ".join(bits).capitalize() + "."

    unres = ind.get("unresolved", {})
    if unres and not demo:
        items = "".join(
            f"<li><b>{html.escape(LABELS.get(k, k))}</b> : mesuré {_fmt(d['value'], 2)}, "
            f"mais le plancher de bruit de la méthode vaut {_fmt(d['noise_floor'], 2)} "
            f"pour un écart-type de norme de {_fmt(d['norm_sd'], 2)}.</li>"
            for k, d in unres.items())
        unres_block = f'''<section class="unres-note">
  <p style="margin-top:0"><strong>Marqueurs mesurés mais non résolus.</strong>
  Pour ceux-ci, le bruit propre de la méthode dépasse la dispersion attendue
  entre individus : la valeur est affichée pour information, mais elle ne peut
  pas distinguer deux sujets et n'entre pas dans les scores.</p>
  <ul>{items}</ul>
</section>'''
    else:
        unres_block = ""

    composite = ind["composite_score"]
    stamp = _dt.datetime.now().strftime("%d/%m/%Y %H:%M")

    doc = f'''<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rapport LongeVis — {html.escape(str(meta["file"]))}</title>
<style>
:root {{
  --paper:#FBF4F2; --paper-2:#FFFDFC; --grid:#F0CBC3; --grid-bold:#E0A196;
  --ink:#241E1C; --ink-2:#6B5D58; --trace:#A81E12; --steel:#2C4E5C;
  --rule:#E7D6D1; --good:#2C6E4A; --warn:#9A5B12;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,"Roboto Mono",monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1080px;margin:0 auto;padding:32px 20px 72px}}
a{{color:var(--steel)}}
:focus-visible{{outline:2px solid var(--steel);outline-offset:3px}}

.eyebrow{{font-family:var(--mono);font-size:11px;letter-spacing:.18em;
  text-transform:uppercase;color:var(--ink-2);margin:0 0 6px}}
h1{{font-family:var(--mono);font-size:clamp(24px,4.2vw,34px);letter-spacing:-.02em;
  font-weight:600;margin:0 0 4px}}
h2{{font-family:var(--mono);font-size:13px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--ink-2);font-weight:600;margin:44px 0 14px;
  padding-bottom:8px;border-bottom:1px solid var(--rule)}}
h3{{font-family:var(--mono);font-size:12px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--ink-2);font-weight:600;margin:26px 0 8px}}
p.lede{{color:var(--ink-2);max-width:62ch;margin:.2em 0 1em}}

.head{{display:flex;flex-wrap:wrap;gap:18px;justify-content:space-between;align-items:flex-end}}
.filemeta{{font-family:var(--mono);font-size:12px;color:var(--ink-2)}}
.filemeta span{{margin-right:14px;white-space:nowrap}}

/* --- Signature : bande d'enregistrement physiologique --- */
.recorder{{margin:26px 0 8px;border:1px solid var(--grid-bold);border-radius:2px;
  background:var(--paper-2);overflow:hidden}}
.strip{{display:block;width:100%;height:auto}}
.trace{{fill:none;stroke:var(--trace);stroke-width:1.9;stroke-linejoin:round;
  stroke-linecap:round}}
.trace--motion{{stroke:var(--steel);stroke-width:1.4;opacity:.85}}
.strip--motion{{border-top:1px dashed var(--grid-bold)}}
.beat{{fill:var(--paper-2);stroke:var(--trace);stroke-width:1.5}}
.gfine{{fill:none;stroke:var(--grid);stroke-width:.7}}
.gbold{{fill:none;stroke:var(--grid-bold);stroke-width:1}}
.tick{{stroke:var(--grid-bold);stroke-width:1}}
.tickval{{font-family:var(--mono);font-size:9px;fill:var(--ink-2);text-anchor:middle}}
.slab{{font-family:var(--mono);font-size:9px;fill:var(--ink-2);letter-spacing:.1em;
  text-transform:uppercase}}
.turn{{fill:var(--grid-bold);opacity:.22}}
.unres{{font-family:var(--mono);font-size:10px;color:var(--warn);white-space:nowrap}}
.unres-note{{margin:10px 0 0;padding:12px 14px;border-left:3px solid var(--warn);
  background:#FFFAF2;font-size:13px;color:var(--ink-2)}}
.striplegend{{font-family:var(--mono);font-size:11px;color:var(--ink-2);
  margin:8px 2px 0;letter-spacing:.04em}}
.striplegend span{{display:inline-block;margin:0 22px 4px 0}}
.striplegend b{{color:var(--ink);font-weight:600}}
@media (prefers-reduced-motion:no-preference){{
  .trace{{stroke-dasharray:6000;stroke-dashoffset:6000;animation:draw 1.6s ease-out forwards}}
  @keyframes draw{{to{{stroke-dashoffset:0}}}}
}}

/* --- Qualité --- */
.quality{{display:grid;grid-template-columns:auto 1fr;gap:20px;align-items:start;
  margin:30px 0 0;padding:18px 20px;background:var(--paper-2);
  border:1px solid var(--rule);border-left:3px solid var(--trace);border-radius:2px}}
.grade{{font-family:var(--mono);font-size:44px;font-weight:600;line-height:1;
  color:var(--trace)}}
.grade small{{display:block;font-size:10px;letter-spacing:.16em;color:var(--ink-2);
  margin-top:6px;font-weight:400}}
.quality ul{{margin:8px 0 0;padding-left:18px;color:var(--ink-2);font-size:14px}}
.quality li{{margin-bottom:3px}}

/* --- Domaines --- */
.doms{{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));
  align-items:start}}
.dom{{background:var(--paper-2);border:1px solid var(--rule);border-radius:2px;padding:14px 15px}}
.dom__name{{font-size:12px;color:var(--ink-2);margin:0 0 6px;letter-spacing:.02em}}
.dom__score{{font-family:var(--mono);font-size:30px;font-weight:600;margin:0;letter-spacing:-.02em}}
.dom__cov{{font-family:var(--mono);font-size:10px;color:var(--ink-2);margin:2px 0 10px}}

/* --- Échelle d'écart --- */
.zbar{{position:relative;display:block;height:16px;width:100%;min-width:110px}}
.zaxis{{position:absolute;top:7px;left:2%;right:2%;height:1px;background:var(--rule)}}
.zbar::before{{content:"";position:absolute;top:2px;bottom:2px;left:50%;
  width:1px;background:var(--grid-bold)}}
.zmark{{position:absolute;top:3px;width:9px;height:9px;margin-left:-4.5px;border-radius:50%;
  border:1.5px solid var(--paper-2)}}
.zmark.hi{{background:var(--good)}} .zmark.mid{{background:var(--ink-2)}}
.zmark.lo{{background:var(--trace)}}
.zbar--empty{{font-family:var(--mono);font-size:10px;color:var(--ink-2)}}

/* --- Tables --- */
.tblock{{margin-bottom:6px}}
table.feat{{width:100%;border-collapse:collapse;font-size:14px}}
table.feat th[scope=col]{{font-family:var(--mono);font-size:10px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-2);font-weight:500;text-align:left;
  padding:6px 10px 6px 0;border-bottom:1px solid var(--rule)}}
table.feat td,table.feat th[scope=row]{{padding:7px 10px 7px 0;
  border-bottom:1px solid var(--rule);vertical-align:middle}}
table.feat th[scope=row]{{font-weight:400;text-align:left;color:var(--ink)}}
.num{{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}}
.unit{{color:var(--ink-2);font-size:11px;margin-left:3px}}
.ref{{font-family:var(--mono);font-size:12px;color:var(--ink-2);white-space:nowrap}}
.zcell{{width:150px}}
tbody tr:hover{{background:#FFF8F6}}

.notice{{margin-top:44px;padding:18px 20px;border:1px solid var(--rule);
  border-radius:2px;background:var(--paper-2);font-size:13px;color:var(--ink-2)}}
.notice strong{{color:var(--ink)}}
.foot{{margin-top:22px;font-family:var(--mono);font-size:11px;color:var(--ink-2);
  letter-spacing:.04em}}
@media (max-width:560px){{
  .quality{{grid-template-columns:1fr}} .zcell{{width:96px}}
  table.feat{{font-size:13px}} .ref{{display:none}}
  table.feat th[scope=col]:nth-child(3){{display:none}}
}}
</style>
</head>
<body>
<main class="wrap">

<header class="head">
  <div>
    <p class="eyebrow">Rapport d'analyse vidéo</p>
    <h1>Biomarqueurs de vieillissement</h1>
  </div>
  <p class="filemeta">
    <span>{html.escape(str(meta["file"]).split("/")[-1])}</span>
    <span>{_fmt(meta["duration_s"], 1)}&#8239;s</span>
    <span>{_fmt(meta["fps"], 1)}&#8239;i/s</span>
    <span>{meta["n_frames"]} images</span>
    <span>{_fmt(meta["processing_s"], 1)}&#8239;s de calcul</span>
  </p>
</header>

<div class="recorder">{recorder}</div>
<p class="striplegend">{legend}</p>
<p style="display:none">
  <span><b>Tracé rouge</b> pouls reconstruit ({html.escape(str(meta["best_method"]))})</span>
  <span><b>Points</b> battements détectés</span>
  <span><b>Tracé bleu</b> déplacement vertical de la tête</span>
  <span><b>{_fmt(feats.get("hr_from_ibi_bpm") or feats.get("hr_bpm"), 1)} bpm</b></span>
</p>

<section class="quality" aria-labelledby="q-title">
  <p class="grade">{html.escape(q["grade"])}<small>Qualité</small></p>
  <div>
    <p id="q-title" style="margin:0 0 2px"><strong>{html.escape(GRADE_TEXT.get(q["grade"], ""))}</strong></p>
    <p class="lede" style="margin:0">{quality_line}</p>
    <ul>{flags}</ul>
  </div>
</section>

<h2>Synthèse par domaine</h2>
<p class="lede">Score sur 100, dérivé de l'écart aux normes de référence. Le repère
sur l'axe montre cet écart&#8239;: à gauche du trait central, en deçà de la norme&#8239;;
à droite, au-delà.</p>
<div class="doms">{"".join(domain_cards)}
  <article class="dom" style="border-color:var(--grid-bold)">
    <p class="dom__name">Composite</p>
    <p class="dom__score" style="color:var(--trace)">{_fmt(composite, 0) if np.isfinite(composite) else "—"}</p>
    <p class="dom__cov">mode {html.escape(str(ind["mode"]))}</p>
  </article>
</div>

<h2>Marqueurs détaillés</h2>
{tables}

<h2>Contrôles</h2>
<section class="tblock">
  <h3>Comparaison des méthodes rPPG</h3>
  <table class="feat">
    <thead><tr><th scope="col">Méthode</th><th scope="col">FC (bpm)</th>
      <th scope="col">SNR (dB)</th><th scope="col">Pureté spectrale</th></tr></thead>
    <tbody>{methods}</tbody>
  </table>
</section>
{gt_block}
{calib_block}

{unres_block}

<section class="notice">
  <p style="margin-top:0"><strong>Ce rapport n'est pas un acte médical.</strong>
  LongeVis est un outil de recherche et de bien-être. Il n'est ni un dispositif
  médical au sens du règlement (UE) 2017/745, ni validé cliniquement, et ne
  permet ni de diagnostiquer, ni de prédire une espérance de vie.</p>
  <p>Les scores composites reposent sur des normes de population publiées et sur
  une pondération choisie a priori. Tant qu'un modèle n'a pas été calibré sur une
  cohorte annotée, ils indiquent une position relative, pas un âge biologique.
  Les mesures optiques dépendent de l'éclairage, du phototype, du maquillage et
  de la caméra&#8239;: comparez d'abord un sujet à lui-même, dans des conditions
  identiques.</p>
  <p style="margin-bottom:0">Les données vidéo faciales sont des données
  biométriques au sens du RGPD (article 9). Leur traitement suppose une base
  légale explicite, une minimisation et une durée de conservation définie.</p>
</section>

<p class="foot">LongeVis v0.1 · rapport généré le {stamp} · méthode retenue&#8239;:
{html.escape(str(meta["best_method"]))}</p>
</main>
</body>
</html>'''

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return out_path


def write_json(result: Dict[str, object], out_path: str) -> str:
    payload = {k: v for k, v in result.items() if not k.startswith("_")}

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(float(o)) else round(float(o), 6)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_clean(payload), fh, indent=2, ensure_ascii=False)
    return out_path
