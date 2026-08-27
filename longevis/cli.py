"""Interface en ligne de commande.

    python -m longevis analyser  video.mp4 --rapport rapport.html
    python -m longevis demo      --duree 90
    python -m longevis lot       dossier/ --csv cohorte.csv
    python -m longevis calibrer  cohorte.csv --cible age --groupe subject_id
"""

from __future__ import annotations
import argparse
import csv
import glob
import os
import sys

import numpy as np

from . import index, pipeline, report
from .config import DEFAULT

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def _print_summary(res: dict) -> None:
    f, ind = res["features"], res["indices"]
    q = ind["quality"]
    print(f"  qualité       {q['grade']}   SNR {q['snr_db']:.1f} dB")
    hr = f.get("hr_from_ibi_bpm")
    hr = hr if hr and np.isfinite(hr) else f.get("hr_bpm", float('nan'))
    print(f"  FC            {hr:.1f} bpm")
    for k, lab in (("hrv_rmssd_ms", "RMSSD        "), ("hrv_sdnn_ms", "SDNN         "),
                   ("resp_rate_cpm", "Respiration  "), ("blink_rate_min", "Clignement   ")):
        v = f.get(k, float("nan"))
        print(f"  {lab} {v:.1f}" if np.isfinite(v) else f"  {lab} —")
    c = ind["composite_score"]
    print(f"  composite     {c:.0f}/100" if np.isfinite(c) else "  composite     —")
    for flag in q["flags"]:
        print(f"  ! {flag}")


def cmd_analyser(args) -> int:
    model = index.load_model(args.modele)
    res = pipeline.analyze(args.video, DEFAULT, model)
    print(f"\n{os.path.basename(args.video)} — {res['meta']['duration_s']:.1f} s")
    _print_summary(res)
    if args.rapport:
        report.render(res, args.rapport)
        print(f"\n  rapport → {args.rapport}")
    if args.json:
        report.write_json(res, args.json)
        print(f"  données → {args.json}")
    return 0


def cmd_demo(args) -> int:
    from . import synthetic
    tmp = args.sortie_video or "demo_longevis.mp4"
    print("Génération d'une vidéo synthétique à vérité terrain connue…")
    gt = synthetic.make_video(tmp, duration_s=args.duree, hr_bpm=args.fc,
                              hrv_sd_ms=args.vfc)
    print(f"  FC injectée   {gt['true_hr_bpm']:.2f} bpm")
    print(f"  SDNN injecté  {gt['true_hrv_sdnn_ms']:.1f} ms")
    res = pipeline.analyze(tmp)
    print("\nAnalyse :")
    _print_summary(res)
    out = args.rapport or "rapport_demo.html"
    report.render(res, out, ground_truth=gt)
    print(f"\n  rapport → {out}")
    return 0


def cmd_mouvement(args) -> int:
    res = pipeline.analyze(args.video, mode="mouvement", task=args.tache,
                           subject_height_m=args.taille, px_per_m=args.echelle,
                           strict=not args.demo)
    f = res["features"]
    print(f"\n{os.path.basename(args.video)} — {res['meta']['duration_s']:.1f} s "
          f"— tâche : {res['meta'].get('task', '?')}")
    g = lambda k: f.get(k, float("nan"))
    if res["meta"].get("task") == "marche":
        print(f"  vitesse       {g('gait_speed_m_s'):.2f} m/s ({g('gait_speed_px_s'):.0f} px/s)")
        print(f"  cadence       {g('cadence_spm'):.1f} pas/min sur {g('n_steps'):.0f} pas")
        print(f"  longueur pas  {g('step_length_m'):.2f} m")
        print(f"  variabilité   {g('step_time_cv_pct'):.1f} %")
        print(f"  asymétrie     {g('step_asymmetry_pct'):.1f} % "
              f"(plancher {g('step_asymmetry_null_pct'):.1f} %, p={g('step_asymmetry_p'):.3f})")
        print(f"  demi-tours    {g('n_turns'):.0f}, {g('turn_mean_dur_s'):.2f} s en moyenne")
        print(f"  IRD           {g('ird_reserve_dynamique'):.2f} pas par demi-tour")
        print(f"  SCF           {g('scf_signature_foulee'):.3f}")
        print(f"  CAX           {g('cax_coherence'):.3f}")
        band = f.get("gait_speed_band")
        if band:
            print(f"  repère        {band}")
    else:
        for k in ("sway_rms_ap_mm", "sway_rms_ml_mm", "sway_path_mm_s",
                  "sts_count", "sts_mean_dur_s"):
            if np.isfinite(g(k)):
                print(f"  {k:16s} {g(k):.2f}")
    if not getattr(args, "demo", False):
        for k, d in res["indices"].get("unresolved", {}).items():
            print(f"  ! {k} non résolu (plancher {d['noise_floor']} > σ norme {d['norm_sd']})")
    if args.rapport:
        report.render(res, args.rapport, demo=getattr(args, "demo", False))
        print(f"\n  rapport → {args.rapport}")
    if args.json:
        report.write_json(res, args.json)
    return 0


def cmd_demo_marche(args) -> int:
    from . import synthetic_gait
    tmp = args.sortie_video or "demo_marche.mp4"
    print("Génération d'une marche synthétique à vérité terrain connue…")
    gt = synthetic_gait.make_walk_video(
        tmp, duration_s=args.duree, speed_m_s=args.vitesse, cadence_spm=args.cadence,
        asymmetry=args.asymetrie, step_cv_pct=args.variabilite, width=960, height=360)
    print(f"  vitesse injectée  {gt['true_speed_m_s']:.2f} m/s")
    print(f"  cadence injectée  {gt['true_cadence_spm']:.0f} pas/min")
    print(f"  asymétrie         {gt['true_step_asymmetry_pct']:.1f} %")
    res = pipeline.analyze(tmp, mode="mouvement", subject_height_m=gt["subject_height_m"])
    f = res["features"]
    print("\nMesuré :")
    print(f"  vitesse       {f.get('gait_speed_m_s', float('nan')):.2f} m/s")
    print(f"  cadence       {f.get('cadence_spm', float('nan')):.1f} pas/min")
    print(f"  asymétrie     {f.get('step_asymmetry_pct', float('nan')):.1f} % "
          f"(nette {f.get('step_asymmetry_net_pct', float('nan')):.1f} %)")
    out = args.rapport or "rapport_marche.html"
    report.render(res, out, ground_truth=gt, demo=args.demo)
    print(f"\n  rapport → {out}")
    return 0


def cmd_lot(args) -> int:
    files = sorted(p for p in glob.glob(os.path.join(args.dossier, "**", "*"), recursive=True)
                   if p.lower().endswith(VIDEO_EXT))
    if not files:
        print(f"Aucune vidéo dans {args.dossier}", file=sys.stderr)
        return 1
    rows = []
    for p in files:
        try:
            res = pipeline.analyze(p)
        except Exception as exc:                      # noqa: BLE001
            print(f"{os.path.basename(p)} : échec — {exc}", file=sys.stderr)
            continue
        row = {"file": os.path.basename(p), "subject_id": os.path.basename(os.path.dirname(p))}
        row.update({k: v for k, v in res["features"].items() if isinstance(v, (int, float))})
        row["quality_grade"] = res["indices"]["quality"]["grade"]
        row["composite_score"] = res["indices"]["composite_score"]
        rows.append(row)
        print(f"{os.path.basename(p):40s} {res['indices']['quality']['grade']}  "
              f"SNR {res['indices']['quality']['snr_db']:.1f} dB")
    if not rows:
        return 1
    keys = list(rows[0].keys())
    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} enregistrements → {args.csv}")
    return 0


def cmd_calibrer(args) -> int:
    from . import calibrate
    m = calibrate.fit_from_csv(args.csv, target=args.cible, group_col=args.groupe,
                               out_path=args.sortie, n_perm=args.permutations)
    print(f"Cible          {m['target']}")
    print(f"Enregistrements {m['n_train']}")
    print(f"MAE (VC)       {m['cv_mae']:.3f}")
    print(f"R² (VC)        {m['cv_r2']:.3f}")
    if m.get("p_value_permutation") is not None:
        p = m["p_value_permutation"]
        print(f"p (permutation) {p:.3f}" + ("  → non distinguable du hasard" if p > 0.05 else ""))
    if not m["grouped_cv"]:
        print("! Validation croisée non groupée : passez --groupe pour éviter la fuite entre "
              "enregistrements d'un même sujet.")
    print(f"modèle → {args.sortie}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="longevis",
                                description="Biomarqueurs de vieillissement par analyse vidéo.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyser", help="analyser une vidéo")
    a.add_argument("video")
    a.add_argument("--rapport", help="chemin du rapport HTML")
    a.add_argument("--json", help="chemin de l'export JSON")
    a.add_argument("--modele", help="modèle calibré (JSON)")
    a.set_defaults(func=cmd_analyser)

    d = sub.add_parser("demo", help="générer une vidéo de test et l'analyser")
    d.add_argument("--duree", type=float, default=90.0)
    d.add_argument("--fc", type=float, default=68.0)
    d.add_argument("--vfc", type=float, default=45.0)
    d.add_argument("--rapport")
    d.add_argument("--sortie-video", dest="sortie_video")
    d.set_defaults(func=cmd_demo)

    m = sub.add_parser("mouvement", help="analyser la marche, l'équilibre ou un transfert")
    m.add_argument("video")
    m.add_argument("--tache", default="auto", choices=["auto", "marche", "posture", "leve"])
    m.add_argument("--taille", type=float, help="stature du sujet en mètres")
    m.add_argument("--echelle", type=float, help="échelle connue en pixels par mètre")
    m.add_argument("--rapport")
    m.add_argument("--json")
    m.add_argument("--demo", action="store_true",
                   help="masque les marqueurs que la méthode ne résout pas")
    m.set_defaults(func=cmd_mouvement)

    dm = sub.add_parser("demo-marche", help="générer une marche de test et l'analyser")
    dm.add_argument("--duree", type=float, default=45.0)
    dm.add_argument("--vitesse", type=float, default=1.20)
    dm.add_argument("--cadence", type=float, default=110.0)
    dm.add_argument("--asymetrie", type=float, default=0.0)
    dm.add_argument("--variabilite", type=float, default=0.0)
    dm.add_argument("--rapport")
    dm.add_argument("--sortie-video", dest="sortie_video")
    dm.add_argument("--demo", action="store_true")
    dm.set_defaults(func=cmd_demo_marche)

    b = sub.add_parser("lot", help="analyser un dossier de vidéos")
    b.add_argument("dossier")
    b.add_argument("--csv", default="cohorte.csv")
    b.set_defaults(func=cmd_lot)

    c = sub.add_parser("calibrer", help="apprendre un modèle sur cohorte annotée")
    c.add_argument("csv")
    c.add_argument("--cible", required=True)
    c.add_argument("--groupe", help="colonne identifiant le sujet")
    c.add_argument("--sortie", default="modele.json")
    c.add_argument("--permutations", type=int, default=200)
    c.set_defaults(func=cmd_calibrer)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
