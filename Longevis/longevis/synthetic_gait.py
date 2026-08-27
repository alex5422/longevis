"""Vidéo de marche synthétique à vérité terrain.

Un marcheur articulé traverse la scène : cadence, vitesse, asymétrie gauche/droite
et amplitude d'oscillation verticale sont imposées, donc vérifiables. Permet de
valider la chaîne de mesure du mouvement sans cohorte humaine.

Le modèle est cinématiquement grossier — il ne sert pas à imiter la marche, mais
à produire des signaux dont la périodicité et l'asymétrie sont exactement connues.
"""

from __future__ import annotations
from typing import Dict

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt


def _background(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """Fond texturé fixe : un aplat uniforme rendrait la soustraction trop facile."""
    bg = np.full((h, w, 3), 96.0, dtype=np.float32)
    bg += cv2.GaussianBlur(rng.normal(0, 16, (h, w)).astype(np.float32), (0, 0), 6)[..., None]
    floor = int(0.86 * h)
    bg[floor:, :] *= 0.72
    for x in range(0, w, 90):                       # repères muraux
        cv2.line(bg, (x, 0), (x, floor), (78, 78, 78), 2)
    cv2.line(bg, (0, floor), (w, floor), (60, 60, 60), 3)
    bg += rng.normal(0, 3, (h, w, 3))
    return np.clip(bg, 0, 255)


def _draw_walker(canvas: np.ndarray, cx: float, cy: float, height_px: float,
                 phase: float, asym: float, arm_swing: float) -> None:
    """Dessine une figure articulée. `phase` avance d'un cycle par foulée."""
    col = (28, 28, 30)
    h = height_px
    head_r = 0.075 * h
    hip = (cx, cy + 0.02 * h)
    shoulder = (cx, cy - 0.26 * h)

    # Tête et tronc
    cv2.circle(canvas, (int(cx), int(cy - 0.36 * h)), int(head_r), col, -1)
    cv2.line(canvas, (int(shoulder[0]), int(shoulder[1])),
             (int(hip[0]), int(hip[1])), col, max(3, int(0.055 * h)))

    leg_len = 0.46 * h
    for side, sign in ((0, 1.0), (1, -1.0)):
        # L'asymétrie allonge la phase d'appui d'un côté : le pied concerné
        # reste plus longtemps au sol, ce que la détection doit retrouver.
        p = phase + (0.0 if side == 0 else np.pi)
        hip_ang = 0.42 * np.sin(p)
        knee_ang = 0.55 * max(0.0, np.sin(p + 0.9))
        kx = hip[0] + leg_len * 0.52 * np.sin(hip_ang)
        ky = hip[1] + leg_len * 0.52 * np.cos(hip_ang)
        fx = kx + leg_len * 0.48 * np.sin(hip_ang - knee_ang)
        fy = ky + leg_len * 0.48 * np.cos(hip_ang - knee_ang)
        cv2.line(canvas, (int(hip[0]), int(hip[1])), (int(kx), int(ky)), col, max(3, int(0.048 * h)))
        cv2.line(canvas, (int(kx), int(ky)), (int(fx), int(fy)), col, max(3, int(0.042 * h)))
        cv2.line(canvas, (int(fx), int(fy)), (int(fx + sign * 0.06 * h), int(fy)), col,
                 max(2, int(0.030 * h)))

        # Bras en opposition de phase avec la jambe homolatérale
        a = p + np.pi
        ax = shoulder[0] + 0.30 * h * arm_swing * np.sin(a)
        ay = shoulder[1] + 0.30 * h * np.cos(0.35 * np.sin(a))
        cv2.line(canvas, (int(shoulder[0]), int(shoulder[1])), (int(ax), int(ay)),
                 col, max(2, int(0.036 * h)))


def make_walk_video(path: str, duration_s: float = 20.0, fps: float = 30.0,
                    cadence_spm: float = 108.0, speed_m_s: float = 1.25,
                    subject_height_m: float = 1.72, asymmetry: float = 0.0,
                    step_cv_pct: float = 0.0,
                    trunk_decoupling: float = 0.0,
                    sway_amp_px: float = 2.0, arm_swing: float = 1.0,
                    turn_dur_s: float = 1.2,
                    width: int = 640, height: int = 360, seed: int = 3) -> Dict[str, float]:
    """Écrit la vidéo et retourne la vérité terrain.

    `cadence_spm` est en pas par minute (deux pas = une foulée).
    `asymmetry` décale la phase du côté droit : 0 = symétrique, 0.2 = 20 % de décalage.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fps)
    t = np.arange(n) / fps

    person_h_px = 0.62 * height
    px_per_m = person_h_px / subject_height_m
    speed_px_s = speed_m_s * px_per_m

    # La phase est construite à partir d'instants de pas explicites : c'est le
    # seul moyen d'imposer une variabilité et une asymétrie exactement connues.
    # Un pas = une demi-foulée = pi de phase.
    step_hz = cadence_spm / 60.0
    stride_hz = step_hz / 2.0
    mean_step = 1.0 / step_hz
    n_steps = int(duration_s * step_hz) + 4
    jitter = rng.normal(0.0, step_cv_pct / 100.0 * mean_step, n_steps)
    intervals = np.full(n_steps, mean_step) + jitter
    # L'asymétrie allonge un pas sur deux et raccourcit l'autre, à cadence
    # moyenne constante.
    intervals[0::2] *= (1.0 + asymmetry / 2.0)
    intervals[1::2] *= (1.0 - asymmetry / 2.0)
    intervals = np.clip(intervals, 0.15, 3.0)
    step_times = np.concatenate([[0.0], np.cumsum(intervals)])
    phase = np.interp(t, step_times, np.pi * np.arange(len(step_times)))

    bg = _background(width, height, rng)
    floor_y = 0.86 * height

    # Oscillation verticale du tronc. `trunk_decoupling` remplace une fraction du
    # mouvement asservi aux jambes par un mouvement de même amplitude et de même
    # bande de fréquence, mais de phase indépendante. Sans ce paramètre, le tronc
    # et les jambes sont pilotés par la même phase : le couplage est parfait par
    # construction et AUCUNE dégradation de la marche ne peut le rompre — un test
    # de coordination sur une telle vidéo ne teste rien.
    couple = np.cos(2 * phase)
    d = float(np.clip(trunk_decoupling, 0.0, 1.0))
    if d > 0:
        libre = rng.normal(0, 1, len(t))
        nyq = fps / 2
        lo, hi = max(0.05, 1.4 * step_hz) / nyq, min(2.6 * step_hz / nyq, 0.95)
        if 0 < lo < hi < 1:
            bb, aa = butter(2, [lo, hi], btype="band")
            libre = filtfilt(bb, aa, libre)
        libre = libre / (np.std(libre) + 1e-9)
        bob_track = sway_amp_px * ((1 - d) * couple + d * libre)
    else:
        bob_track = sway_amp_px * couple

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise IOError("Encodeur vidéo indisponible.")

    # Trajectoire en va-et-vient : le sujet reste dans le cadre et effectue des
    # demi-tours. Le triangle est lissé pour que le demi-tour prenne un temps
    # fini — un changement de sens instantané n'existe pas et rendrait la
    # détection des demi-tours triviale.
    x_min, x_max = 0.12 * width, 0.88 * width
    span = x_max - x_min
    period = 2 * span / speed_px_s
    u = (t % period) / period
    x_track = x_min + span * np.where(u < 0.5, 2 * u, 2 * (1 - u))
    turn_sigma = max(1.0, turn_dur_s * fps / 4.0)
    x_track = gaussian_filter1d(x_track, turn_sigma, mode="nearest")
    for i in range(n):
        x = x_track[i]
        # Oscillation verticale du centre de masse : deux fois par foulée
        bob = bob_track[i]
        cy = floor_y - 0.52 * person_h_px + bob + rng.normal(0, 0.25)

        frame = bg.copy()
        _draw_walker(frame, x, cy, person_h_px, phase[i], asymmetry, arm_swing)
        frame = frame + rng.normal(0, 2.5, (height, width, 3))
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))

    writer.release()
    return {
        "path": path, "duration_s": duration_s, "fps": fps,
        "true_cadence_spm": float(cadence_spm),
        "true_stride_hz": float(stride_hz),
        "true_speed_m_s": float(speed_m_s),
        "true_asymmetry": float(asymmetry),
        "true_step_time_s": float(np.mean(intervals)),
        "true_step_time_cv_pct": float(100.0 * np.std(intervals, ddof=1) / np.mean(intervals)),
        "true_step_asymmetry_pct": float(
            abs(intervals[0::2].mean() - intervals[1::2].mean()) / intervals.mean() * 100.0),
        "true_px_per_m": float(px_per_m),
        "subject_height_m": float(subject_height_m),
        "true_bob_amp_px": float(sway_amp_px),
        "true_trunk_decoupling": float(trunk_decoupling),
        "true_turn_dur_s": float(turn_dur_s),
        "true_n_turns": float(max(0, int(2 * duration_s / period) - 1)),
    }


def make_stand_video(path: str, duration_s: float = 30.0, fps: float = 30.0,
                     sway_rms_px: float = 2.2, sway_hz: float = 0.45,
                     subject_height_m: float = 1.72,
                     width: int = 640, height: int = 360, seed: int = 5) -> Dict[str, float]:
    """Sujet debout immobile : valide la mesure d'oscillation posturale."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fps)
    t = np.arange(n) / fps
    bg = _background(width, height, rng)
    person_h_px = 0.62 * height
    floor_y = 0.86 * height

    # Oscillation lente + composante aléatoire, calibrée pour atteindre le RMS visé
    slow = np.sin(2 * np.pi * sway_hz * t) + 0.6 * np.sin(2 * np.pi * sway_hz * 0.37 * t)
    noise = np.cumsum(rng.normal(0, 0.08, n)); noise -= noise.mean()
    raw = slow + 0.5 * noise / (np.std(noise) + 1e-9)
    ap = raw / (np.std(raw) + 1e-9) * sway_rms_px
    ml = np.roll(ap, 37) * 0.7

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(n):
        frame = bg.copy()
        _draw_walker(frame, width / 2 + ml[i], floor_y - 0.52 * person_h_px + ap[i],
                     person_h_px, 0.0, 0.0, 0.05)
        frame = frame + rng.normal(0, 2.5, (height, width, 3))
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()
    return {
        "path": path, "duration_s": duration_s, "fps": fps,
        "true_sway_rms_px": float(np.sqrt(np.mean(ap ** 2))),
        "true_sway_ml_rms_px": float(np.sqrt(np.mean(ml ** 2))),
        "true_px_per_m": float(person_h_px / subject_height_m),
        "subject_height_m": float(subject_height_m),
    }
