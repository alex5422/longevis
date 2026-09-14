"""Rejeu vidéo + courbe pour les tests tonus, sollicitation et élasticité.

Version allégée du rejeu principal (`hologramme.py`) : pas de musique ni de
biomarqueurs composites, seulement la vidéo et la courbe du signal mesuré,
synchronisées sur l'horloge du lecteur — pour voir, image par image, à quel
moment de la vidéo correspond quel chiffre.
"""

from __future__ import annotations
import json
from typing import Optional, Sequence

import numpy as np

from .hologramme import video_base64


def rejeu_signal(chemin: str, signal: Sequence[float], fps: float,
                  unite: str, titre: str, largeur: int = 700) -> Optional[str]:
    """HTML autonome : vidéo à gauche/au-dessus, courbe du signal en dessous,
    avec une ligne de lecture qui suit la vidéo. Retourne None si la vidéo est
    trop lourde à incruster ou si le signal est inexploitable."""
    b64 = video_base64(chemin)
    if b64 is None:
        return None
    sig = np.asarray(signal, dtype=float)
    if sig.size < 2 or not np.isfinite(sig).any():
        return None

    n = sig.size
    t = np.arange(n) / max(1e-6, fps)
    # sous-échantillonnage à ~200 points : la courbe reste lisible sans
    # alourdir la page avec des milliers de points
    if n > 200:
        idx = np.linspace(0, n - 1, 200).astype(int)
    else:
        idx = np.arange(n)
    t_s = [round(float(x), 2) for x in t[idx]]
    v_s = [round(float(x), 3) if np.isfinite(x) else None for x in sig[idx]]
    data = json.dumps({"t": t_s, "v": v_s, "unite": unite})
    hauteur_courbe = 130

    return f"""
<div style="background:#07070C;border-radius:14px;overflow:hidden;
            font-family:system-ui,-apple-system,sans-serif">
  <video id="rjb-v" src="data:video/mp4;base64,{b64}" controls playsinline
         style="width:100%;display:block;background:#000;max-height:480px;
                object-fit:contain"></video>
  <div style="padding:10px 14px 16px">
    <div style="display:flex;justify-content:space-between;align-items:baseline;
                color:#a8b3c7;font-size:12px;margin-bottom:6px">
      <span>{titre}</span>
      <span id="rjb-val" style="color:#14D6C4;font-weight:600;font-size:14px"></span>
    </div>
    <canvas id="rjb-c" width="{largeur}" height="{hauteur_courbe}"
            style="width:100%;height:{hauteur_courbe}px;display:block"></canvas>
  </div>
</div>
<script>
(function() {{
  const data = {data};
  const v = document.getElementById('rjb-v');
  const cv = document.getElementById('rjb-c');
  const ctx = cv.getContext('2d');
  const valEl = document.getElementById('rjb-val');
  const W = cv.width, H = cv.height;
  const n = data.t.length;
  const tmax = data.t[n - 1] || 1;
  const vals = data.v.filter(x => x !== null);
  const vmin = vals.length ? Math.min(...vals) : 0;
  const vmax = vals.length ? Math.max(...vals) : 1;
  const span = (vmax - vmin) || 1;

  function xAt(i) {{ return (data.t[i] / tmax) * (W - 8) + 4; }}
  function yAt(val) {{ return H - 8 - ((val - vmin) / span) * (H - 16); }}

  function draw() {{
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(168,179,199,0.18)';
    ctx.lineWidth = 1;
    for (let g = 0; g <= 2; g++) {{
      const y = 8 + g * (H - 16) / 2;
      ctx.beginPath(); ctx.moveTo(4, y); ctx.lineTo(W - 4, y); ctx.stroke();
    }}
    ctx.strokeStyle = '#14D6C4';
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < n; i++) {{
      if (data.v[i] === null) {{ started = false; continue; }}
      const x = xAt(i), y = yAt(data.v[i]);
      if (!started) {{ ctx.moveTo(x, y); started = true; }} else {{ ctx.lineTo(x, y); }}
    }}
    ctx.stroke();

    const frac = Math.min(1, Math.max(0, (v.currentTime || 0) / tmax));
    const px = frac * (W - 8) + 4;
    ctx.strokeStyle = '#ff8a3d';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(px, 2); ctx.lineTo(px, H - 2); ctx.stroke();

    let closest = 0, bestD = Infinity;
    for (let i = 0; i < n; i++) {{
      const d = Math.abs(data.t[i] - v.currentTime);
      if (d < bestD) {{ bestD = d; closest = i; }}
    }}
    valEl.textContent = data.v[closest] !== null
      ? data.v[closest].toFixed(2) + ' ' + data.unite : '';
  }}

  v.addEventListener('timeupdate', draw);
  v.addEventListener('loadedmetadata', draw);
  v.addEventListener('seeking', draw);
  draw();
}})();
</script>
"""
