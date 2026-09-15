"""Rejeu vidéo + courbe pour les six gestes complémentaires à la marche.

Version allégée du rejeu principal (`hologramme.py`) : pas de musique ni de
biomarqueurs composites, seulement la vidéo et la courbe du signal mesuré,
synchronisées sur l'horloge du lecteur — pour voir, image par image, à quel
moment de la vidéo correspond quel chiffre.

Pensé pour être agréable à utiliser, pas seulement correct : la courbe est
nette à toute taille d'écran (résolution du canevas recalculée sur sa taille
réelle, pas figée), et on peut glisser le doigt ou la souris dessus pour
parcourir la vidéo image par image, sans repasser par les commandes natives.
"""

from __future__ import annotations
import json
from typing import Optional, Sequence

import numpy as np

from .hologramme import video_base64


def hauteur_composant(facteur: float = 1.0) -> int:
    """Hauteur à réserver pour `components.html` : vidéo + bandeau + courbe."""
    video_max = 480.0 * facteur
    courbe = min(220.0, max(90.0, 130.0 * facteur))
    return int(video_max + courbe + 96)


def rejeu_signal(chemin: str, signal: Sequence[float], fps: float,
                  unite: str, titre: str, largeur: int = 700,
                  facteur: float = 1.0) -> Optional[str]:
    """HTML autonome : vidéo au-dessus, courbe du signal en dessous, avec une
    ligne de lecture qui suit la vidéo et qu'on peut aussi glisser pour
    chercher un instant précis. Retourne None si la vidéo est trop lourde à
    incruster ou si le signal est inexploitable."""
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
    duree_s = float(t[-1]) if n else 0.0
    data = json.dumps({"t": t_s, "v": v_s, "unite": unite, "duree": round(duree_s, 2)})

    video_max = int(480 * facteur)
    hauteur_courbe = int(min(220, max(90, 130 * facteur)))

    return f"""
<style>
  html,body{{margin:0;padding:0;background:transparent}}
  *{{box-sizing:border-box}}
</style>
<div id="rjb-wrap" style="background:#07070C;border-radius:14px;overflow:hidden;
            font-family:'Inter Tight',system-ui,-apple-system,'Segoe UI',sans-serif;
            box-shadow:0 18px 50px rgba(0,0,0,.45)">
  <video id="rjb-v" src="data:video/mp4;base64,{b64}" controls playsinline
         preload="metadata"
         style="width:100%;display:block;background:#000;max-height:{video_max}px;
                object-fit:contain"></video>
  <div style="padding:12px 16px 18px">
    <div style="display:flex;justify-content:space-between;align-items:baseline;
                color:#a8b3c7;font-size:12.5px;margin-bottom:8px">
      <span>{titre}</span>
      <span id="rjb-val" style="color:#14D6C4;font-weight:600;font-size:15px;
            font-variant-numeric:tabular-nums"></span>
    </div>
    <div id="rjb-cwrap" style="position:relative;width:100%;height:{hauteur_courbe}px;
                cursor:pointer;border-radius:8px;overflow:hidden;touch-action:none">
      <canvas id="rjb-c" style="width:100%;height:100%;display:block"></canvas>
    </div>
    <div style="display:flex;justify-content:space-between;margin-top:6px;
                font-size:10.5px;color:#5C6270;letter-spacing:.01em">
      <span id="rjb-t0">0:00</span>
      <span style="opacity:.75">glissez sur la courbe pour parcourir la vidéo</span>
      <span id="rjb-t1">0:00</span>
    </div>
  </div>
</div>
<script>
(function() {{
  const data = {data};
  const v = document.getElementById('rjb-v');
  const wrap = document.getElementById('rjb-cwrap');
  const cv = document.getElementById('rjb-c');
  const ctx = cv.getContext('2d');
  const valEl = document.getElementById('rjb-val');
  const n = data.t.length;
  const tmax = data.t[n - 1] || data.duree || 1;
  const vals = data.v.filter(x => x !== null);
  const vmin = vals.length ? Math.min(...vals) : 0;
  const vmax = vals.length ? Math.max(...vals) : 1;
  const span = (vmax - vmin) || 1;
  const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));

  function fmt(t) {{
    t = Math.max(0, Math.round(t || 0));
    const m = Math.floor(t / 60), s = t % 60;
    return m + ':' + (s < 10 ? '0' : '') + s;
  }}
  document.getElementById('rjb-t1').textContent = fmt(tmax);

  let W = 0, H = 0;

  function xAt(i) {{ return (data.t[i] / tmax) * (W - 8) + 4; }}
  function yAt(val) {{ return H - 8 - ((val - vmin) / span) * (H - 16); }}

  function resize() {{
    const rect = wrap.getBoundingClientRect();
    W = Math.max(40, Math.round(rect.width));
    H = Math.max(30, Math.round(rect.height));
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }}

  function draw() {{
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(168,179,199,0.16)';
    ctx.lineWidth = 1;
    for (let g = 0; g <= 2; g++) {{
      const y = 8 + g * (H - 16) / 2;
      ctx.beginPath(); ctx.moveTo(4, y); ctx.lineTo(W - 4, y); ctx.stroke();
    }}
    ctx.strokeStyle = '#14D6C4';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
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
    ctx.fillStyle = '#ff8a3d';
    ctx.beginPath(); ctx.arc(px, 6, 3, 0, Math.PI * 2); ctx.fill();

    let closest = 0, bestD = Infinity;
    for (let i = 0; i < n; i++) {{
      const d = Math.abs(data.t[i] - v.currentTime);
      if (d < bestD) {{ bestD = d; closest = i; }}
    }}
    valEl.textContent = data.v[closest] !== null
      ? data.v[closest].toFixed(2) + ' ' + data.unite : '';
    document.getElementById('rjb-t0').textContent = fmt(v.currentTime || 0);
  }}

  function seekAt(clientX) {{
    const rect = wrap.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (clientX - rect.left - 4) / Math.max(1, rect.width - 8)));
    const t = frac * tmax;
    if (isFinite(t)) {{
      try {{ v.currentTime = t; }} catch (e) {{}}
      draw();
    }}
  }}

  let glisse = false;
  wrap.addEventListener('pointerdown', function(e) {{
    glisse = true;
    try {{ wrap.setPointerCapture(e.pointerId); }} catch (e) {{}}
    v.pause();
    seekAt(e.clientX);
  }});
  wrap.addEventListener('pointermove', function(e) {{ if (glisse) seekAt(e.clientX); }});
  wrap.addEventListener('pointerup', function() {{ glisse = false; }});
  wrap.addEventListener('pointercancel', function() {{ glisse = false; }});

  if (window.ResizeObserver) {{
    new ResizeObserver(resize).observe(wrap);
  }} else {{
    window.addEventListener('resize', resize);
  }}

  v.addEventListener('timeupdate', draw);
  v.addEventListener('loadedmetadata', resize);
  v.addEventListener('seeking', draw);
  resize();
}})();
</script>
"""
