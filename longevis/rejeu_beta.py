"""Rejeu holographique pour les six gestes complémentaires à la marche.

Même langage visuel que le rejeu principal (`hologramme.py`) : la vidéo est
incrustée d'un repère qui suit, en direct, le point du corps mesuré —
position ou étendue selon le geste — et de cartes de mesures qui montent à
leur valeur pendant la lecture. Pas de partition musicale ici (elle est
propre au caractère de la marche dans `hologramme.py`), mais le même souci :
que l'écran parle par l'image, pas par un graphique de chiffres tout seul.

Le repère se pose sur les coordonnées réelles du corps dans l'image (centre
de masse, tronc ou pied suivi, étendue de la silhouette), normalisées par la
taille de l'image — pas sur une position arbitraire déduite du graphique.
"""

from __future__ import annotations
import base64
import json
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .hologramme import video_base64


def hauteur_composant_geste(facteur: float = 1.0) -> int:
    """Hauteur à réserver pour `components.html` : vidéo + bandeau de cartes."""
    return int(480.0 * facteur + 130)


def _axes_for(signal_cle: str) -> Tuple[str, str, str, Optional[str]]:
    """(genre, clé x, clé y, clé d'étendue) pour poser le repère sur le corps.

    « point » : un repère ponctuel (position). « extent » : une étendue
    verticale (taille de la silhouette), centrée sur le centre de masse.
    """
    return {
        "trunk_y":   ("point",  "cx", "trunk_y", None),
        "foot_y":    ("point",  "cx", "foot_y",  None),
        "cx":        ("point",  "cx", "cy",      None),
        "cy":        ("point",  "cx", "cy",      None),
        "height_px": ("extent", "cx", "cy",      "height_px"),
        "height":    ("extent", "cx", "cy",      "height"),
        "spread":    ("extent", "cx", "cy",      "spread"),
    }.get(signal_cle, ("point", "cx", "cy", None))


def _serie_geste(signaux: Dict[str, object], signal_cle: str, fps: float,
                 duree: float, frame_size, pas_par_s: float = 12.0):
    """Échantillonne la position du repère et la valeur du signal affiché,
    à un rythme fixe — assez pour un mouvement fluide, assez peu pour ne pas
    alourdir la page avec des milliers de points."""
    genre, xk, yk, hk = _axes_for(signal_cle)
    val = np.asarray(signaux.get(signal_cle, []), dtype=float)
    x = np.asarray(signaux.get(xk, []), dtype=float)
    y = np.asarray(signaux.get(yk, []), dtype=float) if yk else np.array([])
    hh = np.asarray(signaux.get(hk, []), dtype=float) if hk else np.array([])
    n = val.size
    if n < 4:
        return [], genre

    fs = frame_size if frame_size else (0, 0)
    lw = float(fs[0]) if fs and fs[0] else 0.0
    lh = float(fs[1]) if fs and len(fs) > 1 and fs[1] else 0.0
    if not lw:
        lw = float(np.nanmax(x)) * 1.25 if x.size and np.isfinite(x).any() else 1.0
    if not lh:
        if y.size and np.isfinite(y).any():
            lh = float(np.nanmax(y)) * 1.4
        elif hh.size and np.isfinite(hh).any():
            lh = float(np.nanmax(hh)) * 1.6
        else:
            lh = lw * 1.4
    lw = lw or 1.0
    lh = lh or 1.0

    sortie = []
    for i in range(int(duree * pas_par_s) + 1):
        ti = i / pas_par_s
        j = int(min(n - 1, round(ti * fps)))
        p = {"t": round(ti, 2)}
        vv = val[j] if j < val.size else float("nan")
        p["v"] = round(float(vv), 2) if np.isfinite(vv) else None
        if x.size:
            jx = min(j, x.size - 1)
            if np.isfinite(x[jx]):
                p["x"] = round(float(np.clip(x[jx] / lw, 0.0, 1.0)), 4)
        if y.size:
            jy = min(j, y.size - 1)
            if np.isfinite(y[jy]):
                p["y"] = round(float(np.clip(y[jy] / lh, 0.0, 1.0)), 4)
        if genre == "extent" and hh.size:
            jh = min(j, hh.size - 1)
            if np.isfinite(hh[jh]):
                p["h"] = round(float(np.clip(hh[jh] / lh, 0.03, 1.4)), 4)
        sortie.append(p)
    return sortie, genre


def _frames_base64(chemin: str, n: int = 14, largeur: int = 440) -> Optional[List[str]]:
    """Secours quand la vidéo dépasse `hologramme.MAX_MO` : échantillonne
    `n` images réduites au lieu d'abandonner l'incrustation.

    Les six gestes partagent une seule vidéo, filmée d'affilée — bien plus
    longue qu'un unique test de marche, elle dépasse vite le plafond
    d'encodage. Un ré-encodage vidéo côté serveur réglerait la taille, mais
    aucun hébergement gratuit ne garantit ffmpeg (voir `hologramme.py`) ; ces
    quelques images fixes, elles, ne dépendent que d'OpenCV, déjà utilisé
    partout ailleurs dans le pipeline."""
    cap = cv2.VideoCapture(chemin)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return None
    images: List[str] = []
    for idx in np.linspace(0, total - 1, num=min(n, total), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > largeur:
            frame = cv2.resize(frame, (largeur, max(1, int(h * largeur / w))))
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 62])
        if ok2:
            images.append(base64.b64encode(buf).decode("ascii"))
    cap.release()
    return images or None


def rejeu_geste(chemin: str, signaux: Dict[str, object], meta: Dict[str, object],
                signal_cle: str, titre: str, unite: str,
                metriques: Sequence[tuple], facteur: float = 1.0) -> Optional[str]:
    """Retourne le lecteur holographique complet, ou None si la vidéo ne peut
    pas être incrustée ou si aucune position n'a pu être suivie.

    `metriques` : liste de tuples (nom, valeur, unité, décimales) — les
    mêmes que celles déjà affichées en cartes statiques au-dessus, réutilisées
    ici pour l'incrustation animée.

    Vidéo dans la limite d'`hologramme.MAX_MO` : rejeu vidéo complet, comme
    pour la marche. Au-delà (attendu ici, une seule vidéo couvrant les six
    gestes est bien plus longue qu'un test de marche), rejeu par images clés
    échantillonnées — l'incrustation reste disponible plutôt que d'échouer."""
    b64 = video_base64(chemin)
    frames = None if b64 is not None else _frames_base64(chemin)
    if b64 is None and frames is None:
        return None

    fps = float(signaux.get("fps") or 25.0)
    duree = float(meta.get("duration_s") or 0.0) or 1.0
    frame_size = meta.get("frame_size") or (0, 0)
    serie, genre = _serie_geste(signaux, signal_cle, fps, duree, frame_size)
    if not serie:
        return None

    vals = [p["v"] for p in serie if p.get("v") is not None]
    vmin = min(vals) if vals else 0.0
    vmax = max(vals) if vals else 1.0

    cartes = []
    for (nom, val, u, dec) in metriques:
        if isinstance(val, (int, float)) and np.isfinite(val):
            cartes.append({"nom": nom, "val": round(float(val), dec),
                           "unite": u, "dec": dec})

    donnees = json.dumps({"support": "video" if b64 is not None else "frames",
                          "serie": serie, "genre": genre, "unite": unite,
                          "vmin": round(float(vmin), 3), "vmax": round(float(vmax), 3),
                          "cartes": cartes, "duree": round(duree, 2),
                          "frames": frames or []})

    video_max = int(480 * facteur)
    return (_GABARIT_GESTE.replace("__B64__", b64 or "").replace("__DATA__", donnees)
            .replace("__TITRE__", titre).replace("__VMAX__", str(video_max)))


_GABARIT_GESTE = """
<style>
html,body{margin:0;padding:0;background:transparent}
*{box-sizing:border-box}
.rjg-scene{position:relative;border-radius:18px;overflow:hidden;background:#07070C;
 font-family:'Inter Tight',-apple-system,'Segoe UI',sans-serif;
 box-shadow:0 20px 60px rgba(0,0,0,.5)}
.rjg-scene video,.rjg-scene img{display:block;width:100%;filter:saturate(.85) contrast(1.05);
 background:#000;max-height:__VMAX__px;object-fit:contain}
.rjg-veil{position:absolute;inset:0;pointer-events:none;
 background:radial-gradient(120% 90% at 50% 40%,transparent 35%,rgba(4,6,14,.68) 100%)}
.rjg-scan{position:absolute;inset:0;pointer-events:none;opacity:.18;
 background:repeating-linear-gradient(180deg,rgba(180,230,255,.14) 0 1px,transparent 1px 4px)}
.rjg-grille{position:absolute;inset:0;pointer-events:none;opacity:.14;
 background:linear-gradient(90deg,rgba(140,220,255,.5) 1px,transparent 1px) 0 0/72px 100%,
            linear-gradient(180deg,rgba(140,220,255,.5) 1px,transparent 1px) 0 0/100% 72px}
.rjg-coin{position:absolute;width:22px;height:22px;border:2px solid rgba(140,220,255,.5);
 pointer-events:none}
.rjg-coin.tl{left:12px;top:12px;border-right:0;border-bottom:0}
.rjg-coin.tr{right:12px;top:12px;border-left:0;border-bottom:0}
.rjg-coin.bl{left:12px;bottom:12px;border-right:0;border-top:0}
.rjg-coin.br{right:12px;bottom:12px;border-left:0;border-top:0}
.rjg-marker{position:absolute;pointer-events:none;display:none;
 transition:left .08s linear,top .08s linear,width .12s linear,height .12s linear}
.rjg-marker i{position:absolute;width:14px;height:14px;border:2px solid rgba(140,220,255,.9);
 filter:drop-shadow(0 0 8px rgba(120,220,255,.7))}
.rjg-marker i:nth-child(1){left:0;top:0;border-right:0;border-bottom:0}
.rjg-marker i:nth-child(2){right:0;top:0;border-left:0;border-bottom:0}
.rjg-marker i:nth-child(3){left:0;bottom:0;border-right:0;border-top:0}
.rjg-marker i:nth-child(4){right:0;bottom:0;border-left:0;border-top:0}
.rjg-halo{position:absolute;left:50%;top:50%;width:34px;height:34px;margin:-17px 0 0 -17px;
 border-radius:50%;border:1px solid rgba(20,214,196,.55);
 box-shadow:0 0 16px rgba(20,214,196,.32) inset,0 0 14px rgba(20,214,196,.24);
 animation:rjgpulse 2.2s ease-in-out infinite}
@keyframes rjgpulse{0%,100%{transform:scale(.9);opacity:.55}50%{transform:scale(1.1);opacity:1}}
.rjg-etiq{position:absolute;left:calc(100% + 10px);top:-4px;white-space:nowrap;
 background:rgba(8,12,22,.55);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
 border:1px solid rgba(255,255,255,.16);border-left:2px solid #14D6C4;border-radius:8px;
 padding:6px 11px;font-size:clamp(10px,1.1vw,14px);letter-spacing:.15em;text-transform:uppercase;
 color:#CFE4FF}
.rjg-etiq b{display:block;font-size:clamp(18px,2.2vw,32px);font-weight:200;line-height:1.05;
 letter-spacing:-.03em;color:#fff;text-shadow:0 0 10px rgba(255,255,255,.28),
 0 0 28px rgba(140,220,255,.4);font-variant-numeric:tabular-nums;text-transform:none}
.rjg-inst{position:absolute;top:14px;left:16px;font-size:clamp(9px,1vw,13px);letter-spacing:.22em;
 text-transform:uppercase;color:#F97316;border:1px solid rgba(249,115,22,.5);border-radius:99px;
 padding:3px 10px;pointer-events:none;z-index:3}
.rjg-plein{position:absolute;top:12px;right:14px;z-index:6;background:rgba(8,12,22,.5);
 backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,.18);
 color:#CFE4FF;border-radius:9px;padding:5px 10px;font-size:11px;letter-spacing:.1em;
 text-transform:uppercase;cursor:pointer;font-family:inherit}
.rjg-plein:hover{border-color:rgba(20,214,196,.7);color:#fff}
.rjg-scene.plein{position:fixed;inset:0;z-index:99999;border-radius:0;display:flex;
 align-items:center;justify-content:center;background:#04060C}
.rjg-scene.plein video,.rjg-scene.plein img{max-height:100vh;max-width:100vw;width:auto;height:100vh}
.rjg-scene:fullscreen{display:flex;align-items:center;justify-content:center;background:#04060C}
.rjg-scene:fullscreen video,.rjg-scene:fullscreen img{max-height:100vh;width:auto;height:100vh}
.rjg-hud{position:absolute;left:0;right:0;bottom:0;padding:12px 14px 14px;display:flex;
 gap:8px;flex-wrap:wrap;align-items:flex-end;pointer-events:none;z-index:3}
.rjg-card{flex:1 1 130px;min-width:118px;border-radius:12px;padding:9px 12px 11px;
 background:rgba(8,12,22,.44);backdrop-filter:blur(12px) saturate(140%);
 -webkit-backdrop-filter:blur(12px) saturate(140%);border:1px solid rgba(255,255,255,.14);
 border-top:2px solid #14D6C4}
.rjg-lab{font-size:clamp(10px,1.1vw,15px);letter-spacing:.18em;text-transform:uppercase;
 color:#B6C8E4;font-weight:500;margin:0}
.rjg-val{font-size:clamp(24px,3.6vw,52px);font-weight:200;letter-spacing:-.04em;color:#fff;
 line-height:1;font-variant-numeric:tabular-nums;margin:3px 0 0}
.rjg-val small{font-size:clamp(10px,1vw,15px);color:#9BB0D0;margin-left:4px;letter-spacing:0}
</style>
<div class="rjg-scene" id="rjgs">
  <video id="rjgv" src="data:video/mp4;base64,__B64__" controls playsinline
         preload="metadata"></video>
  <img id="rjgimg" style="display:none" alt="__TITRE__">
  <div class="rjg-veil"></div><div class="rjg-scan"></div><div class="rjg-grille"></div>
  <span class="rjg-coin tl"></span><span class="rjg-coin tr"></span>
  <span class="rjg-coin bl"></span><span class="rjg-coin br"></span>
  <div class="rjg-marker" id="rjgm">
    <i></i><i></i><i></i><i></i>
    <div class="rjg-halo"></div>
    <div class="rjg-etiq"><span>__TITRE__</span><b id="rjgval">–</b></div>
  </div>
  <div class="rjg-inst">LongeVis · __TITRE__</div>
  <button class="rjg-plein" id="rjgplein" type="button">⛶ agrandir</button>
  <div class="rjg-hud" id="rjghud"></div>
</div>
<script>
(function() {
  var D = __DATA__;
  var v = document.getElementById('rjgv');
  var img = document.getElementById('rjgimg');
  var marker = document.getElementById('rjgm');
  var valEl = document.getElementById('rjgval');
  var hud = document.getElementById('rjghud');
  var genre = D.genre;

  D.cartes.forEach(function(c, i) {
    var d = document.createElement('div');
    d.className = 'rjg-card';
    d.innerHTML = '<p class="rjg-lab">' + c.nom + '</p>' +
                  '<p class="rjg-val"><span id="rjgc' + i + '">0</span>' +
                  '<small>' + c.unite + '</small></p>';
    hud.appendChild(d);
  });

  var monte = null;
  function anime() {
    var t0 = performance.now();
    cancelAnimationFrame(monte);
    (function pas() {
      var k = Math.min(1, (performance.now() - t0) / 1600);
      var e = 1 - Math.pow(1 - k, 3);
      D.cartes.forEach(function(c, i) {
        var el = document.getElementById('rjgc' + i);
        if (el) el.textContent = (c.val * e).toFixed(c.dec);
      });
      if (k < 1) monte = requestAnimationFrame(pas);
    })();
  }

  function poser(p) {
    if (!marker) return;
    if (!p || p.x === undefined) { marker.style.display = 'none'; return; }
    marker.style.display = 'block';
    var y = (p.y === undefined ? 0.5 : p.y);
    var h = (genre === 'extent') ? Math.max(0.05, p.h || 0.3) : 0.10;
    var w = 0.09;
    marker.style.left = (100 * (p.x - w / 2)).toFixed(2) + '%';
    marker.style.top = (100 * (y - h / 2)).toFixed(2) + '%';
    marker.style.width = (100 * w).toFixed(2) + '%';
    marker.style.height = (100 * h).toFixed(2) + '%';
    var vmax = D.vmax, vmin = D.vmin;
    var frac = (vmax > vmin && p.v !== null) ? (p.v - vmin) / (vmax - vmin) : 0.5;
    frac = Math.max(0, Math.min(1, frac));
    var teinte = frac < 0.33 ? '122,162,255' : (frac < 0.66 ? '20,214,196' : '249,115,22');
    var eq = marker.querySelectorAll('i');
    for (var q = 0; q < eq.length; q++) eq[q].style.borderColor = 'rgba(' + teinte + ',.95)';
    var halo = marker.querySelector('.rjg-halo');
    if (halo) {
      halo.style.borderColor = 'rgba(' + teinte + ',.6)';
      halo.style.boxShadow = '0 0 16px rgba(' + teinte + ',.32) inset,0 0 14px rgba(' + teinte + ',.26)';
    }
    if (valEl) valEl.textContent = (p.v === null || p.v === undefined) ? '' : (p.v.toFixed(1) + ' ' + D.unite);
  }

  function suit() {
    var t = v.currentTime, s = D.serie;
    if (!s.length) return;
    var i = Math.min(s.length - 1, Math.max(0, Math.round(t * (s.length - 1) / Math.max(0.01, D.duree))));
    poser(s[i]);
  }

  if (D.support === 'frames' && D.frames && D.frames.length) {
    // Vidéo trop lourde pour être incrustée telle quelle (voir _frames_base64) :
    // un aperçu par images clés remplace la lecture vidéo, mais le repère et
    // les cartes suivent le même chemin que pour un rejeu vidéo complet.
    v.style.display = 'none';
    img.style.display = 'block';
    var trames = D.frames, nT = trames.length;
    var s = D.serie;
    img.src = 'data:image/jpeg;base64,' + trames[0];
    var inst = document.querySelector('.rjg-inst');
    if (inst) inst.textContent += ' · aperçu par images clés';
    var depart = null;
    var dureeMs = Math.max(1000, D.duree * 1000);
    anime();
    (function boucle(ts) {
      if (depart === null) depart = ts;
      var frac = ((ts - depart) % dureeMs) / dureeMs;
      var fi = Math.min(nT - 1, Math.floor(frac * nT));
      img.src = 'data:image/jpeg;base64,' + trames[fi];
      if (s.length) {
        var i = Math.min(s.length - 1, Math.max(0, Math.round(frac * (s.length - 1))));
        poser(s[i]);
      }
      requestAnimationFrame(boucle);
    })();
  } else {
    v.addEventListener('play', anime);
    v.addEventListener('timeupdate', suit);
    v.addEventListener('seeking', suit);
    v.addEventListener('loadeddata', function() { anime(); suit(); });
  }

  var scene = document.getElementById('rjgs');
  var btn = document.getElementById('rjgplein');
  function bascule() {
    var plein = scene.classList.contains('plein') || document.fullscreenElement;
    if (plein) {
      if (document.exitFullscreen && document.fullscreenElement) document.exitFullscreen();
      scene.classList.remove('plein');
      btn.textContent = '⛶ agrandir';
    } else {
      try {
        if (scene.requestFullscreen) scene.requestFullscreen().catch(function() {
          scene.classList.add('plein');
        });
        else scene.classList.add('plein');
      } catch (e) { scene.classList.add('plein'); }
      scene.classList.add('plein');
      btn.textContent = '⛶ réduire';
    }
  }
  if (btn) btn.addEventListener('click', bascule);
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      scene.classList.remove('plein');
      if (btn) btn.textContent = '⛶ agrandir';
    }
  });
})();
</script>
"""
