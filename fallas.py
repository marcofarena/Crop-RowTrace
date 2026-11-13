# -*- coding: utf-8 -*-
"""
Fallas por hilera (tramos sin planta) sobre las hileras ya detectadas.

El análisis de cada cuartel es el de perfil.py y cuarteles.py (del complemento
Crop-RowTrace, github.com/marcofarena/Crop-RowTrace): transectas para el perfil
típico de la hilera, ancho de la vegetación, vigor a lo largo de cada hilera
normalizado entre la entrehilera (0) y una planta típica (1), y fallas donde
baja del umbral en un tramo continuo.

Acá se agrega:
1. El control de resolución. Con píxeles de decímetros el vigor de la hilera
   mezcla planta, sombra y entrehilera, y aparecen fallas que no existen: el
   mismo vuelo de dron llevado a 0,4 m marcó el doble de fallas y solo el 39%
   coincidía con las del vuelo original (a 0,1 m, el 89%). El tamaño de píxel
   no alcanza para saberlo: las imágenes de Google exportadas a 0,125 m tienen
   un detalle real de 0,3-0,4 m. Por eso se mide el detalle real (ver
   fine_detail).
2. Correr el análisis lote por lote (cada lote con su rumbo) y devolver las
   fallas con el lote y el número de hilera del complemento.
"""
import numpy as np

from .cuarteles import infer_blocks
from .perfil import Tile, analyze_block

# Las fallas se buscan con un píxel de menos de esto (m).
FAIL_MAX_PX = 0.2
# Escala del control de detalle (m) y parte mínima del detalle de la imagen
# (escalas < 1,5 m) que tiene que perderse al llevarla a esa escala: si no se
# pierde casi nada, la imagen no tiene detalle más fino que 0,2 m aunque su
# píxel sea más chico. Medido en ventanas de 80 m: vuelo de dron a 0,05 m,
# 6,5%; el mismo llevado a 0,1 m, 3,8%; las 32 imágenes de Google de prueba
# (0,125 m), 0-2%. La escala de 0,2 m es la que mejor los separa (a 0,3 m
# Google ya pierde hasta 4%).
FAIL_DETAIL_S = 0.2
FAIL_MIN_DETAIL = 0.03
FAIL_WINDOW_M = 80.0     # lado de cada ventana de la medición
# Brillo (en 0-255) debajo del cual un píxel es sombra y no se muestrea:
# ahí los cocientes entre bandas son ruido (perfil.Tile).
SHADOW_255 = 30.0
# Contraste mínimo del perfil transversal (pico de la hilera sobre la
# entrehilera, en desvíos): por debajo, las líneas no siguen las plantas.
FAIL_MIN_CONTRAST = 0.8


def detail_loss(gray, px, s=FAIL_DETAIL_S):
    """Parte del detalle de gray (escalas < 1,5 m) que se pierde al llevarla a
    un píxel de s metros y volver."""
    import cv2
    g = gray.astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.5 / px)
    e0 = float((hp ** 2).mean())
    if e0 <= 0:
        return 0.0
    h, w = g.shape
    lo = cv2.resize(g, (max(8, int(round(w * px / s))), max(8, int(round(h * px / s)))),
                    interpolation=cv2.INTER_AREA)
    up = cv2.resize(lo, (w, h), interpolation=cv2.INTER_CUBIC)
    return float(((g - up) ** 2).mean()) / e0


def fine_detail(r, g, b, valid, px, centers_px, max_windows=5):
    """
    Detalle real de la imagen alrededor de las hileras. centers_px: puntos
    (columna, fila) donde medir (p. ej. el centro de algunas hileras).
    Devuelve (ok, pérdida mediana o None, motivo).
    """
    # con el píxel de 0,2 m o más no hay detalle más fino que medir (el
    # mismo vuelo llevado a 0,2 m: solo el 68% de sus fallas coincidía)
    if px >= FAIL_MAX_PX * 0.999:
        return False, None, (f"el píxel es de {px:.3g} m y las fallas necesitan un píxel de "
                             f"menos de {FAIL_MAX_PX:g} m (imagen de dron)")
    H, W = valid.shape
    n = int(min(FAIL_WINDOW_M / px, H, W))
    if n < 32:
        return False, None, "la imagen es demasiado chica para medir su detalle"
    losses = []
    for cx, cy in centers_px[:max_windows]:
        c0 = int(min(max(cx - n // 2, 0), W - n))
        r0 = int(min(max(cy - n // 2, 0), H - n))
        v = valid[r0:r0 + n, c0:c0 + n]
        if v.mean() < 0.8:
            continue
        gray = (r[r0:r0 + n, c0:c0 + n] + g[r0:r0 + n, c0:c0 + n] + b[r0:r0 + n, c0:c0 + n]) / 3.0
        gray = np.where(v, gray, float(gray[v].mean()))
        losses.append(detail_loss(gray, px))
    if not losses:
        return False, None, "no hay una zona con datos suficientes para medir el detalle de la imagen"
    loss = float(np.median(losses))
    if loss < FAIL_MIN_DETAIL:
        return False, loss, (f"el píxel es de {px:.3g} m pero el detalle real de la imagen es más grueso que "
                             f"{FAIL_DETAIL_S:g} m (al llevarla a {FAIL_DETAIL_S:g} m pierde solo el "
                             f"{100 * loss:.0f}% del detalle; un vuelo de dron pierde 4% o más): es una imagen "
                             "ampliada, típica de satélite")
    return True, loss, ""


def dn_scale(r, g, b, valid, n=200000):
    """Escala de los valores de la imagen respecto de 0-255: 1 en 8 bits, 257
    en 16 bits, 1/255 en reflectancia 0-1 (por el percentil 99,9 del canal
    más brillante en una muestra de los píxeles válidos)."""
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return 1.0
    k = np.linspace(0, ys.size - 1, min(n, ys.size)).astype(np.int64)
    hi = float(np.percentile(np.maximum(np.maximum(r[ys[k], xs[k]], g[ys[k], xs[k]]), b[ys[k], xs[k]]), 99.9))
    if hi <= 1.5:
        return 1.0 / 255.0
    return 1.0 if hi <= 255.0 else 257.0


def _tile(r, g, b, valid, gt, ring, margin=3.0):
    """Recorte de las bandas (ya en memoria) alrededor de un cuartel."""
    xs = [q[0] for q in ring]
    ys = [q[1] for q in ring]
    H, W = valid.shape
    c0 = max(0, int((min(xs) - margin - gt[0]) / gt[1]))
    c1 = min(W, int((max(xs) + margin - gt[0]) / gt[1]) + 1)
    r0 = max(0, int((max(ys) + margin - gt[3]) / gt[5]))
    r1 = min(H, int((min(ys) - margin - gt[3]) / gt[5]) + 1)
    if c1 <= c0 or r1 <= r0:
        return None
    sl = (slice(r0, r1), slice(c0, c1))
    return Tile(r[sl], g[sl], b[sl], valid[sl], gt[0] + c0 * gt[1], gt[3] + r0 * gt[5], gt[1], -gt[5])


def failures_by_lot(segs, lots, rows, r, g, b, valid, gt, umbral=0.5, min_falla=2.0,
                    log=print, warn=print, canceled=lambda: False):
    """
    Fallas de las hileras detectadas. segs: [(x1, y1, x2, y2)] en coordenadas
    de mapa; lots y rows: lote y número de hilera de cada línea (o None).
    Devuelve una lista de dicts (lote, hilera, ini_m, largo_m, vigor, borde,
    x1, y1, x2, y2).
    """
    # Píxeles de sombra (brillo < 30 en 0-255) fuera del muestreo: el umbral
    # se lleva a la escala de la imagen (8 bits, 16 bits o reflectancia 0-1).
    shadow = SHADOW_255 * dn_scale(r, g, b, valid)
    groups = {}
    for i, s in enumerate(segs):
        groups.setdefault(lots[i] if lots is not None else None, []).append(i)
    out = []
    for lot, idx in sorted(groups.items(), key=lambda t: (t[0] is None, t[0] or 0)):
        if canceled():
            break
        name = "" if lot is None else f"Lote {lot}: "
        sg = [((segs[i][0], segs[i][1]), (segs[i][2], segs[i][3])) for i in idx]
        blocks, info = infer_blocks(sg, gap=3.0)
        if not blocks:
            warn(f"{name}sin fallas: muy pocas hileras para analizar (hacen falta 8 o más).")
            continue
        for nb, blk in enumerate(blocks, 1):
            if canceled():
                break
            tile = _tile(r, g, b, valid, gt, blk["ring"])
            if tile is None:
                continue
            res = analyze_block(blk, info, tile, step=10.0, win_m=1.0, umbral=umbral, index="auto",
                                min_falla=min_falla, adjust=True, n_breaks=4, kink_deg=3.0,
                                shadow_max=shadow)
            part = f" (parte {nb})" if len(blocks) > 1 else ""
            if "error" in res:
                warn(f"{name}sin fallas{part}: {res['error']}.")
                continue
            if res["contrast"] < FAIL_MIN_CONTRAST:
                warn(f"{name}sin fallas{part}: el perfil de la hilera casi no se distingue de la entrehilera "
                     f"(contraste {res['contrast']:.2f}); las líneas no siguen bien las plantas.")
                continue
            for fa in res["fallas"]:
                hs = [rows[idx[j]] for j in blk["rows"][fa["row"]]] if rows is not None else []
                hs = [h for h in hs if h is not None]
                out.append({"lote": lot, "hilera": max(set(hs), key=hs.count) if hs else None,
                            "ini_m": fa["ini_m"], "largo_m": fa["largo"], "vigor": fa["z"],
                            "borde": fa["borde"], "x1": fa["x1"], "y1": fa["y1"],
                            "x2": fa["x2"], "y2": fa["y2"]})
            log(f"{name}{len(res['fallas'])} fallas{part}, {res['falla_pct']:.1f}% del largo de las hileras "
                f"(índice {res['index']}, vegetación de {res['width']:.2f} m de ancho).")
    return out
