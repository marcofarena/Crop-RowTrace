# -*- coding: utf-8 -*-
"""
Hileras de cultivo y fallas por cuartel, a partir del VERDE.

Proceso (cada etapa usa solo un índice de verdor; la sombra no se usa para
detectar nada, y los píxeles tan oscuros que el índice es puro ruido se
excluyen del muestreo):

  1. Delimitación de cuarteles: zonas con un patrón periódico de hileras en el
     índice de verdor (el verdor sin patrón, como monte o pasto, no es cuartel).
  2. Rumbo y distancia entre hileras de cada cuartel, por separado.
  3. Líneas iniciales: una recta por hilera, paralelas, con ese rumbo y esa
     distancia.
  4. Primer buffer alrededor de cada línea y perfiles transversales de verdor
     para ubicar el pico dentro del buffer.
  5. Líneas nuevas con un número limitado de quiebres leves, que ajustan los
     picos mejor que la línea inicial (se verifica).
  6. Segundo buffer alrededor de las líneas ajustadas: es el que decide dónde
     hay fallas.

NDVI no hay en un ortomosaico RGB; si el raster tiene una banda infrarroja se
puede pasar (nir) y se usa NDVI.
"""
import math

import numpy as np

GREEN_INDICES = ["ExG", "VARI", "GLI", "NGRDI"]
DARK_MAX = 30.0          # brillo medio (0-255) por debajo del cual el índice es ruido


# ---------------------------------------------------------------------------
# Índices de verdor
# ---------------------------------------------------------------------------
def green_index(name, R, G, B, nir=None):
    """Índice con valores altos = más vegetación."""
    eps = 1e-6
    R, G, B = (np.asarray(x, np.float32) for x in (R, G, B))
    if name == "NDVI":
        N = np.asarray(nir, np.float32)
        return (N - R) / (N + R + eps)
    if name == "ExG":
        return (2 * G - R - B) / (R + G + B + eps)
    if name == "VARI":
        return np.clip((G - R) / (G + R - B + eps), -1, 1)
    if name == "GLI":
        return (2 * G - R - B) / (2 * G + R + B + eps)
    if name == "NGRDI":
        return (G - R) / (G + R + eps)
    raise ValueError(name)


def index_images(R, G, B, valid, nir=None, dark_max=DARK_MAX):
    """
    Un mapa por índice de verdor (NaN donde no vale: sin datos o tan oscuro que
    el cociente entre bandas es ruido). Con nir se agrega NDVI y se deja
    primero.
    """
    bright = (np.asarray(R, np.float32) + G + B) / 3.0
    ok = valid & (bright >= dark_max)
    names = (["NDVI"] if nir is not None else []) + GREEN_INDICES
    out = {}
    for nm in names:
        v = green_index(nm, R, G, B, nir).astype(np.float32)
        out[nm] = np.where(ok & np.isfinite(v), v, np.nan)
    return out


# ---------------------------------------------------------------------------
# Etapa 1: delimitación de cuarteles
# ---------------------------------------------------------------------------
def _angle_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def tile_patterns(img, tile=128, stride=64, pmin_px=4.0, pmax_px=40.0, min_valid=0.7):
    """
    Patrón periódico dominante en cada ventana (img con NaN donde no vale).
    Devuelve dicts (ang, per, ring): rumbo de las hileras en coordenadas de
    imagen (grados, mod 180), período en px, y cuántas veces sobresale el pico
    de la mediana de su anillo de frecuencia (un patrón de hileras sobresale
    cientos de veces, el verdor sin patrón unas pocas).
    """
    H, W = img.shape
    fin = np.isfinite(img)
    win = np.outer(np.hanning(tile), np.hanning(tile))
    fy = np.fft.fftfreq(tile)[:, None]
    fx = np.fft.fftfreq(tile)[None, :]
    fr = np.hypot(fx, fy)
    band = (fr >= 1.0 / pmax_px) & (fr <= 1.0 / pmin_px) & (fr <= 0.5)
    rbin = np.round(fr * tile).astype(np.int64)
    out = []
    for y in range(0, max(1, H - tile + 1), stride):
        for x in range(0, max(1, W - tile + 1), stride):
            v = fin[y:y + tile, x:x + tile]
            if v.mean() < min_valid:
                continue
            p = img[y:y + tile, x:x + tile].astype(np.float64)
            p = np.where(v, p, p[v].mean())
            P = np.abs(np.fft.fft2((p - p.mean()) * win)) ** 2
            m = np.where(band, P, 0.0)
            iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
            ys = [(iy + d) % tile for d in (-1, 0, 1)]
            xs = [(ix + d) % tile for d in (-1, 0, 1)]
            blk = P[np.ix_(ys, xs)]
            fyk = fy[ys, 0][:, None] + 0 * blk
            fxk = fx[0, xs][None, :] + 0 * blk
            fyk = np.where(np.abs(fyk - fy[iy, 0]) > 0.5, fyk - np.sign(fyk), fyk)
            fxk = np.where(np.abs(fxk - fx[0, ix]) > 0.5, fxk - np.sign(fxk), fxk)
            s = blk.sum()
            fyc, fxc = float((blk * fyk).sum() / s), float((blk * fxk).sum() / s)
            fc = math.hypot(fxc, fyc)
            if fc <= 0:
                continue
            ring_vals = P[(rbin == rbin[iy, ix]) & band]
            ring = float(P[iy, ix] / max(float(np.median(ring_vals)), 1e-30)) if ring_vals.size else 0.0
            out.append({"ang": (math.degrees(math.atan2(fyc, fxc)) + 90.0) % 180.0,
                        "per": 1.0 / fc, "ring": ring})
    return out


def cluster_patterns(votes, min_ring=100.0, min_votes=8, ang_tol=4.0, per_tol=0.08):
    """Agrupa las ventanas por rumbo y período parecidos; descarta los grupos
    chicos y los armónicos (período mitad o tercio con el mismo rumbo)."""
    vs = sorted((v for v in votes if v["ring"] >= min_ring), key=lambda v: -v["ring"])
    groups = []
    for v in vs:
        for g in groups:
            if _angle_diff(v["ang"], g["ang"]) <= ang_tol and abs(v["per"] - g["per"]) <= per_tol * g["per"]:
                g["m"].append(v)
                break
        else:
            groups.append({"ang": v["ang"], "per": v["per"], "m": [v]})
    out = []
    for g in groups:
        w = np.array([m["ring"] for m in g["m"]])
        a2 = np.radians(2 * np.array([m["ang"] for m in g["m"]]))
        ang = (math.degrees(math.atan2(float((w * np.sin(a2)).sum()), float((w * np.cos(a2)).sum()))) / 2.0) % 180.0
        out.append({"ang": ang, "per": float(np.average([m["per"] for m in g["m"]], weights=w)), "n": len(g["m"])})
    out.sort(key=lambda g: -g["n"])
    keep = []
    for g in out:
        harm = any(_angle_diff(g["ang"], k["ang"]) <= ang_tol and
                   any(abs(g["per"] - k["per"] / n) <= per_tol * k["per"] / n for n in (2, 3)) for k in keep)
        if g["n"] >= min_votes and not harm:
            keep.append(g)
    return keep


def pattern_ratio(img, ang_deg, per_px, xx, yy):
    """
    Qué parte del contraste local de img es un patrón de franjas de rumbo
    ang_deg y período per_px (0 = nada, ~0.8 = hileras claras): demodulación
    local con una gaussiana de un período, contra la energía local.
    """
    import cv2
    fin = np.isfinite(img)
    th = math.radians(ang_deg)
    nx, ny = -math.sin(th), math.cos(th)
    fill = np.where(fin, img, np.nanmean(img)).astype(np.float32)
    hp = fill - cv2.GaussianBlur(fill, (0, 0), 2.5 * per_px)
    ph = (2 * math.pi / per_px) * (nx * xx + ny * yy)
    re = cv2.GaussianBlur(hp * np.cos(ph), (0, 0), per_px)
    im = cv2.GaussianBlur(hp * np.sin(ph), (0, 0), per_px)
    amp = 2.0 * np.hypot(re, im)
    e = np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), per_px))
    floor = 0.08 * float(np.sqrt((hp[fin] ** 2).mean()))
    r = amp / (math.sqrt(2.0) * np.maximum(e, floor))
    r[~fin] = 0.0
    return r


def detect_blocks(imgs, valid, pixel_size, period_range_m=(1.0, 8.0), min_area_ha=0.2,
                  thr_hi=0.45, thr_lo=0.20, info=None):
    """
    Cuarteles: zonas con patrón de hileras en el índice de verdor.
    imgs: {nombre: índice} a la resolución de trabajo (pixel_size en m);
    valid: máscara. Devuelve dicts (ring, ang, per_m, area_ha, ratio, index)
    con el contorno en píxeles de esa imagen, de mayor a menor área.
    """
    import cv2
    H, W = valid.shape
    pmin, pmax = period_range_m[0] / pixel_size, period_range_m[1] / pixel_size
    tile = 128 if min(H, W) >= 128 else 64
    votes = []
    for nm, im in imgs.items():
        votes += tile_patterns(im, tile=tile, stride=tile // 2, pmin_px=max(3.0, pmin),
                               pmax_px=min(pmax, tile / 3.0))
    groups = cluster_patterns(votes)
    if info is not None:
        info["groups"] = [{"ang": g["ang"], "per_m": g["per"] * pixel_size, "n": g["n"]} for g in groups]
    if not groups:
        return []
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    R = np.stack([np.maximum.reduce([pattern_ratio(im, g["ang"], g["per"], xx, yy) for im in imgs.values()])
                  for g in groups])
    best, Rmax = np.argmax(R, axis=0), np.max(R, axis=0)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    min_px = min_area_ha * 10000.0 / (pixel_size ** 2)
    out = []
    for gi, g in enumerate(groups):
        own = (best == gi) & valid
        lo = cv2.morphologyEx((own & (Rmax >= thr_lo)).astype(np.uint8), cv2.MORPH_OPEN, k3)
        hi = own & (Rmax >= thr_hi)
        n, lab = cv2.connectedComponents(lo, connectivity=8)
        for c in range(1, n):
            comp = lab == c
            if not hi[comp].any() or comp.sum() < min_px:
                continue
            m = cv2.morphologyEx(comp.astype(np.uint8), cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            # púas hacia las calles: se quita lo más angosto que una hilera
            ko = max(3, int(round(g["per"])) | 1)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cn = max(cnts, key=cv2.contourArea)
            ap = cv2.approxPolyDP(cn, 0.25 * g["per"], True)[:, 0, :].astype(float)
            if len(ap) < 3:
                continue
            out.append({"ring": [(float(x) + 0.5, float(y) + 0.5) for x, y in ap],
                        "ang": g["ang"], "per_m": g["per"] * pixel_size,
                        "area_ha": float(comp.sum()) * pixel_size ** 2 / 10000.0,
                        "ratio": float(Rmax[comp].mean())})
    # de norte a sur (y de oeste a este entre los de la misma altura)
    out.sort(key=lambda b: (round(float(np.mean([p[1] for p in b["ring"]])) * pixel_size / 50.0),
                            float(np.mean([p[0] for p in b["ring"]]))))
    return out


# ---------------------------------------------------------------------------
# Etapa 2: rumbo y distancia entre hileras de un cuartel
# ---------------------------------------------------------------------------
def rasterize(ring, shape):
    """Máscara booleana de un contorno (lista de (x, y) en píxeles)."""
    import cv2
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.array(ring)).astype(np.int32)], 1)
    return m > 0


def _spectrum_peak(img, mask, pmin_px, pmax_px, tile=128, max_tiles=24):
    """Espectro promedio (Welch) de las ventanas dentro de la máscara y su pico:
    (rumbo imagen en grados, período en px, veces que sobresale de su anillo)."""
    H, W = img.shape
    ok = mask & np.isfinite(img)
    tile = int(min(tile, H, W))
    if tile < 32:
        return None
    step = max(1, tile // 2)
    cands = [(y, x) for y in range(0, H - tile + 1, step) for x in range(0, W - tile + 1, step)
             if ok[y:y + tile, x:x + tile].mean() >= 0.7]
    if not cands:
        return None
    sel = np.unique(np.linspace(0, len(cands) - 1, min(max_tiles, len(cands))).round().astype(int))
    win = np.outer(np.hanning(tile), np.hanning(tile))
    acc = np.zeros((tile, tile))
    for i in sel:
        y, x = cands[i]
        v = ok[y:y + tile, x:x + tile]
        p = img[y:y + tile, x:x + tile].astype(np.float64)
        p = np.where(v, p, p[v].mean())
        acc += np.abs(np.fft.fft2((p - p.mean()) * win)) ** 2
    fy = np.fft.fftfreq(tile)[:, None]
    fx = np.fft.fftfreq(tile)[None, :]
    fr = np.hypot(fx, fy)
    band = (fr >= max(1.0 / pmax_px, 3.0 / tile)) & (fr <= 1.0 / pmin_px) & (fr <= 0.5)
    m = np.where(band, acc, 0.0)
    iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
    ys = [(iy + d) % tile for d in (-1, 0, 1)]
    xs = [(ix + d) % tile for d in (-1, 0, 1)]
    blk = acc[np.ix_(ys, xs)]
    fyk = fy[ys, 0][:, None] + 0 * blk
    fxk = fx[0, xs][None, :] + 0 * blk
    fyk = np.where(np.abs(fyk - fy[iy, 0]) > 0.5, fyk - np.sign(fyk), fyk)
    fxk = np.where(np.abs(fxk - fx[0, ix]) > 0.5, fxk - np.sign(fxk), fxk)
    s = blk.sum()
    fyc, fxc = float((blk * fyk).sum() / s), float((blk * fxk).sum() / s)
    rbin = np.round(fr * tile).astype(np.int64)
    ring_vals = acc[(rbin == rbin[iy, ix]) & band]
    ring = float(acc[iy, ix] / max(float(np.median(ring_vals)), 1e-30))
    return (math.degrees(math.atan2(fyc, fxc)) + 90.0) % 180.0, 1.0 / math.hypot(fxc, fyc), ring


def _profile_vs_angle(img, ok, ang_deg, per_px, max_samples=1500000):
    ys, xs = np.nonzero(ok)
    st = max(1, len(xs) // max_samples)
    ys, xs = ys[::st].astype(np.float64), xs[::st].astype(np.float64)
    vals = img[ok][::st].astype(np.float64)
    return xs, ys, vals


def refine_angle(img, ok, ang0, per_px, span=1.5):
    """Afina el rumbo: el ángulo donde el perfil perpendicular tiene más
    contraste (varianza del verdor medio según la distancia a la hilera)."""
    xs, ys, vals = _profile_vs_angle(img, ok, ang0, per_px)
    bw = max(per_px / 12.0, 0.25)

    def score(a):
        th = math.radians(a)
        t = -xs * math.sin(th) + ys * math.cos(th)
        k = ((t - t.min()) / bw).astype(np.int64)
        c = np.bincount(k).astype(np.float64)
        s = np.bincount(k, vals)
        m = c > 0
        prof = s[m] / c[m]
        w = c[m]
        mu = (prof * w).sum() / w.sum()
        return float((w * (prof - mu) ** 2).sum() / w.sum())

    best, sp, stp = float(ang0), float(span), float(span) / 10.0
    for _ in range(3):
        cs = best + np.arange(-sp, sp + 1e-9, stp)
        best = float(cs[int(np.argmax([score(a) for a in cs]))])
        sp, stp = stp, stp / 5.0
    return best % 180.0


def refine_period(img, ok, ang, per0_px, rel=0.06):
    """Período y fase de las hileras con el rumbo ya fijo: periodograma fino
    del perfil perpendicular. Devuelve (período px, fase v0 px, amplitud
    relativa al contraste del perfil)."""
    xs, ys, vals = _profile_vs_angle(img, ok, ang, per0_px)
    th = math.radians(ang)
    t = -xs * math.sin(th) + ys * math.cos(th)
    bw = per0_px / 16.0
    k = ((t - t.min()) / bw).astype(np.int64)
    c = np.bincount(k).astype(np.float64)
    s = np.bincount(k, vals)
    m = c > 8
    v = (np.nonzero(m)[0] + 0.5) * bw + t.min()
    prof = s[m] / c[m]
    prof = prof - prof.mean()
    w = np.sqrt(c[m])
    best = (0.0, per0_px, 0.0)
    for T in per0_px * np.arange(1 - rel, 1 + rel + 1e-9, 0.001):
        z = (prof * w * np.exp(-2j * np.pi * v / T)).sum() / w.sum()
        if abs(z) > best[0]:
            best = (abs(z), T, float(np.angle(z)))
    amp, T, ph = best
    # los máximos del perfil están en v = v0 + k T con v0 = ph T / 2pi
    v0 = -(ph / (2 * math.pi)) * T
    return T, v0, 2.0 * amp / max(float(prof.std()) * math.sqrt(2.0), 1e-9)


def estimate_rows(imgs, mask, pixel_size, period_range_m=(1.0, 8.0)):
    """
    Rumbo y distancia entre hileras de UN cuartel, solo con verde. Elige el
    índice de verdor cuyo patrón sobresale más. Devuelve dict (index, ang
    [rumbo en coordenadas de imagen, grados], per_px, per_m, v0 [fase,
    px], ring, amp) en la resolución de trabajo, o None.
    """
    pmin, pmax = period_range_m[0] / pixel_size, period_range_m[1] / pixel_size
    best = None
    for nm, im in imgs.items():
        r = _spectrum_peak(im, mask, max(3.0, pmin), pmax)
        if r is None:
            continue
        if best is None or r[2] > best[1][2]:
            best = (nm, r)
    if best is None:
        return None
    nm, (a0, T0, ring) = best
    img = imgs[nm]
    ok = mask & np.isfinite(img)
    ang = refine_angle(img, ok, a0, T0)
    T, v0, amp = refine_period(img, ok, ang, T0)
    return {"index": nm, "ang": ang, "per_px": T, "per_m": T * pixel_size, "v0": v0,
            "ring": ring, "amp": amp, "ang0": a0, "per0_m": T0 * pixel_size}


# ---------------------------------------------------------------------------
# Etapa 3: líneas iniciales
# ---------------------------------------------------------------------------
def initial_rows(ring, ang, per_px, v0, shape=None):
    """
    Una recta por hilera (paralelas, a distancia per_px) con el rumbo ang
    (grados, coordenadas de imagen), recortadas al contorno ring (px). Las
    coordenadas son las de la imagen de trabajo: la hilera k es el conjunto
    de puntos con  v = -x sin(ang) + y cos(ang) = v0 + k per_px.
    Devuelve [(k, [(x1, y1, x2, y2), ...])]: una hilera puede tener varios
    tramos si el contorno es cóncavo.
    """
    from qgis.core import QgsGeometry, QgsPointXY
    th = math.radians(ang)
    d = np.array([math.cos(th), math.sin(th)])
    n = np.array([-math.sin(th), math.cos(th)])
    poly = QgsGeometry.fromPolygonXY([[QgsPointXY(x, y) for x, y in ring]])
    if not poly.isGeosValid():
        poly = poly.makeValid()
    pts = np.array(ring)
    vs, us = pts @ n, pts @ d
    k0 = int(math.ceil((vs.min() - v0) / per_px))
    k1 = int(math.floor((vs.max() - v0) / per_px))
    ext = (us.max() - us.min()) + 10.0
    uc = 0.5 * (us.max() + us.min())
    rows = []
    for k in range(k0, k1 + 1):
        vk = v0 + k * per_px
        a = vk * n + (uc - ext) * d
        b = vk * n + (uc + ext) * d
        inter = poly.intersection(QgsGeometry.fromPolylineXY([QgsPointXY(*a), QgsPointXY(*b)]))
        parts = inter.asMultiPolyline() if inter.isMultipart() else ([inter.asPolyline()] if not inter.isEmpty() else [])
        segs = [(p[0].x(), p[0].y(), p[-1].x(), p[-1].y()) for p in parts if len(p) >= 2]
        if segs:
            rows.append((k, segs))
    return rows


def refine_phase(img, ring, ang, per_px, v0, span=0.3):
    """Corre la fase de las hileras hasta donde el verdor medio sobre las líneas
    es máximo (dentro de +-span períodos): el máximo del perfil real no coincide
    exactamente con la fase de su primera armónica. Devuelve el nuevo v0."""
    H, W = img.shape
    th = math.radians(ang)
    n = np.array([-math.sin(th), math.cos(th)])
    rows = initial_rows(ring, ang, per_px, v0)
    pts = []
    for k, segs in rows:
        for (x1, y1, x2, y2) in segs:
            m = max(2, int(math.hypot(x2 - x1, y2 - y1)))
            t = np.linspace(0, 1, m)
            pts.append(np.stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t], axis=1))
    P = np.concatenate(pts)
    best = (-1e9, 0.0)
    for s in np.arange(-span, span + 1e-9, 0.01) * per_px:
        xs, ys = P[:, 0] + s * n[0], P[:, 1] + s * n[1]
        c = np.clip(xs.astype(int), 0, W - 1)
        r = np.clip(ys.astype(int), 0, H - 1)
        v = img[r, c]
        if np.isfinite(v).any() and np.nanmean(v) > best[0]:
            best = (float(np.nanmean(v)), float(s))
    return v0 + best[1]
