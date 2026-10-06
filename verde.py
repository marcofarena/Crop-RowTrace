# -*- coding: utf-8 -*-
"""
Hileras de cultivo y fallas por cuartel, a partir del VERDE.

Proceso (cada etapa usa solo un índice de verdor; la sombra no se usa para
detectar nada, y los píxeles tan oscuros que el índice es puro ruido se
excluyen del muestreo):

  1. Delimitación de cuarteles: zonas con un patrón periódico de hileras (el
     verdor sin patrón, como monte o pasto, no es cuartel). Por defecto se usa el
     método de bloques.py (patrón en el verdor y en el brillo), que da contornos
     más limpios que detect_blocks, el equivalente solo con verde.
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
import itertools
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


# ---------------------------------------------------------------------------
# Resolución completa: recorte del raster y marco de un cuartel
# ---------------------------------------------------------------------------
class Tile:
    """Recorte del raster en memoria con el índice de verdor de un cuartel."""

    def __init__(self, R, G, B, valid, x0, y0, px, py, nir=None, dark_max=DARK_MAX):
        self.R, self.G, self.B, self.valid, self.nir = R, G, B, valid, nir
        self.x0, self.y0, self.px, self.py = x0, y0, px, py
        self.H, self.W = R.shape
        self.dark_max = dark_max
        self._maps = {}

    def index_map(self, name):
        if name not in self._maps:
            bright = (self.R.astype(np.float32) + self.G + self.B) / 3.0
            v = green_index(name, self.R, self.G, self.B, self.nir)
            self._maps[name] = np.where(self.valid & (bright >= self.dark_max) & np.isfinite(v), v, np.nan).astype(np.float32)
        return self._maps[name]

    def sample(self, name, x, y):
        """Valor del índice en (x, y) de mapa (vecino más próximo) y máscara."""
        m = self.index_map(name)
        c = np.floor((np.asarray(x) - self.x0) / self.px).astype(np.int64)
        r = np.floor((self.y0 - np.asarray(y)) / self.py).astype(np.int64)
        inside = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)
        v = m[np.clip(r, 0, self.H - 1), np.clip(c, 0, self.W - 1)]
        ok = inside & np.isfinite(v)
        return v, ok


def read_tile(ds, bbox, margin=3.0, bands=(1, 2, 3), nir_band=None, dark_max=DARK_MAX):
    """Recorta el raster a bbox (xmin, ymin, xmax, ymax en mapa) más un margen."""
    from osgeo import gdal
    gt = ds.GetGeoTransform()
    x0, y0, x1, y1 = bbox[0] - margin, bbox[3] + margin, bbox[2] + margin, bbox[1] - margin
    c0 = max(0, int((x0 - gt[0]) / gt[1])); c1 = min(ds.RasterXSize, int((x1 - gt[0]) / gt[1]) + 1)
    r0 = max(0, int((y0 - gt[3]) / gt[5])); r1 = min(ds.RasterYSize, int((y1 - gt[3]) / gt[5]) + 1)
    if c1 <= c0 or r1 <= r0:
        return None
    rd = lambda b: ds.GetRasterBand(b).ReadAsArray(c0, r0, c1 - c0, r1 - r0)
    R, G, B = (rd(b) for b in bands)
    valid = np.ones(R.shape, bool)
    for i in range(1, ds.RasterCount + 1):
        if ds.GetRasterBand(i).GetColorInterpretation() == gdal.GCI_AlphaBand:
            valid &= rd(i) > 0
    nir = rd(nir_band) if nir_band else None
    return Tile(R, G, B, valid, gt[0] + c0 * gt[1], gt[3] + r0 * gt[5], gt[1], -gt[5], nir, dark_max)


class Frame:
    """Marco (u a lo largo de las hileras, v a través) de un cuartel, en metros de mapa."""

    def __init__(self, origin, u):
        self.o = np.asarray(origin, float)
        u = np.asarray(u, float)
        self.u = u / np.hypot(*u)
        self.n = np.array([-self.u[1], self.u[0]])

    def to_uv(self, x, y):
        dx, dy = np.asarray(x) - self.o[0], np.asarray(y) - self.o[1]
        return dx * self.u[0] + dy * self.u[1], dx * self.n[0] + dy * self.n[1]

    def to_xy(self, u, v):
        u, v = np.asarray(u), np.asarray(v)
        return self.o[0] + u * self.u[0] + v * self.n[0], self.o[1] + u * self.u[1] + v * self.n[1]


class Row:
    """Un tramo de hilera: polilínea v(u) en el marco del cuartel."""

    def __init__(self, hilera, u_knots, v_knots):
        self.hilera = hilera
        self.uk = np.asarray(u_knots, float)
        self.vk = np.asarray(v_knots, float)

    @property
    def u0(self):
        return float(self.uk[0])

    @property
    def u1(self):
        return float(self.uk[-1])

    def v_at(self, u):
        return np.interp(u, self.uk, self.vk)


def rows_from_segments(frame, segments):
    """segments: [(hilera, x1, y1, x2, y2)] en mapa -> [Row]."""
    out = []
    for h, x1, y1, x2, y2 in segments:
        (ua, ub), (va, vb) = frame.to_uv([x1, x2], [y1, y2])
        if ua > ub:
            ua, ub, va, vb = ub, ua, vb, va
        out.append(Row(h, [ua, ub], [va, vb]))
    return out


# ---------------------------------------------------------------------------
# Etapa 4: buffer y perfiles transversales; picos de verdor
# ---------------------------------------------------------------------------
def lateral_profiles(tile, frame, index, row, half, win_m=2.0, step=None, min_frac=0.5):
    """
    Perfiles transversales de verdor dentro del buffer (+-half m) de una
    hilera, uno por ventana de win_m metros a lo largo de ella. Devuelve
    (centros u de las ventanas, desplazamientos laterales d, matriz [ventana,
    d] con el verdor medio, NaN donde faltan datos).
    """
    px = float(min(tile.px, tile.py))
    step = step or px
    nw = max(1, int(round(win_m / step)))
    n_win = max(1, int(math.floor((row.u1 - row.u0) / win_m)))
    u0 = row.u0 + 0.5 * ((row.u1 - row.u0) - n_win * win_m)       # centrar las ventanas
    us = u0 + (np.arange(n_win * nw) + 0.5) * step
    d = np.arange(-half, half + 1e-9, px)
    vv = row.v_at(us)[:, None] + d[None, :]
    uu = np.broadcast_to(us[:, None], vv.shape)
    x, y = frame.to_xy(uu, vv)
    val, ok = tile.sample(index, x.ravel(), y.ravel())
    val = np.where(ok, val, 0.0).reshape(vv.shape)
    ok = ok.reshape(vv.shape)
    cnt = ok.reshape(n_win, nw, -1).sum(1)
    sm = val.reshape(n_win, nw, -1).sum(1)
    prof = np.where(cnt >= min_frac * nw, sm / np.maximum(cnt, 1), np.nan)
    centers = u0 + (np.arange(n_win) + 0.5) * win_m
    return centers, d, prof


def window_peaks(prof, d, px, sigma_m=0.08):
    """
    Pico de cada perfil: posición lateral (m), altura y prominencia (altura
    menos el cuartil inferior del perfil). NaN donde el perfil no tiene datos.
    """
    k = max(1, int(round(3 * sigma_m / px)))
    kern = np.exp(-0.5 * (np.arange(-k, k + 1) * px / sigma_m) ** 2)
    kern /= kern.sum()
    n, m = prof.shape
    pos = np.full(n, np.nan)
    hgt = np.full(n, np.nan)
    prm = np.full(n, np.nan)
    for i in range(n):
        p = prof[i]
        f = np.isfinite(p)
        if f.sum() < 0.7 * m:
            continue
        q = np.where(f, p, np.nanmin(p))
        s = np.convolve(np.pad(q, k, mode="edge"), kern, mode="valid")
        j = int(np.argmax(s))
        pos[i], hgt[i], prm[i] = d[j], s[j], s[j] - np.percentile(s, 25)
    return pos, hgt, prm


# ---------------------------------------------------------------------------
# Etapa 5: líneas con pocos quiebres leves ajustadas a los picos de verde
# ---------------------------------------------------------------------------
def fit_polyline(uc, vc, w, n_breaks, kink_max, u_lo, u_hi, cap=0.25):
    """
    Poligonal continua con a lo sumo n_breaks quiebres ajustada por mínimos
    cuadrados (pérdida robusta: un pico aislado que se aparta mucho pesa poco)
    a los picos (uc, vc) con pesos w. Los quiebres se eligen entre las
    posiciones de los picos y cada uno cambia la pendiente a lo sumo kink_max
    (m/m). Prueba 0, 1, ..., n_breaks quiebres y se queda con el menor número
    cuya pérdida no es más de 10 % peor que la de más quiebres. Devuelve
    (knots_u, knots_v, n_quiebres, pérdida).
    """
    uc, vc, w = (np.asarray(a, float) for a in (uc, vc, w))
    M = len(uc)
    um = float(uc.mean())
    cap2 = cap * cap

    def design(knots, x):
        return np.array([np.ones(len(x)), np.asarray(x) - um] + [np.maximum(0.0, np.asarray(x) - k) for k in knots]).T

    def solve(knots, wt):
        X = design(knots, uc)
        beta = np.linalg.lstsq(X * np.sqrt(wt)[:, None], vc * np.sqrt(wt), rcond=None)[0]
        return beta, vc - X @ beta

    def loss(res):
        return float((w * np.minimum(res * res, cap2)).sum())

    def robust(knots):
        wt = w.copy()
        for _ in range(3):
            beta, res = solve(knots, wt)
            wt = w / (1.0 + (res / 0.12) ** 2)
        return beta, res

    cands = list(uc[2:-2]) if M > 5 else []
    if len(cands) > 16:
        cands = [cands[i] for i in np.unique(np.linspace(0, len(cands) - 1, 16).round().astype(int))]
    best = {}
    beta, res = robust([])
    best[0] = (loss(res), [], beta)
    chosen = []
    for n in range(1, n_breaks + 1):
        combos = (itertools.combinations(cands, n) if n <= 3
                  else ([c for c in chosen] + [k] for k in cands if k not in chosen))
        bn = None
        for kn in combos:
            kn = sorted(kn)
            beta, res = solve(kn, w)
            if any(abs(c) > kink_max + 1e-9 for c in beta[2:]):
                continue
            L = loss(res)
            if bn is None or L < bn[0]:
                bn = (L, kn)
        if bn is None:
            break
        chosen = list(bn[1])
        beta, res = robust(chosen)
        if all(abs(c) <= kink_max + 1e-9 for c in beta[2:]):
            best[n] = (loss(res), chosen, beta)
    # menor número de quiebres que no pierda más de 10 % contra el mejor
    Lmin = min(b[0] for b in best.values())
    n_sel = min(n for n, b in best.items() if b[0] <= 1.10 * Lmin + 1e-12)
    L, knots, beta = best[n_sel]
    ku = np.array([u_lo] + [k for k in knots if u_lo < k < u_hi] + [u_hi])
    kv = design(knots, ku) @ beta
    return ku, kv, len(knots), L


def adjust_row(tile, frame, index, row, half, win_m=2.0, n_breaks=3, kink_deg=3.0, n_iter=3,
               max_shift=0.9, prom_ref=None, min_gain=0.15):
    """
    Ajusta una hilera: buffer alrededor de su línea -> picos de verdor en los
    perfiles transversales -> línea con pocos quiebres que los ajusta -> nuevo
    buffer alrededor de ella, hasta que converja. Verifica con validación
    cruzada (se ajusta con las ventanas pares y se mide con las impares) que la
    línea nueva queda más cerca de los picos que la inicial; si no, deja la
    inicial. Devuelve (Row nueva, dict con el detalle).
    """
    px = float(min(tile.px, tile.py))
    kink_max = math.tan(math.radians(kink_deg))
    cur = Row(row.hilera, row.uk.copy(), row.vk.copy())
    info = {"n_win": 0, "n_pico": 0, "iter": 0, "quiebres": 0, "aceptada": False,
            "dev_ini": np.nan, "dev_nueva": np.nan, "cv_ini": np.nan, "cv_nueva": np.nan}
    first = None
    for it in range(n_iter):
        cen, d, prof = lateral_profiles(tile, frame, index, cur, half, win_m)
        pos, hgt, prm = window_peaks(prof, d, px)
        ref = prom_ref if prom_ref is not None else np.nanpercentile(prm, 75) if np.isfinite(prm).any() else np.nan
        clear = np.isfinite(pos) & (prm >= 0.35 * ref)
        if clear.sum() < 6:
            break
        vpk = cur.v_at(cen) + pos                          # posición lateral absoluta del pico
        if first is None:
            first = (cen[clear], vpk[clear], pos[clear], prm[clear] / ref)
            info["n_win"], info["n_pico"] = len(cen), int(clear.sum())
        w = np.clip(prm[clear] / ref, 0.2, 1.0)
        uc, vc = cen[clear], vpk[clear]
        # tope: no alejarse de la línea inicial más de max_shift
        v_init = row.v_at(uc)
        vc = np.clip(vc, v_init - max_shift, v_init + max_shift)
        ku, kv, nb, _ = fit_polyline(uc, vc, w, n_breaks, kink_max, row.u0, row.u1)
        new = Row(row.hilera, ku, kv)
        change = float(np.max(np.abs(new.v_at(cen) - cur.v_at(cen))))
        cur = new
        info["iter"], info["quiebres"] = it + 1, nb
        if change < 0.03:
            break
    if first is None:
        return row, info
    # validación cruzada con los picos de la primera pasada (alrededor de la línea inicial)
    uc0, vc0, d0, w0 = first
    ev, od = np.arange(len(uc0)) % 2 == 0, np.arange(len(uc0)) % 2 == 1
    if ev.sum() >= 4 and od.sum() >= 4:
        ku2, kv2, _, _ = fit_polyline(uc0[ev], vc0[ev], np.clip(w0[ev], 0.2, 1), n_breaks, kink_max, row.u0, row.u1)
        cv_new = float(np.median(np.abs(vc0[od] - np.interp(uc0[od], ku2, kv2))))
        cv_ini = float(np.median(np.abs(vc0[od] - row.v_at(uc0[od]))))
        info["cv_ini"], info["cv_nueva"] = cv_ini, cv_new
        info["aceptada"] = bool(cv_new <= (1.0 - min_gain) * cv_ini)
    info["peaks"] = (uc0, vc0, d0)
    info["dev_ini"] = float(np.median(np.abs(d0)))
    info["dev_nueva"] = float(np.median(np.abs(vc0 - cur.v_at(uc0))))
    return (cur if info["aceptada"] else row), info


# ---------------------------------------------------------------------------
# Etapa 6: segundo buffer (alrededor de las líneas ajustadas) y fallas
# ---------------------------------------------------------------------------
def _sample_along(tile, frame, index, row, d, step):
    """Verdor de la hilera en una grilla (u a lo largo cada 'step' m, d lateral):
    devuelve (u, matriz [u, d] con NaN donde no hay dato)."""
    us = np.arange(row.u0 + step / 2.0, row.u1, step)
    if len(us) < 2:
        return us, np.full((len(us), len(d)), np.nan)
    vv = row.v_at(us)[:, None] + d[None, :]
    uu = np.broadcast_to(us[:, None], vv.shape)
    x, y = frame.to_xy(uu, vv)
    val, ok = tile.sample(index, x.ravel(), y.ravel())
    return us, np.where(ok, val, np.nan).reshape(vv.shape)


def vegetation_profile(tile, frame, index, rows, search, max_rows=60):
    """Perfil transversal medio de verdor medido sobre las líneas AJUSTADAS
    (+-search m): (d, perfil suavizado, ancho al 60 % de la altura, nivel de
    entrehilera, altura del pico)."""
    px = float(min(tile.px, tile.py))
    d = np.arange(-search, search + 1e-9, px)
    acc = np.zeros(len(d)); cnt = np.zeros(len(d))
    for row in rows[:: max(1, len(rows) // max_rows)]:
        us, M = _sample_along(tile, frame, index, row, d, 0.25)
        f = np.isfinite(M)
        acc += np.where(f, M, 0.0).sum(0); cnt += f.sum(0)
    P = acc / np.maximum(cnt, 1)
    P[cnt < 30] = np.nan
    k = max(1, int(round(0.1 / px)))
    Ps = np.convolve(np.pad(np.nan_to_num(P, nan=np.nanmin(P)), k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    j = int(np.argmax(Ps))
    base = float(np.percentile(Ps, 20))
    thr = base + 0.6 * (Ps[j] - base)
    a = b = j
    while a > 0 and Ps[a - 1] >= thr:
        a -= 1
    while b < len(Ps) - 1 and Ps[b + 1] >= thr:
        b += 1
    return d, Ps, float(d[b] - d[a] + px), base, float(Ps[j])


def row_vigor(tile, frame, index, row, half, along=0.3, step=0.1):
    """Verdor de la franja medido dentro del buffer (+-half m) de una hilera
    ajustada: en cada punto, el pico (suavizado 0,16 m) del perfil lateral
    promediado 'along' metros a lo largo de la hilera. Devuelve (u, vigor)."""
    px = float(min(tile.px, tile.py))
    d = np.arange(-half, half + 1e-9, px)
    us, M = _sample_along(tile, frame, index, row, d, step)
    if len(us) < 2:
        return us, np.full(len(us), np.nan)
    f = np.isfinite(M)
    k = max(1, int(round(along / step)))
    z = np.zeros((1, M.shape[1]))
    cs = np.concatenate([z, np.cumsum(np.where(f, M, 0.0), axis=0)])
    cc = np.concatenate([z, np.cumsum(f, axis=0)])
    n = len(us)
    lo = np.clip(np.arange(n) - k // 2, 0, n)
    hi = np.clip(np.arange(n) + k // 2 + 1, 0, n)
    sm = (cs[hi] - cs[lo]) / np.maximum(cc[hi] - cc[lo], 1)
    sm = np.where((cc[hi] - cc[lo]) >= 0.5 * (hi - lo)[:, None], sm, np.nan)
    kl = max(1, int(round(0.08 / px)))                       # suavizado lateral (vectorizado)
    fin = np.isfinite(sm)
    rowmin = np.where(fin.any(axis=1), np.nanmin(np.where(fin, sm, np.inf), axis=1), 0.0)
    fill = np.where(fin, sm, rowmin[:, None])
    pad = np.pad(fill, ((0, 0), (kl, kl)), mode="edge")
    cs2 = np.concatenate([np.zeros((pad.shape[0], 1)), np.cumsum(pad, axis=1)], axis=1)
    sl = (cs2[:, 2 * kl + 1:] - cs2[:, :-(2 * kl + 1)]) / (2 * kl + 1)
    sl = np.where((fin.sum(axis=1) > 0.7 * fin.shape[1])[:, None], sl, np.nan)
    vig = np.full(len(us), np.nan)
    ok_rows = np.isfinite(sl).any(axis=1)
    vig[ok_rows] = np.nanmax(sl[ok_rows], axis=1)
    return us, vig


def plant_spacing(series, step=0.1, max_lag=6.0, min_peak=0.10):
    """
    Distancia entre plantas a lo largo de la hilera: primer máximo de la
    autocorrelación del vigor (promediada sobre las hileras) entre 0,4 m y
    max_lag. Con canopia continua no hay máximo claro (< min_peak): devuelve
    (None, altura). Devuelve (distancia en m o None, altura del pico).
    """
    n_l = int(max_lag / step)
    acc, cnt = np.zeros(n_l + 1), np.zeros(n_l + 1)
    for v in series:
        if len(v) < 2 * n_l:
            continue
        x = v - np.nanmean(v)
        f = np.isfinite(x)
        x = np.where(f, x, 0.0)
        for L in range(n_l + 1):
            m = f[:len(x) - L] & f[L:]
            acc[L] += (x[:len(x) - L] * x[L:])[m].sum()
            cnt[L] += m.sum()
    if acc[0] <= 0:
        return None, 0.0
    ac = (acc / np.maximum(cnt, 1)) / (acc[0] / max(cnt[0], 1))
    for i in range(int(0.4 / step), n_l):
        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] >= min_peak:
            return float(i * step), float(ac[i])
    return None, float(ac[int(0.4 / step):].max()) if n_l > int(0.4 / step) else 0.0


def _running_median(x, k):
    """Mediana móvil de k muestras que ignora NaN (NaN donde no hay ninguna)."""
    if k <= 1 or len(x) < k:
        return x.copy()
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
    w = sliding_window_view(pad, k)
    out = np.full(len(x), np.nan)
    okr = np.isfinite(w).sum(axis=1) >= max(1, k // 2)
    out[okr] = np.nanmedian(w[okr], axis=1)
    return out


def failure_runs(u, z, thr=0.5, hyst=0.0, close_m=1.0, min_len=2.0, step=0.1):
    """Tramos de falla de una serie de vigor normalizado z(u): regiones contiguas
    con z < thr + hyst que contienen algún punto con z < thr; se unen los huecos
    de hasta close_m entre ellas y se queda con las de largo >= min_len."""
    n = len(z)
    seed = np.isfinite(z) & (z < thr)
    grow = np.isfinite(z) & (z < thr + hyst)
    low = np.zeros(n, bool)
    i = 0
    while i < n:
        if grow[i]:
            j = i
            while j + 1 < n and grow[j + 1]:
                j += 1
            if seed[i:j + 1].any():
                low[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    gap = int(round(close_m / step))
    idx = np.nonzero(low)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap + 1:
            low[a:b + 1] = True
    runs, i = [], 0
    while i < n:
        if low[i]:
            j = i
            while j + 1 < n and low[j + 1]:
                j += 1
            ua, ub = u[i] - step / 2, u[j] + step / 2
            if ub - ua >= min_len:
                runs.append((float(ua), float(ub), float(np.nanmean(z[i:j + 1]))))
            i = j + 1
        else:
            i += 1
    return runs


def detect_failures(tile, frame, index, rows, T, resid=0.15, thr=0.5, min_len=2.0, search=None,
                    smooth_m=None, hyst=0.0, close_m=None):
    """
    Fallas de un cuartel con las líneas ajustadas. El buffer es el ancho de la
    franja de vegetación (medido sobre estas líneas) más 'resid' a cada lado.
    El verdor del buffer se normaliza entre el nivel de la entrehilera (0) y el
    de una planta típica del cuartel (1, percentil 75); falla = tramo continuo
    con valor < thr y largo >= min_len.
    Devuelve dict: half, width, base, ref, fallas [dict], rows [(Row, u, z)].
    """
    search = search or 0.35 * T
    d, Ps, W, base_prof, peak = vegetation_profile(tile, frame, index, rows, search)
    half = float(min(max(W / 2.0 + resid, 0.25), 0.6))
    ser = []
    for row in rows:
        u, v = row_vigor(tile, frame, index, row, half)
        ser.append((row, u, v))
    # escala de suavizado y de cierre de huecos: la mitad de la distancia entre plantas
    # (para que los huecos naturales entre plantas chicas no se junten en una falla), y
    # 1 m si la hilera es una canopia continua
    fine = [row_vigor(tile, frame, index, row, half, along=0.15)[1] for row in rows[:: max(1, len(rows) // 25)]
            if row.u1 - row.u0 >= 20.0]
    sp_plants, sp_peak = plant_spacing(fine)
    scale = 1.0 if sp_plants is None else float(np.clip(0.5 * sp_plants, 0.3, 1.0))
    smooth_m = scale if smooth_m is None else smooth_m
    close_m = scale if close_m is None else close_m
    allv = np.concatenate([s[2] for s in ser if len(s[2])])
    allv = allv[np.isfinite(allv)]
    ref = float(np.percentile(allv, 75))
    # nivel de entrehilera: verdor medio a media separación entre dos hileras
    mid = []
    for row in rows[:: max(1, len(rows) // 40)]:
        for sgn in (-1, 1):
            r2 = Row(row.hilera, row.uk, row.vk + sgn * 0.5 * T)
            _, M = _sample_along(tile, frame, index, r2, np.array([0.0]), 0.5)
            mid.append(M[:, 0])
    mid = np.concatenate(mid)
    base = float(np.nanmedian(mid[np.isfinite(mid)]))
    if ref - base < 1e-6:
        return None
    out, fallas = [], []
    for row, u, v in ser:
        z = (v - base) / (ref - base)
        zs = _running_median(z, max(1, int(round(smooth_m / 0.1))))
        out.append((row, u, zs))
        for ua, ub, zm in failure_runs(u, zs, thr, hyst, close_m, min_len):
            fallas.append({"row": row, "ua": ua, "ub": ub, "largo": ub - ua, "z": zm,
                           "borde": int(ua - row.u0 < 2.0 or row.u1 - ub < 2.0)})
    return {"half": half, "width": W, "base": base, "ref": ref, "fallas": fallas, "series": out,
            "plant_spacing": sp_plants, "scale": scale,
            "n_samples": int(sum(np.isfinite(s[2]).sum() for s in ser))}


# ---------------------------------------------------------------------------
# Proceso completo
# ---------------------------------------------------------------------------
def previous_blocks(ds, work_px=0.2, log=print, bands=(1, 2, 3)):
    """Cuarteles con el método de bloques.py (patrón de hileras en el verdor y en
    el brillo, a resolución de trabajo), ordenados de norte a sur. Devuelve
    [{ring (px de la imagen de trabajo), area_ha, ang, per_m, ratio}]."""
    import cv2
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("bloques", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bloques.py"))
    bq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bq)
    gt = ds.GetGeoTransform()
    f = max(1, int(round(work_px / abs(gt[1]))))
    r, g, b = (ds.GetRasterBand(i).ReadAsArray().astype(np.float32) for i in bands)
    valid = np.ones(r.shape, bool)
    for i in bands:
        valid &= ds.GetRasterBand(i).GetMaskBand().ReadAsArray() > 0
    rs, gs, bs = (cv2.GaussianBlur(x, (0, 0), 1.0) for x in (r, g, b))
    tot = rs + gs + bs
    exg = (2 * gs - rs - bs) / np.maximum(tot, 1e-6)
    exg[(tot <= 0) | ~valid] = 0.0
    bri = (r + g + b) / 3.0
    bri[~valid] = 0.0
    del r, g, b, rs, gs, bs, tot
    blocks = bq.detect_blocks(exg, bri, valid, abs(gt[1]), work_px=work_px)
    out = [{"ring": [(x / f, y / f) for x, y in bk["rings"][0]], "area_ha": bk["area_ha"], "ang": bk["ang"],
            "per_m": bk["per_m"], "ratio": bk["ratio"]} for bk in blocks]
    out.sort(key=lambda bk: (round(float(np.mean([p[1] for p in bk["ring"]])) * abs(gt[1]) * f / 50.0),
                             float(np.mean([p[0] for p in bk["ring"]]))))
    return out


def run_all(ds, nir_band=None, only=None, work_px=0.2, n_breaks=3, kink_deg=3.0, win_m=2.0,
            search_frac=0.35, thr=0.5, min_len=2.0, resid=0.15, log=print, blocks=None, delimitacion="previa",
            bands=(1, 2, 3), progress=None):
    """
    Corre las seis etapas sobre un raster abierto con GDAL (RGB en las bandas
    1-3; con nir_band se usa NDVI). only: lista de números de cuartel (de norte
    a sur, desde 1) para correr solo esos. Devuelve una lista de dicts, uno por
    cuartel: ring (px de la imagen de trabajo), est (rumbo, distancia, índice),
    frame, rows_ini, rows (ajustadas), info (por hilera), buffer (mitad del
    ancho, m), fallas, base, ref.
    """
    from osgeo import gdal
    from qgis.core import QgsGeometry, QgsPointXY
    gt = ds.GetGeoTransform()
    f = max(1, int(round(work_px / abs(gt[1]))))
    w, h = ds.RasterXSize // f, ds.RasterYSize // f
    rd = lambda b: ds.GetRasterBand(b).ReadAsArray(buf_xsize=w, buf_ysize=h,
                                                   resample_alg=gdal.GRIORA_Average).astype(np.float32)
    R, G, B = (rd(b) for b in bands)
    valid = np.ones(R.shape, bool)
    for i in range(1, ds.RasterCount + 1):
        if ds.GetRasterBand(i).GetColorInterpretation() == gdal.GCI_AlphaBand:
            valid &= rd(i) > 200
    imgs = index_images(R, G, B, valid, nir=rd(nir_band) if nir_band else None)
    wpx = abs(gt[1]) * f
    px2map = lambda x, y: (gt[0] + x * f * gt[1], gt[3] + y * f * gt[5])
    log("Etapa 1: delimitación de cuarteles (%s)" % delimitacion)
    if blocks is None:
        blocks = previous_blocks(ds, work_px, log, bands) if delimitacion == "previa" else detect_blocks(imgs, valid, wpx)
    out = []
    for c, b in enumerate(blocks, 1):
        if only and c not in only:
            continue
        log("Cuartel %d (%.2f ha)" % (c, b["area_ha"]))
        mask = rasterize(b["ring"], valid.shape) & valid
        est = estimate_rows(imgs, mask, wpx)                                   # etapa 2
        est["v0"] = refine_phase(imgs[est["index"]], b["ring"], est["ang"], est["per_px"], est["v0"])
        rows_px = initial_rows(b["ring"], est["ang"], est["per_px"], est["v0"])   # etapa 3
        segs = []
        for k, ss in rows_px:
            for (x1, y1, x2, y2) in ss:
                (ax, ay), (bx, by) = px2map(x1, y1), px2map(x2, y2)
                segs.append((k, ax, ay, bx, by))
        poly = QgsGeometry.fromPolygonXY([[QgsPointXY(*px2map(x, y)) for x, y in b["ring"]]])
        bb = poly.boundingBox()
        frame = Frame(((segs[0][1] + segs[0][3]) / 2, (segs[0][2] + segs[0][4]) / 2),
                      (segs[0][3] - segs[0][1], segs[0][4] - segs[0][2]))
        rows0 = rows_from_segments(frame, segs)
        tile = read_tile(ds, (bb.xMinimum(), bb.yMinimum(), bb.xMaximum(), bb.yMaximum()),
                         bands=bands, nir_band=nir_band)
        T, idx = est["per_m"], est["index"]
        half = search_frac * T
        prm = []                                                                # etapa 4
        for row in rows0:
            cen, d, prof = lateral_profiles(tile, frame, idx, row, half, win_m)
            prm += list(window_peaks(prof, d, tile.px)[2])
        ref_prm = float(np.nanpercentile(prm, 75))
        rows, infos = [], []                                                    # etapa 5
        for row in rows0:
            nr, inf = adjust_row(tile, frame, idx, row, half, win_m, n_breaks, kink_deg, prom_ref=ref_prm)
            rows.append(nr)
            infos.append(inf)
        rows_ok = [r for r in rows if r.u1 - r.u0 >= 4.0]                       # etapa 6
        fl = detect_failures(tile, frame, idx, rows_ok, T, resid, thr, min_len)
        out.append({"cuartel": c, "px_factor": f, "ring": b["ring"], "area_ha": b["area_ha"], "est": est, "frame": frame,
                    "rows_ini": rows0, "rows": rows, "infos": infos, "rows_final": rows_ok,
                    "buffer": fl["half"] if fl else None, "fallas": fl["fallas"] if fl else [],
                    "ancho": fl["width"] if fl else None, "base": fl["base"] if fl else None,
                    "ref": fl["ref"] if fl else None, "n_samples": fl["n_samples"] if fl else 0,
                    "escala": fl["scale"] if fl else None, "dist_plantas": fl["plant_spacing"] if fl else None})
        log("  %d hileras, %d fallas" % (len(rows_ok), len(out[-1]["fallas"])))
        if progress:
            progress(c, len(blocks))
    return out
