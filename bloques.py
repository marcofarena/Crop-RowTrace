# -*- coding: utf-8 -*-
"""
Detección de cuarteles directamente de la imagen: un cuartel es una zona con
un patrón de hileras (periodicidad), no cualquier zona verde.

1. Se mide, en ventanas que se solapan, el espectro 2D del índice de
   vegetación (y del brillo): una zona con hileras tiene un pico estrecho a la
   frecuencia 1/T que sobresale de su anillo de frecuencia; el verdor sin
   patrón (monte, pasto) no.
2. Las ventanas con patrón se agrupan por rumbo y separación parecidos.
3. Para cada grupo se mide, píxel a píxel, cuánto del contraste local es ese
   patrón (demodulación local con un filtro de Gabor): ahí se ve dónde hay
   hileras y dónde no (calles, cabeceras, monte).
4. Cada componente conexa de esa máscara es un cuartel.
"""
import math

import numpy as np


def _angle_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def tile_patterns(img, valid, tile=128, stride=64, pmin_px=4.0, pmax_px=40.0,
                  min_valid=0.7):
    """
    Patrón periódico dominante en cada ventana.
    Devuelve una lista de dicts (y, x, ang, per, ring, frac): centro de la
    ventana en píxeles; ang = rumbo de las hileras en coordenadas de imagen
    (grados, mod 180); per = período en px; ring = cuántas veces sobresale el
    pico de la mediana de su anillo de frecuencia; frac = parte de la energía
    de la banda que está en el pico.
    """
    H, W = img.shape
    win = np.outer(np.hanning(tile), np.hanning(tile))
    fy = np.fft.fftfreq(tile)[:, None]
    fx = np.fft.fftfreq(tile)[None, :]
    fr = np.hypot(fx, fy)
    band = (fr >= 1.0 / pmax_px) & (fr <= 1.0 / pmin_px) & (fr <= 0.5)
    rbin = np.round(fr * tile).astype(np.int64)
    out = []
    for y in range(0, max(1, H - tile + 1), stride):
        for x in range(0, max(1, W - tile + 1), stride):
            v = valid[y:y + tile, x:x + tile]
            if v.mean() < min_valid:
                continue
            p = img[y:y + tile, x:x + tile].astype(np.float64)
            p = np.where(v, p, p[v].mean())
            P = np.abs(np.fft.fft2((p - p.mean()) * win)) ** 2
            m = np.where(band, P, 0.0)
            iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
            # frecuencia del pico con precisión sub-bin (centroide 3x3)
            ys = [(iy + d) % tile for d in (-1, 0, 1)]
            xs = [(ix + d) % tile for d in (-1, 0, 1)]
            blk = P[np.ix_(ys, xs)]
            fyk = fy[ys, 0][:, None] + 0 * blk
            fxk = fx[0, xs][None, :] + 0 * blk
            # corregir el salto de signo en el borde de la grilla
            fyk = np.where(np.abs(fyk - fy[iy, 0]) > 0.5, fyk - np.sign(fyk), fyk)
            fxk = np.where(np.abs(fxk - fx[0, ix]) > 0.5, fxk - np.sign(fxk), fxk)
            s = blk.sum()
            fyc = float((blk * fyk).sum() / s)
            fxc = float((blk * fxk).sum() / s)
            fc = math.hypot(fxc, fyc)
            if fc <= 0:
                continue
            ring_vals = P[(rbin == rbin[iy, ix]) & band]
            ring = float(P[iy, ix] / max(float(np.median(ring_vals)), 1e-30)) if ring_vals.size else 0.0
            frac = float(s / max(float(P[band].sum()), 1e-30))
            ang = (math.degrees(math.atan2(fyc, fxc)) + 90.0) % 180.0
            out.append({"y": y + tile // 2, "x": x + tile // 2, "ang": ang,
                        "per": 1.0 / fc, "ring": ring, "frac": frac})
    return out


def cluster_patterns(votes, min_ring=100.0, min_votes=8, ang_tol=4.0, per_tol=0.08):
    """
    Agrupa las ventanas con patrón por rumbo y período parecidos. Devuelve
    una lista de dicts (ang, per, n), del grupo más numeroso al menos. Un
    período que es la mitad (o un tercio) del de un grupo con el mismo rumbo
    es su armónico, no otro cuartel. Los grupos con pocas ventanas se
    descartan: una ventana aislada con un pico fuerte suele ser un camino o
    un cerco, no un cuartel.
    """
    vs = sorted((v for v in votes if v["ring"] >= min_ring), key=lambda v: -v["ring"])
    groups = []
    for v in vs:
        for g in groups:
            if (_angle_diff(v["ang"], g["ang"]) <= ang_tol
                    and abs(v["per"] - g["per"]) <= per_tol * g["per"]):
                g["m"].append(v)
                break
        else:
            groups.append({"ang": v["ang"], "per": v["per"], "m": [v]})
    out = []
    for g in groups:
        w = np.array([m["ring"] for m in g["m"]])
        # rumbo circular (mod 180) y período, ponderados por la fuerza del pico
        a2 = np.radians(2 * np.array([m["ang"] for m in g["m"]]))
        ang = (math.degrees(math.atan2(float((w * np.sin(a2)).sum()),
                                       float((w * np.cos(a2)).sum()))) / 2.0) % 180.0
        per = float(np.average([m["per"] for m in g["m"]], weights=w))
        out.append({"ang": ang, "per": per, "n": len(g["m"])})
    out.sort(key=lambda g: -g["n"])
    keep = []
    for g in out:
        harm = any(_angle_diff(g["ang"], k["ang"]) <= ang_tol and
                   any(abs(g["per"] - k["per"] / n) <= per_tol * k["per"] / n for n in (2, 3))
                   for k in keep)
        if g["n"] >= min_votes and not harm:
            keep.append(g)
    return keep


def pattern_ratio(img, valid, ang_deg, per_px, xx, yy):
    """
    Qué parte del contraste local de img es un patrón de franjas de rumbo
    ang_deg y período per_px (0 = nada de patrón, ~0.8 = hileras claras).
    Demodulación local: se multiplica por la portadora, se suaviza con una
    gaussiana de un período, y se compara la amplitud con la energía local.
    """
    import cv2
    th = math.radians(ang_deg)
    nx, ny = -math.sin(th), math.cos(th)
    fill = img.astype(np.float32).copy()
    fill[~valid] = float(img[valid].mean())
    hp = fill - cv2.GaussianBlur(fill, (0, 0), 2.5 * per_px)
    ph = (2 * math.pi / per_px) * (nx * xx + ny * yy)
    sig = per_px
    re = cv2.GaussianBlur(hp * np.cos(ph), (0, 0), sig)
    im = cv2.GaussianBlur(hp * np.sin(ph), (0, 0), sig)
    amp = 2.0 * np.hypot(re, im)
    e = np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), sig))
    # piso de energía: en una calle sin contraste el cociente no debe dispararse
    floor = 0.08 * float(np.sqrt((hp[valid] ** 2).mean()))
    r = amp / (math.sqrt(2.0) * np.maximum(e, floor))
    r[~valid] = 0.0
    return r


def detect_blocks(exg, brightness, valid, pixel_size, work_px=0.2,
                  period_range_m=(1.0, 8.0), min_area_ha=0.2,
                  thr_hi=0.5, thr_lo=0.3, info=None):
    """
    Cuarteles de una imagen: zonas con patrón de hileras.
    exg, brightness: índices a resolución completa (se usan los dos: algunas
    canopias se ven en el verdor y otras en el brillo); valid: máscara.
    Devuelve una lista de dicts (rings, ang, per_m, area_ha, ratio) con los
    polígonos en píxeles del raster ORIGINAL (x, y), de mayor a menor patrón.
    info (dict opcional) recibe los grupos de patrón encontrados.
    """
    import cv2
    f = max(1, int(round(work_px / pixel_size)))
    H0, W0 = valid.shape
    H, W = H0 // f, W0 // f

    def down(a):
        a = np.ascontiguousarray(a[:H * f, :W * f], dtype=np.float32)
        return cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)

    ex, br = down(exg), down(brightness)
    va = cv2.resize(valid[:H * f, :W * f].astype(np.float32), (W, H),
                    interpolation=cv2.INTER_AREA) > 0.999
    wpx = pixel_size * f
    pmin, pmax = period_range_m[0] / wpx, period_range_m[1] / wpx
    tile = 128 if min(H, W) >= 128 else 64
    votes = []
    for img in (ex, br):
        votes += tile_patterns(img, va, tile=tile, stride=tile // 2,
                               pmin_px=max(3.0, pmin), pmax_px=min(pmax, tile / 3.0))
    groups = cluster_patterns(votes)
    if info is not None:
        info["groups"] = [{"ang": g["ang"], "per_m": g["per"] * wpx, "n": g["n"]} for g in groups]
        info["work_px"] = wpx
    if not groups:
        return []
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ratios = []
    for g in groups:
        r = np.maximum(pattern_ratio(ex, va, g["ang"], g["per"], xx, yy),
                       pattern_ratio(br, va, g["ang"], g["per"], xx, yy))
        ratios.append(r)
    R = np.stack(ratios)
    best = np.argmax(R, axis=0)
    Rmax = np.max(R, axis=0)
    out = []
    min_px = min_area_ha * 10000.0 / (wpx * wpx)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for gi, g in enumerate(groups):
        own = (best == gi) & va
        lo = (own & (Rmax >= thr_lo)).astype(np.uint8)
        hi = (own & (Rmax >= thr_hi)).astype(np.uint8)
        lo = cv2.morphologyEx(lo, cv2.MORPH_OPEN, k3)
        n, lab = cv2.connectedComponents(lo, connectivity=8)
        for c in range(1, n):
            comp = lab == c
            if not hi[comp].any() or comp.sum() < min_px:
                continue
            m = comp.astype(np.uint8)
            # rellenar huecos chicos (un árbol, una mancha) sin tocar las calles
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (5, 5)))
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rings = []
            for ci, cn in enumerate(cnts):
                if len(cn) < 4:
                    continue
                eps = 0.25 * g["per"]
                ap = cv2.approxPolyDP(cn, eps, True)[:, 0, :].astype(float)
                # centro de píxel (x + 0.5) en la imagen reducida -> original
                rings.append([((x + 0.5) * f, (y + 0.5) * f) for x, y in ap])
            if not rings:
                continue
            out.append({"rings": rings, "ang": g["ang"], "per_m": g["per"] * wpx,
                        "area_ha": float(comp.sum()) * wpx * wpx / 10000.0,
                        "ratio": float(Rmax[comp].mean()), "n_votes": g["n"]})
    out.sort(key=lambda b: -b["area_ha"])
    return out
