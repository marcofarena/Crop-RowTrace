# -*- coding: utf-8 -*-
"""
Inferencia del polígono de cada cuartel a partir de las hileras detectadas.

Idea: en el marco de las hileras (u a lo largo, v a través) cada cuartel es una
banda de hileras paralelas. Para no ajustarse a cada línea:
  1. se agrupan las hileras en cuarteles por proximidad (las calles de cabecera
     y los claros grandes separan cuarteles; las líneas sueltas se descartan);
  2. en cada cuartel, el extremo de cada hilera (inicio y fin) se reemplaza por
     la mediana de las hileras vecinas, de modo que una hilera corta por fallas
     no mete una muesca en el borde, y un saliente real (de varias hileras)
     se conserva;
  3. la envolvente nunca queda por dentro de una línea propia (el polígono
     encierra todas las líneas del cuartel).
"""
import math

import numpy as np


def _principal_angle(P):
    """Dirección dominante de las líneas (ángulo en [0, pi)), ponderada por largo."""
    d = P[:, 1] - P[:, 0]
    L = np.hypot(d[:, 0], d[:, 1])
    a = np.arctan2(d[:, 1], d[:, 0])
    return 0.5 * math.atan2(float((L * np.sin(2 * a)).sum()),
                            float((L * np.cos(2 * a)).sum()))


def _estimate_spacing(vm, lo, hi):
    """
    Separación típica entre hileras: para cada tramo, la distancia transversal
    a la hilera más cercana que se le solapa a lo largo (así cuadros de
    distinta separación en distintos sectores no se mezclan). Mediana.
    """
    order = np.argsort(vm)
    v, a, b = vm[order], lo[order], hi[order]
    n = len(v)
    best = []
    for i in range(n):
        j0 = np.searchsorted(v, v[i] - 8.0)
        j1 = np.searchsorted(v, v[i] + 8.0)
        dv = np.abs(v[j0:j1] - v[i])
        ov = (a[j0:j1] < b[i]) & (b[j0:j1] > a[i]) & (dv > 0.3)
        if ov.any():
            best.append(dv[ov].min())
    return float(np.median(best)) if best else None


def _nanmedian_window(a, half):
    out = np.full(len(a), np.nan)
    for k in range(len(a)):
        w = a[max(0, k - half):k + half + 1]
        w = w[~np.isnan(w)]
        if len(w):
            out[k] = np.median(w)
    return out


def infer_blocks(segs, spacing=None, gap=3.0, win=7, min_rows=8,
                 min_frac=0.1, v_close=2.2, tol_frac=0.3):
    """
    segs: lista de ((x1, y1), (x2, y2)) en un CRS proyectado (metros).
    Devuelve (bloques, info). Cada bloque es un dict con 'ring' (lista de
    (x, y) cerrada), 'n_rows', 'n_segs', 'length', 'ignored_rows',
    'filled_rows'. info trae el rumbo, la separación y lo descartado.
    """
    import cv2

    P = np.array(segs, dtype=float)              # (n, 2, 2)
    n = len(P)
    if n == 0:
        return [], {}
    th = _principal_angle(P)
    uvec = np.array([math.cos(th), math.sin(th)])
    vvec = np.array([-math.sin(th), math.cos(th)])
    o = P.reshape(-1, 2).mean(axis=0)
    U = (P - o) @ uvec
    V = (P - o) @ vvec
    vm = V.mean(axis=1)
    lo = U.min(axis=1)
    hi = U.max(axis=1)
    length = hi - lo

    s = spacing if spacing and spacing > 0 else _estimate_spacing(vm, lo, hi)
    if s is None:
        return [], {}

    # --- 1. agrupar en cuarteles con un raster en el marco (u, v) ----------
    cell = max(s / 5.0, (max(hi.max() - lo.min(), vm.max() - vm.min())) / 6000.0)
    u0, v0 = lo.min() - gap, vm.min() - 2 * s
    H = int((hi.max() + gap - u0) / cell) + 2
    W = int((vm.max() + 2 * s - v0) / cell) + 2
    img = np.zeros((H, W), np.uint8)
    for k in range(n):
        p1 = (int(round((V[k, 0] - v0) / cell)), int(round((U[k, 0] - u0) / cell)))
        p2 = (int(round((V[k, 1] - v0) / cell)), int(round((U[k, 1] - u0) / cell)))
        cv2.line(img, p1, p2, 255, 1)
    kh = max(1, int(round(gap / cell)))
    kw = max(1, int(round(v_close * s / cell)))
    dil = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)))
    _, labels = cv2.connectedComponents(dil, connectivity=8)
    seg_lab = np.array([
        labels[int(round(((U[k].mean()) - u0) / cell)),
               int(round((vm[k] - v0) / cell))] for k in range(n)])

    comps = {}
    for lab in np.unique(seg_lab):
        if lab == 0:
            continue
        idx = np.where(seg_lab == lab)[0]
        comps[lab] = idx
    tot = {lab: float(length[idx].sum()) for lab, idx in comps.items()}
    biggest = max(tot.values()) if tot else 0.0

    blocks = []
    dropped = {'comps': 0, 'segs': 0, 'length': 0.0}
    for lab, idx in comps.items():
        # --- 2. filas del cuartel ----------------------------------------
        order = idx[np.argsort(vm[idx])]
        rows = []                                  # lista de listas de segmentos
        for k in order:
            if rows and vm[k] - np.mean([vm[j] for j in rows[-1]]) <= 0.5 * s:
                rows[-1].append(k)
            else:
                rows.append([k])
        if len(rows) < min_rows or tot[lab] < min_frac * biggest:
            dropped['comps'] += 1
            dropped['segs'] += len(idx)
            dropped['length'] += tot[lab]
            continue
        rv = np.array([np.average(vm[r], weights=np.maximum(length[r], 1e-6))
                       for r in rows])
        nk = len(rows)
        sc = (rv[-1] - rv[0]) / (nk - 1) if nk > 1 else s
        row_lo = np.array([lo[r].min() for r in rows])
        row_hi = np.array([hi[r].max() for r in rows])

        # --- 3. envolvente robusta ---------------------------------------
        half = max(1, win // 2)
        env_lo = _nanmedian_window(row_lo, half)
        env_hi = _nanmedian_window(row_hi, half)
        fin_lo = np.minimum(row_lo, env_lo)
        fin_hi = np.maximum(row_hi, env_hi)
        ignored = int(np.sum((row_hi < fin_hi - 0.5 * s) | (row_lo > fin_lo + 0.5 * s)))
        filled = 0

        # --- 4. polígono en (u, v), simplificado y llevado a mapa ---------
        # cada hilera ocupa desde el punto medio con la vecina hasta el siguiente
        edges = np.empty(nk + 1)
        edges[1:-1] = 0.5 * (rv[:-1] + rv[1:])
        edges[0] = rv[0] - 0.5 * (rv[1] - rv[0] if nk > 1 else s)
        edges[-1] = rv[-1] + 0.5 * (rv[-1] - rv[-2] if nk > 1 else s)
        top, bot = [], []
        for k in range(nk):
            va, vb = edges[k], edges[k + 1]
            top += [(fin_hi[k], va), (fin_hi[k], vb)]
            bot += [(fin_lo[k], va), (fin_lo[k], vb)]
        ring_uv = top + bot[::-1]
        ring_uv.append(ring_uv[0])
        ring = [tuple(o + u * uvec + v * vvec) for u, v in ring_uv]
        blocks.append({
            'ring': ring, 'n_rows': len(rows), 'n_segs': len(idx),
            'length': tot[lab], 'ignored_rows': ignored, 'filled_rows': filled,
            'spacing': sc, 'seg_idx': idx.tolist(), 'tol': tol_frac * sc,
            # hileras ordenadas por v: segmentos, posición v y extremos u
            # suavizados (hasta donde llega el cuartel, no la línea detectada)
            'rows': [[int(j) for j in r] for r in rows], 'row_v': rv,
            'row_lo': fin_lo, 'row_hi': fin_hi,
        })
    info = {'bearing_deg': (math.degrees(th)) % 180, 'spacing': s,
            'dropped': dropped, 'origin': o, 'uvec': uvec, 'vvec': vvec,
            'U': U, 'V': V}
    return blocks, info


def block_polygon(b):
    """Polígono QgsGeometry de un bloque devuelto por infer_blocks."""
    from qgis.core import QgsGeometry, QgsPointXY, Qgis
    g = QgsGeometry.fromPolygonXY([[QgsPointXY(x, y) for x, y in b['ring']]])
    tol = float(b['tol'])
    # simplificar el escalón por hilera y agrandar lo mismo para que ninguna
    # línea quede apenas afuera por el recorte
    return g.simplify(tol).buffer(
        tol, 2, Qgis.EndCapStyle.Flat, Qgis.JoinStyle.Miter, 5.0)
