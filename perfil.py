# -*- coding: utf-8 -*-
"""
Análisis de fallas por hilera dentro de un cuartel (ver cuarteles.py).

1. Transectas perpendiculares a las hileras cada 'step' metros. En cada una se
   muestrean las bandas del raster y se pliegan por la distancia con signo a la
   hilera más cercana: el promedio de todas las transectas da el perfil típico
   de una hilera (pico de vegetación, valle de entrehilera).
2. Con el perfil se elige el índice que mejor separa la hilera de la entrehilera
   y se define el ancho de la vegetación como el ancho del pico al 60 % de su altura
   (así la maleza de la entrehilera queda afuera).
3. A lo largo de cada hilera se promedia el índice en una franja de ese ancho,
   por ventanas de 'win_m' metros, y se normaliza entre el nivel de la
   entrehilera (0) y el de una planta típica del cuartel (1). Donde baja de
   'umbral' hay una falla (faltan plantas).
"""
import itertools
import math

import numpy as np

INDEX_NAMES = ["ExG", "VARI", "GLI", "NGRDI", "Oscuridad"]
AUTO_INDICES = ["ExG", "VARI", "GLI", "NGRDI"]   # solo verdor (la oscuridad mide sombra); con sufijo _r, la cresta del índice
BASE_INDICES = ["ExG", "VARI", "GLI", "NGRDI"]
PEAK_FRAC = 0.6      # el ancho de la vegetación se mide a esta fracción de la altura del pico
# Píxeles con brillo medio menor que esto son sombra: los índices normalizados
# (VARI, NGRDI, GLI, ExG cromático) son puro ruido ahí y pueden dar valores
# altos, así que se excluyen del muestreo.
SHADOW_MAX = 30.0
MAX_OFFSET = 1.0     # m, cuánto puede estar la franja verde corrida de la línea
BIN = 0.1            # m, ancho de clase del perfil transversal
HALF = 1.5           # m, el perfil va de -HALF a +HALF desde la hilera


def compute_index(name, R, G, B):
    """Índice con valores altos = más vegetación (a partir de RGB 0-255)."""
    eps = 1e-6
    if name == "ExG":
        s = R + G + B + eps
        return (2 * G - R - B) / s
    if name == "VARI":
        return np.clip((G - R) / (G + R - B + eps), -1, 1)
    if name == "GLI":
        return (2 * G - R - B) / (2 * G + R + B + eps)
    if name == "NGRDI":
        return (G - R) / (G + R + eps)
    if name == "Oscuridad":
        return 1.0 - (R + G + B) / (3.0 * 255.0)
    raise ValueError(name)


class Tile:
    """Recorte del raster en memoria; muestreo por vecino más próximo."""

    def __init__(self, R, G, B, valid, x0, y0, px, py):
        self.R, self.G, self.B, self.valid = R, G, B, valid
        self.x0, self.y0, self.px, self.py = x0, y0, px, py
        self.H, self.W = R.shape
        self.shadow_max = SHADOW_MAX

    def index_map(self, name, sigma_m=0.5):
        """Mapa del índice 'name' sobre todo el recorte (en caché). Con sufijo
        '_r' es la CRESTA del índice: el índice menos su promedio local
        (gaussiana de sigma_m). Atenúa el pasto o la maleza anchos y parejos de
        la entrehilera y deja las franjas angostas, como la vid."""
        cache = self.__dict__.setdefault("_maps", {})
        if name not in cache:
            ridge = name.endswith("_r")
            val = compute_index(name[:-2] if ridge else name, self.R.astype(np.float32),
                                self.G.astype(np.float32), self.B.astype(np.float32)).astype(np.float32)
            bright = (self.R.astype(np.float32) + self.G.astype(np.float32) + self.B.astype(np.float32)) / 3.0
            val = np.where(np.isfinite(val) & self.valid & (bright >= self.shadow_max), val, np.nan)
            if ridge:
                import cv2
                fill = np.where(np.isnan(val), np.nanmean(val), val).astype(np.float32)
                val = val - cv2.GaussianBlur(fill, (0, 0), max(1.0, sigma_m / self.px))
                val = np.where(self.valid & (bright >= self.shadow_max), val, np.nan)
            cache[name] = val
        return cache[name]

    def sample_index(self, name, x, y):
        """Valor del índice en las coordenadas (x, y) y máscara de válidos."""
        m = self.index_map(name)
        c = np.floor((x - self.x0) / self.px).astype(np.int64)
        r = np.floor((self.y0 - y) / self.py).astype(np.int64)
        ok = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)
        v = m[np.clip(r, 0, self.H - 1), np.clip(c, 0, self.W - 1)]
        ok &= np.isfinite(v)
        return v, ok

    def sample(self, x, y):
        c = np.floor((x - self.x0) / self.px).astype(np.int64)
        r = np.floor((self.y0 - y) / self.py).astype(np.int64)
        ok = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)
        c = np.clip(c, 0, self.W - 1)
        r = np.clip(r, 0, self.H - 1)
        ok &= self.valid[r, c]
        return (self.R[r, c].astype(np.float32), self.G[r, c].astype(np.float32),
                self.B[r, c].astype(np.float32), ok)


def _smooth(p):
    out = np.full(len(p), np.nan)
    for i in range(len(p)):
        w = p[max(0, i - 1):i + 2]
        w = w[~np.isnan(w)]
        if len(w):
            out[i] = w.mean()
    return out


def _profile(vals, d, min_pts):
    nb = int(round(2 * HALF / BIN))
    bi = np.clip(((d + HALF) / BIN).astype(np.int64), 0, nb - 1)
    cnt = np.bincount(bi, minlength=nb)
    sm = np.bincount(bi, weights=vals, minlength=nb)
    mean = np.where(cnt >= min_pts, sm / np.maximum(cnt, 1), np.nan)
    return _smooth(mean)


def _peak_stats(P, centers):
    """(valor del pico, índice del pico, nivel de entrehilera) de un perfil."""
    ok = ~np.isnan(P)
    cand = ok & (np.abs(centers) <= MAX_OFFSET)
    if cand.sum() < 3 or ok.sum() < 8:
        return None
    ip = int(np.argmax(np.where(cand, P, -np.inf)))
    vals = np.sort(P[ok])
    base = float(vals[:max(3, len(vals) // 5)].mean())
    return float(P[ip]), ip, base


def _movavg_nan(x, n):
    """Media móvil de n muestras que ignora NaN (NaN donde no hay ninguna)."""
    if n <= 1:
        return x
    ok = np.isfinite(x)
    k = np.ones(n)
    num = np.convolve(np.where(ok, x, 0.0), k, mode="same")
    den = np.convolve(ok.astype(float), k, mode="same")
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def _fwhm(P, centers, st):
    """Ancho del pico a PEAK_FRAC de su altura y posición de su centro (m)."""
    peak, ip, base = st
    thr = base + PEAK_FRAC * (peak - base)
    l = r_ = ip
    while l - 1 >= 0 and not np.isnan(P[l - 1]) and P[l - 1] >= thr:
        l -= 1
    while r_ + 1 < len(P) and not np.isnan(P[r_ + 1]) and P[r_ + 1] >= thr:
        r_ += 1
    return (float(centers[r_] - centers[l] + BIN),
            float(0.5 * (centers[l] + centers[r_])))


def _peaks(mc, cand, px, adj_m, need):
    """
    Picos de vegetación de una hilera. mc[u, c]: vigor de la franja centrada en
    el candidato lateral c (cand[c], m, relativo a la línea actual). En tramos
    de adj_m metros se toma el candidato de mayor vigor y solo se acepta si
    sobresale del peor al menos 'need' (si no hay planta no hay pico y no se
    persigue ruido ni maleza). Devuelve (muestra central de cada tramo con
    pico, posición lateral del pico, tramos, tramos con pico claro).
    """
    n_u = mc.shape[0]
    nw = max(8, int(round(adj_m / px)))
    ok = np.isfinite(mc)
    mz = np.where(ok, mc, 0.0)
    nwin = max(1, int(np.ceil(n_u / nw)))
    cen, dl = [], []
    for j in range(nwin):
        a, b = j * nw, min(n_u, (j + 1) * nw)
        cnt = ok[a:b].sum(0)
        if cnt.min() < 0.5 * (b - a):
            continue
        sc = mz[a:b].sum(0) / np.maximum(cnt, 1)
        if sc.max() - sc.min() < need:
            continue
        cen.append(0.5 * (a + b))
        dl.append(cand[int(np.argmax(sc))])
    return np.array(cen), np.array(dl), nwin, len(cen)


def _interp_peaks(cen, dl, n_u):
    """Corrimiento libre: mediana de 3 tramos e interpolación a lo largo de la hilera."""
    if len(cen) == 0:
        return np.zeros(n_u)
    if len(dl) >= 3:
        dl = np.array([np.median(dl[max(0, i - 1):i + 2]) for i in range(len(dl))])
    return np.interp(np.arange(n_u), cen, dl)


def _measure(mc, cand, delta, resid):
    """Vigor en la posición lateral delta (por muestra) +-resid de la línea de la que se muestreó mc."""
    ok = np.isfinite(mc)
    msk = np.abs(cand[None, :] - delta[:, None]) <= resid
    mm = np.where(msk & ok, mc, -np.inf).max(axis=1)
    return np.where(np.isfinite(mm), mm, np.nan)


def _fit_pwl(u, uc, t, n_breaks, kink_max):
    """
    Poligonal continua con a lo sumo n_breaks quiebres ajustada a las posiciones
    laterales t de los picos en uc (m a lo largo de la hilera), evaluada en u.
    Cada quiebre cambia la pendiente a lo sumo kink_max (m/m). Los quiebres
    se eligen entre las posiciones de los picos; se prueban todas las
    combinaciones hasta 3 quiebres (con más, se agregan de a uno). Pérdida
    robusta: un pico aislado que se aparta mucho no arrastra el ajuste.
    Devuelve (corrimiento en u, posiciones de los quiebres).
    """
    M = len(uc)
    if M == 0:
        return np.zeros(len(u)), []
    um = float(uc.mean())
    cap2 = 0.25 ** 2

    def design(knots, x):
        cols = [np.ones(len(x)), x - um] + [np.maximum(0.0, x - k) for k in knots]
        return np.array(cols).T

    def fit(knots, w=None):
        X = design(knots, uc)
        if w is None:
            beta = np.linalg.lstsq(X, t, rcond=None)[0]
        else:
            beta = np.linalg.lstsq(X * w[:, None], t * w, rcond=None)[0]
        res = t - X @ beta
        return beta, res

    def ok_kinks(beta):
        return all(abs(c) <= kink_max + 1e-9 for c in beta[2:])

    def loss(res):
        return float(np.minimum(res * res, cap2).sum())

    cands = list(uc[1:-1]) if M > 2 else []
    if len(cands) > 16:
        cands = [cands[i] for i in np.unique(np.linspace(0, len(cands) - 1, 16).round().astype(int))]
    best = (loss(fit([])[1]), [])
    chosen = []
    for n in range(1, n_breaks + 1):
        if n <= 3:
            combos = itertools.combinations(cands, n)
        else:
            combos = ([c for c in chosen] + [k] for k in cands if k not in chosen)
        best_n = None
        for kn in combos:
            kn = sorted(kn)
            beta, res = fit(kn)
            if not ok_kinks(beta):
                continue
            L = loss(res)
            if best_n is None or L < best_n[0]:
                best_n = (L, kn)
        if best_n is None:
            break
        chosen = list(best_n[1])
        # un quiebre más solo si mejora de verdad (evita quiebres por ruido)
        if best_n[0] < best[0] * 0.9 - 1e-9:
            best = best_n
    knots = list(best[1])
    # reponderar: los picos que se apartan mucho pesan menos
    beta, res = fit(knots)
    for _ in range(2):
        w = 1.0 / (1.0 + (res / 0.12) ** 2)
        beta2, res2 = fit(knots, w)
        if ok_kinks(beta2):
            beta, res = beta2, res2
    return design(knots, np.asarray(u, float)) @ beta, knots


def _transects(rows_u, rows_v, lo, hi, step, px, o, uvec, vvec, tile):
    """Muestras (d, x, y) de las transectas y su geometría."""
    nk = len(lo)
    span = hi.max() - lo.min()
    n = max(1, int(span // step))
    u_ts = lo.min() + (span - (n - 1) * step) / 2.0 + step * np.arange(n)
    ds, xs_, ys_, geoms = [], [], [], []
    for ut in u_ts:
        ids = np.where((lo <= ut) & (ut <= hi))[0]
        if len(ids) < 3:
            continue
        vs = np.array([np.interp(ut, rows_u[k], rows_v[k]) for k in ids])
        vs.sort()
        sp = float(np.median(np.diff(vs)))
        va, vb = vs[0] - sp / 2, vs[-1] + sp / 2
        vg = np.arange(va, vb, px)
        pos = np.clip(np.searchsorted(vs, vg), 1, len(vs) - 1)
        left, right = vs[pos - 1], vs[pos]
        near = np.where(np.abs(vg - left) <= np.abs(vg - right), left, right)
        d = vg - near
        x = o[0] + ut * uvec[0] + vg * vvec[0]
        y = o[1] + ut * uvec[1] + vg * vvec[1]
        R, G, B, ok = tile.sample(x, y)
        ds.append(d[ok]); xs_.append(x[ok]); ys_.append(y[ok])
        geoms.append(((x[0], y[0]), (x[-1], y[-1])))
    if not ds:
        return None
    return (np.concatenate(ds), np.concatenate(xs_), np.concatenate(ys_), geoms)


def analyze_block(b, info, tile, step=10.0, win_m=1.0, umbral=0.5,
                  index="auto", width=None, min_pts=30, min_falla=2.0, tol=None, adjust=True, adj_m=8.0,
                  ev_frac=0.35, resid=None, debug=False, n_iter=3, search_frac=0.35, n_breaks=None,
                  kink_deg=3.0, shadow_max=None):
    """
    b / info: de cuarteles.infer_blocks. tile: Tile cubriendo el cuartel.
    Devuelve un dict con el perfil, el ancho, y las ventanas por hilera, o
    {'error': texto}.
    """
    if shadow_max is not None:
        tile.shadow_max = float(shadow_max)
    o, uvec, vvec = info["origin"], info["uvec"], info["vvec"]
    U, V = info["U"], info["V"]
    rows = b["rows"]
    nk = len(rows)
    lo = np.asarray(b["row_lo"], float)
    hi = np.asarray(b["row_hi"], float)
    rows_u, rows_v = [], []
    for r in rows:
        pu = U[r].ravel()
        pv = V[r].ravel()
        order = np.argsort(pu)
        rows_u.append(pu[order])
        rows_v.append(pv[order])
    px = float(min(tile.px, tile.py))

    # --- 1. transectas y perfil transversal --------------------------------
    t = _transects(rows_u, rows_v, lo, hi, step, px, o, uvec, vvec, tile)
    if t is None:
        return {"error": "sin datos de raster en las transectas"}
    d, tx, ty, tgeoms = t
    centers = np.arange(-HALF + BIN / 2, HALF, BIN)

    # --- 2. índice, ancho de la vegetación ----------------------------------
    names = AUTO_INDICES if index == "auto" else [index]
    cands = []
    scores = {}
    for nm in names:
        vals, okv = tile.sample_index(nm, tx, ty)
        vals = vals[okv]
        P = _profile(vals, d[okv], min_pts)
        st = _peak_stats(P, centers)
        if st is None:
            continue
        sd = float(np.std(vals)) or 1e-9
        score = (st[0] - st[2]) / sd
        scores[nm] = round(score, 3)
        w_, off_ = _fwhm(P, centers, st)
        cands.append((score, nm, P, st, w_, off_))
    if not cands:
        return {"error": "perfil transversal sin datos suficientes"}
    cands.sort(key=lambda c: -c[0])
    score, name, P, (peak, ip, base), W, offset = cands[0]
    # el ancho y el corrimiento de la franja no deben depender de un empate
    # entre índices: mediana entre los de contraste comparable al mejor
    near = [c for c in cands if c[0] >= 0.8 * score]
    W = float(np.median([c[4] for c in near]))
    offset = float(np.median([c[5] for c in near]))
    thr = base + PEAK_FRAC * (peak - base)
    sp = float(b["spacing"])
    W = min(max(W, 0.2), 0.8 * sp)
    if width and width > 0:
        W = float(width)

    # --- 3. perfil a lo largo de cada hilera -------------------------------
    nW = max(3, int(round(W / px)))
    # Tolerancia de alineación: la línea puede estar corrida de la planta unos
    # centímetros (y variar a lo largo de la hilera). En cada punto se toma la
    # posición mejor alineada dentro de +-tol, así un corrimiento chico no se
    # confunde con una falla. Tiene que quedar lejos de la entrehilera para no
    # tomar la maleza: por defecto 0,2 separaciones (0,5 m a 2,5 m).
    if tol is None or tol < 0:
        tol = min(0.6, max(0.2, 0.2 * sp))
    # Banda de búsqueda del pico, más ancha que la tolerancia final: una planta a
    # 0,7-1 m de la línea (línea sobre la sombra) queda a medias fuera de una banda
    # de +-0,5 m, su pico dentro de ella es débil y no se la reconoce. 0,35 de la
    # separación llega hasta cerca de la mitad de la entrehilera sin pasarla.
    tol_s = max(tol, search_frac * sp) if adjust else tol
    nT = int(round(tol_s / px))
    nL = nW + 2 * nT
    offs = (np.arange(nL) - (nL - 1) / 2.0) * px
    k_smooth = max(1, int(round(0.3 / px)))
    cand = (np.arange(2 * nT + 1) - nT) * px
    if resid is None:
        resid = max(0.15, 0.3 * tol)
    # cuánto puede alejarse una hilera de su posición nominal (la línea de
    # entrada más el corrimiento global): nunca hasta la hilera vecina
    max_shift = float(min(1.2, 0.45 * sp))

    def sample_mc(k, u_s, shift):
        """Vigor de la franja de ancho W centrada en cada posición lateral
        (+-tol) alrededor de la línea de la hilera k desplazada 'shift' (por
        muestra, m). El buffer se toma SIEMPRE alrededor de la línea actual:
        con cada reajuste se vuelve a leer el raster en torno a la nueva."""
        vc = np.interp(u_s, rows_u[k], rows_v[k]) + offset + shift
        vv = vc[:, None] + offs[None, :]
        uu = np.broadcast_to(u_s[:, None], vv.shape)
        x = o[0] + uu * uvec[0] + vv * vvec[0]
        y = o[1] + uu * uvec[1] + vv * vvec[1]
        val, ok = tile.sample_index(name, x.ravel(), y.ravel())
        ok = (ok & np.isfinite(val)).reshape(vv.shape)
        val = np.where(ok.reshape(-1), val, 0.0).reshape(vv.shape)
        # media de cada franja de ancho W desplazada a lo largo de +-tol
        zc = np.zeros((len(u_s), 1))
        cs = np.concatenate([zc, np.cumsum(val, axis=1)], axis=1)
        cc = np.concatenate([zc, np.cumsum(ok, axis=1)], axis=1)
        sw = cs[:, nW:] - cs[:, :-nW]
        cw = cc[:, nW:] - cc[:, :-nW]
        mc = np.where(cw >= 0.5 * nW, sw / np.maximum(cw, 1), np.nan)
        return mc.astype(np.float32)

    def best_of(mc, lim=None):
        if lim is not None:
            mc = np.where(np.abs(cand)[None, :] <= lim + 1e-9, mc, np.nan)
        mm = np.where(np.isfinite(mc), mc, -np.inf).max(axis=1)
        return np.where(np.isfinite(mm), mm, np.nan)

    # pasada 0: buffer alrededor de la línea de entrada
    prof = []                                   # por hilera: (u_s, m_u)
    mc0 = []                                    # por hilera: vigor por candidato lateral
    for k in range(nk):
        u_s = np.arange(lo[k] + px / 2.0, hi[k], px)
        if len(u_s) < 2:
            prof.append(None)
            mc0.append(None)
            continue
        mc = sample_mc(k, u_s, 0.0)
        mc0.append(mc)
        prof.append((u_s, _movavg_nan(best_of(mc, tol), k_smooth)))
    allm = np.concatenate([p[1] for p in prof if p is not None])
    allm = allm[np.isfinite(allm)]
    if len(allm) == 0:
        return {"error": "sin datos de raster a lo largo de las hileras"}
    # nivel de una planta típica: el cuartel tiene más hilera con planta que sin ella
    ref = float(np.percentile(allm, 75))
    if ref - base < 1e-6:
        return {"error": "no hay contraste entre la hilera y la entrehilera"}

    # --- 3b. ajuste individual de cada hilera al pico de vegetación ---------
    # Se busca el pico dentro del buffer; si lo hay, la línea se corre hacia él
    # y el buffer se vuelve a tomar alrededor de la línea ya corrida, hasta que
    # el pico quede centrado (o se llegue a n_iter): así una hilera cuya línea
    # estaba a más de 'tol' de la planta también se alcanza.
    shifts = [None] * nk
    row_knots = [None] * nk
    kink_max = math.tan(math.radians(kink_deg))
    last_mc = list(mc0)
    adj_stats = {"tramos": 0, "con_pico": 0, "filas_movidas": 0, "desp_med": 0.0,
                 "iter_med": 0.0, "al_limite": 0, "quiebres": 0, "adjust": bool(adjust and nT > 0)}
    if adjust and nT > 0:
        need = ev_frac * (ref - base)
        meds, allm2, iters = [], [], []
        for k in range(nk):
            if mc0[k] is None:
                continue
            u_s = prof[k][0]
            shift = np.zeros(len(u_s))
            mc = mc0[k]
            it = 0
            knots_k = None
            while True:
                cen, dl, nwin_, n_ev = _peaks(mc, cand, px, adj_m, need)
                if it == 0:
                    adj_stats["tramos"] += nwin_
                    adj_stats["con_pico"] += n_ev
                if n_breaks is None:
                    target = shift + _interp_peaks(cen, dl, len(u_s))
                elif n_ev:
                    ci = cen.astype(int)
                    # posición absoluta del pico (respecto de la línea nominal) y ajuste de una
                    # poligonal con pocos quiebres: así el total de quiebres no crece con los reajustes
                    target, knots_k = _fit_pwl(u_s, u_s[ci], shift[ci] + dl, n_breaks, kink_max)
                else:
                    target = shift.copy()
                new = np.clip(target, -max_shift, max_shift)
                if it >= n_iter or np.max(np.abs(new - shift)) < 0.03:
                    m_u = _measure(mc, cand, new - shift, resid)   # el último corrimiento solo se aplica a la medida y al dibujo
                    shift = new
                    break
                shift = new
                mc = sample_mc(k, u_s, shift)
                it += 1
            iters.append(it)
            if n_breaks is not None:
                row_knots[k] = knots_k or []
                adj_stats["quiebres"] += len(row_knots[k])
            shifts[k] = shift
            last_mc[k] = mc
            prof[k] = (u_s, _movavg_nan(m_u, k_smooth))
            meds.append(float(np.median(np.abs(shift))))
            adj_stats["al_limite"] += int(np.sum(np.abs(shift) >= max_shift - 1e-6))
            allm2.append(prof[k][1])
        adj_stats["filas_movidas"] = int(sum(1 for m_ in meds if m_ > 0.1))
        adj_stats["desp_med"] = float(np.median(meds)) if meds else 0.0
        adj_stats["iter_med"] = float(np.mean(iters)) if iters else 0.0
        a2 = np.concatenate(allm2)
        a2 = a2[np.isfinite(a2)]
        if len(a2):
            # el nivel de una planta se mide ahora con la banda angosta de la hilera ajustada
            ref = float(np.percentile(a2, 75))
            if ref - base < 1e-6:
                return {"error": "no hay contraste entre la hilera y la entrehilera"}
    dbg = {k: (prof[k][0], prof[k][1], shifts[k], last_mc[k]) for k in range(nk) if debug and prof[k] is not None}
    mc0 = last_mc = None

    def shift_at(k, u):
        d = shifts[k]
        return 0.0 if d is None else float(np.interp(u, prof[k][0], d))

    win = {k: [] for k in ("row", "i", "ua", "ub", "z", "pres", "falla", "x1", "y1", "x2", "y2")}
    fallas = []
    n_valid = 0
    n_falla_len = 0.0
    for k in range(nk):
        if prof[k] is None:
            continue
        u_s, m_u = prof[k]
        z = (m_u - base) / (ref - base)
        valid = np.isfinite(z)
        low = valid & (z < umbral)
        n_valid += int(valid.sum())
        # rachas continuas sin planta
        inrun = np.zeros(len(u_s), bool)
        j = 0
        n = len(u_s)
        while j < n:
            if low[j]:
                e = j
                while e + 1 < n and low[e + 1]:
                    e += 1
                largo = (e - j + 1) * px
                if largo >= min_falla:
                    inrun[j:e + 1] = True
                    ua, ub = u_s[j] - px / 2, u_s[e] + px / 2
                    va = float(np.interp(ua, rows_u[k], rows_v[k])) + offset + shift_at(k, ua)
                    vb = float(np.interp(ub, rows_u[k], rows_v[k])) + offset + shift_at(k, ub)
                    fallas.append({
                        "row": k, "ua": ua, "ub": ub, "largo": largo,
                        "z": float(np.nanmean(z[j:e + 1])),
                        "ini_m": float(ua - lo[k]), "fin_m": float(hi[k] - ub),
                        "borde": int(j == 0 or e == n - 1 or ua - lo[k] < 2 * win_m
                                     or hi[k] - ub < 2 * win_m),
                        "x1": o[0] + ua * uvec[0] + va * vvec[0],
                        "y1": o[1] + ua * uvec[1] + va * vvec[1],
                        "x2": o[0] + ub * uvec[0] + vb * vvec[0],
                        "y2": o[1] + ub * uvec[1] + vb * vvec[1]})
                    n_falla_len += largo
                j = e + 1
            else:
                j += 1
        # ventanas de 'win_m' para el perfil
        wi = ((u_s - lo[k]) // win_m).astype(np.int64)
        nwin = int(wi[-1]) + 1
        if nwin > 1 and (hi[k] - lo[k]) - (nwin - 1) * win_m < 0.5 * win_m:
            wi = np.minimum(wi, nwin - 2)
            nwin -= 1
        zf = np.where(valid, z, 0.0)
        wn = np.bincount(wi, weights=valid.astype(float), minlength=nwin)
        wz = np.bincount(wi, weights=zf, minlength=nwin) / np.maximum(wn, 1)
        wp = np.bincount(wi, weights=(valid & (z >= umbral)).astype(float),
                         minlength=nwin) / np.maximum(wn, 1)
        wf = np.bincount(wi, weights=inrun.astype(float), minlength=nwin) / \
            np.maximum(np.bincount(wi, minlength=nwin), 1)
        for i in range(nwin):
            ua = lo[k] + i * win_m
            ub = hi[k] if i == nwin - 1 else lo[k] + (i + 1) * win_m
            va = float(np.interp(ua, rows_u[k], rows_v[k])) + offset + shift_at(k, ua)
            vb = float(np.interp(ub, rows_u[k], rows_v[k])) + offset + shift_at(k, ub)
            sin_dato = wn[i] < 0.5 * np.count_nonzero(wi == i)
            win["row"].append(k); win["i"].append(i)
            win["ua"].append(ua); win["ub"].append(ub)
            win["z"].append(np.nan if sin_dato else wz[i])
            win["pres"].append(np.nan if sin_dato else 100.0 * wp[i])
            win["falla"].append(0 if sin_dato else int(wf[i] >= 0.5))
            win["x1"].append(o[0] + ua * uvec[0] + va * vvec[0])
            win["y1"].append(o[1] + ua * uvec[1] + va * vvec[1])
            win["x2"].append(o[0] + ub * uvec[0] + vb * vvec[0])
            win["y2"].append(o[1] + ub * uvec[1] + vb * vvec[1])
    win = {k: np.array(v) for k, v in win.items()}
    total_len = n_valid * px
    adjusted = []
    for k in range(nk):
        if prof[k] is None:
            continue
        if row_knots[k] is not None:
            us = np.array([lo[k]] + [q for q in row_knots[k] if lo[k] < q < hi[k]] + [hi[k]])
        else:
            us = np.append(np.arange(lo[k], hi[k], 2.0), hi[k])
        vs = np.interp(us, rows_u[k], rows_v[k]) + offset + np.array([shift_at(k, u) for u in us])
        adjusted.append({"row": k, "dmed": float(np.median(np.abs(shifts[k]))) if shifts[k] is not None else 0.0,
                         "pts": [(float(o[0] + u * uvec[0] + v * vvec[0]), float(o[1] + u * uvec[1] + v * vvec[1]))
                                 for u, v in zip(us, vs)]})
    return {
        "index": name, "scores": scores, "contrast": float(score),
        "width": W, "offset": offset, "tol": tol, "search": tol_s, "base": base, "peak": peak, "ref": ref,
        "thr_pix": float(thr), "centers": centers, "profile": P,
        "win": win, "fallas": fallas, "transects": tgeoms, "n_rows": nk,
        "n_samples": int(len(d)),
        "falla_pct": 100.0 * n_falla_len / total_len if total_len else 0.0,
        "falla_m": n_falla_len, "row_m": total_len,
        "adjusted": adjusted, "adj": adj_stats, "resid": resid,
        "dbg": dbg if debug else None,
    }
