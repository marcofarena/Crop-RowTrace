# -*- coding: utf-8 -*-
"""
Detección de Hileras de Cultivo
--------------------------------
Flujo:
  1. Lee el raster de entrada con GDAL (bandas R, G, B elegidas por el
     usuario) y su máscara de píxeles válidos (alfa / nodata).
  2. Calcula el índice de vegetación ExG cromático (2G-R-B)/(R+G+B).
  3. Lote por lote (polígonos de la capa de contorno, cuarteles detectados
     en la imagen o toda el área con datos), estima orientación y separación
     de hileras con el espectro de potencia 2D del ExG (FFT, promedio de
     ventanas).
  4. Remuestrea el ExG a una grilla rotada donde las hileras quedan
     verticales; los picos del perfil perpendicular son los centros de
     hilera.
  5. A lo largo de cada hilera, marca como "hilera presente" los tramos
     donde el centro de la hilera es más verde que las dos entrehileras
     vecinas (contraste local, umbral relativo al ruido, con histéresis), o
     donde continúa su patrón de brillo; ubica los extremos a media altura,
     une huecos cortos y descarta tramos cortos.
  6. Otros cultivos: en frutales de copa cerrada ubica las hileras sobre
     las copas iluminadas (verde absoluto), no sobre la sombra entre ellas;
     con hileras finas busca la separación en el espectro blanqueado y, al
     límite de la resolución, marca las hileras por el patrón de varias
     hileras vecinas (modo patrón).
  7. Recorta las líneas al lote, las convierte a coordenadas del mapa con
     el geotransform del raster y escribe una capa vectorial. Opcionales:
     cuarteles detectados y fallas por hilera (fallas.py).
El método Canny + Hough (versiones 0.2-0.23) se sacó en la 0.24: con
hileras angostas y juntas encontraba rectas que cruzan las hileras (F1
medio 0,27 contra 0,98 en la tanda 2).

Dependencias extra (no vienen con QGIS por defecto):
  - opencv-python (o opencv-python-headless)
  - numpy (normalmente ya viene con QGIS)
Instalación (ver README.md del plugin).
"""

import math
import traceback

import numpy as np

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterBand,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterVectorDestination,
    QgsProcessingException,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFields,
    QgsField,
    QgsFeature,
    QgsFeatureRequest,
    QgsGeometry,
    QgsPointXY,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant, QCoreApplication

from .bloques import detect_blocks


def _angle_diff_mod180(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


# ---------------------------------------------------------------------------
# Método "perfil de proyección orientado por FFT"
#
# Por qué existe: con hileras angostas y muy juntas (p. ej. viñedo a
# ~0.2 m/px: hileras de 4-5 px de ancho con un período de ~9 px), Canny
# genera dos bordes por hilera separados por pocos píxeles, y HoughLinesP
# con un max_line_gap mayor que el período "salta" de una hilera a la
# siguiente: cualquier recta que CRUCE las hileras encuentra un píxel de
# borde cada pocos píxeles y la acepta como una línea continua. Esas rectas
# transversales son además más largas que las hileras reales (atraviesan
# toda la parcela), así que dominan la estimación de dirección ponderada
# por longitud — el detector termina eligiendo la dirección perpendicular
# a las hileras. Este método no usa bordes: mide la periodicidad del
# índice de vegetación directamente.
# ---------------------------------------------------------------------------

WHITE_MIN = 20.0  # cuánto tiene que sobresalir un pico de su anillo para ser un patrón periódico
# En el espectro blanqueado, además: potencia propia mínima, relativa a la
# del máximo crudo, y ciclos mínimos por ventana. El granulado del remuestreo
# de la imagen (2-4 px, casi paralelo a los ejes del raster) sobresale miles
# de veces de su anillo con una potencia ínfima: 0,003-0,02% del máximo en
# los lotes de prueba, contra 0,7-2% de las hileras (camellones y hortaliza
# fina). A baja frecuencia el anillo tiene pocos bins y un manchón puede
# sobresalir 20 veces de él.
WHITE_RAW_MIN = 0.002
WHITE_MIN_CYCLES = 8
# Con la distancia entre hileras indicada: fuerza mínima de un pico de la
# ventana de búsqueda, relativa al más fuerte de ella, para poder elegirlo.
HINT_MIN_FRAC = 0.1
# ... y entre los que están a ±HINT_TOL del valor indicado, el más fuerte
# (en un marco de 5 x 4 m, los 4 m quedan a -20%).
HINT_TOL = 0.1
# Marco de plantación (frutal en marco rectangular o tresbolillo): los
# árboles quedan alineados en dos direcciones y el espectro tiene dos picos
# fuertes, el de las hileras y el de la distancia entre plantas a lo largo
# de la hilera. Un segundo pico cuenta como el de las hileras si tiene al
# menos MARCO_MIN_FRAC de la potencia del máximo, forma al menos
# MARCO_MIN_ANG grados con él y su período es entre MARCO_RATIO veces
# mayor (la distancia entre hileras es mayor que la distancia entre
# plantas: por ahí pasa el tractor). Visto: frutal de 5 x 4 m con pasto,
# 4,02 m (plantas) 1,00 contra 4,97 m (hileras) 0,70 en el verdor.
MARCO_MIN_FRAC = 0.3
MARCO_MIN_ANG = 45.0
MARCO_RATIO = (1.1, 3.0)


def estimate_row_orientation_fft(img, valid, period_hint_px=None, tile_px=512, max_tiles=16, info=None):
    """Estima orientación y separación de hileras a partir del espectro de
    potencia 2D del índice de vegetación. Un patrón de franjas paralelas y
    equiespaciadas concentra su energía en un pico del espectro a la
    frecuencia 1/T, en la dirección PERPENDICULAR a las franjas: la
    posición de ese pico da a la vez el ángulo de las hileras y su
    separación T.

    Para no depender de una sola zona ni procesar el raster entero de una
    vez, promedia los espectros de varias ventanas cuadradas con suficientes
    píxeles válidos (método de Welch), con ventana de Hann para evitar el
    "leakage" de los bordes de cada ventana.

    period_hint_px (opcional): separación aproximada conocida; restringe la
    búsqueda del pico a [0.7, 1.4] veces ese valor. Sirve cuando otra
    periodicidad (p. ej. la distancia entre plantas A LO LARGO de la hilera)
    genera un pico más fuerte que el de las hileras.

    Si el máximo del espectro no sobresale de su anillo de frecuencia (fondo
    1/f de manchones grandes, típico con hileras finas), se toma el pico
    periódico del espectro blanqueado (ver abajo). info (dict opcional):
    recibe "blanqueado" (si se usó ese camino) y "sobre_anillo" (cuántas
    veces el pico elegido supera la mediana de su anillo).

    Devuelve (angulo_hilera_deg, periodo_px, contraste_pico), con el ángulo
    en coordenadas de imagen (x a la derecha, y hacia abajo), mod 180, y
    contraste_pico = potencia del pico / mediana de la potencia en la banda
    de búsqueda (cuánto sobresale el pico; valores bajos = sin
    periodicidad clara). Devuelve (None, None, 0.0) si no se puede
    estimar."""
    H, W = img.shape
    if not valid.any():
        return None, None, 0.0
    tile = int(min(tile_px, H, W))
    if tile < 32:
        return None, None, 0.0

    fill = float(img[valid].mean())
    base = np.where(valid, img, fill).astype(np.float64)

    step = max(1, tile // 2)
    ys = list(range(0, H - tile + 1, step))
    xs = list(range(0, W - tile + 1, step))
    if ys[-1] != H - tile:
        ys.append(H - tile)
    if xs[-1] != W - tile:
        xs.append(W - tile)
    cands = []
    for y in ys:
        for x in xs:
            frac = float(valid[y:y + tile, x:x + tile].mean())
            if frac > 0:
                cands.append((frac, y, x))
    if not cands:
        return None, None, 0.0
    good = sorted([(y, x) for frac, y, x in cands if frac >= 0.5])
    if good:
        # repartir las ventanas elegidas a lo largo de todo el lote
        idx = np.unique(np.linspace(0, len(good) - 1, min(max_tiles, len(good))).round().astype(int))
        chosen = [good[i] for i in idx]
    else:
        chosen = [(y, x) for _, y, x in sorted(cands, reverse=True)[:4]]

    win = np.outer(np.hanning(tile), np.hanning(tile))
    acc = np.zeros((tile, tile), dtype=np.float64)
    for y, x in chosen:
        p = base[y:y + tile, x:x + tile]
        p = (p - p.mean()) * win
        acc += np.abs(np.fft.fft2(p)) ** 2

    fy = np.fft.fftfreq(tile)[:, None]
    fx = np.fft.fftfreq(tile)[None, :]
    fr = np.hypot(fx, fy)
    if period_hint_px is not None and period_hint_px > 0:
        band = (fr >= 1.0 / (1.4 * period_hint_px)) & (fr <= 1.0 / (0.7 * period_hint_px))
    else:
        # al menos 3 ciclos dentro de la ventana, y período >= 2 px (Nyquist)
        band = fr >= 3.0 / tile
    band &= (fr > 0) & (fr <= 0.5)
    if not band.any():
        return None, None, 0.0

    masked = np.where(band, acc, 0.0)
    iy, ix = np.unravel_index(int(np.argmax(masked)), masked.shape)
    if not masked[iy, ix] > 0:
        # Sin potencia en la banda de búsqueda: la zona es uniforme (el negro
        # o el blanco de un recorte sin transparencia, un techo, agua) y no hay
        # patrón que medir. Sin este control el "pico" quedaba en la frecuencia
        # cero y el período daba una división por cero, que cortaba la corrida
        # entera (también los demás lotes).
        return None, None, 0.0

    def pk_period_angle(iy_, ix_):
        per_ = 1.0 / float(fr[iy_, ix_])
        ang_ = (math.degrees(math.atan2(float(fy[iy_, 0]), float(fx[0, ix_]))) + 90.0) % 180.0
        return per_, ang_

    # Espectro "blanqueado": potencia de cada frecuencia dividida por la
    # mediana de su anillo (misma frecuencia, todas las direcciones). Una
    # imagen de campo tiene un fondo de potencia que cae como 1/f (manchones
    # de vigor, de suelo, de riego, de decenas de metros), y en hileras
    # finas (hortalizas, a < 1 m) ese fondo supera en potencia al pico de
    # las hileras: el máximo crudo cae en un manchón. Un patrón periódico
    # sobresale de SU anillo (miles de veces en los lotes de prueba); un
    # manchón no (2-3 veces). Si el máximo crudo no sobresale de su anillo,
    # se busca el pico periódico en el espectro blanqueado: solo picos con
    # potencia propia (WHITE_RAW_MIN; el granulado de la imagen, de 2-4 px,
    # sobresale miles de veces de su anillo sin tener potencia) y de al
    # menos WHITE_MIN_CYCLES ciclos por ventana. De ellos se toma el más
    # fuerte y, entre los picos fuertes en su misma dirección a una
    # frecuencia n veces menor, el de menor frecuencia: la fundamental de
    # las hileras. Con un surco angosto entre camellones la fundamental es
    # más débil que sus armónicos (en un lote de camellones cada 5,1 m el
    # más fuerte era el cuarto armónico, 1,28 m). Antes se tomaba el de
    # menor frecuencia entre los fuertes de cualquier dirección, y ahí el
    # granulado, miles de veces más alto que los camellones, era el único
    # fuerte: el lote quedaba a 0,38 m y sin líneas.
    rbin = np.round(fr * tile).astype(np.int64)
    ring_med = np.ones(int(rbin.max()) + 1)
    # Los bins se ordenan por anillo una sola vez (antes, una máscara de toda
    # la ventana por cada anillo, ~360 veces: 0,2 s por llamada). Mismos
    # valores por anillo, misma mediana.
    acc_ring = acc.reshape(-1)[np.argsort(rbin, axis=None, kind="stable")]
    a_ = 0
    for r_, e_ in enumerate(np.cumsum(np.bincount(rbin.reshape(-1), minlength=len(ring_med)))):
        if e_ > a_:
            ring_med[r_] = max(float(np.median(acc_ring[a_:e_])), 1e-30)
        a_ = e_
    white = acc / ring_med[rbin]

    def local_max(m_):
        loc_ = m_ > 0
        for dy_ in (-1, 0, 1):
            for dx_ in (-1, 0, 1):
                if dy_ or dx_:
                    loc_ &= m_ >= np.roll(np.roll(m_, dy_, axis=0), dx_, axis=1)
        return np.nonzero(loc_)

    whitened = False
    if white[iy, ix] < WHITE_MIN:
        wb = np.where(band & (fr >= WHITE_MIN_CYCLES / tile) & (acc >= WHITE_RAW_MIN * float(acc[iy, ix])),
                      white, 0.0)
        pk = local_max(wb)
        strong = wb[pk] >= max(WHITE_MIN, 0.25 * float(wb.max()))
        pk = (pk[0][strong], pk[1][strong])
        if len(pk[0]):
            # vectores de frecuencia en bins de la ventana; un armónico n de
            # la fundamental cae a n veces su vector, con el error de la
            # grilla (~0,7 bin en cada uno)
            vy = fy[pk[0], 0] * tile
            vx = fx[0, pk[1]] * tile
            k0_ = int(np.argmax(wb[pk]))
            best_ = k0_
            f0_ = math.hypot(vy[k0_], vx[k0_])
            for k_ in range(len(pk[0])):
                fk_ = math.hypot(vy[k_], vx[k_])
                n_ = int(round(f0_ / max(fk_, 1e-9)))
                if n_ < 2 or fk_ >= math.hypot(vy[best_], vx[best_]):
                    continue
                d_ = min(math.hypot(n_ * vy[k_] - vy[k0_], n_ * vx[k_] - vx[k0_]),
                         math.hypot(n_ * vy[k_] + vy[k0_], n_ * vx[k_] + vx[k0_]))
                if d_ <= 0.7 * (n_ + 1):
                    best_ = k_
            iy, ix = int(pk[0][best_]), int(pk[1][best_])
            whitened = True

    # marco: la otra alineación de plantas (si la hay) y si se cambió la
    # elección por ella: por="marco" (se tomó la de mayor separación),
    # "separación indicada" (la más cercana al valor del usuario) o None
    # (las hileras ya eran el pico elegido; la otra alineación se guarda
    # igual, porque dice dónde están los árboles: ver lattice_amplitude).
    marco = None
    if not whitened:
        band_all = (fr >= 3.0 / tile) & (fr > 0) & (fr <= 0.5)
        pk_all = local_max(np.where(band_all, acc, 0.0))
        iy0, ix0 = iy, ix
        per0, ang0 = pk_period_angle(iy, ix)
        por = None
        if period_hint_px is not None and period_hint_px > 0:
            # Con la separación indicada por el usuario, entre los picos
            # de la ventana [0.7, 1.4] se toma el más fuerte a ±HINT_TOL de
            # ese valor, y si no hay ninguno, el más cercano (no el más
            # fuerte de la ventana: en un marco rectangular la distancia
            # entre plantas cae dentro de ella y ganaba; tampoco solo el más
            # cercano: a períodos largos la grilla del espectro es gruesa y
            # un pico débil en otra dirección puede quedar más cerca, visto
            # 12,55 m a 170° contra las hileras, 12,8 m a 37°, con 12 m
            # indicados). Solo picos verdaderos del espectro (máximos locales
            # sin recortar a la ventana; recortado, el flanco de un pico
            # fuerte de afuera quedaba como "máximo" en el borde de la
            # ventana: en ese olivar daba 7,8 m) y con al menos HINT_MIN_FRAC
            # de la fuerza del más fuerte de ellos. No se exige que sobresalga
            # de su anillo: el usuario ya dijo dónde buscar, y en ese olivar
            # las hileras apenas se veían en el verdor (3 veces su anillo) y
            # sí en el brillo.
            pk_b = pk_all
            inb = band[pk_b]
            pk = (pk_b[0][inb], pk_b[1][inb])
            ok_ = np.zeros(len(pk[0]), bool)
            if len(pk[0]):
                ok_ = acc[pk] >= HINT_MIN_FRAC * float(acc[pk].max())
                if not (iy, ix) in set(zip(pk[0].tolist(), pk[1].tolist())):
                    # el máximo de la ventana era un borde: se parte del
                    # pico verdadero más fuerte
                    k0_ = int(np.argmax(acc[pk]))
                    iy, ix = int(pk[0][k0_]), int(pk[1][k0_])
                    iy0, ix0 = iy, ix
                    per0, ang0 = pk_period_angle(iy, ix)
            if ok_.any():
                cy_, cx_ = pk[0][ok_], pk[1][ok_]
                dev_ = np.abs(1.0 / fr[cy_, cx_] - period_hint_px) / period_hint_px
                near_ = dev_ <= HINT_TOL
                if near_.any():
                    k_ = int(np.argmax(np.where(near_, acc[cy_, cx_], -1.0)))
                else:
                    k_ = int(np.argmin(dev_))
                per_, ang_ = pk_period_angle(int(cy_[k_]), int(cx_[k_]))
                if abs(per_ - per0) > 0.05 * per0 or _angle_diff_mod180(ang_, ang0) > 5.0:
                    iy, ix = int(cy_[k_]), int(cx_[k_])
                    por = "separación indicada"
        else:
            best_ = None
            for iy_, ix_ in zip(*pk_all):
                frac_ = float(acc[iy_, ix_] / acc[iy, ix])
                if frac_ < MARCO_MIN_FRAC or white[iy_, ix_] < WHITE_MIN:
                    continue
                per_, ang_ = pk_period_angle(iy_, ix_)
                if not (MARCO_RATIO[0] <= per_ / per0 <= MARCO_RATIO[1]):
                    continue
                if _angle_diff_mod180(ang_, ang0) < MARCO_MIN_ANG:
                    continue
                if best_ is None or frac_ > best_[0]:
                    best_ = (frac_, int(iy_), int(ix_))
            if best_ is not None:
                iy, ix = best_[1], best_[2]
                por = "marco"
        if por is not None:
            marco = dict(alt_period=per0, alt_angle=ang0, frac=float(acc[iy, ix] / acc[iy0, ix0]), por=por)
        else:
            # ¿hay otra alineación, la de las plantas a lo largo de la
            # hilera (período menor, en otra dirección)?
            per_r, ang_r = per0, ang0
            best_ = None
            for iy_, ix_ in zip(*pk_all):
                frac_ = float(acc[iy_, ix_] / acc[iy, ix])
                if frac_ < MARCO_MIN_FRAC or white[iy_, ix_] < WHITE_MIN:
                    continue
                per_, ang_ = pk_period_angle(iy_, ix_)
                if not (MARCO_RATIO[0] <= per_r / per_ <= MARCO_RATIO[1]):
                    continue
                if _angle_diff_mod180(ang_, ang_r) < MARCO_MIN_ANG:
                    continue
                if best_ is None or frac_ > best_[0]:
                    best_ = (frac_, per_, ang_)
            if best_ is not None:
                marco = dict(alt_period=best_[1], alt_angle=best_[2], frac=best_[0], por=None)

    period = 1.0 / float(fr[iy, ix])
    angle = (math.degrees(math.atan2(float(fy[iy, 0]), float(fx[0, ix]))) + 90.0) % 180.0
    bg = float(np.median(acc[band]))
    contrast = float(acc[iy, ix] / bg) if bg > 0 else float("inf")
    if info is not None:
        info["blanqueado"] = whitened
        info["sobre_anillo"] = float(white[iy, ix])
        info["marco"] = marco
    return angle, period, contrast


def planting_blocks(img, valid, angle_deg, period_px, tile_t=8.0, min_valid=0.7, max_px=4000):
    """Separación local de las hileras por ventanas de tile_t x tile_t
    separaciones, para avisar cuando un lote junta cuadros de plantación
    distintos. En cada ventana, la amplitud del patrón de franjas en el
    rumbo del lote a separaciones de 0,70 a 1,40 T (coeficiente de Fourier a
    lo ancho de las hileras, con hasta max_px píxeles al azar). Devuelve
    [(fracción válida, amplitud a T, separación más fuerte / T, su
    amplitud)] por ventana con al menos min_valid de píxeles válidos."""
    H, W = img.shape
    tile = max(16, int(round(tile_t * period_px)))
    th = math.radians(angle_deg)
    ratios = np.arange(0.70, 1.4001, 0.05)
    freqs = 1.0 / (ratios * period_px)
    k_T = int(np.argmin(np.abs(ratios - 1.0)))
    out = []
    for r0 in range(0, H - tile // 2, tile):
        for c0 in range(0, W - tile // 2, tile):
            m = valid[r0:r0 + tile, c0:c0 + tile]
            if m.size == 0 or m.mean() < min_valid:
                continue
            ys, xs = np.nonzero(m)
            if len(xs) > max_px:
                # muestra al azar (fija): un paso regular puede alinearse con
                # las hileras
                sel = np.random.default_rng(r0 * 7919 + c0).choice(len(xs), max_px, replace=False)
                ys, xs = ys[sel], xs[sel]
            v = img[r0:r0 + tile, c0:c0 + tile][ys, xs].astype(np.float64)
            v = v - v.mean()
            tt = -(xs + c0) * math.sin(th) + (ys + r0) * math.cos(th)
            amps = np.abs(np.exp(-2j * math.pi * np.outer(freqs, tt)) @ v) / len(v)
            k = int(np.argmax(amps))
            out.append((float(m.mean()), float(amps[k_T]), float(ratios[k]), float(amps[k])))
    return out


def refine_row_angle(img, valid, angle0_deg, period_px, span_deg=1.5, max_samples=2000000, highpass=False):
    """Afina el ángulo de las hileras buscando, alrededor de angle0_deg, el
    ángulo que maximiza el contraste del perfil perpendicular (varianza
    del ExG medio en función de la distancia perpendicular). Con el ángulo
    exacto, cada banda del perfil cae entera sobre hilera o entera sobre
    entrehilera y el perfil tiene máximo contraste; un error de ángulo
    mezcla ambas a lo largo de la hilera y lo aplana. Búsqueda de grueso a
    fino (3 niveles). La resolución angular de la FFT por ventanas es de
    ~T/ventana radianes (~1° en el caso típico); esto la lleva a centésimas
    de grado, lo que importa en hileras largas."""
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return angle0_deg
    stride = max(1, len(xs) // max_samples)
    ys, xs = ys[::stride], xs[::stride]
    vals = img[ys, xs].astype(np.float64)
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    bin_w = max(period_px / 12.0, 0.25)
    # highpass: se mide la varianza del perfil SIN su media móvil de largo
    # 2T. Esa media no contiene nada del patrón de hileras (un filtro de
    # caja de dos períodos anula la fundamental y sus armónicos) y sí los
    # manchones de decenas de metros, que con hileras finas dominan la
    # varianza y fijan el ángulo (visto: 8,1° en vez de 7,1° con hileras a
    # 0,85 m). Solo para hileras finas: en viñedos y frutales el patrón de
    # hileras domina igual y el ángulo no cambia más que centésimas de grado.
    n_ma = max(3, int(round(2.0 * period_px / bin_w)))
    box_ma = np.ones(n_ma)

    def score(a_deg):
        th = math.radians(a_deg)
        t = -xs * math.sin(th) + ys * math.cos(th)
        k = ((t - t.min()) / bin_w).astype(np.int64)
        c = np.bincount(k).astype(np.float64)
        s = np.bincount(k, vals)
        m = c > 0
        prof = s[m] / c[m]
        if highpass:
            ma = np.convolve(s, box_ma, "same") / np.maximum(np.convolve(c, box_ma, "same"), 1e-9)
            prof = prof - ma[m]
        w = c[m]
        mu = (prof * w).sum() / w.sum()
        return float((w * (prof - mu) ** 2).sum() / w.sum())

    best = float(angle0_deg)
    span, step = float(span_deg), float(span_deg) / 10.0
    for _ in range(3):
        cands = best + np.arange(-span, span + 1e-9, step)
        scores = [score(a) for a in cands]
        best = float(cands[int(np.argmax(scores))])
        span, step = step, step / 5.0
    return best % 180.0


def chroma_exg(cv2_module, r, g, b, valid, blur_sigma=1.0):
    """ExG cromático (2G-R-B)/(R+G+B) con las bandas suavizadas blur_sigma
    px (0 = sin suavizar); 0 fuera de `valid`. Ver processAlgorithm."""
    if blur_sigma and blur_sigma > 0:
        r, g, b = (cv2_module.GaussianBlur(x, (0, 0), blur_sigma) for x in (r, g, b))
    total = r + g + b
    out = (2.0 * g - r - b) / np.maximum(total, 1e-6)
    out[(total <= 0) | ~valid] = 0.0
    return out


def perp_profiles(imgs, valid, angle_deg, period_px, max_samples=2000000):
    """Perfil perpendicular a las hileras de cada imagen de `imgs` (bins de
    T/12 a lo ancho, suavizado gaussiano de 1,5 bins = T/8, solo bins con
    datos) y los máximos locales en ±T/2 del primero (las hileras según ese
    índice). Devuelve (perfiles, ancho de bin, medio período en bins,
    máximos), o None si no hay píxeles válidos."""
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return None
    stride = max(1, len(xs) // max_samples)
    ys, xs = ys[::stride], xs[::stride]
    th = math.radians(angle_deg)
    t = -xs * math.sin(th) + ys * math.cos(th)
    bw = max(period_px / 12.0, 0.25)
    k = ((t - t.min()) / bw).astype(np.int64)
    cnt = np.bincount(k).astype(np.float64)
    gk = np.exp(-0.5 * (np.arange(-5, 6) / 1.5) ** 2)
    ok = cnt >= 0.2 * float(np.median(cnt[cnt > 0]))
    w = np.convolve(ok.astype(np.float64), gk, "same")
    profs = []
    for img in imgs:
        s = np.bincount(k, img[ys, xs].astype(np.float64), minlength=len(cnt))
        p = np.where(ok, s / np.maximum(cnt, 1), 0.0)
        profs.append(np.convolve(p, gk, "same") / np.maximum(w, 1e-9))
    p0 = profs[0]
    half = int(round(period_px / 2.0 / bw))
    peaks = [i for i in range(half, len(p0) - half)
             if ok[i - half:i + half + 1].all() and p0[i] >= p0[i - half:i + half + 1].max()]
    return profs, bw, half, peaks


def stripe_levels(imgs, valid, angle_deg, period_px, max_samples=2000000):
    """Niveles de cada imagen de `imgs` en las franjas donde la primera
    tiene sus máximos (las hileras según ese índice) y a mitad de camino
    entre máximos vecinos (perfil de perp_profiles); mitades solo entre
    máximos separados 0,7-1,4 T. Devuelve ([medianas en los máximos],
    [medianas en las mitades], número de máximos), o None si hay menos de 3
    máximos."""
    pp = perp_profiles(imgs, valid, angle_deg, period_px, max_samples)
    if pp is None or len(pp[3]) < 3:
        return None
    profs, bw, _, peaks = pp
    mids = [int(round(0.5 * (a + b))) for a, b in zip(peaks[:-1], peaks[1:])
            if 0.7 * period_px <= (b - a) * bw <= 1.4 * period_px]
    if len(mids) < 2:
        return None
    at_pk = [float(np.median(p[peaks])) for p in profs]
    at_mid = [float(np.median(p[mids])) for p in profs]
    return at_pk, at_mid, len(peaks)


def stripe_offset(c_img, y_img, valid, angle_deg, period_px, max_samples=2000000):
    """Dónde cae la franja más OSCURA (mínimo de y_img, el brillo) respecto
    de la más VERDE (máximo de c_img, el verdor cromático). Mismo perfil
    perpendicular que stripe_levels (perp_profiles); para cada máximo del
    verdor se busca el mínimo del brillo dentro de ±T/2. Devuelve (verdor
    mediano en las franjas más verdes, desplazamiento mediano de la franja
    oscura en fracciones de T, entre 0 y 0,5, y el mismo desplazamiento con
    signo, en la coordenada perpendicular -x sen + y cos), o None si hay
    menos de 3 máximos."""
    pp = perp_profiles((c_img, y_img), valid, angle_deg, period_px, max_samples)
    if pp is None or len(pp[3]) < 3:
        return None
    (pc, py), bw, half, peaks = pp
    offs = [(int(np.argmin(py[i - half:i + half + 1])) - half) * bw / period_px for i in peaks]
    return float(np.median(pc[peaks])), abs(float(np.median(offs))), float(np.median(offs))


def lattice_amplitude(img, valid, angle_deg, period_px, plant_px, ref_img, nph=12, max_samples=2000000):
    """Cuánto "late" `img` a lo largo de la hilera al período entre plantas
    (plant_px), en la franja más alta de `ref_img` (la hilera según ese
    índice) y a media separación. Por hilera y por fase a lo ancho (nph
    fases por período) se toma |suma de img * exp(-2 pi i s / plant_px)| / N
    (s = posición a lo largo de la hilera) y se promedia entre hileras: la
    fase de los árboles cambia de una hilera a otra (tresbolillo) y así no
    se cancela. Devuelve (amplitud en la franja más alta, amplitud a media
    separación), o None si no hay datos."""
    ys, xs = np.nonzero(valid)
    if len(xs) == 0 or plant_px <= 0:
        return None
    stride = max(1, len(xs) // max_samples)
    ys, xs = ys[::stride], xs[::stride]
    th = math.radians(angle_deg)
    s = xs * math.cos(th) + ys * math.sin(th)
    tt = -xs * math.sin(th) + ys * math.cos(th)
    ph = np.minimum((((tt / period_px) % 1.0) * nph).astype(np.int64), nph - 1)
    row = np.floor(tt / period_px).astype(np.int64)
    row -= row.min()
    ref = ref_img[ys, xs].astype(np.float64)
    prof = np.bincount(ph, ref, nph) / np.maximum(np.bincount(ph, minlength=nph), 1)
    pk = int(np.argmax(prof))
    mid = (pk + nph // 2) % nph
    v = img[ys, xs].astype(np.float64)
    v -= v.mean()
    e = np.exp(-2j * math.pi * s / plant_px)
    n_min = max(20.0, 2.0 * plant_px / stride)

    def amp(b):
        out = []
        for b_ in (b - 1, b, b + 1):
            m = ph == (b_ % nph)
            if not m.any():
                continue
            re = np.bincount(row[m], (v[m] * e[m]).real)
            im = np.bincount(row[m], (v[m] * e[m]).imag)
            nn = np.bincount(row[m])
            ok = nn >= n_min
            if ok.any():
                out.append(float(np.mean(np.hypot(re[ok], im[ok]) / nn[ok])))
        return float(np.mean(out)) if out else None

    a_pk, a_mid = amp(pk), amp(mid)
    if a_pk is None or a_mid is None:
        return None
    return a_pk, a_mid


LEVEL_MIN = 0.5   # un tramo: al menos a mitad de camino entre entrehilera y hilera típicas (nivel)
# Tope de nivel en las posiciones agregadas (grilla del marco, continuidad:
# las hileras de borde, que se buscan justamente porque son más débiles):
# un tramo ahí con más de LEVEL_ADDED_MAX veces el contraste de una hilera
# típica no es la hilera de borde sino una cortina o una fila de árboles
# junto al lote. Medido: cortina junto a un viñedo 10,7; posiciones
# agregadas en 31 lotes de viñedo, frutal y hortaliza, hasta 2,1.
LEVEL_ADDED_MAX = 4.0
# Hilera extrema del lote con pico propio (ver "Fila de árboles distintos en
# el borde"): se descarta si su nivel supera EDGE_LEVEL_MIN y EDGE_LEVEL_REL
# veces el percentil 99 de las hileras interiores. Medido: árboles de la
# calle junto a un frutal 31,8 (interiores hasta 3,5); fila de árboles junto
# a un viñedo 3,8 (interiores hasta 1,2); hileras extremas reales en 31
# lotes, hasta 2,1 (y 3,8 en un viñedo casi negro cuyas interiores llegan a
# 5,8).
EDGE_LEVEL_MIN = 3.0
EDGE_LEVEL_REL = 2.5
# Rescate de tramos de plantas chicas o ralas (ver "Nivel" en el paso 5):
# en el BRILLO tienen que ser algo más oscuras que la entrehilera típica
# (al menos LEVEL_DARK_MIN del contraste hilera-entrehilera: copa y sombra
# de un árbol, aunque sea chico) y en el otro índice no pueden quedar más
# allá de la entrehilera, del lado del suelo desnudo, en más de
# LEVEL_FLOOR. Medido en la finca de prueba: árboles chicos de borde de
# frutal +0,27 a +0,47 en el brillo; arbustos claros sueltos sobre un
# sendero -0,11 a +0,12; huellas y bordes sobre la calle -1,0 a -2,4 en uno
# de los dos.
LEVEL_DARK_MIN = 0.2
LEVEL_FLOOR = -0.5
# Con las copas iluminadas como hilera (copa cerrada), la "entrehilera" de
# referencia es la sombra entre copas, que también es vegetación: la escala
# hilera-entrehilera se comprime y una hilera de menor vigor queda en ~0,35.
# El suelo de una calle queda por DEBAJO de esa sombra (fracción < 0), así
# que el umbral más bajo no deja pasar calles.
LEVEL_MIN_SUNLIT = 0.25
# Modo patrón: se activa si menos de esta fracción de las hileras ubicadas
# tiene evidencia propia; promedia tantas hileras vecinas a lo largo de
# tantas separaciones.
PATTERN_MAX_FRAC = 0.3
# Seguimiento lateral de hileras levemente torcidas: corrimiento máximo
# (fracción de T) y niveles a cada lado; costo por nivel de corrimiento y
# por cambio de nivel, por bloque de largo T, en unidades de ruido; recorte
# de la evidencia (una planta muy vigorosa o un hueco no deciden solos).
TRACK_MAX_T = 0.3
# ... y nunca más de TRACK_MAX_M metros: lo que se aparta una hilera
# plantada (a mano, o por la distorsión del ortomosaico) es una cantidad
# en metros, no una fracción de la separación. En una vid a 2 m, 0,6 m son
# 0,3 T (el caso visto); en un frutal a 6 m, 0,3 T serían 1,8 m y el camino
# se enganchaba en arbustos al costado de la hilera (visto: una hilera que
# cruzaba una calle y seguía 18 m dentro de un manchón de arbustos). Con
# copas de 2-4 m, un apartamiento de 0,6 m casi no cambia el contraste.
TRACK_MAX_M = 0.6
TRACK_LEVELS = 3
TRACK_PRIOR = 1.0
TRACK_STEP = 1.0
TRACK_CLIP = (-4.0, 8.0)
# La línea de un tramo se reajusta al camino seguido solo si éste se aparta
# de la recta al menos esta fracción de T (las demás quedan como antes).
TRACK_REFIT_T = 0.2
# Sin seguimiento con hileras de menos de estos píxeles de separación: con
# 4-5 px y contraste al nivel del ruido (hortaliza en Google) el camino
# persigue el ruido.
TRACK_MIN_T_PX = 6.0
# Hueco máximo dentro de una hilera: al menos GAP_T separaciones entre
# hileras (ver el paso 5 de detect_rows_by_profile). En viñedo (2-2,6 m) manda
# el hueco máximo en metros (7,5 m); en frutales más anchos, este piso: a 5 m
# entre hileras, 12 m (3-4 árboles faltantes no cortan la hilera). Con 1,25
# (hasta la 0.24) y 7,5 m, el frutal de Finca 500 se cortaba donde faltaban
# árboles (F1 0,966); con 2,4, 0,978. En las 23 referencias, 0,974 -> 0,975
# (sube un olivar a 12 m 0,936 -> 0,940 y un frutal a 4,9 m 0,951 -> 0,953;
# bajan 0,001 dos frutales a 5-7 m, que unen huecos cortados en su referencia).
# Con 2,0 y 2,2 el olivar bajaba a 0,930.
GAP_T = 2.4
PATTERN_ROWS = 5
PATTERN_LEN_T = 10.0
# Entrehileras alternadas (rastra o cobertura hilera por medio): en la
# imagen auxiliar, la diferencia entre la entrehilera derecha y la izquierda
# de cada hilera (mediana, en ruidos) cambia de signo de una hilera a la
# vecina. Si hay pares con |diferencia| >= ALT_D en al menos ALT_PAIRS de
# las hileras y en al menos ALT_OPP de ellos el signo se invierte, del lado
# cubierto la vid y la cobertura se ven casi iguales, y la hilera tiene que
# superar a cada entrehilera en la imagen auxiliar salvo ALT_TOL ruidos.
# Medido en los 43 lotes de prueba: pasan 4 (dos viñedos con rastra hilera
# por medio, 0,83-0,88 de las hileras en pares y 0,99-1,00 invertidos, un
# viñedo chico y el lote de franjas rojizas, 0,77-0,93 y 0,93-0,96); en los
# demás, menos de 0,66 en pares o menos de 0,7 invertidos.
ALT_D = 2.0
ALT_PAIRS = 0.5
ALT_OPP = 0.9
ALT_TOL = 3.0
# Banda oscura asimétrica (franja oscura elegida porque la canopia no se ve
# verde: vid sin hojas, de otoño o rojiza): lo más oscuro es la sombra, en un
# borde de la banda, y la planta queda del otro lado, menos oscura. La línea de
# salida va al centro a media altura de la banda en vez de a su pico. Se mide
# en el perfil a lo ancho de cada hilera y se aplica a todo el lote la mediana,
# solo si es al menos HALF_MIN_T * T y al menos HALF_SAME de las hileras van
# para el mismo lado (la sombra cae siempre del mismo lado); como mucho
# HALF_MAX_T * T. En las referencias de la tanda 2, la línea del usuario cae en
# ese centro (0,00 y 0,04 T de la banda media) y no en lo más oscuro (0,35 y
# 0,21 T). En los demás lotes la mediana es < 0,08 T o va para los dos lados.
HALF_MIN_T = 0.1
HALF_SAME = 0.9
HALF_MAX_T = 0.25
# Siembra por evidencia débil sostenida: una racha de evidencia débil (la
# hilera más marcada que sus DOS entrehileras y > 1 ruido) de al menos
# WEAK_SEED_T separaciones y con evidencia media >= WEAK_SEED_M ruidos siembra
# un tramo aunque no tenga una racha fuerte (> k_sigma ruidos) de largo T. El
# contraste está suavizado a lo largo de T: en 3T hay ~3 valores casi
# independientes, y que el ruido los dé los tres > 1 con media 1,5 es raro.
# Visto en la tanda 2: viñedos sin hojas y un frutal con hileras parejas pero
# tenues (2-3 ruidos) cortadas en pedazos o sin línea; frutal F1 0,68 -> 0,94,
# viñedo 0,72 -> 0,79; ningún lote con referencia empeora. Riesgo: en un lote
# con cobertura muy cargada (descartado por la imagen) siembra franjas de
# cobertura de la entrehilera (170 m de líneas falsas en 1 ha).
WEAK_SEED_T = 3.0
WEAK_SEED_M = 1.5
# Centro de la banda (franja oscura): cada hilera se ubica en el centro a media
# altura de su banda y no en su pico (lo más oscuro), si así se ve mejor. Casos:
# vid sin hojas (lo más oscuro es la sombra, a un costado de la planta) y
# hortícola en camellones (la banda oscura entre dos líneas claras tiene dos
# jorobas y el pico cae en una de ellas, a ~T/4 del centro). Desde el pico, una
# de las entrehileras (±T/2) cae dentro de la banda y el contraste casi
# desaparece: hileras sin línea o cortadas. Se decide por lote: el largo de
# hilera presente (pasada 1) en el centro tiene que superar al del pico en la
# fracción band_gain (ver _detect_rows_profile). Corrimiento máximo BAND_MAX_T.
# Visto en la tanda 2: hortícola F1 0,62 -> 0,89; vid sin hojas 0,75 -> 0,96 y
# 0,79 -> 0,98. En un frutal y un viñedo de canopia verde la línea del usuario
# está en el pico (la copa) y el centro la corría 0,3-0,5 m: ahí no se aplica.
BAND_MAX_T = 0.35
BAND_MOVED_T = 0.05      # una hilera "se movió" si el centro dista más que esto del pico
# Rumbo propio por hilera: el ángulo es uno por lote, pero las hileras de una
# plantación no son exactamente paralelas (en la tanda 2, hasta 1,5° de
# diferencia dentro de un lote: 0,8 T de deriva en 70 m). La recta coincidía con
# la hilera solo en el medio y la línea se cortaba a mitad de hilera; con
# hileras cortas (< 4 bloques de la pasada 2) la pendiente propia no se
# ajustaba nunca. Antes de la pasada 2, cada hilera prueba inclinaciones de
# hasta SLOPE_MAX_T * T de deriva a lo largo de su largo, en pasos de
# T / SLOPE_RES, sobre el perfil de columnas de la grilla inclinada (una sola
# inclinación de la imagen entera por paso). Se queda con la inclinación si su
# contraste supera al de la recta del lote en SLOPE_GAIN, si al menos SLOPE_NB
# de sus 4 vecinas eligieron una inclinación a no más de SLOPE_NB_K pasos (el
# rumbo cambia de a poco a lo ancho del lote; una inclinación suelta es ruido:
# en un viñedo de hileras finas inclinaba hileras sueltas y cortaba puntas) y si
# su recta no se acerca a menos de T/2 de la de una vecina (visto: líneas
# duplicadas sobre la hilera de al lado). Tanda 2: viñedo 0,86 -> 0,96,
# hortícola 0,89 -> 0,95.
SLOPE_RES = 8.0
SLOPE_MAX_T = 1.0
SLOPE_GAIN = 0.1
SLOPE_NB = 2
SLOPE_NB_K = 2
# Hileras con mucho más rumbo propio (en la tanda 2, un hortícola con una franja
# de 18 hileras a ~2° del resto: 1,5-1,75 T de deriva en 100 m, cada línea
# empezaba en una hilera y terminaba en la siguiente). Dos fallas: (1) a mitad
# de hilera la recta del lote cae entre dos hileras reales (a ±T/2) y la
# búsqueda a ±T/4 se quedaba con una inclinación chica y falsa; se descarta si
# a ±T/2 hay una alineación SLOPE_PHASE_GAIN mejor. (2) La inclinación
# necesaria pasa la de la búsqueda propia (SLOPE_MAX_T): una hilera sin rumbo
# propio junto a una que lo tomó busca su hilera a una separación de la recta
# de la vecina, con una inclinación parecida, hasta SLOPE_PROP_MAX_T de deriva
# (propagación). Ampliar la búsqueda propia a 2 T empeoraba un viñedo (0,964 ->
# 0,959: hileras sueltas tomaban inclinaciones de más).
SLOPE_PHASE_GAIN = 0.25
SLOPE_PROP_MAX_T = 2.0


def detect_rows_by_profile(cv2_module, img, valid, angle_deg, period_px,
                           min_len_px, max_gap_px, k_sigma=3.0, aux=None,
                           aux_localiza=False, row_valid=None, level_min=LEVEL_MIN,
                           wide_rows=False, track_max_px=None, aux_siembra=False, shadow_px=0.0,
                           half_center=False, band_gain=None, canceled=None):
    """Detecta hileras rectas y paralelas, de orientación angle_deg y
    separación period_px (coordenadas de imagen), sobre el índice de
    vegetación img. Devuelve (segmentos, ids_hilera, info):
      - segmentos: lista de (x1, y1, x2, y2) en píxeles de img
      - ids_hilera: número de hilera (1..N, en orden perpendicular) de
        cada segmento; una hilera cortada por un hueco largo da varios
        segmentos con el mismo número
      - info: dict con n_hileras, sigma (ruido del contraste), tau (umbral)

    Pasos:
      1. Remuestrea img a una grilla rotada donde las hileras quedan
         verticales (fila = posición a lo largo de la hilera, columna =
         distancia perpendicular).
      2. Perfil perpendicular (ExG medio por columna) suavizado; sus
         máximos locales separados ~T son los centros de hilera.
      3. Contraste local a lo largo de la hilera: ExG medio en una banda
         de ancho T/2 + 1 px centrada en la hilera (ancho continuo, con
         pesos fraccionales) menos el ExG de las dos entrehileras vecinas
         (±T/2, interpolado). Es
         relativo, así que no depende del vigor absoluto de la zona (a
         diferencia de un umbral global). Hilera "presente" donde el
         contraste medio supera k_sigma veces el ruido Y el centro es más
         verde que AMBAS entrehileras (descarta bordes de árboles, caminos,
         etc., que dan contraste de un solo lado). El ruido se estima con
         la cola negativa de la distribución del contraste, que solo puede
         venir de ruido (una hilera real nunca es menos verde que su
         entrehilera).
      4. Por hilera, reubica el centro en bloques a lo largo de la hilera,
         solo donde el paso 3 la vio presente, y recalcula el contraste
         sobre esa recta (tolera deriva leve de la hilera o del ángulo).
      5. Umbral con histéresis: un tramo necesita una racha de al menos T
         por encima de k_sigma veces el ruido, y se extiende mientras el
         contraste siga por encima de 1 vez el ruido — o, si se pasa una
         imagen auxiliar `aux` (p. ej. brillo), mientras el patrón de la
         hilera en esa imagen siga con el mismo signo que tiene donde el
         índice la ve con claridad (> k_sigma veces el 10% de su contraste
         típico, y respecto de ambas entrehileras). Extremos localizados a
         media altura. Une
         huecos <= max_gap_px (plantas faltantes) y descarta tramos
         < min_len_px.

    aux_localiza=True (usar solo si `aux` es la estructura de brillo y el
    índice principal es el verdor): cada centro de hilera se verifica
    contra el patrón de brillo del lote y, si cayó sobre la entrehilera, se
    mueve a la hilera; y el brillo también siembra tramos.

    shadow_px: con la franja oscura juzgada sombra al costado de la canopia
    (ver C_GREEN_CANOPY), dónde queda la sombra respecto de la canopia en
    columnas de la grilla, con signo (medido en todo el lote). La
    reubicación lleva el centro a la franja oscura menos ese corrimiento: a
    la canopia, no a su sombra.

    row_valid (opcional, misma forma que valid): dónde puede estar el
    CENTRO de una hilera (el lote). `valid` es dónde se puede MUESTREAR la
    imagen, y puede ser más ancho: la hilera de borde de un lote necesita
    la entrehilera que tiene del lado de afuera para medir su contraste.

    level_min: nivel mínimo de un tramo, en fracción del camino entre la
    entrehilera y la hilera típicas del lote (LEVEL_MIN; LEVEL_MIN_SUNLIT
    cuando img es el verde de copas iluminadas).

    wide_rows=True (copas iluminadas): la hilera es una franja ancha (la
    copa, ~2/3 de la separación) y casi pareja, sin un máximo definido; en
    la reubicación por bloques el centro se busca con el perfil transversal
    del bloque suavizado a T/4. Con el máximo sin suavizar caía en
    cualquier punto de la copa (hasta T/5 del centro, sobre la sombra). La
    ubicación inicial, con el perfil de toda la hilera, sigue a T/8: más
    suavizado aplanaba el pico de una hilera de menor vigor.

    Extremos: además de la evidencia suavizada a lo largo de T, cada
    hilera se prolonga planta por planta (evidencia a escala de planta,
    T/3) mientras haya plantas a menos de T una de otra: las plantas chicas
    de la punta (replantes, menor vigor) no alcanzan la evidencia de las
    grandes una vez promediadas, pero la hilera sigue hasta ellas.

    half_center=True (franja oscura porque la canopia no se ve verde): las
    líneas de salida van al centro a media altura de la banda, no a su pico
    (lo más oscuro, la sombra), si la banda es claramente asimétrica en todo
    el lote (ver HALF_MIN_T). La detección no cambia.

    band_gain (franja oscura; None = no): las hileras se ubican en el centro a
    media altura de su banda, en la pasada 1 y en los bloques de la pasada 2,
    si en la pasada 1 eso da al menos (1 + band_gain) veces el largo de
    hilera presente que el pico (ver BAND_MAX_T).

    Antes de la pasada 2 cada hilera puede tomar una pendiente propia (ver
    SLOPE_RES).

    canceled (opcional): función sin argumentos que dice si el usuario
    canceló; se consulta hilera por hilera y, si canceló, se devuelve ([], [],
    info) enseguida (un lote grande tarda minutos y antes "Cancelar" no hacía
    nada hasta que terminaba)."""
    cv2 = cv2_module
    H, W = img.shape
    T = float(period_px)
    info = {"n_hileras": 0, "sigma": None, "tau": None}

    def stop():
        return canceled is not None and canceled()

    th = math.radians(angle_deg)
    u = (math.cos(th), math.sin(th))     # a lo largo de la hilera
    n = (-math.sin(th), math.cos(th))    # perpendicular a la hilera
    corners = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], dtype=np.float64)
    s_c = corners @ np.array(u)
    t_c = corners @ np.array(n)
    s0 = int(math.floor(s_c.min()))
    t0 = int(math.floor(t_c.min()))
    NS = int(math.ceil(s_c.max())) - s0 + 1
    NT = int(math.ceil(t_c.max())) - t0 + 1

    # Grilla rotada: fila i <-> s = s0 + i ; columna j <-> t = t0 + j ;
    # píxel de origen x = s*u_x + t*n_x , y = s*u_y + t*n_y
    M = np.array([[n[0], u[0], s0 * u[0] + t0 * n[0]],
                  [n[1], u[1], s0 * u[1] + t0 * n[1]]], dtype=np.float64)
    R = cv2.warpAffine(img.astype(np.float32), M, (NT, NS),
                       flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    RV = cv2.warpAffine(valid.astype(np.uint8), M, (NT, NS),
                        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    Rv = np.where(RV, R, 0.0)
    if row_valid is None:
        RR = RV
    else:
        RR = cv2.warpAffine(row_valid.astype(np.uint8), M, (NT, NS),
                            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        RR &= RV
    RA = None
    if aux is not None:
        RA = cv2.warpAffine(aux.astype(np.float32), M, (NT, NS),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # --- 2) perfil perpendicular y centros de hilera
    cnt = RV.sum(axis=0)
    good_cols = cnt >= max(min_len_px, 1)
    # columnas donde puede haber un centro de hilera (dentro del lote)
    row_cols = RR.sum(axis=0) >= max(min_len_px, 1)
    if not good_cols.any() or not row_cols.any():
        return [], [], info
    sig = max(0.5, T / 8.0)
    rad = int(math.ceil(3 * sig))
    gk = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sig) ** 2)
    gk /= gk.sum()

    def column_profile(G, sign=1.0):
        """Perfil perpendicular suavizado: media por columna de la grilla
        rotada (solo píxeles válidos)."""
        p = np.where(good_cols, sign * np.where(RV, G, 0.0).sum(axis=0) / np.maximum(cnt, 1), 0.0)
        p = np.where(good_cols, p, p[good_cols].min())
        return np.convolve(np.pad(p, rad, mode="edge"), gk, mode="valid")

    half = max(1, int(round(T / 2.0)))

    def prominence(ps, j):
        """Prominencia de j como máximo local de ps en ±T/2 (None si no lo es,
        o si j cae fuera de donde puede haber hileras)."""
        if not (1 <= j < NT - 1) or not row_cols[j]:
            return None
        lo, hi = max(0, j - half), min(NT, j + half + 1)
        if ps[j] < ps[lo:hi].max() or ps[j] <= ps[j - 1]:
            return None
        left = ps[lo:j].min()
        right = ps[j + 1:hi].min() if j + 1 < hi else ps[j]
        return ps[j] - max(left, right)

    prof_s = column_profile(R)
    cand = [(j, p) for j in range(1, NT - 1) for p in (prominence(prof_s, j),) if p is not None]
    if not cand:
        return [], [], info
    proms = np.array([p for _, p in cand])
    centers = [j for j, p in cand if p > 0.2 * float(np.median(proms))]

    # --- 3) y 4) contraste local a lo largo de cada hilera + reubicación de su centro
    # Semiancho de la banda central, CONTINUO en T: con un número entero de
    # columnas (antes 2*round(T/4)+1), una separación estimada de 9,9 o de
    # 10,1 px daba 5 o 7 columnas, y ese salto cambiaba el ruido estimado
    # (y el umbral) casi un 30% por un error de 0,1 px en el período.
    # Con hileras finas (T < 8 px) el medio píxel extra se reduce hasta 0
    # en T = 4 px: con T/4 + 0,5 la banda cubría el 73% del período a 4,3 px
    # y dejaba pasar solo un tercio del contraste de la hilera.
    hw = T / 4.0 + 0.5 * min(1.0, max(0.0, (T - 4.0) / 4.0))
    K = int(math.ceil(hw + 0.5))
    box = np.ones(max(3, int(round(T))))
    box_p = np.ones(max(3, int(round(T / 3.0))))  # escala de una planta
    ss = np.arange(NS)
    # La grilla como vector: el punto (fila i, columna k) es el elemento
    # i * NT + k. contrast_along se llama miles de veces por lote (un tercio
    # del tiempo total) y leer así es más rápido que con [filas, columnas];
    # da exactamente los mismos valores.
    ss_off = ss.astype(np.int64) * NT
    nt_1 = NT - 1
    R_flat, RV_flat, RR_flat = R.reshape(-1), RV.reshape(-1), RR.reshape(-1)
    RA_flat = None if RA is None else RA.reshape(-1)

    def contrast_along(a0, a1, dt=None):
        """Contrastes centro de hilera vs. entrehileras (±T/2) a lo largo de
        la recta t = a0 + a1*s de la grilla rotada, suavizados a lo largo de
        la hilera. El centro es la media de la banda [t - hw, t + hw] (cada
        columna pesa lo que se superpone con ella) y cada entrehilera se
        interpola linealmente en t ± T/2: nada salta por redondeos de t o de
        T. Para el índice (y, si hay, la imagen auxiliar) devuelve
        (c - media(l, r), c - l, c - r) suavizados a lo largo de T, y
        min(c - l, c - r) suavizado a escala de planta (T/3); NaN en el
        primero y el último donde algún punto de muestreo cae fuera de lo
        válido o el centro cae fuera de donde puede haber hileras."""
        tt = a0 + a1 * ss
        if dt is not None:
            tt = tt + dt   # camino seguido (hilera levemente torcida)
        base = np.floor(tt).astype(np.int64)
        jc = np.round(tt).astype(np.int64)
        # (mínimo y máximo en vez de np.clip, que en vectores cortos tarda
        # varias veces más: mismo resultado)
        ok = (jc >= 0) & (jc < NT) & RR_flat[ss_off + np.minimum(np.maximum(jc, 0), nt_1)]
        taps = {"c": []}
        t_lo, t_hi = tt - hw, tt + hw
        for d in range(-K, K + 1):
            k = base + d
            taps["c"].append((k, np.minimum(np.maximum(
                np.minimum(k + 0.5, t_hi) - np.maximum(k - 0.5, t_lo), 0.0), 1.0)))
        for key, offset in (("l", -T / 2.0), ("r", T / 2.0)):
            x = tt + offset
            k0 = np.floor(x).astype(np.int64)
            f = x - k0
            taps[key] = [(k0, 1.0 - f), (k0 + 1, f)]
        for key in taps:
            clean = []
            for k, w in taps[key]:
                used = w > 1e-9
                kf = ss_off + np.minimum(np.maximum(k, 0), nt_1)   # índice en la grilla como vector
                ok &= ~used | ((k >= 0) & (k < NT) & RV_flat[kf])
                clean.append((kf, np.where(used, w, 0.0)))
            wsum = np.maximum(sum(w for _, w in clean), 1e-12)
            taps[key] = (clean, wsum)
        okf = ok.astype(np.float64)
        norm = np.maximum(np.convolve(okf, box, "same"), 1e-9)
        norm_p = np.maximum(np.convolve(okf, box_p, "same"), 1e-9)

        def smooth(x):
            return np.convolve(np.where(ok, x, 0.0), box, "same") / norm

        out = []
        for G in (R_flat, RA_flat):
            if G is None:
                out.append(None)
                continue
            c, l, r = (sum(G[k] * w for k, w in taps[key][0]) / taps[key][1] for key in ("c", "l", "r"))
            con_s = smooth(c - 0.5 * (l + r))
            con_s[~ok] = np.nan
            plant = np.convolve(np.where(ok, np.minimum(c - l, c - r), 0.0), box_p, "same") / norm_p
            plant[~ok] = np.nan
            lvl_c, lvl_s = smooth(c), smooth(0.5 * (l + r))   # niveles absolutos: centro y entrehileras
            lvl_c[~ok] = np.nan
            out.append((con_s, smooth(c - l), smooth(c - r), plant, lvl_c, lvl_s))
        return out[0], out[1]

    def presence(con_s, cmin_s, tau):
        return (np.nan_to_num(con_s, nan=-np.inf) > tau) & (cmin_s > 0)

    def noise_rms(values):
        """Ruido a partir de la cola negativa de un contraste que, en una
        hilera real, es positivo: la cola negativa solo puede ser ruido."""
        neg = values[values < 0]
        if neg.size >= 50:
            return float(np.sqrt(np.mean(neg ** 2)))
        return 0.1 * float(np.median(np.abs(values))) if values.size else 0.0

    # Pasada 1: recta con el ángulo global por el pico (sub-píxel) del
    # perfil. Alcanza para estimar el ruido y el umbral de presencia.
    rows = []
    band_alt = {}   # hilera -> (centro de la banda, se movió)
    for j in centers:
        if stop():
            return [], [], info
        a_, b_, c_ = prof_s[j - 1], prof_s[j], prof_s[j + 1]
        den = a_ - 2.0 * b_ + c_
        off = 0.5 * (a_ - c_) / den if abs(den) > 1e-12 else 0.0
        a0 = j + float(np.clip(off, -0.5, 0.5))
        if band_gain is not None:
            hc_ = half_height_center(prof_s, j, T)
            if hc_ is not None and abs(hc_ - a0) <= BAND_MAX_T * T:
                band_alt[len(rows)] = (float(hc_), abs(hc_ - a0) > BAND_MOVED_T * T)
        main, auxc = contrast_along(a0, 0.0)
        rows.append([a0, 0.0, main, auxc])

    finite = [r[2][0][np.isfinite(r[2][0])] for r in rows]
    allc = np.concatenate(finite) if finite else np.array([])
    if allc.size == 0:
        return [], [], info
    sigma = noise_rms(allc)
    tau = k_sigma * sigma
    info["sigma"], info["tau"] = sigma, tau

    # Centro de la banda (ver BAND_MAX_T): se compara el largo de hilera
    # presente en el pico y en el centro, con el mismo umbral, en las hileras
    # donde difieren.
    band_on = False
    info["centro_banda"] = 0
    moved_ = [i_ for i_, (_, m_) in band_alt.items() if m_]
    if moved_:
        alt_c = {i_: contrast_along(a_, 0.0) for i_, (a_, _) in band_alt.items()}

        def present_len(m_):
            return float(presence(m_[0], np.minimum(m_[1], m_[2]), tau).sum())

        p_pk = sum(present_len(rows[i_][2]) for i_ in moved_)
        p_bc = sum(present_len(alt_c[i_][0]) for i_ in moved_)
        info["banda_presencia"] = (p_pk, p_bc)
        band_on = p_bc >= (1.0 + band_gain) * p_pk
        if band_on:
            for i_, (a_, m_) in band_alt.items():
                rows[i_][0] = a_
                rows[i_][2], rows[i_][3] = alt_c[i_]
                info["centro_banda"] += int(m_)
            allc = np.concatenate([r[2][0][np.isfinite(r[2][0])] for r in rows])
            sigma = noise_rms(allc)
            tau = k_sigma * sigma
            info["sigma"], info["tau"] = sigma, tau

    def main_presence(row, thr):
        con_s, dl, dr = row[2][:3]
        return presence(con_s, np.minimum(dl, dr), thr)

    # Signo de la hilera en la imagen auxiliar (más oscura o más clara que
    # su entrehilera), tomado de los tramos donde el índice la ve con
    # claridad, hilera por hilera.
    aux_sign = [0.0] * len(rows)
    if RA is not None:
        for i, row in enumerate(rows):
            strong = main_presence(row, tau)
            if strong.sum() >= max(3.0, T):
                aux_sign[i] = float(np.sign(np.nanmedian(row[3][0][strong])))
    sign_global = float(np.sign(sum(aux_sign)))
    own_sign = [s != 0.0 for s in aux_sign]  # signo fijado por la propia hilera

    # --- Verificación de centros con la imagen auxiliar (si ésta es la
    # estructura de brillo). Con el verdor como índice, en una franja del
    # lote donde la entrehilera es más verde que la hilera (pasto o maleza
    # entre hileras, vides debilitadas) el máximo de verdor cae sobre la
    # ENTREHILERA y la línea queda a media separación de la hilera real.
    # El patrón de brillo no se invierte: si en la mayoría de las hileras
    # la hilera es la franja oscura (sign_global), un centro que cae sobre
    # una franja clara se mueve a la franja oscura más cercana.
    info["reubicadas"] = 0
    if aux_localiza and RA is not None and sign_global != 0.0 and len(centers) >= 2:
        pa_loc = column_profile(RA, sign_global)
        wloc = int(math.ceil(T / 2.0)) + 1
        kept_rows, kept_centers, kept_signs, kept_own = [], [], [], []
        for row, j, sgn, own in zip(rows, centers, aux_sign, own_sign):
            lo_, hi_ = max(1, j - wloc), min(NT - 2, j + wloc)
            k = lo_ + int(np.argmax(pa_loc[lo_:hi_ + 1]))
            # Si la franja oscura es la sombra al costado de la canopia
            # (shadow_px), el destino es la canopia: la franja oscura menos el
            # corrimiento de la sombra. Antes iba a la sombra (visto: frutal en
            # seto, 19 líneas a 1 m del centro de la canopia; sin reubicar se
            # perdían 15 hileras, cuyo centro de verdor caía en la maleza).
            # (Probado: mover solo los centros del lado claro. En el mismo
            # seto empeoraba: 12 líneas menos.)
            kc = int(round(k - shadow_px))
            if (abs(kc - j) > T / 4.0 and prominence(pa_loc, k) is not None
                    and 1 <= kc < NT - 1 and row_cols[kc]):
                a_, b_, c_ = pa_loc[k - 1], pa_loc[k], pa_loc[k + 1]
                den = a_ - 2.0 * b_ + c_
                off = 0.5 * (a_ - c_) / den if abs(den) > 1e-12 else 0.0
                a0 = k + float(np.clip(off, -0.5, 0.5)) - shadow_px
                main, auxc = contrast_along(a0, 0.0)
                row, j, sgn, own = [a0, 0.0, main, auxc], kc, sign_global, False
                info["reubicadas"] += 1
            if any(abs(j - c) <= T / 2.0 for c in kept_centers):
                continue  # dos centros terminaron en la misma franja
            kept_rows.append(row); kept_centers.append(j); kept_signs.append(sgn); kept_own.append(own)
        order = np.argsort(kept_centers)
        rows = [kept_rows[i] for i in order]
        centers = [kept_centers[i] for i in order]
        aux_sign = [kept_signs[i] for i in order]
        own_sign = [kept_own[i] for i in order]
        # hileras cuyo verdor no alcanzó para fijar su signo: el del lote
        aux_sign = [s if s != 0.0 else sign_global for s in aux_sign]

    # --- Hileras sin pico propio en el índice (típicamente las de borde,
    # de menor vigor, o alguna suelta en medio): se buscan extendiendo la
    # grilla periódica. A una separación T de la última hilera ubicada (o en
    # un hueco de ~2T entre dos), si el perfil de la imagen auxiliar muestra
    # la misma franja con el mismo signo que el resto de las hileras, se
    # agrega; hacia afuera se sigue hasta que el patrón se termina.
    added = set()
    if RA is not None and sign_global != 0.0 and len(centers) >= 2:
        pa = column_profile(RA, sign_global)
        ref = [p for p in (prominence(pa, j) for j in centers) if p is not None]
        min_prom = 0.3 * float(np.median(ref)) if ref else None

        def find_near(target):
            lo = max(1, int(math.floor(target - T / 4.0)))
            hi = min(NT - 2, int(math.ceil(target + T / 4.0)))
            if hi < lo:
                return None
            k = lo + int(np.argmax(pa[lo:hi + 1]))
            p = prominence(pa, k)
            return k if p is not None and p >= min_prom else None

        new = []
        if min_prom is not None and min_prom > 0:
            for start, step in ((centers[0], -1.0), (centers[-1], 1.0)):
                j = start
                while True:
                    k = find_near(j + step * T)
                    if k is None:
                        break
                    new.append(k)
                    j = k
            for j1, j2 in zip(centers[:-1], centers[1:]):
                m = int(round((j2 - j1) / T)) - 1
                for q in range(1, m + 1):
                    k = find_near(j1 + q * (j2 - j1) / (m + 1))
                    if k is not None and all(abs(k - c) > T / 2.0 for c in (j1, j2)):
                        new.append(k)
        for k in sorted(set(new)):
            if any(abs(k - c) <= T / 2.0 for c in centers):
                continue
            main, auxc = contrast_along(float(k), 0.0)
            rows.append([float(k), 0.0, main, auxc])
            centers.append(k)
            aux_sign.append(sign_global)
            own_sign.append(False)
            added.add(k)
        order = np.argsort(centers)
        rows = [rows[i] for i in order]
        aux_sign = [aux_sign[i] for i in order]
        own_sign = [own_sign[i] for i in order]
        centers = [centers[i] for i in order]

    # --- Grilla completa: el marco de plantación es regular, así que cada
    # posición a un múltiplo de T de las hileras ubicadas (en los huecos
    # entre ellas y hacia afuera, hasta el borde del lote) se prueba como
    # hilera aunque el perfil no muestre ahí un máximo: una hilera de
    # plantas chicas o ralas (típicamente la de borde) casi no pesa en el
    # perfil promedio del lote. La prueba es la de siempre, a lo largo de
    # la hilera (contraste, o secuencia de plantas en el paso 5): en una
    # calle o en suelo desnudo no aparece ningún tramo.
    # La posición es la del marco (múltiplo exacto de T desde las hileras
    # ubicadas, con su posición sub-píxel), sin "afinarla" al máximo del
    # perfil del lote: una hilera de borde que ocupa solo parte del largo
    # del lote casi no pesa en ese perfil, y el máximo dentro de ±T/4 caía
    # donde fuera (visto: hileras de borde a 0,6-0,8 T o a 1,2-1,6 T de su
    # vecina, medidas de costado y después descartadas por fuera del marco).
    # La pasada 2 la reubica con su propia evidencia.
    grid_added = set()
    if len(centers) >= 2:
        fine = T < TRACK_MIN_T_PX
        pos = dict(zip(centers, (float(c) if fine else r[0] for c, r in zip(centers, rows))))
        cs_ = sorted(centers)
        targets = []
        for j1, j2 in zip(cs_[:-1], cs_[1:]):
            p1, p2 = pos[j1], pos[j2]
            m = int(round((p2 - p1) / T)) - 1
            targets += [p1 + q * (p2 - p1) / (m + 1) for q in range(1, m + 1)]
        for start, step in ((pos[cs_[0]], -1.0), (pos[cs_[-1]], 1.0)):
            jf = start + step * T
            while 1 <= jf < NT - 1 and row_cols[int(round(jf))]:
                targets.append(jf)
                jf += step * T
        taken = [r[0] for r in rows]
        for jf in targets:
            if fine:
                # hileras finas (4-5 px): como antes, afinada al máximo del
                # perfil del lote (el modo patrón depende de esa posición)
                lo = max(1, int(math.floor(jf - T / 4.0)))
                hi = min(NT - 2, int(math.ceil(jf + T / 4.0)))
                if hi < lo:
                    continue
                k = lo + int(np.argmax(prof_s[lo:hi + 1]))
                if not row_cols[k] or any(abs(k - c) <= T / 2.0 for c in centers):
                    continue
                jf = float(k)
            else:
                k = int(round(jf))
                if not (1 <= k < NT - 1) or not row_cols[k] or any(abs(jf - p_) <= T / 2.0 for p_ in taken):
                    continue
            main, auxc = contrast_along(float(jf), 0.0)
            taken.append(float(jf))
            rows.append([float(jf), 0.0, main, auxc])
            centers.append(k)
            aux_sign.append(sign_global)
            own_sign.append(False)
            grid_added.add(k)
        if grid_added:
            order = np.argsort(centers)
            rows = [rows[i] for i in order]
            aux_sign = [aux_sign[i] for i in order]
            own_sign = [own_sign[i] for i in order]
            centers = [centers[i] for i in order]
    info["n_agregadas"] = len(added)

    # --- Rumbo propio por hilera (ver SLOPE_RES). Cada paso inclina la grilla
    # entera alrededor de su fila central ic_: la columna x de la grilla
    # inclinada junta los puntos t = x + sg * (i - ic_). Para cada hilera se
    # busca, en cada inclinación, el máximo del contraste del perfil de
    # columnas (centro menos la media de ±T/2) a menos de T/4 de donde la
    # hilera pasa por el medio de su propio largo.
    info["pendiente_propia"] = 0
    if rows:
        len_r, mid_r = [], []   # largo de cada hilera (filas de la grilla) y su fila media
        for r_ in rows:
            ii_ = np.nonzero(RR[:, min(max(int(round(r_[0])), 0), NT - 1)])[0]
            len_r.append(len(ii_))
            mid_r.append(float(np.median(ii_)) if len(ii_) else 0.5 * (NS - 1))
        len_r = np.array(len_r, float)
        len_med = float(np.median(len_r[len_r > 0])) if (len_r > 0).any() else 0.0
        if len_med >= 6.0 * T:
            d_sg = T / (SLOPE_RES * len_med)
            n_sg = int(round(SLOPE_MAX_T * SLOPE_RES))          # búsqueda propia de cada hilera
            n_all = max(n_sg, int(round(SLOPE_PROP_MAX_T * SLOPE_RES)))   # propagación desde vecinas
            ic_ = 0.5 * (NS - 1)
            Rz_ = np.where(RV, R, 0.0).astype(np.float32)
            Vz_ = RV.astype(np.float32)
            xs_ = np.arange(NT, dtype=np.float64)
            Q_ = []   # perfil de contraste por paso: Q_[kk + n_all]
            for kk in range(-n_all, n_all + 1):
                if stop():
                    return [], [], info
                sg_ = kk * d_sg
                Mk_ = np.array([[1.0, sg_, -sg_ * ic_], [0.0, 1.0, 0.0]])
                num_, den_ = (
                    np.convolve(np.pad(cv2.warpAffine(G_, Mk_, (NT, NS),
                                                      flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                       .sum(axis=0).astype(np.float64), rad, mode="edge"), gk, mode="valid")
                    for G_ in (Rz_, Vz_))
                p_ = np.where(den_ >= max(min_len_px, 1), num_ / np.maximum(den_, 1e-9), np.nan)
                Q_.append(p_ - 0.5 * (np.interp(xs_ - T / 2.0, xs_, p_) + np.interp(xs_ + T / 2.0, xs_, p_)))

            def best_in(q_, x_, w_=0.25):
                """Máximo (sub-píxel) de q_ a menos de w_ * T de x_: (posición, valor)."""
                lo_ = max(int(math.ceil(x_ - w_ * T)), 1)
                hi_ = min(int(math.floor(x_ + w_ * T)), NT - 2)
                if hi_ < lo_ or not np.isfinite(q_[lo_:hi_ + 1]).any():
                    return None, -np.inf
                k_ = lo_ + int(np.nanargmax(q_[lo_:hi_ + 1]))
                a_, b_, c_ = q_[k_ - 1], q_[k_], q_[k_ + 1]
                den = a_ - 2.0 * b_ + c_
                off = 0.5 * (a_ - c_) / den if (np.isfinite(den) and abs(den) > 1e-12) else 0.0
                return k_ + float(np.clip(off, -0.5, 0.5)), float(b_)

            cand_ = []   # por hilera: (paso, posición en ic_) de la mejor inclinación, o None
            for r_, im_ in zip(rows, mid_r):
                sc_ = [best_in(Q_[kk + n_all], r_[0] - kk * d_sg * (im_ - ic_)) for kk in range(-n_sg, n_sg + 1)]
                s0_ = sc_[n_sg][1]
                kb_ = int(np.argmax([s_ for _, s_ in sc_]))
                xb_, sb_ = sc_[kb_]
                # a media separación (±T/2) hay una alineación mucho mejor: la
                # recta del lote pasa por el medio de la hilera entre dos hileras
                # reales (ver SLOPE_PHASE_GAIN) y lo de ±T/4 es un falso máximo
                wide_ = max(best_in(Q_[kk + n_all], r_[0] - kk * d_sg * (im_ - ic_), 0.5)[1]
                            for kk in range(-n_sg, n_sg + 1))
                if (kb_ == n_sg or xb_ is None or not np.isfinite(sb_) or sb_ <= 0
                        or (np.isfinite(s0_) and s0_ > 0 and sb_ < (1.0 + SLOPE_GAIN) * s0_)
                        or wide_ > (1.0 + SLOPE_PHASE_GAIN) * sb_):
                    cand_.append(None)
                else:
                    cand_.append((kb_ - n_sg, xb_))
            new_ = {}
            for i_, c_ in enumerate(cand_):
                if c_ is None:
                    continue
                nb_ = [cand_[q_] for q_ in range(i_ - 2, i_ + 3) if q_ != i_ and 0 <= q_ < len(cand_)]
                if sum(1 for d_ in nb_ if d_ is not None and abs(d_[0] - c_[0]) <= SLOPE_NB_K) >= SLOPE_NB:
                    new_[i_] = (c_[1] - c_[0] * d_sg * ic_, c_[0] * d_sg)

            def clash(i_, a0_, a1_, only_new=False):
                """¿La recta (a0_, a1_) de la hilera i_ se acerca a menos de T/2
                de la de una vecina (o la cruza) a lo largo de la hilera?
                only_new: solo contra vecinas que ya tomaron rumbo propio."""
                s_a, s_b = mid_r[i_] - 0.5 * len_r[i_], mid_r[i_] + 0.5 * len_r[i_]
                for q_ in (i_ - 1, i_ + 1):
                    if 0 <= q_ < len(rows) and (q_ in new_ or not only_new):
                        b0_, b1_ = new_.get(q_, (rows[q_][0], rows[q_][1]))
                        g_a = (a0_ - b0_) + (a1_ - b1_) * s_a
                        g_b = (a0_ - b0_) + (a1_ - b1_) * s_b
                        if min(abs(g_a), abs(g_b)) < 0.5 * T or g_a * g_b < 0:
                            return True
                return False

            for i_ in sorted(new_):
                if clash(i_, *new_[i_]):
                    del new_[i_]
            # Propagación (ver SLOPE_PROP_MAX_T): una hilera sin rumbo propio
            # junto a una que lo tomó busca su hilera a una separación de la
            # recta de la vecina, con una inclinación parecida (a lo sumo
            # SLOPE_NB_K pasos de diferencia), hasta que no cambie nada.
            n_first = len(new_)
            s0_all = [best_in(Q_[n_all], r_[0])[1] for r_ in rows]
            changed_ = True
            while changed_:
                changed_ = False
                for i_ in range(len(rows)):
                    if i_ in new_:
                        continue
                    for q_ in (i_ - 1, i_ + 1):
                        if q_ not in new_:
                            continue
                        a0q, a1q = new_[q_]
                        kq = int(round(a1q / d_sg))
                        xq = a0q + a1q * ic_ + (T if i_ > q_ else -T)
                        x_, s_, kb_ = None, -np.inf, 0
                        for kk in range(max(-n_all, kq - SLOPE_NB_K), min(n_all, kq + SLOPE_NB_K) + 1):
                            xk_, sk_ = best_in(Q_[kk + n_all], xq - (kk - kq) * d_sg * (mid_r[i_] - ic_))
                            if xk_ is not None and sk_ > s_:
                                x_, s_, kb_ = xk_, sk_, kk
                        s0_ = s0_all[i_]
                        if x_ is None or not np.isfinite(s_) or s_ <= 0:
                            continue
                        if np.isfinite(s0_) and s0_ > 0 and s_ < (1.0 + SLOPE_GAIN) * s0_:
                            continue
                        line_ = (x_ - kb_ * d_sg * ic_, kb_ * d_sg)
                        # las vecinas que no tomaron rumbo propio todavía pueden
                        # tomarlo en esta propagación: no cuentan
                        if clash(i_, *line_, only_new=True):
                            continue
                        new_[i_] = line_
                        changed_ = True
                        break
            info["pendiente_propagada"] = len(new_) - n_first
            for i_, (a0_, a1_) in new_.items():
                rows[i_][0], rows[i_][1] = a0_, a1_
                rows[i_][2], rows[i_][3] = contrast_along(a0_, a1_)
                info["pendiente_propia"] += 1

    # Pasada 2: reubicar el centro de cada hilera por bloques a lo largo de
    # ella, usando SOLO los bloques donde la pasada 1 vio la hilera presente
    # (en un camino o una cabecera no hay hilera, y el "centro" de ese
    # bloque sería un máximo de ruido). Por defecto la recta sigue paralela
    # al ángulo global, desplazada a la mediana de esos centros; solo con
    # 4+ bloques se ajusta además una pendiente propia, y solo si es
    # significativa: tolera hileras levemente torcidas o distorsión del
    # ortomosaico sin inventar pendientes a partir del ruido.
    # Una hilera con rumbo propio (SLOPE_RES) sigue su recta inclinada: la
    # ventana de cada bloque va donde la recta pasa por el bloque y la recta se
    # corrige con los centros de los bloques respecto de ella. Con el centro de
    # la banda (band_on) el centro de cada bloque es el de media altura.
    blk = int(max(10 * T, 60))
    for row, j in zip(rows, centers):
        if stop():
            return [], [], info
        a0_1, a1_1 = row[0], row[1]
        occ = main_presence(row, tau)
        idx_occ = np.nonzero(occ)[0]
        if len(idx_occ) == 0:
            continue
        s_lo, s_hi = int(idx_occ.min()), int(idx_occ.max()) + 1
        jlo, jhi = max(0, j - half), min(NT, j + half + 1)
        if wide_rows:
            # ventana más ancha para suavizar sin efecto de borde; el
            # máximo se busca igual solo en ±T/2
            sig_w = max(0.5, T / 4.0)
            rad_w = int(math.ceil(3 * sig_w))
            gk_w = np.exp(-0.5 * (np.arange(-rad_w, rad_w + 1) / sig_w) ** 2)
            gk_w /= gk_w.sum()
            jlo_w, jhi_w = max(0, jlo - rad_w), min(NT, jhi + rad_w)
        pts_s, pts_t = [], []
        for b0 in range(s_lo, s_hi, blk):
            b1 = min(b0 + blk, s_hi)
            jb = j
            if a1_1 != 0.0:
                jb = int(round(a0_1 + a1_1 * 0.5 * (b0 + b1)))
                jlo, jhi = max(0, jb - half), min(NT, jb + half + 1)
                if wide_rows:
                    jlo_w, jhi_w = max(0, jlo - rad_w), min(NT, jhi + rad_w)
            if occ[b0:b1].mean() < 0.5:
                continue
            c = RV[b0:b1, jlo:jhi].sum(axis=0)
            if c.min() < max(3, (b1 - b0) // 4):
                continue
            colmean = Rv[b0:b1, jlo:jhi].sum(axis=0) / c
            if wide_rows:
                cw = RV[b0:b1, jlo_w:jhi_w].sum(axis=0)
                cmw = Rv[b0:b1, jlo_w:jhi_w].sum(axis=0) / np.maximum(cw, 1)
                sm = np.convolve(np.pad(cmw, rad_w, mode="edge"), gk_w, mode="valid")
                colmean = sm[jlo - jlo_w:jlo - jlo_w + (jhi - jlo)]
            k = int(np.argmax(colmean))
            if not 0 < k < len(colmean) - 1:
                continue  # máximo en el borde de la ventana: no es un centro de hilera
            a_, b_, c_ = colmean[k - 1], colmean[k], colmean[k + 1]
            den = a_ - 2.0 * b_ + c_
            off = 0.5 * (a_ - c_) / den if abs(den) > 1e-12 else 0.0
            tk = jlo + k + float(np.clip(off, -0.5, 0.5))
            if band_on:
                # perfil del bloque más ancho (±1,5 T), suavizado como el del
                # lote, para medir la banda a media altura
                blo, bhi = max(0, jb - int(math.ceil(1.5 * T))), min(NT, jb + int(math.ceil(1.5 * T)) + 1)
                cb = RV[b0:b1, blo:bhi].sum(axis=0)
                if cb.min() >= max(3, (b1 - b0) // 4):
                    smb = np.convolve(np.pad(Rv[b0:b1, blo:bhi].sum(axis=0) / cb, rad, mode="edge"), gk, mode="valid")
                    hb = half_height_center(smb, jlo + k - blo, T)
                    if hb is not None and abs(blo + hb - tk) <= BAND_MAX_T * T:
                        tk = blo + hb
            pts_s.append(0.5 * (b0 + b1))
            pts_t.append(tk)
        if not pts_t:
            continue
        pts_s, pts_t = np.array(pts_s), np.array(pts_t)
        if a1_1 != 0.0:
            # rumbo propio: corrimiento y, si es significativa, pendiente de
            # los centros de los bloques respecto de la recta inclinada
            res_ = pts_t - (a0_1 + a1_1 * pts_s)
            d0_, d1_ = 0.0, 0.0
            med_ = float(np.median(res_))
            if abs(med_) <= T / 4.0:
                d0_ = med_
            if len(pts_s) >= 4:
                keep = np.abs(res_ - d0_) <= T / 4.0
                if keep.sum() >= 4:
                    ps, pt = pts_s[keep], res_[keep]
                    A = np.vstack([np.ones(len(ps)), ps]).T
                    (e0, e1), *_ = np.linalg.lstsq(A, pt, rcond=None)
                    resid = pt - (e0 + e1 * ps)
                    sxx = float(((ps - ps.mean()) ** 2).sum())
                    se_e1 = (math.sqrt(float((resid ** 2).sum()) / (len(ps) - 2) / sxx)
                             if sxx > 0 and len(ps) > 2 else float("inf"))
                    mid = 0.5 * (s_lo + s_hi)
                    if (abs(e1) > 2.0 * se_e1 and abs(e1) * (s_hi - s_lo) <= T / 2.0
                            and abs(e0 + e1 * mid) <= T / 2.0):
                        d0_, d1_ = float(e0), float(e1)
            if d0_ != 0.0 or d1_ != 0.0:
                row[0], row[1] = a0_1 + d0_, a1_1 + d1_
                row[2], row[3] = contrast_along(row[0], row[1])
            continue
        # con el centro de la banda, la referencia es la posición de la
        # pasada 1 (ya en el centro), no la columna del pico
        j_ref = a0_1 if band_on else j
        a0, a1 = a0_1, 0.0
        med = float(np.median(pts_t))
        if abs(med - j_ref) <= T / 4.0:
            a0 = med
        if len(pts_s) >= 4:
            keep = np.abs(pts_t - a0) <= T / 4.0
            if keep.sum() >= 4:
                ps, pt = pts_s[keep], pts_t[keep]
                A = np.vstack([np.ones(len(ps)), ps]).T
                (c0, c1), *_ = np.linalg.lstsq(A, pt, rcond=None)
                resid = pt - (c0 + c1 * ps)
                sxx = float(((ps - ps.mean()) ** 2).sum())
                se_c1 = math.sqrt(float((resid ** 2).sum()) / (len(ps) - 2) / sxx) if sxx > 0 else float("inf")
                mid = 0.5 * (s_lo + s_hi)
                # pendiente significativa (> 2 errores estándar) y sin
                # derivar hacia la hilera vecina
                if (abs(c1) > 2.0 * se_c1
                        and abs(c1) * (s_hi - s_lo) <= T / 2.0
                        and abs(c0 + c1 * mid - j_ref) <= T / 2.0):
                    a0, a1 = float(c0), float(c1)
        if abs(a0 - a0_1) > 1e-6 or a1 != 0.0:
            row[0], row[1] = a0, a1
            row[2], row[3] = contrast_along(a0, a1)

    # --- Evidencia auxiliar (p. ej. brillo): la hilera sigue presente
    # mientras siga su patrón periódico, aunque el índice de vegetación se
    # debilite (vides de menor vigor con pasto verde en la entrehilera, que
    # invierte el contraste de verdor). Se usa con el signo de cada hilera
    # (calculado arriba) y se exige que el centro difiera de AMBAS
    # entrehileras en ese sentido (una mancha uniforme, como la copa y la
    # sombra de un árbol, no cumple).
    sigma_a = None
    if RA is not None:
        # Escala de la evidencia auxiliar: el 10% del contraste típico (la
        # mediana) de las hileras cuyo signo fijó su propio verdor (las
        # ubicadas por otros medios no deben influir). Con k_sigma = 3, la
        # estructura auxiliar cuenta donde su contraste supera el 30% del
        # típico. No se usa la cola negativa, como en el índice: en estas
        # hileras el contraste auxiliar casi nunca es negativo (< 0,1% de
        # las muestras), así que esa cola no es ruido sino un puñado de
        # anomalías (un hueco, un árbol) y una sola hilera podía fijar el
        # umbral de todo el lote (visto: 7 en una imagen, 27 en la misma
        # imagen remuestreada, con la mitad de las hileras perdidas).
        vals = [
            (sgn * row[3][0])[np.isfinite(row[3][0])]
            for row, sgn, own in zip(rows, aux_sign, own_sign) if sgn != 0.0 and own
        ]
        if vals:
            sigma_a = 0.1 * float(np.median(np.abs(np.concatenate(vals))))
            info["sigma_aux"] = sigma_a

    # Ruido de la evidencia a escala de planta (índice principal): cola
    # negativa de min(c - l, c - r) suavizado a T/3. Acá la cola negativa
    # SÍ es ruido y abundante: entre planta y planta, en las fallas y en
    # las cabeceras dentro del lote el centro no difiere de las entrehileras.
    pvals = [row[2][3][np.isfinite(row[2][3])] for row in rows]
    sigma_p = noise_rms(np.concatenate(pvals)) if pvals else 0.0
    info["sigma_planta"] = sigma_p
    half_p = (len(box_p) - 1) / 2.0
    gap_p = max(3.0, T)             # plantas de una misma hilera: a menos de T
    min_solo = max(min_len_px, 3.0 * T)
    info["extendidas_por_plantas"] = 0

    # Niveles típicos del lote (índice y auxiliar), en las hileras que dio el
    # perfil, donde el índice las ve con claridad: el nivel de la hilera y el
    # de su entrehilera. Un tramo tiene que parecerse a una hilera también en
    # NIVEL, no solo en contraste con sus costados: una huella de tractor o
    # la sombra de un borde sobre una calle es algo más oscura (o verde) que
    # la calle a sus lados, pero queda lejos del nivel de una hilera.
    def typical_levels(k_idx, sgn_fn):
        cs_, ss_ = [], []
        for row, j, s_ in zip(rows, centers, aux_sign):
            if j in added or j in grid_added or row[k_idx] is None:
                continue
            sgn_ = sgn_fn(s_)
            if sgn_ == 0.0:
                continue
            st = main_presence(row, tau)
            cs_.append(sgn_ * row[k_idx][4][st]); ss_.append(sgn_ * row[k_idx][5][st])
        if not cs_:
            return None
        cv_, sv_ = np.concatenate(cs_), np.concatenate(ss_)
        cv_, sv_ = cv_[np.isfinite(cv_)], sv_[np.isfinite(sv_)]
        if cv_.size < 10 or sv_.size < 10:
            return None
        hi_, lo_ = float(np.median(cv_)), float(np.median(sv_))
        return (hi_, lo_) if hi_ - lo_ > 1e-9 else None

    lvl_main = typical_levels(2, lambda s_: 1.0)
    lvl_aux = typical_levels(3, lambda s_: sign_global) if RA is not None and sign_global != 0.0 else None
    info["descartados_por_nivel"] = 0

    def level_frac(arr, sgn_, lv, sl):
        v = sgn_ * arr[sl]
        v = v[np.isfinite(v)]
        if lv is None or v.size == 0:
            return -np.inf
        return (float(np.median(v)) - lv[1]) / (lv[0] - lv[1])

    # --- Seguimiento lateral: la recta de cada hilera (ángulo del lote,
    # desplazamiento y a lo sumo una pendiente propia) no sigue una hilera
    # levemente torcida (plantación a mano, distorsión del ortomosaico).
    # Donde la hilera real se aparta ~T/4 de la recta, la banda central cae
    # sobre el borde de la entrehilera y los costados casi sobre las hileras
    # vecinas: el contraste se invierte y el tramo se perdía (visto: 0,6 m
    # de apartamiento a lo largo de 45 m en una vid a 2 m). Se mide la
    # hilera corrida hasta TRACK_MAX_T*T a cada lado y, por bloques de largo
    # T, se elige el camino con más evidencia de los dos lados (índice o
    # imagen auxiliar) que sea continuo: programación dinámica con a lo sumo
    # un nivel de corrimiento por bloque, un costo por cambio de nivel y
    # otro por nivel de apartamiento de la recta (una hilera recta, o una
    # zona sin hilera, se queda en la recta). La evidencia de los pasos
    # siguientes se mide sobre ese camino; la línea de salida sigue siendo
    # una recta por tramo.
    track = {}
    info["seguidas"] = 0
    max_tr = TRACK_MAX_T * T if track_max_px is None else min(TRACK_MAX_T * T, track_max_px)
    offs_tr = np.linspace(-max_tr, max_tr, 2 * TRACK_LEVELS + 1)
    blk_tr = max(3, int(round(T)))
    nb_tr = NS // blk_tr
    lv_ix = np.arange(len(offs_tr))
    prior_tr = -TRACK_PRIOR * np.abs(lv_ix - TRACK_LEVELS)
    if nb_tr >= 3 and T >= TRACK_MIN_T_PX:
        for row_num, (row, sgn, j) in enumerate(zip(rows, aux_sign, centers), start=1):
            if stop():
                return [], [], info
            a0, a1 = row[0], row[1]
            # Solo hileras con pico propio en el perfil que ya se ven sobre su
            # recta (al menos la longitud mínima, con el mismo criterio que
            # siembra los tramos): las posiciones agregadas por la grilla o
            # por continuidad (bordes del lote, junto a calles y cortinas) no
            # se "buscan" de costado, porque ahí siempre aparece alguna sombra.
            if j in added or j in grid_added:
                continue
            pres = main_presence(row, tau)
            if aux_localiza and row[3] is not None and sgn != 0.0 and sigma_a:
                am_ = np.minimum(sgn * row[3][1], sgn * row[3][2])
                pres = pres | ((np.nan_to_num(sgn * row[3][0], nan=-np.inf) / sigma_a > k_sigma) & (am_ > 0))
            if pres.sum() < max(min_len_px, 3.0 * T):
                continue
            sc = []
            for o in offs_tr:
                m_, x_ = (row[2], row[3]) if o == 0.0 else contrast_along(a0 + o, a1)
                e_ = np.nan_to_num(np.minimum(m_[1], m_[2]), nan=-np.inf) / max(sigma, 1e-12)
                if x_ is not None and sgn != 0.0 and sigma_a:
                    e_ = np.maximum(e_, np.nan_to_num(np.minimum(sgn * x_[1], sgn * x_[2]), nan=-np.inf) / sigma_a)
                sc.append(np.clip(e_, TRACK_CLIP[0], TRACK_CLIP[1]))
            B_ = np.stack(sc)[:, :nb_tr * blk_tr].reshape(len(offs_tr), nb_tr, blk_tr).mean(axis=2)
            V = B_[:, 0] + prior_tr
            back = np.zeros((len(offs_tr), nb_tr), dtype=np.int64)
            for b_ in range(1, nb_tr):
                cand_v = np.full((3, len(offs_tr)), -np.inf)
                cand_v[1] = V
                cand_v[0, 1:] = V[:-1] - TRACK_STEP     # viene del nivel de abajo
                cand_v[2, :-1] = V[1:] - TRACK_STEP     # viene del nivel de arriba
                w_ = np.argmax(cand_v, axis=0)
                back[:, b_] = lv_ix + (w_ - 1)
                V = cand_v[w_, lv_ix] + B_[:, b_] + prior_tr
            path = np.zeros(nb_tr, dtype=np.int64)
            path[-1] = int(np.argmax(V))
            for b_ in range(nb_tr - 1, 0, -1):
                path[b_ - 1] = back[path[b_], b_]
            if not (path != TRACK_LEVELS).any():
                continue
            dt = np.interp(ss, (np.arange(nb_tr) + 0.5) * blk_tr, offs_tr[path])
            row[2], row[3] = contrast_along(a0, a1, dt)
            track[row_num] = dt
            info["seguidas"] += 1

    # Ruido recalculado con las rectas ya ubicadas (rumbo propio, pasada 2,
    # seguimiento): el de la pasada 1 sale de rectas con el rumbo del lote, que
    # en hileras levemente inclinadas se salen de la hilera y dan contrastes
    # negativos que no son ruido (visto: viñedo de la tanda 2, 21 -> 14,7; sus
    # hileras más tenues quedaban en 1-2 "ruidos" y se cortaban a pedazos).
    # Solo con las hileras de pico propio, y solo si baja.
    fin2 = [r_[2][0][np.isfinite(r_[2][0])] for r_, j_ in zip(rows, centers)
            if j_ not in added and j_ not in grid_added]
    fin2 = [x_ for x_ in fin2 if x_.size]
    if fin2:
        s2_ = noise_rms(np.concatenate(fin2))
        if 0 < s2_ < sigma:
            info["sigma_pasada1"] = sigma
            sigma = s2_
            tau = k_sigma * sigma
            info["sigma"], info["tau"] = sigma, tau

    # --- 5) tramos por hilera, con umbral por histéresis (como en Canny):
    # un tramo tiene que contener al menos una racha "fuerte" del índice
    # (> tau, de largo >= T) para existir, pero se extiende mientras haya
    # evidencia "débil" de hilera: índice > 1 sigma, o estructura auxiliar
    # consistente > k_sigma. Así el extremo cae donde la hilera realmente se
    # termina (cabecera, camino), y no donde el vigor baja del umbral
    # estricto, que es lo que dejaba los extremos irregulares.
    min_run = max(3.0, T)
    tau_low = min(tau, 1.0 * sigma)
    # El hueco máximo (plantas faltantes) es un parámetro en píxeles; en un
    # frutal de hileras muy separadas y buena resolución quedaba menor que
    # la distancia entre árboles (olivar a 12 m sobre 0,125 m/px: 7,5 m,
    # con árboles cada 7,8 m). Al menos GAP_T separaciones entre hileras.
    max_gap_px = max(max_gap_px, GAP_T * T)

    def runs(mask):
        d = np.diff(np.r_[0, mask.astype(np.int8), 0])
        return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))  # fin exclusivo

    segments, row_ids = [], []
    row_iv = []  # (row_num, a0, a1, [[i1, i2], ...]) por hilera con tramos
    tails = {}   # row_num -> [inicio, fin] candidatos de punta (fragmentos terminales)
    low_lv = []  # tramos descartados por nivel: (row_num, a0, a1, [i1, i2], f_índice, f_aux)
    w_end = int(max(3, round(3 * T)))

    def half_trim(ev_, a, e, start=True, end=True):
        """Extremos de un tramo [a, e) a media altura de su evidencia local:
        la mitad de la mediana de los primeros (o últimos) 3T, y al menos 1
        ruido. Los dos umbrales se miden antes de recortar. Devuelve (a, e)."""
        def thr_of(x_):
            return max(1.0, 0.5 * float(np.median(x_[np.isfinite(x_)]))) if np.isfinite(x_).any() else 1.0
        thr_a, thr_e = thr_of(ev_[a:min(e, a + w_end)]), thr_of(ev_[max(a, e - w_end):e])
        if start:
            while a < e - 1 and not ev_[a] >= thr_a:
                a += 1
        if end:
            while e - 1 > a and not ev_[e - 1] >= thr_e:
                e -= 1
        return a, e

    def half_start(ev_, a, e):
        """Inicio de un tramo [a, e) a media altura de su evidencia local."""
        return half_trim(ev_, a, e, end=False)[0]

    def half_end(ev_, a, e):
        """Último índice de un tramo [a, e) a media altura de su evidencia."""
        return half_trim(ev_, a, e, start=False)[1] - 1
    # Entrehileras alternadas (ver ALT_D): tolerancia de un lado en la
    # imagen auxiliar. En un viñedo con rastra hilera por medio la vid se ve
    # 8-20 ruidos más oscura que la entrehilera rastreada y casi igual a la
    # cubierta, y la exigencia de superar a las dos cortaba las hileras
    # (visto: 0,8 km de puntas cortas en 1,3 ha).
    alt_tol = 0.0
    if sigma_a:
        d_ = []
        for row in rows:
            x_ = row[3]
            v_ = (x_[1] - x_[2])[np.isfinite(x_[0])] if x_ is not None else np.zeros(0)
            d_.append(float(np.median(v_)) / sigma_a if v_.size >= 10 else np.nan)
        d_ = np.array(d_)
        if d_.size >= 2:
            ok_ = (np.isfinite(d_[:-1]) & np.isfinite(d_[1:])
                   & (np.abs(d_[:-1]) >= ALT_D) & (np.abs(d_[1:]) >= ALT_D))
            if (ok_.sum() >= ALT_PAIRS * len(rows)
                    and np.mean(np.sign(d_[:-1][ok_]) != np.sign(d_[1:][ok_])) >= ALT_OPP):
                alt_tol = ALT_TOL
    info["alternadas"] = alt_tol > 0
    for row_num, (row, sgn, j) in enumerate(zip(rows, aux_sign, centers), start=1):
        if stop():
            return [], [], info
        a0, a1 = row[0], row[1]
        con_s, dl, dr = row[2][:3]
        cmin = np.minimum(dl, dr)
        strong = presence(con_s, cmin, tau)
        # evidencia en unidades de ruido, para la histéresis y los extremos
        ev = np.where(cmin > 0, np.nan_to_num(con_s, nan=-np.inf) / max(sigma, 1e-12), -np.inf)
        weak = presence(con_s, cmin, tau_low)
        if sgn != 0.0 and sigma_a:
            acon, adl, adr = row[3][:3]
            amin = np.minimum(sgn * adl, sgn * adr) + alt_tol * sigma_a
            aev = np.where(amin > 0, np.nan_to_num(sgn * acon, nan=-np.inf) / sigma_a, -np.inf)
            weak |= aev > k_sigma
            ev = np.maximum(ev, aev)
            if j in added:
                # hilera agregada por continuidad del patrón: no tiene
                # tramos fuertes en el índice, la siembra la imagen auxiliar
                strong = aev > k_sigma
            elif aux_localiza or aux_siembra:
                # con la estructura de brillo como referencia de posición,
                # también siembra tramos en hileras bien ubicadas cuyo
                # verdor no alcanza (vides débiles, pasto entre hileras). Y al
                # revés (aux_siembra): con la franja oscura como índice, el
                # verdor siembra donde la franja oscura se ve débil (visto:
                # canopia rojiza, 36 hileras sin línea con el verdor 10-17
                # veces el ruido y la franja oscura casi sin rachas fuertes)
                strong = strong | (aev > k_sigma)
        # Microcortes de la evidencia débil (hasta T/4: una planta chica, un
        # píxel de ruido) no parten la histéresis. Partida, la continuación
        # tenía que sembrarse sola con una racha fuerte de largo T, y en un
        # frutal joven (árboles chicos, contraste intermitente) se perdía
        # entera: visto, un corte de 0,4 m y 60 m de hilera clara sin línea.
        gap_fill = int(T / 4.0)
        if gap_fill >= 1:
            rw_ = runs(weak)
            for (_, e1_), (a2_, _) in zip(rw_[:-1], rw_[1:]):
                if a2_ - e1_ <= gap_fill:
                    weak[e1_:a2_] = True
        pieces, cand = [], []
        for a, e in runs(weak):
            # rachas fuertes sueltas más cortas que T (textura de un árbol,
            # ruido) no alcanzan para sembrar un tramo; sí varias que juntas
            # suman T: en un frutal de árboles sueltos muy separados (olivar
            # a 12 m, árboles cada 7,8 m) cada árbol da una racha de 4-7 m y
            # nunca una de 12 (visto: una hilera de 256 m sin línea)
            st_ = runs(strong[a:e])
            tot_ = sum(se - sa for sa, se in st_)
            if any(se - sa >= min_run for sa, se in st_) or tot_ >= min_run:
                pieces.append((a, e))
            elif (e - a >= WEAK_SEED_T * T
                  and float(np.mean(np.clip(ev[a:e], -5.0, 20.0))) >= WEAK_SEED_M):
                # evidencia débil pero sostenida (ver WEAK_SEED_T)
                pieces.append((a, e))
                info["sembradas_debiles"] = info.get("sembradas_debiles", 0) + 1
            elif tot_ >= 3:
                cand.append((a, e))
        # Rachas débiles con algo de evidencia fuerte, a menos del hueco
        # máximo de un tramo sembrado: son la continuación de la hilera tras
        # uno o dos árboles faltantes (visto: 70 m de hilera clara de olivos
        # sin línea, con rachas fuertes de 8 y 3 m, a 10 m del tramo).
        grew = True
        while cand and pieces and grew:
            grew = False
            for c_ in list(cand):
                if any(0 <= c_[0] - pe <= max_gap_px or 0 <= pa - c_[1] <= max_gap_px for pa, pe in pieces):
                    pieces.append(c_)
                    cand.remove(c_)
                    grew = True
            pieces.sort()
        # Fragmentos terminales más cortos que la longitud mínima (p. ej.
        # un árbol verde en la prolongación de la hilera, pasando la
        # cabecera) no se unen al resto: se descartan antes de unir huecos.
        # Los fragmentos cortos INTERNOS (entre plantas faltantes) se
        # conservan. Pero el fragmento también puede ser la punta de la
        # propia hilera separada por una planta débil (visto: un corte de
        # 0,4 m y se perdían los últimos 10 m): queda como candidato y se
        # decide después, con las hileras vecinas (regularize_row_ends): se
        # suma si no pasa del borde de lo plantado que forman ellas.
        pop_a, pop_e = [], []
        while len(pieces) >= 2 and pieces[0][1] - pieces[0][0] < min_len_px:
            pop_a.append(pieces.pop(0))
        while len(pieces) >= 2 and pieces[-1][1] - pieces[-1][0] < min_len_px:
            pop_e.append(pieces.pop())
        tail = [None, None]
        nxt = pieces[0][0] if pieces else None
        for a, e in reversed(pop_a):          # del más cercano al más lejano
            if nxt - e > max_gap_px:
                break
            tail[0] = half_start(ev, a, e)
            nxt = a
        prv = pieces[-1][1] if pieces else None
        for a, e in reversed(pop_e):
            if a - prv > max_gap_px:
                break
            tail[1] = half_end(ev, a, e)
            prv = e
        merged = []
        for a, e in pieces:
            if merged and a - merged[-1][1] <= max_gap_px:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((a, e))
        intervals = []
        for a, e in merged:
            # Localizar cada extremo a media altura: el suavizado de largo T
            # convierte el final abrupto de la hilera en una rampa de largo
            # T, y un umbral bajo corta al pie de esa rampa, más allá del
            # final real. Con un filtro de caja el valor suavizado vale
            # exactamente la mitad del escalón en la posición del borde, así
            # que se recorta cada extremo hasta donde la evidencia alcanza
            # la mitad de su amplitud local.
            if e - a > 2 * w_end:
                a, e = half_trim(ev, a, e)
            i1, i2 = float(a), float(e - 1)
            if i2 - i1 >= min_len_px:
                intervals.append([i1, i2])
        # Planta por planta: la evidencia promediada a lo largo de T no ve
        # las plantas chicas (replantes, menor vigor, típicamente en la
        # punta de la hilera), pero cada una sí se ve a su escala. Plantas a
        # menos de T una de otra forman una secuencia; una secuencia que toca
        # un tramo (o queda a menos de T de él) lo prolonga o une dos tramos,
        # y una suelta cuenta sola si mide al menos 3T (una hilera entera de
        # plantas chicas). Un árbol aislado en la cabecera, a más de T de la
        # última planta, no se une.
        if sigma_p > 0:
            pmask = np.nan_to_num(row[2][3], nan=-np.inf) > k_sigma * sigma_p
            groups = []
            for pa, pe in runs(pmask):
                if groups and pa - groups[-1][1] <= gap_p:
                    groups[-1][1] = pe
                else:
                    groups.append([pa, pe])
            cov = np.zeros(NS, dtype=bool)
            for i1, i2 in intervals:
                cov[int(i1):int(i2) + 1] = True
            before = int(cov.sum())
            for pa, pe in groups:
                # el suavizado a escala de planta ensancha cada planta half_p
                ga, ge = int(math.ceil(pa + half_p)), int(math.floor(pe - 1 - half_p))
                if ge < ga:
                    continue
                lo_, hi_ = max(0, ga - int(gap_p)), min(NS, ge + int(gap_p) + 1)
                near = np.nonzero(cov[lo_:hi_])[0]
                if near.size:
                    cov[min(ga, lo_ + int(near.min())):max(ge, lo_ + int(near.max())) + 1] = True
                elif ge - ga >= min_solo:
                    cov[ga:ge + 1] = True
            if int(cov.sum()) > before:
                info["extendidas_por_plantas"] += 1
                intervals = []
                for a, e in runs(cov):
                    if intervals and a - intervals[-1][1] - 1 <= max_gap_px:
                        intervals[-1][1] = float(e - 1)
                    else:
                        intervals.append([float(a), float(e - 1)])
                intervals = [iv for iv in intervals if iv[1] - iv[0] >= min_len_px]
        # Nivel: el tramo tiene que estar, en el índice o en la imagen
        # auxiliar, al menos a mitad de camino entre la entrehilera típica y
        # la hilera típica del lote.
        if intervals and (lvl_main is not None or lvl_aux is not None):
            def lv_ok(i1_, i2_):
                # sin las rampas de las puntas (T/2 de cada lado) en tramos
                # largos: la punta a media altura es más débil y bajaba el
                # promedio justo por debajo de los umbrales
                if i2_ - i1_ > 3.0 * T:
                    i1_, i2_ = i1_ + 0.5 * T, i2_ - 0.5 * T
                sl = slice(int(i1_), int(i2_) + 1)
                f_m_ = level_frac(row[2][4], 1.0, lvl_main, sl)
                f_a_ = level_frac(row[3][4], sign_global, lvl_aux, sl) if row[3] is not None else -np.inf
                return max(f_m_, f_a_) >= level_min, f_m_, f_a_

            keep = []
            # Posiciones agregadas (grilla del marco, continuidad del patrón:
            # las de borde) donde el tramo es mucho MÁS marcado que una hilera
            # típica: una cortina o una fila de árboles junto al lote, no una
            # hilera de plantas chicas (ver LEVEL_ADDED_MAX).
            add_row = j in added or j in grid_added
            for iv in intervals:
                ok_lv, f_m, f_a = lv_ok(iv[0], iv[1])
                if add_row and max(f_m, f_a) > LEVEL_ADDED_MAX:
                    info["descartados_cortina"] = info.get("descartados_cortina", 0) + 1
                    continue
                if ok_lv:
                    keep.append(iv)
                    continue
                # Si el tramo entero no llega, se prueban sus partes con
                # evidencia continua (las rachas débiles que se unieron por
                # huecos): una parte más pobre de la hilera (plantas chicas,
                # un manchón) bajaba el promedio y se descartaba la hilera
                # entera, también la parte con nivel de hilera.
                parts = []
                for pa, pe in runs(weak[int(iv[0]):int(iv[1]) + 1]):
                    p1, p2 = iv[0] + pa, iv[0] + pe - 1
                    if p2 - p1 >= min_len_px and lv_ok(p1, p2)[0]:
                        if parts and p1 - parts[-1][1] <= max_gap_px:
                            parts[-1][1] = p2
                        else:
                            parts.append([p1, p2])
                # el tramo entero sigue siendo candidato al rescate de plantas
                # chicas (más abajo), que lo recupera completo si corresponde
                info["descartados_por_nivel"] += 1
                low_lv.append((row_num, a0, a1, list(iv), f_m, f_a))
                keep.extend(parts)
            intervals = keep
        if intervals:
            row_iv.append((row_num, a0, a1, intervals))
            if tail[0] is not None or tail[1] is not None:
                tails[row_num] = tail

    # --- Fila de árboles distintos en el borde del lote (árboles de la
    # calle, cortina): el perfil la ubica como una hilera más, pero su nivel
    # (contraste con la entrehilera, relativo al de una hilera típica) es
    # muy superior al de las hileras interiores del mismo lote. Se descarta
    # la hilera extrema de cada lado si supera EDGE_LEVEL_MIN y EDGE_LEVEL_REL
    # veces el percentil 99 de las interiores (relativo: en lotes con
    # canopia casi negra las interiores también dan valores altos).
    edge_dropped = set()   # hileras extremas descartadas: el rescate no las revive
    if len(row_iv) >= 6 and (lvl_main is not None or lvl_aux is not None):
        def row_level(r_):
            rn_, _, _, ivs_ = r_
            row_ = rows[rn_ - 1]
            tot_, acc_ = 0.0, 0.0
            for i1_, i2_ in ivs_:
                sl_ = slice(int(i1_), int(i2_) + 1)
                f_m_ = level_frac(row_[2][4], 1.0, lvl_main, sl_)
                f_a_ = level_frac(row_[3][4], sign_global, lvl_aux, sl_) if row_[3] is not None else -np.inf
                v_ = max(f_m_, f_a_)
                if np.isfinite(v_):
                    tot_ += i2_ - i1_ + 1
                    acc_ += (i2_ - i1_ + 1) * v_
            return acc_ / tot_ if tot_ > 0 else float("nan")

        lv_rows = np.array([row_level(r_) for r_ in row_iv])
        inner_ = lv_rows[1:-1][np.isfinite(lv_rows[1:-1])]
        if inner_.size >= 4:
            thr_ = max(EDGE_LEVEL_MIN, EDGE_LEVEL_REL * float(np.percentile(inner_, 99)))
            for k_ in (len(row_iv) - 1, 0):
                if np.isfinite(lv_rows[k_]) and lv_rows[k_] > thr_:
                    edge_dropped.add(row_iv.pop(k_)[0])
                    info["descartados_cortina"] = info.get("descartados_cortina", 0) + 1

    # --- Plantas chicas o ralas (típicamente las hileras de borde de un
    # frutal, o una hilera replantada): su tramo tiene contraste claro con
    # las dos entrehileras, pero su NIVEL promedio a lo largo de la hilera
    # queda cerca del de la entrehilera, porque entre planta y planta hay
    # suelo, y el control de nivel lo descartaba junto con las huellas de
    # tractor. Se rescata si (1) en el brillo es algo más oscuro que la
    # entrehilera típica (LEVEL_DARK_MIN: copa y sombra de un árbol, aunque
    # sea chico; unos arbustos claros sueltos sobre un sendero no lo son),
    # (2) en el otro índice no queda más allá de la entrehilera del lado del
    # suelo desnudo (LEVEL_FLOOR: una huella o un borde sobre la calle, que
    # es suelo desnudo y claro, queda muy por debajo en uno de los dos) y
    # (3) está dentro del largo que ocupan las hileras vecinas (±T): no se
    # agregan tramos en la cabecera. Sin imagen auxiliar, o con hileras
    # finas, no se rescata nada.
    info["recuperados_por_nivel"] = 0
    if low_lv and T >= TRACK_MIN_T_PX:
        acc = {r_[0]: r_ for r_ in row_iv}
        new_iv = {}
        for rn, a0_, a1_, iv, f_m, f_a in low_lv:
            if rn in edge_dropped or not (np.isfinite(f_a) and np.isfinite(f_m)):
                continue
            # cuál de los dos es el brillo (franja oscura = positivo): la
            # imagen auxiliar, salvo en los lotes donde el índice principal
            # ya es el brillo
            f_dark, f_other = (f_a, f_m) if aux_localiza else (f_m, f_a)
            if f_dark < LEVEL_DARK_MIN or f_other < LEVEL_FLOOR:
                continue
            nbs = [acc[q] for q in range(rn - 3, rn + 4) if q != rn and q in acc]
            if len(nbs) < 2:
                continue
            lo_ = min(x[0] for r_ in nbs for x in r_[3]); hi_ = max(x[1] for r_ in nbs for x in r_[3])
            if iv[0] < lo_ - T or iv[1] > hi_ + T:
                continue
            new_iv.setdefault(rn, (a0_, a1_, []))[2].append(iv)
            info["recuperados_por_nivel"] += 1
        for rn, (a0_, a1_, ivs) in new_iv.items():
            own = acc.get(rn)
            allv = sorted([list(x) for x in (own[3] if own else [])] + ivs)
            merged_iv = []
            for a, e in allv:
                if merged_iv and a <= merged_iv[-1][1] + max_gap_px:
                    merged_iv[-1][1] = max(merged_iv[-1][1], e)
                else:
                    merged_iv.append([a, e])
            acc[rn] = (rn, own[1] if own else a0_, own[2] if own else a1_, merged_iv)
        if new_iv:
            row_iv = [acc[k_] for k_ in sorted(acc)]

    # --- Modo patrón (hileras finas, al límite de la resolución): el lote
    # muestra un patrón de hileras clarísimo en conjunto (el perfil ubicó
    # las hileras) pero casi ninguna hilera tiene evidencia propia: su
    # contraste, a lo largo de un tramo de largo T, apenas supera el ruido
    # (visto: hortaliza a 0,85 m en una imagen de 0,2 m, relación
    # señal/ruido ~1,2 por hilera; hacen falta ~40 m de hilera para llegar a
    # 3). Se promedia el contraste de PATTERN_ROWS hileras vecinas a lo largo
    # de PATTERN_LEN_T separaciones y cada hilera se marca donde ese patrón
    # conjunto supera k_sigma veces SU ruido (el del promedio). La posición
    # de cada línea es la del perfil del lote; los extremos tienen la
    # precisión del promedio y no se ven fallas de plantas. En viñedos y
    # frutales no se activa: ahí casi todas las hileras tienen evidencia
    # propia.
    info["modo_patron"] = False
    n_rows_all = len(rows)
    if n_rows_all >= 10 and len(row_iv) < PATTERN_MAX_FRAC * n_rows_all and sigma > 0:
        Lp = int(round(max(PATTERN_LEN_T * T, min_len_px)))
        ksz = (PATTERN_ROWS, Lp)  # (ancho = hileras, alto = a lo largo)
        full = PATTERN_ROWS * Lp

        def pattern_t(C):
            """Prueba t local del contraste C (NS x hileras): media en la
            ventana dividida por su error, con la varianza de la PROPIA
            ventana (la textura de árboles o techos es mucho mayor que la
            del cultivo; con el ruido del lote daba falsos positivos). El
            contraste ya está suavizado a lo largo de T: ~N/T muestras
            independientes."""
            ok_ = np.isfinite(C)
            Cz = np.where(ok_, C, 0.0)
            S = cv2.boxFilter(Cz, -1, ksz, normalize=False, borderType=cv2.BORDER_CONSTANT)
            S2 = cv2.boxFilter(Cz * Cz, -1, ksz, normalize=False, borderType=cv2.BORDER_CONSTANT)
            Nn = cv2.boxFilter(ok_.astype(np.float64), -1, ksz, normalize=False, borderType=cv2.BORDER_CONSTANT)
            m_ = S / np.maximum(Nn, 1e-9)
            var = np.maximum(S2 / np.maximum(Nn, 1e-9) - m_ * m_, 1e-30)
            t_ = m_ / np.sqrt(var / np.maximum(Nn / max(T, 1.0), 1.0))
            return np.where(Nn >= 0.5 * full, t_, -np.inf)

        # contraste contra AMBOS costados (el borde de un árbol o de un
        # camino contrasta de un solo lado), en el índice y, con su signo,
        # en la imagen auxiliar: en una parte del lote las hileras finas se
        # ven por verdor y en otra por brillo
        # (los contrastes contra cada costado valen 0, no NaN, fuera de lo
        # válido: se enmascaran con la validez de la hilera para que no
        # cuenten como muestras en las ventanas que tocan el borde)
        okc = np.isfinite(np.stack([r[2][0] for r in rows], axis=1))
        cm_main = np.where(okc, np.stack([np.minimum(r[2][1], r[2][2]) for r in rows], axis=1), np.nan)
        tstat = pattern_t(cm_main.astype(np.float64))
        if RA is not None and sign_global != 0.0:
            cm_aux = np.where(okc, np.stack([np.minimum(sign_global * r[3][1], sign_global * r[3][2])
                                             for r in rows], axis=1), np.nan)
            tstat = np.maximum(tstat, pattern_t(cm_aux.astype(np.float64)))
        mean = tstat
        present = (tstat > k_sigma) & okc
        p_iv = []
        for k, row in enumerate(rows):
            ivs = []
            for a, e in runs(present[:, k]):
                if ivs and a - ivs[-1][1] <= max(max_gap_px, Lp / 2.0):
                    ivs[-1][1] = e
                else:
                    ivs.append([a, e])
            out_iv = []
            for a, e in ivs:
                m_ = mean[a:e, k]
                half_h = 0.5 * float(np.median(m_[np.isfinite(m_)])) if np.isfinite(m_).any() else 0.0
                while a < e - 1 and not mean[a, k] >= half_h:
                    a += 1
                while e - 1 > a and not mean[e - 1, k] >= half_h:
                    e -= 1
                if e - 1 - a >= max(min_len_px, Lp):
                    out_iv.append([float(a), float(e - 1)])
            if out_iv:
                p_iv.append((k + 1, row[0], row[1], out_iv))
        # Se suma a lo que cada hilera ya tenía: los tramos propios conservan
        # sus extremos (más precisos que los del patrón) y el patrón completa
        # el resto de la hilera (unión de intervalos).
        own = {r_[0]: r_ for r_ in row_iv}
        n_changed = 0
        for rn, a0p, a1p, ivp in p_iv:
            if rn not in own:
                own[rn] = (rn, a0p, a1p, ivp)
                n_changed += 1
                continue
            _, a0o, a1o, ivo = own[rn]
            merged_iv = []
            for a, e in sorted([list(iv) for iv in ivo] + [list(iv) for iv in ivp]):
                if merged_iv and a <= merged_iv[-1][1] + max_gap_px:
                    merged_iv[-1][1] = max(merged_iv[-1][1], e)
                else:
                    merged_iv.append([a, e])
            if sum(e - a for a, e in merged_iv) > sum(e - a for a, e in ivo) + 1:
                n_changed += 1
            own[rn] = (rn, a0o, a1o, merged_iv)
        if n_changed:
            row_iv = [own[k_] for k_ in sorted(own)]
            info["modo_patron"] = True
            info["patron_largo_px"] = Lp
            info["n_patron"] = n_changed

    # hileras de la grilla completa: cuentan solo las que tienen tramos
    with_iv = {r[0] for r in row_iv}
    grid_with = sum(1 for n_, j in enumerate(centers, start=1) if j in grid_added and n_ in with_iv)
    info["n_grilla"] = grid_with
    info["n_hileras"] = len(centers) - len(grid_added) + grid_with

    regularize_row_ends(row_iv, T, info, tails=tails)
    fill_by_neighbors(row_iv, rows, T, sigma, info)
    cut_aligned_gaps(row_iv, rows, T, NS, info)
    # diagnóstico: posición perpendicular (px de img, t = -x*sin + y*cos del
    # ángulo) de cada hilera ubicada, y cuáles tienen tramos
    info["filas_t"] = [t0 + row[0] for row in rows]
    info["filas_con_tramos"] = sorted(r[0] for r in row_iv)

    info["tramos_reajustados"] = 0
    # Línea de salida al centro a media altura de la banda (ver HALF_MIN_T).
    # El perfil a lo ancho de cada hilera se toma siguiendo su propia recta (y
    # el camino seguido) solo donde tiene tramos: el perfil de columnas de todo
    # el lote borronea la forma de la banda en hileras largas (corría 0,15 m
    # donde hacían falta 0,9 m).
    hc_shift = {}
    if half_center:
        step_o = 0.25
        oo = np.arange(-T, T + 1e-9, step_o)
        sig_o = max(1.0, (T / 16.0) / step_o)
        rk = int(math.ceil(3 * sig_o))
        gko = np.exp(-0.5 * (np.arange(-rk, rk + 1) / sig_o) ** 2)
        gko /= gko.sum()
        c0i = len(oo) // 2
        w4 = int(math.ceil((T / 4.0) / step_o))
        for row_num, a0, a1, intervals in row_iv:
            if not intervals:
                continue
            sidx = np.concatenate([np.arange(max(0, int(i1)), min(NS - 1, int(i2)) + 1) for i1, i2 in intervals])
            if len(sidx) < 3 * T:
                continue
            dtr = track.get(row_num)
            tt = a0 + a1 * sidx + (dtr[sidx] if dtr is not None else 0.0)
            cc = tt[:, None] + oo[None, :]
            k0 = np.floor(cc).astype(np.int64)
            fr = cc - k0
            okc = (k0 >= 0) & (k0 + 1 < NT)
            k0c = np.clip(k0, 0, NT - 2)
            s_ = sidx[:, None]
            okc &= RV[s_, k0c] & RV[s_, k0c + 1]
            cnt_ = okc.sum(axis=0)
            if cnt_.min() < 0.5 * len(sidx):
                continue
            vv = R[s_, k0c] * (1 - fr) + R[s_, k0c + 1] * fr
            pr = np.where(okc, vv, 0.0).sum(axis=0) / np.maximum(cnt_, 1)
            pr = np.convolve(np.pad(pr, rk, mode="edge"), gko, mode="valid")
            k_ = c0i - w4 + int(np.argmax(pr[c0i - w4:c0i + w4 + 1]))
            if not 0 < k_ < len(pr) - 1:
                continue
            hc = half_height_center(pr, k_, T / step_o)
            if hc is None:
                continue
            # la asimetría de la banda: cuánto se aparta su centro a media
            # altura de su punto más oscuro (sub-muestra). La línea se corre eso
            # mismo: medido desde la línea (y no desde el pico) corría menos y
            # quedaba más lejos de la corrección a mano (F1 0,595 -> 0,543 y
            # 0,793 -> 0,779 en los dos viñedos)
            a_, b_, c_ = pr[k_ - 1], pr[k_], pr[k_ + 1]
            den = a_ - 2.0 * b_ + c_
            pk = k_ + (float(np.clip(0.5 * (a_ - c_) / den, -0.5, 0.5)) if abs(den) > 1e-12 else 0.0)
            hc_shift[row_num] = float(np.clip((hc - pk) * step_o, -HALF_MAX_T * T, HALF_MAX_T * T))
        # un solo corrimiento para todo el lote (la sombra cae siempre del
        # mismo lado), solo si es claro y casi todas las hileras lo comparten
        vals = np.array(list(hc_shift.values()))
        med = float(np.median(vals)) if len(vals) >= 5 else 0.0
        same = float(np.mean(np.sign(vals) == np.sign(med))) if len(vals) else 0.0
        if abs(med) >= HALF_MIN_T * T and same >= HALF_SAME:
            hc_shift = {k_: med for k_ in hc_shift}
            info["centro_media_altura"] = med
        else:
            hc_shift = {}
    for row_num, a0, a1, intervals in row_iv:
        dt = track.get(row_num)
        a0 = a0 + hc_shift.get(row_num, 0.0)
        for i1, i2 in intervals:
            # la alineación de extremos puede llevar una punta a la
            # prolongación del borde de las vecinas, fuera de la grilla
            i1, i2 = max(float(i1), 0.0), min(float(i2), NS - 1.0)
            if i2 - i1 < min_len_px:
                continue
            c0_, c1_ = a0, a1
            if dt is not None:
                seg_i = np.arange(int(i1), int(i2) + 1)
                if np.abs(dt[seg_i]).max() >= TRACK_REFIT_T * T:
                    # recta del tramo ajustada al camino seguido
                    c1_, c0_ = np.polyfit(seg_i.astype(np.float64), a0 + a1 * seg_i + dt[seg_i], 1)
                    info["tramos_reajustados"] += 1
            pts = []
            for i in (i1, i2):
                s = s0 + i
                t = t0 + c0_ + c1_ * i
                pts.append((s * u[0] + t * n[0], s * u[1] + t * n[1]))
            segments.append((pts[0][0], pts[0][1], pts[1][0], pts[1][1]))
            row_ids.append(row_num)

    return segments, row_ids, info


def half_height_center(p, k, T):
    """Centro a media altura del pico k del perfil p (T en muestras de p):
    punto medio entre los dos cruces del nivel (pico + fondo)/2, con el fondo
    = media de los mínimos a cada lado dentro de ±0,9 T (la franja clara de un
    lado puede quedar a más de T/2 del pico si la banda es asimétrica). None
    si no cruza a ambos lados."""
    n = len(p)
    h = max(2, int(round(0.9 * T)))
    lo, hi = max(0, k - h), min(n - 1, k + h)
    if k - lo < 1 or hi - k < 1:
        return None
    base = 0.5 * (float(p[lo:k].min()) + float(p[k + 1:hi + 1].min()))
    lev = 0.5 * (float(p[k]) + base)
    i = k
    while i > lo and p[i] >= lev:
        i -= 1
    if p[i] >= lev:
        return None
    xl = i + (lev - p[i]) / (p[i + 1] - p[i])
    i = k
    while i < hi and p[i] >= lev:
        i += 1
    if p[i] >= lev:
        return None
    xr = (i - 1) + (p[i - 1] - lev) / (p[i - 1] - p[i])
    return 0.5 * (xl + xr)


# Relleno con las vecinas: una hilera que se ve de punta a punta pero más
# tenue que sus vecinas (en la tanda 2, hileras a 1-3 ruidos entre vecinas a
# 10-14) salía con huecos de 10-30 m o le faltaba una punta entera. Donde sus
# DOS vecinas tienen línea (al menos NB_COVER del tramo), un hueco interno de la
# hilera, o una punta que le falta (al menos NB_END_T separaciones), se
# completa si ahí la hilera sigue siendo más oscura que sus dos entrehileras en
# al menos NB_POS del tramo, con contraste mediano >= NB_EV ruidos. Donde la
# sombra de una fila de árboles o una reguera cortan la hilera, cortan también
# a las vecinas: eso no se rellena.
NB_COVER = 0.9
NB_POS = 0.6
NB_EV = 0.5
NB_END_T = 3.0


def fill_by_neighbors(row_iv, rows, T, sigma, info):
    """Completa en el lugar los huecos y puntas faltantes de cada hilera de
    row_iv donde sus dos vecinas (números de hilera ±1) tienen línea (ver
    NB_COVER). rows: filas de detect_rows_by_profile (row[2][:3] =
    contrastes con las entrehileras); sigma: ruido del contraste."""
    info["rellenos_vecinas"] = 0
    by_rn = {r_[0]: r_[3] for r_ in row_iv}

    def covered(ivs, a, e):
        """Fracción de [a, e] cubierta por los tramos ivs."""
        if e <= a:
            return 1.0
        return sum(max(0.0, min(e, i2) - max(a, i1)) for i1, i2 in ivs) / (e - a)

    def weak_ok(rn, a, e):
        """¿La hilera rn sigue más oscura que sus entrehileras en [a, e]?"""
        con, dl, dr = rows[rn - 1][2][:3]
        sl = slice(int(max(a, 0)), int(e) + 1)
        c_, m_ = con[sl], np.minimum(dl[sl], dr[sl])
        ok = np.isfinite(c_)
        if ok.sum() < 0.5 * max(1, len(c_)):
            return False
        return (float(np.mean(m_[ok] > 0)) >= NB_POS
                and float(np.median(c_[ok])) / max(sigma, 1e-12) >= NB_EV)

    new_all = {}
    for k, (rn, a0, a1, ivs) in enumerate(row_iv):
        nb = [by_rn.get(rn - 1), by_rn.get(rn + 1)]
        if any(v is None for v in nb):
            continue
        merged = [[x[0], x[1]] for x in ivs[:1]]
        for i1, i2 in ivs[1:]:
            a, e = merged[-1][1], i1
            if all(covered(v, a, e) >= NB_COVER for v in nb) and weak_ok(rn, a, e):
                merged[-1][1] = i2
                info["rellenos_vecinas"] += 1
            else:
                merged.append([i1, i2])
        st = max(v[0][0] for v in nb)   # hasta donde llegan las dos vecinas
        if (merged[0][0] - st >= NB_END_T * T and all(covered(v, st, merged[0][0]) >= NB_COVER for v in nb)
                and weak_ok(rn, st, merged[0][0])):
            merged[0][0] = float(st)
            info["rellenos_vecinas"] += 1
        en = min(v[-1][1] for v in nb)
        if (en - merged[-1][1] >= NB_END_T * T and all(covered(v, merged[-1][1], en) >= NB_COVER for v in nb)
                and weak_ok(rn, merged[-1][1], en)):
            merged[-1][1] = float(en)
            info["rellenos_vecinas"] += 1
        new_all[k] = merged
    for k, merged in new_all.items():
        rn, a0, a1, _ = row_iv[k]
        row_iv[k] = (rn, a0, a1, merged)


# Cortes transversales: una reguera, acequia o camino angosto que cruza las
# hileras en diagonal deja en cada hilera un hueco de 2-5 m que la histéresis
# une (como una planta faltante) y la línea lo atravesaba (visto: viñedo de la
# tanda 2 partido por una reguera; el usuario corta las líneas a los dos lados).
# Los huecos de una reguera caen todos sobre una misma recta que atraviesa
# muchas hileras; los de plantas faltantes, sueltos. Se buscan rectas de huecos
# (contraste a escala de planta < 0, de al menos T/3) que junten al menos
# CUT_MIN_ROWS hileras y al menos CUT_FRAC de las hileras que la recta cruza
# (pendiente |ds/dt| <= CUT_M_MAX: no casi paralela a las hileras; hasta
# CUT_MAX_LINES rectas por lote), y ahí se cortan las líneas. Solo cuentan
# hileras casi continuas (huecos < CUT_MAX_GAPFRAC del largo): en un frutal de
# árboles sueltos los huecos entre árboles también se alinean, pero son la
# mitad de la hilera.
CUT_MIN_ROWS = 8
CUT_FRAC = 0.5
CUT_MAX_GAPFRAC = 0.15
CUT_M_MAX = 4.0
CUT_MAX_LINES = 5


def cut_aligned_gaps(row_iv, rows, T, NS, info):
    """Corta los tramos de row_iv (en el lugar) en los huecos alineados de
    un corte transversal (ver CUT_MIN_ROWS). rows: filas de
    detect_rows_by_profile (row[2][3] = contraste a escala de planta)."""
    info["cortes_transversales"] = 0
    gaps = []   # (índice en row_iv, t, s, inicio, fin) de cada hueco
    for k, (rn, a0, a1, ivs) in enumerate(row_iv):
        pl = rows[rn - 1][2][3]
        tot, rg = 0.0, []
        for i1, i2 in ivs:
            i1, i2 = int(max(i1, 0)), int(min(i2, NS - 1))
            m = np.nan_to_num(pl[i1:i2 + 1], nan=0.0) < 0
            d = np.diff(np.r_[0, m.astype(np.int8), 0])
            for a, e in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
                tot += e - a
                if e - a >= T / 3.0:
                    sm = i1 + 0.5 * (a + e)
                    rg.append((k, a0 + a1 * sm, sm, i1 + a, i1 + e - 1))
        L = sum(i2 - i1 for i1, i2 in ivs)
        if L > 0 and tot <= CUT_MAX_GAPFRAC * L:
            gaps += rg
    if len(gaps) < CUT_MIN_ROWS:
        return
    G = np.array(gaps)
    t_rows = np.array([a0 + a1 * 0.5 * (ivs[0][0] + ivs[-1][1]) for _, a0, a1, ivs in row_iv])
    span = max(float(np.ptp(t_rows)), T)
    # pendientes de a T/4 de diferencia a lo ancho de todo el lote
    ms = np.arange(-CUT_M_MAX, CUT_M_MAX + 1e-9, 0.25 * T / span)
    used = np.zeros(len(G), bool)
    cuts = {}   # índice en row_iv -> [(inicio, fin)]
    ub = None   # por pendiente: el mejor conteo de una vuelta anterior (cota superior)
    for _ in range(CUT_MAX_LINES):
        idx = np.nonzero(~used)[0]
        if len(idx) < CUT_MIN_ROWS or len(ms) == 0:
            break
        g_t, g_s = G[idx, 1], G[idx, 2]
        ar_ = np.arange(len(idx))

        def best_at(m, g_t=g_t, g_s=g_s, ar_=ar_):
            """Más huecos a menos de T/2 de una recta de pendiente m (barrido
            de b): (cuántos, b de la recta)."""
            bs = np.sort(g_s - m * g_t)
            cnt = np.searchsorted(bs, bs + T, side="right") - ar_
            q = int(np.argmax(cnt))
            return int(cnt[q]), bs[q] + 0.5 * T

        # recta s = m t + b con más huecos: la primera pendiente con el máximo
        if ub is None:
            found = [best_at(m) for m in ms]
            ub = np.array([n for n, _ in found])
            k_best = int(np.argmax(ub))
            n_, b = found[k_best]
        else:
            # En las vueltas siguientes solo se sacaron huecos (los de la recta
            # anterior), así que el conteo de cada pendiente solo pudo bajar: el
            # de antes es una cota superior. Se recalculan de mayor a menor cota
            # hasta que ninguna de las que quedan puede alcanzar al mejor; el
            # resultado es el mismo que barriendo todas (antes, cinco barridos
            # completos por lote: una cuarta parte del tiempo en un lote de 400
            # hileras).
            n_, k_best, b = 0, None, None
            for k in np.argsort(-ub, kind="stable"):
                if ub[k] < n_:
                    break
                nk, bk = best_at(ms[k])
                ub[k] = nk
                if nk > n_ or (nk == n_ and k < k_best):
                    n_, k_best, b = nk, int(k), bk
        m = ms[k_best]
        if n_ < CUT_MIN_ROWS:
            break
        sup = idx[np.abs(g_s - (m * g_t + b)) <= 0.5 * T]
        rows_sup = set(G[sup, 0].astype(int))
        crossed = 0
        for k, (rn, a0, a1, ivs) in enumerate(row_iv):
            s_ = m * (a0 + a1 * 0.5 * (ivs[0][0] + ivs[-1][1])) + b
            if any(i1 <= s_ <= i2 for i1, i2 in ivs) or k in rows_sup:
                crossed += 1
        used[sup] = True
        if len(rows_sup) < CUT_MIN_ROWS or len(rows_sup) < CUT_FRAC * crossed:
            continue
        info["cortes_transversales"] += 1
        for g in G[sup]:
            cuts.setdefault(int(g[0]), []).append((g[3], g[4]))
    for k, cl in cuts.items():
        rn, a0, a1, ivs = row_iv[k]
        for a, e in cl:
            new = []
            for i1, i2 in ivs:
                if i1 < a and e < i2:
                    new += [[i1, a - 1.0], [e + 1.0, i2]]
                else:
                    new.append([i1, i2])
            ivs = new
        row_iv[k] = (rn, a0, a1, ivs)


def regularize_row_ends(row_iv, T, info=None, n_neighbors=4, tails=None):
    """Alinea con el borde de lo plantado los extremos de hilera que se
    apartan de él. row_iv: [(row_num, a0, a1, [[i1, i2], ...]), ...] en
    orden perpendicular, con posiciones i a lo largo de la hilera (misma
    coordenada para todas las hileras, porque son paralelas). Modifica los
    intervalos en el lugar.

    El borde de un lote plantado (la línea donde terminan las hileras,
    contra la calle de cabecera) es continuo: el extremo de cada hilera
    debería estar alineado con los de sus vecinas. Para cada extremo
    exterior (el primero y el último de la hilera) se ajusta una recta
    local s = a + b*t con los extremos de hasta n_neighbors hileras a cada
    lado (sin la propia), con un paso de descarte de atípicos. Si las
    vecinas forman un borde consistente (residuo RMS <= 2 max(1, T/4)) y el
    extremo se aparta más de T/2 y de 3 RMS (y no más de 3T), se lo lleva a
    ese borde:
    recorta hileras que se meten en la calle y completa las que quedan
    cortas por una o dos plantas débiles en la punta. Donde el borde tiene
    un escalón real (p. ej. un rincón sin plantar) las vecinas no son
    consistentes y el extremo no se toca; desvíos mayores a 3T se dejan
    como están (pueden ser reales: plantas arrancadas en la punta).

    tails: {row_num: [inicio, fin]} puntas candidatas (fragmentos
    terminales cortos separados del resto de la hilera por un corte chico).
    Se suman si llegan como máximo hasta el borde de las vecinas (+T/2): una
    punta separada por una planta débil sí, un árbol pasando la cabecera no
    (queda más allá del borde). Sin un borde consistente de las vecinas no
    se suman."""
    if info is not None:
        info["puntas_recuperadas"] = 0
    if len(row_iv) < 3:
        return
    tol_fit = max(1.0, T / 4.0)
    dev_min, dev_max = T / 2.0, 3.0 * T
    moved = 0

    def border(tt, ss, k):
        """Borde de lo plantado que forman las vecinas de k, en su posición,
        y el desvío cuadrático medio de las vecinas respecto de él ((None,
        None) si las vecinas no forman un borde común)."""
        nb = [q for q in range(max(0, k - n_neighbors), min(len(row_iv), k + n_neighbors + 1)) if q != k]
        if len(nb) < 4:
            return None, None
        t_nb, s_nb = tt[nb], ss[nb]
        keep = np.ones(len(nb), dtype=bool)
        resid = None
        for _ in range(2):
            if keep.sum() < 4 or np.ptp(t_nb[keep]) < 1e-9:
                break
            b, a = np.polyfit(t_nb[keep], s_nb[keep], 1)
            resid = s_nb - (a + b * t_nb)
            keep = np.abs(resid) <= max(2.0 * tol_fit, 2.5 * float(np.median(np.abs(resid[keep]))))
        if resid is None or keep.sum() < max(4, int(0.75 * len(nb))):
            return None, None  # vecinas sin un borde común (escalón real)
        rms = float(np.sqrt(np.mean(resid[keep] ** 2)))
        if rms > 2.0 * tol_fit:
            return None, None
        return a + b * tt[k], rms

    for which in (0, 1):  # 0: primer extremo (inicio), 1: último (fin)
        # puntas candidatas: hasta dos vueltas (una punta recuperada puede
        # completar el borde que necesita la de la hilera de al lado)
        for _ in range(2):
            tt = np.array([a0 + a1 * (iv[0][0] if which == 0 else iv[-1][1]) for _, a0, a1, iv in row_iv])
            ss = np.array([iv[0][0] if which == 0 else iv[-1][1] for _, _, _, iv in row_iv])
            took = 0
            for k, (rn, _, _, iv) in enumerate(row_iv):
                cand = (tails or {}).get(rn, [None, None])[which]
                if cand is None:
                    continue
                pred, rms = border(tt, ss, k)
                if pred is None or rms > tol_fit:
                    continue
                if which == 1 and ss[k] < cand <= pred + dev_min:
                    iv[-1][1] = float(cand)
                elif which == 0 and pred - dev_min <= cand < ss[k]:
                    iv[0][0] = float(cand)
                else:
                    continue
                tails[rn][which] = None
                took += 1
            if info is not None:
                info["puntas_recuperadas"] += took
            if not took:
                break
        tt = np.array([a0 + a1 * (iv[0][0] if which == 0 else iv[-1][1]) for _, a0, a1, iv in row_iv])
        ss = np.array([iv[0][0] if which == 0 else iv[-1][1] for _, _, _, iv in row_iv])
        new_ss = ss.copy()
        for k in range(len(row_iv)):
            pred, rms = border(tt, ss, k)
            if pred is None:
                continue
            # el borde de las vecinas tiene su propio ruido (plantas de la
            # punta más o menos grandes): se tolera hasta 2 x T/4 de desvío
            # medio, pero entonces el extremo tiene que apartarse al menos 3
            # veces ese ruido (antes, con más de T/4 no se alineaba nada:
            # visto, una hilera 3,5 m metida en la calle de cabecera de un
            # viñedo con puntas que variaban ±0,5 m)
            dev = ss[k] - pred
            if max(dev_min, 3.0 * rms) < abs(dev) <= dev_max:
                new_ss[k] = pred
        for k, (_, _, _, iv) in enumerate(row_iv):
            if new_ss[k] != ss[k]:
                if which == 0 and new_ss[k] < iv[0][1]:
                    iv[0][0] = float(new_ss[k])
                    moved += 1
                elif which == 1 and new_ss[k] > iv[-1][0]:
                    iv[-1][1] = float(new_ss[k])
                    moved += 1
    if info is not None:
        info["extremos_regularizados"] = moved


def fit_intervals_to_contour(intervals, pieces, max_ext):
    """Ajusta los tramos detectados de UNA hilera al contorno de la parcela,
    trabajando en 1D sobre la recta de la hilera.

    intervals: tramos detectados, [(a, b), ...] en posición a lo largo de
        la hilera.
    pieces: partes de esa misma recta que quedan dentro del contorno,
        [(c, d), ...] en la misma coordenada (más de una si el contorno es
        cóncavo o tiene huecos).
    max_ext: extensión máxima de un extremo hasta el borde del contorno
        (misma unidad). None o <= 0: no extender, solo recortar.

    Dentro de cada parte del contorno, el primer tramo se extiende hasta el
    borde de entrada y el último hasta el de salida, si a cada extremo le
    falta como máximo max_ext; si le falta más, se deja donde terminó la
    hilera (p. ej. contra un rincón con árboles o construcciones que el
    contorno incluye pero donde no hay hilera). Los huecos internos entre
    tramos no se tocan. Finalmente todo se recorta estrictamente a las
    partes del contorno: nada queda afuera. Devuelve [(a, b), ...]."""
    out = []
    for c, d in sorted(pieces):
        inside = [(max(a, c), min(b, d)) for a, b in sorted(intervals) if min(b, d) > max(a, c)]
        if not inside:
            continue
        if max_ext is not None and max_ext > 0:
            a0, b0 = inside[0]
            if a0 - c <= max_ext:
                inside[0] = (c, b0)
            a1, b1 = inside[-1]
            if d - b1 <= max_ext:
                inside[-1] = (a1, d)
        out.extend(inside)
    return out


class DetectCropRowsAlgorithm(QgsProcessingAlgorithm):

    INPUT = "INPUT"
    RED_BAND = "RED_BAND"
    GREEN_BAND = "GREEN_BAND"
    BLUE_BAND = "BLUE_BAND"
    ROW_SPACING = "ROW_SPACING"
    SENSITIVITY = "SENSITIVITY"
    CONTOUR = "CONTOUR"
    AUTO_BLOCKS = "AUTO_BLOCKS"
    OUT_BLOCKS = "OUT_BLOCKS"
    DETECT_FAILURES = "DETECT_FAILURES"
    FAIL_THRESHOLD = "FAIL_THRESHOLD"
    FAIL_MIN_LEN = "FAIL_MIN_LEN"
    OUT_FAILURES = "OUT_FAILURES"
    EXTEND_MAX = "EXTEND_MAX"
    MIN_LENGTH_M = "MIN_LENGTH_M"
    MAX_GAP_M = "MAX_GAP_M"
    SPACING_FIELD = "SPACING_FIELD"
    STRIPE = "STRIPE"
    OUTPUT = "OUTPUT"

    # Qué franja es la hilera (método de perfil)
    STRIPE_AUTO = 0      # la más verde, salvo copa cerrada (ver _detect_rows_profile)
    STRIPE_CHROMA = 1    # siempre la más verde (verdor cromático; comportamiento hasta v0.12)
    STRIPE_SUNLIT = 2    # siempre las copas iluminadas (verde absoluto 2G-R-B)
    STRIPE_DARK = 3      # siempre la franja oscura (brillo)
    # Franja oscura en automático: el brillo marca la hilera como la franja
    # oscura (canopia + su sombra). Se usa si la canopia no es verde (verdor
    # cromático de la franja más verde < C_NOT_GREEN: viñedos sin hoja,
    # imagen de otoño, canopia rojiza; ahí la franja "más verde" es solo la
    # menos rojiza, a menudo el suelo), o si el patrón de brillo es mucho más
    # fuerte (BRIGHTNESS_SWITCH) y la franja oscura coincide con la más verde
    # (canopias casi negras, donde el verdor es ruido). Si el patrón de brillo
    # es más fuerte pero la canopia es claramente verde (>= C_GREEN_CANOPY) y
    # la franja oscura está corrida más de DARK_OFFSET_MAX T de la más verde,
    # la oscura es la SOMBRA al costado de la canopia y se usa la más verde.
    # En la zona gris (canopia apenas verde, entre C_NOT_GREEN y
    # C_GREEN_CANOPY) con la franja oscura corrida más de DARK_OFFSET_MAX T,
    # la más verde es el suelo o la cobertura entre hileras y se usa la
    # oscura, venga el índice del verdor o del brillo (antes dependía de si
    # el brillo era 3 veces más fuerte: en tres frutales jóvenes con esa
    # firma, verdor 0,02-0,03 y corrimiento 0,25-0,33, dos iban a la oscura
    # y el tercero caía en el medio de la entrehilera). Si las dos franjas
    # coinciden se sigue con el verdor (cambiarlo acortaba 10% las líneas en
    # dos frutales de CANTERA, verdor 0,05-0,07).
    # Medido (verdor de la franja más verde, promedio del lote, a la
    # resolución de trabajo): sin hoja o rojizos -0,08 a -0,004; frutales
    # jóvenes o ralos con hoja 0,03-0,07 (el suelo entre árboles diluye el
    # promedio); viñedos y frutales con hoja 0,04-0,44; frutal en seto con la
    # sombra al costado 0,27. Corrimiento (en pasos de T/12): viñedos 0-1
    # paso, el seto 2 pasos (0,17), frutal joven con sombra y pasto 3 pasos.
    # Vuelo de dron (0,05 m/px) sobre un viñedo con hojas: canopia angosta y
    # verde con su sombra negra al costado (verdor 0,086, corrimiento 0,25):
    # con el límite en 0,10 caía en la zona gris y las líneas iban a la
    # sombra (0,65 m de la vid); con 0,06 sigue con el verdor, y la sombra se
    # declara como tal para que la reubicación por brillo no lleve las hileras
    # a ella (sin eso, 53 de 108 terminaban en la sombra). Los frutales
    # jóvenes con cobertura (0,02-0,03) siguen en la zona gris. Ninguno de los
    # 32 lotes de prueba cambia.
    C_NOT_GREEN = 0.0
    C_GREEN_CANOPY = 0.06
    DARK_OFFSET_MAX = 0.15
    # Copa cerrada: separación mínima de frutal, verdor mínimo de la franja
    # a media separación (fracción del de la más verde) y cuánto más verde
    # en valor absoluto tiene que ser. Medido: frutales con suelo entre
    # hileras 9-18% y 0,1-0,4 veces; viñedos 13-20% y ~1,4 (pero < 3,5 m);
    # frutal de copa cerrada 59% y 1,4 veces.
    ORCHARD_MIN_M = 3.5
    SUNLIT_VEG_FRAC = 0.3
    SUNLIT_GREEN_RATIO = 1.1
    # Con árboles sueltos en marco (dos alineaciones de plantas), los
    # árboles están en la franja que "late" al período entre plantas; una
    # cobertura verde entre hileras (pasto) es pareja a lo largo de la
    # hilera. Si a media separación late menos que TREE_BEAT_FRAC de lo que
    # late la franja más verde, los árboles son la franja más verde y la
    # franja verde de al lado es pasto, no copas al sol. Visto: frutal de
    # 5 x 4 m con pasto, 0,20 en el brillo.
    TREE_BEAT_FRAC = 0.5
    # Hileras finas (hortalizas): el ángulo se afina sin los manchones grandes
    FINE_ROWS_M = 1.2
    # Hileras de menos de FINE_T_PX píxeles: verdor con suavizado FINE_BLUR
    FINE_T_PX = 6.0
    FINE_BLUR = 0.5

    # Debajo de esto, el pico de la FFT no sobresale lo suficiente del
    # fondo como para confiar en el rumbo/separación detectados.
    MIN_SPECTRAL_PEAK = 20.0
    # Aviso de revisión (campo 'revisar'): marco casi cuadrado (la
    # separación entre hileras menos de MARCO_SQUARE veces la de plantas:
    # la elección de la dirección de mayor separación tiene poco margen) y
    # lotes donde menos de COVER_MIN_FRAC de las hileras ubicadas tienen
    # línea (visto: frutal en seto con las líneas sobre la sombra, 27 de 61).
    MARCO_SQUARE = 1.3
    COVER_MIN_FRAC = 0.7
    # Cuadros mezclados: el complemento usa una sola separación por lote. Una
    # ventana de ~8 x 8 separaciones tiene OTRA separación si la más fuerte
    # dista al menos BLOCK_DEV de T, es al menos BLOCK_X veces más fuerte que
    # la de T y es un patrón claro (BLOCK_AMP_MIN de la amplitud típica a T).
    # Se avisa si son al menos BLOCK_MIN ventanas y BLOCK_FRAC del lote.
    # Medido: un frutal que juntaba tres cuadros (uno a 0,88 T), 20% de las
    # ventanas; en los frutales y viñedos de un solo cuadro, 0-5% (manchones,
    # plantas faltantes, cobertura cargada).
    BLOCK_DEV = 0.075
    BLOCK_X = 2.0
    BLOCK_AMP_MIN = 0.3
    BLOCK_MIN = 3
    BLOCK_FRAC = 0.10

    # Si el pico espectral del brillo supera en este factor al del verdor,
    # las hileras se ubican por brillo (franjas oscuras). En la finca de
    # prueba separa limpio los lotes con canopia poco verde (3,7-6,5) de
    # los normales (0,8-2,9).
    BRIGHTNESS_SWITCH = 3.0
    # Centro de la banda (ver BAND_MAX_T): mejora mínima del largo de hilera
    # presente para usarlo, según por qué se eligió la franja oscura. Canopia
    # nada verde (vid sin hojas): lo más oscuro suele ser la sombra, 25% (en la
    # tanda 2 los viñedos que ganan dan 33-124%; uno que daba 13% perdía un
    # poco). Brillo mucho más marcado que el verdor: la canopia puede ser verde
    # y oscura y el usuario marca lo más oscuro (frutal y viñedo de la tanda 1:
    # 16% y 0%, y el centro los empeoraba), 50%; el hortícola en camellones da
    # 138%. Canopia apenas verde (zona gris): no se usa (un frutal con 74%
    # empeoraba: ahí la copa, algo verde, es lo más oscuro).
    BAND_GAIN_NOT_GREEN = 0.25
    BAND_GAIN_BRIGHT = 0.5
    # Hileras sobre el borde del contorno, en fracciones de la separación T
    # y solo A LO ANCHO de las hileras (a lo largo, el contorno se respeta
    # estricto): una hilera cuyo eje queda hasta 0,4 T afuera del contorno es
    # la hilera de borde del lote (el contorno se dibujó sobre ella) y se
    # conserva; la siguiente hacia afuera estaría a más de 0,5 T y no entra.
    # Para medir su contraste se muestrea hasta 1 T afuera (su entrehilera
    # exterior).
    # Largo mínimo de un tramo y hueco máximo dentro de una hilera (m): los
    # valores con que se ajustaron las tandas de prueba (Google, 0,125 m/px).
    MIN_LENGTH_M_DEF = 6.25
    MAX_GAP_M_DEF = 7.5
    ROW_TOL = 0.4
    SAMPLE_TOL = 1.0

    def tr(self, string):
        return QCoreApplication.translate("DetectCropRowsAlgorithm", string)

    def _square_doubt(self, doubts, per_rows, per_plants, px):
        """Agrega a doubts el aviso de marco casi cuadrado (ver MARCO_SQUARE)."""
        if per_rows < self.MARCO_SQUARE * per_plants:
            doubts.append((
                "rumbo",
                f"marco casi cuadrado ({per_rows * px:.1f} x {per_plants * px:.1f} m): se tomó "
                "como hileras la dirección de mayor separación, con poca diferencia; si van en "
                "la otra dirección, indicá la 'Distancia entre hileras'",
            ))

    def createInstance(self):
        return DetectCropRowsAlgorithm()

    def name(self):
        return "detectar_hileras"

    def displayName(self):
        return self.tr("Detectar hileras de cultivo")

    def group(self):
        return self.tr("Detección de Hileras de Cultivo")

    def groupId(self):
        return "crop_row_detector"

    def shortHelpString(self):
        # Guía de uso: cada renglón es un párrafo en el panel de ayuda. El
        # detalle técnico está en el README; la explicación de cada parámetro,
        # en su ayuda emergente (initAlgorithm).
        return self.tr(
            "Dibuja una línea por hilera de cultivo (viñedo, frutal, hortalizas en línea) "
            "sobre una imagen RGB de dron o satelital. No hace falta entrenar nada.\n"
            "<b>Uso básico</b>\n"
            "1. <b>Imagen</b>: el ortomosaico, en un sistema de coordenadas en metros (UTM).\n"
            "2. <b>Lotes</b> (recomendado): un polígono por lote o cuartel. Cada uno se "
            "procesa por separado y las líneas no salen de él: dibujá la cabecera sobre la "
            "calle.\n"
            "3. <b>Distancia entre hileras</b>: dejala en -1. Indicala solo si el log muestra "
            "una separación equivocada.\n"
            "El log muestra, por lote, el rumbo, la separación y cuántas hileras encontró.\n"
            "<b>Qué más se puede obtener</b> (casillas, todas desactivadas por defecto)\n"
            "• <b>Cuarteles automáticos</b>: si no tenés los lotes dibujados, los busca en la "
            "imagen (zonas con patrón de hileras) y procesa cada uno con su propio rumbo y "
            "distancia.\n"
            "• <b>Fallas por hilera</b>: tramos sin planta. Solo con imágenes de dron; con "
            "imágenes satelitales se omite con un aviso.\n"
            "<b>Si algo sale mal</b>\n"
            "• Separación o rumbo equivocados en el log: indicá la distancia entre hileras.\n"
            "• Las líneas caen al costado de las plantas (sobre la sombra o la entrehilera): "
            "en Parámetros avanzados, 'Franja de la hilera'.\n"
            "• Faltan tramos de hileras débiles: bajá la 'Sensibilidad'; aparecen tramos "
            "sobre cosas que no son hileras: subila.\n"
            "• Un lote junta cuadros con otra separación o rumbo: dibujá un polígono por "
            "cuadro.\n"
            "<b>Campo 'revisar'</b>: cuando una decisión automática queda con poco margen, "
            "el log lo avisa en rojo y cada línea lleva el motivo: 'rumbo' (marco casi "
            "cuadrado o periodicidad débil), 'franja' (dudosa qué franja es la hilera), "
            "'cobertura' (menos del 70% de las hileras con línea) o 'cuadros' (parte del "
            "lote con otra separación). Coloreá la capa por ese campo y mirá esos lotes.\n"
            "<b>Parámetros avanzados</b>: pasá el mouse sobre cada uno para ver qué hace. En "
            "general no hace falta tocarlos.\n"
            "<b>Cómo funciona</b>: mide el rumbo y la separación de las hileras con la FFT "
            "del verdor, ubica cada hilera con el perfil perpendicular y la sigue a lo largo "
            "mientras sea más verde (o más oscura) que sus dos entrehileras. Las hileras "
            "terminan donde termina lo plantado, sin entrar en la calle de cabecera. El "
            "detalle está en el README del complemento."
        )

    def initAlgorithm(self, config=None):
        # Diálogo: a la vista solo lo que cualquiera necesita (imagen, lotes,
        # distancia y qué obtener); el resto en "Parámetros avanzados". Cada
        # parámetro tiene un título corto y su explicación en la ayuda
        # emergente (setHelp, se ve al pasar el mouse).
        def add(param, help_text=None, advanced=False):
            if advanced:
                param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
            if help_text:
                param.setHelp(self.tr(help_text))
            self.addParameter(param)

        Number = QgsProcessingParameterNumber

        # --- a la vista -----------------------------------------------------
        add(
            QgsProcessingParameterRasterLayer(self.INPUT, self.tr("Imagen (ortomosaico RGB)")),
            "Imagen con bandas roja, verde y azul, de dron o satelital, en un sistema de "
            "coordenadas en metros (UTM). Tiene que ser un archivo: una capa web (Google, XYZ) "
            "hay que guardarla antes con 'Convertir mapa a ráster'. Los píxeles sin datos "
            "(transparentes o nodata, como el área fuera de un recorte) se ignoran.",
        )
        add(
            QgsProcessingParameterFeatureSource(
                self.CONTOUR, self.tr("Lotes o cuarteles (polígonos, opcional)"),
                [QgsProcessing.TypeVectorPolygon], optional=True,
            ),
            "Un polígono por lote o cuartel (recomendado). Cada uno se procesa por separado, "
            "con su propio rumbo, separación y numeración (campo 'lote'), y las líneas no salen "
            "de él: dibujá la cabecera sobre la calle, no antes de la última planta. Una hilera "
            "dibujada sobre el borde del polígono se conserva entera. Se reproyecta solo. Si no "
            "das lotes, se usa toda el área con datos de la imagen (o los cuarteles automáticos, "
            "si activás esa opción).",
        )
        add(
            Number(self.ROW_SPACING, self.tr("Distancia entre hileras en m (-1 = automática)"),
                   Number.Double, defaultValue=-1, minValue=-1),
            "Dejala en -1: el complemento la mide sola. Indicala (con un valor aproximado "
            "alcanza) solo si el log muestra un rumbo o una separación equivocados, por "
            "ejemplo la mitad de la real cuando hay pasto verde entre hileras. La búsqueda se "
            "acota a entre 0,7 y 1,4 veces este valor.",
        )
        add(
            QgsProcessingParameterBoolean(
                self.AUTO_BLOCKS, self.tr("Sin lotes: detectar los cuarteles automáticamente"),
                defaultValue=False),
            "Solo actúa si no das lotes. Busca en la imagen las zonas con patrón de hileras "
            "(el monte o el pasto no cuentan, las calles separan cuarteles) y procesa cada "
            "cuartel por separado, con su propio rumbo y su propia distancia entre hileras. "
            "Busca hileras de 1 a 8 m. Los polígonos salen en la capa 'Cuarteles detectados'.",
        )
        add(
            QgsProcessingParameterBoolean(
                self.DETECT_FAILURES, self.tr("Detectar fallas por hilera (solo imágenes de dron)"),
                defaultValue=False),
            "Marca los tramos de cada hilera sin planta, en la capa 'Fallas por hilera'. Hace "
            "falta un píxel de menos de 0,2 m y un detalle real igual de fino: con imágenes "
            "satelitales (aunque digan 0,125 m por píxel, su detalle real es de 0,3-0,4 m) se "
            "omite con un aviso. Las hileras no cambian con esta opción.",
        )
        # --- parámetros avanzados (plegados) ----------------------------------
        add(QgsProcessingParameterBand(self.RED_BAND, self.tr("Banda roja"), 1, self.INPUT), advanced=True)
        add(QgsProcessingParameterBand(self.GREEN_BAND, self.tr("Banda verde"), 2, self.INPUT), advanced=True)
        add(QgsProcessingParameterBand(self.BLUE_BAND, self.tr("Banda azul"), 3, self.INPUT), advanced=True)
        add(
            QgsProcessingParameterField(
                self.SPACING_FIELD, self.tr("Campo de los lotes con su distancia entre hileras"),
                parentLayerParameterName=self.CONTOUR,
                type=QgsProcessingParameterField.Numeric, optional=True,
            ),
            "Campo numérico de la capa de lotes con la distancia entre hileras de cada lote, "
            "en metros. Donde es mayor que 0 reemplaza a la distancia general; vacío o 0, "
            "automática. Sirve para correr juntos cultivos de separación muy distinta "
            "(un frutal y una hortaliza).",
            advanced=True,
        )
        add(
            Number(self.EXTEND_MAX, self.tr("Extender hileras hasta el borde del lote (m)"),
                   Number.Double, defaultValue=0.0, minValue=0.0),
            "Prolonga cada hilera hasta el borde del polígono si le faltan como máximo estos "
            "metros. 0 = no extender: la hilera termina donde termina lo plantado. Usalo solo si "
            "el polígono es el del área plantada; con el del lote, las líneas atravesarían las "
            "calles de cabecera.",
            advanced=True,
        )
        add(
            Number(self.SENSITIVITY, self.tr("Sensibilidad (contraste mínimo, en ruidos)"),
                   Number.Double, defaultValue=3.0, minValue=0.5, maxValue=20.0),
            "Cuánto más verde (u oscura) que sus entrehileras tiene que ser la hilera, en "
            "múltiplos del ruido de la imagen. Bajala si faltan tramos de hileras débiles; "
            "subila si aparecen tramos sobre cosas que no son hileras.",
            advanced=True,
        )
        add(
            QgsProcessingParameterEnum(
                self.STRIPE, self.tr("Franja de la hilera"),
                options=[
                    self.tr("Automática (recomendada)"),
                    self.tr("La más verde (viñedo, frutal con suelo entre hileras)"),
                    self.tr("Copas iluminadas (frutal de copa cerrada)"),
                    self.tr("La más oscura (canopia sin hojas o rojiza)"),
                ],
                defaultValue=self.STRIPE_AUTO,
            ),
            "Qué franja de la imagen es la hilera. En automático: la más verde; la oscura si la "
            "canopia no es verde (sin hojas, otoño, rojiza); las copas iluminadas en un frutal "
            "de copa cerrada; y si la oscura es la sombra al costado de una canopia verde, la "
            "verde. Cambiala solo si las líneas caen al costado de las plantas (el log dice qué "
            "franja eligió y por qué).",
            advanced=True,
        )
        add(
            Number(self.MIN_LENGTH_M, self.tr("Largo mínimo de un tramo de hilera (m)"),
                   Number.Double, defaultValue=self.MIN_LENGTH_M_DEF, minValue=0.1),
            "Tramos de hilera más cortos que esto se descartan.",
            advanced=True,
        )
        add(
            Number(self.MAX_GAP_M, self.tr("Hueco máximo dentro de una hilera (m)"),
                   Number.Double, defaultValue=self.MAX_GAP_M_DEF, minValue=0.0),
            "Huecos (plantas faltantes) de hasta este largo se unen dentro de la misma "
            "hilera; los más largos la cortan en dos tramos. Nunca menos de 2,4 separaciones "
            "entre hileras: en un frutal a 5 m, faltan 3-4 árboles y la hilera sigue entera.",
            advanced=True,
        )
        add(
            Number(self.FAIL_THRESHOLD, self.tr("Fallas: umbral de vigor (0 = entrehilera, 1 = planta)"),
                   Number.Double, defaultValue=0.5, minValue=0.05, maxValue=0.95),
            "El verdor de la hilera se mide entre la entrehilera (0) y una planta típica del "
            "cuartel (1). Falla es un tramo continuo por debajo de este valor. Más alto marca "
            "más fallas.",
            advanced=True,
        )
        add(
            Number(self.FAIL_MIN_LEN, self.tr("Fallas: largo mínimo (m)"),
                   Number.Double, defaultValue=2.0, minValue=0.1),
            "Tramos bajos más cortos que esto no se marcan como falla.",
            advanced=True,
        )
        # --- salidas -----------------------------------------------------------
        add(QgsProcessingParameterVectorDestination(self.OUTPUT, self.tr("Hileras detectadas")))
        add(
            QgsProcessingParameterVectorDestination(
                self.OUT_BLOCKS, self.tr("Cuarteles detectados (si se activó la opción)"),
                QgsProcessing.TypeVectorPolygon, optional=True, createByDefault=True,
            )
        )
        add(
            QgsProcessingParameterVectorDestination(
                self.OUT_FAILURES, self.tr("Fallas por hilera (si se activó la opción)"),
                QgsProcessing.TypeVectorLine, optional=True, createByDefault=True,
            )
        )

    @staticmethod
    def _out_of_memory(err):
        """¿err es una falta de memoria (de numpy, de GDAL o de OpenCV)?"""
        return (isinstance(err, MemoryError) or "Insufficient memory" in str(err)
                or "Failed to allocate" in str(err))

    def processAlgorithm(self, parameters, context, feedback):
        try:
            return self._process(parameters, context, feedback)
        except Exception as err:
            if not self._out_of_memory(err):
                raise
            # la imagen entera se lee en memoria: con un ortomosaico muy grande
            # salía un error técnico en inglés
            raise QgsProcessingException(
                self.tr(
                    "No alcanzó la memoria para procesar la imagen (se lee completa). Recortala a la "
                    "zona de interés (Ráster > Extracción > Cortar ráster por extensión) o exportala "
                    "con un píxel más grande, y volvé a ejecutar."
                )
            ) from None

    def _process(self, parameters, context, feedback):
        try:
            import cv2
        except ImportError:
            raise QgsProcessingException(
                self.tr(
                    "Falta la librería opencv-python. Instalala en el "
                    "Python de QGIS (ver README.md del plugin) y reiniciá QGIS."
                )
            )
        try:
            from osgeo import gdal
        except ImportError:
            raise QgsProcessingException(
                self.tr("No se pudo importar GDAL desde el entorno de QGIS.")
            )

        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if raster_layer is None:
            raise QgsProcessingException(self.tr("Raster de entrada inválido."))

        red_idx = self.parameterAsInt(parameters, self.RED_BAND, context)
        green_idx = self.parameterAsInt(parameters, self.GREEN_BAND, context)
        blue_idx = self.parameterAsInt(parameters, self.BLUE_BAND, context)
        contour_source = self.parameterAsSource(parameters, self.CONTOUR, context)
        auto_blocks = self.parameterAsBool(parameters, self.AUTO_BLOCKS, context)
        extend_max_m = self.parameterAsDouble(parameters, self.EXTEND_MAX, context)
        row_spacing_m = self.parameterAsDouble(parameters, self.ROW_SPACING, context)
        sensitivity = self.parameterAsDouble(parameters, self.SENSITIVITY, context)
        min_length_m = self.parameterAsDouble(parameters, self.MIN_LENGTH_M, context)
        max_gap_m = self.parameterAsDouble(parameters, self.MAX_GAP_M, context)
        detect_failures = self.parameterAsBool(parameters, self.DETECT_FAILURES, context)
        fail_threshold = self.parameterAsDouble(parameters, self.FAIL_THRESHOLD, context)
        fail_min_len = self.parameterAsDouble(parameters, self.FAIL_MIN_LEN, context)
        spacing_fields = self.parameterAsFields(parameters, self.SPACING_FIELD, context) or []
        spacing_field = spacing_fields[0] if spacing_fields else None
        stripe_mode = self.parameterAsEnum(parameters, self.STRIPE, context)

        source_path = raster_layer.source()
        feedback.pushInfo(self.tr(f"Abriendo raster: {source_path}"))

        try:
            ds = gdal.Open(source_path)
        except RuntimeError:   # GDAL con excepciones activadas (otro complemento)
            ds = None
        if ds is None:
            if raster_layer.providerType() != "gdal":
                # una capa web (Google, XYZ, WMS...) no es un archivo que se pueda leer
                raise QgsProcessingException(
                    self.tr(
                        "La imagen elegida no es un archivo: es una capa web o de otro origen (Google, "
                        f"XYZ, WMS...; proveedor '{raster_layer.providerType()}'). El complemento necesita un "
                        "archivo ráster: guardá primero la zona con 'Convertir mapa a ráster' (caja de "
                        "herramientas de Procesos), con el proyecto en un sistema de coordenadas en "
                        "metros, y elegí ese archivo."
                    )
                )
            raise QgsProcessingException(self.tr("GDAL no pudo abrir el raster."))
        if raster_layer.crs().isGeographic():
            # En grados el tamaño de píxel no está en metros: el largo mínimo de
            # 6,25 m pasaba a ser de millones de píxeles y la corrida terminaba
            # sin ninguna línea y sin decir por qué. Además el píxel en grados no
            # es cuadrado en el terreno (a 33° de latitud, 16% más angosto).
            raise QgsProcessingException(
                self.tr(
                    "La imagen está en grados (sistema de coordenadas geográfico "
                    f"{raster_layer.crs().authid()}) y el complemento mide en metros. Reproyectala a "
                    "un sistema en metros (UTM o POSGAR; por ejemplo EPSG:32719 en Mendoza) con "
                    "Ráster > Proyecciones > Combar (reproyectar) y volvé a ejecutar con la imagen "
                    "reproyectada."
                )
            )
        if raster_layer.crs().projectionAcronym() == "merc":
            # Mercator (EPSG:3857 es el CRS de los mapas web y el que toma un
            # proyecto de QGIS que empieza con Google o XYZ): sus unidades se
            # llaman metros pero se estiran con la latitud, 19% en Mendoza. Las
            # líneas salen bien; las distancias, no.
            k_merc = None
            try:
                to_geo = QgsCoordinateTransform(
                    raster_layer.crs(), QgsCoordinateReferenceSystem("EPSG:4326"), context.transformContext())
                lat = to_geo.transform(raster_layer.extent().center()).y()
                if abs(lat) < 85.0:
                    k_merc = 1.0 / math.cos(math.radians(lat))
            except Exception:
                pass
            feedback.reportError(
                self.tr(
                    f"AVISO: la imagen está en proyección Mercator ({raster_layer.crs().authid()}, la de los "
                    "mapas web). Sus unidades no son metros reales"
                    + (f": acá 1 m del terreno mide {k_merc:.2f}" if k_merc else "")
                    + ". Las hileras salen igual, pero las distancias del log, las longitudes y los "
                    "parámetros en metros quedan en esa escala. Para medidas reales, reproyectá la "
                    "imagen a UTM o POSGAR (Ráster > Proyecciones > Combar (reproyectar))."
                ),
                fatalError=False,
            )

        band_count = ds.RasterCount
        if band_count < 3:
            raise QgsProcessingException(
                self.tr(
                    f"El raster de entrada tiene solo {band_count} banda(s). Este algoritmo "
                    "necesita al menos 3 bandas (Roja, Verde, Azul) de un ortomosaico RGB, como "
                    "los que salen de vuelos de drone. Si es una banda individual de un satélite "
                    "(por ejemplo, una sola banda de Sentinel-2), primero hay que armar un "
                    "compuesto RGB apilando las bandas correspondientes — y aun así, tené en "
                    "cuenta que este algoritmo está pensado para imágenes de resolución "
                    "centimétrica/decimétrica (drone): con la resolución típica de satélites "
                    "(varios metros por píxel) las hileras individuales no se van a poder "
                    "distinguir."
                )
            )
        for nombre, idx in (("Roja", red_idx), ("Verde", green_idx), ("Azul", blue_idx)):
            if idx < 1 or idx > band_count:
                raise QgsProcessingException(
                    self.tr(
                        f"La banda {nombre} pedida (índice {idx}) no existe: el raster solo "
                        f"tiene {band_count} banda(s). Revisá la selección de bandas."
                    )
                )

        geotransform = ds.GetGeoTransform()
        px_w = abs(geotransform[1])
        px_h = abs(geotransform[5])
        pixel_size_avg = (px_w + px_h) / 2.0
        # largo mínimo y hueco máximo en metros -> píxeles del raster (hasta la
        # 0.24 se daban en píxeles: 50 y 60 px eran 6,25 y 7,5 m en las imágenes
        # de Google a 0,125 m, 10 y 12 m a 0,2 m y 2,7 y 3,2 m en un dron a 0,054)
        min_line_length = min_length_m / pixel_size_avg
        merge_max_gap_param = max_gap_m / pixel_size_avg

        # Los lotes, antes de leer la imagen: una capa de lotes sin polígonos
        # válidos (vacía, o con "solo objetos seleccionados" y nada
        # seleccionado) se avisa ya, no después de procesar la imagen entera.
        lots = (
            self._lot_geometries(contour_source, raster_layer.crs(), context, spacing_field)
            if contour_source is not None else []
        )
        if contour_source is not None and not lots and not auto_blocks:
            raise QgsProcessingException(
                self.tr(
                    "La capa de lotes no tiene ningún polígono válido (¿está vacía, o se marcó "
                    "'Solo objetos seleccionados' sin nada seleccionado?). Elegí otra capa, o dejá "
                    "el campo vacío para usar toda el área con datos de la imagen."
                )
            )

        def read_band(idx):
            band = ds.GetRasterBand(idx)
            if band is None:
                raise QgsProcessingException(
                    self.tr(f"No se pudo leer la banda {idx} del raster (el raster tiene {band_count} banda(s)).")
                )
            arr = band.ReadAsArray()
            if arr is None:   # GDAL no pudo reservar la memoria de la banda
                raise MemoryError()
            return arr.astype(np.float32)

        feedback.pushInfo(self.tr("Leyendo bandas R, G, B..."))
        r = read_band(red_idx)
        g = read_band(green_idx)
        b = read_band(blue_idx)

        # Píxeles válidos según GDAL (banda alfa, nodata, o todo válido si no
        # hay ninguno de los dos). Sin esto, el área transparente de un raster
        # recortado (RGB = 0) entra en los cálculos como si fuera suelo, y el
        # borde del recorte aparece como un borde más.
        valid = np.ones(r.shape, dtype=bool)
        for idx in (red_idx, green_idx, blue_idx):
            valid &= ds.GetRasterBand(idx).GetMaskBand().ReadAsArray() > 0
        # NaN o infinito (el "sin datos" de algunos rasters de coma flotante):
        # sin datos, y a 0 para que el suavizado de las bandas no los propague
        # a los píxeles válidos de al lado (el verdor quedaba en NaN y no se
        # podía estimar nada).
        for band_arr in (r, g, b):
            bad = ~np.isfinite(band_arr)
            if bad.any():
                valid &= ~bad
                band_arr[bad] = 0.0
        del bad
        if not valid.any():
            raise QgsProcessingException(
                self.tr("El raster no tiene píxeles válidos (todo es nodata/transparente).")
            )
        n_invalid = int((~valid).sum())
        if n_invalid > 0:
            feedback.pushInfo(
                self.tr(
                    f"Píxeles sin datos (alfa/nodata) ignorados: "
                    f"{100.0 * n_invalid / valid.size:.1f}% del raster."
                )
            )

        feedback.setProgress(20)
        if feedback.isCanceled():
            return {}

        # Índice ExG (Excess Green): resalta vegetación frente a suelo/residuos.
        # La versión cromática (la definición original de Woebbecke et al.,
        # 1995) divide por R+G+B, así que mide QUÉ TAN VERDE es un píxel y no
        # cuánto brilla. Importa cuando la canopia es más oscura que la
        # entrehilera (imágenes satelitales o aéreas con sol alto, canopia
        # densa): sin normalizar, 2G-R-B escala con el brillo, y un suelo
        # claro apenas verdoso da más que una canopia verde oscura, con lo
        # que las "hileras" detectadas caen sobre la entrehilera.
        # Las bandas se suavizan levemente (sigma 1 px) antes de dividir: en
        # píxeles casi negros (canopia en sombra, 0-3 DN) la cromaticidad está
        # cuantizada y es puro ruido (p. ej. RGB 0,1,0 da ExG = 2). Promediar
        # con los vecinos antes del cociente estabiliza el índice sin mover
        # las hileras.
        exg = chroma_exg(cv2, r, g, b, valid, 1.0)

        # Sin variación de verdor (bandas mal elegidas, raster uniforme) no hay
        # hileras que buscar. Percentiles en vez de mín/máx: unos pocos píxeles
        # casi negros pueden dar valores extremos.
        exg_min, exg_max = (float(v) for v in np.percentile(exg[valid], [0.5, 99.5]))
        if exg_max - exg_min < 1e-6:
            raise QgsProcessingException(
                self.tr("El índice ExG no tiene variación (imagen uniforme). Revisá las bandas elegidas.")
            )
        feedback.setProgress(45)
        if feedback.isCanceled():
            return {}

        row_ids = None
        row_period_px = None
        line_lots = None      # lote de cada línea (con varios polígonos de contorno)
        line_periods = None   # separación entre hileras de cada línea, en px
        line_flags = None     # motivos de revisión de cada línea (método de perfil)
        auto_info = []        # cuarteles detectados de la imagen (ver _auto_lots)
        clipped_by_lot = False
        # Brillo como evidencia auxiliar de continuidad de la hilera
        brightness = (r + g + b) / 3.0
        brightness[~valid] = 0.0
        # verde absoluto (2G-R-B), para la franja de la hilera en frutales
        # de copa cerrada
        rgb = (r, g, b)
        if not lots and auto_blocks:
            lots, auto_info = self._auto_lots(
                exg, brightness, valid, pixel_size_avg, geotransform, feedback)
        if lots:
            # Cada polígono es un lote: su propia orientación, separación
            # y alineación de hileras (dos lotes uno detrás del otro no
            # tienen por qué tener las hileras alineadas), estimadas solo
            # con sus píxeles, y recortado a su polígono, así que las
            # calles entre lotes quedan afuera. También con un solo
            # polígono: si el raster tiene otros cultivos alrededor, no
            # entran en la estimación (antes sí: un frutal vecino fijaba el
            # rumbo de una hortaliza).
            if auto_info:
                msg = (f"{len(lots)} cuartel(es) detectado(s) de la imagen: cada uno "
                       "estima su propio rumbo y su propia separación entre hileras.")
            elif len(lots) >= 2:
                msg = f"Capa de contorno con {len(lots)} polígonos: se procesa cada uno como un lote independiente."
            else:
                msg = ("Contorno de 1 polígono: orientación y separación se estiman "
                       "solo con los píxeles de adentro.")
            feedback.pushInfo(self.tr(msg))
            if spacing_field:
                n_sp = sum(1 for lot in lots if lot[2])
                feedback.pushInfo(
                    self.tr(f"Distancia entre hileras por lote (campo '{spacing_field}'): {n_sp} de {len(lots)} lotes la indican.")
                )
            result = self._detect_rows_by_lot(
                cv2, gdal, lots, exg, valid, brightness, pixel_size_avg, geotransform,
                row_spacing_m, sensitivity, min_line_length, merge_max_gap_param,
                extend_max_m, feedback, rgb=rgb, stripe_mode=stripe_mode,
            )
            if result is None:
                return {}
            lines, row_ids, line_lots, line_periods, line_flags = result
            clipped_by_lot = True
            feedback.pushInfo(
                self.tr(f"Total: {len(lines)} líneas en {len(set(line_lots))} lote(s) con hileras.")
            )
        else:
            result = self._detect_rows_profile(
                cv2, exg, valid, pixel_size_avg, geotransform, row_spacing_m,
                sensitivity, min_line_length, merge_max_gap_param, feedback,
                aux=brightness,
                green_fn=(None if rgb is None else (lambda: 2.0 * rgb[1] - rgb[0] - rgb[2])),
                stripe_mode=stripe_mode,
                chroma_fn=(None if rgb is None else (lambda s: chroma_exg(cv2, rgb[0], rgb[1], rgb[2], valid, s))),
            )
            if result is None:
                return {}
            lines, row_ids, row_period_px = result
            line_flags = [", ".join(self._doubts)] * len(lines)
        del brightness

        # --- Recorte estricto al contorno de la parcela (y, en el método de
        # perfil, extensión de cada hilera hasta el borde del contorno) ---
        if feedback.isCanceled():
            return {}
        contour_geom = None
        if not clipped_by_lot:
            # sin lotes (ni cuarteles detectados): el área con datos de la imagen
            contour_geom = self._data_area_geometry(gdal, valid, geotransform, feedback)
        if contour_geom is not None and len(lines) > 0:
            ext_px = (
                extend_max_m / pixel_size_avg
                if extend_max_m is not None and extend_max_m > 0
                else 0.0
            )
            len_before = sum(math.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in lines) * pixel_size_avg
            n_before = len(lines)
            lines, row_ids = self._fit_lines_to_contour(
                gdal, lines, row_ids, contour_geom, geotransform, ext_px,
                valid.shape, min_line_length,
                cross_tol_px=self.ROW_TOL * row_period_px if row_period_px else 0.0,
                period_px=row_period_px,
            )
            len_after = sum(math.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in lines) * pixel_size_avg
            if ext_px > 0:
                msg = (
                    f"Ajuste al contorno: hileras extendidas hasta el borde si les "
                    f"faltaban <= {extend_max_m:g} m, y recortadas al contorno. "
                )
            else:
                msg = "Ajuste al contorno: líneas recortadas al contorno. "
            feedback.pushInfo(
                self.tr(
                    msg + f"{n_before} -> {len(lines)} líneas; longitud total "
                    f"{len_before:.0f} m -> {len_after:.0f} m."
                )
            )

        def pixel_to_map(col, row):
            # col/row son índices de píxel (centro del píxel = índice + 0.5
            # en el sistema del geotransform, que mide desde la esquina).
            col, row = col + 0.5, row + 0.5
            x = geotransform[0] + col * geotransform[1] + row * geotransform[2]
            y = geotransform[3] + col * geotransform[4] + row * geotransform[5]
            return x, y

        fields = QgsFields()
        fields.append(QgsField("id", QVariant.Int))
        fields.append(QgsField("longitud_m", QVariant.Double))
        fields.append(QgsField("angulo_deg", QVariant.Double))
        if line_lots is not None:
            fields.append(QgsField("lote", QVariant.Int))
        if row_ids is not None:
            fields.append(QgsField("hilera", QVariant.Int))
        if line_flags is not None and len(line_flags) != len(lines):
            # sin lotes (un solo raster) el motivo es el mismo para todas y el
            # recorte al contorno puede cambiar la cantidad de líneas
            line_flags = [line_flags[0]] * len(lines) if len(set(line_flags)) == 1 else None
        if line_flags is not None:
            # vacío = sin dudas; si no, motivos: rumbo, franja, cobertura
            fields.append(QgsField("revisar", QVariant.String))

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            fields,
            QgsWkbTypes.LineString,
            raster_layer.crs(),
        )
        if sink is None:
            raise QgsProcessingException(self.tr("No se pudo crear la capa de salida."))

        feedback.pushInfo(self.tr(f"Escribiendo {len(lines)} segmentos detectados..."))
        for i, (x1, y1, x2, y2) in enumerate(lines):
            mx1, my1 = pixel_to_map(x1, y1)
            mx2, my2 = pixel_to_map(x2, y2)

            length_m = math.hypot(mx2 - mx1, my2 - my1)
            angle_deg = math.degrees(math.atan2(my2 - my1, mx2 - mx1))

            attrs = [i, length_m, angle_deg]
            if line_lots is not None:
                attrs.append(line_lots[i])
            if row_ids is not None:
                attrs.append(row_ids[i])
            if line_flags is not None:
                attrs.append(line_flags[i])

            feat = QgsFeature(fields)
            feat.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(mx1, my1), QgsPointXY(mx2, my2)]))
            feat.setAttributes(attrs)
            sink.addFeature(feat)

        results = {self.OUTPUT: dest_id}
        if auto_info:
            bfields = QgsFields()
            for nm, tp in (("cuartel", QVariant.Int), ("area_ha", QVariant.Double),
                           ("rumbo_deg", QVariant.Double), ("separacion_m", QVariant.Double),
                           ("patron", QVariant.Double)):
                bfields.append(QgsField(nm, tp))
            bsink, bdest = self.parameterAsSink(
                parameters, self.OUT_BLOCKS, context, bfields,
                QgsWkbTypes.Polygon, raster_layer.crs())
            if bsink is not None:
                for lot_id, geom, info_b in auto_info:
                    ft = QgsFeature(bfields)
                    ft.setGeometry(geom)
                    ft.setAttributes([lot_id, round(geom.area() / 10000.0, 3),
                                      round(info_b["azimut"], 2), round(info_b["per_m"], 2),
                                      round(info_b["ratio"], 2)])
                    bsink.addFeature(ft)
                results[self.OUT_BLOCKS] = bdest

        if detect_failures and len(lines) > 0 and not feedback.isCanceled():
            # Las hileras ya están escritas: un problema en las fallas solo da
            # un aviso y no toca las líneas.
            try:
                self._failures(parameters, context, feedback, results, lines, line_lots, row_ids,
                               pixel_to_map, r, g, b, valid, geotransform, pixel_size_avg,
                               raster_layer.crs(), fail_threshold, fail_min_len)
            except Exception as e:
                feedback.reportError(
                    self.tr(f"No se pudieron calcular las fallas ({e}); las hileras no se ven afectadas."),
                    fatalError=False,
                )

        feedback.setProgress(100)
        ds = None  # cerrar dataset de GDAL

        return results

    def _failures(self, parameters, context, feedback, results, lines, line_lots, row_ids,
                  pixel_to_map, r, g, b, valid, gt, px, crs, umbral, min_len):
        """Fallas por hilera sobre las líneas detectadas (ver fallas.py), solo si
        la imagen tiene detalle real suficiente (dron)."""
        from .fallas import fine_detail, failures_by_lot
        feedback.pushInfo(self.tr("Fallas por hilera: midiendo el detalle real de la imagen..."))
        if gt[2] != 0 or gt[4] != 0 or gt[1] <= 0 or gt[5] >= 0:
            # el recorte por cuartel (fallas._tile) supone el norte arriba; con
            # un raster "sur arriba" no encontraba ninguna falla, sin avisar
            feedback.reportError(self.tr(
                "Fallas por hilera omitidas: el raster está rotado o no tiene el norte arriba."),
                fatalError=False)
            return
        # el detalle se mide donde están las hileras: en el centro de las más largas
        order = sorted(range(len(lines)), key=lambda i: -math.hypot(lines[i][2] - lines[i][0], lines[i][3] - lines[i][1]))
        pick = [order[int(k)] for k in np.unique(np.linspace(0, min(len(order), 40) - 1, 5).round())]
        centers = [((lines[i][0] + lines[i][2]) / 2.0, (lines[i][1] + lines[i][3]) / 2.0) for i in pick]
        ok, loss, why = fine_detail(r, g, b, valid, px, centers)
        if not ok:
            feedback.reportError(self.tr(
                f"Fallas por hilera omitidas: {why}. Las fallas solo se pueden ver con imágenes de dron; "
                "las hileras salen igual."), fatalError=False)
            return
        feedback.pushInfo(self.tr(
            f"Detalle real suficiente (al llevar la imagen a 0,2 m pierde el {100 * loss:.0f}% del detalle): "
            "se buscan fallas."))
        segs = []
        for x1, y1, x2, y2 in lines:
            (ax, ay), (bx, by) = pixel_to_map(x1, y1), pixel_to_map(x2, y2)
            segs.append((ax, ay, bx, by))
        fallas = failures_by_lot(
            segs, line_lots, row_ids, r, g, b, valid, gt, umbral=umbral, min_falla=min_len,
            log=lambda m: feedback.pushInfo(self.tr(m)),
            warn=lambda m: feedback.reportError(self.tr(m), fatalError=False),
            canceled=feedback.isCanceled)
        ffields = QgsFields()
        for nm, tp in (("lote", QVariant.Int), ("hilera", QVariant.Int), ("ini_m", QVariant.Double),
                       ("largo_m", QVariant.Double), ("vigor", QVariant.Double), ("borde", QVariant.Int)):
            ffields.append(QgsField(nm, tp))
        fsink, fdest = self.parameterAsSink(
            parameters, self.OUT_FAILURES, context, ffields, QgsWkbTypes.LineString, crs)
        if fsink is None:
            return
        for fa in fallas:
            ft = QgsFeature(ffields)
            ft.setGeometry(QgsGeometry.fromPolylineXY(
                [QgsPointXY(fa["x1"], fa["y1"]), QgsPointXY(fa["x2"], fa["y2"])]))
            ft.setAttributes([fa["lote"], fa["hilera"], round(fa["ini_m"], 2), round(fa["largo_m"], 2),
                              round(fa["vigor"], 3), int(fa["borde"])])
            fsink.addFeature(ft)
        results[self.OUT_FAILURES] = fdest
        n_b = sum(1 for fa in fallas if fa["borde"])
        feedback.pushInfo(self.tr(
            f"Fallas por hilera: {len(fallas)} ({sum(fa['largo_m'] for fa in fallas):.0f} m); {n_b} de borde "
            "(a menos de 2 m de la punta de la hilera: dependen de dónde termina la línea)."))

    def _auto_lots(self, exg, brightness, valid, pixel_size_avg, geotransform, feedback):
        """Cuarteles detectados de la imagen: zonas con patrón de hileras (ver
        bloques.py). Devuelve (lots, info): lots en el formato de
        _lot_geometries (número, geometría, None: cada cuartel estima su propia
        separación), numerados de norte a sur; info, (número, geometría,
        datos del patrón) para la capa de cuarteles. Con [] se sigue con toda
        el área con datos."""
        feedback.pushInfo(self.tr("Buscando cuarteles: zonas con patrón de hileras..."))
        gt = geotransform
        if gt[2] != 0 or gt[4] != 0:
            feedback.pushInfo(self.tr("Raster rotado: no se detectan cuarteles, se usa toda el área con datos."))
            return [], []
        det_info = {}
        try:
            blocks = detect_blocks(exg, brightness, valid, pixel_size_avg, info=det_info)
        except Exception as err:
            if self._out_of_memory(err):
                raise
            # p. ej. una imagen de pocos metros de lado: sin cuarteles, se sigue
            # con toda el área con datos en vez de cortar la corrida
            feedback.reportError(
                self.tr(
                    f"No se pudieron buscar los cuarteles ({type(err).__name__}: {err}): se usa toda "
                    "el área con datos."
                ),
                fatalError=False,
            )
            feedback.pushInfo(traceback.format_exc())
            return [], []
        if not blocks:
            feedback.pushInfo(self.tr(
                "No se encontró ninguna zona con patrón de hileras (hileras de 1 a 8 m): "
                "se usa toda el área con datos."))
            return [], []
        out = []
        for b in blocks:
            geom = QgsGeometry.fromPolygonXY([
                [QgsPointXY(gt[0] + x * gt[1], gt[3] + y * gt[5]) for x, y in ring]
                for ring in b["rings"][:1]
            ])
            geom = self._valid_polygon(geom)
            if geom.isEmpty():
                continue
            th = math.radians(b["ang"])
            azimut = math.degrees(math.atan2(gt[1] * math.cos(th), gt[5] * math.sin(th))) % 180.0
            out.append((geom, {"azimut": azimut, "per_m": b["per_m"], "ratio": b["ratio"]}))
        # de norte a sur, y de oeste a este entre los de la misma altura
        out.sort(key=lambda t: (-round(t[0].centroid().asPoint().y() / 50.0), t[0].centroid().asPoint().x()))
        lots = [(i + 1, g, None) for i, (g, _) in enumerate(out)]
        info = [(i + 1, g, d) for i, (g, d) in enumerate(out)]
        for n, g, d in info:
            feedback.pushInfo(self.tr(
                f"  Cuartel {n}: {g.area() / 10000.0:.2f} ha, rumbo {d['azimut']:.1f}°, hileras cada "
                f"~{d['per_m']:.2f} m (el patrón explica {100 * d['ratio']:.0f}% del contraste local)."))
        return lots, info

    def _data_area_geometry(self, gdal, valid, geotransform, feedback):
        """Área con datos del raster (máscara alfa/nodata vectorizada), en su
        CRS: el contorno cuando no hay lotes. None si no hay píxeles sin datos
        (nada que recortar). Con una capa de lotes, cada polígono se procesa y
        se recorta por su cuenta (ver _detect_rows_by_lot) y esto no se usa."""
        if valid.all():
            return None

        from osgeo import ogr
        H, W = valid.shape
        mem = gdal.GetDriverByName("MEM").Create("", W, H, 1, gdal.GDT_Byte)
        mem.SetGeoTransform(geotransform)
        band = mem.GetRasterBand(1)
        band.WriteArray(valid.astype(np.uint8))
        vds = ogr.GetDriverByName("Memory").CreateDataSource("contorno")
        layer = vds.CreateLayer("contorno", geom_type=ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
        gdal.Polygonize(band, band, layer, 0)  # la banda como máscara: solo los píxeles válidos
        geoms = []
        for feat in layer:
            if feat.GetField("v") == 1:
                geoms.append(QgsGeometry.fromWkt(feat.GetGeometryRef().ExportToWkt()))
        layer, vds, band, mem = None, None, None, None
        if not geoms:
            return None
        geom = QgsGeometry.unaryUnion(geoms)
        feedback.pushInfo(
            self.tr(
                f"Contorno de la parcela: sin capa de contorno, se usa el área con "
                f"datos del raster ({geom.area() / 10000.0:.2f} ha)."
            )
        )
        return geom

    def _inv_geotransform(self, gdal, gt):
        inv = gdal.InvGeoTransform(gt)
        if inv is not None and len(inv) == 2:  # bindings viejos de GDAL: (ok, gt_inv)
            inv = inv[1] if inv[0] else None
        if inv is None:
            raise QgsProcessingException(self.tr("Geotransform del raster no invertible."))
        return inv

    def _lot_geometries(self, source, raster_crs, context, spacing_field=None):
        """Polígonos de la capa de contorno, cada uno en el CRS del raster,
        numerados 1..N en el orden de la capa, con la distancia entre
        hileras del lote en metros si spacing_field la da (> 0; si no,
        None). Un objeto multiparte cuenta como un solo lote.
        Devuelve [(número, geometría, distancia o None), ...]."""
        request = QgsFeatureRequest().setDestinationCrs(raster_crs, context.transformContext())
        out = []
        for f in source.getFeatures(request):
            if not f.hasGeometry():
                continue
            geom = self._valid_polygon(f.geometry())
            if geom.isEmpty() or geom.area() <= 0:
                continue
            spacing = None
            if spacing_field:
                try:
                    v = f[spacing_field]
                    spacing = float(v) if v is not None and float(v) > 0 else None
                except (KeyError, TypeError, ValueError):
                    spacing = None
            out.append((geom, spacing))
        return [(i + 1, geom, spacing) for i, (geom, spacing) in enumerate(out)]

    @staticmethod
    def _valid_polygon(geom):
        """geom reparada (makeValid) y reducida a sus partes poligonales. Al
        reparar un polígono dibujado con una "espina" (un vértice que va y
        vuelve por la misma línea, un desliz común al digitalizar), makeValid
        devuelve una colección con el polígono y una línea suelta, que no se
        puede rasterizar: cortaba la corrida entera con un error técnico."""
        if not geom.isGeosValid():
            geom = geom.makeValid()
        if geom.type() != QgsWkbTypes.PolygonGeometry:
            parts = [p for p in geom.asGeometryCollection() if p.type() == QgsWkbTypes.PolygonGeometry]
            geom = QgsGeometry.unaryUnion(parts) if parts else QgsGeometry()
        return geom

    @staticmethod
    def _sweep_across_rows(geom, gt, angle_deg, tol_px, steps=4):
        """El polígono barrido ±tol_px (píxeles) en la dirección perpendicular
        a las hileras (angle_deg: rumbo de las hileras en coordenadas de
        imagen). Los bordes paralelos a las hileras se corren tol_px hacia
        afuera; los perpendiculares (cabeceras) quedan donde están. Unión de
        copias trasladadas (error <= tol_px / steps en las esquinas)."""
        th = math.radians(angle_deg)
        nx, ny = -math.sin(th), math.cos(th)
        dx = (gt[1] * nx + gt[2] * ny) * tol_px
        dy = (gt[4] * nx + gt[5] * ny) * tol_px
        parts = [geom]
        for k in range(-steps, steps + 1):
            if k == 0:
                continue
            g = QgsGeometry(geom)
            g.translate(dx * k / steps, dy * k / steps)
            parts.append(g)
        return DetectCropRowsAlgorithm._valid_polygon(QgsGeometry.unaryUnion(parts))

    @staticmethod
    def _polygon_mask(cv2, geom, inv, r0, c0, h, w):
        """Máscara booleana (h x w, con origen en la fila r0 / columna c0 del
        raster) de los píxeles cuyo centro cae dentro del polígono."""
        mask = np.zeros((h, w), dtype=np.uint8)
        polys = geom.asMultiPolygon() if geom.isMultipart() else [geom.asPolygon()]
        for poly in polys:
            for k, ring in enumerate(poly):  # anillo 0 = exterior, el resto = huecos
                pts = np.array([
                    [inv[0] + p.x() * inv[1] + p.y() * inv[2] - 0.5 - c0,
                     inv[3] + p.x() * inv[4] + p.y() * inv[5] - 0.5 - r0]
                    for p in ring
                ])
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1 if k == 0 else 0)
        return mask > 0

    def _detect_rows_by_lot(self, cv2, gdal, lots, exg, valid, aux, pixel_size_avg,
                            geotransform, row_spacing_m, sensitivity, min_line_length,
                            max_gap_param, extend_max_m, feedback, rgb=None, stripe_mode=0):
        """Método de perfil lote por lote: cada polígono se procesa solo con
        sus propios píxeles (su propia orientación, separación, alineación de
        hileras y numeración) y sus líneas se recortan a él. lots: salida de
        _lot_geometries (la distancia entre hileras de un lote, si la tiene,
        reemplaza a row_spacing_m). rgb: (r, g, b) para el verde absoluto
        (ver _detect_rows_profile). Devuelve (líneas, hilera de cada línea,
        lote de cada línea, separación en px de cada línea, motivos de
        revisión de cada línea), o None si el usuario canceló."""
        gt = geotransform
        inv = self._inv_geotransform(gdal, gt)
        H, W = valid.shape
        ext_px = extend_max_m / pixel_size_avg if extend_max_m is not None and extend_max_m > 0 else 0.0
        lines, ids, lots_out, periods, flags = [], [], [], [], []
        to_review = []
        for n, (lot_id, geom, lot_spacing) in enumerate(lots):
            if feedback.isCanceled():
                return None
            head = f"--- Lote {lot_id} ({geom.area() / 10000.0:.2f} ha)"
            if lot_spacing:
                head += f", distancia entre hileras indicada {lot_spacing:g} m"
            feedback.pushInfo(self.tr(head + " ---"))
            bb = geom.boundingBox()
            corners = [
                (inv[0] + x * inv[1] + y * inv[2], inv[3] + x * inv[4] + y * inv[5])
                for x, y in ((bb.xMinimum(), bb.yMinimum()), (bb.xMinimum(), bb.yMaximum()),
                             (bb.xMaximum(), bb.yMinimum()), (bb.xMaximum(), bb.yMaximum()))
            ]
            cmin_, cmax_ = min(c for c, _ in corners), max(c for c, _ in corners)
            rmin_, rmax_ = min(r for _, r in corners), max(r for _, r in corners)
            if min(cmax_, W) - max(cmin_, 0) < 32 or min(rmax_, H) - max(rmin_, 0) < 32:
                feedback.reportError(
                    self.tr(f"Lote {lot_id}: fuera del raster o demasiado chico, se omite."),
                    fatalError=False,
                )
                continue
            # margen alrededor del lote: la hilera de borde se mide contra su
            # entrehilera exterior, que queda fuera del polígono
            m = int(min(600, math.ceil(15.0 / max(pixel_size_avg, 1e-9))))
            c0 = max(0, int(math.floor(cmin_)) - 2 - m)
            c1 = min(W, int(math.ceil(cmax_)) + 3 + m)
            r0 = max(0, int(math.floor(rmin_)) - 2 - m)
            r1 = min(H, int(math.ceil(rmax_)) + 3 + m)
            hh, ww = r1 - r0, c1 - c0
            valid_crop = valid[r0:r1, c0:c1]
            mask = self._polygon_mask(cv2, geom, inv, r0, c0, hh, ww) & valid_crop
            if mask.sum() < 1024:
                feedback.reportError(
                    self.tr(f"Lote {lot_id}: casi sin píxeles con datos, se omite."),
                    fatalError=False,
                )
                continue
            gt_crop = (gt[0] + c0 * gt[1] + r0 * gt[2], gt[1], gt[2],
                       gt[3] + c0 * gt[4] + r0 * gt[5], gt[4], gt[5])

            def cross_masks(angle_deg, period_px, geom=geom, r0=r0, c0=c0, hh=hh, ww=ww, vc=valid_crop):
                out = []
                for tol in (self.SAMPLE_TOL, self.ROW_TOL):
                    g = self._sweep_across_rows(geom, gt, angle_deg, tol * period_px)
                    out.append(self._polygon_mask(cv2, g, inv, r0, c0, hh, ww) & vc)
                return out

            try:
                res = self._detect_rows_profile(
                    cv2, exg[r0:r1, c0:c1], mask, pixel_size_avg, gt_crop,
                    lot_spacing if lot_spacing else row_spacing_m,
                    sensitivity, min_line_length, max_gap_param, feedback,
                    aux=None if aux is None else aux[r0:r1, c0:c1],
                    cross_masks=cross_masks,
                    green_fn=(None if rgb is None else
                              (lambda r0=r0, r1=r1, c0=c0, c1=c1:
                               2.0 * rgb[1][r0:r1, c0:c1] - rgb[0][r0:r1, c0:c1] - rgb[2][r0:r1, c0:c1])),
                    stripe_mode=stripe_mode,
                    chroma_fn=(None if rgb is None else
                               (lambda s, r0=r0, r1=r1, c0=c0, c1=c1, vc=valid_crop:
                                chroma_exg(cv2, rgb[0][r0:r1, c0:c1], rgb[1][r0:r1, c0:c1],
                                           rgb[2][r0:r1, c0:c1], vc, s))),
                )
                if res is None:
                    return None
                l_lines, l_ids, period = res
                l_lines = [(x1 + c0, y1 + r0, x2 + c0, y2 + r0) for x1, y1, x2, y2 in l_lines]
                if l_lines:
                    l_lines, l_ids = self._fit_lines_to_contour(
                        gdal, l_lines, l_ids, geom, gt, ext_px, valid.shape, min_line_length,
                        cross_tol_px=self.ROW_TOL * period, period_px=period,
                    )
            except QgsProcessingException as err:
                feedback.reportError(self.tr(f"Lote {lot_id}: {err}"), fatalError=False)
                continue
            except Exception as err:
                # Un error inesperado en un lote no tiene que perder los demás
                # (antes cortaba la corrida entera, sin ninguna línea): se avisa,
                # se deja el detalle en el log y se sigue.
                if self._out_of_memory(err):
                    feedback.reportError(
                        self.tr(
                            f"Lote {lot_id}: no alcanzó la memoria para procesarlo. Se omite y se sigue "
                            "con los demás; probá con un lote más chico o una imagen de menor resolución."
                        ),
                        fatalError=False,
                    )
                    continue
                feedback.reportError(
                    self.tr(
                        f"Lote {lot_id}: error inesperado ({type(err).__name__}: {err}). Se omite este "
                        "lote y se sigue con los demás; el detalle técnico queda en el log."
                    ),
                    fatalError=False,
                )
                feedback.pushInfo(traceback.format_exc())
                continue
            lines += l_lines
            ids += l_ids
            lots_out += [lot_id] * len(l_lines)
            periods += [period] * len(l_lines)
            flag = ", ".join(self._doubts)
            flags += [flag] * len(l_lines)
            if flag:
                to_review.append(f"{lot_id} ({flag})")
            feedback.setProgress(45 + int(25 * (n + 1) / len(lots)))
        if to_review:
            feedback.reportError(
                self.tr(
                    f"Lotes a revisar sobre la imagen ({len(to_review)} de {len(lots)}): "
                    + "; ".join(to_review) + ". Sus líneas tienen el motivo en el campo 'revisar'."
                ),
                fatalError=False,
            )
        return lines, ids, lots_out, periods, flags

    def _fit_lines_to_contour(self, gdal, lines, row_ids, contour_geom, geotransform,
                              ext_px, shape, min_len_px, cross_tol_px=0.0, period_px=None):
        """Recorta las líneas (píxeles) al contorno y, con ext_px > 0,
        extiende los extremos de cada hilera hasta el borde (ver
        fit_intervals_to_contour). Los tramos de una misma hilera (row_ids)
        se tratan juntos; sin row_ids, cada línea por separado. Con
        cross_tol_px > 0, una hilera de
        borde (dibujada sobre el borde del contorno, o apenas afuera, y por
        eso cortada por él) se recorta con la paralela a cross_tol_px hacia
        adentro del lote, y se conserva entera; las demás se recortan contra
        el contorno tal cual (estricto a lo largo de la hilera, también donde
        la cabecera está oblicua). Una hilera es de borde si el contorno le
        deja menos de la mitad del largo que le deja a esa paralela interior,
        o si la corta en varios pedazos y le deja menos del 90%. Con
        period_px, una hilera de borde tiene que estar en el marco de
        plantación: a un múltiplo de la separación (±T/4) de la hilera
        interior más cercana; si no, es otra cosa a lo largo del borde (un
        cerco, una cortina, una fila de árboles de la calle) y se descarta.
        Devuelve (líneas, row_ids)."""
        gt = geotransform
        inv = self._inv_geotransform(gdal, gt)
        tol = cross_tol_px if (cross_tol_px > 0 and row_ids is not None) else 0.0

        def to_map(x, y):  # índices de píxel -> mapa (centro del píxel)
            x, y = x + 0.5, y + 0.5
            return gt[0] + x * gt[1] + y * gt[2], gt[3] + x * gt[4] + y * gt[5]

        def to_px(X, Y):
            return inv[0] + X * inv[1] + Y * inv[2] - 0.5, inv[3] + X * inv[4] + Y * inv[5] - 0.5

        groups = {}
        for i, seg in enumerate(lines):
            key = row_ids[i] if row_ids is not None else i
            groups.setdefault(key, []).append(seg)

        big = 2.0 * (shape[0] + shape[1])
        out_by_key = {}
        edge_keys = set()
        offsets = {}
        for key, segs in groups.items():
            x1, y1, x2, y2 = segs[0]
            L = math.hypot(x2 - x1, y2 - y1)
            if L < 1e-9:
                continue
            u = ((x2 - x1) / L, (y2 - y1) / L)

            def param(x, y):
                return (x - x1) * u[0] + (y - y1) * u[1]

            if not offsets:
                n_ref = (-u[1], u[0])  # normal común (las hileras son paralelas)
            offsets[key] = x1 * n_ref[0] + y1 * n_ref[1]

            intervals = sorted(
                (min(param(a, b), param(c, d)), max(param(a, b), param(c, d)))
                for a, b, c, d in segs
            )

            def contour_pieces(offset):
                """Tramos (en el parámetro de la hilera) de la paralela a la
                hilera corrida `offset` px a lo ancho, dentro del contorno."""
                ox, oy = -u[1] * offset, u[0] * offset
                A = to_map(x1 - big * u[0] + ox, y1 - big * u[1] + oy)
                B = to_map(x1 + big * u[0] + ox, y1 + big * u[1] + oy)
                full_line = QgsGeometry.fromPolylineXY([QgsPointXY(*A), QgsPointXY(*B)])
                pieces = []
                for part in contour_geom.intersection(full_line).asGeometryCollection():
                    if part.type() != QgsWkbTypes.LineGeometry:
                        continue
                    polylines = part.asMultiPolyline() if part.isMultipart() else [part.asPolyline()]
                    for pl in polylines:
                        if len(pl) >= 2:
                            # la proyección sobre la hilera no cambia con el corrimiento
                            ts = [param(*to_px(p.x(), p.y())) for p in pl]
                            pieces.append((min(ts), max(ts)))
                # GEOS puede devolver en varios pedazos contiguos un mismo
                # tramo (p. ej. al tocar vértices del contorno): unirlos
                pieces.sort()
                merged_ = []
                for c, d in pieces:
                    if merged_ and c <= merged_[-1][1] + 1e-6:
                        merged_[-1] = (merged_[-1][0], max(merged_[-1][1], d))
                    else:
                        merged_.append((c, d))
                return merged_

            merged = contour_pieces(0.0)
            if tol > 0:
                # Hilera de borde: queda mayormente afuera del contorno, o el
                # contorno la corta en varios pedazos (un borde dibujado sobre
                # ella, que la cruza de un lado al otro). Se toma la paralela
                # interior (la de más largo). Una hilera de esquina, que el
                # borde oblicuo corta una sola vez, se recorta estricto.
                inner = max((contour_pieces(s_ * tol) for s_ in (-1.0, 1.0)),
                            key=lambda pcs: sum(d - c for c, d in pcs))
                l_in, l_str = sum(d - c for c, d in inner), sum(d - c for c, d in merged)
                if l_str < 0.5 * l_in or (len(merged) >= 2 and l_str < 0.9 * l_in):
                    merged = inner
                    edge_keys.add(key)

            out_by_key[key] = [
                (x1 + a * u[0], y1 + a * u[1], x1 + b * u[0], y1 + b * u[1])
                for a, b in fit_intervals_to_contour(intervals, merged, ext_px)
                if b - a >= min_len_px
            ]

        if period_px and edge_keys:
            inner_offs = [offsets[k] for k in groups if k not in edge_keys and out_by_key.get(k)]
            for k in edge_keys:
                if not inner_offs:
                    break
                q = min(abs(offsets[k] - o) for o in inner_offs) / period_px
                if round(q) < 1 or abs(q - round(q)) > 0.25:
                    out_by_key[k] = []
        new_lines, new_ids = [], []
        for key in groups:
            for seg in out_by_key.get(key, []):
                new_lines.append(seg)
                new_ids.append(key)
        return new_lines, (new_ids if row_ids is not None else None)

    def _detect_rows_profile(self, cv2, exg, valid, pixel_size_avg, geotransform,
                             row_spacing_m, sensitivity, min_line_length,
                             max_gap_param, feedback, aux=None, cross_masks=None,
                             green_fn=None, stripe_mode=0, chroma_fn=None):
        """Método de perfil de proyección. Devuelve (líneas en píxeles del
        raster completo, número de hilera de cada línea, separación entre
        hileras en píxeles), o None si el usuario canceló.

        cross_masks (opcional): función (ángulo, período en px) -> (máscara
        de muestreo, máscara de centros de hilera), para un lote: el lote
        ensanchado a lo ancho de las hileras (ver _detect_rows_by_lot). La
        orientación y la separación se estiman igual solo con `valid`.

        green_fn (opcional): función sin argumentos que devuelve el verde
        absoluto 2G-R-B (misma forma que exg), para decidir qué franja es la
        hilera en frutales de copa cerrada (ver stripe_mode).

        chroma_fn (opcional): función (sigma de suavizado en px) -> verdor
        cromático (misma forma que exg), para recalcularlo con menos
        suavizado cuando las hileras son finas."""
        hint_px = row_spacing_m / pixel_size_avg if row_spacing_m is not None and row_spacing_m > 0 else None
        # Decisiones automáticas tomadas con poco margen: (motivo, texto).
        # Al final se avisan juntas y cada línea las lleva en el campo
        # 'revisar' (ver self._doubts).
        doubts = []
        self._doubts = []
        feedback.pushInfo(self.tr("Estimando orientación y separación de hileras (FFT 2D del ExG)..."))
        fft_info = {}
        angle, period, peak = estimate_row_orientation_fft(exg, valid, hint_px, info=fft_info)
        if angle is None:
            raise QgsProcessingException(
                self.tr(
                    "No se pudo estimar la orientación de las hileras: la zona es "
                    "demasiado chica, casi no tiene píxeles con datos o es uniforme "
                    "(sin nada que distinguir, como el fondo de un recorte)."
                )
            )
        mc = fft_info.get("marco")
        if mc and mc["por"]:
            th_ = math.radians(mc["alt_angle"])
            az_alt = math.degrees(math.atan2(
                geotransform[1] * math.cos(th_) + geotransform[2] * math.sin(th_),
                geotransform[4] * math.cos(th_) + geotransform[5] * math.sin(th_))) % 180.0
            if mc["por"] == "marco":
                why = ("se tomó el de mayor separación como las hileras (la distancia "
                       "entre hileras es mayor que la distancia entre plantas)")
            else:
                why = "se tomó el más cercano a la distancia entre hileras indicada"
            feedback.pushInfo(
                self.tr(
                    f"Marco de plantación: la imagen muestra dos alineaciones de plantas, "
                    f"una cada {period * pixel_size_avg:.2f} m y otra cada "
                    f"{mc['alt_period'] * pixel_size_avg:.2f} m a "
                    f"{az_alt:.1f}° ({100 * mc['frac']:.0f}% "
                    f"de la fuerza del patrón elegido: {why}). Si las hileras van en la otra "
                    f"dirección, indicá su distancia en 'Distancia entre hileras'."
                )
            )
            if mc["por"] == "marco":
                self._square_doubt(doubts, period, mc["alt_period"], pixel_size_avg)
        if fft_info.get("blanqueado"):
            feedback.pushInfo(
                self.tr(
                    "El máximo del espectro era un manchón grande, no un patrón de "
                    "hileras (no sobresalía de su frecuencia): la separación se tomó "
                    "del espectro blanqueado (típico de hileras finas, < 1 m)."
                )
            )

        # Índice para ubicar las hileras: el verdor, salvo que el patrón de
        # brillo sea mucho más fuerte (y con el mismo rumbo y separación).
        # Pasa cuando la canopia casi no es verde en la imagen (sin hojas,
        # en sombra profunda, imagen de otra fecha): el verdor de píxeles
        # casi negros es ruido de cuantización y los centros de hilera
        # caían en cualquier lado. En ese caso la hilera es la franja
        # OSCURA (canopia + su sombra, de menor reflectancia que el suelo).
        # (Después de afinar el ángulo se revisa si la franja oscura es la
        # hilera o la sombra al costado de ella: ver C_NOT_GREEN.)
        def azimuth_of(a_):
            th_ = math.radians(a_)
            return math.degrees(math.atan2(
                geotransform[1] * math.cos(th_) + geotransform[2] * math.sin(th_),
                geotransform[4] * math.cos(th_) + geotransform[5] * math.sin(th_))) % 180.0

        aux_is_brightness = aux is not None
        if aux is not None:
            b_info = {}
            b_angle, b_period, b_peak = estimate_row_orientation_fft(aux, valid, hint_px, info=b_info)
            same = (b_angle is not None and abs(b_period - period) <= 0.1 * period
                    and _angle_diff_mod180(b_angle, angle) <= 2.0)
            # Marco de plantación entre índices: el verdor y el brillo ven
            # alineaciones de plantas distintas. En un olivar de 12,25 x 7,8 m
            # el verdor mostraba la alineación cruzada de los árboles (7,4 m,
            # sus hileras apenas 12% de esa fuerza) y el brillo las hileras
            # (árboles + sombra contra el suelo claro entre hileras). Como en
            # el marco con un solo índice, las hileras son la de mayor
            # separación. También si el verdor ya eligió entre dos de sus
            # alineaciones: con pasto en el centro de la entrehilera el verdor
            # ve franjas cada media separación, y entre esas y la distancia
            # entre plantas tomaba las plantas.
            cross = (not same and b_angle is not None and hint_px is None
                     and not fft_info.get("blanqueado") and not b_info.get("blanqueado")
                     and b_info.get("sobre_anillo", 0.0) >= WHITE_MIN
                     and MARCO_RATIO[0] <= b_period / period <= MARCO_RATIO[1]
                     and _angle_diff_mod180(b_angle, angle) >= MARCO_MIN_ANG)
            if cross:
                doubts[:] = [d_ for d_ in doubts if d_[0] != "rumbo"]
                feedback.pushInfo(
                    self.tr(
                        f"Marco de plantación: en el verdor sobresale una alineación de plantas "
                        f"cada {period * pixel_size_avg:.2f} m a {azimuth_of(angle):.1f}°, y en el "
                        f"brillo otra cada {b_period * pixel_size_avg:.2f} m a "
                        f"{azimuth_of(b_angle):.1f}°: se tomó la de mayor separación como las "
                        "hileras (la distancia entre hileras es mayor que la distancia entre "
                        "plantas). Si las hileras van en la otra dirección, indicá su distancia "
                        "en 'Distancia entre hileras'."
                    )
                )
                fft_info["marco"] = mc = dict(alt_period=period, alt_angle=angle, frac=float("nan"), por="brillo")
                self._square_doubt(doubts, b_period, period, pixel_size_avg)
                angle, period, peak = b_angle, b_period, b_peak
            if stripe_mode == self.STRIPE_DARK:
                swap = True
            elif stripe_mode == self.STRIPE_CHROMA:
                swap = False
            else:
                swap = cross or (same and b_peak > self.BRIGHTNESS_SWITCH * peak)
            if swap:
                if stripe_mode == self.STRIPE_DARK:
                    feedback.pushInfo(self.tr("Franja de la hilera: la oscura (brillo), elegida por el usuario."))
                elif not cross:
                    feedback.pushInfo(
                        self.tr(
                            f"El patrón de brillo es {b_peak / peak:.1f} veces más fuerte que el "
                            "de verdor: las hileras se ubican como las franjas oscuras. "
                            "Revisá que coincidan con la canopia."
                        )
                    )
                if same:
                    angle, period, peak = b_angle, b_period, b_peak
                exg, aux = -aux, exg
                aux_is_brightness = False

        # Hileras finas (T < FINE_T_PX): el verdor se calculó con las bandas
        # suavizadas sigma = 1 px (estabiliza el verdor de píxeles casi
        # negros), y a un período de 4 px ese suavizado deja pasar solo un
        # tercio del patrón de las hileras. Se recalcula con FINE_BLUR.
        if chroma_fn is not None and aux_is_brightness and period < self.FINE_T_PX:
            exg = chroma_fn(self.FINE_BLUR)
            feedback.pushInfo(
                self.tr(
                    f"Hileras finas ({period:.1f} px por hilera): verdor recalculado con "
                    f"menos suavizado (sigma {self.FINE_BLUR:g} px) para no borrarlas."
                )
            )

        # Si cada período ocupa muchos píxeles, trabajar a resolución
        # reducida: ~12 px por período alcanzan para ubicar las hileras, y
        # todo lo que sigue escala con el área. Se re-estima la FFT a esa
        # resolución porque cada ventana cubre más terreno (más períodos ->
        # mejor resolución angular).
        f = int(period // 12) if period > 24 else 1
        Hc, Wc = (exg.shape[0] // f) * f, (exg.shape[1] // f) * f
        if f > 1:
            work = cv2.resize(exg[:Hc, :Wc], (Wc // f, Hc // f), interpolation=cv2.INTER_AREA)
            work_valid = cv2.resize(
                valid[:Hc, :Wc].astype(np.float32), (Wc // f, Hc // f), interpolation=cv2.INTER_AREA
            ) > 0.999
            if aux is not None:
                aux = cv2.resize(aux[:Hc, :Wc], (Wc // f, Hc // f), interpolation=cv2.INTER_AREA)
            angle_w, period_w, peak_w = estimate_row_orientation_fft(work, work_valid, period / f)
            # Solo para afinar: si la re-estimación se va a otro patrón (otra
            # dirección del marco de plantación), se queda la primera. A
            # resolución completa, con pocos períodos por ventana, el ángulo
            # tiene una resolución de hasta ~12°.
            if (angle_w is not None and _angle_diff_mod180(angle_w, angle) <= 15.0
                    and abs(period_w - period / f) <= 0.25 * period / f):
                angle, peak = angle_w, peak_w
            else:
                period_w = period / f
        else:
            work, work_valid, period_w = exg, valid, period

        period_full = period_w * f
        if period_w < 3.0:
            raise QgsProcessingException(
                self.tr(
                    f"La separación entre hileras estimada es de {period_full:.1f} px "
                    f"({period_full * pixel_size_avg:.2f} m): con menos de ~3 píxeles por "
                    "hilera no hay forma de distinguirlas a esta resolución. Si la "
                    "separación real es otra, indicala en 'Distancia entre hileras'."
                )
            )
        if work.shape[0] + work.shape[1] > 32000:
            raise QgsProcessingException(
                self.tr(
                    "El raster es demasiado grande para el método de perfil a esta "
                    "resolución. Recortalo al lote de interés."
                )
            )

        angle = refine_row_angle(
            work, work_valid, angle, period_w,
            highpass=bool(fft_info.get("blanqueado")) or period_full * pixel_size_avg < self.FINE_ROWS_M,
        )

        # ¿Franja oscura o más verde? (automático, índice cromático; ver
        # C_NOT_GREEN). Con el ángulo ya afinado, para que el perfil a lo
        # ancho no se borronee.
        shadow_frac = 0.0   # sombra respecto de la canopia (fracción de T, con signo)
        dark_not_green = False   # franja oscura porque la canopia no se ve verde (ver HALF_MIN_T)
        dark_no_green_at_all = False   # ídem, con la canopia nada verde (verdor < C_NOT_GREEN)
        if green_fn is not None and aux is not None and stripe_mode == self.STRIPE_AUTO:
            dark_now = not aux_is_brightness
            c_w, y_w = (aux, -work) if dark_now else (work, aux)
            so = stripe_offset(c_w, y_w, work_valid, angle, period_w)
            if so is not None:
                c_pk, off, off_sgn = so
                if not dark_now and c_pk < self.C_NOT_GREEN:
                    work, aux = -y_w, c_w
                    aux_is_brightness = False
                    dark_not_green = True
                    dark_no_green_at_all = True
                    feedback.pushInfo(
                        self.tr(
                            "Franja de la hilera: la oscura. La canopia no se ve verde en la "
                            f"imagen (verdor de la franja más verde {c_pk:.2f}, negativo: más "
                            "rojiza que verde): sin hojas, de otoño o rojiza. Así, la franja "
                            "\"más verde\" es solo la menos rojiza, a "
                            "menudo el suelo; las plantas absorben más luz que el suelo y "
                            "quedan como la franja oscura. Revisá que las líneas caigan sobre "
                            "las plantas; si no, elegí la franja en 'Franja de la hilera'."
                        )
                    )
                elif not dark_now and c_pk < self.C_GREEN_CANOPY and off > self.DARK_OFFSET_MAX:
                    work, aux = -y_w, c_w
                    aux_is_brightness = False
                    dark_not_green = True
                    feedback.pushInfo(
                        self.tr(
                            "Franja de la hilera: la oscura. La canopia apenas se ve verde "
                            f"(verdor de la franja más verde {c_pk:.2f}) y la franja oscura está en "
                            f"otro lugar (corrida {off:.0%} de la separación): la más verde es el "
                            "suelo o la cobertura entre hileras, y las plantas, que absorben más luz, "
                            "quedan como la franja oscura. Revisá que las líneas caigan sobre las "
                            "plantas; si no, elegí la franja en 'Franja de la hilera'."
                        )
                    )
                elif not dark_now and c_pk >= self.C_GREEN_CANOPY and off > self.DARK_OFFSET_MAX:
                    # canopia verde con la sombra al costado, con el verdor ya
                    # como índice: se sigue con el verdor, pero la reubicación
                    # por el brillo tiene que apuntar a la canopia y no a la
                    # sombra (ver C_GREEN_CANOPY)
                    shadow_frac = off_sgn
                    feedback.pushInfo(
                        self.tr(
                            "Franja de la hilera: la más verde. La franja oscura está corrida "
                            f"{off:.0%} de la separación respecto de la más verde, que es "
                            f"canopia verde (verdor {c_pk:.2f}): la oscura es la sombra al "
                            "costado de la canopia, no la hilera."
                        )
                    )
                elif dark_now and c_pk >= self.C_GREEN_CANOPY and off > self.DARK_OFFSET_MAX:
                    work, aux = c_w, y_w
                    aux_is_brightness = True
                    shadow_frac = off_sgn
                    feedback.pushInfo(
                        self.tr(
                            "Franja de la hilera: la más verde. La franja oscura está corrida "
                            f"{off:.0%} de la separación respecto de la más verde, que es "
                            f"canopia verde (verdor {c_pk:.2f}): la oscura es la sombra al "
                            "costado de la canopia, no la hilera."
                        )
                    )
                # Dudoso: las dos candidatas están en lugares distintos y la
                # decisión quedó en la zona gris (canopia apenas verde) o,
                # con canopia verde, cerca del umbral de corrimiento (a un
                # paso del perfil). Sin verde (verdor < 0) la oscura va igual.
                if off > self.DARK_OFFSET_MAX and (
                        self.C_NOT_GREEN <= c_pk < self.C_GREEN_CANOPY
                        or (c_pk >= self.C_GREEN_CANOPY and off <= self.DARK_OFFSET_MAX + 1.0 / 12.0)):
                    doubts.append((
                        "franja",
                        f"la franja más verde y la más oscura están a "
                        f"{off * period_full * pixel_size_avg:.1f} m una de otra y la elección "
                        f"quedó cerca del límite (verdor {c_pk:.2f}, corrimiento {off:.0%}): se "
                        f"tomó la {'oscura' if not aux_is_brightness else 'más verde'}; si las "
                        "líneas no caen sobre las plantas, elegí la otra en 'Franja de la hilera'",
                    ))

        # ¿Qué franja es la hilera? El verdor cromático ubica la hilera en la
        # franja MÁS VERDE, que en viñedos y frutales con suelo desnudo entre
        # hileras es la canopia (oscura, con su sombra). En un frutal de copa
        # cerrada no hay suelo entre hileras: las copas iluminadas llenan casi
        # todo el ancho y el hueco entre ellas queda en sombra, que es verde
        # oscura y da el verdor cromático MÁS ALTO (en un píxel oscuro, la luz
        # que queda es la que filtran y reflejan las hojas). Ahí la franja
        # más verde es la sombra y la línea caía entre copas. Se reconoce
        # porque la franja a media separación también es vegetación (verdor
        # >= 30% del de la franja más verde; con suelo desnudo es 5-20%), es
        # más clara y es más verde en valor absoluto (2G-R-B >= 1,1 veces):
        # son las copas al sol. Solo con separación de frutal (>= 3,5 m):
        # en un viñedo con cubierta verde entre hileras la vid es la franja
        # oscura y el pasto la clara, con la misma firma.
        sunlit = False
        if green_fn is not None and (
                stripe_mode == self.STRIPE_SUNLIT
                or (stripe_mode == self.STRIPE_AUTO and aux_is_brightness
                    and period_full * pixel_size_avg >= self.ORCHARD_MIN_M)):
            green = green_fn()
            if f > 1:
                green = cv2.resize(green[:Hc, :Wc], (Wc // f, Hc // f), interpolation=cv2.INTER_AREA)
            use_green = stripe_mode == self.STRIPE_SUNLIT
            if not use_green and aux is not None:
                lv = stripe_levels([work, aux, green], work_valid, angle, period_w)
                if lv is not None:
                    (c_pk, y_pk, g_pk), (c_mid, y_mid, g_mid), _ = lv
                    use_green = (y_mid > y_pk and c_pk > 0 and c_mid >= self.SUNLIT_VEG_FRAC * c_pk
                                 and g_mid > 0 and g_mid >= self.SUNLIT_GREEN_RATIO * max(g_pk, 0.0))
                    mc_ = fft_info.get("marco")
                    beat = None
                    if use_green and mc_:
                        # período entre plantas A LO LARGO de la hilera: el del
                        # otro pico del marco, proyectado sobre la hilera
                        sin_ = abs(math.sin(math.radians(angle - mc_["alt_angle"])))
                        if sin_ > 0.3:
                            beat = lattice_amplitude(aux, work_valid, angle, period_w,
                                                     mc_["alt_period"] / f / sin_, work)
                    if beat is not None and beat[1] < self.TREE_BEAT_FRAC * beat[0]:
                        use_green = False
                        feedback.pushInfo(
                            self.tr(
                                "Franja de la hilera: la más verde. La franja a media "
                                "separación también es verde, pero es pareja a lo largo de "
                                f"la hilera (late {beat[1] / beat[0]:.0%} de lo que late la "
                                "franja más verde al ritmo de las plantas): es pasto o "
                                "cobertura entre hileras, y los árboles, sueltos, están en la "
                                "franja más verde."
                            )
                        )
                    elif not use_green:
                        if c_pk > 0 and g_pk > 0:
                            why_ = (f"a media separación, verdor {c_mid / c_pk:.0%} del de la hilera "
                                    f"y verde absoluto {g_mid / g_pk:.1f} veces: suelo o cobertura "
                                    "baja entre hileras")
                        else:
                            why_ = "la franja a media separación no es vegetación más verde"
                        feedback.pushInfo(self.tr(f"Franja de la hilera: la más verde ({why_})."))
                    else:
                        feedback.pushInfo(
                            self.tr(
                                f"Copa cerrada: la franja más verde es la sombra entre copas "
                                f"(la franja a media separación también es vegetación: verdor "
                                f"{c_mid / c_pk:.0%} del de la más verde, y es más verde en "
                                f"valor absoluto: {g_mid / max(g_pk, 1e-9):.1f} veces). Las "
                                "hileras se ubican sobre las copas iluminadas (verde absoluto "
                                "2G-R-B). Si en este lote la franja clara es pasto entre "
                                "hileras, elegí 'La más verde' en 'Franja de la hilera'."
                            )
                        )
                        doubts.append((
                            "franja",
                            "se aplicó copa cerrada (líneas sobre las copas al sol); si la "
                            "franja clara es pasto entre hileras, elegí 'La más verde' en 'Franja de la hilera'",
                        ))
            elif use_green:
                feedback.pushInfo(self.tr("Franja de la hilera: copas iluminadas (verde absoluto 2G-R-B), elegida por el usuario."))
            if use_green:
                sunlit = True
                # Sin brillo como evidencia auxiliar: con copa cerrada el
                # brillo marca sombras, y con la hilera como franja CLARA el
                # suelo desnudo de una calle (más claro todavía que las
                # copas) pasaba el control de nivel y daba hileras falsas.
                work = green
                aux = None
                aux_is_brightness = False
            del green

        # Rumbo respecto del norte (0-180°), usando el geotransform para
        # pasar la dirección de píxeles a coordenadas del mapa.
        th = math.radians(angle)
        ux, uy = math.cos(th), math.sin(th)
        dxm = geotransform[1] * ux + geotransform[2] * uy
        dym = geotransform[4] * ux + geotransform[5] * uy
        azimuth = math.degrees(math.atan2(dxm, dym)) % 180.0
        feedback.pushInfo(
            self.tr(
                f"Hileras: rumbo {azimuth:.1f}° respecto del norte, separación "
                f"{period_full:.1f} px = {period_full * pixel_size_avg:.2f} m "
                f"(pico espectral {peak:.0f} veces el fondo)."
            )
        )
        if peak < self.MIN_SPECTRAL_PEAK:
            feedback.reportError(
                self.tr(
                    "El pico espectral es bajo: no se ve una periodicidad clara de "
                    "hileras. Revisá que el rumbo y la separación del mensaje "
                    "anterior sean los reales; si no, indicá la 'Distancia "
                    "entre hileras'."
                ),
                fatalError=False,
            )
            doubts.append(("rumbo", "pico espectral bajo: el rumbo y la separación pueden no ser los reales"))
        if f > 1:
            feedback.pushInfo(
                self.tr(f"Trabajando a 1/{f} de la resolución original (~{period_w:.0f} px por hilera).")
            )

        feedback.setProgress(55)
        if feedback.isCanceled():
            return None

        max_gap_px = max_gap_param if max_gap_param is not None and max_gap_param > 0 else 0.0
        # Centro de la banda (ver BAND_MAX_T y BAND_GAIN_NOT_GREEN): solo con
        # la franja oscura (el índice es el brillo) y no en la zona gris.
        band_gain = None
        if aux is not None and not aux_is_brightness and not sunlit:
            if dark_no_green_at_all:
                band_gain = self.BAND_GAIN_NOT_GREEN
            elif not dark_not_green:
                band_gain = self.BAND_GAIN_BRIGHT
        sample_w, row_w = work_valid, None
        if cross_masks is not None:
            sample_m, row_m = cross_masks(angle, period_full)
            if f > 1:
                def down(m, thr):
                    return cv2.resize(m[:Hc, :Wc].astype(np.float32), (Wc // f, Hc // f),
                                      interpolation=cv2.INTER_AREA) > thr
                sample_w, row_w = down(sample_m, 0.999), down(row_m, 0.5)
            else:
                sample_w, row_w = sample_m, row_m
        segs_w, row_ids, info = detect_rows_by_profile(
            cv2, work, sample_w, angle, period_w,
            min_line_length / f, max_gap_px / f, sensitivity, aux=aux,
            aux_localiza=aux_is_brightness, row_valid=row_w,
            level_min=LEVEL_MIN_SUNLIT if sunlit else LEVEL_MIN, wide_rows=sunlit,
            track_max_px=TRACK_MAX_M / max(pixel_size_avg * f, 1e-9),
            aux_siembra=aux is not None and not aux_is_brightness,
            shadow_px=shadow_frac * period_w,
            half_center=dark_not_green,
            band_gain=band_gain,
            canceled=feedback.isCanceled,
        )
        if feedback.isCanceled():
            return None
        # centro de píxel a centro de píxel entre resoluciones
        lines = [tuple((v + 0.5) * f - 0.5 for v in seg) for seg in segs_w]

        feedback.setProgress(70)
        if info["tau"] is not None:
            feedback.pushInfo(
                self.tr(
                    f"Resultado: {info['n_hileras']} hileras ubicadas "
                    f"({info.get('n_agregadas', 0)} agregadas por continuidad del "
                    f"patrón, típicamente de borde; {info.get('n_grilla', 0)} por la "
                    f"grilla del marco de plantación; {info.get('reubicadas', 0)} "
                    f"reubicadas de la entrehilera a la hilera por el patrón de brillo), "
                    f"{len(set(row_ids))} con vegetación detectada, "
                    f"{info.get('extremos_regularizados', 0)} extremos alineados "
                    f"con el borde de lo plantado, "
                    f"{info.get('puntas_recuperadas', 0)} puntas recuperadas tras un corte chico, "
                    f"{info.get('recuperados_por_nivel', 0)} tramos de plantas chicas recuperados, "
                    f"{info.get('extendidas_por_plantas', 0)} hileras completadas planta por "
                    f"planta (plantas chicas), {info.get('seguidas', 0)} hileras seguidas de "
                    f"costado (levemente torcidas; {info.get('tramos_reajustados', 0)} tramos "
                    f"reajustados), {info.get('sembradas_debiles', 0)} tramos de evidencia "
                    f"débil pero sostenida, {info.get('descartados_por_nivel', 0)} tramos "
                    f"descartados por no tener el nivel de una hilera (huellas, sombras en la "
                    f"calle), {len(lines)} "
                    f"tramos (umbral de contraste {info['tau']:.3g} = "
                    f"{sensitivity:g} x ruido {info['sigma']:.3g})."
                )
            )
        if info.get("alternadas"):
            feedback.pushInfo(
                self.tr(
                    "Entrehileras alternadas (rastra o cobertura hilera por medio): de un "
                    "lado de cada hilera el suelo es claro y del otro la cobertura se ve casi "
                    "igual a las plantas; alcanza con que la hilera se distinga del lado claro."
                )
            )
        if info.get("centro_banda"):
            p_pk, p_bc = info["banda_presencia"]
            feedback.pushInfo(
                self.tr(
                    f"Hileras en el centro de su banda oscura ({info['centro_banda']} movidas desde lo "
                    "más oscuro): la banda tiene dos jorobas (camellón) o lo más oscuro es la sombra, "
                    f"a un costado; desde el centro la hilera se ve en {p_bc / max(p_pk, 1.0):.1f} veces "
                    "más largo."
                )
            )
        if info.get("pendiente_propia"):
            prop_ = info.get("pendiente_propagada", 0)
            feedback.pushInfo(
                self.tr(
                    f"{info['pendiente_propia']} hileras con rumbo propio, levemente distinto del "
                    "del lote (hileras no del todo paralelas)"
                    + (f"; {prop_} de ellas, siguiendo el rumbo de su vecina." if prop_ else ".")
                )
            )
        if info.get("rellenos_vecinas"):
            feedback.pushInfo(
                self.tr(
                    f"{info['rellenos_vecinas']} huecos o puntas de hileras tenues completados donde "
                    "sus dos vecinas siguen (la hilera sigue siendo más oscura que sus entrehileras)."
                )
            )
        if info.get("cortes_transversales"):
            feedback.pushInfo(
                self.tr(
                    f"{info['cortes_transversales']} cortes transversales (reguera, acequia o camino "
                    "que cruza las hileras): las líneas se cortan donde los huecos de muchas hileras "
                    "seguidas caen sobre una misma recta."
                )
            )
        if info.get("centro_media_altura"):
            feedback.pushInfo(
                self.tr(
                    "Banda oscura asimétrica: lo más oscuro es la sombra, a un costado, y la "
                    "planta queda del otro lado. Las líneas se corrieron "
                    f"{abs(info['centro_media_altura']) * f * pixel_size_avg:.2f} m al centro de la banda."
                )
            )
        if info.get("descartados_cortina"):
            feedback.pushInfo(
                self.tr(
                    f"{info['descartados_cortina']} tramos o hileras de borde descartados por ser "
                    "mucho más marcados que las hileras del cultivo: cortina o fila de árboles "
                    "de la calle junto al lote."
                )
            )
        if info.get("modo_patron"):
            lp_m = info.get("patron_largo_px", 0) * f * pixel_size_avg
            feedback.pushInfo(
                self.tr(
                    "Hileras al límite de la resolución: casi ninguna se ve sola "
                    "(su contraste apenas supera el ruido), pero el patrón del lote es "
                    f"claro. {info.get('n_patron', 0)} hileras se marcaron o completaron donde el patrón "
                    f"de {PATTERN_ROWS} hileras vecinas "
                    f"es claro a lo largo de ~{lp_m:.0f} m: la posición de cada línea es la "
                    "del perfil del lote, los extremos tienen esa precisión y no se ven "
                    "fallas ni plantas faltantes. Para detalle hilera por hilera hace "
                    "falta una imagen de mayor resolución (drone)."
                )
            )
        if not lines:
            feedback.reportError(
                self.tr(
                    "No se detectó ningún tramo de hilera. Probá bajar la "
                    "'Sensibilidad' (parámetros avanzados) o revisá en el log "
                    "que el rumbo y la separación detectados sean los reales."
                ),
                fatalError=False,
            )
        n_loc, n_seg = info.get("n_hileras", 0), len(set(row_ids))
        if n_loc >= 5 and n_seg < self.COVER_MIN_FRAC * n_loc:
            doubts.append((
                "cobertura",
                f"{n_loc - n_seg} de {n_loc} hileras ubicadas quedaron sin línea: puede que "
                "la franja o el rumbo no sean los reales, o que la imagen no alcance",
            ))
        # ¿Junta el lote cuadros de plantación con otra separación?
        pb = planting_blocks(work, sample_w, angle, period_w)
        if pb:
            med_ = float(np.median([b_[1] for b_ in pb]))
            otra = [b_ for b_ in pb if abs(b_[2] - 1.0) >= self.BLOCK_DEV and b_[3] >= self.BLOCK_X * b_[1]
                    and b_[3] >= self.BLOCK_AMP_MIN * med_]
            if len(otra) >= self.BLOCK_MIN and len(otra) >= self.BLOCK_FRAC * len(pb):
                sep_ = float(np.median([b_[2] for b_ in otra])) * period_full * pixel_size_avg
                doubts.append((
                    "cuadros",
                    f"en {100.0 * len(otra) / len(pb):.0f}% del lote las hileras parecen ir cada "
                    f"~{sep_:.1f} m y no cada {period_full * pixel_size_avg:.2f} m: puede juntar cuadros "
                    "de plantación distintos, y el complemento usa una sola separación por lote; "
                    "conviene dividirlo en un polígono por cuadro",
                ))
        if doubts:
            feedback.reportError(
                self.tr("REVISAR este lote sobre la imagen: " + "; ".join(t_ for _, t_ in doubts) + "."),
                fatalError=False,
            )
        self._doubts = sorted({c_ for c_, _ in doubts})
        return lines, row_ids, period_full

