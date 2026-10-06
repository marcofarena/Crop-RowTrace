# -*- coding: utf-8 -*-
"""
Detección de Hileras de Cultivo
--------------------------------
Flujo común:
  1. Lee el raster de entrada con GDAL (bandas R, G, B elegidas por el
     usuario) y su máscara de píxeles válidos (alfa / nodata).
  2. Calcula el índice de vegetación ExG cromático (2G-R-B)/(R+G+B) (o, como
     opción, el ExG sin normalizar 2G-R-B) para resaltar la vegetación.
  3. Binariza (Otsu automático sobre los píxeles válidos, o umbral manual).

Método "perfil de proyección" (default, hileras rectas y paralelas):
  4. Estima orientación y separación de hileras con el espectro de
     potencia 2D del ExG (FFT, promedio de ventanas).
  5. Remuestrea el ExG a una grilla rotada donde las hileras quedan
     verticales; los picos del perfil perpendicular son los centros de
     hilera.
  6. A lo largo de cada hilera, marca como "hilera presente" los tramos
     donde el centro de la hilera es más verde que las dos entrehileras
     vecinas (contraste local, umbral relativo al ruido, con histéresis), o
     donde continúa su patrón de brillo; ubica los extremos a media altura,
     une huecos cortos y descarta tramos cortos.
  7. Otros cultivos: en frutales de copa cerrada ubica las hileras sobre
     las copas iluminadas (verde absoluto), no sobre la sombra entre ellas;
     con hileras finas busca la separación en el espectro blanqueado y, al
     límite de la resolución, marca las hileras por el patrón de varias
     hileras vecinas (modo patrón).

Método "Canny + Hough" (anterior):
  4. Opcionalmente aplica un cierre morfológico, detecta bordes con Canny y
     segmentos con la Transformada de Hough probabilística (cv2.HoughLinesP).
  5. Opcionalmente filtra los segmentos por dirección dominante.
  6. Fusiona los segmentos de una misma hilera en una sola línea (PCA +
     tope de longitud + tope de hueco longitudinal).

Final común:
  7. Recorta las líneas al contorno de la parcela (capa opcional o área con
     datos del raster) y, en el método de perfil, extiende cada hilera
     hasta el borde si le falta poco.
  8. Convierte cada línea de coordenadas de píxel a coordenadas del mapa
     usando el geotransform del raster, y escribe una capa vectorial.

Dependencias extra (no vienen con QGIS por defecto):
  - opencv-python (o opencv-python-headless)
  - numpy (normalmente ya viene con QGIS)
Instalación (ver README.md del plugin).
"""

import math
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
    QgsFields,
    QgsField,
    QgsFeature,
    QgsFeatureRequest,
    QgsGeometry,
    QgsPointXY,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant, QCoreApplication


def _angle_deg_mod180(dx, dy):
    ang = math.degrees(math.atan2(dy, dx)) % 180.0
    return ang


def _angle_diff_mod180(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _recompute_cluster(cluster):
    """Recalcula dirección (PCA), punto de referencia y ángulo del cluster
    a partir de todos los puntos (extremos de segmentos) que contiene."""
    pts = np.array(cluster["points"], dtype=np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]  # eigenvector con mayor varianza
    norm = np.linalg.norm(direction)
    if norm > 1e-9:
        direction = direction / norm
    cluster["point"] = mean
    cluster["direction"] = direction
    cluster["angle"] = _angle_deg_mod180(direction[0], direction[1])


def _point_perp_distance(px, py, line_point, direction):
    vx, vy = px - line_point[0], py - line_point[1]
    cross = vx * direction[1] - vy * direction[0]
    return abs(cross)


def _projection_interval(points, ref_point, direction):
    """Proyecta una lista de puntos sobre `direction` (a partir de
    ref_point) y devuelve (min, max) de esas proyecciones — el rango que
    ocupan esos puntos A LO LARGO de la dirección de la hilera."""
    pts = np.array(points, dtype=np.float64)
    proj = (pts - ref_point) @ direction
    return float(proj.min()), float(proj.max())


def estimate_dominant_angle_deg(lines):
    """Estima la orientación dominante de un conjunto de segmentos, ponderando
    cada uno por su longitud (no por cantidad). Usa el truco de 'ángulo
    duplicado' para tratar la orientación (que es periódica cada 180°, no
    360°) como una cantidad circular: cada segmento aporta un vector de
    longitud proporcional a su largo real, en dirección 2*ángulo; el ángulo
    dominante es la dirección del vector resultante (dividida por 2).

    Esto funciona incluso si hay muchísimo ruido en direcciones dispersas
    (p. ej. bordes de copas de plantas individuales, que no tienen una
    dirección preferente): esos vectores tienden a cancelarse entre sí,
    mientras que los segmentos que sí siguen una dirección común (p. ej.
    bordes reales de hilera) se refuerzan.

    Devuelve (angulo_dominante_deg, concentracion) donde concentracion está
    en [0, 1]: 0 = direcciones completamente dispersas (sin una orientación
    dominante clara), 1 = todos los segmentos perfectamente alineados.
    Devuelve (None, 0.0) si no hay segmentos."""
    if len(lines) == 0:
        return None, 0.0

    sum_cos = 0.0
    sum_sin = 0.0
    total_len = 0.0
    for (x1, y1, x2, y2) in lines:
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1e-9:
            continue
        ang = _angle_deg_mod180(x2 - x1, y2 - y1)
        theta2 = math.radians(2.0 * ang)
        sum_cos += length * math.cos(theta2)
        sum_sin += length * math.sin(theta2)
        total_len += length

    if total_len < 1e-9:
        return None, 0.0

    mean_theta2 = math.atan2(sum_sin, sum_cos)
    dominant_angle = (math.degrees(mean_theta2) / 2.0) % 180.0
    concentration = math.hypot(sum_cos, sum_sin) / total_len
    return dominant_angle, concentration


def filter_lines_by_angle(lines, target_angle_deg, tol_deg):
    """Descarta los segmentos cuyo ángulo (mod 180°) se aleja de
    target_angle_deg más que tol_deg. Se aplica ANTES de la fusión, para
    que el ruido en otras direcciones (bordes de copas individuales,
    caminos, límites de parcela con otra orientación) no llegue siquiera
    a la etapa de fusión."""
    kept = []
    for seg in lines:
        x1, y1, x2, y2 = seg
        ang = _angle_deg_mod180(x2 - x1, y2 - y1)
        if _angle_diff_mod180(ang, target_angle_deg) <= tol_deg:
            kept.append(seg)
    return kept


def _normalize_vec(vx, vy):
    n = math.hypot(vx, vy)
    if n < 1e-9:
        return 0.0, 0.0
    return vx / n, vy / n


def estimate_perp_spacing_px(lines, dominant_angle_deg, bin_px=2.0):
    """Estima la separación perpendicular típica entre hileras vecinas a
    partir de los segmentos SIN fusionar (ya filtrados por dirección),
    usando autocorrelación: proyecta el punto medio de cada segmento sobre
    el eje perpendicular a la dirección dominante, arma un histograma de
    densidad a lo largo de ese eje, y busca la periodicidad dominante de
    ese histograma. Es la misma técnica que separar las notas de un
    acorde: si hay hileras regularmente espaciadas, la densidad de puntos
    va a repetirse con ese período.

    A diferencia de estimate_row_spacing_px (que mide distancia entre
    hileras YA fusionadas, una línea por hilera), esta función mide sobre
    los segmentos crudos — sirve para calibrar la tolerancia de fusión
    ANTES de fusionar, que es lo que realmente hace falta.

    Devuelve la separación en píxeles, o None si no hay suficientes datos
    o no se detecta una periodicidad clara."""
    if len(lines) < 6:
        return None

    theta = math.radians(dominant_angle_deg)
    perp = (-math.sin(theta), math.cos(theta))
    offsets = []
    for (x1, y1, x2, y2) in lines:
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        offsets.append(mx * perp[0] + my * perp[1])
    offsets = np.array(offsets, dtype=np.float64)

    lo, hi = offsets.min(), offsets.max()
    span = hi - lo
    if span < bin_px * 6:
        return None

    n_bins = int(span / bin_px) + 1
    hist, _ = np.histogram(offsets, bins=n_bins, range=(lo, hi))
    density = hist.astype(np.float64)
    density -= density.mean()
    if np.allclose(density, 0.0):
        return None

    # Autocorrelación vía FFT (mucho más rápido y estable que calcularla
    # "a mano" desplazando el array), quedándonos solo con lags positivos.
    n = len(density)
    padded = np.zeros(2 * n)
    padded[:n] = density
    f = np.fft.rfft(padded)
    power = (f * np.conj(f)).real
    autocorr = np.fft.irfft(power)[:n]

    min_lag = max(2, int(round(1.0 / bin_px)) + 1)
    if n <= min_lag + 2:
        return None

    peak_lag = min_lag + int(np.argmax(autocorr[min_lag:]))
    if autocorr[peak_lag] <= 0:
        return None

    return peak_lag * bin_px


def merge_segments_into_rows(lines, angle_tol_deg, dist_tol_px, max_length_px=None, max_gap_px=None):
    """
    Agrupa segmentos de Hough que pertenecen a la misma hilera (ángulo
    similar + cercanos perpendicularmente + cercanos a lo largo de la
    hilera) y devuelve UNA línea por grupo, ajustada por PCA y extendida
    al alcance real de todos sus segmentos.

    max_length_px (opcional): tope duro a cuánto puede "crecer" una hilera
    fusionada en total.

    max_gap_px (opcional, la corrección más importante): espacio máximo,
    medido A LO LARGO de la hilera (no perpendicular), entre el tramo ya
    fusionado y el próximo segmento candidato. Sin este chequeo, dos
    segmentos con ángulo y offset perpendicular parecidos se fusionan
    aunque estén a cientos de píxeles de distancia entre sí a lo largo de
    la hilera (con un camino, un límite de parcela, o cualquier zona sin
    relación en el medio) — incluso en un campo con una única dirección
    real consistente en toda la escena. Es la causa más común de líneas
    absurdamente largas que cruzan zonas sin relación entre sí.
    """
    clusters = []

    for (x1, y1, x2, y2) in lines:
        seg_angle = _angle_deg_mod180(x2 - x1, y2 - y1)
        seg_mid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        assigned = False
        for cluster in clusters:
            if _angle_diff_mod180(seg_angle, cluster["angle"]) <= angle_tol_deg:
                d = _point_perp_distance(seg_mid[0], seg_mid[1], cluster["point"], cluster["direction"])
                if d <= dist_tol_px:
                    if max_gap_px is not None:
                        c_min, c_max = _projection_interval(
                            cluster["points"], cluster["point"], cluster["direction"]
                        )
                        s_min, s_max = _projection_interval(
                            [(x1, y1), (x2, y2)], cluster["point"], cluster["direction"]
                        )
                        gap = max(0.0, s_min - c_max, c_min - s_max)
                        if gap > max_gap_px:
                            continue  # mismo angulo/offset, pero demasiado lejos a lo largo de la hilera
                    if max_length_px is not None:
                        trial_pts = np.array(cluster["points"] + [(x1, y1), (x2, y2)], dtype=np.float64)
                        proj = (trial_pts - cluster["point"]) @ cluster["direction"]
                        trial_length = float(proj.max() - proj.min())
                        if trial_length > max_length_px:
                            continue  # rechazar esta fusión, probar con otro cluster
                    cluster["points"].append((x1, y1))
                    cluster["points"].append((x2, y2))
                    _recompute_cluster(cluster)
                    assigned = True
                    break

        if not assigned:
            new_cluster = {"points": [(x1, y1), (x2, y2)]}
            _recompute_cluster(new_cluster)
            clusters.append(new_cluster)

    merged_lines = []
    for cluster in clusters:
        pts = np.array(cluster["points"], dtype=np.float64)
        direction = cluster["direction"]
        point = cluster["point"]
        # proyectar cada punto sobre la dirección principal para hallar los extremos
        projections = (pts - point) @ direction
        p_min = pts[np.argmin(projections)]
        p_max = pts[np.argmax(projections)]
        merged_lines.append((p_min[0], p_min[1], p_max[0], p_max[1]))

    return merged_lines


def estimate_row_spacing_px(lines):
    """Distancia mediana (en píxeles) entre hileras vecinas, proyectando el
    punto medio de cada una sobre el eje perpendicular a la dirección
    dominante de todas las hileras. Devuelve None si hay menos de 2 hileras
    (no se puede estimar un espaciado). Se usa DESPUÉS de fusionar (una
    línea por hilera) — para calibrar el ancho de franja cultivable del
    cálculo de cobertura/conteo de plantas, no para la fusión en sí (eso
    lo hace estimate_perp_spacing_px, sobre los segmentos sin fusionar)."""
    if len(lines) < 2:
        return None

    dirs = []
    for (x1, y1, x2, y2) in lines:
        dx, dy = _normalize_vec(x2 - x1, y2 - y1)
        if dx < 0:
            dx, dy = -dx, -dy
        dirs.append((dx, dy))
    avg_dx = float(np.mean([d[0] for d in dirs]))
    avg_dy = float(np.mean([d[1] for d in dirs]))
    avg_dx, avg_dy = _normalize_vec(avg_dx, avg_dy)
    perp = (-avg_dy, avg_dx)

    offsets = []
    for (x1, y1, x2, y2) in lines:
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        offsets.append(mx * perp[0] + my * perp[1])
    offsets.sort()
    gaps = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
    if not gaps:
        return None
    return float(np.median(gaps))


def _strip_mask_bbox(x1, y1, x2, y2, half_width_px, shape):
    """Devuelve (row_min, row_max, col_min, col_max) y una máscara booleana
    (del tamaño de ese recorte) True para los píxeles que caen dentro de la
    franja cultivable de la hilera: cerca perpendicularmente (<= half_width_px)
    y dentro del alcance a lo largo de la línea."""
    H, W = shape
    dx, dy = _normalize_vec(x2 - x1, y2 - y1)
    perp = (-dy, dx)
    length = math.hypot(x2 - x1, y2 - y1)

    xs = [x1, x2, x1 + perp[0]*half_width_px, x1 - perp[0]*half_width_px,
          x2 + perp[0]*half_width_px, x2 - perp[0]*half_width_px]
    ys = [y1, y2, y1 + perp[1]*half_width_px, y1 - perp[1]*half_width_px,
          y2 + perp[1]*half_width_px, y2 - perp[1]*half_width_px]
    col_min = max(0, int(math.floor(min(xs))))
    col_max = min(W - 1, int(math.ceil(max(xs))))
    row_min = max(0, int(math.floor(min(ys))))
    row_max = min(H - 1, int(math.ceil(max(ys))))
    if col_max <= col_min or row_max <= row_min:
        return (row_min, row_max, col_min, col_max), None

    cols = np.arange(col_min, col_max + 1)
    rows = np.arange(row_min, row_max + 1)
    cc, rr = np.meshgrid(cols, rows)

    vx, vy = cc - x1, rr - y1
    along = vx * dx + vy * dy
    perp_dist = vx * perp[0] + vy * perp[1]

    mask = (np.abs(perp_dist) <= half_width_px) & (along >= 0) & (along <= length)
    return (row_min, row_max, col_min, col_max), mask


def compute_row_coverage_and_count(cv2_module, binary_u8, x1, y1, x2, y2,
                                    half_width_px, min_plant_area_px, pixel_area_m2):
    """Calcula, para la franja cultivable alrededor de una hilera:
      - % de cobertura de vegetación dentro de esa franja
      - número estimado de plantas (componentes conectados en la máscara
        binaria, filtrando blobs más chicos que min_plant_area_px)
      - área promedio de esos blobs (m2), como dato de diagnóstico: si es
        mucho más grande de lo esperado para una sola planta, es señal de
        que el dosel está cerrado y el conteo está subestimado.
    Devuelve un dict con claves cobertura_pct, num_plantas, area_prom_m2
    (todas en None si no hay píxeles válidos en la franja)."""
    H, W = binary_u8.shape
    (row_min, row_max, col_min, col_max), mask = _strip_mask_bbox(
        x1, y1, x2, y2, half_width_px, (H, W)
    )
    if mask is None or mask.sum() == 0:
        return {"cobertura_pct": None, "num_plantas": None, "area_prom_m2": None}

    sub_bin = binary_u8[row_min:row_max + 1, col_min:col_max + 1]
    total_px = int(mask.sum())
    veg_px = int(((sub_bin > 0) & mask).sum())
    cobertura_pct = 100.0 * veg_px / total_px

    sub_masked = np.where(mask, sub_bin, 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2_module.connectedComponentsWithStats(sub_masked, connectivity=8)
    areas = stats[1:, cv2_module.CC_STAT_AREA]  # excluir el fondo (label 0)
    areas = areas[areas >= min_plant_area_px]
    num_plantas = int(len(areas))
    area_prom_m2 = float(areas.mean() * pixel_area_m2) if num_plantas > 0 else 0.0

    return {
        "cobertura_pct": cobertura_pct,
        "num_plantas": num_plantas,
        "area_prom_m2": area_prom_m2,
    }


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
    for r_ in range(len(ring_med)):
        v_ = acc[rbin == r_]
        if v_.size:
            ring_med[r_] = max(float(np.median(v_)), 1e-30)
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
            pk_b = local_max(np.where(band_all, acc, 0.0))
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
# hileras (ver el paso 5 de detect_rows_by_profile).
GAP_T = 1.25
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


def detect_rows_by_profile(cv2_module, img, valid, angle_deg, period_px,
                           min_len_px, max_gap_px, k_sigma=3.0, aux=None,
                           aux_localiza=False, row_valid=None, level_min=LEVEL_MIN,
                           wide_rows=False, track_max_px=None, aux_siembra=False, shadow_px=0.0):
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
    grandes una vez promediadas, pero la hilera sigue hasta ellas."""
    cv2 = cv2_module
    H, W = img.shape
    T = float(period_px)
    info = {"n_hileras": 0, "sigma": None, "tau": None}

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
        ok = (jc >= 0) & (jc < NT) & RR[ss, np.clip(jc, 0, NT - 1)]
        taps = {"c": []}
        for d in range(-K, K + 1):
            k = base + d
            taps["c"].append((k, np.clip(np.minimum(k + 0.5, tt + hw) - np.maximum(k - 0.5, tt - hw), 0.0, 1.0)))
        for key, offset in (("l", -T / 2.0), ("r", T / 2.0)):
            x = tt + offset
            k0 = np.floor(x).astype(np.int64)
            f = x - k0
            taps[key] = [(k0, 1.0 - f), (k0 + 1, f)]
        for key in taps:
            clean = []
            for k, w in taps[key]:
                used = w > 1e-9
                kc = np.clip(k, 0, NT - 1)
                ok &= ~used | ((k >= 0) & (k < NT) & RV[ss, kc])
                clean.append((kc, np.where(used, w, 0.0)))
            wsum = np.maximum(sum(w for _, w in clean), 1e-12)
            taps[key] = (clean, wsum)
        okf = ok.astype(np.float64)
        norm = np.maximum(np.convolve(okf, box, "same"), 1e-9)
        norm_p = np.maximum(np.convolve(okf, box_p, "same"), 1e-9)

        def smooth(x):
            return np.convolve(np.where(ok, x, 0.0), box, "same") / norm

        out = []
        for G in (R, RA):
            if G is None:
                out.append(None)
                continue
            c, l, r = (sum(G[ss, k] * w for k, w in taps[key][0]) / taps[key][1] for key in ("c", "l", "r"))
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
    for j in centers:
        a_, b_, c_ = prof_s[j - 1], prof_s[j], prof_s[j + 1]
        den = a_ - 2.0 * b_ + c_
        off = 0.5 * (a_ - c_) / den if abs(den) > 1e-12 else 0.0
        a0 = j + float(np.clip(off, -0.5, 0.5))
        main, auxc = contrast_along(a0, 0.0)
        rows.append([a0, 0.0, main, auxc])

    finite = [r[2][0][np.isfinite(r[2][0])] for r in rows]
    allc = np.concatenate(finite) if finite else np.array([])
    if allc.size == 0:
        return [], [], info
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

    # Pasada 2: reubicar el centro de cada hilera por bloques a lo largo de
    # ella, usando SOLO los bloques donde la pasada 1 vio la hilera presente
    # (en un camino o una cabecera no hay hilera, y el "centro" de ese
    # bloque sería un máximo de ruido). Por defecto la recta sigue paralela
    # al ángulo global, desplazada a la mediana de esos centros; solo con
    # 4+ bloques se ajusta además una pendiente propia, y solo si es
    # significativa: tolera hileras levemente torcidas o distorsión del
    # ortomosaico sin inventar pendientes a partir del ruido.
    blk = int(max(10 * T, 60))
    for row, j in zip(rows, centers):
        a0_1 = row[0]
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
            pts_s.append(0.5 * (b0 + b1))
            pts_t.append(jlo + k + float(np.clip(off, -0.5, 0.5)))
        if not pts_t:
            continue
        pts_s, pts_t = np.array(pts_s), np.array(pts_t)
        a0, a1 = a0_1, 0.0
        med = float(np.median(pts_t))
        if abs(med - j) <= T / 4.0:
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
                        and abs(c0 + c1 * mid - j) <= T / 2.0):
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
    # diagnóstico: posición perpendicular (px de img, t = -x*sin + y*cos del
    # ángulo) de cada hilera ubicada, y cuáles tienen tramos
    info["filas_t"] = [t0 + row[0] for row in rows]
    info["filas_con_tramos"] = sorted(r[0] for r in row_iv)

    info["tramos_reajustados"] = 0
    for row_num, a0, a1, intervals in row_iv:
        dt = track.get(row_num)
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
    METHOD = "METHOD"
    VEG_INDEX = "VEG_INDEX"
    ROW_SPACING = "ROW_SPACING"
    SENSITIVITY = "SENSITIVITY"
    CONTOUR = "CONTOUR"
    EXTEND_MAX = "EXTEND_MAX"
    THRESHOLD = "THRESHOLD"
    MIN_LINE_LENGTH = "MIN_LINE_LENGTH"
    MAX_LINE_GAP = "MAX_LINE_GAP"
    HOUGH_THRESHOLD = "HOUGH_THRESHOLD"
    CLOSING_PX = "CLOSING_PX"
    FILTER_ANGLE = "FILTER_ANGLE"
    ANGLE_FILTER_TOL = "ANGLE_FILTER_TOL"
    MERGE_LINES = "MERGE_LINES"
    MERGE_ANGLE_TOL = "MERGE_ANGLE_TOL"
    MERGE_DIST_TOL = "MERGE_DIST_TOL"
    MERGE_MAX_GAP = "MERGE_MAX_GAP"
    MERGE_MAX_LENGTH = "MERGE_MAX_LENGTH"
    CALC_COVERAGE = "CALC_COVERAGE"
    COVERAGE_WIDTH = "COVERAGE_WIDTH"
    CALC_PLANT_COUNT = "CALC_PLANT_COUNT"
    MIN_PLANT_AREA = "MIN_PLANT_AREA"
    SPACING_FIELD = "SPACING_FIELD"
    STRIPE = "STRIPE"
    OUTPUT = "OUTPUT"

    METHOD_PROFILE = 0
    METHOD_HOUGH = 1

    INDEX_EXG_CHROMATIC = 0
    INDEX_EXG_RAW = 1

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
    C_NOT_GREEN = 0.0
    C_GREEN_CANOPY = 0.10
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
    # Hileras sobre el borde del contorno, en fracciones de la separación T
    # y solo A LO ANCHO de las hileras (a lo largo, el contorno se respeta
    # estricto): una hilera cuyo eje queda hasta 0,4 T afuera del contorno es
    # la hilera de borde del lote (el contorno se dibujó sobre ella) y se
    # conserva; la siguiente hacia afuera estaría a más de 0,5 T y no entra.
    # Para medir su contraste se muestrea hasta 1 T afuera (su entrehilera
    # exterior).
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
        return self.tr(
            "Detecta hileras de cultivo en un raster (ortomosaico RGB) sin "
            "necesidad de entrenar un modelo. Usa el índice ExG cromático "
            "(2G-R-B)/(R+G+B), que mide qué tan verde es cada píxel "
            "independientemente de su brillo, y georreferencia el resultado "
            "automáticamente. Los píxeles transparentes o nodata (p. ej. "
            "fuera del recorte de la parcela) se ignoran.\n\n"
            "Extremos de hilera: cada hilera termina donde termina lo "
            "plantado, sin entrar en la calle de cabecera. Los extremos que "
            "se apartan del borde que forman las hileras vecinas (más de "
            "media separación entre hileras, hasta 3 separaciones) se "
            "alinean con ese borde; donde hay un escalón real (un rincón sin "
            "plantar) no se tocan. Las plantas chicas de la punta (replantes, "
            "menor vigor) se incluyen: la hilera se prolonga planta por planta "
            "mientras haya plantas a menos de una separación entre hileras una "
            "de otra; un árbol aislado más lejos, en la cabecera, no se une. "
            "Una punta separada del resto por una o dos plantas débiles o "
            "faltantes se suma si no pasa del borde que forman las hileras "
            "vecinas.\n\n"
            "Frutal en marco (árboles sueltos alineados en dos direcciones): se "
            "toma como hileras la alineación de mayor separación, mirando el "
            "verdor y el brillo (en un olivar las hileras solo se veían en el "
            "brillo), o, si se indica la 'Distancia entre hileras', la más "
            "fuerte a ±10% de ese valor; el log muestra las dos. Con pasto "
            "verde entre árboles sueltos no se aplica copa cerrada.\n\n"
            "REVISAR: cuando una decisión automática queda con poco margen, el "
            "log lo avisa en rojo al final de cada lote y cada línea lleva el "
            "motivo en el campo 'revisar' (vacío = sin dudas): 'rumbo' (marco "
            "casi cuadrado o periodicidad débil), 'franja' (la franja más verde "
            "y la más oscura están en lugares distintos y la elección quedó "
            "cerca del límite, o se aplicó copa cerrada), 'cobertura' (menos "
            "del 70% de las hileras ubicadas tienen línea) o 'cuadros' (en parte "
            "del lote las hileras van a otra separación: dividilo en un polígono "
            "por cuadro de plantación). Coloreá la capa por "
            "ese campo y mirá esos lotes sobre la imagen.\n\n"
            "Hileras de borde y plantas chicas: las posiciones del marco de "
            "plantación se prueban en el múltiplo exacto de la separación, y un "
            "tramo de plantas chicas o ralas (más oscuro que la entrehilera, "
            "dentro del largo de las hileras vecinas) se conserva aunque su "
            "nivel promedio sea bajo.\n\n"
            "Hileras levemente torcidas: cada hilera se sigue de costado hasta "
            "0,3 separaciones (y no más de 0,6 m) para medirla aunque se "
            "aparte de su recta; la salida sigue siendo una recta por tramo.\n\n"
            "Importante: las líneas nunca salen del polígono del lote. Si el "
            "borde de cabecera pasa antes de la última planta, la hilera "
            "termina en ese borde: dibujá la cabecera sobre la calle.\n\n"
            "Contorno de la parcela o lotes (opcional): las líneas se "
            "recortan a estos polígonos, estrictamente a lo largo de las "
            "hileras. A lo ancho, una hilera dibujada sobre el borde del "
            "polígono (o hasta 0,4 separaciones afuera) es la hilera de borde "
            "del lote y se conserva entera; su entrehilera exterior se usa "
            "para medirla. El polígono puede tomar parte de la calle (hasta "
            "su eje): cada hilera igual termina donde terminan las plantas, y "
            "los tramos sin el nivel de una hilera (huellas de tractor, "
            "sombras) se descartan; sí conviene dejar afuera cortinas "
            "rompevientos y árboles en fila. Si no se da contorno, se usa el "
            "área con datos del raster. Con el método de perfil, cada polígono "
            "(uno por lote; también si hay uno solo) se procesa por separado, "
            "solo con sus píxeles, con su propia orientación, separación, "
            "alineación de hileras y numeración (campo 'lote'): así funcionan "
            "escenas con varios lotes, calles entre ellos o cultivos "
            "distintos (viñedo, frutal, hortaliza), y las calles entre lotes "
            "quedan afuera. Una hilera de borde tiene que estar en el marco de "
            "plantación (a un múltiplo de la separación de las hileras "
            "interiores): un cerco o una fila de árboles a lo largo del borde "
            "no se toma como hilera. "
            "'Extender hasta el contorno' (default 0) "
            "prolonga cada hilera hasta el borde del contorno si le faltan "
            "como máximo esos metros: usalo solo si el contorno es el del "
            "área plantada; con el contorno del lote, las líneas "
            "atravesarían las calles.\n\n"
            "MÉTODO 'Perfil de proyección' (default): pensado para hileras "
            "rectas y paralelas (viñedo, frutal, cultivos en línea). Mide "
            "la orientación y la separación de las hileras con la FFT 2D "
            "del ExG, ubica el centro de cada hilera con el perfil "
            "perpendicular, y a lo largo de cada hilera marca los tramos "
            "donde la hilera es más verde que las dos entrehileras vecinas "
            "(contraste local, independiente del vigor absoluto de la "
            "zona). Donde el verdor se debilita (vides de menor vigor, pasto "
            "en la entrehilera, plantas de borde) la hilera se sigue "
            "mientras continúe su patrón de brillo (franja más oscura o más "
            "clara que ambas entrehileras, con el mismo signo que en el resto "
            "de la hilera). El brillo también controla la posición: si en "
            "un lote la canopia no es verde en la imagen (sin hojas, de otoño, "
            "rojiza: la franja 'más verde' es solo la menos rojiza, a menudo el "
            "suelo), las hileras se ubican como las franjas oscuras (el log lo "
            "avisa); si el brillo es más fuerte pero la canopia es verde y la "
            "franja oscura está corrida de ella, la oscura es la sombra y se "
            "usa la más verde; y si en "
            "una franja del lote la entrehilera es más verde que la hilera "
            "(pasto entre hileras), los centros se reubican sobre la franja "
            "oscura. Como el marco de plantación es regular, también se "
            "prueba cada posición a un múltiplo de la separación de las "
            "hileras ubicadas, aunque el perfil no muestre ahí una hilera: "
            "así aparecen hileras de plantas chicas o ralas (típicamente la "
            "de borde); en una calle o en suelo desnudo no se marca nada.\n\n"
            "Otros cultivos: en un frutal de copa cerrada (sin suelo entre "
            "hileras) la franja más verde es la sombra entre copas; el "
            "complemento lo reconoce y ubica las hileras sobre las copas "
            "iluminadas (el log lo avisa; 'Franja de la hilera' permite "
            "forzar el criterio). Con hileras finas (hortalizas, < 1 m) la "
            "separación se busca en el espectro blanqueado (un manchón de "
            "vigor grande no la tapa) y, si las hileras están al límite de la "
            "resolución, se marcan donde el patrón de varias hileras vecinas "
            "es claro (modo patrón: posición correcta, extremos aproximados, "
            "sin fallas de plantas; el log lo avisa). "
            "Da una línea por tramo continuo de hilera, con "
            "el campo 'hilera' numerando cada hilera en orden. No usa bordes ni "
            "Hough, así que no sufre el problema de líneas que cruzan las "
            "hileras cuando éstas son angostas y están muy juntas. El log "
            "muestra el rumbo detectado, la separación entre hileras y "
            "cuántas se encontraron.\n\n"
            "  - 'Distancia entre hileras' (default -1 = automático): solo "
            "hace falta si el rumbo o la separación detectados en el log "
            "son incorrectos (p. ej. si la distancia entre plantas a lo "
            "largo de la hilera genera una periodicidad más fuerte que la "
            "de las hileras). Con un valor aproximado alcanza.\n"
            "  - 'Campo de distancia entre hileras por lote' (opcional): un "
            "campo numérico de la capa de lotes con la distancia de cada lote "
            "en metros; donde es > 0 reemplaza a la distancia general. Sirve "
            "para correr juntos cultivos de separación muy distinta.\n"
            "  - 'Franja de la hilera' (avanzado): Automático, Franja más "
            "verde (viñedos y frutales con suelo entre hileras), Copas "
            "iluminadas (frutal de copa cerrada) o Franja oscura (canopia poco "
            "verde o rojiza sobre suelo claro).\n"
            "  - 'Longitud mínima de línea' y 'Espacio longitudinal máximo' "
            "se usan también en este método: tramos más cortos que la "
            "longitud mínima se descartan, y huecos (plantas faltantes) de "
            "hasta el espacio longitudinal máximo se unen.\n"
            "  - 'Sensibilidad' (avanzado, default 3): umbral de contraste "
            "en múltiplos del ruido. Bajalo si faltan tramos de hileras "
            "débiles; subilo si aparecen tramos sobre cosas que no son "
            "hileras.\n"
            "Limitación: sin polígonos de lote, supone un único lote (una "
            "orientación y una alineación de hileras) por raster. Si la "
            "escena tiene varios lotes o cultivos, dibujá un polígono por "
            "lote en la capa de contorno.\n\n"
            "MÉTODO 'Canny + Hough' (anterior): binariza, detecta bordes con "
            "Canny y ajusta líneas rectas con la Transformada de Hough "
            "probabilística. Sus parámetros específicos están en "
            "'Parámetros avanzados'. Funciona con franjas de vegetación "
            "anchas y bien separadas; con hileras angostas y juntas "
            "(menos de ~15 px de período) tiende a detectar líneas que "
            "cruzan las hileras.\n\n"
            "Consejo (Hough): si las hileras salen muy fragmentadas, subí "
            "'Distancia máxima entre segmentos' (max_line_gap). Si aparecen "
            "líneas espurias, subí 'Umbral de Hough' o 'Longitud mínima'.\n\n"
            "Cierre morfológico (Hough, viñedo/frutal): si el cultivo son plantas "
            "individuales bien separadas entre sí (no una franja continua), "
            "Canny va a detectar el contorno de cada planta por separado, en "
            "todas direcciones, en vez de un borde de hilera continuo. Subir "
            "este parámetro (probá 3-7 px) une plantas cercanas en blobs más "
            "continuos antes de Canny y reduce ese ruido direccional.\n\n"
            "Filtro de dirección dominante (Hough): antes de fusionar, detecta "
            "automáticamente la orientación predominante entre todos los "
            "segmentos (ponderada por longitud) y descarta los que no estén "
            "cerca de esa dirección. Es clave cuando el cultivo tiene plantas "
            "individuales (viñedo/frutal): el contorno de cada planta genera "
            "segmentos en direcciones dispersas que, sin este filtro, "
            "contaminan la fusión y producen una maraña de líneas cruzadas "
            "en vez de hileras limpias. Si el mensaje de 'concentración' sale "
            "muy bajo, no hay una dirección clara — el ángulo detectado puede "
            "ser incorrecto; conviene revisar visualmente el resultado.\n\n"
            "Fusión de segmentos (Hough): con 'Fusionar segmentos' activado, los "
            "tramos de Hough que comparten ángulo (dentro de la tolerancia), "
            "están cerca perpendicularmente y cerca a lo largo de la hilera "
            "se combinan en una sola línea por hilera, ajustada por PCA y "
            "extendida al alcance real de todos sus segmentos.\n\n"
            "  - 'Distancia perpendicular máxima' (default -1 = automático): "
            "qué tan cerca de costado tienen que estar dos segmentos para "
            "fusionarse. El automático mide la separación real entre "
            "hileras vecinas a partir de los segmentos detectados y usa el "
            "40% de esa distancia — pensado para hileras muy juntas "
            "(viñedo/frutal de alta densidad), donde un valor fijo grande "
            "termina fusionando varias hileras vecinas en una sola.\n\n"
            "  - 'Espacio longitudinal máximo' (default 60 px): la "
            "salvaguarda más importante. Espacio máximo, medido A LO LARGO "
            "de la hilera (no de costado), entre el tramo ya fusionado y el "
            "próximo segmento candidato. Sin este chequeo, dos segmentos con "
            "ángulo y distancia perpendicular parecidos se fusionan aunque "
            "estén a cientos de píxeles de distancia entre sí a lo largo de "
            "la hilera — con un camino, un límite de parcela, o cualquier "
            "zona sin relación en el medio — incluso en un campo con una "
            "única dirección real consistente en toda la escena. Es la "
            "causa más común de líneas absurdamente largas que cruzan zonas "
            "sin relación entre sí. Subilo solo si tus hileras tienen huecos "
            "reales (plantas faltantes) más largos que el default.\n\n"
            "  - 'Longitud máxima de fusión' (default -1 = sin límite, en "
            "metros): tope duro adicional a cuánto puede crecer una hilera "
            "fusionada en total. Ya no suele hacer falta si el espacio "
            "longitudinal máximo está bien calibrado, pero sirve como límite "
            "absoluto extra si conocés la longitud real máxima de tus "
            "hileras.\n\n"
            "% de cobertura (opcional): mide, dentro de una franja cultivable "
            "alrededor de cada hilera (excluyendo calles/pasillos), qué "
            "porcentaje de esa franja es vegetación según la máscara ExG. "
            "El ancho de la franja puede ser automático (mitad de la distancia "
            "a la hilera vecina) o manual en metros; el automático no distingue "
            "calles de tránsito bien anchas, así que si las tenés, conviene "
            "poner el ancho real a mano.\n\n"
            "Conteo de plantas (opcional, aproximado): cuenta componentes "
            "conectados (manchas de vegetación) dentro de esa misma franja. "
            "Solo es confiable en etapas donde las plantas están separadas "
            "entre sí. Se agrega un campo de área promedio por planta como "
            "diagnóstico: si sale mucho más grande de lo esperado, es señal "
            "de que el dosel ya está cerrado y el conteo está subestimado "
            "(varias plantas fusionadas en un solo blob)."
        )

    def initAlgorithm(self, config=None):
        # Parámetros que solo usa el método Canny + Hough: van a "Parámetros
        # avanzados" para no recargar el diálogo con el método por defecto.
        hough_only = {
            self.MAX_LINE_GAP, self.HOUGH_THRESHOLD, self.CLOSING_PX,
            self.FILTER_ANGLE, self.ANGLE_FILTER_TOL, self.MERGE_LINES,
            self.MERGE_ANGLE_TOL, self.MERGE_DIST_TOL, self.MERGE_MAX_LENGTH,
        }
        advanced = hough_only | {self.SENSITIVITY, self.VEG_INDEX, self.STRIPE}

        def add(param):
            if param.name() in advanced:
                param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
            self.addParameter(param)

        add(
            QgsProcessingParameterRasterLayer(self.INPUT, self.tr("Raster de entrada (ortomosaico RGB)"))
        )
        add(
            QgsProcessingParameterBand(
                self.RED_BAND, self.tr("Banda Roja"), 1, self.INPUT
            )
        )
        add(
            QgsProcessingParameterBand(
                self.GREEN_BAND, self.tr("Banda Verde"), 2, self.INPUT
            )
        )
        add(
            QgsProcessingParameterBand(
                self.BLUE_BAND, self.tr("Banda Azul"), 3, self.INPUT
            )
        )
        add(
            QgsProcessingParameterEnum(
                self.METHOD,
                self.tr("Método de detección"),
                options=[
                    self.tr(
                        "Perfil de proyección orientado por FFT (hileras "
                        "rectas y paralelas: viñedo, frutal, cultivos en línea)"
                    ),
                    self.tr("Canny + Hough (método anterior)"),
                ],
                defaultValue=self.METHOD_PROFILE,
            )
        )
        add(
            QgsProcessingParameterFeatureSource(
                self.CONTOUR,
                self.tr(
                    "Contorno de la parcela o lotes (polígonos, opcional). Las "
                    "líneas se recortan estrictamente a este contorno. Cada "
                    "polígono (uno o varios) se procesa como un lote "
                    "independiente (método de perfil). Si no se da, se usa el "
                    "área con datos del raster (alfa/nodata)."
                ),
                [QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.EXTEND_MAX,
                self.tr(
                    "[Perfil] Extender cada hilera hasta el borde del contorno "
                    "si le faltan como máximo estos metros (default 0 = no "
                    "extender: la hilera termina donde termina lo plantado, "
                    "sin entrar en la calle de cabecera). Usalo solo si el "
                    "contorno es el del área plantada, no el del lote con "
                    "calles."
                ),
                QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.ROW_SPACING,
                self.tr(
                    "[Perfil] Distancia aproximada entre hileras, en metros "
                    "(-1 = automático por FFT). Solo si el automático se "
                    "equivoca: la búsqueda se acota a entre 0,7 y 1,4 veces este "
                    "valor, y se prefiere el patrón más fuerte a ±10%."
                ),
                QgsProcessingParameterNumber.Double,
                defaultValue=-1,
                minValue=-1,
            )
        )
        add(
            QgsProcessingParameterField(
                self.SPACING_FIELD,
                self.tr(
                    "[Perfil] Campo de la capa de contorno con la distancia entre "
                    "hileras de cada lote, en metros (opcional). En los lotes "
                    "donde el campo tiene un valor > 0, reemplaza a la distancia "
                    "de arriba; vacío o <= 0 = automático. Sirve para correr "
                    "juntos cultivos de separación muy distinta (frutal y "
                    "hortaliza)."
                ),
                parentLayerParameterName=self.CONTOUR,
                type=QgsProcessingParameterField.Numeric,
                optional=True,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.SENSITIVITY,
                self.tr(
                    "[Perfil] Umbral de contraste hilera/entrehilera, en "
                    "múltiplos del ruido (default 3). Más bajo = detecta "
                    "hileras más débiles, pero con más falsos positivos."
                ),
                QgsProcessingParameterNumber.Double,
                defaultValue=3.0,
                minValue=0.5,
                maxValue=20.0,
            )
        )
        add(
            QgsProcessingParameterEnum(
                self.STRIPE,
                self.tr("[Perfil] Franja de la hilera"),
                options=[
                    self.tr(
                        "Automático: la franja más verde; la oscura si la canopia no "
                        "es verde; las copas iluminadas en frutales de copa cerrada"
                    ),
                    self.tr(
                        "Franja más verde (verdor cromático): canopia oscura o con "
                        "su sombra; viñedos, frutales con suelo entre hileras"
                    ),
                    self.tr(
                        "Copas iluminadas (verde absoluto 2G-R-B): frutal de copa "
                        "cerrada, sin suelo visible entre hileras"
                    ),
                    self.tr(
                        "Franja oscura (brillo): canopia poco verde o rojiza (sin "
                        "hojas, otoño), sobre suelo claro"
                    ),
                ],
                defaultValue=self.STRIPE_AUTO,
            )
        )
        add(
            QgsProcessingParameterEnum(
                self.VEG_INDEX,
                self.tr("Índice de vegetación"),
                options=[
                    self.tr(
                        "ExG cromático: (2G-R-B)/(R+G+B) (Woebbecke et al. "
                        "1995; independiente del brillo)"
                    ),
                    self.tr("ExG sin normalizar: 2G-R-B (versiones <= 0.6)"),
                ],
                defaultValue=self.INDEX_EXG_CHROMATIC,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.THRESHOLD,
                self.tr("Umbral de binarización (-1 = automático, Otsu)"),
                QgsProcessingParameterNumber.Double,
                defaultValue=-1,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.MIN_LINE_LENGTH,
                self.tr("Longitud mínima de línea / tramo de hilera (píxeles)"),
                QgsProcessingParameterNumber.Integer,
                defaultValue=50,
                minValue=1,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.MAX_LINE_GAP,
                self.tr("[Hough] Distancia máxima entre segmentos para unirlos (píxeles)"),
                QgsProcessingParameterNumber.Integer,
                defaultValue=20,
                minValue=0,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.HOUGH_THRESHOLD,
                self.tr("[Hough] Umbral de Hough (votos mínimos para aceptar una línea)"),
                QgsProcessingParameterNumber.Integer,
                defaultValue=80,
                minValue=1,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.CLOSING_PX,
                self.tr(
                    "[Hough] Cierre morfológico antes de Canny, en píxeles (0 = "
                    "desactivado). Útil en viñedo/frutal: une plantas "
                    "individuales muy próximas en un blob más continuo, "
                    "antes de que Canny fragmente cada una por separado."
                ),
                QgsProcessingParameterNumber.Integer,
                defaultValue=0,
                minValue=0,
            )
        )
        add(
            QgsProcessingParameterBoolean(
                self.FILTER_ANGLE,
                self.tr(
                    "[Hough] Filtrar segmentos por dirección dominante antes de fusionar"
                ),
                defaultValue=True,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.ANGLE_FILTER_TOL,
                self.tr(
                    "[Hough] Tolerancia del filtro de dirección dominante (grados)"
                ),
                QgsProcessingParameterNumber.Double,
                defaultValue=15.0,
                minValue=0.5,
                maxValue=90.0,
            )
        )
        add(
            QgsProcessingParameterBoolean(
                self.MERGE_LINES,
                self.tr("[Hough] Fusionar segmentos en una sola línea por hilera"),
                defaultValue=True,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.MERGE_ANGLE_TOL,
                self.tr("[Hough] Tolerancia de ángulo para fusionar (grados)"),
                QgsProcessingParameterNumber.Double,
                defaultValue=5.0,
                minValue=0.1,
                maxValue=90.0,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.MERGE_DIST_TOL,
                self.tr(
                    "[Hough] Distancia perpendicular máxima para fusionar, en "
                    "píxeles (-1 = automático). El automático mide la "
                    "separación real entre hileras a partir de los "
                    "segmentos detectados y usa un 40% de esa distancia — "
                    "pensado para hileras muy juntas (viñedo/frutal de alta "
                    "densidad), donde un valor fijo grande termina "
                    "fusionando varias hileras vecinas en una sola. Si tus "
                    "hileras son muy onduladas (no rectas), el automático "
                    "puede ser demasiado estricto y fragmentarlas: en ese "
                    "caso poné un valor manual más alto."
                ),
                QgsProcessingParameterNumber.Double,
                defaultValue=-1,
                minValue=-1,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.MERGE_MAX_GAP,
                self.tr(
                    "Espacio longitudinal máximo, en píxeles a lo largo de "
                    "la hilera (default 60). Método de perfil: hueco máximo "
                    "(plantas faltantes) que se une dentro de una misma "
                    "hilera; huecos más largos cortan la hilera en dos "
                    "tramos. Método Hough: espacio máximo entre segmentos "
                    "para fusionarlos — la salvaguarda más importante: sin "
                    "ella, dos segmentos con ángulo y distancia perpendicular "
                    "parecidos se fusionan aunque estén a cientos de "
                    "píxeles de distancia entre sí a lo largo de la hilera "
                    "— con un camino, un límite de parcela o cualquier zona "
                    "sin relación en el medio — que es la causa más común "
                    "de líneas absurdamente largas cruzando toda la "
                    "imagen. Subí este valor solo si tus hileras tienen "
                    "huecos reales (plantas faltantes) más largos que el "
                    "default."
                ),
                QgsProcessingParameterNumber.Double,
                defaultValue=60,
                minValue=0,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.MERGE_MAX_LENGTH,
                self.tr(
                    "[Hough] Longitud máxima de una hilera fusionada, en metros "
                    "(-1 = sin límite). Salvaguarda adicional, en general "
                    "ya no hace falta si el 'espacio longitudinal máximo' "
                    "de arriba está funcionando bien — pero puede servir "
                    "como límite absoluto extra si conocés la longitud "
                    "real máxima de tus hileras."
                ),
                QgsProcessingParameterNumber.Double,
                defaultValue=-1,
            )
        )
        add(
            QgsProcessingParameterBoolean(
                self.CALC_COVERAGE,
                self.tr("Calcular % de cobertura de suelo (fracción cultivable)"),
                defaultValue=False,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.COVERAGE_WIDTH,
                self.tr(
                    "Ancho de franja cultivable por hilera, en metros "
                    "(-1 = automático, según distancia entre hileras vecinas)"
                ),
                QgsProcessingParameterNumber.Double,
                defaultValue=-1,
            )
        )
        add(
            QgsProcessingParameterBoolean(
                self.CALC_PLANT_COUNT,
                self.tr(
                    "Estimar número de plantas por hilera (aproximado: solo "
                    "confiable con plantas separadas, no con dosel cerrado)"
                ),
                defaultValue=False,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.MIN_PLANT_AREA,
                self.tr("Área mínima para contar como planta (m²), filtra ruido"),
                QgsProcessingParameterNumber.Double,
                defaultValue=0.01,
                minValue=0.0001,
            )
        )
        add(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT, self.tr("Hileras detectadas (líneas)")
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
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
        method = self.parameterAsEnum(parameters, self.METHOD, context)
        veg_index = self.parameterAsEnum(parameters, self.VEG_INDEX, context)
        contour_source = self.parameterAsSource(parameters, self.CONTOUR, context)
        extend_max_m = self.parameterAsDouble(parameters, self.EXTEND_MAX, context)
        row_spacing_m = self.parameterAsDouble(parameters, self.ROW_SPACING, context)
        sensitivity = self.parameterAsDouble(parameters, self.SENSITIVITY, context)
        threshold_val = self.parameterAsDouble(parameters, self.THRESHOLD, context)
        min_line_length = self.parameterAsInt(parameters, self.MIN_LINE_LENGTH, context)
        max_line_gap = self.parameterAsInt(parameters, self.MAX_LINE_GAP, context)
        hough_threshold = self.parameterAsInt(parameters, self.HOUGH_THRESHOLD, context)
        closing_px = self.parameterAsInt(parameters, self.CLOSING_PX, context)
        filter_angle = self.parameterAsBool(parameters, self.FILTER_ANGLE, context)
        angle_filter_tol = self.parameterAsDouble(parameters, self.ANGLE_FILTER_TOL, context)
        merge_lines = self.parameterAsBool(parameters, self.MERGE_LINES, context)
        merge_angle_tol = self.parameterAsDouble(parameters, self.MERGE_ANGLE_TOL, context)
        merge_dist_tol_param = self.parameterAsDouble(parameters, self.MERGE_DIST_TOL, context)
        merge_max_gap_param = self.parameterAsDouble(parameters, self.MERGE_MAX_GAP, context)
        merge_max_length_m = self.parameterAsDouble(parameters, self.MERGE_MAX_LENGTH, context)
        calc_coverage = self.parameterAsBool(parameters, self.CALC_COVERAGE, context)
        coverage_width_m = self.parameterAsDouble(parameters, self.COVERAGE_WIDTH, context)
        calc_plant_count = self.parameterAsBool(parameters, self.CALC_PLANT_COUNT, context)
        min_plant_area_m2 = self.parameterAsDouble(parameters, self.MIN_PLANT_AREA, context)
        spacing_fields = self.parameterAsFields(parameters, self.SPACING_FIELD, context) or []
        spacing_field = spacing_fields[0] if spacing_fields else None
        stripe_mode = self.parameterAsEnum(parameters, self.STRIPE, context)

        source_path = raster_layer.source()
        feedback.pushInfo(self.tr(f"Abriendo raster: {source_path}"))

        ds = gdal.Open(source_path)
        if ds is None:
            raise QgsProcessingException(self.tr("GDAL no pudo abrir el raster."))

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
        pixel_area_m2 = px_w * px_h
        raster_is_geographic = raster_layer.crs().isGeographic()

        def read_band(idx):
            band = ds.GetRasterBand(idx)
            if band is None:
                raise QgsProcessingException(
                    self.tr(f"No se pudo leer la banda {idx} del raster (el raster tiene {band_count} banda(s)).")
                )
            arr = band.ReadAsArray().astype(np.float32)
            return arr

        feedback.pushInfo(self.tr("Leyendo bandas R, G, B..."))
        r = read_band(red_idx)
        g = read_band(green_idx)
        b = read_band(blue_idx)

        # Píxeles válidos según GDAL (banda alfa, nodata, o todo válido si no
        # hay ninguno de los dos). Sin esto, el área transparente de un raster
        # recortado (RGB = 0) entra en el Otsu como si fuera suelo y corre el
        # umbral, y el borde del recorte aparece como un borde más.
        valid = np.ones(r.shape, dtype=bool)
        for idx in (red_idx, green_idx, blue_idx):
            valid &= ds.GetRasterBand(idx).GetMaskBand().ReadAsArray() > 0
        valid &= np.isfinite(r) & np.isfinite(g) & np.isfinite(b)
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
        if veg_index == self.INDEX_EXG_RAW:
            exg = 2.0 * g - r - b
        else:
            # Las bandas se suavizan levemente (sigma 1 px) antes de dividir:
            # en píxeles casi negros (canopia en sombra, 0-3 DN) la
            # cromaticidad está cuantizada y es puro ruido (p. ej. RGB 0,1,0
            # da ExG = 2), lo que moteaba la máscara de vegetación y
            # arrastraba el umbral de Otsu. Promediar con los vecinos antes
            # del cociente estabiliza el índice sin mover las hileras.
            rs, gs, bs = (cv2.GaussianBlur(x, (0, 0), 1.0) for x in (r, g, b))
            total = rs + gs + bs
            exg = (2.0 * gs - rs - bs) / np.maximum(total, 1e-6)
            exg[total <= 0] = 0.0
            del rs, gs, bs, total
        exg[~valid] = 0.0

        # Normalizar a 0-255 (uint8) para trabajar con OpenCV. Percentiles
        # en vez de mín/máx: en el índice cromático unos pocos píxeles casi
        # negros pueden dar valores extremos y comprimirían todo el rango.
        exg_min, exg_max = (float(v) for v in np.percentile(exg[valid], [0.5, 99.5]))
        if exg_max - exg_min < 1e-6:
            raise QgsProcessingException(
                self.tr("El índice ExG no tiene variación (imagen uniforme). Revisá las bandas elegidas.")
            )
        # La máscara binaria solo la usan el método Hough y los cálculos
        # opcionales de cobertura y conteo de plantas.
        binary = None
        if method == self.METHOD_HOUGH or calc_coverage or calc_plant_count:
            exg_norm = np.zeros(exg.shape, dtype=np.uint8)
            exg_norm[valid] = (np.clip((exg[valid] - exg_min) / (exg_max - exg_min), 0.0, 1.0) * 255.0).astype(np.uint8)
            feedback.pushInfo(self.tr("Binarizando (Otsu si corresponde)..."))
            if threshold_val is None or threshold_val < 0:
                # Otsu calculado solo sobre los píxeles válidos
                threshold_val, _ = cv2.threshold(
                    exg_norm[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            binary = np.where(valid & (exg_norm > threshold_val), 255, 0).astype(np.uint8)
            del exg_norm

        feedback.setProgress(45)
        if feedback.isCanceled():
            return {}

        row_ids = None
        row_period_px = None
        line_lots = None      # lote de cada línea (con varios polígonos de contorno)
        line_periods = None   # separación entre hileras de cada línea, en px
        line_flags = None     # motivos de revisión de cada línea (método de perfil)
        clipped_by_lot = False
        if method == self.METHOD_PROFILE:
            # Brillo como evidencia auxiliar de continuidad de la hilera
            brightness = (r + g + b) / 3.0
            brightness[~valid] = 0.0
            # verde absoluto (2G-R-B), para la franja de la hilera en frutales
            # de copa cerrada; con el índice sin normalizar ya es el índice
            rgb = (r, g, b) if veg_index != self.INDEX_EXG_RAW else None
            lots = (
                self._lot_geometries(contour_source, raster_layer.crs(), context, spacing_field)
                if contour_source is not None else []
            )
            if lots:
                # Cada polígono es un lote: su propia orientación, separación
                # y alineación de hileras (dos lotes uno detrás del otro no
                # tienen por qué tener las hileras alineadas), estimadas solo
                # con sus píxeles, y recortado a su polígono, así que las
                # calles entre lotes quedan afuera. También con un solo
                # polígono: si el raster tiene otros cultivos alrededor, no
                # entran en la estimación (antes sí: un frutal vecino fijaba el
                # rumbo de una hortaliza).
                if len(lots) >= 2:
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
        else:
            if closing_px and closing_px > 0:
                feedback.pushInfo(
                    self.tr(f"Aplicando cierre morfológico ({closing_px}px) antes de Canny...")
                )
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_px, closing_px))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            feedback.pushInfo(self.tr("Detectando bordes (Canny) y líneas (Hough)..."))
            edges = cv2.Canny(binary, 50, 150)

            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=hough_threshold,
                minLineLength=min_line_length,
                maxLineGap=max_line_gap,
            )

            feedback.setProgress(70)
            if feedback.isCanceled():
                return {}

            if lines is None:
                feedback.reportError(
                    self.tr(
                        "No se detectó ninguna línea. Probá bajar 'Umbral de Hough' "
                        "o 'Longitud mínima de línea', o revisar el umbral de binarización."
                    ),
                    fatalError=False,
                )
                lines = []
            else:
                lines = lines.reshape(-1, 4)

            # La dirección dominante se calcula siempre que haya segmentos (no
            # solo si el filtro de ángulo está activado), porque también hace
            # falta para auto-calibrar la distancia de fusión más abajo.
            dominant_angle, concentration = (None, 0.0)
            if len(lines) > 0:
                dominant_angle, concentration = estimate_dominant_angle_deg(lines)

            if len(lines) > 0 and filter_angle and dominant_angle is not None:
                n_before = len(lines)
                lines = filter_lines_by_angle(lines, dominant_angle, angle_filter_tol)
                feedback.pushInfo(
                    self.tr(
                        f"Dirección dominante detectada: {dominant_angle:.1f}° "
                        f"(concentración: {concentration:.2f}, 0=dispersa, 1=muy "
                        f"alineada). Segmentos: {n_before} -> {len(lines)} tras "
                        f"filtrar a ±{angle_filter_tol}°."
                    )
                )
                if concentration < 0.15:
                    feedback.reportError(
                        self.tr(
                            "Concentración muy baja: no hay una dirección "
                            "claramente dominante entre los segmentos detectados. "
                            "El ángulo elegido puede no corresponder a la hilera "
                            "real — revisá el resultado visualmente, o probá "
                            "activar el cierre morfológico y/o bajar el umbral "
                            "de Hough para que se detecten más bordes de hilera "
                            "reales antes de filtrar."
                        ),
                        fatalError=False,
                    )
                if len(lines) == 0:
                    feedback.reportError(
                        self.tr(
                            "El filtro de dirección dominante eliminó todos los "
                            "segmentos. Subí la tolerancia de ángulo o desactivá "
                            "el filtro."
                        ),
                        fatalError=False,
                    )

            if len(lines) > 0 and merge_lines:
                feedback.pushInfo(
                    self.tr(f"Fusionando {len(lines)} segmentos en líneas de hilera completas...")
                )
                if merge_dist_tol_param is not None and merge_dist_tol_param > 0:
                    merge_dist_tol = merge_dist_tol_param
                else:
                    merge_dist_tol = None
                    if dominant_angle is not None:
                        spacing_px = estimate_perp_spacing_px(lines, dominant_angle)
                        if spacing_px is not None and spacing_px > 0:
                            merge_dist_tol = spacing_px * 0.4
                            feedback.pushInfo(
                                self.tr(
                                    f"Distancia de fusión automática: separación "
                                    f"real entre hileras estimada en {spacing_px:.1f} px "
                                    f"-> tolerancia de fusión = {merge_dist_tol:.1f} px "
                                    "(40% de esa separación, para no fusionar hileras "
                                    "vecinas entre sí)."
                                )
                            )
                    if merge_dist_tol is None:
                        merge_dist_tol = 30.0
                        feedback.reportError(
                            self.tr(
                                "No se pudo estimar automáticamente la separación "
                                "entre hileras (pocos segmentos o sin dirección "
                                "clara). Usando 30 px por defecto — si tus hileras "
                                "están muy juntas, esto puede fusionar de más: "
                                "poné un valor manual más chico."
                            ),
                            fatalError=False,
                        )

                max_gap_px = merge_max_gap_param if merge_max_gap_param is not None and merge_max_gap_param > 0 else None

                if merge_max_length_m is not None and merge_max_length_m > 0:
                    if raster_is_geographic:
                        feedback.reportError(
                            self.tr(
                                "El raster está en un CRS geográfico (grados, no metros): la "
                                "'Longitud máxima de fusión' en metros no se puede convertir "
                                "correctamente a píxeles. Reproyectá el raster a un CRS proyectado."
                            ),
                            fatalError=False,
                        )
                    max_length_px = merge_max_length_m / pixel_size_avg
                else:
                    max_length_px = None

                lines = merge_segments_into_rows(
                    lines, merge_angle_tol, merge_dist_tol, max_length_px, max_gap_px
                )
                feedback.pushInfo(self.tr(f"Resultado: {len(lines)} hileras."))

        # --- Recorte estricto al contorno de la parcela (y, en el método de
        # perfil, extensión de cada hilera hasta el borde del contorno) ---
        if feedback.isCanceled():
            return {}
        contour_geom = None
        if not clipped_by_lot:
            contour_geom = self._contour_geometry(
                contour_source, raster_layer.crs(), context, gdal, valid, geotransform, feedback
            )
        if contour_geom is not None and len(lines) > 0:
            ext_px = (
                extend_max_m / pixel_size_avg
                if method == self.METHOD_PROFILE and extend_max_m is not None and extend_max_m > 0
                else 0.0
            )
            len_before = sum(math.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in lines) * pixel_size_avg
            n_before = len(lines)
            lines, row_ids = self._fit_lines_to_contour(
                gdal, lines, row_ids, contour_geom, geotransform, ext_px,
                valid.shape, min_line_length,
                cross_tol_px=(self.ROW_TOL * row_period_px
                              if method == self.METHOD_PROFILE and row_period_px else 0.0),
                period_px=row_period_px if method == self.METHOD_PROFILE else None,
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

        # --- Preparar cálculo de cobertura / conteo de plantas, si corresponde ---
        half_width_px = None
        half_widths = None  # por línea, cuando cada lote tiene su separación
        min_plant_area_px = None

        if calc_coverage or calc_plant_count:
            if raster_is_geographic:
                feedback.reportError(
                    self.tr(
                        "El raster está en un CRS geográfico (grados, no metros). "
                        "Los cálculos de cobertura/conteo van a ser incorrectos: "
                        "reproyectá el raster a un CRS proyectado (metros) antes de correr esto."
                    ),
                    fatalError=False,
                )

            if (coverage_width_m is None or coverage_width_m < 0) and line_periods is not None:
                half_widths = [p / 2.0 for p in line_periods]
                feedback.pushInfo(
                    self.tr(
                        "Ancho automático de franja cultivable: media separación entre "
                        "hileras de cada lote, a cada lado de la hilera."
                    )
                )
            elif coverage_width_m is None or coverage_width_m < 0:
                if row_period_px is not None:
                    spacing_px = row_period_px  # método de perfil: medido por FFT
                else:
                    spacing_px = estimate_row_spacing_px(lines)
                if spacing_px is None:
                    feedback.reportError(
                        self.tr(
                            "No se pudo estimar el ancho automático de la franja "
                            "cultivable (se necesitan al menos 2 hileras detectadas). "
                            "Especificá un ancho manual en metros, o revisá la detección."
                        ),
                        fatalError=False,
                    )
                else:
                    half_width_px = spacing_px / 2.0
                    feedback.pushInfo(
                        self.tr(
                            f"Ancho automático de franja cultivable: "
                            f"{half_width_px * pixel_size_avg:.2f} m a cada lado de la hilera "
                            f"(según espaciado detectado entre hileras)."
                        )
                    )
            else:
                half_width_px = coverage_width_m / pixel_size_avg
                feedback.pushInfo(
                    self.tr(f"Ancho manual de franja cultivable: {coverage_width_m} m a cada lado de la hilera.")
                )

            min_plant_area_px = min_plant_area_m2 / pixel_area_m2

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
        if calc_coverage:
            fields.append(QgsField("cobertura_pct", QVariant.Double))
        if calc_plant_count:
            fields.append(QgsField("num_plantas_est", QVariant.Int))
            fields.append(QgsField("area_prom_planta_m2", QVariant.Double))

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
        coberturas = []
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

            if calc_coverage or calc_plant_count:
                hw = half_widths[i] if half_widths is not None else half_width_px
                if hw is not None:
                    row_stats = compute_row_coverage_and_count(
                        cv2, binary, x1, y1, x2, y2,
                        hw, min_plant_area_px, pixel_area_m2,
                    )
                else:
                    row_stats = {"cobertura_pct": None, "num_plantas": None, "area_prom_m2": None}

                if calc_coverage:
                    attrs.append(row_stats["cobertura_pct"])
                    if row_stats["cobertura_pct"] is not None:
                        coberturas.append(row_stats["cobertura_pct"])
                if calc_plant_count:
                    attrs.append(row_stats["num_plantas"])
                    attrs.append(row_stats["area_prom_m2"])

            feat = QgsFeature(fields)
            feat.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(mx1, my1), QgsPointXY(mx2, my2)]))
            feat.setAttributes(attrs)
            sink.addFeature(feat)

        if calc_coverage and coberturas:
            feedback.pushInfo(
                self.tr(f"Cobertura promedio de todas las hileras: {sum(coberturas) / len(coberturas):.1f}%")
            )

        feedback.setProgress(100)
        ds = None  # cerrar dataset de GDAL

        return {self.OUTPUT: dest_id}

    def _contour_geometry(self, source, raster_crs, context, gdal, valid, geotransform, feedback):
        """Geometría del contorno de la parcela, en el CRS del raster: la capa
        de contorno si se dio (unión de todos sus polígonos), o si no el área
        con datos del raster (máscara alfa/nodata vectorizada). None si no
        hay contorno ni píxeles sin datos (nada que recortar)."""
        if source is not None:
            request = QgsFeatureRequest().setDestinationCrs(raster_crs, context.transformContext())
            geoms = [f.geometry() for f in source.getFeatures(request) if f.hasGeometry()]
            if not geoms:
                raise QgsProcessingException(self.tr("La capa de contorno no tiene geometrías."))
            geom = QgsGeometry.unaryUnion(geoms)
            if not geom.isGeosValid():
                geom = geom.makeValid()
            feedback.pushInfo(
                self.tr(
                    f"Contorno de la parcela: {len(geoms)} polígono(s), "
                    f"{geom.area() / 10000.0:.2f} ha."
                )
            )
            return geom

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
            geom = f.geometry()
            if not geom.isGeosValid():
                geom = geom.makeValid()
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
        out = QgsGeometry.unaryUnion(parts)
        if not out.isGeosValid():
            out = out.makeValid()
        return out

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
            except QgsProcessingException as err:
                feedback.reportError(self.tr(f"Lote {lot_id}: {err}"), fatalError=False)
                continue
            if res is None:
                return None
            l_lines, l_ids, period = res
            l_lines = [(x1 + c0, y1 + r0, x2 + c0, y2 + r0) for x1, y1, x2, y2 in l_lines]
            if l_lines:
                l_lines, l_ids = self._fit_lines_to_contour(
                    gdal, l_lines, l_ids, geom, gt, ext_px, valid.shape, min_line_length,
                    cross_tol_px=self.ROW_TOL * period, period_px=period,
                )
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
        fit_intervals_to_contour). Con row_ids (método de perfil) los tramos
        de una misma hilera se tratan juntos; sin row_ids (Hough) cada línea
        por separado. Con cross_tol_px > 0 (método de perfil), una hilera de
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
                    "No se pudo estimar la orientación de las hileras (raster "
                    "demasiado chico o sin suficientes píxeles válidos)."
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
                    "separación real es otra, indicala en 'Distancia aproximada entre "
                    "hileras'."
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
        if green_fn is not None and aux is not None and stripe_mode == self.STRIPE_AUTO:
            dark_now = not aux_is_brightness
            c_w, y_w = (aux, -work) if dark_now else (work, aux)
            so = stripe_offset(c_w, y_w, work_valid, angle, period_w)
            if so is not None:
                c_pk, off, off_sgn = so
                if not dark_now and c_pk < self.C_NOT_GREEN:
                    work, aux = -y_w, c_w
                    aux_is_brightness = False
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
                                "hileras, elegí 'Franja más verde' en 'Franja de la hilera'."
                            )
                        )
                        doubts.append((
                            "franja",
                            "se aplicó copa cerrada (líneas sobre las copas al sol); si la "
                            "franja clara es pasto entre hileras, elegí 'Franja más verde'",
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
                    "aproximada entre hileras'."
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
        )
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
                    f"reajustados), {info.get('descartados_por_nivel', 0)} tramos "
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
