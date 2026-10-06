# -*- coding: utf-8 -*-
"""
Algoritmo de Processing: perfil de vegetación por hilera y detección de fallas.
El método está en perfil.py; los cuarteles se infieren con cuarteles.py.
"""
import numpy as np

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterBand,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingParameterBoolean,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsWkbTypes,
    Qgis,
)
from qgis.PyQt.QtCore import QVariant

from .cuarteles import infer_blocks
from .perfil import analyze_block, Tile, INDEX_NAMES

# por debajo de este contraste el perfil transversal casi no tiene pico
MIN_CONTRAST = 0.8


class RowProfileAlgorithm(QgsProcessingAlgorithm):

    INPUT = "INPUT"
    LINES = "LINES"
    RED_BAND = "RED_BAND"
    GREEN_BAND = "GREEN_BAND"
    BLUE_BAND = "BLUE_BAND"
    STEP = "STEP"
    INDEX = "INDEX"
    WIDTH = "WIDTH"
    TOLERANCE = "TOLERANCE"
    THRESHOLD = "THRESHOLD"
    MIN_GAP = "MIN_GAP"
    WINDOW = "WINDOW"
    BLOCK_GAP = "BLOCK_GAP"
    ADJUST = "ADJUST"
    N_BREAKS = "N_BREAKS"
    SHADOW = "SHADOW"
    KINK_DEG = "KINK_DEG"
    OUT_LINES = "OUT_LINES"
    OUT_BUFFER = "OUT_BUFFER"
    OUT_PROFILE = "OUT_PROFILE"
    OUT_GAPS = "OUT_GAPS"
    OUT_TRANSECTS = "OUT_TRANSECTS"

    def name(self):
        return "fallas_por_hilera"

    def displayName(self):
        return "Perfil de vegetación y fallas por hilera"

    def group(self):
        return ""

    def groupId(self):
        return ""

    def createInstance(self):
        return RowProfileAlgorithm()

    def shortHelpString(self):
        return (
            "Mide la vegetación a lo largo de cada hilera y marca los tramos donde faltan plantas.\n\n"
            "1) Se infieren los cuarteles a partir de las hileras. En cada uno se trazan transectas "
            "perpendiculares a las hileras cada N metros y se pliegan por la distancia a la hilera "
            "más cercana: el promedio da el perfil típico de una hilera (pico de vegetación, valle de "
            "entrehilera).\n"
            "2) Con ese perfil se elige el índice de verdor que mejor separa hilera de entrehilera "
            "(el raster es solo RGB, no hay NDVI), y el ancho de la vegetación es el ancho del pico al "
            "60 % de su altura; la maleza de la entrehilera queda afuera. El pico puede estar corrido "
            "de la línea (p. ej. si la línea cayó sobre la sombra); el análisis sigue al pico.\n"
            "3) A lo largo de cada hilera se promedia el índice en una franja de ese ancho y se "
            "normaliza entre la entrehilera (0) y una planta típica del cuartel (1). Como la línea puede "
            "estar corrida de la planta unos centímetros, en cada punto se toma la posición mejor alineada "
            "dentro de una tolerancia lateral (por defecto 0,2 de la separación entre hileras): un pequeño "
            "desalineamiento no se confunde con una falla. Además cada hilera se reajusta sola: en tramos "
            "de 8 m se busca dentro de esa tolerancia el pico de vegetación y, si es claro, la hilera se "
            "corre hacia él (si no hay planta no hay pico y no se la mueve) y el buffer se vuelve a tomar "
            "alrededor de la línea ya corrida, hasta que el pico quede centrado, así se alcanzan también "
            "hileras cuya línea quedó a más de la tolerancia de la planta. Con un máximo de quiebres, cada hilera "
            "ajustada es una recta con a lo sumo ese número de cambios leves de dirección (poligonal por mínimos "
            "cuadrados): líneas más limpias y menos sensibles al ruido, a costa de seguir menos fielmente una "
            "hilera muy irregular. Las salidas opcionales "
            "'Hileras ajustadas' y 'Buffer' tienen las líneas corregidas y la franja medida. Una falla es un tramo continuo con vigor menor "
            "que el umbral y de al menos el largo mínimo.\n\n"
            "Si en un cuartel el perfil casi no tiene contraste, las líneas probablemente no siguen "
            "las plantas (p. ej. el cuartel tiene otra distancia entre hileras): volvé a detectar las "
            "hileras en ese cuartel con su distancia real antes de interpretar las fallas.\n\n"
            "El raster y las hileras tienen que estar en el mismo CRS proyectado."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, "Ortomosaico RGB"))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.LINES, "Hileras detectadas (líneas)", [QgsProcessing.TypeVectorLine]))
        self.addParameter(QgsProcessingParameterBand(
            self.RED_BAND, "Banda roja", 1, self.INPUT))
        self.addParameter(QgsProcessingParameterBand(
            self.GREEN_BAND, "Banda verde", 2, self.INPUT))
        self.addParameter(QgsProcessingParameterBand(
            self.BLUE_BAND, "Banda azul", 3, self.INPUT))
        self.addParameter(QgsProcessingParameterNumber(
            self.STEP, "Distancia entre transectas (m)",
            QgsProcessingParameterNumber.Double, 10.0, minValue=1.0))
        self.addParameter(QgsProcessingParameterEnum(
            self.INDEX, "Índice de vegetación",
            ["Automático (el de mayor contraste entre ExG, VARI, GLI, NGRDI)"] + INDEX_NAMES,
            defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(
            self.WIDTH, "Ancho de la vegetación (m, -1 = automático)",
            QgsProcessingParameterNumber.Double, -1.0))
        self.addParameter(QgsProcessingParameterNumber(
            self.TOLERANCE, "Tolerancia de alineación de la línea con la planta (m, -1 = 0,2 de la separación)",
            QgsProcessingParameterNumber.Double, -1.0))
        self.addParameter(QgsProcessingParameterBoolean(
            self.ADJUST, "Reajustar cada hilera al pico de vegetación dentro de la tolerancia",
            defaultValue=True))
        self.addParameter(QgsProcessingParameterNumber(
            self.N_BREAKS, "Máximo de quiebres por hilera al reajustar (-1 = libre, sin límite; 0 = recta)",
            QgsProcessingParameterNumber.Integer, 4, minValue=-1, maxValue=8))
        self.addParameter(QgsProcessingParameterNumber(
            self.SHADOW, "Brillo máximo de un píxel de sombra (0-255): se excluye del muestreo, porque ahí los índices de verdor son ruido",
            QgsProcessingParameterNumber.Double, 30.0, minValue=0.0, maxValue=120.0))
        self.addParameter(QgsProcessingParameterNumber(
            self.KINK_DEG, "Cambio máximo de dirección en cada quiebre (grados)",
            QgsProcessingParameterNumber.Double, 3.0, minValue=0.1, maxValue=15.0))
        self.addParameter(QgsProcessingParameterNumber(
            self.THRESHOLD, "Umbral de falla (0 = entrehilera, 1 = planta típica)",
            QgsProcessingParameterNumber.Double, 0.5, minValue=0.05, maxValue=0.95))
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_GAP, "Largo mínimo de una falla (m)",
            QgsProcessingParameterNumber.Double, 2.0, minValue=0.1))
        self.addParameter(QgsProcessingParameterNumber(
            self.WINDOW, "Ventana del perfil a lo largo de la hilera (m)",
            QgsProcessingParameterNumber.Double, 1.0, minValue=0.1))
        self.addParameter(QgsProcessingParameterNumber(
            self.BLOCK_GAP, "Calle mínima que separa cuarteles (m)",
            QgsProcessingParameterNumber.Double, 3.0, minValue=0.5))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_LINES, "Hileras ajustadas a la vegetación (líneas)", QgsProcessing.TypeVectorLine,
            optional=True, createByDefault=False))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_BUFFER, "Buffer de la franja de vegetación alrededor de la hilera ajustada (polígonos)",
            QgsProcessing.TypeVectorPolygon, optional=True, createByDefault=False))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_PROFILE, "Perfil por ventana (líneas)", QgsProcessing.TypeVectorLine))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_GAPS, "Fallas (líneas)", QgsProcessing.TypeVectorLine))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_TRANSECTS, "Transectas (líneas)", QgsProcessing.TypeVectorLine,
            optional=True, createByDefault=False))

    # ------------------------------------------------------------------
    @staticmethod
    def _open_tile(ds, gt, b, bands, margin=3.0):
        xs = [q[0] for q in b["ring"]]
        ys = [q[1] for q in b["ring"]]
        x0, x1 = min(xs) - margin, max(xs) + margin
        y0, y1 = max(ys) + margin, min(ys) - margin
        c0 = max(0, int((x0 - gt[0]) / gt[1]))
        c1 = min(ds.RasterXSize, int((x1 - gt[0]) / gt[1]) + 1)
        r0 = max(0, int((y0 - gt[3]) / gt[5]))
        r1 = min(ds.RasterYSize, int((y1 - gt[3]) / gt[5]) + 1)
        if c1 <= c0 or r1 <= r0:
            return None
        from osgeo import gdal
        planes = []
        for bn in bands:
            planes.append(ds.GetRasterBand(bn).ReadAsArray(c0, r0, c1 - c0, r1 - r0))
        valid = np.ones(planes[0].shape, bool)
        for i in range(1, ds.RasterCount + 1):
            bd = ds.GetRasterBand(i)
            if bd.GetColorInterpretation() == gdal.GCI_AlphaBand:
                valid &= bd.ReadAsArray(c0, r0, c1 - c0, r1 - r0) > 0
        nd = ds.GetRasterBand(bands[0]).GetNoDataValue()
        if nd is not None:
            valid &= planes[0] != nd
        return Tile(planes[0], planes[1], planes[2], valid,
                    gt[0] + c0 * gt[1], gt[3] + r0 * gt[5], gt[1], -gt[5])

    def processAlgorithm(self, parameters, context, feedback):
        from osgeo import gdal
        raster = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        source = self.parameterAsSource(parameters, self.LINES, context)
        if raster is None:
            raise QgsProcessingException(self.invalidRasterError(parameters, self.INPUT))
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.LINES))
        if source.sourceCrs().isGeographic():
            raise QgsProcessingException(
                "Las hileras están en un CRS geográfico; reproyectalas a uno en metros (UTM).")
        if source.sourceCrs() != raster.crs():
            raise QgsProcessingException(
                "El raster (%s) y las hileras (%s) tienen distinto CRS; reproyectá uno de los dos."
                % (raster.crs().authid(), source.sourceCrs().authid()))
        bands = [self.parameterAsInt(parameters, k, context)
                 for k in (self.RED_BAND, self.GREEN_BAND, self.BLUE_BAND)]
        step = self.parameterAsDouble(parameters, self.STEP, context)
        idx_i = self.parameterAsEnum(parameters, self.INDEX, context)
        index = "auto" if idx_i == 0 else INDEX_NAMES[idx_i - 1]
        width = self.parameterAsDouble(parameters, self.WIDTH, context)
        tol = self.parameterAsDouble(parameters, self.TOLERANCE, context)
        adjust = self.parameterAsBool(parameters, self.ADJUST, context)
        n_breaks = self.parameterAsInt(parameters, self.N_BREAKS, context)
        shadow = self.parameterAsDouble(parameters, self.SHADOW, context)
        kink_deg = self.parameterAsDouble(parameters, self.KINK_DEG, context)
        umbral = self.parameterAsDouble(parameters, self.THRESHOLD, context)
        min_gap = self.parameterAsDouble(parameters, self.MIN_GAP, context)
        win_m = self.parameterAsDouble(parameters, self.WINDOW, context)
        block_gap = self.parameterAsDouble(parameters, self.BLOCK_GAP, context)

        ds = gdal.Open(raster.dataProvider().dataSourceUri().split("|")[0])
        if ds is None:
            raise QgsProcessingException("No se pudo abrir el raster con GDAL.")
        gt = ds.GetGeoTransform()
        if gt[2] != 0 or gt[4] != 0:
            raise QgsProcessingException("El raster está rotado; no se admite.")

        segs = []
        for f in source.getFeatures():
            g = f.geometry()
            if g.isNull() or g.isEmpty():
                continue
            parts = g.asMultiPolyline() if g.isMultipart() else [g.asPolyline()]
            for pl in parts:
                if len(pl) >= 2:
                    segs.append(((pl[0].x(), pl[0].y()), (pl[-1].x(), pl[-1].y())))
        if not segs:
            raise QgsProcessingException("La capa de hileras no tiene líneas.")
        blocks, info = infer_blocks(segs, gap=block_gap)
        if not blocks:
            raise QgsProcessingException("No se encontró ningún cuartel en las hileras.")
        feedback.pushInfo("%d cuartel(es) inferidos a partir de %d tramos." % (len(blocks), len(segs)))

        f_prof = QgsFields()
        for nm, tp in (("cuartel", QVariant.Int), ("hilera", QVariant.Int),
                       ("ventana", QVariant.Int), ("pos_m", QVariant.Double),
                       ("vigor", QVariant.Double), ("presencia_pct", QVariant.Double),
                       ("falla", QVariant.Int)):
            f_prof.append(QgsField(nm, tp))
        f_gap = QgsFields()
        for nm, tp in (("cuartel", QVariant.Int), ("hilera", QVariant.Int),
                       ("ini_m", QVariant.Double), ("fin_m", QVariant.Double),
                       ("largo_m", QVariant.Double), ("vigor", QVariant.Double),
                       ("borde", QVariant.Int)):
            f_gap.append(QgsField(nm, tp))
        f_tr = QgsFields()
        f_tr.append(QgsField("cuartel", QVariant.Int))
        crs = source.sourceCrs()
        sink_p, dest_p = self.parameterAsSink(
            parameters, self.OUT_PROFILE, context, f_prof, QgsWkbTypes.LineString, crs)
        sink_g, dest_g = self.parameterAsSink(
            parameters, self.OUT_GAPS, context, f_gap, QgsWkbTypes.LineString, crs)
        f_adj = QgsFields()
        for nm, tp in (("cuartel", QVariant.Int), ("hilera", QVariant.Int),
                       ("desplaz_med_m", QVariant.Double)):
            f_adj.append(QgsField(nm, tp))
        sink_a, dest_a = self.parameterAsSink(
            parameters, self.OUT_LINES, context, f_adj, QgsWkbTypes.LineString, crs)
        f_buf = QgsFields()
        for nm, tp in (("cuartel", QVariant.Int), ("hilera", QVariant.Int), ("ancho_m", QVariant.Double)):
            f_buf.append(QgsField(nm, tp))
        sink_b, dest_b = self.parameterAsSink(
            parameters, self.OUT_BUFFER, context, f_buf, QgsWkbTypes.Polygon, crs)
        sink_t, dest_t = self.parameterAsSink(
            parameters, self.OUT_TRANSECTS, context, f_tr, QgsWkbTypes.LineString, crs)

        def line(x1, y1, x2, y2):
            return QgsGeometry.fromPolylineXY([QgsPointXY(x1, y1), QgsPointXY(x2, y2)])

        results = {self.OUT_PROFILE: dest_p, self.OUT_GAPS: dest_g}
        if dest_t:
            results[self.OUT_TRANSECTS] = dest_t
        if dest_a:
            results[self.OUT_LINES] = dest_a
        if dest_b:
            results[self.OUT_BUFFER] = dest_b
        for n, b in enumerate(blocks, 1):
            if feedback.isCanceled():
                break
            feedback.setProgress(100.0 * (n - 1) / len(blocks))
            tile = self._open_tile(ds, gt, b, bands)
            if tile is None:
                feedback.reportError("Cuartel %d: fuera del raster." % n)
                continue
            r = analyze_block(b, info, tile, step=step, win_m=win_m, umbral=umbral,
                              index=index, width=width if width > 0 else None,
                              min_falla=min_gap, tol=tol if tol >= 0 else None, adjust=adjust,
                              n_breaks=n_breaks if n_breaks >= 0 else None, kink_deg=kink_deg,
                              shadow_max=shadow)
            if "error" in r:
                feedback.reportError("Cuartel %d: %s." % (n, r["error"]))
                continue
            w = r["win"]
            for j in range(len(w["row"])):
                ft = QgsFeature(f_prof)
                ft.setGeometry(line(w["x1"][j], w["y1"][j], w["x2"][j], w["y2"][j]))
                z, pr = w["z"][j], w["pres"][j]
                ft.setAttributes([
                    n, int(w["row"][j]) + 1, int(w["i"][j]),
                    round(float(w["ua"][j] - b["row_lo"][w["row"][j]]), 2),
                    None if np.isnan(z) else round(float(z), 3),
                    None if np.isnan(pr) else round(float(pr), 1),
                    int(w["falla"][j])])
                sink_p.addFeature(ft, QgsFeatureSink.FastInsert)
            for g in r["fallas"]:
                ft = QgsFeature(f_gap)
                ft.setGeometry(line(g["x1"], g["y1"], g["x2"], g["y2"]))
                ft.setAttributes([
                    n, g["row"] + 1, round(g["ini_m"], 2), round(g["fin_m"], 2),
                    round(g["largo"], 2), round(g["z"], 3), g["borde"]])
                sink_g.addFeature(ft, QgsFeatureSink.FastInsert)
            if sink_b:
                half = r["width"] / 2.0 + r["resid"]
                for a_ in r["adjusted"]:
                    ft = QgsFeature(f_buf)
                    ft.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(x, y) for x, y in a_["pts"]]).buffer(
                        half, 4, Qgis.EndCapStyle.Flat, Qgis.JoinStyle.Round, 2.0))
                    ft.setAttributes([n, a_["row"] + 1, round(2 * half, 2)])
                    sink_b.addFeature(ft, QgsFeatureSink.FastInsert)
            if sink_a:
                for a_ in r["adjusted"]:
                    ft = QgsFeature(f_adj)
                    ft.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(x, y) for x, y in a_["pts"]]))
                    ft.setAttributes([n, a_["row"] + 1, round(a_["dmed"], 3)])
                    sink_a.addFeature(ft, QgsFeatureSink.FastInsert)
            if sink_t:
                for (a, c) in r["transects"]:
                    ft = QgsFeature(f_tr)
                    ft.setGeometry(line(a[0], a[1], c[0], c[1]))
                    ft.setAttributes([n])
                    sink_t.addFeature(ft, QgsFeatureSink.FastInsert)

            feedback.pushInfo(
                "Cuartel %d: índice %s (contraste %.2f), vegetación de %.2f m de ancho "
                "corrida %+.2f m de la línea (tolerancia de alineación ±%.2f m); %d fallas, %.1f%% del largo de hileras "
                "(%.0f m de %.0f m)." % (
                    n, r["index"], r["contrast"], r["width"], r["offset"], r["tol"],
                    len(r["fallas"]), r["falla_pct"], r["falla_m"], r["row_m"]))
            ad = r["adj"]
            if ad["adjust"]:
                feedback.pushInfo(
                    "Cuartel %d: ajuste individual de las hileras: pico claro en %d de %d tramos de 8 m; "
                    "%d de %d hileras se corrieron más de 10 cm (desplazamiento mediano %.2f m, "
                    "%.1f reajustes en promedio, con el buffer retomado alrededor de la línea ya corrida; "
                    "el pico se busca hasta ±%.2f m de la línea%s)." % (
                        n, ad["con_pico"], ad["tramos"], ad["filas_movidas"], r["n_rows"], ad["desp_med"],
                        ad["iter_med"], r["search"],
                        ("; %d quiebres en total, hasta %d por hilera, de a lo sumo %.1f°" % (
                            ad["quiebres"], n_breaks, kink_deg)) if n_breaks >= 0 else "; línea libre, un vértice cada 2 m"))
            if r["contrast"] < MIN_CONTRAST:
                feedback.reportError(
                    "Cuartel %d: el perfil transversal casi no tiene pico (contraste %.2f). "
                    "Las líneas probablemente no siguen las plantas (¿otra distancia entre "
                    "hileras en este cuartel?); no interpretes sus fallas hasta rehacer las "
                    "hileras." % (n, r["contrast"]))
        return results
