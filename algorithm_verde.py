# -*- coding: utf-8 -*-
"""
Algoritmo de Processing: hileras y fallas por cuartel, basado en el verdor.
El método está en verde.py.
"""
import math

import numpy as np

from qgis.core import (
    Qgis,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBand,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant

from . import verde as V


class HilerasFallasVerdeAlgorithm(QgsProcessingAlgorithm):

    INPUT = "INPUT"
    RED_BAND = "RED_BAND"
    GREEN_BAND = "GREEN_BAND"
    BLUE_BAND = "BLUE_BAND"
    NIR_BAND = "NIR_BAND"
    DELIM = "DELIM"
    N_BREAKS = "N_BREAKS"
    KINK_DEG = "KINK_DEG"
    WINDOW = "WINDOW"
    SEARCH = "SEARCH"
    RESID = "RESID"
    THRESHOLD = "THRESHOLD"
    MIN_GAP = "MIN_GAP"
    OUT_BLOCKS = "OUT_BLOCKS"
    OUT_INIT = "OUT_INIT"
    OUT_ADJ = "OUT_ADJ"
    OUT_BUFFER = "OUT_BUFFER"
    OUT_GAPS = "OUT_GAPS"
    OUT_PEAKS = "OUT_PEAKS"

    def name(self):
        return "hileras_y_fallas_verde"

    def displayName(self):
        return "Hileras y fallas por cuartel (basado en verde)"

    def group(self):
        return ""

    def groupId(self):
        return ""

    def createInstance(self):
        return HilerasFallasVerdeAlgorithm()

    def shortHelpString(self):
        return (
            "Delimita los cuarteles de un ortomosaico, detecta las hileras de cada uno y marca las "
            "fallas (tramos sin planta), todo a partir del verdor. Puede tardar un par de minutos en una finca de ~13 ha.\n\n"
            "1) Delimitación de cuarteles: zonas con patrón de hileras (el verdor sin patrón, como monte o "
            "pasto, no es cuartel).\n"
            "2) Rumbo y distancia entre hileras de cada cuartel, por separado, con el índice de verdor que "
            "mejor lo marca (ExG, VARI, GLI o NGRDI; NDVI si se da una banda infrarroja).\n"
            "3) Líneas iniciales: una recta por hilera.\n"
            "4) Primer buffer alrededor de cada línea y perfiles transversales de verdor cada 2 m: el "
            "pico de cada perfil dice dónde está la planta.\n"
            "5) Líneas nuevas con un número limitado de quiebres leves, ajustadas a esos picos. Una línea "
            "nueva solo se acepta si, medida con ventanas que no se usaron para ajustar (validación "
            "cruzada), queda más cerca de los picos que la inicial.\n"
            "6) Segundo buffer alrededor de las líneas ajustadas: el que decide las fallas. El verdor del "
            "buffer se normaliza entre la entrehilera (0) y una planta típica del cuartel (1); falla es un "
            "tramo continuo bajo el umbral y de al menos el largo mínimo. Las fallas a menos de 2 m del "
            "extremo de la hilera se marcan aparte (campo 'borde'): dependen de cuánto se pasa el polígono "
            "del cuartel de la última planta.\n\n"
            "No usa la sombra para detectar nada; solo excluye del muestreo los píxeles tan oscuros que el "
            "cociente entre bandas es ruido (brillo < 30).\n\n"
            "El raster tiene que estar en un CRS proyectado (metros)."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT, "Ortomosaico RGB"))
        self.addParameter(QgsProcessingParameterBand(self.RED_BAND, "Banda roja", 1, self.INPUT))
        self.addParameter(QgsProcessingParameterBand(self.GREEN_BAND, "Banda verde", 2, self.INPUT))
        self.addParameter(QgsProcessingParameterBand(self.BLUE_BAND, "Banda azul", 3, self.INPUT))
        self.addParameter(QgsProcessingParameterBand(
            self.NIR_BAND, "Banda infrarroja (opcional: con ella se usa NDVI)", None, self.INPUT, optional=True))
        self.addParameter(QgsProcessingParameterEnum(
            self.DELIM, "Delimitación de cuarteles",
            ["Patrón de hileras en verde y brillo (contornos más limpios)", "Solo verde"], defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(
            self.N_BREAKS, "Máximo de quiebres por hilera (0 = recta)",
            QgsProcessingParameterNumber.Integer, 3, minValue=0, maxValue=6))
        self.addParameter(QgsProcessingParameterNumber(
            self.KINK_DEG, "Cambio máximo de dirección en cada quiebre (grados)",
            QgsProcessingParameterNumber.Double, 3.0, minValue=0.1, maxValue=15.0))
        self.addParameter(QgsProcessingParameterNumber(
            self.WINDOW, "Largo de cada perfil transversal a lo largo de la hilera (m)",
            QgsProcessingParameterNumber.Double, 2.0, minValue=0.5, maxValue=10.0))
        self.addParameter(QgsProcessingParameterNumber(
            self.SEARCH, "Semiancho del primer buffer, en fracción de la distancia entre hileras",
            QgsProcessingParameterNumber.Double, 0.35, minValue=0.1, maxValue=0.45))
        self.addParameter(QgsProcessingParameterNumber(
            self.RESID, "Margen del buffer final a cada lado de la franja de vegetación (m)",
            QgsProcessingParameterNumber.Double, 0.15, minValue=0.0, maxValue=0.5))
        self.addParameter(QgsProcessingParameterNumber(
            self.THRESHOLD, "Umbral de falla (0 = entrehilera, 1 = planta típica)",
            QgsProcessingParameterNumber.Double, 0.5, minValue=0.05, maxValue=0.95))
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_GAP, "Largo mínimo de una falla (m)",
            QgsProcessingParameterNumber.Double, 2.0, minValue=0.1))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_BLOCKS, "Cuarteles (polígonos)", QgsProcessing.TypeVectorPolygon))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_ADJ, "Hileras ajustadas (líneas)", QgsProcessing.TypeVectorLine))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_GAPS, "Fallas (líneas)", QgsProcessing.TypeVectorLine))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_BUFFER, "Buffer final alrededor de las hileras ajustadas (polígonos)",
            QgsProcessing.TypeVectorPolygon, optional=True, createByDefault=False))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_INIT, "Hileras iniciales (líneas)", QgsProcessing.TypeVectorLine,
            optional=True, createByDefault=False))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_PEAKS, "Picos de verde de la primera pasada (puntos)", QgsProcessing.TypeVectorPoint,
            optional=True, createByDefault=False))

    @staticmethod
    def _fields(*spec):
        fs = QgsFields()
        for nm, tp in spec:
            fs.append(QgsField(nm, tp))
        return fs

    def processAlgorithm(self, parameters, context, feedback):
        from osgeo import gdal
        raster = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if raster is None:
            raise QgsProcessingException(self.invalidRasterError(parameters, self.INPUT))
        if raster.crs().isGeographic():
            raise QgsProcessingException("El raster está en un CRS geográfico; reproyectalo a uno en metros (UTM).")
        bands = tuple(self.parameterAsInt(parameters, k, context)
                      for k in (self.RED_BAND, self.GREEN_BAND, self.BLUE_BAND))
        nir = self.parameterAsInt(parameters, self.NIR_BAND, context) if parameters.get(self.NIR_BAND) not in (None, "") else None
        nir = nir if nir and nir > 0 else None
        ds = gdal.Open(raster.dataProvider().dataSourceUri().split("|")[0])
        if ds is None:
            raise QgsProcessingException("No se pudo abrir el raster con GDAL.")
        gt = ds.GetGeoTransform()
        if gt[2] != 0 or gt[4] != 0:
            raise QgsProcessingException("El raster está rotado; no se admite.")
        delim = "previa" if self.parameterAsEnum(parameters, self.DELIM, context) == 0 else "verde"

        res = V.run_all(
            ds, nir_band=nir, bands=bands, delimitacion=delim,
            n_breaks=self.parameterAsInt(parameters, self.N_BREAKS, context),
            kink_deg=self.parameterAsDouble(parameters, self.KINK_DEG, context),
            win_m=self.parameterAsDouble(parameters, self.WINDOW, context),
            search_frac=self.parameterAsDouble(parameters, self.SEARCH, context),
            resid=self.parameterAsDouble(parameters, self.RESID, context),
            thr=self.parameterAsDouble(parameters, self.THRESHOLD, context),
            min_len=self.parameterAsDouble(parameters, self.MIN_GAP, context),
            log=feedback.pushInfo,
            progress=lambda i, n: feedback.setProgress(int(100.0 * i / max(1, n))))
        if not res:
            raise QgsProcessingException(
                "No se encontró ningún cuartel (zonas con patrón de hileras de 1 a 8 m).")

        crs = raster.crs()
        I, D, S = QVariant.Int, QVariant.Double, QVariant.String
        f_blk = self._fields(("cuartel", I), ("area_ha", D), ("rumbo_deg", D), ("distancia_m", D), ("indice", S),
                             ("dist_plantas_m", D), ("buffer_m", D), ("hileras", I), ("fallas_pct", D), ("fallas_sin_borde_pct", D))
        f_line = self._fields(("cuartel", I), ("hilera", I), ("ajustada", I), ("quiebres", I),
                              ("desvio_ini_m", D), ("desvio_nueva_m", D))
        f_gap = self._fields(("cuartel", I), ("hilera", I), ("ini_m", D), ("largo_m", D), ("vigor", D), ("borde", I))
        f_buf = self._fields(("cuartel", I), ("hilera", I), ("ancho_m", D))
        f_ini = self._fields(("cuartel", I), ("hilera", I), ("longitud_m", D))
        f_pk = self._fields(("cuartel", I), ("hilera", I), ("desvio_m", D))
        sk = {}
        for key, fields, geom in ((self.OUT_BLOCKS, f_blk, QgsWkbTypes.Polygon), (self.OUT_ADJ, f_line, QgsWkbTypes.LineString),
                                  (self.OUT_GAPS, f_gap, QgsWkbTypes.LineString), (self.OUT_BUFFER, f_buf, QgsWkbTypes.Polygon),
                                  (self.OUT_INIT, f_ini, QgsWkbTypes.LineString), (self.OUT_PEAKS, f_pk, QgsWkbTypes.Point)):
            sk[key] = self.parameterAsSink(parameters, key, context, fields, geom, crs)
        out = {key: dest for key, (sink, dest) in sk.items() if dest}

        def add(key, fields, geom, attrs):
            sink = sk[key][0]
            if sink is None:
                return
            ft = QgsFeature(fields)
            ft.setGeometry(geom)
            ft.setAttributes(attrs)
            sink.addFeature(ft, QgsFeatureSink.FastInsert)

        def xy(frame, u, v):
            x, y = frame.to_xy(u, v)
            return [QgsPointXY(float(a), float(b)) for a, b in zip(np.atleast_1d(x), np.atleast_1d(y))]

        for r in res:
            c, f, est, frame = r["cuartel"], r["px_factor"], r["est"], r["frame"]
            px2map = lambda x, y, f=f: (gt[0] + x * f * gt[1], gt[3] + y * f * gt[5])
            th = math.radians(est["ang"])
            az = math.degrees(math.atan2(gt[1] * math.cos(th), gt[5] * math.sin(th))) % 180.0
            tot = r["n_samples"] * 0.1
            fa = r["fallas"]
            pct = 100.0 * sum(g["largo"] for g in fa) / tot if tot else None
            pct_sb = 100.0 * sum(g["largo"] for g in fa if not g["borde"]) / tot if tot else None
            poly = QgsGeometry.fromPolygonXY([[QgsPointXY(*px2map(x, y)) for x, y in r["ring"]]])
            add(self.OUT_BLOCKS, f_blk, poly, [
                c, round(r["area_ha"], 3), round(az, 2), round(est["per_m"], 3), est["index"],
                r["dist_plantas"], None if r["buffer"] is None else round(r["buffer"], 2),
                len(r["rows_final"]), None if pct is None else round(pct, 2), None if pct_sb is None else round(pct_sb, 2)])
            for row in r["rows_ini"]:
                p = xy(frame, row.uk, row.vk)
                add(self.OUT_INIT, f_ini, QgsGeometry.fromPolylineXY([p[0], p[-1]]), [c, int(row.hilera), round(p[0].distance(p[-1]), 2)])
            for row, inf in zip(r["rows"], r["infos"]):
                if row.u1 - row.u0 < 4.0:                       # fragmentos del borde del polígono
                    continue
                ok = bool(inf["aceptada"])
                add(self.OUT_ADJ, f_line, QgsGeometry.fromPolylineXY(xy(frame, row.uk, row.vk)),
                    [c, int(row.hilera), int(ok), int(inf["quiebres"]) if ok else 0,
                     None if np.isnan(inf["cv_ini"]) else round(inf["cv_ini"], 3),
                     None if np.isnan(inf["cv_nueva"]) else round(inf["cv_nueva"], 3)])
                if "peaks" in inf and sk[self.OUT_PEAKS][0] is not None:
                    uc, vc, dd = inf["peaks"]
                    for a, b, e in zip(*frame.to_xy(uc, vc), dd):
                        add(self.OUT_PEAKS, f_pk, QgsGeometry.fromPointXY(QgsPointXY(float(a), float(b))),
                            [c, int(row.hilera), round(float(e), 3)])
            if r["buffer"] is not None:
                for row in r["rows_final"]:
                    g = QgsGeometry.fromPolylineXY(xy(frame, row.uk, row.vk)).buffer(
                        r["buffer"], 4, Qgis.EndCapStyle.Flat, Qgis.JoinStyle.Round, 2.0)
                    add(self.OUT_BUFFER, f_buf, g, [c, int(row.hilera), round(2 * r["buffer"], 2)])
            for g in fa:
                us = np.array([g["ua"], g["ub"]])
                pts = xy(frame, us, np.interp(us, g["row"].uk, g["row"].vk))
                add(self.OUT_GAPS, f_gap, QgsGeometry.fromPolylineXY(pts),
                    [c, int(g["row"].hilera), round(float(g["ua"] - g["row"].uk[0]), 2), round(g["largo"], 2),
                     round(g["z"], 3), g["borde"]])
            feedback.pushInfo(
                "Cuartel %d: %.2f ha, %s, hileras cada %.2f m (rumbo %.1f°), buffer ±%.2f m; %d fallas = %.1f%% del largo de "
                "hileras (%.1f%% sin las de borde)." % (
                    c, r["area_ha"], est["index"], est["per_m"], az, r["buffer"] or 0.0, len(fa), pct or 0.0, pct_sb or 0.0))
        return out
