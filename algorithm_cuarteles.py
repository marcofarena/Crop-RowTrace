# -*- coding: utf-8 -*-
"""
Algoritmo de Processing: infiere el polígono de cada cuartel a partir de las
hileras detectadas por 'Detectar hileras de cultivo'. El método está en cuarteles.py.
"""
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterNumber,
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

from .cuarteles import infer_blocks, block_polygon


class InferBlocksAlgorithm(QgsProcessingAlgorithm):

    INPUT = "INPUT"
    GAP = "GAP"
    WIN = "WIN"
    MIN_ROWS = "MIN_ROWS"
    MIN_PCT = "MIN_PCT"
    SPACING = "SPACING"
    OUTPUT = "OUTPUT"

    def name(self):
        return "inferir_cuarteles"

    def displayName(self):
        return "Inferir cuarteles a partir de las hileras"

    def group(self):
        return ""

    def groupId(self):
        return ""

    def createInstance(self):
        return InferBlocksAlgorithm()

    def shortHelpString(self):
        return (
            "Genera un polígono por cuartel a partir de las líneas de hileras.\n\n"
            "Las hileras se agrupan en cuarteles (las calles de cabecera separan cuarteles; "
            "las líneas sueltas fuera de lo plantado se descartan). El extremo de cada hilera "
            "se suaviza con la mediana de las hileras vecinas: una hilera que se corta antes "
            "por fallas no deforma el borde, y un saliente real de varias hileras se conserva. "
            "El polígono nunca queda por dentro de una línea del cuartel.\n\n"
            "La capa de entrada tiene que estar en un CRS proyectado (metros)."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT, "Hileras detectadas (líneas)",
            [QgsProcessing.TypeVectorLine]))
        self.addParameter(QgsProcessingParameterNumber(
            self.GAP, "Calle mínima que separa cuarteles (m)",
            QgsProcessingParameterNumber.Double, 3.0, minValue=0.5))
        self.addParameter(QgsProcessingParameterNumber(
            self.WIN, "Hileras vecinas para suavizar el borde (impar)",
            QgsProcessingParameterNumber.Integer, 7, minValue=3))
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_ROWS, "Mínimo de hileras de un cuartel",
            QgsProcessingParameterNumber.Integer, 8, minValue=2))
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_PCT, "Largo mínimo de un cuartel, % del mayor",
            QgsProcessingParameterNumber.Double, 10.0, minValue=0.0, maxValue=100.0))
        self.addParameter(QgsProcessingParameterNumber(
            self.SPACING, "Distancia entre hileras (m, -1 = automática)",
            QgsProcessingParameterNumber.Double, -1.0))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, "Cuarteles inferidos", QgsProcessing.TypeVectorPolygon))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))
        if source.sourceCrs().isGeographic():
            raise QgsProcessingException(
                "La capa está en un CRS geográfico; reproyectala a uno en metros (UTM).")
        gap = self.parameterAsDouble(parameters, self.GAP, context)
        win = self.parameterAsInt(parameters, self.WIN, context)
        min_rows = self.parameterAsInt(parameters, self.MIN_ROWS, context)
        min_frac = self.parameterAsDouble(parameters, self.MIN_PCT, context) / 100.0
        spacing = self.parameterAsDouble(parameters, self.SPACING, context)

        geoms = []
        for f in source.getFeatures():
            g = f.geometry()
            if g.isNull() or g.isEmpty():
                continue
            parts = g.asMultiPolyline() if g.isMultipart() else [g.asPolyline()]
            for pl in parts:
                if len(pl) >= 2:
                    geoms.append(QgsGeometry.fromPolylineXY([pl[0], pl[-1]]))
        if not geoms:
            raise QgsProcessingException("La capa no tiene líneas.")
        segs = [((g.asPolyline()[0].x(), g.asPolyline()[0].y()),
                 (g.asPolyline()[1].x(), g.asPolyline()[1].y())) for g in geoms]
        feedback.pushInfo("Hileras (tramos) leídas: %d" % len(segs))

        blocks, info = infer_blocks(
            segs, spacing=spacing if spacing > 0 else None, gap=gap,
            win=win, min_rows=min_rows, min_frac=min_frac)
        if not blocks:
            raise QgsProcessingException(
                "No se encontró ningún cuartel: probá bajar el mínimo de hileras o el % del mayor.")
        d = info["dropped"]
        feedback.pushInfo(
            "Rumbo de las hileras %.1f°, separación %.2f m. %d cuartel(es); descartados "
            "%d grupos sueltos (%d tramos, %.0f m)." % (
                info["bearing_deg"], info["spacing"], len(blocks),
                d["comps"], d["segs"], d["length"]))

        fields = QgsFields()
        fields.append(QgsField("cuartel", QVariant.Int))
        fields.append(QgsField("hileras", QVariant.Int))
        fields.append(QgsField("hileras_acortadas", QVariant.Int))
        fields.append(QgsField("tramos", QVariant.Int))
        fields.append(QgsField("longitud_m", QVariant.Double))
        fields.append(QgsField("separacion_m", QVariant.Double))
        fields.append(QgsField("area_ha", QVariant.Double))
        fields.append(QgsField("cobertura_pct", QVariant.Double))
        sink, dest = self.parameterAsSink(
            parameters, self.OUTPUT, context, fields,
            QgsWkbTypes.Polygon, source.sourceCrs())

        # Los cuarteles se numeran de un extremo al otro de las hileras.
        for n, b in enumerate(blocks, 1):
            if feedback.isCanceled():
                break
            g = block_polygon(b)
            tot = sum(geoms[k].length() for k in b["seg_idx"])
            ins = sum(geoms[k].intersection(g).length() for k in b["seg_idx"])
            ft = QgsFeature(fields)
            ft.setGeometry(g)
            ft.setAttributes([
                n, int(b["n_rows"]), int(b["ignored_rows"]), int(b["n_segs"]),
                round(float(b["length"]), 1), round(float(b["spacing"]), 2),
                round(g.area() / 10000.0, 3),
                round(100.0 * ins / tot, 2) if tot else None])
            sink.addFeature(ft, QgsFeatureSink.FastInsert)
            feedback.pushInfo(
                "Cuartel %d: %d hileras (%d con el extremo acortado por el borde general), "
                "%.2f ha, %.1f%% de las líneas adentro." % (
                    n, b["n_rows"], b["ignored_rows"], g.area() / 10000.0,
                    100.0 * ins / tot if tot else 0.0))
        return {self.OUTPUT: dest}
