# -*- coding: utf-8 -*-
from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon
import os

from .algorithm import DetectCropRowsAlgorithm
from .algorithm_cuarteles import InferBlocksAlgorithm
from .algorithm_perfil import RowProfileAlgorithm
from .algorithm_verde import HilerasFallasVerdeAlgorithm


class CropRowProvider(QgsProcessingProvider):

    def id(self):
        return "crop_row_detector"

    def name(self):
        return "Detección de Hileras de Cultivo"

    def icon(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return QgsProcessingProvider.icon(self)

    def loadAlgorithms(self):
        self.addAlgorithm(DetectCropRowsAlgorithm())
        self.addAlgorithm(InferBlocksAlgorithm())
        self.addAlgorithm(RowProfileAlgorithm())
        self.addAlgorithm(HilerasFallasVerdeAlgorithm())
