# -*- coding: utf-8 -*-
from qgis.core import QgsApplication
from .provider import CropRowProvider


class CropRowDetectorPlugin:
    """
    Clase mínima que registra el proveedor de Processing al cargar QGIS
    y lo desregistra al descargar el plugin.
    """

    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initGui(self):
        self.provider = CropRowProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def unload(self):
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)
