# -*- coding: utf-8 -*-
"""
Plugin: Detección de Hileras de Cultivo
Punto de entrada requerido por QGIS.
"""


def classFactory(iface):
    from .crop_row_detector import CropRowDetectorPlugin
    return CropRowDetectorPlugin(iface)
