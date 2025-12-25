"""
Dökülen Sıva / Nem İzleri Tespiti Modülü
Bina duvarları ve kolonlar üzerindeki dökülen sıva parçalarını tespit eder

Bu modül plaster_detection.py modülünü kullanır.
"""

# Yeni modülden tüm fonksiyonları import et
from Opencv.plaster_detection import (
    detect_plaster_pipeline as detect_moisture_pipeline,
    find_plaster_regions as find_moisture_regions,
    draw_plaster_regions as draw_moisture_regions,
    analyze_plaster_properties as analyze_moisture_properties,
    detect_edges_canny,
    detect_color_anomalies,
    detect_brightness_changes,
    detect_texture_changes,
    fill_edge_regions,
    filter_plaster_contours
)
