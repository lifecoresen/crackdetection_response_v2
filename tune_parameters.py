"""
Çatlak Tespiti Parametrelerini İnteraktif Olarak Ayarlama
OpenCV Trackbar ile parametreleri slider ile değiştir ve sonucu gerçek zamanlı görmek
"""

import cv2
import numpy as np
import os
from Opencv.preprocessing import preprocess_image
from Opencv.crack_detection import (
    detect_edges_canny, detect_edges_sobel, detect_edges_laplacian, remove_noise_from_edges, 
    find_cracks, draw_cracks
)


# Global değişkenler
current_image = None
original_image = None
processed_image = None
current_result = None

# Varsayılan parametreler
canny_threshold1 = 5
canny_threshold2 = 15
min_contour_area = 30
morphology_kernel = 1
use_sobel = 1  # 0 = Canny only, 1 = Canny + Sobel


def update_detection(x=None):
    """
    Parametreler değiştiğinde çatlak tespitini güncelle (SADECE HESAPLA)
    """
    global current_result, processed_image, original_image, current_image
    
    if processed_image is None:
        return
    
    # KENAR TESPİTİ PARAMETRELERİ
    threshold1 = cv2.getTrackbarPos('Canny Threshold1', 'Parametreler')
    threshold2 = cv2.getTrackbarPos('Canny Threshold2', 'Parametreler')
    
    # ÇATLAK FİLTRELEME
    min_area = cv2.getTrackbarPos('Min Contour Area', 'Parametreler')
    max_area = cv2.getTrackbarPos('Max Contour Area', 'Parametreler')
    
    # MORFOLOJİK İŞLEMLER
    kernel_val = cv2.getTrackbarPos('Morphology Kernel', 'Parametreler')
    erode_iter = cv2.getTrackbarPos('Erode Iterations', 'Parametreler')
    dilate_iter = cv2.getTrackbarPos('Dilate Iterations', 'Parametreler')
    
    # GÖRÜNTÜ İŞLEME
    bilateral_d = cv2.getTrackbarPos('Bilateral D', 'Parametreler')
    bilateral_sigma = cv2.getTrackbarPos('Bilateral Sigma', 'Parametreler')
    clahe_clip = cv2.getTrackbarPos('CLAHE Clip', 'Parametreler') / 10.0
    clahe_tile = cv2.getTrackbarPos('CLAHE Tile Size', 'Parametreler')
    clahe_tile_size = (clahe_tile * 2, clahe_tile * 2) if clahe_tile > 0 else (2, 2)
    
    # ALGORİTMA SEÇİMİ
    use_sobel_val = cv2.getTrackbarPos('Canny+Sobel (0=No,1=Yes)', 'Parametreler')
    use_laplacian = cv2.getTrackbarPos('Use Laplacian (0=No,1=Yes)', 'Parametreler')
    
    # VİZÜALİZASYON
    line_thickness = cv2.getTrackbarPos('Line Thickness', 'Parametreler')
    
    # Doğrulama
    if threshold2 <= threshold1:
        threshold2 = threshold1 + 1
        cv2.setTrackbarPos('Canny Threshold2', 'Parametreler', threshold2)
    
    if min_area < 1:
        min_area = 1
    
    if max_area <= min_area:
        max_area = min_area + 100
        cv2.setTrackbarPos('Max Contour Area', 'Parametreler', max_area)
    
    # Kernel en az 3 olmalı
    if kernel_val < 3:
        kernel_val = 3
    if kernel_val % 2 == 0:
        kernel_val = kernel_val + 1 if kernel_val < 9 else kernel_val - 1
    
    if bilateral_d < 3:
        bilateral_d = 3
    if bilateral_d % 2 == 0:
        bilateral_d = bilateral_d + 1 if bilateral_d < 25 else bilateral_d - 1
    
    # GÖRÜNTÜ ÖN İŞLEME
    working_image = processed_image.copy()
    
    # Bilateral filtreleme
    if bilateral_d > 0:
        working_image = cv2.bilateralFilter(working_image, bilateral_d, bilateral_sigma, bilateral_sigma)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile_size)
    working_image = clahe.apply(working_image)
    
    # KENAR TESPİTİ
    edges_list = []
    
    # Canny kenar tespiti
    edges_canny = detect_edges_canny(working_image, threshold1, threshold2)
    edges_list.append(edges_canny)
    
    # Sobel kenar tespiti
    if use_sobel_val == 1:
        edges_sobel = detect_edges_sobel(working_image, kernel_size=3)
        edges_list.append(edges_sobel)
    
    # Laplacian kenar tespiti
    if use_laplacian == 1:
        edges_laplacian = detect_edges_laplacian(working_image)
        edges_list.append(edges_laplacian)
    
    # Kenarları birleştir
    edges = edges_list[0] if len(edges_list) == 1 else cv2.bitwise_or(edges_list[0], edges_list[1])
    if len(edges_list) > 2:
        edges = cv2.bitwise_or(edges, edges_list[2])
    
    # Gürültü çıkar
    edges_cleaned = remove_noise_from_edges(edges, kernel_size=max(1, kernel_val))
    
    # Morfolojik işlemler
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_val, kernel_val))
    if erode_iter > 0:
        edges_cleaned = cv2.erode(edges_cleaned, kernel, iterations=erode_iter)
    if dilate_iter > 0:
        edges_cleaned = cv2.dilate(edges_cleaned, kernel, iterations=dilate_iter)
    
    # Çatlakları bul
    cracks = find_cracks(edges_cleaned, min_contour_area=min_area)
    
    # Alan filtrelemesi
    cracks = [c for c in cracks if min_area <= cv2.contourArea(c) <= max_area]
    
    # Sonucu çiz
    result = original_image.copy() if original_image is not None else processed_image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    result = draw_cracks(result, cracks, color=(0, 255, 0), thickness=line_thickness)
    
    # İstatistikleri ekle
    num_cracks = len(cracks)
    total_area = sum([cv2.contourArea(c) for c in cracks]) if cracks else 0
    avg_area = total_area / num_cracks if num_cracks > 0 else 0
    
    text_lines = [
        f"Cracks: {num_cracks}",
        f"Total Area: {total_area:.0f}",
        f"Avg Area: {avg_area:.0f}",
        f"T1: {threshold1} | T2: {threshold2}",
        f"Min: {min_area} | Max: {max_area}",
        f"Kernel: {kernel_val} | Erode: {erode_iter} | Dilate: {dilate_iter}",
        f"Bilateral: D={bilateral_d} Sigma={bilateral_sigma}",
        f"CLAHE: Clip={clahe_clip:.1f} Tile={clahe_tile_size}",
    ]
    
    algorithm = []
    if True:
        algorithm.append("Canny")
    if use_sobel_val:
        algorithm.append("Sobel")
    if use_laplacian:
        algorithm.append("Laplacian")
    text_lines.append(f"Method: {'+'.join(algorithm)}")
    
    y_pos = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    for text in text_lines:
        cv2.putText(result, text, (10, y_pos), font, 0.5, (255, 255, 0), 1)
        y_pos += 20
    
    # Kenarları da göster (küçük)
    edges_display = cv2.resize(edges_cleaned, (300, 300))
    
    # Global değişkenlere kaydet (pencere gösterimi için)
    current_result = (result, edges_display)


def load_and_process_image(image_path):
    """
    Görüntüyü yükle ve ön işleme yap
    """
    global current_image, original_image, processed_image
    
    if not os.path.exists(image_path):
        print(f"Hata: Görüntü bulunamadı: {image_path}")
        return False
    
    try:
        print(f"📥 Görüntü yükleniyor: {image_path}")
        original_image, _, processed_image = preprocess_image(image_path, resize_width=800)
        current_image = processed_image.copy()
        print("✓ Görüntü başarıyla yüklendi")
        return True
    except Exception as e:
        print(f"Hata: {e}")
        return False


def create_trackbars():
    """
    Trackbar'ları oluştur
    """
    window_name = 'Parametreler'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 500, 500)
    
    # KENAR TESPİTİ PARAMETRELERİ
    cv2.createTrackbar('Canny Threshold1', window_name, 5, 100, update_detection)
    cv2.createTrackbar('Canny Threshold2', window_name, 15, 200, update_detection)
    
    # ÇATLAK FİLTRELEME
    cv2.createTrackbar('Min Contour Area', window_name, 30, 500, update_detection)
    cv2.createTrackbar('Max Contour Area', window_name, 5000, 10000, update_detection)
    
    # MORFOLOJİK İŞLEMLER
    cv2.createTrackbar('Morphology Kernel', window_name, 1, 7, update_detection)
    cv2.createTrackbar('Erode Iterations', window_name, 0, 5, update_detection)
    cv2.createTrackbar('Dilate Iterations', window_name, 0, 5, update_detection)
    
    # GÖRÜNTÜ İŞLEME
    cv2.createTrackbar('Bilateral D', window_name, 9, 25, update_detection)
    cv2.createTrackbar('Bilateral Sigma', window_name, 75, 150, update_detection)
    cv2.createTrackbar('CLAHE Clip', window_name, 60, 100, update_detection)
    cv2.createTrackbar('CLAHE Tile Size', window_name, 2, 8, update_detection)
    
    # ALGORİTMA SEÇİMİ
    cv2.createTrackbar('Canny+Sobel (0=No,1=Yes)', window_name, 1, 1, update_detection)
    cv2.createTrackbar('Use Laplacian (0=No,1=Yes)', window_name, 0, 1, update_detection)
    
    # VİZÜALİZASYON
    cv2.createTrackbar('Line Thickness', window_name, 2, 10, update_detection)


def save_parameters():
    """
    Optimal parametreleri konsola yazdır
    """
    threshold1 = cv2.getTrackbarPos('Canny Threshold1', 'Parametreler')
    threshold2 = cv2.getTrackbarPos('Canny Threshold2', 'Parametreler')
    min_area = cv2.getTrackbarPos('Min Contour Area', 'Parametreler')
    max_area = cv2.getTrackbarPos('Max Contour Area', 'Parametreler')
    kernel_val = cv2.getTrackbarPos('Morphology Kernel', 'Parametreler')
    erode_iter = cv2.getTrackbarPos('Erode Iterations', 'Parametreler')
    dilate_iter = cv2.getTrackbarPos('Dilate Iterations', 'Parametreler')
    bilateral_d = cv2.getTrackbarPos('Bilateral D', 'Parametreler')
    bilateral_sigma = cv2.getTrackbarPos('Bilateral Sigma', 'Parametreler')
    clahe_clip = cv2.getTrackbarPos('CLAHE Clip', 'Parametreler') / 10.0
    clahe_tile = cv2.getTrackbarPos('CLAHE Tile Size', 'Parametreler')
    clahe_tile_size = (clahe_tile * 2, clahe_tile * 2)
    use_sobel_val = cv2.getTrackbarPos('Canny+Sobel (0=No,1=Yes)', 'Parametreler')
    use_laplacian = cv2.getTrackbarPos('Use Laplacian (0=No,1=Yes)', 'Parametreler')
    line_thickness = cv2.getTrackbarPos('Line Thickness', 'Parametreler')
    
    print("\n" + "="*60)
    print("OPTIMAL PARAMETRELER")
    print("="*60)
    
    print("\n[KENAR TESPİTİ]")
    print(f"  canny_threshold1: {threshold1}")
    print(f"  canny_threshold2: {threshold2}")
    
    print("\n[ÇATLAK FİLTRELEME]")
    print(f"  min_contour_area: {min_area}")
    print(f"  max_contour_area: {max_area}")
    
    print("\n[MORFOLOJİK İŞLEMLER]")
    print(f"  morphology_kernel: {kernel_val}")
    print(f"  erode_iterations: {erode_iter}")
    print(f"  dilate_iterations: {dilate_iter}")
    
    print("\n[GÖRÜNTÜ İŞLEME]")
    print(f"  bilateral_d: {bilateral_d}")
    print(f"  bilateral_sigma: {bilateral_sigma}")
    print(f"  clahe_clip_limit: {clahe_clip:.1f}")
    print(f"  clahe_tile_size: {clahe_tile_size}")
    
    print("\n[ALGORİTMA SEÇİMİ]")
    print(f"  use_sobel: {use_sobel_val == 1}")
    print(f"  use_laplacian: {use_laplacian == 1}")
    
    print("\n[VİZÜALİZASYON]")
    print(f"  line_thickness: {line_thickness}")
    
    print("\n" + "="*60)
    print("config.py'ye KOPİ ETMEK İÇİN:")
    print("="*60)
    
    config_code = f"""
CRACK_DETECTION_CONFIG = {{
    'canny_threshold1': {threshold1},
    'canny_threshold2': {threshold2},
    'min_contour_area': {min_area},
    'max_contour_area': {max_area},
    'morphology_kernel': {kernel_val},
    'erode_iterations': {erode_iter},
    'dilate_iterations': {dilate_iter},
    'use_sobel': {use_sobel_val == 1},
    'use_laplacian': {use_laplacian == 1},
}}

PREPROCESSING_CONFIG = {{
    'bilateral_d': {bilateral_d},
    'sigma_color': {bilateral_sigma},
    'sigma_space': {bilateral_sigma},
    'clahe_clip_limit': {clahe_clip:.1f},
    'clahe_tile_size': {clahe_tile_size},
}}

VISUALIZATION_CONFIG = {{
    'line_thickness': {line_thickness},
}}
"""
    print(config_code)
    print("="*60)


def main():
    """
    Ana program
    """
    print("\n" + "="*60)
    print("🎛️  ÇATLAK TESPİTİ PARAMETRE AYARLAYICI")
    print("="*60)
    
    # Görüntü yolunu sor
    image_path = input("\n📁 Görüntü dosya yolunu gir (örn: images/catlak8.jpg): ").strip()
    
    if not image_path:
        image_path = "images/catlak8.jpg"
        print(f"Varsayılan kullanılıyor: {image_path}")
    
    # Görüntüyü yükle
    if not load_and_process_image(image_path):
        return
    
    # Trackbar'ları oluştur
    create_trackbars()
    
    # Pencereleri SADECE BİR KEZ oluştur
    cv2.namedWindow('Sonuç', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Sonuç', 800, 600)
    
    cv2.namedWindow('Kenarlar', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Kenarlar', 400, 400)
    
    # İlk detectioni çalıştır
    update_detection()
    
    # İlk sonuçları göster
    if current_result is not None:
        result_img, edges_img = current_result
        cv2.imshow('Sonuç', result_img)
        cv2.imshow('Kenarlar', edges_img)
    
    print("\n" + "="*60)
    print("⌨️  KONTROLLER:")
    print("="*60)
    print("  • Slider'ları hareket ettirerek parametreleri ayarla")
    print("  • Sonuç penceresinde çatlakları gerçek zamanlı göreceksin")
    print("  • 's' tuşu: Parametreleri konsola yazdır & kaydet")
    print("  • 'r' tuşu: Parametreleri sıfırla")
    print("  • 'q' tuşu: Çık")
    print("="*60 + "\n")
    
    # Tuş kontrolleri
    while True:
        key = cv2.waitKey(50) & 0xFF
        
        # Her frame'de sonuçları güncelle ve göster
        if current_result is not None:
            result_img, edges_img = current_result
            cv2.imshow('Sonuç', result_img)
            cv2.imshow('Kenarlar', edges_img)
        
        if key == ord('q'):  # Çık
            print("\n👋 Program kapatılıyor...")
            break
        elif key == ord('s'):  # Kaydet
            save_parameters()
        elif key == ord('r'):  # Reset
            cv2.setTrackbarPos('Canny Threshold1', 'Parametreler', 5)
            cv2.setTrackbarPos('Canny Threshold2', 'Parametreler', 15)
            cv2.setTrackbarPos('Min Contour Area', 'Parametreler', 30)
            cv2.setTrackbarPos('Morphology Kernel', 'Parametreler', 1)
            cv2.setTrackbarPos('Canny+Sobel (0=No,1=Yes)', 'Parametreler', 1)
            print("\n🔄 Parametreler sıfırlandı")
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
