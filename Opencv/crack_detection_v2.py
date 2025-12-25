"""
Çatlak Tespiti V2 - Gelişmiş Yöntemler
Line Segment Detection + Ridge Detection + Multi-scale Analysis
"""

import cv2
import numpy as np
from scipy import ndimage


def detect_ridges(gray, sigma=1.0):
    """
    Ridge detection - ince çatlak yapıları için
    Hessian matrix eigenvalue analizi
    """
    # Gaussian smoothing
    smoothed = cv2.GaussianBlur(gray.astype(np.float64), (0, 0), sigma)
    
    # Hessian matrix bileşenleri
    Ixx = cv2.Sobel(smoothed, cv2.CV_64F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(smoothed, cv2.CV_64F, 1, 1, ksize=3)
    
    # Eigenvalue hesaplama
    tmp = np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2)
    lambda1 = 0.5 * (Ixx + Iyy + tmp)
    lambda2 = 0.5 * (Ixx + Iyy - tmp)
    
    # Ridge strength (en büyük eigenvalue)
    ridge = np.maximum(np.abs(lambda1), np.abs(lambda2))
    
    # Normalize
    ridge = ((ridge - ridge.min()) / (ridge.max() - ridge.min() + 1e-8) * 255).astype(np.uint8)
    
    return ridge


def detect_lines_lsd(gray):
    """
    Line Segment Detector - düz çatlak çizgilerini tespit et
    """
    lsd = cv2.createLineSegmentDetector(0)
    lines, widths, precs, nfas = lsd.detect(gray)
    
    if lines is None:
        return [], np.zeros_like(gray)
    
    # Çizgi maskesi oluştur
    line_mask = np.zeros_like(gray)
    
    filtered_lines = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Sadece yeterince uzun çizgileri al
        if length > 30:
            filtered_lines.append(line)
            cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
    
    return filtered_lines, line_mask


def multi_scale_crack_detection(gray, scales=[1.0, 2.0, 3.0]):
    """
    Multi-scale çatlak tespiti - farklı kalınlıklardaki çatlaklar için
    """
    combined = np.zeros_like(gray, dtype=np.float64)
    
    for sigma in scales:
        ridge = detect_ridges(gray, sigma)
        combined += ridge.astype(np.float64)
    
    # Normalize
    combined = ((combined - combined.min()) / (combined.max() - combined.min() + 1e-8) * 255).astype(np.uint8)
    
    return combined


def detect_dark_lines(gray, threshold=30):
    """
    Koyu çizgileri tespit et - çatlaklar genelde çevreden daha koyu
    """
    # Yerel ortalama
    kernel_size = 21
    local_mean = cv2.blur(gray.astype(np.float64), (kernel_size, kernel_size))
    
    # Yerel ortalamadan ne kadar koyu?
    diff = local_mean - gray.astype(np.float64)
    
    # Sadece koyu olanları al
    dark_mask = (diff > threshold).astype(np.uint8) * 255
    
    return dark_mask


def skeletonize(binary_mask):
    """
    İskelet çıkarma - çatlakların merkez çizgisini bul
    """
    skeleton = np.zeros_like(binary_mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    img = binary_mask.copy()
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        
        if cv2.countNonZero(img) == 0:
            break
    
    return skeleton


def filter_crack_contours(contours, min_length=50, max_width=30, min_aspect_ratio=3.0):
    """
    Çatlak benzeri konturları filtrele
    Çatlaklar: uzun, ince, düzensiz
    """
    cracks = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:
            continue
            
        # Minimum bounding rectangle
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        
        if width == 0 or height == 0:
            continue
        
        # Uzun kenar ve kısa kenar
        long_side = max(width, height)
        short_side = min(width, height)
        
        if short_side == 0:
            continue
        
        aspect_ratio = long_side / short_side
        
        # Çatlak kriterleri:
        # - Yeterince uzun
        # - Yeterince ince (geniş değil)
        # - Aspect ratio yüksek (uzun ve ince)
        if long_side >= min_length and short_side <= max_width and aspect_ratio >= min_aspect_ratio:
            cracks.append(contour)
    
    return cracks


def detect_cracks_v2(image, config=None):
    """
    Ana çatlak tespiti fonksiyonu - V2
    
    Args:
        image: BGR veya grayscale görüntü
        config: Konfigürasyon dict
    
    Returns:
        cracks: Çatlak konturları listesi
        visualization: Görselleştirme için maske
    """
    if config is None:
        config = {
            'ridge_scales': [1.0, 1.5, 2.0],
            'dark_threshold': 20,
            'min_crack_length': 40,
            'max_crack_width': 25,
            'min_aspect_ratio': 2.5,
            'use_lsd': True,
            'combine_methods': True
        }
    
    # Grayscale'e çevir
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Gürültü azaltma
    denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    # === YÖNTEM 1: Multi-scale Ridge Detection ===
    ridge_mask = multi_scale_crack_detection(denoised, config['ridge_scales'])
    _, ridge_binary = cv2.threshold(ridge_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # === YÖNTEM 2: Dark Line Detection ===
    dark_mask = detect_dark_lines(denoised, config['dark_threshold'])
    
    # === YÖNTEM 3: Line Segment Detection ===
    lsd_mask = np.zeros_like(gray)
    if config['use_lsd']:
        _, lsd_mask = detect_lines_lsd(denoised)
    
    # === Yöntemleri birleştir ===
    if config['combine_methods']:
        # En az 2 yöntemin hemfikir olduğu pikseller
        combined = np.zeros_like(gray, dtype=np.float64)
        combined += (ridge_binary > 0).astype(np.float64)
        combined += (dark_mask > 0).astype(np.float64)
        combined += (lsd_mask > 0).astype(np.float64)
        
        # En az 2 yöntem hemfikir olmalı
        final_mask = (combined >= 2).astype(np.uint8) * 255
    else:
        # Sadece ridge detection
        final_mask = ridge_binary
    
    # Morfolojik temizlik
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Konturları bul
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Çatlak benzeri konturları filtrele
    cracks = filter_crack_contours(
        contours,
        min_length=config['min_crack_length'],
        max_width=config['max_crack_width'],
        min_aspect_ratio=config['min_aspect_ratio']
    )
    
    # Görselleştirme maskesi
    visualization = np.zeros_like(gray)
    cv2.drawContours(visualization, cracks, -1, 255, -1)
    
    return cracks, visualization


def detect_moisture_v2(image, config=None):
    """
    Nem/Dökülen sıva tespiti - V2
    Renk anomalisi + Texture analizi
    
    Args:
        image: BGR görüntü
        config: Konfigürasyon dict
    
    Returns:
        regions: Nem bölgesi konturları
        visualization: Görselleştirme maskesi
    """
    if config is None:
        config = {
            'color_sensitivity': 1.2,
            'texture_sensitivity': 1.5,
            'min_region_area': 1000,
            'max_region_area': 100000,
        }
    
    # Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bgr = image
    else:
        gray = image
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # === YÖNTEM 1: Renk Anomalisi (LAB) ===
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    
    # Arka plan tahmini
    L_bg = cv2.medianBlur(L, 51)
    
    # Fark
    L_diff = cv2.absdiff(L, L_bg)
    
    mean_diff = np.mean(L_diff)
    std_diff = np.std(L_diff)
    threshold = mean_diff + config['color_sensitivity'] * std_diff
    _, color_mask = cv2.threshold(L_diff, threshold, 255, cv2.THRESH_BINARY)
    
    # === YÖNTEM 2: Texture Analizi ===
    # Yerel varyans
    blur = cv2.GaussianBlur(gray.astype(np.float32), (15, 15), 0)
    blur_sq = cv2.GaussianBlur((gray.astype(np.float32))**2, (15, 15), 0)
    variance = np.sqrt(np.abs(blur_sq - blur**2))
    
    mean_var = np.mean(variance)
    std_var = np.std(variance)
    var_threshold = mean_var + config['texture_sensitivity'] * std_var
    texture_mask = (variance > var_threshold).astype(np.uint8) * 255
    
    # === Birleştir ===
    combined = cv2.bitwise_and(color_mask, texture_mask)
    
    # Morfolojik işlemler
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Konturları bul
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Boyut filtresi
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if config['min_region_area'] <= area <= config['max_region_area']:
            regions.append(contour)
    
    # Görselleştirme
    visualization = np.zeros_like(gray)
    cv2.drawContours(visualization, regions, -1, 255, -1)
    
    return regions, visualization


def draw_results_v2(image, cracks, moisture_regions):
    """
    Sonuçları görüntü üzerine çiz
    """
    result = image.copy()
    
    # Çatlakları yeşil çiz
    cv2.drawContours(result, cracks, -1, (0, 255, 0), 2)
    
    # Nem bölgelerini mavi çiz
    cv2.drawContours(result, moisture_regions, -1, (255, 100, 0), 2)
    
    # İstatistik ekle
    h, w = image.shape[:2]
    cv2.rectangle(result, (10, 10), (250, 80), (0, 0, 0), -1)
    cv2.putText(result, f"Catlak: {len(cracks)}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(result, f"Nem/Siva: {len(moisture_regions)}", (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
    
    return result


def analyze_results_v2(cracks, moisture_regions):
    """
    Sonuçları analiz et
    """
    crack_stats = {
        'count': len(cracks),
        'total_area': sum(cv2.contourArea(c) for c in cracks) if cracks else 0,
        'total_length': sum(cv2.arcLength(c, True) for c in cracks) if cracks else 0
    }
    
    moisture_stats = {
        'count': len(moisture_regions),
        'total_area': sum(cv2.contourArea(r) for r in moisture_regions) if moisture_regions else 0
    }
    
    return crack_stats, moisture_stats
