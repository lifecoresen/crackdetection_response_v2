"""
Dökülen Sıva Tespiti Modülü
Bina duvarları ve kolonlar üzerindeki dökülen/ayrışan sıva parçalarını tespit eder

Özellikler:
- Kenar tespiti (sıva dökülmesi keskin kenarlar oluşturur)
- Renk farklılığı (dökülen sıvanın altındaki yüzey farklı renk)
- Texture değişimi (sıva vs alttaki yüzey)
- Kontur analizi
"""

import cv2
import numpy as np


def detect_edges_canny(gray_image, threshold1=50, threshold2=150):
    """
    Canny kenar tespiti - dökülen sıvanın keskin kenarlarını bulur
    
    Args:
        gray_image (numpy.ndarray): Gri görüntü
        threshold1 (int): Alt eşik
        threshold2 (int): Üst eşik
    
    Returns:
        numpy.ndarray: Kenar maskesi
    """
    # Blur uygula gürültüyü azaltmak için
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1, threshold2)
    return edges


def detect_color_anomalies(image_bgr, sensitivity=1.5):
    """
    Renk anomalisi tespiti - duvar renginden farklı bölgeler
    Dökülen sıvanın altındaki yüzey farklı renkte olur
    
    Args:
        image_bgr (numpy.ndarray): BGR görüntü
        sensitivity (float): Hassasiyet çarpanı
    
    Returns:
        numpy.ndarray: Anomali maskesi
    """
    # LAB renk uzayına çevir (renk farklılıkları daha belirgin)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    
    # Median filtreleme ile arka plan tahmin et
    L_bg = cv2.medianBlur(L, 51)
    A_bg = cv2.medianBlur(A, 51)
    B_bg = cv2.medianBlur(B, 51)
    
    # Arka plandan farkı hesapla
    L_diff = cv2.absdiff(L, L_bg)
    A_diff = cv2.absdiff(A, A_bg)
    B_diff = cv2.absdiff(B, B_bg)
    
    # Toplam renk farklılığı
    total_diff = (L_diff.astype(np.float32) + 
                  A_diff.astype(np.float32) + 
                  B_diff.astype(np.float32)) / 3
    total_diff = total_diff.astype(np.uint8)
    
    # Eşik uygula
    mean_diff = np.mean(total_diff)
    std_diff = np.std(total_diff)
    threshold = mean_diff + sensitivity * std_diff
    
    _, mask = cv2.threshold(total_diff, threshold, 255, cv2.THRESH_BINARY)
    
    return mask.astype(np.uint8)


def detect_brightness_changes(gray_image, block_size=51):
    """
    Parlaklık değişimi tespiti
    Dökülen sıva alanları çevresinden daha koyu veya açık olabilir
    
    Args:
        gray_image (numpy.ndarray): Gri görüntü
        block_size (int): Yerel ortalama pencere boyutu
    
    Returns:
        numpy.ndarray: Parlaklık anomali maskesi
    """
    # Yerel ortalama
    local_mean = cv2.blur(gray_image, (block_size, block_size))
    
    # Fark hesapla (hem koyu hem açık alanlar)
    diff = cv2.absdiff(gray_image, local_mean)
    
    # Eşik
    mean_val = np.mean(diff)
    std_val = np.std(diff)
    threshold = mean_val + 1.5 * std_val
    
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    return mask


def detect_texture_changes(gray_image):
    """
    Texture değişimi tespiti
    Sıva ve alttaki yüzey farklı doku gösterir
    
    Args:
        gray_image (numpy.ndarray): Gri görüntü
    
    Returns:
        numpy.ndarray: Texture maskesi
    """
    # Laplacian ile texture analizi
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    laplacian = np.abs(laplacian)
    
    # Yerel varyans (texture ölçüsü)
    blur = cv2.GaussianBlur(gray_image.astype(np.float32), (15, 15), 0)
    blur_sq = cv2.GaussianBlur((gray_image.astype(np.float32))**2, (15, 15), 0)
    variance = blur_sq - blur**2
    variance = np.sqrt(np.abs(variance)).astype(np.uint8)
    
    # Yüksek varyans = texture değişimi
    mean_var = np.mean(variance)
    std_var = np.std(variance)
    threshold = mean_var + std_var
    
    _, mask = cv2.threshold(variance, threshold, 255, cv2.THRESH_BINARY)
    
    return mask


def fill_edge_regions(edge_mask, kernel_size=7):
    """
    Kenar maskesindeki kapalı bölgeleri doldur
    
    Args:
        edge_mask (numpy.ndarray): Kenar maskesi
        kernel_size (int): Kernel boyutu
    
    Returns:
        numpy.ndarray: Doldurulmuş maske
    """
    # Kenarları kalınlaştır ve birleştir
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(edge_mask, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Kapalı bölgeleri doldur
    # Flood fill kullan
    h, w = closed.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = closed.copy()
    
    # Köşelerden flood fill
    cv2.floodFill(filled, mask, (0, 0), 255)
    
    # Tersini al - iç bölgeler
    filled_inv = cv2.bitwise_not(filled)
    
    # Orijinal kenarlarla birleştir
    result = cv2.bitwise_or(closed, filled_inv)
    
    return result


def filter_plaster_contours(contours, min_area=500, max_area=None, 
                            min_solidity=0.3, max_solidity=0.95,
                            min_extent=0.2):
    """
    Dökülen sıva özelliklerine göre konturları filtrele
    
    Args:
        contours (list): Konturlar
        min_area (int): Minimum alan
        max_area (int): Maksimum alan (None = sınırsız)
        min_solidity (float): Minimum doluluk oranı
        max_solidity (float): Maksimum doluluk oranı
        min_extent (float): Minimum kaplama oranı
    
    Returns:
        list: Filtrelenmiş konturlar
    """
    filtered = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Alan filtresi
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        
        # Convex hull hesapla
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            continue
        
        # Solidity (doluluk) - dökülen sıva düzensiz şekilli
        solidity = area / hull_area
        if solidity < min_solidity or solidity > max_solidity:
            continue
        
        # Bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        
        if rect_area == 0:
            continue
        
        # Extent (kaplama oranı)
        extent = area / rect_area
        if extent < min_extent:
            continue
        
        filtered.append(contour)
    
    return filtered


def find_plaster_regions(mask, min_area=500, max_area=None):
    """
    Dökülen sıva bölgelerini bul
    
    Args:
        mask (numpy.ndarray): İkili maske
        min_area (int): Minimum alan
        max_area (int): Maksimum alan
    
    Returns:
        list: Sıva bölgeleri (konturlar)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrele
    filtered = filter_plaster_contours(
        contours, 
        min_area=min_area, 
        max_area=max_area,
        min_solidity=0.2,
        max_solidity=0.98,
        min_extent=0.15
    )
    
    return filtered


def draw_plaster_regions(image, regions, color=(0, 0, 255), thickness=2, draw_filled=False):
    """
    Dökülen sıva bölgelerini çiz
    
    Args:
        image (numpy.ndarray): Orijinal görüntü
        regions (list): Sıva bölgeleri
        color (tuple): BGR renk
        thickness (int): Çizgi kalınlığı
        draw_filled (bool): Dolu çizim
    
    Returns:
        numpy.ndarray: İşaretlenmiş görüntü
    """
    result = image.copy()
    
    if draw_filled:
        # Yarı saydam overlay
        overlay = result.copy()
        cv2.drawContours(overlay, regions, -1, color, -1)
        result = cv2.addWeighted(overlay, 0.4, result, 0.6, 0)
        # Kenarları da çiz
        cv2.drawContours(result, regions, -1, color, thickness)
    else:
        cv2.drawContours(result, regions, -1, color, thickness)
    
    return result


def analyze_plaster_properties(regions):
    """
    Dökülen sıva bölgelerinin özelliklerini analiz et
    
    Args:
        regions (list): Sıva bölgeleri
    
    Returns:
        dict: İstatistikler
    """
    if len(regions) == 0:
        return {
            'total_regions': 0,
            'total_area': 0,
            'average_area': 0,
            'max_area': 0
        }
    
    areas = [cv2.contourArea(region) for region in regions]
    
    return {
        'total_regions': len(regions),
        'total_area': sum(areas),
        'average_area': sum(areas) / len(regions),
        'max_area': max(areas),
        'areas': areas
    }


def detect_plaster_pipeline(original_image, gray_image, min_region_area=500):
    """
    Dökülen sıva tespiti ana pipeline'ı
    
    Yöntem:
    1. Kenar tespiti (keskin sınırlar)
    2. Renk anomalisi tespiti (farklı renkli bölgeler)
    3. Parlaklık değişimi tespiti
    4. Texture analizi
    5. Sonuçları birleştir ve filtrele
    
    Args:
        original_image (numpy.ndarray): Orijinal BGR görüntü
        gray_image (numpy.ndarray): Gri görüntü
        min_region_area (int): Minimum bölge alanı
    
    Returns:
        dict: Sonuçlar
    """
    h, w = gray_image.shape[:2]
    max_region_area = h * w * 0.4  # Resmin %40'ından büyük olamaz
    
    # 1. Kenar tespiti
    edges = detect_edges_canny(gray_image, threshold1=30, threshold2=100)
    
    # 2. Renk anomalisi tespiti
    color_anomaly = detect_color_anomalies(original_image, sensitivity=1.2)
    
    # 3. Parlaklık değişimi
    brightness_change = detect_brightness_changes(gray_image, block_size=41)
    
    # 4. Texture değişimi
    texture_change = detect_texture_changes(gray_image)
    
    # 5. Kenar bölgelerini doldur
    edge_regions = fill_edge_regions(edges, kernel_size=5)
    
    # 6. Birleştirme stratejisi:
    # Renk anomalisi VEYA parlaklık değişimi olan bölgeler
    anomaly_mask = cv2.bitwise_or(color_anomaly, brightness_change)
    
    # Kenar bölgeleri ile kesişim (kenarlarla çevrili anomaliler)
    kenar_confirmed = cv2.bitwise_and(anomaly_mask, edge_regions)
    
    # Ana maske
    combined = cv2.bitwise_or(anomaly_mask, kenar_confirmed)
    
    # 7. Morfolojik temizlik
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Küçük gürültüleri temizle
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small, iterations=2)
    
    # Yakın bölgeleri birleştir
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    
    # 8. Bölgeleri bul ve filtrele
    regions = find_plaster_regions(cleaned, min_area=min_region_area, max_area=max_region_area)
    
    # 9. Özellikleri analiz et
    properties = analyze_plaster_properties(regions)
    
    return {
        'edges': edges,
        'color_anomaly': color_anomaly,
        'brightness_change': brightness_change,
        'texture_change': texture_change,
        'edge_regions': edge_regions,
        'combined_mask': combined,
        'cleaned_mask': cleaned,
        'regions': regions,
        'properties': properties
    }


# Geriye uyumluluk için eski fonksiyon isimleri
def detect_moisture_pipeline(original_image, gray_image, min_region_area=100):
    """Geriye uyumluluk - plaster pipeline'a yönlendir"""
    return detect_plaster_pipeline(original_image, gray_image, min_region_area)


def find_moisture_regions(mask, min_region_area=100):
    """Geriye uyumluluk"""
    return find_plaster_regions(mask, min_area=min_region_area)


def draw_moisture_regions(image, regions, color=(0, 0, 255), thickness=2, draw_filled=False):
    """Geriye uyumluluk"""
    return draw_plaster_regions(image, regions, color, thickness, draw_filled)


def analyze_moisture_properties(regions):
    """Geriye uyumluluk"""
    return analyze_plaster_properties(regions)
