"""
Çatlak Tespiti Modülü
Bina duvarları ve kolonlar üzerindeki çatlakları tespit eder
"""

import cv2
import numpy as np


def detect_edges_canny(image, threshold1=50, threshold2=150):
    """
    Canny kenar tespiti ile çatlakları bulma
    
    Args:
        image (numpy.ndarray): Gri görüntü
        threshold1 (int): Alt eşik değeri
        threshold2 (int): Üst eşik değeri
    
    Returns:
        numpy.ndarray: Kenar tespiti sonucu
    """
    edges = cv2.Canny(image, threshold1, threshold2)
    return edges


def detect_edges_sobel(image, kernel_size=3):
    """
    Sobel operatörü ile kenar tespiti
    
    Args:
        image (numpy.ndarray): Gri görüntü
        kernel_size (int): Kernel boyutu (1, 3, 5, 7)
    
    Returns:
        numpy.ndarray: Sobel kenarları
    """
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    
    sobel = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
    return cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY)[1]


def detect_edges_laplacian(image):
    """
    Laplacian operatörü ile kenar tespiti
    
    Args:
        image (numpy.ndarray): Gri görüntü
    
    Returns:
        numpy.ndarray: Laplacian kenarları
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.absolute(laplacian).astype(np.uint8)
    return cv2.threshold(laplacian, 50, 255, cv2.THRESH_BINARY)[1]


def remove_noise_from_edges(edges, kernel_size=5):
    """
    Kenarlardan gürültü çıkar
    
    Args:
        edges (numpy.ndarray): Kenar görüntüsü
        kernel_size (int): Kernel boyutu
    
    Returns:
        numpy.ndarray: Temizlenmiş kenarlar
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Gürültü çıkar (açılış işlemi)
    cleaned = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Kırık çizgileri birleştir (kapanış işlemi)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return cleaned


def find_cracks(edges, min_contour_area=50, max_solidity=0.5, max_aspect_ratio=8.0):
    """
    Çatlak konturlarını bulma ve analiz etme
    Borular gibi düzgün geometrik şekilleri filtreler
    
    Args:
        edges (numpy.ndarray): Temizlenmiş kenar görüntüsü
        min_contour_area (int): Minimum kontur alanı
        max_solidity (float): Maksimum doluluk oranı (düşük = düzensiz çatlak)
        max_aspect_ratio (float): Maksimum en-boy oranı
    
    Returns:
        list: Bulunan çatlaklar (konturlar)
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cracks = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        
        # Extent: kontur alanı / bounding box alanı
        extent = float(area) / bbox_area if bbox_area > 0 else 0
        
        # Convex hull ile solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Konturun düzlüğünü kontrol et - approxPolyDP
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertex_count = len(approx)
        
        # Rotated rectangle analizi
        rect_fill = 0
        rect_aspect = 1
        if len(contour) >= 5:
            rect = cv2.minAreaRect(contour)
            rect_w, rect_h = rect[1]
            rect_area = rect_w * rect_h
            if rect_area > 0:
                rect_fill = float(area) / rect_area
            if min(rect_w, rect_h) > 0:
                rect_aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
        
        # Minimum boyut (en dar kenar)
        min_dim = min(w, h)
        max_dim = max(w, h)
        
        # ===== BORU FİLTRELEME KURALLARI (Agresif) =====
        
        is_pipe = False
        
        # Kural 1: Rotated rectangle'ı %50'den fazla dolduran = düzgün şekil
        if rect_fill > 0.5:
            is_pipe = True
        
        # Kural 2: Extent yüksek = bounding box'a iyi oturuyor
        if extent > 0.4:
            is_pipe = True
        
        # Kural 3: Solidity yüksek = convex hull'a yakın
        if solidity > 0.4:
            is_pipe = True
        
        # Kural 4: Az köşeli şekiller (8'den az köşe)
        if vertex_count <= 8 and (extent > 0.3 or solidity > 0.35):
            is_pipe = True
        
        # Kural 5: Geniş şekiller (minimum boyut 10 pikselden fazla)
        if min_dim > 10 and rect_fill > 0.4:
            is_pipe = True
        
        # ===== ÇATLAK KURTARMA KURALLARI =====
        # Gerçek çatlaklar: çok ince, uzun ve düzensiz
        
        is_crack = False
        
        # Çatlak özellikleri: çok ince (min_dim küçük) ve uzun
        if min_dim <= 8 and max_dim > 50:
            is_crack = True
        
        # Çok düzensiz şekiller (düşük solidity VE düşük extent)
        if solidity < 0.25 and extent < 0.2:
            is_crack = True
        
        # Çok fazla köşeli (düzensiz) şekiller
        if vertex_count > 15 and solidity < 0.35:
            is_crack = True
        
        # Perimeter/area oranı yüksek = ince uzun şekil
        thin_ratio = perimeter / area if area > 0 else 0
        if thin_ratio > 0.3 and min_dim < 15:
            is_crack = True
        
        # Karar
        if is_crack:
            cracks.append(contour)
        elif not is_pipe:
            cracks.append(contour)
    
    return cracks


def draw_cracks(image, cracks, color=(0, 255, 0), thickness=2):
    """
    Tespit edilen çatlakları görüntüye çiz
    
    Args:
        image (numpy.ndarray): Orijinal görüntü
        cracks (list): Çatlak konturları
        color (tuple): BGR renk kodu
        thickness (int): Çizgi kalınlığı
    
    Returns:
        numpy.ndarray: Çatlakları işaretlenmiş görüntü
    """
    result = image.copy()
    cv2.drawContours(result, cracks, -1, color, thickness)
    return result


def analyze_crack_properties(cracks):
    """
    Çatlakların özelliklerini analiz et
    
    Args:
        cracks (list): Çatlak konturları
    
    Returns:
        dict: Çatlak istatistikleri
    """
    if len(cracks) == 0:
        return {
            'total_cracks': 0,
            'total_area': 0,
            'average_area': 0,
            'max_area': 0,
            'crack_lengths': []
        }
    
    areas = [cv2.contourArea(crack) for crack in cracks]
    lengths = [cv2.arcLength(crack, True) for crack in cracks]
    
    return {
        'total_cracks': len(cracks),
        'total_area': sum(areas),
        'average_area': sum(areas) / len(areas),
        'max_area': max(areas),
        'crack_lengths': lengths,
        'average_length': sum(lengths) / len(lengths) if lengths else 0
    }


def detect_cracks_pipeline(processed_image, min_contour_area=50, canny_threshold1=15, canny_threshold2=60):
    """
    Tam çatlak tespiti pipeline'ı
    
    Args:
        processed_image (numpy.ndarray): Ön işlenmiş gri görüntü
        min_contour_area (int): Minimum kontur alanı
        canny_threshold1 (int): Canny alt eşiği
        canny_threshold2 (int): Canny üst eşiği
    
    Returns:
        dict: Sonuçlar
    """
    # Canny kenar tespiti
    edges_canny = detect_edges_canny(processed_image, canny_threshold1, canny_threshold2)
    
    # Sobel kenar tespiti (alternatif)
    edges_sobel = detect_edges_sobel(processed_image, kernel_size=3)
    
    # Her iki kenarı birleştir
    edges_combined = cv2.bitwise_or(edges_canny, edges_sobel)
    
    # Gürültü çıkar
    edges_cleaned = remove_noise_from_edges(edges_combined, kernel_size=3)
    
    # Çatlakları bul
    cracks = find_cracks(edges_cleaned, min_contour_area=min_contour_area)
    
    # Özellikleri analiz et
    properties = analyze_crack_properties(cracks)
    
    return {
        'edges': edges_combined,
        'cleaned_edges': edges_cleaned,
        'cracks': cracks,
        'properties': properties
    }


def detect_cracks_adaptive_pipeline(processed_image, blur_kernel=7, adaptive_block=11, 
                                   adaptive_constant=2, morph_kernel=3, morph_iterations=2,
                                   min_contour_area=300, max_solidity=0.5, max_aspect_ratio=8.0):
    """
    Adaptive Threshold yöntemi ile çatlak tespiti (improved method)
    
    Args:
        processed_image (numpy.ndarray): Ön işlenmiş gri görüntü
        blur_kernel (int): Gaussian blur kernel boyutu
        adaptive_block (int): Adaptive threshold block boyutu
        adaptive_constant (int): Adaptive threshold sabit
        morph_kernel (int): Morfoloji kernel boyutu
        morph_iterations (int): Morfoloji işlem sayısı
        min_contour_area (int): Minimum kontur alanı
        max_solidity (float): Maksimum doluluk oranı (düşük = çatlak)
        max_aspect_ratio (float): Maksimum en-boy oranı
    
    Returns:
        dict: Sonuçlar
    """
    # 1. Gaussian Blur (mikro dokular)
    if blur_kernel % 2 == 0:
        blur_kernel = blur_kernel + 1
    blurred = cv2.GaussianBlur(processed_image, (blur_kernel, blur_kernel), 0)
    
    # 2. Adaptive Threshold
    if adaptive_block % 2 == 0:
        adaptive_block = adaptive_block + 1
    
    threshold = cv2.adaptiveThreshold(blurred, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 
                                      adaptive_block, 
                                      adaptive_constant)
    
    # 3. Morphological Closing (kopukları bağla)
    if morph_kernel % 2 == 0:
        morph_kernel = morph_kernel + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                       (morph_kernel, morph_kernel))
    closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, 
                              iterations=morph_iterations)
    
    # 4. Kontur bulma (şekil filtreleme ile)
    cracks = find_cracks(closing, min_contour_area=min_contour_area, 
                        max_solidity=max_solidity, max_aspect_ratio=max_aspect_ratio)
    
    # 5. Özellikleri analiz et
    properties = analyze_crack_properties(cracks)
    
    return {
        'binary': threshold,
        'morphology': closing,
        'cracks': cracks,
        'properties': properties
    }


def detect_cracks_orb_pipeline(gray_image, blur_kernel=3, bilateral_d=5, bilateral_sigma=75,
                               canny_threshold1=100, canny_threshold2=200, 
                               morph_kernel=5, orb_features=1500, min_contour_area=200,
                               max_solidity=0.5, max_aspect_ratio=8.0):
    """
    ORB Feature Detection ile gelişmiş çatlak tespiti
    Logaritmik dönüşüm + Bilateral filter + Canny + ORB
    
    Args:
        gray_image (numpy.ndarray): Gri görüntü
        blur_kernel (int): Blur kernel boyutu
        bilateral_d (int): Bilateral filter diameter
        bilateral_sigma (int): Bilateral filter sigma değeri
        canny_threshold1 (int): Canny alt eşik
        canny_threshold2 (int): Canny üst eşik
        morph_kernel (int): Morfoloji kernel boyutu
        orb_features (int): ORB feature sayısı
        min_contour_area (int): Minimum kontur alanı
        max_solidity (float): Maksimum solidity
        max_aspect_ratio (float): Maksimum aspect ratio
    
    Returns:
        dict: Sonuçlar
    """
    # 1. Averaging blur
    blur = cv2.blur(gray_image, (blur_kernel, blur_kernel))
    
    # 2. Logaritmik dönüşüm (kontrast iyileştirme)
    # Koyu alanları (çatlaklar) daha belirgin yapar
    blur_float = blur.astype(np.float64)
    max_val = np.max(blur_float)
    if max_val > 0:
        img_log = (np.log1p(blur_float) / np.log1p(max_val)) * 255
    else:
        img_log = blur_float
    img_log = np.clip(img_log, 0, 255).astype(np.uint8)
    
    # 3. Bilateral filter (kenarları koruyarak gürültü azaltma)
    bilateral = cv2.bilateralFilter(img_log, bilateral_d, bilateral_sigma, bilateral_sigma)
    
    # 4. Canny Edge Detection
    edges = cv2.Canny(bilateral, canny_threshold1, canny_threshold2)
    
    # 5. Morphological işlemler (kırık çizgileri birleştir)
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    
    # Dilation - çatlak çizgilerini kalınlaştır
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Closing - kırık parçaları birleştir
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 6. DÜZ KENARLARI TESPİT ET (Hough Line Transform)
    # SADECE YATAY çizgileri filtrele (borular genelde yatay)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                            minLineLength=40, maxLineGap=10)
    
    # Düz kenar maskesi oluştur - sadece YATAY çizgiler için
    straight_edge_mask = np.zeros_like(gray_image)
    exclusion_distance = 25  # Düz kenarlardan bu kadar piksel uzakta çatlak ara
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Çizginin açısını hesapla
            if x2 - x1 != 0:
                angle = np.abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
            else:
                angle = 90  # Dikey çizgi
            
            # Sadece yatay çizgileri al (açı < 30 derece)
            # Dikey çizgiler (çatlaklar) korunacak
            if angle < 30:
                # Yatay çizgiyi kalın çiz (exclusion zone)
                cv2.line(straight_edge_mask, (x1, y1), (x2, y2), 255, exclusion_distance * 2)
    
    # Maskeyi genişlet
    exclude_kernel = np.ones((exclusion_distance, exclusion_distance), np.uint8)
    straight_edge_mask = cv2.dilate(straight_edge_mask, exclude_kernel, iterations=1)
    
    # 7. ORB Feature Detection
    orb = cv2.ORB_create(nfeatures=orb_features)
    keypoints, descriptors = orb.detectAndCompute(closing, None)
    
    # 8. Closing'den konturları bul (akıllı filtreleme ile)
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Görüntü boyutları
    img_height, img_width = gray_image.shape[:2]
    
    # Akıllı filtreleme - düz kenarlardan uzak olanları çatlak olarak al
    cracks = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue
            
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Konturun düz kenar maskesine ne kadar yakın olduğunu kontrol et
        # Kontur merkezini al
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        # Konturu kapsayan bölgedeki düz kenar yoğunluğunu hesapla
        contour_mask = np.zeros_like(gray_image)
        cv2.drawContours(contour_mask, [contour], 0, 255, -1)
        
        # Kontur alanının ne kadarı düz kenar bölgesinde?
        overlap = cv2.bitwise_and(contour_mask, straight_edge_mask)
        overlap_area = np.sum(overlap > 0)
        contour_pixel_count = np.sum(contour_mask > 0)
        
        if contour_pixel_count > 0:
            overlap_ratio = overlap_area / contour_pixel_count
        else:
            overlap_ratio = 0
        
        # Eğer konturun %50'den fazlası düz kenar bölgesindeyse = boru/kenar, atla
        if overlap_ratio > 0.5:
            continue
        
        # Perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Convex hull ile solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Extent: kontur alanı / bbox alanı
        bbox_area = w * h
        extent = float(area) / bbox_area if bbox_area > 0 else 0
        
        # Rotated rectangle analizi
        rect_fill = 0
        if len(contour) >= 5:
            rect = cv2.minAreaRect(contour)
            rect_w, rect_h = rect[1]
            rect_area = rect_w * rect_h
            if rect_area > 0:
                rect_fill = float(area) / rect_area
        
        # Kontur düzgünlüğü
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertex_count = len(approx)
        
        # Yatay mı dikey mi?
        is_horizontal = w > h * 2
        is_vertical = h > w * 1.5
        
        # ===== EK DÜZGÜN ŞEKİL FİLTRESİ =====
        # Düz kenar kontrolünden geçen ama hala düzgün olan şekilleri filtrele
        is_too_regular = False
        
        # Çok düzgün dikdörtgen + yatay
        if rect_fill > 0.6 and solidity > 0.6 and is_horizontal:
            is_too_regular = True
        
        # Çok yüksek extent + yatay
        if extent > 0.55 and is_horizontal:
            is_too_regular = True
        
        # Yatay ve düzgün
        if is_horizontal and rect_fill > 0.45 and solidity > 0.5:
            is_too_regular = True
        
        # ===== ÇATLAK DOĞRULAMA =====
        is_crack = False
        
        # Dikey şekiller çatlak olabilir
        if is_vertical:
            is_crack = True
        
        # Düzensiz şekiller çatlak
        if solidity < 0.5:
            is_crack = True
        
        # Çok fazla köşe = düzensiz kenar
        if vertex_count > 8:
            is_crack = True
        
        # Düşük rect_fill = düzensiz
        if rect_fill < 0.45:
            is_crack = True
        
        # Düşük extent
        if extent < 0.4:
            is_crack = True
        
        # Karar: Çatlak özellikleri var VE çok düzgün değil
        if is_crack and not is_too_regular:
            cracks.append(contour)
    
    # 9. Featured image oluştur (görselleştirme için)
    featured_img = cv2.drawKeypoints(closing, keypoints, None, color=(0, 255, 0))
    
    # 10. Özellikleri analiz et
    properties = analyze_crack_properties(cracks)
    
    return {
        'log_transform': img_log,
        'bilateral': bilateral,
        'edges': edges,
        'morphology': closing,
        'straight_edge_mask': straight_edge_mask,  # Debug için
        'featured_img': featured_img,
        'keypoints': keypoints,
        'keypoint_count': len(keypoints),
        'cracks': cracks,
        'properties': properties
    }
