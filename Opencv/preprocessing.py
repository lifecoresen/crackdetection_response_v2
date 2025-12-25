"""
Görüntü Ön İşleme Modülü
Bina duvarları ve kolonlar üzerindeki çatlak ve nem tespiti için görüntü hazırlama
"""

import cv2
import numpy as np


def load_image(image_path, resize_width=None):
    """
    Görüntüyü yükle ve opsiyonel olarak yeniden boyutlandır
    
    Args:
        image_path (str): Görüntü dosyasının yolu
        resize_width (int): Yeniden boyutlandırma genişliği (None ise orijinal boyut)
    
    Returns:
        numpy.ndarray: Yüklenen görüntü
    """
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")
    
    if resize_width is not None:
        height = int(image.shape[0] * (resize_width / image.shape[1]))
        image = cv2.resize(image, (resize_width, height))
    
    return image


def to_grayscale(image):
    """
    Renkli görüntüyü gri tona dönüştür
    
    Args:
        image (numpy.ndarray): Giriş görüntüsü
    
    Returns:
        numpy.ndarray: Gri tonlu görüntü
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    İkili filtre uygula - kenarları korurken gürültü azalt
    
    Args:
        image (numpy.ndarray): Giriş görüntüsü
        d (int): Çap (9 önerilir)
        sigma_color (int): Renk uzayında sigma
        sigma_space (int): Koordinat uzayında sigma
    
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if len(image.shape) == 3:  # Renkli görüntü ise
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    else:  # Gri görüntü ise
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_clahe(image):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) uygula
    Kontrast ve detay tespitini iyileştir
    
    Args:
        image (numpy.ndarray): Gri tonlu görüntü
    
    Returns:
        numpy.ndarray: İyileştirilmiş görüntü
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def apply_morphological_operations(image, operation_type='open', kernel_size=5):
    """
    Morfolojik işlemler uygula
    
    Args:
        image (numpy.ndarray): Giriş görüntüsü
        operation_type (str): 'open', 'close', 'erode', 'dilate'
        kernel_size (int): Kernel boyutu
    
    Returns:
        numpy.ndarray: İşlem uygulanmış görüntü
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation_type == 'open':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation_type == 'close':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    elif operation_type == 'erode':
        return cv2.erode(image, kernel)
    elif operation_type == 'dilate':
        return cv2.dilate(image, kernel)
    else:
        return image


def normalize_image(image):
    """
    Görüntüyü normalize et (0-255 aralığına)
    
    Args:
        image (numpy.ndarray): Giriş görüntüsü
    
    Returns:
        numpy.ndarray: Normalize edilmiş görüntü
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def preprocess_image(image_path, resize_width=800, clahe_clip_limit=5.0, clahe_tile_size=(4, 4)):
    """
    Tam ön işleme pipeline'ı
    
    Args:
        image_path (str): Görüntü dosyası yolu
        resize_width (int): Yeniden boyutlandırma genişliği
        clahe_clip_limit (float): CLAHE clip limit
        clahe_tile_size (tuple): CLAHE tile size
    
    Returns:
        tuple: (orijinal görüntü, gri görüntü, ön işlenmiş görüntü)
    """
    # Görüntüyü yükle
    original = load_image(image_path, resize_width=resize_width)
    
    # Gri tona dönüştür
    gray = to_grayscale(original)
    
    # İkili filtreyi uygula
    filtered = apply_bilateral_filter(gray)
    
    # CLAHE uygula - parametreleri güncelle
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_size)
    enhanced = clahe.apply(filtered)
    
    # Normalize et
    processed = normalize_image(enhanced)
    
    return original, gray, processed
