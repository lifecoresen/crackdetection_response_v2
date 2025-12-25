"""
Özel Konfigürasyon - Parametreleri Ayarlama
Çatlak ve nem tespiti hassasiyetini kontrol et
"""

# Çatlak Tespiti Parametreleri (Adaptive Threshold - İyileştirilmiş Yöntem)
CRACK_DETECTION_CONFIG = {
    # Yöntem seçimi: 'adaptive', 'canny' veya 'orb'
    'method': 'orb',  # 'canny', 'adaptive' veya 'orb'
    
    # Adaptive Threshold yöntemi parametreleri
    'blur_kernel': 5,              # Gaussian blur kernel (yüksek = daha az gürültü)
    'adaptive_block': 21,          # Adaptive threshold block size
    'adaptive_constant': 5,        # Adaptive threshold sabit
    'morph_kernel': 3,             # Morfoloji kernel (küçük = ince çatlaklar)
    'morph_iterations': 2,         # Morfoloji işlem sayısı
    'min_contour_area': 150,       # Minimum kontur alanı (px²) - yükseltildi
    
    # Şekil filtreleme parametreleri (boruları ayırt etmek için)
    'max_solidity': 0.25,          # Maksimum doluluk (düşük = çatlak, yüksek = boru)
    'max_aspect_ratio': 10.0,      # Maksimum en-boy oranı
    
    # ORB yöntemi parametreleri (yeni - gelişmiş)
    'orb_features': 3000,          # ORB feature sayısı
    'bilateral_d': 5,              # Bilateral filter diameter
    'bilateral_sigma': 50,         # Bilateral filter sigma
    'canny_threshold1': 30,        # Canny alt eşik - yükseltildi
    'canny_threshold2': 100,       # Canny üst eşik - yükseltildi
}

# Nem Tespiti Parametreleri
MOISTURE_DETECTION_CONFIG = {
    'dark_threshold': 100,       # Koyu alan eşiği - yükseltildi
    'min_region_area': 500,      # Minimum nem alanı (piksel²) - yükseltildi
    'morphology_kernel': 9,      # Daha büyük kernel
    'window_size': 31,           # Kontrast analizi pencere boyutu
}

# Görüntü Ön İşleme Parametreleri
PREPROCESSING_CONFIG = {
    'bilateral_filter_d': 5,      # Daha az filtreleme
    'bilateral_filter_sigma_color': 50,
    'bilateral_filter_sigma_space': 50,
    'clahe_clip_limit': 6.0,      # Maksimum kontrast iyileştirme
    'clahe_tile_size': (2, 2),    # Çok küçük tile = maksimum detay
    'resize_width': 800,
}

# Görselleştirme Parametreleri
VISUALIZATION_CONFIG = {
    'crack_color': (0, 255, 0),      # Yeşil - Çatlaklar
    'moisture_color': (0, 0, 255),   # Kırmızı - Nem
    'text_color': (255, 255, 255),   # Beyaz - Yazı
    'line_thickness': 2,
}

print("Konfigürasyon yüklendi!")
