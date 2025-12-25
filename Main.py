"""
Ana Program - Bina Duvar ve Kolon Teşhis Sistemi
Çatlak ve Nem İzlerini Tespit Eden Görüntü İşleme Uygulaması

Kullanım:
    python Main.py <görüntü_yolu>
    
Örnek:
    python Main.py images/building.jpg
"""

import sys
import os
import cv2

# Modülleri import et
from Opencv.preprocessing import preprocess_image
from Opencv.crack_detection import detect_cracks_pipeline, detect_cracks_adaptive_pipeline, detect_cracks_orb_pipeline, draw_cracks
from Opencv.moisture_detection import detect_moisture_pipeline, draw_moisture_regions
from Opencv.utils import (
    save_image, display_multiple_images, draw_statistics,
    create_comparison_image, get_image_statistics, print_report,
    cleanup_windows
)
from config import CRACK_DETECTION_CONFIG, MOISTURE_DETECTION_CONFIG, PREPROCESSING_CONFIG


def process_building_image(image_path, output_dir='results'):
    """
    Bina görüntüsünü işle ve çatlak + nem tespiti yap
    
    Args:
        image_path (str): İnput görüntü yolu
        output_dir (str): Çıkış dosyaları dizini
    
    Returns:
        dict: Tüm sonuçlar
    """
    # Input dosya adından output dosya adı oluştur
    input_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(input_filename)
    output_filename = f"{name}_result{ext}"
    
    print("\n" + "="*60)
    print("🏗️  BINA DUVAR VE KOLON TEŞHİS SİSTEMİ")
    print("="*60)
    
    # 1. GÖRÜNTÜ YÜKLEMESİ VE ÖN İŞLEME
    print("\n📥 Adım 1: Görüntü yükleniyor...")
    try:
        original, gray, processed = preprocess_image(
            image_path, 
            resize_width=PREPROCESSING_CONFIG['resize_width'],
            clahe_clip_limit=PREPROCESSING_CONFIG['clahe_clip_limit'],
            clahe_tile_size=PREPROCESSING_CONFIG['clahe_tile_size']
        )
        print("✓ Görüntü başarıyla yüklendi ve ön işlemesi yapıldı")
    except FileNotFoundError as e:
        print(f"✗ Hata: {e}")
        return None
    
    # 2. ÇATLAK TESPİTİ
    print("\n🔍 Adım 2: Çatlaklar tespit ediliyor...")
    
    method = CRACK_DETECTION_CONFIG.get('method', 'adaptive')
    
    # ORB Feature Detection yöntemi (yeni - gelişmiş)
    if method == 'orb':
        print("  (ORB Feature Detection yöntemi)")
        crack_results = detect_cracks_orb_pipeline(
            processed,
            blur_kernel=CRACK_DETECTION_CONFIG.get('blur_kernel', 3),
            bilateral_d=CRACK_DETECTION_CONFIG.get('bilateral_d', 5),
            bilateral_sigma=CRACK_DETECTION_CONFIG.get('bilateral_sigma', 75),
            canny_threshold1=CRACK_DETECTION_CONFIG.get('canny_threshold1', 100),
            canny_threshold2=CRACK_DETECTION_CONFIG.get('canny_threshold2', 200),
            morph_kernel=CRACK_DETECTION_CONFIG.get('morph_kernel', 5),
            orb_features=CRACK_DETECTION_CONFIG.get('orb_features', 1500),
            min_contour_area=CRACK_DETECTION_CONFIG.get('min_contour_area', 200),
            max_solidity=CRACK_DETECTION_CONFIG.get('max_solidity', 0.5),
            max_aspect_ratio=CRACK_DETECTION_CONFIG.get('max_aspect_ratio', 8.0)
        )
    # Adaptive Threshold yöntemi
    elif method == 'adaptive':
        print("  (Adaptive Threshold yöntemi)")
        crack_results = detect_cracks_adaptive_pipeline(
            processed,
            blur_kernel=CRACK_DETECTION_CONFIG['blur_kernel'],
            adaptive_block=CRACK_DETECTION_CONFIG['adaptive_block'],
            adaptive_constant=CRACK_DETECTION_CONFIG['adaptive_constant'],
            morph_kernel=CRACK_DETECTION_CONFIG['morph_kernel'],
            morph_iterations=CRACK_DETECTION_CONFIG['morph_iterations'],
            min_contour_area=CRACK_DETECTION_CONFIG['min_contour_area'],
            max_solidity=CRACK_DETECTION_CONFIG.get('max_solidity', 0.5),
            max_aspect_ratio=CRACK_DETECTION_CONFIG.get('max_aspect_ratio', 8.0)
        )
    else:
        # Fallback: Canny yöntemi
        print("  (Canny Edge Detection yöntemi)")
        crack_results = detect_cracks_pipeline(
            processed, 
            min_contour_area=CRACK_DETECTION_CONFIG['min_contour_area'],
            canny_threshold1=CRACK_DETECTION_CONFIG['canny_threshold1'],
            canny_threshold2=CRACK_DETECTION_CONFIG['canny_threshold2']
        )
    
    crack_image = draw_cracks(original, crack_results['cracks'], color=(0, 255, 0), thickness=2)
    crack_stats = crack_results['properties']
    print(f"✓ {crack_stats['total_cracks']} çatlak tespit edildi")
    
    # 3. NEM TESPİTİ
    print("\n💧 Adım 3: Nem izleri tespit ediliyor...")
    moisture_results = detect_moisture_pipeline(
        original, gray, 
        min_region_area=MOISTURE_DETECTION_CONFIG['min_region_area']
    )
    moisture_image = draw_moisture_regions(original, moisture_results['regions'], 
                                          color=(0, 0, 255), thickness=2, draw_filled=False)
    moisture_stats = moisture_results['properties']
    print(f"✓ {moisture_stats['total_regions']} nem bölgesi tespit edildi")
    
    # 4. BİRLEŞTİRİLMİŞ SONUÇ
    print("\n🎯 Adım 4: Birleştirilmiş sonuç oluşturuluyor...")
    combined_image = original.copy()
    
    # Çatlakları yeşille çiz
    for crack in crack_results['cracks']:
        cv2.drawContours(combined_image, [crack], 0, (0, 255, 0), 2)
    
    # Nem bölgelerini kırmızıyla çiz
    for moisture in moisture_results['regions']:
        cv2.drawContours(combined_image, [moisture], 0, (0, 0, 255), 2)
    
    # İstatistikleri ve açıklamaları ekle
    h, w = combined_image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Arka plan kutusu için yarı saydam overlay
    overlay = combined_image.copy()
    cv2.rectangle(overlay, (5, 5), (280, 100), (0, 0, 0), -1)
    combined_image = cv2.addWeighted(overlay, 0.6, combined_image, 0.4, 0)
    
    # Başlık
    cv2.putText(combined_image, "TESPIT SONUCLARI", (10, 25), font, 0.6, (255, 255, 255), 2)
    
    # Yeşil - Çatlak açıklaması
    cv2.rectangle(combined_image, (10, 35), (30, 55), (0, 255, 0), -1)  # Yeşil kutu
    cv2.putText(combined_image, f"Catlak: {crack_stats['total_cracks']} adet", (35, 50), font, 0.5, (0, 255, 0), 2)
    
    # Kırmızı - Nem/Dökülen sıva açıklaması
    cv2.rectangle(combined_image, (10, 60), (30, 80), (0, 0, 255), -1)  # Kırmızı kutu
    cv2.putText(combined_image, f"Dokulen Siva: {moisture_stats['total_regions']} adet", (35, 75), font, 0.5, (0, 0, 255), 2)
    
    print("✓ Birleştirilmiş sonuç hazır")
    
    # 5. SONUÇLARI KAYDET
    print("\n💾 Adım 5: Sonuçlar kaydediliyor...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Sadece birleştirilmiş sonucu kaydet
    save_image(combined_image, output_dir, output_filename)
    
    # 6. RAPOR YAZDIR
    image_stats = get_image_statistics(gray)
    print_report(crack_stats, moisture_stats, image_stats)
    
    # Sonuçları döndür
    results = {
        'original': original,
        'gray': gray,
        'processed': processed,
        'crack_results': crack_results,
        'crack_image': crack_image,
        'crack_stats': crack_stats,
        'moisture_results': moisture_results,
        'moisture_image': moisture_image,
        'moisture_stats': moisture_stats,
        'combined_image': combined_image,
        'image_stats': image_stats
    }
    
    return results


def display_results(results):
    """
    Sonuçları görüntüle
    
    Args:
        results (dict): İşleme sonuçları
    """
    print("\n📺 Sonuçlar gösteriliyor...")
    print("Pencereyi kapatmak için herhangi bir tuşa basın...")
    
    # Karşılaştırma görüntüleri oluştur
    comparison1 = create_comparison_image(results['original'], results['crack_image'], 
                                         "Orijinal", "Çatlaklar")
    comparison2 = create_comparison_image(results['original'], results['moisture_image'], 
                                         "Orijinal", "Nem İzleri")
    
    # Tüm sonuçları göster
    cv2.imshow('Çatlak Tespiti Karşılaştırması', comparison1)
    cv2.imshow('Nem Tespiti Karşılaştırması', comparison2)
    cv2.imshow('Birleştirilmiş Sonuç', results['combined_image'])
    
    cv2.waitKey(0)
    cleanup_windows()


def main():
    """Ana fonksiyon"""
    
    # Komut satırı argümanlarını kontrol et
    if len(sys.argv) < 2:
        print("\n📌 Kullanım:")
        print("   python Main.py <görüntü_yolu>")
        print("\n📝 Örnek:")
        print("   python Main.py images/building.jpg")
        print("\nÖrnek bir test görüntüsü oluşturmak istiyorsanız:")
        print("   python Main.py --create-test")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Test görüntüsü oluştur (--create-test flag'ı ile)
    if image_path == '--create-test':
        create_test_image()
        image_path = 'images/test_image.jpg'
        print(f"✓ Test görüntüsü oluşturuldu: {image_path}")
    
    # Görüntüyü işle
    results = process_building_image(image_path)
    
    if results is None:
        print("\n✗ İşlem başarısız oldu")
        sys.exit(1)
    
    print("\n✅ Tüm işlemler tamamlandı!")


def create_test_image():
    """Test için örnek bir görüntü oluştur"""
    import numpy as np
    
    os.makedirs('images', exist_ok=True)
    
    # Gri görüntü oluştur
    img = np.ones((400, 600, 3), dtype=np.uint8) * 150
    
    # Çatlakları simüle et (koyu çizgiler)
    cv2.line(img, (100, 100), (200, 250), (80, 80, 80), 3)
    cv2.line(img, (150, 80), (320, 180), (70, 70, 70), 2)
    cv2.line(img, (400, 150), (500, 300), (60, 60, 60), 2)
    
    # Nem izlerini simüle et (benekli koyu alanlar)
    cv2.circle(img, (300, 200), 40, (100, 100, 100), -1)
    cv2.circle(img, (450, 250), 35, (105, 105, 105), -1)
    
    # Gürültü ekle
    noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    cv2.imwrite('images/test_image.jpg', img)


if __name__ == '__main__':
    main()
