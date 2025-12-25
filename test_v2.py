"""
Test V2 - Yeni Çatlak ve Nem Tespiti Sistemi
"""

import cv2
import sys
import os
import numpy as np

# V2 modülünü import et
from Opencv.crack_detection_v2 import (
    detect_cracks_v2, 
    detect_moisture_v2, 
    draw_results_v2,
    analyze_results_v2
)


def process_image_v2(image_path, output_dir='results'):
    """
    V2 algoritması ile görüntü işle
    """
    print("\n" + "="*60)
    print("🏗️  BINA HASAR TESPİT SİSTEMİ V2")
    print("="*60)
    
    # Görüntü yükle
    print("\n📥 Görüntü yükleniyor...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Görüntü yüklenemedi: {image_path}")
        return None
    
    # Boyutlandır
    max_width = 1000
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, None, fx=scale, fy=scale)
        print(f"  Boyut: {w}x{h} -> {image.shape[1]}x{image.shape[0]}")
    else:
        print(f"  Boyut: {w}x{h}")
    
    # Çatlak tespiti konfigürasyonu
    crack_config = {
        'ridge_scales': [1.0, 1.5, 2.0],
        'dark_threshold': 15,          # Düşük = daha hassas
        'min_crack_length': 30,        # Minimum çatlak uzunluğu
        'max_crack_width': 20,         # Maksimum çatlak genişliği
        'min_aspect_ratio': 2.0,       # En-boy oranı
        'use_lsd': True,
        'combine_methods': True
    }
    
    # Nem tespiti konfigürasyonu
    moisture_config = {
        'color_sensitivity': 1.0,      # Düşük = daha hassas
        'texture_sensitivity': 1.2,
        'min_region_area': 500,
        'max_region_area': 150000,
    }
    
    # Çatlak tespiti
    print("\n🔍 Çatlaklar tespit ediliyor...")
    print("   (Ridge Detection + Dark Lines + LSD)")
    cracks, crack_vis = detect_cracks_v2(image, crack_config)
    print(f"   ✓ {len(cracks)} çatlak tespit edildi")
    
    # Nem tespiti
    print("\n💧 Nem/Dökülen sıva tespit ediliyor...")
    print("   (Renk Anomalisi + Texture Analizi)")
    moisture, moisture_vis = detect_moisture_v2(image, moisture_config)
    print(f"   ✓ {len(moisture)} bölge tespit edildi")
    
    # Sonuçları çiz
    result = draw_results_v2(image, cracks, moisture)
    
    # İstatistikler
    crack_stats, moisture_stats = analyze_results_v2(cracks, moisture)
    
    # Kaydet
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    output_path = os.path.join(output_dir, f"{base_name}_v2_result.jpg")
    cv2.imwrite(output_path, result)
    print(f"\n💾 Sonuç kaydedildi: {output_path}")
    
    # Debug görüntüleri
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_v2_cracks.jpg"), crack_vis)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_v2_moisture.jpg"), moisture_vis)
    
    # Rapor
    print("\n" + "="*60)
    print("📊 RAPOR")
    print("="*60)
    print(f"\n🔍 ÇATLAK TESPİTİ:")
    print(f"   Sayı: {crack_stats['count']}")
    print(f"   Toplam Alan: {crack_stats['total_area']:.0f} px²")
    print(f"   Toplam Uzunluk: {crack_stats['total_length']:.0f} px")
    
    print(f"\n💧 NEM/DÖKÜLEN SIVA:")
    print(f"   Sayı: {moisture_stats['count']}")
    print(f"   Toplam Alan: {moisture_stats['total_area']:.0f} px²")
    
    print("\n" + "="*60)
    
    return result


def test_all_images(image_dir='images'):
    """
    Tüm görüntüleri test et
    """
    if not os.path.exists(image_dir):
        print(f"❌ Klasör bulunamadı: {image_dir}")
        return
    
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\n📁 {len(images)} görüntü bulundu\n")
    
    for img_name in sorted(images):
        img_path = os.path.join(image_dir, img_name)
        print(f"\n{'='*60}")
        print(f"📷 İşleniyor: {img_name}")
        process_image_v2(img_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\nKullanım:")
        print("  python test_v2.py <görüntü_yolu>")
        print("  python test_v2.py --all  (tüm görüntüleri test et)")
        sys.exit(1)
    
    if sys.argv[1] == '--all':
        test_all_images()
    else:
        process_image_v2(sys.argv[1])
