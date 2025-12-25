"""
Yardımcı Fonksiyonlar Modülü
Ortak işlevler ve görselleme araçları
"""

import cv2
import numpy as np
import os


def save_image(image, output_path, filename):
    """
    Görüntüyü dosyaya kaydet
    
    Args:
        image (numpy.ndarray): Kaydedilecek görüntü
        output_path (str): Çıkış dizini
        filename (str): Dosya adı
    
    Returns:
        bool: Başarı durumu
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    full_path = os.path.join(output_path, filename)
    success = cv2.imwrite(full_path, image)
    
    if success:
        print(f"✓ Görüntü kaydedildi: {full_path}")
    else:
        print(f"✗ Görüntü kaydetme başarısız: {full_path}")
    
    return success


def display_image(image, window_name="Görüntü", wait_time=0):
    """
    Görüntüyü ekranda göster
    
    Args:
        image (numpy.ndarray): Gösterilecek görüntü
        window_name (str): Pencere başlığı
        wait_time (int): Bekleme süresi (ms), 0 = sonsuz
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(wait_time)


def display_multiple_images(images_dict, window_size=(400, 400)):
    """
    Birden fazla görüntüyü yan yana göster
    
    Args:
        images_dict (dict): {başlık: görüntü} şeklinde sözlük
        window_size (tuple): Pencere boyutu
    """
    cv2.namedWindow('Sonuçlar', cv2.WINDOW_NORMAL)
    
    # Görüntüleri yatay olarak birleştir
    combined = None
    for title, image in images_dict.items():
        # Görüntüyü yeniden boyutlandır
        resized = cv2.resize(image, window_size)
        
        # Başlığı ekle
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(resized, title, (10, 30), font, 0.7, (0, 255, 0), 2)
        
        # Birleştir
        if combined is None:
            combined = resized
        else:
            combined = np.hstack([combined, resized])
    
    cv2.imshow('Sonuçlar', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_statistics(image, stats, position=(10, 30), font_size=0.6, color=(0, 255, 0)):
    """
    Tespit istatistiklerini görüntüye çiz
    
    Args:
        image (numpy.ndarray): Hedef görüntü
        stats (dict): İstatistikler sözlüğü
        position (tuple): Başlangıç pozisyonu
        font_size (float): Font boyutu
        color (tuple): Renk (BGR)
    
    Returns:
        numpy.ndarray: İstatistikler eklenen görüntü
    """
    result = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = position[1]
    
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
            cv2.putText(result, text, (position[0], y_offset), font, font_size, color, 1)
            y_offset += 25
    
    return result


def apply_heatmap(image, colormap_type=cv2.COLORMAP_JET):
    """
    Görüntüye ısı haritası renklendirmesi uygula
    
    Args:
        image (numpy.ndarray): Gri görüntü
        colormap_type: OpenCV colormap sabiti
    
    Returns:
        numpy.ndarray: Renklendirilen görüntü
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    heatmap = cv2.applyColorMap(image, colormap_type)
    return heatmap


def create_comparison_image(original, processed, label1="Orijinal", label2="İşlenmiş"):
    """
    Orijinal ve işlenmiş görüntüyü yan yana karşılaştır
    
    Args:
        original (numpy.ndarray): Orijinal görüntü
        processed (numpy.ndarray): İşlenmiş görüntü
        label1 (str): Birinci etiket
        label2 (str): İkinci etiket
    
    Returns:
        numpy.ndarray: Karşılaştırma görüntüsü
    """
    # Boyutları eşitle
    h = max(original.shape[0], processed.shape[0])
    w = max(original.shape[1], processed.shape[1])
    
    orig_resized = cv2.resize(original, (w, h))
    proc_resized = cv2.resize(processed, (w, h))
    
    # Gri görüntüleri renkli yap
    if len(proc_resized.shape) == 2:
        proc_resized = cv2.cvtColor(proc_resized, cv2.COLOR_GRAY2BGR)
    
    # Etiketleri ekle
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(orig_resized, label1, (10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.putText(proc_resized, label2, (10, 30), font, 0.8, (0, 255, 0), 2)
    
    # Yan yana birleştir
    comparison = np.hstack([orig_resized, proc_resized])
    return comparison


def get_image_statistics(image):
    """
    Görüntü istatistiklerini al
    
    Args:
        image (numpy.ndarray): Görüntü
    
    Returns:
        dict: İstatistikler
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return {
        'mean': np.mean(image),
        'std': np.std(image),
        'min': np.min(image),
        'max': np.max(image),
        'width': image.shape[1],
        'height': image.shape[0]
    }


def print_report(crack_stats, moisture_stats, image_stats):
    """
    Detaylı rapor yazdır
    
    Args:
        crack_stats (dict): Çatlak istatistikleri
        moisture_stats (dict): Nem istatistikleri
        image_stats (dict): Görüntü istatistikleri
    """
    print("\n" + "="*60)
    print("BINA DUVAR VE KOLON TEŞHİS RAPORU")
    print("="*60)
    
    print("\n📊 GÖRÜNTÜ BİLGİSİ:")
    print(f"  Boyut: {image_stats['width']}x{image_stats['height']}")
    print(f"  Ortalama Parlaklık: {image_stats['mean']:.2f}")
    print(f"  Standart Sapma: {image_stats['std']:.2f}")
    
    print("\n🔍 ÇATLAK TESPİTİ:")
    print(f"  Bulunan Çatlak Sayısı: {crack_stats['total_cracks']}")
    print(f"  Toplam Çatlak Alanı: {crack_stats['total_area']:.2f} px²")
    if crack_stats['total_cracks'] > 0:
        print(f"  Ortalama Çatlak Alanı: {crack_stats['average_area']:.2f} px²")
        print(f"  Maksimum Çatlak Alanı: {crack_stats['max_area']:.2f} px²")
        print(f"  Ortalama Çatlak Uzunluğu: {crack_stats['average_length']:.2f} px")
    
    print("\n💧 NEM TESPİTİ:")
    print(f"  Bulunan Nem Bölgesi Sayısı: {moisture_stats['total_regions']}")
    print(f"  Toplam Nem Alanı: {moisture_stats['total_area']:.2f} px²")
    if moisture_stats['total_regions'] > 0:
        print(f"  Ortalama Nem Bölgesi Alanı: {moisture_stats['average_area']:.2f} px²")
        print(f"  Maksimum Nem Bölgesi Alanı: {moisture_stats['max_area']:.2f} px²")
    
    print("\n" + "="*60 + "\n")


def cleanup_windows():
    """Tüm OpenCV pencerelerini kapat"""
    cv2.destroyAllWindows()
