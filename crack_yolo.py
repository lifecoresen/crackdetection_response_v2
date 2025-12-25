"""
YOLOv8 ile Çatlak Tespiti
Segmentasyon modeli kullanarak çatlakları tespit eder
"""

import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO


def _normalize_u8(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32)
    mn = float(np.min(img_f))
    mx = float(np.max(img_f))
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img_f - mn) * (255.0 / (mx - mn))
    return np.clip(out, 0, 255).astype(np.uint8)


def _morph_skeleton(binary: np.ndarray) -> np.ndarray:
    """Morphological skeletonization for a binary mask (uint8 0/255)."""
    img = (binary > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break

    return skel


def _compute_crack_response(gray_u8: np.ndarray) -> np.ndarray:
    """Compute a crack-likeliness response map (uint8 0..255)."""
    # Denoise while preserving edges.
    den = cv2.bilateralFilter(gray_u8, 7, 60, 60)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g = clahe.apply(den)

    # Multi-scale black-hat emphasizes thin dark lines.
    blackhat_sum = np.zeros_like(g, dtype=np.float32)
    for k in (9, 13, 17):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)
        blackhat_sum += bh.astype(np.float32)

    # Local darkness: (local_mean - pixel) highlights locally darker structures.
    local = cv2.blur(g, (21, 21))
    darker = cv2.subtract(local, g)

    # Gradient magnitude helps when cracks are more edge-like than dark.
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(sx, sy)

    # Weighted fusion (kept conservative to avoid flooding).
    resp = 0.55 * blackhat_sum + 0.30 * darker.astype(np.float32) + 0.15 * grad
    return _normalize_u8(resp)


def _threshold_by_percentile(resp_u8: np.ndarray, percentile: float) -> np.ndarray:
    flat = resp_u8.reshape(-1)
    # Avoid extreme thresholds for low-contrast images.
    thr = int(np.percentile(flat, percentile))
    thr = max(thr, 10)
    _, binary = cv2.threshold(resp_u8, thr, 255, cv2.THRESH_BINARY)
    return binary


def _filter_by_skeleton(binary_mask: np.ndarray, min_length_px: int, max_width_px: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Filter candidate crack mask by skeleton length and estimated width."""
    mask = (binary_mask > 0).astype(np.uint8) * 255

    # Clean tiny speckles but keep thin structures.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    skel = _morph_skeleton(mask)

    # Distance transform gives radius; width≈2*radius at skeleton pixels.
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((skel > 0).astype(np.uint8), connectivity=8)
    keep_skel = np.zeros_like(skel)

    kept = 0
    lengths = []
    widths = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_length_px:
            continue

        ys, xs = np.where(labels == label)
        if ys.size == 0:
            continue

        w = float(np.mean(dist[ys, xs]) * 2.0)
        if w > max_width_px:
            continue

        keep_skel[ys, xs] = 255
        kept += 1
        lengths.append(area)
        widths.append(w)

    # Reconstruct a slightly thicker mask from skeleton for visualization/mask output.
    recon = cv2.dilate(keep_skel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    summary = {
        'components': kept,
        'skeleton_length_px': int(np.sum(keep_skel > 0)),
        'avg_width_px': float(np.mean(widths)) if widths else 0.0,
        'avg_component_length_px': float(np.mean(lengths)) if lengths else 0.0,
    }
    return recon, keep_skel, summary


def detect_cracks_yolo(image_path, output_dir='results', conf_threshold=0.25):
    """
    YOLOv8 segmentasyon modeli ile çatlak tespiti
    
    Args:
        image_path: Görüntü yolu
        output_dir: Çıktı klasörü
        conf_threshold: Güven eşiği (0-1)
    """
    print("\n" + "="*60)
    print("🔍 YOLO ÇaTLAK TESPİT SİSTEMİ")
    print("="*60)
    
    # Görüntü yükle
    print("\n📥 Görüntü yükleniyor...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Görüntü yüklenemedi: {image_path}")
        return None
    
    h, w = image.shape[:2]
    print(f"   Boyut: {w}x{h}")
    
    # YOLOv8 segmentasyon modeli yükle
    print("\n🤖 YOLO modeli yükleniyor...")
    model = YOLO('yolov8n-seg.pt')  # Nano segmentasyon modeli
    
    # Inference yap
    print("\n🔍 Çatlak analizi yapılıyor...")
    results = model(image, conf=conf_threshold, verbose=False)
    
    # Sonuçları işle
    result = results[0]
    result_image = image.copy()
    
    crack_count = 0
    total_area = 0
    
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes
        
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            # Maske boyutunu görüntüye uyarla
            mask_resized = cv2.resize(mask, (w, h))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            
            # Alan hesapla
            area = np.sum(mask_binary)
            total_area += area
            
            # Kontur bul ve çiz
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Yeşil renk ile çiz
            cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
            
            # Yarı saydam maske overlay
            overlay = result_image.copy()
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), -1)
            result_image = cv2.addWeighted(result_image, 0.7, overlay, 0.3, 0)
            
            crack_count += 1
    
    # Sonuçları kaydet
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_yolo.jpg")
    
    # İstatistik ekle
    cv2.rectangle(result_image, (10, 10), (300, 60), (0, 0, 0), -1)
    cv2.putText(result_image, f"Tespit: {crack_count} bolge", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, result_image)
    
    # Rapor
    print(f"\n✅ Tespit tamamlandı!")
    print(f"   Bulunan bölge: {crack_count}")
    print(f"   Toplam alan: {total_area} px²")
    print(f"   Sonuç: {output_path}")
    
    return result_image, crack_count


def create_severity_colormap():
    """Çatlak şiddeti için renk haritası oluştur"""
    # Sarı -> Turuncu -> Kırmızı gradyanı
    colors = []
    for i in range(256):
        if i < 128:
            # Sarı -> Turuncu
            r = 255
            g = 255 - i
            b = 0
        else:
            # Turuncu -> Kırmızı
            r = 255
            g = 127 - (i - 128)
            b = 0
        colors.append([b, g, r])
    return np.array(colors, dtype=np.uint8)


def detect_cracks_edge_based(image_path, output_dir='results'):
    """
    Adaptif çatlak tespiti (daha anlamlı test/çıktı):
    - Çoklu response map (black-hat + local darkness + gradient)
    - Percentile tabanlı threshold (görüntüye göre otomatik)
    - Bu sürümde: binary'de yakalananı direkt çiz (ek filtre yok)
    
    Çıktılar:
    - *_panel.jpg: original | response-color | binary | overlay
    """
    print("\n" + "="*60)
    print("🔍 ÇATLAK TESPİT SİSTEMİ")
    print("="*60)
    
    # Görüntü yükle
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Görüntü yüklenemedi: {image_path}")
        return None
    
    h, w = image.shape[:2]
    print(f"\n📥 Görüntü: {w}x{h}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("🔍 Response haritası hesaplanıyor...")
    resp = _compute_crack_response(gray)

    # Threshold selection: prefer high percentile to reduce noise, but keep sensitivity.
    # Images that are very clean/bright (e.g. low std) need lower percentile.
    std = float(np.std(gray))
    percentile = 98.5 if std > 20 else 96.5
    binary = _threshold_by_percentile(resp, percentile)

    # Minimal cleanup only (do not distort detections).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Draw what binary found directly: thin red contours (no fills).
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Basic binary stats
    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats((binary_clean > 0).astype(np.uint8), connectivity=8)
    binary_components = int(max(0, n_lbl - 1))
    binary_ratio = float(np.mean(binary_clean > 0))
    largest_area = int(np.max(stats[1:, cv2.CC_STAT_AREA])) if n_lbl > 1 else 0

    result = image.copy()
    if contours:
        cv2.drawContours(result, contours, -1, (0, 0, 255), 1)

    # Minimal label
    cv2.rectangle(result, (5, 5), (420, 35), (0, 0, 0), -1)
    cv2.putText(
        result,
        f"BinCC: {binary_components}  Bin%: {binary_ratio*100:.2f}  MaxA: {largest_area}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    # Colorful response heatmap (the "weird colors" view).
    resp_color = cv2.applyColorMap(resp, cv2.COLORMAP_TURBO)

    # Panel: original | response-color | binary | overlay
    h = image.shape[0]
    def _tile(x):
        if x.ndim == 2:
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        if x.shape[0] != h:
            scale = h / float(x.shape[0])
            w = max(1, int(round(x.shape[1] * scale)))
            x = cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)
        return x

    panel = cv2.hconcat([
        _tile(image),
        _tile(resp_color),
        _tile(binary_clean),
        _tile(result),
    ])
    output_path = os.path.join(output_dir, f"{base_name}_panel.jpg")
    cv2.imwrite(output_path, panel)

    print("\n✅ Tespit tamamlandı!")
    print(f"   Binary bileşen: {binary_components}")
    print(f"   Binary oran: {binary_ratio:.6f}")
    print(f"   Max alan: {largest_area}")
    print(f"   Panel: {output_path}")

    return result, binary_components


def test_all(image_dir='images', method='edge'):
    """
    Tüm görüntüleri test et
    
    Args:
        method: 'yolo' veya 'edge'
    """
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\n📁 {len(images)} görüntü bulundu")
    print(f"📋 Yöntem: {method.upper()}")
    
    for img_name in sorted(images):
        img_path = os.path.join(image_dir, img_name)
        
        if method == 'yolo':
            detect_cracks_yolo(img_path)
        else:
            detect_cracks_edge_based(img_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\nKullanım:")
        print("  python crack_yolo.py <görüntü>")
        print("  python crack_yolo.py --all")
        print("  python crack_yolo.py --all --yolo")
        sys.exit(1)
    
    if sys.argv[1] == '--all':
        method = 'yolo' if '--yolo' in sys.argv else 'edge'
        test_all(method=method)
    else:
        # Varsayılan olarak edge-based kullan
        detect_cracks_edge_based(sys.argv[1])
