"""
Görüntü analizi - çatlakların nasıl göründüğünü anlamak için
"""
import cv2
import numpy as np
import os

def analyze_image(image_path):
    """Görüntüyü analiz et ve farklı işlemlerle kaydet"""
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Yüklenemedi: {image_path}")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    print(f"\n{'='*50}")
    print(f"Görüntü: {os.path.basename(image_path)}")
    print(f"Boyut: {w}x{h}")
    print(f"Min/Max piksel: {gray.min()} / {gray.max()}")
    print(f"Ortalama: {gray.mean():.1f}")
    print(f"Std: {gray.std():.1f}")
    
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = "results/debug"
    os.makedirs(out_dir, exist_ok=True)
    
    # Orijinal
    cv2.imwrite(f"{out_dir}/{base}_0_original.jpg", image)
    cv2.imwrite(f"{out_dir}/{base}_1_gray.jpg", gray)
    
    # Histogram equalization
    eq = cv2.equalizeHist(gray)
    cv2.imwrite(f"{out_dir}/{base}_2_equalized.jpg", eq)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    cv2.imwrite(f"{out_dir}/{base}_3_clahe.jpg", cl)
    
    # Canny - farklı eşikler
    for t1, t2 in [(10, 30), (20, 60), (30, 90), (50, 150)]:
        canny = cv2.Canny(gray, t1, t2)
        cv2.imwrite(f"{out_dir}/{base}_4_canny_{t1}_{t2}.jpg", canny)
    
    # Adaptive threshold
    adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite(f"{out_dir}/{base}_5_adaptive.jpg", adapt)
    
    # Black-hat
    for k in [5, 10, 15, 20]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        cv2.imwrite(f"{out_dir}/{base}_6_blackhat_{k}.jpg", bh)
    
    # Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = (sobel / sobel.max() * 255).astype(np.uint8)
    cv2.imwrite(f"{out_dir}/{base}_7_sobel.jpg", sobel)
    
    # Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.uint8(np.abs(lap))
    cv2.imwrite(f"{out_dir}/{base}_8_laplacian.jpg", lap)
    
    # Koyu bölgeler (yerel ortalamadan koyu)
    local_mean = cv2.blur(gray, (31, 31))
    darker = cv2.subtract(local_mean, gray)
    cv2.imwrite(f"{out_dir}/{base}_9_darker.jpg", darker)
    
    # Threshold darker
    _, dark_thresh = cv2.threshold(darker, 10, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"{out_dir}/{base}_9_darker_thresh.jpg", dark_thresh)
    
    print(f"Debug görüntüleri kaydedildi: {out_dir}/")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_image(sys.argv[1])
    else:
        for f in sorted(os.listdir("images")):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                analyze_image(f"images/{f}")
