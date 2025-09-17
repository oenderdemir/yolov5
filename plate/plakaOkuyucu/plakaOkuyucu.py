from ultralytics import YOLO
import cv2
import os
import glob
import re
import shutil
import time
from collections import deque
from datetime import datetime, timedelta

# === Ayarlar ===
model_path = 'Charcter-LP.pt'
images_folder = 'video_results/plakalar'
plaka_regex = r'[0-9]{2}[A-ZÇŞĞÜİÖ]{1,3}[0-9]{2,4}'  # Türk plaka formatı
output_file = "../guvenilir_plakalar.txt"
guvenilir_folder = "guvenilir_plakalar"
cooldown_seconds = 15   # Aynı plaka için bekleme süresi

os.makedirs(guvenilir_folder, exist_ok=True)

# === Model yükle ===
model = YOLO(model_path)

# Sonuçları takip etmek için kuyruk
last_results = deque(maxlen=2)  # son 2 sonucu sakla
last_images = deque(maxlen=2)   # son 2 görsel yolunu sakla
seen_files = set()              # daha önce işlenen dosyalar

# Plaka -> Son kayıt zamanı
plaka_kayit_zamanlari = {}


def oku_plaka(image_path):
    """Tek resimden plaka okuma (regex uymazsa None döner)"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"HATA: Resim yüklenemedi: {image_path}")
        return None

    results = model(image)
    boxes = results[0].boxes

    detected_chars = []
    if boxes is not None and boxes.xyxy is not None:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cls_id = int(boxes.cls[i].cpu().numpy()) if boxes.cls is not None else -1
            char_label = model.names[cls_id] if cls_id >= 0 else '?'

            detected_chars.append((x1, char_label))

    detected_chars.sort(key=lambda x: x[0])
    detected_text = ''.join([char for (_, char) in detected_chars])
    detected_text = detected_text.replace(" ", "").strip()

    # Regex ile Türk plaka formatını kontrol et
    matches = re.findall(plaka_regex, detected_text)
    if matches:
        return matches[0]  # plakaya uygun format
    else:
        return None  # regex uymuyorsa sonuç üretme


def kaydet_guvenilir_plaka(plaka_text, image_path):
    """Bulunan güvenilir plakayı dosyaya ve klasöre kaydet"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} -> {plaka_text}\n")

    # Aynı isimde dosya varsa üzerine yazmamak için zaman damgası ekle
    safe_name = f"{plaka_text}_{datetime.now().strftime('%H%M%S')}.jpg"
    dest_path = os.path.join(guvenilir_folder, safe_name)

    shutil.copy(image_path, dest_path)
    print(f"📂 Guvenilir plaka dosyaya kaydedildi: {plaka_text}")
    print(f"🖼️ Resim kopyalandı: {dest_path}")

    # Son kayıt zamanını güncelle
    plaka_kayit_zamanlari[plaka_text] = datetime.now()


def main():
    print("Sürekli tarama başlatıldı. Çıkmak için CTRL+C basın.\n")

    while True:
        # Yeni dosyaları bul
        image_paths = glob.glob(os.path.join(images_folder, '**', '*.jpg'), recursive=True)
        image_paths += glob.glob(os.path.join(images_folder, '**', '*.png'), recursive=True)
        image_paths.sort()

        for img_path in image_paths:
            if img_path in seen_files:
                continue  # zaten işlenmiş
            seen_files.add(img_path)

            result = oku_plaka(img_path)

            if result:
                print(f"[OK] {os.path.basename(img_path)} -> {result}")
                last_results.append(result)
                last_images.append(img_path)

                # Eğer son 2 sonuç aynıysa, plakayı güvenilir kabul et
                if len(last_results) == 2 and len(set(last_results)) == 1:
                    son_kayit = plaka_kayit_zamanlari.get(result, None)

                    if son_kayit and datetime.now() - son_kayit < timedelta(seconds=cooldown_seconds):
                        print(f"⏳ {result} plakası için bekleme süresi dolmadı, tekrar kaydedilmeyecek.\n")
                    else:
                        print(f"✅ GÜVENİLİR PLAKA BULUNDU: {result}\n")
                        kaydet_guvenilir_plaka(result, last_images[-1])

                    last_results.clear()
                    last_images.clear()
            else:
                print(f"[X] {os.path.basename(img_path)} -> Geçerli plaka bulunamadı.")

        time.sleep(2)  # her 2 saniyede bir yeni dosya var mı kontrol et


if __name__ == "__main__":
    main()
