import torch
import yolov5
import cv2
import numpy as np
import os
import logging
import json
from datetime import datetime, timedelta
import hashlib

# === Logging Ayarı ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("plaka_tespit.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# === Config dosyası ===
config_file = "config.json"


def load_config():
    """Konfigürasyonu dosyadan yükle"""
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_config(config):
    """Konfigürasyonu dosyaya kaydet"""
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    logging.info("Konfigürasyon dosyaya kaydedildi.")


# === Klasör Ayarları ===
BASE_OUTPUT_FOLDER = 'video_results'
CROPPED_BASE_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, 'plakalar')

# Son kaydedilen plakalar (hash + zaman)
recent_hashes = []


def create_output_folders():
    """Tarih bazlı çıktı klasörlerini hazırla"""
    today = datetime.now().strftime("%Y-%m-%d")
    cropped_folder = os.path.join(CROPPED_BASE_FOLDER, today)
    os.makedirs(cropped_folder, exist_ok=True)
    return cropped_folder

def load_model(weights_path: str, min_confidence: float):
    """YOLOv5 modelini yükle"""
    logging.info("YOLOv5 modeli yükleniyor...")

    import torch.serialization
    import torch.nn as nn
    from yolov5.models.yolo import DetectionModel
    from yolov5 import models
    from yolov5.models.yolo import Detect
    from torch.nn.modules.pooling import MaxPool2d, AdaptiveAvgPool2d

    # YOLOv5'te kullanılan tüm katmanları whitelist'e ekle
    torch.serialization.add_safe_globals([
        DetectionModel,
        Detect,
        nn.Sequential,
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.SiLU,
        nn.ModuleList,
        nn.Upsample,
        nn.Identity,
        nn.Hardswish,
        nn.LeakyReLU,
        MaxPool2d,
        AdaptiveAvgPool2d,
        models.common.Conv,
        models.common.C3,
        models.common.Bottleneck,
        models.common.SPPF,
        models.common.Concat,
        models.common.AutoShape,
        models.common.DetectMultiBackend,
    ])

    # Modeli yükle
    model = yolov5.load(weights_path)

    model.conf = min_confidence
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 100
    logging.info("Model başarıyla yüklendi.")
    return model






def get_image_hash(image):
    """Küçük boyutlu gri resimden hash üret"""
    resized = cv2.resize(image, (32, 16))  # küçült
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # hashlib md5 girdisi bytes olmalı
    return hashlib.md5(gray.tobytes()).hexdigest()


def is_duplicate(image_hash, ttl: int):
    """Son kaydedilen plakalar arasında duplicate kontrolü"""
    global recent_hashes
    now = datetime.now()

    # Süresi geçmiş kayıtları temizle
    recent_hashes = [(h, t) for h, t in recent_hashes if now - t < timedelta(seconds=ttl)]

    # Aynı hash var mı?
    for h, _ in recent_hashes:
        if h == image_hash:
            return True

    # Yeni hash ekle
    recent_hashes.append((image_hash, now))
    return False


def save_cropped_plate(cropped, score, output_folder, ttl: int):
    """Plaka görüntüsünü dosyaya kaydet"""
    image_hash = get_image_hash(cropped)
    if is_duplicate(image_hash, ttl):
        logging.warning("Duplicate plaka tespit edildi, kaydedilmedi.")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"plaka_{timestamp}_score_{score:.2f}.jpg"
    filepath = os.path.join(output_folder, filename)
    cv2.imwrite(filepath, cropped)
    logging.info(f"Plaka kaydedildi: {filepath}")


def process_predictions(predictions, roi_frame, roi_coords, model, output_folder, frame, ttl: int):
    """Model tahminlerini işle, kırp ve kaydet"""
    if predictions is None or predictions.shape[0] == 0:
        return

    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    roi_h, roi_w, _ = roi_frame.shape
    pad = 5
    x1_roi, y1_roi = roi_coords

    for i in range(len(boxes)):
        if scores[i] >= model.conf:
            x1, y1, x2, y2 = boxes[i].int().tolist()

            # Güvenli kırpma
            x1_pad = max(0, x1 - pad)
            y1_pad = max(0, y1 - pad)
            x2_pad = min(roi_w, x2 + pad)
            y2_pad = min(roi_h, y2 + pad)

            cropped = roi_frame[y1_pad:y2_pad, x1_pad:x2_pad]

            # Kontrol: boyut ve oran
            ch, cw = cropped.shape[:2]
            aspect_ratio = (cw / ch) if ch != 0 else 0
            area = cw * ch

            if area >= 500 and (2.0 < aspect_ratio < 6.0):
                save_cropped_plate(cropped, scores[i].item(), output_folder, ttl)

            # Ana frame üzerine kutu çiz
            cv2.rectangle(
                frame,
                (x1_pad + x1_roi, y1_pad + y1_roi),
                (x2_pad + x1_roi, y2_pad + y1_roi),
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"{scores[i].item():.2f}",
                (x1_pad + x1_roi, y1_pad + y1_roi - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )


def main():
    # === Konfigürasyon yükle ===
    config = load_config()
    print(config)
    FRAME_SKIP = config.get("frame_skip", 10)
    MIN_CONFIDENCE = config.get("min_confidence", 0.70)
    OUTPUT_SIZE = config.get("output_size", 640)
    DUPLICATE_TTL= config.get("duplicate_ttl", 3)
    RTSP_USER = config.get("rtsp_user", "admin")
    RTSP_PASS = config.get("rtsp_pass", "")
    CAMERA_IP = config.get("camera_ip", "0.0.0.0")
    API_BASE = config.get("api_base", "http://localhost:5236")
    USERNAME = config.get("username", "admin")
    PASSWORD = config.get("password", "1")
    cooldown_seconds = 15
    RTSP_URL = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{CAMERA_IP}:554"
    RTSP_STREAM_NAME = config.get("rtsp_stream_name", "")

    if RTSP_STREAM_NAME != "":
        RTSP_URL+="/"+RTSP_STREAM_NAME

    print("RTSP_URL:", RTSP_URL)

    # === Model yükle ===
    model = load_model('./best.pt', MIN_CONFIDENCE)

    # === RTSP bağlantısı ===
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        logging.error("Video açılamadı.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    cap.set(cv2.CAP_PROP_FPS, 15)

    # İlk frame al
    ret, frame = cap.read()
    if not ret:
        logging.error("İlk frame alınamadı.")
        cap.release()
        return

    # ROI kontrolü
    roi = config.get("roi", None)
    if roi:
        x_roi, y_roi, w_roi, h_roi = roi
        logging.info(f"Konfigürasyondan ROI yüklendi: ({x_roi}, {y_roi}, {w_roi}, {h_roi})")
    else:
        logging.info("ROI seçimi için frame açılıyor...")
        roi = cv2.selectROI("ROI Seçimi", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("ROI Seçimi")
        config["roi"] = [int(v) for v in roi]
        save_config(config)

    x_roi, y_roi, w_roi, h_roi = roi
    roi_coords = (int(x_roi), int(y_roi))
    x1_roi, y1_roi = roi_coords
    x2_roi, y2_roi = int(x_roi + w_roi), int(y_roi + h_roi)

    # Çıktı klasörü
    cropped_output_folder = create_output_folders()

    frame_count = 0
    processed_count = 0

    while True:
        ret = cap.grab()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        ret, frame = cap.retrieve()
        if not ret:
            continue

        processed_count += 1
        logging.info(f"Frame {frame_count} işleniyor (Toplam işlenen: {processed_count})")

        roi_frame = frame[y1_roi:y2_roi, x1_roi:x2_roi]
        results = model(roi_frame, size=OUTPUT_SIZE)
        predictions = results.pred[0]

        process_predictions(predictions, roi_frame, roi_coords, model, cropped_output_folder, frame, DUPLICATE_TTL)

        # ROI kutusu çiz
        cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 0, 0), 2)
        cv2.imshow('Plaka Tespiti (ROI Uygulandı)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("İşlem tamamlandı.")


if __name__ == "__main__":
    main()
