import torch
import yolov5
import cv2
import numpy as np
import os
import logging
import json
import hashlib
import shutil
import re
from datetime import datetime, timedelta
from ultralytics import YOLO

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("plaka_tespit.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# === Config ===
config_file = "config.json"

def load_config():
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_config(config):
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    logging.info("Konfigürasyon dosyaya kaydedildi.")

# === Yol ayarları ===
BASE_OUTPUT_FOLDER = 'video_results'
CROPPED_BASE_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, 'plakalar')
guvenilir_folder = "guvenilir_plakalar"
output_file = "guvenilir_plakalar.txt"
os.makedirs(guvenilir_folder, exist_ok=True)

recent_hashes = []
plaka_regex = r'[0-9]{2}[A-ZÇŞĞÜİÖ]{1,3}[0-9]{2,4}'
plaka_kayit_zamanlari = {}
cooldown_seconds = 15

# === Modeller ===
ocr_model = YOLO("Charcter-LP.pt")

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

# === Yardımcılar ===
def get_image_hash(image):
    resized = cv2.resize(image, (32, 16))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return hashlib.md5(gray.tobytes()).hexdigest()

def is_duplicate(image_hash, ttl: int):
    global recent_hashes
    now = datetime.now()
    recent_hashes = [(h, t) for h, t in recent_hashes if now - t < timedelta(seconds=ttl)]
    for h, _ in recent_hashes:
        if h == image_hash:
            return True
    recent_hashes.append((image_hash, now))
    return False

def oku_plaka(cropped):
    results = ocr_model(cropped)
    boxes = results[0].boxes
    detected_chars = []
    if boxes is not None and boxes.xyxy is not None:
        for i in range(len(boxes)):
            x1 = int(boxes.xyxy[i][0].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy()) if boxes.cls is not None else -1
            char_label = ocr_model.names[cls_id] if cls_id >= 0 else '?'
            detected_chars.append((x1, char_label))
    detected_chars.sort(key=lambda x: x[0])
    detected_text = ''.join([c for (_, c) in detected_chars]).replace(" ", "").strip()
    matches = re.findall(plaka_regex, detected_text)
    return matches[0] if matches else None

def kaydet_guvenilir_plaka(plaka_text, cropped):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} -> {plaka_text}\n")
    safe_name = f"{plaka_text}_{datetime.now().strftime('%H%M%S')}.jpg"
    dest_path = os.path.join(guvenilir_folder, safe_name)
    cv2.imwrite(dest_path, cropped)
    logging.info(f"✅ Güvenilir plaka: {plaka_text}, kaydedildi: {dest_path}")
    plaka_kayit_zamanlari[plaka_text] = datetime.now()

# === Prediction işleme ===
def process_predictions(predictions, roi_frame, roi_coords, model, frame, ttl: int):
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
            x1_pad, y1_pad = max(0, x1 - pad), max(0, y1 - pad)
            x2_pad, y2_pad = min(roi_w, x2 + pad), min(roi_h, y2 + pad)
            cropped = roi_frame[y1_pad:y2_pad, x1_pad:x2_pad]
            ch, cw = cropped.shape[:2]
            if ch == 0 or cw == 0:
                continue
            aspect_ratio = cw / ch
            area = cw * ch
            if area >= 500 and (2.0 < aspect_ratio < 6.0):
                image_hash = get_image_hash(cropped)
                if is_duplicate(image_hash, ttl):
                    continue
                plaka_text = oku_plaka(cropped)
                if plaka_text:
                    son_kayit = plaka_kayit_zamanlari.get(plaka_text, None)
                    if not son_kayit or datetime.now() - son_kayit > timedelta(seconds=cooldown_seconds):
                        kaydet_guvenilir_plaka(plaka_text, cropped)
            cv2.rectangle(frame, (x1_pad+x1_roi, y1_pad+y1_roi),
                          (x2_pad+x1_roi, y2_pad+y1_roi), (0,255,0), 2)

# === Main ===
def main():
    config = load_config()
    print(config)
    FRAME_SKIP = config.get("frame_skip", 10)
    MIN_CONFIDENCE = config.get("min_confidence", 0.70)
    OUTPUT_SIZE = config.get("output_size", 640)
    DUPLICATE_TTL= config.get("duplicate_ttl", 3)
    RTSP_USER = config.get("rtsp_user", "admin")
    RTSP_PASS = config.get("rtsp_pass", "")
    CAMERA_IP = config.get("camera_ip", "0.0.0.0")
    RTSP_STREAM_NAME = config.get("rtsp_stream_name", "")
    RTSP_URL = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{CAMERA_IP}:554"
    if RTSP_STREAM_NAME != "":
        RTSP_URL += "/" + RTSP_STREAM_NAME

    logging.info(f"RTSP_URL: {RTSP_URL}")
    model = load_model('./best.pt', MIN_CONFIDENCE)

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        logging.error("Video açılamadı.")
        return

    ret, frame = cap.read()
    if not ret:
        logging.error("İlk frame alınamadı.")
        return

    roi = config.get("roi", None)
    if not roi:
        roi = cv2.selectROI("ROI Seçimi", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("ROI Seçimi")
        config["roi"] = [int(v) for v in roi]
        save_config(config)
    x_roi, y_roi, w_roi, h_roi = roi
    roi_coords = (int(x_roi), int(y_roi))
    x1_roi, y1_roi = roi_coords
    x2_roi, y2_roi = int(x_roi+w_roi), int(y_roi+h_roi)

    frame_count = 0
    while True:
        ret = cap.grab()
        if not ret: break
        frame_count += 1
        if frame_count % FRAME_SKIP != 0: continue
        ret, frame = cap.retrieve()
        if not ret: continue
        roi_frame = frame[y1_roi:y2_roi, x1_roi:x2_roi]
        results = model(roi_frame, size=OUTPUT_SIZE)
        predictions = results.pred[0]
        process_predictions(predictions, roi_frame, roi_coords, model, frame, DUPLICATE_TTL)
        cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (255,0,0), 2)
        cv2.imshow('Plaka Tespiti ve Okuma', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("İşlem tamamlandı.")

if __name__ == "__main__":
    main()
