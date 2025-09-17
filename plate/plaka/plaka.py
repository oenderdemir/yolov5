import cv2
import os
import logging
import json
import requests
import re
import shutil
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from datetime import datetime, timedelta
from collections import defaultdict

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
auth_token = None

def load_config():
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# === API Fonksiyonları ===
def authenticate(api_base, username, password):
    global auth_token
    url = f"{api_base}/api/Auth/login"
    payload = {"kullaniciAdi": username, "parola": password}
    try:
        resp = requests.post(url, json=payload, headers={"accept": "text/plain"})
        resp.raise_for_status()
        data = resp.json()
        if data.get("authenticateResult") and data.get("authToken"):
            auth_token = data["authToken"]
            logging.info("✅ API authentication başarılı, token alındı.")
        else:
            logging.error("❌ API authentication başarısız!")
    except Exception as e:
        logging.error(f"Auth hatası: {e}")

def send_plate(api_base, camera_ip, plate):
    global auth_token
    if not auth_token:
        logging.warning("⚠️ Token yok, plaka gönderilemedi.")
        return

    url = f"{api_base}/service/VehiclePassCheck/viaCamera"
    headers = {
        "accept": "text/plain",
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "requestDate": datetime.utcnow().isoformat() + "Z",
        "cameraIp": camera_ip,
        "plate": plate
    }
    try:
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        logging.info(f"📡 API'ye plaka gönderildi: {plate} -> {resp.status_code}")
    except Exception as e:
        logging.error(f"Plaka gönderim hatası: {e}")

# === Klasör Ayarları ===
BASE_OUTPUT_FOLDER = 'video_results'
CROPPED_BASE_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, 'plakalar')
guvenilir_folder = "guvenilir_plakalar"
output_file = "guvenilir_plakalar.txt"
plaka_regex = r'[0-9]{2}[A-ZÇŞĞÜİÖ]{1,3}[0-9]{2,4}'  # Türk plaka formatı

os.makedirs(guvenilir_folder, exist_ok=True)
plaka_kayit_zamanlari = {}

def create_output_folders():
    today = datetime.now().strftime("%Y-%m-%d")
    cropped_folder = os.path.join(CROPPED_BASE_FOLDER, today)
    os.makedirs(cropped_folder, exist_ok=True)
    return cropped_folder

# === ONNX Model Yükleme ===
def load_onnx_model(path):
    logging.info(f"ONNX modeli yükleniyor: {path}")
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return session, input_name

def run_onnx_inference(session, input_name, frame, size=640):
    img_resized = cv2.resize(frame, (size, size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    outputs = session.run(None, {input_name: img_input})
    return outputs

# === OCR Plaka Okuma (YOLOv8 PT) ===
def oku_plaka(image_path, char_model):
    results = char_model.predict(image_path, verbose=False)
    boxes = results[0].boxes

    detected_chars = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
        cls_id = int(boxes.cls[i].cpu().numpy())
        char_label = char_model.names[cls_id]
        detected_chars.append((int(x1), char_label))

    detected_chars.sort(key=lambda x: x[0])
    detected_text = ''.join([char for (_, char) in detected_chars])
    detected_text = detected_text.replace(" ", "").strip()

    matches = re.findall(plaka_regex, detected_text)
    return matches[0] if matches else None

def kaydet_guvenilir_plaka(plaka_text, image_path, cooldown_seconds, camera_ip, api_base):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} -> {plaka_text}\n")

    safe_name = f"{plaka_text}_{datetime.now().strftime('%H%M%S')}.jpg"
    dest_path = os.path.join(guvenilir_folder, safe_name)
    shutil.copy(image_path, dest_path)

    logging.info(f"📂 Guvenilir plaka dosyaya kaydedildi: {plaka_text}")
    logging.info(f"🖼️ Resim kopyalandı: {dest_path}")

    send_plate(api_base, camera_ip, plaka_text)
    plaka_kayit_zamanlari[plaka_text] = datetime.now()

# === Gelişmiş Plaka Kontrol ===
plaka_sayac = defaultdict(lambda: {"ilk": None, "say": 0, "confs": []})

def gecerli_tr_plaka(plaka_text):
    """Regex sonrası TR plaka kodu kontrolü (01–81)."""
    if not re.match(plaka_regex, plaka_text):
        return False
    try:
        kod = int(plaka_text[:2])
        return 1 <= kod <= 81
    except:
        return False

def process_plate(plaka_text, conf, image_path, cooldown_seconds, camera_ip, api_base, required_matches=3):
    """Plakayı sayaç mantığı ile işle."""
    if not gecerli_tr_plaka(plaka_text):
        logging.info(f"❌ Geçersiz TR plakası: {plaka_text}")
        return

    simdi = datetime.now()
    kayit = plaka_sayac[plaka_text]

    if kayit["ilk"] is None or (simdi - kayit["ilk"]).seconds > 10:
        # Yeni sayaç başlat
        kayit["ilk"] = simdi
        kayit["say"] = 1
        kayit["confs"] = [conf]
    else:
        kayit["say"] += 1
        kayit["confs"].append(conf)

        ort_conf = sum(kayit["confs"]) / len(kayit["confs"])

        if kayit["say"] >= required_matches and ort_conf >= 0.75:
            son_kayit = plaka_kayit_zamanlari.get(plaka_text, None)
            if son_kayit and datetime.now() - son_kayit < timedelta(seconds=cooldown_seconds):
                logging.info(f"⏳ {plaka_text} için cooldown sürüyor.")
            else:
                logging.info(f"✅ GÜVENİLİR PLAKA BULUNDU: {plaka_text} "
                             f"(ortalama güven: {ort_conf:.2f}, tekrar: {kayit['say']})")
                kaydet_guvenilir_plaka(plaka_text, image_path, cooldown_seconds, camera_ip, api_base)

            # sayaç sıfırla
            plaka_sayac[plaka_text] = {"ilk": None, "say": 0, "confs": []}


# === Main ===
def main():
    config = load_config()

    FRAME_SKIP = config.get("frame_skip", 10)
    MIN_CONFIDENCE = config.get("min_confidence", 0.70)
    OUTPUT_SIZE = config.get("output_size", 640)

    RTSP_USER = config.get("rtsp_user", "admin")
    RTSP_PASS = config.get("rtsp_pass", "")
    CAMERA_IP = config.get("camera_ip", "0.0.0.0")
    API_BASE = config.get("api_base", "http://localhost:5236")
    USERNAME = config.get("username", "admin")
    PASSWORD = config.get("password", "1")
    cooldown_seconds = 15
    RTSP_URL = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{CAMERA_IP}:554"

    # API login
    authenticate(API_BASE, USERNAME, PASSWORD)

    # ONNX plaka tespit modeli yükle
    det_session, det_input = load_onnx_model("best.onnx")

    # YOLOv8 OCR modeli yükle
    char_model = YOLO("Charcter-LP.pt")

    # RTSP aç
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        logging.error("Video açılamadı.")
        return

    # ROI
    ret, frame = cap.read()
    if not ret:
        logging.error("İlk frame alınamadı.")
        return

    roi = config.get("roi", None)
    if roi:
        x_roi, y_roi, w_roi, h_roi = roi
    else:
        logging.info("ROI seçimi için frame açılıyor...")
        roi = cv2.selectROI("ROI Seçimi", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("ROI Seçimi")
        config["roi"] = [int(v) for v in roi]
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        x_roi, y_roi, w_roi, h_roi = roi

    x1_roi, y1_roi = int(x_roi), int(y_roi)
    x2_roi, y2_roi = int(x_roi + w_roi), int(y_roi + h_roi)

    create_output_folders()
    frame_count = 0

    logging.info("Sürekli tarama başlatıldı. Çıkmak için 'q' basın.\n")

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

        roi_frame = frame[y1_roi:y2_roi, x1_roi:x2_roi]

        # Ölçek faktörleri
        scale_x = roi_frame.shape[1] / OUTPUT_SIZE
        scale_y = roi_frame.shape[0] / OUTPUT_SIZE

        # ONNX plaka tespiti
        outputs = run_onnx_inference(det_session, det_input, roi_frame, size=OUTPUT_SIZE)
        detections = outputs[0]
        if len(detections.shape) == 3:
            detections = detections[0]

        for det in detections:
            x, y, w, h, obj_conf, cls_conf = det[:6]
            conf = float(obj_conf * cls_conf)
            if conf < MIN_CONFIDENCE:
                continue

            # xywh → xyxy
            x1 = int((x - w/2) * scale_x)
            y1 = int((y - h/2) * scale_y)
            x2 = int((x + w/2) * scale_x)
            y2 = int((y + h/2) * scale_y)

            x1 = max(0, min(x1, roi_frame.shape[1]))
            y1 = max(0, min(y1, roi_frame.shape[0]))
            x2 = max(0, min(x2, roi_frame.shape[1]))
            y2 = max(0, min(y2, roi_frame.shape[0]))
            if x2 <= x1 or y2 <= y1:
                continue

            cropped = roi_frame[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            temp_file = "temp.jpg"
            cv2.imwrite(temp_file, cropped)

            plaka_text = oku_plaka(temp_file, char_model)
            if plaka_text:
                logging.info(f"[OK] -> {plaka_text} (conf={conf:.2f})")
                process_plate(plaka_text, conf, temp_file, cooldown_seconds, CAMERA_IP, API_BASE, required_matches=config.get("required_matches", 3))

        # ROI dikdörtgenini sürekli çiz
        cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 0), 2)

        cv2.imshow("Plaka Tespiti", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
