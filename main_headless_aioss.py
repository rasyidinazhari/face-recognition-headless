import cv2
import os
import numpy as np
import time
import requests
import logging
import threading
import json
import sys
import signal
from datetime import datetime

import mediapipe as mp
from keras_facenet import FaceNet

CONFIG_FILE = "config_aioss.json"
running_system = True

# ==========================================
# 1. SETUP LOGGING & BACKGROUND READER
# ==========================================
def setup_custom_logging():
    base_log_path = "logs"
    for folder in ["system", "error", "attendance"]:
        os.makedirs(os.path.join(base_log_path, folder), exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    logger_sys = logging.getLogger('system_logger')
    logger_sys.setLevel(logging.INFO)
    if not logger_sys.hasHandlers():
        sys_h = logging.FileHandler(f"logs/system/{today}_headless_AIOSS.log")
        sys_h.setFormatter(log_format)
        logger_sys.addHandler(sys_h)
        logger_sys.addHandler(logging.StreamHandler())

    logger_err = logging.getLogger('error_logger')
    logger_err.setLevel(logging.ERROR)
    if not logger_err.hasHandlers():
        err_h = logging.FileHandler(f"logs/error/error_{today}_headless_AIOSS.log")
        err_h.setFormatter(log_format)
        logger_err.addHandler(err_h)
        logger_err.addHandler(logging.StreamHandler())

    logger_att = logging.getLogger('attendance_logger')
    logger_att.setLevel(logging.INFO)
    if not logger_att.hasHandlers():
        att_h = logging.FileHandler(f"logs/attendance/attendance_{today}_headless_AIOSS.log")
        att_h.setFormatter(log_format)
        logger_att.addHandler(att_h)
    
    return logger_sys, logger_err, logger_att

logger_sys, logger_err, logger_att = setup_custom_logging()

class RTSPVideoReader:
    def __init__(self, src=0):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame
            else:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()

# ==========================================
# 2. FUNGSI ALIGNMENT WAJAH
# ==========================================
def align_and_crop_face(image, detection):
    h, w, _ = image.shape
    bboxC = detection.location_data.relative_bounding_box
    x, y = int(bboxC.xmin * w), int(bboxC.ymin * h)
    bw, bh = int(bboxC.width * w), int(bboxC.height * h)
    
    keypoints = detection.location_data.relative_keypoints
    re_x, re_y = int(keypoints[0].x * w), int(keypoints[0].y * h) 
    le_x, le_y = int(keypoints[1].x * w), int(keypoints[1].y * h) 
    
    angle = np.degrees(np.arctan2(le_y - re_y, le_x - re_x))
    
    m_x, m_y = int(bw * 0.5), int(bh * 0.5)
    x_min, y_min = max(0, x - m_x), max(0, y - m_y)
    x_max, y_max = min(w, x + bw + m_x), min(h, y + bh + m_y)
    
    padded_crop = image[y_min:y_max, x_min:x_max]
    if padded_crop.size == 0: return None
    
    center_x, center_y = (re_x + le_x) // 2 - x_min, (re_y + le_y) // 2 - y_min
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    rotated_crop = cv2.warpAffine(padded_crop, M, (padded_crop.shape[1], padded_crop.shape[0]), flags=cv2.INTER_CUBIC)
    
    new_x, new_y = x - x_min, y - y_min
    strict_m_x, strict_m_y = int(bw * 0.05), int(bh * 0.05)
    final_x_min, final_y_min = max(0, new_x - strict_m_x), max(0, new_y - strict_m_y)
    final_x_max, final_y_max = min(rotated_crop.shape[1], new_x + bw + strict_m_x), min(rotated_crop.shape[0], new_y + bh + strict_m_y)
    
    return rotated_crop[final_y_min:final_y_max, final_x_min:final_x_max]

# ==========================================
# 3. CORE HEADLESS CONFIG & UTILITIES (AIOSS)
# ==========================================
class HeadlessAttendanceAioss:
    def __init__(self):
        self.known_face_encodings, self.known_face_nips, self.known_face_names = [], [], []
        self.last_attendance, self.tracked_faces = {}, []
        self.frame_count = 0
        self.COOLDOWN_DETIK = 30
        self.need_reload = False
        
        self.load_config()
        
        logger_sys.info("🧠 Memuat Model FaceNet dan MediaPipe (Headless Mode AIOSS)...")
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)
        self.embedder = FaceNet()
        
        self.load_database_wajah()
        threading.Thread(target=self.background_sync_worker, daemon=True).start()
        
        logger_sys.info(f"Mencoba membuka kamera via Headless AIOSS: {self.VIDEO_SOURCE}")
        self.cap = RTSPVideoReader(self.VIDEO_SOURCE)

    def load_config(self):
        self.VIDEO_SOURCE = 0
        self.LATITUDE, self.LONGITUDE = "-7.7693546", "110.3956848"
        self.CITY, self.IP_ADDRESS = "Yogyakarta", "127.0.0.1"
        self.API_URL, self.SYNC_API_URL = "", ""
        self.FACENET_THRESHOLD = 0.75
        self.NAMA_MESIN = "HEADLESS-PC-AIOSS"
        self.COMPANY_CODE = "DMO"

        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as file:
                data = json.load(file)
                source_type = data.get("video_source", "")
                if "Webcam Laptop" in source_type: self.VIDEO_SOURCE = 0
                elif "Webcam External" in source_type: self.VIDEO_SOURCE = 1
                else: self.VIDEO_SOURCE = data.get("rtsp_url", "")
                
                self.COMPANY_CODE = data.get("company_code", self.COMPANY_CODE)
                self.LATITUDE, self.LONGITUDE = data.get("latitude", self.LATITUDE), data.get("longitude", self.LONGITUDE)
                self.CITY, self.IP_ADDRESS = data.get("city", self.CITY), data.get("ip_address", self.IP_ADDRESS)
                self.API_URL, self.SYNC_API_URL = data.get("api_url", ""), data.get("sync_url", "")
                
                threshold_str = data.get("facenet_threshold", "Normal (0.75)")
                if "0.60" in threshold_str: self.FACENET_THRESHOLD = 0.60
                elif "0.90" in threshold_str: self.FACENET_THRESHOLD = 0.90
                else: self.FACENET_THRESHOLD = 0.75

    def background_sync_worker(self):
        logger_sys.info("🔄 Background Sync Worker AIOSS harian berjalan.")
        while running_system:
            if self.SYNC_API_URL != "":
                try:
                    response = requests.get(self.SYNC_API_URL, timeout=10)
                    if response.status_code == 200:
                        karyawan_list = response.json().get("data", [])
                        ada_foto_baru = False
                        for kar in karyawan_list:
                            nip = kar.get("student_id", kar.get("student_code", ""))
                            nama = kar.get("name", "Unknown")
                            foto_url = kar.get("photo_url", "")
                            
                            if append_validation(nip, foto_url):
                                filename = f"Picture_{nip}_{nama}.jpg"
                                filepath = os.path.join("known_faces", filename)
                                if not os.path.exists(filepath):
                                    try:
                                        img_data = requests.get(foto_url, timeout=10)
                                        if img_data.status_code == 200:
                                            with open(filepath, 'wb') as f:
                                                f.write(img_data.content)
                                            logger_sys.info(f"⬇️ Berhasil unduh foto baru via Headless AIOSS: {filename}")
                                            ada_foto_baru = True
                                    except Exception:
                                        pass
                        if ada_foto_baru:
                            self.need_reload = True
                except Exception as e:
                    logger_err.error(f"⚠️ Gagal sinkronisasi harian Headless AIOSS: {e}")
            time.sleep(86400) 

    def kirim_data_ke_backend(self, nip, frame_crop):
        try:
            success, buffer = cv2.imencode('.jpg', frame_crop)
            if not success: return
                
            data_text = {
                "company_code": self.COMPANY_CODE,
                "student_id": nip,
                "latitude": self.LATITUDE, 
                "longitude": self.LONGITUDE,
                "scan_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "machine": self.NAMA_MESIN, 
                "ip_address": self.IP_ADDRESS, 
                "city": self.CITY,
                "remark": "Data From Headless CCTV System (AIOSS)"
            }
            data_file = {"image": (f"{nip}_capture.jpg", buffer.tobytes(), "image/jpeg")}

            def worker_api():
                try:
                    response = requests.post(self.API_URL, data=data_text, files=data_file, timeout=10)
                    if response.status_code in [200, 201]:
                        logger_att.info(f"✅ SUKSES API [Headless AIOSS]: Data {nip} terkirim.")
                    else:
                        logger_err.error(f"⚠️ API GAGAL [Headless AIOSS]: {response.text}")
                except Exception as e:
                    logger_err.error(f"⚠️ API Background Error [Headless AIOSS]: {e}")

            if self.API_URL != "":
                threading.Thread(target=worker_api, daemon=True).start()
        except Exception as e:
            logger_err.error(f"❌ GAGAL memproses API Headless AIOSS: {str(e)}")

    def load_database_wajah(self):
        path_foto = "known_faces"
        if not os.path.exists(path_foto): os.makedirs(path_foto)
        self.known_face_encodings, self.known_face_nips, self.known_face_names = [], [], []
        
        for filename in os.listdir(path_foto):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                clean_name = os.path.splitext(filename)[0] 
                parts = clean_name.split("_")
                nip = parts[1] if len(parts) >= 3 else "UNKNOWN"
                
                if nip not in self.known_face_nips:
                    image_path = os.path.join(path_foto, filename)
                    try:
                        img = cv2.imread(image_path)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        results = self.face_detector.process(img_rgb)
                        if results.detections:
                            face_crop_aligned = align_and_crop_face(img_rgb, results.detections[0])
                            if face_crop_aligned is not None:
                                face_crop_resized = cv2.resize(face_crop_aligned, (160, 160))
                                embedding = self.embedder.embeddings([face_crop_resized])[0]
                                self.known_face_encodings.append(embedding)
                                nama = " ".join(parts[2:]) if len(parts) >= 3 else clean_name
                                self.known_face_nips.append(nip)
                                self.known_face_names.append(nama)
                    except Exception:
                        pass
        logger_sys.info(f"✅ Load database sukses! {len(self.known_face_nips)} wajah aktif di memori Headless AIOSS.")

    def run(self):
        global running_system
        logger_sys.info("🚀 Mesin utama Headless AIOSS sukses mengudara...")
        
        while running_system:
            if self.need_reload:
                self.load_database_wajah()
                self.need_reload = False

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            self.frame_count += 1
            h_cam, w_cam, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_time = time.time()
            
            results = self.face_detector.process(rgb_frame)
            new_tracked_faces = []
            
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x, y = int(bboxC.xmin * w_cam), int(bboxC.ymin * h_cam)
                    bw, bh = int(bboxC.width * w_cam), int(bboxC.height * h_cam)
                    cx, cy = x + bw // 2, y + bh // 2
                    
                    matched_face = None
                    for tf in self.tracked_faces:
                        dist = np.sqrt((cx - tf['centroid'][0])**2 + (cy - tf['centroid'][1])**2)
                        if dist < 100:
                            matched_face = tf
                            break

                    if matched_face and matched_face['nip'] != "":
                        new_tracked_faces.append({'centroid': (cx, cy), 'nip': matched_face['nip'], 'name': matched_face['name']})
                    else:
                        if self.frame_count % 10 == 0 or matched_face is None:
                            face_crop_aligned = align_and_crop_face(rgb_frame, detection)
                            if face_crop_aligned is not None and face_crop_aligned.size != 0:
                                face_crop_resized = cv2.resize(face_crop_aligned, (160, 160))
                                try:
                                    emb = self.embedder.embeddings([face_crop_resized])[0]
                                    if len(self.known_face_encodings) > 0:
                                        distances = np.linalg.norm(self.known_face_encodings - emb, axis=1)
                                        best_match_index = np.argmin(distances)
                                        
                                        if distances[best_match_index] < self.FACENET_THRESHOLD:
                                            display_nip = self.known_face_nips[best_match_index]
                                            display_name = self.known_face_names[best_match_index]
                                            
                                            new_tracked_faces.append({'centroid': (cx, cy), 'nip': display_nip, 'name': display_name})
                                            
                                            last_seen = self.last_attendance.get(display_nip, 0)
                                            if current_time - last_seen > self.COOLDOWN_DETIK:
                                                self.last_attendance[display_nip] = current_time
                                                logger_att.info(f"[Headless AIOSS] Terdeteksi: {display_nip} - {display_name}")
                                                
                                                margin_y, margin_x = int(bh * 0.1), int(bw * 0.1)
                                                c_top, c_bottom = max(0, y - margin_y), min(h_cam, y + bh + margin_y)
                                                c_left, c_right = max(0, x - margin_x), min(w_cam, x + bw + margin_x)
                                                
                                                if c_top >= 0 and c_bottom <= h_cam and c_left >= 0 and c_right <= w_cam:
                                                    bgr_crop = frame[c_top:c_bottom, c_left:c_right].copy()
                                                    self.kirim_data_ke_backend(display_nip, bgr_crop)
                                        else:
                                            new_tracked_faces.append({'centroid': (cx, cy), 'nip': "", 'name': "Unknown"})
                                except Exception:
                                    pass
                        else:
                            new_tracked_faces.append({'centroid': (cx, cy), 'nip': "", 'name': "Unknown"})

            self.tracked_faces = new_tracked_faces
            time.sleep(0.030)

    def clean_exit(self):
        self.cap.release()
        logger_sys.info("✅ Kamera berhasil dilepas. Headless AIOSS mati dengan aman.")

def append_validation(nip, url):
    return True if nip and url else False

def sigterm_handler(signum, frame):
    global running_system
    logger_sys.info("🛑 Menerima sinyal shutdown (SIGTERM/SIGINT). Menghentikan sistem AIOSS...")
    running_system = False

if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigterm_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)
    
    app = HeadlessAttendanceAioss()
    app.run()
    app.clean_exit()
