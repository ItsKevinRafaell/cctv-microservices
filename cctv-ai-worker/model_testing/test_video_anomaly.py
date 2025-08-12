# RUNNING : py test_video_anomaly.py <video_source_to_be_tested>
import cv2
import numpy as np
from collections import deque
import tensorflow as tf
import os
import sys

# --- Config sesuai training ---
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 50
MODEL_PATH = "mod.h5"   # ganti path kalau beda
THRESHOLD = 0.8         # ambang batas deteksi anomaly

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' tidak ditemukan.")

# --- Load model ---
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# --- Load video file ---
if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} path_ke_video.mp4")
    sys.exit(1)

video_path = sys.argv[1]
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file '{video_path}' tidak ditemukan.")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Tidak bisa membuka video.")

# --- Frame buffer ---
frames_deque = deque(maxlen=SEQUENCE_LENGTH)

print("[INFO] Starting video test...")
while True:
    ret, frame = cap.read()
    if not ret:
        break  # selesai

    # Resize untuk model & normalisasi
    small = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    small = small.astype(np.float32) / 255.0
    frames_deque.append(small)

    status_text = "Collecting..."
    color = (0, 255, 0)

    if len(frames_deque) == SEQUENCE_LENGTH:
        seq = np.array(frames_deque, dtype=np.float32)
        seq = np.expand_dims(seq, axis=0)  # (1, seq, h, w, c)
        preds = model.predict(seq, verbose=0)  # (1, 2)
        anomaly_prob = float(preds[0][0])  # index 0 = anomaly
        normal_prob = float(preds[0][1])

        if anomaly_prob >= THRESHOLD:
            status_text = f"ANOMALY ({anomaly_prob:.2f})"
            color = (0, 0, 255)  # merah
        else:
            status_text = f"Normal ({normal_prob:.2f})"
            color = (0, 255, 0)  # hijau

    # Tampilkan di video
    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Video Anomaly Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
