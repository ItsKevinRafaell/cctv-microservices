import cv2
import numpy as np
from collections import deque
import tensorflow as tf
import time
import os

# --- Config (match your training) ---
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 50
MODEL_PATH = "mod.h5"   # change this to the correct path if needed
WINDOW_NAME = "CCTV - Anomaly Detection"
# initial threshold (you can adjust with trackbar)
INITIAL_THRESHOLD = 0.5

# --- Utility: load model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Put mod.h5 in this folder or update MODEL_PATH.")

model = tf.keras.models.load_model(MODEL_PATH)
# optional: warm up model
_dummy = np.zeros((1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
try:
    model.predict(_dummy, verbose=0)
except Exception:
    pass

# --- Prepare deque for frames (sliding window) ---
frames_deque = deque(maxlen=SEQUENCE_LENGTH)

# --- Trackbar callback ---
def nothing(x):
    pass

cv2.namedWindow(WINDOW_NAME)
cv2.createTrackbar('Threshold x100', WINDOW_NAME, int(INITIAL_THRESHOLD * 100), 100, nothing)

# --- Open webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam (device 0). Make sure camera is connected.")

# For FPS display
prev_time = time.time()
fps = 0.0

print("Starting live CCTV. Press 'q' to quit.")

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from webcam. Exiting.")
        break

    # Mirror to look like CCTV viewer (optional)
    frame = cv2.flip(frame, 1)

    # Make a copy for display
    display_frame = frame.copy()

    # Preprocess: resize to model input size and normalize
    small = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    small = small.astype(np.float32) / 255.0

    # Append to deque (will auto-drop oldest when full)
    frames_deque.append(small)

    anomaly_prob = None
    status_text = "Warming up..."
    color = (0, 255, 0)  # green by default

    # If we have enough frames, run prediction
    if len(frames_deque) == SEQUENCE_LENGTH:
        # Prepare batch of shape (1, SEQ, H, W, 3)
        seq = np.array(frames_deque, dtype=np.float32)
        seq = np.expand_dims(seq, axis=0)
        preds = model.predict(seq, verbose=0)  # shape (1, 2)
        # Based on your training labels mapping: [anomaly, normal]
        anomaly_prob = float(preds[0][0])
        normal_prob = float(preds[0][1])

        # Read threshold from trackbar (0-100)
        thresh = cv2.getTrackbarPos('Threshold x100', WINDOW_NAME) / 100.0

        # Decide status
        if anomaly_prob >= thresh:
            status_text = f"ANOMALY ({anomaly_prob:.2f})"
            color = (0, 0, 255)  # red
        else:
            status_text = f"Normal ({1.0 - anomaly_prob:.2f})"
            color = (0, 255, 0)  # green

        # Draw the threshold bar and probabilities
        cv2.putText(display_frame, f"Anomaly prob: {anomaly_prob:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Normal prob: {normal_prob:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Threshold: {thresh:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # If anomaly, put a big ALERT box
        if anomaly_prob >= thresh:
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 60), (0, 0, 255), -1)
            cv2.putText(display_frame, f"!!! ANOMALY ALERT: {anomaly_prob:.2f} !!!", (10, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    else:
        # show progress to filling sequence
        k = len(frames_deque)
        cv2.putText(display_frame, f"Collecting frames: {k}/{SEQUENCE_LENGTH}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

    # Draw small status background and text bottom-left
    cv2.rectangle(display_frame, (5, display_frame.shape[0]-45),
                  (260, display_frame.shape[0]-5), (0,0,0), -1)
    cv2.putText(display_frame, f"Status: {status_text}", (10, display_frame.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # FPS calculation
    curr_time = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else fps
    prev_time = curr_time
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1]-120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    # Show frame
    cv2.imshow(WINDOW_NAME, display_frame)

    # keyboard
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Optional quick threshold adjustments via keys
    elif key == ord('=') or key == ord('+'):
        pos = cv2.getTrackbarPos('Threshold x100', WINDOW_NAME)
        cv2.setTrackbarPos('Threshold x100', WINDOW_NAME, min(100, pos + 5))
    elif key == ord('-') or key == ord('_'):
        pos = cv2.getTrackbarPos('Threshold x100', WINDOW_NAME)
        cv2.setTrackbarPos('Threshold x100', WINDOW_NAME, max(0, pos - 5))

# cleanup
cap.release()
cv2.destroyAllWindows()
