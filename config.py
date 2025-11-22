import os

WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720
YOLO_CONFIDENCE = 0.25
GAZE_DEVICE = "cpu" # MediaPipe chạy CPU cực tốt trên Mac

EXPORT_BASE_DIR = os.path.join(os.path.dirname(__file__), "exports")
os.makedirs(EXPORT_BASE_DIR, exist_ok=True)
