from typing import Any, Dict
import cv2
import numpy as np
import torch
from config import YOLO_CONFIDENCE

_original_torch_load = torch.load
def torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs: kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = torch_load_compat

try:
    from ultralytics import YOLO as UltralyticsYOLO
except ImportError:
    UltralyticsYOLO = None

try:
    import yolov5
except ImportError:
    yolov5 = None

class TiltDetector:
    def __init__(self, model_path: str, conf_thres: float = YOLO_CONFIDENCE):
        if UltralyticsYOLO is None: raise RuntimeError("Chưa cài ultralytics")
        self.model = UltralyticsYOLO(model_path)
        self.conf_thres = conf_thres

    def infer(self, frame_bgr) -> Dict[str, Any]:
        if frame_bgr is None: return {}
        results = self.model.predict(frame_bgr, verbose=False, conf=self.conf_thres)
        if not results: return {}

        r = results[0]
        label, conf, keypoints = None, None, None

        if r.boxes and len(r.boxes) > 0:
            box = r.boxes[0]
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = r.names.get(cls_id, str(cls_id))

        if r.keypoints is not None and r.keypoints.xy is not None and len(r.keypoints.xy) > 0:
            kpts = r.keypoints.xy[0].cpu().numpy()
            # Lấy 7 điểm đầu
            limited_kpts = kpts[:7]
            keypoints = [(float(x), float(y)) for x, y in limited_kpts]

        return {"label": label, "confidence": conf, "keypoints": keypoints}

class PostureDetector:
    def __init__(self, model_path: str, conf_thres: float = YOLO_CONFIDENCE):
        if yolov5 is None: raise RuntimeError("Chưa cài yolov5")
        self.model = yolov5.load(model_path)
        self.model.conf = conf_thres

    def infer(self, frame_bgr) -> Dict[str, Any]:
        if frame_bgr is None: return {}
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, size=640)

        if len(results.xywh[0]) == 0: return {}

        det = results.xywh[0][0]
        x_c, y_c, w, h, conf, cls = det.tolist()
        label = results.names[int(cls)]

        return {"label": label, "confidence": float(conf), "bbox": (float(x_c), float(y_c), float(w), float(h))}
