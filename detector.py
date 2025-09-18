"""
detector.py
YOLO 封装：返回车辆检测结果并带上类别名（car/motorcycle/bus/truck）
"""
from ultralytics import YOLO
import numpy as np

# COCO 中车辆类的 id
VEHICLE_CLASS_MAP = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}
VEHICLE_CLASS_IDS = set(VEHICLE_CLASS_MAP.keys())

class Detector:
    def __init__(self, model_path="yolov8n.pt", device="cpu", conf=0.35):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf

    def detect(self, frame):
        """
        返回 list of detections:
        {
          "bbox": (x1,y1,x2,y2),
          "conf": float,
          "cls": int,
          "cls_name": str,
          "centroid": (cx,cy)
        }
        """
        results = self.model(frame, device=self.device, conf=self.conf)[0]

        # 兼容不同返回类型（torch / numpy）
        try:
            xyxys = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
        except Exception:
            xyxys = results.boxes.xyxy.numpy()
            clss = results.boxes.cls.numpy()
            confs = results.boxes.conf.numpy()

        detections = []
        for xyxy, cls, conf in zip(xyxys, clss, confs):
            cls = int(cls)
            if cls not in VEHICLE_CLASS_IDS:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": float(conf),
                "cls": cls,
                "cls_name": VEHICLE_CLASS_MAP.get(cls, "other"),
                "centroid": (cx, cy)
            })
        return detections
