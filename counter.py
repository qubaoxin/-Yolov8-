# counter.py
"""
包含：
 - CentroidTracker: 基于质心的轻量跟踪器，保留 bbox + class_id + class_name
 - LineCounter: 基于跟踪结果按类别计数（横向/纵向过线）并做简易变道检测
兼容输入 detections 列表，detection 的单项可包含键：
  - "bbox": (x1,y1,x2,y2)
  - "class_id": int
  - "class_name": str
  - "confidence": float
或已包含 "centroid": (cx,cy)（若不存在会自动计算）
"""
from collections import OrderedDict
import numpy as np

# 可扩展的 COCO 类名映射（如果需要）
CLS_NAME = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=60):
        """
        简单的基于质心的跟踪器
        :param max_disappeared: 对象允许多少帧未被检测到后注销
        :param max_distance: 匹配时的最大质心距离（像素）
        """
        self.nextObjectID = 0
        # objects: objectID -> {"centroid":(x,y),"bbox":(...),"class_id":int,"class_name":str}
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox=None, class_id=None, class_name=None):
        self.objects[self.nextObjectID] = {
            "centroid": tuple(centroid),
            "bbox": tuple(bbox) if bbox is not None else None,
            "class_id": int(class_id) if class_id is not None else None,
            "class_name": class_name
        }
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        if objectID in self.objects:
            del self.objects[objectID]
        if objectID in self.disappeared:
            del self.disappeared[objectID]

    def _compute_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return (cx, cy)

    def update(self, detections):
        """
        更新 tracker
        :param detections: list of dict, each dict 包含 bbox / class_id / class_name / confidence (centroid 可选)
        :return: self.objects (OrderedDict)
        """
        # 如果没有检测到任何目标，增加 disappeared 计数并可能 deregister
        if len(detections) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        # 构建输入质心数组与其他信息
        input_centroids = []
        input_bboxes = []
        input_cls_ids = []
        input_cls_names = []

        for det in detections:
            if "centroid" in det and det["centroid"] is not None:
                centroid = tuple(det["centroid"])
            elif "bbox" in det and det["bbox"] is not None:
                centroid = self._compute_centroid(det["bbox"])
            else:
                # 如果连 bbox 都没有，跳过该检测项
                continue
            input_centroids.append(centroid)
            input_bboxes.append(tuple(det.get("bbox")) if det.get("bbox") is not None else None)
            input_cls_ids.append(det.get("class_id"))
            input_cls_names.append(det.get("class_name"))

        if len(self.objects) == 0:
            # 直接注册所有输入
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], bbox=input_bboxes[i],
                              class_id=input_cls_ids[i], class_name=input_cls_names[i])
            return self.objects

        # 现存对象的质心数组
        objectIDs = list(self.objects.keys())
        objectCentroids = np.array([self.objects[oid]["centroid"] for oid in objectIDs])

        # 计算距离矩阵 (num_existing x num_input)
        D = np.linalg.norm(objectCentroids[:, None] - np.array(input_centroids)[None, :], axis=2)

        # 贪心匹配：先按每个 existing 的最近距离排序
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.max_distance:
                continue
            objectID = objectIDs[row]
            # 更新对象信息
            self.objects[objectID]["centroid"] = tuple(input_centroids[col])
            self.objects[objectID]["bbox"] = tuple(input_bboxes[col]) if input_bboxes[col] is not None else self.objects[objectID].get("bbox")
            self.objects[objectID]["class_id"] = int(input_cls_ids[col]) if input_cls_ids[col] is not None else self.objects[objectID].get("class_id")
            self.objects[objectID]["class_name"] = input_cls_names[col] if input_cls_names[col] is not None else self.objects[objectID].get("class_name")
            self.disappeared[objectID] = 0
            usedRows.add(row)
            usedCols.add(col)

        # 未匹配到的 existing -> disappeared++
        unusedRows = set(range(0, D.shape[0])) - usedRows
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.max_disappeared:
                self.deregister(objectID)

        # 未匹配到的 input -> register new
        unusedCols = set(range(0, D.shape[1])) - usedCols
        for col in unusedCols:
            self.register(input_centroids[col], bbox=input_bboxes[col],
                          class_id=input_cls_ids[col], class_name=input_cls_names[col])

        return self.objects


class LineCounter:
    def __init__(self, line_position, direction="horizontal"):
        """
        :param line_position: 横线(y)或竖线(x)像素位置
        :param direction: "horizontal" 或 "vertical"
        """
        self.line_position = int(line_position)
        self.direction = direction
        # counts_per_class: {"car": int, ...}
        self.counts_per_class = {}
        # lane_changes_per_class: {"car": int, ...}
        self.lane_changes_per_class = {}
        # 用于判断是否已被计数，防止重复计数
        self.prev_centroids = {}  # objectID -> last centroid
        self.prev_lane = {}       # objectID -> lane index (for change detection)

    def _get_lane(self, centroid_x, frame_width, n_lanes=2):
        """
        简单把图像按垂直方向平均分成 n_lanes 车道，返回 lane index (0..n_lanes-1)
        """
        if frame_width is None or frame_width == 0:
            return None
        lane_width = frame_width / n_lanes
        idx = int(centroid_x // lane_width)
        if idx < 0: idx = 0
        if idx >= n_lanes: idx = n_lanes - 1
        return idx

    def update_counts(self, tracked_objects, frame_width=None):
        """
        tracked_objects: OrderedDict { objectID: {"centroid":(x,y),"bbox":(...),"class_name":str, ...}, ... }
        frame_width: 用于车道划分（可为 None）
        """
        for oid, info in tracked_objects.items():
            centroid = info.get("centroid")
            bbox = info.get("bbox")
            cls_name = info.get("class_name") or ("unknown")

            if centroid is None:
                # 如果没有质心但有 bbox，可以计算（冗余保障）
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    centroid = (int((x1 + x2)/2), int((y1 + y2)/2))
                else:
                    continue

            prev = self.prev_centroids.get(oid)
            # 首次见到仅记录，不计数
            if prev is None:
                self.prev_centroids[oid] = centroid
                # 记录 lane
                if frame_width is not None:
                    self.prev_lane[oid] = self._get_lane(centroid[0], frame_width)
                # ensure keys exist
                self.counts_per_class.setdefault(cls_name, 0)
                self.lane_changes_per_class.setdefault(cls_name, 0)
                continue

            # 横向过线计数（检查 y 坐标从一侧穿过 line_position）
            if self.direction == "horizontal":
                y_prev = prev[1]
                y_cur = centroid[1]
                crossed = (y_prev < self.line_position <= y_cur) or (y_prev > self.line_position >= y_cur)
                if crossed:
                    self.counts_per_class[cls_name] = self.counts_per_class.get(cls_name, 0) + 1
            else:
                x_prev = prev[0]
                x_cur = centroid[0]
                crossed = (x_prev < self.line_position <= x_cur) or (x_prev > self.line_position >= x_cur)
                if crossed:
                    self.counts_per_class[cls_name] = self.counts_per_class.get(cls_name, 0) + 1

            # 简易变道检测（基于按图像宽度平分车道）
            if frame_width is not None:
                cur_lane = self._get_lane(centroid[0], frame_width)
                prev_lane = self.prev_lane.get(oid)
                if prev_lane is not None and cur_lane is not None and prev_lane != cur_lane:
                    # 记录一次变道（按类别）
                    self.lane_changes_per_class[cls_name] = self.lane_changes_per_class.get(cls_name, 0) + 1
                self.prev_lane[oid] = cur_lane

            # 更新最后质心
            self.prev_centroids[oid] = centroid
            # ensure keys exist
            self.counts_per_class.setdefault(cls_name, 0)
            self.lane_changes_per_class.setdefault(cls_name, 0)

    def get_counts(self):
        return {
            "counts_per_class": dict(self.counts_per_class),
            "lane_changes_per_class": dict(self.lane_changes_per_class)
        }
