import numpy as np
from collections import defaultdict

class LineCounter:
    def __init__(self, line_y, frame_width):
        # 用于车流量计数的参考线
        self.line_y = line_y
        self.frame_width = frame_width

        # 三条车道的边界（根据视频可调整）
        self.lane_boundaries = [
            (0, int(frame_width / 3)),                   # 左车道
            (int(frame_width / 3), int(2 * frame_width / 3)),  # 中车道
            (int(2 * frame_width / 3), frame_width)      # 右车道
        ]

        # 每类车辆的总数统计
        self.counts_per_class = defaultdict(int)

        # 每类车辆的车道变更次数
        self.lane_changes_per_class = defaultdict(int)

        # 当前追踪的对象：object_id -> (last_lane, last_y)
        self.tracked_objects = {}

        # 车道内的车辆分布计数
        self.lane_counts = [0, 0, 0]

    def _get_lane(self, bbox):
        """根据 bbox 的中心点确定所属车道"""
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        for i, (x_min, x_max) in enumerate(self.lane_boundaries):
            if x_min <= cx < x_max:
                return i
        return None

    def update(self, detections):
        """
        detections: [
            {"bbox": [x1, y1, x2, y2], "class_name": "car", "id": 15}, ...
        ]
        """
        # 每帧重置车道统计
        self.lane_counts = [0, 0, 0]

        for det in detections:
            bbox = det["bbox"]
            class_name = det["class_name"]
            obj_id = det.get("id", None)

            if obj_id is None:
                continue

            lane = self._get_lane(bbox)
            if lane is None:
                continue

            # 更新该车道车辆数量
            self.lane_counts[lane] += 1

            # 记录或更新对象的历史车道
            if obj_id not in self.tracked_objects:
                self.tracked_objects[obj_id] = lane
                self.counts_per_class[class_name] += 1  # 首次出现计数一次
            else:
                last_lane = self.tracked_objects[obj_id]
                if last_lane != lane:
                    self.lane_changes_per_class[class_name] += 1
                    self.tracked_objects[obj_id] = lane

    def get_counts(self):
        """返回完整的计数结果"""
        lane_counts_data = [
            {"id": i + 1, "count": count} for i, count in enumerate(self.lane_counts)
        ]
        return {
            "counts_per_class": dict(self.counts_per_class),
            "lane_changes_per_class": dict(self.lane_changes_per_class),
            "lane_counts": lane_counts_data
        }
