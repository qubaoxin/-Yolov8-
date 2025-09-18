import cv2
import numpy as np
import json
import time
import threading
from collections import deque, defaultdict
from flask import Flask, Response, jsonify, render_template
from detector import Detector
from counter import CentroidTracker, LineCounter
import datetime

app = Flask(__name__)

from flask_cors import CORS

CORS(app)  # 允许所有域访问


class TrafficMonitor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.detector = Detector()
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=50)
        self.line_counter = LineCounter(line_position=400, direction="horizontal")

        # 数据存储
        self.vehicle_history = deque(maxlen=50)  # 限制历史记录长度
        self.class_counts = defaultdict(int)
        self.lane_change_counts = defaultdict(int)
        self.last_positions = {}
        self.frame_width = None
        self.latest_frame = None
        self.latest_stats = {
            "counts_per_class": {},
            "lane_changes_per_class": {},
            "line_counts": {"counts_per_class": {}, "lane_changes_per_class": {}},
            "history": [],
            "lane_counts": [],
            "timestamp": time.time(),
            # 新增趋势数据
            "traffic_trend": {
                "timestamps": [],
                "car": [], "motorcycle": [], "bus": [], "truck": [], "total": []
            },
            "lane_change_trend": {
                "timestamps": [],
                "car": [], "motorcycle": [], "bus": [], "truck": [], "total": []
            }
        }

        # 车道定义
        self.lanes = [
            {"id": 1, "y_min": 0, "y_max": 200, "count": 0},
            {"id": 2, "y_min": 200, "y_max": 400, "count": 0},
            {"id": 3, "y_min": 400, "y_max": 600, "count": 0}
        ]

        # 启动视频处理线程
        self.running = True
        self.thread = threading.Thread(target=self.process_video)
        self.thread.daemon = True
        self.thread.start()

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {self.video_path}")
            return

        # 获取视频的FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 0.033

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # 循环播放视频
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # 处理帧
            processed_frame, stats = self.process_frame(frame)
            self.latest_frame = processed_frame
            self.latest_stats = stats

            # 保存统计信息
            with open('counts.json', 'w') as f:
                json.dump(stats, f)

            # 控制处理速度以匹配视频FPS
            time.sleep(frame_delay)

        cap.release()

    def process_frame(self, frame):
        # 如果是第一帧，获取视频宽度用于车道检测
        if self.frame_width is None:
            self.frame_width = frame.shape[1]

        # 检测车辆
        detections = self.detector.detect(frame)

        # 格式化检测结果以适应CentroidTracker
        formatted_detections = []
        for detection in detections:
            formatted_detections.append({
                "bbox": detection["bbox"],
                "class_id": detection["cls"],
                "class_name": detection["cls_name"],
                "confidence": detection["conf"],
                "centroid": detection["centroid"]
            })

        # 更新跟踪器
        objects = self.tracker.update(formatted_detections)

        # 更新线路计数器
        self.line_counter.update_counts(objects, frame_width=self.frame_width)
        line_counts = self.line_counter.get_counts()

        # 更新车道计数和检测车道变更
        lane_changes = self._update_lane_counts(objects)

        # 更新历史数据
        timestamp = time.time()
        self._update_history(timestamp, objects, lane_changes, line_counts)

        # 更新趋势数据
        self._update_trend_data()

        # 绘制检测结果和统计信息
        output_frame = self._draw_results(frame, objects, lane_changes, line_counts)

        return output_frame, self._get_stats()

    def _update_lane_counts(self, objects):
        lane_changes = defaultdict(int)
        current_positions = {}

        for object_id, object_info in objects.items():
            centroid = object_info.get("centroid")
            if centroid is None:
                continue

            class_name = object_info.get("class_name", "unknown")
            current_positions[object_id] = centroid[1]  # 保存当前Y坐标

            # 确定当前车道
            current_lane = None
            for lane in self.lanes:
                if lane["y_min"] <= centroid[1] < lane["y_max"]:
                    current_lane = lane["id"]
                    break

            # 检查车道变更
            if object_id in self.last_positions:
                last_y = self.last_positions[object_id]
                last_lane = None
                for lane in self.lanes:
                    if lane["y_min"] <= last_y < lane["y_max"]:
                        last_lane = lane["id"]
                        break

                if last_lane is not None and current_lane != last_lane:
                    lane_changes[class_name] += 1
                    self.lane_change_counts[class_name] += 1

            # 更新车道计数
            if current_lane is not None:
                for lane in self.lanes:
                    if lane["id"] == current_lane:
                        lane["count"] += 1
                        break

        self.last_positions = current_positions
        return lane_changes

    def _update_history(self, timestamp, objects, lane_changes, line_counts):
        # 按类别计数
        class_counts = defaultdict(int)
        for object_info in objects.values():
            class_name = object_info.get("class_name", "unknown")
            class_counts[class_name] += 1

        # 更新总计数
        for class_name, count in class_counts.items():
            self.class_counts[class_name] = count

        # 保存历史记录
        self.vehicle_history.append({
            "timestamp": timestamp,
            "class_counts": dict(class_counts),
            "lane_changes": dict(lane_changes),
            "line_counts": line_counts,
            "total_vehicles": len(objects)
        })

    def _update_trend_data(self):
        """更新趋势数据"""
        # 清空趋势数据
        self.latest_stats["traffic_trend"] = {
            "timestamps": [],
            "car": [], "motorcycle": [], "bus": [], "truck": [], "total": []
        }
        self.latest_stats["lane_change_trend"] = {
            "timestamps": [],
            "car": [], "motorcycle": [], "bus": [], "truck": [], "total": []
        }

        # 填充趋势数据
        for record in self.vehicle_history:
            # 时间戳
            ts = datetime.datetime.fromtimestamp(record["timestamp"]).strftime('%H:%M:%S')
            self.latest_stats["traffic_trend"]["timestamps"].append(ts)
            self.latest_stats["lane_change_trend"]["timestamps"].append(ts)

            # 车流量数据
            self.latest_stats["traffic_trend"]["car"].append(record["class_counts"].get("car", 0))
            self.latest_stats["traffic_trend"]["motorcycle"].append(record["class_counts"].get("motorcycle", 0))
            self.latest_stats["traffic_trend"]["bus"].append(record["class_counts"].get("bus", 0))
            self.latest_stats["traffic_trend"]["truck"].append(record["class_counts"].get("truck", 0))
            self.latest_stats["traffic_trend"]["total"].append(record["total_vehicles"])

            # 车道变更数据
            self.latest_stats["lane_change_trend"]["car"].append(record["lane_changes"].get("car", 0))
            self.latest_stats["lane_change_trend"]["motorcycle"].append(record["lane_changes"].get("motorcycle", 0))
            self.latest_stats["lane_change_trend"]["bus"].append(record["lane_changes"].get("bus", 0))
            self.latest_stats["lane_change_trend"]["truck"].append(record["lane_changes"].get("truck", 0))
            self.latest_stats["lane_change_trend"]["total"].append(
                sum(record["lane_changes"].values()) if record["lane_changes"] else 0
            )

    def _get_stats(self):
        # 获取线路计数器的当前计数
        line_counts = self.line_counter.get_counts()

        # 更新趋势数据
        self._update_trend_data()

        return {
            "counts_per_class": dict(self.class_counts),
            "lane_changes_per_class": dict(self.lane_change_counts),
            "line_counts": line_counts,
            "history": list(self.vehicle_history),
            "lane_counts": [{"id": lane["id"], "count": lane["count"]} for lane in self.lanes],
            "timestamp": time.time(),
            "traffic_trend": self.latest_stats["traffic_trend"],
            "lane_change_trend": self.latest_stats["lane_change_trend"]
        }

    def _draw_results(self, frame, objects, lane_changes, line_counts):
        # 绘制检测框和ID
        for object_id, object_info in objects.items():
            centroid = object_info.get("centroid")
            if centroid is None:
                continue

            class_name = object_info.get("class_name", "unknown")

            # 绘制质心
            cv2.circle(frame, centroid, 4, (0, 255, 0), -1)

            # 绘制ID和类别
            text = f"ID:{object_id} {class_name}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 绘制计数线
        cv2.line(frame, (0, self.line_counter.line_position),
                 (frame.shape[1], self.line_counter.line_position),
                 (0, 255, 255), 2)

        # 绘制车道分隔线
        for lane in self.lanes:
            cv2.line(frame, (0, lane["y_max"]), (frame.shape[1], lane["y_max"]), (255, 0, 0), 1)

        # 绘制统计信息
        y_offset = 30
        stats_text = [
            f"Total: {len(objects)}",
            f"Line Crossings: {sum(line_counts.get('counts_per_class', {}).values())}",
            f"Lane Changes: {sum(lane_changes.values())}"
        ]

        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

        # 绘制各类别计数
        for class_name, count in self.class_counts.items():
            text = f"{class_name}: {count}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


# 创建全局监控器实例
monitor = TrafficMonitor("traffic.mp4")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if monitor.latest_frame is not None:
                # 编码帧为JPEG
                ret, jpeg = cv2.imencode('.jpg', monitor.latest_frame)
                if ret:
                    frame = jpeg.tobytes()
                    # 使用MJPEG格式流式传输帧
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            time.sleep(0.033)  # 约30FPS

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/traffic_data')
def traffic_data():
    return jsonify(monitor.latest_stats)


# 新增接口：获取车流量趋势数据
@app.route('/traffic_trend')
def traffic_trend():
    return jsonify(monitor.latest_stats["traffic_trend"])


# 新增接口：获取车道变更趋势数据
@app.route('/lane_change_trend')
def lane_change_trend():
    return jsonify(monitor.latest_stats["lane_change_trend"])


if __name__ == '__main__':
    try:
        app.run(host='localhost', port=5000, debug=True, threaded=True)
    except KeyboardInterrupt:
        print("正在停止服务器...")
    finally:
        monitor.stop()