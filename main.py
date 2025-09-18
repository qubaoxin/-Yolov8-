"""
main.py
主循环：读取视频/摄像头，运行 background-subtraction + YOLO 检测，
更新 centroid tracker，执行过线计数，并把 counts 写到 counts.json
"""
import cv2
import argparse
import json
import time
from detector import Detector
from counter import CentroidTracker, LineCounter

def draw_info(frame, tracked_objects, counts, line_y):
    # 画过线
    h, w = frame.shape[:2]
    cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)

    # 画每个 tracked 对象 bbox、id
    for oid, info in tracked_objects.items():
        bbox = info.get("bbox")
        centroid = info.get("centroid")
        cls = info.get("cls")
        if bbox is not None:
            x1,y1,x2,y2 = map(int, bbox)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            label = f"ID {oid} {cls}"
            cv2.putText(frame, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        if centroid is not None:
            cv2.circle(frame, tuple(map(int,centroid)), 3, (255,0,0), -1)

    # 在左上角显示 counts
    y = 20
    for k, v in counts.items():
        cv2.putText(frame, f"{k}: {v}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        y += 25

    return frame

def main(args):
    # 初始化
    detector = Detector(model_path=args.model, device=args.device, conf=args.conf)
    tracker = CentroidTracker(max_disappeared=30, max_distance=60)

    cap = cv2.VideoCapture(args.source if args.source is not None else 0)
    if not cap.isOpened():
        print("无法打开视频源：", args.source)
        return

    # 背景减法器（可视化与辅助）
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # 初始线（横向）放在底部 60% 处，或者由参数 override
    ret, frame = cap.read()
    if not ret:
        print("视频读取失败")
        return
    h, w = frame.shape[:2]
    line_y = int(h * args.line_ratio)

    line_counter = LineCounter(line_position=line_y, direction="horizontal")

    # 主循环
    frame_idx = 0
    counts_write_interval = 1.0  # seconds
    last_write_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 前景掩码（可用于调试/进一步过滤）
        fgmask = back_sub.apply(frame)
        # 一些形态学操作去噪
        fgmask = cv2.medianBlur(fgmask, 5)
        fgmask = cv2.dilate(fgmask, None, iterations=1)

        # 检测（为了实时可以每 N 帧检测一次；这里简单每帧检测）
        detections = detector.detect(frame)  # list of {bbox, conf, cls, centroid}

        # 更新 tracker（会 register / match / deregister）
        tracked = tracker.update(detections)

        # 更新计数器（基于 tracked objects）
        line_counter.update_counts(tracked)

        # 绘制信息
        counts = line_counter.get_counts()
        out = draw_info(frame, tracked, counts, line_y)

        cv2.imshow("Traffic Monitor", out)
        # 定期把 counts 写到 counts.json（供 server.py 读取）
        now = time.time()
        if now - last_write_time >= counts_write_interval:
            try:
                with open("counts.json", "w", encoding="utf-8") as f:
                    json.dump({"timestamp": time.time(), "counts": counts}, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print("写 counts.json 出错：", e)
            last_write_time = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="traffic.mp4",
                        help="视频路径，或摄像头索引 0（传 0）")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO 模型路径/名称")
    parser.add_argument("--device", type=str, default="cpu", help="设备：cpu 或 GPU id，如 '0'")
    parser.add_argument("--conf", type=float, default=0.35, help="检测置信度阈值")
    parser.add_argument("--line_ratio", type=float, default=0.60, help="过线在高度比例（0~1）")
    args = parser.parse_args()
    main(args)
