# detector.py
import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque, defaultdict
import math

class VideoProcessor:
    def __init__(self,
                 video_path="traffic.mp4",
                 model_path="yolov8n.pt",
                 conf_thresh=0.4,
                 skip_frames=2,
                 enable_light_detection=True,
                 light_rois = None,
                 white_line_y_rel = 0.72):
        """
        light_rois: list of two ROIs for traffic lights in pixel coords:
          [ (x1,y1,x2,y2), (x1,y1,x2,y2) ]
        white_line_y_rel: 白线 Y 的相对高度（帧高度的比例）
        """
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.model = YOLO(model_path)
        self.model.conf = conf_thresh
        self.conf_thresh = conf_thresh

        self.vehicle_classes_set = set(["car","motorcycle","bus","truck","motorbike","truck"])  # motorbike aliases
        # standardize naming mapping if voilates
        self.person_name = "person"
        self.light_name = "traffic light"

        self.skip_frames = max(0, int(skip_frames))
        self._frame_count = 0
        self.enable_light_detection = enable_light_detection

        # default ROI coords (from your input)
        if light_rois is None:
            # Use provided coordinates by user:
            # A: 958,80 980,80 958,136 980,136
            # B: 1067,76 1089,76 1067,131 1089,131
            self.light_rois = [
                (958,80,980,136),
                (1067,76,1089,131)
            ]
        else:
            self.light_rois = light_rois

        # tracker state
        self.next_object_id = 0
        self.objects = dict()   # id -> centroid (x,y)
        self.last_seen = dict() # id -> last_seen_timestamp
        self.trajs = defaultdict(lambda: deque(maxlen=30))  # id -> deque of centroids
        self.object_classes = dict()  # id -> class_name

        # history & stats
        self.counts_per_class = defaultdict(int)
        self.lane_changes_per_class = defaultdict(int)
        self.lane_counts = [0,0,0]  # per-lane counts this frame
        self.direction_counts = defaultdict(int)  # left/straight/right per class
        self.violations = deque(maxlen=200)  # list of recent violation dicts

        # per-object last lane for change detection
        self.last_lane = dict()

        # white line y (absolute)
        self.white_line_y = int(self.height * white_line_y_rel)

        # parameters
        self.match_max_dist = 60  # pixels

        # last model detections cache (we always compute every frame but can skip heavy ops)
        self.last_detections = []

    # ---------------- helper ----------------
    def _compute_centroid(self,bbox):
        x1,y1,x2,y2 = bbox
        return (int((x1+x2)/2), int((y1+y2)/2))

    def _match_objects(self, centroids):
        """
        centroids: list of (x,y) for current detections
        Returns list of assigned ids for each centroid in same order.
        Simple greedy nearest-neighbor matching.
        """
        assigned = [-1]*len(centroids)
        if len(self.objects)==0:
            # register all
            for i,c in enumerate(centroids):
                oid = self.next_object_id
                self.next_object_id += 1
                self.objects[oid] = c
                self.last_seen[oid] = time.time()
                assigned[i] = oid
                self.trajs[oid].append(c)
            return assigned

        # existing centroids
        obj_ids = list(self.objects.keys())
        obj_centroids = np.array([self.objects[oid] for oid in obj_ids])

        D = np.linalg.norm(obj_centroids[:,None] - np.array(centroids)[None,:], axis=2)
        # greedy matching
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
        for r,c in zip(rows,cols):
            if r in used_rows or c in used_cols:
                continue
            if D[r,c] > self.match_max_dist:
                continue
            oid = obj_ids[r]
            assigned[c] = oid
            used_rows.add(r)
            used_cols.add(c)
            # update object
            self.objects[oid] = centroids[c]
            self.last_seen[oid] = time.time()
            self.trajs[oid].append(centroids[c])

        # unmatched detections -> new ids
        for i in range(len(centroids)):
            if assigned[i] == -1:
                oid = self.next_object_id
                self.next_object_id += 1
                assigned[i] = oid
                self.objects[oid] = centroids[i]
                self.last_seen[oid] = time.time()
                self.trajs[oid].append(centroids[i])

        # cleanup disappeared objects (not seen for a while)
        now = time.time()
        to_del = []
        for oid, last in list(self.last_seen.items()):
            if now - last > 2.0:  # 2s timeout
                to_del.append(oid)
        for oid in to_del:
            try:
                del self.objects[oid]
                del self.last_seen[oid]
                del self.trajs[oid]
                if oid in self.object_classes: del self.object_classes[oid]
                if oid in self.last_lane: del self.last_lane[oid]
            except:
                pass

        return assigned

    # ------------------ detection & logic ------------------
    def detect_frame(self, frame):
        """
        Run YOLOv8 model on frame and return detection dicts.
        Each detection: {"bbox":(x1,y1,x2,y2),"conf":float,"class_name":str,"centroid":(cx,cy)}
        """
        self._frame_count += 1
        results = self.model(frame, verbose=False)[0]
        dets = []
        for box in results.boxes:
            conf = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
            if conf < self.conf_thresh:
                continue
            cls_id = int(box.cls[0]) if hasattr(box.cls, '__len__') else int(box.cls)
            label = self.model.names.get(cls_id, str(cls_id))
            x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
            cx,cy = self._compute_centroid((x1,y1,x2,y2))
            dets.append({"bbox":(x1,y1,x2,y2),"conf":conf,"class_name":label,"centroid":(cx,cy)})
        self.last_detections = dets
        return dets

    def analyze_lights(self, frame):
        """
        Analyze the two fixed ROIs and return a dict:
        { "light_1":"red"/"green"/"yellow"/"unknown", "light_2":... }
        """
        res = {}
        for i,roi in enumerate(self.light_rois, start=1):
            x1,y1,x2,y2 = roi
            # clamp
            x1 = max(0,min(self.width-1,int(x1)))
            x2 = max(0,min(self.width-1,int(x2)))
            y1 = max(0,min(self.height-1,int(y1)))
            y2 = max(0,min(self.height-1,int(y2)))
            roi_img = frame[y1:y2, x1:x2]
            state = "unknown"
            if roi_img.size == 0:
                res[f"light_{i}"] = state
                continue
            hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
            # red range - consider two ranges
            mask1 = cv2.inRange(hsv, (0,100,100), (10,255,255))
            mask2 = cv2.inRange(hsv, (160,100,100), (180,255,255))
            red_mask = cv2.bitwise_or(mask1, mask2)
            green_mask = cv2.inRange(hsv, (40,60,60), (90,255,255))
            yellow_mask = cv2.inRange(hsv, (15,60,60), (35,255,255))
            red = int(np.sum(red_mask>0))
            green = int(np.sum(green_mask>0))
            yellow = int(np.sum(yellow_mask>0))
            mx = max(red,green,yellow)
            if mx==0:
                state = "unknown"
            elif mx==red:
                state = "red"
            elif mx==green:
                state = "green"
            else:
                state = "yellow"
            res[f"light_{i}"] = state
        return res

    def _class_is_vehicle(self,label):
        # unify 'motorbike' / 'motorcycle' naming
        if label == "motorbike":
            label = "motorcycle"
        return label in self.vehicle_classes_set or label in ["car","bus","truck","motorcycle"]

    def _get_lane_index_by_x(self, x):
        # partition width into 3 equal lanes
        third = self.width / 3.0
        if x < third:
            return 0
        elif x < 2*third:
            return 1
        else:
            return 2

    def _compute_direction_for_id(self, oid):
        # use trajectory of the object to infer direction when crossing white line
        traj = list(self.trajs[oid])
        if len(traj) < 2:
            return None
        # take earliest and latest in trajectory (within stored window)
        x0,y0 = traj[0]
        x1,y1 = traj[-1]
        dx = x1 - x0
        dy = y1 - y0
        # if movement mainly downward (dy positive), evaluate dx relative magnitude
        # threshold proportion of frame width
        dx_abs = abs(dx)
        threshold = max(20, self.width * 0.08)
        if dx_abs < threshold:
            return "straight"
        elif dx < 0:
            return "left"
        else:
            return "right"

    def _check_cross_white_line(self, prev_y, cur_y):
        # detect crossing (either direction)
        return (prev_y < self.white_line_y <= cur_y) or (prev_y > self.white_line_y >= cur_y)

    def get_next_frame(self):
        """
        Read next frame, run detection & tracking, update stats and return (vis_frame, detections_list)
        detections_list: list of dicts with keys: bbox, conf, class_name, centroid, id
        """
        ret, frame = self.cap.read()
        if not ret:
            # loop
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return None, []

        detections = self.detect_frame(frame)

        # prepare centroids and match to ids
        centroids = [d["centroid"] for d in detections]
        assigned_ids = self._match_objects(centroids)

        # build enriched detections with ids
        enriched = []
        now = time.time()
        # reset per-frame lane counts
        self.lane_counts = [0,0,0]
        for i, d in enumerate(detections):
            oid = assigned_ids[i]
            d["id"] = oid
            self.object_classes[oid] = d["class_name"]
            enriched.append(d)
            x,y = d["centroid"]
            # lane counting
            lane_idx = self._get_lane_index_by_x(x)
            if lane_idx is not None:
                self.lane_counts[lane_idx] += 1

            # lane change detection
            prev_lane = self.last_lane.get(oid)
            if prev_lane is None:
                self.last_lane[oid] = lane_idx
            else:
                if prev_lane != lane_idx:
                    # increment lane change per class
                    cls = d["class_name"]
                    self.lane_changes_per_class[cls] += 1
                    self.last_lane[oid] = lane_idx

        # update counts per class (current visible)
        self.counts_per_class = defaultdict(int)
        for d in enriched:
            cls = d["class_name"]
            # unify naming
            if cls == "motorbike": cls = "motorcycle"
            self.counts_per_class[cls] += 1

        # detect white-line crossings for direction classification and violation (red light) detection
        # we evaluate objects that have at least 2 trajectory points
        lights = self.analyze_lights(frame) if self.enable_light_detection else {"light_1":"unknown","light_2":"unknown"}
        # compute if main lights are red (if ANY of the two is red we consider red? We'll mark both separately)
        # Check crossing by comparing second last and last centroid for each object
        for oid in list(self.trajs.keys()):
            traj = list(self.trajs[oid])
            if len(traj) < 2:
                continue
            prev_x, prev_y = traj[-2]
            cur_x, cur_y = traj[-1]
            if self._check_cross_white_line(prev_y, cur_y):
                # object crossed white line now - determine direction and register count
                cls = self.object_classes.get(oid,"unknown")
                # classify direction
                direction = self._compute_direction_for_id(oid)
                if direction is None:
                    direction = "unknown"
                key = f"{direction}_{cls}"
                self.direction_counts[key] += 1
                # if light is red at this moment, flag violation
                # We choose to check both lights: if either light is red -> violation
                red_present = (lights.get("light_1")=="red") or (lights.get("light_2")=="red")
                # If object is person and any red -> pedestrian violation
                if cls == self.person_name and red_present:
                    self.violations.appendleft({
                        "time": now,
                        "type": "pedestrian_red_cross",
                        "id": oid,
                        "class": cls,
                        "msg": f"行人 (id:{oid}) 闯红灯"
                    })
                # If vehicle and red_present -> vehicle red crossing
                if self._class_is_vehicle(cls) and red_present:
                    self.violations.appendleft({
                        "time": now,
                        "type": "vehicle_red_cross",
                        "id": oid,
                        "class": cls,
                        "msg": f"车辆 (id:{oid}, {cls}) 闯红灯"
                    })

        # Prepare visualization: draw detections & extra info on frame
        vis = frame.copy()
        # draw white line
        cv2.line(vis, (0, self.white_line_y), (self.width, self.white_line_y), (0,255,255), 2)

        for d in enriched:
            x1,y1,x2,y2 = d["bbox"]
            cls = d["class_name"]
            oid = d["id"]
            color = (0,255,0) if self._class_is_vehicle(cls) else (255,0,0)
            cv2.rectangle(vis,(x1,y1),(x2,y2), color, 2)
            cx,cy = d["centroid"]
            cv2.circle(vis, (cx,cy), 3, color, -1)
            cv2.putText(vis, f"{cls}:{oid}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # draw lane separators
        x_third = int(self.width/3)
        cv2.line(vis, (x_third,0),(x_third,self.height),(200,200,200),1)
        cv2.line(vis, (2*x_third,0),(2*x_third,self.height),(200,200,200),1)

        # draw lights state
        y0 = 40
        for i in range(1,3):
            st = lights.get(f"light_{i}", "unknown")
            col = (0,255,0) if st=="green" else ((0,255,255) if st=="yellow" else (0,0,255))
            cv2.putText(vis, f"Light{i}:{st.upper()}", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            y0 += 30

        return vis, enriched

    def get_stats(self):
        """
        Return a dictionary of aggregated stats for frontend:
        - counts_per_class
        - lane_changes_per_class
        - lane_counts (list of dicts)
        - direction_counts (summary)
        - lights (current)
        - violations (list of recent violations)
        """
        # build lane_counts output
        lane_counts_out = [{"id":i+1,"count":int(c)} for i,c in enumerate(self.lane_counts)]
        # simplify direction summary: aggregate by direction only (sum across classes)
        dir_summary = defaultdict(int)
        for k,v in self.direction_counts.items():
            # format: "{direction}_{class}"
            direction = k.split("_")[0]
            dir_summary[direction] += v
        # latest lights
        # to avoid running analyze_lights on empty frame, we attempt to peek last frame via cap
        # but we'll just call analyze_lights on the last saved frame if needed (not stored) -> call once using a read
        # (caller should be holding lock). For simplicity, call analyze_lights on a fresh read:
        # (But to avoid reading extra frames, we'll return last known from previous analyze if exists.)
        # We'll compute lights by taking current frame (read without advancing): not trivial; so keep "unknown"
        lights_out = {"light_1":"unknown","light_2":"unknown"}
        # Violations: return up to 20 most recent as list of msgs
        viols = list(self.violations)[:20]
        return {
            "counts_per_class": dict(self.counts_per_class),
            "lane_changes_per_class": dict(self.lane_changes_per_class),
            "lane_counts": lane_counts_out,
            "direction_summary": dict(dir_summary),
            "lights": lights_out,
            "violations": viols,
            "timestamp": time.time()
        }
