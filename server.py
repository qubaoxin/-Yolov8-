# server.py
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from detector import VideoProcessor
import cv2, threading, time

app = Flask(__name__)
app.secret_key = "traffic_monitor_secret_key"

# simple users store
users = {"admin":"123456"}

# initialize video processor with user-specified light ROIs (pixel coords)
light_rois = [
    (958,80,980,136),
    (1067,76,1089,131)
]
# default conf and skip frames can be tuned by frontend via /config (not implemented persistent here)
video_processor = VideoProcessor(video_path="traffic.mp4",
                                 model_path="yolov8n.pt",
                                 conf_thresh=0.4,
                                 skip_frames=2,
                                 enable_light_detection=True,
                                 light_rois=light_rois,
                                 white_line_y_rel=0.72)

frame_lock = threading.Lock()
latest_frame = None

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if users.get(username)==password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="用户名或密码错误")
    return render_template('login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method=='POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users:
            return render_template('register.html', error="用户名已存在")
        users[username]=password
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username',None)
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username'))

def mjpeg_generator():
    global latest_frame
    while True:
        with frame_lock:
            vis, dets = video_processor.get_next_frame()
            latest_frame = vis
        if vis is None:
            time.sleep(0.03)
            continue
        ret, buf = cv2.imencode('.jpg', vis)
        if not ret:
            continue
        frame = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    if 'username' not in session:
        return redirect(url_for('login'))
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/traffic_data')
def traffic_data():
    if 'username' not in session:
        return jsonify({"error":"未登录"}),403
    # read stats from processor
    with frame_lock:
        stats = video_processor.get_stats()
        # for lights we compute on latest_frame if available
        if latest_frame is not None:
            lights = video_processor.analyze_lights(latest_frame)
            stats['lights'] = lights
    return jsonify(stats)

# optional: API to update config (called by front-end controls)
@app.route('/config', methods=['GET','POST'])
def config():
    if 'username' not in session:
        return jsonify({"error":"未登录"}),403
    if request.method=='GET':
        return jsonify({
            "conf_thresh": video_processor.conf_thresh,
            "skip_frames": video_processor.skip_frames,
            "enable_light_detection": video_processor.enable_light_detection,
            "white_line_y_rel": float(video_processor.white_line_y)/video_processor.height
        })
    else:
        payload = request.json or {}
        c = payload.get("conf_thresh")
        if c is not None:
            video_processor.conf_thresh = float(c)
            video_processor.model.conf = float(c)
        s = payload.get("skip_frames")
        if s is not None:
            video_processor.skip_frames = int(s)
        en = payload.get("enable_light_detection")
        if en is not None:
            video_processor.enable_light_detection = bool(en)
        w = payload.get("white_line_y_rel")
        if w is not None:
            video_processor.white_line_y = int(float(w)*video_processor.height)
        return jsonify({"status":"ok"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
