#!/usr/bin/env python3
import sys, os, time, threading, cv2, numpy as np, subprocess, json, requests
from flask import Flask, Response, jsonify
from collections import deque

sys.path.append(os.path.expanduser('~/.local/lib/python3.10/site-packages'))
from rknnlite.api import RKNNLite

# ---------- 初始化 NPU ----------
rknn = RKNNLite()
rknn.load_rknn('/home/elf/yolo_pose/yolov8n-pose_rk3588.rknn')
rknn.init_runtime()

# ---------- GStreamer 摄像头 ----------
def open_camera():
    pipeline = (
        "v4l2src device=/dev/video11 ! "
        "video/x-raw,format=NV12,width=640,height=480 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ GStreamer 摄像头打开失败")
        return None
    print("✅ GStreamer 摄像头已打开")
    return cap

cap = open_camera()
if cap is None:
    print("❌ 摄像头打开失败")
    sys.exit(1)

# ---------- 生成网格（YOLO后处理）----------
def get_map():
    grids, strides = [], []
    for s in [8, 16, 32]:
        h = w = 640 // s
        for y in range(h):
            for x in range(w):
                grids.append([x + 0.5, y + 0.5])
                strides.append(s)
    return np.array(grids), np.array(strides)
GRIDS, STRIDES = get_map()

# ---------- 全局数据 ----------
class DataNode:
    def __init__(self):
        self.rat = 0.0
        self.score = 100.0
        self.level = "Normal"
        self.frame = None
        self.lock = threading.Lock()
        self.session_start = time.time()
        self.distraction_counter = 0
        self.focus_sum = 0.0
        self.frame_cnt = 0
        self.eeg_state = 'NORMAL'
        self.eeg_score = 1.0
node = DataNode()

last_llm_message = ""

# ---------- 脑电后台接收监听 ----------
def start_eeg_listener():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 9999))
    while True:
        try:
            data, _ = sock.recvfrom(1024)
            st_str, sc_str = data.decode('utf-8').split(',')
            with node.lock:
                node.eeg_state = st_str
                node.eeg_score = float(sc_str)
        except:
            pass
threading.Thread(target=start_eeg_listener, daemon=True).start()

# ---------- 自适应校准 ----------
CALIB_FILE = "/home/elf/yolo_pose/config/user_calib.json"

def load_thresholds():
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, 'r') as f:
            data = json.load(f)
            return data.get("mild_thr", 0.2), data.get("severe_thr", 0.4)
    return 0.2, 0.4

def save_thresholds(mild, severe):
    os.makedirs(os.path.dirname(CALIB_FILE), exist_ok=True)
    with open(CALIB_FILE, 'w') as f:
        json.dump({"mild_thr": mild, "severe_thr": severe}, f)

def calibrate():
    print("🧘 请保持标准坐姿，校准开始... 保持10秒不要动")
    rat_samples = []
    start = time.time()
    old_mild, old_severe = load_thresholds()
    save_thresholds(1.0, 1.0)
    time.sleep(0.5)
    while time.time() - start < 10.0:
        with node.lock:
            rat_samples.append(node.rat)
        time.sleep(0.05)
    save_thresholds(old_mild, old_severe)
    if len(rat_samples) < 10:
        print("校准失败：采集数据不足")
        return False
    mean_rat = np.mean(rat_samples)
    std_rat = np.std(rat_samples)
    mild_thr = mean_rat + 1.5 * std_rat
    severe_thr = mean_rat + 3.0 * std_rat
    mild_thr = max(0.05, min(0.5, mild_thr))
    severe_thr = max(0.1, min(0.8, severe_thr))
    save_thresholds(mild_thr, severe_thr)
    print(f"✅ 校准完成！新阈值：轻度={mild_thr:.3f} 重度={severe_thr:.3f}")
    return True

MILD_THR, SEVERE_THR = load_thresholds()

# ---------- AI 推理线程 ----------
def ai_worker():
    global MILD_THR, SEVERE_THR, last_llm_message
    last_warn_time = 0
    last_light_warn = 0
    distraction_timer = 0.0

    os.system("amixer -c 1 sset 'Speaker' mute > /dev/null 2>&1")
    os.system("amixer -c 1 sset 'PCM' 64 > /dev/null 2>&1")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        img = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB)
        res = rknn.inference(inputs=[np.expand_dims(img, 0)])

        data = res[3][0]
        scores = np.sum(data[:, 2, :], axis=0)
        idx = np.argmax(scores)
        best = data[:, :, idx]
        g, s = GRIDS[idx], STRIDES[idx]

        def get_conf(val):
            if val > 1.0 or val < 0.0:
                return 1.0 / (1.0 + np.exp(-val))
            return val

        c0 = get_conf(best[0][2])
        c5 = get_conf(best[5][2])
        c6 = get_conf(best[6][2])

        with node.lock:
            current_eeg = node.eeg_state

        if c0 > 0.4 and c5 > 0.4 and c6 > 0.4:
            p0_x = (best[0][0] + g[0]) * s
            p5_x = (best[5][0] + g[0]) * s
            p6_x = (best[6][0] + g[0]) * s
            rat = abs(p0_x - (p5_x + p6_x)/2) / (abs(p5_x - p6_x) + 1e-6)

            p0_y = (best[0][1] + g[1]) * s
            p5_y = (best[5][1] + g[1]) * s
            p6_y = (best[6][1] + g[1]) * s
            scale_y = 480 / 640.0
            nx, ny = int(p0_x), int(p0_y * scale_y)
            lx, ly = int(p5_x), int(p5_y * scale_y)
            rx, ry = int(p6_x), int(p6_y * scale_y)

            cv2.circle(frame, (nx, ny), 6, (0, 0, 255), -1)
            cv2.circle(frame, (lx, ly), 6, (0, 255, 0), -1)
            cv2.circle(frame, (rx, ry), 6, (0, 255, 0), -1)
            cv2.line(frame, (lx, ly), (rx, ry), (255, 255, 0), 2)
            cv2.line(frame, (nx, ny), (int((lx+rx)/2), int((ly+ry)/2)), (0, 255, 255), 2)
        else:
            rat = 0.0

        with node.lock:
            node.rat = float(rat)
            node.frame = frame
            node.frame_cnt += 1
            node.focus_sum += (1.0 - rat)

            if rat > SEVERE_THR:
                node.level = "Severe"
                node.score = max(0.0, node.score - 0.5)
                distraction_timer += 0.05
                node.distraction_counter += 1
            elif rat > MILD_THR:
                node.level = "Mild"
                node.score = max(0.0, node.score - 0.1)
                distraction_timer += 0.05
                node.distraction_counter += 1
            else:
                node.level = "Normal"
                node.score = min(100.0, node.score + 0.2)
                distraction_timer = max(0, distraction_timer - 0.02)

            now = time.time()

            if 3 <= distraction_timer < 10 and (now - last_light_warn) > 3:
                if current_eeg == 'FOCUS':
                    print('📊 [脑电拦截] 坐姿微偏，但脑电高能专注，已拦截轻度提示。')
                else:
                    subprocess.Popen(["espeak", "-a", "150", "-s", "160", "-v", "zh", "注意坐姿哦"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                last_light_warn = now

            elif distraction_timer >= 5 and (now - last_warn_time) > 5:
                if current_eeg == 'FOCUS':
                    print('📊 [脑电拦截] 坐姿严重塌陷，但当前为极度专注状态，推迟大模型打扰。')
                    last_warn_time = now
                else:
                    elapsed = int(now - node.session_start)
                    avg_focus = node.focus_sum / max(1, node.frame_cnt)
                    profile = {
                        "name": "同学",
                        "total_study_minutes": elapsed // 60,
                        "avg_focus": avg_focus,
                        "distraction_count": node.distraction_counter
                    }
                    try:
                        resp = requests.post("http://127.0.0.1:8020/chat", json=profile, timeout=3)
                        if resp.status_code == 200:
                            reply = resp.json().get("reply", "注意休息，保持专注。")
                        else:
                            reply = "看起来有点累了，休息一下吧。"
                    except Exception as e:
                        reply = "调整一下坐姿，我们继续学习。"

                    last_llm_message = reply
                    subprocess.Popen(["espeak", "-a", "200", "-s", "170", "-v", "zh", reply],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    last_warn_time = now
                    distraction_timer = 0
                    node.distraction_counter = 0

threading.Thread(target=ai_worker, daemon=True).start()

# ---------- Flask 服务 ----------
app = Flask(__name__)

@app.route('/status')
def status():
    with node.lock:
        return jsonify({
            "rat": node.rat,
            "sc": int(node.score),
            "lvl": node.level,
            "eeg_st": node.eeg_state,
            "eeg_sc": round(node.eeg_score, 4)
        })

@app.route('/stream')
def stream():
    def gen():
        while True:
            with node.lock:
                if node.frame is None: continue
                f = node.frame.copy()
                level = node.level
                score = int(node.score)

            color = (0, 0, 255) if level == "Severe" else (0, 255, 255) if level == "Mild" else (0, 255, 0)
            cv2.rectangle(f, (5, 5), (320, 50), (20, 20, 20), -1)
            cv2.putText(f, f"{level} Score:{score}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            _, e = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 40])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + e.tobytes() + b'\r\n')
            time.sleep(0.04)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/calibrate')
def do_calibrate():
    if calibrate():
        global MILD_THR, SEVERE_THR
        MILD_THR, SEVERE_THR = load_thresholds()
        return "<html><body><h2>校准成功！阈值已生效。</h2></body></html>"
    return "<html><body><h2>校准失败。</h2></body></html>"

@app.route('/last_message')
def last_message():
    return jsonify({"message": last_llm_message})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
