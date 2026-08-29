import time
import os
import numpy as np
import cv2
import threading
import sys
from arduino.app_utils import App, Bridge
from flask import Flask, Response, jsonify

# Line-buffer stdout so App Lab's log panel doesn't batch/withhold prints
sys.stdout.reconfigure(line_buffering=True)

import logging

# Protect the low-power CPU from OpenCV's internal thread pool bloating
#cv2.setNumThreads(1)
# ---- Thread Locks ----
kf_lock = threading.Lock()
frame_lock = threading.Lock()
detections_lock = threading.Lock()
bridge_lock = threading.Lock()  # Serializes every outbound Bridge.call()

# Suppress Flask request logs from cluttering the terminal
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


# CONFIG
CAMERA_INDEX=2
MODEL_INPUT_SIZE = 160  # Must match the deployed .onnx file's export resolution

# ---- Kalman Filter ----
class FusionKalman:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.array([50.0, 0.0])      # State: [distance, velocity]
        self.P = np.eye(2) * 100
        self.F = np.array([[1, -dt],
                           [0,  1]])
        self.H = np.array([[1, 0]])
        self.Q = np.array([[0.1, 0],
                           [0,   0.1]])

    def predict(self, a_vehicle=0.0):
        B = np.array([-self.dt**2 / 2, -self.dt])
        self.x = self.F @ self.x + B * a_vehicle
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, R):
        H = self.H
        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T / S
        self.x = self.x + K.flatten() * innovation
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P

    def get_TTC(self):
        d, v = self.x
        if v <= 0:
            return float('inf')
        return d / v

kf = FusionKalman(dt=0.02)

ALERT_COLORS = {0: "#1a7a1a", 1: "#b8860b", 2: "#cc5500", 3: "#cc0000"}
ALERT_LABELS = {0: "CLEAR", 1: "WARNING", 2: "SLOW DOWN", 3: "STOP"}

dashboard_state = {
    "fused_dist": 0.0,
    "velocity": 0.0,
    "ttc": float('inf'),
    "alert": 0
}

def get_alert_level(ttc):
    if ttc == float('inf') or ttc > 3.0: return 0
    elif ttc > 1.5: return 1
    elif ttc > 0.5: return 2
    else: return 3

last_print_time = 0

# ---- Bridge callback — called by MCU 50x/second ----
# No Bridge.call() happens inside here — outbound calls are handled
# entirely by alert_sender_worker() on an independent thread, avoiding
# the inbound/outbound reentrancy deadlock hit earlier tonight.
def update_sensors(us1, us2, ir, ax):
    global last_print_time

    with kf_lock:
        kf.predict(a_vehicle=ax)
        if us1 > 0: kf.update(z=us1, R=4.0)
        if us2 > 0: kf.update(z=us2, R=4.0)
        if ir > 0:  kf.update(z=ir, R=9.0)

        ttc = kf.get_TTC()
        fused_distance, closing_velocity = kf.x
        alert = get_alert_level(ttc)

    dashboard_state["fused_dist"] = fused_distance
    dashboard_state["velocity"] = closing_velocity
    dashboard_state["ttc"] = ttc
    dashboard_state["alert"] = alert

    current_time = time.time()
    if current_time - last_print_time > 0.2:
        print(f"Dist:{fused_distance:.1f}cm  V:{closing_velocity:.2f}cm/s  "
              f"TTC:{ttc:.2f}s  Alert:{ALERT_LABELS[alert]}")
        last_print_time = current_time

Bridge.provide("update_sensors", update_sensors)

# ---- Alert sender — independent thread, fires only on state change ----
def alert_sender_worker():
    last_sent_alert = -1
    while True:
        current_alert = dashboard_state["alert"]
        if current_alert != last_sent_alert:
            try:
                with bridge_lock:
                    Bridge.call("set_alert", current_alert)
                last_sent_alert = current_alert
            except Exception as e:
                print(f"[BRIDGE ERROR] set_alert failed: {e}")
        time.sleep(0.05)  # 20Hz poll rate

alert_thread = threading.Thread(target=alert_sender_worker, daemon=True)
alert_thread.start()

# ---- Autonomous Drive Ignition Thread ----
def start_autonomous_drive():
    time.sleep(2)  # Let Bridge fully stabilize before first call
    try:
        with bridge_lock:
            Bridge.call("set_manual_command", 1)  # 1 = Drive Forward
        print("[DRIVE] Autonomous forward command sent")
    except Exception as e:
        print(f"[BRIDGE ERROR] set_manual_command failed: {e}")

drive_thread = threading.Thread(target=start_autonomous_drive, daemon=True)
drive_thread.start()

# ==========================================================
# Camera Setup — Dedicated Grabber Loop
# ==========================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

latest_raw_frame = None

def camera_grabber_worker():
    global latest_raw_frame
    while True:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                with frame_lock:
                    latest_raw_frame = frame
        time.sleep(0.01)

camera_thread = threading.Thread(target=camera_grabber_worker, daemon=True)
camera_thread.start()

# ---- YOLO NN Engine Setup ----
MODEL_PATH = '/app/python/yolov5n.onnx'
print("Model file exists:", os.path.exists(MODEL_PATH))
if os.path.exists(MODEL_PATH):
    print("Model file size:", os.path.getsize(MODEL_PATH))

net = cv2.dnn.readNetFromONNX(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

if net is not None:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane",
           "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign", "parking meter", "bench",
           "bird", "cat", "dog", "horse", "sheep", "cow"]

last_detections = []

def inference_worker():
    global last_detections
    num_classes = len(CLASSES)

    while True:
        if net is None:
            time.sleep(1)
            continue

        frame_to_process = None
        with frame_lock:
            if latest_raw_frame is not None:
                frame_to_process = latest_raw_frame.copy()

        if frame_to_process is None:
            time.sleep(0.05)
            continue

        try:
            blob = cv2.dnn.blobFromImage(frame_to_process, 1/255.0,
                                         (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                                         swapRB=True)
            net.setInput(blob)
            output = net.forward()

            predictions = output[0]
            if predictions.ndim == 3:
                predictions = predictions[0]

            obj_confs = predictions[:, 4]
            class_scores = predictions[:, 5:5+num_classes]
            pred_class_ids = np.argmax(class_scores, axis=1)
            max_class_scores = class_scores[np.arange(len(class_scores)), pred_class_ids]
            total_scores = obj_confs * max_class_scores

            mask = total_scores > 0.4
            valid_indices = np.where(mask)[0]

            boxes, conf_scores, class_ids = [], [], []
            for idx in valid_indices:
                pred = predictions[idx]
                cx, cy, w, h = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
                x = int(cx - w/2)
                y = int(cy - h/2)
                boxes.append([x, y, int(w), int(h)])
                conf_scores.append(float(total_scores[idx]))
                class_ids.append(int(pred_class_ids[idx]))

            indices = cv2.dnn.NMSBoxes(boxes, conf_scores, score_threshold=0.4, nms_threshold=0.4)

            local_detections = []
            largest_area = 0
            primary_distance = None

            if len(indices) > 0:
                for i in np.array(indices).flatten():
                    det_box = boxes[i]
                    local_detections.append({
                        "box": det_box,
                        "confidence": conf_scores[i],
                        "class_id": class_ids[i]
                    })
                    w_b, h_b = det_box[2], det_box[3]
                    area = w_b * h_b
                    if area > largest_area:
                        largest_area = area
                        if area > 1.0:
                            primary_distance = 5000.0 / (area ** 0.5)

            with detections_lock:
                last_detections = local_detections

            if primary_distance is not None:
                with kf_lock:
                    kf.update(z=primary_distance, R=25.0)
                print(f"[CAM CALIBRATION] area:{largest_area:.0f}px  raw_dist:{primary_distance:.1f}cm")

            time.sleep(0.08)

        except Exception as e:
            print(f"[INFERENCE ERROR] {type(e).__name__}: {e}")
            time.sleep(0.05)

inference_thread = threading.Thread(target=inference_worker, daemon=True)
inference_thread.start()

# # ==========================================================
# # Flask Dashboard
# # ==========================================================
# flask_app = Flask(__name__)
# latest_frame = {"jpeg": None}

# @flask_app.route('/')
# def index():
#     return """
#     <html>
#     <head>
#         <title>Collision Avoidance Dashboard</title>
#         <style>
#             * { box-sizing: border-box; margin: 0; padding: 0; }
#             body { background:#111; color:#eee; font-family:sans-serif; text-align:center; padding:20px; }
#             h2 { color:#888; font-size:13px; letter-spacing:3px; margin-bottom:15px; }
#             .status { display:inline-block; padding:12px 40px; border-radius:8px; color:white; font-size:22px; font-weight:bold; margin-bottom:15px; transition:background 0.3s; }
#             .metrics { display:flex; justify-content:center; gap:20px; margin-bottom:15px; flex-wrap:wrap; }
#             .metric { background:#1e1e1e; border:1px solid #333; border-radius:8px; padding:14px 22px; min-width:130px; }
#             .metric-label { font-size:11px; color:#666; letter-spacing:1px; margin-bottom:6px; }
#             .metric-value { font-size:24px; font-weight:bold; color:#ddd; }
#             img { border:2px solid #2a2a2a; border-radius:6px; max-width:100%; }
#         </style>
#     </head>
#     <body>
#         <h2>SENSOR FUSION COLLISION AVOIDANCE SYSTEM</h2>
#         <div class="status" id="alert-box" style="background:#1a7a1a">CLEAR</div>
#         <div class="metrics">
#             <div class="metric"><div class="metric-label">FUSED DISTANCE</div><div class="metric-value" id="dist">-- cm</div></div>
#             <div class="metric"><div class="metric-label">CLOSING SPEED</div><div class="metric-value" id="vel">-- cm/s</div></div>
#             <div class="metric"><div class="metric-label">TIME TO COLLISION</div><div class="metric-value" id="ttc">-- s</div></div>
#         </div>
#         <img src="/video_feed" width="640"><br>
#         <script>
#             function updateData() {
#                 fetch('/data').then(r => r.json()).then(d => {
#                     document.getElementById('dist').textContent = d.dist + ' cm';
#                     document.getElementById('vel').textContent = d.vel + ' cm/s';
#                     document.getElementById('ttc').textContent = d.ttc + 's';
#                     const box = document.getElementById('alert-box');
#                     box.textContent = d.label; box.style.background = d.color;
#                 }).catch(() => {});
#             }
#             setInterval(updateData, 200); updateData();
#         </script>
#     </body>
#     </html>
#     """

# @flask_app.route('/data')
# def data():
#     state = dashboard_state
#     ttc = state['ttc']
#     return jsonify({
#         "dist": f"{state['fused_dist']:.1f}",
#         "vel": f"{state['velocity']:.2f}",
#         "ttc": f"{ttc:.2f}" if ttc != float('inf') else "inf",
#         "label": ALERT_LABELS[state['alert']],
#         "color": ALERT_COLORS[state['alert']]
#     })

# @flask_app.route('/video_feed')
# def video_feed():
#     def generate():
#         while True:
#             if latest_frame["jpeg"] is not None:
#                 yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + latest_frame["jpeg"] + b'\r\n')
#             time.sleep(0.1)
#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def run_flask():
#     flask_app.run(host='0.0.0.0', port=5000, threaded=True)

# flask_thread = threading.Thread(target=run_flask, daemon=True)
# flask_thread.start()

# last_jpeg_time = 0

# ==========================================================
# Main loop
# ==========================================================
def loop():
    global last_jpeg_time

    frame = None
    with frame_lock:
        if latest_raw_frame is not None:
            frame = latest_raw_frame.copy()

    if frame is None:
        time.sleep(0.02)
        return

    current_time = time.time()
    if current_time - last_jpeg_time > 0.1:
        draw_frame = frame.copy()

        with detections_lock:
            active_detections = list(last_detections)

        for detection in active_detections:
            x, y, w, h = detection["box"]
            confidence = detection["confidence"]
            class_id = detection["class_id"]

            H_f, W_f = draw_frame.shape[:2]
            x1 = int(x * W_f / MODEL_INPUT_SIZE)
            y1 = int(y * H_f / MODEL_INPUT_SIZE)
            x2 = int((x + w) * W_f / MODEL_INPUT_SIZE)
            y2 = int((y + h) * H_f / MODEL_INPUT_SIZE)

            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = CLASSES[class_id] if class_id < len(CLASSES) else f"obj_{class_id}"
            cv2.putText(draw_frame, f"{label} {confidence:.2f}", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        ret2, buffer = cv2.imencode('.jpg', draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 35])
        if ret2:
            latest_frame["jpeg"] = buffer.tobytes()

        last_jpeg_time = current_time

    time.sleep(0.04)

App.run(user_loop=loop)