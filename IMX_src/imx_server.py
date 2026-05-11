import socket
import struct
import numpy as np
import cv2
import onnxruntime as ort
import subprocess
import time
import os
import threading

PORT_IN = 9001   
PORT_OUT = 9002  

def free_port(port):

    try:
        cmd = f"netstat -tulpn 2>/dev/null | grep :{port} | awk '{{print $7}}'"
        out = subprocess.check_output(cmd, shell=True).decode().strip()
        if out:
            for item in out.split("\n"):
                pid = item.split("/")[0]
                if pid.isdigit():
                    print(f"[CLEANUP] Killing PID {pid} on port {port}")
                    os.system(f"kill -9 {pid}")
    except Exception:
        pass

free_port(PORT_IN)
free_port(PORT_OUT)
time.sleep(0.1)

IMG_SIZE = 512
NUM_CLASSES = 5

MASK_MODEL = "/home/ECA/Models/mask2former_tiny.onnx"
YOLO_COCO_MODEL = "/home/ECA/Models/yolov8n.onnx"
YOLO_AERIAL_MODEL = "/home/ECA/Models/car_aerial_detection_yolo7_ITCVD_deepness.onnx"

CLASS_NAMES = ["Background", "Building", "Woodland", "Water", "Road"]
SEG_VISIBLE = {
    "Building": True,
    "Woodland": True,
    "Water": True,
    "Road": True
}

PALETTE_5 = np.array([
    [0, 0, 0],
    [255, 87, 51],
    [46, 213, 115],
    [52, 172, 224],
    [162, 155, 254],
], dtype=np.uint8)
PALETTE_BGR = PALETTE_5[:, ::-1]

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird",
    "cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

print("[INFO] Loading ONNX models...")
mask_sess = ort.InferenceSession(MASK_MODEL, providers=["CPUExecutionProvider"])
mask_input = mask_sess.get_inputs()[0].name

yolo_coco_sess = ort.InferenceSession(YOLO_COCO_MODEL, providers=["CPUExecutionProvider"])
yolo_coco_input = yolo_coco_sess.get_inputs()[0].name

yolo_aerial = ort.InferenceSession(YOLO_AERIAL_MODEL, providers=["CPUExecutionProvider"])
yolo_aerial_input = yolo_aerial.get_inputs()[0].name

print("[OK] Models loaded.")

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def nms_xyxy(boxes, iou_th=0.45):

    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        rem = []
        for b in boxes:
            if b[5] != best[5]:
                rem.append(b)
                continue
            if iou_xyxy(best[:4], b[:4]) < iou_th:
                rem.append(b)
        boxes = rem
    return keep

def decode_yolo_coco(outputs, img_w, img_h, conf_th=0.25, input_size=640, nms_th=0.45):

    out = np.array(outputs[0])

    if out.ndim == 3:
        out = out[0]

    if out.shape[0] == 84 and out.shape[1] != 84:
        out = out.T

    if out.ndim != 2 or out.shape[1] != 84:
        print("[ERR] Unexpected YOLO COCO output shape:", out.shape)
        return []

    boxes = []
    sx = img_w / float(input_size)
    sy = img_h / float(input_size)

    for i in range(out.shape[0]):
        cx, cy, bw, bh = out[i, 0:4]
        cls_scores = out[i, 4:]  # 80

        cls_id = int(np.argmax(cls_scores))
        conf = float(cls_scores[cls_id])
        if conf < conf_th:
            continue

        x1 = int((cx - bw * 0.5) * sx)
        y1 = int((cy - bh * 0.5) * sy)
        x2 = int((cx + bw * 0.5) * sx)
        y2 = int((cy + bh * 0.5) * sy)

        # clamp
        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(0, min(img_w - 1, x2))
        y2 = max(0, min(img_h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append((x1, y1, x2, y2, conf, cls_id))

    boxes = nms_xyxy(boxes, iou_th=nms_th)
    return boxes

def draw_yolo_coco(frame, boxes):
    for (x1, y1, x2, y2, conf, cls) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = COCO_CLASSES[cls] if 0 <= cls < len(COCO_CLASSES) else str(cls)
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def decode_yolo_aerial(outputs, img_w, img_h, conf_th=0.3):

    dets = outputs[0][0]
    boxes = []
    for xc, yc, w, h, conf, cls in dets:
        if conf < conf_th:
            continue

        x1 = int((xc - w / 2) * img_w / 640)
        y1 = int((yc - h / 2) * img_h / 640)
        x2 = int((xc + w / 2) * img_w / 640)
        y2 = int((yc + h / 2) * img_h / 640)

        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(0, min(img_w - 1, x2))
        y2 = max(0, min(img_h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append((x1, y1, x2, y2, float(conf)))
    return boxes

def draw_yolo_aerial(frame, boxes):
    overlay = frame.copy()
    for (x1, y1, x2, y2, conf) in boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 190, 255), 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 190, 255), -1)
        cv2.putText(overlay, f"{conf:.2f}", (x1, max(0, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return cv2.addWeighted(frame, 1.0, overlay, 0.35, 0)

mode_yolo = 0          
mask_enabled = True   

def console_commands():
    global mode_yolo, mask_enabled, SEG_VISIBLE
    print(">> Console command mode ON (type 'help')")
    while True:
        try:
            cmd = input(">> ").strip().lower()
        except EOFError:
            time.sleep(0.2)
            continue
        except Exception:
            time.sleep(0.2)
            continue

        if cmd.startswith("yolo "):
            try:
                mode_yolo = int(cmd.split()[1])
                mode_yolo = max(0, min(3, mode_yolo))
                print("[CMD] YOLO mode set to", mode_yolo)
            except Exception:
                print("[ERR] Use: yolo 0/1/2/3")

        elif cmd == "seg on":
            mask_enabled = True
            print("[CMD] Segmentation ON")

        elif cmd == "seg off":
            mask_enabled = False
            print("[CMD] Segmentation OFF")

        elif cmd.startswith("cls "):
            try:
                _, cname, state = cmd.split()
                cname = cname.capitalize()
                if cname in SEG_VISIBLE:
                    SEG_VISIBLE[cname] = (state == "on")
                    print(f"[CMD] {cname} visibility -> {state.upper()}")
                else:
                    print(f"[ERR] Unknown class: {cname}")
            except Exception:
                print("[ERR] Use: cls building on/off")

        elif cmd == "help":
            print("""
Commands:
  yolo 0/1/2/3
  seg on/off
  cls building on/off
  cls woodland on/off
  cls water on/off
  cls road on/off
""")
        else:
            print("[ERR] Unknown command. Type help")

threading.Thread(target=console_commands, daemon=True).start()

sock_in = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_in.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock_in.bind(("0.0.0.0", PORT_IN))
sock_in.listen(1)

print("[INFO] Waiting for Windows client...")
conn_in, addr = sock_in.accept()
print("[INFO] Connected FROM Windows:", addr)

sock_out = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_out.connect((addr[0], PORT_OUT))
print("[INFO] Connected BACK to Windows.")

print("[INFO] Streaming loop started.")

while True:
    size_data = conn_in.recv(4)
    if not size_data:
        break

    size = struct.unpack(">I", size_data)[0]
    buf = b""
    while len(buf) < size:
        chunk = conn_in.recv(size - len(buf))
        if not chunk:
            break
        buf += chunk

    if not buf:
        break

    frame_bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        continue

    h, w = frame_bgr.shape[:2]
    seg_vis = frame_bgr.copy()

    if mask_enabled:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

        norm = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (norm - mean) / std

        inp = np.transpose(norm, (2, 0, 1))[None].astype(np.float32)
        logits = mask_sess.run(None, {mask_input: inp})[0][0] 

        logits_up = np.zeros((NUM_CLASSES, h, w), np.float32)
        for c in range(NUM_CLASSES):
            logits_up[c] = cv2.resize(logits[c], (w, h))

        preds = np.argmax(logits_up.transpose(1, 2, 0), axis=-1)

        overlay = np.zeros_like(frame_bgr)
        alpha = np.zeros((h, w), np.float32)

        for c in range(1, NUM_CLASSES):
            if SEG_VISIBLE[CLASS_NAMES[c]]:
                overlay[preds == c] = PALETTE_BGR[c]
                alpha[preds == c] = 0.55

        alpha = cv2.GaussianBlur(alpha, (15, 15), 0)
        seg_vis = (seg_vis * (1 - alpha[..., None]) + overlay * alpha[..., None]).astype(np.uint8)

    if mode_yolo in [1, 3]:
        resized640 = cv2.resize(frame_bgr, (640, 640))
        resized640 = cv2.cvtColor(resized640, cv2.COLOR_BGR2RGB) 
        inp_y = resized640.astype(np.float32) / 255.0
        inp_y = inp_y.transpose(2, 0, 1)[None].astype(np.float32)

        outputs = yolo_aerial.run(None, {yolo_aerial_input: inp_y})
        boxes = decode_yolo_aerial(outputs, w, h, conf_th=0.3)
        seg_vis = draw_yolo_aerial(seg_vis, boxes)

    if mode_yolo in [2, 3]:
        resized640 = cv2.resize(frame_bgr, (640, 640))
        resized640 = cv2.cvtColor(resized640, cv2.COLOR_BGR2RGB)
        inp_coco = resized640.astype(np.float32) / 255.0
        inp_coco = inp_coco.transpose(2, 0, 1)[None].astype(np.float32)

        outputs = yolo_coco_sess.run(None, {yolo_coco_input: inp_coco})


        boxes_coco = decode_yolo_coco(
            outputs, w, h,
            conf_th=0.25,    
            input_size=640,
            nms_th=0.45
        )
        seg_vis = draw_yolo_coco(seg_vis, boxes_coco)

    # HUD text
    yolo_mode_text = ["OFF", "AERIAL", "COCO", "BOTH"][mode_yolo]
    seg_text = "SEG ON" if mask_enabled else "SEG OFF"

    cv2.rectangle(seg_vis, (10, 10), (260, 90), (0, 0, 0), -1)
    cv2.putText(seg_vis, f"YOLO: {yolo_mode_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(seg_vis, seg_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

    y_offset = 120
    for cname in ["Building", "Woodland", "Water", "Road"]:
        state = "ON" if SEG_VISIBLE[cname] else "OFF"
        color = (0, 255, 0) if SEG_VISIBLE[cname] else (0, 0, 255)
        cv2.putText(seg_vis, f"{cname}: {state}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25

    # Send to Windows
    ok, encoded = cv2.imencode(".jpg", seg_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        continue
    data = encoded.tobytes()
    sock_out.sendall(struct.pack(">I", len(data)) + data)

# cleanup
print("[INFO] Closing sockets...")
try:
    conn_in.close()
except:
    pass
try:
    sock_out.close()
except:
    pass
try:
    sock_in.close()
except:
    pass
print("[DONE]")
