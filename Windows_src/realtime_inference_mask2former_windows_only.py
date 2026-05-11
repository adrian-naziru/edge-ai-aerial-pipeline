import time
import os
import numpy as np

from PIL import ImageGrab
import cv2
import onnxruntime as ort
from ultralytics import YOLO

# CONFIG

IMG_SIZE = 512
NUM_CLASSES = 5

MASK_MODEL = "models/mask2former_tiny.onnx"
YOLO_COCO_MODEL = "models/yolov8n.onnx"
YOLO_AERIAL_MODEL = "models/car_aerial_detection_yolo7_ITCVD_deepness.onnx"

CLASS_NAMES = ["Background", "Building", "Woodland", "Water", "Road"]
SEG_VISIBLE = {
    "Building": True,
    "Woodland": True,
    "Water": True,
    "Road": True
}

PALETTE_5 = np.array([
    [0, 0, 0],           # Background transparent
    [255, 87,  51],      # Building - coral/orange
    [ 46,213,115],       # Woodland - green
    [ 52,172,224],       # Water - cyan/blue
    [162,155,254],       # Road - purple
], dtype=np.uint8)

PALETTE_BGR = PALETTE_5[:, ::-1]

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# LOAD MODELS

def load_mask_model():
    return ort.InferenceSession(
        MASK_MODEL,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )


def load_yolo_aerial():
    return ort.InferenceSession(
        YOLO_AERIAL_MODEL,
        providers=["CPUExecutionProvider"]
    )
def decode_yolo_aerial(output, img_w, img_h, conf_th=0.3):

    dets = output[0][0]  # (25200, 6)

    boxes = []
    for xc, yc, w, h, conf, cls in dets:

        if conf < conf_th:
            continue

        x1 = int(xc - w/2)
        y1 = int(yc - h/2)
        x2 = int(xc + w/2)
        y2 = int(yc + h/2)

        x1 = int(x1 * img_w / 640)
        y1 = int(y1 * img_h / 640)
        x2 = int(x2 * img_w / 640)
        y2 = int(y2 * img_h / 640)

        boxes.append((x1, y1, x2, y2, float(conf)))

    return boxes



def draw_yolo_aerial(frame, boxes):
    overlay = frame.copy()

    for (x1, y1, x2, y2, conf) in boxes:

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,190,255), 2)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,190,255), -1)

        cv2.putText(
            overlay, f"{conf:.2f}", (x1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1
        )

    frame = cv2.addWeighted(frame, 1.0, overlay, 0.35, 0)
    return frame

# MAIN LOOP

def main():

    mask_sess = load_mask_model()
    mask_input = mask_sess.get_inputs()[0].name

    yolo_coco = YOLO(YOLO_COCO_MODEL)
    yolo_aerial = load_yolo_aerial()

    mode_yolo = 0
    mask_enabled = True
    
    fps = 0.0

    print("""
F1 = YOLO Aerial ON
F2 = YOLO COCO ON
F3 = BOTH YOLO
F4 = YOLO OFF

F5 = SEG ON
F6 = SEG OFF
""")

    while True:
        loop_start = time.time()
        # SCREEN CAPTURE
        img_pil = ImageGrab.grab()
        frame = np.array(img_pil)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h,w = frame_bgr.shape[:2]

        seg_vis = frame_bgr.copy()

        # MASK2FORMER SEGMENTATION
        if mask_enabled:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (IMG_SIZE,IMG_SIZE))
            norm = resized.astype(np.float32) / 255.0

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            norm = (norm - mean) / std

            inp = np.transpose(norm, (2, 0, 1))[None, :, :, :]

            logits = mask_sess.run(None,{mask_input:inp})[0][0]

            logits_up = np.zeros((NUM_CLASSES,h,w),dtype=np.float32)
            for c in range(NUM_CLASSES):
                logits_up[c] = cv2.resize(logits[c], (w,h))

            preds = np.argmax(logits_up.transpose(1,2,0), axis=-1)

            # background transparent
            overlay = np.zeros_like(frame_bgr, dtype=np.uint8)
            alpha_mask = np.zeros((h,w), dtype=np.float32)

            for c in range(1, NUM_CLASSES):
                class_name = CLASS_NAMES[c]
                if SEG_VISIBLE[class_name]:
                    overlay[preds == c] = PALETTE_BGR[c]
                    alpha_mask[preds == c] = 0.55

            alpha_mask = cv2.GaussianBlur(alpha_mask, (15,15), 0)

            seg_vis = (seg_vis * (1 - alpha_mask[...,None]) +
                       overlay * alpha_mask[...,None]).astype(np.uint8)

        # YOLO AERIAL
        if mode_yolo in [1, 3]:

            resized640 = cv2.resize(frame_bgr, (640, 640))

            inp_y = resized640.astype(np.float32) / 255.0
            # NCHW
            inp_y = inp_y.transpose(2, 0, 1)[None]

            outputs = yolo_aerial.run(
                None,
                {yolo_aerial.get_inputs()[0].name: inp_y}
            )

            boxes = decode_yolo_aerial(outputs, w, h)
            seg_vis = draw_yolo_aerial(seg_vis, boxes)

        # YOLO COCO

        if mode_yolo in [2, 3]:
            res = yolo_coco(frame_bgr, verbose=False)[0].boxes

            for box in res:
                cls = int(box.cls[0])
                name = COCO_CLASSES[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                cv2.rectangle(seg_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(seg_vis, f"{name} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # FPS
        loop_end = time.time()
        dt = loop_end - loop_start
        fps = 1.0 / dt if dt > 0 else 0.0

        # HUD

        yolo_mode_text = ["OFF","AERIAL","COCO","BOTH"][mode_yolo]
        seg_text = "SEG ON" if mask_enabled else "SEG OFF"

        cv2.rectangle(seg_vis, (10, 10), (300, 130), (0, 0, 0), -1)

        cv2.putText(seg_vis, f"YOLO: {yolo_mode_text}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        cv2.putText(seg_vis, seg_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,200,0), 2)

        cv2.putText(seg_vis, f"FPS: {fps:.2f}", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        y0 = 155

        for cname in ["Building", "Woodland", "Water", "Road"]:
            state = "ON " if SEG_VISIBLE[cname] else "OFF"
            cv2.putText(seg_vis, f"{cname}: {state}", (20, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if SEG_VISIBLE[cname] else (0, 0, 255), 2)
            y0 += 25

        cv2.imshow("AI View", seg_vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('1'):
            mode_yolo = 1
        elif key == ord('2'):
            mode_yolo = 2
        elif key == ord('3'):
            mode_yolo = 3
        elif key == ord('4'):
            mode_yolo = 0
        elif key == ord('5'):
            mask_enabled = True
        elif key == ord('6'):
            mask_enabled = False

        #  TOGGLE SEGMENTATION CLASSES

        elif key == ord('7'):  # Building
            SEG_VISIBLE["Building"] = not SEG_VISIBLE["Building"]
            print("Building:", SEG_VISIBLE["Building"])

        elif key == ord('8'):  # Woodland
            SEG_VISIBLE["Woodland"] = not SEG_VISIBLE["Woodland"]
            print("Woodland:", SEG_VISIBLE["Woodland"])

        elif key == ord('9'):  # Water
            SEG_VISIBLE["Water"] = not SEG_VISIBLE["Water"]
            print("Water:", SEG_VISIBLE["Water"])

        elif key == ord('0'):  # Road
            SEG_VISIBLE["Road"] = not SEG_VISIBLE["Road"]
            print("Road:", SEG_VISIBLE["Road"])

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
