#!/usr/bin/env python3
import subprocess
import numpy as np
import cv2
import onnxruntime as ort
import time

# CONFIG

IMX_IN_PORT = 7000
WIN_OUT_IP = "192.168.0.105"
WIN_OUT_PORT = 6000

IMG_SIZE = 512
NUM_CLASSES = 5

MASK_MODEL = "/home/ECA/Models/mask2former_tiny.onnx"
YOLO_COCO_MODEL = "/home/ECA/Models/yolov8n.onnx"
YOLO_AERIAL_MODEL = "/home/ECA/Models/car_aerial_detection_yolo7_ITCVD_deepness.onnx"

CLASS_NAMES = ["Background", "Building", "Woodland", "Water", "Road"]

PALETTE_5 = np.array([
    [0,0,0], [255,87,51], [46,213,115], [52,172,224], [162,155,254]
], dtype=np.uint8)

PALETTE_BGR = PALETTE_5[:, ::-1]

# LOAD MODELS

def load_mask():
    sess = ort.InferenceSession(MASK_MODEL, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name

def load_yolo_coco():
    sess = ort.InferenceSession(YOLO_COCO_MODEL, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name

def load_yolo_aerial():
    sess = ort.InferenceSession(YOLO_AERIAL_MODEL, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name

# MAIN

def main():
    print("[INFO] Loading models...")
    mask_sess, mask_in = load_mask()
    yolo_aerial, y_a_in = load_yolo_aerial()
    yolo_coco, y_c_in = load_yolo_coco()

    width, height = 1280, 720  # OBS output

    ffmpeg_cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", f"udp://@:{IMX_IN_PORT}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-"
    ]

    print("[INFO] Starting ffmpeg input pipeline...")
    pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)

    out_url = f"udp://{WIN_OUT_IP}:{WIN_OUT_PORT}"
    out_writer = cv2.VideoWriter(
        out_url,
        cv2.VideoWriter_fourcc(*"H264"),
        30,
        (1280,720)
    )

    print("[INFO] Ready.")

    frame_size = width * height * 3

    while True:
        raw = pipe.stdout.read(frame_size)
        if len(raw) != frame_size:
            print("[WARN] Frame lost or incomplete.")
            continue

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb,(IMG_SIZE,IMG_SIZE))
        norm = resized.astype(np.float32)/255.0
        mean = np.array([0.485,0.456,0.406])
        std  = np.array([0.229,0.224,0.225])
        norm = (norm-mean)/std
        inp = np.transpose(norm,(2,0,1))[None]

        logits = mask_sess.run(None,{mask_in:inp})[0][0]

        logits_up = np.zeros((NUM_CLASSES,height,width))
        for c in range(NUM_CLASSES):
            logits_up[c] = cv2.resize(logits[c],(width,height))

        preds = np.argmax(logits_up.transpose(1,2,0),axis=-1)

        overlay = np.zeros_like(frame)
        alpha   = np.zeros((height,width),dtype=np.float32)

        for c in range(1,NUM_CLASSES):
            overlay[preds==c] = PALETTE_BGR[c]
            alpha[preds==c] = 0.55

        alpha = cv2.GaussianBlur(alpha,(15,15),0)
        seg_vis = (frame*(1-alpha[...,None]) + overlay*alpha[...,None]).astype(np.uint8)

        # SEND OUTPUT
        out_writer.write(seg_vis)

        print("[INFO] Frame processed")

if __name__ == "__main__":
    main()
