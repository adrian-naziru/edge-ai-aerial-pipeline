# Edge AI Pipeline for Vision Tasks on Aerial and Satellite Imagery

End-to-end edge AI pipeline for semantic segmentation and multi-task object detection on aerial imagery. A lightweight Mask2Former (1.165M params, mIoU 0.7862) combined with YOLOv8n and YOLO7-ITCVD, deployed via ONNX Runtime on the NXP i.MX 8M Plus EVK. Fully offline, no cloud dependency.

> **Paper:** *End-to-end Edge AI Pipeline for Vision Tasks on Aerial and Satellite Imagery*
> Adrian Naziru, Beatrice Gherghel, Ștefan-Daniel Achirei —
> Faculty of Automatic Control and Computer Engineering, Gheorghe Asachi Technical University of Iași, Romania

---

## Repository Structure

```
edge-ai-aerial-pipeline/
│
├── IMX_src/                          # Runs on NXP i.MX 8M Plus EVK (Linux, Cortex-A53)
│   ├── imx_server.py                 # TCP server: receives frames, runs all models, sends back result
│   ├── realtime_inference_mask2former.py  # Standalone local inference (USB camera)
│   └── test_connection.py            # FFmpeg UDP pipeline test
│
├── Windows_src/                      # Runs on Windows desktop
│   ├── benchmark.py                  # Reproduces Tables 3 & 4 from the paper
│   ├── realtime_inference_mask2former_windows_only.py  # Live demo via screen capture
│   ├── windows_client.py             # Sends frames to IMX8, displays segmented result
│   ├── train_mask2former.py          # Full training loop (GPU workstation)
│   ├── mask2former_flexible.py       # Model architecture
│   ├── dataset_torch.py              # Dataset and DataLoader
│   ├── losses_full.py                # Focal + Dice + Lovász hybrid loss
│   └── split_after_tiles.py          # Train/val/test split from tiles
│
├── models/                           # ONNX model weights (see Pre-trained Models below)
│   ├── mask2former_tiny.onnx
│   ├── yolov8n.onnx
│   └── car_aerial_detection_yolo7_ITCVD_deepness.onnx
│
└── README.md
```

---

## Hardware Requirements

| Component | Training | Inference (desktop) | Inference (edge) |
|-----------|----------|-------------------|-----------------|
| Platform  | GPU workstation | Windows x86 CPU | NXP i.MX 8M Plus EVK |
| CPU | Any modern x86 | Intel @ 2.4 GHz (single-thread) | Quad Cortex-A53 @ 1.8 GHz |
| GPU | NVIDIA (CUDA) | Not required | Not used (CPU only) |
| RAM | ≥ 16 GB | ≥ 8 GB | 4 GB LPDDR4 |
| Storage | ≥ 20 GB (dataset) | ≥ 2 GB | ≥ 1 GB |

---

## Installation

### Windows / Desktop (training + benchmark)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime opencv-python pillow numpy psutil tqdm ultralytics
pip install onnx  # optional, for parameter counting in benchmark
```

### NXP i.MX 8M Plus EVK (inference only)

```bash
pip3 install onnxruntime opencv-python numpy
# psutil optional, only needed for benchmark
pip3 install psutil
```

> The EVK runs Linux (Yocto / Ubuntu). ONNX Runtime CPU Execution Provider is used.
> No NPU delegation is active in the current release — NPU (INT8/TFLite) is planned as future work.

---

## Pre-trained Models

Download the three ONNX models and place them in the `models/` folder:

| Model | Task | Params | Size | Download |
|-------|------|--------|------|----------|
| `mask2former_tiny.onnx` | Semantic segmentation | 1.165 M | 4.63 MB | [Link](...) |
| `yolov8n.onnx` | Street-level detection (COCO) | 3.2 M | 12.26 MB | [Link](...) |
| `car_aerial_detection_yolo7_ITCVD_deepness.onnx` | Aerial vehicle detection | ~6.0 M | 23.01 MB | [Link](...) |

> `yolov8n.onnx` and `yolov8n.pt` can also be exported directly from [Ultralytics](https://github.com/ultralytics/ultralytics):
> ```bash
> yolo export model=yolov8n.pt format=onnx imgsz=640
> ```
> `car_aerial_detection_yolo7_ITCVD_deepness.onnx` is sourced from [PUTvision/qgis-plugin-deepness](https://github.com/PUTvision/qgis-plugin-deepness).

---

## Dataset Preparation

The segmentation model is trained on [LandCover.ai](https://landcover.ai/).

1. Download the dataset and extract tiles to `dataset_landcover/output/images/` and `dataset_landcover/output/masks/`.
2. Run the split script (70% train / 15% val / 15% test):

```bash
python Windows_src/split_after_tiles.py
```

This creates `train.txt`, `val.txt`, `test.txt` under `dataset_landcover/output/`.

**Mask encoding:** each mask PNG contains integer class labels per pixel:

| Value | Class |
|-------|-------|
| 0 | Background |
| 1 | Building |
| 2 | Woodland |
| 3 | Water |
| 4 | Road |

---

## Training

```bash
python Windows_src/train_mask2former.py
```

Key hyperparameters (edit at the top of `train_mask2former.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BACKBONE` | `"tiny"` | MobileNet-style CNN backbone |
| `IMAGE_SIZE` | `512` | Input resolution |
| `NUM_QUERIES` | `25` | Transformer decoder queries |
| `TRANSFORMER_LAYERS` | `3` | Decoder depth |
| `BATCH_SIZE` | `20` | Training batch size |
| `EPOCHS` | `40` | Total epochs |
| `DATASET_FRACTION` | `0.99` | Fraction of training data to use |

Checkpoints are saved to `checkpoints_mask2former/` after every epoch.
Qualitative results (overlay PNGs) are saved to `results_mask2former/`.

---

## Export to ONNX

After training, export the best checkpoint to ONNX:

```python
import torch
from mask2former_flexible import Mask2Former

model = Mask2Former(num_classes=5, backbone_name="tiny", num_queries=8, num_layers=1)
ckpt = torch.load("checkpoints_mask2former/best.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

dummy = torch.randn(1, 3, 512, 512)
torch.onnx.export(
    model, dummy, "models/mask2former_tiny.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=17
)
```

---

## Reproducing Benchmark Results (Tables 3 & 4)

Run on the Windows desktop to reproduce the single-thread CPU latency and FPS values:

```bash
python Windows_src/benchmark.py
```

Results are saved to `benchmark_results_windows.json` and `benchmark_results_windows.txt`.

> **Note:** The paper reports values measured on the NXP i.MX 8M Plus EVK (Cortex-A53 @ 1.8 GHz, single-thread). Desktop values differ; EVK values are estimated via the scaling factor in Table 2 of the paper. To obtain exact EVK numbers, copy `benchmark.py` to the EVK and run it there.

---

## Running the Live Demo

### Option A — Standalone on IMX8 (local USB camera)

```bash
# On the EVK
python3 IMX_src/realtime_inference_mask2former.py
```

Key bindings:

| Key | Action |
|-----|--------|
| `1` | YOLO Aerial ON |
| `2` | YOLO COCO ON |
| `3` | Both YOLO ON |
| `4` | YOLO OFF |
| `5` | Segmentation ON |
| `6` | Segmentation OFF |
| `7/8/9/0` | Toggle Building / Woodland / Water / Road overlay |
| `q` | Quit |

---

### Option B — Windows screen capture only

```bash
# On Windows (no EVK needed)
python Windows_src/realtime_inference_mask2former_windows_only.py
```

Captures the full screen, runs all three models locally, and displays the annotated output.

---

### Option C — Client/server (Windows → IMX8 → Windows)

```bash
# Step 1: On the EVK — start the inference server
python3 IMX_src/imx_server.py

# Step 2: On Windows — start the client (captures Google Maps window, sends to EVK)
# Edit IMX_IP in windows_client.py first
python Windows_src/windows_client.py
```

The server accepts console commands while running:

```
>> yolo 0          # 0=off, 1=aerial, 2=coco, 3=both
>> seg on/off
>> cls building on/off
>> cls woodland on/off
>> cls water on/off
>> cls road on/off
```

---

## Segmentation Results

Evaluated on the LandCover.ai test split at 512×512 resolution (FP32, CPU):

| Class | IoU | Precision | Recall | F1 |
|-------|-----|-----------|--------|----|
| Background | 0.9051 | 0.9698 | 0.9313 | 0.9502 |
| Building | 0.6916 | 0.8261 | 0.8095 | 0.8177 |
| Woodland | 0.8762 | 0.9098 | 0.9596 | 0.9340 |
| Water | 0.8931 | 0.9215 | 0.9667 | 0.9435 |
| Road | 0.5652 | 0.6673 | 0.7869 | 0.7222 |
| **mIoU** | **0.7862** | 0.8589 | 0.8908 | 0.8735 |

---

## Deployment Scenarios (EVK, CPU-only, single-thread)

| Scenario | Latency (ms) | FPS | RAM (MB) |
|----------|-------------|-----|----------|
| Seg only | 760 | 1.32 | 307.9 |
| YOLOv8n only | 913 | 1.10 | 307.9 |
| YOLO7-ITCVD only | 970 | 1.03 | 307.9 |
| Seg + YOLOv8n | 1536 | 0.65 | 307.9 |
| Seg + YOLOv8n + YOLO7 | 2513 | 0.40 | 307.9 |

Real-time (≥ 10 FPS) is not achievable in multi-model CPU-only mode. NPU acceleration (INT8, TFLite + Vx Delegate) is the planned next step.

---

## Citation

```bibtex
@inproceedings{naziru2025edgeai,
  title     = {End-to-end Edge {AI} Pipeline for Vision Tasks on Aerial and Satellite Imagery},
  author    = {Naziru, Adrian and Gherghel, Beatrice and Achirei, \c{S}tefan-Daniel},
  booktitle = {Proceedings of ...},
  year      = {2025}
}
```

---

## License

This project is released for academic and research purposes.
The YOLO7-ITCVD model is subject to the license of [PUTvision/qgis-plugin-deepness](https://github.com/PUTvision/qgis-plugin-deepness).
YOLOv8 is subject to the [Ultralytics license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
