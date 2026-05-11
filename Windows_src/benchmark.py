import time
import os
import sys
import json
import threading
import statistics
import platform

import numpy as np
import psutil


IS_WINDOWS = platform.system() == "Windows"


try:
    import onnxruntime as ort
    ORT_OK = True

    PROVIDERS_AVAILABLE = ort.get_available_providers()
    HAS_CUDA = "CUDAExecutionProvider" in PROVIDERS_AVAILABLE
except ImportError:
    print("[ERROR] onnxruntime is not installed.")
    print("  Run: pip install onnxruntime  or  pip install onnxruntime-gpu")
    sys.exit(1)

MODELS = {
    "seg":          r"models\mask2former_tiny.onnx",
    "yolo_coco":    r"models\yolov8n.onnx",
    "yolo_aerial":  r"models\car_aerial_detection_yolo7_ITCVD_deepness.onnx",
}

IMG_SIZE_SEG   = 512
IMG_SIZE_YOLO  = 640
NUM_CLASSES    = 5
WARMUP_FRAMES  = 10
BENCH_FRAMES   = 100

OUTPUT_JSON = "benchmark_results_windows.json"
OUTPUT_TXT  = "benchmark_results_windows.txt"


def get_file_size_mb(path: str) -> float:
    if not os.path.isfile(path):
        return -1.0
    return os.path.getsize(path) / (1024 ** 2)


def get_process_ram_mb() -> float:
    """Return current process RSS memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def get_system_info() -> dict:
    vm = psutil.virtual_memory()
    return {
        "os":                 platform.platform(),
        "python":             sys.version.split()[0],
        "ort_version":        ort.__version__,
        "cpu_cores_logical":  psutil.cpu_count(logical=True),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_freq_mhz":       round(psutil.cpu_freq().max, 0) if psutil.cpu_freq() else "N/A",
        "ram_total_gb":       round(vm.total / 1024**3, 1),
        "ram_avail_gb":       round(vm.available / 1024**3, 1),
        "cuda_available":     HAS_CUDA,
        "ort_providers":      PROVIDERS_AVAILABLE,
    }


class CPUMonitor:
    """Monitors CPU% of the current process in a background thread."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.samples: list = []
        self._stop   = threading.Event()
        self._proc   = psutil.Process(os.getpid())

        self._proc.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._stop.clear()
        self.samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list:
        self._stop.set()
        self._thread.join(timeout=3)
        return self.samples

    def _run(self):
        while not self._stop.is_set():
            sample = self._proc.cpu_percent(interval=self.interval)
            self.samples.append(sample)


def make_dummy_seg() -> np.ndarray:
    img = np.random.randint(0, 256, (IMG_SIZE_SEG, IMG_SIZE_SEG, 3), dtype=np.uint8)
    img_f = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_f = (img_f - mean) / std
    return img_f.transpose(2, 0, 1)[None]  # (1, 3, 512, 512)


def make_dummy_yolo() -> np.ndarray:
    img = np.random.randint(0, 256, (IMG_SIZE_YOLO, IMG_SIZE_YOLO, 3), dtype=np.uint8)
    return img.astype(np.float32).transpose(2, 0, 1)[None] / 255.0  # (1, 3, 640, 640)


def count_onnx_params(model_path: str) -> int:
    try:
        import onnx
        m = onnx.load(model_path)
        total = 0
        for init in m.graph.initializer:
            n = 1
            for d in init.dims:
                n *= d
            total += n
        return total
    except ImportError:
        return -1  # onnx package not installed; non-blocking
    except Exception:
        return -1


def load_session(model_key: str, use_cuda: bool = False) -> tuple:
    path = MODELS[model_key]
    if not os.path.isfile(path):
        print(f"  [SKIP] Model not found: {path}")
        return None, None

    providers = []
    if use_cuda and HAS_CUDA:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1

    sess = ort.InferenceSession(path, sess_options=opts, providers=providers)
    inp_name = sess.get_inputs()[0].name
    actual_provider = sess.get_providers()[0]
    print(f"  {model_key:15s} | {os.path.basename(path)} | provider: {actual_provider}")
    return sess, inp_name


def bench_single_model(label: str, sess, inp_name: str, dummy: np.ndarray) -> dict:
    if sess is None:
        return {"error": "model not loaded"}

    print(f"\n  Warmup [{label}] ({WARMUP_FRAMES} frames) ...", end=" ", flush=True)
    for _ in range(WARMUP_FRAMES):
        sess.run(None, {inp_name: dummy})
    print("done")

    print(f"  Benchmark [{label}] ({BENCH_FRAMES} frames) ...", end=" ", flush=True)

    ram_before = get_process_ram_mb()
    cpu_mon = CPUMonitor()
    cpu_mon.start()

    latencies = []
    for _ in range(BENCH_FRAMES):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: dummy})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    cpu_samples = cpu_mon.stop()
    ram_after = get_process_ram_mb()

    print("done")

    avg_lat = statistics.mean(latencies)
    p95_lat = sorted(latencies)[int(0.95 * len(latencies))]
    fps     = 1000.0 / avg_lat

    return {
        "avg_latency_ms": round(avg_lat, 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "p95_latency_ms": round(p95_lat, 2),
        "fps":            round(fps, 2),
        "peak_ram_mb":    round(max(ram_before, ram_after), 1),
        "ram_delta_mb":   round(ram_after - ram_before, 1),
        "avg_cpu_pct":    round(statistics.mean(cpu_samples), 1) if cpu_samples else -1,
        "max_cpu_pct":    round(max(cpu_samples), 1) if cpu_samples else -1,
    }


def bench_scenario(scenario_label: str, model_list: list) -> dict:
    valid = [(s, n, d) for s, n, d in model_list if s is not None]
    if not valid:
        return {"error": "no models loaded"}

    print(f"\n  Warmup [{scenario_label}] ({WARMUP_FRAMES} frames) ...", end=" ", flush=True)
    for sess, inp_name, dummy in valid:
        for _ in range(WARMUP_FRAMES):
            sess.run(None, {inp_name: dummy})
    print("done")

    print(f"  Benchmark [{scenario_label}] ({BENCH_FRAMES} frames) ...", end=" ", flush=True)

    ram_before = get_process_ram_mb()
    cpu_mon = CPUMonitor()
    cpu_mon.start()

    latencies = []
    for _ in range(BENCH_FRAMES):
        t0 = time.perf_counter()
        for sess, inp_name, dummy in valid:
            sess.run(None, {inp_name: dummy})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    cpu_samples = cpu_mon.stop()
    ram_after = get_process_ram_mb()

    print("done")

    avg_lat = statistics.mean(latencies)
    fps     = 1000.0 / avg_lat
    rt_ok   = fps >= 10.0

    return {
        "avg_latency_ms": round(avg_lat, 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "fps":            round(fps, 2),
        "ram_mb":         round(ram_after, 1),
        "ram_delta_mb":   round(ram_after - ram_before, 1),
        "avg_cpu_pct":    round(statistics.mean(cpu_samples), 1) if cpu_samples else -1,
        "real_time":      "YES" if rt_ok else "NO",
        "real_time_bool": rt_ok,
    }


def fmt_table(title: str, headers: list, rows: list) -> str:
    col_w = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_w[i] = max(col_w[i], len(str(cell)))

    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    hdr = "|" + "|".join(f" {h:<{col_w[i]}} " for i, h in enumerate(headers)) + "|"
    lines = [f"\n{'='*60}", f"  {title}", "=" * 60, sep, hdr, sep]
    for row in rows:
        lines.append("|" + "|".join(f" {str(v):<{col_w[i]}} " for i, v in enumerate(row)) + "|")
    lines.append(sep)
    return "\n".join(lines)


def main():
    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log("=" * 60)
    log("  BENCHMARK – Edge AI Paper (Windows desktop reference)")
    log("=" * 60)

    sysinfo = get_system_info()
    for k, v in sysinfo.items():
        log(f"  {k:<25}: {v}")
    log()

    all_results = {"system": sysinfo}

    log("[1/4] Model info ...")
    model_info = {}
    for key, path in MODELS.items():
        size_mb = get_file_size_mb(path)
        params  = count_onnx_params(path)
        model_info[key] = {
            "path":     path,
            "exists":   os.path.isfile(path),
            "size_mb":  round(size_mb, 2) if size_mb > 0 else "N/A",
            "params_M": round(params / 1e6, 3) if params > 0 else "N/A",
        }
        if size_mb > 0:
            param_str = f" | {params/1e6:.3f} M params" if params > 0 else " | params: N/A (pip install onnx)"
            log(f"  {key:15s}: {size_mb:.2f} MB{param_str}")
        else:
            log(f"  [MISSING] {key}: model not found at {path}")
    all_results["model_info"] = model_info

    log(f"\n[2/4] Loading ORT sessions (single-threaded, CPU) ...")

    sess_seg,    inp_seg    = load_session("seg")
    sess_coco,   inp_coco   = load_session("yolo_coco")
    sess_aerial, inp_aerial = load_session("yolo_aerial")

    dummy_seg  = make_dummy_seg()
    dummy_yolo = make_dummy_yolo()

    log("\n[3/4] Single-model benchmarks (Table 3) ...")

    r_seg    = bench_single_model("Seg (Mask2Former)", sess_seg,    inp_seg,    dummy_seg)
    r_coco   = bench_single_model("YOLOv8n",           sess_coco,   inp_coco,   dummy_yolo)
    r_aerial = bench_single_model("YOLO7-ITCVD",       sess_aerial, inp_aerial, dummy_yolo)

    single_results = {
        "seg":         r_seg,
        "yolo_coco":   r_coco,
        "yolo_aerial": r_aerial,
    }
    all_results["single_model"] = single_results

    r_seg["model_size_mb"]    = model_info["seg"]["size_mb"]
    r_coco["model_size_mb"]   = model_info["yolo_coco"]["size_mb"]
    r_aerial["model_size_mb"] = model_info["yolo_aerial"]["size_mb"]

    t3 = fmt_table(
        "Table 3 – Single-Model Runtime Benchmarks (Windows CPU)",
        ["Model", "Avg (ms)", "Min (ms)", "Max (ms)", "p95 (ms)", "FPS", "RAM (MB)", "Size (MB)", "CPU%"],
        [
            ["Seg (Mask2Former)",
             r_seg.get("avg_latency_ms", "N/A"), r_seg.get("min_latency_ms", "N/A"),
             r_seg.get("max_latency_ms", "N/A"), r_seg.get("p95_latency_ms", "N/A"),
             r_seg.get("fps", "N/A"), r_seg.get("peak_ram_mb", "N/A"),
             r_seg.get("model_size_mb", "N/A"), r_seg.get("avg_cpu_pct", "N/A")],
            ["YOLOv8n",
             r_coco.get("avg_latency_ms", "N/A"), r_coco.get("min_latency_ms", "N/A"),
             r_coco.get("max_latency_ms", "N/A"), r_coco.get("p95_latency_ms", "N/A"),
             r_coco.get("fps", "N/A"), r_coco.get("peak_ram_mb", "N/A"),
             r_coco.get("model_size_mb", "N/A"), r_coco.get("avg_cpu_pct", "N/A")],
            ["YOLO7-ITCVD",
             r_aerial.get("avg_latency_ms", "N/A"), r_aerial.get("min_latency_ms", "N/A"),
             r_aerial.get("max_latency_ms", "N/A"), r_aerial.get("p95_latency_ms", "N/A"),
             r_aerial.get("fps", "N/A"), r_aerial.get("peak_ram_mb", "N/A"),
             r_aerial.get("model_size_mb", "N/A"), r_aerial.get("avg_cpu_pct", "N/A")],
        ]
    )
    log(t3)

    log("\n[4/4] Multi-model scenarios (Table 4) ...")

    scenarios = [
        ("Seg only",
         [(sess_seg, inp_seg, dummy_seg)]),
        ("YOLOv8n only",
         [(sess_coco, inp_coco, dummy_yolo)]),
        ("YOLO7-ITCVD only",
         [(sess_aerial, inp_aerial, dummy_yolo)]),
        ("Seg + YOLOv8n",
         [(sess_seg, inp_seg, dummy_seg),
          (sess_coco, inp_coco, dummy_yolo)]),
        ("Seg + YOLOv8n + YOLO7-ITCVD",
         [(sess_seg, inp_seg, dummy_seg),
          (sess_coco, inp_coco, dummy_yolo),
          (sess_aerial, inp_aerial, dummy_yolo)]),
    ]

    def sum_mb(*keys):
        vals = [model_info[k]["size_mb"] for k in keys]
        if all(isinstance(v, (float, int)) for v in vals):
            return round(sum(float(v) for v in vals), 2)
        return "N/A"

    disk = {
        "Seg only":                     model_info["seg"]["size_mb"],
        "YOLOv8n only":                 model_info["yolo_coco"]["size_mb"],
        "YOLO7-ITCVD only":             model_info["yolo_aerial"]["size_mb"],
        "Seg + YOLOv8n":                sum_mb("seg", "yolo_coco"),
        "Seg + YOLOv8n + YOLO7-ITCVD": sum_mb("seg", "yolo_coco", "yolo_aerial"),
    }

    multi_results = {}
    t4_rows = []

    for sc_name, sc_models in scenarios:
        res = bench_scenario(sc_name, sc_models)
        multi_results[sc_name] = res
        t4_rows.append([
            sc_name,
            res.get("avg_latency_ms", "N/A"),
            res.get("ram_mb", "N/A"),
            disk[sc_name],
            res.get("real_time", "N/A"),
            res.get("fps", "N/A"),
        ])

    all_results["multi_model"] = multi_results

    t4 = fmt_table(
        "Table 4 – Multi-Model Deployment Scenarios (Windows CPU, single-thread)",
        ["Scenario", "Latency (ms)", "RAM (MB)", "Disk (MB)", "RT (>=10FPS)?", "FPS"],
        t4_rows
    )
    log(t4)
    log("\n  NOTE: RT = real-time defined as >=10 FPS (<=100 ms/frame).")

    log("\n" + "=" * 60)
    log("  VALUES FOR THE PAPER (copy-paste into LaTeX)")
    log("=" * 60)

    log("\n-- Table 3 -------------------------------------------------")
    for label, r in [("Seg (Mask2Former)", r_seg),
                     ("YOLOv8n",           r_coco),
                     ("YOLO7-ITCVD",       r_aerial)]:
        if "error" not in r:
            log(f"  {label}:")
            log(f"    Avg latency : {r.get('avg_latency_ms', '?')} ms")
            log(f"    Min / Max   : {r.get('min_latency_ms', '?')} / {r.get('max_latency_ms', '?')} ms")
            log(f"    FPS         : {r.get('fps', '?')}")
            log(f"    Peak RAM    : {r.get('peak_ram_mb', '?')} MB")
            log(f"    Model size  : {r.get('model_size_mb', '?')} MB")
            log(f"    CPU util    : {r.get('avg_cpu_pct', '?')} %")

    log("\n-- Table 4 -------------------------------------------------")
    for sc_name, res in multi_results.items():
        if "error" not in res:
            log(f"  {sc_name}:")
            log(f"    Latency : {res['avg_latency_ms']} ms | FPS: {res['fps']} | RT: {res['real_time']}")
            log(f"    RAM     : {res['ram_mb']} MB")

    log("\n-- Table 2 (params & size) ---------------------------------")
    for key, info in model_info.items():
        log(f"  {key:15s}: {info['size_mb']} MB | {info['params_M']} M params")

    log(f"\n  JSON saved to : {OUTPUT_JSON}")
    log(f"  Text saved to : {OUTPUT_TXT}")
    log()
    log("  NOTE: These values are from the Windows desktop reference run.")
    log("  For final paper tables, run benchmark_imx8_paper.py")
    log("  directly on the NXP i.MX 8M Plus EVK (Cortex-A53).")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\n  Output files: {OUTPUT_JSON}  |  {OUTPUT_TXT}")


if __name__ == "__main__":
    main()