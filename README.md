# 🎯 Real-Time Object Detection — YOLOv8 + OpenCV

A clean, modular Python project that performs **real-time object detection** using a
pre-trained **YOLOv8** model and an **OpenCV** video pipeline.  
Supports **webcam**, **video files**, and **static images** as input sources.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔍 Object Detection | YOLOv8 — detects 80 COCO classes out of the box |
| 🎨 Bounding Boxes | Colour-coded per class, consistent across frames |
| 🏷️ Labels + Confidence | E.g. `person  94.3%` displayed as a pill above each box |
| ⚡ FPS Counter | Live frames-per-second overlay (top-right corner) |
| 📷 Multiple Sources | Webcam · Video file · Image file |
| 💾 Save Output | Annotated video exported to `output.avi` with `--save` |
| 🧩 Modular Code | Detector, drawing utils, and entry-point kept fully separate |

---

## 📁 Project Structure

```
realtime-object-detection/
│
├── main.py                   ← Entry point; argument parsing + source routing
│
├── detector/
│   ├── __init__.py
│   └── yolo_detector.py      ← YOLODetector class; model loading + inference
│
├── utils/
│   ├── __init__.py
│   └── draw_utils.py         ← draw_detections(), draw_fps(), draw_header()
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup

### 1 — Prerequisites

- Python **3.9 or higher**
- A working webcam (for live mode)
- `pip` (or `pip3`)

> **GPU (optional):** If you have an NVIDIA GPU, install
> [CUDA-enabled PyTorch](https://pytorch.org/get-started/locally/) first,
> then install the rest. The project works fine on CPU — just slower.

---

### 2 — Clone / Download

```bash
git clone https://github.com/your-username/realtime-object-detection.git
cd realtime-object-detection
```

---

### 3 — Create a Virtual Environment (recommended)

```bash
# Create
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

---

### 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **First run:** Ultralytics will automatically download `yolov8n.pt`
> (~3 MB) into `~/.config/Ultralytics/` the first time the model is used.
> An internet connection is required for this one-time download.

---

## 🚀 Usage

### Live Webcam (default)

```bash
python main.py
```

```bash
# Use a specific webcam index (if you have multiple cameras)
python main.py --source webcam --device 1
```

---

### Video File

```bash
python main.py --source path/to/video.mp4
```

---

### Image File

```bash
python main.py --source path/to/photo.jpg
```

---

### Save Annotated Output

```bash
# Works for webcam and video sources
python main.py --source path/to/video.mp4 --save
# Output saved to: output.avi
```

---

### Adjust Confidence Threshold

```bash
# Only show detections with ≥ 70% confidence
python main.py --conf 0.7
```

---

### Choose a Different Model

| Model | Flag | Speed | Accuracy |
|---|---|---|---|
| Nano (default) | `--model yolov8n.pt` | ⚡ Fastest | ★★☆☆☆ |
| Small | `--model yolov8s.pt` | ⚡⚡ Fast | ★★★☆☆ |
| Medium | `--model yolov8m.pt` | ⚡⚡⚡ | ★★★★☆ |
| Large | `--model yolov8l.pt` | 🐢 Slow | ★★★★★ |
| XLarge | `--model yolov8x.pt` | 🐢🐢 Slowest | ★★★★★ |

```bash
python main.py --model yolov8s.pt
```

---

### Full Options

```
usage: main.py [-h] [--source SOURCE] [--model MODEL] [--conf CONF] [--save] [--device DEVICE]

options:
  -h, --help       Show this help message and exit
  --source SOURCE  'webcam', path/to/video.mp4, or path/to/image.jpg
  --model MODEL    YOLOv8 weights file (default: yolov8n.pt)
  --conf CONF      Confidence threshold 0–1 (default: 0.5)
  --save           Save annotated output to output.avi
  --device DEVICE  Webcam index (default: 0)
```

---

## 🕹️ Controls

| Key | Action |
|---|---|
| `q` | Quit the application |
| Any key | Close (image mode only) |

---

## 📦 Detected Object Classes (COCO — 80 classes)

A few examples:

`person · bicycle · car · motorcycle · airplane · bus · train · truck · boat · traffic light ·
fire hydrant · stop sign · parking meter · bench · bird · cat · dog · horse · sheep · cow ·
elephant · bear · zebra · giraffe · backpack · umbrella · handbag · tie · suitcase · frisbee ·
skis · snowboard · sports ball · kite · baseball bat · baseball glove · skateboard · surfboard ·
tennis racket · bottle · wine glass · cup · fork · knife · spoon · bowl · banana · apple ·
sandwich · orange · broccoli · carrot · hot dog · pizza · donut · cake · chair · couch ·
potted plant · bed · dining table · toilet · tv · laptop · mouse · remote · keyboard ·
cell phone · microwave · oven · toaster · sink · refrigerator · book · clock · vase · scissors ·
teddy bear · hair drier · toothbrush`

---

## 🧠 How It Works

```
main.py (parse args)
    │
    ├── YOLODetector.detect(frame)         ← yolo_detector.py
    │       └── ultralytics YOLO(frame)
    │           └── returns [Detection(bbox, label, conf, class_id), ...]
    │
    └── draw_detections(frame, detections) ← draw_utils.py
            ├── draw_bounding_box()
            ├── draw_label()
            ├── draw_fps()
            └── draw_header()
```

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot open source: 0` | Try `--device 1` or check camera permissions |
| `Failed to load model` | Delete cached `.pt` file and re-run to re-download |
| Very low FPS | Use `yolov8n.pt` (nano) and/or lower resolution |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your venv |

---

## 📄 License

MIT — free to use, modify, and distribute.
