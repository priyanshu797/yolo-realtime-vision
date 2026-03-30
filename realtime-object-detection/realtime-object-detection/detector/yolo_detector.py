"""
detector/yolo_detector.py

Handles YOLOv8 model loading and running inference on frames.
Uses the Ultralytics YOLOv8 library under the hood.
"""

import numpy as np
from ultralytics import YOLO


class Detection:
    """
    Lightweight data class representing a single detected object.

    Attributes:
        bbox    (tuple): Bounding box as (x1, y1, x2, y2) in pixel coords.
        label   (str):   Class name (e.g. 'person', 'car').
        conf    (float): Confidence score in [0, 1].
        class_id(int):   Numeric class index.
    """

    def __init__(
        self,
        bbox: tuple[int, int, int, int],
        label: str,
        conf: float,
        class_id: int
    ):
        self.bbox     = bbox       # (x1, y1, x2, y2)
        self.label    = label
        self.conf     = conf
        self.class_id = class_id

    def __repr__(self) -> str:
        x1, y1, x2, y2 = self.bbox
        return (
            f"Detection(label='{self.label}', conf={self.conf:.2f}, "
            f"bbox=({x1},{y1},{x2},{y2}))"
        )


class YOLODetector:
    """
    Wraps a YOLOv8 model for straightforward object detection.

    Usage:
        detector = YOLODetector("yolov8n.pt", conf_threshold=0.5)
        detections = detector.detect(frame)   # frame is a BGR numpy array

    Args:
        model_path     (str):   Path to a .pt weights file, or a model name
                                that Ultralytics will auto-download
                                (e.g. 'yolov8n.pt', 'yolov8s.pt').
        conf_threshold (float): Minimum confidence to keep a detection.
        device         (str):   Inference device — 'cpu', 'cuda', or 'mps'.
                                Defaults to 'cpu' for broadest compatibility.
    """
    AVAILABLE_MODELS = [
        "yolov8n.pt", 
        "yolov8s.pt",   
        "yolov8m.pt",   
        "yolov8l.pt",   
        "yolov8x.pt",   
    ]

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        conf_threshold: float = 0.5,
        device: str = "cpu"
    ):
        self.model_path     = model_path
        self.conf_threshold = conf_threshold
        self.device         = device

        print(f"[YOLODetector] Loading '{model_path}' on device='{device}' ...")
        self._model = self._load_model(model_path)
        print(f"[YOLODetector] Model ready. Classes: {self._model.names}")

    def _load_model(self, model_path: str) -> YOLO:
        """Load the YOLO model; raises RuntimeError on failure."""
        try:
            model = YOLO(model_path)
            return model
        except FileNotFoundError:
            raise RuntimeError(
                f"Model file not found: '{model_path}'. "
                "Provide a valid path or a model name like 'yolov8n.pt' "
                "so Ultralytics can download it automatically."
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load model '{model_path}': {exc}") from exc

    def _parse_results(self, results) -> list[Detection]:
        """
        Convert raw Ultralytics result objects into a list of Detection instances.

        Args:
            results: The return value of model(frame, ...).

        Returns:
            List of Detection objects.
        """
        detections: list[Detection] = []

        for result in results:
            boxes = result.boxes  # Ultralytics Boxes object

            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # Bounding box in xyxy format, rounded to integers
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2))

                # Confidence score
                conf = float(box.conf[0])

                # Class index and name
                class_id = int(box.cls[0])
                label    = result.names[class_id]

                detections.append(Detection(bbox, label, conf, class_id))

        return detections


    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single BGR frame.

        Args:
            frame: A BGR image as a NumPy array (e.g. from cv2.VideoCapture).

        Returns:
            A list of Detection objects (may be empty if nothing is found).

        Raises:
            ValueError: If `frame` is None or not a NumPy array.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a non-None NumPy array (BGR image).")

        results = self._model(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False       # suppress per-frame console output
        )

        return self._parse_results(results)

    @property
    def class_names(self) -> dict[int, str]:
        """Return the model's class-index → class-name mapping."""
        return self._model.names
