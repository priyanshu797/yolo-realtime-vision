"""
Real-Time Object Detection using YOLOv8 + OpenCV
Entry point for the application.
"""

import argparse
import sys
import cv2
from detector.yolo_detector import YOLODetector
from utils.draw_utils import draw_detections, draw_fps, draw_header


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Object Detection using YOLOv8",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        help=(
            "Input source:\n"
            "  'webcam'       → Live webcam feed (default)\n"
            "  'video'        → Path to a video file (e.g. sample.mp4)\n"
            "  'image'        → Path to an image file (e.g. photo.jpg)"
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLOv8 model to use (default: yolov8n.pt — nano, fastest)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the output video to 'output.avi'"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Webcam device index (default: 0)"
    )
    return parser.parse_args()


def run_on_image(detector: YOLODetector, image_path: str):
    """Run detection on a single image and display the result."""
    print(f"[INFO] Loading image: {image_path}")
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"[ERROR] Could not load image at: {image_path}")
        sys.exit(1)

    detections = detector.detect(frame)
    output = draw_detections(frame, detections)
    draw_header(output, f"YOLOv8 — {len(detections)} object(s) detected")

    cv2.imshow("Object Detection — Press any key to exit", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("[INFO] Done. Press any key in the window to close.")


def run_on_video_or_webcam(
    detector: YOLODetector,
    source,
    save_output: bool = False
):
    """
    Run detection on a video file or live webcam feed.

    Args:
        detector:    YOLODetector instance
        source:      int (webcam index) or str (video path)
        save_output: whether to save annotated frames to output.avi
    """
    print(f"[INFO] Opening source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    # --- VideoWriter setup (optional) ---
    writer = None
    if save_output:
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter("output.avi", fourcc, fps_src, (width, height))
        print("[INFO] Saving output to output.avi")

    # --- FPS tracking ---
    import time
    prev_time = time.time()

    print("[INFO] Detection running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream or frame read error. Exiting.")
            break

        # Run YOLO detection
        detections = detector.detect(frame)

        # Draw bounding boxes / labels
        output = draw_detections(frame, detections)

        # Compute and draw FPS
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        draw_fps(output, fps)

        # Draw header bar
        draw_header(output, f"YOLOv8 Detection  |  {len(detections)} object(s)")

        # Show frame
        cv2.imshow("Real-Time Object Detection — Press 'q' to quit", output)

        # Save frame if requested
        if writer is not None:
            writer.write(output)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] 'q' pressed — exiting.")
            break

    cap.release()
    if writer is not None:
        writer.release()
        print("[INFO] Output saved to output.avi")
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    # Load the YOLO model
    print(f"[INFO] Loading model: {args.model}")
    try:
        detector = YOLODetector(model_path=args.model, conf_threshold=args.conf)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    print(f"[INFO] Model loaded. Confidence threshold: {args.conf}")

    # Route to appropriate handler
    source_lower = args.source.lower()

    if source_lower == "webcam":
        run_on_video_or_webcam(detector, args.device, save_output=args.save)

    elif source_lower == "image":
        print("[ERROR] Please provide the image path, e.g. --source path/to/image.jpg")
        sys.exit(1)

    elif source_lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        run_on_image(detector, args.source)

    elif source_lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        run_on_video_or_webcam(detector, args.source, save_output=args.save)

    else:
        # Try treating it as a webcam index or unknown video source
        try:
            src = int(args.source)
            run_on_video_or_webcam(detector, src, save_output=args.save)
        except ValueError:
            print(f"[ERROR] Unrecognized source: '{args.source}'")
            print("  Use 'webcam', a video path, or an image path.")
            sys.exit(1)


if __name__ == "__main__":
    main()
