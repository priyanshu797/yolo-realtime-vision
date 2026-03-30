"""
utils/draw_utils.py

Utility functions for drawing bounding boxes, labels, FPS counter,
and a header bar onto OpenCV frames.
"""

import cv2
import numpy as np
import colorsys


def _generate_color_palette(n: int = 80) -> list[tuple[int, int, int]]:
    """
    Generate `n` visually distinct BGR colours using the HSV colour wheel.

    Args:
        n: Number of colours to generate (80 covers all COCO classes).

    Returns:
        List of (B, G, R) tuples.
    """
    palette: list[tuple[int, int, int]] = []
    for i in range(n):
        hue        = i / n
        saturation = 0.85
        value      = 0.95
        r, g, b    = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert 0-1 floats to 0-255 ints, in BGR order for OpenCV
        palette.append((int(b * 255), int(g * 255), int(r * 255)))
    return palette


# Pre-build once at module level
_COLOR_PALETTE = _generate_color_palette(80)


def get_class_color(class_id: int) -> tuple[int, int, int]:
    """Return a consistent BGR colour for a given class index."""
    return _COLOR_PALETTE[class_id % len(_COLOR_PALETTE)]


def draw_bounding_box(
    frame:    np.ndarray,
    bbox:     tuple[int, int, int, int],
    color:    tuple[int, int, int],
    thickness: int = 2
) -> None:
    """
    Draw a rectangle around a detected object.

    Args:
        frame:     BGR image to draw on (modified in-place).
        bbox:      (x1, y1, x2, y2) pixel coordinates.
        color:     BGR colour tuple.
        thickness: Line thickness in pixels.
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_label(
    frame:   np.ndarray,
    text:    str,
    bbox:    tuple[int, int, int, int],
    color:   tuple[int, int, int],
    font_scale: float = 0.55,
    thickness:  int   = 1
) -> None:
    """
    Draw a filled label pill above the bounding box.

    The pill background uses the same colour as the box, keeping things tidy.

    Args:
        frame:      BGR image to draw on (modified in-place).
        text:       Label string, e.g. "person 0.93".
        bbox:       (x1, y1, x2, y2) — only x1/y1 is used for positioning.
        color:      BGR background colour for the pill.
        font_scale: cv2 font scale.
        thickness:  Text stroke thickness.
    """
    x1, y1, _, _ = bbox
    font          = cv2.FONT_HERSHEY_SIMPLEX

    # Measure text so we can size the pill correctly
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Padding around the text
    pad = 4

    # Keep pill inside the frame
    pill_y1 = max(y1 - text_h - 2 * pad, 0)
    pill_y2 = pill_y1 + text_h + 2 * pad

    # Clamp right edge
    pill_x2 = min(x1 + text_w + 2 * pad, frame.shape[1])

    # Filled background rectangle
    cv2.rectangle(frame, (x1, pill_y1), (pill_x2, pill_y2), color, cv2.FILLED)

    # White text on top
    text_x = x1 + pad
    text_y = pill_y2 - pad - baseline
    cv2.putText(
        frame, text,
        (text_x, text_y),
        font, font_scale,
        (255, 255, 255),    # white
        thickness,
        cv2.LINE_AA
    )


def draw_detections(frame: np.ndarray, detections) -> np.ndarray:
    """
    Draw all detections on a copy of the frame.

    For each Detection object, this draws:
      - A coloured bounding box
      - A label pill showing  "<class>  <conf%>"

    Args:
        frame:      Original BGR frame (NOT modified).
        detections: List of Detection objects from YOLODetector.detect().

    Returns:
        A new BGR frame with all annotations overlaid.
    """
    output = frame.copy()

    for det in detections:
        color = get_class_color(det.class_id)
        label = f"{det.label}  {det.conf * 100:.1f}%"

        draw_bounding_box(output, det.bbox, color, thickness=2)
        draw_label(output, label, det.bbox, color)

    return output


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """
    Overlay an FPS counter in the top-right corner (in-place).

    Args:
        frame: BGR frame to annotate.
        fps:   Current frames-per-second value.
    """
    text       = f"FPS: {fps:.1f}"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 2
    margin     = 10

    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Position: top-right
    x = frame.shape[1] - w - margin
    y = h + margin

    # Dark shadow for readability on any background
    cv2.putText(frame, text, (x + 1, y + 1), font, font_scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Bright green text
    cv2.putText(frame, text, (x, y), font, font_scale,
                (0, 230, 0), thickness, cv2.LINE_AA)


def draw_header(frame: np.ndarray, title: str) -> None:
    """
    Draw a slim dark banner at the very top of the frame with a title string.

    Args:
        frame: BGR frame to annotate (in-place).
        title: Text to display in the banner.
    """
    banner_h   = 30
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1

    # Semi-transparent dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], banner_h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Centred text
    (tw, th), _ = cv2.getTextSize(title, font, font_scale, thickness)
    tx = (frame.shape[1] - tw) // 2
    ty = (banner_h + th) // 2

    cv2.putText(frame, title, (tx, ty), font, font_scale,
                (200, 200, 200), thickness, cv2.LINE_AA)
