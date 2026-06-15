import argparse
import csv
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class GazeResult:
    direction: str
    pupil_center: tuple[int, int] | None
    ratio_x: float | None
    ratio_y: float | None
    confidence: float


@dataclass
class FrameResult:
    frame: np.ndarray
    raw_direction: str
    stable_direction: str
    ratio_x: float | None
    ratio_y: float | None
    eyes_found: int


class GazeEstimator:
    def __init__(self, history_size: int = 9) -> None:
        self.center_x = 0.5
        self.center_y = 0.5
        self.history: deque[str] = deque(maxlen=history_size)

    def calibrate(self, ratio_x: float | None, ratio_y: float | None) -> bool:
        if ratio_x is None or ratio_y is None:
            return False

        self.center_x = ratio_x
        self.center_y = ratio_y
        self.history.clear()
        return True

    def classify(self, ratio_x: float | None, ratio_y: float | None) -> str:
        if ratio_x is None or ratio_y is None:
            return "unknown"

        dx = ratio_x - self.center_x
        dy = ratio_y - self.center_y

        if dx < -0.14:
            horizontal = "left"
        elif dx > 0.14:
            horizontal = "right"
        else:
            horizontal = "center"

        if dy < -0.13:
            vertical = "up"
        elif dy > 0.16:
            vertical = "down"
        else:
            vertical = "center"

        if horizontal == "center" and vertical == "center":
            return "center"
        if vertical == "center":
            return horizontal
        if horizontal == "center":
            return vertical
        return f"{vertical}-{horizontal}"

    def smooth(self, direction: str) -> str:
        if direction != "unknown":
            self.history.append(direction)

        if not self.history:
            return "unknown"

        return Counter(self.history).most_common(1)[0][0]


class CsvLogger:
    def __init__(self, path: Path | None) -> None:
        self.file = None
        self.writer = None

        if path is None:
            return

        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ["timestamp", "raw_direction", "stable_direction", "ratio_x", "ratio_y", "eyes_found"]
        )

    def write(self, result: FrameResult) -> None:
        if self.writer is None:
            return

        self.writer.writerow(
            [
                f"{time.time():.3f}",
                result.raw_direction,
                result.stable_direction,
                "" if result.ratio_x is None else f"{result.ratio_x:.4f}",
                "" if result.ratio_y is None else f"{result.ratio_y:.4f}",
                result.eyes_found,
            ]
        )

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


def load_cascade(name: str) -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + name
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Cannot load cascade: {path}")
    return cascade


def detect_pupil(eye_gray: np.ndarray, threshold_offset: int = 0) -> GazeResult:
    blurred = cv2.GaussianBlur(eye_gray, (7, 7), 0)
    otsu_value, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_value = int(np.clip(otsu_value + threshold_offset, 5, 250))
    _, threshold = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return GazeResult("unknown", None, None, None, 0.0)

    height, width = eye_gray.shape[:2]
    eye_area = width * height
    min_area = max(8, int(eye_area * 0.01))
    max_area = int(eye_area * 0.45)
    candidates = [
        cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area
    ]

    if not candidates:
        return GazeResult("unknown", None, None, None, 0.0)

    pupil = max(candidates, key=cv2.contourArea)
    moments = cv2.moments(pupil)
    if moments["m00"] == 0:
        return GazeResult("unknown", None, None, None, 0.0)

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    ratio_x = center_x / max(1, width)
    ratio_y = center_y / max(1, height)
    confidence = min(1.0, cv2.contourArea(pupil) / max(1, eye_area * 0.12))

    return GazeResult("detected", (center_x, center_y), ratio_x, ratio_y, confidence)


def draw_eye_result(frame: np.ndarray, eye_box: tuple[int, int, int, int], result: GazeResult) -> None:
    x, y, width, height = eye_box
    color = (80, 220, 120) if result.pupil_center else (90, 90, 255)
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

    if result.pupil_center is None:
        return

    pupil_x = x + result.pupil_center[0]
    pupil_y = y + result.pupil_center[1]
    cv2.circle(frame, (pupil_x, pupil_y), 4, (0, 0, 255), -1)


def average_gaze(results: list[GazeResult]) -> tuple[float | None, float | None]:
    known = [item for item in results if item.ratio_x is not None and item.ratio_y is not None]
    if not known:
        return None, None

    weights = np.array([max(0.1, item.confidence) for item in known], dtype=np.float32)
    xs = np.array([item.ratio_x for item in known], dtype=np.float32)
    ys = np.array([item.ratio_y for item in known], dtype=np.float32)
    return float(np.average(xs, weights=weights)), float(np.average(ys, weights=weights))


def draw_hud(
    frame: np.ndarray,
    result: FrameResult,
    estimator: GazeEstimator,
    threshold_offset: int,
    log_enabled: bool,
) -> None:
    lines = [
        f"Gaze: {result.stable_direction}  raw: {result.raw_direction}",
        f"Eyes: {result.eyes_found}  Center: {estimator.center_x:.2f}, {estimator.center_y:.2f}",
        f"Threshold offset: {threshold_offset}  Log: {'on' if log_enabled else 'off'}",
        "C - calibrate center | +/- threshold | Q - exit",
    ]

    y = 34
    for line in lines:
        cv2.putText(
            frame,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (40, 240, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28


def process_frame(
    frame: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    eye_cascade: cv2.CascadeClassifier,
    estimator: GazeEstimator,
    threshold_offset: int,
    log_enabled: bool,
) -> FrameResult:
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )

    gaze_results: list[GazeResult] = []

    for face_x, face_y, face_w, face_h in faces[:1]:
        cv2.rectangle(
            frame,
            (face_x, face_y),
            (face_x + face_w, face_y + face_h),
            (255, 180, 70),
            2,
        )

        upper_face_gray = gray[face_y : face_y + face_h // 2, face_x : face_x + face_w]
        eyes = eye_cascade.detectMultiScale(
            upper_face_gray,
            scaleFactor=1.15,
            minNeighbors=8,
            minSize=(22, 16),
        )

        eyes = sorted(eyes, key=lambda box: box[2] * box[3], reverse=True)[:2]

        for eye_x, eye_y, eye_w, eye_h in eyes:
            absolute_box = (face_x + eye_x, face_y + eye_y, eye_w, eye_h)
            eye_gray = upper_face_gray[eye_y : eye_y + eye_h, eye_x : eye_x + eye_w]
            result = detect_pupil(eye_gray, threshold_offset)
            gaze_results.append(result)
            draw_eye_result(frame, absolute_box, result)

    ratio_x, ratio_y = average_gaze(gaze_results)
    raw_direction = estimator.classify(ratio_x, ratio_y)
    stable_direction = estimator.smooth(raw_direction)

    result = FrameResult(
        frame=frame,
        raw_direction=raw_direction,
        stable_direction=stable_direction,
        ratio_x=ratio_x,
        ratio_y=ratio_y,
        eyes_found=len(gaze_results),
    )
    draw_hud(frame, result, estimator, threshold_offset, log_enabled)
    return result


def run(camera_index: int, log_path: Path | None, threshold_offset: int) -> None:
    face_cascade = load_cascade("haarcascade_frontalface_default.xml")
    eye_cascade = load_cascade("haarcascade_eye.xml")
    estimator = GazeEstimator()
    logger = CsvLogger(log_path)

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(
            f"Cannot open camera #{camera_index}. Try another index with --camera."
        )

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Cannot read frame from camera.")

            result = process_frame(
                frame,
                face_cascade,
                eye_cascade,
                estimator,
                threshold_offset,
                log_path is not None,
            )
            logger.write(result)
            cv2.imshow("Simple Eye Tracking", result.frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                estimator.calibrate(result.ratio_x, result.ratio_y)
            if key in (ord("+"), ord("=")):
                threshold_offset = min(60, threshold_offset + 2)
            if key in (ord("-"), ord("_")):
                threshold_offset = max(-60, threshold_offset - 2)
    finally:
        logger.close()
        camera.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple webcam eye tracking demo.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index, usually 0 for the built-in webcam.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional CSV file for saving gaze measurements.",
    )
    parser.add_argument(
        "--threshold-offset",
        type=int,
        default=0,
        help="Manual pupil threshold correction from -60 to 60.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        camera_index=args.camera,
        log_path=args.log,
        threshold_offset=int(np.clip(args.threshold_offset, -60, 60)),
    )
