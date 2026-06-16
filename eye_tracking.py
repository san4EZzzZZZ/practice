import argparse
import ctypes
import ctypes.wintypes
import csv
import os
import time
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MATPLOTLIB_CACHE_DIR = Path(".cache/matplotlib").resolve()
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import mediapipe as mp
except ImportError:
    mp = None


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
    confidence: float
    eyes_found: int


LEFT_EYE_CONTOUR = [
    33, 246, 161, 160, 159, 158, 157, 173,
    133, 155, 154, 153, 145, 144, 163, 7,
]
RIGHT_EYE_CONTOUR = [
    362, 398, 384, 385, 386, 387, 388, 466,
    263, 249, 390, 373, 374, 380, 381, 382,
]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = Path(".cache/mediapipe/face_landmarker.task")

COLOR_PANEL = (18, 22, 30)
COLOR_PANEL_2 = (29, 34, 45)
COLOR_TEXT = (235, 242, 248)
COLOR_MUTED = (160, 172, 184)
COLOR_ACCENT = (51, 217, 178)
COLOR_WARN = (80, 170, 255)
COLOR_DANGER = (95, 95, 255)
COLOR_BLUE = (255, 180, 90)
VIGNETTE_CACHE: dict[tuple[int, int], np.ndarray] = {}


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


class CursorController:
    def __init__(self, active_by_default: bool, speed: int) -> None:
        self.available = hasattr(ctypes, "windll")
        self.active = active_by_default and self.available
        self.speed = speed
        self.user32 = ctypes.windll.user32 if self.available else None

    def toggle(self) -> None:
        if self.available:
            self.active = not self.active

    def status(self) -> str:
        if not self.available:
            return "unavailable"
        return "on" if self.active else "off"

    def move(self, direction: str) -> None:
        if not self.available or not self.active or direction in ("center", "unknown"):
            return

        dx = 0
        dy = 0
        step = self.speed

        if "left" in direction:
            dx = -step
        if "right" in direction:
            dx = step
        if "up" in direction:
            dy = -step
        if "down" in direction:
            dy = step

        point = ctypes.wintypes.POINT()
        self.user32.GetCursorPos(ctypes.byref(point))
        screen_w = self.user32.GetSystemMetrics(0)
        screen_h = self.user32.GetSystemMetrics(1)
        new_x = int(np.clip(point.x + dx, 0, screen_w - 1))
        new_y = int(np.clip(point.y + dy, 0, screen_h - 1))
        self.user32.SetCursorPos(new_x, new_y)


def landmark_to_point(landmark, frame_width: int, frame_height: int) -> tuple[int, int]:
    return int(landmark.x * frame_width), int(landmark.y * frame_height)


def landmark_points(
    landmarks,
    indexes: list[int],
    frame_width: int,
    frame_height: int,
) -> list[tuple[int, int]]:
    return [
        landmark_to_point(landmarks[index], frame_width, frame_height)
        for index in indexes
        if index < len(landmarks)
    ]


def ensure_face_landmarker_model() -> Path:
    if FACE_LANDMARKER_MODEL_PATH.exists():
        return FACE_LANDMARKER_MODEL_PATH

    FACE_LANDMARKER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(
            FACE_LANDMARKER_MODEL_URL,
            FACE_LANDMARKER_MODEL_PATH,
        )
    except OSError as exc:
        raise RuntimeError(
            "Cannot download MediaPipe Face Landmarker model. "
            f"Download it manually from {FACE_LANDMARKER_MODEL_URL} and save it to "
            f"{FACE_LANDMARKER_MODEL_PATH}."
        ) from exc

    return FACE_LANDMARKER_MODEL_PATH


def create_face_landmarker():
    if mp is None:
        raise RuntimeError(
            "MediaPipe is not installed. Use Python 3.10-3.12, then run "
            "`pip install -r requirements.txt`."
        )

    model_path = ensure_face_landmarker_model()
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.55,
        min_face_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def estimate_eye_from_landmarks(
    landmarks,
    eye_indexes: list[int],
    iris_indexes: list[int],
    frame_width: int,
    frame_height: int,
) -> tuple[GazeResult, list[tuple[int, int]]]:
    eye_points = landmark_points(landmarks, eye_indexes, frame_width, frame_height)
    iris_points = landmark_points(landmarks, iris_indexes, frame_width, frame_height)

    if len(eye_points) < 4 or not iris_points:
        return GazeResult("unknown", None, None, None, 0.0), eye_points

    eye_array = np.array(eye_points, dtype=np.int32)
    iris_array = np.array(iris_points, dtype=np.float32)
    min_x = int(np.min(eye_array[:, 0]))
    max_x = int(np.max(eye_array[:, 0]))
    min_y = int(np.min(eye_array[:, 1]))
    max_y = int(np.max(eye_array[:, 1]))

    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)
    center_x = float(np.mean(iris_array[:, 0]))
    center_y = float(np.mean(iris_array[:, 1]))
    ratio_x = float(np.clip((center_x - min_x) / width, 0.0, 1.0))
    ratio_y = float(np.clip((center_y - min_y) / height, 0.0, 1.0))

    iris_center = np.mean(iris_array, axis=0)
    iris_spread = float(np.mean(np.linalg.norm(iris_array - iris_center, axis=1)))
    confidence = float(
        np.clip(iris_spread / max(1.0, min(width, height) * 0.18), 0.35, 1.0)
    )

    return (
        GazeResult(
            "detected",
            (int(center_x), int(center_y)),
            ratio_x,
            ratio_y,
            confidence,
        ),
        eye_points,
    )


def draw_landmark_eye(
    frame: np.ndarray,
    eye_points: list[tuple[int, int]],
    result: GazeResult,
) -> None:
    if len(eye_points) >= 3:
        cv2.polylines(
            frame,
            [np.array(eye_points, dtype=np.int32)],
            isClosed=True,
            color=(80, 220, 120),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    if result.pupil_center is not None:
        cv2.circle(frame, result.pupil_center, 4, (0, 0, 255), -1)
        cv2.circle(frame, result.pupil_center, 8, (40, 240, 255), 1, cv2.LINE_AA)


def average_gaze(results: list[GazeResult]) -> tuple[float | None, float | None, float]:
    known = [item for item in results if item.ratio_x is not None and item.ratio_y is not None]
    if not known:
        return None, None, 0.0

    weights = np.array([max(0.1, item.confidence) for item in known], dtype=np.float32)
    xs = np.array([item.ratio_x for item in known], dtype=np.float32)
    ys = np.array([item.ratio_y for item in known], dtype=np.float32)
    confidence = float(np.clip(np.mean([item.confidence for item in known]), 0.0, 1.0))
    return (
        float(np.average(xs, weights=weights)),
        float(np.average(ys, weights=weights)),
        confidence,
    )


def blend_rect(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def put_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = COLOR_TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def apply_vignette(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    cache_key = (width, height)
    mask = VIGNETTE_CACHE.get(cache_key)
    if mask is None:
        x = np.linspace(-1.0, 1.0, width)
        y = np.linspace(-1.0, 1.0, height)
        xv, yv = np.meshgrid(x, y)
        mask = 1.0 - np.clip((xv * xv + yv * yv - 0.28) * 0.42, 0.0, 0.35)
        VIGNETTE_CACHE[cache_key] = mask
    frame[:] = (frame.astype(np.float32) * mask[:, :, None]).astype(np.uint8)


def draw_status_chip(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    active: bool,
    accent: tuple[int, int, int] = COLOR_ACCENT,
) -> int:
    x, y = origin
    width = max(92, 14 + len(text) * 10)
    color = accent if active else COLOR_PANEL_2
    blend_rect(frame, (x, y), (x + width, y + 28), color, 0.72 if active else 0.82)
    cv2.rectangle(frame, (x, y), (x + width, y + 28), color, 1, cv2.LINE_AA)
    put_text(frame, text, (x + 11, y + 19), 0.46, COLOR_TEXT)
    return width


def draw_progress_bar(
    frame: np.ndarray,
    label: str,
    value: float,
    origin: tuple[int, int],
    width: int,
    accent: tuple[int, int, int],
) -> None:
    x, y = origin
    value = float(np.clip(value, 0.0, 1.0))
    put_text(frame, label, (x, y), 0.44, COLOR_MUTED)
    cv2.rectangle(frame, (x, y + 10), (x + width, y + 20), (64, 70, 82), -1)
    cv2.rectangle(frame, (x, y + 10), (x + int(width * value), y + 20), accent, -1)
    cv2.rectangle(frame, (x, y + 10), (x + width, y + 20), (92, 100, 112), 1, cv2.LINE_AA)


def draw_gaze_pad(
    frame: np.ndarray,
    result: FrameResult,
    estimator: GazeEstimator,
    origin: tuple[int, int],
    size: int,
) -> None:
    x, y = origin
    blend_rect(frame, (x, y), (x + size, y + size), COLOR_PANEL_2, 0.82)
    cv2.rectangle(frame, (x, y), (x + size, y + size), (80, 90, 104), 1, cv2.LINE_AA)
    cv2.line(frame, (x + size // 2, y + 12), (x + size // 2, y + size - 12), (70, 78, 90), 1)
    cv2.line(frame, (x + 12, y + size // 2), (x + size - 12, y + size // 2), (70, 78, 90), 1)

    center_x = x + int(estimator.center_x * size)
    center_y = y + int(estimator.center_y * size)
    cv2.circle(frame, (center_x, center_y), 5, COLOR_BLUE, 1, cv2.LINE_AA)

    if result.ratio_x is None or result.ratio_y is None:
        put_text(frame, "no face", (x + 18, y + size // 2 + 5), 0.5, COLOR_MUTED)
        return

    gaze_x = x + int(result.ratio_x * size)
    gaze_y = y + int(result.ratio_y * size)
    cv2.circle(frame, (gaze_x, gaze_y), 10, COLOR_ACCENT, -1, cv2.LINE_AA)
    cv2.circle(frame, (gaze_x, gaze_y), 16, COLOR_ACCENT, 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    result: FrameResult,
    estimator: GazeEstimator,
    log_enabled: bool,
    cursor_status: str,
) -> None:
    height, width = frame.shape[:2]
    margin = 16
    top_h = 76
    bottom_h = 48

    apply_vignette(frame)
    blend_rect(frame, (0, 0), (width, top_h), COLOR_PANEL, 0.78)
    blend_rect(frame, (0, height - bottom_h), (width, height), COLOR_PANEL, 0.78)

    put_text(frame, "Gaze Studio", (margin, 29), 0.72, COLOR_TEXT, 2)
    put_text(frame, "MediaPipe Face Landmarker", (margin, 56), 0.46, COLOR_MUTED)

    chips = [
        (f"cursor {cursor_status}", cursor_status == "on", COLOR_ACCENT),
        ("log on" if log_enabled else "log off", log_enabled, COLOR_BLUE),
        (f"eyes {result.eyes_found}", result.eyes_found > 0, COLOR_WARN),
    ]
    chip_widths = [max(92, 14 + len(text) * 10) for text, _, _ in chips]
    chip_x = max(margin, width - margin - sum(chip_widths) - 8 * (len(chips) - 1))
    for index, (text, active, accent) in enumerate(chips):
        chip_x += draw_status_chip(frame, text, (chip_x, 22), active, accent)
        if index < len(chips) - 1:
            chip_x += 8

    panel_w = min(310, max(240, width // 3))
    panel_h = 206
    panel_x = margin
    panel_y = top_h + 16
    blend_rect(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), COLOR_PANEL, 0.72)
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (68, 78, 92),
        1,
        cv2.LINE_AA,
    )

    direction = result.stable_direction.upper().replace("-", " ")
    direction_color = COLOR_DANGER if result.stable_direction == "unknown" else COLOR_ACCENT
    put_text(frame, "CURRENT GAZE", (panel_x + 18, panel_y + 28), 0.43, COLOR_MUTED)
    put_text(frame, direction, (panel_x + 18, panel_y + 64), 0.86, direction_color, 2)
    put_text(frame, f"raw: {result.raw_direction}", (panel_x + 18, panel_y + 91), 0.45, COLOR_MUTED)

    draw_progress_bar(
        frame,
        "confidence",
        result.confidence,
        (panel_x + 18, panel_y + 124),
        panel_w - 36,
        COLOR_ACCENT if result.confidence >= 0.5 else COLOR_DANGER,
    )
    ratio_text = "x --  y --"
    if result.ratio_x is not None and result.ratio_y is not None:
        ratio_text = f"x {result.ratio_x:.2f}  y {result.ratio_y:.2f}"
    put_text(frame, ratio_text, (panel_x + 18, panel_y + 176), 0.5, COLOR_TEXT)

    pad_size = 132
    draw_gaze_pad(frame, result, estimator, (width - pad_size - margin, top_h + 16), pad_size)

    footer = "C  calibrate    M  cursor toggle    Q  quit"
    put_text(frame, footer, (margin, height - 18), 0.52, COLOR_TEXT)


def process_frame(
    frame: np.ndarray,
    face_landmarker,
    estimator: GazeEstimator,
    log_enabled: bool,
    cursor_status: str,
) -> FrameResult:
    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmark_result = face_landmarker.detect(mp_image)

    gaze_results: list[GazeResult] = []

    if landmark_result.face_landmarks:
        landmarks = landmark_result.face_landmarks[0]
        for eye_indexes, iris_indexes in (
            (LEFT_EYE_CONTOUR, LEFT_IRIS),
            (RIGHT_EYE_CONTOUR, RIGHT_IRIS),
        ):
            result, eye_points = estimate_eye_from_landmarks(
                landmarks,
                eye_indexes,
                iris_indexes,
                frame_width,
                frame_height,
            )
            gaze_results.append(result)
            draw_landmark_eye(frame, eye_points, result)

    ratio_x, ratio_y, confidence = average_gaze(gaze_results)
    raw_direction = estimator.classify(ratio_x, ratio_y)
    stable_direction = estimator.smooth(raw_direction)

    result = FrameResult(
        frame=frame,
        raw_direction=raw_direction,
        stable_direction=stable_direction,
        ratio_x=ratio_x,
        ratio_y=ratio_y,
        confidence=confidence,
        eyes_found=len(gaze_results),
    )
    draw_hud(frame, result, estimator, log_enabled, cursor_status)
    return result


def run(
    camera_index: int,
    log_path: Path | None,
    control_cursor: bool,
    cursor_speed: int,
) -> None:
    estimator = GazeEstimator()
    logger = CsvLogger(log_path)
    cursor = CursorController(control_cursor, cursor_speed)
    face_landmarker = create_face_landmarker()

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(
            f"Cannot open camera #{camera_index}. Try another index with --camera."
        )

    window_name = "Gaze Studio"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Cannot read frame from camera.")

            result = process_frame(
                frame,
                face_landmarker,
                estimator,
                log_path is not None,
                cursor.status(),
            )
            logger.write(result)
            cursor.move(result.stable_direction)
            cv2.imshow(window_name, result.frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                estimator.calibrate(result.ratio_x, result.ratio_y)
            if key == ord("m"):
                cursor.toggle()
    finally:
        face_landmarker.close()
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
        "--control-cursor",
        action="store_true",
        help="Start with gaze-based cursor control enabled. Press M in the video window to toggle it.",
    )
    parser.add_argument(
        "--cursor-speed",
        type=int,
        default=18,
        help="Cursor movement step in pixels per frame.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        camera_index=args.camera,
        log_path=args.log,
        control_cursor=args.control_cursor,
        cursor_speed=max(1, args.cursor_speed),
    )
