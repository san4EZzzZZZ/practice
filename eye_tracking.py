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
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

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

COLOR_PANEL = (28, 29, 31)
COLOR_PANEL_2 = (45, 47, 50)
COLOR_TEXT = (242, 242, 240)
COLOR_MUTED = (178, 178, 174)
COLOR_ACCENT = (84, 204, 170)
COLOR_WARN = (84, 176, 232)
COLOR_DANGER = (92, 112, 235)
COLOR_BLUE = (220, 170, 92)
COLOR_LINE = (92, 94, 98)
VIGNETTE_CACHE: dict[tuple[int, int], np.ndarray] = {}
FONT_CACHE: dict[tuple[str, int], object] = {}

DIRECTION_LABELS = {
    "unknown": "неизвестно",
    "center": "центр",
    "left": "влево",
    "right": "вправо",
    "up": "вверх",
    "down": "вниз",
    "up-left": "вверх-влево",
    "up-right": "вверх-вправо",
    "down-left": "вниз-влево",
    "down-right": "вниз-вправо",
    "detected": "обнаружено",
}

STATUS_LABELS = {
    "unavailable": "недоступно",
    "unmapped": "нет калибровки",
    "on": "вкл",
    "off": "выкл",
    "calibrating": "калибровка",
    "paused": "пауза",
    "need calibration": "нужна калибровка",
    "no gaze": "нет взгляда",
    "low confidence": "низкая уверенность",
    "tracking": "отслеживание",
    "inactive": "неактивна",
}


def direction_label(direction: str) -> str:
    return DIRECTION_LABELS.get(direction, direction).upper()


def status_label(status: str) -> str:
    if "," in status:
        return status
    return STATUS_LABELS.get(status, status)


class RussianArgumentParser(argparse.ArgumentParser):
    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "использование:")

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "использование:")


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


class ScreenMapper:
    def __init__(
        self,
        gaze_axis_x: np.ndarray,
        screen_axis_x: np.ndarray,
        gaze_axis_y: np.ndarray,
        screen_axis_y: np.ndarray,
    ) -> None:
        self.gaze_axis_x = gaze_axis_x
        self.screen_axis_x = screen_axis_x
        self.gaze_axis_y = gaze_axis_y
        self.screen_axis_y = screen_axis_y

    @staticmethod
    def interp_axis(gaze_value: float, gaze_axis: np.ndarray, screen_axis: np.ndarray) -> float:
        if len(gaze_axis) < 2:
            return float(screen_axis[0])

        if gaze_value <= float(gaze_axis[0]):
            slope = (screen_axis[1] - screen_axis[0]) / max(1e-5, gaze_axis[1] - gaze_axis[0])
            return float(screen_axis[0] + (gaze_value - gaze_axis[0]) * slope)

        if gaze_value >= float(gaze_axis[-1]):
            slope = (screen_axis[-1] - screen_axis[-2]) / max(1e-5, gaze_axis[-1] - gaze_axis[-2])
            return float(screen_axis[-1] + (gaze_value - gaze_axis[-1]) * slope)

        return float(np.interp(gaze_value, gaze_axis, screen_axis))

    def map(self, gaze_x: float, gaze_y: float, screen_w: int, screen_h: int) -> tuple[int, int]:
        x = self.interp_axis(gaze_x, self.gaze_axis_x, self.screen_axis_x)
        y = self.interp_axis(gaze_y, self.gaze_axis_y, self.screen_axis_y)
        return (
            int(np.clip(x, 0, screen_w - 1)),
            int(np.clip(y, 0, screen_h - 1)),
        )

    @staticmethod
    def remove_target_outliers(
        samples: list[tuple[float, float, int, int]],
    ) -> list[tuple[float, float, int, int]]:
        cleaned: list[tuple[float, float, int, int]] = []
        grouped: dict[tuple[int, int], list[tuple[float, float, int, int]]] = {}
        for sample in samples:
            grouped.setdefault((sample[2], sample[3]), []).append(sample)

        for target_samples in grouped.values():
            if len(target_samples) < 4:
                cleaned.extend(target_samples)
                continue

            gaze = np.array([(x, y) for x, y, _, _ in target_samples], dtype=np.float32)
            median = np.median(gaze, axis=0)
            distances = np.linalg.norm(gaze - median, axis=1)
            median_distance = float(np.median(distances))
            mad = float(np.median(np.abs(distances - median_distance)))
            limit = median_distance + max(0.035, 2.5 * mad)

            kept = [
                sample
                for sample, distance in zip(target_samples, distances)
                if float(distance) <= limit
            ]
            cleaned.extend(kept or target_samples)

        return cleaned

    @classmethod
    def fit(
        cls,
        samples: list[tuple[float, float, int, int]],
    ) -> "ScreenMapper":
        samples = cls.remove_target_outliers(samples)
        by_screen_x: dict[int, list[tuple[float, float, int, int]]] = {}
        by_screen_y: dict[int, list[tuple[float, float, int, int]]] = {}
        for sample in samples:
            by_screen_x.setdefault(sample[2], []).append(sample)
            by_screen_y.setdefault(sample[3], []).append(sample)

        axis_x = sorted(
            (
                float(np.median([sample[0] for sample in target_samples])),
                float(screen_x),
            )
            for screen_x, target_samples in by_screen_x.items()
        )
        axis_y = sorted(
            (
                float(np.median([sample[1] for sample in target_samples])),
                float(screen_y),
            )
            for screen_y, target_samples in by_screen_y.items()
        )

        gaze_axis_x = np.array([item[0] for item in axis_x], dtype=np.float32)
        screen_axis_x = np.array([item[1] for item in axis_x], dtype=np.float32)
        gaze_axis_y = np.array([item[0] for item in axis_y], dtype=np.float32)
        screen_axis_y = np.array([item[1] for item in axis_y], dtype=np.float32)
        if len(gaze_axis_x) < 3 or len(gaze_axis_y) < 3:
            raise RuntimeError("Калибровка не собрала все области экрана.")
        if float(np.min(np.diff(gaze_axis_x))) < 0.012 or float(np.min(np.diff(gaze_axis_y))) < 0.012:
            raise RuntimeError(
                "Калибровка получилась слишком слабой: взгляд почти не отличался между точками. "
                "Пройдите ее заново, глядя точно в центр каждого маркера."
            )
        return cls(gaze_axis_x, screen_axis_x, gaze_axis_y, screen_axis_y)


class CursorController:
    def __init__(
        self,
        active_by_default: bool,
        smoothing: float,
        min_confidence: float,
    ) -> None:
        self.available = hasattr(ctypes, "windll")
        self.active = active_by_default and self.available
        self.smoothing = smoothing
        self.min_confidence = min_confidence
        self.mapper: ScreenMapper | None = None
        self.user32 = ctypes.windll.user32 if self.available else None
        self.current_x: float | None = None
        self.current_y: float | None = None
        self.gaze_x: float | None = None
        self.gaze_y: float | None = None
        self.target_x: float | None = None
        self.target_y: float | None = None
        self.gaze_history: deque[tuple[float, float]] = deque(maxlen=5)

    def toggle(self) -> None:
        if self.available:
            self.active = not self.active
            self.reset_motion()

    def status(self) -> str:
        if not self.available:
            return "unavailable"
        if self.mapper is None:
            return "unmapped"
        return "on" if self.active else "off"

    def movement_status(self, result: FrameResult, calibration_active: bool) -> str:
        if calibration_active:
            return "calibrating"
        if not self.available:
            return "unavailable"
        if not self.active:
            return "paused"
        if self.mapper is None:
            return "need calibration"
        if result.ratio_x is None or result.ratio_y is None:
            return "no gaze"
        if result.confidence < self.min_confidence:
            return "low confidence"
        if self.current_x is None or self.current_y is None:
            return "tracking"
        return f"{int(self.current_x)}, {int(self.current_y)}"

    def set_mapper(self, mapper: ScreenMapper) -> None:
        self.mapper = mapper
        self.reset_motion()

    def reset_motion(self) -> None:
        self.current_x = None
        self.current_y = None
        self.gaze_x = None
        self.gaze_y = None
        self.target_x = None
        self.target_y = None
        self.gaze_history.clear()

    def move(
        self,
        ratio_x: float | None,
        ratio_y: float | None,
        confidence: float,
    ) -> None:
        if not self.available or not self.active or self.mapper is None:
            return
        if ratio_x is None or ratio_y is None or confidence < self.min_confidence:
            self.gaze_x = None
            self.gaze_y = None
            self.target_x = None
            self.target_y = None
            self.gaze_history.clear()
            return

        self.gaze_history.append((ratio_x, ratio_y))
        stable_gaze = np.array(self.gaze_history, dtype=np.float32)
        gaze_x = float(np.median(stable_gaze[:, 0]))
        gaze_y = float(np.median(stable_gaze[:, 1]))

        if self.gaze_x is None or self.gaze_y is None:
            self.gaze_x = gaze_x
            self.gaze_y = gaze_y
        else:
            gaze_alpha = 0.82
            self.gaze_x = self.gaze_x * gaze_alpha + gaze_x * (1.0 - gaze_alpha)
            self.gaze_y = self.gaze_y * gaze_alpha + gaze_y * (1.0 - gaze_alpha)

        screen_w = self.user32.GetSystemMetrics(0)
        screen_h = self.user32.GetSystemMetrics(1)
        target_x, target_y = self.mapper.map(self.gaze_x, self.gaze_y, screen_w, screen_h)

        if self.target_x is None or self.target_y is None:
            self.target_x = float(target_x)
            self.target_y = float(target_y)
        else:
            target_alpha = 0.88
            self.target_x = self.target_x * target_alpha + float(target_x) * (1.0 - target_alpha)
            self.target_y = self.target_y * target_alpha + float(target_y) * (1.0 - target_alpha)

        alpha = float(np.clip(self.smoothing, 0.0, 0.95))
        if self.current_x is None or self.current_y is None:
            point = ctypes.wintypes.POINT()
            self.user32.GetCursorPos(ctypes.byref(point))
            self.current_x = float(point.x)
            self.current_y = float(point.y)

        distance = float(np.hypot(self.target_x - self.current_x, self.target_y - self.current_y))
        if distance < 8:
            return
        if distance > 400:
            alpha = min(alpha, 0.70)
        elif distance > 200:
            alpha = min(alpha, 0.80)
        elif distance < 70:
            alpha = max(alpha, 0.94)

        self.current_x = self.current_x * alpha + self.target_x * (1.0 - alpha)
        self.current_y = self.current_y * alpha + self.target_y * (1.0 - alpha)
        self.user32.SetCursorPos(int(round(self.current_x)), int(round(self.current_y)))


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


def get_screen_size() -> tuple[int, int]:
    if hasattr(ctypes, "windll"):
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    return 1920, 1080


class CalibrationSession:
    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        samples_per_point: int,
        settle_frames: int = 16,
        sample_stride: int = 3,
    ) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.samples_per_point = samples_per_point
        self.settle_frames = settle_frames
        self.sample_stride = sample_stride
        self.targets = [
            (0.5, 0.5),
            (0.15, 0.15),
            (0.85, 0.15),
            (0.85, 0.85),
            (0.15, 0.85),
            (0.5, 0.15),
            (0.5, 0.85),
            (0.15, 0.5),
            (0.85, 0.5),
        ]
        self.active = False
        self.completed = False
        self.target_index = 0
        self.samples: list[tuple[float, float, int, int]] = []
        self.point_samples = 0
        self.point_frames = 0
        self.point_buffer: deque[tuple[float, float]] = deque(maxlen=10)

    def start(self) -> None:
        self.active = True
        self.completed = False
        self.target_index = 0
        self.samples.clear()
        self.point_samples = 0
        self.point_frames = 0
        self.point_buffer.clear()

    def stop(self) -> None:
        self.active = False

    def current_target(self) -> tuple[int, int]:
        ratio_x, ratio_y = self.targets[self.target_index]
        return int(ratio_x * self.screen_w), int(ratio_y * self.screen_h)

    def progress_label(self) -> str:
        if self.completed:
            return "завершена"
        if not self.active:
            return "неактивна"
        return (
            f"точка {self.target_index + 1}/{len(self.targets)} "
            f"образец {self.point_samples}/{self.samples_per_point}"
        )

    def ready_for_sample(self) -> bool:
        if self.point_frames < self.settle_frames or len(self.point_buffer) < 6:
            return False
        values = np.array(self.point_buffer, dtype=np.float32)
        spread_x = float(np.std(values[:, 0]))
        spread_y = float(np.std(values[:, 1]))
        return spread_x < 0.040 and spread_y < 0.045

    def add_sample(self, ratio_x: float | None, ratio_y: float | None, confidence: float) -> bool:
        if not self.active or ratio_x is None or ratio_y is None:
            self.point_frames += 1
            return False
        if confidence < 0.35:
            self.point_frames += 1
            return False

        self.point_frames += 1
        self.point_buffer.append((ratio_x, ratio_y))
        if not self.ready_for_sample():
            return False
        if (self.point_frames - self.settle_frames) % self.sample_stride != 0:
            return False

        screen_x, screen_y = self.current_target()
        stable_values = np.array(self.point_buffer, dtype=np.float32)
        stable_x = float(np.median(stable_values[:, 0]))
        stable_y = float(np.median(stable_values[:, 1]))
        self.samples.append((stable_x, stable_y, screen_x, screen_y))
        self.point_samples += 1
        if self.point_samples >= self.samples_per_point:
            self.point_samples = 0
            self.target_index += 1
            self.point_frames = 0
            self.point_buffer.clear()
            if self.target_index >= len(self.targets):
                self.active = False
                self.completed = True
                return True
        return False

    def render(self) -> np.ndarray:
        frame = np.full((self.screen_h, self.screen_w, 3), 18, dtype=np.uint8)
        panel_w = min(560, self.screen_w - 80)
        panel_x = 40
        panel_y = 34
        draw_panel(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + 210), 0.92)

        put_text(frame, "Калибровка экрана", (panel_x + 24, panel_y + 40), 0.84, COLOR_TEXT, 2)
        put_text(frame, "Смотрите на маркер и держите голову ровно", (panel_x + 24, panel_y + 73), 0.50, COLOR_MUTED)
        put_text(frame, self.progress_label(), (panel_x + 24, panel_y + 108), 0.58, COLOR_ACCENT, 2)

        if self.active:
            target_x, target_y = self.current_target()
            marker_color = COLOR_ACCENT if self.ready_for_sample() else COLOR_WARN
            radius = 42 if self.ready_for_sample() else 32
            cv2.circle(frame, (target_x, target_y), radius + 10, COLOR_PANEL_2, 2, cv2.LINE_AA)
            cv2.circle(frame, (target_x, target_y), radius, marker_color, 3, cv2.LINE_AA)
            cv2.circle(frame, (target_x, target_y), 14, COLOR_TEXT, 2, cv2.LINE_AA)
            cv2.circle(frame, (target_x, target_y), 5, COLOR_TEXT, -1, cv2.LINE_AA)

            bar_x = panel_x + 24
            bar_y = panel_y + 144
            bar_w = panel_w - 48
            fill = float(np.clip(self.point_frames / max(1, self.settle_frames), 0.0, 1.0))
            draw_progress_bar(frame, "стабилизация", fill, (bar_x, bar_y), bar_w, marker_color)
            draw_progress_bar(
                frame,
                "образцы",
                self.point_samples / max(1, self.samples_per_point),
                (bar_x, bar_y + 38),
                bar_w,
                COLOR_BLUE,
            )

        blend_rect(frame, (0, self.screen_h - 42), (self.screen_w, self.screen_h), COLOR_PANEL, 0.90)
        put_text(frame, "Q - отменить калибровку", (40, self.screen_h - 15), 0.50, COLOR_TEXT)
        return frame

    def build_mapper(self) -> ScreenMapper:
        if len(self.samples) < 12:
            raise RuntimeError("Собрано недостаточно образцов калибровки.")
        return ScreenMapper.fit(self.samples)


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
            "Не удалось скачать модель MediaPipe Face Landmarker. "
            f"Скачайте ее вручную с {FACE_LANDMARKER_MODEL_URL} и сохраните в "
            f"{FACE_LANDMARKER_MODEL_PATH}."
        ) from exc

    return FACE_LANDMARKER_MODEL_PATH


def create_face_landmarker():
    if mp is None:
        raise RuntimeError(
            "MediaPipe не установлен. Используйте Python 3.10-3.12, затем выполните "
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


def draw_panel(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    alpha: float = 0.82,
) -> None:
    blend_rect(frame, top_left, bottom_right, COLOR_PANEL, alpha)
    cv2.rectangle(frame, top_left, bottom_right, COLOR_LINE, 1, cv2.LINE_AA)


def put_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = COLOR_TEXT,
    thickness: int = 1,
) -> None:
    if Image is not None and ImageDraw is not None and ImageFont is not None:
        font_size = max(10, int(round(scale * 32)))
        font_key = ("bold" if thickness > 1 else "regular", font_size)
        font = FONT_CACHE.get(font_key)
        if font is None:
            font_names = ["segoeuib.ttf", "arialbd.ttf"] if thickness > 1 else ["segoeui.ttf", "arial.ttf"]
            for font_name in font_names:
                try:
                    font = ImageFont.truetype(f"C:/Windows/Fonts/{font_name}", font_size)
                    break
                except OSError:
                    font = None
            if font is None:
                font = ImageFont.load_default()
            FONT_CACHE[font_key] = font

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        draw.text(origin, text, font=font, fill=(color[2], color[1], color[0]))
        frame[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return

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
    width = max(88, 24 + len(text) * 8)
    draw_panel(frame, (x, y), (x + width, y + 26), 0.76)
    dot_color = accent if active else COLOR_MUTED
    cv2.circle(frame, (x + 13, y + 13), 4, dot_color, -1, cv2.LINE_AA)
    put_text(frame, text, (x + 24, y + 18), 0.42, COLOR_TEXT if active else COLOR_MUTED)
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
    put_text(frame, label, (x, y), 0.42, COLOR_MUTED)
    put_text(frame, f"{int(value * 100):3d}%", (x + width - 42, y), 0.42, COLOR_MUTED)
    bar_y = y + 10
    cv2.rectangle(frame, (x, bar_y), (x + width, bar_y + 8), COLOR_PANEL_2, -1)
    cv2.rectangle(frame, (x, bar_y), (x + int(width * value), bar_y + 8), accent, -1)
    cv2.rectangle(frame, (x, bar_y), (x + width, bar_y + 8), COLOR_LINE, 1, cv2.LINE_AA)


def draw_metric(
    frame: np.ndarray,
    label: str,
    value: str,
    origin: tuple[int, int],
    width: int,
) -> None:
    x, y = origin
    put_text(frame, label, (x, y), 0.40, COLOR_MUTED)
    put_text(frame, value, (x + width - 124, y), 0.43, COLOR_TEXT)


def draw_gaze_pad(
    frame: np.ndarray,
    result: FrameResult,
    estimator: GazeEstimator,
    origin: tuple[int, int],
    size: int,
) -> None:
    x, y = origin
    draw_panel(frame, (x, y), (x + size, y + size), 0.70)
    put_text(frame, "карта взгляда", (x + 12, y + 22), 0.38, COLOR_MUTED)

    grid_top = y + 34
    grid_bottom = y + size - 12
    grid_left = x + 12
    grid_right = x + size - 12
    cv2.rectangle(frame, (grid_left, grid_top), (grid_right, grid_bottom), COLOR_PANEL_2, -1)
    cv2.rectangle(frame, (grid_left, grid_top), (grid_right, grid_bottom), COLOR_LINE, 1, cv2.LINE_AA)
    cv2.line(
        frame,
        (grid_left + (grid_right - grid_left) // 2, grid_top + 8),
        (grid_left + (grid_right - grid_left) // 2, grid_bottom - 8),
        (74, 76, 80),
        1,
    )
    cv2.line(
        frame,
        (grid_left + 8, grid_top + (grid_bottom - grid_top) // 2),
        (grid_right - 8, grid_top + (grid_bottom - grid_top) // 2),
        (74, 76, 80),
        1,
    )

    usable_w = grid_right - grid_left
    usable_h = grid_bottom - grid_top

    center_x = grid_left + int(estimator.center_x * usable_w)
    center_y = grid_top + int(estimator.center_y * usable_h)
    cv2.circle(frame, (center_x, center_y), 5, COLOR_BLUE, 1, cv2.LINE_AA)

    if result.ratio_x is None or result.ratio_y is None:
        put_text(frame, "лица нет", (grid_left + 17, grid_top + usable_h // 2 + 5), 0.46, COLOR_MUTED)
        return

    gaze_x = grid_left + int(result.ratio_x * usable_w)
    gaze_y = grid_top + int(result.ratio_y * usable_h)
    cv2.circle(frame, (gaze_x, gaze_y), 7, COLOR_ACCENT, -1, cv2.LINE_AA)
    cv2.circle(frame, (gaze_x, gaze_y), 13, COLOR_ACCENT, 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    result: FrameResult,
    estimator: GazeEstimator,
    log_enabled: bool,
    cursor_status: str,
    cursor_motion: str,
    cursor_hint: str,
    calibration_status: str,
) -> None:
    height, width = frame.shape[:2]
    margin = 14
    bottom_h = 38
    sidebar_w = min(328, max(278, width // 3))

    draw_panel(frame, (margin, margin), (margin + sidebar_w, height - margin - bottom_h), 0.76)
    blend_rect(frame, (0, height - bottom_h), (width, height), COLOR_PANEL, 0.84)

    panel_x = margin
    panel_y = margin
    content_x = panel_x + 18
    content_w = sidebar_w - 36

    put_text(frame, "Студия взгляда", (content_x, panel_y + 30), 0.68, COLOR_TEXT, 2)
    put_text(frame, "отслеживание взгляда с веб-камеры", (content_x, panel_y + 55), 0.42, COLOR_MUTED)

    direction = direction_label(result.stable_direction)
    direction_color = COLOR_DANGER if result.stable_direction == "unknown" else COLOR_ACCENT
    put_text(frame, "текущий взгляд", (content_x, panel_y + 92), 0.40, COLOR_MUTED)
    put_text(frame, direction, (content_x, panel_y + 132), 0.80, direction_color, 2)
    put_text(frame, f"сырой: {direction_label(result.raw_direction).lower()}", (content_x, panel_y + 158), 0.42, COLOR_MUTED)

    draw_progress_bar(
        frame,
        "уверенность",
        result.confidence,
        (content_x, panel_y + 192),
        content_w,
        COLOR_ACCENT if result.confidence >= 0.5 else COLOR_DANGER,
    )

    ratio_text = "x --  y --"
    if result.ratio_x is not None and result.ratio_y is not None:
        ratio_text = f"x {result.ratio_x:.2f}  y {result.ratio_y:.2f}"
    draw_metric(frame, "координаты", ratio_text, (content_x, panel_y + 238), content_w)
    draw_metric(frame, "глаза", str(result.eyes_found), (content_x, panel_y + 266), content_w)
    draw_metric(frame, "курсор", status_label(cursor_status), (content_x, panel_y + 294), content_w)
    draw_metric(frame, "журнал", "вкл" if log_enabled else "выкл", (content_x, panel_y + 322), content_w)

    put_text(frame, "движение", (content_x, panel_y + 364), 0.40, COLOR_MUTED)
    put_text(frame, status_label(cursor_motion), (content_x, panel_y + 388), 0.43, COLOR_TEXT)
    put_text(frame, "калибровка", (content_x, panel_y + 426), 0.40, COLOR_MUTED)
    put_text(frame, calibration_status, (content_x, panel_y + 450), 0.43, COLOR_TEXT)

    pad_size = min(content_w, max(132, height - panel_y - bottom_h - 490))
    if pad_size >= 120:
        draw_gaze_pad(frame, result, estimator, (content_x, panel_y + 478), pad_size)

    chips = [
        (f"курсор {status_label(cursor_status)}", cursor_status == "on", COLOR_ACCENT),
        ("журнал вкл" if log_enabled else "журнал выкл", log_enabled, COLOR_BLUE),
        (f"глаза {result.eyes_found}", result.eyes_found > 0, COLOR_WARN),
    ]
    chip_x = margin + sidebar_w + 12
    chip_y = margin
    for text, active, accent in chips:
        chip_x += draw_status_chip(frame, text, (chip_x, chip_y), active, accent) + 8

    footer = f"C калибровка   M курсор   R сброс   Q выход   {cursor_hint}"
    put_text(frame, footer, (margin, height - 14), 0.46, COLOR_TEXT)


def process_frame(
    frame: np.ndarray,
    face_landmarker,
    estimator: GazeEstimator,
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
    return result


class GazeStudioApp:
    def __init__(
        self,
        camera_index: int,
        log_path: Path | None,
        control_cursor: bool,
        cursor_smoothing: float,
        cursor_min_confidence: float,
        calibration_samples: int,
    ) -> None:
        self.root = tk.Tk()
        self.root.title("Студия отслеживания взгляда")
        self.root.geometry("1180x760")
        self.root.minsize(980, 650)

        self.camera_index = camera_index
        self.log_path = log_path
        self.calibration_samples = calibration_samples
        self.camera = None
        self.face_landmarker = None
        self.logger: CsvLogger | None = None
        self.estimator = GazeEstimator()
        self.cursor = CursorController(control_cursor, cursor_smoothing, cursor_min_confidence)
        self.screen_w, self.screen_h = get_screen_size()
        self.calibration = CalibrationSession(self.screen_w, self.screen_h, calibration_samples)
        self.calibration_window: tk.Toplevel | None = None
        self.calibration_label: tk.Label | None = None
        self.video_image = None
        self.calibration_image = None
        self.running = False
        self.closed = False

        self.cursor_enabled = tk.BooleanVar(value=control_cursor)
        self.log_enabled = tk.BooleanVar(value=log_path is not None)
        self.smoothing = tk.DoubleVar(value=cursor_smoothing)
        self.min_confidence = tk.DoubleVar(value=cursor_min_confidence)
        self.direction_var = tk.StringVar(value="-")
        self.raw_var = tk.StringVar(value="-")
        self.confidence_var = tk.StringVar(value="0%")
        self.eyes_var = tk.StringVar(value="0")
        self.ratio_var = tk.StringVar(value="x --  y --")
        self.cursor_var = tk.StringVar(value=status_label(self.cursor.status()))
        self.motion_var = tk.StringVar(value="пауза")
        self.calibration_var = tk.StringVar(value="неактивна")
        self.status_var = tk.StringVar(value="Готово")

        self._build_style()
        self._build_menu()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Control-q>", lambda _event: self.close())
        self.root.bind("<F5>", lambda _event: self.start())
        self.root.bind("<F6>", lambda _event: self.stop())
        self.root.bind("<Control-r>", lambda _event: self.reset_calibration())
        self.root.bind("c", lambda _event: self.start_calibration())
        self.root.bind("m", lambda _event: self.toggle_cursor_control())
        self.root.bind("r", lambda _event: self.reset_calibration())
        self.root.bind("q", lambda _event: self.close())
        self.root.after(100, self.start)

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Value.TLabel", font=("Segoe UI", 11))
        style.configure("BigValue.TLabel", font=("Segoe UI", 22, "bold"))
        style.configure("Status.TLabel", padding=(8, 4))

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)

        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Запустить камеру", command=self.start, accelerator="F5")
        file_menu.add_command(label="Остановить камеру", command=self.stop, accelerator="F6")
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.close, accelerator="Ctrl+Q")
        menu.add_cascade(label="Файл", menu=file_menu)

        tracking_menu = tk.Menu(menu, tearoff=False)
        tracking_menu.add_command(label="Калибровать экран", command=self.start_calibration)
        tracking_menu.add_command(label="Сбросить калибровку", command=self.reset_calibration, accelerator="Ctrl+R")
        tracking_menu.add_separator()
        tracking_menu.add_checkbutton(
            label="Управление курсором",
            variable=self.cursor_enabled,
            command=self.apply_cursor_toggle,
        )
        menu.add_cascade(label="Отслеживание", menu=tracking_menu)

        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="О программе", command=self.show_about)
        menu.add_cascade(label="Справка", menu=help_menu)

        self.root.config(menu=menu)

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)
        main.rowconfigure(0, weight=1)

        video_frame = ttk.LabelFrame(main, text="Камера")
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        self.video_label = ttk.Label(video_frame, anchor=tk.CENTER)
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        side = ttk.Frame(main, width=300)
        side.grid(row=0, column=1, sticky="ns")
        side.columnconfigure(0, weight=1)

        status_box = ttk.LabelFrame(side, text="Отслеживание")
        status_box.grid(row=0, column=0, sticky="ew")
        status_box.columnconfigure(1, weight=1)
        ttk.Label(status_box, text="Студия взгляда", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 2)
        )
        ttk.Label(status_box, textvariable=self.direction_var, style="BigValue.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 12)
        )
        self._add_status_row(status_box, 2, "Сырой сигнал", self.raw_var)
        self._add_status_row(status_box, 3, "Уверенность", self.confidence_var)
        self._add_status_row(status_box, 4, "Глаза", self.eyes_var)
        self._add_status_row(status_box, 5, "Координаты", self.ratio_var)
        self._add_status_row(status_box, 6, "Курсор", self.cursor_var)
        self._add_status_row(status_box, 7, "Движение", self.motion_var)
        self._add_status_row(status_box, 8, "Калибровка", self.calibration_var)

        control_box = ttk.LabelFrame(side, text="Управление")
        control_box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        control_box.columnconfigure(0, weight=1)
        control_box.columnconfigure(1, weight=1)
        ttk.Button(control_box, text="Старт", command=self.start).grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        ttk.Button(control_box, text="Стоп", command=self.stop).grid(row=0, column=1, sticky="ew", padx=8, pady=(8, 4))
        ttk.Button(control_box, text="Калибровка", command=self.start_calibration).grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        ttk.Button(control_box, text="Сброс", command=self.reset_calibration).grid(row=1, column=1, sticky="ew", padx=8, pady=4)
        ttk.Checkbutton(
            control_box,
            text="Управление курсором",
            variable=self.cursor_enabled,
            command=self.apply_cursor_toggle,
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 4))
        ttk.Checkbutton(
            control_box,
            text="Записывать CSV-журнал",
            variable=self.log_enabled,
            command=self.apply_logging_toggle,
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=4)

        settings_box = ttk.LabelFrame(side, text="Настройки")
        settings_box.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        settings_box.columnconfigure(0, weight=1)
        ttk.Label(settings_box, text="Сглаживание курсора").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 0))
        ttk.Scale(settings_box, from_=0.0, to=0.95, variable=self.smoothing, command=self.apply_settings).grid(
            row=1, column=0, sticky="ew", padx=8, pady=(0, 8)
        )
        ttk.Label(settings_box, text="Минимальная уверенность").grid(row=2, column=0, sticky="w", padx=8)
        ttk.Scale(settings_box, from_=0.0, to=1.0, variable=self.min_confidence, command=self.apply_settings).grid(
            row=3, column=0, sticky="ew", padx=8, pady=(0, 8)
        )

        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _add_status_row(self, parent: ttk.LabelFrame, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=3)
        ttk.Label(parent, textvariable=variable, style="Value.TLabel").grid(row=row, column=1, sticky="e", padx=10, pady=3)

    def apply_settings(self, _event=None) -> None:
        self.cursor.smoothing = float(np.clip(self.smoothing.get(), 0.0, 0.95))
        self.cursor.min_confidence = float(np.clip(self.min_confidence.get(), 0.0, 1.0))

    def apply_cursor_toggle(self) -> None:
        self.cursor.active = bool(self.cursor_enabled.get()) and self.cursor.available
        self.cursor.reset_motion()
        self.cursor_var.set(status_label(self.cursor.status()))

    def toggle_cursor_control(self) -> None:
        self.cursor_enabled.set(not self.cursor_enabled.get())
        self.apply_cursor_toggle()

    def apply_logging_toggle(self) -> None:
        if self.log_enabled.get():
            if self.logger is None:
                path = self.log_path or Path("gaze_log.csv")
                self.log_path = path
                self.logger = CsvLogger(path)
            self.status_var.set(f"Запись журнала: {self.log_path}")
            return

        if self.logger is not None:
            self.logger.close()
            self.logger = None
        self.status_var.set("Запись журнала выключена")

    def start(self) -> None:
        if self.running:
            return
        try:
            if self.face_landmarker is None:
                self.face_landmarker = create_face_landmarker()
            if self.camera is None:
                self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                raise RuntimeError(f"Не удалось открыть камеру #{self.camera_index}.")
            if self.log_enabled.get() and self.logger is None:
                self.logger = CsvLogger(self.log_path or Path("gaze_log.csv"))
            self.running = True
            self.status_var.set("Камера запущена")
            self.update_frame()
        except Exception as exc:
            self.status_var.set(str(exc))
            messagebox.showerror("Студия отслеживания взгляда", str(exc))

    def stop(self) -> None:
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.close_calibration_window()
        self.status_var.set("Камера остановлена")

    def start_calibration(self) -> None:
        self.calibration.start()
        self.cursor.reset_motion()
        self.open_calibration_window()
        self.update_calibration_window()
        self.status_var.set("Калибровка запущена")

    def reset_calibration(self) -> None:
        self.calibration.stop()
        self.cursor.mapper = None
        self.cursor.reset_motion()
        self.close_calibration_window()
        self.calibration_var.set("неактивна")
        self.cursor_var.set(status_label(self.cursor.status()))
        self.status_var.set("Калибровка сброшена")

    def open_calibration_window(self) -> None:
        if self.calibration_window is not None:
            return
        self.calibration_window = tk.Toplevel(self.root)
        self.calibration_window.title("Калибровка экрана")
        self.calibration_window.attributes("-fullscreen", True)
        self.calibration_window.configure(background="black")
        self.calibration_label = tk.Label(
            self.calibration_window,
            anchor=tk.CENTER,
            background="black",
            borderwidth=0,
            highlightthickness=0,
        )
        self.calibration_label.pack(fill=tk.BOTH, expand=True)
        self.calibration_window.lift()
        self.calibration_window.focus_force()
        self.calibration_window.bind("<Escape>", lambda _event: self.reset_calibration())
        self.calibration_window.bind("q", lambda _event: self.reset_calibration())
        self.calibration_window.protocol("WM_DELETE_WINDOW", self.reset_calibration)

    def close_calibration_window(self) -> None:
        if self.calibration_window is not None:
            self.calibration_window.destroy()
            self.calibration_window = None
            self.calibration_label = None

    def update_calibration_window(self) -> None:
        if self.calibration_window is None or self.calibration_label is None:
            return
        frame = self.calibration.render()
        self.calibration_image = self.frame_to_photo(frame, self.screen_w, self.screen_h)
        self.calibration_label.configure(image=self.calibration_image)

    def update_frame(self) -> None:
        if self.closed or not self.running or self.camera is None:
            return

        ok, frame = self.camera.read()
        if not ok:
            self.status_var.set("Не удалось получить кадр с камеры")
            self.stop()
            return

        result = process_frame(frame, self.face_landmarker, self.estimator)
        if self.logger is not None:
            self.logger.write(result)

        if self.calibration.active:
            done = self.calibration.add_sample(result.ratio_x, result.ratio_y, result.confidence)
            self.update_calibration_window()
            self.cursor.reset_motion()
            if done:
                self.cursor.set_mapper(self.calibration.build_mapper())
                self.cursor.active = bool(self.cursor_enabled.get()) and self.cursor.available
                self.close_calibration_window()
                self.status_var.set("Калибровка завершена")
        else:
            self.cursor.move(result.ratio_x, result.ratio_y, result.confidence)

        self.update_status(result)
        self.video_image = self.frame_to_photo(result.frame, 860, 640)
        self.video_label.configure(image=self.video_image)
        self.root.after(15, self.update_frame)

    def update_status(self, result: FrameResult) -> None:
        self.direction_var.set(direction_label(result.stable_direction))
        self.raw_var.set(direction_label(result.raw_direction).lower())
        self.confidence_var.set(f"{int(result.confidence * 100)}%")
        self.eyes_var.set(str(result.eyes_found))
        if result.ratio_x is None or result.ratio_y is None:
            self.ratio_var.set("x --  y --")
        else:
            self.ratio_var.set(f"x {result.ratio_x:.2f}  y {result.ratio_y:.2f}")
        self.cursor_var.set(status_label(self.cursor.status()))
        self.motion_var.set(status_label(self.cursor.movement_status(result, self.calibration.active)))
        self.calibration_var.set(self.calibration.progress_label())

    def frame_to_photo(self, frame: np.ndarray, max_w: int, max_h: int) -> tk.PhotoImage:
        height, width = frame.shape[:2]
        scale = min(max_w / width, max_h / height, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ok, buffer = cv2.imencode(".ppm", rgb)
        if not ok:
            raise RuntimeError("Не удалось отрисовать кадр.")
        return tk.PhotoImage(data=buffer.tobytes(), format="PPM")

    def show_about(self) -> None:
        messagebox.showinfo(
            "О программе",
            "Студия отслеживания взгляда\n\nДемонстрация отслеживания взгляда через веб-камеру на OpenCV и MediaPipe.",
        )

    def close(self) -> None:
        self.closed = True
        self.running = False
        self.close_calibration_window()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        if self.face_landmarker is not None:
            self.face_landmarker.close()
            self.face_landmarker = None
        if self.logger is not None:
            self.logger.close()
            self.logger = None
        cv2.destroyAllWindows()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_desktop_app(
    camera_index: int,
    log_path: Path | None,
    control_cursor: bool,
    cursor_smoothing: float,
    cursor_min_confidence: float,
    calibration_samples: int,
) -> None:
    app = GazeStudioApp(
        camera_index=camera_index,
        log_path=log_path,
        control_cursor=control_cursor,
        cursor_smoothing=cursor_smoothing,
        cursor_min_confidence=cursor_min_confidence,
        calibration_samples=calibration_samples,
    )
    app.run()


def run(
    camera_index: int,
    log_path: Path | None,
    control_cursor: bool,
    cursor_smoothing: float,
    cursor_min_confidence: float,
    calibration_samples: int,
) -> None:
    estimator = GazeEstimator()
    logger = CsvLogger(log_path)
    cursor = CursorController(
        active_by_default=control_cursor,
        smoothing=cursor_smoothing,
        min_confidence=cursor_min_confidence,
    )
    face_landmarker = create_face_landmarker()
    screen_w, screen_h = get_screen_size()
    calibration = CalibrationSession(screen_w, screen_h, calibration_samples)

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(
            f"Не удалось открыть камеру #{camera_index}. Попробуйте другой индекс через --camera."
        )

    window_name = "Студия взгляда"
    calibration_window = "Калибровка"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)
    cursor_hint = f"сгл {cursor_smoothing:.2f}  ув {cursor_min_confidence:.2f}"

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Не удалось получить кадр с камеры.")

            result = process_frame(frame, face_landmarker, estimator)
            logger.write(result)

            if calibration.active:
                done = calibration.add_sample(result.ratio_x, result.ratio_y, result.confidence)
                cv2.namedWindow(calibration_window, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(calibration_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow(calibration_window, calibration.render())
                cursor.reset_motion()
                if done:
                    cursor.set_mapper(calibration.build_mapper())
                    cursor.active = True
                    calibration.stop()
                    cv2.destroyWindow(calibration_window)
            else:
                cursor.move(result.ratio_x, result.ratio_y, result.confidence)

            draw_hud(
                result.frame,
                result,
                estimator,
                log_path is not None,
                cursor.status(),
                cursor.movement_status(result, calibration.active),
                cursor_hint,
                calibration.progress_label(),
            )
            cv2.imshow(window_name, result.frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                if not calibration.active:
                    calibration.start()
            if key == ord("m"):
                cursor.toggle()
            if key == ord("r"):
                calibration.stop()
                cursor.mapper = None
                cursor.reset_motion()
                try:
                    cv2.destroyWindow(calibration_window)
                except cv2.error:
                    pass
    finally:
        face_landmarker.close()
        logger.close()
        camera.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = RussianArgumentParser(
        description="Демонстрационная система отслеживания взгляда через веб-камеру.",
        add_help=False,
    )
    parser.usage = "%(prog)s [-h] [--camera CAMERA] [--log LOG] [--control-cursor] [--cursor-smoothing VALUE] [--cursor-min-confidence VALUE] [--calibration-samples N] [--opencv-ui]"
    parser._positionals.title = "позиционные аргументы"
    parser._optionals.title = "параметры"
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="показать эту справку и выйти.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Индекс камеры, обычно 0 для встроенной веб-камеры.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Необязательный CSV-файл для записи измерений взгляда.",
    )
    parser.add_argument(
        "--control-cursor",
        action="store_true",
        help="Включить управление курсором взглядом после калибровки. Клавиша M переключает режим.",
    )
    parser.add_argument(
        "--cursor-smoothing",
        type=float,
        default=0.90,
        help="Сглаживание положения курсора от 0.0 до 0.95. Чем выше, тем плавнее.",
    )
    parser.add_argument(
        "--cursor-min-confidence",
        type=float,
        default=0.30,
        help="Минимальная уверенность взгляда для движения курсора.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=10,
        help="Количество стабильных образцов взгляда для каждой точки калибровки.",
    )
    parser.add_argument(
        "--opencv-ui",
        action="store_true",
        help="Использовать старый интерфейс OpenCV вместо настольного окна.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = run if args.opencv_ui else run_desktop_app
    runner(
        camera_index=args.camera,
        log_path=args.log,
        control_cursor=args.control_cursor,
        cursor_smoothing=float(np.clip(args.cursor_smoothing, 0.0, 0.95)),
        cursor_min_confidence=float(np.clip(args.cursor_min_confidence, 0.0, 1.0)),
        calibration_samples=max(4, args.calibration_samples),
    )
