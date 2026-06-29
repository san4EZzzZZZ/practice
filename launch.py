"""Точка входа для запуска приложения."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from eye_tracking import (
    GAZE_ALGORITHM_VALUES,
    run,
    run_desktop_app,
    run_image_mode,
)


class RussianArgumentParser(argparse.ArgumentParser):
    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "использование:")

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "использование:")


def parse_args() -> argparse.Namespace:
    parser = RussianArgumentParser(
        description="Демонстрационная система отслеживания взгляда через веб-камеру.",
        add_help=False,
    )
    parser.usage = "%(prog)s [-h] [--camera CAMERA] [--log LOG] [--control-cursor] [--cursor-mode MODE] [--gaze-algorithm MODE] [--cursor-smoothing VALUE] [--cursor-min-confidence VALUE] [--calibration-samples N] [--opencv-ui] [--image IMAGE | --image-dir IMAGE_DIR] [--output-dir OUTPUT_DIR] [--preview]"
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
        help="Включить управление курсором взглядом сразу, без калибровки. Клавиша M переключает режим.",
    )
    parser.add_argument(
        "--gaze-algorithm",
        choices=GAZE_ALGORITHM_VALUES,
        default="threshold",
        help="Алгоритм слежения за взглядом: threshold, majority, adaptive, ema или hysteresis.",
    )
    parser.add_argument(
        "--cursor-mode",
        choices=("direct", "relative", "calibrated"),
        default="direct",
        help="Алгоритм управления курсором: direct работает по сырым координатам, relative двигает курсор по отклонению, calibrated использует экранную калибровку.",
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
        default=6,
        help="Количество стабильных образцов взгляда для каждой точки калибровки.",
    )
    parser.add_argument(
        "--opencv-ui",
        action="store_true",
        help="Использовать старый интерфейс OpenCV вместо настольного окна.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Анализировать одно изображение вместо живой камеры.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Анализировать все изображения в папке.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Папка для сохранения размеченных изображений.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Показывать окно предпросмотра для одного изображения.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.image is not None or args.image_dir is not None:
            run_image_mode(
                image_path=args.image,
                image_dir=args.image_dir,
                output_dir=args.output_dir,
                preview=bool(args.preview),
            )
        else:
            runner = run if args.opencv_ui else run_desktop_app
            runner(
                camera_index=args.camera,
                log_path=args.log,
                control_cursor=args.control_cursor,
                gaze_algorithm=args.gaze_algorithm,
                cursor_mode=args.cursor_mode,
                cursor_smoothing=float(np.clip(args.cursor_smoothing, 0.0, 0.95)),
                cursor_min_confidence=float(np.clip(args.cursor_min_confidence, 0.0, 1.0)),
                calibration_samples=max(3, args.calibration_samples),
            )
    except KeyboardInterrupt:
        print("\nЗапуск прерван пользователем.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Ошибка запуска: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
