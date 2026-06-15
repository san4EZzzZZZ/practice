import argparse
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GazeResult:
    direction: str
    pupil_center: tuple[int, int] | None
    ratio_x: float | None
    ratio_y: float | None


def load_cascade(name: str) -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + name
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Cannot load cascade: {path}")
    return cascade


def detect_pupil(eye_gray: np.ndarray) -> GazeResult:
    blurred = cv2.GaussianBlur(eye_gray, (7, 7), 0)

    # The pupil is usually one of the darkest areas in the eye crop.
    _, threshold = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return GazeResult("unknown", None, None, None)

    height, width = eye_gray.shape[:2]
    min_area = max(8, int(width * height * 0.01))
    candidates = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    if not candidates:
        return GazeResult("unknown", None, None, None)

    pupil = max(candidates, key=cv2.contourArea)
    moments = cv2.moments(pupil)
    if moments["m00"] == 0:
        return GazeResult("unknown", None, None, None)

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    ratio_x = center_x / max(1, width)
    ratio_y = center_y / max(1, height)

    if ratio_x < 0.38:
        horizontal = "left"
    elif ratio_x > 0.62:
        horizontal = "right"
    else:
        horizontal = "center"

    if ratio_y < 0.36:
        vertical = "up"
    elif ratio_y > 0.68:
        vertical = "down"
    else:
        vertical = "center"

    if horizontal == "center" and vertical == "center":
        direction = "center"
    elif vertical == "center":
        direction = horizontal
    elif horizontal == "center":
        direction = vertical
    else:
        direction = f"{vertical}-{horizontal}"

    return GazeResult(direction, (center_x, center_y), ratio_x, ratio_y)


def draw_eye_result(frame: np.ndarray, eye_box: tuple[int, int, int, int], result: GazeResult) -> None:
    x, y, width, height = eye_box
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 220, 120), 2)

    if result.pupil_center is None:
        return

    pupil_x = x + result.pupil_center[0]
    pupil_y = y + result.pupil_center[1]
    cv2.circle(frame, (pupil_x, pupil_y), 4, (0, 0, 255), -1)


def combine_directions(results: list[GazeResult]) -> str:
    known = [item for item in results if item.direction != "unknown"]
    if not known:
        return "unknown"

    avg_x = float(np.mean([item.ratio_x for item in known if item.ratio_x is not None]))
    avg_y = float(np.mean([item.ratio_y for item in known if item.ratio_y is not None]))

    if avg_x < 0.38:
        horizontal = "left"
    elif avg_x > 0.62:
        horizontal = "right"
    else:
        horizontal = "center"

    if avg_y < 0.36:
        vertical = "up"
    elif avg_y > 0.68:
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


def process_frame(
    frame: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    eye_cascade: cv2.CascadeClassifier,
) -> np.ndarray:
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

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
            result = detect_pupil(eye_gray)
            gaze_results.append(result)
            draw_eye_result(frame, absolute_box, result)

    gaze = combine_directions(gaze_results)
    cv2.putText(
        frame,
        f"Gaze: {gaze}",
        (24, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (40, 240, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Press Q to exit",
        (24, frame.shape[0] - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    return frame


def run(camera_index: int) -> None:
    face_cascade = load_cascade("haarcascade_frontalface_default.xml")
    eye_cascade = load_cascade("haarcascade_eye.xml")

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

            result = process_frame(frame, face_cascade, eye_cascade)
            cv2.imshow("Simple Eye Tracking", result)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.camera)
