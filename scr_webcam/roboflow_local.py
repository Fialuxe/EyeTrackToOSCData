import math
from pathlib import Path

import cv2
import numpy as np
import torch
from l2cs import Pipeline

DISTANCE_TO_OBJECT = 460  # mm
HEIGHT_OF_HUMAN_FACE = 250  # mm

pipeline = Pipeline(
    weights=Path("models/L2CSNet_gaze360.pkl"),
    arch="ResNet50",
    device=torch.device("cpu"),
    include_detector=True,
    confidence_threshold=0.5,
)


def detect_gazes(frame: np.ndarray):
    try:
        results = pipeline.step(frame)
    except Exception:
        return []

    if results.pitch.size == 0:
        return []

    gazes = []
    for i in range(len(results.pitch)):
        box = results.bboxes[i]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1
        gazes.append({
            "yaw":   float(results.yaw[i]),
            "pitch": float(results.pitch[i]),
            "face":  {"x": cx, "y": cy, "width": w, "height": h},
        })
    return gazes


def draw_gaze(img: np.ndarray, gaze: dict):
    # draw face bounding box
    face = gaze["face"]
    x_min = int(face["x"] - face["width"] / 2)
    x_max = int(face["x"] + face["width"] / 2)
    y_min = int(face["y"] - face["height"] / 2)
    y_max = int(face["y"] + face["height"] / 2)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    # draw gaze arrow
    _, imgW = img.shape[:2]
    arrow_length = imgW / 2
    dx = -arrow_length * np.sin(gaze["yaw"]) * np.cos(gaze["pitch"])
    dy = -arrow_length * np.sin(gaze["pitch"])
    cv2.arrowedLine(
        img,
        (int(face["x"]), int(face["y"])),
        (int(face["x"] + dx), int(face["y"] + dy)),
        (0, 0, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.18,
    )

    # draw label and score
    label = "yaw {:.2f}  pitch {:.2f}".format(
        gaze["yaw"] / np.pi * 180, gaze["pitch"] / np.pi * 180
    )
    cv2.putText(
        img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
    )

    return img


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        gazes = detect_gazes(frame)

        if len(gazes) == 0:
            cv2.imshow("gaze", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        gaze = gazes[0]
        draw_gaze(frame, gaze)

        image_height, image_width = frame.shape[:2]

        length_per_pixel = HEIGHT_OF_HUMAN_FACE / gaze["face"]["height"]

        dx = -DISTANCE_TO_OBJECT * np.tan(gaze["yaw"]) / length_per_pixel
        dx = dx if not np.isnan(dx) else 100000000
        dy = (
            -DISTANCE_TO_OBJECT
            * np.arccos(gaze["yaw"])
            * np.tan(gaze["pitch"])
            / length_per_pixel
        )
        dy = dy if not np.isnan(dy) else 100000000
        gaze_point = int(image_width / 2 + dx), int(image_height / 2 + dy)

        quadrants = [
            (
                "center",
                (
                    int(image_width / 4),
                    int(image_height / 4),
                    int(image_width / 4 * 3),
                    int(image_height / 4 * 3),
                ),
            ),
            ("top_left", (0, 0, int(image_width / 2), int(image_height / 2))),
            (
                "top_right",
                (int(image_width / 2), 0, image_width, int(image_height / 2)),
            ),
            (
                "bottom_left",
                (0, int(image_height / 2), int(image_width / 2), image_height),
            ),
            (
                "bottom_right",
                (
                    int(image_width / 2),
                    int(image_height / 2),
                    image_width,
                    image_height,
                ),
            ),
        ]

        for quadrant, (x_min, y_min, x_max, y_max) in quadrants:
            if x_min <= gaze_point[0] <= x_max and y_min <= gaze_point[1] <= y_max:
                print(quadrant)
                break

        cv2.circle(frame, gaze_point, 25, (0, 0, 255), -1)

        cv2.imshow("gaze", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
