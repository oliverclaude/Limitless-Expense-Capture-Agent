"""Detect a slip rectangle, perspective-correct, and apply a light contrast pass."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

MIN_AREA_FRAC = 0.12
MIN_SIDE_PX = 200
MAX_ASPECT = 15.0
MIN_GRAY_STD = 8.0


def straighten_slip(path: Path) -> tuple[bytes | None, str]:
    image = _load_bgr(path)
    if image is None:
        return None, "failed (unreadable image)"

    proc, scale = _downscale(image)
    quad = _find_quad(proc)
    if quad is None:
        return None, "failed (no slip rectangle)"

    full_quad = quad / scale
    h, w = image.shape[:2]
    if _near_full_frame(full_quad, w, h):
        out = _light_contrast(image)
        status = "contrast-only"
    else:
        warped = _four_point_transform(image, full_quad)
        if warped is None:
            return None, "failed (bad warp)"
        if not _acceptable(warped):
            return None, "failed (poor crop)"
        out = _light_contrast(warped)
        status = "warped"

    jpeg = _encode_jpeg(out)
    if jpeg is None:
        return None, "failed (jpeg encode)"
    return jpeg, status


def _load_bgr(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            rgb = np.array(im.convert("RGB"))
    except (OSError, ValueError):
        return None
    if rgb.ndim != 3 or rgb.size == 0:
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _downscale(image: np.ndarray, max_side: int = 1080) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _find_quad(proc: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    quad = _best_quad(edges, proc.shape)
    if quad is not None:
        return quad
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
    )
    thresh = cv2.bitwise_not(thresh)
    return _best_quad(thresh, proc.shape)


def _best_quad(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray | None:
    img_area = float(shape[0] * shape[1])
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA_FRAC * img_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2).astype(np.float32)
            if len(np.unique(np.round(pts, 1), axis=0)) == 4:
                return pts
        if area >= 0.18 * img_area:
            box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
            if len(np.unique(np.round(box, 1), axis=0)) == 4:
                return box
    return None


def _near_full_frame(pts: np.ndarray, width: int, height: int, margin: float = 0.04) -> bool:
    xs, ys = pts[:, 0], pts[:, 1]
    return bool(
        xs.min() <= margin * width
        and ys.min() <= margin * height
        and xs.max() >= (1.0 - margin) * width
        and ys.max() >= (1.0 - margin) * height
    )


def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]
    return np.stack([top_left, top_right, bottom_right, bottom_left]).astype(np.float32)


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray | None:
    rect = _order_points(pts)
    if len(np.unique(np.round(rect, 1), axis=0)) < 4:
        return None
    (tl, tr, br, bl) = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < MIN_SIDE_PX or height < MIN_SIDE_PX:
        return None
    dest = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dest)
    return cv2.warpPerspective(image, matrix, (width, height))


def _acceptable(warped: np.ndarray) -> bool:
    h, w = warped.shape[:2]
    if min(h, w) < MIN_SIDE_PX:
        return False
    aspect = max(h, w) / float(min(h, w))
    if aspect > MAX_ASPECT:
        return False
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < MIN_GRAY_STD:
        return False
    return True


def _light_contrast(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
    return cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)


def _encode_jpeg(bgr: np.ndarray) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return None
    return buf.tobytes()
