"""Detect a slip, perspective-correct, and apply a light contrast pass.

Real photos are messy: folds, a finger on the slip, a book used as a weight.
We prefer a paper region with dense text over a clean four-sided outline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

MIN_AREA_FRAC = 0.06
MIN_SIDE_PX = 160
MAX_ASPECT = 15.0
MIN_GRAY_STD = 8.0
MIN_TEXT_SCORE = 6.0


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
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    candidates: list[np.ndarray] = []
    candidates.extend(_contour_quads(edges, proc.shape))
    candidates.extend(_contour_quads(_adaptive_ink(blurred), proc.shape))
    band = _text_band_quad(proc, edges)
    if band is not None:
        candidates.append(band)
    paper_rect = _paper_min_rect(proc, edges)
    if paper_rect is not None:
        candidates.append(paper_rect)

    best: np.ndarray | None = None
    best_score = MIN_TEXT_SCORE
    for quad in candidates:
        score = _quad_text_score(edges, quad, proc.shape)
        if score > best_score:
            best_score = score
            best = quad
    return best


def _adaptive_ink(blurred: np.ndarray) -> np.ndarray:
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
    )
    return cv2.bitwise_not(thresh)


def _contour_quads(mask: np.ndarray, shape: tuple[int, ...]) -> list[np.ndarray]:
    img_area = float(shape[0] * shape[1])
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    found: list[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA_FRAC * img_area:
            continue
        for frac in (0.02, 0.04, 0.06):
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, frac * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2).astype(np.float32)
                if _unique_pts(pts):
                    found.append(pts)
                    break
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.05 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2).astype(np.float32)
            if _unique_pts(pts):
                found.append(pts)
        box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
        if _unique_pts(box):
            found.append(box)
    return found


def _paper_masks(proc: np.ndarray) -> list[np.ndarray]:
    light = cv2.cvtColor(proc, cv2.COLOR_BGR2LAB)[:, :, 0]
    otsu_t, otsu = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    percentile_t = float(np.percentile(light, 60))
    relative = ((light >= max(otsu_t, percentile_t)).astype(np.uint8)) * 255
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masks: list[np.ndarray] = []
    for raw in (otsu, relative):
        mask = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, close, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
        masks.append(mask)
    return masks


def _components(mask: np.ndarray) -> list[np.ndarray]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    img_area = float(mask.shape[0] * mask.shape[1])
    out: list[np.ndarray] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA_FRAC * img_area or area > 0.92 * img_area:
            continue
        out.append((labels == i).astype(np.uint8) * 255)
    return out


def _text_band_quad(proc: np.ndarray, edges: np.ndarray) -> np.ndarray | None:
    best: np.ndarray | None = None
    best_score = -1.0
    for paper in _paper_masks(proc):
        for comp in _components(paper):
            quad = _fit_text_band(comp, edges)
            if quad is None:
                continue
            score = _quad_text_score(edges, quad, proc.shape)
            if score > best_score:
                best_score = score
                best = quad
    return best


def _fit_text_band(mask: np.ndarray, edges: np.ndarray) -> np.ndarray | None:
    h, w = mask.shape
    lefts = np.full(h, np.nan, dtype=np.float64)
    rights = np.full(h, np.nan, dtype=np.float64)
    density = np.zeros(h, dtype=np.float64)
    for y in range(h):
        xs = np.flatnonzero(mask[y] > 0)
        if xs.size < 8:
            continue
        lefts[y] = float(xs[0])
        rights[y] = float(xs[-1])
        strip = edges[y, int(xs[0]) : int(xs[-1]) + 1]
        density[y] = float(strip.mean()) if strip.size else 0.0

    valid = np.isfinite(lefts) & ((rights - lefts) > 0.08 * w)
    if int(valid.sum()) < max(24, int(0.08 * h)):
        return None

    dens_valid = density[valid]
    floor = max(float(np.median(dens_valid)) * 0.45, 4.0)
    text = valid & (density >= floor)
    y0, y1 = _longest_true_run(text)
    if y1 - y0 < max(24, int(0.12 * h)):
        ys = np.flatnonzero(valid)
        y0, y1 = int(ys.min()), int(ys.max())
    band = np.arange(y0, y1 + 1)
    band = band[np.isfinite(lefts[band]) & np.isfinite(rights[band])]
    if band.size < 16:
        return None

    left_fit = np.polyfit(band, lefts[band], 1)
    right_fit = np.polyfit(band, rights[band], 1)
    top, bottom = float(band[0]), float(band[-1])

    def edge_x(fit: np.ndarray, y: float) -> float:
        return float(np.clip(fit[0] * y + fit[1], 0, w - 1))

    tl, tr = edge_x(left_fit, top), edge_x(right_fit, top)
    bl, br = edge_x(left_fit, bottom), edge_x(right_fit, bottom)
    if tr - tl < 0.08 * w or br - bl < 0.08 * w:
        return None
    quad = np.array([[tl, top], [tr, top], [br, bottom], [bl, bottom]], dtype=np.float32)
    if not _unique_pts(quad):
        return None
    return quad


def _paper_min_rect(proc: np.ndarray, edges: np.ndarray) -> np.ndarray | None:
    best: np.ndarray | None = None
    best_score = -1.0
    for paper in _paper_masks(proc):
        for comp in _components(paper):
            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
            if not _unique_pts(box):
                continue
            score = _quad_text_score(edges, box, proc.shape)
            if score > best_score:
                best_score = score
                best = box
    return best


def _longest_true_run(flags: np.ndarray) -> tuple[int, int]:
    best_start = 0
    best_end = 0
    start = None
    for i, value in enumerate(flags.tolist()):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - 1 - start > best_end - best_start:
                best_start, best_end = start, i - 1
            start = None
    if start is not None and len(flags) - 1 - start > best_end - best_start:
        best_start, best_end = start, len(flags) - 1
    return best_start, best_end


def _quad_text_score(edges: np.ndarray, quad: np.ndarray, shape: tuple[int, ...]) -> float:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    area = int(cv2.countNonZero(mask))
    img_area = shape[0] * shape[1]
    if area < MIN_AREA_FRAC * img_area:
        return -1.0
    dens = float(edges[mask > 0].mean()) if area else 0.0
    height = float(quad[:, 1].max() - quad[:, 1].min())
    width = float(max(quad[:, 0].max() - quad[:, 0].min(), 1.0))
    aspect = height / width
    return dens * (1.0 + 0.2 * min(aspect, 4.0))


def _unique_pts(pts: np.ndarray) -> bool:
    return len(np.unique(np.round(pts.reshape(-1, 2), 1), axis=0)) == 4


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
    if not _unique_pts(rect):
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
