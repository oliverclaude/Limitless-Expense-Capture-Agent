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


def straighten_slip_hybrid(path: Path) -> tuple[bytes | None, str]:
    """OpenCV first; xAI corners if the crop looks clipped or the box hits the photo edges."""
    opencv_jpeg, opencv_status = straighten_slip(path)
    if not _needs_xai_corners(path, opencv_jpeg):
        return opencv_jpeg, opencv_status
    try:
        from extract import CreditExhaustedError, slip_corners
    except ImportError:
        return opencv_jpeg, opencv_status
    try:
        corners = slip_corners(path)
    except CreditExhaustedError:
        raise
    except RuntimeError:
        corners = None
    if not corners:
        if opencv_jpeg is not None:
            return opencv_jpeg, opencv_status
        return None, "failed (no slip rectangle)"
    image = _load_bgr(path)
    if image is None:
        if opencv_jpeg is not None:
            return opencv_jpeg, opencv_status
        return None, "failed (unreadable image)"
    h, w = image.shape[:2]
    quad = np.array([[x * w, y * h] for x, y in corners], dtype=np.float32)
    jpeg, status = _finish_from_quad(image, quad)
    if jpeg is None:
        return opencv_jpeg, opencv_status if opencv_jpeg is not None else (None, status)
    if opencv_jpeg is not None and _ink_flush_to_side(_decode_jpeg(jpeg)) and not _ink_flush_to_side(
        _decode_jpeg(opencv_jpeg)
    ):
        return opencv_jpeg, opencv_status
    return jpeg, "warped (xai)"


def _needs_xai_corners(path: Path, opencv_jpeg: bytes | None) -> bool:
    if opencv_jpeg is None:
        return True
    image = _load_bgr(path)
    if image is None:
        return False
    proc, _scale = _downscale(image)
    quad = _find_quad(proc)
    if quad is not None and _quad_hits_opposite_borders(quad, proc.shape):
        return True
    decoded = _decode_jpeg(opencv_jpeg)
    return decoded is not None and _ink_flush_to_side(decoded)


def _quad_hits_opposite_borders(quad: np.ndarray, shape: tuple[int, ...], tol: float = 0.03) -> bool:
    height, width = shape[:2]
    xs, ys = quad[:, 0], quad[:, 1]
    top = float(ys.min()) <= tol * height
    bottom = float(ys.max()) >= (1.0 - tol) * height
    left = float(xs.min()) <= tol * width
    right = float(xs.max()) >= (1.0 - tol) * width
    return (top and bottom) or (left and right)


def _ink_flush_to_side(bgr: np.ndarray, frac: float = 0.03, thresh: float = 2.4) -> bool:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    h, w = edges.shape
    mx, my = max(6, int(w * frac)), max(6, int(h * frac))
    return bool(
        float(edges[:, :mx].mean()) > thresh
        or float(edges[:, w - mx :].mean()) > thresh
        or float(edges[:my, :].mean()) > thresh
        or float(edges[h - my :, :].mean()) > thresh
    )


def _decode_jpeg(jpeg: bytes) -> np.ndarray | None:
    arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return arr


def _finish_from_quad(image: np.ndarray, quad: np.ndarray) -> tuple[bytes | None, str]:
    h, w = image.shape[:2]
    padded = _inflate_quad(quad, (h, w), 0.03)
    warped = _four_point_transform(image, padded)
    if warped is None:
        return None, "failed (bad warp)"
    warped = _upright_slip(warped)
    if not _acceptable(warped):
        return None, "failed (poor crop)"
    out = _deskew(warped)
    trimmed = _trim_to_content(out)
    if _acceptable(trimmed):
        out = trimmed
    out = _flip_if_upside_down(out)
    out = _light_contrast(out)
    jpeg = _encode_jpeg(out)
    if jpeg is None:
        return None, "failed (jpeg encode)"
    return jpeg, "warped (xai)"


def _upright_slip(bgr: np.ndarray) -> np.ndarray:
    if bgr.shape[0] >= bgr.shape[1] * 0.95:
        return bgr
    cw = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    ccw = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cw if _horizontal_line_score(cw) >= _horizontal_line_score(ccw) else ccw


def _top_whiteness(bgr: np.ndarray, frac: float = 0.22) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height = max(1, int(gray.shape[0] * frac))
    top = gray[:height]
    return float((top > 180).mean())


def _flip_if_upside_down(bgr: np.ndarray) -> np.ndarray:
    """Till slips have a pale header; dense totals at the top means 180° off."""
    rotated = cv2.rotate(bgr, cv2.ROTATE_180)
    if _top_whiteness(rotated) > _top_whiteness(bgr) + 0.04:
        return rotated
    return bgr


def _horizontal_line_score(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    min_len = max(40, int(0.3 * bgr.shape[1]))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=min_len, maxLineGap=16)
    if lines is None:
        return 0.0
    score = 0.0
    for item in lines:
        x1, y1, x2, y2 = item[0]
        angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        if angle <= 15 or abs(angle - 180) <= 15:
            score += 1.0
    return score


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
        out = image
        status = "contrast-only"
    else:
        padded = _inflate_quad(full_quad, (h, w), 0.02)
        warped = _four_point_transform(image, padded)
        if warped is None:
            return None, "failed (bad warp)"
        if not _acceptable(warped):
            return None, "failed (poor crop)"
        out = warped
        status = "warped"

    out = _deskew(out)
    trimmed = _trim_to_content(out)
    if _acceptable(trimmed):
        out = trimmed
    out = _flip_if_upside_down(out)
    out = _light_contrast(out)

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
    paper_rect = _paper_min_rect(proc, edges)
    tilted = paper_rect is not None and _tilt_degrees(paper_rect) >= 1.5
    if not tilted:
        band = _text_band_quad(proc, edges)
        if band is not None:
            candidates.append(band)
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
    _, otsu = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masks: list[np.ndarray] = []
    raw_masks = [otsu]
    for percentile in (70.0, 75.0):
        thresh = float(np.percentile(light, percentile))
        raw_masks.append(((light >= thresh).astype(np.uint8)) * 255)
    for raw in raw_masks:
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


def _peel_background(values: np.ndarray, ink: np.ndarray, paper_level: float, ink_floor: float) -> tuple[int, int]:
    """Drop dark, ink-free margins only. Never eat a white receipt edge."""
    n = len(values)
    bg = (values < paper_level - 20.0) & (ink < ink_floor)
    lo, hi = 0, n - 1
    limit = int(0.4 * n)
    while lo < limit and bg[lo]:
        lo += 1
    while hi > n - 1 - limit and bg[hi]:
        hi -= 1
    return lo, hi


def _trim_to_content(warped: np.ndarray) -> np.ndarray:
    """Peel leftover table/background after warp. Keep white paper margins."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    h, w = gray.shape
    col_ink = edges.mean(axis=0)
    row_ink = edges.mean(axis=1)
    col_mean = gray.mean(axis=0)
    row_mean = gray.mean(axis=1)
    paper = float(np.percentile(col_mean, 65))
    ink_floor = max(float(np.median(col_ink)) * 0.6, 3.0)
    x0, x1 = _peel_background(col_mean, col_ink, paper, ink_floor)
    y0, y1 = _peel_background(row_mean, row_ink, paper, ink_floor)
    pad = 4
    x0 = max(0, x0 - pad)
    x1 = min(w, x1 + pad + 1)
    y0 = max(0, y0 - pad)
    y1 = min(h, y1 + pad + 1)
    if x1 - x0 < 0.25 * w or y1 - y0 < 0.25 * h:
        return warped
    return warped[y0:y1, x0:x1]


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


def _tilt_degrees(quad: np.ndarray) -> float:
    ordered = _order_points(quad)
    top = ordered[1] - ordered[0]
    angle = abs(float(np.degrees(np.arctan2(top[1], top[0]))))
    return min(angle, 180.0 - angle)


def _inflate_quad(quad: np.ndarray, shape: tuple[int, int], frac: float) -> np.ndarray:
    height, width = shape[:2]
    center = quad.mean(axis=0)
    out = center + (quad - center) * (1.0 + frac)
    out[:, 0] = np.clip(out[:, 0], 0, width - 1)
    out[:, 1] = np.clip(out[:, 1], 0, height - 1)
    return out.astype(np.float32)


def _deskew(bgr: np.ndarray) -> np.ndarray:
    """Rotate so printed text lines run horizontally."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    min_len = max(40, int(0.25 * bgr.shape[1]))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=min_len, maxLineGap=20)
    if lines is None:
        return bgr
    angles: list[float] = []
    for item in lines:
        x1, y1, x2, y2 = item[0]
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if abs(angle) <= 25:
            angles.append(angle)
    if len(angles) < 5:
        return bgr
    median = float(np.median(angles))
    if abs(median) < 0.4:
        return bgr
    h, w = bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, median, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    fill = tuple(int(v) for v in np.median(bgr.reshape(-1, 3), axis=0))
    return cv2.warpAffine(bgr, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=fill)


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
