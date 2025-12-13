"""
Refactored bar-chart shape extractor.

This module detects bar-like regions in a chart image and provides a set of
parameterized splitting strategies (color clustering in LAB, vertical projection,
edge-based separators, Hough vertical-line splitting) to split merged regions.

Public functions:
 - detect_and_refine(image_path, **params)
 - extract_bars_and_colors(image, **params)  # accepts path or ndarray
 - find_bars_by_color(image, target_rgb, **params)
 - determine_distinct_colors(bars_data, **params)
 - visualize_bars_on_blank(image_path, bars_data, output_path=None, ...)

The functions accept parameter knobs to adapt behavior per-dataset.
"""

from typing import List, Tuple, Dict, Optional, Union
import os
import json
import cv2
import numpy as np
from sklearn.cluster import KMeans


def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def get_dominant_color(img_rgb: np.ndarray, k: int = 1, sample_size: int = 2000) -> Tuple[Tuple[int, int, int], str]:
    """Return dominant RGB color and hex string for an RGB image crop.

    Parameters
    - img_rgb: ndarray HxWx3 (RGB)
    - k: number of clusters for KMeans (1 returns mean-like color)
    - sample_size: maximum pixels to sample for clustering
    """
    if img_rgb is None or img_rgb.size == 0:
        return (0, 0, 0), "#000000"

    pixels = img_rgb.reshape((-1, 3)).astype(np.uint8)
    if pixels.shape[0] > sample_size:
        idx = np.linspace(0, pixels.shape[0] - 1, sample_size).astype(int)
        pixels = pixels[idx]

    if k <= 1:
        mean = pixels.mean(axis=0).astype(int)
        rgb = (int(mean[0]), int(mean[1]), int(mean[2]))
        return rgb, _rgb_to_hex(rgb)

    try:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(pixels)
        center = kmeans.cluster_centers_[0].astype(int)
        rgb = (int(center[0]), int(center[1]), int(center[2]))
        return rgb, _rgb_to_hex(rgb)
    except Exception:
        mean = pixels.mean(axis=0).astype(int)
        rgb = (int(mean[0]), int(mean[1]), int(mean[2]))
        return rgb, _rgb_to_hex(rgb)


def _ensure_image(image_or_path: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(image_or_path, str):
        if not os.path.exists(image_or_path):
            raise FileNotFoundError(f"Image not found: {image_or_path}")
        img = cv2.imread(image_or_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_or_path}")
        return img
    elif isinstance(image_or_path, np.ndarray):
        return image_or_path
    else:
        raise TypeError("image_or_path must be path or ndarray")


def _abs_bbox(x: int, y: int, w: int, h: int) -> List[int]:
    return [int(x), int(y), int(x + w), int(y + h)]


def extract_bars_and_colors(image: Union[str, np.ndarray], *,
                            thresh_val: int = 200,
                            blur_ksize: int = 5,
                            morph_kernel: int = 3,
                            min_area_ratio: float = 0.005,
                            min_size: int = 6,
                            use_adaptive: bool = False,
                            dominant_frac_thresh: float = 0.95,
                            dominant_color_tol: float = 18.0,
                            bg_tol: float = 40.0) -> List[Dict]:
    """Detect candidate bar regions from an image.

    Parameters (important ones):
    - image: file path or BGR ndarray
    - thresh_val: fixed threshold used when `use_adaptive` is False
    - blur_ksize: gaussian blur kernel size (odd)
    - morph_kernel: morphological opening kernel size
    - min_area_ratio: minimal contour area relative to image area
    - min_size: minimal width/height in pixels

    Returns: list of dicts containing: type='bar', bbox=[x1,y1,x2,y2], width, height, area, color_rgb, color_hex
    """
    img_bgr = _ensure_image(image)
    img_rgb = _to_rgb(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    min_area = max(1.0, (h * w) * float(min_area_ratio))

    k = int(blur_ksize) if blur_ksize > 0 else 1
    k = k if k % 2 == 1 else k + 1
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    if use_adaptive:
        mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
    else:
        _, mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

    if morph_kernel and morph_kernel > 0:
        mk = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, mk)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results: List[Dict] = []

    # compute a global/background dominant color once to compare against
    global_bg_rgb, _ = get_dominant_color(img_rgb, k=1)

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < min_area or cw < min_size or ch < min_size:
            continue

        crop_rgb = img_rgb[y:y+ch, x:x+cw]

        # get dominant color for the crop
        dom_rgb, hexc = get_dominant_color(crop_rgb, k=1)

        # compute fraction of pixels within `dominant_color_tol` distance
        arr = crop_rgb.astype(int)
        tr, tg, tb = dom_rgb
        dist = np.sqrt((arr[:, :, 0] - tr) ** 2 + (arr[:, :, 1] - tg) ** 2 + (arr[:, :, 2] - tb) ** 2)
        match_frac = float((dist <= float(dominant_color_tol)).sum()) / float(dist.size)

        # ensure dominant color is not just the chart background
        bg_dist = float(np.linalg.norm(np.array(dom_rgb, dtype=float) - np.array(global_bg_rgb, dtype=float)))

        if match_frac >= float(dominant_frac_thresh) and bg_dist > float(bg_tol):
            # single-color bar (dominant color covers almost entire crop and isn't background)
            results.append({
                "type": "bar",
                "bbox": _abs_bbox(x, y, cw, ch),
                "width": int(cw),
                "height": int(ch),
                "area": float(area),
                "color_rgb": dom_rgb,
                "color_hex": hexc
            })
            continue

        # Otherwise, try to split the crop by color/structure to extract sub-bars
        subs: List[Dict] = []
        # try non-background connected components first
        try:
            subs.extend(_extract_non_bg_subboxes(crop_rgb, x, y, tuple(global_bg_rgb), bg_tol=bg_tol))
        except Exception:
            pass
        # try perceptual LAB clustering
        if not subs:
            try:
                subs.extend(_split_by_lab_clusters(crop_rgb, x, y, tuple(global_bg_rgb), top_k=4, bg_tol=bg_tol, merge_thresh=12.0, color_tol=dominant_color_tol))
            except Exception:
                pass
        # try projection-based split
        if not subs:
            try:
                subs.extend(_split_by_projection(crop_rgb, x, y, tuple(global_bg_rgb), bg_tol=bg_tol))
            except Exception:
                pass
        # try edge/hough separators as a last resort
        if not subs:
            try:
                subs.extend(_split_by_edge_separators(crop_rgb, x, y, tuple(global_bg_rgb)))
            except Exception:
                pass

        if subs:
            results.extend(subs)
        else:
            # fallback: keep the coarse box but tag with dominant color
            results.append({
                "type": "bar",
                "bbox": _abs_bbox(x, y, cw, ch),
                "width": int(cw),
                "height": int(ch),
                "area": float(area),
                "color_rgb": dom_rgb,
                "color_hex": hexc
            })

    results.sort(key=lambda b: b["bbox"][0])
    return results


def _mask_non_background(crop_rgb: np.ndarray, bg_rgb: Tuple[int, int, int], bg_tol: float) -> np.ndarray:
    arr = crop_rgb.astype(int)
    tr, tg, tb = bg_rgb
    dist = np.sqrt((arr[:, :, 0] - tr) ** 2 + (arr[:, :, 1] - tg) ** 2 + (arr[:, :, 2] - tb) ** 2)
    return (dist > float(bg_tol)).astype(np.uint8) * 255


def _extract_non_bg_subboxes(img_rgb: np.ndarray, abs_x: int, abs_y: int,
                             bg_rgb: Tuple[int, int, int], *, bg_tol: float = 40,
                             min_area_px: Optional[float] = None) -> List[Dict]:
    ph, pw = img_rgb.shape[:2]
    if ph == 0 or pw == 0:
        return []
    if min_area_px is None:
        H, W = img_rgb.shape[:2]
        min_area_px = max(1.0, (H * W) * 0.0005)

    mask = _mask_non_background(img_rgb, bg_rgb, bg_tol)
    mk = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, mk)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, mk)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    subboxes: List[Dict] = []
    for c in contours:
        x2, y2, w2, h2 = cv2.boundingRect(c)
        if w2 < 3 or h2 < 3:
            continue
        area2 = float(w2 * h2)
        if area2 < min_area_px:
            continue
        sx = abs_x + x2
        sy = abs_y + y2
        crop2 = img_rgb[sy: sy + h2, sx: sx + w2]
        rgb, hexc = get_dominant_color(crop2, k=1)
        subboxes.append({
            "type": "bar",
            "bbox": _abs_bbox(sx, sy, w2, h2),
            "width": int(w2),
            "height": int(h2),
            "area": area2,
            "color_rgb": rgb,
            "color_hex": hexc
        })
    return subboxes


def _split_by_lab_clusters(crop_rgb: np.ndarray, abs_x: int, abs_y: int, bg_rgb: Tuple[int, int, int], *,
                           top_k: int = 4, bg_tol: float = 28.0, merge_thresh: float = 12.0,
                           color_tol: float = 18.0, min_area_px: Optional[float] = None) -> List[Dict]:
    """Split a crop by perceptual (LAB-ab) clustering into color regions.

    Parameters similar to earlier heuristics; returns list of subboxes (absolute coords).
    """
    ph, pw = crop_rgb.shape[:2]
    if ph == 0 or pw == 0:
        return []
    if min_area_px is None:
        H, W = crop_rgb.shape[:2]
        min_area_px = max(1.0, (H * W) * 0.0002)

    try:
        lab = cv2.cvtColor(crop_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)
    except Exception:
        return []

    ab = lab[:, :, 1:3].reshape((-1, 2)).astype(float)
    # compute background AB
    bg_arr = np.uint8([[[bg_rgb[0], bg_rgb[1], bg_rgb[2]]]])
    try:
        bg_lab = cv2.cvtColor(bg_arr, cv2.COLOR_RGB2LAB)[0, 0]
        bg_ab = np.array(bg_lab[1:3], dtype=float)
    except Exception:
        bg_ab = np.array([0.0, 0.0])

    dist_bg = np.linalg.norm(ab - bg_ab.reshape((1, 2)), axis=1)
    idx = np.where(dist_bg > bg_tol)[0]
    if idx.size < 30:
        return []

    sample = ab[idx]
    k = min(top_k, max(2, int(len(sample) / 2000))) if len(sample) >= 200 else min(2, top_k)
    k = max(2, int(k))
    try:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(sample)
    except Exception:
        return []

    centers = km.cluster_centers_.astype(float)
    merged = []
    for c in centers:
        placed = False
        for grp in merged:
            if np.linalg.norm(grp['center'] - c) <= merge_thresh:
                grp['members'].append(c)
                grp['center'] = np.mean(grp['members'], axis=0)
                placed = True
                break
        if not placed:
            merged.append({'center': c.copy(), 'members': [c.copy()]})

    centers_merged = [g['center'] for g in merged]
    filtered = [c for c in centers_merged if np.linalg.norm(c - bg_ab) > max(6.0, bg_tol * 0.5)]
    if not filtered:
        return []

    lab_ab_map = lab[:, :, 1:3].astype(float)
    results: List[Dict] = []
    for center in filtered:
        dist_map = np.linalg.norm(lab_ab_map - center.reshape((1, 1, 2)), axis=2)
        mask = (dist_map <= float(color_tol)).astype(np.uint8) * 255
        mk = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, mk)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, mk)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x2, y2, w2, h2 = cv2.boundingRect(c)
            if w2 < 4 or h2 < 4:
                continue
            area2 = float(w2 * h2)
            if area2 < min_area_px:
                continue
            sx = abs_x + x2
            sy = abs_y + y2
            crop2 = crop_rgb[y2:y2 + h2, x2:x2 + w2]
            rgb, hexc = get_dominant_color(crop2, k=1)
            results.append({
                'type': 'bar', 'bbox': _abs_bbox(sx, sy, w2, h2), 'width': int(w2), 'height': int(h2),
                'area': area2, 'color_rgb': rgb, 'color_hex': hexc
            })
    return results


def _split_by_projection(crop_rgb: np.ndarray, abs_x: int, abs_y: int, bg_rgb: Tuple[int, int, int], *,
                         bg_tol: float = 30.0, min_seg_width: int = 4, min_area_px: Optional[float] = None) -> List[Dict]:
    ph, pw = crop_rgb.shape[:2]
    if ph == 0 or pw == 0:
        return []
    if min_area_px is None:
        min_area_px = max(1.0, (ph * pw) * 0.00012)

    try:
        gray = cv2.cvtColor(crop_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    except Exception:
        gray = np.mean(crop_rgb, axis=2).astype(np.uint8)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if gray.mean() < 80:
        mask_bin = 255 - mask_bin

    mk = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, mk)
    mk_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(ph * 0.05))))
    mask_bin = cv2.dilate(mask_bin, mk_vert, iterations=1)

    col_sum = np.sum(mask_bin > 0, axis=0)
    if col_sum.size == 0 or col_sum.max() == 0:
        return []

    kernel_size = min(15, max(3, pw // 10))
    kernel = np.ones(kernel_size) / float(kernel_size)
    smooth = np.convolve(col_sum.astype(float), kernel, mode='same')

    valley = max(1, int(smooth.max() * 0.10))
    content = smooth >= valley

    segments = []
    s = None
    for i, v in enumerate(content):
        if v and s is None:
            s = i
        elif not v and s is not None:
            segments.append((s, i - 1))
            s = None
    if s is not None:
        segments.append((s, len(content) - 1))

    boxes: List[Dict] = []
    for (s0, s1) in segments:
        seg_w = s1 - s0 + 1
        if seg_w >= min_seg_width:
            sx = abs_x + s0
            sw = seg_w
            sub = crop_rgb[:, s0:s1 + 1]
            subs = _extract_non_bg_subboxes(sub, sx, abs_y, bg_rgb, bg_tol=bg_tol, min_area_px=min_area_px)
            if subs:
                boxes.extend(subs)
                continue
            color_subs = _split_by_lab_clusters(sub, sx, abs_y, bg_rgb, top_k=3, bg_tol=bg_tol, merge_thresh=10, color_tol=18, min_area_px=min_area_px)
            if color_subs:
                boxes.extend(color_subs)
                continue
            rgb, hexc = get_dominant_color(sub, k=1)
            boxes.append({'type': 'bar', 'bbox': _abs_bbox(sx, abs_y, sw, ph), 'width': int(sw), 'height': int(ph), 'area': float(sw * ph), 'color_rgb': rgb, 'color_hex': hexc})
    return boxes


def _split_by_edge_separators(crop_rgb: np.ndarray, abs_x: int, abs_y: int, bg_rgb: Tuple[int, int, int], *,
                              edge_thresh_ratio: float = 0.12, min_sep_width: int = 2, min_area_px: Optional[float] = None) -> List[Dict]:
    ph, pw = crop_rgb.shape[:2]
    if ph == 0 or pw == 0:
        return []
    if min_area_px is None:
        min_area_px = max(1.0, (ph * pw) * 0.00012)

    try:
        gray = cv2.cvtColor(crop_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    except Exception:
        gray = np.mean(crop_rgb, axis=2).astype(np.uint8)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    col_grad = np.sum(np.abs(gx), axis=0)
    if col_grad.size == 0 or col_grad.max() == 0:
        return []

    kernel_size = min(15, max(3, pw // 15))
    kernel = np.ones(kernel_size) / float(kernel_size)
    smooth = np.convolve(col_grad.astype(float), kernel, mode='same')

    thresh = max(1.0, smooth.max() * float(edge_thresh_ratio))
    sep_mask = smooth >= thresh
    if not sep_mask.any():
        return []
    dil_k = max(1, int(min_sep_width))
    sep_mask = np.convolve(sep_mask.astype(int), np.ones(dil_k), mode='same') > 0

    runs = []
    rs = None
    for i, v in enumerate(sep_mask):
        if v and rs is None:
            rs = i
        elif not v and rs is not None:
            runs.append((rs, i - 1))
            rs = None
    if rs is not None:
        runs.append((rs, len(sep_mask) - 1))

    if not runs:
        return []

    cuts = [0] + [int((s0 + s1) / 2) for (s0, s1) in runs] + [pw]
    boxes: List[Dict] = []
    for i in range(len(cuts) - 1):
        s0 = cuts[i]
        s1 = cuts[i + 1]
        seg_w = s1 - s0
        if seg_w < 3:
            continue
        sx = abs_x + s0
        sub = crop_rgb[:, s0:s1]
        subs = _extract_non_bg_subboxes(sub, sx, abs_y, bg_rgb, bg_tol=30, min_area_px=min_area_px)
        if subs:
            boxes.extend(subs)
            continue
        color_subs = _split_by_lab_clusters(sub, sx, abs_y, bg_rgb, top_k=3, bg_tol=28, merge_thresh=10, color_tol=16, min_area_px=min_area_px)
        if color_subs:
            boxes.extend(color_subs)
            continue
        rgb, hexc = get_dominant_color(sub, k=1)
        boxes.append({'type': 'bar', 'bbox': _abs_bbox(sx, abs_y, seg_w, ph), 'width': int(seg_w), 'height': int(ph), 'area': float(seg_w * ph), 'color_rgb': rgb, 'color_hex': hexc})
    return boxes


def _to_json_serializable(obj):
    # convert numpy types to python native for json
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def step_coarse_detection(image_path: Union[str, np.ndarray], *, output_dir: Optional[str] = None, debug: bool = False, **kwargs) -> List[Dict]:
    img_bgr = _ensure_image(image_path)
    bars = extract_bars_and_colors(img_bgr, **kwargs)
    if debug:
        out_dir = output_dir or os.path.join(os.getcwd(), 'output_process')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'step1_bars.json'), 'w', encoding='utf-8') as f:
            json.dump(bars, f, default=_to_json_serializable, indent=2)
        try:
            visualize_bars_on_blank(img_bgr, bars, output_path=os.path.join(out_dir, 'step1_vis.png'))
        except Exception:
            pass
    return bars


def step_cluster_colors(bars: List[Dict], *, eps: float = 40.0, output_dir: Optional[str] = None, debug: bool = False) -> List[Dict]:
    clusters = determine_distinct_colors(bars, eps=eps)
    # produce simplified list: mean color per cluster (no member lists)
    simple = []
    for c in clusters:
        cid = c.get('cluster_id')
        col = c.get('color_rgb')
        try:
            color_rgb = tuple(int(x) for x in col)
        except Exception:
            color_rgb = tuple(c.get('color_rgb', (0, 0, 0)))
        simple.append({'cluster_id': cid, 'color_rgb': color_rgb})

    if debug:
        out_dir = output_dir or os.path.join(os.getcwd(), 'output_process')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'step3_clusters.json'), 'w', encoding='utf-8') as f:
            json.dump(simple, f, default=_to_json_serializable, indent=2)
        # also create a visual legend image of cluster colors
        try:
            legend_h = 40
            legend_w = max(200, 40 * len(simple))
            legend = np.full((legend_h, legend_w, 3), 255, dtype=np.uint8)
            for i, s in enumerate(simple):
                col = s.get('color_rgb', (0, 0, 0))
                x0 = i * 40
                x1 = x0 + 40
                cv2.rectangle(legend, (x0, 0), (x1, legend_h), (int(col[2]), int(col[1]), int(col[0])), -1)
            cv2.imwrite(os.path.join(out_dir, 'step3_clusters_legend.png'), legend)
        except Exception:
            pass

    return simple


def step_extract_colors_per_box(img_bgr: np.ndarray, coarse_boxes: List[Dict], *, top_k: int = 3, sample_size: int = 2000, output_dir: Optional[str] = None, debug: bool = False) -> List[Dict]:
    """For each coarse box, compute up to `top_k` dominant colors (RGB) using KMeans.

    Returns list of color candidates: each entry has 'color_rgb', 'color_hex', 'bbox' (parent box), 'parent_idx'.
    """
    img_rgb = _to_rgb(img_bgr)
    candidates: List[Dict] = []
    for idx, b in enumerate(coarse_boxes):
        bbox = b.get('bbox')
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        pixels = crop.reshape((-1, 3)).astype(np.float32)
        npix = pixels.shape[0]
        if npix == 0:
            continue
        # sample to keep KMeans fast
        if npix > sample_size:
            idxs = np.linspace(0, npix - 1, sample_size).astype(int)
            sample = pixels[idxs]
        else:
            sample = pixels
        k = min(int(top_k), max(1, int(len(sample) / 500)))
        k = max(1, k)
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(sample)
            centers = km.cluster_centers_.astype(int)
            labels = km.predict(pixels)
            for ci, center in enumerate(centers):
                count = int((labels == ci).sum())
                rgb = (int(center[0]), int(center[1]), int(center[2]))
                candidates.append({'type': 'color_candidate', 'parent_idx': idx, 'bbox': bbox, 'color_rgb': rgb, 'color_hex': _rgb_to_hex(rgb), 'count': count})
        except Exception:
            # fallback: single dominant
            rgb, hexc = get_dominant_color(crop, k=1)
            candidates.append({'type': 'color_candidate', 'parent_idx': idx, 'bbox': bbox, 'color_rgb': rgb, 'color_hex': hexc, 'count': int(npix)})

    # filter out candidates similar to global background
    try:
        img_rgb_full = _to_rgb(img_bgr)
        global_bg_rgb, _ = get_dominant_color(img_rgb_full, k=3)
        print("Global background RGB:", global_bg_rgb)
        bg_tr, bg_tg, bg_tb = global_bg_rgb
        filtered = []
        for c in candidates:
            cr, cg, cb = c['color_rgb']
            dist = np.sqrt((int(cr) - int(bg_tr)) ** 2 + (int(cg) - int(bg_tg)) ** 2 + (int(cb) - int(bg_tb)) ** 2)
            if dist > float(40.0):
                filtered.append(c)
        candidates = filtered
    except Exception:
        pass

    if debug:
        out_dir = output_dir or os.path.join(os.getcwd(), 'output_process')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'step2_color_candidates.json'), 'w', encoding='utf-8') as f:
            json.dump(candidates, f, default=_to_json_serializable, indent=2)
    return candidates


def step_refine_by_color(img_bgr: np.ndarray, bars: List[Dict], clusters: List[Dict], *, small_ratio: float = 0.6, tolerance: float = 40.0, output_dir: Optional[str] = None, debug: bool = False) -> List[Dict]:
    all_smalls: List[Dict] = []
    if not clusters:
        return all_smalls
    areas = [b['area'] for b in bars] if bars else [0]
    median_area = float(np.median(areas)) if areas else 0.0
    for cl in clusters:
        target = cl['color_rgb']
        found = find_bars_by_color(img_bgr, target, tolerance=tolerance)
        smalls = [f for f in found if f.get('area', 0) < median_area * small_ratio]
        all_smalls.extend(smalls)
    if debug:
        out_dir = output_dir or os.path.join(os.getcwd(), 'output_process')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'step4_smalls.json'), 'w', encoding='utf-8') as f:
            json.dump(all_smalls, f, default=_to_json_serializable, indent=2)
        # visualize the small regions for debugging
        try:
            vis_boxes = []
            for s in all_smalls:
                vis_boxes.append({'bbox': s.get('bbox'), 'color_rgb': tuple(s.get('color_rgb', (0, 0, 0)))})
            visualize_bars_on_blank(img_bgr, vis_boxes, output_path=os.path.join(out_dir, 'step4_smalls.png'))
        except Exception:
            pass
    return all_smalls


def step_merge_smalls(bars: List[Dict], all_smalls: List[Dict], *, output_dir: Optional[str] = None, debug: bool = False) -> List[Dict]:
    def iou(boxA, boxB):
        ax1, ay1, ax2, ay2 = boxA
        bx1, by1, bx2, by2 = boxB
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        areaA = (ax2 - ax1) * (ay2 - ay1)
        areaB = (bx2 - bx1) * (by2 - by1)
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0.0

    combined = list(bars)
    for s in all_smalls:
        s_box = s['bbox']
        s_area = s.get('area', (s['bbox'][2] - s['bbox'][0]) * (s['bbox'][3] - s['bbox'][1]))
        overlaps = []
        for idx, ex in enumerate(combined):
            if iou(s_box, ex['bbox']) > 0:
                overlaps.append((idx, ex))
        if not overlaps:
            combined.append(s)
        else:
            replaced = False
            for idx, ex in overlaps:
                ex_area = ex.get('area', (ex['bbox'][2] - ex['bbox'][0]) * (ex['bbox'][3] - ex['bbox'][1]))
                if ex_area >= s_area:
                    replaced = True
                    break
            if replaced:
                continue
            for idx, _ in sorted(overlaps, key=lambda x: x[0], reverse=True):
                combined.pop(idx)
            combined.append(s)

    combined.sort(key=lambda b: b['bbox'][0])
    if debug:
        out_dir = output_dir or os.path.join(os.getcwd(), 'output_process')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'step5_merged.json'), 'w', encoding='utf-8') as f:
            json.dump(combined, f, default=_to_json_serializable, indent=2)
    return combined


def step_find_legends(img_bgr: np.ndarray, bars: List[Dict], clusters: List[Dict], *,
                      legend_ratio: float = 0.4, tolerance: float = 30.0,
                      output_dir: Optional[str] = None, debug: bool = False) -> List[Dict]:
    """Find small legend markers matching cluster colors.

    Looks for small regions matching each cluster color that are noticeably
    smaller than typical bar areas. Writes `step5_legends.json` and
    `step5_legends.png` when `debug=True`.
    """
    legends: List[Dict] = []
    if not clusters:
        return legends

    # derive size thresholds from provided bars
    areas = [b.get('area', 0.0) for b in bars] if bars else []
    widths = [b.get('width', 0) for b in bars] if bars else []
    median_area = float(np.median(areas)) if areas else 0.0
    median_width = float(np.median(widths)) if widths else 0.0

    # detection tolerance for legends tends to be a bit tighter
    tol = float(tolerance) * 0.6

    # compute a smaller min-area ratio to allow detecting tiny legend markers
    H, W = img_bgr.shape[:2]
    if H > 0 and W > 0:
        # prefer legends that are much smaller than the median bar area
        if median_area > 0:
            desired_min_px = max(1.0, median_area * 0.01)
        else:
            desired_min_px = 1.0
        min_area_ratio_for_legends = float(desired_min_px) / float(H * W)
    else:
        min_area_ratio_for_legends = None

    # (no brightness/background filtering — allow dark/black candidates)

    # collect candidates per cluster first
    candidates_by_cluster = []  # list of tuples (cluster_id, cluster_color, [candidates])
    for cl in clusters:
        cid = cl.get('cluster_id')
        target = tuple(cl.get('color_rgb', (0, 0, 0)))
        if min_area_ratio_for_legends:
            found = find_bars_by_color(img_bgr, target, tolerance=tol, min_area_ratio=min_area_ratio_for_legends)
        else:
            found = find_bars_by_color(img_bgr, target, tolerance=tol)
        cand_list = []
        for f in found:
            cand_rgb = tuple(f.get('color_rgb', (0, 0, 0)))
            area = float(f.get('area', 0.0))
            w = float(f.get('width', f.get('bbox')[2] - f.get('bbox')[0]))
            if median_area > 0 and area < max(1.0, median_area * float(legend_ratio)):
                if median_width > 0 and w > max(1.0, median_width * 0.6):
                    continue
                bbox = tuple(map(int, f.get('bbox', [0, 0, 0, 0])))
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                cand_list.append({'bbox': list(bbox), 'cx': cx, 'cy': cy, 'area': area, 'width': w, 'color_rgb': cand_rgb, 'color_hex': f.get('color_hex')})
        if cand_list:
            candidates_by_cluster.append((cid, tuple(target), cand_list))
    # debug: write all candidate lists for inspection and a visualization
    if debug:
        out_dir = output_dir or os.path.join(os.getcwd(), 'output_process')
        os.makedirs(out_dir, exist_ok=True)
        serializable = []
        for cid, target, lst in candidates_by_cluster:
            serializable.append({
                'cluster_id': cid,
                'cluster_color': list(target) if isinstance(target, (list, tuple)) else target,
                'candidates': lst
            })
        try:
            with open(os.path.join(out_dir, 'step5_can_legends.json'), 'w', encoding='utf-8') as f:
                json.dump(serializable, f, default=_to_json_serializable, indent=2)
        except Exception:
            pass
        try:
            vis_boxes = []
            for _, _, lst in candidates_by_cluster:
                for p in lst:
                    vis_boxes.append({'bbox': p.get('bbox'), 'color_rgb': tuple(p.get('color_rgb', (0,0,0)))})
            visualize_bars_on_blank(img_bgr, vis_boxes, output_path=os.path.join(out_dir, 'step5_can_legends.png'))
        except Exception:
            pass
    
    # select one candidate per cluster such that their x-positions are as close together as possible
    if not candidates_by_cluster:
        return legends
    # compute total combination size
    import math, itertools
    sizes = [len(c[2]) for c in candidates_by_cluster]
    total_combinations = 1
    for s in sizes:
        total_combinations *= s

    chosen = []
    if total_combinations <= 2000:
        # exhaustive search: minimize maximal pairwise Euclidean distance between selected centers
        best_span = float('inf')
        best_combo = None
        pools = [c[2] for c in candidates_by_cluster]
        for combo in itertools.product(*pools):
            pts = [(p['cx'], p.get('cy', 0.0)) for p in combo]
            maxd = 0.0
            for i in range(len(pts)):
                xi, yi = pts[i]
                for j in range(i + 1, len(pts)):
                    xj, yj = pts[j]
                    d = math.hypot(xi - xj, yi - yj)
                    if d > maxd:
                        maxd = d
                        if maxd >= best_span:
                            break
                if maxd >= best_span:
                    break
            if maxd < best_span:
                best_span = maxd
                best_combo = combo
        if best_combo is not None:
            for sel in best_combo:
                legends.append({
                    'type': 'legend', 'bbox': sel['bbox'], 'width': int(sel['width']), 'height': int(round(sel['area'] / max(1.0, sel['width']))),
                    'area': sel['area'], 'color_rgb': tuple(sel['color_rgb']), 'color_hex': sel.get('color_hex')
                })
    else:
        # greedy: choose candidate closest to global median x across all candidates
        all_pts = [(p['cx'], p.get('cy', 0.0)) for _, _, lst in candidates_by_cluster for p in lst]
        if all_pts:
            median_x = float(np.median([pt[0] for pt in all_pts]))
            median_y = float(np.median([pt[1] for pt in all_pts]))
        else:
            median_x = median_y = None
        for _, _, lst in candidates_by_cluster:
            if median_x is None:
                sel = lst[0]
            else:
                sel = min(lst, key=lambda p: math.hypot(p['cx'] - median_x, p.get('cy', 0.0) - median_y))
            legends.append({
                'type': 'legend', 'bbox': sel['bbox'], 'width': int(sel['width']), 'height': int(round(sel['area'] / max(1.0, sel['width']))),
                'area': sel['area'], 'color_rgb': tuple(sel['color_rgb']), 'color_hex': sel.get('color_hex')
            })

    if debug:
        out_dir = output_dir or os.path.join(os.getcwd(), 'output_process')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'step5_legends.json'), 'w', encoding='utf-8') as f:
            json.dump(legends, f, default=_to_json_serializable, indent=2)
        try:
            vis_boxes = []
            for s in legends:
                vis_boxes.append({'bbox': s.get('bbox'), 'color_rgb': tuple(s.get('color_rgb', (0, 0, 0)))} )
            visualize_bars_on_blank(img_bgr, vis_boxes, output_path=os.path.join(out_dir, 'step5_legends.png'))
        except Exception:
            pass

    return legends


def step_assign_classes(combined: List[Dict], clusters: List[Dict], *, output_dir: Optional[str] = None, debug: bool = False) -> List[Dict]:
    class_means: List[Tuple[int, int, int]] = []
    for cl in clusters:
        class_means.append(tuple(cl.get('color_rgb', (0, 0, 0))))

    if class_means:
        cm = np.array(class_means, dtype=float)
        for b in combined:
            rgb = tuple(b.get('color_rgb', (0, 0, 0)))
            dists = np.linalg.norm(cm - np.array(rgb, dtype=float), axis=1)
            idx = int(np.argmin(dists))
            b['color_class'] = f'class_{idx + 1}'
            b['cluster_idx'] = idx
            b['cluster_color'] = tuple(class_means[idx])
    else:
        for b in combined:
            b['color_class'] = None
            b['cluster_idx'] = None
            b['cluster_color'] = None

    if debug:
        out_dir = output_dir or os.path.join(os.getcwd(), 'output_process')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'step6_classified.json'), 'w', encoding='utf-8') as f:
            json.dump(combined, f, default=_to_json_serializable, indent=2)
    return combined



def find_bars_by_color(image: Union[str, np.ndarray], target_rgb: Tuple[int, int, int], *, tolerance: float = 40.0, morph_kernel: int = 3, min_area_ratio: Optional[float] = None) -> List[Dict]:
    img_bgr = _ensure_image(image)
    img_rgb = _to_rgb(img_bgr)
    arr = img_rgb.astype(int)
    tr, tg, tb = target_rgb
    dist = np.sqrt((arr[:, :, 0] - tr) ** 2 + (arr[:, :, 1] - tg) ** 2 + (arr[:, :, 2] - tb) ** 2)
    mask = (dist <= float(tolerance)).astype(np.uint8) * 255
    if morph_kernel and morph_kernel > 0:
        mk = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, mk)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img_rgb.shape[:2]
    # allow caller to request a smaller minimum area via keyword arg (backwards compatible)
    # if `min_area_ratio` isn't provided, fall back to original heuristic
    if min_area_ratio is None:
        min_area_px = max(1.0, (H * W) * 0.0008)
    else:
        try:
            min_area_px = max(1.0, (H * W) * float(min_area_ratio))
        except Exception:
            min_area_px = max(1.0, (H * W) * 0.0008)
    results: List[Dict] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_px:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < 3 or ch < 3:
            continue
        crop = img_rgb[y:y + ch, x:x + cw]
        rgb, hexc = get_dominant_color(crop, k=1)
        results.append({'type': 'region', 'bbox': _abs_bbox(x, y, cw, ch), 'width': int(cw), 'height': int(ch), 'area': float(area), 'color_rgb': rgb, 'color_hex': hexc})
    results.sort(key=lambda b: b['bbox'][0])
    return results


def determine_distinct_colors(bars_data: List[Dict], *, eps: float = 40.0, min_samples: int = 1) -> List[Dict]:
    from sklearn.cluster import DBSCAN
    if not bars_data:
        return []
    colors = [tuple(b.get('color_rgb', (0, 0, 0))) for b in bars_data]
    X = np.array(colors)
    db = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(X)
    labels = db.labels_
    clusters = {}
    for i, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append({'index': i, 'color_rgb': tuple(map(int, X[i].tolist())), 'bbox': bars_data[i].get('bbox')})
    results: List[Dict] = []
    for lbl, members in clusters.items():
        if lbl == -1:
            for m in members:
                results.append({'cluster_id': f'noise_{m["index"]}', 'color_rgb': m['color_rgb'], 'members': [m]})
        else:
            mean_color = tuple(map(int, np.mean([m['color_rgb'] for m in members], axis=0).tolist()))
            results.append({'cluster_id': int(lbl), 'color_rgb': mean_color, 'members': members})
    results.sort(key=lambda c: c['members'][0]['bbox'][0] if c['members'][0].get('bbox') else 0)
    return results


def visualize_bars_on_blank(image_path: Union[str, np.ndarray], bars_data: List[Dict], *, output_path: Optional[str] = None, background_color: Tuple[int, int, int] = (255, 255, 255), thickness: int = 2, show_index: bool = True) -> str:
    if isinstance(image_path, np.ndarray):
        src = image_path
    else:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        src = cv2.imread(image_path)
        if src is None:
            raise ValueError(f"Could not read image: {image_path}")
    h, w = src.shape[:2]
    blank = np.full((h, w, 3), background_color, dtype=np.uint8)
    for idx, bar in enumerate(bars_data):
        bbox = bar.get('bbox')
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        rgb = bar.get('color_rgb', (0, 0, 0))
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.rectangle(blank, (x1, y1), (x2, y2), bgr, thickness)
        if show_index:
            label = str(idx + 1)
            brightness = (int(rgb[0]) * 0.299 + int(rgb[1]) * 0.587 + int(rgb[2]) * 0.114)
            text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
            text_bgr = (text_color[2], text_color[1], text_color[0])
            cv2.putText(blank, label, (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_bgr, 1, cv2.LINE_AA)
    if output_path is None:
        out_dir = os.path.join(os.getcwd(), 'output_process')
        os.makedirs(out_dir, exist_ok=True)
        base = 'visualization'
        output_path = os.path.join(out_dir, f"{base}.png")
    else:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, blank)
    return output_path


def detect_and_refine(image_path: Union[str, np.ndarray], *,
                      refine_color: bool = True,
                      tolerance: float = 20.0,
                      visual: bool = True,
                      output_dir: Optional[str] = None,
                      small_ratio: float = 0.8,
                      aggressive_top_k: int = 5,
                      debug: bool = False,
                      # forwarded params for extract_bars_and_colors
                      dominant_frac_thresh: float = 0.95,
                      dominant_color_tol: float = 18.0,
                      bg_tol: float = 40.0) -> Tuple[List[Dict], Optional[str]]:
    """Full detection pipeline.

    Steps:
    1. coarse detection via `extract_bars_and_colors`
    2. cluster distinct colors via `determine_distinct_colors`
    3. per-cluster color-find for small boxes
    4. merge smalls into main set preferring larger boxes on overlap
    5. assign `color_class` and `cluster_idx` fields

    Returns: (combined_bars, visualization_path_or_None)
    """
    img_bgr = _ensure_image(image_path)
    img_rgb = _to_rgb(img_bgr)
    h, w = img_rgb.shape[:2]

    # Step 1: coarse detection
    bars = step_coarse_detection(img_bgr, output_dir=output_dir, debug=debug,
                                 dominant_frac_thresh=float(dominant_frac_thresh),
                                 dominant_color_tol=float(dominant_color_tol),
                                 bg_tol=float(bg_tol))
    if not bars:
        return [], None

    # Step 2: extract colors inside each coarse box
    color_candidates = step_extract_colors_per_box(img_bgr, bars, top_k=aggressive_top_k, output_dir=output_dir, debug=debug)

    # Step 3: cluster the extracted colors
    clusters = step_cluster_colors(color_candidates, eps=tolerance, output_dir=output_dir, debug=debug)
    # Step 4: refine by color (find smalls)
    bars = step_refine_by_color(img_bgr, bars, clusters, small_ratio=small_ratio, tolerance=tolerance, output_dir=output_dir, debug=debug) if refine_color else []

    # Step 5: find legends (small markers of cluster colors)
    legends = step_find_legends(img_bgr, bars, clusters, legend_ratio=0.4, tolerance=20, output_dir=output_dir, debug=debug) if refine_color else []

    # Step 6: assign classes
    combined = step_assign_classes(bars, clusters, output_dir=output_dir, debug=debug)

    vis_path = None
    if visual:
        out_dir = output_dir or os.path.join(os.getcwd(), 'output_image')
        os.makedirs(out_dir, exist_ok=True)
        vis_path = os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0] + '_visualization.png')
        vis_items = list(combined) + list(legends)
        visualize_bars_on_blank(img_bgr, vis_items, output_path=vis_path)

    return combined, vis_path