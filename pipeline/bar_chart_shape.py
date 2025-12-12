import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

def get_dominant_color(image_crop, k=1):
    if image_crop.size == 0:
        return (0, 0, 0), "#000000"

    pixels = image_crop.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    
    r, g, b = dominant_color
    hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
    
    return (int(r), int(g), int(b)), hex_color

def extract_bars_and_colors(image_path):
    """
    Simple extraction: detect bar-like contours and return their bounding boxes
    and dominant colors. No tuning parameters — sensible defaults are used.

    Returns: list of dicts with keys: type, bbox, color_rgb, color_hex, width, height, area
    """
    # sensible defaults
    thresh_val = 200
    min_area_ratio = 0.005
    min_size = 5
    blur_ksize = 5
    use_adaptive = False
    morph_kernel_size = 3

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy ảnh tại {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        return []

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Determine background color as the dominant color of the whole image
    try:
        bg_rgb, bg_hex = get_dominant_color(image_rgb, k=1)
    except Exception:
        bg_rgb, bg_hex = (255, 255, 255), "#ffffff"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # prepare mask
    k = int(blur_ksize) if blur_ksize and blur_ksize > 0 else 1
    k = k if k % 2 == 1 else k + 1
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    if use_adaptive:
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
    else:
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

    if morph_kernel_size and morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bars_data = []

    height, width = gray.shape
    min_area = (height * width) * float(min_area_ratio)

    rects = [cv2.boundingRect(cnt) for cnt in contours]
    widths = [r[2] for r in rects if r[2] > 0]
    median_width = int(np.median(widths)) if widths else 0

    # helper: find vertical content segments inside a binary crop
    def _vertical_segments_from_mask(crop_mask, abs_x):
        col_sum = np.sum(crop_mask > 0, axis=0)
        if col_sum.size == 0 or col_sum.max() == 0:
            return []
        valley_thresh = max(1, int(col_sum.max() * 0.10))
        content_mask = col_sum > valley_thresh
        segments = []
        seg_start = None
        for i, v in enumerate(content_mask):
            if v and seg_start is None:
                seg_start = i
            elif not v and seg_start is not None:
                segments.append((seg_start, i - 1))
                seg_start = None
        if seg_start is not None:
            segments.append((seg_start, len(content_mask) - 1))

        min_seg_width = max(3, int(median_width * 0.25))
        boxes = []
        for (s0, s1) in segments:
            seg_w = s1 - s0 + 1
            if seg_w >= min_seg_width:
                boxes.append((abs_x + s0, seg_w))
        return boxes

    # helper: extract subboxes within a crop that are not background-like
    def _extract_non_bg_subboxes(crop, abs_x, abs_y, bg_rgb, bg_tol=40, min_area_px= max(1.0, (height * width) * 0.0005)):
        # returns list of dicts for subboxes (absolute coords)
        ph, pw = crop.shape[:2]
        if ph == 0 or pw == 0:
            return []
        pixels = crop.reshape((-1, 3))
        total = pixels.shape[0]
        if total == 0:
            return []
        arr = np.asarray(pixels, dtype=int)
        tr, tg, tb = bg_rgb
        dist = np.sqrt((arr[:,0]-tr)**2 + (arr[:,1]-tg)**2 + (arr[:,2]-tb)**2)
        bg_ratio = float(np.sum(dist <= bg_tol)) / float(total)
        if bg_ratio <= 0.7:
            return []

        arr_img = np.asarray(crop, dtype=int)
        dist_map = np.sqrt((arr_img[:,:,0]-tr)**2 + (arr_img[:,:,1]-tg)**2 + (arr_img[:,:,2]-tb)**2)
        mask = (dist_map > bg_tol).astype(np.uint8) * 255
        mk = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, mk)
        contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for c2 in contours2:
            x2, y2, w2, h2 = cv2.boundingRect(c2)
            if w2 < 3 or h2 < 3:
                continue
            area2 = float(w2 * h2)
            if area2 < min_area_px:
                continue
            sx_abs = abs_x + x2
            sy_abs = abs_y + y2
            crop2 = image_rgb[sy_abs: sy_abs + h2, sx_abs: sx_abs + w2]
            rgb2, hex2 = get_dominant_color(crop2)
            results.append({
                "type": "bar",
                "bbox": [int(sx_abs), int(sy_abs), int(sx_abs + w2), int(sy_abs + h2)],
                "color_rgb": rgb2,
                "color_hex": hex2,
                "width": int(w2),
                "height": int(h2),
                "area": area2
            })
        return results

    # helper: color-based splitting inside a crop using KMeans on non-background pixels
    def _color_split_box(crop_rgb, abs_x, abs_y, image_rgb, bg_rgb, bg_tol=40, min_area_px=max(1.0, (height * width) * 0.0005), k_clusters=4):
        ph, pw = crop_rgb.shape[:2]
        if ph == 0 or pw == 0:
            return []

        pixels = crop_rgb.reshape((-1, 3)).astype(int)
        tr, tg, tb = bg_rgb
        dist = np.sqrt((pixels[:, 0] - tr) ** 2 + (pixels[:, 1] - tg) ** 2 + (pixels[:, 2] - tb) ** 2)
        mask_idx = np.where(dist > bg_tol)[0]
        if mask_idx.size < 50:
            return []

        sample = pixels[mask_idx]
        # choose k reasonably
        k = min(k_clusters, max(2, int(len(sample) / 5000))) if len(sample) >= 500 else min(2, k_clusters)
        try:
            k = max(2, int(k))
        except Exception:
            k = 2

        try:
            kmeans = KMeans(n_clusters=k, n_init=10).fit(sample)
        except Exception:
            return []

        # assign cluster labels for all pixels in crop
        try:
            all_labels = kmeans.predict(pixels)
        except Exception:
            return []

        label_map = all_labels.reshape((ph, pw))
        results = []
        mk = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for ci in range(kmeans.n_clusters):
            mask = (label_map == ci).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, mk)
            contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c2 in contours2:
                x2, y2, w2, h2 = cv2.boundingRect(c2)
                if w2 < 3 or h2 < 3:
                    continue
                area2 = float(w2 * h2)
                if area2 < min_area_px:
                    continue
                sx_abs = abs_x + x2
                sy_abs = abs_y + y2
                crop2 = image_rgb[sy_abs: sy_abs + h2, sx_abs: sx_abs + w2]
                rgb2, hex2 = get_dominant_color(crop2)
                results.append({
                    "type": "bar",
                    "bbox": [int(sx_abs), int(sy_abs), int(sx_abs + w2), int(sy_abs + h2)],
                    "color_rgb": rgb2,
                    "color_hex": hex2,
                    "width": int(w2),
                    "height": int(h2),
                    "area": area2
                })
        return results

    # iterate contours and produce bars
    for cnt, (x, y, w, h) in zip(contours, rects):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if w < min_size or h < min_size:
            continue

        split_boxes = []
        try_split = (median_width > 0 and w > max(median_width * 1.2, median_width + 10))

        if try_split:
            crop_mask = thresh[y:y+h, x:x+w]
            segs = _vertical_segments_from_mask(crop_mask, x)
            # segs yields (abs_x, seg_w) per content segment
            if len(segs) >= 2:
                split_boxes = [(sx, y, sw, h) for (sx, sw) in segs]

        if split_boxes:
            for sx, sy, sw, sh in split_boxes:
                pad = 2
                if sw > 2*pad and sh > 2*pad:
                    bar_crop = image_rgb[sy+pad: sy+sh-pad, sx+pad: sx+sw-pad]
                else:
                    bar_crop = image_rgb[sy: sy+sh, sx: sx+sw]

                subs = _extract_non_bg_subboxes(bar_crop, sx, sy, bg_rgb)
                if subs:
                    bars_data.extend(subs)
                    continue

                # if vertical segmentation failed to find inner colored parts,
                # try color-based clustering inside this sub-crop
                color_subs = _color_split_box(bar_crop, sx, sy, image_rgb, bg_rgb)
                if color_subs:
                    bars_data.extend(color_subs)
                    continue

                rgb, hex_code = get_dominant_color(bar_crop)
                bars_data.append({
                    "type": "bar",
                    "bbox": [int(sx), int(sy), int(sx + sw), int(sy + sh)],
                    "color_rgb": rgb,
                    "color_hex": hex_code,
                    "width": int(sw),
                    "height": int(sh),
                    "area": float(sw * sh)
                })
        else:
            pad = 2
            if w > 2*pad and h > 2*pad:
                bar_crop = image_rgb[y+pad : y+h-pad, x+pad : x+w-pad]
            else:
                bar_crop = image_rgb[y : y+h, x : x+w]

            subs = _extract_non_bg_subboxes(bar_crop, x, y, bg_rgb)
            if subs:
                bars_data.extend(subs)
                continue

            # try color-based splitting for wide/merged regions
            color_subs = _color_split_box(bar_crop, x, y, image_rgb, bg_rgb)
            if color_subs:
                bars_data.extend(color_subs)
                continue

            rgb, hex_code = get_dominant_color(bar_crop)
            bars_data.append({
                "type": "bar",
                "bbox": [x, y, x + w, y + h],
                "color_rgb": rgb,
                "color_hex": hex_code,
                "width": w,
                "height": h,
                "area": area
            })

    bars_data.sort(key=lambda b: b["bbox"][0])
    return bars_data


def get_main_colors_from_boxes(bars_data, top_n=3):
    """
    Given `bars_data` (list of dicts with `color_hex`), return the top N most
    frequent colors across the detected boxes. Returns a list of dicts:
    {"color_hex": str, "color_rgb": (r,g,b), "count": int}
    """
    from collections import Counter

    hex_list = [b.get("color_hex", "#000000").lower() for b in bars_data]
    counter = Counter(hex_list)
    most = counter.most_common(top_n)
    results = []
    for hex_code, cnt in most:
        # convert hex to rgb tuple
        try:
            r = int(hex_code[1:3], 16)
            g = int(hex_code[3:5], 16)
            b = int(hex_code[5:7], 16)
            rgb = (r, g, b)
        except Exception:
            rgb = (0, 0, 0)
        results.append({"color_hex": hex_code, "color_rgb": rgb, "count": cnt})
    return results


def find_bars_by_color(image_path, target_rgb, tolerance=40, morph_kernel_size=3):
    """
    Find regions in `image_path` whose color is within `tolerance` (Euclidean distance)
    of `target_rgb` (r,g,b). Returns a list of bar-like dicts similar to
    `extract_bars_and_colors`.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy ảnh tại {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        return []

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # compute distance map
    arr = image_rgb.astype(int)
    tr, tg, tb = target_rgb
    dist = np.sqrt((arr[:,:,0]-tr)**2 + (arr[:,:,1]-tg)**2 + (arr[:,:,2]-tb)**2)
    mask = (dist <= float(tolerance)).astype(np.uint8) * 255

    if morph_kernel_size and morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bars_data = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 3 or h < 3:
            continue
        pad = 1
        h_img, w_img = image_rgb.shape[:2]
        y1 = max(0, y+pad); y2 = min(h_img, y+h-pad)
        x1 = max(0, x+pad); x2 = min(w_img, x+w-pad)
        crop = image_rgb[y1:y2, x1:x2]
        rgb, hex_code = get_dominant_color(crop)
        bars_data.append({
            "type": "region",
            "bbox": [x, y, x + w, y + h],
            "color_rgb": rgb,
            "color_hex": hex_code,
            "width": w,
            "height": h,
            "area": cv2.contourArea(cnt)
        })

    bars_data.sort(key=lambda b: b["bbox"][0])
    return bars_data


def determine_distinct_colors(bars_data, eps=40, min_samples=1):
    """
    Cluster the detected box colors to determine how many distinct chart colors exist.

    - `eps` controls color distance (Euclidean in RGB space) for clustering.
    - Returns a list of clusters: each is {"cluster_id": int_or_str, "color_rgb": (r,g,b), "members": [ {index, color_rgb, bbox} ]}
    """
    from sklearn.cluster import DBSCAN
    if not bars_data:
        return []

    colors = [tuple(b.get("color_rgb", (0, 0, 0))) for b in bars_data]
    X = np.array(colors)

    db = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(X)
    labels = db.labels_

    clusters = {}
    for i, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append({
            "index": i,
            "color_rgb": tuple(map(int, X[i].tolist())),
            "bbox": bars_data[i].get("bbox")
        })

    results = []
    for lbl, members in clusters.items():
        if lbl == -1:
            # treat each noise member as its own cluster
            for m in members:
                results.append({"cluster_id": f"noise_{m['index']}", "color_rgb": m["color_rgb"], "members": [m]})
        else:
            mean_color = tuple(map(int, np.mean([m["color_rgb"] for m in members], axis=0).tolist()))
            results.append({"cluster_id": int(lbl), "color_rgb": mean_color, "members": members})

    # sort clusters by x position of first member for consistent order
    results.sort(key=lambda c: c["members"][0]["bbox"][0] if c["members"][0].get("bbox") else 0)
    return results

def visualize_bars_on_blank(image_path, bars_data, output_path=None, background_color=(255,255,255), thickness=2, show_index=True):
    """
    Create a blank image with the same size as `image_path` and draw the bounding boxes
    from `bars_data` using their RGB colors. Saves the image to `output_path` if provided
    or to `output_image/bars_visualization.png` by default. Returns the output path.

    - `bars_data` is expected to be a list of dicts with keys: "bbox" and "color_rgb".
    """
    import cv2
    import numpy as np
    import os

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy ảnh tại {image_path}")

    src = cv2.imread(image_path)
    if src is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    h, w = src.shape[:2]
    blank = np.full((h, w, 3), background_color, dtype=np.uint8)

    for idx, bar in enumerate(bars_data):
        bbox = bar.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)

        rgb = bar.get("color_rgb", (0, 0, 0))
        # OpenCV uses BGR ordering for colors
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))

        cv2.rectangle(blank, (x1, y1), (x2, y2), bgr, thickness)

        if show_index:
            label = str(idx + 1)
            # Choose text color as white or black depending on brightness
            brightness = (int(rgb[0]) * 0.299 + int(rgb[1]) * 0.587 + int(rgb[2]) * 0.114)
            text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
            # text_color in BGR
            text_bgr = (text_color[2], text_color[1], text_color[0])
            cv2.putText(blank, label, (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_bgr, 1, cv2.LINE_AA)

    if output_path is None:
        output_dir = os.path.join(os.getcwd(), "output_process")
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base}_visualization.png")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cv2.imwrite(output_path, blank)
    return output_path


def detect_and_refine(image_path, *, refine_color=True, tolerance=40, visual=True, output_dir=None, small_ratio=0.6):
    """
    Run full detection pipeline: extract bars, cluster colors, find small boxes by color,
    merge boxes (prefer larger boxes on overlap), and optionally visualize final result.

    Returns: (combined_bars, visualization_path_or_None)
    - combined_bars: list of dicts with keys: bbox, color_rgb, color_hex, width, height, area
    - visualization_path_or_None: path to saved visualization image if visual=True, else None
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy ảnh tại {image_path}")

    # Step 1: extract bars
    bars = extract_bars_and_colors(image_path)

    if not bars:
        return [], None

    # Step 2: cluster distinct colors
    clusters = determine_distinct_colors(bars, eps=tolerance)

    # Step 3: find small boxes per cluster if requested
    all_smalls = []
    if refine_color and clusters:
        areas = [b['area'] for b in bars] if bars else [0]
        median_area = float(np.median(areas)) if areas else 0

        for cl in clusters:
            target = cl['color_rgb']
            found = find_bars_by_color(image_path, target, tolerance=tolerance)
            smalls = [f for f in found if f.get('area', 0) < median_area * small_ratio]
            all_smalls.extend(smalls)

    # Merge original bars and smalls: prefer larger boxes on overlap
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
        s_area = s.get('area', (s['bbox'][2]-s['bbox'][0])*(s['bbox'][3]-s['bbox'][1]))
        overlaps = []
        for idx, ex in enumerate(combined):
            ex_box = ex['bbox']
            if iou(s_box, ex_box) > 0:
                overlaps.append((idx, ex))
        if not overlaps:
            combined.append(s)
        else:
            replaced = False
            for idx, ex in overlaps:
                ex_area = ex.get('area', (ex['bbox'][2]-ex['bbox'][0])*(ex['bbox'][3]-ex['bbox'][1]))
                if ex_area >= s_area:
                    replaced = True
                    break
            if replaced:
                continue
            for idx, _ in sorted(overlaps, key=lambda x: x[0], reverse=True):
                combined.pop(idx)
            combined.append(s)

    # sort final boxes by x coordinate
    combined.sort(key=lambda b: b['bbox'][0])

    # Assign color class to each combined box based on nearest cluster mean
    class_means = []
    for cl in clusters:
        # ensure color_rgb is a tuple
        class_means.append(tuple(cl.get('color_rgb', (0, 0, 0))))

    if class_means:
        cm = np.array(class_means, dtype=float)
        for b in combined:
            rgb = tuple(b.get('color_rgb', (0, 0, 0)))
            dists = np.linalg.norm(cm - np.array(rgb, dtype=float), axis=1)
            idx = int(np.argmin(dists))
            b['color_class'] = f'class_{idx+1}'
            b['cluster_idx'] = idx
            b['cluster_color'] = tuple(class_means[idx])
    else:
        for b in combined:
            b['color_class'] = None
            b['cluster_idx'] = None
            b['cluster_color'] = None

    vis_path = None
    if visual:
        out_dir = output_dir or os.path.join(os.getcwd(), 'output_process')
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(out_dir, f"{base}_visualization.png")
        visualize_bars_on_blank(image_path, combined, output_path=vis_path)

    return combined, vis_path