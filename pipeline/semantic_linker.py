import numpy as np
import math

class SemanticLinker:
    def _get_center(self, box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def _is_number(self, text):
        clean_text = text.replace(',', '').replace('.', '').replace('%', '').replace('$', '').strip()
        if clean_text.startswith('-'):
            clean_text = clean_text[1:]
        return clean_text.isdigit()

    def _parse_number(self, text):
        try:
            clean_text = text.replace(',', '').replace('%', '').replace('$', '').strip()
            return float(clean_text)
        except ValueError:
            return None

    def link_bar_chart(self, ocr_data, bar_objects):
        final_results = []

        # Prepare OCR centers and apply confidence filtering / normalization
        ocr_items = []
        for item in ocr_data:
            hbox = item.get('hbox')
            if not hbox:
                continue
            conf = float(item.get('confidence', 1.0))
            text = (item.get('text') or '').strip()
            # normalize common OCR mistake: letter O as zero when confidence is low
            if text == 'O' and conf < 0.75:
                text = '0'
            # discard low confidence results
            if conf < 0.5:
                continue
            cx, cy = self._get_center(hbox)
            ocr_items.append({**item, 'text': text, 'confidence': conf, 'center_x': cx, 'center_y': cy})

        # Compute overall vertical span to help decide top/title and bottom/x-axis
        all_ys = []
        for it in ocr_items:
            all_ys.append(it['center_y'])
        for b in bar_objects:
            all_ys.append(b['bbox'][1])
            all_ys.append(b['bbox'][3])
        if not all_ys:
            return {'chart_title': None, 'class_map': {}, 'bars': []}
        top_y = min(all_ys)
        bottom_y = max(all_ys)
        height = max(1.0, bottom_y - top_y)

        # Separate numeric (potential y-axis ticks) and non-numeric texts
        numeric_items = [it for it in ocr_items if self._is_number(it.get('text', ''))]
        non_numeric = [it for it in ocr_items if not self._is_number(it.get('text', ''))]

        # Detect which numeric group corresponds to y-axis ticks by grouping numerics by X
        y_axis_candidates = []
        if numeric_items:
            xs = [it['center_x'] for it in numeric_items]
            min_x, max_x = min(xs), max(xs)
            x_range = max(1.0, max_x - min_x)
            tol_x = max(10.0, 0.05 * x_range)
            groups = []
            for it in numeric_items:
                placed = False
                for g in groups:
                    if abs(g['mean_x'] - it['center_x']) <= tol_x:
                        g['items'].append(it)
                        g['mean_x'] = float(np.mean([x['center_x'] for x in g['items']]))
                        placed = True
                        break
                if not placed:
                    groups.append({'mean_x': it['center_x'], 'items': [it]})

            # Evaluate each group: look for arithmetic progression in parsed numeric values or good linear fit
            best_group = None
            best_score = -1.0
            for g in groups:
                items = sorted(g['items'], key=lambda x: x['center_y'])
                vals = [self._parse_number(it['text']) for it in items]
                if any(v is None for v in vals):
                    continue
                if len(vals) >= 3:
                    diffs = np.diff(vals)
                    # arithmetic progression check: diffs roughly equal
                    diff_std = float(np.std(diffs))
                    diff_mean = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
                    arith_score = 1.0 / (1.0 + diff_std / (abs(diff_mean) + 1e-6))
                else:
                    arith_score = 0.0

                # linear fit quality (pixel_y -> value)
                pixel_coords = np.array([it['center_y'] for it in items])
                values = np.array(vals)
                if len(pixel_coords) > 1:
                    coeffs = np.polyfit(pixel_coords, values, 1)
                    pred = np.polyval(coeffs, pixel_coords)
                    ss_res = np.sum((values - pred) ** 2)
                    ss_tot = np.sum((values - np.mean(values)) ** 2) + 1e-9
                    r2 = 1 - ss_res / ss_tot
                else:
                    r2 = 0.0

                score = 0.7 * r2 + 0.3 * arith_score
                if score > best_score:
                    best_score = score
                    best_group = items

            if best_group and len(best_group) >= 2 and best_score > 0.1:
                y_axis_candidates = best_group
            else:
                y_axis_candidates = numeric_items

        # Determine chart title: top-most non-numeric text within top 15% of image
        chart_title = None
        non_numeric_sorted = sorted(non_numeric, key=lambda x: x['center_y'])
        if non_numeric_sorted:
            candidate = non_numeric_sorted[0]
            if candidate['center_y'] <= top_y + 0.15 * height:
                chart_title = candidate['text']
                # Extend title with nearby long texts (many letters) that are close vertically
                extensions = []
                for it in non_numeric_sorted[1:5]:
                    if it.get('text') == chart_title:
                        continue
                    # if near vertically (within 10% height) and contains many letters, append
                    if abs(it['center_y'] - candidate['center_y']) <= 0.1 * height and sum(c.isalpha() for c in it.get('text', '')) >= 5:
                        extensions.append(it['text'])
                if extensions:
                    chart_title = chart_title + ' ' + ' '.join(extensions)

        # Compute median bottom of bars to detect x-axis labels
        bar_bottoms = [b['bbox'][3] for b in bar_objects] if bar_objects else [bottom_y]
        bar_bottom_median = float(np.median(bar_bottoms))

        # Detect x-axis labels by grouping OCR items (numeric or text) that share similar Y (same horizontal line)
        tol_y = max(8.0, 0.02 * height)
        y_groups = {}
        for it in ocr_items:
            assigned = False
            for gy in list(y_groups.keys()):
                if abs(gy - it['center_y']) <= tol_y:
                    y_groups[gy].append(it)
                    assigned = True
                    break
            if not assigned:
                y_groups[it['center_y']] = [it]

        x_axis_labels = []
        for gy, items in y_groups.items():
            # consider as x-axis if group contains >=2 labels and is located near or below bar bottoms
            if len(items) >= 2 and gy >= bar_bottom_median - 0.05 * height:
                x_axis_labels.extend(items)

        # fallback: any OCR item below median bottom
        if not x_axis_labels:
            x_axis_labels = [it for it in ocr_items if it['center_y'] >= bar_bottom_median]

        # legend candidates: any OCR items (numeric or text) that are not x-axis labels and are not the title
        legend_candidates = [it for it in ocr_items if it not in x_axis_labels and it.get('text') != chart_title]

        # Group bars by color_class and compute centroids and median area
        class_groups = {}
        areas = []
        for b in bar_objects:
            cls = b.get('color_class')
            cx, cy = ((b['bbox'][0] + b['bbox'][2]) / 2, (b['bbox'][1] + b['bbox'][3]) / 2)
            area = (b['bbox'][2] - b['bbox'][0]) * (b['bbox'][3] - b['bbox'][1])
            areas.append(area)
            class_groups.setdefault(cls, []).append({'bar': b, 'center_x': cx, 'center_y': cy, 'area': area})

        median_area = float(np.median(areas)) if areas else 0.0

        class_name_map = {}
        min_bar_per_class = {}
        for cls, items in class_groups.items():
            try:
                min_item = min(items, key=lambda it: it['area'])
                min_bar_per_class[cls] = min_item['bar']
            except ValueError:
                continue

        used_legend_idxs = set()
        for cls, bar in min_bar_per_class.items():
            bx0, by0, bx1, by1 = bar['bbox']
            bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
            mapped_name = None
            best_score = float('inf')
            # Strict: only consider legend OCR boxes that lie to the right of the legend swatch
            for i, legend in enumerate(legend_candidates):
                if i in used_legend_idxs:
                    continue
                lx, ly = legend['center_x'], legend['center_y']
                if lx > bcx:
                    score = (lx - bcx) + 0.01 * abs(ly - bcy)
                    if score < best_score and score < max(80, 0.1 * height):
                        best_score = score
                        mapped_name = legend['text']
                        mapped_idx = i

            if mapped_name is not None:
                class_name_map[cls] = mapped_name
                used_legend_idxs.add(mapped_idx)

        # Fallback: if some classes not assigned, use nearest legend (even if further away) or centroid heuristic
        for cls in class_groups.keys():
            if cls in class_name_map:
                continue
            # only map if there is an OCR box strictly to the right of the class centroid
            mapped_name = None
            mapped_idx = None
            bx = float(np.mean([it['center_x'] for it in class_groups[cls]]))
            by = float(np.mean([it['center_y'] for it in class_groups[cls]]))
            best_score = float('inf')
            for i, legend in enumerate(legend_candidates):
                if i in used_legend_idxs:
                    continue
                lx, ly = legend['center_x'], legend['center_y']
                if lx > bx:
                    score = (lx - bx) + 0.01 * abs(ly - by)
                    if score < best_score and score < max(80, 0.2 * height):
                        best_score = score
                        mapped_name = legend['text']
                        mapped_idx = i
            if mapped_name is not None:
                class_name_map[cls] = mapped_name
                used_legend_idxs.add(mapped_idx)

        # If we only found one legend mapping, apply it to all classes
        if len(class_name_map) == 1:
            only_name = list(class_name_map.values())[0]
            for cls in list(class_groups.keys()):
                class_name_map[cls] = only_name

        # Build final bar results and attach class name
        # Determine which boxes are legend swatches (smallest per class)
        legend_bboxes = set()
        for cls, bar in min_bar_per_class.items():
            legend_bboxes.add(tuple(bar['bbox']))

        # Chart boxes = boxes that are not legend swatches
        chart_boxes = [b for b in bar_objects if tuple(b['bbox']) not in legend_bboxes]

        # Collect any OCR text not used for legend mapping or title as additional info
        used_legend_texts = set()
        for i in used_legend_idxs:
            if 0 <= i < len(legend_candidates):
                used_legend_texts.add(legend_candidates[i].get('text'))

        additional_info = []
        for it in ocr_items:
            txt = it.get('text')
            if not txt:
                continue
            if txt == chart_title:
                continue
            if txt in used_legend_texts:
                continue
            additional_info.append(txt)

        # Ox axis Y is median bottom Y of chart boxes
        if chart_boxes:
            ox_bottoms = [b['bbox'][3] for b in chart_boxes]
            Ox_y = float(np.median(ox_bottoms))
        else:
            Ox_y = bar_bottom_median

        # Candidate labels for chart bars: OCR items (numeric or text) below Ox
        labels_below_ox = [it for it in ocr_items if it['center_y'] > Ox_y]

        for b in bar_objects:
            bar_box = b['bbox']
            bar_center = self._get_center(bar_box)

            # Decide if this is a legend swatch; skip legend boxes from output
            if tuple(bar_box) in legend_bboxes:
                continue

            # For chart boxes: pick the x-axis label nearest to the bottom-middle point of the bar
            closest_label_text = 'Unknown'
            bar_bottom = float(bar_box[3])
            bottom_mid = ( (bar_box[0] + bar_box[2]) / 2.0, bar_bottom )
            best_label = None
            if x_axis_labels:
                best_dist = float('inf')
                for xl in x_axis_labels:
                    lx, ly = xl['center_x'], xl['center_y']
                    dist = math.hypot(lx - bottom_mid[0], ly - bottom_mid[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_label = xl
                if best_label is not None:
                    closest_label_text = best_label['text']
            else:
                best_score = None
                best_label = None
                for xl in labels_below_ox:
                    dy = float(xl['center_y']) - bar_bottom
                    if dy >= 0:
                        dx = abs(xl['center_x'] - bar_center[0])
                        score = (dy, dx)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_label = xl
                if best_label is None and labels_below_ox:
                    best_abs = None
                    for xl in labels_below_ox:
                        dy = abs(float(xl['center_y']) - bar_bottom)
                        dx = abs(xl['center_x'] - bar_center[0])
                        score = (dy, dx)
                        if best_abs is None or score < best_abs:
                            best_abs = score
                            best_label = xl
                if best_label is not None:
                    closest_label_text = best_label['text']

            # compute value using y-axis regression if possible
            scaling_factor = None
            y_intercept = None
            if len(y_axis_candidates) >= 2:
                pixel_coords = np.array([t['center_y'] for t in y_axis_candidates])
                values = np.array([self._parse_number(t['text']) for t in y_axis_candidates])
                if len(pixel_coords) > 1:
                    slope, intercept = np.polyfit(pixel_coords, values, 1)
                    scaling_factor = slope
                    y_intercept = intercept

            if scaling_factor is not None:
                y_top_pixel = bar_box[1]
                calculated_value = scaling_factor * y_top_pixel + y_intercept
                if calculated_value < 0 and calculated_value > -5:
                    calculated_value = 0.0
                calculated_value = round(calculated_value, 2)
            else:
                calculated_value = float(b.get('height', 0))

            cls = b.get('color_class')
            class_name = class_name_map.get(cls)

            final_results.append({
                'label': closest_label_text,
                'value': calculated_value,
                # 'color': b.get('color_hex', '#000000'),
                # 'color_class': cls,
                'class_name': class_name,
                # 'cluster_color': b.get('cluster_color'),
                # 'bar_bbox': bar_box
            })

        return {
            'chart_title': chart_title,
            'class_map': class_name_map,
            'bars': final_results,
            'additional_info': additional_info
        }