"""Parameter tuner for `detect_and_refine`.

Usage:
    python trainer/tune_detect_params.py --img1 my_images/bar2.png --img2 my_images/bar1.jpg

The script performs a small grid-search over a few `detect_and_refine` knobs
(`tolerance`, `small_ratio`, `aggressive_top_k`, `refine_color`) and stops when
both target conditions are met:
 - `bar2.png` -> 15 boxes & 3 classes
 - `bar1.jpg` -> 12 boxes & 2 classes

If successful, best parameters are saved to `trainer/best_detect_params.json`.
"""

import json
import os
import time
from itertools import product
from typing import Tuple

from pipeline.bar_chart_shape import detect_and_refine


def evaluate(image_path: str, params: dict) -> Tuple[int, int]:
    """Run detect_and_refine and return (boxes_count, distinct_classes).
    distinct_classes is number of unique `cluster_idx` (non-None).
    """
    # forward tuning params to detect_and_refine (including extract_bars_and_colors knobs)
    combined, vis = detect_and_refine(
            image_path,
            refine_color=params['refine_color'],
            tolerance=params['tolerance'],
            visual=False,
            small_ratio=params['small_ratio'],
            aggressive_top_k=params['aggressive_top_k'],
            dominant_frac_thresh=params.get('dominant_frac_thresh', 0.95),
            dominant_color_tol=params.get('dominant_color_tol', 18.0),
            bg_tol=params.get('bg_tol', 40.0),
            # extract_bars_and_colors parameters
            thresh_val=params.get('thresh_val', 200),
            blur_ksize=params.get('blur_ksize', 5),
            morph_kernel=params.get('morph_kernel', 3),
            min_area_ratio=params.get('min_area_ratio', 0.005),
            min_size=params.get('min_size', 6),
            use_adaptive=params.get('use_adaptive', False)
        )
    boxes = combined or []
    boxes_count = len(boxes)
    cluster_idxs = set([b.get('cluster_idx') for b in boxes if b.get('cluster_idx') is not None])
    distinct = len(cluster_idxs)
    return boxes_count, distinct


def grid_search(image_targets: list, out_path: str, max_iters: int = 200):
    """Grid-search over parameter combos.

    - image_targets: list of tuples (image_path, target_boxes, target_classes)
    """
    # Expanded grid ranges — increase coverage when results aren't changing
    tolerances = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
    small_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9]
    aggressive_k = [1, 3, 5, 7, 9]
    refine_opts = [True, False]
    # new params controlling dominant-color heuristics in extract_bars_and_colors
    dominant_fracs = [0.8, 0.9, 0.95, 0.98]
    dominant_color_tols = [12, 18, 24, 36]
    bg_tols = [20, 28, 40, 56]
    # knobs for extract_bars_and_colors to fine-tune detection
    thresh_vals = [120, 160, 200, 220]
    blur_ksizes = [3, 5, 7]
    morph_kernels = [1, 3, 5]
    min_area_ratios = [0.002, 0.005, 0.01]
    min_sizes = [4, 6, 10]
    use_adaptives = [False, True]

    tried = 0
    best = None
    best_score = None

    for tol, sratio, atk, refc, dfrac, dctol, bgt, tval, bksz, mker, mar, msz, uad in product(
            tolerances, small_ratios, aggressive_k, refine_opts,
            dominant_fracs, dominant_color_tols, bg_tols,
            thresh_vals, blur_ksizes, morph_kernels, min_area_ratios, min_sizes, use_adaptives):
        params = {
            'tolerance': float(tol),
            'small_ratio': float(sratio),
            'aggressive_k': int(atk),
            'refine_color': bool(refc),
            'dominant_frac_thresh': float(dfrac),
            'dominant_color_tol': float(dctol),
            'bg_tol': float(bgt),
            # forwarded extract params
            'thresh_val': int(tval),
            'blur_ksize': int(bksz),
            'morph_kernel': int(mker),
            'min_area_ratio': float(mar),
            'min_size': int(msz),
            'use_adaptive': bool(uad)
        }
        tried += 1
        t0 = time.time()
        results = []
        ok = True
        try:
            for (imgp, tb, tc) in image_targets:
                b, c = evaluate(imgp, params)
                results.append({'image': imgp, 'boxes': b, 'classes': c, 'target_boxes': tb, 'target_classes': tc})
        except Exception as e:
            print(f"Error running detect on params {params}: {e}")
            ok = False
        dt = time.time() - t0
        if not ok:
            continue

        # print concise summary
        summary = ", ".join([f"{os.path.basename(r['image'])}(boxes={r['boxes']},cls={r['classes']})" for r in results])
        print(f"Tried {tried}: tol={tol} small_ratio={sratio} top_k={atk} -> {summary} time={dt:.2f}s")

        # score: sum of absolute differences to targets
        score = sum([abs(r['boxes'] - r['target_boxes']) + abs(r['classes'] - r['target_classes']) for r in results])
        if best is None or score < best_score:
            best = dict(params)
            best_score = score
            best['result'] = results

        # stop if all exact
        if all((r['boxes'] == r['target_boxes'] and r['classes'] == r['target_classes']) for r in results):
            print("Found exact matching parameters:", params)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump({'params': params, 'results': results}, f, indent=2)
            return params

        if tried >= max_iters:
            break

    # Save best found
    if best is not None:
        print("No exact match found. Saving best found:", best)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'best': best, 'score': best_score}, f, indent=2)
    else:
        print("No valid runs completed.")
    return best


def main():
    image_targets = [
        (os.path.join('my_images', 'bar2.png'), 15, 3),
        (os.path.join('my_images', 'bar1.jpg'), 12, 2),
    ]

    out_path = os.path.join('trainer', 'best_detect_params.json')
    # Allow many more tries when exploring a larger grid
    max_iters = 1000

    print('Grid searching detect_and_refine parameters...')
    best = grid_search(image_targets, out_path, max_iters=max_iters)
    print('Done. Best:', best)


if __name__ == '__main__':
    main()
