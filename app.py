from chart_class_model import load_model_and_predict
from bar_chart_ocr import extract_text_and_boxes
from bar_chart_shape import detect_and_refine
from semantic_linker import SemanticLinker
import json
import os

SAVED_MODEL_PATH = 'chart_classifier_model.pth'
IMAGE_TO_TEST = './my_images/bar_chart.jpg'
def process(image_path=IMAGE_TO_TEST, run_ocr=True, run_shape=True):
    print(f"--- 1. Phân loại Ảnh ---")
    print(f"Đang tải ảnh: {image_path}")

    label, confidence = load_model_and_predict(image_path, SAVED_MODEL_PATH)

    print(f"\n--- Kết quả Phân loại ---")
    print(f"Dự đoán: {label}")
    print(f"Độ tin cậy: {confidence:.2f}%")

    combined = []
    vis_path = None
    if run_shape:
        out_dir = os.path.join(os.getcwd(), "output_image")
        os.makedirs(out_dir, exist_ok=True)
        combined, vis_path = detect_and_refine(image_path, refine_color=True, tolerance=40, visual=True, output_dir=out_dir)
        if vis_path:
            print(f"Hình minh họa cuối cùng đã lưu tại: {vis_path}")
        else:
            print("Không tìm thấy thanh nào.")
    else:
        print("\n--- Trích xuất hình dạng bị tắt bởi cấu hình ---")

    ocr_results = []
    if run_ocr:
        print(f"\n--- 2. Trích xuất OCR ---")
        ocr_results = extract_text_and_boxes(image_path)
        if ocr_results:
            print(f"Tìm thấy {len(ocr_results)} mục OCR")
        else:
            print("Không tìm thấy văn bản nào.")
    else:
        print("\n--- OCR bị tắt bởi cấu hình ---")

    # Semantic linking: only if we have both shape and OCR results
    if run_shape and run_ocr and combined and ocr_results:
        linker = SemanticLinker()
        linked = linker.link_bar_chart(ocr_results, combined)
        if isinstance(linked, dict):
            title = linked.get('chart_title')
            if title:
                print(f"Chart Title: {title}")
            bars = linked.get('bars', [])

            # Build JSON output using semantic linker results
            chart_name = os.path.basename(image_path)
            metadata = {
                "chart_type": "Bar Chart (Vertical)",
                "title": title or "",
                "input_image": chart_name
            }
            raw_data_table = []
            for it in bars:
                category = it.get('label') or it.get('class_name') or "Unknown"
                value = it.get('value')
                # try to coerce numeric values
                try:
                    if value is None:
                        value_num = 0
                    else:
                        value_num = float(value)
                        # represent as int if whole
                        if abs(value_num - int(value_num)) < 1e-9:
                            value_num = int(value_num)
                except Exception:
                    value_num = it.get('value')

                raw_data_table.append({
                    "category": category,
                    "value": value_num,
                    "class": it.get('class_name'),
                })

            output_json = {
                "metadata": metadata,
                "raw_data_table": raw_data_table
            }

            out_dir = os.path.join(os.getcwd(), "output_image")
            os.makedirs(out_dir, exist_ok=True)
            out_name = os.path.splitext(chart_name)[0] + ".json"
            out_path = os.path.join(out_dir, out_name)
            try:
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(output_json, f, ensure_ascii=False, indent=2)
                print(f"JSON output saved to: {out_path}")
            except Exception as e:
                print(f"Failed to save JSON: {e}")
        else:
            for item in linked:
                print(item)

if __name__ == '__main__':
    image_array = [
        './my_images/bar2.png',
        './my_images/bar3.png',
        './my_images/bar4.png',
        './my_images/bar5.png',
    ]
    for img_path in image_array:
        process(image_path=img_path, run_ocr=True, run_shape=True)