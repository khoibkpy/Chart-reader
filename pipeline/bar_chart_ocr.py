from paddleocr import PaddleOCR
import os
import json

ocr_engine = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang='en'
)

OUTPUT_JSON_DIR = './output_ocr'
OUTPUT_IMAGE_DIR = './output_ocr'
POSTFIX_TO_DELETE = 'preprocessed_img.png'

def extract_text_and_boxes(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy ảnh tại {image_path}")

    if not os.path.exists(OUTPUT_JSON_DIR):
        os.makedirs(OUTPUT_JSON_DIR)
        print(f"Đã tạo thư mục: {OUTPUT_JSON_DIR}")

    image_basename = os.path.basename(image_path)
    image_name_without_ext = os.path.splitext(image_basename)[0]
    result = ocr_engine.predict(image_path, text_rec_score_thresh=0.4, text_det_thresh=0.5)
    json_filename = f"{image_name_without_ext}_res.json"
    json_file_path = os.path.join(OUTPUT_JSON_DIR, json_filename)
    useful_data = []

    if not (result and result[0]):
        print(f"PaddleOCR không tìm thấy text nào trong {image_path}.")
        return useful_data

    res = result[0]

    res.save_to_json(OUTPUT_JSON_DIR)
    with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    texts = data.get('rec_texts', [])
    scores = data.get('rec_scores', [])
    polygons = data.get('rec_polys', [])

    if not (len(texts) == len(scores) == len(polygons)):
        print(f"Lỗi: Dữ liệu trong kết quả OCR không đồng nhất.")
        return useful_data

    for text, score, poly_box in zip(texts, scores, polygons):
        try:
            x_coords = [p[0] for p in poly_box]
            y_coords = [p[1] for p in poly_box]
            
            x_min = int(min(x_coords))
            y_min = int(min(y_coords))
            x_max = int(max(x_coords))
            y_max = int(max(y_coords))
            
            simple_box = [x_min, y_min, x_max, y_max]
            
            useful_data.append({
                "text": text,
                "confidence": score,
                "hbox": simple_box
            })
        except Exception:
            pass

    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(useful_data, f, ensure_ascii=False, indent=4)
        print(f"Đã lưu useful_data vào: {json_file_path}")
    except Exception as e:
        print(f"Lỗi khi lưu file useful_data JSON: {e}")

    try:
        res.save_to_img(OUTPUT_IMAGE_DIR)
    except Exception as e:
        print(f"Lỗi khi PaddleOCR lưu ảnh: {e}")

    if os.path.exists(OUTPUT_IMAGE_DIR):
        try:
            for filename in os.listdir(OUTPUT_IMAGE_DIR):
                if filename.endswith(POSTFIX_TO_DELETE):
                    file_to_delete = os.path.join(OUTPUT_IMAGE_DIR, filename)
                    os.remove(file_to_delete)
        except Exception as e:
            print(f"Lỗi khi dọn dẹp thư mục output_visual: {e}")
    return useful_data