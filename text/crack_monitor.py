import os
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "runs" / "segment" / "train-4" / "weights" / "best.pt"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "new_cracks_output"
STATE_DIR = BASE_DIR / "monitoring_state"
IOU_THRESHOLD = 0.3

SEVERITY_LOW_MAX = 1.0
SEVERITY_MEDIUM_MAX = 5.0

_model = None


def load_model():
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model


def run_yolo_on_image(model, image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    results = model(str(image_path))
    result = results[0]

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        masks_bin = (masks > 0.5).astype(np.uint8)
        boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else np.array([])
        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
    else:
        masks_bin = np.array([])
        boxes = np.array([])
        confs = np.array([])

    return {
        "image": img,
        "masks": masks_bin,
        "boxes": boxes,
        "confs": confs,
        "image_shape": img.shape,
        "total_pixels": img.shape[0] * img.shape[1],
        "has_cracks": len(masks_bin) > 0,
    }


def compute_mask_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def find_new_cracks(current_masks, previous_masks):
    if previous_masks is None or len(previous_masks) == 0:
        return []

    H_curr, W_curr = current_masks.shape[1], current_masks.shape[2]
    H_prev, W_prev = previous_masks.shape[1], previous_masks.shape[2]

    if H_prev != H_curr or W_prev != W_curr:
        prev_resized = np.array([
            cv2.resize(m.astype(np.uint8), (W_curr, H_curr), interpolation=cv2.INTER_NEAREST)
            for m in previous_masks
        ]).astype(bool)
    else:
        prev_resized = previous_masks.astype(bool)

    new_indices = []
    for i, curr_mask in enumerate(current_masks):
        max_iou = 0.0
        for prev_mask in prev_resized:
            iou = compute_mask_iou(curr_mask.astype(bool), prev_mask)
            if iou > max_iou:
                max_iou = iou
        if max_iou < IOU_THRESHOLD:
            new_indices.append(i)

    return new_indices


def classify_severity(pixel_count, total_pixels):
    ratio = (pixel_count / total_pixels) * 100
    if ratio < SEVERITY_LOW_MAX:
        return "low"
    elif ratio < SEVERITY_MEDIUM_MAX:
        return "medium"
    else:
        return "high"


def annotate_image(img, masks, new_indices, boxes=None, confs=None):
    annotated = img.copy()
    overlay = annotated.copy()
    H, W = img.shape[:2]

    green = (0, 255, 0)
    red = (0, 0, 255)
    green_fill = (0, 180, 0)
    red_fill = (0, 0, 180)
    alpha = 0.45

    for idx, mask in enumerate(masks):
        mask_resized = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_resized.astype(bool)

        if idx in new_indices:
            color = red
            fill_color = red_fill
        else:
            color = green
            fill_color = green_fill

        overlay[mask_bool] = fill_color

        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, color, 2)

        if boxes is not None and len(boxes) > idx:
            x1, y1, x2, y2 = boxes[idx].tolist()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{'NEW' if idx in new_indices else 'exist'} #{idx + 1}"
            if confs is not None and len(confs) > idx:
                label += f" {confs[idx]:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)
    return annotated


def load_previous_state():
    state_file = STATE_DIR / "state.json"
    masks_file = STATE_DIR / "previous_masks.npy"
    if not state_file.exists() or not masks_file.exists():
        return None, None
    with open(state_file, "r") as f:
        meta = json.load(f)
    masks = np.load(masks_file)
    return masks, meta


def save_current_state(masks, meta):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(STATE_DIR / "previous_masks.npy", masks)
    with open(STATE_DIR / "state.json", "w") as f:
        json.dump({**meta, "timestamp": datetime.now().isoformat()}, f, ensure_ascii=False)


def build_report(all_results):
    images_with_new = sum(1 for r in all_results if r["new_cracks_count"] > 0)
    return {
        "run_timestamp": datetime.now().isoformat(),
        "total_images_processed": len(all_results),
        "images_with_new_cracks": images_with_new,
        "results": all_results,
    }


def process_uploads():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "annotated").mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    processed_dir = UPLOAD_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [f for f in UPLOAD_DIR.iterdir() if f.suffix.lower() == ".jpg" and f.is_file()],
        key=lambda x: x.name,
    )

    if not image_files:
        return {"message": "No images found in uploads/", "total_images_processed": 0, "images_with_new_cracks": 0, "results": []}

    model = load_model()
    previous_masks, prev_meta = load_previous_state()
    all_results = []

    for idx, image_path in enumerate(image_files):
        current_result = run_yolo_on_image(model, image_path)
        if current_result is None:
            image_path.rename(processed_dir / image_path.name)
            continue

        current_masks = current_result["masks"]
        filename = image_path.name

        is_baseline = (idx == 0 and previous_masks is None)

        if current_result["has_cracks"]:
            if is_baseline:
                new_indices = []
            else:
                new_indices = find_new_cracks(current_masks, previous_masks)
        else:
            new_indices = []
            image_path.rename(processed_dir / image_path.name)
            all_results.append({
                "filename": filename,
                "is_baseline": is_baseline,
                "total_cracks": 0,
                "new_cracks_count": 0,
                "new_cracks_details": [],
                "message": "未检测到裂缝，跳过状态更新",
            })
            continue

        new_cracks_details = []
        for crack_idx in new_indices:
            pixel_count = int(current_masks[crack_idx].sum())
            ratio = (pixel_count / current_result["total_pixels"]) * 100
            severity = classify_severity(pixel_count, current_result["total_pixels"])
            box = current_result["boxes"][crack_idx].tolist() if len(current_result["boxes"]) > crack_idx else []
            conf = float(current_result["confs"][crack_idx]) if len(current_result["confs"]) > crack_idx else 0.0
            new_cracks_details.append({
                "crack_index": crack_idx,
                "confidence": round(conf, 4),
                "severity": severity,
                "mask_pixel_count": pixel_count,
                "mask_ratio_percent": round(ratio, 3),
                "bbox": box,
            })

        result_entry = {
            "filename": filename,
            "is_baseline": is_baseline,
            "total_cracks": len(current_masks),
            "new_cracks_count": len(new_indices),
            "new_cracks_details": new_cracks_details,
        }

        if len(new_indices) > 0:
            annotated = annotate_image(
                current_result["image"],
                current_masks,
                new_indices,
                current_result["boxes"],
                current_result["confs"],
            )
            output_filename = f"new_{filename}"
            cv2.imwrite(str(OUTPUT_DIR / "annotated" / output_filename), annotated)
            result_entry["annotated_path"] = str(OUTPUT_DIR / "annotated" / output_filename)

        all_results.append(result_entry)

        previous_masks = current_masks

        image_path.rename(processed_dir / image_path.name)

    if previous_masks is not None:
        save_current_state(previous_masks, {"filename": filename, "image_shape": current_result["image_shape"]})

    report = build_report(all_results)
    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    report = process_uploads()
    print(json.dumps(report, indent=2, ensure_ascii=False))
