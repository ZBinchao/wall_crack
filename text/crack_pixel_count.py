import os
import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = YOLO(os.path.join(BASE_DIR, "..", "runs", "segment", "train-4", "weights", "best.pt"))
img_path = os.path.join(BASE_DIR, "text-2.jpg")
img = cv2.imread(img_path)

results = model(img_path)

result = results[0]
if result.masks is not None:
    masks = result.masks.data.cpu().numpy()
    total_img_pixels = img.shape[0] * img.shape[1]

    for idx, mask in enumerate(masks):
        mask_bin = (mask > 0.5).astype(np.uint8)
        pixel_count = cv2.countNonZero(mask_bin)
        ratio = (pixel_count / total_img_pixels) * 100

        print(f"==================================")
        print(f"第 {idx + 1} 条裂缝")
        print(f"裂缝像素点数：{pixel_count} 像素")
        print(f"占图片总面积：{ratio:.3f} %")
        print(f"==================================")

    # 生成标记后的图片
    annotated = result.plot()

    output_path = os.path.join(BASE_DIR, "output.jpg")
    cv2.imwrite(output_path, annotated)
    print(f"标记结果已保存至：{output_path}")

    cv2.imshow("result", annotated)
else:
    print("未检测到裂缝")
    cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
