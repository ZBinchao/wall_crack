import cv2
import numpy as np
from ultralytics import YOLO

# 加载模型
model = YOLO(r"D:\wall_crack_m\runs\segment\train-4\weights\last.pt")

# 你的图片
img_path = r"D:\wall_crack_m\text\text-2.jpg"
img = cv2.imread(img_path)

# 推理
results = model(img_path)

# 遍历每个裂缝
for res in results:
    if res.masks is not None:
        masks = res.masks.data.cpu().numpy()

        for idx, mask in enumerate(masks):
            # 转为二值图
            mask_bin = (mask > 0.5).astype(np.uint8)
            
            # ====================== 核心：计算像素点数量 ======================
            pixel_count = cv2.countNonZero(mask_bin)  # 裂缝占多少像素
            total_img_pixels = mask_bin.size         # 整张图片总像素
            ratio = (pixel_count / total_img_pixels) * 100  # 占比 %

            print(f"==================================")
            print(f"第 {idx+1} 条裂缝")
            print(f"裂缝像素点数：{pixel_count} 像素")
            print(f"占图片总面积：{ratio:.3f} %")
            print(f"==================================")

# 显示图片
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
