from ultralytics import YOLO
# 加载训练好的模型
model = YOLO(r"runs/segment/train-4/weights/best.pt")

# 预测单张图片
results = model.predict(
    source=r"D:\wall_crack_m\text\text-4.jpg",  # 改成你自己的图片路径
    save=True,  # 保存预测结果
    conf=0.25  # 置信度阈值，低于这个的检测结果会过滤掉
)

# 弹出窗口显示结果
results[0].show()