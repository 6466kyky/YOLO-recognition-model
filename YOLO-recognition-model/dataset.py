# 导入必要的库
from ultralytics import YOLO

if __name__ == '__main__':  # Windows 多进程必须加这一行
    # 初始化模型
    model = YOLO("yolov8n.pt")
    # 第一次运行代码会自动从官网下载yolov8n.pt文件

    # 执行训练，指定到你手动创建的 runs 目录
    results = model.train(
        data="C:/Users/Lenovo/Desktop/Research group meeting/YOLO/configuration.yaml", #configuration.yaml文件路径
        epochs=500,
        batch=16,
        workers=8,
        device="0",
        # 核心：指定 project 为你手动创建的 runs 目录绝对路径
        project="C:/Users/Lenovo/Desktop/Research group meeting/YOLO/runs",
        # 自定义目录名
        name="train"
    )