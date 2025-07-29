#接下来就可以用训练出来的训练集识别视频或图片（这里以视频为例）
import os
import cv2
from ultralytics import YOLO
import time

def process_video(model_path, source_video, output_dir, output_name, resolution=(640, 384), batch_size=1):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # 加载模型
    model = YOLO(model_path)
    # 打开视频文件
    cap = cv2.VideoCapture(source_video)
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_dir, f"{output_name}.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    frame_count = 0
    start_time = time.time()
    try:
        while cap.isOpened():
            # 批量读取帧
            frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                # 调整帧大小以降低内存占用
                frame = cv2.resize(frame, resolution)
                frames.append(frame)
            if not frames:
                break
            # 模型推理
            results = model(frames, verbose=False)
            # 处理结果并写入输出视频
            for i, result in enumerate(results):
                annotated_frame = result.plot()
                out.write(annotated_frame)
                frame_count += 1
                # 打印进度
                if frame_count % 100 == 0:
                    elapsed_time = time.time() - start_time
                    fps_processing = frame_count / elapsed_time
                    print(f"已处理 {frame_count} 帧，处理速度: {fps_processing:.2f} FPS")
    finally:
        # 释放资源
        cap.release()
        out.release()
        print(f"视频处理完成，结果保存在: {output_path}")

if __name__ == "__main__":
    # 模型路径
    model_path = r"C:\Users\Lenovo\Desktop\Research group meeting\YOLO\runs\train\weights\best.pt" #这个是训练之后的模型 如果之前步骤一样模型路径为runs\train\weights\best.pt 一定要用best.pt 不要用last.pt
    # 源视频路径
    source_video = r"D:\surveillance video\6.6\J2-1-3748\20250607023611.mp4" #待检测视频路径
    # 输出目录
    output_dir = r"D:\surveillance video\6.6\J2-1-3748" #输出视频保存路径
    # 输出文件名
    output_name = "v22"
    # 处理参数 - 可根据系统性能调整
    resolution = (640, 384)  # 降低分辨率以减少内存使用
    batch_size = 1  # 批处理大小，可根据内存情况调整
    # 执行视频处理
    process_video(model_path, source_video, output_dir, output_name, resolution, batch_size)