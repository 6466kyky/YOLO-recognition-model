#利用官方数据集测试能否正常使用
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

yolo = YOLO(model="./yolov8n.pt", task="detect")
#第一次运行代码会自动从官网下载yolov8n.pt文件

img = cv2.imread(r"C:\Users\Lenovo\Desktop\YOLO\car.jpg") #目标识别图片路径
result = yolo(source=img)
plotted_img = result[0].plot()

scaled_img = cv2.resize(plotted_img, (0, 0), fx=0.5, fy=0.5)

cv2.imshow("Detection Result", scaled_img)
cv2.waitKey(0)
cv2.destroyAllWindows()