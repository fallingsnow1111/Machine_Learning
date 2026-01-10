import ultralytics
import os

print("YOLO 版本:", ultralytics.__version__)
print("实际引用的文件路径:", os.path.dirname(ultralytics.__file__))