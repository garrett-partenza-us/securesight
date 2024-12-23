from ultralytics import YOLO

model = YOLO("weights/yolov11n-face.pt")

model.export(format="onnx", opset=12, imgsz=640)
