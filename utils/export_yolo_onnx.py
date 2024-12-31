from ultralytics import YOLO

model = YOLO("../../weights/yolov11m-face.pt")

model.export(format="onnx", opset=12, imgsz=640)
