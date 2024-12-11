from ultralytics import YOLO

pt_name = ["best.pt", "yolov11s.pt"]

model = YOLO(pt_name[1])

model.predict(source=0, show=True, conf=0.45)