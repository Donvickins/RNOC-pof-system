from ultralytics import YOLO

model - YOLO("runs/segment/train/weights/best.pt")

results = model.predict(source="workspace/images", save=True, conf=0.5, hide_labels=False, hide_conf=False)

for result in results:
    result.show()