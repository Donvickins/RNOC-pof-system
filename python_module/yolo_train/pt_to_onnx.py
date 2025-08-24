from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("runs/segment/train/weights/best.pt")
    
# Export the model to ONNX format
model.export(format='onnx')
