from ultralytics import YOLO
from pathlib import Path

# Load the YOLOv8 model
model_path = Path.cwd().parent / 'models' / 'YOLO' / 'best.pt'
model = YOLO(model_path)
    
# Export the model to ONNX format
model.export(format='onnx')
