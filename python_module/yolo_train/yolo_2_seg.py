from ultralytics.data.converter import yolo_bbox2segment
import os

workspace = os.path.join(os.getcwd(), 'merge')
yolo_bbox2segment(
    im_dir= workspace,
    save_dir= os.path.join(workspace, 'detect_label'),
    sam_model='model/sam_b.pt'
)