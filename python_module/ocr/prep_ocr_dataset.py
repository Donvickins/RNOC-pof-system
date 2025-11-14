import logging
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python_module.utils.utils import get_class_name, get_site_id_from_image, site_id_2_binary, extract_text

logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

workspace = Path.cwd().parent / 'workspace'
image_dir = workspace / 'images'
yolo_model_path = Path.cwd().parent / 'models' / 'YOLO' / 'best.pt'
save_img_dir = workspace / 'OCR' / 'images'

if not workspace.exists():
    logger.error('Workspace directory not found')
    sys.exit(1)

if not yolo_model_path.exists():
    logger.error(f'YOLO model not found in {yolo_model_path.parent}')
    sys.exit(1)

if not next((True for img in image_dir.iterdir() if img.suffix.lower() in ['.png', '.jpg', '.jpeg']),False):
    logger.error(f'No images found in {image_dir}')
    sys.exit(1)

Path.mkdir(save_img_dir, exist_ok=True)
yolo_model = YOLO(yolo_model_path)

results = yolo_model.predict(
    source=image_dir,
    save=False,
)

for idx, result in enumerate(results):
    for key, item in enumerate(result.boxes.data):
        class_id = item[-1]
        class_name = get_class_name(result=result, c_id=class_id)
        bbox = result.boxes.data[key][:4]
        image_name = f'image_{idx}{key}.png'
        txt_file = f'image_{idx}{key}.gt.txt'

        if any(class_name.startswith(p) for p in ['ATN', 'RTN', 'Router', 'Switch']):
            site_id_image =  get_site_id_from_image(image_path=result.path, node_bbox=bbox)
            if isinstance(site_id_image, str) or site_id_image is None:
                continue

            site_id_binary = site_id_2_binary(site_id_image)
            cv2.imwrite(f'{str(save_img_dir)}/{image_name}', site_id_binary)
            label = Path(save_img_dir/txt_file)
            label.touch()
            with open(label, 'w') as f:
                f.write(extract_text(site_id_binary))


logger.info(f'Images saved to {save_img_dir}')
