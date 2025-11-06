import shutil
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

workspace = Path.cwd().parent / 'workspace'
image_dir = workspace / 'images'
save_img_dir = workspace / 'OCR' / 'images'
data_dir = save_img_dir.parent / 'data' / 'pof_ocr-ground-truth'

if not workspace.exists():
    logger.error('Workspace directory not found')
    sys.exit(1)

if not image_dir.exists() or not next((True for img in image_dir.iterdir() if img.suffix.lower() in ['.png', '.jpg', '.jpeg']),False):
    logger.error(f'No images found in {image_dir}')
    sys.exit(1)

if not save_img_dir.exists() or not next((True for img in save_img_dir.iterdir() if img.suffix.lower() in ['.png', '.jpg', '.jpeg']),False):
    logger.error(f'No images found in {save_img_dir}')
    sys.exit(1)

Path.mkdir(data_dir, exist_ok=True, parents=True)

images = [file for file in save_img_dir.iterdir() if file.suffix.lower() in ['.png', '.jpg', '.jpeg']]
labels = [file.name for file in save_img_dir.iterdir() if file.suffix.lower() == '.txt']

logger.info(f"Found {len(images)} images and {len(labels)} labels")
logger.info('Sorting Images and label...')

for image_file in images:
    label_file = image_file.stem + '.gt.txt'
    if label_file in labels:
        with open(Path(save_img_dir/label_file), 'r') as label:
            for line in label:
                if not isinstance(line, str):
                    continue
                src_img_path = save_img_dir / image_file.name
                dst_img_path = data_dir / image_file.name
                shutil.copy2(src_img_path, dst_img_path)

                src_label_path = save_img_dir / label_file
                dst_label_path = data_dir / label_file
                shutil.copy2(src_label_path, dst_label_path)

logger.info('Sorting Completed Successfully...')
logger.info(f'Check sorted data in {data_dir}')
