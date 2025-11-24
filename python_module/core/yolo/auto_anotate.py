import torch
import sys
import logging
import shutil
from ultralytics import YOLO
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

def move_and_merge(src, dst):
    if not dst.exists():
        shutil.move(src, dst)
    else:
        for src_path in src.iterdir():
            dst_path = dst/ src_path.name
            if src_path.is_dir():
                move_and_merge(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)

workspace = Path.cwd().parents[2] / 'workspace'
image_dir = workspace/ 'images'
output_dir = workspace / 'auto_labelled'
labelled_image_dir = output_dir / 'labelled_images'
auto_labelled_dir = output_dir / 'labels'
temp_annotated = output_dir / 'temp'

if not image_dir.is_dir():
    logger.info('Images directory does not exist')
    sys.exit(0)

total_images = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
if not next((True for file in total_images), False):
    logger.info('Images directory does not contain any image files')
    sys.exit(0)

output_dir.mkdir(parents=True, exist_ok=True)
labelled_image_dir.mkdir(parents=True, exist_ok=True)

model_path = Path.cwd().parent / 'models' / 'YOLO' / 'best.pt'
model = YOLO(model_path)

model.predict(
    source=image_dir,
    save=True,
    save_txt=True,
    show_labels=True,
    show_conf=True,
    project=output_dir,
    name=temp_annotated,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

runs = []
for content in output_dir.iterdir():
    if content.is_dir() and content.name.startswith('temp'):
        runs.append(content.name[4:])

if len(runs) != 0:
    max_run = max(runs)
    temp_annotated = Path(str(temp_annotated) + max_run)

a_label_dir = temp_annotated / 'labels'
move_and_merge(a_label_dir, auto_labelled_dir)
move_and_merge(temp_annotated, labelled_image_dir)
shutil.rmtree(temp_annotated)

for content in output_dir.iterdir():
    if content.is_dir() and content.name.startswith('temp'):
        shutil.rmtree(content)

logger.info(f'Completed Successfully: Check: {output_dir}')