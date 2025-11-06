from ultralytics import YOLO
import sys
from pathlib import Path
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

def move_and_merge(src, dst):
    if not Path.exists(dst):
        shutil.move(src, dst)
    else:
        for item in Path.iterdir(src):
            src_path = Path(src) / item.name
            dst_path = Path(dst) / item.name
            if Path.is_dir(src_path):
                move_and_merge(src_path, dst_path)
            else:
                shutil.copy(src_path, dst_path)
        shutil.rmtree(src)

workspace = Path.cwd().parent / 'workspace'
image_dir = workspace/ 'images'
output_dir = workspace / 'auto_labelled'
labelled_image_dir = output_dir / 'labelled_images'
auto_labelled_dir = output_dir / 'labels'

temp_annotated = output_dir / 'temp'

if not image_dir.is_dir():
    logger.info('Images directory does not exist')
    sys.exit(0)

Path.mkdir(output_dir, exist_ok=True)
Path.mkdir(labelled_image_dir, exist_ok=True)

total_images = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
if not next((True for file in total_images), False):
    logger.info('Images directory does not contain any .png files')
    sys.exit(0)

output_dir.mkdir(parents=True, exist_ok=True)

model_path = Path.cwd().parent / 'models' / 'YOLO' / 'best.pt'
model = YOLO(model_path)

model.predict(
    source=image_dir,
    save=True,
    save_txt=True,
    show_labels=True,
    show_conf=True,
    project=output_dir,
    name=temp_annotated
)

a_label_dir = temp_annotated / 'labels'
move_and_merge(a_label_dir, labelled_image_dir)
temp_annotated.rename(labelled_image_dir)

logger.info('Done...')
logger.info(f'Check {output_dir}')