import sys
from ultralytics.data.converter import yolo_bbox2segment
import os
import logging
from pathlib import Path

FILE = Path(__file__).resolve()
# Determine the base directory depending on whether the script is running as a standalone executable or from source
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle/executable, the base dir is the executable's directory
    BASE_DIR = Path(sys.executable).parent
else:
    # If running from source, the base dir is the project root (3 levels up from this script)
    BASE_DIR = FILE.parents[2]
    if str(BASE_DIR) not in sys.path:
        sys.path.append(str(BASE_DIR))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')

workspace = Path('workspace') / 'merge' if Path('workspace').exists() else BASE_DIR / 'workspace' / 'merge'
bbox_dir = workspace / 'bbox_2_segment'
images_dir = workspace / 'images'

sam_b_model_path = Path.cwd().parent / 'models' / 'sam_b.pt'

if not os.path.exists(workspace):
    logger.warning(f'Ensure workspace directory exists in {workspace}')
    sys.exit(1)

if not os.path.exists(images_dir):
    logger.warning('Ensure images directory exists in workspace directory')
    sys.exit(1)
else:
    images = next(
        (True for path in Path(images_dir).rglob('*') if path.suffix.lower() in ['.png', '.jpg']),
        False
    )
    if not images:
        logger.warning('No images files found in the directory...')
        sys.exit(1)

if not os.path.exists(bbox_dir):
    os.makedirs(bbox_dir)

yolo_bbox2segment(
    im_dir= workspace,
    save_dir= os.path.join(workspace, 'bbox_2_segment'),
    sam_model=str(sam_b_model_path)
)