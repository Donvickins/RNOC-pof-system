"""
Author: Victor Chukwujekwu vwx1423235

This utility script prepares txt files containing site_id down with corresponding pof predictions.
This is done in preparation for a personnel to fill in the pof per image.
This will be used as ground truth for training of the GNN model

This script assumes that workspace directory is 2 directories above it.
If you wish to run this script outside current directory. You need to modify workspace directory path to match
you desired directory.

Note: This script assumes the site_down id is in the file name bearing the structure: topology_AK0031_20250704_191908.png

The output of the operation will be saved workspace/pof/
"""

from pathlib import Path

FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[2]

workspace = PROJECT_ROOT/ 'workspace'
image_path = Path(workspace, 'images')
if not image_path.is_dir() and next((True for file in image_path.glob('*.png') if file.is_file()), False):
    print(f'Images folder does not exist or not image files in {image_path}')
    exit()

pof_path = Path(workspace/ 'pof')
pof_path.mkdir(parents=True, exist_ok=True)

for image in image_path.glob('*.png'):
    site_id = image.stem.split('_')[1]
    text = f'down: {site_id}\npof: '
    file_name = f'{image.stem}.txt'
    file_path = pof_path / file_name
    file_path.touch()
    with open(file_path, 'w') as f:
        f.write(text)

print('Task completed successfully.')