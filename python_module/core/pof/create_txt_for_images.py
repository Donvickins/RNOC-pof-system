from pathlib import Path

workspace = Path.cwd().parent / 'workspace'
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