import os
import shutil
import yaml
from pathlib import Path

workspace = os.path.join(os.getcwd(), 'workspace')

#Paths
class_names = os.path.join(workspace, 'classes.txt')
save_dir = os.path.join(workspace, 'export_dir')

if not os.path.exists(save_dir):
   os.makedirs(save_dir)

if not os.path.exists(os.path.join(save_dir, 'images', 'train')) or not os.path.exists(os.path.join(save_dir, 'labels', 'train')):
    os.makedirs(os.path.join(save_dir, 'images', 'train'))
    os.makedirs(os.path.join(save_dir, 'labels', 'train'))

for image_file in os.listdir(os.path.join(workspace, 'images')):
   shutil.copy(os.path.join(workspace, 'images', image_file), os.path.join(save_dir, 'images', 'train', image_file))
   file_label = Path(image_file)
   label_name = file_label.stem + '.txt'
   shutil.copy(os.path.join(workspace, 'labels', label_name), os.path.join(save_dir, 'labels', 'train', label_name))
   path = Path(os.path.join('images', 'train', image_file))
   with open(os.path.join(save_dir,'train.txt'), 'a') as train:
    train.write(f'{path.as_posix()}\n')

# Extract class names to create yaml config
with open(class_names, 'r') as class_names_file:
    classes = []
    for line in class_names_file.readlines():
      if len(line.strip()) == 0: continue
      classes.append(line.strip())

base_path = os.getcwd()
data = {
    'path': './',
    'train': 'train.txt',
    'names': {key:value for key, value in enumerate(classes)}
}

# Write data to YAML file
with open(os.path.join(save_dir, 'data.yaml'), 'w') as f:
    yaml.dump(data, f, sort_keys=False)