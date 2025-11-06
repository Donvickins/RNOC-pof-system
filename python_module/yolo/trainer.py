from pathlib import Path
import random
import os
import sys
import shutil
import argparse
import subprocess as command

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', help='Path to data folder containing image and annotation files',
                    required=True)
parser.add_argument('--train-pct', help='Ratio of images to go to train folder; \
                    the rest go to validation folder (example: ".8")',
                    default=.8)

parser.add_argument('--install-deps', help='Do you want to install requirements',choices=['yes', 'no'], default='yes')

args = parser.parse_args()

data_path = args.datapath
train_percentage = float(args.train_pct)
install_deps = args.install_deps

data_path = Path(data_path).resolve()

# Check for valid entries
if not os.path.isdir(data_path):
   print('Directory specified by --datapath not found. Verify the path is correct (and uses double back slashes if on Windows) and try again.')
   sys.exit(1)
if train_percentage < .01 or train_percentage > 0.99:
   print('Invalid entry for train_pct. Please enter a number between .01 and .99.')
   sys.exit(1)

# Define path to input dataset 
input_image_path = os.path.join(data_path,'images')
input_label_path = os.path.join(data_path,'labels')
input_class_names_path = os.path.join(data_path, 'classes.txt')
config_yaml = os.path.join(data_path, 'config.yaml')

if not os.path.exists(input_image_path) or not os.path.exists(input_label_path) or not os.path.exists(input_class_names_path):
    print(f'Either Image Path: {input_image_path}, Label Path: {input_label_path}, or Class Name {input_class_names_path} does not exist in current directory')
    sys.exit(0)

#Prep workspace for dependencies 
root_dir = Path.cwd()
env_dir = root_dir / '.venv'
req_dir = Path.resolve(root_dir/'requirements.txt')
model_path = Path(root_dir) / 'model/best.pt'

if not Path.exists(model_path):
    model_path = Path(root_dir) / 'model/yolov8l-seg.pt'

#Create new venv
if not os.path.exists(env_dir):
    try:
        command.run([sys.executable, '-m' , 'venv', '.venv'], cwd=root_dir ,check=True)
        print("Environment Created Successfully")
    except command.CalledProcessError as e:
        print(f'Environment Creation failed with error: {e}')
        sys.exit(1)

if os.name == 'posix':
    venv_pip = os.path.join(env_dir, 'bin', 'pip')
    venv_python = os.path.join(env_dir, 'bin', 'python')
    yolo_exec = os.path.join(env_dir, 'bin', 'yolo')
elif os.name == 'nt':
    venv_pip = os.path.join(env_dir, 'Scripts', 'pip.exe')
    venv_python = os.path.join(env_dir, 'Scripts', 'python.exe')
    yolo_exec = os.path.join(env_dir, 'Scripts', 'yolo.exe')
else:
    print("Unknown Operating system")
    sys.exit(1)

if not os.path.exists(venv_pip) or not os.path.exists(venv_python):
    print(f'pip and python not found in: {venv_pip} and {venv_python}')

if not os.path.exists(req_dir):
    print(f'Requirements.txt is not found in: {req_dir}')
    sys.exit(1)

#Restart app in venv
if sys.executable != venv_python:
    os.execv(venv_python, [venv_python] + sys.argv)

#Install Dependencies
if install_deps == 'yes':   
    print('Installing Dependencies...')
    command.run([venv_python, '-m', 'ensurepip', '--upgrade'])
    command.run([venv_python, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
    dep_install = command.run([venv_python, '-m', 'pip','install', '-r', req_dir])
    if dep_install.returncode != 0:
        print("Installation failed: Try again...")
        sys.exit(1)

# Get list of all images and annotation files
img_file_list = [path for path in Path(input_image_path).rglob('*') if path.suffix.lower() in ['.png', '.jpg']]
txt_file_list = [path for path in Path(input_label_path).rglob('*') if path.suffix.lower() == '.txt']

if len(img_file_list) <= 0:
    print('No images files found in the directory...')
    sys.exit(0)
else:
    print(f'Number of image files: {len(img_file_list)}')
    print(f'Number of annotation files: {len(txt_file_list)}')

# Define paths to image and annotation folders
workspace = os.path.abspath(os.path.join(data_path, 'training_data'))
train_img_path = os.path.join(workspace, 'train', 'images')
train_txt_path = os.path.join(workspace, 'train', 'labels')
validation_img_path = os.path.join(workspace, 'validation', 'images')
validation_txt_path = os.path.join(workspace, 'validation', 'labels')

# Create folders if they don't already exist
for dir_path in [train_img_path, train_txt_path, validation_img_path, validation_txt_path]:
   if not os.path.exists(dir_path):
      os.makedirs(dir_path)
      print(f'Created folder at: {dir_path}.')

# Determine number of files to move to each folder
total_image_files = len(img_file_list)
number_of_files_to_train = int(total_image_files*train_percentage)
number_of_validation_files = total_image_files - number_of_files_to_train

print(f'Total Images to be used for training: {number_of_files_to_train}')
print(f'Total Images to be used for validation: {number_of_validation_files}')

# # Select files randomly and copy them to train or val folders
for i, set_num in enumerate([number_of_files_to_train, number_of_validation_files]):
  for ii in range(set_num):
    img_path = random.choice(img_file_list)
    img_fn = img_path.name
    base_fn = img_path.stem
    txt_fn = base_fn + '.txt'
    txt_path = os.path.join(input_label_path,txt_fn)

    if i == 0: # Copy first set of files to train folders
      new_img_path, new_txt_path = train_img_path, train_txt_path
    elif i == 1: # Copy second set of files to the validation folders
      new_img_path, new_txt_path = validation_img_path, validation_txt_path

    shutil.copy(img_path, os.path.join(new_img_path,img_fn))
    #os.rename(img_path, os.path.join(new_img_path,img_fn))
    if os.path.exists(txt_path): # If txt path does not exist, this is a background image, so skip txt file
      shutil.copy(txt_path,os.path.join(new_txt_path,txt_fn))
      #os.rename(txt_path,os.path.join(new_txt_path,txt_fn))

    img_file_list.remove(img_path)

# Extract class names to create yaml config
with open(input_class_names_path, 'r') as class_names_file:
    classes = []
    for line in class_names_file.readlines():
      if len(line.strip()) == 0: continue
      classes.append(line.strip())

number_of_classes = len(classes)

# Data to write to config.yaml

data = {
    'path': os.path.join(workspace),
    'train': os.path.join(workspace, 'train', 'images'),
    'val': os.path.join(workspace, 'validation','images'),
    'nc': number_of_classes,
    'names': {key:value for key, value in enumerate(classes)}
}

import yaml

# Write data to YAML file
with open(config_yaml, 'w') as f:
    yaml.dump(data, f, sort_keys=False)

print(f'Created config file at: {config_yaml}')
model = Path(model_path)
model_name = model.name
print(f'Training Model {model_name} at: {model_path}')

try:
    train = command.run([
        yolo_exec, 'task=segment', 'mode=train', f'data={config_yaml}', f'model={model_path}', 'patience=100', 'epochs=200', 'imgsz=640'
    ], text=True, check=True)
except command.CalledProcessError as e:
    print(f"Failed to train: {e}")
