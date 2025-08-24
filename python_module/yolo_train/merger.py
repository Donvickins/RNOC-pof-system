import os

workspace = os.path.join(os.getcwd(), 'merge')

detect_label_dir = os.path.join(workspace, 'detect_label')
segment_label_dir = os.path.join(workspace, 'segment_label')
merged_label_dir = os.path.join(workspace, 'all_labels_in_seg_form')

for filename in os.listdir(detect_label_dir):
    if filename.endswith('.txt'):
        detect_file_path = os.path.join(detect_label_dir, filename)
        segment_file_path = os.path.join(segment_label_dir, filename)
        save_path = os.path.join(merged_label_dir, filename)

        with open(detect_file_path, 'r') as detect_label_file:
            detect_content = detect_label_file.read()

        if os.path.exists(segment_file_path):
            with open(segment_file_path, 'r') as segment_label_file:
                segment_content = segment_label_file.read()
            with open(save_path, 'w') as merged_file:
                merged_file.write(detect_content + segment_content)
        else:
            with open(save_path, 'w') as merged_file:
                merged_file.write(detect_content)

print("All files merged successfully")