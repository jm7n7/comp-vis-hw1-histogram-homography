import os
import random
import shutil

# Set random seed for reproducibility
random.seed(2025)

dataset_dir = 'Dataset'
output_dir = 'SelectedImages'
num_folders = 15
num_images_per_folder = 40

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

folders = sorted([f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))])[:num_folders]

for folder in folders:
    folder_path = os.path.join(dataset_dir, folder)
    images = [img for img in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, img))]
    selected_images = random.sample(images, num_images_per_folder)
    output_folder = os.path.join(output_dir, folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for img in selected_images:
        src = os.path.join(folder_path, img)
        dst = os.path.join(output_folder, img)
        shutil.copy2(src, dst)
