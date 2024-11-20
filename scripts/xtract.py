import os
import shutil

def filter_dataset(source_folder, destination_folder, num_data):
    # Ensure destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    
    # Get all images and their corresponding text files
    files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]
    
    # Sort files to ensure consistency in selection
    files.sort()
    
    # Select the desired number of data pairs
    selected_files = files[:num_data]
    
    for img_file in selected_files:
        # Copy image file
        shutil.copy(os.path.join(source_folder, img_file), destination_folder)
        
        # Copy corresponding .txt file
        txt_file = img_file.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(source_folder, txt_file)):
            shutil.copy(os.path.join(source_folder, txt_file), destination_folder)

# Configuration
source_folder = '../downloaded_images/person'  # Replace with the path to your dataset folder
destination_folder = '../downloaded_images/person-xtract'  # Replace with the desired destination folder
num_data = 4500  # Number of data pairs to extract

filter_dataset(source_folder, destination_folder, num_data)

print("done")
