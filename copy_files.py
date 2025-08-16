import os
import shutil
import random

def copy_random_files(source_folder, destination_folder, number_of_files):
    # Ensure source folder exists
    if not os.path.isdir(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return

    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Get all files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # Handle case where requested number is more than available files
    if number_of_files > len(files):
        print(f"Requested {number_of_files} files, but only {len(files)} available.")
        number_of_files = len(files)

    # Randomly select files
    selected_files = random.sample(files, number_of_files)

    # Copy files
    for file_name in selected_files:
        src_path = os.path.join(source_folder, file_name)
        dst_path = os.path.join(destination_folder, file_name)
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {file_name}")

    print(f"\nSuccessfully copied {len(selected_files)} files to '{destination_folder}'.")

# ==== Configuration ====
source = "data/CelebDF-V2/Celeb-synthesis"
destination = "data_subset/fake"
num_files_to_copy = 100  # Change this to however many files you want to copy

# ==== Run the script ====
copy_random_files(source, destination, num_files_to_copy)
