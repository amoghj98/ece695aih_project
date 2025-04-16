# partitions data_50 folder into two folders of separated forward pass data

import os
import shutil

# Define source and destination directories
source_folder = 'data_50'
destination_folder = 'data_50_olmo'

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
    print(f"Created folder: {destination_folder}")

# Iterate through the files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file name starts with 'forward_pass_best_of_n' and is a .csv file
    if filename.startswith("forward_pass_best_of_n") and filename.endswith(".csv"):
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        
        # Move the file
        shutil.move(source_file, destination_file)
        print(f"Moved {filename} to {destination_folder}")

print("Script completed.")

