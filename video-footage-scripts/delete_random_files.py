import os
import random
import shutil

dirPath = "D:/extracted_frames"
destDirectory = "random_selected_images"

# Randomly select 600 frames from the dirPath directory
# Store extracted frames in the destDirectory
filenames = random.sample(os.listdir(dirPath), 600)
for fname in filenames:
    srcpath = os.path.join(dirpath, fname)
    shutil.copy(srcpath, destDirectory)
