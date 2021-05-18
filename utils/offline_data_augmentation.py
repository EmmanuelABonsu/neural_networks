import random
import cv2
import os

"""

This script is used to generate more images from a set of images. It randomly selects an image
from a given folder, applies some augmentation and writes it to an output folder
"""

from albumentations import (
    Compose, HorizontalFlip,
    Rotate, RandomCrop, RandomBrightnessContrast
)

# Augmentation to apply to random image selected
transform = Compose([
    Rotate(limit=20),
    RandomBrightnessContrast(p=0.5),
    HorizontalFlip(),
    RandomCrop(220, 220, always_apply=False, p=1.0)
])

folder_path = '/INPUT_FOLDER_PATH'
num_files_desired = 6000

to_folder = '/OUTPUT_FOLDER_PATH'
# find all files paths from the folder
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

num_generated_files = 1
while num_generated_files <= num_files_desired:
    # random image from the folder
    image_path = random.choice(images)

    # Read an image with OpenCV and convert it to the RGB colorspace
    image = cv2.imread(image_path)
    # print(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Augment an image
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    temp = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)

    new_file_path = '%s/augmented_image_%s.jpg' % (to_folder, num_generated_files)
    print("Writing File Number", num_generated_files)

    # write image to folder
    cv2.imwrite(new_file_path, temp)
    num_generated_files += 1
