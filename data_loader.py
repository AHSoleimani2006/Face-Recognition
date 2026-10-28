import os
import numpy as np
import cv2


def load_dataset(dataset_path):
    """Read PGM images, flattens them, and builds the dataset matrix A."""
    images = []
    labels = []
    img_shape = None

    if not os.path.exists(dataset_path):
        print(f"Error: The path '{dataset_path}' does not exist.")
        return None, None, None

    for person_id in range(1, 41):
        person_folder = os.path.join(dataset_path, f"s{person_id}")
        if not os.path.exists(person_folder):
            continue

        for file_name in sorted(os.listdir(person_folder)):
            if file_name.endswith(".pgm"):
                img_path = os.path.join(person_folder, file_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                if img_shape is None:
                    img_shape = img.shape

                # Convert 2D image into 1D vector
                images.append(img.flatten())
                labels.append(person_id)

    # Transpose so each column represents one image
    A = np.array(images).T
    return A, np.array(labels), img_shape