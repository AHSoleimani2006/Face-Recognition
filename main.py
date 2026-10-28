from data_loader import load_dataset
from recognizer import compute_svd, reconstruct_face, recognize_faces

# --- ADJUSTABLE PARAMETERS ---
DATASET_PATH = "Dataset"
TEST_FOLDER_PATH = "Dataset/recognition"
RECONSTRUCTION_INDEX = 40  # Index of the image to test reconstruction
RECONSTRUCTION_K_VALUES = [3, 6]  # Number of eigenfaces to use for reconstruction
RECOGNITION_K_VALUE = 5  # Rank-k value used for distance-based recognition


# -----------------------------

def main():
    print("Loading dataset...")
    A, labels, image_shape = load_dataset(DATASET_PATH)

    if A is None:
        return

    print("Computing SVD...")
    U, S, Vt, mean_vector, A_centered = compute_svd(A)

    print("Running facial reconstruction test...")
    reconstruct_face(RECONSTRUCTION_INDEX, RECONSTRUCTION_K_VALUES, U, mean_vector, A, image_shape)

    print("Running facial recognition on test images...")
    recognize_faces(RECOGNITION_K_VALUE, U, mean_vector, A_centered, labels, A, image_shape, TEST_FOLDER_PATH)


if __name__ == "__main__":
    main()