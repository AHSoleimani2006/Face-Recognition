import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

def compute_svd(A):
    """Calculate the mean vector, centers the matrix, and applies SVD."""
    mean_vector = np.mean(A, axis=1, keepdims=True)
    A_centered = A - mean_vector
    U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)
    return U, S, Vt, mean_vector, A_centered

def reconstruct_face(person_idx, k_values, U, mean_vec, original_matrix, img_shape):
    """Reconstruct a specific face using different rank-k approximations."""
    original_face = original_matrix[:, person_idx]
    centered_face = original_face.reshape(-1, 1) - mean_vec

    plt.figure(figsize=(12, 5))
    plt.subplot(1, len(k_values) + 1, 1)
    plt.imshow(original_face.reshape(img_shape), cmap='gray')
    plt.title("Original Face")
    plt.axis('off')

    for i, k in enumerate(k_values):
        U_k = U[:, :k]
        weights = U_k.T @ centered_face
        reconstructed_face = (U_k @ weights) + mean_vec

        plt.subplot(1, len(k_values) + 1, i + 2)
        plt.imshow(reconstructed_face.reshape(img_shape), cmap='gray')
        plt.title(f"Reconstruction (k={k})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def recognize_faces(k, U, mean_vec, A_centered, train_labels, A_original, image_shape, test_folder_path):
    """Project test images into the k-dimensional space and find the closest match."""
    U_k = U[:, :k]
    Weights_Train = U_k.T @ A_centered
    test_image_paths = glob.glob(f"{test_folder_path}/*.jpg")

    if not test_image_paths:
        print("No images found for recognition.")
        return

    for test_path in test_image_paths:
        img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img_resized = cv2.resize(img, (image_shape[1], image_shape[0]))
        img_centered = img_resized.flatten().reshape(-1, 1) - mean_vec
        weight_test = U_k.T @ img_centered

        # Euclidean distance matching
        diff = Weights_Train - weight_test
        distances = np.linalg.norm(diff, axis=0)

        min_dist_idx = np.argmin(distances)
        predicted_label = train_labels[min_dist_idx]
        min_distance = distances[min_dist_idx]
        matched_img_vec = A_original[:, min_dist_idx]

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img_resized, cmap='gray')
        plt.title(f"Test Image\n({test_path.split(r'/')[-1]})")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(matched_img_vec.reshape(image_shape), cmap='gray')
        plt.title(f"Identified: Person {predicted_label}\n(Dist: {min_distance:.2f})")
        plt.axis('off')
        plt.show()