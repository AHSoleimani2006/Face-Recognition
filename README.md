# Eigenface Facial Recognition System

This project implements a facial recognition and image reconstruction system using advanced linear algebra techniques. By applying Singular Value Decomposition (SVD), the system extracts the most critical visual features from a dataset to identify individuals, even when presented with highly noisy or damaged test images.

## Data Processing
The dataset consists of 400 `.pgm` grayscale images, representing 40 individuals with 10 images each[cite: 9]. The pixel intensities range from 0 to 255[cite: 9]. To process this data mathematically, each 2D image array is flattened into a single large 1D vector[cite: 9]. These vectors are stacked to form a matrix $A$ containing 400 columns, where each column represents a single face[cite: 9].

## Mathematical Methodology

### 1. Mean Centering
To highlight the differences between individuals rather than their commonalities, a "mean face" is calculated by averaging all images[cite: 10]. This mean face is subtracted from every column in matrix $A$ to create a centered matrix ($A_{centered}$)[cite: 10].

### 2. SVD and Rank-k Approximation
Matrix $A_{centered}$ has a full rank of 400[cite: 10]. To compress the data and eliminate noise, Singular Value Decomposition is applied, and a Rank-$k$ approximation is used[cite: 10]. 
* The $S$ matrix contains singular values sorted in descending order ($\lambda_{i}=\sigma_{i}^{2}$), dictating the importance of each feature[cite: 10]. 
* The $U$ matrix contains the eigenvectors, also known as Eigenfaces[cite: 10]. By retaining only the first $k$ columns of $U$ (the largest singular values), the system isolates primary facial structures and discards smaller values that represent noise or insignificant details[cite: 10]. This minimizes the Euclidean distance (Frobenius norm) error between the approximation and the original images[cite: 10].

### 3. Face Reconstruction
A specific face can be mathematically reconstructed as a linear combination of the mean face and the weighted Eigenfaces[cite: 11]:

$$Face_{approx}=Mean~Face+\sum_{i=1}^{k}w_{i}\cdot u_{i}$$

Testing on image 55 (Person 5) demonstrates the impact of $k$[cite: 11]:
* **$k=3$:** Yields a very blurry image showing only basic head shape and major shadows without clear eyes or mouth details[cite: 11].
* **$k=6$:** Produces a clearer, albeit ghost-like, image with more distinct lighting details, though still insufficient for exact identification[cite: 11].

### 4. Facial Recognition
To identify a new, noisy test image, it is projected into the calculated $k$-dimensional sub-space[cite: 10, 11]. The system calculates the Euclidean distance between the test image's weights and the weights of the known training images[cite: 10]. The system accurately matched heavily damaged test images to Person 15, Person 10, and Person 1 using a Rank-5 ($k=5$) approximation, which provided sufficient precision for identification[cite: 11].
