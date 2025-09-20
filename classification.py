import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 1. Define Constants ---
# Ensure these match the directories and parameters from your kmeans.py script
FEATURE_DIR = "features"
IMAGE_DIR = "SelectedImages"
# The number of classes is needed for validation.
# The assignment specifies using the first 15 classes.
NUM_CLASSES = 15

# --- 2. Function to Generate Labels ---
def generate_labels():
    """
    Generates a list of numerical labels for the images based on the folder structure.
    This version is more robust as it dynamically counts images instead of using a
    hardcoded number. The order must match the order used to generate the
    feature histograms in kmeans.py.
    """
    labels = []
    # Sort the class directories to ensure a consistent order
    class_dirs = sorted([d for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))])

    # NEW: Check if the number of classes matches the assignment
    if len(class_dirs) != NUM_CLASSES:
        print(f"  ⚠️ Warning: Found {len(class_dirs)} class directories, but expected {NUM_CLASSES}.")
        print("  This might cause a mismatch if kmeans.py was run with a different dataset.")

    for i, class_name in enumerate(class_dirs):
        class_path = os.path.join(IMAGE_DIR, class_name)
        # Dynamically count the number of valid image files in the directory
        num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        # Add the label 'i' for each image found in this class
        labels.extend([i] * num_images)

    return np.array(labels), class_dirs

# --- 3. Main Classification Function (Updated for Leave-One-Out) ---
def run_classification(k_value, num_neighbors=1):
    """
    Loads data, trains a KNN classifier using Leave-One-Out cross-validation,
    and evaluates its performance.
    """
    print(f"\n--- Starting Classification for K={k_value} Clusters (Leave-One-Out) ---")

    # Load the data matrix
    feature_file = os.path.join(FEATURE_DIR, f"global_histograms_k{k_value}.npy")
    if not os.path.exists(feature_file):
        print(f"  ❌ Error: Feature file not found at {feature_file}")
        print("  Please make sure you have run kmeans.py successfully.")
        return

    print(f"  Loading feature data from {feature_file}...")
    features = np.load(feature_file)

    # Generate labels
    print("  Generating labels...")
    labels, class_names = generate_labels()

    # Verify data shapes
    if features.shape[0] != len(labels):
        print(f"  ❌ Error: Mismatch between features ({features.shape[0]}) and labels ({len(labels)}).")
        print("  Ensure that the IMAGE_DIR contains the exact same images as when you ran kmeans.py.")
        return

    print(f"  Total samples to process: {features.shape[0]}")

    # Initialize the KNN classifier and the LeaveOneOut cross-validator
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    loo = LeaveOneOut()

    # Use cross_val_predict to get predictions for each sample when it's used as a test set.
    # This is an efficient way to perform LOO cross-validation.
    print(f"  Training and predicting with KNN (n_neighbors={num_neighbors}) using Leave-One-Out...")
    y_pred = cross_val_predict(knn, features, labels, cv=loo)
    print("  ...LOO prediction complete.")

    # Calculate and report accuracy
    accuracy = accuracy_score(labels, y_pred)
    print("\n  --- Results ---")
    print(f"  ✅ Accuracy for K={k_value} model: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Generate and save the confusion matrix
    print("  Generating confusion matrix...")
    cm = confusion_matrix(labels, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix for K={k_value}, Neighbors={num_neighbors} (Leave-One-Out)', fontsize=14)
    plt.tight_layout()

    # Save the figure
    output_filename = f"confusion_matrix_k{k_value}_n{num_neighbors}_loo.png"
    plt.savefig(output_filename)
    print(f"  Saved confusion matrix to {output_filename}")
    plt.close() # Close the plot to free up memory


# --- 4. Main Execution Block ---
if __name__ == "__main__":
    # Run the classification for both k=64 and k=128
    # Using k=1 for the nearest neighbor is a common baseline.
    run_classification(k_value=64, num_neighbors=1)
    run_classification(k_value=128, num_neighbors=1)
