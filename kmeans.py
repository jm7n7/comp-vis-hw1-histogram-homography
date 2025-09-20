import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings
# Suppress matplotlib warnings
warnings.filterwarnings("ignore", message="More than 20 figures have been opened.")

#--- SETUP AND CONSTANTS
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU is available. Using device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ No GPU found. Using device: CPU")
print("-" * 60)

IMAGE_DIR = "SelectedImages"
ORIGINAL_HEATMAP_DIR = "Original_Heatmaps"
KMEANS_IMAGE_DIR = "kmeans_images"
KMEANS_GLOBAL_IMAGE_DIR = "kmeans_images_global"
QUANTIZED_HISTOGRAM_DIR = "Quantized_Histograms"
FEATURE_DIR = "features"

# Create all necessary output directories
for dir_path in [
    ORIGINAL_HEATMAP_DIR,
    KMEANS_IMAGE_DIR,
    KMEANS_GLOBAL_IMAGE_DIR,
    QUANTIZED_HISTOGRAM_DIR,
    FEATURE_DIR
]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

#--- FUNCTION FOR ORIGINAL IMAGE HEATMAP GENERATION
def generate_original_image_heatmaps():
    """
    Generates a visual 2D histogram (heatmap) of the Hue and Saturation
    channels for each ORIGINAL image. Saves output to ORIGINAL_HEATMAP_DIR.
    """
    print("\n--- Generating Original Image HSV Heatmaps ---")
    class_names = sorted(os.listdir(IMAGE_DIR))
    for class_name in class_names:
        class_path = os.path.join(IMAGE_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"  Processing class: {class_name}")
        # Create a subdirectory for the class in the output folder
        output_class_path = os.path.join(ORIGINAL_HEATMAP_DIR, class_name)
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        image_files = os.listdir(class_path)
        for filename in image_files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_path, filename)
                img = cv2.imread(image_path)
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # Extract H and S channels
                h, s, _ = cv2.split(img_hsv)

                # Create 2D histogram
                hist = cv2.calcHist([h, s], [0, 1], None, [180, 256], [0, 180, 0, 256])

                # Plotting
                plt.figure(figsize=(8, 6))
                plt.imshow(hist, interpolation='nearest', aspect='auto', cmap='viridis')
                plt.title(f'HSV Color Histogram for {filename}')
                plt.xlabel('Saturation')
                plt.ylabel('Hue')
                plt.colorbar()

                # Save the figure
                base_name, _ = os.path.splitext(filename)
                save_path = os.path.join(output_class_path, f'{base_name}_heatmap.png')
                plt.savefig(save_path)
                plt.close() # Close the plot to free up memory

    print("--- Original Image Heatmap Generation Complete ---")


#--- FUNCTION TO GENERATE 2X2 QUANTIZED BAR GRAPH PLOTS
def plot_color_bar_graph(ax, image_path, title):
    """
    Helper function to generate a single bar graph of a quantized image's colors.
    The bars are colored with the actual pixel color they represent.
    """
    if not os.path.exists(image_path):
        ax.text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=10, color='red')
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        return

    # Load image (OpenCV loads in BGR format)
    img = cv2.imread(image_path)
    # Reshape to a list of pixels
    pixels = img.reshape(-1, 3)
    # Get unique BGR colors and their counts
    unique_colors_bgr, counts = np.unique(pixels, axis=0, return_counts=True)

    # Sort everything by count in descending order for a cleaner look
    sorted_indices = np.argsort(counts)[::-1]
    unique_colors_bgr = unique_colors_bgr[sorted_indices]
    counts = counts[sorted_indices]

    # Convert BGR colors to RGB for Matplotlib and normalize to [0, 1] range
    unique_colors_rgb = unique_colors_bgr[:, ::-1] / 255.0

    # Plotting
    bar_indices = np.arange(len(unique_colors_rgb))
    ax.bar(bar_indices, counts, color=unique_colors_rgb, width=1.0)

    # Formatting
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Pixel Count")
    ax.set_xticks([])  # X-ticks have no meaning here
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)


def generate_quantized_bar_graphs():
    """
    Creates a composite 2x2 image for each source image, showing bar graphs
    of the color palettes from four different K-Means quantization models.
    """
    print("\n--- Generating Quantized Color Bar Graph Visualizations ---")
    class_names = sorted([d for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))])

    for class_name in class_names:
        print(f"  Processing class: {class_name}")
        class_path = os.path.join(IMAGE_DIR, class_name)
        output_class_path = os.path.join(QUANTIZED_HISTOGRAM_DIR, class_name)
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        image_files = sorted([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        for filename in image_files:
            base_name, _ = os.path.splitext(filename)
            print(f"    Generating plot for {filename}...")
            output_file = os.path.join(output_class_path, f"{base_name}_histograms.png")

            # Define the paths for the four quantized images to be plotted
            paths = {
                "class_64": os.path.join(KMEANS_IMAGE_DIR, class_name, f"{base_name}_k64.png"),
                "class_128": os.path.join(KMEANS_IMAGE_DIR, class_name, f"{base_name}_k128.png"),
                "global_64": os.path.join(KMEANS_GLOBAL_IMAGE_DIR, class_name, f"{base_name}_k64.png"),
                "global_128": os.path.join(KMEANS_GLOBAL_IMAGE_DIR, class_name, f"{base_name}_k128.png")
            }
            titles = {
                "class_64": "Class-Specific K=64",
                "class_128": "Class-Specific K=128",
                "global_64": "Global K=64",
                "global_128": "Global K=128"
            }

            # Create the 2x2 plot grid
            fig, axes = plt.subplots(2, 2, figsize=(14, 11))
            fig.suptitle(f'Quantized Color Histograms for: {filename}', fontsize=18, fontweight='bold')

            # Populate the grid with plots
            plot_color_bar_graph(axes[0, 0], paths["class_64"], titles["class_64"])
            plot_color_bar_graph(axes[0, 1], paths["class_128"], titles["class_128"])
            plot_color_bar_graph(axes[1, 0], paths["global_64"], titles["global_64"])
            plot_color_bar_graph(axes[1, 1], paths["global_128"], titles["global_128"])

            # Final adjustments and saving
            plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle
            plt.savefig(output_file)
            plt.close(fig)

    print("--- Quantized Bar Graph Generation Complete ---")


#--- FUNCTION TO TRAIN A K-MEANS MODEL (CLASS-SPECIFIC)
def train_kmeans_class_model():
    """
    Trains a separate K-Means model for each image class.
    """
    print("\n--- Training Class-Specific K-Means Models ---")
    class_names = sorted(os.listdir(IMAGE_DIR))
    all_class_models = {}
    for class_name in class_names:
        class_path = os.path.join(IMAGE_DIR, class_name)
        if not os.path.isdir(class_path): continue
        print(f"  Processing class: {class_name}")
        pixel_data = []
        image_files = os.listdir(class_path)
        for filename in image_files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_path, filename)
                img = cv2.imread(image_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pixels = img_rgb.reshape(-1, 3)
                if pixels.shape[0] > 1000:
                    sample_indices = np.random.choice(pixels.shape[0], 500, replace=False)
                    pixel_data.append(pixels[sample_indices, :])
                else:
                    pixel_data.append(pixels)
        class_pixels = np.vstack(pixel_data)
        all_class_models[class_name] = {}
        for k in [64, 128]:
            print(f"    Training K-Means with K={k}...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(class_pixels)
            all_class_models[class_name][k] = kmeans
            print(f"    ✅ Done.")
    print("--- Class-Specific K-Means Training Complete ---")
    return all_class_models

#--- FUNCTION TO TRAIN A SINGLE GLOBAL K-MEANS MODEL
def train_kmeans_global_model():
    """
    Trains a single K-Means model on pixels from ALL images across ALL classes.
    """
    print("\n--- Training Global K-Means Models ---")
    all_pixels = []
    class_names = sorted(os.listdir(IMAGE_DIR))

    for class_name in class_names:
        class_path = os.path.join(IMAGE_DIR, class_name)
        if not os.path.isdir(class_path): continue
        print(f"  Loading pixels from class: {class_name}")
        image_files = os.listdir(class_path)
        for filename in image_files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_path, filename)
                img = cv2.imread(image_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pixels = img_rgb.reshape(-1, 3)
                # To manage memory, we take a random sample of 1000 pixels from each image
                sample_indices = np.random.choice(pixels.shape[0], 500, replace=False)
                all_pixels.append(pixels[sample_indices, :])

    # Combine all pixel data
    print("  Combining all pixel data...")
    global_pixel_data = np.vstack(all_pixels)
    print(f"  Total pixels for global training: {global_pixel_data.shape[0]}")

    global_models = {}
    for k in [64, 128]:
        print(f"  Training Global K-Means with K={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', verbose=1)
        kmeans.fit(global_pixel_data)
        global_models[k] = kmeans
        print(f"  ✅ Global model for K={k} training complete.")

    print("--- Global K-Means Training Complete ---")
    return global_models


#--- FUNCTION TO GENERATE QUANTIZED IMAGES (CLASS-SPECIFIC)
def generate_quantized_images_class(class_models):
    """
    Generates new images using the CLASS-SPECIFIC models.
    """
    print("\n--- Generating Quantized Images (Class-Specific Models) ---")
    class_names = sorted(os.listdir(IMAGE_DIR))
    for class_name in class_names:
        class_path = os.path.join(IMAGE_DIR, class_name)
        if not os.path.isdir(class_path): continue

        print(f"  Processing class: {class_name}")
        output_class_path = os.path.join(KMEANS_IMAGE_DIR, class_name)
        if not os.path.exists(output_class_path): os.makedirs(output_class_path)

        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_path, filename)
                img = cv2.imread(image_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, _ = img.shape
                pixels = img_rgb.reshape(-1, 3)

                # Use the models specific to this class
                models_for_this_class = class_models.get(class_name)
                if not models_for_this_class:
                    print(f"    - Warning: No class model found for {class_name}. Skipping.")
                    continue

                for k, model in models_for_this_class.items():
                    labels = model.predict(pixels)
                    new_pixels = model.cluster_centers_[labels].astype('uint8')
                    new_img_rgb = new_pixels.reshape(h, w, 3)
                    new_img_bgr = cv2.cvtColor(new_img_rgb, cv2.COLOR_RGB2BGR)
                    base_name, _ = os.path.splitext(filename)
                    save_path = os.path.join(output_class_path, f"{base_name}_k{k}.png")
                    cv2.imwrite(save_path, new_img_bgr)
    print("--- Quantized Image Generation (Class-Specific) Complete ---")

#--- FUNCTION TO GENERATE QUANTIZED IMAGES
def generate_quantized_images_global(global_models):
    """
    Generates new images where each pixel's color is replaced by the
    centroid color of the cluster it belongs to, using the GLOBAL models.
    """
    print("\n--- Generating Quantized Images (Global Models) ---")
    class_names = sorted(os.listdir(IMAGE_DIR))
    for class_name in class_names:
        class_path = os.path.join(IMAGE_DIR, class_name)
        if not os.path.isdir(class_path): continue

        print(f"  Processing class: {class_name}")
        output_class_path = os.path.join(KMEANS_GLOBAL_IMAGE_DIR, class_name)
        if not os.path.exists(output_class_path): os.makedirs(output_class_path)

        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_path, filename)
                img = cv2.imread(image_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, _ = img.shape
                pixels = img_rgb.reshape(-1, 3)

                for k, model in global_models.items():
                    labels = model.predict(pixels)
                    new_pixels = model.cluster_centers_[labels].astype('uint8')
                    new_img_rgb = new_pixels.reshape(h, w, 3)
                    new_img_bgr = cv2.cvtColor(new_img_rgb, cv2.COLOR_RGB2BGR)
                    base_name, _ = os.path.splitext(filename)
                    save_path = os.path.join(output_class_path, f"{base_name}_k{k}.png")
                    cv2.imwrite(save_path, new_img_bgr)
    print("--- Quantized Image Generation (Global) Complete ---")


#--- FUNCTION TO COMPUTE FEATURE HISTOGRAMS
def compute_feature_histograms(model_type, models):
    """
    Computes the feature histogram for every image in the dataset.
    This creates the final N x K data matrix needed for classification.
    """
    print(f"\n--- Computing Feature Histograms ({model_type.capitalize()} Models) ---")
    for k, model in models.items():
        print(f"  Processing model with K={k}")
        all_histograms = []
        class_names = sorted(os.listdir(IMAGE_DIR))
        for class_name in class_names:
            class_path = os.path.join(IMAGE_DIR, class_name)
            if not os.path.isdir(class_path): continue
            for filename in os.listdir(class_path):
                 if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_path, filename)
                    img = cv2.imread(image_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pixels = img_rgb.reshape(-1, 3)
                    labels = model.predict(pixels)
                    hist = np.bincount(labels, minlength=k)
                    all_histograms.append(hist)
            print(f"    Computed feature histograms for class: {class_name}")
        data_matrix = np.vstack(all_histograms)
        save_path = os.path.join(FEATURE_DIR, f"{model_type}_histograms_k{k}.npy")
        np.save(save_path, data_matrix)
        print(f"  ✅ Saved data matrix of shape {data_matrix.shape} to {save_path}")

    print(f"--- Feature Histogram Computation ({model_type.capitalize()}) Complete ---")


#--- MAIN EXECUTION BLOCK
def main():
    """
    Main function to run the entire processing pipeline.
    Uncomment the steps you wish to run.
    """
    # Step 1: Generate visualizations of original image colors
    generate_original_image_heatmaps()

    # Step 2: Train the class-specific K-Means models
    class_models = train_kmeans_class_model()

    # Step 3: Train the global K-Means models
    global_models = train_kmeans_global_model()

    # Step 4: Generate the quantized images using the class-specific models
    generate_quantized_images_class(class_models)

    # Step 5: Generate the quantized images using the global models
    generate_quantized_images_global(global_models)

    # Step 6: Generate the 2x2 comparison plot of quantized image histograms
    generate_quantized_bar_graphs()

    # Step 7: Compute the feature histograms for classification (classification.py)
    compute_feature_histograms('global', global_models)

    print("\n✅ Pipeline Finished.")


if __name__ == "__main__":
    main()
