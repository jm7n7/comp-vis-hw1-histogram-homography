import cv2
import numpy as np
import os

# --- Define Constants ---
IMAGE_DIR = "SelectedImages"
KMEANS_CLASS_DIR = "kmeans_images"
KMEANS_GLOBAL_DIR = "kmeans_images_global"
COMPARISON_DIR = "comparison_images_full"
PADDING = 10  # 10 pixels of white padding

def create_comparison_image(original, k64_class, k128_class, k64_global, k128_global):
    """
    Creates a side-by-side comparison image from five input images
    with labels and padding.
    """
    images = [original.copy(), k64_class.copy(), k128_class.copy(), k64_global.copy(), k128_global.copy()]
    labels = ["Original", "K=64 (Class)", "K=128 (Class)", "K=64 (Global)", "K=128 (Global)"]
    
    # Add text labels to each image
    labeled_images = []
    for img, label in zip(images, labels):
        # Create a white canvas for the label
        h, w, _ = img.shape
        label_canvas = np.full((50, w, 3), 255, dtype=np.uint8)
        # Center the text on the canvas
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_x = (w - text_size[0]) // 2
        cv2.putText(label_canvas, label, (text_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        # Combine the label canvas and the image
        labeled_images.append(cv2.vconcat([label_canvas, img]))

    # Create padding
    height = labeled_images[0].shape[0]
    padding_canvas = np.full((height, PADDING, 3), 255, dtype=np.uint8)
    
    # Stitch all five images together with padding
    final_image = cv2.hconcat([
        labeled_images[0], padding_canvas,
        labeled_images[1], padding_canvas,
        labeled_images[2], padding_canvas,
        labeled_images[3], padding_canvas,
        labeled_images[4]
    ])
    
    return final_image

def main():
    """
    Main function to find original and processed images and create comparisons.
    """
    if not os.path.exists(COMPARISON_DIR):
        os.makedirs(COMPARISON_DIR)
        print(f"Created directory: {COMPARISON_DIR}")

    print("\n--- Starting Full Comparison Image Generation ---")
    
    # Walk through the original images directory
    for root, _, files in os.walk(IMAGE_DIR):
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            base_name, _ = os.path.splitext(filename)
            class_name = os.path.basename(root)

            # Define paths for all required images
            original_path = os.path.join(root, filename)
            k64_class_path = os.path.join(KMEANS_CLASS_DIR, class_name, f"{base_name}_k64.png")
            k128_class_path = os.path.join(KMEANS_CLASS_DIR, class_name, f"{base_name}_k128.png")
            k64_global_path = os.path.join(KMEANS_GLOBAL_DIR, class_name, f"{base_name}_k64.png")
            k128_global_path = os.path.join(KMEANS_GLOBAL_DIR, class_name, f"{base_name}_k128.png")

            all_paths = [original_path, k64_class_path, k128_class_path, k64_global_path, k128_global_path]
            
            # Check if all files exist before processing
            if all(os.path.exists(p) for p in all_paths):
                # Load the images
                original_img = cv2.imread(original_path)
                k64_class_img = cv2.imread(k64_class_path)
                k128_class_img = cv2.imread(k128_class_path)
                k64_global_img = cv2.imread(k64_global_path)
                k128_global_img = cv2.imread(k128_global_path)

                all_images = [original_img, k64_class_img, k128_class_img, k64_global_img, k128_global_img]
                
                if any(img is None for img in all_images):
                    print(f"  Warning: Could not read one of the images for {filename}. Skipping.")
                    continue

                # Create the comparison
                comparison_img = create_comparison_image(original_img, k64_class_img, k128_class_img, k64_global_img, k128_global_img)
                
                # Save the final image
                save_dir = os.path.join(COMPARISON_DIR, class_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir, f"{base_name}_full_comparison.png"), comparison_img)
                print(f"  Generated full comparison for {filename}")
            else:
                print(f"  Warning: Missing one or more processed images for {filename}. Skipping.")
    
    print("--- Full Comparison Image Generation Complete ---")


if __name__ == "__main__":
    main()

