import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_sift_matches(img1, img2):
    """
    Finds SIFT keypoints and matches between two images.
    
    Args:
        img1: The first image (source).
        img2: The second image (destination).
        
    Returns:
        A tuple containing:
        - all_matches: A list of all raw (best) matches.
        - good_matches: A list of good matches after Lowe's ratio test.
        - kp1: Keypoints from the first image.
        - kp2: Keypoints from the second image.
    """
    try:
        sift = cv2.SIFT_create()
    except cv2.error as e:
        raise ImportError("Could not create SIFT object. Make sure you have 'opencv-contrib-python' installed.") from e

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches_knn = bf.knnMatch(des1, des2, k=2)

    all_matches = [m for m, n in matches_knn]
    good_matches = [m for m, n in matches_knn if m.distance < 0.75 * n.distance]
            
    return all_matches, good_matches, kp1, kp2

def compute_homography_svd(matches, kp_src, kp_dst):
    """
    Computes the homography matrix from a set of matches using SVD.
    """
    if len(matches) < 4:
        raise ValueError("Not enough matches found to compute homography. Need at least 4 pairs.")

    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    
    A = []
    for i in range(len(src_pts)):
        x, y = src_pts[i]
        xp, yp = dst_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    A = np.asarray(A)

    print(f"Shape of matrix A: {A.shape}")

    # Perform SVD on A
    U, S, Vh = np.linalg.svd(A)
    
    # --- Solution Details for Assignment ---
    print("\n[Q] What are the singular values of A?")
    print("Singular values (S):", S)
    print(f"The smallest singular value is {S[-1]:.4f}, which corresponds to the 'zero' singular value in the ideal case.")

    # The solution h is the last row of Vh, which corresponds to the smallest singular value.
    h = Vh[-1, :]

    print("\n[Q] What is the null space solution to Ah=0?")
    print("The non-zero vector h (last row of Vh) is:", h)
    
    H = h.reshape((3, 3))
    H = H / H[2, 2]
    
    return H

def count_inliers(H, matches, kp_src, kp_dst, threshold=4.0):
    """
    Counts the number of inlier matches for a given homography.
    """
    inlier_count = 0
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Project source points to destination plane using the homography
    projected_dst_pts = cv2.perspectiveTransform(src_pts, H)

    # Calculate the distance between projected points and actual destination points
    for i in range(len(dst_pts)):
        dist = np.linalg.norm(dst_pts[i] - projected_dst_pts[i])
        if dist < threshold:
            inlier_count += 1
            
    return inlier_count

def create_mosaic(img_warp, img_base, H):
    """
    Warps img_warp and stitches it onto img_base to create a mosaic.
    """
    h_warp, w_warp = img_warp.shape[:2]
    h_base, w_base = img_base.shape[:2]

    corners_warp = np.float32([[0, 0], [0, h_warp], [w_warp, h_warp], [w_warp, 0]]).reshape(-1, 1, 2)
    corners_transformed = cv2.perspectiveTransform(corners_warp, H)
    corners_base = np.float32([[0, 0], [0, h_base], [w_base, h_base], [w_base, 0]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners_base, corners_transformed), axis=0)

    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
    
    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    canvas_size = (x_max - x_min, y_max - y_min)
    warped_img = cv2.warpPerspective(img_warp, T.dot(H), canvas_size)

    mosaic = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    mosaic[-y_min:-y_min + h_base, -x_min:-x_min + w_base] = img_base
    
    mask = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY) > 0
    mosaic[mask] = warped_img[mask]
    
    return mosaic

def run_and_visualize_analysis(img_base_rgb, img_warp_rgb, mode, common_data):
    """
    Runs a specific homography analysis (all matches vs. best M) and generates plots.
    
    Args:
        img_base_rgb: The base image in RGB format.
        img_warp_rgb: The image to warp in RGB format.
        mode: Either 'all' to use all good matches or an integer (e.g., 8) to use the best M matches.
        common_data: A dictionary containing pre-computed matches and keypoints.
    """
    all_matches = common_data['all_matches']
    good_matches = common_data['good_matches']
    kp_warp = common_data['kp_warp']
    kp_base = common_data['kp_base']
    
    title_suffix = ""
    matches_for_H = []

    if mode == 'all':
        title_suffix = "(from All Good Matches)"
        matches_for_H = good_matches
        print(f"Using all {len(good_matches)} good matches for homography.")
    elif isinstance(mode, int):
        if len(good_matches) < mode:
            raise ValueError(f"Not enough good matches to select the best {mode}. Found only {len(good_matches)}.")
        title_suffix = f"(from Best {mode} Matches)"
        matches_for_H = sorted(good_matches, key=lambda x: x.distance)[:mode]
        print(f"Using the best {mode} matches for homography.")
    
    # Compute Homography
    H = compute_homography_svd(matches_for_H, kp_warp, kp_base)
    print("Final 3x3 Homography Matrix (H):\n", H)

    # --- Inlier Calculation for Assignment ---
    inlier_count = count_inliers(H, good_matches, kp_warp, kp_base)
    print("\n[Q] How many matched SIFT points agree with this homography?")
    print(f"Number of inliers: {inlier_count} out of {len(good_matches)} total good matches.")
    
    # Create visualizations
    img_all_matches = cv2.drawMatches(img_warp_rgb, kp_warp, img_base_rgb, kp_base, all_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_good_matches = cv2.drawMatches(img_warp_rgb, kp_warp, img_base_rgb, kp_base, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_used_matches = cv2.drawMatches(img_warp_rgb, kp_warp, img_base_rgb, kp_base, matches_for_H, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    mosaic = create_mosaic(img_warp_rgb, img_base_rgb, H)

    # Plotting
    plt.figure(figsize=(20, 15))
    plt.suptitle(f"Homography Analysis {title_suffix}", fontsize=20)

    plt.subplot(2, 2, 1)
    plt.imshow(img_all_matches)
    plt.title(f'All {len(all_matches)} Raw SIFT Matches')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img_good_matches)
    plt.title(f'All {len(good_matches)} Good Matches (Ratio Test)')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(img_used_matches)
    plt.title(f'Matches Used for Homography: {len(matches_for_H)}')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(mosaic)
    plt.title('Stitched Mosaic')
    plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])


def main():
    """Main function to run the complete homography analysis."""
    try:
        # Load images once
        img_base_bgr = cv2.imread('my_image-1.jpg')
        img_warp_bgr = cv2.imread('my_image-3.jpg')

        if img_base_bgr is None or img_warp_bgr is None:
            raise FileNotFoundError("Could not load 'sift-1.jpg' or 'sift-2.jpg'.")

        img_base_rgb = cv2.cvtColor(img_base_bgr, cv2.COLOR_BGR2RGB)
        img_warp_rgb = cv2.cvtColor(img_warp_bgr, cv2.COLOR_BGR2RGB)
        
        # Pre-compute SIFT matches once to be efficient
        print("--- Finding SIFT Matches (once) ---")
        img_base_gray = cv2.cvtColor(img_base_bgr, cv2.COLOR_BGR2GRAY)
        img_warp_gray = cv2.cvtColor(img_warp_bgr, cv2.COLOR_BGR2GRAY)
        all_matches, good_matches, kp_warp, kp_base = find_sift_matches(img_warp_gray, img_base_gray)
        
        common_data = {
            'all_matches': all_matches,
            'good_matches': good_matches,
            'kp_warp': kp_warp,
            'kp_base': kp_base
        }
        
        print(f"Total raw matches found: {len(all_matches)}")
        print(f"Total good matches found (after ratio test): {len(good_matches)}")

        # --- Run Analysis for Task 1: Over-determined solution ---
        print("\n\n--- Running Task 1: Homography from ALL Good Matches ---")
        run_and_visualize_analysis(img_base_rgb, img_warp_rgb, mode='all', common_data=common_data)

        # --- Run Analysis for Task 2: Best M=8 solution ---
        print("\n\n--- Running Task 2: Homography from Best M=8 Matches ---")
        run_and_visualize_analysis(img_base_rgb, img_warp_rgb, mode=8, common_data=common_data)
        
        # Display all generated figures
        plt.show()

    except (ImportError, ValueError, FileNotFoundError) as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

