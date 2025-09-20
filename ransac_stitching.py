import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_sift_matches(img1, img2):
    """
    Finds SIFT keypoints and matches between two images using Lowe's ratio test.
    
    Args:
        img1: The first image (grayscale).
        img2: The second image (grayscale).
        
    Returns:
        A tuple containing:
        - good_matches: A list of good matches after Lowe's ratio test.
        - kp1: Keypoints from the first image.
        - kp2: Keypoints from the second image.
    """
    try:
        # Use SIFT to detect keypoints and descriptors
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Use a Brute-Force Matcher with k-Nearest Neighbors
        bf = cv2.BFMatcher()
        matches_knn = bf.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio test to find good matches
        good_matches = [m for m, n in matches_knn if m.distance < 0.75 * n.distance]
            
        return good_matches, kp1, kp2
    except cv2.error as e:
        raise ImportError("Could not create SIFT object. Make sure you have 'opencv-contrib-python' installed.") from e

def compute_homography_ransac(matches, kp_src, kp_dst):
    """
    Computes the homography matrix from a set of matches using RANSAC.
    This is a robust method that is resilient to outlier matches.

    Args:
        matches: A list of DMatch objects.
        kp_src: Keypoints from the source image.
        kp_dst: Keypoints from the destination image.
    
    Returns:
        A tuple containing:
        - H: The 3x3 homography matrix.
        - mask: A mask indicating which matches are inliers.
    """
    if len(matches) < 4:
        raise ValueError("Not enough matches found to compute homography. Need at least 4 pairs.")

    # Extract the coordinates of the matched keypoints
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Use cv2.findHomography with RANSAC to find the best homography
    # The 5.0 is the RANSAC reprojection threshold. It's the maximum allowed
    # reprojection error to classify a point pair as an inlier.
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H, mask

def create_mosaic(img_warp, img_base, H):
    """
    Warps img_warp and stitches it onto img_base to create a mosaic.
    
    Args:
        img_warp: The image to be warped (e.g., the right image).
        img_base: The base image to stitch onto (e.g., the left image).
        H: The homography matrix that maps points from img_warp to img_base.
        
    Returns:
        The final stitched mosaic image.
    """
    # Get dimensions of the images
    h_warp, w_warp = img_warp.shape[:2]
    h_base, w_base = img_base.shape[:2]

    # Get the corners of the image to be warped
    corners_warp = np.float32([[0, 0], [0, h_warp-1], [w_warp-1, h_warp-1], [w_warp-1, 0]]).reshape(-1, 1, 2)
    
    # Transform the corners of the warped image to the perspective of the base image
    corners_transformed = cv2.perspectiveTransform(corners_warp, H)

    # Get the corners of the base image
    corners_base = np.float32([[0, 0], [0, h_base-1], [w_base-1, h_base-1], [w_base-1, 0]]).reshape(-1, 1, 2)

    # Combine all corner points to find the bounding box of the final mosaic
    all_corners = np.concatenate((corners_base, corners_transformed), axis=0)

    # Find the min and max coordinates to determine the size of the output canvas
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
    
    # Create a translation matrix to shift the mosaic to fit into the new canvas
    # This is needed if x_min or y_min are negative
    translation_dist = [-x_min, -y_min]
    T = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # Calculate the size of the new canvas
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min

    # Warp the source image using the combined transformation (translation + homography)
    # The result is the image warped into its correct place in the final mosaic
    warped_img = cv2.warpPerspective(img_warp, T.dot(H), (canvas_width, canvas_height))

    # Create a new canvas (mosaic) and place the base image on it
    mosaic = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    mosaic[translation_dist[1]:translation_dist[1] + h_base, translation_dist[0]:translation_dist[0] + w_base] = img_base
    
    # Create a mask of the non-black pixels in the warped image
    gray_warped = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, mask_warped = cv2.threshold(gray_warped, 0, 255, cv2.THRESH_BINARY)
    
    # Use the mask to remove the corresponding area in the mosaic
    mosaic[mask_warped > 0] = [0,0,0]

    # Add the warped image to the mosaic
    mosaic = cv2.add(mosaic, warped_img)
    
    return mosaic

def main():
    """Main function to load images, find matches, and create a mosaic."""
    try:
        # Load the images you want to stitch
        # NOTE: Make sure these file names are correct in your directory
        img_left_bgr = cv2.imread('sift-1.jpg') # This will be the base image
        img_right_bgr = cv2.imread('sift-2.jpg') # This will be warped

        if img_left_bgr is None or img_right_bgr is None:
            raise FileNotFoundError("Could not load one or both images. Check file paths.")

        # Convert images to RGB for Matplotlib and grayscale for SIFT
        img_left_rgb = cv2.cvtColor(img_left_bgr, cv2.COLOR_BGR2RGB)
        img_right_rgb = cv2.cvtColor(img_right_bgr, cv2.COLOR_BGR2RGB)
        img_left_gray = cv2.cvtColor(img_left_bgr, cv2.COLOR_BGR2GRAY)
        img_right_gray = cv2.cvtColor(img_right_bgr, cv2.COLOR_BGR2GRAY)
        
        # Find SIFT matches between the two images
        # We find features in the right image and match them to the left image.
        good_matches, kp_right, kp_left = find_sift_matches(img_right_gray, img_left_gray)
        print(f"Found {len(good_matches)} good matches after ratio test.")

        # Compute the homography using RANSAC
        H, mask = compute_homography_ransac(good_matches, kp_right, kp_left)
        
        # The mask returns a list of lists, so we flatten it
        inlier_matches = [m for i, m in enumerate(good_matches) if mask[i][0]]
        print(f"Found {len(inlier_matches)} inlier matches using RANSAC.")

        # Create the mosaic
        mosaic = create_mosaic(img_right_rgb, img_left_rgb, H)

        # --- Visualization ---
        plt.figure(figsize=(20, 10))

        # 1. Show the inlier matches found by RANSAC
        draw_params = dict(matchColor=(0, 255, 0), # draw matches in green
                           singlePointColor=None,
                           matchesMask=mask.ravel().tolist(), # draw only inliers
                           flags=2)
        img_matches = cv2.drawMatches(img_right_rgb, kp_right, img_left_rgb, kp_left, good_matches, None, **draw_params)
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_matches)
        plt.title(f'RANSAC Inlier Matches ({len(inlier_matches)})')
        plt.axis('off')

        # 2. Show the final stitched mosaic
        plt.subplot(1, 2, 2)
        plt.imshow(mosaic)
        plt.title('Stitched Mosaic')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    except (ImportError, ValueError, FileNotFoundError) as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
