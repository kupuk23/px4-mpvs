import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

def detect_markers(reference_img_path, test_img_path, visualization=True, debug=False):
    """
    Detect and match circular markers (white circles with black dots) between reference and test images
    using SIFT algorithm and feature matching.
    
    Args:
        reference_img_path: Path to the reference image
        test_img_path: Path to the test image
        visualization: Whether to visualize the results
        debug: Whether to show debug visualizations
        
    Returns:
        matches: List of good matches
        ref_markers: Detected marker keypoints in reference image
        test_markers: Detected marker keypoints in test image
    """
    # Load images
    ref_img = cv2.imread(reference_img_path)
    test_img = cv2.imread(test_img_path)
    
    if ref_img is None or test_img is None:
        raise ValueError("Could not load images")
    
    if debug:
        # Display both images to verify loading
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
        plt.title('Reference Image')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        plt.title('Test Image')
        plt.show()
    
    # Convert to grayscale
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    ref_keypoints, ref_descriptors = sift.detectAndCompute(ref_gray, None)
    test_keypoints, test_descriptors = sift.detectAndCompute(test_gray, None)
    
    if debug:
        # Show all SIFT keypoints before filtering
        ref_kp_img = cv2.drawKeypoints(ref_img, ref_keypoints, None, color=(0, 255, 0), 
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        test_kp_img = cv2.drawKeypoints(test_img, test_keypoints, None, color=(0, 255, 0), 
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(ref_kp_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Reference Image: {len(ref_keypoints)} SIFT keypoints')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(test_kp_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Test Image: {len(test_keypoints)} SIFT keypoints')
        plt.show()
    
    # Filter keypoints that might correspond to circular markers and get their indices
    ref_markers, ref_indices = filter_circular_markers(ref_img, ref_keypoints, debug=debug, image_label="Reference")
    test_markers, test_indices = filter_circular_markers(test_img, test_keypoints, debug=debug, image_label="Test")
    
    # Extract corresponding descriptors using indices
    ref_marker_descriptors = ref_descriptors[ref_indices] if len(ref_indices) > 0 else []
    test_marker_descriptors = test_descriptors[test_indices] if len(test_indices) > 0 else []
    
    # Match descriptors using FLANN
    good_matches = []
    if len(ref_marker_descriptors) > 0 and len(test_marker_descriptors) > 0:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        bf = cv2.BFMatcher()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Perform matching
        matches = flann.knnMatch(ref_marker_descriptors, test_marker_descriptors, k=2)
        bf_matches = bf.knnMatch(ref_marker_descriptors, test_marker_descriptors, k=2)
            
        # Filter good matches using Lowe's ratio test
        good_matches = []
        for m, n in bf_matches:
            if m.distance < 0.9 * n.distance:  # Slightly more lenient ratio
                good_matches.append(m)
    
    if visualization and good_matches:
        # Draw matches
        result = cv2.drawMatches(ref_img, ref_markers, test_img, test_markers, 
                                good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Display results
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('SIFT Marker Matches')
        plt.show()
        
        # Draw keypoints on original images
        ref_with_kp = cv2.drawKeypoints(ref_img, ref_markers, None, color=(0, 255, 0), 
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        test_with_kp = cv2.drawKeypoints(test_img, test_markers, None, color=(0, 255, 0), 
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(ref_with_kp, cv2.COLOR_BGR2RGB))
        plt.title('Reference Image Markers')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(test_with_kp, cv2.COLOR_BGR2RGB))
        plt.title('Test Image Markers')
        plt.show()
        
    
    return good_matches, ref_markers, test_markers


def filter_circular_markers(img, keypoints, debug=False, image_label=""):
    """
    Filter keypoints to identify those that might correspond to white circular markers with black dots
    
    Args:
        img: Original image
        keypoints: Detected keypoints
        debug: Whether to show debug visualizations
        image_label: Label for debug visualizations
        
    Returns:
        filtered_keypoints: Keypoints likely to be markers
        indices: Original indices of these keypoints
    """
    # Convert to HSV for better color filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for white color in HSV - MORE LENIENT
    lower_white = np.array([0, 0, 180])  # Reduced brightness threshold
    upper_white = np.array([180, 50, 255])  # Increased saturation threshold
    
    # Threshold the image to get white regions
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    if debug:
        # Display the white mask
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'{image_label} Image')
        plt.subplot(1, 2, 2)
        plt.imshow(white_mask, cmap='gray')
        plt.title(f'{image_label} White Mask')
        plt.show()
    
    # Find contours in the mask
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by circularity
    circular_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 150:  # Reduced minimum area to 50 pixels
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if 0.8 < circularity:  # More lenient circularity range
                    circular_contours.append(contour)
    
    if debug:
        # Display detected circular contours
        debug_img = img.copy()
        cv2.drawContours(debug_img, circular_contours, -1, (0, 255, 0), 2)
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title(f'{image_label} Detected Circular Contours: {len(circular_contours)}')
        plt.show()
    
    # Create a mask of circular regions
    circle_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(circle_mask, circular_contours, -1, 255, -1)
    
    # Filter keypoints based on their location within circular regions
    filtered_keypoints = []
    indices = []
    
    for i, kp in enumerate(keypoints):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            if circle_mask[y, x] > 0:
                filtered_keypoints.append(kp)
                indices.append(i)  # Store original index instead of modifying KeyPoint
    
    if debug:
        print(f"{image_label} image: Found {len(filtered_keypoints)} keypoints inside {len(circular_contours)} circular regions")
    
    return filtered_keypoints, np.array(indices)


def main():
    # Example usage
    reference_img_path = "/home/tafarrel/discower_ws/src/px4_mpvs/px4_mpvs/px4_mpvs/aligned_image.jpg"
    test_img_path = "/home/tafarrel/discower_ws/src/px4_mpvs/px4_mpvs/px4_mpvs/docked_image.jpg"
    
    matches, ref_markers, test_markers = detect_markers(reference_img_path, test_img_path, debug=True)
    
    print(f"Found {len(matches)} matches between {len(ref_markers)} reference markers and {len(test_markers)} test markers")


if __name__ == "__main__":
    main()