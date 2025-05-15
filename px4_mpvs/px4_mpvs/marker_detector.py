#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class CircleMarkerDetector:
    """
    Class for detecting and tracking circular markers in images.
    Designed for integration with ROS2 for visual servoing applications.
    """
    
    def __init__(self, reference_img_path, expected_markers=4, debug=False, vis_debug=False):
        """
        Initialize the detector with a reference image containing the markers.
        
        Args:
            reference_img_path: Path to the reference image
            expected_markers: Number of markers expected (default: 4)
            debug: Whether to print debug information
            vis_debug: Whether to show visualization windows
        """
        self.expected_markers = expected_markers
        self.debug = debug
        self.vis_debug = vis_debug
        self.cv_bridge = CvBridge()
        
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()
        
        # Initialize matchers
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.bf = cv2.BFMatcher()
        
        # Process reference image
        self._load_reference(reference_img_path)
        
        if debug:
            print(f"CircleMarkerDetector initialized with {len(self.ref_markers)} reference markers")

    def _load_reference(self, reference_img_path):
        """
        Load reference image and extract features.
        """
        # Load and process reference image
        self.ref_img = cv2.imread(reference_img_path)
        if self.ref_img is None:
            raise ValueError(f"Could not load reference image from {reference_img_path}")
        
        self.ref_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)
        
        # Detect SIFT features in reference image
        self.ref_keypoints, self.ref_descriptors = self.sift.detectAndCompute(self.ref_gray, None)
        
        # Filter keypoints to find markers
        self.ref_markers, self.ref_indices = self._filter_circular_markers(
            self.ref_img, 
            self.ref_keypoints, 
            debug=self.debug, 
            image_label="Reference"
        )
        
        # Extract descriptors for the markers
        if len(self.ref_indices) > 0:
            self.ref_marker_descriptors = self.ref_descriptors[self.ref_indices]
        else:
            self.ref_marker_descriptors = []
            if self.debug:
                print("Warning: No markers detected in reference image!")
        
        # Calculate centroids of reference markers for sorting
        self.ref_marker_centroids = []
        if self.ref_markers:
            for kp in self.ref_markers:
                self.ref_marker_centroids.append((kp.pt[0], kp.pt[1]))
        
        # Create debug window if requested
        if self.vis_debug:
            self._show_reference_markers()

    def detect_markers_from_compressed(self, compressed_img_msg):
        """
        Process a ROS CompressedImage message to detect markers.
        
        Args:
            compressed_img_msg: ROS CompressedImage message
            
        Returns:
            marker_positions: List of (x,y) positions of the detected markers, sorted
            success: Boolean indicating if the expected number of markers was found
        """
        # Convert CompressedImage to OpenCV format
        test_img = self.cv_bridge.compressed_imgmsg_to_cv2(compressed_img_msg)
        return self.detect_markers(test_img)
    
    def detect_markers(self, test_img):
        """
        Process an OpenCV image to detect markers.
        
        Args:
            test_img: OpenCV image
            
        Returns:
            marker_positions: List of (x,y) positions of the detected markers, sorted
            success: Boolean indicating if the expected number of markers was found
        """
        if self.debug:
            print("Processing new image for marker detection...")
        
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        
        # Detect SIFT features in test image
        test_keypoints, test_descriptors = self.sift.detectAndCompute(test_gray, None)
        
        # Filter keypoints to find markers
        test_markers, test_indices = self._filter_circular_markers(
            test_img, 
            test_keypoints, 
            debug=self.debug, 
            image_label="Current"
        )
        
        # If not enough markers found, try with more lenient parameters
        if len(test_markers) < self.expected_markers:
            if self.debug:
                print(f"Only found {len(test_markers)} markers, trying with more lenient parameters...")
            test_markers, test_indices = self._filter_circular_markers_lenient(
                test_img, 
                test_keypoints, 
                debug=self.debug, 
                image_label="Current (Lenient)"
            )
        
        # Extract descriptors for the markers
        if len(test_indices) > 0:
            test_marker_descriptors = test_descriptors[test_indices]
        else:
            test_marker_descriptors = []
            if self.debug:
                print("No markers detected in current image!")
            return [], False
        
        # Match markers between reference and current image
        good_matches = self._match_markers(test_marker_descriptors)
        
        if self.debug:
            print(f"Found {len(good_matches)} good matches between reference and current markers")
            
        if self.vis_debug:
            self._visualize_matches(test_img, test_markers, good_matches)
        
        # Extract and sort marker positions
        marker_positions = self._extract_marker_positions(test_markers, good_matches)
        
        success = len(marker_positions) == self.expected_markers
        
        if self.debug and not success:
            print(f"Warning: Found {len(marker_positions)} markers, expected {self.expected_markers}")
            
        return marker_positions, success

    def _filter_circular_markers(self, img, keypoints, debug=False, image_label=""):
        """
        Filter keypoints to identify those that might correspond to circular markers.
        
        Args:
            img: Input image
            keypoints: Detected keypoints
            debug: Whether to print debug info
            image_label: Label for debug output
            
        Returns:
            filtered_keypoints: Keypoints likely to be markers
            indices: Original indices of these keypoints
        """
        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range for white color in HSV
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        
        # Threshold the image to get white regions
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by circularity
        circular_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 150:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if 0.8 < circularity:
                        circular_contours.append(contour)
        
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
                    indices.append(i)
        
        if debug:
            print(f"{image_label} image: Found {len(filtered_keypoints)} keypoints inside {len(circular_contours)} circular regions")
        
        if self.vis_debug:
            # Create debug visualization
            debug_img = img.copy()
            cv2.drawContours(debug_img, circular_contours, -1, (0, 255, 0), 2)
            cv2.drawKeypoints(debug_img, filtered_keypoints, debug_img, color=(0, 0, 255), 
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            window_name = f"{image_label} Markers"
            cv2.imshow(window_name, debug_img)
            cv2.waitKey(1)  # Update the window without blocking
            
        return filtered_keypoints, np.array(indices) if indices else np.array([])

    def _filter_circular_markers_lenient(self, img, keypoints, debug=False, image_label=""):
        """
        More lenient version of filter_circular_markers for challenging images.
        """
        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define very lenient range for white color
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 80, 255])
        
        # Threshold the image to get white regions
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.dilate(white_mask, kernel, iterations=1)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by circularity - VERY LENIENT
        circular_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 30:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if 0.6 < circularity < 1.3:
                        circular_contours.append(contour)
        
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
                    indices.append(i)
        
        if debug:
            print(f"{image_label} image (lenient): Found {len(filtered_keypoints)} keypoints inside {len(circular_contours)} circular regions")
        
        if self.vis_debug:
            # Create debug visualization
            debug_img = img.copy()
            cv2.drawContours(debug_img, circular_contours, -1, (0, 255, 0), 2)
            cv2.drawKeypoints(debug_img, filtered_keypoints, debug_img, color=(0, 0, 255), 
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            window_name = f"{image_label} Markers (Lenient)"
            cv2.imshow(window_name, debug_img)
            cv2.waitKey(1)  # Update the window without blocking
        
        return filtered_keypoints, np.array(indices) if indices else np.array([])

    def _match_markers(self, test_marker_descriptors):
        """
        Match markers between reference and test images.
        
        Args:
            test_marker_descriptors: Descriptors of test markers
            
        Returns:
            good_matches: List of good matches
        """
        good_matches = []
        
        if len(self.ref_marker_descriptors) > 0 and len(test_marker_descriptors) > 0:
            # Try both FLANN and Brute Force matchers
            try:
                # FLANN matcher
                flann_matches = self.flann.knnMatch(self.ref_marker_descriptors, test_marker_descriptors, k=2)
                
                flann_good = []
                for m, n in flann_matches:
                    if m.distance < 0.85 * n.distance:
                        flann_good.append(m)
                
                # Brute Force matcher
                bf_matches = self.bf.knnMatch(self.ref_marker_descriptors, test_marker_descriptors, k=2)
                
                bf_good = []
                for m, n in bf_matches:
                    if m.distance < 0.85 * n.distance:
                        bf_good.append(m)
                
                # Use whichever found more good matches
                good_matches = flann_good if len(flann_good) > len(bf_good) else bf_good
                
            except Exception as e:
                if self.debug:
                    print(f"Error during marker matching: {e}")
        
        return good_matches

    def _extract_marker_positions(self, test_markers, good_matches):
        """
        Extract and sort marker positions from matches.
        
        Args:
            test_markers: Detected markers in test image
            good_matches: Good matches between reference and test markers
            
        Returns:
            sorted_positions: List of (x,y) positions of the markers, sorted
        """
        # Extract positions from matches
        positions = []
        
        for match in good_matches:
            # Get the index of the matched test keypoint
            test_idx = match.trainIdx
            
            if test_idx < len(test_markers):
                # Get the keypoint
                kp = test_markers[test_idx]
                # Add its position to our list
                positions.append((kp.pt[0], kp.pt[1]))
        
        # Ensure we only return up to expected_markers positions
        positions = positions[:self.expected_markers]
        
        # Sort markers by position (left to right, top to bottom)
        sorted_positions = self._sort_marker_positions(positions)
        
        return sorted_positions

    def _sort_marker_positions(self, positions):
        """
        Sort marker positions from left to right, top to bottom.
        
        Args:
            positions: List of (x,y) positions
            
        Returns:
            sorted_positions: Sorted list of positions
        """
        if not positions:
            return []
            
        # Calculate average y-coordinate
        if len(positions) >= 2:
            avg_y = sum(p[1] for p in positions) / len(positions)
            
            # Split into top and bottom groups
            top_markers = [p for p in positions if p[1] < avg_y]
            bottom_markers = [p for p in positions if p[1] >= avg_y]
            
            # Sort each group by x-coordinate
            top_markers.sort(key=lambda p: p[0])
            bottom_markers.sort(key=lambda p: p[0])
            
            # Combine the sorted groups
            sorted_positions = top_markers + bottom_markers
        else:
            # If only one marker, just return it
            sorted_positions = positions
        
        return sorted_positions

    def _show_reference_markers(self):
        """
        Display the reference image with detected markers.
        """
        if self.ref_img is not None and self.ref_markers:
            debug_img = self.ref_img.copy()
            cv2.drawKeypoints(debug_img, self.ref_markers, debug_img, color=(0, 0, 255),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            cv2.imshow("Reference Markers", debug_img)
            cv2.waitKey(1)

    def _visualize_matches(self, test_img, test_markers, good_matches):
        """
        Visualize matches between reference and test images.
        
        Args:
            test_img: Current image
            test_markers: Detected markers in test image
            good_matches: Good matches between reference and test
        """
        if good_matches:
            # Draw matches
            result = cv2.drawMatches(self.ref_img, self.ref_markers, test_img, test_markers, 
                                    good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            cv2.imshow("Marker Matches", result)
            cv2.waitKey(1)

    def calculate_interaction_matrix(self, marker_positions, camera_params):
        """
        Calculate the interaction matrix based on marker positions.
        To be implemented in the next step.
        """
        # Placeholder - will be implemented later
        pass
        
    def __del__(self):
        """
        Clean up any resources when the object is deleted.
        """
        if self.vis_debug:
            cv2.destroyAllWindows()