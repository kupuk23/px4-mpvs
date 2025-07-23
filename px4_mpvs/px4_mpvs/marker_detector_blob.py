#!/usr/bin/env python3

import cv2
import numpy as np
from itertools import combinations, permutations
import os
import matplotlib

# matplotlib.use("Agg")  # Use a non-interactive backend


img_path = "/home/tafarrel/discower_ws/src/px4_mpvs/px4_mpvs/resource/docked_image_hw.jpg"



class CircleFeatureDetector:
    """
    Circle feature detector that finds and orders circular markers in images.
    Uses contour detection, circularity filtering, and homography matching.
    """

    def __init__(
        self,
        min_circle_radius=20,
        max_circle_radius=50,
        circularity_threshold=0.99,
        match_threshold=10.0,
        visualize=False,
        debug=False,
        window_name="Circle Detection",
    ):
        """
        Initialize the detector with configuration parameters.

        Args:
            reference_img_path: Path to reference image containing markers (optional)
            min_circle_radius: Minimum radius of circles to detect
            max_circle_radius: Maximum radius of circles to detect
            circularity_threshold: Threshold for circularity (1.0 is perfect circle)
            match_threshold: Maximum error allowed for matched points
            visualize: Whether to show debug visualizations
            debug: Whether to print debug information
            window_name: Name of the OpenCV window for visualization
        """

        self.tune_hsv = False  # Enable real-time HSV tuning
        self.match_threshold = match_threshold

        self.params = cv2.SimpleBlobDetector_Params()
        self.params.filterByArea = True
        self.params.minArea = min_circle_radius
        self.params.maxArea = max_circle_radius


        self.params.filterByCircularity = True
        self.params.minCircularity = circularity_threshold
        self.params.maxCircularity = 1.0

        self.params.filterByConvexity = True
        self.params.minConvexity = 0.95

        self.params.filterByInertia = True
        self.params.minInertiaRatio = 0.75

        self.params.minDistBetweenBlobs = 10

        self.detector = cv2.SimpleBlobDetector_create(self.params)

        self.visualize = visualize
        self.debug = debug
        self.window_name = window_name

        # Initialize reference image and target points
        self.reference_img = None
        self.target_points = None
        self.ref_visualization = None
        self.matched_centers = None

        # HSV threshold parameters for real-time tuning
        self.hsv_lower = [0, 0, 0]
        self.hsv_upper = [179, 153, 101]
        
        # Create trackbars for HSV tuning if visualization is enabled
        if self.visualize and self.tune_hsv:
            self._create_hsv_trackbars()

        if self.debug:
            print("CircleFeatureDetector initialized")

        # Analyze reference image if provided
        # if reference_img_path is not None:
        #     self.analyze_reference(reference_img_path)

    def _create_hsv_trackbars(self):
        """Create trackbars for HSV threshold tuning"""


        trackbar_window = "HSV Tuning"
        cv2.namedWindow(trackbar_window, cv2.WINDOW_AUTOSIZE)
        
        # Create trackbars for HSV lower bounds
        cv2.createTrackbar("H_min", trackbar_window, self.hsv_lower[0], 179, self._on_trackbar_change)
        cv2.createTrackbar("S_min", trackbar_window, self.hsv_lower[1], 255, self._on_trackbar_change)
        cv2.createTrackbar("V_min", trackbar_window, self.hsv_lower[2], 255, self._on_trackbar_change)
        
        # Create trackbars for HSV upper bounds
        cv2.createTrackbar("H_max", trackbar_window, self.hsv_upper[0], 179, self._on_trackbar_change)
        cv2.createTrackbar("S_max", trackbar_window, self.hsv_upper[1], 255, self._on_trackbar_change)
        cv2.createTrackbar("V_max", trackbar_window, self.hsv_upper[2], 255, self._on_trackbar_change)
        
        # Add some instructions text
        instructions = np.zeros((150, 700, 3), dtype=np.uint8)
        cv2.putText(instructions, "HSV Threshold Real-time Tuning", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(instructions, "Keyboard shortcuts:", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(instructions, "  'p' - Print current HSV values", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(instructions, "  's' - Save HSV config to file", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(instructions, "  'l' - Load HSV config from file", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(trackbar_window, instructions)

        # self.hsv_lower = default_hsv_lower
        # self.hsv_upper = default_hsv_upper
        # self._update_trackbars()

    def _on_trackbar_change(self, val):
        """Callback function for trackbar changes"""
        # Get current trackbar values
        trackbar_window = "HSV Tuning"
        self.hsv_lower[0] = cv2.getTrackbarPos("H_min", trackbar_window)
        self.hsv_lower[1] = cv2.getTrackbarPos("S_min", trackbar_window)
        self.hsv_lower[2] = cv2.getTrackbarPos("V_min", trackbar_window)
        
        self.hsv_upper[0] = cv2.getTrackbarPos("H_max", trackbar_window)
        self.hsv_upper[1] = cv2.getTrackbarPos("S_max", trackbar_window)
        self.hsv_upper[2] = cv2.getTrackbarPos("V_max", trackbar_window)
        
        if self.debug:
            print(f"HSV Lower: {self.hsv_lower}, HSV Upper: {self.hsv_upper}")

    def print_current_hsv_values(self):
        """Print current HSV threshold values - useful for saving good configurations"""
        print("="*50)
        print("Current HSV Threshold Values:")
        print(f"Lower bound: ({self.hsv_lower[0]}, {self.hsv_lower[1]}, {self.hsv_lower[2]})")
        print(f"Upper bound: ({self.hsv_upper[0]}, {self.hsv_upper[1]}, {self.hsv_upper[2]})")
        print("Copy this to your code:")
        print(f"hsv_lower = {self.hsv_lower}")
        print(f"hsv_upper = {self.hsv_upper}")
        print("="*50)

    def save_hsv_config(self, filename="hsv_config.txt"):
        """Save current HSV configuration to a file"""
        try:
            with open(filename, 'w') as f:
                f.write(f"# HSV Configuration\n")
                f.write(f"hsv_lower = {self.hsv_lower}\n")
                f.write(f"hsv_upper = {self.hsv_upper}\n")
            print(f"HSV configuration saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving HSV config: {e}")
            return False

    def load_hsv_config(self, filename="hsv_config.txt"):
        """Load HSV configuration from a file"""
        try:
            # set the filename into the full path in resources directory
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("hsv_lower"):
                        self.hsv_lower = eval(line.split("=")[1].strip())
                    elif line.startswith("hsv_upper"):
                        self.hsv_upper = eval(line.split("=")[1].strip())
            
            # Update trackbars if visualization is enabled
            if self.visualize:
                self._update_trackbars()
            
            print(f"HSV configuration loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading HSV config: {e}")
            return False

    def _update_trackbars(self):
        """Update trackbar positions to match current HSV values"""
        trackbar_window = "HSV Tuning"
        cv2.setTrackbarPos("H_min", trackbar_window, self.hsv_lower[0])
        cv2.setTrackbarPos("S_min", trackbar_window, self.hsv_lower[1])
        cv2.setTrackbarPos("V_min", trackbar_window, self.hsv_lower[2])
        cv2.setTrackbarPos("H_max", trackbar_window, self.hsv_upper[0])
        cv2.setTrackbarPos("S_max", trackbar_window, self.hsv_upper[1])
        cv2.setTrackbarPos("V_max", trackbar_window, self.hsv_upper[2])

    def set_target_points(self, target_points):
        """
        Set the target points for circle matching.

        Args:
            target_points: Array of [x,y] points in desired pattern (np.float32)
        """
        self.target_points = np.array(target_points, dtype=np.float32)
        if self.debug:
            print(f"Set target points: {self.target_points}")

    def detect(self, img):
        """
        Detect 4 circle features and return their centroids ordered from
        top-left to bottom-right.

        Args:
            img: Input image (BGR)

        Returns:
            ordered_centers: Numpy array of shape (4, 2) containing the ordered
                centroids of the detected circles. If 4 circles are not found, returns None.
        """
        if self.target_points is None:
            if self.debug:
                print("Target points not set. Using default ordering.")

        img_blur = cv2.GaussianBlur(img, (3, 3), 0)

        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv, tuple(self.hsv_lower), tuple(self.hsv_upper)
        ) 

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)


        # cv2.imshow("Morphology Mask", mask)

        kp = self.detector.detect(mask)
        img_with_keypoints = cv2.drawKeypoints(
            img,
            kp,
            np.array([]),
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        circle_centers = cv2.KeyPoint_convert(kp)

        # If visualization is enabled, draw the circles that were found
        if self.visualize:
            # Visualize the blobbed image with circles

            cv2.putText(
                img_with_keypoints,
                f"Found {len(circle_centers)} circles",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        # Show intermediate processing steps for debugging
        if self.visualize:
            cv2.imshow("Keypoints", img_with_keypoints)
            cv2.imshow("Masked Image", mask)
            # cv2.imshow("HSV Image", hsv)
            
            # Add HSV values to the keypoints image
            # cv2.putText(
            #     img_with_keypoints,
            #     f"HSV Lower: {self.hsv_lower}",
            #     (20, 60),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 0),
            #     1,
            # )
            # cv2.putText(
            #     img_with_keypoints,
            #     f"HSV Upper: {self.hsv_upper}",
            #     (20, 80),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 0),
            #     1,
            # )
            # cv2.putText(
            #     img_with_keypoints,
            #     "Press 'p'=print, 's'=save, 'l'=load HSV",
            #     (20, 100),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.4,
            #     (255, 255, 255),
            #     1,
            # )
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                self.print_current_hsv_values()
            elif key == ord('s'):
                self.save_hsv_config()
            elif key == ord('l'):
                self.load_hsv_config()


        # We need exactly 4 circles
        if len(circle_centers) < 4:
            if self.debug:
                print(f"Not enough circles detected: {len(circle_centers)}")
            return None, img

        # If more than 4 circles, match with target points
        self.matched_centers = self._match_circles(circle_centers, self.target_points)
        if self.matched_centers is not None:
            ordered_centers = self._order_circles(self.matched_centers)
        else:
            # If matching failed, use default ordering with all detected circles
            ordered_centers = self._order_circles(circle_centers[:4])
            if self.debug:
                print("Matching failed, using default ordering")

        ordered_centers = np.array(ordered_centers, dtype=np.int16)

        viz_img = img.copy()

        

        # Draw circles and IDs
        for i, center in enumerate(ordered_centers):
            cv2.circle(viz_img, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
            center = np.array(center, dtype=np.int16)
            cv2.putText(
                viz_img,
                str(i),
                (int(center[0]) + 15, int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            # offset = 30 if i % 2 == 0 else 0
            # cv2.putText(
            #     viz_img,
            #     "(" + str(center[0]) + "," + str(center[1]) + ")",
            #     (int(center[0]) - offset, int(center[1]) + 20),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 255),
            #     2,
            # )

        # Show final detected circles if visualization is enabled

        return ordered_centers, viz_img

    def _order_circles(self, circle_centers):
        """
        Order the circle centers from top-left to bottom-right

        Args:
            circle_centers: List of (x, y) tuples

        Returns:
            ordered_centers: Numpy array with ordered centers
        """
        # Convert to numpy array
        centers = np.array(circle_centers, dtype=np.float32)

        # Order the points: top-left, top-right, bottom-left, bottom-right
        # First, sort by y-coordinate to separate top and bottom pairs
        centers = sorted(centers, key=lambda p: p[1])

        # Get top and bottom pairs
        top_pair = sorted(centers[:2], key=lambda p: p[0])  # Sort by x for top pair
        bottom_pair = sorted(
            centers[2:], key=lambda p: p[0]
        )  # Sort by x for bottom pair

        # Combine into the final order: [top-left, top-right, bottom-left, bottom-right]
        ordered_centers = np.array(
            [top_pair[0], top_pair[1], bottom_pair[0], bottom_pair[1]], dtype=np.float32
        )

        return ordered_centers

    def _match_circles(self, detected_points, target_points):
        """
        Match detected circle centers to target pattern using homography

        Args:
            detected_points: Array of points detected in the current frame
            target_points: Array of the 4 points in desired configuration

        Returns:
            matched_points: Best 4 points matching the target pattern or None if no good match
        """
        if len(detected_points) < 4:
            return None

        # Convert to numpy arrays
        detected_points = np.array(detected_points, dtype=np.float32)
        target_points = np.array(target_points, dtype=np.float32)

        # Try all combinations of 4 detected points
        best_error = float("inf")
        best_points = None

        for indices in combinations(range(len(detected_points)), 4):
            src_points = detected_points[list(indices)]

            # Try all permutations of these 4 points
            for perm in permutations(range(4)):
                ordered_src = src_points[list(perm)]

                # Find homography
                try:
                    H, _ = cv2.findHomography(ordered_src, target_points, cv2.RANSAC)

                    if H is None:
                        continue

                    # Transform points
                    transformed = cv2.perspectiveTransform(
                        ordered_src.reshape(-1, 1, 2), H
                    ).reshape(-1, 2)

                    # Calculate error
                    error = np.mean(np.linalg.norm(transformed - target_points, axis=1))

                    if error < best_error:
                        best_error = error
                        best_points = ordered_src
                except:
                    continue

        # Only return if error is below threshold
        if best_error > self.match_threshold:
            return None

        return best_points


    def save_reference_visualization(self, output_path):
        """Save the reference visualization to disk."""
        if self.ref_visualization is not None:
            cv2.imwrite(output_path, self.ref_visualization)
            if self.debug:
                print(f"Saved reference visualization to {output_path}")
            return True
        return False

    def __del__(self):
        """Clean up OpenCV windows when object is destroyed"""
        if self.visualize:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = CircleFeatureDetector(
        min_circle_radius=20,
        max_circle_radius=1200,
        circularity_threshold=0.85,
        match_threshold=5.0,
        visualize=True,
        debug=True,
    )

    image = cv2.imread(img_path)
    # cv2.imshow("Input Image", image)
    
    centroids, _ = detector.detect(image)
    cv2.waitKey(0)
    print(centroids)
