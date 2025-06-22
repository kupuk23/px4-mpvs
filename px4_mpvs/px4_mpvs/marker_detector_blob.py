#!/usr/bin/env python3

import cv2
import numpy as np
from itertools import combinations, permutations
import os
import matplotlib

matplotlib.use("TkAgg")  # Use a non-interactive backend


img_path = "/home/tafarrel/discower_ws/src/px4_mpvs/px4_mpvs/resource/docked_image_v3.jpg"



class CircleFeatureDetector:
    """
    Circle feature detector that finds and orders circular markers in images.
    Uses contour detection, circularity filtering, and homography matching.
    """

    def __init__(
        self,
        min_circle_radius=5,
        max_circle_radius=50,
        circularity_threshold=0.8,
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

        self.match_threshold = match_threshold

        self.params = cv2.SimpleBlobDetector_Params()
        self.params.filterByArea = False
        self.params.minArea = min_circle_radius
        self.params.maxArea = max_circle_radius

        self.params.filterByColor = False
        self.params.minThreshold = 0
        self.params.maxThreshold = 255

        self.params.filterByCircularity = False
        self.params.minCircularity = circularity_threshold
        self.params.maxCircularity = 1.2

        self.params.filterByConvexity = False
        self.params.minConvexity = 0.8

        self.params.filterByInertia = False
        self.params.minInertiaRatio = 0.8

        self.params.minDistBetweenBlobs = 10

        self.detector = cv2.SimpleBlobDetector_create(self.params)

        self.visualize = visualize
        self.debug = debug
        self.window_name = window_name

        # Initialize reference image and target points
        self.reference_img = None
        self.target_points = None
        self.ref_visualization = None

        if self.debug:
            print("CircleFeatureDetector initialized")

        # Analyze reference image if provided
        # if reference_img_path is not None:
        #     self.analyze_reference(reference_img_path)

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
            hsv, (0, 0, 0), (200, 255, 40)  # only accept low V
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
        # if self.visualize:
        #     # Visualize the blobbed image with circles

        #     cv2.putText(
        #         img_with_keypoints,
        #         f"Found {len(circle_centers)} circles",
        #         (20, 30),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.7,
        #         (0, 255, 255),
        #         2,
        #     )

        # Show intermediate processing steps for debugging
        # cv2.imshow("Keypoints", img_with_keypoints)
        # cv2.imshow("Masked Image", mask)

        # cv2.imshow(self.window_name, viz_img)

        # We need exactly 4 circles
        if len(circle_centers) < 4:
            if self.debug:
                print(f"Not enough circles detected: {len(circle_centers)}")
            return None, img

        # If more than 4 circles, match with target points
        matched_centers = self._match_circles(circle_centers, self.target_points)
        if matched_centers is not None:
            ordered_centers = self._order_circles(matched_centers)
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
        if self.visualize:

            # If we have reference visualization, show it side by side with current detection

            cv2.imshow(self.window_name, viz_img)

            cv2.waitKey(1)

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

    def detect_lines(self, image, lsd_parameters=None):
        """
        Detect lines in the input image using OpenCV's LineSegmentDetector

        Args:
            image: Input image
            lsd_parameters: Parameters for LineSegmentDetector

        Returns:
            filtered_lines: Array of detected line segments
        """
        # Make a copy for visualization
        viz_img = image.copy()
        color = (0, 255, 0)
        thickness = 2

        # Convert to grayscale if the image is in color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Create LSD detector with custom parameters if provided
        if lsd_parameters is None:
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        else:
            lsd = cv2.createLineSegmentDetector(**lsd_parameters)

        # Detect lines
        lines, width, prec, nfa = lsd.detect(gray)

        # Filter lines if needed (e.g., by length, orientation, etc.)
        filtered_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # Filter short lines and horizontal lines
                if length > 10 and abs(x2 - x1) < abs(
                    y2 - y1
                ):  # Minimum length threshold abs(x2 - x1) < abs(y2 - y1)
                    filtered_lines.append(line)

                    # Draw lines on visualization image
                    if self.visualize:
                        cv2.line(
                            viz_img,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color,
                            thickness,
                        )

        if self.visualize:
            cv2.imshow("Line Detection", viz_img)

        return np.array(filtered_lines) if filtered_lines else None

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
        min_circle_radius=5,
        max_circle_radius=50,
        circularity_threshold=0.8,
        match_threshold=10.0,
        visualize=True,
        debug=True,
    )

    image = cv2.imread(img_path)
    cv2.imshow("Input Image", image)
    
    centroids, _ = detector.detect(image)
    cv2.waitKey(0)
    print(centroids)
