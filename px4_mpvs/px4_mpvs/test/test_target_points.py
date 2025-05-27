from px4_mpvs.marker_detector_blob import CircleFeatureDetector

import numpy as np
import cv2
import matplotlib

# matplotlib.use("TkAgg")  # Use a non-interactive backend


ref_image = (
    "/home/tafarrel/discower_ws/src/px4_mpvs/px4_mpvs/resource/docked_image.jpg"
)


def test_class():
    detector = CircleFeatureDetector(
        min_circle_radius=10,
        max_circle_radius=100,
        circularity_threshold=0.9,
        match_threshold=5.0,
        visualize=True,
        debug=True,
    )

    image = cv2.imread(ref_image)
    centroids = detector.detect(image)
    print(centroids)

if __name__ == "__main__":
    cv2.imshow("Reference Image", cv2.imread(ref_image))
    test_class()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # img = cv2.imread(ref_image)
    # img_blur = cv2.GaussianBlur(img, (3, 3), 0)
