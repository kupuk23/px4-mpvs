from px4_mpvs.marker_detector_blob import CircleFeatureDetector

import numpy as np
import cv2
import matplotlib

matplotlib.use("TkAgg")  # Use a non-interactive backend


ref_image = (
    "/home/tafarrel/discower_ws/src/px4_mpvs/px4_mpvs/resource/aligned_image.jpg"
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
    test_class()
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()
    # img = cv2.imread(ref_image)
    # img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    # hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(
    #     hsv, (0, 0, 0), (200, 255, 40)  # lower-H, lower-S, **very low V**
    # )  # upcentroidsper-H, upper-S, **dark V only**

    # contours,_   = cv2.findContours(mask, cv2.RETR_EXTERNAL,
    #                             cv2.CHAIN_APPROX_SIMPLE)
    # mask_filled  = np.zeros_like(mask)
    # for c in contours:
    #     cv2.drawContours(mask_filled, [c], -1, 255, -1)  # -1 = filled

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # mask   = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel, iterations=2)

    # cv2.imshow("masked with kernel", mask)
    # cv2.waitKey(0)  # Wait indefinitely until a key is pressed

    # params = cv2.SimpleBlobDetector_Params()
    # params.filterByArea = False
    # params.minArea = 20
    # params.maxArea = 500

    # params.filterByColor = False
    # params.minThreshold = 200
    # params.maxThreshold = 255

    # params.filterByCircularity = False
    # params.minCircularity = 0.6
    # params.maxCircularity = 1.2

    # params.filterByConvexity = False
    # params.minConvexity = 0.6

    # params.filterByInertia = False
    # params.minInertiaRatio = 0.5


    # params.minDistBetweenBlobs = 10 

    # detector = cv2.SimpleBlobDetector_create(params)

    # # cv2.imshow("Thresholded Image", img_thresh)
    # # cv2.waitKey(0)  # Wait indefinitely until a key is pressed

    # # Apply blur to reduce noise
    
    # kp = detector.detect(mask)
    # img_with_keypoints = cv2.drawKeypoints(
    #     mask,
    #     kp,
    #     np.array([]),
    #     (0, 0, 255),
    #     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    # )
    # pts = cv2.KeyPoint_convert(kp)
    

    # cv2.imshow("Keypoints", img_with_keypoints)
    # cv2.waitKey(0)  # Wait indefinitely until a key is pressed
