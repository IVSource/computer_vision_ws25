import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def main():
    upper = 200
    lower = 100
    ESC_KEY = 27
    img = cv.imread('./data/classroom wall/20221115_113319.jpg',
                    cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"

    # Create a black image, a window
    # cv.namedWindow('image', cv.WINDOW_NORMAL)

    # Load the ArUco dictionary and detector parameters
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_100)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(img)
    color_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    # Draw detected markers
    if ids is not None:
        color_img = cv.aruco.drawDetectedMarkers(
            color_img, corners, ids, borderColor=(0, 255, 0))

        # Print marker information
        for i, marker_id in enumerate(ids.flatten()):
            corner = corners[i][0]
            center_x = np.mean(corner[:, 0])
            center_y = np.mean(corner[:, 1])
            print(
                f"Marker ID: {marker_id}, Center: ({center_x:.2f}, {center_y:.2f})")

    cv.namedWindow('ArUco Detection', cv.WINDOW_NORMAL)
    cv.imshow('ArUco Detection', color_img)

    key = cv.waitKey(0)


if __name__ == "__main__":
    main()
