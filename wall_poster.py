import numpy as np
import cv2 as cv


def main():
    upper = 200
    lower = 100
    ESC_KEY = 27
    # wall_grey = cv.imread('./data/classroom wall/20221115_113340.jpg', cv.IMREAD_GRAYSCALE)
    wall_grey = cv.imread('./data/classroom wall/20221115_113346.jpg', cv.IMREAD_GRAYSCALE)
    assert wall_grey is not None, "WALL file could not be read, check with os.path.exists()"

    # wall_color = cv.imread('./data/classroom wall/20221115_113340.jpg', cv.IMREAD_COLOR)
    wall_color = cv.imread('./data/classroom wall/20221115_113346.jpg', cv.IMREAD_COLOR)
    assert wall_color is not None, "WALL file could not be read, check with os.path.exists()"

    poster = cv.imread('./starry_night.jpg', cv.IMREAD_COLOR)
    assert poster is not None, "POSTER file could not be read, check with os.path.exists()"

    # Create a black image, a window
    # cv.namedWindow('image', cv.WINDOW_NORMAL)

    # Load the ArUco dictionary and detector parameters
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_100)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(wall_grey)
    print('Detected corners: ', corners)

    source_points = np.zeros((4, 2), dtype=np.float32)
    print('Original corners: ', source_points)
    source_points[0] = [0, 0]
    source_points[1] = [100, 0]
    source_points[2] = [100, 100]
    source_points[3] = [0, 100]
    print(source_points)
    target_points = np.array(corners).reshape((4, 2)).astype(np.float32)

    projection_matrix = cv.getPerspectiveTransform(source_points, target_points)
    print(projection_matrix)

    # Draw detected markers
    if ids is not None:
        wall_color = cv.aruco.drawDetectedMarkers(
            wall_color, corners, ids, borderColor=(0, 255, 0))

        # Print marker information
        for i, marker_id in enumerate(ids.flatten()):
            corner = corners[i][0]
            center_x = np.mean(corner[:, 0])
            center_y = np.mean(corner[:, 1])
            print(
                f"Marker ID: {marker_id}, Center: ({center_x:.2f}, {center_y:.2f})")

    projection_matrix @= np.array([[1, 0, -100], [0, 1, -300], [0, 0, 1]])

    projection_pane = np.zeros((*wall_grey.shape, 3), dtype=np.uint8)
    cv.warpPerspective(poster, projection_matrix,
                       (wall_grey.shape[1], wall_grey.shape[0]), wall_color, borderMode=cv.BORDER_TRANSPARENT)

    cv.namedWindow('ArUco Detection', cv.WINDOW_NORMAL)
    # cv.namedWindow('Poster Projection', cv.WINDOW_NORMAL)
    cv.imshow('ArUco Detection', wall_color)
    # cv.imshow('Poster Projection', projection_pane)

    key = cv.waitKey(0)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
