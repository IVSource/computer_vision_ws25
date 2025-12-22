import numpy as np
import cv2 as cv
import os

images_directory = './data/classroom wall/'
image_paths = []

with os.scandir(images_directory) as entries:
    for entry in entries:
        if entry.is_file() and entry.name.endswith('.jpg'):
            image_paths.append(entry.path)

window_name = 'ArUco Detection'
current_image_index = 0

source_points = np.zeros((4, 2), dtype=np.float32)
source_points[0] = [0, 0]
source_points[1] = [100, 0]
source_points[2] = [100, 100]
source_points[3] = [0, 100]

x_offset = 512
y_offset = 1024
translation_matrix = np.array([[1, 0, -x_offset], [0, 1, -y_offset], [0, 0, 1]])


def update_target_x(value: int):
    global current_image_index
    global x_offset
    translation_matrix[0, 2] = value - x_offset
    update_window(current_image_index)


def update_target_y(value: int):
    global current_image_index
    global y_offset
    translation_matrix[1, 2] = value - y_offset
    update_window(current_image_index)


def update_window(image: int):
    global current_image_index
    global translation_matrix

    current_image_index = image
    wall_file = image_paths[current_image_index]
    print(f"Loading image: {wall_file}")
    wall_grey = cv.imread(wall_file, cv.IMREAD_GRAYSCALE)
    assert wall_grey is not None, "WALL file could not be read, check with os.path.exists()"

    wall_color = cv.imread(wall_file, cv.IMREAD_COLOR)
    assert wall_color is not None, "WALL file could not be read, check with os.path.exists()"

    poster = cv.imread('./starry_night.jpg', cv.IMREAD_COLOR)
    assert poster is not None, "POSTER file could not be read, check with os.path.exists()"

    # Load the ArUco dictionary and detector parameters
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_100)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(wall_grey)
    print('Detected corners: ', corners)

    if len(corners) < 1:
        print("No markers detected.")
        cv.imshow('ArUco Detection', wall_color)
        return

    target_points = np.array(corners).reshape((4, 2)).astype(np.float32)

    projection_matrix = cv.getPerspectiveTransform(source_points, target_points)

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

    # apply translation to projection matrix
    projection_matrix = projection_matrix @ translation_matrix

    # apply the projection to the poster image
    cv.warpPerspective(poster, projection_matrix,
                       (wall_grey.shape[1], wall_grey.shape[0]), wall_color, borderMode=cv.BORDER_TRANSPARENT)

    cv.imshow('ArUco Detection', wall_color)


def main():
    global current_image_index
    global x_offset
    global y_offset
    ESC_KEY = 27

    print(f"Found {len(image_paths)} images in {images_directory}")
    print(image_paths)

    file_selector_name = 'Image Selector'
    x_pos_slider_name = 'X Position'
    y_pos_slider_name = 'Y Position'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.createTrackbar(file_selector_name, window_name, 0, len(image_paths) - 1, update_window)
    cv.createTrackbar(x_pos_slider_name, window_name, 0, 1024, update_target_x)
    cv.createTrackbar(y_pos_slider_name, window_name, 0, 2048, update_target_y)
    cv.setTrackbarPos(x_pos_slider_name, window_name, x_offset)
    cv.setTrackbarPos(y_pos_slider_name, window_name, y_offset)
    update_window(current_image_index)

    while cv.waitKey(-1) != ESC_KEY:
        pass

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
