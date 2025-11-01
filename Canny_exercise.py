import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('messi5.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"


def nothing(x):
    pass


def main():
    upper = 200
    lower = 100

    # Create a black image, a window
    cv.namedWindow('image')

    # create trackbars to change the hysteresis thresholds
    cv.createTrackbar('Lower Threshold', 'image', lower, 500, nothing)
    cv.createTrackbar('Upper Threshold', 'image', upper, 500, nothing)

    while (1):
        canny_edges = cv.Canny(img, upper, lower)
        sobel_edges = cv.Sobel(img, cv.CV_8U, 1, 1, ksize=5)
        display_img = np.hstack(
            (img, canny_edges, sobel_edges.astype(np.uint8)))
        cv.imshow('image', display_img)

        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        upper = cv.getTrackbarPos('Lower Threshold', 'image')
        lower = cv.getTrackbarPos('Upper Threshold', 'image')

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
