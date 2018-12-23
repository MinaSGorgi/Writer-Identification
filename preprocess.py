import argparse
import matplotlib.pyplot as plt
import numpy as np

import skimage
from skimage import io
from skimage import morphology
from skimage import transform


def binarize_image(grey_image):
    """
    Performs: Gaussian blur -> Otsu threshold -> Closing kernel.

    Args:
        grey_image: The grey image to process.

    Returns:
        The resultant new binary image.

    Raises:
        None
    """
    gaussian_image = skimage.filters.gaussian(grey_image)

    threshold = skimage.filters.threshold_otsu(gaussian_image)
    thresh_image = gaussian_image <= threshold

    closed_image = skimage.morphology.binary_closing(thresh_image)

    return closed_image


def get_cords(contour):
    """
    TODO: add documentation here
    """
    return np.min(contour[:, 0]), np.max(contour[:, 0]), np.min(contour[:, 1]), np.max(contour[:, 1])


def get_contours(binary_image):
    """
    TODO: add documentation here
    """
    # get all contours
    contours = skimage.measure.find_contours(binary_image, 0, fully_connected='high')

    separators = []
    # get 3 separator lines
    for contour in contours:
        xmin, xmax, ymin, ymax = get_cords(contour)

        if ymax - ymin > binary_image.shape[1] * 0.8:
            separators.append((int(xmin), int(xmax)))
    rmin = separators[1][1]
    rmax = separators[2][0]

    contours_image = binary_image.copy()
    final_contours = []
    # get contours of extracted text only
    for contour in contours:
        xmin, xmax, ymin, ymax = get_cords(contour)

        if xmax > rmin:
            final_contours.append(contour)

            r = [xmin, xmax, xmax, xmin, xmin]
            c = [ymax, ymax, ymin, ymin, ymax]

            rr, cc = skimage.draw.polygon_perimeter(r, c, contours_image.shape)
            contours_image[rr, cc] = 1  # set color white

    return final_contours, contours_image[rmin:rmax]


def preprocess_image(image_path, debug=False):
    """
    TODO: add documentation here
    """
    # load the image from disk
    input_image = skimage.io.imread(image_path, as_gray=True)

    # perform operations
    binary_image = binarize_image(input_image)
    contours, contours_image = get_contours(binary_image)

    if debug:
        # show results
        rows = 2
        cols = 2
        figure, axes = plt.subplots(rows, cols)

        axes[0][0].imshow(input_image, cmap=plt.cm.gray)
        axes[0][0].set_title('Input Image')

        axes[0][1].imshow(binary_image, cmap=plt.cm.gray)
        axes[0][1].set_title('Binary Image')

        axes[1][0].imshow(contours_image, cmap=plt.cm.gray)
        axes[1][0].set_title('Contours Image')

        plt.show()

    return contours_image


if __name__ == "__main__":
    # for manual testing purposes
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to input image file")
    args = vars(parser.parse_args())

    preprocess_image(args["image"], debug=True)
