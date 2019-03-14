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
        grey_image: The grey image to binarize.

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
    # xmin, xmax, ymin, ymax
    return list(map(
        int, [np.min(contour[:, 0]), np.max(contour[:, 0]), np.min(contour[:, 1]), np.max(contour[:, 1])]
    ))


def get_paragraph(binary_image):
    """
    TODO: add documentation here
    """
    # get all contours
    contours = skimage.measure.find_contours(binary_image, 0.5)

    separators = []
    # get 3 separator lines
    for contour in contours:
        xmin, xmax, ymin, ymax = get_cords(contour)
        if ymax - ymin > binary_image.shape[0] * 0.5:
            separators.append(get_cords(contour))
    rmin = separators[1][1] + 5
    rmax = separators[2][0] - 5

    paragraph = binary_image[rmin:rmax]

    # remove padding
    rows = [row for row in range(paragraph.shape[0]) if paragraph[row, :].any()]
    cols = [col for col in range(paragraph.shape[1]) if paragraph[:, col].any()]
    
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]

    return paragraph[top:bottom, left:right]


def preprocess_image(input_image):
    """
    TODO: add documentation here
    """
    binary_image = binarize_image(input_image)
    paragraph = get_paragraph(binary_image)

    return paragraph


if __name__ == "__main__":
    # for manual testing purposes
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to input image file")
    args = vars(parser.parse_args())

    # load the image from disk
    input_image = skimage.io.imread(args["image"], as_gray=True)

    preprocessed_image = preprocess_image(input_image)
    plt.imshow(preprocessed_image, cmap=plt.cm.gray)
    plt.show()
