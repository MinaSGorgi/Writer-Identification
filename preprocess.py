import argparse
import matplotlib.pyplot as plt
import numpy as np

import skimage
from skimage import io
from skimage import morphology


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


def build_texture(binary_image, thresh_area=128):
    """
    TODO: add documentation here
    """
    # get contours
    contours = skimage.measure.find_contours(binary_image, 0, fully_connected='high', positive_orientation='low')
    contours_image = binary_image.copy()

    # draw bounding boxes
    for contour in contours:
        xmin = np.min(contour[:, 0])
        xmax = np.max(contour[:, 0])
        ymin = np.min(contour[:, 1])
        ymax = np.max(contour[:, 1])

        r = [xmin, xmax, xmax, xmin, xmin]
        c = [ymax, ymax, ymin, ymin, ymax]
        rr, cc = skimage.draw.polygon_perimeter(r, c, contours_image.shape)
        contours_image[rr, cc] = 1  # set color white

    return contours_image


if __name__ == "__main__":
    # for manual testing purposes
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to input image file")
    args = vars(parser.parse_args())

    # load the image from disk
    input_image = skimage.io.imread(args["image"], as_gray=True)
    print(input_image.shape)

    # test operations
    binary_image = binarize_image(input_image)
    contours_image = build_texture(binary_image)

    # show results
    rows = 1
    cols = 3
    figure, axes = plt.subplots(rows, cols)

    axes[0].imshow(input_image, cmap=plt.cm.gray)
    axes[0].set_title('Input Image')

    axes[1].imshow(binary_image, cmap=plt.cm.gray)
    axes[1].set_title('Binary Image')

    axes[2].imshow(contours_image, cmap=plt.cm.gray)
    axes[2].set_title('Contours Image')

    plt.show()
