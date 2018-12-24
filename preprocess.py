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
    # xmin, xmax, ymin, ymax
    return list(map(
        int, [np.min(contour[:, 0]), np.max(contour[:, 0]), np.min(contour[:, 1]), np.max(contour[:, 1])]
    ))


def get_contours(binary_image):
    """
    TODO: add documentation here
    """
    # get all contours
    contours = skimage.measure.find_contours(binary_image, 0.5)

    separators = []
    # get 3 separator lines
    for contour in contours:
        xmin, xmax, ymin, ymax = get_cords(contour)

        if ymax - ymin > binary_image.shape[1] * 0.8:
            separators.append(get_cords(contour))
    rmin = separators[1][1] + 5
    rmax = separators[2][0] - 5

    contours_image = binary_image.copy()
    final_contours = []
    # get contours of extracted text only
    for contour in contours:
        xmin, xmax, ymin, ymax = get_cords(contour)

        if xmin > rmin and xmax < rmax:
            final_contours.append(contour)

            r = [xmin, xmax, xmax, xmin, xmin]
            c = [ymax, ymax, ymin, ymin, ymax]

            rr, cc = skimage.draw.polygon_perimeter(r, c, contours_image.shape)
            contours_image[rr, cc] = 1  # set color white

    return final_contours, (separators[1][1], separators[1][2]), contours_image[rmin:rmax]


def build_texture(grey_image, contours, transposed_center):
    """
    TODO: add documentation here
    """
    texture_image = np.zeros(shape=grey_image.shape)
    xtransposed, ytransposed = transposed_center
    xcenter, ycenter = 0, 0

    xdist_max = 0
    for contour in contours:
        xcontour = np.array(list(map(int, contour[:, 0])))
        ycontour = np.array(list(map(int, contour[:, 1])))

        xmin, xmax, ymin, ymax = get_cords(contour)
        xdist = xmax + 1 - xmin
        ydist = ymax + 1 - ymin
        xdist_max = max(xdist_max, xdist)

        if ycenter + ydist > texture_image.shape[1]:
            ycenter = 0
            xcenter += xdist_max
            xdist_max = 0

        iso_contour = np.zeros(shape=(xdist, ydist))
        for point in contour:
            iso_contour[int(point[0]) - xmin, int(point[1]) - ymin] = grey_image[int(point[0]), int(point[1])]

        texture_image[xcenter:xcenter+xdist, ycenter:ycenter+ydist] = iso_contour
        ycenter += ydist

    skimage.io.imsave('image.png', texture_image[:xcenter+xdist_max], cmap=plt.cm.gray)
    return texture_image[:xcenter+xdist_max]

def preprocessImage(input_image, debug=False):
    """
    TODO: add documentation here
    """
    #return [input_image[0:255,0:255]]

    # perform operations
    binary_image = binarize_image(input_image)
    contours, transposed_center, contours_image = get_contours(binary_image)
    texture_image = build_texture(input_image, contours, transposed_center)
    
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

        axes[1][1].imshow(texture_image, cmap=plt.cm.gray)
        axes[1][1].set_title('Texture Image')
    
        plt.show()
    
    return texture_image[0:256, 0:256]


if __name__ == "__main__":
    # for manual testing purposes
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to input image file")
    args = vars(parser.parse_args())

    # load the image from disk
    input_image = skimage.io.imread(args["image"], as_gray=True)

    preprocessImage(input_image, debug=True)
