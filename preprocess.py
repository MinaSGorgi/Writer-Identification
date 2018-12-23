import argparse
import matplotlib.pyplot as plt
import skimage
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


def build_texture(image, thresh_area=128):
    """
    TODO: add documentation here
    """

    # disconnect text from border by forcing white border
    border = (20,) * 4
    border_image = cv2.copyMakeBorder(image, *border, cv2.BORDER_CONSTANT, value=0)
    contour_image = border_image[5:-5, 5:-5].copy()

    # get contours image
    __, contours, hierarchy = cv2.findContours(contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    i = 0
    for contour in contours:
        i = i + 1
        # get rectangle bounding contour
        x, y, w, h = cv2.boundingRect(contour)

        # draw rectangle around contour on original image and discard small artifacts
        if w * h > thresh_area:
            cv2.imshow(str(i), border_image[y:y + h, x:x + w])
            cv2.rectangle(border_image, (x, y), (x + w, y + h), 255, 2)

    return border_image


if __name__ == "__main__":
    # for manual testing purposes
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to input image file")
    args = vars(parser.parse_args())

    # load the image from disk
    input_image = skimage.io.imread(args["image"], as_gray=True)

    # test operations
    binary_image = binarize_image(input_image)

    # show results
    rows = 1
    cols = 2
    fig, axs = plt.subplots(rows, cols, constrained_layout=True)

    axs[0].imshow(input_image, cmap=plt.cm.gray)
    axs[0].set_title('Input Image')

    axs[1].imshow(binary_image, cmap=plt.cm.gray)
    axs[1].set_title('Binary Image')

    plt.show()
