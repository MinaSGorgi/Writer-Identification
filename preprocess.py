import argparse
import cv2


def binarize_image(image, gsize=(3, 3), csize=(3, 3)):
    """
    Performs: Gaussian blur -> Otsu binarization -> Closing .

    Args:
        image: The BGR image to process.
        gsize: The Gaussian kernel size, defaults to (3, 3).
        csize:  The closing kernel size, defaults to (3, 3).

    Returns:
        The resultant new binary image where foreground is white.

    Raises:
        None
    """

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, gsize, 0)
    __, thresh_image = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, csize)
    closed_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, closing_kernel)

    inverted_image = cv2.bitwise_not(closed_image)

    return inverted_image


if __name__ == "__main__":
    # for manual testing purposes
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to input image file")
    args = vars(parser.parse_args())

    # load the image from disk
    input_image = cv2.imread(args["image"])

    # test operations
    binary_image = binarize_image(input_image)

    # show results
    cv2.imshow("Input Image", input_image)
    cv2.imshow("Binary Image", binary_image)
    cv2.waitKey(0)
