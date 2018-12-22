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
    input_image = cv2.imread(args["image"])

    # test operations
    binary_image = binarize_image(input_image)
    textured_image = build_texture(binary_image)

    # show results
    cv2.imshow("Input Image", input_image)
    cv2.imshow("Binary Image", binary_image)
    cv2.imshow("Textured Image", textured_image)
    cv2.waitKey(0)
