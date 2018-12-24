import skimage

from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.util import invert
import numpy as np

line_width_height_ratio = 50
max_same_line_diff = 30
initially_cropped_area = 30


def preprocess_image(image):
    image = image[initially_cropped_area:, initially_cropped_area:]
    thresh = threshold_otsu(image)
    binary_otsu_image = closing(image > thresh, square(3))
    binary_otsu_image = invert(binary_otsu_image)
    plt.imshow(binary_otsu_image, cmap=plt.cm.gray)
    plt.show()
    binary_otsu_image = clear_border(binary_otsu_image)
    connected_component_labels, num_labels = label(binary_otsu_image, connectivity=2, return_num=True)
    plt.imshow(connected_component_labels, cmap=plt.cm.gray)
    plt.show()
    image_label_overlay = label2rgb(connected_component_labels, image=image)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)
    lineCounter = 0
    crop_points = []
    last_distance = 0
    wanted_regions = []
    image_min_row = 10000
    image_min_column = 10000
    image_max_row = 0
    image_max_column = 0
    for region in regionprops(connected_component_labels):
        minr, minc, maxr, maxc = region.bbox
        if (maxc - minc) / (maxr - minr) > line_width_height_ratio:
            if lineCounter >= 1 and maxr - last_distance > max_same_line_diff:
                crop_points.append(maxr)
                last_distance = maxr
                lineCounter += 1
            elif maxr - last_distance > max_same_line_diff:
                lineCounter += 1
                last_distance = maxr
        else:
            wanted_regions.append(region)
            if lineCounter == 2:
                image_max_row = max(image_max_row, maxr)
                image_max_column = max(image_max_column, maxc)
                image_min_column = min(image_min_column, minc)
                image_min_row = min(image_min_row, minr)
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='blue', linewidth=1)
            ax.add_patch(rect)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    cropped_image = image[image_min_row:image_max_row, image_min_column:image_max_column]
    plt.imshow(cropped_image)
    plt.show()
    new_image = np.zeros(cropped_image.shape)
    x_value = 0
    y_value = 0
    prev_max_row = wanted_regions[0].bbox[1]
    average_line_height = 0
    lineCounter = 0
    for wanted_region in wanted_regions:
        min_row, max_row, min_column, max_column = wanted_region.bbox
        if max_row - prev_max_row > 30:
            y_value += np.floor((average_line_height / lineCounter) * 0.5)
        prev_max_row = max_row
        row_count = max_row - min_row
        column_count = max_column - min_column
        print((min_row,max_row,max_column,min_column))
        new_image[x_value:x_value + row_count, y_value:y_value + column_count] = image[min_row:max_row,
                                                                                 min_column:max_column]
        x_value += row_count
        average_line_height += row_count
        lineCounter += 1
    plt.imshow(new_image)
    plt.show()


if __name__ == '__main__':
    input_image = skimage.io.imread('/home/hassan/Documents/PatternProject/iamDB/forms/a01-003u.png', as_gray=True)
    preprocess_image(input_image)
