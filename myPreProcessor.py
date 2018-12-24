import skimage

from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.morphology import closing, square, remove_small_holes
from skimage.segmentation import clear_border
from skimage.util import invert
import numpy as np

line_width_height_ratio = 50
max_same_line_diff = 30
initially_cropped_area = 30


def preprocessImage(image):
    image = image[initially_cropped_area:, initially_cropped_area:]
    thresh = threshold_otsu(image)
    binary_otsu_image = closing(image > thresh, square(3))
    binary_otsu_image = remove_small_holes(binary_otsu_image, 400, connectivity=2)
    # plt.imshow(binary_otsu_image, cmap=plt.cm.gray)
    # plt.show()
    connected_component_labels, num_labels = label(binary_otsu_image, connectivity=2, return_num=True, background=True)
    connected_component_labels = clear_border(connected_component_labels)
    # plt.imshow(connected_component_labels, cmap=plt.cm.gray)
    # plt.show()
    # image_label_overlay = label2rgb(connected_component_labels, image=image)
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image_label_overlay)
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
    #         rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='blue', linewidth=1)
    #         ax.add_patch(rect)
    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()
    cropped_image = image[image_min_row:image_max_row, image_min_column:image_max_column]
    # # return [cropped_image]
    # plt.imshow(cropped_image)
    # plt.show()
    new_image = np.full(cropped_image.shape, 255)
    x_value = 0
    y_value = 0
    prev_max_row = wanted_regions[0].bbox[1]
    sum_line_height = 0
    max_line_height = 0
    lineCounter = 0
    for wanted_region in wanted_regions:
        min_row, min_column, max_row, max_column = wanted_region.bbox
        if image_min_row < min_row < image_max_row and image_min_row < max_row < image_max_row and image_min_column < min_column < image_max_column and image_min_column < max_column < image_max_column:
            hull = wanted_region.convex_image
            # print(hull.dtype)
            row_count = max_row - min_row
            column_count = max_column - min_column
            if x_value + column_count > new_image.shape[1]:
                y_value += int(np.floor((sum_line_height / lineCounter) * 0.5))
                x_value = 0
                sum_line_height = 0
                lineCounter = 0
                max_line_height = 0
            # print((min_row,max_row,max_column,min_column))
            # print (np.count_nonzero(new_image == 0))
            if y_value + row_count > new_image.shape[0] or x_value + column_count > new_image.shape[1]:
                raise RuntimeError
            # new_image_slice =new_image[y_value:y_value + row_count, x_value:x_value + column_count]
            new_image[y_value:(y_value + row_count), x_value:(x_value + column_count)][hull] = \
            image[min_row:max_row, min_column:max_column][hull]
            # new_image_slice = image_slice
            x_value += column_count
            sum_line_height += row_count
            max_line_height = max(max_line_height, row_count)
            lineCounter += 1
    y_value += max_line_height
    new_image = new_image[:y_value, :]
    returned_images = []
    if new_image.shape[0] < 128:
        return returned_images
    else:
        x_value = 0
        while x_value < new_image.shape[1]:
            if x_value + 265 > new_image.shape[1]:
                return returned_images
            returned_images.append(new_image[:, x_value:x_value + 256])
            x_value += 265
    # plt.imshow(new_image,cmap=plt.cm.gray)
    # plt.show()
    return returned_images


if __name__ == '__main__':
    input_image = skimage.io.imread('/home/hassan/Documents/PatternProject/iamDB/forms/a01-003u.png', as_gray=True)
    preprocessImage(input_image)
    input_image = skimage.io.imread('/home/hassan/Documents/PatternProject/iamDB/forms/b01-122.png', as_gray=True)
    preprocessImage(input_image)
    input_image = skimage.io.imread('/home/hassan/Documents/PatternProject/iamDB/forms/f04-064.png', as_gray=True)
    preprocessImage(input_image)
    input_image = skimage.io.imread('/home/hassan/Documents/PatternProject/iamDB/forms/c02-012.png', as_gray=True)
    preprocessImage(input_image)
