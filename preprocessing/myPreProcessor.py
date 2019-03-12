from functools import cmp_to_key

import skimage

from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.morphology import closing, square, remove_small_holes
from skimage.segmentation import clear_border
import numpy as np

line_width_height_ratio = 50
max_same_line_diff = 30
initially_cropped_area = 30


def compare_regions(region1, region2):
    centroid1 = region1.local_centroid
    centroid2 = region2.local_centroid
    if abs(centroid1[0] - centroid2[0]) < 100:
        return centroid1[1] - centroid2[1]
    return centroid1[0] - centroid2[0]


def preprocessImage(image):
    image = skimage.img_as_float(image)
    image = image[initially_cropped_area:, initially_cropped_area:]
    thresh = threshold_otsu(image)
    binary_otsu_image = closing(image > thresh, square(3))
    binary_otsu_image = remove_small_holes(binary_otsu_image, 200, connectivity=2)
    # plt.imshow(binary_otsu_image, cmap=plt.cm.gray)
    # plt.show()
    connected_component_labels, num_labels = label(binary_otsu_image, connectivity=2, return_num=True, background=True)
    connected_component_labels = clear_border(connected_component_labels)
    image_label_overlay = label2rgb(connected_component_labels, image=image)
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image_label_overlay)
    lineCounter = 0
    crop_points = []
    last_distance = 0
    line_regions = []
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
            line_regions.append(region)
            if lineCounter == 2:
                image_max_row = max(image_max_row, maxr)
                image_max_column = max(image_max_column, maxc)
                image_min_column = min(image_min_column, minc)
                image_min_row = min(image_min_row, minr)
            # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='blue', linewidth=1)
            # ax.add_patch(rect)
    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()
    cropped_image = image[image_min_row:image_max_row, image_min_column:image_max_column]
    # # return [cropped_image]
    # plt.imshow(cropped_image)
    # plt.show()
    new_image = np.full(cropped_image.shape, fill_value=1.0, dtype=image.dtype)
    x_value = 0
    y_value = 0
    sum_line_height = 0
    max_line_height = 0
    lineCounter = 0
    thresh = threshold_otsu(cropped_image)
    binary_otsu_image = closing(cropped_image > thresh, square(3))
    connected_component_labels, num_labels = label(binary_otsu_image, connectivity=2, return_num=True, background=True)
    connected_component_labels = clear_border(connected_component_labels)
    line_regions = []
    for wanted_region in regionprops(connected_component_labels):
        min_row, min_column, max_row, max_column = wanted_region.bbox
        if True or image_min_row < min_row < image_max_row and image_min_row < max_row < image_max_row and image_min_column < min_column < image_max_column and image_min_column < max_column < image_max_column:
            row_count = max_row - min_row
            column_count = max_column - min_column
            if x_value + column_count > new_image.shape[1]:
                # Now fill the image and align center of gravity
                x_value = 0
                for index, line_region in enumerate(line_regions):
                    region_min_row, region_min_column, region_max_row, region_max_column = line_region.bbox
                    region_row_count = region_max_row - region_min_row
                    region_column_count = region_max_column - region_min_column
                    hull = line_region.image
                    starting_y_value = int(np.floor(max_line_height / 2)) - int(
                        np.floor((region_max_row - region_min_row) / 2))
                    if starting_y_value < 0:
                        raise RuntimeError
                    starting_y_value += y_value
                    if starting_y_value + region_row_count > new_image.shape[0] or x_value + region_column_count > \
                            new_image.shape[1]:
                        raise RuntimeError
                    new_image[starting_y_value:(starting_y_value + region_row_count),
                    x_value:(x_value + region_column_count)][hull] = \
                    cropped_image[region_min_row:region_max_row, region_min_column:region_max_column][hull]
                    x_value += region_column_count
                    # plt.imshow(new_image)
                    # plt.show()
                # advance y value with average /2
                y_value += int(np.ceil((sum_line_height / len(line_regions)) * 1))
                # print(y_value)
                x_value = 0
                sum_line_height = 0
                lineCounter = 0
                max_line_height = 0
                line_regions = []
            # print((min_row,max_row,max_column,min_column))
            # print (np.count_nonzero(new_image == 0))

            # new_image_slice =new_image[y_value:y_value + row_count, x_value:x_value + column_count]
            line_regions.append(wanted_region)
            # print('Shit ' + str(np.count_nonzero(new_image[y_value:(y_value + row_count), x_value:(x_value + column_count)] != image[min_row:max_row, min_column:max_column])))
            # plt.imshow(image[min_row:max_row, min_column:max_column])
            # plt.show()
            # plt.imshow(new_image[y_value:(y_value + row_count), x_value:(x_value + column_count)])
            # plt.show()
            # new_image_slice = image_slice
            x_value += column_count
            sum_line_height += row_count
            max_line_height = max(max_line_height, row_count)
            lineCounter += 1
    y_value += max_line_height
    new_image = new_image[:y_value, :]
    # skimage.io.imsave('texture.png', new_image, cmap=plt.cm.gray)
    returned_images = []
    # plt.imshow(new_image, cmap=plt.cm.gray)
    # plt.show()
    if new_image.shape[0] < 128:
        return returned_images
    else:
        x_value = 0
        y_value = 0
        # skimage.io.imsave('slices/texture.png', new_image, cmap=plt.cm.gray)
        while x_value < new_image.shape[1]:
            if x_value + 265 > new_image.shape[1]:
                if new_image.shape[0] - (y_value + 128) < 128:
                    #     for i in range(len(returned_images)):
                    #         skimage.io.imsave('slices/' + str(i) + '.png', returned_images[i], cmap=plt.cm.gray)
                    return returned_images
                x_value = 0
                y_value += 128
            returned_images.append(new_image[y_value:y_value + 128, x_value:x_value + 256])
            x_value += 265

    return returned_images


if __name__ == '__main__':
    input_image = skimage.io.imread('1.png', as_gray=True)
    preprocessImage(input_image)
    input_image = skimage.io.imread('/home/hassan/Documents/PatternProject/iamDB/forms/b01-122.png', as_gray=True)
    preprocessImage(input_image)
    input_image = skimage.io.imread('/home/hassan/Documents/PatternProject/iamDB/forms/f04-064.png', as_gray=True)
    preprocessImage(input_image)
    input_image = skimage.io.imread('/home/hassan/Documents/PatternProject/iamDB/forms/c02-012.png', as_gray=True)
    preprocessImage(input_image)
