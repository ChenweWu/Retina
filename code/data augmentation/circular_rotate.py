# import packages
import cv2
from math import sqrt
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

# helper functions
# circular rotate (cv2 package)
# https://stackoverflow.com/questions/61516526/how-to-use-opencv-to-crop-circular-image
def circular_rotate(img, scaled_box):
    # find the center of fovea region
    x_center = int(scaled_box[0] + 1/2 * (scaled_box[2] - scaled_box[0]))
    y_center = int(scaled_box[1] + 1/2 * (scaled_box[3] - scaled_box[1]))
    radius = int(max([(scaled_box[2] - scaled_box[0]), (scaled_box[3] - scaled_box[1])]) / 2)
    # print("(x, y)", (x_center, y_center))
    # print("radius", radius)

    # create mask
    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (x_center, y_center), radius, (255,255,255), thickness = -1)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)

    # put mask into alpha channel of input
    result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask[:, :, 0]

    # display result
    print("result shape (before rotation)", result.shape)
    # the following code cannot display the correct image
    # cv2.imshow('Masked image', result)
    # cv2.waitKey(0)

    # rotate image centered at the given center
    # result = cv2.rotate(result, cv2.ROTATE_180) # without a center
    image_center = (x_center, y_center)
    angle = 180
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(result, rot_mat, result.shape[1::-1], flags=cv2.INTER_LINEAR)
    print("result shape (after rotation)", result.shape)
    cv2.imwrite('/Users/yangxiyu/Desktop/test_masked_image.png', result)

    # display the whole rotated image
    rotated_img = cv2.warpAffine(img, rot_mat, result.shape[1::-1], flags=cv2.INTER_LINEAR)
    cv2.imshow('Rotated image', rotated_img)
    cv2.waitKey(0)

    # paste the rotated fovea region with the original image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) # add an alpha channel to img
    mask = np.dstack((mask, mask[:, :, 0])) # add an alpha channel to mask
    # print("Stacked mask shape", mask.shape)
    locs = np.where(mask != 0) # get the non-zero mask locations
    # print("Locations of nonzero mask values", locs)

    img[locs[0], locs[1], locs[2]] = result[locs[0], locs[1], locs[2]]
    cv2.imwrite('/Users/yangxiyu/Desktop/test_masked_rotated_image.png', img)

# apply circular rotation on images
if __name__ == "__main__":
    # read in bounding boxes
    bounding_boxes = pd.read_csv("/Users/yangxiyu/Desktop/harvard/2022 Fall/ac297r/images/result_box.csv", header=None)
    print(bounding_boxes.head(3))
    bounding_boxes = bounding_boxes.iloc[:, 1:] # drop the first index column
    print(bounding_boxes.head(3))
    print("Value type:", type(bounding_boxes.iloc[0, 0]))

    # crop the fovea region from all sample images
    for i in range(1):
        # create image id
        if len(str(i+1)) < 3:
            num = "0" * (3-len(str(i+1))) + str(i+1)
        else:
            num = str(i+1)
        img_id = "img00" + num

        # import sample images
        path_name = '/Users/yangxiyu/Desktop/harvard/2022 Fall/ac297r/images/Image/' + img_id + '.jpg'
        img = cv2.imread(path_name)
        # cv2.imshow('Sample image', img)
        # cv2.waitKey(0)
        # print("Sample image shape", img.shape)

        # identify and scale the bounding box
        box = list(bounding_boxes.iloc[i, :4])
        # print("bounding box", box)
        x_scale = img.shape[1]/800
        y_scale = img.shape[0]/800
        scaled_box = [box[0]*x_scale, box[1]*y_scale, box[2]*x_scale, box[3]*y_scale]
        print("scaled bounding box", scaled_box)

        # crop the fovea region according to its scaled bounding box (rectangle)
        cropped_img = img[int(scaled_box[1]):int(scaled_box[3]), int(scaled_box[0]):int(scaled_box[2])]
        print("Cropped image (fovea) shape", cropped_img.shape)

        # circular rotation
        x_center = scaled_box[0] + 1/2 * (scaled_box[2] - scaled_box[0])
        y_center = scaled_box[1] + 1/2 * (scaled_box[3] - scaled_box[1])
        radius = int(max([(scaled_box[2] - scaled_box[0]), (scaled_box[3] - scaled_box[1])]) / 2)
        print("Scaled bounding box:", scaled_box)
        print("Radius:", radius)

        # add the alpha channel (transparency) to the img
        if len(img.shape) == 3 and img.shape[2] == 3:
            w, h = img.shape[0], img.shape[1]
            print("RGB to RGBA")
            new_img = np.dstack((img, np.full((w, h), 255)))

        # rotated_img_arr, rotated_img = circle_rotate(image=new_img, x=x_center, y=y_center, radius=radius, degree=45)
        circular_rotate(img, scaled_box)
        
