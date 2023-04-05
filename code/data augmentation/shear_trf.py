# import packages
import cv2
from math import sqrt
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from skimage import transform as tf

# helper functions
# shear transformation
def shear_trf(img, scaled_box):
    # step 1: crop the fovea region
    # find the center and the radius of the fovea region
    x_center = int(scaled_box[0] + 1/2 * (scaled_box[2] - scaled_box[0]))
    y_center = int(scaled_box[1] + 1/2 * (scaled_box[3] - scaled_box[1]))
    radius = int(max([(scaled_box[2] - scaled_box[0]), (scaled_box[3] - scaled_box[1])]) / 2)

    # create mask
    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (x_center, y_center), radius, (255,255,255), thickness = -1)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)

    # put mask into alpha channel of input
    cropped_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    cropped_img[:, :, 3] = mask[:, :, 0]

    # display the cropped image
    print("new image shape (before shear transformation)", cropped_img.shape)
    cv2.imwrite('/Users/yangxiyu/Desktop/shear_trf_masked_image.png', cropped_img)

    # step 2: shear transform the cropped image
    # define radian (/angle) for shear transformation
    sign = np.random.choice([-1, 1], size=1)
    range = np.random.choice([np.pi/6, np.pi/5, np.pi/4], size=1)
    radian = np.random.random(size=1) * range * sign
    print("Radian:", radian)

    # shear transform the cropped image
    afine_tf = tf.AffineTransform(shear=radian)
    img_shear = tf.warp(cropped_img, inverse_map=afine_tf)
    img_shear = np.uint8(img_shear*255)

    # save the correctly sheared image
    cv2.imwrite('/Users/yangxiyu/Desktop/shear_trf_sheared_image.png', img_shear)
    ## cannot properly display the image
    # cv2.imshow('Sheared image', img_shear)
    # cv2.waitKey(0)

    # step 3: crop the sheared fovea region as a rectangle image
    # find rectangle containing the sheared fovea
    mask_shear = img_shear[:, :, 3]
    print("Sheared mask shape", mask_shear.shape)
    locs = np.where(mask_shear != 0) # get the non-zero mask locations
    print("Locations of nonzero mask values", locs)

    # find the end points of the rectangle
    x_min = min(locs[0]); x_max = max(locs[0])
    y_min = min(locs[1]); y_max = max(locs[1])

    # crop the rectangle containing the sheared fovea
    fovea_shear = img_shear[x_min:x_max, y_min:y_max, :]
    cv2.imwrite('/Users/yangxiyu/Desktop/shear_trf_sheared_fovea.png', fovea_shear)

    # step 4: resize and paste the sheared fovea rectangle to the bounding box image
    # resize the sheared fovea rectangle
    width = int(scaled_box[3]) - int(scaled_box[1])
    height = int(scaled_box[2]) - int(scaled_box[0])
    dim = (height, width) # width and height are computed in step 3
    fovea_shear_resized = cv2.resize(fovea_shear, dim, interpolation = cv2.INTER_AREA)
    print("dimension of resized fovea rectangle", fovea_shear_resized.shape)
    cv2.imwrite('/Users/yangxiyu/Desktop/shear_trf_sheared_resized_fovea.png', fovea_shear_resized)

    # step 5: paste the resized sheared fovea rectangle back to the original image
    # version 1: with transparency channel
    # new_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # new_img[int(scaled_box[1]):int(scaled_box[3]), int(scaled_box[0]):int(scaled_box[2])] = fovea_shear_resized
    # cv2.imwrite('/Users/yangxiyu/Desktop/shear_trf_new_image.png', new_img)

    # version 2: no transparency channel
    new_img = img[:, :, :]
    new_img[int(scaled_box[1]):int(scaled_box[3]), int(scaled_box[0]):int(scaled_box[2])] = fovea_shear_resized[:, :, :3]
    cv2.imwrite('/Users/yangxiyu/Desktop/shear_trf_new_image.png', new_img)

# apply shear transformation on image
if __name__ == "__main__":
    # read in bounding boxes
    bounding_boxes = pd.read_csv("/Users/yangxiyu/Desktop/harvard/2022 Fall/ac297r/images/result_box.csv", header=None)
    bounding_boxes = bounding_boxes.iloc[:, 1:] # drop the first index column

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

        # identify and scale the bounding box
        box = list(bounding_boxes.iloc[i, :4])
        x_scale = img.shape[1]/800
        y_scale = img.shape[0]/800
        scaled_box = [box[0]*x_scale, box[1]*y_scale, box[2]*x_scale, box[3]*y_scale]
        print("scaled bounding box", scaled_box)

        # conduct shear transformation
        np.random.seed(0)
        shear_trf(img, scaled_box)
