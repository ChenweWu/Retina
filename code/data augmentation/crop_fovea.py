# import packages
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

# run some functions
if __name__ == "__main__":
    # read in bounding boxes
    bounding_boxes = pd.read_csv("/Users/yangxiyu/Desktop/harvard/2022 Fall/ac297r/images/result_box.csv", header=None)
    print(bounding_boxes.head(3))
    bounding_boxes = bounding_boxes.iloc[:, 1:] # drop the first index column
    print(bounding_boxes.head(3))
    print("Value type:", type(bounding_boxes.iloc[0, 0]))

    # crop the fovea region from all sample images
    for i in range(100):
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
        # print("scaled bounding box", scaled_box)

        # crop the fovea region according to its scaled bounding box (rectangle)
        cropped_img = img[int(scaled_box[1]):int(scaled_box[3]), int(scaled_box[0]):int(scaled_box[2])]

        # save cropped image (fovea region to a folder)
        cropped_img = Image.fromarray(cropped_img[:,:,::-1].astype('uint8'), 'RGB')
        cropped_path_name = '/Users/yangxiyu/Desktop/harvard/2022 Fall/ac297r/images/cropped_fovea_rec/cropped' + num + '.jpg' # rectangle shape
        cropped_img.save(cropped_path_name)
