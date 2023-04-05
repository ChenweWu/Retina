# import packages
import cv2
from math import sqrt
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

# helper functions
# reference code: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
# add gaussian noise
def add_gaussian_noises(img, mean=0, var=1):
      row, col, ch= img.shape
      mean = mean
      var = var
      sigma = var**0.5
      gauss = np.random.normal(mean, sigma, (row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy_img = img + gauss
      return noisy_img

# run some functions
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

        # add gaussian noises to the cropped fovea region
        new_cropped_img = add_gaussian_noises(cropped_img, mean=0, var=200)

        # paste the new cropped_img with gaussian noises back to the original image
        img[int(scaled_box[1]):int(scaled_box[3]), int(scaled_box[0]):int(scaled_box[2])] = new_cropped_img
        # cv2.imshow('New image with noises', img)
        # cv2.waitKey(0)
        # print("New image shape", img.shape)

        # save the new images (fovea region with gaussian noises) to a folder
        new_img = Image.fromarray(img[:,:,::-1].astype('uint8'), 'RGB')
        new_path_name = '/Users/yangxiyu/Desktop/harvard/2022 Fall/ac297r/images/img_with_gaussian_noises/new' + num + '.jpg' # original image shape
        new_img.save(new_path_name)
