# import packages
import cv2
from math import sqrt
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

# fisheye transformation
# Cite from: https://github.com/Gil-Mor/iFish/blob/master/fish.py
def get_fish_xn_yn(source_x, source_y, radius, distortion):
    """
    Get normalized x, y pixel coordinates from the original image and return normalized
    x, y pixel coordinates in the destination fished image.
    :param distortion: Amount in which to move pixels from/to center.
    As distortion grows, pixels will be moved further from the center, and vice versa.
    """

    if 1 - distortion*(radius**2) == 0:
        return source_x, source_y

    return source_x / (1 - (distortion*(radius**2))), source_y / (1 - (distortion*(radius**2)))

def fish(img, distortion_coefficient):
    """
    :type img: numpy.ndarray
    :param distortion_coefficient: The amount of distortion to apply.
    :return: numpy.ndarray - the image with applied effect.
    """

    # If input image is only BW or RGB convert it to RGBA
    # So that output 'frame' can be transparent.
    w, h = img.shape[0], img.shape[1]
    if len(img.shape) == 2:
        # Duplicate the one BW channel twice to create Black and White
        # RGB image (For each pixel, the 3 channels have the same value)
        bw_channel = np.copy(img)
        img = np.dstack((img, bw_channel))
        img = np.dstack((img, bw_channel))
    if len(img.shape) == 3 and img.shape[2] == 3:
        print("RGB to RGBA")
        img = np.dstack((img, np.full((w, h), 255)))

    # prepare array for dst image
    dstimg = np.zeros_like(img)

    # floats for calcultions
    w, h = float(w), float(h)

    # easier calcultion if we traverse x, y in dst image
    for x in range(len(dstimg)):
        for y in range(len(dstimg[x])):

            # normalize x and y to be in interval of [-1, 1]
            xnd, ynd = float((2*x - w)/w), float((2*y - h)/h)

            # get xn and yn distance from normalized center
            rd = sqrt(xnd**2 + ynd**2)

            # new normalized pixel coordinates
            xdu, ydu = get_fish_xn_yn(xnd, ynd, rd, distortion_coefficient)

            # convert the normalized distorted xdn and ydn back to image pixels
            xu, yu = int(((xdu + 1)*w)/2), int(((ydu + 1)*h)/2)

            # if new pixel is in bounds copy from source pixel to destination pixel
            if 0 <= xu and xu < img.shape[0] and 0 <= yu and yu < img.shape[1]:
                dstimg[x][y] = img[xu][yu]

    return dstimg.astype(np.uint8)

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
        print("Cropped image (fovea) shape", cropped_img.shape)

        # apply fisheye transformation to the fovea region
        new_cropped_img = fish(cropped_img, distortion_coefficient=0.8)

        # trim the black borders
        # https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
        gray = cv2.cvtColor(new_cropped_img, cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)

        trimmed = new_cropped_img[y:y+h,x:x+w]

        # zoom out the cropped image
        trimmed_scaled = cv2.resize(trimmed, dsize=(new_cropped_img.shape[1], new_cropped_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        # paste the new cropped_img with gaussian noises back to the original image
        w, h = img.shape[0], img.shape[1]
        img = np.dstack((img, np.full((w, h), 255))) # add one more channel to the image
        # img[int(scaled_box[1]):int(scaled_box[3]), int(scaled_box[0]):int(scaled_box[2])] = new_cropped_img
        img[int(scaled_box[1]):int(scaled_box[3]), int(scaled_box[0]):int(scaled_box[2])] = trimmed_scaled
        print("New image shape", img.shape)

        # save the new images (fisheye transformed fovea region) to a folder
        new_path_name = '/Users/yangxiyu/Desktop/harvard/2022 Fall/ac297r/images/img_with_fisheye_trf/new' + num + '.png' # original image shape
        cv2.imwrite(new_path_name, img)

        # crop the zero paddings from the fisheye transformed images (using PIL)
        # https://stackoverflow.com/questions/10615901/trim-whitespace-using-pil/10616717#10616717
        # from PIL import Image, ImageChops
        #
        # def trim(im):
        #     bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        #     diff = ImageChops.difference(im, bg)
        #     diff = ImageChops.add(diff, diff, 2.0, -100)
        #     bbox = diff.getbbox()
        #     if bbox:
        #         return im.crop(bbox)
        #
        # im = Image.fromarray(new_cropped_img, mode='RGBA')
        # im = trim(im)
        # im.show()
