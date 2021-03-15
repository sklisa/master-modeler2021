import json
from json.decoder import JSONDecodeError
import os
import glob
import cv2
import dlib
from skimage import io
import pandas as pd
import numpy as np
from urllib.parse import urlparse


dir = os.getcwd()
in_files = glob.glob(dir+'/PrepData0313/'+'*.json')
# in_files = dir+'/dataset0313.csv'
out_dir = dir+'/PrepImage/'
list_of_dct = []
unavailable_json = []

# https://towardsdatascience.com/simple-face-detection-in-python-1fcda0ea648e
# https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php


# Input data
# input = pd.read_csv(in_files)
# print(sum(pd.notnull(input['photo_url'])))  # 562 with photo type

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_print(image_url, filename):
    print(image_url)
    # Read image from URL
    img = io.imread(image_url)
    # Turn into gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Save annotated image
    cv2.imwrite(out_dir + filename, img)
    return


def main():
    for file in in_files:
        with open(file, 'r') as f:
            filename = os.path.basename(file)[:-5]    # only number
            try:
                dct = json.load(f)
                if dct['photo_url'] is not None:
                    if type(dct['photo_url']) is str:
                        print(filename, 'has one image')
                        image_url = dct['photo_url']
                        detect_print(image_url, filename+'.png')
                    elif type(dct['photo_url']) is list:
                        print(filename, 'has multiple image')
                        for url in dct['photo_url']:
                            image_url = url
                            detect_print(image_url, filename + '-' + str(dct['photo_url'].index(url)) + '.png')
            except JSONDecodeError:
                unavailable_json.append(filename)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# print('# faces:', len(faces))



