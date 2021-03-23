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

# Image processing (photo & thumbnail)


dir = os.getcwd()
in_files = glob.glob(dir+'/PrepData0313/'+'*.json')
# in_files = dir+'/dataset0313.csv'
image_out_dir = dir+'/PrepImage/'
tn_out_dir = dir+'/PrepTN/'
list_of_dct = []
list_of_dct2 = []
unavailable_json = []

# https://towardsdatascience.com/simple-face-detection-in-python-1fcda0ea648e
# https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php


# Input data
# input = pd.read_csv(in_files)
# print(sum(pd.notnull(input['thumbnail_url'])))  # 562 with photo type; # 1512 thumbnail

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



def detect_print(image_url, filename, out_dir):
    print(image_url)
    if image_url == 'https://external-lax3-1.xx.fbcdn.net/safe_image.php?d=AQFwNwXgZM3XiZ1R&w=698&h=698&url=https%3A%2F%2Fwww.swcbulletin.com%2Fsites%2Fdefault%2Ffiles%2Fstyles%2F16x9_1240%2Fpublic%2F1_2DMvVxOlxEkjoJViBAdWP1k5_d1ufEo.jpg%3Fitok%3D0U78mR2S&cfs=1&sx=0&sy=0&sw=698&sh=698&_nc_cb=1&_nc_hash=AQGH0XiLaHJ14MGQ':
        return

    try:
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
        if len(faces) > 0:
            return 1
        else:
            return 0
    except Exception as e:
        print(filename, e)
        unavailable_json.append(filename+out_dir)


def main():
    for file in in_files:
        with open(file, 'r') as f:
            filename = os.path.basename(file)[:-5]    # only number
            try:
                dct = json.load(f)

                # Extract from photo
                # if dct['photo_url'] is not None:
                #     if type(dct['photo_url']) is str:   # single image
                #         print(filename, 'has one image')
                #         image_url = dct['photo_url']
                #         face_present = detect_print(image_url, filename+'.png', image_out_dir)
                #     elif type(dct['photo_url']) is list:    # album / multiple image
                #         face_present_list = []
                #         print(filename, 'has multiple image')
                #         for url in dct['photo_url']:
                #             image_url = url
                #             face_present = detect_print(image_url, filename+'-'+str(dct['photo_url'].index(url))+'.png', image_out_dir)
                #             face_present_list.append(face_present)
                #     new_dct = {'json_id': filename, 'face_present': face_present_list}
                #     # print(new_dct)
                #     list_of_dct.append(new_dct)

                # Extract from thumbnail
                # if dct['thumbnail_url'] is not None:
                #     if type(dct['thumbnail_url']) is str:
                #         image_url = dct['thumbnail_url']
                #         face_present = detect_print(image_url, filename+'.png', tn_out_dir)
                #     else:
                #         print(filename, 'thumbnail is not str but [], pass')
                #     new_dct = {'json_id': filename, 'face_present': face_present}
                #     list_of_dct2.append(new_dct)

            except JSONDecodeError:
                unavailable_json.append(filename)

    cv2.destroyAllWindows()
    # df = pd.DataFrame(list_of_dct)
    # df.to_csv('json_face.csv', index=False)
    # df2 = pd.DataFrame(list_of_dct2)
    # df2.to_csv('json_tn_face.csv', index=False)

    print(unavailable_json)


if __name__ == "__main__":
    main()




