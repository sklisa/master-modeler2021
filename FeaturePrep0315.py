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
in_face = dir+'/json_face_modified.csv'
in_tn_face = dir+'/json_tn_face_modified.csv'
out_dir = dir+'/PrepData0315/'
list_of_dct = []
list_of_modified = []
unavailable_json = []


face = pd.read_csv(in_face)
tn_face = pd.read_csv(in_tn_face)

# 2074; 741.json new_album is duplicated; cannot retrieve image, set face_present = 0

# merge datasets
json_face = pd.concat([face, tn_face], ignore_index=True)
print(json_face.loc[json_face['json_id']==741])
print(len(json_face))
print(len(face))
print(len(tn_face))

# key_diff = set(tn_face.json_id).difference(set(face.json_id))
# print(len(key_diff))
# print(set(tn_face.json_id).difference(key_diff))

for file in in_files:
    with open(file, 'r') as f:
        filename = os.path.basename(file)
        try:
            dct = json.load(f)
            new_dct = dct

            new_dct['face_present'] = 0
            # print(json_face[json_face['json_id']== 1618])
            if pd.to_numeric(filename[:-5]) in list(json_face['json_id']):
                # print(filename)
                # print(json_face['face_present'].loc[json_face['json_id'] == pd.to_numeric(filename[:-5])].values[0])
                if json_face['face_present'].loc[json_face['json_id'] == pd.to_numeric(filename[:-5])].values[0] == 1:
                    new_dct['face_present'] = 1
                elif json_face['face_present'].loc[json_face['json_id'] == pd.to_numeric(filename[:-5])].values[0] == 2:
                    new_dct['face_present'] = 2
                    list_of_modified.append(filename)
            out_file = open(out_dir + filename, 'w')
            json.dump(new_dct, out_file)
            list_of_dct.append(new_dct)
        except JSONDecodeError:
            unavailable_json.append(filename)

df = pd.DataFrame(list_of_dct)
print(len(list_of_modified))  # 746 face_present==1; 65 face_present==2
df.to_csv('dataset0315.csv', index=False)  # sep='\t'


