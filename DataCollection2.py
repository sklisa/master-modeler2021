#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 22:14:19 2021

@author: plutosirius
"""

import facebook
import json
from json.decoder import JSONDecodeError
import os
import glob
import pandas as pd

dir = os.getcwd()
df = pd.read_excel('Master Modeler Competition 2021.xlsx', usecols='D', names = ['url']) 
all_urls = list(df['url'])
raw_files = glob.glob(dir+'/RawData/'+'*.json')
unavailable_urls = []

token = 'EAADCE73DYfwBAELYvPEdKCpadWcwQdXRENab72OBnrZBQB4mjIC1rHZA49hKEoSZBPejrHErRe4cYd1cc0OPudzxm7AZBNG7VdOZB2aFdmyJ5aN6slzVRwaHEMFZCZAfbVmxRs6W7F878cEK4jDL4p8yU417ufKaBC18TJGOGw7tHRvcGWVlgo637aMEa12vzh1c0g45ujWKphnftmEdqYQOWAhxY5ByRoU0IgZAAwBzgwZDZD'
graph = facebook.GraphAPI(access_token=token)
info = open('unavailable_urls.txt', 'w')

page_id = '1003344626383879'

for file in raw_files:
    with open(file, 'r') as f:
        try:
            dct = json.load(f)
        except JSONDecodeError:
            idx = int(os.path.basename(file).replace('.json', ''))
            url = all_urls[idx]
            unavailable_urls.append((idx, all_urls[idx]))
            if 'helperase' in url or page_id in url:
                info.write(url)
                info.write('\n')
info.close()


for tup in unavailable_urls[::-1]:
    i, url = tup[0], tup[1]
    if 'helperase/posts' in url:
        fid = page_id + '_' + url[41:].strip('/').strip(':0')
    elif 'helperase/videos' in url:
        fid = page_id + '_' + url[42:].strip('/').strip(':0')
    elif '/?type=3' in url:
        fid = page_id + '_' + url[-24:-8].strip('/').strip(':0')
    else:
        continue
    output_file = open(dir+'/RawData/'+str(i)+'.json', 'w')
    try:
        event = graph.get_object(id=fid, fields='created_time,message,message_tags,place,permalink_url,attachments')
        print('success', url)
        json.dump(event, output_file)
    except facebook.GraphAPIError as e:
        # print(str(e))
        if str(e) == '(#100) Pages Public Content Access requires either app secret proof or an app token': 
            print('not accessible', url)
        elif str(e) == '(#4) Application request limit reached':
            print('limit reached')
            break
        else:
            print(str(e), url)