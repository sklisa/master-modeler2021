import json
from json.decoder import JSONDecodeError
import os
import glob
import pandas as pd
import numpy as np
from urllib.parse import urlparse


dir = os.getcwd()
filter_files = glob.glob(dir+'/FilteredData/'+'*.json')
out_dir = dir+'/PrepData0313/'
engagement_data = dir+'/Master Modeler Competition 2021 - ERASE - FB post collection (1_2017 to current).csv'
list_of_dct = []
list_of_filtered = []
unavailable_json = []

# Events
# January Human Trafficking Awareness Month
# SuperBowl: Feb. 5, 2017; Feb. 4, 2018; Feb. 3, 2019; Feb. 2, 2020; Feb. 7, 2021
# Marti Gras: February 28, 2017; February 13, 2018; March 5, 2019; February 25, 2020; February 16, 2021
# March Madness: second half of March + first week of April; 2020 cancelled
# July 30 World Day Against Trafficking in Person

engagement = pd.read_csv(engagement_data, usecols=['URL', 'total engagement', 'engagement rate', 'reactions', 'shares', 'comments'])
engagement = engagement.rename({"total engagement": "total_engagement", "engagement rate": "engagement_rate"}, axis=1)


class MediaTypeError(Exception):
    pass


class URLMatchError(Exception):
    pass


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


for file in filter_files:
    with open(file, 'r') as f:
        filename = os.path.basename(file)
        try:
            dct = json.load(f)
            new_dct = dct
            new_dct['date'] = dct['created_time'][0:10]
            # keep date as string bcs datetime format is not supported by JSON
            new_dct['date_dt'] = pd.to_datetime(new_dct['date'])

            # day of week
            new_dct['day_week'] = new_dct['date_dt'].dayofweek
            new_dct['Mon'] = 0
            new_dct['Tue'] = 0
            new_dct['Wed'] = 0
            new_dct['Thur'] = 0
            new_dct['Fri'] = 0
            new_dct['Sat'] = 0
            if new_dct['day_week'] == 0:
                new_dct['Mon'] = 1
            elif new_dct['day_week'] == 1:
                new_dct['Tue'] = 1
            elif new_dct['day_week'] == 2:
                new_dct['Wed'] = 1
            elif new_dct['day_week'] == 3:
                new_dct['Thur'] = 1
            elif new_dct['day_week'] == 4:
                new_dct['Fri'] = 1
            elif new_dct['day_week'] == 5:
                new_dct['Sat'] = 1

            # event
            new_dct['event'] = 0
            if new_dct['date_dt'].month == 1:
                new_dct['event'] = 1
            elif new_dct['date_dt'].month == 7 and new_dct['date_dt'].day == 30:
                new_dct['event'] = 1

            # season
            new_dct['winter'] = 0
            new_dct['spring'] = 0
            new_dct['summer'] = 0
            if new_dct['date_dt'].month == 12 or new_dct['date_dt'].month <= 2:
                new_dct['winter'] = 1
            elif 3 <= new_dct['date_dt'].month <= 5:
                new_dct['spring'] = 1
            elif 6 <= new_dct['date_dt'].month <= 8:
                new_dct['summer'] = 1

            # time of day
            new_dct['hour'] = dct['created_time'][11:13]
            new_dct['hour'] = pd.to_numeric(new_dct['hour'])
            new_dct['morning'] = 0
            new_dct['afternoon'] = 0
            new_dct['evening'] = 0
            if 6 <= new_dct['hour'] < 12:
                new_dct['morning'] = 1
            elif 12 <= new_dct['hour'] < 18:
                new_dct['afternoon'] = 1
            elif 18 <= new_dct['hour'] <= 23:
                new_dct['evening'] = 1

            # remove 'date_dt' bcs it is not supported by JSON
            new_dct.pop('date_dt')

            # media type
            new_dct['share'] = 0
            new_dct['share_url'] = None
            new_dct['photo'] = 0
            new_dct['photo_url'] = None
            new_dct['video'] = 0
            new_dct['video_url'] = None
            new_dct['media_desc'] = []
            new_dct['thumbnail_url'] = []
            if dct['attachments'] is not None:
                for media in dct['attachments']:
                    if media['media_type'] in ['avatar', 'profile_media', 'cover_photo']:
                        raise MediaTypeError

                    if media['media_type'] == 'share':
                        new_dct['share'] = 1
                        new_dct['media_desc'] = media['media_description']
                        new_dct['thumbnail_url'] = media['thumbnail_url']
                        if media['media_url'] is not None:
                            new_dct['share_url'] = media['media_url']
                        else:
                            print('Media URL Error: ', filename, new_dct['url'], 'share_url is missing')
                    elif media['media_type'] in ['photo', 'album', 'new_album']:
                        new_dct['photo'] = 1
                        new_dct['media_desc'] = media['media_description']
                        new_dct['thumbnail_url'] = media['thumbnail_url']
                        if media['media_url'] is not None:
                            new_dct['photo_url'] = media['media_url']
                        else:
                            print('Media URL Error: ', filename, new_dct['url'], 'photo_url is missing')
                    elif media['media_type'] in ['video_inline', 'video_direct_response', 'native_templates', 'video', 'map']:
                        new_dct['video'] = 1
                        new_dct['media_desc'] = media['media_description']
                        new_dct['thumbnail_url'] = media['thumbnail_url']
                        if media['media_url'] is not None:
                            new_dct['video_url'] = media['media_url']
                        else:
                            print('Media URL Error: ', filename, new_dct['url'], 'video_url is missing')

            new_dct['link'] = 0
            new_dct['link_url'] = None
            if 'urls' in dct.keys() and dct['urls'] is not None:
                new_dct['link'] = 1
                new_dct['link_url'] = dct['urls']

            # check url from json and original data
            url1 = new_dct['url']
            url2 = engagement.loc[pd.to_numeric(filename[:-5])]['URL']
            # if urlparse(url1).netloc != urlparse(url2).netloc:
            if url1 != url2:
                raise URLMatchError
            else:
                # engagements
                new_dct.update(engagement.loc[pd.to_numeric(filename[:-5]), engagement.columns != 'URL'])

            out_file = open(out_dir + filename, 'w')
            json.dump(new_dct, out_file, default=np_encoder)
            list_of_dct.append(new_dct)
        except JSONDecodeError:
            unavailable_json.append(filename)
        except MediaTypeError:
            print('Media Type Error: ', filename, new_dct['url'], 'that has type', media['media_type'], 'is removed')
        except URLMatchError:
            print('URL Matching Error: Could not match', filename[:-5], 'in original data')

df = pd.DataFrame(list_of_dct)
df.to_csv('dataset0313.csv', index=False)  #sep='\t'


