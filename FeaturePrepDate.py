import json
from json.decoder import JSONDecodeError
import os
import glob
import pandas as pd


dir = os.getcwd()
filter_files = glob.glob(dir+'/FilteredData/'+'*.json')
out_dir = dir+'/PrepData0313/'
list_of_dct = []
unavailable_json = []

# Events
# January Human Trafficking Awareness Month
# SuperBowl: Feb. 5, 2017; Feb. 4, 2018; Feb. 3, 2019; Feb. 2, 2020; Feb. 7, 2021
# Marti Gras: February 28, 2017; February 13, 2018; March 5, 2019; February 25, 2020; February 16, 2021
# March Madness: second half of March + first week of April; 2020 cancelled
# July 30 World Day Against Trafficking in Person

for file in filter_files:
    with open(file, 'r') as f:
        # new_dct = {'url': None,
        #            'created_time': None,
        #            'date': None,
        #            'time_day': None,   # time of day: morning(6-12), afternoon(12-18), evening(18-0), dawn(0-6)
        #            'day_week': None,   # day of week
        #            'season': None,
        #            #'event': None,  # 1 if in Jan, Feb, March, and July 30
        #            'message': None,
        #            'cleaned_message': None,
        #            'emojis': None,
        #            'mentions': None,
        #            'names': None,
        #            'message_tags': None,
        #            'attachments': None,
        #            'urls': None
        #            }
        filename = os.path.basename(file)
        try:
            dct = json.load(f)
            new_dct = dct
            new_dct['date'] = dct['created_time'][0:10]
            new_dct['date'] = pd.to_datetime(new_dct['date'])

            # day of week
            new_dct['day_week'] = new_dct['date'].dayofweek
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

            # season
            new_dct['winter'] = 0
            new_dct['spring'] = 0
            new_dct['summer'] = 0
            if new_dct['date'].month == 12 or new_dct['date'].month <= 2:
                new_dct['winter'] = 1
            elif 3 <= new_dct['date'].month <= 5:
                new_dct['spring'] = 1
            elif 6 <= new_dct['date'].month <= 8:
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

            # media type
            new_dct['share'] = 0
            new_dct['share_url'] = None
            new_dct['photo'] = 0
            new_dct['photo_url'] = None
            new_dct['video'] = 0
            new_dct['video_url'] = None

            new_dct['link'] = 0
            new_dct['link_url'] = None
            if dct['urls'] is not None:
                new_dct['link'] = 1
                new_dct['link_url'] = dct['urls']


            # out_file = open(out_dir + filename, 'w')
            # json.dump(new_dct, out_file)
            list_of_dct.append(new_dct)
        except JSONDecodeError:
            unavailable_json.append(filename)

df = pd.DataFrame(list_of_dct)
print(list_of_dct)
# print(type(df['day_week']))
print(df.columns)
# df.to_csv('dataset0313.csv', sep='\t', index=False)


