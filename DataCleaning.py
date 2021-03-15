#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from json.decoder import JSONDecodeError
import os
import glob
import pandas as pd
from TextPreprocessor import TextPreprocessor
import spacy

dir = os.getcwd()
raw_files = glob.glob(dir+'/RawData/'+'*.json')
out_dir = dir+'/FilteredData/'
list_of_dct = []
unavailable_urls = []
nlp = spacy.load('en_core_web_sm')
dataset = pd.read_excel('Master Modeler Competition 2021.xlsx', usecols='D', names = ['url']) 
all_urls = dataset['url']

'''
Meaning of each field in dict:
    url: str, url of the post
    created_time: str, timeframe that the post is created
    message: str, the original content of the post
    cleaned_message: list of str, the content of the post after cleaning (in a near bag-of-words form)
        we will use a seperate TextPreprocessor.py for preprocessing: 
            retrieve emojis, mentions, urls, use of names in the message
            remove emojis, mentions, urls, hashtags, and non-english characters -> lower case -> expand contractions
            -> remove stopwords -> remove numbers -> remove blank spaces -> tokenize and stem words -> remove stopwords again
    urls: list of urls in the message
    emojis: list of emojis in the message
    mentions: list of str, list of mentions in the message (@ removed)
    names: list of str, list of person names in the message
    message_tags: list of str, list of tags that appeared in the post if any (# removed)
    attachments: list of dict, list of dictionary that represents the info of each media (normally there is only one media, but in case there are multiple, I will use a list)
        media_type: str, type of the media
        media_url: str, url of the media (photo/sharing url depending on the media_type) attached to the post
            list of url if media_type is album
        media_description: str, title of media if media_type is not photo, else None
        thumbnail_url: str, url of the thumbnail if media_type is not photo, else None
    --- Note: 
        1) we said in the presentation that we plan to calculate the frequency of links, mentions and emojis,
           i will keep them as a list of contents for now and see if we want to retrieve more meaningful info or frequency only later
        2) if the information that corresponds to a field does not exist in the url, the corresponding field will be None
'''

for file in raw_files:
    with open(file, 'r') as f:
        new_dct = {'url': None, 'created_time': None, 'message': None, 'cleaned_message': None, 'emojis': None, 'mentions': None, 'names': None, 
                   'message_tags': None, 'attachments': None}
        filename = os.path.basename(file)
        actual_url = all_urls[int(filename[:-5])]
        # print(filename)
        try:
            dct = json.load(f)
            new_dct['url'] = actual_url
            new_dct['created_time'] = dct['created_time']
            if 'message' in dct and dct['message'] is not None:
                new_dct['message'] = dct['message']
                pr = TextPreprocessor(dct['message'], nlp)
                new_dct['emojis'] = pr.retrieve_emojis()
                pr.remove_emojis()
                new_dct['mentions'] = pr.retrieve_mentions()
                pr.remove_mentions()
                new_dct['urls'] = pr.retrieve_urls()
                pr.remove_urls()
                pr.remove_hashtags()
                pr.remove_special_characters()
                new_dct['names'] = pr.retrieve_names()
                pr.fully_preprocess()
                new_dct['cleaned_message'] = pr.text.split() if len(pr.text)>0 else None
                # print('cleaned text')
            if 'message_tags' in dct:
                message_tags = dct['message_tags']
                new_dct['message_tags'] = [i['name'].replace('#','') for i in message_tags] if len(message_tags)>0 else None
            if 'attachments' in dct:
                attachments = dct['attachments']['data']
                new_dct['attachments'] = [{'media_type': None, 'media_url': None, 'media_description': None, 'thumbnail_url': None} for _ in range(len(attachments))]
                for i, attachment in enumerate(attachments):
                    new_dct['attachments'][i]['media_type'] = attachment['type']
                    if 'url' in attachment:
                        new_dct['attachments'][i]['media_url'] = attachment['url']
                    image = None
                    if 'media' in attachment and 'image' in attachment['media']:
                        image = attachment['media']['image']
                    if image and attachment['type'] == 'photo':
                        new_dct['attachments'][i]['media_url'] = image['src']
                    elif attachment['type'] == 'album':
                        subattachments = attachment['subattachments']['data']
                        new_dct['attachments'][i]['media_url'] = [s['media']['image']['src'] for s in subattachments]
                        # print(new_dct['attachments'][i]['media_url'])
                    else:
                        if 'url' in attachment:
                            new_dct['attachments'][i]['media_url'] = attachment['url']
                        if image and 'src' in image.keys():
                            new_dct['attachments'][i]['thumbnail_url'] = image['src']
                        if 'title' in attachment:
                            new_dct['attachments'][i]['media_description'] = attachment['title']
                        # print(new_dct['attachments'][i]['thumbnail_url'])
            out_file = open(out_dir+filename, 'w')
            json.dump(new_dct, out_file)
            list_of_dct.append(new_dct)
        except JSONDecodeError:
            unavailable_urls.append(filename)
df = pd.DataFrame(list_of_dct)
df.to_csv('dataset.csv', sep='\t', index=False)
