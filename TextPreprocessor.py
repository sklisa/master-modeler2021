#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import string

import nltk
from nltk.corpus import stopwords
from nltk import re
import spacy
from nltk.stem import PorterStemmer

def get_emojis_pattern():
    try:
        # UCS-4
        emojis_pattern = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    except re.error:
        # UCS-2
        emojis_pattern = re.compile(
            u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
    return emojis_pattern

def get_negations_pattern():
    negations_ = {"isnt": "is not", "cant": "can not", "couldnt": "could not", "hasnt": "has not",
                  "hadnt": "had not", "wont": "will not",
                  "wouldnt": "would not", "arent": "are not",
                  "havent": "have not", "doesnt": "does not", "didnt": "did not",
                  "dont": "do not", "shouldnt": "should not", "wasnt": "was not", "werent": "were not",
                  "mightnt": "might not", "mustnt": "must not",
                  'ive': 'i have', 'im': 'i am', 'gimme': 'give me', 'gonna': 'going to', 'gotta': 'got to', 'hows': 'how is',
                  'ill': 'i will', 'lets': 'let us', 'hes': 'he is', 'shes': 'she is', 'thats': 'that is', 'wanna': 'want to'}
    return re.compile(r'\b(' + '|'.join(negations_.keys()) + r')\b')

def get_mentions_pattern():
    return re.compile(r'@\w*')

def get_hashtags_pattern():
    return re.compile(r'#\w*')

def get_urls_pattern():
    return re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

class TextPreprocessor:
    
    def __init__(self, text, nlp):
        self.text = text
        self.nlp = nlp
        self.stemmer = PorterStemmer()
    
    def retrieve_urls(self):
        urls = re.findall(pattern=get_urls_pattern(), string=self.text)
        urls = [i for i in urls if len(i)>0]
        return urls if len(urls)>0 else None
    
    def retrieve_emojis(self):
        emojis_list = []
        emojis = re.findall(pattern=get_emojis_pattern(), string=self.text)
        for tup in emojis:
            for w in tup:
                if w != '' and w != ',' and len(w)>=1:
                    emojis_list.append(w)
        return emojis_list if len(emojis_list)>0 else None
    
    def retrieve_names(self):
        # reference: https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
        celebrities = []
        doc = self.nlp(self.text)
        for w in doc.ents:
            if w.label_ == 'PERSON' and w.text != 'Repost  \n':
                celebrities.append(w.text)
        return celebrities if len(celebrities)>0 else None
    
    def retrieve_events(self):
        events = []
        doc = self.nlp(self.text)
        for w in doc.ents:
            if w.label_ == 'EVENT':
                events.append(w.text)
        return events if len(events)>0 else None
    
    def retrieve_mentions(self):
        mentions = re.findall(pattern=get_mentions_pattern(), string=self.text)
        mentions = [i.replace('@', '') for i in mentions if len(i)>0]
        return mentions if len(mentions)>0 else None
    
    def fully_preprocess(self):
        return self.lowercase().handle_negations().remove_stopwords().remove_numbers().stem_words().remove_stopwords()
    
    def remove_urls(self):
        self.text = re.sub(pattern=get_urls_pattern(), repl=' ', string=self.text)
        return self
    
    def remove_mentions(self):
        self.text = re.sub(pattern=get_mentions_pattern(), repl=' ', string=self.text)
        return self
    
    def remove_hashtags(self):
        self.text = re.sub(pattern=get_hashtags_pattern(), repl=' ', string=self.text)
        return self
    
    def remove_emojis(self):
        self.text = re.sub(pattern=get_emojis_pattern(), repl=' ', string=self.text)
        return self
    
    def remove_special_characters(self):
        self.text = re.sub('[^a-zA-Z_-]', repl=' ', string=self.text)
        return self
    
    def remove_stopwords(self):
        text = nltk.word_tokenize(self.text)
        stop_words = set(stopwords.words('english'))

        new_sentence = []
        for w in text:
            if w not in stop_words:
                new_sentence.append(w)
        self.text = ' '.join(new_sentence)
        return self
    
    def remove_numbers(self):
        text_list = self.text.split(' ')
        for text in text_list:
            if text.isnumeric():
                text_list.remove(text)

        self.text = ' '.join(text_list)
        return self
    
    def remove_blank_spaces(self):
        self.text = re.sub(r'\s{2,}|\t', repl=' ', string=self.text)
        return self
    
    def lowercase(self):
        self.text = self.text.lower()
        return self
    
    def handle_negations(self):  
        self.text = re.sub(pattern=get_negations_pattern(), repl='', string=self.text)
        return self
    
    def stem_words(self):
        text_list = self.text.split(' ')
        new_list = []
        for text in text_list:
            text = text.encode("ascii", "ignore")
            decode_text = text.decode()
            new_list.append(self.stemmer.stem(decode_text))
        self.text =' '.join(new_list)
        return self

# def preprocess(conv):
#     p = TextPreprocessor(conv, spacy.load('en_core_web_sm'))
#     emojis = p.retrieve_emojis()
#     p.fully_preprocess()
#     conv = p.text
#     return emojis, conv

# ans = preprocess('🤔 🙈 me así, bla es se 😌 ds 💕👭👙')