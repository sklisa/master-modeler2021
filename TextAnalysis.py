#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter

dir = os.getcwd()
files = glob.glob(dir+'/PrepData0316/'+'*.json')

def to_list(corpus, mode):
    texts = {0: [], 1:[], 2:[]}
    for label, text in zip(df['weighted_engagement_label3'], df[mode]):
        if text is None:
            texts[label].append(' ')
        else:
            texts[label].append(' '.join(text))
    return texts

def generate_ngram(corpus, n, top_n):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(n,n)) 
    corpus_ngram = vectorizer.fit_transform(corpus)
    ngram_values = corpus_ngram.toarray().sum(axis=0)
    features = vectorizer.vocabulary_
    leng = len(corpus)
    counts = {}
    for k, i in features.items():
        counts[k] = ngram_values[i]/leng
    counts = {k: v for k, v in sorted(counts.items(), key=lambda item: -item[1])[:top_n]}
    return list(counts.keys())

def generate_tfidf(corpus, max_features, top_n):
    vectorizer = TfidfVectorizer(ngram_range=(1,5), max_features=max_features) 
    corpus_ngram = vectorizer.fit_transform(corpus)
    ngram_values = corpus_ngram.toarray().sum(axis=0)
    features = vectorizer.vocabulary_
    leng = len(corpus)
    counts = {}
    for k, i in features.items():
        counts[k] = ngram_values[i]/leng
    counts = {k: v for k, v in sorted(counts.items(), key=lambda item: -item[1])[:top_n]}
    return counts

def filter_duplicates(l, index):
    intersection = Counter(l[0]) & Counter(l[1])
    print('\n'.join(list((Counter(l[index]) - intersection).elements())))

def frequent_items(mode, top_n):
    emoji_list = []
    for sublist in df[mode]:
        if sublist is not None:
            for i in sublist:
                if len(i)>0: emoji_list.append(i)
    freq = Counter(emoji_list)
    return {k:v for k, v in sorted(freq.items(), key=lambda item: -item[1])[:top_n]}

dct_list = []
for file in files:
    with open(file, 'r') as f:
        dct = json.load(f)
        dct_list.append(dct)
df = pd.DataFrame(dct_list)

# emojis = frequent_items('emojis', 13)
# words = frequent_items('cleaned_message', 10)
# hashtags = frequent_items('message_tags', 10)
# mentions = frequent_items('mentions', 10)
# names = frequent_items('names', 10)

cleaned_texts = to_list(df, 'cleaned_message')

_1gram_0 = generate_ngram(cleaned_texts[0], 1, 50)
_1gram_1 = generate_ngram(cleaned_texts[1], 1, 50)
l = {0: _1gram_0, 1: _1gram_1}
print('*'*30, '1-gram for class 0')
filter_duplicates(l, 0)
print('*'*30, '1-gram for class 1')
filter_duplicates(l, 1)