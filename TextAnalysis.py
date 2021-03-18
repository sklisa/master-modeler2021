#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

dir = os.getcwd()
files = glob.glob(dir+'/FilteredData/'+'*.json')

def to_list(corpus, ctype):
    texts = []
    for i in list(corpus):
        if i is None:
            texts.append('')
        else:
            if ctype == 'cleaned_message':
                texts.append(' '.join(i))
            elif ctype == 'message':
                pr = TextPreprocessor(i, nlp)
                pr.remove_urls()
                pr.remove_mentions()
                pr.remove_hashtags()
                pr.remove_emojis()
                pr.remove_special_characters()
                # print(pr.text)
                texts.append(pr.text)
            else:
                texts.append(i)
    return texts


def generate_ngram(corpus, top_n):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(3,3)) 
    corpus_ngram = vectorizer.fit_transform(corpus)
    ngram_values = corpus_ngram.toarray().sum(axis=0)
    features = vectorizer.vocabulary_
    leng = len(corpus)
    counts = {}
    for k, i in features.items():
        counts[k] = ngram_values[i]/leng
    counts = {k: v for k, v in sorted(counts.items(), key=lambda item: -item[1])[:top_n]}
    return counts

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

dct_list = []
for file in files:
    with open(file, 'r') as f:
        dct = json.load(f)
        dct_list.append(dct)
df = pd.DataFrame(dct_list)

cleaned_texts = to_list(df['cleaned_message'], 'cleaned_message')
ngram = generate_ngram(cleaned_texts, 100)

