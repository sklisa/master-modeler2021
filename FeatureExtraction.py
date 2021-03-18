#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from TextPreprocessor import TextPreprocessor
import spacy
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import numpy as np

dir = os.getcwd()
files = glob.glob(dir+'/PrepData0316/'+'*.json')
nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
analyzer = SentimentIntensityAnalyzer()

def load_dct():
    dct_list = []
    for file in files:
        new_dct = {}
        with open(file, 'r') as f:
            dct = json.load(f)
            try:
                new_dct['message'] = dct['message']
                new_dct['cleaned_message'] = dct['cleaned_message']
                new_dct['message_tags'] = dct['message_tags']
                new_dct['Mon'] = dct['Mon']
                new_dct['Tue'] = dct['Tue']
                new_dct['Wed'] = dct['Wed']
                new_dct['Thur'] = dct['Thur']
                new_dct['Fri'] = dct['Fri']
                new_dct['Sat'] = dct['Sat']
                new_dct['special_day'] = dct['special_day']
                new_dct['winter'] = dct['winter']
                new_dct['summer'] = dct['summer']
                new_dct['morning'] = dct['morning']
                new_dct['afternoon'] = dct['afternoon']
                new_dct['evening'] = dct['evening']
                new_dct['emoji_num'] = dct['emoji_num']
                new_dct['mention_num'] = dct['mention_num']
                new_dct['name_num'] = dct['name_num']
                new_dct['share'] = dct['share']
                new_dct['photo'] = dct['photo']
                new_dct['video'] = dct['video']
                new_dct['link'] = dct['link']
                new_dct['face_present'] = dct['face_present']
                new_dct['engagement_rate_label'] = dct['engagement_rate_label']
                new_dct['total_engagement_label'] = dct['total_engagement_label']
                dct_list.append(new_dct)
            except KeyError as e:
                print(str(e))
                print(file)
    df = pd.DataFrame(dct_list, columns=dct_list[0].keys())
    return df

def to_list(corpus, ctype):
    texts = []
    events = []
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
                events.append(len(pr.retrieve_events()) if pr.retrieve_events() is not None else 0)
                # print(pr.text)
                texts.append(pr.text)
            else:
                texts.append(i)
    if len(events)>0:
        df_events = pd.DataFrame(events, columns=['events_num'])
        return df_events, texts
    return texts

def normalized_BoW(corpus):
    tv = TfidfVectorizer(
        binary=False, norm='l1', use_idf=False, smooth_idf=False,
        lowercase=True, stop_words='english', ngram_range=(1,1))
    df = pd.DataFrame(tv.fit_transform(corpus).toarray(), columns=tv.get_feature_names())
    return df

def bert_sentence_embedding(corpus):
    # reference: https://www.sbert.net/docs/quickstart.html
    sentence_embeddings = model.encode(corpus)
    num_features = len(sentence_embeddings[0]) # 768
    feature_names = ['embed_'+str(i) for i in range(num_features)]
    df = pd.DataFrame(sentence_embeddings, columns=feature_names)
    return df

def length_of_post(corpus):
    res = []
    for i in corpus:
        res.append(len(i.split()))
    df = pd.DataFrame(res, columns=['post_len'])
    return df

def var_sentiment_score(corpus):
    # reference: https://github.com/cjhutto/vaderSentiment
    # The compound score is computed by summing the valence scores of each word in the lexicon, adjusted according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive).
    # normalized, weighted composite score
    sentiment_scores = []
    for sentence in corpus:
        vs = analyzer.polarity_scores(sentence)
        sentiment_scores.append(vs['compound'])
    df = pd.DataFrame(sentiment_scores, columns=['sentiment_score'])
    return df

def seperate_label(df, mode):
    df.drop(['message', 'cleaned_message', 'message_tags'], axis=1, inplace=True)
    if mode == 'engagement_rate':
        y = df['engagement_rate_label']
        df.drop(['engagement_rate_label', 'total_engagement_label'], axis=1, inplace=True)
    elif mode == 'total_engagement':
        y = df['total_engagement_label']
        df.drop(['engagement_rate_label', 'total_engagement_label'], axis=1, inplace=True)
    return df, y

def scale(X):
    scaler = MinMaxScaler()
    X_rescaled = scaler.fit_transform(X)
    return X_rescaled

def dim_reduction(X):
    # pca = PCA().fit(X)
    # plt.rcParams["figure.figsize"] = (12,6)
    
    # fig, ax = plt.subplots()
    # xi = np.arange(1, 2094, step=1)
    # y = np.cumsum(pca.explained_variance_ratio_)
    
    # plt.ylim(0.0,1.1)
    # plt.plot(xi, y, marker='o', linestyle='--', color='b')
    
    # plt.xlabel('Number of Components')
    # plt.xticks(np.arange(0, 2093, step=100)) #change from 0-based array index to 1-based human-readable label
    # plt.ylabel('Cumulative variance (%)')
    # plt.title('The number of components needed to explain variance')
    
    # plt.axhline(y=0.95, color='r', linestyle='-')
    # plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)
    
    # ax.grid(axis='x')
    plt.show()
    pca = PCA(n_components=0.95)
    pca.fit(X)
    transformed_X = pca.transform(X)
    return transformed_X

if __name__ == '__main__':
    df = load_dct()
    cleaned_texts = to_list(df['cleaned_message'], 'cleaned_message')
    df_events, raw_texts = to_list(df['message'], 'message')
    # text -> normalized bag-of-words post
    # df_bow = normalized_BoW(cleaned_texts)
    # text -> bert sentence embedding
    df_bert = bert_sentence_embedding(raw_texts)
    
    # retrieve sentiment score from the text
    df_sentiment_score = var_sentiment_score(raw_texts)
    
    # categorize the hashtags
    hashtags = to_list(df['message_tags'], 'cleaned_message')
    df_bow_hashtags = normalized_BoW(hashtags)
    
    # calculate length of post
    df_lop = length_of_post(raw_texts)
    df = pd.concat([df, df_bert, df_sentiment_score, df_bow_hashtags, df_lop], axis=1)
    X, y = seperate_label(df, 'engagement_rate')
    rescaled_X = scale(X)
    # dataset = pd.concat([pd.DataFrame(rescaled_X), y], axis=1)
    PCA_X = dim_reduction(rescaled_X)
    PCA_dataset = pd.concat([pd.DataFrame(PCA_X), y], axis=1)
    train, test = train_test_split(PCA_dataset, test_size=0.3, shuffle=True, random_state=123)
    train.to_csv('train_bert_PCA.csv')
    test.to_csv('test_bert_PCA.csv')