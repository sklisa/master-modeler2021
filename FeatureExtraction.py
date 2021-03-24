#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import json
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from TextPreprocessor import TextPreprocessor
import spacy
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

dir = os.getcwd()
files = glob.glob(dir+'/PrepData0316/'+'*.json')
nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
analyzer = SentimentIntensityAnalyzer()

def load_dct():
    dct_list = []
    message_dct = {}
    for file in files:
        new_dct = {}
        with open(file, 'r') as f:
            file_key = int(os.path.basename(file)[:-5])
            dct = json.load(f)
            try:
                # new_dct['key'] = file_key
                # new_dct['message'] = dct['message']
                # new_dct['cleaned_message'] = dct['cleaned_message']
                # new_dct['message_tags'] = dct['message_tags']
                new_dct['Mon'] = int(dct['Mon'])
                new_dct['Tue'] = int(dct['Tue'])
                new_dct['Wed'] = int(dct['Wed'])
                new_dct['Thur'] = int(dct['Thur'])
                new_dct['Fri'] = int(dct['Fri'])
                new_dct['Sat'] = int(dct['Sat'])
                new_dct['winter'] = int(dct['winter'])
                new_dct['spring'] = int(dct['spring'])
                new_dct['summer'] = int(dct['summer'])
                new_dct['morning'] = int(dct['morning'])
                new_dct['afternoon'] = int(dct['afternoon'])
                new_dct['evening'] = int(dct['evening'])
                new_dct['emoji_num'] = int(dct['emoji_num'])
                new_dct['mention_num'] = int(dct['mention_num'])
                new_dct['name_num'] = int(dct['name_num'])
                new_dct['hashtag_num'] = int(dct['hashtag_num'])
                new_dct['share'] = int(dct['share'])
                new_dct['photo'] = int(dct['photo'])
                new_dct['video'] = int(dct['video'])
                new_dct['link'] = int(dct['link'])
                new_dct['face_present'] = int(dct['face_present'])
                new_dct['face_vague'] = int(dct['face_vague'])
                new_dct['recovered'] = int(dct['recovered'])
                new_dct['missing'] = int(dct['missing'])
                new_dct['asterisk'] = int(dct['asterisk'])
                new_dct['federal'] = int(dct['federal'])
                new_dct['special_day'] = int(dct['special_day'])
                new_dct['engagement_rate_label3'] = int(dct['engagement_rate_label3'])
                # new_dct['total_engagement_label'] = int(dct['total_engagement_label'])
                new_dct['total_engagement_label3'] = int(dct['total_engagement_label3'])
                new_dct['weighted_engagement_label3'] = int(dct['weighted_engagement_label3'])
                new_dct['shares_label3'] = int(dct['shares_label3'])

                # new_dct['total_engagement'] = int(dct['total_engagement'])
                # new_dct['engagement_rate'] = int(dct['engagement_rate'])
                # new_dct['weighted_engagement'] = int(dct['weighted_engagement'])
                dct_list.append(new_dct)
                message_dct[file_key] = (dct['message'], dct['cleaned_message'], dct['message_tags'])
            except KeyError as e:
                print(str(e))
                print(file)
    df = pd.DataFrame(dct_list, columns=dct_list[0].keys(), dtype='float64')
    # df.sort_values(by=['index'])
    return df, message_dct

def load_sentiment():
    df = pd.read_csv('SentimentAnalysis.csv', 
                     usecols=['filename', 'negative_adjectives_component', 'social_order_component', 'action_component', 
                                                       'positive_adjectives_component', 'joy_component', 'affect_friends_and_family_component', 
                                                       'fear_and_digust_component', 'politeness_component', 'polarity_nouns_component',
                                                       'polarity_verbs_component', 'positive_nouns_component', 'respect_component', 'trust_verbs_component',
                                                       'well_being_component', 'economy_component', 'certainty_component', 'positive_verbs_component', 'objects_component'])
    df['key'] = df.apply(lambda row: int(row['filename'][:-4]), axis=1)
    df.drop(columns=['filename'], axis=1, inplace=True)
    df.fillna(0, inplace=True)
    df = df.astype('float64')
    return df

def to_list(corpus, ctype):
    texts = {}
    events = {}
    for key, val in corpus.items():
        if ctype == 'cleaned_message':
            i = val[1]
        elif ctype == 'message':
            i = val[0]
        else:
            i = val[2]
        # outfile = open(dir+'/MessageContent/'+str(key)+'.txt', 'w')
        if i is None:
            texts[key] = ' '
        else:
            if ctype == 'cleaned_message':
                texts[key] = ' '.join(i)
            elif ctype == 'message':
                pr = TextPreprocessor(i, nlp)
                pr.remove_urls()
                pr.remove_mentions()
                pr.remove_hashtags()
                pr.remove_emojis()
                pr.remove_special_characters()
                pr.remove_blank_spaces()
                events[key] = len(pr.retrieve_events()) if pr.retrieve_events() is not None else 0
                # print(pr.text)
                texts[key] = pr.text
            else:
                texts[key] = ' '.join(i)
        # outfile.write(texts[key])
        # outfile.close()
    if len(events) > 0:
        df_events = pd.DataFrame(events.items(), columns=['key', 'events_num'])
        return df_events, texts
    return texts

def normalized_BoW(corpus, top_n):
    corpus = list(corpus.values())
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,1), max_features=top_n) 
    corpus_ngram = vectorizer.fit_transform(corpus)
    ngram_values = corpus_ngram.toarray()
    features = vectorizer.vocabulary_
    df = pd.DataFrame(ngram_values, columns=features)
    return df

def bert_sentence_embedding(corpus):
    # reference: https://www.sbert.net/docs/quickstart.html
    corpus = list(corpus.values())
    sentence_embeddings = model.encode(corpus)
    num_features = len(sentence_embeddings[0]) # 768
    feature_names = ['embed_'+str(i) for i in range(num_features)]
    df = pd.DataFrame(sentence_embeddings, columns=feature_names)
    return df

def length_of_post(corpus):
    res = {}
    for idx, text in corpus.items():
        res[idx] = len(text.split())
    df = pd.DataFrame(res.items(), columns=['key', 'post_len'], dtype='float64')
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
    # df.drop(['message', 'cleaned_message', 'message_tags'], axis=1, inplace=True)
    df.drop(['key'], axis=1, inplace=True)
    y = df[mode]
    df.drop(['engagement_rate_label3', 'total_engagement_label3', 'weighted_engagement_label3', 'shares_label3'], axis=1, inplace=True)
    return df, y

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
    # plt.show()
    scaled_X = pd.DataFrame(scale(X),columns=X.columns) 
    pca = PCA(n_components=0.95)
    transformed_X = pca.fit_transform(scaled_X)
    lst = ['PCA_'+str(i) for i in range(len(pca.components_))]
    print(pd.DataFrame(pca.components_,columns=scaled_X.columns,index=lst))
    return transformed_X

if __name__ == '__main__':
    df, message_dct = load_dct()

    df.to_csv('dataset_0320.csv', index=False)
    # cleaned_texts = to_list(message_dct, 'cleaned_message')
    # df_events, raw_texts = to_list(message_dct, 'message')
    # text -> normalized bag-of-words post
    # df_bow = normalized_BoW(cleaned_texts, 500)
    # text -> bert sentence embedding
    # df_bert = bert_sentence_embedding(raw_texts)
    
    # retrieve sentiment score from the text
    # df_sentiment_score = var_sentiment_score(raw_texts)
    
    # load other sentiment analysis
    # df_sentiment_analysis = load_sentiment()
    
    # categorize the hashtags
    # hashtags = to_list(message_dct, 'message_tags')
    # df_bow_hashtags = normalized_BoW(hashtags, 50)
    
    # calculate length of post
    # df_lop = length_of_post(raw_texts)
    # df = df.join(df_sentiment_analysis.set_index('key'), on='key')
    # # df = pd.concat([df, df_bow, df_bow_hashtags], axis=1)
    # df = df.join(df_lop.set_index('key'))
    # df = df.join(df_events.set_index('key'))
    # df = df.fillna(0)
    # df.drop(columns=['key'], axis=1)
    # # df = df.astype('float')
    # # df = pd.concat([df, df_bert, df_sentiment_score, df_bow_hashtags, df_lop], axis=1)
    # X, y = seperate_label(df, 'total_engagement_label3')
    # scaled_X = pd.DataFrame(StandardScaler().fit_transform(X),columns=X.columns) 
    # dataset = pd.concat([pd.DataFrame(scaled_X), y], axis=1)
    # # PCA_X = dim_reduction(X)
    # # PCA_dataset = pd.concat([pd.DataFrame(PCA_X), y], axis=1)
    # train, test = train_test_split(dataset, test_size=0.3, shuffle=True, random_state=123)
    # train.to_csv('train_new.csv', index=False)
    # test.to_csv('test_new.csv', index=False)