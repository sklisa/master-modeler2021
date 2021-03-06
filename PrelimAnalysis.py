import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import json
from json.decoder import JSONDecodeError
import glob
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import scipy.stats as ss
from statsmodels.stats.proportion import proportions_ztest


dir = os.getcwd()
out_dir = dir + '/PrelimAnalysisChart/'
out_dir2 = dir + '/StatsChart/'
data_input = pd.read_csv('dataset0316.csv')
data_input2 = pd.read_csv('dataset_0320.csv')

pd.set_option('display.max_columns', None)
df = pd.read_csv('SentimentAnalysis.csv')
df['key'] = df.apply(lambda row: int(row['filename'][:-4]), axis=1)
df.drop(columns=['filename'], axis=1, inplace=True)
df_joined = df.join(data_input.set_index('key'), on='key')
df_joined2 = df.join(data_input2.set_index('key'), on='key')


# data_input['date'] = pd.to_datetime(data_input['date'], format="%Y-%m-%d")

# post freq by month (ts) -----------------
# post_date = data_input.groupby(pd.Grouper(key='date', freq='M')).size().to_frame(name='freq').reset_index()
# plt.figure(figsize=(8, 5))
# ax = sns.lineplot(x='date', y='freq', data=post_date)
# ax.set(xlabel='Date by Month', ylabel='Number of Posts', title='Number of Posts by Month')
# plt.savefig("posts_by_month_ts.png")

# plt.plot(post_date['date'], post_date['freq'])
# plt.xlabel('Date by Month')
# plt.ylabel('Number of Posts')
# plt.title('Number of Posts by Month')


# engagement rate by month -----------------
# engage_date = data_input[['date', 'engagement_rate']].groupby(pd.Grouper(key='date', freq='M')).mean().reset_index()
# plt.figure(figsize=(8, 5))
# ax = sns.lineplot(x='date', y='engagement_rate', data=engage_date)
# ax.set(xlabel='Date by Month', ylabel='Average Engagement Rate', title='Average Engagement Rate by Month')
# plt.savefig("engage_rate_by_month.png")


# engagement by month -----------------
# engage_date = data_input[['date', 'total_engagement']].groupby(pd.Grouper(key='date', freq='M')).mean().reset_index()
# plt.figure(figsize=(8, 5))
# ax = sns.lineplot(x='date', y='total_engagement', data=engage_date)
# ax.set(xlabel='Date by Month', ylabel='Average Total Engagement', title='Average Total Engagement by Month')
# plt.savefig("total_engage_by_month.png")


# post freq by day of week ---------------
# plt.figure(figsize=(8, 5))
# sns.histplot(data=data_input, x='day_week', bins=7, binwidth=1, discrete=True)
# plt.xlabel('Date of Week')
# plt.xticks(np.arange(7), ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'])
# plt.ylabel('Number of Posts')
# plt.title('Number of Posts by Day of Week')
# plt.savefig(out_dir+"posts_by_weekday.png")


# post freq by month -----------------
# plt.figure(figsize=(8, 5))
# sns.histplot(pd.DatetimeIndex(data_input['date']).month, stat='frequency', bins=12, binwidth=1, discrete=True)
# plt.xlabel('Month')
# plt.ylabel('Number of Posts')
# plt.title('Number of Posts by Month')
# plt.savefig(out_dir+"posts_by_month.png")


# post freq by season -----------------
# plt.figure(figsize=(8, 5))
# ax = sns.histplot(data=data_input, x='season', stat='frequency', bins=4, discrete=True)
# ax.set(xlabel='Season', ylabel='Number of Posts', title='Number of Posts by Season')
# plt.xticks(np.arange(4), ['Winter', 'Spring', 'Summer', 'Fall'])
# plt.savefig(out_dir+"posts_by_season.png")


# post freq by time of day -----------------
# plt.figure(figsize=(8, 5))
# ax = sns.histplot(data=data_input, x='time_day', stat='frequency', bins=4, discrete=True)
# ax.set(xlabel='Time of Day', ylabel='Number of Posts', title='Number of Posts by Time of Day')
# plt.xticks(np.arange(4), ['Dawn', 'Morning', 'Afternoon', 'Evening'])
# plt.savefig(out_dir+"posts_by_time_day.png")


# engagement rate hist --------------
# plt.figure(figsize=(8, 5))
# sns.histplot(data=data_input, x='engagement_rate')
# plt.xlabel('Engagement Rate')
# plt.ylabel('Number of Posts')
# plt.title('Engagement Rate Distribution')
# plt.savefig(out_dir+"engagement_rate_hist.png")


# engagement rate hist (removing outlier) --------------
# plt.figure(figsize=(8, 5))
# sns.histplot(data=data_input.loc[data_input['engagement_rate'] <= 2000], x='engagement_rate')
# plt.xlabel('Engagement Rate')
# plt.ylabel('Number of Posts')
# plt.title('Engagement Rate Distribution (Under 2000)')
# plt.savefig(out_dir+"engagement_rate_hist<2000.png")


# total engagement hist --------------
# plt.figure(figsize=(8, 5))
# sns.histplot(data=data_input, x='total_engagement')
# plt.xlabel('Total Engagement')
# plt.ylabel('Number of Posts')
# plt.title('Total Engagement Distribution')
# plt.savefig(out_dir+"total_engagement_hist.png")


# total engagement hist --------------
# plt.figure(figsize=(8, 5))
# sns.histplot(data=data_input.loc[data_input['total_engagement'] <= 600], x='total_engagement')
# plt.xlabel('Total Engagement')
# plt.ylabel('Number of Posts')
# plt.title('Total Engagement Distribution (Under 600)')
# plt.savefig(out_dir+"total_engagement_hist<600.png")


# Correlation
# print(data_input.columns)
# corr_var = data_input[['total_engagement', 'engagement_rate', 'reactions',
#        'shares', 'comments', 'day_week', 'Mon', 'Tue', 'Wed', 'Thur',
#        'Fri', 'Sat', 'event', 'season', 'winter', 'spring', 'summer', 'hour',
#        'time_day', 'morning', 'afternoon', 'evening', 'emoji_num',
#        'mention_num', 'name_num', 'share', 'photo', 'video', 'link',
#        'face_present', 'engagement_rate_label', 'total_engagement_label']]
#
# pear = corr_var.corr(method='pearson')
# sns.heatmap(pear, cmap="YlGnBu", xticklabels=True, yticklabels=True)
# plt.gcf().subplots_adjust(left=0.3, bottom=0.3)
# plt.savefig(out_dir+"corr_mat.png", dpi=300)
# abs_corr = pear.abs()
# for a in range(len(abs_corr)):
#     for b in range(len(abs_corr)):
#          if abs_corr.iloc[a, b] > 0.8 and a != b:
#                 print(corr_var.columns[a], corr_var.columns[b])



# Find outlier
# print('Outlier by engagement_rate')
# # outlier1 = data_input.loc[np.abs(stats.zscore(data_input['engagement_rate'])) >= 3]
# quartile_1, quartile_3 = np.percentile(data_input['engagement_rate'], [25, 75])
# iqr = quartile_3 - quartile_1
# lower_bound = quartile_1 - (iqr * 1.5)
# upper_bound = quartile_3 + (iqr * 1.5)
# outlier1 = data_input.loc[(data_input['engagement_rate'] > upper_bound) | (data_input['engagement_rate'] < lower_bound)]
# print(outlier1)
# print('Outlier by total_engagement')
# # outlier2 = data_input.loc[np.abs(stats.zscore(data_input['total_engagement'])) >= 3]
# quartile_1, quartile_3 = np.percentile(data_input['total_engagement'], [25, 75])
# iqr = quartile_3 - quartile_1
# lower_bound = quartile_1 - (iqr * 1.5)
# upper_bound = quartile_3 + (iqr * 1.5)
# outlier2 = data_input.loc[(data_input['total_engagement'] > upper_bound) | (data_input['total_engagement'] < lower_bound)]
# print(outlier2)
# # merge1 = pd.merge(outlier1, outlier2['url'], how='outer', on='url')
# # print('Union of outliers', len(merge1))
# # print(merge1)
# merge2 = pd.merge(outlier1, outlier2['url'], how='inner', on='url')
# print('Intersection of outliers', len(merge2))
# print(merge2)
# merge2 = merge2.sort_values(by=['total_engagement', 'engagement_rate'], ascending=False)
# # merge1.to_csv('union_outliers2.csv')
# # merge2.to_csv('inters_outliers2.csv')


# Outlier stats
# print('face_present', len(merge2[merge2['face_present']==1])/len(merge2))
# print('total face_present', len(data_input[data_input['face_present']==1])/len(data_input))
#
# print('missing', len(merge2[merge2['missing']==1])/len(merge2))
# print('total missing', len(data_input[data_input['missing']==1])/len(data_input))
#
# print('missing+recovered', len(merge2[(merge2['missing']==1) | (merge2['recovered']==1)])/len(merge2))
# print('total missing+recovered', len(data_input[(data_input['missing']==1) | (data_input['recovered']==1)])/len(data_input))




# Regression

# rm = ['reactions', 'shares', 'comments',
#           'total_engagement', 'engagement_rate', 'weighted_engagement',
#           'total_engagement_label', 'engagement_rate_label',
#           'total_engagement_label2', 'engagement_rate_label2',
#           'total_engagement_label3', 'engagement_rate_label3',
#           'weighted_engagement_label3', 'shares_label3',
#       'url', 'created_time', 'message', 'cleaned_message', 'emojis',
#       'mentions', 'names', 'message_tags', 'attachments', 'urls',
#       'original_date', 'date', 'day_week', 'season', 'hour', 'time_day',
#       'share_url', 'photo_url', 'video_url', 'media_desc', 'thumbnail_url']
# cols = [col for col in df_joined.columns if col not in rm]
# features = df_joined[cols]
# print(features.columns)
# output = 'shares'
#
# scaler = MinMaxScaler()
# feat_scaled = scaler.fit_transform(X=features, y=df_joined[output])
# # reg = LinearRegression().fit(X=feat_scaled, y=data_input[output])
# # print(output, reg.score(X=feat_scaled, y=data_input[output]))
# # print(output, reg.coef_)
#
# lasso = Lasso(alpha=0.5)
# lasso.fit(feat_scaled, df_joined[output])
# # lasso = LassoCV(cv=2, n_alphas=100, random_state=0).fit(feat_scaled, df_joined[output])
# # alphas=[0.01, 0.05, 0.1, 0.5]
# print(lasso.score(X=feat_scaled, y=df_joined[output]))
# print('coef', lasso.coef_)

# Statistical test

# x="weighted_engagement_label3", y="positive_adjectives_component"
output_var = 'weighted_engagement_label3'
print(df_joined2[output_var].unique())
cat1 = df_joined2[df_joined2[output_var] == 0]
cat2 = df_joined2[df_joined2[output_var] == 1]
nobs1 = len(cat1)
nobs2 = len(cat2)

print(nobs1, nobs2)

for col in df_joined2.columns:
    if len(df_joined2[col].unique()) > 2:
          t_stat, p = ss.ttest_ind(cat1[col], cat2[col])
          if p < 0.05:
                 print('Numerical:', col, 't is ', t_stat, 'p value is', p, '******')
    elif len(df_joined2[col].unique()) == 2:
          count1 = len(cat1[cat1[col] == 1])
          count2 = len(cat2[cat2[col] == 1])
          z_stat, p = proportions_ztest([count1, count2], [nobs1, nobs2])
          if p < 0.05:
                 print('Binary:', col, 'z is ', z_stat, ', p value is', p, '******')
          # else:
          #        print('Binary:', col, ', p value is', p)



# Statistical test chart

# Binary
# sns.catplot(x="weighted_engagement_label3", hue="face_present", kind='count', data=df_joined2)
# plt.title('Frequency of Face Present Among High and Low Engagement Groups')
# plt.xlabel('Engagement Label')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.savefig(out_dir2+"face_present_freq.png")
#
# # Numerical
# ax = sns.catplot(y="positive_adjectives_component", x="weighted_engagement_label3", kind='bar', data=df_joined2)
# plt.title('Average Positive Adjectives Component Among Engagement Groups')
# plt.xlabel('Engagement Label')
# plt.ylabel('Average Positive Adjectives Component')
# plt.tight_layout()
# plt.savefig(out_dir2+"positive_adj_comp_avg.png")
#
# ax = sns.catplot(y="negative_adjectives_component", x="weighted_engagement_label3", kind='bar', data=df_joined2)
# plt.title('Average Negative Adjectives Component Among Engagement Groups')
# plt.xlabel('Engagement Label')
# plt.ylabel('Average Negative Adjectives Component')
# plt.tight_layout()
# plt.savefig(out_dir2+"negative_adj_comp_avg.png")



# Case study
# print(len(data_input[data_input['face_present']==1])/len(data_input))
# print(len(data_input[data_input['share']==1])/len(data_input))
# print(len(data_input[data_input['missing']==1])/len(data_input))
# print(len(data_input[(data_input['missing']==1) | (data_input['recovered']==1)])/len(data_input))


plt.show()