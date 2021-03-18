import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import json
from json.decoder import JSONDecodeError
import glob


dir = os.getcwd()
out_dir = dir+'/PrelimAnalysisChart/'
data_input = pd.read_csv('dataset0316.csv')
data_input['date'] = pd.to_datetime(data_input['date'], format="%Y-%m-%d")


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
corr_var = data_input[['total_engagement', 'engagement_rate', 'reactions',
       'shares', 'comments', 'day_week', 'Mon', 'Tue', 'Wed', 'Thur',
       'Fri', 'Sat', 'event', 'season', 'winter', 'spring', 'summer', 'hour',
       'time_day', 'morning', 'afternoon', 'evening', 'emoji_num',
       'mention_num', 'name_num', 'share', 'photo', 'video', 'link',
       'face_present', 'engagement_rate_label', 'total_engagement_label']]

pear = corr_var.corr(method='pearson')
sns.heatmap(pear, cmap="YlGnBu", xticklabels=True, yticklabels=True)
plt.gcf().subplots_adjust(left=0.3, bottom=0.3)
plt.savefig(out_dir+"corr_mat.png", dpi=300)
abs_corr = pear.abs()
for a in range(len(abs_corr)):
    for b in range(len(abs_corr)):
         if abs_corr.iloc[a, b] > 0.8 and a != b:
                print(corr_var.columns[a], corr_var.columns[b])
plt.show()