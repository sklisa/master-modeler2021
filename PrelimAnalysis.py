import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

dir = os.getcwd()
out_dir = dir+'/PrelimAnalysisChart/'
data0313 = pd.read_csv('dataset0315.csv')
data0313['date'] = pd.to_datetime(data0313['date'], format="%Y-%m-%d")

# post freq by month (ts) -----------------
# post_date = data0313.groupby(pd.Grouper(key='date', freq='M')).size().to_frame(name='freq').reset_index()
# plt.figure(figsize=(8, 5))
# ax = sns.lineplot(x='date', y='freq', data=post_date)
# ax.set(xlabel='Date by Month', ylabel='Number of Posts', title='Number of Posts by Month')
# plt.savefig("posts_by_month_ts.png")

# plt.plot(post_date['date'], post_date['freq'])
# plt.xlabel('Date by Month')
# plt.ylabel('Number of Posts')
# plt.title('Number of Posts by Month')


# engagement rate by month -----------------
# engage_date = data0313[['date', 'engagement_rate']].groupby(pd.Grouper(key='date', freq='M')).mean().reset_index()
# plt.figure(figsize=(8, 5))
# ax = sns.lineplot(x='date', y='engagement_rate', data=engage_date)
# ax.set(xlabel='Date by Month', ylabel='Average Engagement Rate', title='Average Engagement Rate by Month')
# plt.savefig("engage_rate_by_month.png")


# engagement by month -----------------
# engage_date = data0313[['date', 'total_engagement']].groupby(pd.Grouper(key='date', freq='M')).mean().reset_index()
# plt.figure(figsize=(8, 5))
# ax = sns.lineplot(x='date', y='total_engagement', data=engage_date)
# ax.set(xlabel='Date by Month', ylabel='Average Total Engagement', title='Average Total Engagement by Month')
# plt.savefig("total_engage_by_month.png")


# post freq by day of week ---------------
# plt.figure(figsize=(8, 5))
# sns.histplot(data=data0313, x='day_week', bins=7, binwidth=1, discrete=True)
# plt.xlabel('Date of Week')
# plt.xticks(np.arange(7), ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'])
# plt.ylabel('Number of Posts')
# plt.title('Number of Posts by Day of Week')
# plt.savefig(out_dir+"posts_by_weekday.png")


# post freq by month -----------------
# plt.figure(figsize=(8, 5))
# sns.histplot(pd.DatetimeIndex(data0313['date']).month, stat='frequency', bins=12, binwidth=1, discrete=True)
# plt.xlabel('Month')
# plt.ylabel('Number of Posts')
# plt.title('Number of Posts by Month')
# plt.savefig(out_dir+"posts_by_month.png")


# post freq by season -----------------
# plt.figure(figsize=(8, 5))
# ax = sns.histplot(data=data0313, x='season', stat='frequency', bins=4, discrete=True)
# ax.set(xlabel='Season', ylabel='Number of Posts', title='Number of Posts by Season')
# plt.xticks(np.arange(4), ['Winter', 'Spring', 'Summer', 'Fall'])
# plt.savefig(out_dir+"posts_by_season.png")


# post freq by time of day -----------------
# plt.figure(figsize=(8, 5))
# ax = sns.histplot(data=data0313, x='time_day', stat='frequency', bins=4, discrete=True)
# ax.set(xlabel='Time of Day', ylabel='Number of Posts', title='Number of Posts by Time of Day')
# plt.xticks(np.arange(4), ['Dawn', 'Morning', 'Afternoon', 'Evening'])
# plt.savefig(out_dir+"posts_by_time_day.png")


# engagement rate hist --------------
# plt.figure(figsize=(8, 5))
# sns.histplot(data=data0313, x='engagement_rate')
# plt.xlabel('Engagement Rate')
# plt.ylabel('Number of Posts')
# plt.title('Engagement Rate Distribution')
# plt.savefig(out_dir+"engagement_rate_hist.png")


# engagement rate hist (removing outlier) --------------
# plt.figure(figsize=(8, 5))
# sns.histplot(data=data0313.loc[data0313['engagement_rate'] <= 2000], x='engagement_rate')
# plt.xlabel('Engagement Rate')
# plt.ylabel('Number of Posts')
# plt.title('Engagement Rate Distribution (Under 2000)')
# plt.savefig(out_dir+"engagement_rate_hist<2000.png")


# total engagement hist --------------
# plt.figure(figsize=(8, 5))
# sns.histplot(data=data0313, x='total_engagement')
# plt.xlabel('Total Engagement')
# plt.ylabel('Number of Posts')
# plt.title('Total Engagement Distribution')
# plt.savefig(out_dir+"total_engagement_hist.png")


# total engagement hist --------------
# plt.figure(figsize=(8, 5))
# sns.histplot(data=data0313.loc[data0313['total_engagement'] <= 600], x='total_engagement')
# plt.xlabel('Total Engagement')
# plt.ylabel('Number of Posts')
# plt.title('Total Engagement Distribution (Under 600)')
# plt.savefig(out_dir+"total_engagement_hist<600.png")


# Examine engagement quantile
variable = 'total_engagement'
# plt.boxplot(data0313[variable])
outlier = data0313[variable].loc[np.abs(stats.zscore(data0313[variable])) >= 3]
print(variable, len(outlier), 'outlier removed: \n', outlier)
outlier_rm = data0313[np.abs(stats.zscore(data0313[variable])) < 3]
quantile = outlier_rm[variable].quantile([.1, .25, .5, .75, .9])
print('Quantile after outlier removed: \n', quantile)


plt.show()
