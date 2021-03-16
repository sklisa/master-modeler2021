import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data0313 = pd.read_csv('dataset0313.csv')

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
# plt.savefig("posts_by_weekday.png")


# post freq by month -----------------
# plt.figure(figsize=(8, 5))
# sns.histplot(pd.DatetimeIndex(data0313['date']).month, stat='frequency', bins=12, binwidth=1, discrete=True)
# plt.xlabel('Month')
# plt.ylabel('Number of Posts')
# plt.title('Number of Posts by Month')
# plt.savefig("posts_by_month.png")


# post freq by season -----------------
# plt.figure(figsize=(8, 5))
# ax = sns.histplot(data=data0313, x='season', stat='frequency', bins=4, discrete=True)
# ax.set(xlabel='Season', ylabel='Number of Posts', title='Number of Posts by Season')
# plt.xticks(np.arange(4), ['Winter', 'Spring', 'Summer', 'Fall'])
# plt.savefig("posts_by_season.png")


# post freq by time of day -----------------
# plt.figure(figsize=(8, 5))
# ax = sns.histplot(data=data0313, x='time_day', stat='frequency', bins=4, discrete=True)
# ax.set(xlabel='Time of Day', ylabel='Number of Posts', title='Number of Posts by Time of Day')
# plt.xticks(np.arange(4), ['Dawn', 'Morning', 'Afternoon', 'Evening'])
# plt.savefig("posts_by_time_day.png")


# engagement rate hist --------------
# plt.figure(figsize=(8, 5))
# sns.histplot(data=data0313, x='engagement_rate')
# plt.xlabel('Engagement Rate')
# plt.ylabel('Number of Posts')
# plt.title('Engagement Rate Distribution')
# plt.savefig("engagement_rate_hist.png")


# engagement rate hist (removing outlier) --------------
# plt.figure(figsize=(8, 5))
# sns.histplot(data=data0313.loc[data0313['engagement_rate'] <= 2000], x='engagement_rate')
# plt.xlabel('Engagement Rate')
# plt.ylabel('Number of Posts')
# plt.title('Engagement Rate Distribution (Under 2000)')
# plt.savefig("engagement_rate_hist<2000.png")


plt.show()
