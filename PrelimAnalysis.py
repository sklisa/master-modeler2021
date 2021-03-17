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
data_input = pd.read_csv('dataset0315.csv')
data_input['date'] = pd.to_datetime(data_input['date'], format="%Y-%m-%d")

out_data_dir = dir+'/PrepData0316/'
in_files = glob.glob(dir+'/PrepData0315/'+'*.json')
list_of_dct = []
unavailable_json = []


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


plt.show()


# Examine engagement quantile
def quantile(variable):
    # plt.boxplot(data_input[variable])
    outlier = data_input[variable].loc[np.abs(stats.zscore(data_input[variable])) >= 3]
    print(variable, len(outlier), 'outlier removed: \n', outlier)
    outlier_rm = data_input[np.abs(stats.zscore(data_input[variable])) < 3]
    quantile = outlier_rm[variable].quantile([.25, .75, .9])
    print('Quantile after outlier removed: \n', quantile)
    return quantile


def label(quantile1, quantile2):
    for file in in_files:
        with open(file, 'r') as f:
            filename = os.path.basename(file)
            try:
                dct = json.load(f)
                new_dct = dct

                if new_dct['engagement_rate'] <= quantile1.iloc[0]:
                    new_dct['engagement_rate_label'] = 0
                elif quantile1.iloc[0] < new_dct['engagement_rate'] <= quantile1.iloc[1]:
                    new_dct['engagement_rate_label'] = 1
                else:
                    new_dct['engagement_rate_label'] = 2

                if new_dct['total_engagement'] <= quantile2.iloc[0]:
                    new_dct['total_engagement_label'] = 0
                elif quantile2.iloc[0] < new_dct['total_engagement'] <= quantile2.iloc[1]:
                    new_dct['total_engagement_label'] = 1
                else:
                    new_dct['total_engagement_label'] = 2

                out_file = open(out_data_dir + filename, 'w')
                json.dump(new_dct, out_file)
                list_of_dct.append(new_dct)

            except JSONDecodeError:
                unavailable_json.append(filename)

    df = pd.DataFrame(list_of_dct)
    print(len(list_of_dct))
    df.to_csv('dataset0316.csv', index=False)  # sep='\t'


def main():
    quantile1 = quantile('engagement_rate')
    quantile2 = quantile('total_engagement')
    label(quantile1, quantile2)

if __name__ == "__main__":
    main()
