import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data0313 = pd.read_csv('dataset0313.csv')

data0313['date'] = pd.to_datetime(data0313['date'], format="%Y-%m-%d")

# post freq by month
post_date = data0313.groupby(pd.Grouper(key='date', freq='M')).size().to_frame(name='freq').reset_index()
ax = sns.lineplot(x='date', y='freq', data=post_date)
ax.set(xlabel='Date by Month', ylabel='Number of Posts', title='Number of Posts by Month')
# plt.plot(post_date['date'], post_date['freq'])
# plt.xlabel('Date by Month')
# plt.ylabel('Number of Posts')
# plt.title('Number of Posts by Month')
plt.show()
# print(data0313.dtypes)
# print(post_date.columns)