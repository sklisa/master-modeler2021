import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import json
from json.decoder import JSONDecodeError
import glob


data_input = pd.read_csv('dataset0315.csv')
out_data_dir = dir+'/PrepData0316/'
in_files = glob.glob(dir+'/PrepData0315/'+'*.json')
list_of_dct = []
unavailable_json = []


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