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
data_input = pd.read_csv('dataset0315.csv')
out_data_dir = dir+'/PrepData0316/'
in_files = glob.glob(dir+'/PrepData0315/'+'*.json')
unavailable_json = []


# Examine engagement quantile
def quantile(variable):
    # plt.boxplot(data_input[variable])
    outlier = data_input[variable].loc[np.abs(stats.zscore(data_input[variable])) >= 3]
    print(variable, len(outlier), 'outlier removed: \n', outlier)
    outlier_rm = data_input[np.abs(stats.zscore(data_input[variable])) < 3]
    quantile = outlier_rm[variable].quantile([.1, .25, .5, .75, .9])
    print('Quantile after outlier removed: \n', quantile)
    return quantile


def label(quantile1, quantile2):
    list_of_dct = []
    for file in in_files:
        with open(file, 'r') as f:
            filename = os.path.basename(file)
            try:
                dct = json.load(f)
                new_dct = dct

                # label outcome var
                if new_dct['engagement_rate'] <= quantile1.iloc[1]:
                    new_dct['engagement_rate_label'] = 0
                elif quantile1.iloc[1] < new_dct['engagement_rate'] <= quantile1.iloc[3]:
                    new_dct['engagement_rate_label'] = 1
                else:
                    new_dct['engagement_rate_label'] = 2

                if new_dct['total_engagement'] <= quantile2.iloc[1]:
                    new_dct['total_engagement_label'] = 0
                elif quantile2.iloc[1] < new_dct['total_engagement'] <= quantile2.iloc[3]:
                    new_dct['total_engagement_label'] = 1
                else:
                    new_dct['total_engagement_label'] = 2

                if new_dct['engagement_rate'] <= quantile1.iloc[0]:
                    new_dct['engagement_rate_label2'] = 0
                elif quantile1.iloc[0] < new_dct['engagement_rate'] <= quantile1.iloc[4]:
                    new_dct['engagement_rate_label2'] = 1
                else:
                    new_dct['engagement_rate_label2'] = 2

                if new_dct['total_engagement'] <= quantile2.iloc[0]:
                    new_dct['total_engagement_label2'] = 0
                elif quantile2.iloc[0] < new_dct['total_engagement'] <= quantile2.iloc[4]:
                    new_dct['total_engagement_label2'] = 1
                else:
                    new_dct['total_engagement_label2'] = 2

                if new_dct['engagement_rate'] <= quantile1.iloc[2]:
                    new_dct['engagement_rate_label3'] = 0
                else:
                    new_dct['engagement_rate_label3'] = 1

                if new_dct['total_engagement'] <= quantile2.iloc[2]:
                    new_dct['total_engagement_label3'] = 0
                else:
                    new_dct['total_engagement_label3'] = 1

                # weighted total engagement
                new_dct['weighted_engagement'] = new_dct['shares'] + .75 * new_dct['comments'] + 0.5 * new_dct['reactions']


                # flag
                new_dct['recovered'] = 0
                new_dct['missing'] = 0
                new_dct['asterisk'] = 0
                new_dct['federal'] = 0

                if new_dct['message'] is not None:
                    if 'missing' in new_dct['message'].lower():
                        print('missing')
                        new_dct['missing'] = 1

                    if 'recovered' in new_dct['message'].lower():
                        print('recovered')
                        new_dct['recovered'] = 1

                    if 'federal' in new_dct['message'].lower():
                        print('federal')
                        new_dct['federal'] = 1

                    if '*' in new_dct['message']:
                        print('asterisk')
                        new_dct['asterisk'] = 1

                # rename event to special_day
                new_dct['special_day'] = new_dct['event']
                new_dct.pop('event')

                # face_present == 2
                new_dct['face_vague'] = 0
                if new_dct['face_present'] == 2:
                    new_dct['face_vague'] = 1
                    new_dct['face_present'] = 1

                out_file = open(out_data_dir + filename, 'w')
                json.dump(new_dct, out_file)
                list_of_dct.append(new_dct)

            except JSONDecodeError:
                unavailable_json.append(filename)

    print(len(list_of_dct))
    return list_of_dct


def main():
    quantile1 = quantile('engagement_rate')
    quantile2 = quantile('total_engagement')
    list_of_dct = label(quantile1, quantile2)

    df = pd.DataFrame(list_of_dct)

    # add


    df.to_csv('dataset0316.csv', index=False)  # sep='\t'


if __name__ == "__main__":
    main()