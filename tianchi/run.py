import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from collections import defaultdict


def pre_process(file_path):
    # data load
    dataset = pd.read_csv(file_path, delimiter=",",
                          header=0, nrows=2000,
                          dtype={'behavior_type': np.int8}, parse_dates=[5])
    dataset.columns = ['user_id', 'item_id', 'behavior_type', 'user_geohash', 'item_category', 'time']

    dataset['date'] = dataset['time'].values.astype('datetime64[D]')
    dataset['add_collect'] = np.where(dataset['behavior_type'].between(2, 3, inclusive=True), True, False)
    dataset['buy'] = dataset['behavior_type'] == 4

    print(dataset.loc[
              dataset.behavior_type >= 2, ['user_id', 'item_id', 'date', 'add_collect', 'buy', 'behavior_type']]
          .sort_values('date'))


if __name__ == '__main__':
    file_path = 'fresh_comp_offline/tianchi_fresh_comp_train_user.csv'
    calculate(file_path)
