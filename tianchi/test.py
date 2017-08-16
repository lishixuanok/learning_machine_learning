import pandas as pd


def data_load(file_path):
    cur = pd.read_csv(file_path, delimiter=",", header=0, nrows=100, dtype=bytes, chunksize=10)
    for line in cur:
        print(line)

if __name__ == '__main__':
    file_path = 'fresh_comp_offline/tianchi_fresh_comp_train_user.csv'
    data_load(file_path)
