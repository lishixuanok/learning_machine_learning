import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime


def _parser(date_string):
    return datetime.strptime(date_string, '%Y-%m-%d %H')


def split_data_to_train_and_test():
    """ split data in to test data set and train data set."""
    total_df = pd.read_csv('tianchi_fresh_comp_train_user.csv',
                           delimiter=',',
                           header=0,
                           parse_dates=['time'], date_parser=_parser,
                           )
    train_df = total_df.loc[total_df.time < datetime(year=2014, month=12, day=18)]
    test_df = total_df.loc[total_df.time == datetime(year=2014, month=12, day=18)]
    return train_df, test_df


def pre_process_data(df):
    # buy
    buy_col_name = 'if_buy'
    df.loc[df.behavior_type == 4, buy_col_name] = 1
    df.loc[df.behavior_type != 4, buy_col_name] = 0

    # add to buy
    buy_col_name = 'if_add_buy'
    df.loc[df.behavior_type == 3, buy_col_name] = 1
    df.loc[df.behavior_type != 3, buy_col_name] = 0
    df[buy_col_name] = df[buy_col_name].astype(np.int)

    # collect
    collect_col_name = 'if_collect'
    df.loc[df.behavior_type == 2, collect_col_name] = 1
    df.loc[df.behavior_type != 2, collect_col_name] = 0
    df[collect_col_name] = df[collect_col_name].astype(np.int)

    # watch
    watch_col_name = 'if_watch'
    df.loc[df.behavior_type == 1, watch_col_name] = 1
    df.loc[df.behavior_type != 1, watch_col_name] = 0

    # 升维
    watch_df = df.loc[df[watch_col_name] == 1]
    buy_df = df.loc[df[buy_col_name] == 1]

    # item
    watch_item_df = watch_df \
        .groupby(['item_id']) \
        .size().to_frame('item_total_watch')

    buy_item_df = buy_df \
        .groupby(['item_id']) \
        .size().to_frame('item_total_buy')
    item_result = pd.concat([watch_item_df, buy_item_df],
                            axis=1).fillna(value=0)
    item_result['item_total_count'] = item_result['item_total_buy'] + item_result['item_total_watch']
    item_result['item_buy_rate'] = item_result['item_total_buy'] / item_result['item_total_count']

    # user
    watch_item_df = watch_df \
        .groupby(['user_id']) \
        .size().to_frame('user_total_watch')

    buy_item_df = buy_df \
        .groupby(['user_id']) \
        .size().to_frame('user_total_buy')
    user_result = pd.concat([watch_item_df, buy_item_df], axis=1
                            ).fillna(value=0)
    user_result['user_total_count'] = user_result['user_total_buy'] + user_result['user_total_watch']
    user_result['user_buy_rate'] = user_result['user_total_buy'] / user_result['user_total_count']

    # merge
    basic_df = df[['user_id', 'item_id', 'if_collect', 'if_add_buy', 'if_buy']] \
        .drop_duplicates()
    item_result.reset_index(level=[0], inplace=True)
    df1 = pd.merge(basic_df, item_result, how='inner', on='item_id')
    user_result.reset_index(level=[0], inplace=True)
    df2 = pd.merge(df1, user_result, how='inner', on='user_id')

    # normalize
    items = ['user_id', 'item_id']
    for item in items:
        df2[item] = df2[item].astype(str)
    cols_to_norm = ['item_total_watch', 'item_total_buy',
                    'item_total_count',
                    'user_total_watch', 'user_total_buy',
                    'user_total_count']
    df2[cols_to_norm] = df2[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return df2


def split_data_label(df):
    data = df[['if_collect', 'if_add_buy',
               'item_total_watch', 'item_total_buy',
               'item_total_count', 'item_buy_rate',
               'user_total_watch', 'user_total_buy',
               'user_total_count', 'user_buy_rate'
               ]]
    df['if_not_buy'] = df.apply(lambda row: 0 if row.if_buy else 1, axis=1)
    df['if_buy'] = df['if_buy'].astype(np.int)
    label = df[['if_not_buy', 'if_buy']]
    return data, label


def main():
    train_df, test_df = split_data_to_train_and_test()
    process_train_df = pre_process_data(train_df)
    process_test_df = pre_process_data(test_df)
    train_data, train_label = split_data_label(process_train_df)
    test_data, test_label = split_data_label(process_test_df)

    # Parameters
    learning_rate = 0.01
    training_epochs = 25
    batch_size = 100
    display_step = 1

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, train_data.shape[1]])
    y = tf.placeholder(tf.float32, [None, train_label.shape[1]])

    # Set model weights
    W = tf.Variable(tf.zeros([train_data.shape[1], train_label.shape[1]]), name="weight")
    b = tf.Variable(tf.zeros([train_label.shape[1]]), name='bias')

    # Construct model
    pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver(tf.global_variables())

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(train_data.shape[0] / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                # Run optimization op (backprop) and cost op (to get loss value)
                _, _loss = sess.run([optimizer, cost],
                                    feed_dict={x: train_data.iloc[i * batch_size: (i + 1) * batch_size],
                                               y: train_label.iloc[i * batch_size: (i + 1) * batch_size]})
                # Compute average loss
                avg_cost += _loss / total_batch
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")
        save_path = saver.save(sess, "./model/log.model")
        print("Model saved in file: %s" % save_path)
        print('Weight is :%s' % W.eval())

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: test_data, y: test_label}))

        result = tf.argmax(pred, 1).eval({x: test_data, y: test_label})

        # to csv
        predict_result_df = pd.DataFrame({'predict_buy': result})
        res = process_test_df.merge(predict_result_df, left_index=True, right_index=True)
        temp = res.loc[res.predict_buy == 1]
        temp.to_csv('tianchi_mobile_recommendation_predict.csv', columns=['user_id', 'item_id'], index=False)


def dump_result():
    read_df = pd.read_csv('tianchi_fresh_comp_train_user.csv',
                          delimiter=',',
                          header=0,
                          parse_dates=['time'],
                          date_parser=_parser)
    y_true = read_df.loc[(read_df.time == datetime(year=2014, month=12, day=18)) &
                         (read_df.behavior_type == 4)]
    y_true.to_csv('tianchi_mobile_recommendation_true.csv', columns=['user_id', 'item_id'], index=False)


def verify():
    y_pred = pd.read_csv('tianchi_mobile_recommendation_predict.csv')
    y_true = pd.read_csv('tianchi_mobile_recommendation_true.csv')

    s1 = pd.merge(y_pred, y_true, how='inner', on=['user_id', 'item_id'])
    print(s1)


if __name__ == '__main__':
    main()
    # dump_result()
    # verify()
