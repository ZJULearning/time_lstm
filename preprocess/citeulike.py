from __future__ import print_function
import pandas as pd
import pickle
import os

BASE_DIR = 'data'
DATA_SOURCE = 'citeulike'

# awk -F "|" '{print $1"|"$2"|"$3}' citeulike-origin | uniq > citeulike-origin-filtered
data_path = os.path.join(BASE_DIR, DATA_SOURCE,'citeulike-origin-filtered')
user_item_record = os.path.join(BASE_DIR, DATA_SOURCE, 'user-item.lst')
user_item_delta_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'user-item-delta-time.lst')
user_item_accumulate_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'user-item-accumulate-time.lst')
index2item_path = os.path.join(BASE_DIR, DATA_SOURCE, 'index2item')
item2index_path = os.path.join(BASE_DIR, DATA_SOURCE, 'item2index')


def generate_data():
    out1 = open(user_item_record, 'w')
    out2 = open(user_item_delta_time_record, 'w')
    out3 = open(user_item_accumulate_time_record, 'w')


    data = pd.read_csv(data_path, sep='|',
                       error_bad_lines=False,
                       header=None,
                       names=['item_id', 'user_id', 'timestamp'])
    if os.path.exists(index2item_path) and os.path.exists(item2index_path):
        index2item = pickle.load(open(index2item_path, 'rb'))
        item2index = pickle.load(open(item2index_path, 'rb'))
        print('Total music %d' % len(index2item))
    else:
        print('Build index2item')
        sorted_series = data.groupby(['item_id']).size().sort_values(ascending=False)
        index2item = sorted_series.keys().tolist()
        print('Most common item is "%s":%d' % (index2item[0],sorted_series[index2item[0]]))
        print('build item2index')
        item2index = dict((v, i) for i, v in enumerate(index2item))
        pickle.dump(index2item, open(index2item_path, 'wb'))
        pickle.dump(item2index, open(item2index_path, 'wb'))

    print('start loop')

    count = 0
    user_group = data.groupby(['user_id'])
    total = len(user_group)
    # short sequence comes first
    for userid, length in user_group.size().sort_values().iteritems():
        if count % 10 == 0:
            print("=====count %d/%d======" % (count, total))
        count += 1
        print('%s %d' % (userid, length))
        # oldest data comes first
        user_data = user_group.get_group(userid).sort_values(by='timestamp')
        # user_data = user_data[user['tranname'].notnull()]
        music_seq = user_data['item_id']
        time_seq = user_data['timestamp']
        # filter the null data. 
        music_seq = music_seq[music_seq.notnull()]
        time_seq = time_seq[time_seq.notnull()]
        # calculate the difference between adjacent items. -1 means using t[i] = t[i] - t[i+1]
        delta_time = pd.to_datetime(time_seq).diff(-1).astype('timedelta64[s]') * -1
        # map music to index
        item_seq = music_seq.apply(lambda x: item2index[x] if pd.notnull(x) else -1).tolist()
        delta_time = delta_time.tolist()
        delta_time[-1] = 0
        time_accumulate = [0]
        for delta in delta_time[:-1]:
            next_time = time_accumulate[-1] + delta
            time_accumulate.append(next_time)

        out1.write(userid + ',')
        out1.write(' '.join(str(x) for x in item_seq) + '\n')
        out2.write(userid + ',')
        out2.write(' '.join(str(x) for x in delta_time) + '\n')
        out3.write(userid + ',')
        out3.write(' '.join(str(x) for x in time_accumulate) + '\n')

    out1.close()
    out2.close()
    out3.close()

if __name__ == '__main__':
    generate_data()
