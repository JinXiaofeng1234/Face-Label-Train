import pickle
from pandas import read_csv
from csv import writer as csv_writer

with open("source_face_labels.pkl", "rb") as fir_f:
    labels_data = pickle.load(fir_f)

labels = [str(i.tolist()) for i in labels_data]

tag_count = dict()

for tag in labels:
    if tag in tag_count:
        tag_count[tag] += 1
    else:
        tag_count[tag] = 1

g_dic = {}
# 打印统计结果
max_label_len = max(tag_count.values())

for tag, count in tag_count.items():
    aug_num = round(max_label_len / count)
    g_dic[tag] = aug_num

data = read_csv('source_face_labels.csv')
data_list = data.values.tolist()

ls = list()
for index, label in enumerate(labels):
    if g_dic[label]:
        ls += [data_list[index]] * g_dic[label]

with open('face_labels.csv', "w", newline='', encoding='utf-8') as csvfile:
    for i in ls:
        csvwriter = csv_writer(csvfile)
        csvwriter.writerow(i)



