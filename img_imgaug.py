import pickle
import sys
# import time
from re import search
from tqdm import tqdm
import cv2
import imgaug.augmenters as iaa
import numpy as np
from os import listdir

# import hashlib

with open("backup/source_face_labels.pkl", "rb") as fir_f:
    labels_data = pickle.load(fir_f)


# data = read_csv('face_labels.csv')
#
# data_list = data.values.tolist()

def custom_sort_key(s):
    math = search(f"face\((\d+)\)_(\d+)\.jpg", s)
    if math:
        return int(math.group(1))
    else:
        return s


def img_imgaug(path, augment=True, seed=None):
    """
    使用 imgaug 库对输入的图片数组进行增强，并将增强后的图片叠加在一起。
    """
    image = cv2.imread(path)
    image = np.expand_dims(image, axis=0)
    if seed is not None:
        np.random.seed(seed)

    if augment:
        # 定义 imgaug 增强序列
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # 水平翻转, 概率为 0.5
            iaa.Flipud(0.5),  # 垂直翻转, 概率为 0.5
            iaa.Rotate((-45, 45)),  # 随机旋转, 角度在 -45 到 45 度之间
            iaa.Affine(scale=(0.8, 1.2)),  # 随机缩放, 比例因子在 0.8 到 1.2 之间
            iaa.GaussianBlur(sigma=(0, 3.0)),  # 高斯模糊, 标准差在 0 到 3.0 之间
            iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace="RGB"),  # 随机灰度化
            iaa.AddToHueAndSaturation(value_hue=(-50, 50), value_saturation=(-20, 20))
        ])
        augmented_images = seq(images=image)
    else:
        augmented_images = image

    return augmented_images[0].astype(np.uint8)


file_names = listdir('backup/source_modified_img')
file_names.remove('_system~.ini')
file_names = sorted(file_names, key=custom_sort_key)
# print(file_names)

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

if not (len(file_names) == len(labels_data)):
    print('长度不匹配')
    sys.exit()
for index, label in tqdm(enumerate(labels)):
    if label in g_dic:
        if g_dic[label] != 1:
            img_file_name = file_names[index]
            face_count = 0
            for _ in range(g_dic[label] - 1):
                new_img = img_imgaug(f'modified_img/{img_file_name}')
                cv2.imwrite(f'modified_img/{img_file_name[:-4]}_{face_count}.jpg', new_img)
                face_count += 1
                # with open('face_labels.csv', "a", newline='', encoding='utf-8') as csvfile:
                #     csvwriter = csv_writer(csvfile)
                # csvwriter.writerow(data_list[index])
