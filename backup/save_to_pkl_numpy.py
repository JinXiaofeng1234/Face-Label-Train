from pandas import read_csv
import pickle
from torch import stack as to_tensor
from torch import tensor
from torch import float32 as torch_float32
from tqdm import tqdm


def convert_to_2d_tensor(input_list, num_classes=6):
    output_list = [[0] * num_classes for _ in range(len(input_list))]
    for i, value in enumerate(input_list):
        if 0 <= value < num_classes:
            output_list[i][value - 1] = 1
    return tensor(output_list, dtype=torch_float32)


def turn_to_pkl(csv_filename):
    data = read_csv(csv_filename)

    data.drop(['image_name', 'character_name', 'face_pos_1x', 'face_pos_1y', 'face_pos_2x', 'face_pos_2y'],
              axis=1, inplace=True)

    data_list = data.values.tolist()
    saved_ls = list()
    for ls in tqdm(data_list):
        label_tensor = convert_to_2d_tensor(ls)
        saved_ls.append(label_tensor)
    labels_tensors = to_tensor(saved_ls, dim=0)
    pkl_filename = csv_filename[:(len(csv_filename) - 4)] + ".pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(labels_tensors, file)


if __name__ == '__main__':
    filename = input("请输入csv文件(不加后缀名):\n") + ".csv"
    turn_to_pkl(filename)
