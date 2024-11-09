import sys
from img_read import img_process
from os import listdir
from re import findall
import pickle
from tqdm import tqdm


def custom_sort_key(filename):
    # 使用正则表达式提取所有的数字
    numbers = findall(r'\d+', filename)

    # 将提取的数字转换为整数
    numbers = [int(num) for num in numbers]

    # 根据提取到的数字数量，返回不同长度的元组
    # 不足的部分用 -1 补充，确保文件名可以正确排序
    if len(numbers) == 1:
        return numbers[0], -1, -1  # 只有括号中的数字
    elif len(numbers) == 2:
        return numbers[0], numbers[1], -1  # 括号中的数字和一个额外的数字
    elif len(numbers) >= 3:
        return numbers[0], numbers[1], numbers[2]  # 括号中的数字和两个额外的数字


def group_read(file_path):
    global count
    global unrecognized_img_ls
    file_ls = listdir("../deep learning model")
    file_names = listdir(file_path)
    file_names.remove('_system~.ini')
    file_names = sorted(file_names, key=custom_sort_key)
    # print(file_names)

    # 准备统计未装量化图片的列表
    not_into_tensors_img_ls = []

    # 检查face_tensors.pt是否存在于当前目录下

    if "face_tensors.pkl" in file_ls:
        # 导入图片张量文件
        with open('face_tensors.pkl', 'rb') as file:
            face_tensors = pickle.load(file)
        # 检查face_tensors长度
        face_tensors_len = len(face_tensors)
        if face_tensors_len < len(file_names):
            # 统计未张量化的图片
            not_into_tensors_img_ls = file_names[face_tensors_len:]
        elif face_tensors_len == len(file_names):
            print("图片已经全部转化完成,自动结束程序")
            sys.exit()
        else:
            pass
    else:
        pass

    # 加载人脸检测器

    res = list()
    # 如果统计未张量化列表不为空列表，则在接下来的图片张量化过程中，张量化那些未被张量化的图片

    if not_into_tensors_img_ls:
        file_names = not_into_tensors_img_ls

    for i in tqdm(file_names):
        try:
            img_path = fr'{file_path}\{i}'
            img_tensor = img_process(img_path)
        except Exception as e:
            print(f"图片{i}报错!")
            continue
        if img_tensor is not None:
            res.append(img_tensor)
        else:
            count += 1
            unrecognized_img_ls.append(i)

    if "face_tensors.pkl" in file_ls:
        with open('face_tensors.pkl', 'rb') as file:
            face_tensors_backup = pickle.load(file)
            new_res = face_tensors_backup + res
    else:
        new_res = res
    if unrecognized_img_ls:
        print(f"共有{count}张人脸照片未被处理")
        print(f'以下图片未被识别:{" ".join(unrecognized_img_ls)}')

    with open('face_tensors.pkl', 'wb') as file:
        print('保存中...')
        pickle.dump(new_res, file)
        print('保存成功!')


if __name__ == '__main__':
    unrecognized_img_ls = []
    count = 0
    img_file_path = input('请输入图片文件夹地址')
    group_read(img_file_path)
