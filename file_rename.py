from PIL import Image
import os
from tqdm import tqdm
import shutil

current_folder = 'source_image'


def img_file_rename():
    file_ls = os.listdir(current_folder)
    num = len(os.listdir('debug_img'))
    for file in tqdm(file_ls):
        if file[-3:] != 'ini':
            os.rename(os.path.join(current_folder, file), os.path.join(current_folder, f'data ({num}){file[-4:]}'))
            num = num + 1


def move_img_file():
    # 源文件夹路径
    src_folder = 'source_image/'

    # 目标文件夹路径
    dst_folder = 'debug_img/'

    # 获取源文件夹中的所有文件
    files = os.listdir(src_folder)

    # 逐个移动文件
    for file in tqdm(files):
        src_file = os.path.join(src_folder, file)
        dst_file = os.path.join(dst_folder, file)
        shutil.move(src_file, dst_file)


def to_jpg():
    file_ls = os.listdir(current_folder)
    not_jpg_img_ls = [img_name for img_name in file_ls if not img_name.endswith('.jpg')]

    if not_jpg_img_ls:
        for filename in tqdm(not_jpg_img_ls):
            im = Image.open(f'source_image/{filename}')
            # 如果模式不是 'RGB',则转换
            if im.mode != 'RGB':
                im = im.convert('RGB')

            # 保存为 JPG 格式
            im.save(f'source_image/{filename[:-4]}.jpg', 'JPEG')
            # 删除原文件
            os.remove(f'source_image/{filename}')
    else:
        print('没有图片需要处理')


while True:
    user_choice = input('请输入你的行动 -0 规范化命名文件 -1 移动文件至debug文件夹 -2 统一图片格式\n')
    if user_choice == '0':
        img_file_rename()
    elif user_choice == '1':
        move_img_file()
    elif user_choice == '2':
        to_jpg()
    elif user_choice == 'q':
        break
    else:
        pass
