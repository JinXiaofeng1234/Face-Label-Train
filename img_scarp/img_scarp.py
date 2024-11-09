import sys
from selenium import webdriver
import subprocess
import html5lib
from bs4 import BeautifulSoup
from os import listdir
from requests import get
import zipfile
import shutil
import re
from multiprocessing.dummy import Pool

""" 查找数据集的最终图片名字,方便让爬取的图片加入数据集"""


def custom_sort_key(s):
    math = re.search(f"data \((\d+)\)\.jpg", s)
    if math:
        return int(math.group(1))
    else:
        return s


file_ls = listdir('../img_scarp')
img_ls = sorted(listdir('../debug_img')[:-1], key=custom_sort_key)  # 最后一个是ini文件,故取到-1

headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,"
              "image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
}


def web_drive_download():
    # 定义命令
    command = r'wmic datafile where name="C:\\Program Files (x86)\\' \
              r'Microsoft\\Edge\\Application\\msedge.exe" get Version'
    # 运行命令并获取输出
    output = subprocess.check_output(command, shell=True, text=True)
    version = output.split()[1]
    url = f'https://msedgedriver.azureedge.net/{version}/edgedriver_win64.zip'  # 指定下载地址
    file_data = get(url=url).content  # 获得压缩包
    save_path = r'C:\Users\Cara  al sol\Downloads\edgedriver_win64.zip'  # 指定保存路径和文件名
    with open(save_path, 'wb') as fp:
        fp.write(file_data)  # 写入数据
        print("驱动压缩包下载成功!")

    zipfile_path = save_path  # 指定压缩包文件地址
    unzip_path = r'C:/Users\Cara  al sol/Downloads/'  # 指定释放解压后文件的放置地址
    unzip_file = zipfile.ZipFile(zipfile_path)  # 解压压缩包
    print("开始解压....")
    unzip_file.extractall(unzip_path)  # 释放解压文件到指定位置
    print('解压结束')
    unzip_file.close()  # 关闭文件,释放存储
    shutil.copy(unzip_path + 'msedgedriver.exe', r'C:\Program Files\EDGE driver')  # 将被解压的文件复制并转移到浏览器驱动文件夹


def img_scarp(key_word, count):
    web_driver = webdriver.Edge()  # 启动edge内核
    url = fr"https://cn.bing.com/images/search?q={key_word}&first={count}"  # 指定爬取网址
    print("开始爬取.....")
    try:
        web_driver.get(url=url)  # 向目标网址发送get请求
    except Exception as e:
        print(f"报错:{e}")
        web_drive_download()
        print("浏览器驱动更新成功!将自动结束程序")
        sys.exit()

    print("爬取成功!")
    page_text = web_driver.execute_script("return document.documentElement.outerHTML")  # 抓取js动态网页
    soup = BeautifulSoup(page_text, 'html5lib')  # 利用html5lib,结构化网页源码
    formatted_html = soup.prettify()
    # time.sleep(10)
    # web_driver.refresh()
    # time.sleep(5)
    web_driver.quit()  # 退出驱动
    print("页面保存中........")
    with open("html.txt", "a", encoding="utf-8") as fp:  # 保存网页
        fp.write(formatted_html)
    print("页面保存成功!")
    print('-' * 20)


def save_image(url_num_tup):
    try:
        img_data = get(url=url_num_tup[0], headers=headers).content  # get请求获得图片内容
        img_name = f"data ({url_num_tup[2]}).{url_num_tup[1]}"
        save_path = f'../source_image/{img_name}'
        with open(save_path, "wb") as fp:  # 保存为原图片的格式,防止下载后无法查看, 二进制流
            fp.write(img_data)  # 保存图片
            print(f"{img_name}", "下载成功！")
    except Exception as e:
        print(e)


def scarp_loop():
    key_word = input("请输入要搜索的关键字:")
    start_page = int(input("请输入起始页:"))
    end_page = int(input("请输入终止页:"))
    count_ls = []
    for i in range(start_page, end_page + 1):
        if i == 1:
            count_ls.append(1)
        else:
            count_ls.append(35 * (i - 1))  # 计算first取值的集合
    for count in count_ls:
        img_scarp(key_word, count)


def data_filter():
    if not img_ls:
        num = 0
    else:
        existing_num = re.findall(r'\d+', img_ls[-1])  # 从最后的图片名称中截取序号
        num = int(existing_num[0])
    html_text = open("html.txt", encoding='utf-8').read()  # 读取网页源码文件
    pattern = r'murl":"(https?://[^"]+\.(jpg|png|webp))'  # 利用正则表达式查找图片地址
    img_url = re.findall(pattern, html_text)

    filtered_img_url = list(set(img_url))  # 去重操作
    save_num_ls = [num + b for b in range(1, len(filtered_img_url) + 1)]  # 定义图片赋名列表

    res_ls = [(item[0][0], item[0][1], item[1]) for item in zip(filtered_img_url, save_num_ls)]  # 每个元组的内容为地址,格式,名字
    pool = Pool(7)
    pool.map(save_image, res_ls)


while True:
    user_input = input("想清理html文件吗?:")
    if user_input == 'y':
        with open('html.txt', 'w') as file:
            file.truncate(0)  # 清空html文件
    elif user_input == 'q':
        break
    else:
        pass

    if 'html.txt' in file_ls:
        messagebox = input("是否需要爬取新的Html(y/n)?:")
        if messagebox == 'y':
            scarp_loop()
        elif messagebox == 'q':
            break
        else:
            data_filter()
    else:
        scarp_loop()
