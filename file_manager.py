from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFormLayout
from PyQt5.QtWidgets import QFileDialog
import sys
from torch import stack
import csv
from numpy import array as to_array


class FilePathOutPut(QtWidgets.QWidget):
    def __init__(self, window_title, file_filter):
        super(FilePathOutPut, self).__init__()
        # 读取Excel文件路径按钮
        self.chose_file_button = QtWidgets.QPushButton()  # 实例化按钮图像方法
        self.chose_file_button.setObjectName("GetFilePathButton")
        self.chose_file_button.setText("请选择文件以读取路径")  # 设定按钮显示文字信息
        # 工具界面日志
        self.log_TextEdit = QtWidgets.QTextEdit()  # 类似于文本输入框
        # 业务相关
        self.file_path = None
        self.window_title = window_title  # 定义窗口名字和文件类型过滤器
        self.file_filter = file_filter

    def main_window(self):
        self.setWindowTitle(self.window_title)
        form_layout = QFormLayout()
        form_layout.addRow(self.chose_file_button)
        self.chose_file_button.setCheckable(True)
        self.chose_file_button.clicked.connect(lambda: self.click_find_file_path(self.chose_file_button))
        form_layout.addRow("日志信息：", self.log_TextEdit)
        self.setLayout(form_layout)

    def click_find_file_path(self, button):
        # 设置文件扩展名过滤，同一个类型的不同格式如xlsx和xls 用空格隔开
        read_file_path, filetype = QFileDialog.getOpenFileName(self, self.window_title, "./data",
                                                               self.file_filter)

        if button.isChecked():
            self.file_path = read_file_path
            self.log_TextEdit.append("需要读取的文件路径为:" + read_file_path)
            self.log_TextEdit.append("文件格式为:" + filetype)
        button.toggle()


def get_file_path(window_title, file_filter):
    app = QtWidgets.QApplication(sys.argv)
    main = FilePathOutPut(window_title, file_filter)
    main.main_window()
    main.show()
    app.exec_()
    return main.file_path


def data_choose(source_female_features, female_picture_tensor):
    with open('face_labels.csv', 'r', encoding="utf-8-sig") as f:  # 防止开头出现 /u eff
        reader = csv.reader(f)
        label_names = next(reader)  # 接收循环中第一行的值

    label_dict = {item[0]: item[1] for item in enumerate(label_names)}  # 取出标签,按索引值和标签名放入一个字典

    for key in label_dict.keys():
        print(f"标签索引值:{key}, 标签名:{label_dict[key]}")  # 打印这个标签字典

    # 0 青年 1 中年 2 老年
    index = list(map(int, input("请输入筛选条件").split()))  # split是将输入内容按照空格分隔开,再用map函数进行整形化
    condition = list(f"item[1][{index[i]}] and" for i in range(len(index)))
    condition[-1] = condition[-1].replace(" and", '')
    try:
        index_list = [item[0] for item in enumerate(source_female_features) if eval(" ".join(condition))]
    except Exception as e:
        print(e)
        print("程序出错!执行退出操作")
        sys.exit()
    if index_list:
        tensors = stack([female_picture_tensor[index] for index in index_list])
        labels = to_array([source_female_features[index] for index in index_list])
        return tensors, labels
    else:
        print("没有找到!")
        return -1, -1






