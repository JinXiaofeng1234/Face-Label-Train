from os import listdir
from os import path
import numpy as np
import cv2
from cv2 import imread, imwrite, resize, cvtColor, COLOR_BGR2RGB, rectangle, VideoCapture, flip
from PyFaceDet import facedetectcnn
from sys import argv as sys_argv
from sys import exit as sys_exit
import json
import torchvision.transforms as transforms
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsScene, QGraphicsView,
                             QVBoxLayout, QHBoxLayout, QPushButton, QWidget,
                             QFileDialog, QMessageBox, QLabel, QScrollBar, QLineEdit,
                             QComboBox, QGraphicsPixmapItem, QTableWidget, QTableWidgetItem,
                             QDialog, QTabWidget, QListWidget, QListWidgetItem, QTextBrowser)
from PyQt5.QtGui import QPixmap, QPen, QColor, QImage, QIcon
from re import search
from re import findall
from PyQt5.QtCore import Qt, QRectF, QPointF, QEventLoop, QSize, QTimer
from csv import writer as csv_writer
from csv import reader as csv_reader
import pandas as pd


def source_data_custom_sort_key(s):
    math = search(f"data \((\d+)\)\.jpg", s)
    if math:
        return int(math.group(1))
    else:
        return s


def data_custom_sort_key(filename):
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


def qt_image_to_array(qt_image):
    """将QImage转换为NumPy数组"""
    # 获取QImage的尺寸
    width, height, channels = qt_image.width(), qt_image.height(), qt_image.depth()
    # 获取QImage的字节串
    bytes_str = qt_image.bits().asstring(width * height * channels // 8)
    # 根据通道数和图像尺寸创建NumPy数组
    res_array = np.frombuffer(bytes_str, dtype=np.uint8).reshape((height, width, channels // 8))
    img_array = res_array[:, :, :3]
    return img_array


def img_process(img):
    # 转换成张量前的图片预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x / 255.0),

    ])
    # 转换为pytorch张量
    tensor_img = transform(img).unsqueeze(0)

    return tensor_img


def label_predict(ai_model, img, labels_table):
    ls = list()
    with torch.no_grad():
        input_tensor = img_process(img)
        # print(input_tensor.shape)
        input_tensor = input_tensor.cuda()
        # ai_model.cuda()
        output = ai_model(input_tensor)
    for i in output:
        ls.append(torch.max(i, dim=0)[1].item())
    print('模型预测的图片属性:')
    for cla, index in enumerate(ls):
        print(labels_table['table'][cla][index])
    return ls


def load_labels():
    # 读入标签数据集
    df = pd.read_csv('backup/source_face_labels.csv')
    # 计算每种标签组合的数量
    count = df.groupby(['gender', 'age_group', 'emoticon', 'ornaments']).size().reset_index(name='count_num')

    # 输出结果
    res_ls = count.values.tolist()
    # print(self.labels_ls)
    return res_ls


def load_labeled_count():
    with open('archive.json', 'r') as file:
        archive_dic = json.load(file)
    return archive_dic['num']


# 获取标注记录,但是只读取单个人脸图片的记录
def get_labeled_labels_dic():
    df = pd.read_csv('backup/source_face_labels.csv')

    df2 = df.set_index('image_name').T

    label_labels_dic = dict()
    for i in df2.keys():
        if isinstance(df2[i].count(), np.int32):
            label_labels_dic[i] = df2[i].to_list()[:-5]
    return label_labels_dic


# 这个函数专门为摄像头检测窗口准备
def label_face(model, source_img, labels_t, show_browser):
    show_browser.setPlainText("")
    res = source_img
    faces = facedetectcnn.facedetect_cnn(source_img)
    if len(faces) >= 1:
        faces = [face for face in faces if face[4] >= 99]
        for face_index, face in enumerate(faces):
            res_ls = list()
            x1, y1, x2, y2 = face[0], face[1], face[0] + face[2], face[1] + face[3]
            face_region = source_img[y1:y2, x1:x2]
            pre_ls = label_predict(model, face_region, labels_t)
            for cla, index in enumerate(pre_ls):
                res_ls.append(labels_t['table'][cla][index])
            show_browser.append(f"face{face_index + 1}:{' '.join(res_ls)}\n")
            res = rectangle(source_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(res, f'face{face_index + 1}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)
    return res


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Window")
        self.setGeometry(100, 100, 1920, 1080)

        # 创建选项卡窗口
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # 创建四个窗口类
        self.window1 = ImageCropApp()
        self.window2 = LabelTableView()
        self.window3 = ImageViewer()
        self.window4 = FaceDetectorApp(self.window1.model, self.window1.labels_table)

        # 将两个窗口类添加到选项卡窗口中
        self.tab_widget.addTab(self.window1, "Image Crop App")
        self.tab_widget.addTab(self.window2, "Label Table View")
        self.tab_widget.addTab(self.window3, "Image View")
        self.tab_widget.addTab(self.window4, "Detector")

        # 绑定选项卡切换事件
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index):
        # 根据当前选项卡索引执行相应的操作
        if index == 1:
            self.window2.update_labels()
        elif index == 2:
            self.window3.update_images()

    def closeEvent(self, event):
        data = {'num': self.window1.num}
        # 将字典保存为 JSON 文件
        with open("archive.json", "w") as f:
            json.dump(data, f, indent=4)
        event.accept()


class NonModalDialog(QDialog):
    def __init__(self, parent=None):
        super(NonModalDialog, self).__init__(parent)
        self.setWindowTitle("标记窗口")
        self.setWindowModality(Qt.NonModal)  # 设置为非模态

        layout = QVBoxLayout(self)
        self.label = QLabel("在完成图片标注操作后点击确定", self)
        layout.addWidget(self.label)
        self.ok_button = QPushButton("确认", self)
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)


class CustomGraphicsView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.scene = scene
        self.startPoint = None
        self.endPoint = None
        self.rectItem = None

        self.startPoint_copy = None
        self.endPoint_copy = None
        self.rectPen = QPen(QColor(255, 0, 0), 2)  # 红色，2px宽

    def clear_rect(self):
        if self.rectItem:
            self.scene.removeItem(self.rectItem)
            self.rectItem = None
        self.startPoint = None
        self.endPoint = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 将鼠标点击位置映射到场景坐标
            scene_pos = self.mapToScene(event.pos())

            # 如果这是第一次点击，记录起始点
            if self.startPoint is None:
                self.startPoint = self.startPoint_copy = scene_pos

            else:
                # 如果这是第二次点击，记录结束点并绘制矩形
                self.endPoint = self.endPoint_copy = scene_pos
                self.draw_rectangle()
                # 重置起始点以便下次绘制
                self.startPoint = None
                self.endPoint = None

    def draw_rectangle(self):
        # 如果存在之前的矩形，移除它
        if self.rectItem:
            self.scene.removeItem(self.rectItem)
            self.rectItem = None
        # 根据起始点和结束点绘制矩形
        if self.startPoint and self.endPoint:
            # print(self.startPoint.x(), self.startPoint.y(), self.endPoint.x(), self.endPoint.y())
            rect = QRectF(self.startPoint, self.endPoint)
            self.rectItem = self.scene.addRect(rect, self.rectPen)


class ImageCropApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化UI
        self.marked_flag = None
        self.current_image_name = None
        self.data_score_btn = None
        self.data_label_compare_btn = None
        self.sort_labels_btn = None
        self.labels_data_ls = None
        self.tableWidget = None
        self.label_confirm_btn = None
        self.sp_text_box = None
        self.sp_name_label = None
        self.ai_assisted_btn = None
        self.scroll_bar = None
        self.confirmButton = None
        self.turn_page_btn = None
        self.text_box = None
        self.view = None
        self.scene = None
        self.prevButton = None
        self.nextButton = None
        self.display_label = None
        self.selectFolderButton = None
        self.img_ls = list()
        self.label_combo = list()
        self.folder_path = None
        self.num = -1
        self.counter = -1

        self.marked = False
        self.labeled = False

        self.label_name_ls = ['性别:', '年龄段:', '面部表情:', '饰物:']
        self.combo_items = [['男', '女', '无'], ['儿童', '青年', '年轻人', '中年人', '老年人', '无'],
                            ['惊讶', '愤怒', '哀愁', '欢喜', '无表情', '严肃'],
                            ['眼镜', '口罩', '眼罩', '面罩', '无', '口罩+眼镜']]

        # 读入标签表
        with open('labels_table.json', encoding='utf-8') as f:
            self.labels_table = json.load(f)
        if self.labels_table:
            self.msg_box('标签表读入成功')
        self.labels_dic = {item: ls.index(item) + 1 for ls in self.combo_items for item in ls}
        self.sp_ls = list()
        self.labels_ls = list()
        self.prop = 100
        self.init_pic_name = None
        self.init_ui()

        # 启动模型,由于模型本身就是以cuda模式保存,所以不需要额外开启cuda推理
        self.model = torch.load('resnet_model.pth')
        # self.model.cuda()
        if self.model:
            self.msg_box('模型启动成功')
            try:
                self.model.eval()
                self.msg_box('模型开启推理模式成功')
            except Exception as e:
                self.msg_box(f'模型开启推理模式失败,报错:{e}')
            try:
                dummy_input = torch.randn(1, 3, 224, 224).cuda()  # 生成一个随机输入
                with torch.no_grad():
                    self.model(dummy_input)
                self.msg_box('模型预热预测成功')
            except Exception as e:
                self.msg_box(f'模型预热预测失败,报错:{e}')

        self.labeled_labels_dic = get_labeled_labels_dic()  # 图片标注记录字典,根据图片名锁定标签,但不能查找一个图片多条记录情况

    def msg_box(self, content):
        message = QMessageBox.information(self, '事件', content)

    def display_img(self, num):
        if self.img_ls and self.folder_path:
            length = len(self.img_ls)
            if -length <= num <= length - 1:
                self.view.clear_rect()
                self.local_image(f'{self.folder_path}/{self.img_ls[num]}', resize_bool=False, qt=None)
                self.show_name()
            else:
                self.msg_box('已超过边界值!')
        else:
            self.msg_box('你还没有选择图片文件夹!')

    def put_button(self, rp, ct):
        rp.addWidget(ct)
        if ct != self.scroll_bar:
            ct.setFixedWidth(120)

    def get_column_data(self):
        row_dic = dict()
        for row in range(self.tableWidget.rowCount()):
            tmp_ls = [self.tableWidget.item(row, 0).text(), self.tableWidget.item(row, 1).text(),
                      self.tableWidget.item(row, 2).text(), self.tableWidget.item(row, 3).text()]
            label_count = self.tableWidget.item(row, 4)
            if label_count is not None and label_count.text().isnumeric():
                row_dic[tuple(tmp_ls)] = (int(label_count.text()))
        return row_dic

    def sort_by_count_ascending(self):
        data = self.get_column_data()
        sorted_data = sorted(data.items(), key=lambda x: x[1])
        row_item_ls = [list(item[0]) + [str(item[1])] for item in sorted_data]
        for row in range(self.tableWidget.rowCount()):
            row_ls = [self.tableWidget.item(row, i) for i in range(5)]
            for index, item in enumerate(row_ls):
                item.setText(row_item_ls[row][index])

    def init_ui(self):
        # 设置窗口标题和大小
        self.setWindowTitle('Image Crop App')
        self.setGeometry(100, 100, 800, 600)

        # 创建主要布局
        main_layout = QHBoxLayout()

        # 创建图像显示区域
        self.scene = QGraphicsScene()
        self.view = CustomGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)  # 设置拖动模式

        # 创建右侧面板
        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()
        data_analysis_panel = QVBoxLayout()

        # 创建子容器
        sub_panel = QHBoxLayout()  # 放置跳转按钮和所以输入框,位于left_panel中
        sub_panel_data_analysis = QHBoxLayout()
        sub_panel_img_statue = QHBoxLayout()  # 放置图片的大小缩放滚动条和标注标记,位于left_panel中

        # 创建按钮
        self.selectFolderButton = QPushButton('选择文件夹')
        self.confirmButton = QPushButton('提交')
        self.prevButton = QPushButton('上一页')
        self.nextButton = QPushButton('下一页')
        self.display_label = QLabel('此处显示图片名')

        self.scroll_bar = QScrollBar()
        self.scroll_bar.setStyleSheet("QScrollBar:vertical { width: 30px; }")
        self.scroll_bar.setMinimum(10)
        self.scroll_bar.setMaximum(500)
        self.scroll_bar.setSingleStep(1)
        self.scroll_bar.setPageStep(50)
        self.scroll_bar.setValue(100)

        self.marked_flag = QLabel()
        self.marked_flag.setFixedWidth(20)
        self.marked_flag.setFixedHeight(20)
        self.marked_flag.setStyleSheet("QLabel { background-color : green; }")

        self.text_box = QLineEdit()
        self.text_box.setFixedWidth(40)

        self.turn_page_btn = QPushButton('跳转')
        self.turn_page_btn.setFixedWidth(60)

        """ 创建最右侧布局控件 """

        for i in range(1, len(self.label_name_ls) + 1):
            label_type = QLabel()
            combo_type = QComboBox()
            setattr(self, f'label_box{i}', combo_type)
            setattr(self, f'label_name{i}', label_type)
            label_type.setText(self.label_name_ls[i - 1])
            label_type.setFixedWidth(100)
            combo_type.addItems(self.combo_items[i - 1])
            combo_type.setFixedWidth(100)
            self.label_combo += [label_type, combo_type]

        self.sp_name_label = QLabel()
        self.sp_name_label.setFixedWidth(100)

        self.sp_text_box = QLineEdit()
        self.sp_text_box.setFixedWidth(100)

        self.sp_name_label.setText('人物名字')

        self.ai_assisted_btn = QPushButton('人工智能检测')
        self.ai_assisted_btn.setFixedWidth(100)

        self.label_confirm_btn = QPushButton('提交标签')
        self.label_confirm_btn.setFixedWidth(100)

        self.labels_data_ls = load_labels()
        self.tableWidget = QTableWidget(self)
        self.tableWidget.setRowCount(len(self.labels_data_ls))  # 假设有四个标签
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setHorizontalHeaderLabels(['性别', '年龄段', '表情', '饰物', '数量'])
        labels_table_ls = self.labels_table['table']
        for row, (label1, label2, label3, label4, label5) in enumerate(self.labels_data_ls):
            self.tableWidget.setItem(row, 0, QTableWidgetItem(str(labels_table_ls[0][label1 - 1])))
            self.tableWidget.setItem(row, 1, QTableWidgetItem(str(labels_table_ls[1][label2 - 1])))
            self.tableWidget.setItem(row, 2, QTableWidgetItem(str(labels_table_ls[2][label3 - 1])))
            self.tableWidget.setItem(row, 3, QTableWidgetItem(str(labels_table_ls[3][label4 - 1])))
            self.tableWidget.setItem(row, 4, QTableWidgetItem(str(label5)))
        self.sort_labels_btn = QPushButton('升序排列')
        self.data_label_compare_btn = QPushButton('比对标签和图片数量')
        self.data_score_btn = QPushButton('数据集打分')

        for cb in self.label_combo:
            right_panel.addWidget(cb)

        right_panel.addWidget(self.sp_name_label)
        right_panel.addWidget(self.sp_text_box)
        right_panel.addWidget(self.ai_assisted_btn)
        right_panel.addWidget(self.label_confirm_btn)

        sub_panel.addWidget(self.text_box)
        sub_panel.addWidget(self.turn_page_btn)

        sub_panel_img_statue.addWidget(self.marked_flag)
        sub_panel_img_statue.addWidget(self.scroll_bar)

        left_panel.addLayout(sub_panel_img_statue)
        # 将组件添加到右侧面板
        control_ls = [self.display_label, self.selectFolderButton,
                      self.confirmButton, self.prevButton, self.nextButton]
        for control in control_ls:
            self.put_button(left_panel, control)
        left_panel.addLayout(sub_panel)

        data_analysis_panel.addWidget(self.tableWidget)
        sub_panel_data_analysis.addWidget(self.sort_labels_btn)
        sub_panel_data_analysis.addWidget(self.data_label_compare_btn)
        sub_panel_data_analysis.addWidget(self.data_score_btn)
        data_analysis_panel.addLayout(sub_panel_data_analysis)

        # 连接信号和槽
        self.selectFolderButton.clicked.connect(self.select_folder)
        self.confirmButton.clicked.connect(self.confirm_selection)
        self.prevButton.clicked.connect(self.prev_image)
        self.nextButton.clicked.connect(self.next_image)
        self.scroll_bar.valueChanged.connect(self.change_img_size)
        self.turn_page_btn.clicked.connect(self.skip_pic)
        self.ai_assisted_btn.clicked.connect(self.ai_label)
        self.label_confirm_btn.clicked.connect(self.confirm_labels)
        self.sort_labels_btn.clicked.connect(self.sort_by_count_ascending)
        self.data_label_compare_btn.clicked.connect(self.compare_data_label)
        self.data_score_btn.clicked.connect(self.get_train_dataset_score)

        # 将左侧和右侧布局添加到主布局
        main_layout.addWidget(self.view)
        main_layout.addLayout(left_panel)
        main_layout.addLayout(right_panel)
        main_layout.addLayout(data_analysis_panel)

        # 设置中心窗口
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def confirm_labels(self):
        if self.folder_path:
            length = len(self.img_ls)
            if (self.view.startPoint is None and self.view.endPoint is None) and -length <= self.num <= length - 1:
                start_point = self.view.startPoint_copy
                end_point = self.view.endPoint_copy
                if start_point and end_point:
                    if not self.labeled:
                        self.labeled = True
                    else:
                        pass
                    for i in range(1, len(self.label_combo), 2):
                        self.labels_ls += [self.labels_dic[self.label_combo[i].currentText()]]
                    input_sp_name = self.sp_text_box.text()
                    if input_sp_name:
                        if input_sp_name not in self.sp_ls:
                            self.sp_ls.append(input_sp_name)
                        self.labels_ls += [self.sp_ls.index(input_sp_name)]
                    else:
                        self.labels_ls += [-1]

                    point_ls = [start_point.x(), start_point.y(), end_point.x(), end_point.y()]
                    if self.prop != 100:
                        point_ls = [int(item / (self.prop / 100)) for item in point_ls]

                    self.labels_ls += point_ls
                    self.labels_ls.insert(0, self.img_ls[self.num])

                    """
                    标签表的动态添加
                    """
                    # print(self.labels_ls)
                    int_index_ls = self.labels_ls[1:5]
                    ls = [str(i) for i in int_index_ls]
                    # 清洗标签并字符串化
                    convert_ls = [labels_ls[int_index_ls[i] - 1] for i, labels_ls in enumerate(self.combo_items)]
                    # print(convert_ls)
                    row_count = self.tableWidget.rowCount()  # 统计现有标签表行数
                    for row in range(row_count):
                        row_data = list()
                        for col in range(self.tableWidget.columnCount() - 1):
                            item = self.tableWidget.item(row, col)
                            if item:
                                row_data.append(item.text())
                        if row_data == convert_ls:
                            match_index = row
                            new_value = int(self.tableWidget.item(match_index, 4).text()) + 1
                            self.tableWidget.setItem(match_index, 4, QTableWidgetItem(str(new_value)))
                            break
                    else:
                        row_count += 1
                        self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
                        self.tableWidget.setItem(row_count - 1, 4, QTableWidgetItem('1'))
                        for index in range(len(convert_ls)):
                            self.tableWidget.setItem(row_count - 1, index, QTableWidgetItem(convert_ls[index]))

                    self.label_confirm_btn.setEnabled(False)
                else:
                    self.msg_box('没有框选人脸!')
            else:
                self.labeled = False
                self.msg_box('没有框选人脸!')
        else:
            self.msg_box('没有选择文件夹!')

    def skip_pic(self):
        length = len(self.img_ls)
        index = self.text_box.text()
        if index:
            turn_num = int(eval(index))
            if -length <= turn_num <= length - 1:
                self.num = turn_num
                self.view.clear_rect()
                self.local_image(f'{self.folder_path}/{self.img_ls[turn_num]}', resize_bool=False, qt=None)
                self.prop = 100
                self.scroll_bar.setValue(100)
                self.show_name()
                self.load_labels(self.current_image_name)
            else:
                self.msg_box('输入的索引值越过边界')
        else:
            self.msg_box('你没有输入跳转索引值')

    def show_name(self):
        self.current_image_name = self.img_ls[self.num]
        self.display_label.setText(f'{self.current_image_name}')

    def change_img_size(self):
        self.prop = self.scroll_bar.value()
        length = len(self.img_ls)
        if length and -length <= self.num <= length - 1:
            prop = self.scroll_bar.value()
            img = imread(f'{self.folder_path}/{self.img_ls[self.num]}')
            img_resized = resize(img, dsize=None, fx=prop / 100, fy=prop / 100)
            rgb_image = cvtColor(img_resized, COLOR_BGR2RGB)

            # 创建 QImage 对象
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.view.clear_rect()
            self.local_image(image_path=None, resize_bool=True, qt=qt_image)
        else:
            self.msg_box('图片文件夹未选择或图片索引号不合法')

    def local_image(self, image_path, resize_bool, qt):
        # 加载并显示图像
        if not resize_bool:
            pixmap = QPixmap(image_path)
        else:
            pixmap = QPixmap.fromImage(qt)

        self.scene.clear()
        self.view.startPoint_copy = None
        self.view.endPoint_copy = None

        self.scene.addPixmap(pixmap)

        # self.view.setSceneRect(pixmap.rect())

    def select_folder(self):
        # 选择包含图像的文件夹
        self.folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")

        if self.folder_path:
            self.img_ls = listdir(self.folder_path)
            filtered_img_ls = [name for name in self.img_ls if name.lower().endswith('.jpg')]
            self.img_ls = sorted(filtered_img_ls, key=source_data_custom_sort_key)
            # 取得最新的图片索引
            self.num = load_labeled_count()
            self.display_img(self.num)
            self.init_pic_name = self.img_ls[self.num]
            # print(self.init_pic_name)
            self.msg_box('图片文件夹已经导入')

    def confirm_selection(self):
        # 确认裁剪区域和属性
        if self.folder_path:
            if self.view.startPoint is None and self.view.endPoint is None:
                start_point = self.view.startPoint_copy
                end_point = self.view.endPoint_copy
                length = len(self.img_ls)
                if start_point and end_point:
                    s_x = start_point.x()
                    s_y = start_point.y()
                    e_x = end_point.x()
                    e_y = end_point.y()
                    if -length <= self.num <= length - 1:
                        if self.labeled:
                            image = imread(f'{self.folder_path}/{self.img_ls[self.num]}')

                            if self.prop == 100:
                                pass
                            elif 10 <= self.prop < 100 or 101 <= self.prop <= 500:
                                k = self.prop / 100
                                s_y /= k
                                e_y /= k
                                s_x /= k
                                e_x /= k
                            res_image = image[int(s_y): int(e_y),
                                        int(s_x): int(e_x)]
                            image_shape = res_image.shape
                            if image_shape[0] >= 20 and image_shape[1] >= 20:
                                if self.init_pic_name == self.img_ls[self.num]:
                                    self.counter += 1
                                else:
                                    self.counter = 0
                                    self.init_pic_name = self.img_ls[self.num]

                                path_name = f'backup/source_modified_img/face({self.num})_{self.counter}.jpg'
                                # res_image_resized = resize(res_image, (224, 224))
                                if self.labels_ls:  # 防止空标签列表提交
                                    with open('backup/source_face_labels.csv', "a", newline='',
                                              encoding='utf-8') as csvfile:
                                        csvwriter = csv_writer(csvfile)
                                        csvwriter.writerow(self.labels_ls)
                                    imwrite(path_name, res_image)
                                    # self.counter += 1
                                    # self.labeled = False
                                    self.labeled_labels_dic[self.current_image_name] = self.labels_ls[1:-5]
                                    self.labels_ls.clear()
                                else:
                                    self.msg_box('请勿提交空属性!')
                            else:
                                self.msg_box('人脸图片过小')
                            self.label_confirm_btn.setEnabled(True)
                        else:
                            self.msg_box('标签未提交')
                    else:
                        self.msg_box('请选择图片!')
                else:
                    self.msg_box('你还没有框选人脸!')
            else:
                self.msg_box('请框选完整的人脸!')
                self.label_confirm_btn.setEnabled(True)
        else:
            self.msg_box('你还没有选择文件夹')

    def get_pixmap_from_scene(self):
        for item in self.scene.items():
            if isinstance(item, QGraphicsPixmapItem):
                return item.pixmap()
        return None  # 如果没有找到 QGraphicsPixmapItem，返回 None

    def load_labels(self, image_name):  # 读取标注过的标签
        try:
            labeled_labels_ls = self.labeled_labels_dic[image_name]
            if labeled_labels_ls:
                labeled_ls = [int(index) - 1 for index in labeled_labels_ls]  # 由于标签列表索引从1开始,且为浮点值.所以要处理
                j = 0
                for i in range(1, len(self.label_combo), 2):
                    self.label_combo[i].setCurrentIndex(labeled_ls[j])
                    j += 1
                self.marked_flag.setStyleSheet("QLabel { background-color : green; }")
            else:
                self.msg_box('标签存储表读取出问题')
        except KeyError:
            self.marked_flag.setStyleSheet("QLabel { background-color : red; }")
        except Exception as e:
            self.msg_box(f'出现意外错误:{e}')

    def ai_process_img(self, x1, y1, x2, y2, pos_ls, img_array):
        pos_ls += [x1, y1, x2, y2]
        if sum(pos_ls) > 0:
            face_region = img_array[y1:y2, x1:x2]
            predict_ls = label_predict(self.model, face_region, self.labels_table)
            if predict_ls:
                j = 0
                for i in range(1, len(self.label_combo), 2):
                    self.label_combo[i].setCurrentIndex(predict_ls[j])
                    j += 1
            start_point = QPointF(x1, y1)
            end_point = QPointF(x2, y2)
            self.view.startPoint = self.view.startPoint_copy = start_point
            self.view.endPoint = self.view.endPoint_copy = end_point
            self.view.draw_rectangle()

    def ai_label(self):
        pos_ls = list()
        pixmap = self.get_pixmap_from_scene()
        if pixmap:
            qt_img = pixmap.toImage()
            img_array = qt_image_to_array(qt_img)
            faces = facedetectcnn.facedetect_cnn(img_array)
            faces = [face for face in faces if face[4] >= 99]
            if faces:
                if len(faces) == 1:
                    x1, y1, x2, y2 = faces[0][0], faces[0][1], faces[0][0] + faces[0][2], faces[0][1] + faces[0][3]
                    self.ai_process_img(x1, y1, x2, y2, pos_ls, img_array)
                    self.view.startPoint = None
                    self.view.endPoint = None
                else:
                    for face in faces:
                        x1, y1, x2, y2 = face[0], face[1], face[0] + face[2], face[1] + face[3]
                        self.ai_process_img(x1, y1, x2, y2, pos_ls, img_array)
                        self.view.startPoint = None
                        self.view.endPoint = None
                        dialog = NonModalDialog(self)
                        dialog.finished.connect(self.on_dialog_closed)  # 连接对话框关闭的信号
                        dialog.show()

                        # 使用 QEventLoop 暂停主线程,等待对话框关闭
                        event_loop = QEventLoop()
                        dialog.finished.connect(event_loop.quit)
                        event_loop.exec_()
            else:
                self.msg_box('自动标注失败')
        else:
            self.msg_box('没有读取到图片')

    def on_dialog_closed(self, result):
        pass

    def compare_data_label(self):
        df = pd.read_csv('backup/source_face_labels.csv')
        file_ls = listdir('backup/source_modified_img')
        if len(df) == len(file_ls) - 1:
            content = '训练集标签与图片数量匹配'
        else:
            content = '训练集标签与图片数量不匹配'
        self.msg_box(content)

    def get_train_dataset_score(self):
        label_count_ls = list()
        for row in range(self.tableWidget.rowCount()):
            label_count = self.tableWidget.item(row, 4)
            if label_count:
                label_count_ls.append(int(label_count.text()))
        # 计算总标签数
        total_labels = sum(label_count_ls)
        # 计算每个标签的相对频率
        label_frequencies = np.array([count / total_labels for count in label_count_ls])
        # 计算 Softmax 输出
        score_output = 1 - (np.max(label_frequencies) - np.average(label_frequencies)) / np.max(label_frequencies)
        self.msg_box(f'训练集标签平衡指标:{score_output}')

    def prev_image(self):
        # 显示上一张图像
        self.num -= 1
        self.display_img(self.num)
        self.prop = 100
        self.scroll_bar.setValue(100)
        self.load_labels(self.current_image_name)

    def next_image(self):
        self.num += 1
        # 显示下一张图像
        self.display_img(self.num)
        self.prop = 100
        self.scroll_bar.setValue(100)
        self.load_labels(self.current_image_name)


class LabelTableView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.labels_count = None
        self.data = None
        self.save_btn = None
        self.table_widget = None
        self.button_layout = None
        self.table_layout = None
        self.main_layout = None
        self.setWindowTitle("Main Window")
        self.setGeometry(100, 100, 1920, 1080)
        self.init_ui()

    def get_the_labels_data(self):
        # 读取 CSV 文件
        filename = 'backup/source_face_labels.csv'
        self.data = []
        with open(filename, 'r', encoding='utf-8') as file:
            csv_reade = csv_reader(file)
            for row in csv_reade:
                self.data.append(row)

    def init_ui(self):
        # 创建选项卡窗口
        self.main_layout = QHBoxLayout()

        self.table_layout = QVBoxLayout()
        self.button_layout = QVBoxLayout()

        self.table_widget = QTableWidget()
        self.save_btn = QPushButton('保存')

        self.table_layout.addWidget(self.table_widget)
        self.button_layout.addWidget(self.save_btn)

        self.main_layout.addLayout(self.table_layout)
        self.main_layout.addLayout(self.button_layout)

        self.save_btn.clicked.connect(self.save_labels)
        # 设置中心窗口
        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)
        self.get_the_labels_data()

        # 创建 QTableWidget
        self.labels_count = len(self.data) - 1
        self.table_widget.setColumnCount(len(self.data[0]))
        self.table_widget.setRowCount(self.labels_count)
        for i, row in enumerate(self.data[1:]):
            for j, cell in enumerate(row):
                item = QTableWidgetItem(cell)
                self.table_widget.setItem(i, j, item)

    def update_labels(self):
        self.get_the_labels_data()
        current_labels_count = len(self.data) - 1
        if current_labels_count > self.labels_count:
            self.table_widget.setRowCount(current_labels_count)
            for i, row in enumerate(self.data[self.labels_count + 1:]):
                for j, cell in enumerate(row):
                    item = QTableWidgetItem(cell)
                    self.table_widget.setItem(self.labels_count + i, j, item)
            self.labels_count = current_labels_count

    def save_labels(self):
        nested_list = []
        for row in range(self.table_widget.rowCount()):
            row_data = []
            for col in range(self.table_widget.columnCount()):
                item = self.table_widget.item(row, col)
                if item:
                    row_data.append(item.text())
                else:
                    row_data.append(None)
            nested_list.append(row_data)
        with open('backup/source_face_labels.csv', 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv_writer(csvfile)
            writer.writerow(
                ['image_name', 'gender', 'age_group', 'emoticon', 'ornaments', 'character_name', 'face_pos_1x',
                 'face_pos_1y', 'face_pos_2x', 'face_pos_2y'])
            for row in nested_list:
                row = [str(cell) for cell in row]
                writer.writerow(row)


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.path = None
        self.listWidget = None
        self.image_count = len(listdir('backup/source_modified_img')) - 1
        self.img_ls = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Image Viewer')
        self.setGeometry(100, 100, 800, 600)

        # 创建一个 QListWidget 用于显示缩略图
        self.listWidget = QListWidget()
        self.listWidget.setViewMode(QListWidget.IconMode)
        self.listWidget.setIconSize(QSize(120, 120))
        self.listWidget.setResizeMode(QListWidget.Adjust)
        self.listWidget.setSpacing(10)

        # 创建一个 QVBoxLayout 并添加 QListWidget
        layout = QVBoxLayout()
        layout.addWidget(self.listWidget)

        # 创建一个 QWidget 并设置布局
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.path = 'backup/source_modified_img'
        # 加载图片文件夹
        self.load_images(self.path)

    def get_the_img_ls(self, folder_path):
        self.img_ls = listdir(folder_path)
        self.img_ls.remove('_system~.ini')
        self.img_ls = sorted(self.img_ls, key=data_custom_sort_key)[-100:]

    def put_image(self, index_img_path_tup):
        # 创建一个 QListWidgetItem 并设置缩略图
        item = QListWidgetItem()
        pixmap = QPixmap(index_img_path_tup[1])
        pixmap = pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        item.setIcon(QIcon(pixmap))
        item.setToolTip(index_img_path_tup[1])
        self.listWidget.insertItem(index_img_path_tup[0], item)

    def load_images(self, folder_path):
        self.get_the_img_ls(folder_path)
        index_img_path_ls = [(index, path.join(folder_path, filename)) for index, filename in enumerate(self.img_ls)]
        for i in index_img_path_ls:
            self.put_image(i)

    def update_images(self):
        current_image_count = len(listdir('backup/source_modified_img')) - 1
        if current_image_count > self.image_count:
            difference = current_image_count - self.image_count
            self.get_the_img_ls(self.path)
            index_img_path_ls = [(self.image_count + index, path.join(self.path, filename)) for index, filename
                                 in enumerate(self.img_ls[-difference:])]
            for i in index_img_path_ls:
                self.put_image(i)
            self.image_count = current_image_count


class FaceDetectorApp(QWidget):
    def __init__(self, model, labels_table, parent=None):
        super().__init__(parent)  # 父类的构造函数
        self.image = None
        self.timer_camera = QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = VideoCapture()  # 视频流
        self.CAM_NUM = 0  # 为0时表示视频流来自笔记本内置摄像头

        self.__layout_main = QHBoxLayout()  # 水平总布局
        self.__layout_fun_layout = QVBoxLayout()  # 按键布局
        self.__layout_data_show = QVBoxLayout()  # 数据(视频)显示布局
        self.button_open_camera = QPushButton('打开摄像头')  # 建立用于打开摄像头的按键
        self.button_screenshot = QPushButton('截取人脸')

        self.label_show_camera = QLabel()  # 定义显示视频的Label
        self.face_image_label = QLabel()  # 定义最终人像效果的Label
        self.prediction_browser = QTextBrowser()
        self.set_ui()  # 初始化程序界面
        self.slot_init()  # 初始化槽函数
        self.setGeometry(50, 50, 1600, 800)

        self.model = model
        self.labels_table = labels_table

    '''程序界面布局'''

    def set_ui(self):

        self.button_open_camera.setMinimumHeight(50)  # 设置按键大小
        self.button_open_camera.setFixedWidth(100)

        self.button_screenshot.setMinimumHeight(50)
        self.button_screenshot.setFixedWidth(100)

        '''信息显示'''

        self.label_show_camera.setFixedSize(781, 641)  # 给显示视频的Label设置大小为641x481
        '''把按键加入到按键布局中'''
        self.__layout_fun_layout.addWidget(self.button_open_camera)  # 把打开摄像头的按键放到按键布局中
        self.__layout_fun_layout.addWidget(self.button_screenshot)  # 把截图的按键放到按键布局中
        '''把某些控件加入到总布局中'''
        self.__layout_main.addLayout(self.__layout_fun_layout)  # 把按键布局加入到总布局中
        self.__layout_main.addWidget(self.label_show_camera)  # 把用于显示视频的Label加入到总布局中
        self.__layout_main.addWidget(self.face_image_label)  # 把用于显示人像照片的Label加入到总布局中
        self.__layout_main.addWidget(self.prediction_browser)
        '''总布局布置好后就可以把总布局作为参数传入下面函数'''
        self.setLayout(self.__layout_main)  # 到这步才会显示所有控件

    '''初始化所有槽函数'''

    def slot_init(self):
        self.button_open_camera.clicked.connect(
            self.button_open_camera_clicked)  # 若该按键被点击，则调用button_open_camera_clicked()
        self.timer_camera.timeout.connect(self.out_ret)  # 若定时器结束，则调用show_camera()
        self.button_screenshot.clicked.connect(self.screen_shot)  # 让截图按钮连接相应函数

    '''槽函数之一'''

    def button_open_camera_clicked(self):
        if not self.timer_camera.isActive():  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if not flag:  # flag表示open()成不成功
                QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确",
                                    buttons=QMessageBox.Ok)
            else:
                self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open_camera.setText('关闭摄像头')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.label_show_camera.clear()  # 清空视频显示区域
            self.button_open_camera.setText('打开摄像头')

    def out_ret(self):
        flag, self.image = self.cap.read()  # 从视频流中读取
        ret = resize(self.image, (780, 640))  # 把读到的帧的大小重新设置为 640x480
        ret = flip(cvtColor(ret, COLOR_BGR2RGB), 1)  # 视频色彩转换回RGB，这样才是现实的颜色
        ret = label_face(self.model, ret, self.labels_table, self.prediction_browser)  # 进行人脸检测
        convert_image = QImage(ret.data, ret.shape[1], ret.shape[0],
                               QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_camera.setPixmap(QPixmap.fromImage(convert_image))  # 往显示视频的Label里 显示QImage

    def screen_shot(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys_argv)
    ex = MainWindow()
    ex.show()
    sys_exit(app.exec_())
