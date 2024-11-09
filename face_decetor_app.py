from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2 as cv2
from PyFaceDet import facedetectcnn
from face_labeler import label_predict
from face_labeler import ImageCropApp


def label_face(model, source_img, labels_t, show_browser):
    show_browser.setPlainText("")
    source_img = cv2.flip(source_img, 1)
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
            res = cv2.rectangle(source_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(res, f'face{face_index + 1}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)
    return res


class FaceDetectorApp(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)  # 父类的构造函数
        self.image = None
        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()  # 视频流
        self.CAM_NUM = 0  # 为0时表示视频流来自笔记本内置摄像头

        self.__layout_main = QtWidgets.QHBoxLayout()  # 水平总布局
        self.__layout_fun_layout = QtWidgets.QVBoxLayout()  # 按键布局
        self.__layout_data_show = QtWidgets.QVBoxLayout()  # 数据(视频)显示布局
        self.button_open_camera = QtWidgets.QPushButton('打开摄像头')  # 建立用于打开摄像头的按键
        self.button_screenshot = QtWidgets.QPushButton('截取人脸')
        self.button_close = QtWidgets.QPushButton('退出')  # 建立用于退出程序的按键

        self.label_show_camera = QtWidgets.QLabel()  # 定义显示视频的Label
        self.face_image_label = QtWidgets.QLabel()  # 定义最终人像效果的Label
        self.prediction_browser = QtWidgets.QTextBrowser()
        self.set_ui()  # 初始化程序界面
        self.slot_init()  # 初始化槽函数
        self.setGeometry(50, 50, 1600, 800)

        sub_app = ImageCropApp()
        self.model = sub_app.model
        self.labels_table = sub_app.labels_table

    '''程序界面布局'''

    def set_ui(self):

        self.button_open_camera.setMinimumHeight(50)  # 设置按键大小
        self.button_open_camera.setFixedWidth(100)

        self.button_close.setMinimumHeight(50)
        self.button_close.setFixedWidth(100)

        self.button_screenshot.setMinimumHeight(50)
        self.button_screenshot.setFixedWidth(100)

        self.button_close.move(10, 100)  # 移动按键
        '''信息显示'''

        self.label_show_camera.setFixedSize(781, 641)  # 给显示视频的Label设置大小为641x481
        '''把按键加入到按键布局中'''
        self.__layout_fun_layout.addWidget(self.button_open_camera)  # 把打开摄像头的按键放到按键布局中
        self.__layout_fun_layout.addWidget(self.button_screenshot)  # 把截图的按键放到按键布局中
        self.__layout_fun_layout.addWidget(self.button_close)  # 把退出程序的按键放到按键布局中
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
        self.button_close.clicked.connect(self.close)  # 若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序

    '''槽函数之一'''

    def button_open_camera_clicked(self):
        if not self.timer_camera.isActive():  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM, cv2.CAP_DSHOW)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if not flag:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok)
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
        ret = cv2.resize(self.image, (780, 640))  # 把读到的帧的大小重新设置为 640x480
        ret = label_face(self.model, ret, self.labels_table, self.prediction_browser)  # 进行人脸检测
        ret = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        convert_image = QtGui.QImage(ret.data, ret.shape[1], ret.shape[0],
                                     QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(convert_image))  # 往显示视频的Label里 显示QImage

    def screen_shot(self):
        pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 固定的，表示程序应用
    ui = FaceDetectorApp()  # 实例化Ui_MainWindow
    ui.show()  # 调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())  # 不加这句，程序界面会一闪而过
