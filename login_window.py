import os.path
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sqlite3
import hashlib
from face_labeler import MainWindow


def md5_lock(text):
    md5 = hashlib.md5()
    md5.update(text.encode('utf-8'))
    hashed_text = md5.hexdigest()
    return hashed_text


def init_db():
    conn = sqlite3.connect('user_db/database.db')

    cursor = conn.cursor()  # 创建游标

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT, 
        password TEXT
    )
    ''')

    username = 'admin'
    password = 'Jxf123456'

    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, md5_lock(password)))

    # 提交事务并关闭数据库连接
    conn.commit()
    conn.close()


class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 窗口标题
        self.face_labeler_app = None
        self.setWindowTitle('欢迎使用JinLabeler')

        # 布局初始化
        layout = QVBoxLayout()
        # 设置固定长度
        self.setFixedWidth(350)
        # logo图片
        logo_layout = QHBoxLayout()
        self.logo_label = QLabel()
        logo_pixmap = QPixmap('logo.png')
        self.logo_label.setPixmap(logo_pixmap)
        logo_layout.addWidget(self.logo_label)
        logo_layout.setAlignment(Qt.AlignCenter)
        # 用户名
        self.username_label = QLabel('用户名:')
        self.username_input = QLineEdit(self)

        # 密码
        self.password_label = QLabel('密码:')
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)

        # 按钮
        self.login_button = QPushButton('登录')
        self.clear_button = QPushButton('清空')

        # 将元素添加到布局
        layout.addLayout(logo_layout)
        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)

        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.login_button)
        button_layout.addWidget(self.clear_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # 绑定按钮事件
        self.login_button.clicked.connect(self.handle_login)
        self.clear_button.clicked.connect(self.clear_inputs)

        if not os.path.exists('user_db/database.db'):
            init_db()

        conn = sqlite3.connect('user_db/database.db')
        self.cursor = conn.cursor()  # 创建游标

    def show_message_box(self, text):
        # 创建一个消息框
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle('提示')
        msg_box.setText(text)
        # msg_box.setInformativeText('您可以在此添加更多的信息')
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.show()

    def run_labeler(self):
        self.face_labeler_app = MainWindow()
        self.face_labeler_app.show()

    def handle_login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        if username and password:
            # 这里可以添加验证逻辑
            self.cursor.execute("""
            SELECT * FROM users WHERE username = ?
            """, (username,))
            search_res = self.cursor.fetchone()
            if search_res:
                if md5_lock(password) == search_res[1]:
                    # self.show_message_box('登录成功!')
                    self.hide()
                    self.run_labeler()
                else:
                    self.show_message_box('账户正确, 密码错误!')
            else:
                self.show_message_box('没有这个账户名!')
        elif username and (not password):
            self.show_message_box('请输入密码!')
        elif password and (not username):
            self.show_message_box('请输入用户名!')
        else:
            self.show_message_box('请输入账户和密码!')

    def clear_inputs(self):
        self.username_input.clear()
        self.password_input.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    login_window = LoginWindow()
    login_window.show()
    sys.exit(app.exec_())
