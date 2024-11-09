import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit
import paramiko


class LinuxClientApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linux Client")
        self.setGeometry(300, 300, 600, 400)

        # 添加布局
        self.layout = QVBoxLayout()

        # 服务器信息输入
        self.server_ip_label = QLabel('Server IP:')
        self.server_ip_input = QLineEdit()
        self.username_label = QLabel('Username:')
        self.username_input = QLineEdit()
        self.password_label = QLabel('Password:')
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.port_label = QLabel('Port:')
        self.port_input = QLineEdit()

        self.layout.addWidget(self.server_ip_label)
        self.layout.addWidget(self.server_ip_input)
        self.layout.addWidget(self.username_label)
        self.layout.addWidget(self.username_input)
        self.layout.addWidget(self.password_label)
        self.layout.addWidget(self.password_input)
        self.layout.addWidget(self.port_label)
        self.layout.addWidget(self.port_input)

        # 连接按钮
        self.connect_button = QPushButton('Connect')
        self.connect_button.clicked.connect(self.connect_to_server)
        self.layout.addWidget(self.connect_button)

        # 命令输入和执行
        self.command_label = QLabel('Command:')
        self.command_input = QLineEdit()
        self.execute_button = QPushButton('Execute')
        self.execute_button.clicked.connect(self.execute_command)

        self.layout.addWidget(self.command_label)
        self.layout.addWidget(self.command_input)
        self.layout.addWidget(self.execute_button)

        # 命令输出
        self.output_text = QTextEdit()
        self.layout.addWidget(self.output_text)

        self.setLayout(self.layout)

    def connect_to_server(self):
        server_ip = self.server_ip_input.text()
        username = self.username_input.text()
        password = self.password_input.text()
        port = self.port_input.text()

        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.ssh_client.connect(server_ip, port=port, username=username, password=password)
            self.output_text.append("Connected to server successfully.\n")
        except Exception as e:
            self.output_text.append(f"Failed to connect to server: {e}\n")

    def execute_command(self):
        command = self.command_input.text()
        if not command or not hasattr(self, 'ssh_client'):
            return

        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            output = stdout.read().decode()
            error = stderr.read().decode()
            if output:
                self.output_text.append(f"Output:\n{output}\n")
            if error:
                self.output_text.append(f"Error:\n{error}\n")
        except Exception as e:
            self.output_text.append(f"Failed to execute command: {e}\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LinuxClientApp()
    window.show()
    sys.exit(app.exec_())