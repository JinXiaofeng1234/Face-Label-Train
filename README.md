# Face-Label-Train
FaceLabel-Train
基于PyQt5的人脸识别标注与训练一体化工具

📝 项目简介
FaceLabel-Train 是一个开源的人脸识别标注与训练工具，采用PyQt5构建界面，集成了PyTorch深度学习框架。本项目支持边标注边训练的工作模式，适合需要迭代优化数据集和模型的场景。项目文档和代码注释均采用中文，方便中文开发者理解和使用。

✨ 核心特性
🖥️ 基于PyQt5的可视化标注界面
📸 支持实时摄像头采集样本
✏️ 手动框选标注功能
🤖 集成PyTorch训练框架
☁️ 支持本地/云端训练模式切换
🇨🇳 全中文注释和文档
🛠️ 技术栈
Python 3.8+
PyQt5
PyTorch 1.8+
OpenCV
NumPy
Pandas
PIL (Pillow)
[其他依赖库...]
💻 系统要求
最低配置（本地标注）
CPU: Intel Core i5 或同等性能
内存: 8GB RAM
存储: 10GB可用空间
操作系统: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
推荐配置（本地训练）
CPU: Intel Core i7/AMD Ryzen 7 或更高
GPU: NVIDIA GTX 1660 或更高
内存: 16GB RAM
CUDA支持（用于GPU训练）
# 注意，face_labeler是启动标注软件的，并不是集成训练程序的，cloud文件夹中是云训练脚本，不过我用的平台是AutoDL
最后提醒，这是我的毕业设计，不怎么好
