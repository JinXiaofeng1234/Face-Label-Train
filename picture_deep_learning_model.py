import sys
import random
from file_manager import get_file_path
import pickle
from save_to_pkl_numpy import turn_to_pkl
from img_read import img_process
from img_group_read import group_read
import json
from torch import max as torch_max
from torch import save
from torch import load
from torch import nn
from torch.optim import SGD
from torch import no_grad
from torch import stack as to_tensor
from neural_net import pyramid_resnet18_2d_model

from os import listdir


# from sys import exit


def csv_get():
    csv_name = get_file_path("请选择csv文件:", "Csv Files (*.csv);;All Files (*)").split('/')[-1]
    turn_to_pkl(csv_name)


def train_test_split(features, labels, prop=0.2):
    length = len(labels)
    test_num = int(length * prop)
    test_index = random.sample(range(length), test_num)

    test_features = [features[i] for i in test_index]
    test_labels = to_tensor([labels[i] for i in test_index], dim=0)

    train_index = list(set(range(length)) - set(test_index))

    train_features = [features[i] for i in train_index]
    train_labels = to_tensor([labels[i] for i in train_index], dim=0)
    return train_labels, train_features, test_labels, test_features


file_ls = listdir("../deep learning model")
if "face_labels.pkl" not in file_ls:
    csv_get()
# 这里是保存为pkl文件,而不是读取
if "face_tensors.pkl" not in file_ls:
    group_read('source_modified_img')
else:
    ans = input("是否需要重新读取人脸图片训练集(y/n):")
    if ans == 'y':
        group_read('source_modified_img')
    else:
        pass

"""
读取人脸数据集和标签数据集
人脸数据集是列表嵌套张量
标签数据集是张量
"""
try:
    with open("face_labels.pkl", "rb") as fir_f:
        labels_data = pickle.load(fir_f)
    with open("face_tensors.pkl", "rb") as sec_f:
        face_data = pickle.load(sec_f)
except Exception as e:
    print(e)
    print("训练集读取失败,程序退出!")
    sys.exit()


def correct_rate(tensor1, tensor2):
    output_convert_list = [torch_max(i, dim=0)[1].item() for i in tensor1]
    label_convert_list = [torch_max(i, dim=0)[1].item() for i in tensor2]
    correct_ls = [i for i, j in zip(output_convert_list, label_convert_list) if i == j]
    if output_convert_list == label_convert_list:
        total_cor_rate = 1
    else:
        total_cor_rate = 0
    # if len(correct_ls) == 5 and correct_ls[2] == 3:
    #     print(correct_ls)
    cor_rate = len(correct_ls) / 5
    return cor_rate, total_cor_rate


def model_train(features, labels, features_test, labels_test):
    # 5个属性,假设每个属性有6个类别
    num_classes_per_attribute = [6] * 5

    # 创建模型
    model = pyramid_resnet18_2d_model(num_classes_per_attribute)
    model.cuda()
    # 损失函数设置
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.cuda()
    # 优化器
    learning_rate = float(input('请输入学习率:'))
    weight_decay = float(input('请输入权衰值:'))
    optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    user_input = input("-0 继续训练模型/ -1 训练新模型:\n")

    model_file_name = str()

    if user_input == '0':
        model_file_name = input("请输入字典模型名字:")
        model.load_state_dict(load(f"{model_file_name}_dict.pth"))
    elif user_input == '1':
        model_file_name = input("请输入新模型名字:") or "new_untitled_model"
    elif user_input != '0' or user_input != '1':
        print("请勿输入其它内容")
        return
    round_num = 1
    iteration_count = 0
    # num_epochs=40#*31才是总的训练次数

    save_interval = 100

    while True:
        print(f"{'-' * 10}第{round_num}轮训练开始{'-' * 10}")
        round_num += 1
        model.train()
        # round_loss_list = list()  # 定义一个统计一轮每批次损失值的列表
        loss_list = list()
        cr_list = list()
        tcr_list = list()
        for input_tensor, label_tensor in zip(features, labels):
            input_tensor = input_tensor.cuda()
            label_tensor = label_tensor.cuda()
            outputs = model(input_tensor)
            """ 计算损失函数 """
            loss_value = loss_function(outputs, label_tensor)
            """ 计算正确率 """
            cr, tcr = correct_rate(outputs, label_tensor)
            loss_list.append(loss_value)  # 统计每批次单次计算的损失值
            cr_list.append(cr)
            tcr_list.append(tcr)
            # round_loss_list.append(loss_value)
            # 设置优化器

            loss_value.backward()

            iteration_count += 1
            if iteration_count % 64 == 0 or iteration_count == len(labels):
                optimizer.step()
                optimizer.zero_grad()
                avg_loss = sum(loss_list) / len(loss_list)
                avg_cr = sum(cr_list) / len(cr_list)
                avg_tcr = sum(tcr_list) / len(tcr_list)
                print("迭代次数:{},单批次平均损失值:{:.4f},准确率:{:.4f},绝对正确率:{:.4f}".
                      format(iteration_count, avg_loss, avg_cr, avg_tcr))
                loss_list.clear()
                cr_list.clear()
                tcr_list.clear()
        # 当损失函数不下降时，改变学习率和权重衰退
        if round_num == save_interval:
            q = input("检测到损失函数没有降低,需要调整参数吗? -0 由程序安排 -1 自己输入 - 2 保持默认\n") or "0"
            if q == "0":
                learning_rate *= 0.1
                weight_decay *= 0.1
                optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif q == "1":
                learning_rate = float(input("请输入学习率:"))
                weight_decay = float(input("请输入权重衰退:"))
                optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            else:
                pass
            qs = input("训练次数已达100轮，请选择:-0 暂停训练并退出保存模型 -1 继续训练:\n")
            if qs == '0':
                save(model, f"{model_file_name}.pth")
                save(model.state_dict(), f"{model_file_name}_dict.pth")
                print("模型已经保存")
                break
            elif qs == '1':
                save_interval += 100
            else:
                print("请不要输入其它内容")
                break
        print('正在验证模型...')
        model.eval()
        cr_list = list()
        tcr_list = list()
        with no_grad():
            for test_input_tensor, test_label_tensor in zip(features_test, labels_test):
                test_input_tensor = test_input_tensor.cuda()
                test_label_tensor = test_label_tensor.cuda()
                outputs = model(test_input_tensor)
                cr, tcr = correct_rate(outputs, test_label_tensor)
                cr_list.append(cr)
                tcr_list.append(tcr)
        print("平均正确率:{:.4f},平均绝对正确率:{:.4f}".format(sum(cr_list) / len(cr_list),
                                                               sum(tcr_list) / len(tcr_list)))
        print('正在保存模型...')
        save(model, f"{model_file_name}.pth")
        save(model.state_dict(), f"{model_file_name}_dict.pth")
        print('保存成功...')
        # model.train()


# model_train()

# 读取模型并定义模型预测
def model_predict(model_dic_name, features, labels):
    # labels_list = open("labels_name.txt", encoding="utf-8").read().split("\n")  # 创建一个标签列表增强可读性

    model_pth = model_dic_name
    # 5个属性,假设每个属性有6个类别
    num_classes_per_attribute = [6] * 5
    # 创建模型
    model = pyramid_resnet18_2d_model(num_classes_per_attribute)
    model = model.cuda()
    model.load_state_dict(load(model_pth))
    model.eval()
    cr_list = list()
    tcr_list = list()
    with no_grad():
        for test_input_tensor, test_label_tensor in zip(features, labels):
            test_input_tensor = test_input_tensor.cuda()
            test_label_tensor = test_label_tensor.cuda()
            outputs = model(test_input_tensor)
            cr, tcr = correct_rate(outputs, test_label_tensor)
            cr_list.append(cr)
            tcr_list.append(tcr)
    print("平均正确率:{:.4f},平均绝对正确率:{:.4f}".format(sum(cr_list) / len(cr_list), sum(tcr_list) / len(tcr_list)))


def model_application(input_tensor, model_dict_path):
    # 5个属性,假设每个属性有6个类别
    num_classes_per_attribute = [6] * 5
    # 创建模型
    model = pyramid_resnet18_2d_model(num_classes_per_attribute)
    model = model.cuda()
    model.load_state_dict(load(model_dict_path))
    model.eval()
    with no_grad():
        inputs = input_tensor
        inputs = inputs.cuda()
        outputs = model(inputs)
    return outputs


if len(face_data) != len(labels_data):
    print('源数据特征与标签数量不匹配')
    sys.exit()
Y_train, X_train, Y_test, X_test = train_test_split(face_data, labels_data)
if len(Y_train) != len(X_train) or len(Y_test) != len(X_test):
    print('数据集分割有问题')
    sys.exit()
print("请选择你要做什么\n")
print("-0 训练模型\n"
      "-1 读取模型并测试\n"
      "-2 应用模型\n"
      "-3 判断人脸相似度")
choice = input("请输入你的选择:")
if choice == '0':

    model_train(features=X_train, labels=Y_train, features_test=X_test, labels_test=Y_test)
elif choice == '1':
    model_name = input("请输入模型名字:")
    model_predict(model_name, X_test, Y_test)
elif choice == '2':
    with open('labels_table.json', encoding='utf-8') as f:
        labels_table = json.load(f)
    ls = list()
    img_path = get_file_path("请选择图片:", "Image Files (*.png *.jpg *.bmp);;All Files (*)")
    print(f"捕获的图片地址:{img_path}")

    model_dict = get_file_path("请选择模型字典:", "Model Files (*.pth);;All Files (*)").split("/")[-1]
    print(f"捕获到的模型字典:{model_dict}")

    face_tensor = img_process(img_path)
    res = model_application(face_tensor, model_dict)
    print(res)
    for i in res:
        ls.append(torch_max(i, dim=0)[1].item())
    print('模型预测的图片属性:')
    for cla, index in enumerate(ls):
        print(labels_table['table'][cla][index])

elif choice == '3':
    pass

else:
    print("请不要输入其它内容")
