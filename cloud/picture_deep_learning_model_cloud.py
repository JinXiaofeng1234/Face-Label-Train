import sys
import random
import pickle
from torch import max as torch_max
from torch import save
from torch import load
from torch import nn
from torch.optim import Adam, SGD
from torch import no_grad
from torch import stack as to_tensor
from torch.utils.data import DataLoader, Dataset
from neural_net_cloud import RestNet18, init_weights
import os


# from sys import exit

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


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


"""
读取人脸数据集和标签数据集
人脸数据集是列表嵌套张量
标签数据集是张量
"""
try:
    with open("face_labels.pkl", "rb") as fir_f:
        full_labels_data = pickle.load(fir_f)
        print(f'标签集数据量:{len(full_labels_data)}')
except Exception as e:
    print(e)
    print("训练标签集读取失败,程序退出!")
    sys.exit()


def optim_choose(model_parameters, lr, wd):
    dialogue = input('请输入你要的优化器 -0 SGD -1 Adam \n')
    while dialogue not in ['0', '1']:
        print('请不要输入其它内容')
        dialogue = input('请输入你要的优化器 -0 SGD -1 Adam \n')
    if dialogue == '0':
        optimizer = SGD(model_parameters, lr=lr, weight_decay=wd)
        return optimizer
    elif dialogue == '1':
        optimizer = Adam(model_parameters, lr=lr, weight_decay=wd)
        return optimizer


def correct_rate(tensor1, tensor2):
    output_convert_list = [torch_max(i, dim=0)[1].item() for i in tensor1]
    label_convert_list = [torch_max(i, dim=0)[1].item() for i in tensor2]
    correct_ls = [i for i, j in zip(output_convert_list, label_convert_list) if i == j]
    if output_convert_list == label_convert_list:
        total_cor_rate = 1
    else:
        total_cor_rate = 0
    # print(output_convert_list)
    # print(label_convert_list)
    # if len(correct_ls) == 5 and correct_ls[2] == 3:
    #     print(correct_ls)
    cor_rate = len(correct_ls) / 4
    return cor_rate, total_cor_rate


def model_train():
    # 创建模型
    model = RestNet18()
    model.apply(init_weights)
    model.cuda()
    # 损失函数设置
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.cuda()
    # 优化器
    learning_rate = float(input('请输入学习率:'))
    weight_decay = float(input('请输入权衰值:'))
    optimizer = optim_choose(model.parameters(), lr=learning_rate, wd=weight_decay)

    model_file_name = str()
    user_input = input("-0 继续训练模型/ -1 训练新模型:\n")
    while user_input not in ['0', '1']:
        print("请勿输入其它内容")
        user_input = input("-0 继续训练模型/ -1 训练新模型:\n")

    if user_input == '0':
        model_file_name = input("请输入字典模型名字:")
        model.load_state_dict(load(f"{os.path.expanduser('~/autodl-tmp/')}{model_file_name}_dict.pth"))

    elif user_input == '1':
        model_file_name = input("请输入新模型名字:") or "new_untitled_model"

    train_file_ls = [i for i in os.listdir('/root/autodl-tmp/') if i.endswith('.pkl')]
    round_num = 0
    save_interval = 20
    batch_size = 128
    while True:
        loss_value_ls = list()
        train_num_ls = list()
        range_cr_ls = list()
        range_tcr_ls = list()
        model.train()
        round_num += 1
        print(f"{'-' * 10}第{round_num}轮训练开始{'-' * 10}")
        for index, file in enumerate(train_file_ls):
            print(f'载入训练数据文件:{file}')
            with open(os.path.expanduser(f'/root/autodl-tmp/{file}'), "rb") as sec_f:
                face_data = pickle.load(sec_f)
            labels_data = full_labels_data[10000 * index:10000 * (index + 1)]
            if len(face_data) != len(labels_data):
                print(f'特征集数量:{len(face_data)},标签集数量:{len(labels_data)}')
                print("数据集划分数量不对等!")
                sys.exit()
            # 随机打乱数据集
            Y_train, X_train, Y_test, X_test = train_test_split(face_data, labels_data)
            # 创建数据集
            date_set = MyDataset(X_train, Y_train)
            train_num_ls.append(len(date_set))
            dataloader = DataLoader(date_set, batch_size=batch_size, shuffle=True)
            for input_tensor, label_tensor in dataloader:
                input_tensor = input_tensor.squeeze(1).cuda()
                label_tensor = label_tensor.cuda()
                optimizer.zero_grad()
                outputs = model(input_tensor)

                """ 计算损失函数 """
                loss_value = loss_function(outputs, label_tensor)
                """ 计算正确率 """
                # cr, tcr = correct_rate(outputs, label_tensor)
                # 设置优化器
                loss_value.backward()
                optimizer.step()
                loss_value_ls.append(loss_value)
            print(f"单批次平均损失值:{sum(loss_value_ls) / len(date_set):.4f}")
            print('正在验证模型...')
            model.eval()
            cr_list = list()
            tcr_list = list()
            with no_grad():
                for test_input_tensor, test_label_tensor in zip(X_test, Y_test):
                    test_input_tensor = test_input_tensor.cuda()
                    test_label_tensor = test_label_tensor.cuda()
                    outputs = model(test_input_tensor)
                    cr, tcr = correct_rate(outputs, test_label_tensor)
                    cr_list.append(cr)
                    tcr_list.append(tcr)
            avg_cr = sum(cr_list) / len(cr_list)
            avg_tcr = sum(tcr_list) / len(tcr_list)

            range_cr_ls.append(avg_cr)
            range_tcr_ls.append(avg_tcr)
            print("单批次平均正确率:{:.4f},单批次平均绝对正确率:{:.4f}".format(avg_cr, avg_tcr))
            del Y_train, X_train, Y_test, X_test
            model.train()
        print(f"单轮平均损失值:{sum(loss_value_ls) / sum(train_num_ls):.4f}",
              f"单轮平均准确率:{sum(range_cr_ls) / len(range_cr_ls):.4f}",
              f"单轮平均绝对准确率:{sum(range_tcr_ls) / len(range_tcr_ls):.4f}")
        print('正在保存模型...')
        save(model, f"{os.path.expanduser('~/autodl-tmp/')}{model_file_name}.pth")
        save(model.state_dict(), f"{os.path.expanduser('~/autodl-tmp/')}{model_file_name}_dict.pth")
        print('保存成功...')
        # 当损失函数不下降时，改变学习率和权重衰退
        if round_num == save_interval:
            qs = input("已达到间隔轮数,请选择:-0 暂停训练并退出保存模型 -1 继续训练:\n")
            if qs == '0':
                save(model, f"{model_file_name}.pth")
                save(model.state_dict(), f"{model_file_name}_dict.pth")
                print("模型已经保存")
                break
            elif qs == '1':
                save_interval += 20
            else:
                print("请不要输入其它内容")
                break
            q = input("需要调整参数吗? -0 由程序安排 -1 自己输入 - 2 保持默认\n") or "0"
            if q == "0":
                learning_rate *= 0.1
                weight_decay *= 0.1
                optimizer = optim_choose(model.parameters(), lr=learning_rate, wd=weight_decay)
            elif q == "1":
                learning_rate = float(input("请输入学习率:"))
                weight_decay = float(input("请输入权重衰退:"))
                batch_size = int(input("请输入新的批次数"))
                optimizer = optim_choose(model.parameters(), lr=learning_rate, wd=weight_decay)
            else:
                pass


model_train()
