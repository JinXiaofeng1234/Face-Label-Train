from cv2 import imread
import torchvision.transforms as transforms


def img_process(img_path):
    img = imread(img_path)  # 读取人脸CV2

    # 转换成张量前的图片预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x / 255.0),

    ])
    # 转换为pytorch张量
    tensor_img = transform(img).unsqueeze(0)

    return tensor_img


