'''
功能：训练一个车牌字符识别模型
备注：训练出来的模型文件为 Model.pth 可加载这个模型使用
    目前的准确率在 89% 以后可以选择较好的模型提高准确率
时间：2021-4-12
'''
import torch
# import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import os
import time
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import math
import cv2

LR = 0.00004     # 设置学习率
EPOCH_NUM = 30


def time_since(since):
    s = time.time() - since
    m = math.floor(s/60)
    s -= m*60
    return '%dm %ds' % (m, s)


# 初始化根目录
train_path  = 'charData\\mydata\\train\\'
test_path   = 'charData\\mydata\\test\\'



# 定义读取文件的格式
# def default_loader(path):
#     return Image.open(path).convert('RGB')

# 数据集类
class MyDataSet(Dataset):
    def __init__(self, data_path:str, transform=None):  # 传入训练样本路径
        super(MyDataSet, self).__init__()
        self.data_path = data_path
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(32, 40)), # 原本就是 32x40 不需要修改尺寸
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transform = transform
        self.path_list = os.listdir(data_path)

    def __getitem__(self, idx:int):
        img_path = self.path_list[idx]
        label = int(img_path.split('.')[1])
        label = torch.as_tensor(label, dtype=torch.int64)
        img_path = os.path.join(self.data_path, img_path)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label

    def __len__(self)->int:
        return len(self.path_list)


train_ds = MyDataSet(train_path)
test_data = MyDataSet(test_path)
# for i, item in enumerate(tqdm(train_ds)):
#     print(item)
#     break

# 数据加载
new_train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True, num_workers=0)
new_test_loader = DataLoader(test_data, batch_size=32, shuffle=False, pin_memory=True, num_workers=0)

# for i, item in enumerate(new_train_loader):
#     print(item[0].shape)
#     break
#
# img_PIL_Tensor = train_ds[1][0]
# new_img_PIL = transforms.ToPILImage()(img_PIL_Tensor).convert('RGB')
# plt.imshow(new_img_PIL)
# plt.show()


if 0:
    # 设置训练类
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.maxpooling = torch.nn.MaxPool2d(2)
            self.avgpool = torch.nn.AvgPool2d(2)
            self.globalavgpool = torch.nn.AvgPool2d((8, 10))
            self.bn1 = torch.nn.BatchNorm2d(64)
            self.bn2 = torch.nn.BatchNorm2d(128)
            self.bn3 = torch.nn.BatchNorm2d(256)
            self.dropout50 = torch.nn.Dropout(0.5)
            self.dropout10 = torch.nn.Dropout(0.1)

            self.fc1 = torch.nn.Linear(256, 40)

        def forward(self, x):
            batch_size = x.size(0)
            x = self.bn1(F.relu(self.conv1(x)))
            x = self.bn1(F.relu(self.conv2(x)))
            x = self.maxpooling(x)
            x = self.dropout10(x)
            x = self.bn2(F.relu(self.conv3(x)))
            x = self.bn2(F.relu(self.conv4(x)))
            x = self.maxpooling(x)
            x = self.dropout10(x)
            x = self.bn3(F.relu(self.conv5(x)))
            x = self.bn3(F.relu(self.conv6(x)))
            x = self.globalavgpool(x)
            x = self.dropout50(x)

            x = x.view(batch_size, -1)

            x = self.fc1(x)
            return x


    model = Net()
else:
    # 方法一：增减卷积 要修改网络中的层次结构，这时只能用参数覆盖的方法，即自己先定义一个类似的网络，再将预训练中的参数提取到自己的网络中来
    class Net(torch.nn.Module):
        def __init__(self, num_classes=40):
            super(Net, self).__init__()

            # pretrained=True 加载网络结构和预训练参数，
            # pretrained=False 时代表只加载网络结构，不加载预训练参数，即不需要用预训练模型的参数来初始化
            # pretrained 参数默认是False,为了代码清晰，最好还是加上参数赋值
            net = models.vgg16(pretrained=True)
            # net = models.mnist(pretrained=True)
            net.classifier = torch.nn.Sequential()    # 将分类层（fc）置空
            self.features = net
            self.classifier = torch.nn.Sequential(    # 定义一个卷积网络结构
                torch.nn.Linear(512*7*7, 512),
                torch.nn.ReLU(True),
                torch.nn.Dropout(),
                torch.nn.Linear(512, 128),
                torch.nn.ReLU(True),
                torch.nn.Dropout(),
                torch.nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def train(epoch, loss_list):
    running_loss = 0.0
    for batch_idx, data in enumerate(new_train_loader, 0):
        inputs, target = data[0], data[1]
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        # print(outputs.shape, target.shape)
        # print(outputs, target)
        # break
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        running_loss += loss.item()
        if batch_idx % 10 == 9:
            print(f'[{time_since(start)}] Epoch {epoch}', end='')
            print('[%d, %5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 10))
            running_loss = 0.0

    return loss_list

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for _, data in enumerate(new_test_loader, 0):
            inputs, target = data[0], data[1]
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, prediction = torch.max(outputs.data, dim=1)

            total += target.size(0)
            correct += (prediction == target).sum().item()
    print('Accuracy on test set: (%d/%d)%d %%' % (correct, total, 100 * correct / total))


if __name__ == '__main__':
    start = time.time()
    loss_list = []
    for epoch in range(EPOCH_NUM):
        train(epoch, loss_list)
        test()
    torch.save(model.state_dict(), "Model.pth")     # 训练完成 保存训练好的模型

    x_ori = []
    for i in range(len(loss_list)):
        x_ori.append(i)
    plt.title("Graph")
    plt.plot(x_ori, loss_list)
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.show()
