'''
功能：测试训练出的模型的准确率，直接识别分割出的车牌字符
备注：此处是直接加载的生成的模型文件 Model.pth
时间：2021-4-12
'''
import torch
from trainsinpic import SingleCharNet1
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

# 差点翻车，车牌训练的数据集里面没有字母 ‘I’ 和 'O'
SINGLE_CHAR_LIST = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A',
                    'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L',
                    'M', 'N', 'P', 'Q', 'R' ,'S', 'T', 'U', 'V', 'W',
                    'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵', '桂', '黑',
                    '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁',
                    '青', '琼', '陕', '苏', '晋', '皖', '湘', '新', '豫', '渝',
                    '粤', '云', '藏', '浙']

# 数据集类
class MyDataSet(Dataset):
    def __init__(self, data_path:str, transform=None):  # 传入训练样本路径
        super(MyDataSet, self).__init__()
        self.data_path = data_path
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)), # 原本就是 224x224 不需要修改尺寸
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
        label = int(img_path.split('.')[0])
        label = torch.as_tensor(label, dtype=torch.int64)
        img_path = os.path.join(self.data_path, img_path)
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label

    def __len__(self)->int:
        return len(self.path_list)

# test_path = 'D:\\DeapLearn Project\\Face_Recognition\\data\\test\\'
test_path = 'sincharpic\\resizepic\\'

test_data = MyDataSet(test_path)
new_test_loader = DataLoader(test_data, batch_size=32, shuffle=False, pin_memory=True, num_workers=0)


# model = SingleCharNet1(num_classes=65)
# model.load_state_dict(torch.load('SingleCharModel.pth'))
# 加载预训练模型的参数

model1 = SingleCharNet1(num_classes=31)  # 创造一个一模一样的识别省份模型
model1.load_state_dict(torch.load('SingleCharShengModel.pth'))

model2 = SingleCharNet1(num_classes=34)  # 创造一个一模一样的识别字符模型
model2.load_state_dict(torch.load('SingleCharZifuModel.pth'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1.to(device)
model2.to(device)

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for _, data in enumerate(new_test_loader, 0):
            inputs, _ = data[0], data[1]
            inputs = inputs.to(device)
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            # print(outputs.shape)
            _, prediction1 = torch.max(outputs1.data, dim=1)
            _, prediction2 = torch.max(outputs2.data, dim=1)
            print('-'*40)
            # print(target)
            # print(prediction)
            print(f'Predicted license plate number:'
                  f'{SINGLE_CHAR_LIST[prediction1[0]+34]}'
                  f'{SINGLE_CHAR_LIST[prediction2[1]]}'
                  f'{SINGLE_CHAR_LIST[prediction2[2]]}'
                  f'{SINGLE_CHAR_LIST[prediction2[3]]}'
                  f'{SINGLE_CHAR_LIST[prediction2[4]]}'
                  f'{SINGLE_CHAR_LIST[prediction2[5]]}'
                  f'{SINGLE_CHAR_LIST[prediction2[6]]}')
            # total += target.size(0)
            # correct += (prediction == target).sum().item()
            # print(target.shape)

    # print('Accuracy on test set: (%d/%d)%d %%' % (correct, total, 100 * correct / total))

if __name__ == '__main__':
    test()