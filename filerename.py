'''
功能：将文件批量重命名
备注：处理数据需要重命名
时间：2021-4-12
'''

import os
import cv2



ORIGIN_IMG_PATH = 'D:\\DeapLearn Project\\DeepLab V3 Plate Recognition\\sincharpic\\resizepic\\z\\train\\'

#批量重命名文件 数字和子母
def RenameFileShuzizimu():
    index_no = 0
    for file in os.listdir(ORIGIN_IMG_PATH):
        imagepath = os.path.join(ORIGIN_IMG_PATH, file)
        newname = 'jpg.33.' + str(index_no) + '.jpg'
        newimagename = os.path.join(ORIGIN_IMG_PATH, newname)
        os.rename(imagepath, newimagename)
        print("以前的文件名和对应现在的名字", file + '---' + str(newname))
        index_no += 1

#批量重命名文件 省份
def RenameFileShengfen():
    index_no = 0
    for file in os.listdir(ORIGIN_IMG_PATH):
        imagepath = os.path.join(ORIGIN_IMG_PATH, file)
        label = int(imagepath.split('.')[1])
        # label = imagepath.split('.')[0]
        newname = 'jpg.'+ str(label-33) + '.' + str(index_no) + '.jpg'
        newimagename = os.path.join(ORIGIN_IMG_PATH, newname)
        os.rename(imagepath, newimagename)
        # print("以前的文件名和对应现在的名字", file + '---' + str(newname))
        index_no += 1


if __name__ == '__main__':
    RenameFileShengfen()
    # load_dataset('D:\\DeapLearn Project\\Face_Recognition\\moreface\\7219face\\test\\originface\\')