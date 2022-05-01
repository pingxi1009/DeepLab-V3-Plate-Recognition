#2022-5-1
#绘制输出面积大于阈值的连通域实心轮廓
#用于寻找车牌轮廓

import cv2
import numpy as np
from PIL import Image

img = cv2.imread('palat.jpg', 0)

def Clear_Micor_Areas(img):
    print(img.shape)
    conyours, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("count = ",len(conyours))
    # for index in range(len(conyours)):
    #     area = cv2.contourArea(conyours[index])
    #     print(area)

    kernel_X = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))  # 定义矩形卷积核
    mark = cv2.dilate(img, kernel_X, (-1, -1), iterations=1)  # 膨胀操作
    # cv2.imshow('erode', mark)

    kernel_Y = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))  # 定义矩形卷积核
    mark = cv2.dilate(mark, kernel_Y, (-1, -1), iterations=1)  # 膨胀操作
    # cv2.imshow('erode', mark)

    conyours, h = cv2.findContours(mark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("count = ", len(conyours))
    maxAreaRect = 0
    rect = cv2.boundingRect(conyours[0])
    for jndex in range(len(conyours)):
        area = cv2.contourArea(conyours[jndex])
        rect = cv2.boundingRect(conyours[jndex])
        print(rect)
        curRectArea = rect[2]*rect[3]
        maxAreaRect = maxAreaRect if maxAreaRect > curRectArea else curRectArea

    print("maxarea = ",maxAreaRect)

    cv2.waitKey(0)
    return rect

def GetBigstArea():
    # 轮廓检测时，对象必须是白色的，背景是黑色的。
    #img = cv2.imread('D:\\Git Store\\Transfer_Learning_2021-11-28\\hu.bmp', 0)
    # img = cv2.imread('palat.jpg', 0)
    print(np.shape(img))
    thres_label = 300
    new_label = np.zeros(img.shape,np.uint8)
    contours,_ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #_, contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n=len(contours)  #轮廓的个数
    print('n = ',n)
    for i in range(n):
        temp = np.zeros(img.shape,np.uint8)
        temp[img == i] = i
        for num,values in enumerate(contours):
            if cv2.contourArea(values) < thres_label:
                # cv2.drawContours(temp,contours,num,0,thickness=-1)
                cv2.drawContours(temp, contours, 0, 0, thickness=-1)
        new_label += temp
    img1 = new_label
    cv2.imshow("yes", img1)
    cv2.imwrite('mianjijiance06.jpg',img1)

    cv2.waitKey(0)

    return img1

def GetBigArea2():
    # img = cv2.imread('21biyunsuan14.bmp', 0)
    print(np.shape(img))
    Laplacian = cv2.Laplacian(img, cv2.CV_64F)
    Laplacian = cv2.convertScaleAbs(Laplacian)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(Laplacian)
    print(centroids)
    print("123")
    print("stats = ", stats)
    i = 0
    for istat in stats:
        if istat[4] < 4830:
            # print(i)
            print(istat[0:2])
            if istat[3] > istat[4]:
                r = istat[3]
            else:
                r = istat[4]
            cv2.rectangle(Laplacian, tuple(istat[0:2]), tuple(istat[0:2] + istat[2:4]), (0, 0, 255), thickness=-1)
        i = i + 1

    cv2.imshow("yes", Laplacian)
    cv2.imwrite('mianjijiance06.jpg', Laplacian)

    cv2.imwrite('105.jpg', Laplacian)

    cv2.waitKey(0)

if __name__ == '__main__':
    # GetBigstArea()
    # GetBigArea2()
    Clear_Micor_Areas(img)

