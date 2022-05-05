# 2022-5-2
# 首先二值化车牌
# 将抠出的车牌中的单个字符分割开来
# 并将单个字符二值图图存起来

import cv2
import numpy as np
from PIL import Image

root_path = 'D:\\DeapLearn Project\\DeepLab V3 Plate Recognition\\' #根目录，自己的工程需要自己配置，即工程所在目录
plate_path = 'plate.jpg'

def ReadPlate():
    img = cv2.imread(plate_path)

    # 稍微裁剪车牌图片
    print(img.shape)
    img_width = img.shape[1]
    img_height = img.shape[0]

    del_width = img_width//55
    del_height = img_height//8
    print('del_w = ', del_width)
    print('del_h = ', del_height)

    after_del_x = del_width
    after_del_y = del_height

    after_del_width = img_width - 2*del_width
    after_del_height = img_height - 2*del_height

    print('after_del_width = ', after_del_width)
    print('after_del_height = ', after_del_height)

    plate_after_del = img[after_del_y:after_del_y+after_del_height, after_del_x:after_del_x+after_del_width]

    plate_img = cv2.cvtColor(plate_after_del, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
    _, plate_img = cv2.threshold(plate_img, 125, 255, cv2.THRESH_BINARY)  # 二值化
    # _, plate_img = cv2.threshold(plate_img, 180, 255, cv2.THRESH_BINARY)  # 二值化
    # cv2.imshow('palat1', plate_img)  # 打印出被抠出来的车牌

    # # 腐蚀和膨胀
    # kernel_X = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义矩形卷积核
    # mark = cv2.dilate(plate_img, kernel_X, (-1, -1), iterations=1)  # 膨胀操作
    # mark = cv2.erode(mark, kernel_X, (-1, -1), iterations=1)  # 腐蚀操作

    kernel_X = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # 定义矩形卷积核
    mark = cv2.dilate(plate_img, kernel_X, (-1, -1), iterations=1)  # 膨胀操作
    # cv2.imshow('erode', mark)

    kernel_Y = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))  # 定义矩形卷积核
    mark = cv2.dilate(mark, kernel_Y, (-1, -1), iterations=1)  # 膨胀操作
    cv2.imshow('palat2', mark)  # 打印出被抠出来的车牌

    # 阴影法分割每个字符，首先统计x轴上的投影
    pix_list = []
    for i in range(after_del_width):
        num_pix = 0
        for j in range(after_del_height):
            if mark[j][i] > 0:
                num_pix += 1
        pix_list.append(num_pix)

        # print('num_pix = ', num_pix)

    # 算出开始有字符的头和尾
    start_char_index = 0
    end_char_index = 0
    for i in range(after_del_width):
        if pix_list[i] > 3:
            start_char_index = i
            break
    for i in range(after_del_width-1, -1, -1):
        if pix_list[i] > 3:
            end_char_index = i
            break
    print('start_char_index:', start_char_index)
    print('end_char_index:', end_char_index)

    all_char_lenth = end_char_index - start_char_index + 1
    top2width = (end_char_index-start_char_index+1)*102/409 # 前2个字符宽度
    end5width = (end_char_index-start_char_index+1)*273/409 # 后5个字符宽度
    top2width = int(top2width+0.5)  #四舍五入
    end5width = int(end5width+0.5)
    # top2width = int(top2width)
    # end5width = int(end5width)
    print('end5width:', end5width)
    print('top2width:', top2width)

    # 计算每一个字符起始点和终点
    # 前两个字符
    char_1_start_index = start_char_index
    char_2_start_index = start_char_index + int(all_char_lenth*(45+12)/409+0.5)

    # 后五个字符
    char_7_start_index = end_char_index - int(all_char_lenth * (45 ) / 409 + 0.5) + 1
    print('char_7_start_index = ', char_7_start_index)
    char_6_start_index = end_char_index - int(all_char_lenth * (45 + 12 + 45) / 409 + 0.5) + 1
    print('char_6_start_index = ', char_6_start_index)
    char_5_start_index = end_char_index - int(all_char_lenth * (45 + 12 + 45 + 12 + 45) / 409 + 0.5) + 1
    print('char_5_start_index = ', char_5_start_index)
    char_4_start_index = end_char_index - int(all_char_lenth * (45 + 12 + 45 + 12 + 45 + 12 + 45) / 409 + 0.5) + 1
    print('char_4_start_index = ', char_4_start_index)
    char_3_start_index = end_char_index - int(all_char_lenth * (45 + 12 + 45 + 12 + 45 + 12 + 45 + 12 + 45) / 409 + 0.5) + 1
    print('char_3_start_index = ', char_3_start_index)

    single_char_width = int(all_char_lenth * 45 / 409 + 0.5)
    print('single_char_width = ', single_char_width)


    # 开始分割字符
    char_1_img = mark[:, char_1_start_index:char_1_start_index + single_char_width + 2]
    cv2.imshow('charpic1', char_1_img)  # 打印出被抠出来的车牌单个字符
    cv2.imwrite(f'{root_path}\\sincharpic\\{1}.jpg',char_1_img)

    char_2_img = mark[:, char_2_start_index - 2:char_2_start_index + single_char_width + 2]
    cv2.imshow('charpic2', char_2_img)  # 打印出被抠出来的车牌单个字符
    cv2.imwrite(f'{root_path}\\sincharpic\\{2}.jpg', char_2_img)

    char_3_img = mark[:, char_3_start_index - 2:char_3_start_index + single_char_width + 2]
    cv2.imshow('charpic3', char_3_img)  # 打印出被抠出来的车牌单个字符
    cv2.imwrite(f'{root_path}\\sincharpic\\{3}.jpg', char_3_img)

    char_4_img = mark[:, char_4_start_index - 2:char_4_start_index + single_char_width + 2]
    cv2.imshow('charpic4', char_4_img)  # 打印出被抠出来的车牌单个字符
    cv2.imwrite(f'{root_path}\\sincharpic\\{4}.jpg', char_4_img)

    char_5_img = mark[:, char_5_start_index - 2:char_5_start_index + single_char_width + 2]
    cv2.imshow('charpic5', char_5_img)  # 打印出被抠出来的车牌单个字符
    cv2.imwrite(f'{root_path}\\sincharpic\\{5}.jpg', char_5_img)

    char_6_img = mark[:, char_6_start_index - 2:char_6_start_index + single_char_width + 2]
    cv2.imshow('charpic6', char_6_img)  # 打印出被抠出来的车牌单个字符
    cv2.imwrite(f'{root_path}\\sincharpic\\{6}.jpg', char_6_img)

    char_7_img = mark[:, char_7_start_index - 2:char_7_start_index + single_char_width]
    cv2.imshow('charpic7', char_7_img)  # 打印出被抠出来的车牌单个字符
    cv2.imwrite(f'{root_path}\\sincharpic\\{7}.jpg', char_7_img)


    cv2.waitKey(0)
if __name__ == '__main__':
    ReadPlate()