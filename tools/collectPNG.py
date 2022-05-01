'''
功能：将生成的 0_json 等文件夹下的 label.png 统一放到一个文件夹下
时间：2021-11-29
'''
import os
import shutil

JSON_folder = r"G:\数据集\自己收集的车牌图片2\json转化为png"          # 生成的 0_json 等文件夹所在的文件夹
Paste_label_folder = r"G:\数据集\自己收集的车牌图片2\集合label"  # lable.png 统一存放的文件夹
#  获取文件夹内的文件名
FileNameList = os.listdir(JSON_folder)
NewFileName = 1
for i in range(len(FileNameList)):
    #  复制label文件
    jpg_file_name = FileNameList[i].split(".", 1)[0]
    print(jpg_file_name)
    label_file = JSON_folder + "\\" + jpg_file_name + "\\label.png"
    NewFileName = jpg_file_name.split("_", 1)[0]
    print(NewFileName)
    new_label_file = Paste_label_folder + "\\" + str(NewFileName) + ".png"
    shutil.copyfile(label_file, new_label_file)