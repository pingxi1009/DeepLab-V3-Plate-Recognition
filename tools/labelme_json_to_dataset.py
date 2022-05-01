'''
功能：将批量 json 格式标签为 png 格式
时间：2021-11-29
'''
import os

json_folder = r"G:\数据集\自己收集的车牌图片2\生成的json" # 生成的 json 文件所在的文件夹
#  获取文件夹内的文件名
FileNameList = os.listdir(json_folder)
#  激活labelme环境
os.system("activate labelme")
for i in range(len(FileNameList)):
    #  判断当前文件是否为json文件
    if(os.path.splitext(FileNameList[i])[1] == ".json"):
        json_file = json_folder + "\\" + FileNameList[i]
        #  将该json文件转为png
        os.system("labelme_json_to_dataset " + json_file)