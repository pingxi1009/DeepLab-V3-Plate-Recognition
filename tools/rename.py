'''
功能：批量化修改文件名称
时间：2021-11-29
'''
import os,shutil

file_path = "G:/数据集/自己收集的车牌图片2/第一次收集的/label"    # 文件所在的文件夹
# 改变工作文件夹
os.chdir(file_path)

# 遍历该文件夹所有的文件，并for循环
index = 201
for name in os.listdir(file_path):
    print(name)

    # 修改文件名
    if(name.split(".")[1] == "png"):
        # print(name)
        new_name = str(index) + ".png"
        print(new_name)
        index += 1

        # 文件名加上文件夹构成绝对路径
        before_file = os.path.join(file_path, name)
        after_file = os.path.join(file_path, new_name)

        print('rename "%s" to "%s"......'%(before_file,after_file))
        # 利用shutil.move将文件移动到原来位置(重命名的效果)
        shutil.move(before_file, after_file)