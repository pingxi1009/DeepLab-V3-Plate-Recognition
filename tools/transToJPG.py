'''
功能：批量化将文件后缀修改为 jpg
时间：2021-11-29
'''
import os,shutil

file_path = "G:/数据集/自己收集的车牌图片2"
# 改变工作文件夹
os.chdir(file_path)

# 遍历该文件夹所有的文件，并for循环
index = 0
for name in os.listdir(file_path):
    print(name)

    # 修改文件名
    if(name.split(".")[1] == "jpeg"):
        # print(name)
        new_name = str(index) + ".jpg"
        print(new_name)
        index += 1

        # 文件名加上文件夹构成绝对路径
        before_file = os.path.join(file_path, name)
        after_file = os.path.join(file_path, new_name)

        print('rename "%s" to "%s"......'%(before_file,after_file))
        # 利用shutil.move将文件移动到原来位置(重命名的效果)
        shutil.move(before_file, after_file)