from skimage import io                                  # scikit-image是基于scipy的一款图像处理包，它将图片作为numpy数组进行处理，正好与matlab一样,用来图片输入输出操作
import os                                               # 处理文件和目录
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split    # 决定划分训练集测试集比例
from PIL import Image                                   # PIL (Python Image Library) 是 Python 平台处理图片的事实标准,兼具强大的功能和简洁的 API

product_image_dir = r'./product_image/'
path_list = os.listdir(product_image_dir)               # 对产品图片按名字排序
path_list.sort(key=lambda x: str(x[:-4]))
# for k in range(0,len(path_list)):
#     print(path_list[k])

test_dir = r'test/'

# 切割图片，并保存到./test_dir/文件夹下
def cropImage(path):
    assert os.path.exists(path),print( path + " ：待测试图片文件夹不存在")   # 判断待测试图片文件夹是否存在
    assert os.path.exists(test_dir), os.makedirs(test_dir)                # 判断切割图片存储的文件夹是否存在，不存在则创建
    path_list = os.listdir(path)
    # 按顺序读取图片并切割
    path_list.sort(key=lambda x:str(x[:-4]))
    i = 0
    for imageName in path_list:
        print(imageName)
        if (imageName == ".DS_Store"):      # 删除mac自动生成的.DS_Store
            os.remove(imageName)
        im = Image.open(product_image_dir + imageName)
        pro1 = (420, 0, 920, 2048)          # 设置图像裁剪区域
        image1 = im.crop(pro1)              # 图像裁剪
        image1.save(test_dir +str(i) + '-1.jpg')
        pro2 = (980, 0, 1480, 2048)
        image2 = im.crop(pro2)
        image2.save(test_dir +str(i) + '-2.jpg')
        pro3 = (1560, 0, 2060, 2048)
        image3 = im.crop(pro3)
        image3.save(test_dir +str(i) + '-3.jpg')
        i = i + 1

cropImage(product_image_dir)

data_x = []     # 定义数据集列表,图片
data_y = []     # 定义数据集列表，图片标签

dir_list_1 = os.listdir('./cut_image/negative/')        # 列出文件夹下所有的目录与文件，返回一个由文件名和目录名组成的列表，
for name in dir_list_1:
    if (name == "DS_Store"):                            # 删除mac自动生成的.DS_Store
        os.remove(name)
    path = './cut_image/negative/' + name               # 循环
    img = io.imread(path)                               # 从文件读取图象
    img = np.reshape(img, (-1))                         # 创建一个改变了尺寸的新数组，原数组的shape保持不变，变成只有一行数据的矩阵
    # print(img)                                        # 打印查看图片变成数组
    data_x.append(img)                                  # 将数据添加入列表
    data_y.append(-1)                                   # 将数据添加入列表

dir_list_2 = os.listdir('./cut_image/positive/')        # 列出文件夹下所有的目录与文件，不好的图片
for name in dir_list_2:                                 # 同理也是读取图片变成一维的列表
    path = './cut_image/positive/' + name
    img = io.imread(path)
    img = np.reshape(img, (-1))
    # print(img)
    data_x.append(img)
    data_y.append(1)

data_x = np.array(data_x)                  # 将所有列表数据集合变成矩阵数据
data_y = np.array(data_y)                  # 同上
# print(data_x)

# 测试集图片处理
test_x = []                                # 定义数据集列表,图片
dir_list_test = os.listdir('test/')        # 按顺序列出文件夹下所有的目录与文件
def comp(x):
    dir_list_test = x[0:x.find('.')].split('-')
    return (int(dir_list_test[0]), int(dir_list_test[1]))
dir_list_test = sorted(dir_list_test, key=comp)
for name in dir_list_test:
    # print(name)
    if (name == ".DS_Store"):               # 删除mac自动生成的.DS_Store
        os.remove(name)
    path = 'test/' + name
    img = io.imread(path)
    img = np.reshape(img, (-1))
    # print(img)
    test_x.append(img)
test_x = np.array(test_x)                   # 将所有列表数据集合变成矩阵数据


x_train,y_train = data_x, data_y
x_test = test_x
clf = RandomForestClassifier(n_estimators=40,max_depth=8)    # 利用随机森林分类器模型对数据进行分类
clf.fit(x_train, y_train)                                    # 用训练数据拟合分类器模型开始训练
predict = clf.predict(x_test)                                # 返回预测标签
# print(predict,'\n',dir_list_test)

# 写入txt文档
def save_lst_to_file(predict,file_path,path_list):
    with open(file_path, 'w') as f:
        i = 0
        leng = (len(dir_list_test))/3
        # print(leng)
        j=0
        for i in range(0,int(leng)):
            # print(i)
            content = []
            # content.append(str(i)+" ")
            im = cv2.imread(r'./product_image/'+path_list[i])
            # im = cv2.imread(r'./test/'+dir_list_test[i])
            # print(im)
            if ((predict[j] == 1) & (predict[j+1] == 1) & (predict[j+2] == 1)):
                content.append("合格")
            else:content.append("不合格 ")
            if (predict[j] == -1):
                content.append("1")

                cv2.rectangle(im, (460, 650), (920, 2000),(255,0,0),3)  # 画矩形框
                # cv2.imshow("im", im)
                cv2.waitKey(0)
                cv2.imwrite(path_list[i],im)  # 存储结果图像

            if (predict[j+1] == -1):
                if(predict[j] == -1):
                    content.append(",2")
                else:
                    content.append("2")
                cv2.rectangle(im, (980, 650), (1480, 2000),(255,0,0),3)  # 画矩形框
                cv2.imwrite(path_list[i],im)  # 存储结果图像

            if (predict[j+2] == -1):
                if ((predict[j] == 1)&(predict[j+1] == 1)):
                    content.append("3")
                else:
                    content.append(",3")
                cv2.rectangle(im, (1560, 650), (2060, 2000),(255,0,0),3)  # 画矩形框
                cv2.imwrite(path_list[i],im)  # 存储结果图像
            content.append("\n")
            j=j+3
            f.writelines(content)
    print("预测结果写入 product_prediction.txt 文档成功！\n")
    print("预测图片载入当前目录成功！\n")
save_lst_to_file(predict,'./product_prediction.txt',path_list)
