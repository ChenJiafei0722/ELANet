# converting trainIDs to labelIDs for evaluating the test set segmenatation results of the cityscapes dataset

import numpy as np
import os
from PIL import Image

# index: trainId from 0 to 18, 19 semantic class   val: labelIDs
cityscapes_trainIds2labelIds = np.array([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
                                        dtype=np.uint8)

# 这个函数有两个参数：trainID_png_dir和save_dir
# trainID_png_dir是包含以trainID格式存储的预测分割掩膜的目录。
# save_dir是转换后以labelID格式存储的分割掩膜所保存的目录。
# 函数首先检查save_dir是否存在。如果不存在，函数将使用os.makedirs()函数创建该目录。
# 接下来，函数使用os.listdir()函数获取trainID_png_dir中所有文件的列表。
# 该列表包含以trainID格式存储的所有预测分割掩膜的文件名。
def trainIDs2LabelID(trainID_png_dir, save_dir):
    print('save_dir:  ', save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    png_list = os.listdir(trainID_png_dir)
    for index, png_filename in enumerate(png_list):
        # 使用 os.path.join() 函数将 trainID_png_dir 和 png_filename 字符串拼接在一起，创建一个字符串对象 png_path。
        png_path = os.path.join(trainID_png_dir, png_filename)
        # print(png_path)
        print('processing(', index, '/', len(png_list), ') ....')
        image = Image.open(png_path)  # image is a PIL #image
        pngdata = np.array(image)
        trainID = pngdata  # model prediction
        row, col = pngdata.shape
        labelID = np.zeros((row, col), dtype=np.uint8)
        for i in range(row):
            for j in range(col):
                labelID[i][j] = cityscapes_trainIds2labelIds[trainID[i][j]]

        res_path = os.path.join(save_dir, png_filename)
        new_im = Image.fromarray(labelID)
        new_im.save(res_path)


if __name__ == '__main__':
    trainID_png_dir = 'D:/ELANet-master/result/cityscapes/predict/ELANet'
    save_dir = './result/cityscapes_submit/'
    trainIDs2LabelID(trainID_png_dir, save_dir)
