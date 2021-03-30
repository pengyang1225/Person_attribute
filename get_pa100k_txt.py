import os
import random
import cv2
import numpy as np


name_dict = {
                "gender":['男', '女'], #0
                "age":["老年", "中年", "少年"],# 1 ,2 ,3
                "orientation":['正面', '侧面', '背面'],#4,5,6
                "hat":["不戴帽子", "戴帽子"],#7
                "glasses":["不戴眼镜", "戴眼镜"],#8
                "handBag":["没有", "有"],#9
                "shoulderBag":["没有", "有"], #10
                "backBag":["没有", "有"],#11
                "upClothing":["短袖", "长袖"],# 13,14
                "downClothing":["长裤", "短裤", "裙子"]#22,23,24
            }


index_list =[0,1,2,3,4,5,6,7,8,9,10,11,13,14,22,23,24]
one_labe_size=[1,3,3,1,1,1,1,1,2,3]
source_name_txt="/home/py/code/JDAI/fast-reid/datasets/PA-100K/test_images_name.txt"
source_label_txt="/home/py/code/JDAI/fast-reid/datasets/PA-100K/test_label.txt"
result_txt="/home/py/code/JDAI/fast-reid/datasets/PA-100K/annotation/test.txt"
file = open(result_txt, 'w')
with open(os.path.join(source_name_txt)) as f:
    imgs = f.readlines()
imgs = [img.rstrip("\n") for img in imgs]


with open(os.path.join(source_label_txt)) as f:
    labels = f.readlines()
labels = [label.rstrip("\n")for label in labels]
all_result=[]
for i,label in enumerate (labels):

    list_label=label.split(",")

    tmp_index = 0
    ok_index = []

    for index in index_list:
        ok_index.append(label.split(",")[index])
    print("label14", ok_index[13])
    print("label15", ok_index[14])
    obj_list =[]
    obj_list.append(imgs[i])
    for size in one_labe_size:
        if size !=1:
            adsasd =ok_index[tmp_index:tmp_index+size]
            id =ok_index[tmp_index:tmp_index+size].index(max(ok_index[tmp_index:tmp_index+size]))
            print(id)
        else:
            id =max(ok_index[tmp_index:tmp_index+size])
        adsassd = ok_index[tmp_index:tmp_index + size]
        tmp_index +=size
        obj_list.append(id)
    all_result.append(obj_list)
    s = str(all_result[i]).replace(',', '').replace("'", '').replace('[', '').replace(']', '') + '\n'
    file.write(s)
# 方法二
file.close()








