"""
author:guopei
"""
import os
import cv2
import torch
from torch.nn import DataParallel
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import transforms
from models import resnest50


class Person_Attribute(object):
    def __init__(self, weights="resnest50.pth"):
        self.device = torch.device("cuda")
        self.net = resnest50().to(self.device)
        self.net = DataParallel(self.net)
        self.weights = weights
        self.net.load_state_dict(torch.load(self.weights))

        TRAIN_MEAN = [0.485, 0.499, 0.432]
        TRAIN_STD = [0.232, 0.227, 0.266]
        self.transforms = transforms.Compose([
                    transforms.ToCVImage(),
                    transforms.Resize((128,256)),
                    transforms.ToTensor(),
                    transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
        ])

    def recog(self, img_path):
        img = cv2.imread(img_path)
        img = self.transforms(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            self.net.eval()
            img_input = img.to(self.device)
            outputs = self.net(img_input)
            results = []
            for output in outputs:
                output = torch.softmax(output, 1)
                output = np.array(output[0].cpu())
                label = np.argmax(output)
                score = output[label]
                results.append((label, score))
        return results


name_dict = {
                "gender":['男', '女'],
                "age":["老年", "中年", "少年"],
                "orientation":['正面', '侧面', '背面'],
                "hat":["不戴帽子", "戴帽子"],
                "glasses":["不戴眼镜", "戴眼镜"],
                "handBag":["没有", "有"],
                "shoulderBag":["没有", "有"],
                "backBag":["没有", "有"],
                "upClothing":["短袖", "长袖"],
                "downClothing":["长裤", "短裤", "裙子"]
            }
# index =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,23,24,25]-1
# one_labe_size=[1,3,3,1,1,1,1,1,2,3]


if __name__ == "__main__":
    atts = ["gender","age", "orientation", "hat", "glasses",
            "handBag", "shoulderBag", "backBag", "upClothing", "downClothing"]

    person_attribute = Person_Attribute("/home/py/code/mana/attribute/checkpoints/resnest50/2021-02-27T18:11:21.385036/resnest50-47-best.pth")
    img_path = "/home/py/dukto/Market/pytorch/train/0896/0896_c1s4_049706_01.jpg"
    results = person_attribute.recog(img_path)
    print(results)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128,256))
    img1 = img*0 +255
    img1[:,:,0] *= 255
    img1[:,:,2] *= 255

    line =[]
    dict_result={}
    labels = [i[0] for i in results]
    for att, label in zip(atts, labels):
        if label == -1:
            continue

        dict_result.update({str(att):name_dict[att][label]})
       # line.append(dict_one)


    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # img1 = Image.fromarray(img1)
    # draw = ImageDraw.Draw(img1)
   # font = ImageFont.truetype("consola.ttf", 40, encoding="unic"  )
   #  font = ImageFont.truetype(
   #      "font/simsun.ttc", 10, encoding="utf-8")
  #  draw.text((0, 0), line, (255, 0, 0), font=font)
 #   cv2.putText(img1, line, (255, 0, 0))
 #    cv2.putText(img1, line, (0,0 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    print(line)

    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)

    img_rst = np.hstack([img, img1])
    cv2.imshow("attribute.jpg",img_rst)
    cv2.waitKey(0)

