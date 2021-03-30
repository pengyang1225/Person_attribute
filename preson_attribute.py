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
        self.size =(128,256)
        self.transforms = transforms.Compose([
                    transforms.ToCVImage(),
                    transforms.Resize((128,256)),
                    transforms.ToTensor(),
                    transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
        ])

        self.mean = torch.tensor([0.485, 0.499, 0.432], dtype=torch.float32)
        self.std = torch.tensor([0.232, 0.227, 0.266], dtype=torch.float32)

        self.atts = ["gender", "age", "orientation", "hat", "glasses",
                "handBag", "shoulderBag", "backBag", "upClothing", "downClothing"]

    def detect(self, img):
        #imgss = self.transforms(img)
        image = img.astype('uint8')
        image = cv2.resize(image, self.size,cv2.INTER_LINEAR )
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        image = image.float() / 255.0





        image = image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])

        image = image.unsqueeze(0)

        with torch.no_grad():
            self.net.eval()
            img_input = image.to(self.device)
            outputs = self.net(img_input)
            results = []
            for output in outputs:
                output = torch.softmax(output, 1)
                output = np.array(output[0].cpu())
                label = np.argmax(output)
                score = output[label]
                results.append((label, score))
            labels = [i[0] for i in results]
            dict_result={}

            for att, label in zip( self.atts, labels):
                if label == -1:
                    continue
                dict_result.update({str(att): name_dict[att][label]})
        return dict_result


name_dict = {
                "gender":['mela', 'female'],
                "age":["old age", "middle age", "young person"],
                "orientation":['positive', 'side', '背面'],
                "hat":["no", "yes"],
                "glasses":["no", "yes"],
                "handBag":["no", "yes"],
                "shoulderBag":["no", "yes"],
                "backBag":["no", "yes"],
                "upClothing":["short sleeve", "long sleeve"],
                "downClothing":["trousers", "shorts", "skirt"]
            }
if __name__ == "__main__":
    atts = ["gender","age", "orientation", "hat", "glasses",
            "handBag", "shoulderBag", "backBag", "upClothing", "downClothing"]

    person_attribute = Person_Attribute("/home/py/code/mana/attribute/checkpoints/resnest50/2021-02-27T18:11:21.385036/resnest50-47-best.pth")
    img_path = "/home/py/dukto/Market/pytorch/train/0896/0896_c1s4_049706_01.jpg"
    img =cv2.imread(img_path)
    results = person_attribute.detect(img)
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

    print(line)

    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)

    img_rst = np.hstack([img, img1])
    cv2.imshow("attribute.jpg",img_rst)
    cv2.waitKey(0)

