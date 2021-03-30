import pandas as pd
import scipy
from scipy import io

data = scipy.io.loadmat('annotation.mat')


def mat2txt(data, key):
    subdata = data[key]
    dfdata = pd.DataFrame(subdata)
    dfdata.to_csv("%s.txt" % key, index=False)


if __name__ == "__main__":
    data = scipy.io.loadmat("annotation.mat")
    key_list = ["attributes", "test_images_name", "test_label",
                "train_images_name", "train_label",
                "val_images_name", "val_label"]
    for key in key_list:
        mat2txt(data, key)
