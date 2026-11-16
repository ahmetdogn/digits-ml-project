import numpy as np
import gzip
import os
import urllib.request

link = "https://storage.googleapis.com/cvdf-datasets/mnist/"
dateien = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def download():
    if not os.path.exists("data"):
        os.mkdir("data")
    for d in dateien:
        if not os.path.exists("data/" + d):
            print("downloade " + d)
            urllib.request.urlretrieve(link + d, "data/" + d)


def bilder_laden(datei):
    f = gzip.open("data/" + datei, "rb")
    inhalt = np.frombuffer(f.read(), np.uint8, offset=16)
    f.close()
    return inhalt.reshape(-1, 28, 28)


def labels_laden(datei):
    f = gzip.open("data/" + datei, "rb")
    inhalt = np.frombuffer(f.read(), np.uint8, offset=8)
    f.close()
    return inhalt


def laden():
    download()
    train_x = bilder_laden(dateien[0])
    train_y = labels_laden(dateien[1])
    test_x = bilder_laden(dateien[2])
    test_y = labels_laden(dateien[3])
    return train_x, train_y, test_x, test_y
