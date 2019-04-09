import os
from PIL import Image
import numpy as np
from tvm.contrib.download import download_testdata

def get_mobilenet():
    url = 'https://docs-assets.developer.apple.com/coreml/models/MobileNet.mlmodel'
    dst = 'mobilenet.mlmodel'
    real_dst = download_testdata(url, dst, module='coreml')
    return real_dst

def get_resnet50():
    url = 'https://docs-assets.developer.apple.com/coreml/models/Resnet50.mlmodel'
    dst = 'resnet50.mlmodel'
    real_dst = download_testdata(url, dst, module='coreml')
    return real_dst

def get_cat_image():
    url = 'https://gist.githubusercontent.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/fa7ef0e9c9a5daea686d6473a62aacd1a5885849/cat.png'
    dst = 'cat.png'
    real_dst = download_testdata(url, dst, module='data')
    img = Image.open(real_dst).resize((224, 224))
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
    return np.asarray(img)
