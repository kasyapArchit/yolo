#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_img(len, data):
    return data.reshape(len, 3, 32, 32).transpose([0, 2, 3, 1])

def get_nm():
    data = unpickle('cifar-10/batches.meta')
    lt_nm = data[b'label_names']
    del data
    return lt_nm

def get_test_data():
    lt_img = np.empty([0])
    lt_lab = []
    path = 'cifar-10/data_batch_'
    for i in range(5):
        data = unpickle(path+str(i+1))
        tmp_img = data[b'data']
        tmp_lab = data[b'labels']
        lt_lab.extend(tmp_lab)
        lt_img = np.append(lt_img, tmp_img)
        del data, tmp_img, tmp_lab

    lt_img = get_img(50000, lt_img)
    lt_img = lt_img.astype(int)
    return lt_img, lt_lab


#%%
lt_nm = get_nm()
lt_img, lt_lab = get_test_data()

#%%
