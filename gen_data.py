#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
#%%
def unpickle(file):
    '''
    Load the pickle data
    '''
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_img(len, data):
    '''
    Convert the matrix row into (32,32,3)
    '''
    return data.reshape(len, 3, 32, 32).transpose([0, 2, 3, 1])

def get_nm():
    '''
    Load the class list
    '''
    data = unpickle('cifar-10/batches.meta')
    lt_nm = data[b'label_names']
    del data
    return lt_nm

def get_test_data():
    '''
    Generate a list of all the test images in CIFAR-10
    '''
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
def img_resize(img, factor):
    '''
    Resize the image given the scaling factor
    cv2 works on float values
    while matplotlib works on int values
    '''
    img = img.astype('float32') 
    w = int(32*factor)
    res = cv2.resize(img, (w, w))
    return res.astype('int')

#%%
def get_random_img():
    '''
    Get a random list of image number and their scaling factor
    the number of images vary between 2 and 5
    '''
    n = random.randint(2, 5)
    idx = random.sample(range(50000), n)
    fact = np.empty([0])
    ht = np.empty([0])
    for i in range(n):
        x = random.uniform(1.0, 2.2)
        fact = np.append(fact, x)
        ht = np.append(ht, int(32*x))
    return idx, fact, ht 
#%%
idx, fact, ht = get_random_img()