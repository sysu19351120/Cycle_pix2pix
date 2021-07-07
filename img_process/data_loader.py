import os

import cv2
import numpy as np
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import SDoG
def label_generator(path,image_path):

    print("generate labels...")
    f = open(path, "w")
    image_names = os.listdir(image_path)
    for img in image_names:
        lines = img + '\n'
        f.write(lines)
    f.close()
class Cartoon(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(Cartoon, self).__init__()
        self.root = root
        self.train = train
        self.transform_ = transform
        self._path=[]
        if self.train:
            self.txt="../train_label.txt"
        else:
            self.txt="../test_label.txt"
        self._path_append()
    def _path_append(self):
        for line in open(self.txt):
            line = line.rstrip()
            words = line.split()
            self._path.append(words)
    def __getitem__(self, index):
        img_name=self._path[index]
        image_path=os.path.join(self.root,img_name[0])
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        if self.transform_ is not None:
            img = self.transform_(img)
        return img
    def __len__(self):
        return len(self._path)
def Concatenate(root, path1, path2):
    j=0
    for i in tqdm(range(20000)):
        sketch_path = os.path.join(path1, 'sketch'+str(i)+'.jpg')
        origin_path = os.path.join(path2, 'origin'+str(i)+'.jpg')
        if(os.path.exists(sketch_path) and os.path.exists(origin_path)):
            img1 = Image.open(sketch_path)
            rgb = img1.convert('RGB')
            rgb = rgb.resize((64,64), Image.ANTIALIAS)
            rgb_npy = np.array(rgb)
            img2=Image.open(origin_path)
            img2 = img2.resize((64,64), Image.ANTIALIAS)
            img2_npy = np.array(img2)
            img = np.concatenate((rgb_npy,img2_npy), 1)
            img = Image.fromarray(img)
            try:
                os.makedirs(root + 'cartoon/')
            except OSError:
                pass
            img.save(root + 'cartoon/img' + str(j) + '.jpg')
            j+=1
if __name__ == '__main__':
    label_generator("../train_label.txt","../Cartoon/")
    # label_generator("../test_label.txt","../Cartoon/")
    c_train_root = '../Cartoon/'
    # c_test_root='../Cartoon/'
    save_root = '../dataset/'
    origin_cub_train = Cartoon(c_train_root)
    sketch_cub_train = Cartoon(c_train_root, transform=SDoG.XDOG)
    # origin_cub_test = Cartoon(c_test_root, train=None)
    # sketch_cub_test = Cartoon(c_test_root, train=None, transform=SDoG.XDOG)
    sketch_path = save_root + 'sketch/'
    origin_path = save_root + 'origin/'
    try:
        os.makedirs(sketch_path)
        os.makedirs(origin_path)
    except OSError:
        pass

    i = 0
    for img in tqdm(origin_cub_train):
        img.save(origin_path+'origin'+str(i)+'.jpg')
        i = i+1
        if i >= origin_cub_train.__len__():
            break
    # stop = i + origin_cub_test.__len__()
    # for img in tqdm(origin_cub_test):
    #     img.save(origin_path+'origin'+str(i)+'.jpg')
    #     i = i+1
    #     if i >= stop:
    #         break

    i = 0
    for img in tqdm(sketch_cub_train):
        img.save(sketch_path+'sketch'+str(i)+'.jpg')
        i = i+1
        if i >= sketch_cub_train.__len__():
            break

    # stop = i + sketch_cub_test.__len__()
    # for img in tqdm(sketch_cub_test):
    #     img.save(sketch_path+'sketch'+str(i)+'.jpg')
    #     i = i+1
    #     if i >= stop:
    #         break
            
    Concatenate(save_root, sketch_path, origin_path)
