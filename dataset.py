import glob
import torch as th 

from libraries.strategies import read_image
from torchvision import transforms as T 
from torch.utils.data import Dataset
from os import path 


class Source(Dataset):
    def __init__(self, root, shape):
        width, height = shape
        self.lr_resize = T.Resize((width // 4, height // 4))
        self.hr_resize = T.Resize((width, height))
        self.files = sorted(glob.glob(path.join(root, '*.jpg')))

    def normalize(self, img, is_hr):
        img = img / th.max(img)
        if is_hr:
            return T.Normalize(mean=[0.5], std=[0.5])(img) 
        return img 

    def __getitem__(self, index):
        I = read_image(self.files[index], by='th')
        I = I / th.max(I)
        I_LR = self.lr_resize(I)
        I_HR = self.hr_resize(I)
        return self.normalize(I_LR, False), self.normalize(I_HR,True)  

    def __len__(self):
        return len(self.files)