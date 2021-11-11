import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageEnhance
import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2


class SingleCoDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase, opt.folder, 'imgs')

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)

        self.A_size = len(self.A_paths)
        # self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index]

        A_img = Image.open(A_path).convert('RGB')
        # enhancer = ImageEnhance.Brightness(A_img)
        # A_img = enhancer.enhance(1.5)
        if os.path.exists(A_path.replace('imgs','line')[:-4]+'.jpg'):
            # L_img = Image.open(A_path.replace('imgs','line')[:-4]+'.png')
            L_img = cv2.imread(A_path.replace('imgs','line')[:-4]+'.jpg')
            kernel = np.ones((3,3), np.uint8)
            L_img = cv2.erode(L_img, kernel, iterations=1)
            L_img = Image.fromarray(L_img)
        else:
            L_img = A_img
        if A_img.size!=L_img.size:
            # L_img = L_img.resize(A_img.size, Image.ANTIALIAS)
            A_img = A_img.resize(L_img.size, Image.ANTIALIAS)
        if A_img.size[1]>2500:
            A_img = A_img.resize((A_img.size[0]//2, A_img.size[1]//2), Image.ANTIALIAS)

        ow, oh = A_img.size
        transform_params = get_params(self.opt, A_img.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=False)
        L_transform = get_transform(self.opt, transform_params, grayscale=True)
        A = A_transform(A_img)
        L = L_transform(L_img)

        # base = 2**9
        # h = int((oh+base-1) // base * base)
        # w = int((ow+base-1) // base * base)
        # A = F.pad(A.unsqueeze(0), (0,w-ow, 0,h-oh), 'replicate').squeeze(0)
        # L = F.pad(L.unsqueeze(0), (0,w-ow, 0,h-oh), 'replicate').squeeze(0)

        tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        Ai = tmp.unsqueeze(0)
        
        return {'A': A, 'Ai': Ai, 'L': L, 
                'B': torch.zeros(1), 'Bs': torch.zeros(1), 'Bi': torch.zeros(1), 'Bl': torch.zeros(1), 
                'A_paths': A_path, 'h': oh, 'w': ow}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'SingleCoDataset'


def M_transform(feat, opt, params=None):
    outfeat = feat.copy()
    oh,ow = feat.shape[1:]
    x1, y1 = params['crop_pos']
    tw = th = opt.crop_size
    if (ow > tw or oh > th):
        outfeat = outfeat[:,y1:y1+th,x1:x1+tw]
    if params['flip']:
        outfeat = np.flip(outfeat, 2)#outfeat[:,:,::-1]
    return torch.from_numpy(outfeat.copy()).float()*2-1.0