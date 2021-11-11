import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch
import torch.nn.functional as F


class SingleSrDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_B = os.path.join(opt.dataroot, opt.phase, opt.folder, 'imgs')
        # self.dir_B = os.path.join(opt.dataroot, opt.phase, 'test/imgs', opt.folder)

        self.B_paths = make_dataset(self.dir_B)

        self.B_paths = sorted(self.B_paths)

        self.B_size = len(self.B_paths)
        # self.transform = get_transform(opt)
        # print(self.B_size)

    def __getitem__(self, index):
        B_path = self.B_paths[index]

        B_img = Image.open(B_path).convert('RGB')
        if os.path.exists(B_path.replace('imgs','line').replace('.jpg','.png')):
            L_img = Image.open(B_path.replace('imgs','line').replace('.jpg','.png'))#.convert('RGB')
        else:
            L_img = Image.open(B_path.replace('imgs','line').replace('.png','.jpg'))#.convert('RGB')
        B_img = B_img.resize(L_img.size, Image.ANTIALIAS)

        ow, oh = B_img.size
        transform_params = get_params(self.opt, B_img.size)
        B_transform = get_transform(self.opt, transform_params, grayscale=True)
        B = B_transform(B_img)
        L = B_transform(L_img)

        # base = 2**8
        # h = int((oh+base-1) // base * base)
        # w = int((ow+base-1) // base * base)
        # B = F.pad(B.unsqueeze(0), (0,w-ow, 0,h-oh), 'replicate').squeeze(0)
        # L = F.pad(L.unsqueeze(0), (0,w-ow, 0,h-oh), 'replicate').squeeze(0)

        return {'B': B, 'Bs': B, 'Bi': B, 'Bl': L, 
                'A': torch.zeros(1), 'Ai': torch.zeros(1), 'L': torch.zeros(1), 
                'A_paths': B_path, 'h': oh, 'w': ow}

    def __len__(self):
        return self.B_size

    def name(self):
        return 'SingleSrDataset'


def M_transform(feat, opt, params=None):
    outfeat = feat.copy()
    if params is not None:
        oh,ow = feat.shape[1:]
        x1, y1 = params['crop_pos']
        tw = th = opt.crop_size
        if (ow > tw or oh > th):
            outfeat = outfeat[:,y1:y1+th,x1:x1+tw]
        if params['flip']:
            outfeat = np.flip(outfeat, 2).copy()#outfeat[:,:,::-1]
    return torch.from_numpy(outfeat).float()*2-1.0