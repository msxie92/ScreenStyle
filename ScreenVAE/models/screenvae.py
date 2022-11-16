from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import functools
import os
# from models.SuperPixelPool.suppixpool_layer import AveSupPixPool,SupPixUnpool
import random
import models.networks as networks

class ScreenVAE(nn.Module):
    def __init__(self,inc=1,outc=4, outplanes=64, downs=5, blocks=2,load_ext=True, save_dir='checkpoints/ScreenVAE',
        init_type="normal", init_gain=0.02, gpu_ids=[]):
        super(ScreenVAE, self).__init__()
        self.inc = inc
        self.outc = outc
        self.save_dir = save_dir
        self.model_names=['enc','dec']
        self.enc=networks.define_C(inc+1, outc*2, 24, netC='resnet_6blocks', 
                                      norm='layer', nl='lrelu', use_dropout=True, 
                                      gpu_ids=gpu_ids, upsample='bilinear')
        self.dec=networks.define_G(outc, inc, 48, netG='unet_128_G', 
                                      norm='layer', nl='lrelu', use_dropout=True, 
                                      gpu_ids=gpu_ids, where_add='input', upsample='bilinear', use_noise=True)
        self.load_networks('latest')
        for param in self.parameters():
            param.requires_grad = False

    def load_networks(self, epoch):

        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(
                    load_path, map_location=lambda storage, loc: storage.cuda())
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)
                del state_dict

    def load_gaborext(self, gpu_ids=[]):
        self.gaborext = GaborWavelet()
        self.gaborext.eval()
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.gaborext.to(gpu_ids[0])

    def npad(self, im, pad=128, value=0):
        h,w = im.shape[-2:]
        hp = h //pad*pad+pad
        wp = w //pad*pad+pad
        return F.pad(im, (0, wp-w, 0, hp-h), mode='constant',value=value)

    def forward(self, x, line=None, screen=False, rep=False):
        if line is None:
            line = torch.ones_like(x)
        else:
            line = torch.sign(line)
            x = torch.clamp(x + (1-line),-1,1)
        if not screen:
            h,w = x.shape[-2:]
            input = torch.cat([x, line], 1)
            input = self.npad(input,value=1)
            inter = self.enc(input)[:,:,:h,:w]
            scr, logvar = torch.split(inter, (self.outc, self.outc), dim=1)
            if rep:
                return scr
            recons = self.dec(scr)
            return recons, scr
        else:
            h,w = x.shape[-2:]
            x = self.npad(x,value=0)
            recons = self.dec(x)[:,:,:h,:w]
            recons = (recons+1)*(line+1)/2-1
            return torch.clamp(recons,-1,1)
