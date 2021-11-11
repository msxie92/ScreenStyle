import random
import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


class CycleGANSTFTModel(BaseModel):

    def __init__(self, opt):

        BaseModel.__init__(self, opt)

        use_vae = True
        self.interchnnls = 4
        use_noise = False
        self.half_size = opt.batch_size //2
        self.device=opt.local_rank
        self.gpu_ids=[self.device]
        self.local_rank = opt.local_rank
        self.cropsize = opt.crop_size

        self.model_names = ['G_INTSCR2RGB','G_RGB2INTSCR','E']
        self.netG_INTSCR2RGB = networks.define_G(self.interchnnls + 1, 3, opt.nz, opt.ngf, netG='unet_256', 
                                      norm='layer', nl='lrelu', use_dropout=opt.use_dropout, init_type='kaiming', init_gain=opt.init_gain,
                                      gpu_ids=self.gpu_ids, where_add='all', upsample='bilinear', use_noise=use_noise)
        self.netG_RGB2INTSCR = networks.define_G(4, self.interchnnls, 0, opt.ngf, netG='unet_256', 
                                      norm='layer', nl='lrelu', use_dropout=opt.use_dropout, init_type='kaiming', init_gain=opt.init_gain,
                                      gpu_ids=self.gpu_ids, where_add='input', upsample='bilinear', use_noise=use_noise)        
        self.netE = networks.define_E(opt.output_nc, opt.nz, opt.nef, netE=opt.netE, norm='none', nl='lrelu',
                                      init_type='xavier', init_gain=opt.init_gain, gpu_ids=self.gpu_ids, vaeLike=use_vae)
        self.nets = [self.netG_INTSCR2RGB, self.netG_RGB2INTSCR, self.netE]

        self.netSVAE = networks.define_SVAE(inc=self.interchnnls, outc=1, outplanes=64, blocks=3, netVAE='SVAE', 
            save_dir='checkpoints/ScreenVAE',init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)


    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_RGB = input['A'].to(self.device)
        self.real_Ai = self.grayscale(self.real_RGB)
        self.real_L = input['L'].to(self.device)
        self.real_ML = input['Bl'].to(self.device)
        self.real_M = input['B'].to(self.device)

        self.h = input['h']
        self.w = input['w']

    def grayscale(self, input_image):
        rate = torch.Tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1).to(input_image.device)
        # tmp = input_image[:,0, ...] * 0.299 + input_image[:,1, ...] * 0.587 + input_image[:,2, ...] * 0.114
        return (input_image*rate).sum(1,keepdims=True)

    def forward(self, AtoB=True, sty=None):
        if AtoB:
            real_LRGB = torch.cat([self.real_L, self.real_RGB],1)
            fake_SCR = self.netG_RGB2INTSCR(real_LRGB)
            fake_M = self.netSVAE(fake_SCR, line=self.real_L, screen=False)
            fake_M = torch.clamp(fake_M, -1,1)
            fake_M2 = self.norm(torch.mul(self.denorm(fake_M), self.denorm(self.real_L)))#*self.mask2
            return fake_M[:,:,:self.h, :self.w], fake_M2[:,:,:self.h, :self.w], fake_SCR[:,:,:self.h, :self.w]
        else:
            if sty is None:  # use encoded z
                z0, _ = self.netE(self.real_RGB)
            else:
                z0 = sty
                # z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
            real_SCR = self.netSVAE(self.real_M, self.real_ML, rep=True) #8
            real_LSCR = torch.cat([self.real_ML, real_SCR], 1)
            fake_nRGB = self.netG_INTSCR2RGB(real_LSCR, z0)
            fake_nRGB = torch.clamp(fake_nRGB, -1,1)
            fake_RGB = self.norm(torch.mul(self.denorm(fake_nRGB), self.denorm(self.real_ML)))
            return fake_RGB[:,:,:self.h, :self.w], real_SCR[:,:,:self.h, :self.w], self.real_ML[:,:,:self.h, :self.w]

    def norm(self, im):
        return im * 2.0 - 1

    def denorm(self, im):
        return (im + 1) / 2.0

    def optimize_parameters(self):
        pass

    def get_z_random(self, batch_size, nz, random_type='gauss', truncation=False, tvalue=1):
        z = None
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz) * tvalue
            # do the truncation trick
            if truncation:
                k = 0
                while (k < 15 * nz):
                    if torch.max(z) <= tvalue:
                        break
                    zabs = torch.abs(z)
                    zz = torch.randn(batch_size, nz)
                    z[zabs > tvalue] = zz[zabs > tvalue]
                    k += 1
                z = torch.clamp(z, -tvalue, tvalue)

        return z.detach().to(self.device)