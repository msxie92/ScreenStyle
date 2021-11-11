import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html
import cv2

seed = 10
import torch
import numpy as np
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# options
opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

model = create_model(opt)
model.setup(opt)
model.eval() 
print('Loading model %s' % opt.model)

testdata = ['manga_paper']
# fake_sty = model.get_z_random(1, 64, truncation=True)

opt.dataset_mode = 'singleSr'
for folder in testdata:
    opt.folder = folder
    # create dataset
    dataset = create_dataset(opt)
    web_dir = os.path.join(opt.results_dir, opt.folder + '_Sr2Co')
    webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))
    # fake_sty = model.get_z_random(1, 64, truncation=True)
    for i, data in enumerate(islice(dataset, opt.num_test)):
        h = data['h']
        w = data['w']
        model.set_input(data)
        fake_sty = model.get_z_random(1, 64, truncation=True, tvalue=1.25)
        fake_B, SCR, line = model.forward(AtoB=False, sty=fake_sty)
        images=[fake_B[:,:,:h,:w]]
        names=['color']

        img_path = 'input_%3.3d' % i
        save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)
    webpage.save()

testdata = ['western_paper']

opt.dataset_mode = 'singleCo'
for folder in testdata:
    opt.folder = folder
    # create dataset
    dataset = create_dataset(opt)
    web_dir = os.path.join(opt.results_dir, opt.folder + '_Sr2Co')
    webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))
    for i, data in enumerate(islice(dataset, opt.num_test)):
        h = data['h']
        w = data['w']
        model.set_input(data)
        fake_B, fake_B2, SCR = model.forward(AtoB=True)
        images=[fake_B2[:,:,:h,:w]]
        names=['manga']

        img_path = 'input_%3.3d' % i
        save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)
    webpage.save()
