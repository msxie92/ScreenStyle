from __future__ import print_function
import torch
import torch.nn.functional as F
import os
from torch.autograd import Variable
from models.screenvae import ScreenVAE

from tkinter import *
from PIL import ImageTk, Image

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import numpy as np
np.random.seed(seed)


torch.cuda.set_device(0)


def test(X,line=None):
    X = X/127.5-1.0
    X = torch.from_numpy(X.transpose(2,0,1)).unsqueeze(0).float().cuda()
    line = torch.ones_like(X)
    outs = model(X, line, img_input=False)
    # Loss
    return torch.clamp(outs,-1,1).cpu().numpy()[0,0]

# =============================== TRAINING ====================================
device = [0]
outf = 'ScreenVAE'
model = ScreenVAE(inc=1, outc=4, blocks=3, save_dir='checkpoints/%s'%outf, gpu_ids=device)

# model.load_networks('latest')
# model.cuda()
# model.eval()


def Generate(event):
    c1 = w1.get()
    c2 = w2.get()
    c3 = w3.get()
    c4 = w4.get()
    c5 = w5.get()
    c6 = w6.get()
    c7 = w7.get()
    c8 = w8.get()
    cl1 = np.array((c1,c2,c3,c4))
    cl2 = np.array((c5,c6,c7,c8))
    X = generate_gradient(cl1,cl2)
    with torch.no_grad():
        im = test(X)
    im = im*127.5+127.5
    img2 = ImageTk.PhotoImage(Image.fromarray(im[32:-32,:].astype(np.uint8)))
    panel1.configure(image=img2)
    panel1.image = img2


def generate_gradient(cl1,cl2):
    out = np.zeros((w,4))
    cl1 = cl1[np.newaxis,:]
    cl2 = cl2[np.newaxis,:]
    tmp = np.arange(w).astype(np.float32)[:,np.newaxis]/w
    out = tmp*(cl2-cl1)+cl1

    return out[np.newaxis,:,:].repeat(h+64, axis=0).astype(np.uint8)

    
master = Tk()

w = 768
h = 256
X = np.zeros((h,w))

img2 = ImageTk.PhotoImage(Image.fromarray(X))
panel1 = Label(master, image=img2)
panel1.grid(row=1, columnspan=2)

w1 = Scale(master, from_=0, to=255, length=256, orient=HORIZONTAL, command=Generate)
w1.grid(row=3, column=0)
w2 = Scale(master, from_=0, to=255, length=256, orient=HORIZONTAL, command=Generate)
w2.grid(row=4, column=0)
w3 = Scale(master, from_=0, to=255, length=256, orient=HORIZONTAL, command=Generate)
w3.grid(row=5, column=0)
w4 = Scale(master, from_=0, to=255, length=256, orient=HORIZONTAL, command=Generate)
w4.grid(row=6, column=0)
w5 = Scale(master, from_=0, to=255, length=256, orient=HORIZONTAL, command=Generate)
w5.grid(row=3, column=1)
w6 = Scale(master, from_=0, to=255, length=256, orient=HORIZONTAL, command=Generate)
w6.grid(row=4, column=1)
w7 = Scale(master, from_=0, to=255, length=256, orient=HORIZONTAL, command=Generate)
w7.grid(row=5, column=1)
w8 = Scale(master, from_=0, to=255, length=256, orient=HORIZONTAL, command=Generate)
w8.grid(row=6, column=1)

mainloop()
