from __future__ import print_function
import torch
import torch.nn.functional as F
import os
from models.screenvae import ScreenVAE

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, io
from skimage.segmentation import flood, flood_fill
from sklearn.decomposition import PCA

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import numpy as np
np.random.seed(seed)

torch.cuda.set_device(0)

def get_screenmap(img, line=None):
    img = torch.from_numpy(img[np.newaxis,np.newaxis,...]).float()*2-1.0
    line = torch.from_numpy(line[np.newaxis,np.newaxis,...]).float()*2-1.0
    scr = model(img.cuda(), line.cuda(), rep=True).cpu().detach()
    scr = scr*(line+1)/2
    return scr.numpy()[0]

def get_recons(scr):
    scr = torch.from_numpy(scr).unsqueeze(0).float()
    outs = model(scr.cuda(), None, screen=True)*0.5+0.5
    return torch.clamp(outs,0,1).cpu().detach().numpy()[0,0]

def getpca(scr):
    result = np.concatenate([im.reshape(1,-1) for im in scr], axis=0)
    pca = PCA(n_components=3)
    pca.fit(result)
    result = pca.components_.copy()
    result = result.transpose().reshape((scr.shape[1], scr.shape[2], 3))
    for i in range(3):
        tmppic = result[:,:,i]
        result[:,:,i] = (tmppic - tmppic.min()) / (tmppic.max() - tmppic.min())
    #     cv2.normalize(tmppic,resultPic[:,:,i],0,255,dtype=cv2.NORM_MINMAX)
    return result

def process(pts=(10,10)):
    filled = np.ones((scr.shape[1],scr.shape[2]))
    nscr = scr.copy()
    # scr[line[np.newaxis,:,:].repeat(4,axis=0)<0.75]=-1
    for i in range(4):
        filled_img = flood(scr[i,:,:], pts, tolerance=0.15)
        filled[~filled_img]=0
    plt.close(2)
    figi, axi = plt.subplots(figsize=(10, 8))

    rnd = np.random.randn(4)*0.5
    for t in range(4):
        nscr[t][filled[:]==1]=rnd[t]
    out = get_recons(nscr)
    # nimg = img.copy()
    nimg[filled[:]==1]=out[filled[:]==1]
    nimg[:] = nimg[:]*line[:]
    
    io.imsave('saved.png',nimg)

    axi.imshow(nimg, cmap=plt.cm.gray)
    axi.set_title('Edited')
    axi.axis('off')
    figi.show()


# =============================== TRAINING ====================================
device = [0]
outf = 'ScreenVAE'
model = ScreenVAE(inc=1, outc=4, blocks=3, save_dir='checkpoints/%s'%outf, gpu_ids=device)

img = io.imread('examples/manga.png', as_gray=True)/255.0
line = io.imread('examples/line.png', as_gray=True)/255.0

nimg = img.copy()
scr = get_screenmap(img,line)
resultPic = getpca(scr)
fig, ax = plt.subplots(ncols=2, figsize=(16, 8), dpi=100)

def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))# Refresh the plot
    process((int(event.ydata),int(event.xdata)))
    fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')

io.imsave('PCA.png',resultPic)

ax[1].imshow(resultPic)
ax[1].set_title('PCA')
ax[1].axis('off')

plt.show()
