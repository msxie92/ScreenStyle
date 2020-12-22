# Bidirectional Translation

Pytorch implementation for bidirectional translation between color comic and manga. 

**Note**: The current software works well with PyTorch 1.1+. 

## Example results


## Prerequisites
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN


## Getting Started ###
### Installation
- Clone this repo:
```bash
git clone https://github.com/msxie92/ScreenStyle.git
cd ScreenStyle/BidirectionalTranslation
```
- Install PyTorch and dependencies from http://pytorch.org
- Install python libraries [tensorboardX](https://github.com/lanpa/tensorboardX).


### Dataset
- We cannot release the whole dataset due to copyright issues. But you can generate synthesis manga dataset using [MangaLineExtraction](https://github.com/ljsabc/MangaLineExtraction) or download public available manga dataset [Manga109 dataset](http://www.manga109.org/en/).

The training requires paired data (including manga image, line drawing, corresponding smoothed manga image and encoded screentone map). 
The line drawing can be extracted using [MangaLineExtraction](https://github.com/ljsabc/MangaLineExtraction).
The smoothed manga image can be obtained applying [Image Smoothing via L0 Gradient Minimization](www.cse.cuhk.edu.hk/~leojia/projects/L0smoothing/).
The encoded screentone map is generated with a variational autoencoder and please refer to the code of screenVAE.


### Model Training
Coming soon ...


## Models
Coming soon ...


