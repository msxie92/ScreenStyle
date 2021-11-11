# Bidirectional Translation

Pytorch implementation for multimodal comic-to-manga translation. 

**Note**: The current software works well with PyTorch 1.6.0+. 

## Prerequisites
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started ###
### Installation
- Clone this repo:
```bash
git clone https://github.com/msxie/ScreenStyle.git
cd ScreenStyle/MangaScreening
```
- Install PyTorch and dependencies from http://pytorch.org
- Install python libraries [tensorboardX](https://github.com/lanpa/tensorboardX)
- Install other libraries
For pip users:
```
pip install -r requirements.txt
```

## Data praperation
The training requires paired data (including manga image, western image and their line drawings). 
The line drawing can be extracted using [MangaLineExtraction](https://github.com/ljsabc/MangaLineExtraction).

  ```
${DATASET} 
|-- color2manga 
|   |-- val 
|   |   |-- ${FOLDER}
|   |   |   |-- imgs
|   |   |   |   |-- 0001.png 
|   |   |   |   |-- ...
|   |   |   |-- line
|   |   |   |   |-- 0001.png 
|   |   |   |   |-- ...
  ```

### Use a Pre-trained Model
- Download the pre-trained [ScreenVAE](https://drive.google.com/file/d/1QaXqR4KWl_lxntSy32QpQpXb-1-EP7_L/view?usp=sharing) model and place under `checkpoints/ScreenVAE/` folder.

- Download the pre-trained [color2manga](https://drive.google.com/file/d/18-N1W0t3igWLJWFyplNZ5Fa2YHWASCZY/view?usp=sharing) model and place under `checkpoints/color2manga/` folder.
- Generate results with the model
```bash
bash ./scripts/test_western2manga.sh
```

## Copyright and License
You are granted with the [LICENSE](LICENSE) for both academic and commercial usages.

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@article{xie-2020-manga,
        author   = {Minshan Xie and Chengze Li and Xueting Liu and Tien-Tsin Wong},
        title    = {Manga Filling Style Conversion with Screentone Variational Autoencoder},
        journal  = {ACM Transactions on Graphics (SIGGRAPH Asia 2020 issue)},
        month    = {December},
        year     = {2020},
        volume   = {39},
        number   = {6},
        pages    = {226:1--226:15}
    }
```

### Acknowledgements
This code borrows heavily from the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository.
