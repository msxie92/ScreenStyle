## Manga Filling with ScreenVAE
### [SIGGRAPH ASIA 2020](https://dl.acm.org/doi/abs/10.1145/3414685.3417873) | [Project Website](https://www.cse.cuhk.edu.hk/~ttwong/papers/screenstyle/screenstyle.html) | [BibTex](#citation)

This repository is for ScreenVAE introduced in the following paper
"Manga Filling Style Conversion with Screentone Variational Autoencoder", SIGGRAPH AISA 2020 (TOG) 

## Introduction
Western color comics and Japanese-style screened manga are two popular comic styles. They mainly differ in the style of region-filling. However, the conversion between the two region-filling styles is very challenging, and manually done currently. In this paper, we identify that the major obstacle in the conversion between the two filling styles stems from the difference between the fundamental properties of screened region-filling and colored region-filling. To resolve this obstacle, we propose a screentone variational autoencoder, ScreenVAE, to map the screened manga to an intermediate do- main. This intermediate domain can summarize local texture characteristics and is interpolative. With this domain, we effectively unify the properties of screening and color-filling, and ease the learning for bidirectional translation between screened manga and color comics. To carry out the bidirectional translation, we further propose a network to learn the translation between the intermediate domain and color comics. Our model can generate quality screened manga given a color comic, and generate color comic that retains the original screening intention by the bitonal manga artist. Several results are shown to demonstrate the effectiveness and convenience of the proposed method. We also demonstrate how the intermediate domain can assist other applications such as manga inpainting and photo-to-comic conversion. 

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
