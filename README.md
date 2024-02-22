# Detecting Deepfakes with Patchwise Training on Self-Blended Images
Code is adapted from
The official PyTorch implementation for the following paper: 
> [**Detecting Deepfakes with Self-Blended Images**](https://arxiv.org/abs/2204.08376),  
> Kaede Shiohara and Toshihiko Yamasaki,  
> *CVPR 2022 Oral*

# Methodology

We randomly crop the images into 64x64 patches and perform FFT on the patches to extract the frequency features to append them with spatial features. We use constrastive learning as a pretraining task. You can visualize the embeddings from the pretrained model using TSNE in src/visualize_embedding 
