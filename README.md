# Awesome-TokenMixer-pytorch

This project is inspired by [Fighting CV](https://github.com/xmu-xiaoma666/External-Attention-pytorch)'s project. Also, other references are included to get code insights.

Other references: 

*  [xFormers](https://github.com/facebookresearch/xformers)
*  [OpenMixup](https://github.com/Westlake-AI/openmixup)
*  [Awesome-Vision-Attentions](https://github.com/MenghaoGuo/Awesome-Vision-Attentions)
*  [vit-pytorch](https://github.com/lucidrains/vit-pytorch)



```bash
# code test env.
python == 3.10.8
pytorch == 1.12.1
```



***

# Contents

- [Spatial Attentions](#spatial-attentions)
- [Channel Attentions](#channel-attentions)
- [MLPs](#mlps)
- [Convolutions](#convolutions)
- [Spectral Features](#spectral-features)
- [Activations](#activations)
- [Patch Embedding](#patch-embedding )
- [Branch Scaling](#branch-scaling)
- [Normalization](#normalization) 
- [Backbones](#backbones)



***

# Spatial Attentions
* Inception Mixer (iFormer, [2022](https://github.com/sail-sg/iFormer)) --- ([pytorch_v1](https://github.com/DoranLyong/Awesome-TokenMixer-pytorch/blob/main/model/attention/inceptionMixer_iFormer.py))(graph)
* HiLo Attention (LITv2, [2022](https://github.com/ziplab/litv2)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/attention/HiLo_LITv2.py))(graph)
* Pay Less Attention (LITv1, [2022](https://github.com/ziplab/LIT)) --- (pytorch_v1)(graph)
* External Attention (EANet, [2021](https://github.com/MenghaoGuo/EANet)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/attention/External_Attention.py))(graph)
* Non-local Multi-head Self-Attention (Transformer, [2017](https://paperswithcode.com/method/multi-head-attention)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/attention/non-local_MHSA.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/attention/non-local_MHSA.png))

# Channel Attentions 

* Frequency Channel Attention (FcaNet, [2021](https://arxiv.org/abs/2012.11879)) --- ([pytorch_v1](https://github.com/DoranLyong/Awesome-TokenMixer-pytorch/blob/main/model/channel_attention/FCA_FcaNet.py))(graph)
* Global Context Attention (GCNet, [2019](https://arxiv.org/abs/1904.11492)) --- ([pytorch_v1](https://github.com/DoranLyong/Awesome-TokenMixer-pytorch/blob/main/model/channel_attention/GCA_GCNet.py))(graph)
* Efficient Channel Attention (ECA-Net, [2019](https://arxiv.org/abs/1910.03151)) --- ([pytorch_v1](https://github.com/DoranLyong/Awesome-TokenMixer-pytorch/blob/main/model/channel_attention/ECA_ECANet.py))(graph)
* Squeeze-and-Excitation (SE-Net, [2018](https://arxiv.org/abs/1709.01507)) --- ([pytorch_v1](https://github.com/DoranLyong/Awesome-TokenMixer-pytorch/blob/main/model/channel_attention/SE_SENet.py))(graph)

# MLPs

* Mixing and Shifting for global and local (MS-MLP, [2022](https://github.com/JegZheng/MS-MLP)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/mlp/MS-MLP.py))(graph)
* Phase-Aware Token Mixing (Wave-MLP, [2022](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/wavemlp_pytorch)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/mlp/wave-MLP.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/MLP/wave-MLP.png))
* An all-MLP Architecture for Vision (MLP-Mixer, [2021](https://arxiv.org/abs/2105.01601)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/mlp/MLP-mixer.py))(graph)

# Convolutions

* Focal Module (FocalNets, [2022](https://github.com/microsoft/FocalNet)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/focal_module.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/conv/focal_module.png))
* gnConv (HorNet, [2022](https://github.com/raoyongming/HorNet)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/gnConv_HorNet.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/conv/gnConv_HorNet.png))
* MSCA (SegNeXt, [2022](https://github.com/Visual-Attention-Network/SegNeXt)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/MSCA_SegNeXt.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/conv/MSCA_SegNeXt.png))
* LKA (VAN, [2022](https://github.com/Visual-Attention-Network/VAN-Classification)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/VAN.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/conv/VAN.png))
* Pooling (MetaFormer, [2022](https://arxiv.org/abs/2210.13452)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/Pooling_metaformer.py))(graph)
* RandomMixing (MetaFormer, [2022](https://arxiv.org/abs/2210.13452)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/RandomMixing_metaformer.py))(graph)
* Inverted SepConv (MobileNetV2, [2018](https://arxiv.org/abs/1801.04381)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/SepConv_metaformer.py))(graph)

# Spectral Features
* Fourier_test --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/spectral/fourier_test.py))
* img_FFT --- ([.py](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/spectral/2D_FFT/img_FFT.py))([.ipynb](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/spectral/2D_FFT/img_FFT.ipynb))([2D FFT](https://github.com/DoranLyong/TokenMixer-pytorch/tree/main/model/spectral/2D_FFT))
* Global Filter (GFNet, [2021](https://github.com/raoyongming/GFNet)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/spectral/globalfilter_GFNet.py))(graph)
* Fourier Mixer (FNet, [2021](https://github.com/google-research/google-research/tree/master/f_net)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/spectral/fouriermixer_FNet.py))(graph)

# Activations 

* StarReLU ([2022](https://arxiv.org/abs/2210.13452)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/activation/StarReLU.py))
* SquaredReLU ([2021](https://arxiv.org/abs/2109.08668)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/activation/SquaredReLU.py))
* GELU ([2016](https://arxiv.org/abs/1606.08415)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/activation/GELU.py))([scratch_code](https://github.com/DoranLyong/Awesome-TokenMixer-pytorch/blob/main/model/activation/GELU_from_scratch.py))
* ReLU ([2010](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/activation/ReLU.py))

# Patch Embedding 

* patch embed --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/patch_emb/patch_embed.py))

# Branch Scaling 

* Layer-/Residual-branch scaling ([LayerScale](https://arxiv.org/abs/2103.17239) 2021, [ResScale](https://arxiv.org/abs/2110.09456) 2021) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/scale/layer_res_scale.py))([timm](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py))

# Normalization 

* GroupNorm
* LayerNorm ([2016](https://arxiv.org/abs/1607.06450)) --- ([pytorch_v1](https://github.com/DoranLyong/Awesome-TokenMixer-pytorch/blob/main/model/norm/layer_norm.py))
* BatchNorm ([2015](https://arxiv.org/abs/1502.03167)) --- ([pytorch_v1](https://github.com/DoranLyong/Awesome-TokenMixer-pytorch/blob/main/model/norm/batch_norm1D.py))

# Backbones 

