# TokenMixer-pytorch

This project is inspired by [Fighting CV](https://github.com/xmu-xiaoma666/External-Attention-pytorch)'s project. Also, other references are included to get code insights.

Other references: 

*  [xFormers](https://github.com/facebookresearch/xformers)
*  [OpenMixup](https://github.com/Westlake-AI/openmixup)



```bash
# code test env.
python == 3.10.8
pytorch == 1.12.1
```



***

# Contents

- [Attentions](#attentions)
- [MLPs](#mlps)
- [Convolutions](#convolutions)
- [Activations](#activations)
- [Backbones](#backbones)



***

# Attentions
* HiLo Attention (LITv2, [2022](https://github.com/ziplab/litv2)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/attention/HiLo_LITv2.py))(graph)
* Pay Less Attention (LITv1, [2022](https://github.com/ziplab/LIT)) --- (pytorch_v1)(graph)
* External Attention (EANet, [2021](https://github.com/MenghaoGuo/EANet)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/attention/External_Attention.py))(graph)
* Non-local Multi-head Self-Attention (Transformer, [2017](https://paperswithcode.com/method/multi-head-attention)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/attention/non-local_MHSA.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/attention/non-local_MHSA.png))

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
* Inverted SepConv (MobileNetV2, [2018](https://arxiv.org/abs/1801.04381)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/SepConv_metaformer.py))(graph)

# Spectral Features
* Fourier_test --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/spectral/fourier_test.py))
* img_FFT --- ([.py](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/spectral/img_FFT.py))([.ipynb](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/spectral/img_FFT.ipynb))
* Global Filter (GFNet, [2021](https://github.com/raoyongming/GFNet)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/spectral/globalfilter_GFNet.py))(graph)
* Fourier Mixer (FNet, [2021](https://github.com/google-research/google-research/tree/master/f_net)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/spectral/fouriermixer_FNet.py))(graph)

# Activations 

* ReLU --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/activation/ReLU.py))
* GELU --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/activation/GELU.py))
* SquaredReLU --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/activation/SquaredReLU.py))
* StarReLU --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/activation/StarReLU.py))

# Backbones 

