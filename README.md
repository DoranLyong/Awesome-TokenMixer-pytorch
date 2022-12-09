# TokenMixer-pytorch

This project is inspired by [Fighting CV](https://github.com/xmu-xiaoma666/External-Attention-pytorch)'s project. Also, refer to [xFormers](https://github.com/facebookresearch/xformers) repository of the Meta research to get code insights.

Other references: 
* [OpenMixup](https://github.com/Westlake-AI/openmixup)



***

# Contents

- [Attentions](#attentions)
- [MLPs](#mlps)
- [Convolutions](#convolutions)
- [Backbones](#backbones)



***

# Attentions
* HiLo Attention (LITv2, [2022](https://github.com/ziplab/litv2)) --- (pytorch)(graph)
* Pay Less Attention (LITv1, [2022](https://github.com/ziplab/LIT)) --- (pytorch)(graph)
* External Attention (EANet, [2021](https://github.com/MenghaoGuo/EANet)) --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/attention/External_Attention.py))(graph)
* Non-local Multi-head Self-Attention (Transformer, [2017](https://paperswithcode.com/method/multi-head-attention)) --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/attention/non-local_MHSA.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/attention/non-local_MHSA.png))

# MLPs

* Mixing and Shifting for global and local (MS-MLP, [2022](https://github.com/JegZheng/MS-MLP)) --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/mlp/MS-MLP.py))(graph)
* Phase-Aware Token Mixing (Wave-MLP, [2022](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/wavemlp_pytorch)) --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/mlp/wave-MLP.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/MLP/wave-MLP.png))

# Convolutions

* Focal Module (FocalNets, [2022](https://github.com/microsoft/FocalNet)) --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/focal_module.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/conv/focal_module.png))
* gnConv (HorNet, [2022](https://github.com/raoyongming/HorNet)) --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/gnConv_HorNet.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/conv/gnConv_HorNet.png))
* MSCA (SegNeXt, [2022](https://github.com/Visual-Attention-Network/SegNeXt)) --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/MSCA_SegNeXt.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/conv/MSCA_SegNeXt.png))
* LKA (VAN, [2022](https://github.com/Visual-Attention-Network/VAN-Classification)) --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/VAN.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/conv/VAN.png))

# Spectral Features
* Fourier_test --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/spectral/fourier_test.py))
* Global Filter (GFNet, [2021](https://github.com/raoyongming/GFNet)) --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/spectral/globalfilter_GFNet.py))(graph)
* Fourier Mixer (FNet, [2021](https://github.com/google-research/google-research/tree/master/f_net)) --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/spectral/fouriermixer_FNet.py))(graph)

# Backbones 

