# Awesome-TokenMixer-pytorch

This project is inspired by [Fighting CV](https://github.com/xmu-xiaoma666/External-Attention-pytorch)'s project. Also, other references are included to get code insights.

Other references: 
*  [x-transformers](https://github.com/lucidrains/x-transformers)
*  [xFormers](https://github.com/facebookresearch/xformers)
*  [OpenMixup](https://github.com/Westlake-AI/openmixup)
*  [Awesome-Vision-Attentions](https://github.com/MenghaoGuo/Awesome-Vision-Attentions)
*  [vit-pytorch](https://github.com/lucidrains/vit-pytorch)
*  [Networks-Beyond-Attention](https://github.com/FocalNet/Networks-Beyond-Attention)
*  [Efficient-AI-Backbones](https://github.com/huawei-noah/Efficient-AI-Backbones)



```bash
# code test env.
python == 3.10.8
pytorch == 1.12.1
```



***

# Contents
- [Kolmogorov-Arnold Models](#kolmogorov-arnold-models)
- [State Space Models](#state-space-models)
- [Spatial Attentions](#spatial-attentions)
- [Channel Attentions](#channel-attentions)
- [MLPs](#mlps)
- [Convolutions](#convolutions)
- [Spectral Features](#spectral-features)
- [Graph](#graph)
- [Hybrid](#hybrid)
- [Spatio-Temporal (ST)](#spatio-temporal-st)
- [Activations](#activations)
- [Patch Embedding](#patch-embedding )
- [Branch Scaling](#branch-scaling)
- [Normalization](#normalization) 
- [Backbones](#backbones)



***
# Kolmogorov-Arnold Models 
* Kolmogorov-Arnold Transformer (KAT, [2024](https://arxiv.org/abs/2409.10594)) --- (pytorch_v1)

# State Space Models 
* CSM (VMamba, [2024](https://github.com/MzeroMiko/VMamba)) --- (pytorch_v1)(graph)
* Bidirectional Mamba (Vim, [2024](https://github.com/hustvl/Vim)) --- (pytorch_v1)(graph)
* ConvSSM (ConvS5, [2023](https://arxiv.org/abs/2310.19694)) --- (pytorch_v1)(graph)
* Selective SSMs (Mamba, [2023](https://arxiv.org/abs/2312.00752)) --- (pytorch_v1)(graph)

# Spatial Attentions
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
* GatedCNNBlock (MambaOut, [2024](https://github.com/yuweihao/MambaOut)) --- ([pytorch_v1](https://github.com/DoranLyong/Awesome-TokenMixer-pytorch/blob/main/model/conv/GatedCNN_MambaOut.py))(graph)
* Efficient Modulation (EfficientMod. [2024](https://arxiv.org/abs/2403.19963)) --- ([pytorch_v1](./model/attention/EfficientMod_Attention.py))
* SMT (Scale-Aware Modulation Meet Transformer, [2023](https://github.com/AFeng-x/SMT)) --- ([pytorch_v1](./model/conv/SMT.py))
* InceptionDWConv2d (InceptionNeXt, [2023](https://github.com/sail-sg/inceptionnext)) --- ([pytorch_v1](https://github.com/DoranLyong/Awesome-TokenMixer-pytorch/blob/main/model/conv/InceptionDWConv2d_InceptionNeXt.py))(graph)
* ParC operator (ParC-Net, [2022](https://github.com/hkzhang91/parc-net)) --- ([pytorch_v1](https://github.com/DoranLyong/Awesome-TokenMixer-pytorch/blob/main/model/conv/ParC_convnext.py))(graph)
* Dynamic DWConv (DWNet, [2022](https://arxiv.org/abs/2106.04263#)) --- ([pytorch_v1](https://github.com/DoranLyong/Awesome-TokenMixer-pytorch/blob/main/model/conv/DynamicDWConv.py))(graph)
* Focal Module (FocalNets, [2022](https://github.com/microsoft/FocalNet)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/focal_module.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/conv/focal_module.png))
* gnConv (HorNet, [2022](https://github.com/raoyongming/HorNet)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/gnConv_HorNet.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/conv/gnConv_HorNet.png))
* MSCA (SegNeXt, [2022](https://github.com/Visual-Attention-Network/SegNeXt)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/MSCA_SegNeXt.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/conv/MSCA_SegNeXt.png))
* LKA (VAN, [2022](https://github.com/Visual-Attention-Network/VAN-Classification)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/VAN.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/conv/VAN.png))
* Pooling (MetaFormer, [2022](https://arxiv.org/abs/2210.13452)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/Pooling_metaformer.py))(graph)
* RandomMixing (MetaFormer, [2022](https://arxiv.org/abs/2210.13452)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/RandomMixing_metaformer.py))(graph)
* Inverted SepConv (MobileNetV2, [2018](https://arxiv.org/abs/1801.04381)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/SepConv_metaformer.py))(graph)
* Deformable Conv (Deformable ConvNet, [2017](https://pytorch.org/vision/main/generated/torchvision.ops.deform_conv2d.html)) --- ([pytorch_v1](https://github.com/DoranLyong/Awesome-TokenMixer-pytorch/blob/main/model/conv/DeformableConv2d.py))

# Spectral Features
* Dynamic Filter (DFFormer, [2024](https://github.com/okojoalg/dfformer)) --- ([pytorch_v1](./model/spectral/DynamicFilter.py))
* Global Filter (GFNet, [2021](https://github.com/raoyongming/GFNet)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/spectral/globalfilter_GFNet.py))(graph)
* Fourier Mixer (FNet, [2021](https://github.com/google-research/google-research/tree/master/f_net)) --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/spectral/fouriermixer_FNet.py))(graph)
* Fourier_test --- ([pytorch_v1](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/spectral/fourier_test.py))
* img_FFT --- ([.py](./model/spectral/2D_FFT/img_FFT.py))([.ipynb](./model/spectral/2D_FFT/img_FFT.ipynb))([2D FFT](./model/spectral/2D_FFT))

# Graph
* Mobile ViG ([SVGA](./model/graph/MobileViG/svga.py), [2023](https://github.com/SLDGroup/MobileViG)) --- ([pytorch_v1](./model/graph/MobileViG/mobilevig.py))(graph)
* Vision GNN (ViG, [2022](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch)) --- ([pytorch_v1](./model/graph/ViG))(graph)
* GINconv2d (GIN, [2019](https://arxiv.org/abs/1810.00826)) --- ([pytorch_v1](./model/graph/GIN))(graph)
* DeepGCNs (MR-GCN, [2019](https://github.com/lightaime/deep_gcns_torch)) --- ([pytorch_v1](./model/graph/DeepGCNs))(graph)

# Hybrid

* Inception Mixer (iFormer, [2022](https://github.com/sail-sg/iFormer)) --- ([pytorch_v1](./model/attention/inceptionMixer_iFormer.py))(graph)
* MHRA (Uniformer v1, [2022](https://github.com/Sense-X/UniFormer)) --- ([pytorch_v1](./model/hybrid/MHRA_uniformer_v1.py))(graph)

# Spatio-Temporal (ST)

* MHRA (Uniformer v2, [2022](https://github.com/OpenGVLab/UniFormerV2/blob/main/slowfast/models/uniformerv2_model.py)) --- (pytorch_v1)(graph)([mmaction2](https://mmaction2.readthedocs.io/en/latest/model_zoo/recognition.html#uniformerv2))
* MHRA (Uniformer v1, [2022](https://github.com/Sense-X/UniFormer/blob/main/video_classification/slowfast/models/uniformer.py)) --- ([pytorch_v1](./model/spatio_temporal/MHRA_uniformer_v1.py))(graph)([mmaction2](https://mmaction2.readthedocs.io/en/latest/model_zoo/recognition.html#uniformer))

# Activations 

* StarReLU ([2022](https://arxiv.org/abs/2210.13452)) --- ([pytorch_v1](./model/activation/StarReLU.py))
* SquaredReLU ([2021](https://arxiv.org/abs/2109.08668)) --- ([pytorch_v1](./model/activation/SquaredReLU.py))
* Swish ([2017](https://arxiv.org/abs/1710.05941v2)) --- ([pytorch_v1](./model/activation/Swish.py))
* GELU ([2016](https://arxiv.org/abs/1606.08415)) --- ([pytorch_v1](./model/activation/GELU_likes/GELU.py))([scratch_code](./model/activation/GELU_likes/GELU_from_scratch.py))([quick_version](./model/activation/GELU_likes/quickGELU.py))
* ReLU ([2010](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)) --- ([pytorch_v1](./model/activation/ReLU.py))
* Softsign ([2009](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)) --- ([pytorch_v1](./model/activation/Modified_S-shaped_functions/Softsign.py))
* Zero-centered s-shaped functions --- ([pytorch_v1](./model/activation/Modified_S-shaped_functions))
* S-shape activation function comparision --- ([pytorch_v1](./model/activation/Modified_S-shaped_functions/s-shape_activation.py))

# Patch Embedding 

* patch embed --- ([pytorch_v1](./model/patch_emb/patch_embed.py))

# Branch Scaling 

* Layer-/Residual-branch scaling ([LayerScale](https://arxiv.org/abs/2103.17239) 2021, [ResScale](https://arxiv.org/abs/2110.09456) 2021) --- ([pytorch_v1](./model/scale/layer_res_scale.py))([timm](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py))

# Normalization 

* GroupNorm
* LayerNorm ([2016](https://arxiv.org/abs/1607.06450)) --- ([pytorch_v1](./model/norm/layer_norm.py))
* BatchNorm ([2015](https://arxiv.org/abs/1502.03167)) --- ([pytorch_v1](./model/norm/batch_norm1D.py))

# Backbones 
* Efficient Modulation Network ([2024](https://arxiv.org/abs/2403.19963)) --- ([pytorch_v1](./model/backbones/EfficientModNet.py))
* VanillaNet ([2023](https://github.com/huawei-noah/VanillaNet/tree/main)) --- ([pytorch_v1](./model/backbones/VanillaNet.py))
* InceptionNeXt; MetaNeXt ([2023](https://github.com/sail-sg/inceptionnext/tree/main)) --- ([pytorch_v1](./model/backbones/MetaNeXt.py))
* MetaFormer baseline ([2022](https://github.com/sail-sg/metaformer/tree/main)) --- ([pytorch_v1](./model/backbones/MetaFormer.py))
* PoolFormer ([2022](https://github.com/sail-sg/poolformer)) --- ([pytorch_v1](./model/backbones/PoolFormer.py))
