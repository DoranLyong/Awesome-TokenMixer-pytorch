# TokenMixer-pytorch

This project is inspired by [Fighting CV](https://github.com/xmu-xiaoma666/External-Attention-pytorch)'s project. 



***

# Contents

- [Attentions](#attentions)
- [MLPs](#mlps)
- [Convolutions](#convolutions)
- [Backbones](#backbones)



***

# Attentions

* Non-local Muliti-head Self-Attention --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/attention/non-local_MHSA.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/attention/non-local_MHSA.png))
* External Attention --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/attention/External_Attention.py))(graph)

# MLPs

* MS-MLP --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/mlp/MS-MLP.py))(graph)
* Wave-MLP --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/mlp/wave-MLP.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/MLP/wave-MLP.png))

# Convolutions

* Focal Module (FocalNets) --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/focal_module.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/conv/focal_module.png))
* gnConv (HorNet) --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/gnConv_HorNet.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/conv/gnConv_HorNet.png))
* MSCA (SegNeXt) --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/MSCA_SegNeXt.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/conv/MSCA_SegNeXt.png))
* LKA (VAN) --- ([pytorch](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/conv/VAN.py))([graph](https://github.com/DoranLyong/TokenMixer-pytorch/blob/main/model/ComputationGraph_imgs/conv/VAN.png))

# Backbones 

