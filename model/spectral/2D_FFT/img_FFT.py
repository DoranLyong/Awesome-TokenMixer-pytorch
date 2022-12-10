# (ref) https://kai760.medium.com/how-to-use-torch-fft-to-apply-a-high-pass-filter-to-an-image-61d01c752388
# (ref) https://pytorch.org/docs/stable/generated/torch.fft.fftshift.html

import numpy as np 
import cv2 
from PIL import Image

import torch 
import torch.nn as nn
import torchvision.transforms as T 


# == Load image == # 
img = Image.open("Lenna.png").convert('RGB')
cv2.imshow("img_orig", np.array(img))
print(np.array(img).shape)

# == Convert to tensor == #
img_T = T.ToTensor()(img) # (C,H,W)
img_T = img_T.unsqueeze(0) # (B, C, H, W)
B,C,H,W = img_T.shape 
print(img_T.shape)

# == FFT == # 
img_T = img_T.to(torch.float32)
img_FFT = torch.fft.rfft2(img_T, dim=(2, 3), norm='ortho') # (B, C, H, W//2+1)


# == FFTShift == # 
# low_frequency components are at the center
img_FFT = torch.fft.fftshift(img_FFT)
temp_FFT = torch.zeros_like(img_FFT)

# == High pass filter == # 
filter_rate = 1
b,c, h, w = img_FFT.shape # height and width
cy, cx = int(h/2), int(w/2) # centerness
rh, rw = int(filter_rate * cy), int(filter_rate * cx) # filter_size

#img_FFT[:, :, cy-rh:cy+rh, cx-rw:cx+rw] = 0 # the value of center pixel is zero.
temp_FFT[:, :, cy-rh:cy+rh, cx-rw:cx+rw] = img_FFT[:, :, cy-rh:cy+rh, cx-rw:cx+rw]
img_FFT = temp_FFT  


# == Inverse FFTShift == #
#img_FFT = torch.fft.ifftshift(img_FFT)

# == Inverse FFT == # 
img_filtered = torch.fft.irfft2(img_FFT, s=(H, W), dim=(2, 3), norm='ortho') # (B, C, H, W)
img_filtered = torch.view_as_real(img_FFT)
img_filtered = img_filtered[...,0]

# == Image show == # 
img_filtered = img_filtered.squeeze(0) # (H, W)
img_filtered = img_filtered.to('cpu').detach().numpy()

img_filtered = img_filtered.transpose(1,2,0) # (H, W, C)
cv2.imshow("img", img_filtered)
cv2.waitKey(0)
