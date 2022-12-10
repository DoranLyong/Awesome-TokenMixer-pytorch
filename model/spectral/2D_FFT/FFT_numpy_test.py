# (ref) https://bo-10000.tistory.com/160

import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

# ---------------- #
# == Load image == #
# ---------------- #
img = cv2.imread('../Lenna.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (224, 224))


# --------- #
# == FFT == #
# --------- #
fft_img = np.fft.fft2(img, norm='ortho')
fft_img = np.fft.fftshift(fft_img) # low_frequency components are at the center

print(fft_img.shape, fft_img.dtype)

f = np.abs(fft_img) + 1e-6
f = np.log(f) # FFT for visualization 


# ---------- #
# == iFFT == # 
# ---------- #
ifft_img = np.fft.ifftshift(fft_img) # inverse shift 
ifft_img = np.fft.ifft2(ifft_img, norm='ortho')

img_back = np.abs(ifft_img)

print(img_back.shape, img_back.dtype)


# ------------- #
# == Compare == # 
# ------------- #
diff = img - img_back 



# ---------- # 
# == Plot == #
# ---------- # 
columns = 4
fig = plt.figure(figsize=[columns*8,12])
title_fontsize = 20 

ax = fig.add_subplot(1, 4, 1)
ax.set_title('Image',fontsize=title_fontsize)
plt.imshow(img, cmap='gray') 

ax = fig.add_subplot(1, 4, 2)
ax.set_title('FFT',fontsize=title_fontsize)
plt.imshow(f, cmap='gray')  

ax = fig.add_subplot(1, 4, 3)
ax.set_title('iFFT',fontsize=title_fontsize)
plt.imshow(img_back, cmap='gray')  

ax = fig.add_subplot(1, 4, 4)
ax.set_title('Diff=Orig - iFFT',fontsize=title_fontsize)
plt.imshow(diff, cmap='gray')  

plt.show()
