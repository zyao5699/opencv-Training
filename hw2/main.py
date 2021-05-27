import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# reference:https://docs.opencv.org/master/de/dbc/tutorial_py_fourier_transform.html
img = cv2.imread('Bird 2.tif',0)
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.show()

rows, cols = img.shape
crow,ccol = rows//2 , cols//2
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
fshift_lp = dft_shift*mask
f_ishift_lp = np.fft.ifftshift(fshift_lp)
img_lp = cv2.idft(f_ishift_lp)
img_lp = cv2.magnitude(img_lp[:,:,0],img_lp[:,:,1])
plt.imshow(img_lp, cmap = 'gray')
plt.show()

mask = np.ones((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0
fshift_hp = dft_shift*mask
f_ishift_hp = np.fft.ifftshift(fshift_hp)
img_hp = cv2.idft(f_ishift_hp)
img_hp = cv2.magnitude(img_hp[:,:,0],img_hp[:,:,1])
plt.imshow(img_hp, cmap = 'gray')
plt.show()

W = img.shape[0]
H = img.shape[1]
data=[]
top25=[]
for i in range(W//2):
    for j in range(H):
        data.append([magnitude_spectrum[j][i], j, i])
data.sort(reverse= True)
for i in range(25):
    top25.append(data[i])
d = pd.DataFrame({"Magnitude" : [item[0] for item in top25], "u" : [item[1] for item in top25],"v" : [item[2] for item in top25]})
d.to_csv("result.csv",index=False)
