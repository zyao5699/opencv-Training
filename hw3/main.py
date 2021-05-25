import numpy as np
import cv2
import matplotlib.pyplot as plt

img_input = cv2.imread('Bird 3 blurred.tif')
img = cv2.cvtColor(img_input,cv2.COLOR_BGR2RGB)
r = img[:,:,0]
g = img[:,:,1]
b = img[:,:,2]
plt.imsave('R.jpg',r,cmap ='gray')
plt.imsave('G.jpg',g,cmap ='gray')
plt.imsave('B.jpg',b,cmap ='gray')

img_hsi = cv2.cvtColor(img_input,cv2.COLOR_BGR2HSV)
h = img_hsi[:,:,0]
s = img_hsi[:,:,1]
i = img_hsi[:,:,2]
plt.imsave('H.jpg',h,cmap ='gray')
plt.imsave('S.jpg',s,cmap ='gray')
plt.imsave('I.jpg',i,cmap ='gray')

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
rgb_sharp = cv2.filter2D(img,-1,kernel)
hsi_sharp = cv2.cvtColor(cv2.filter2D(img_hsi,-1,kernel),cv2.COLOR_HSV2RGB)
dif = hsi_sharp - rgb_sharp
plt.imsave('RGB.jpg',rgb_sharp)
plt.imsave('HSI.jpg',hsi_sharp)
plt.imsave('dif.jpg',dif)
