import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

#intensity transformation function
def tf(x):
    y = (np.arctan((x - 128.0) / 32.0))
    y_op = np.interp(y, (y.min(), y.max()), (0, 255))
    return(y_op)

#plot diagram of the transformation function
r = np.arange(0, 255, 1)
s = tf(r)
plt.plot(r, s)
plt.show()
#output data to excel
data = pd.DataFrame({"r" : r, "s" : s})
data.to_csv("result.csv",index=False)

#read image and show transformed image
img = cv2.imread('Bird feeding 3 low contrast.tif')
img_tf = tf(img)
plt.imshow(img_tf.astype(np.uint8))
plt.show()
#show image histogram
img1d = img.ravel()
plt.hist(img1d, 256, [0, 256])
plt.title('origin')
plt.show()
#show transformed image histogram
img1d_tf =img_tf.ravel()
plt.hist(img1d_tf, 256, [0, 256])
plt.title('output')
plt.show()