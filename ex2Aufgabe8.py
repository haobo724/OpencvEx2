import numpy as np
from scipy.ndimage import minimum_filter1d
import cv2
from scipy.signal import convolve
from matplotlib import pyplot as plt

def imagetoshow2D(img):

    plt.figure()
    if len(img.shape)==2:
        plt.imshow(img,cmap='gray')
    else:
        plt.imshow(img)
    plt.show()

img_v=cv2.imread(r'data/e-codices_acv-P-Antitus_027v_large.jpg')
img_v=cv2.cvtColor(img_v,cv2.COLOR_BGR2RGB).astype(np.uint16)
gray=cv2.cvtColor(img_v,cv2.COLOR_BGR2GRAY)
imagetoshow2D(gray)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
# compute the Scharr gradient of the blackhat image and scale the
# result into the range [0, 255]
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
# apply a closing operation using the rectangular kernel to close
# gaps in between letters -- then apply Otsu's thresholding method
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
imagetoshow2D(thresh)
