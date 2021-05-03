import numpy as np
from scipy.ndimage import minimum_filter1d
import cv2
from scipy.signal import convolve
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import median_filter as median


def ComputerEnergy(img_input):
    test_img = []
    Energy_kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    if (len(img_input.shape)!=3):
        img_input=np.expand_dims(img_input,axis=2)
    for i in range(img_input.shape[-1]):
        temp = convolve(img_input[:, :, i], Energy_kernel, mode='same')

        test_img.append(temp)
    test_img=np.array(test_img)
    test_img=np.sum(test_img,axis=0)
    return np.abs(test_img)
def seam_carving_row(arr,peaks):
    arrc=arr.copy()
    # minlist = np.argmin(arrc[:,0])
    minlist=peaks
    for i in range(arr.shape[1]-1):
        output= minimum_filter1d(arr[:,i].copy(),size=3,mode='reflect')
        arr[:,i+1]=output+arr[:,i+1]
    minindex_list=[]
    print(minlist)
    allpath=[]
    for min in minlist:
        minindex_list.append(min)
        result=None
        for i in range(0,arr.shape[1]-1):
            min = minindex_list[-1]
            output= minimum_filter1d(arr[:,i+1].copy(),size=3,mode='reflect')
            index=np.argwhere(arr[:,i+1]==output[min])
            index=np.squeeze(index,axis=1)
            for inner in index:
                if abs(inner-min)<2:
                    result=inner
                    break
            minindex_list.append(result)
        path=minindex_list
        minindex_list=[]
        # pathcheck(path)
        allpath.append(path)
    return allpath
def pathcheck(path):
    for i in range (len(path)-1):
        if abs(path[i]-path[i+1])>2:
            print(i,abs(path[i]-path[i+1]))
            raise ("WRONG")
def imagetoshow2D(img):

    plt.figure()
    if len(img.shape)==2:
        plt.imshow(img,cmap='gray')
    else:
        plt.imshow(img)
    plt.show()
def imageprocessing(img_input):
    gray = cv2.GaussianBlur(img_input, (3, 3), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    rectKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
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
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rectKernel2)
    thresh = cv2.erode(thresh, rectKernel2, iterations=4)

    return thresh

def lineStart(img_input,rate=0.25):
    row,col=img_input.shape
    img_input=img_input/np.max(img_input)
    temp=0.0
    grad=10
    result=0
    for i in range (int(row*rate)):
        col_sum=np.sum(img_input[:,i])
        if (col_sum-temp) > grad:
            grad=col_sum-temp
            print(grad)
            result = i
            break

        temp=col_sum
    return result


img_v2=cv2.imread(r'data/e-codices_kba-0016-2_006v_large.jpg')
img_v2=cv2.cvtColor(img_v2,cv2.COLOR_BGR2RGB).astype(np.uint16)
gray2=cv2.cvtColor(img_v2,cv2.COLOR_BGR2GRAY)
img_mean = np.mean(gray2)
#todo Exercise 8.1: computing projection profiles
img_contrast=cv2.normalize(gray2,gray2,0,255,cv2.NORM_MINMAX)
img_contrast = img_contrast-img_mean

imagetoshow2D(img_contrast)
projection = np.sum(img_contrast,axis=1)
#todo Exercise 8.2: searching for line starts

subImg = img_contrast[:,:int(img_contrast.shape[1]*0.25)]
temp = np.sum(subImg,axis=1)
cost = median(temp, size=5)
cost[cost>0]=0
x = abs(cost)
peaks, _ = find_peaks(x, height=0)
idx = np.argmin(x)
plt.figure()
plt.plot(x)
plt.plot(peaks, x[peaks], "x")
plt.show()
#todo Exercise 8.3: compute seams
img_cost=ComputerEnergy(img_contrast)
m = np.max(np.sum(img_cost, axis=0))
img_cost[:,0] = m
img_cost[peaks,0] = -m
allpath=seam_carving_row(img_cost,peaks)
for path in allpath:
    for col ,row in enumerate(path):
        img_v2[row,col,0]=255
        img_v2[row,col,1]=0
        img_v2[row,col,2]=0
imagetoshow2D(img_v2)
