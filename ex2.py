import cv2
import numpy as np
from scipy.signal import convolve
from matplotlib import pyplot as plt
from tqdm import tqdm
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
def Cost(img_input):
    Cost_onRow=np.sum(img_input,axis=1)
    Cost_onCol=np.sum(img_input,axis=0)

    return Cost_onRow,Cost_onCol
def maskgenerator(img_input):
    if img_input.shape[-1]==3:
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    otsuThe, dst_Otsu = cv2.threshold(img_input, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst_Otsu=255-dst_Otsu
    dst_Otsu=cv2.dilate(dst_Otsu,kernel,iterations=8)
    return dst_Otsu
def Remove(img_input,REDUCE_NUM=1,Mask_On=True,mask_input=None,Row_Switch_oN=True,Col_Switch_oN=True):
    templist=[]
    if (len(img_input.shape) != 3):
        print("Gray INPUT")
        img_input = np.expand_dims(img_input, axis=2)
    img_copy = img_input.copy().astype(np.uint16)
    if Mask_On:
        if mask_input is not None:
            mask=mask_input
        else:
            mask = maskgenerator(img_copy)
        # for index in range(img_input.shape[-1]):
        #     img_copy[..., index] = img_copy[..., index] + mask
    temp = ComputerEnergy(img_copy)
    if Mask_On:
        temp = temp+mask
    Cost_R, Cost_C = Cost(temp)

    for i in tqdm(range(REDUCE_NUM)):
        for channel in range(img_input.shape[-1]):
            if Row_Switch_oN:
                ROW=np.delete(img_input[...,channel], np.argmin(Cost_R), 0)
            else:
                ROW=img_input[...,channel]
            if Col_Switch_oN:
                COL=np.delete(ROW, np.argmin(Cost_C), 1)
            else:
                COL=ROW
            templist.append(COL)
        Cost_R = np.delete(Cost_R, np.argmin(Cost_R))
        Cost_C = np.delete(Cost_C, np.argmin(Cost_C))
        img_input=np.array(templist).transpose((1,2,0))
        templist=[]
    return img_input

def Removebetter(img_input,REDUCE_NUM=0,Mask_On=True,mask_input=None,Row_Switch_oN=True,Col_Switch_oN=True):
    templist=[]
    if (len(img_input.shape) != 3):
        print("Gray INPUT")
        img_input = np.expand_dims(img_input, axis=2)

    # BREAKPOINT_R=400
    # BREAKPOINT_C=400
    if Mask_On:
        if mask_input is not None:
            mask = mask_input
        else:
            img_copy = img_input.copy()
            mask = maskgenerator(img_copy)

    for i in tqdm(range(REDUCE_NUM)):
        img_copy = img_input.copy()

        temp = ComputerEnergy(img_copy.copy())
        if Mask_On:

            temp = temp + mask

        Cost_R, Cost_C = Cost(temp)
        Cost_R_index=np.argsort(Cost_R)
        Cost_C_index=np.argsort(Cost_C)


        for channel in range(img_input.shape[-1]):
            if Row_Switch_oN:
                ROW = np.delete(img_input[..., channel], Cost_R_index[0:10], 0)
            else:
                ROW = img_input[..., channel]
            if Col_Switch_oN:
                COL = np.delete(ROW, Cost_C_index[0:10], 1)
            else:
                COL = ROW
            templist.append(COL)
        if Mask_On:

            if Row_Switch_oN:
                ROW_mask = np.delete(mask,Cost_R_index[0:10], 0)
            else:
                ROW_mask = mask
            COL_mask = np.delete(ROW_mask, Cost_C_index[0:10], 1)
            mask=np.array(COL_mask)
        img_input=np.array(templist).transpose((1,2,0))

        templist=[]

    return img_input
def imagetoshow2DMulit(img,img2):

    plt.figure()
    plt.subplot(211)
    if len(img.shape)==2:
        plt.imshow(img,cmap='gray')
    else:
        plt.imshow(img,cmap='RdYlGn')

    plt.subplot(212)
    if len(img2.shape)==2:
        plt.imshow(img2,cmap='gray')
    else:
        plt.imshow(img2)
    plt.show()

def imagetoshow2D(img):

    plt.figure()
    if len(img.shape)==2:
        plt.imshow(img,cmap='gray')
    else:
        plt.imshow(img)
    plt.show()

img=cv2.imread('data/common-kestrel.jpg')
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


img_bird=cv2.imread('data/kingfishers.jpg')
img_bird_mask=cv2.imread('data/kingfishers-mask.png')
img_bird_mask=cv2.cvtColor(img_bird_mask,cv2.COLOR_BGR2GRAY)
print(img_bird_mask.shape)
result_bird=Remove(img_bird,60,Mask_On=False,Row_Switch_oN=False)
result_bird2=Removebetter(img_bird,60,Mask_On=True,mask_input=img_bird_mask,Row_Switch_oN=False)
imagetoshow2DMulit(result_bird,result_bird2)
