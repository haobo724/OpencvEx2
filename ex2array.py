import numpy as np
from scipy.ndimage import minimum_filter1d
import cv2
from scipy.signal import convolve
from matplotlib import pyplot as plt
# arr = np.array([[4,3,2,1,3,5,4],[2,5,4,3,5,1,3],[4,1,3,2,4,4,2],[1,5,3,2,5,1,1],[4,2,1,3,2,2,4],[5,2,5,5,2,4,1],[3,5,1,4,1,2,5]])
def seam_carving(arr):
    for i in range(arr.shape[0]-1):
        output= minimum_filter1d(arr[i],size=3,mode='reflect')
        arr[i+1]=output+arr[i+1]
    minindex_list=[]
    min = np.argmin(arr[-1])
    minindex_list.append(min)
    for i in reversed(range(1,arr.shape[0])):
        output= minimum_filter1d(arr[i-1],size=3,mode='reflect')
        index=np.argwhere(arr[i-1]==output[minindex_list[-1]])[0][0]
        minindex_list.append(index)
    return list(reversed(minindex_list))
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
    Cost_onRow=np.sum(img_input,axis=0)
    Cost_onCol=np.sum(img_input,axis=1)

    return Cost_onRow,Cost_onCol
def imagetoshow2D(img):

    plt.figure()
    if len(img.shape)==2:
        plt.imshow(img,cmap='gray')
    else:
        plt.imshow(img)
    plt.show()
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

def removeRECUSIV(img_input,NUM,mask_ON=False,mask_input=None):
    if (len(img_input.shape) != 3):
        print("Gray INPUT")
        img_input = np.expand_dims(img_input, axis=2)
    img_v = ComputerEnergy(img_input)
    if mask_ON:
        img_v += mask_input
    index_list = seam_carving(img_v)

    row,col,channel=img_input.shape
    mask=np.empty_like(img_input).astype(bool)
    mask.fill(True)
    for x,y in enumerate(index_list):
        mask[x,y,:]=False
    pic=img_input[mask].reshape((row,col-1,channel))
    if mask_ON:
        mask_input=mask_input[mask[:,:,0]].reshape((row,col-1))
    if NUM>0:
        NUM-=1
        pic=removeRECUSIV(pic,NUM,mask_ON,mask_input)
    return pic
def removenew(img_input,NUM,STEP,mask_ON=False,mask_input=None):
    if (len(img_input.shape) != 3):
        print("Gray INPUT")
        img_input = np.expand_dims(img_input, axis=2)
    img_COST = ComputerEnergy(img_input)

    while NUM>0:
        if NUM % STEP == 0:
            img_COST = ComputerEnergy(img_input)
        if mask_ON:
            img_COST += mask_input
        index_list = seam_carving(img_COST)

        row, col, channel = img_input.shape
        mask = np.empty_like(img_input).astype(bool)
        mask.fill(True)
        for x, y in enumerate(index_list):
            mask[x, y, :] = False
        img_input = img_input[mask].reshape((row, col - 1, channel))

        if mask_ON:
            mask_input = mask_input[mask[:, :, 0]].reshape((row, col - 1))
        img_COST = img_COST[mask[:, :, 0]].reshape((row, col - 1))
        NUM-=1
    return img_input

def extand(img_input,NUM,STEP,mask_ON=False,mask_input=None):
    if (len(img_input.shape) != 3):
        print("Gray INPUT")
        img_input = np.expand_dims(img_input, axis=2)
    img_COST = ComputerEnergy(img_input)
    while NUM > 0:
        if NUM % STEP == 0:
            img_COST = ComputerEnergy(img_input)
        if mask_ON:
            img_COST += mask_input
        row, col, channel = img_input.shape
        index_list = seam_carving(img_COST)
        # index_list_right=[i+1 for i in index_list ]
        # index_list_right=[i if i<col else col for i in index_list_right]
        #
        mask = np.empty_like(img_input).astype(bool)
        mask.fill(False)
        mask2=mask.copy()
        # for x, y in enumerate(index_list):
        #     mask[x, y, :] = True
        # img_insert_part = img_input[mask]

        # for x1, y1 in enumerate(index_list_right):
        #     mask2[x1, y1, :] = True
        # img_insert_part2 = img_input[mask2]
        # img_insert=((img_insert_part+img_insert_part2)//2).reshape((row, 1, channel))
        temp= np.zeros([row, col+1, channel]).astype(np.int)
        for x, y in enumerate(index_list):
            temp[x,0:y,:]=img_input[x,0:y,:]

            if y+1>col:
                grenz=col
            else:
                grenz = y + 1
            temp[x,y,:]=(img_input[x,y,:]+img_input[x,grenz,:])//2
            temp[x,y+1:col+1,:]=img_input[x,y:col+1,:]
        img_input=temp
        if mask_ON:
            mask_input = mask_input[mask[:, :, 0]].reshape((row, col - 1))
        # img_COST = img_COST[mask[:, :, 0]].reshape((row, col - 1))
        NUM-=1

    return img_input


img_v=cv2.imread('data/vincent-on-cliff.jpg')
img_v=cv2.cvtColor(img_v,cv2.COLOR_BGR2RGB).astype(np.uint16)
img_gg=cv2.cvtColor(img_v,cv2.COLOR_BGR2GRAY)

img_mask=cv2.imread('data/kingfishers-mask.png')
img_mask=cv2.cvtColor(img_mask,cv2.COLOR_BGR2GRAY)
print(img_mask.shape)
imgc=img_v.copy()

# Cost_R, Cost_C=Cost(img_v)
img_g=cv2.cvtColor(imgc,cv2.COLOR_RGB2GRAY)
imgtest=extand(img_v,100,10)
print(imgtest.shape)
# imgtest=removenew(imgc,450,10,True,img_mask)
imagetoshow2DMulit(imgc,imgtest)
