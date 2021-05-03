import numpy as np
from scipy.ndimage import minimum_filter1d
import cv2
from scipy.signal import convolve
from matplotlib import pyplot as plt
from tqdm import tqdm

# arr = np.array([[4,3,2,1,3,5,4],[2,5,4,3,5,1,3],[4,1,3,2,4,4,2],[1,5,3,2,5,1,1],[4,2,1,3,2,2,4],[5,2,5,5,2,4,1],[3,5,1,4,1,2,5]])
def seam_carving(arr):
    for i in range(arr.shape[0]-1):
        output= minimum_filter1d(arr[i].copy(),size=3,mode='reflect')
        arr[i+1]=output+arr[i+1]
    minindex_list=[]
    min = np.argmin(arr[-1])
    minindex_list.append(min)
    checklist=[]
    result=None
    for i in reversed(range(1,arr.shape[0])):
        min = minindex_list[-1]
        output= minimum_filter1d(arr[i-1].copy(),size=3,mode='reflect')

        # number=np.argmin([arr[i-1][min-1],arr[i-1][min],arr[i-1][min+1]])
        index=np.argwhere(arr[i-1]==output[min])
        index=np.squeeze(index,axis=1)
        for inner in index:
            if abs(inner-min)<2:
                result=inner
                break
            checklist.append(abs(inner-min))
        checklist=[]
        minindex_list.append(result)
    path=list(reversed(minindex_list))
    pathcheck(path)
    return list(reversed(minindex_list))
def pathcheck(path):
    for i in range (len(path)-1):
        if abs(path[i]-path[i+1])>2:
            print(i,abs(path[i]-path[i+1]))
            raise ("WRONG")
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
    img_COST = ComputerEnergy(img_input.copy())
    for _ in tqdm(range(NUM)):
        if NUM % STEP == 0:
            img_COST = ComputerEnergy(img_input.copy())
        if mask_ON:
            img_COST += mask_input
        index_list = seam_carving(img_COST.copy())
        row, col, channel = img_input.shape
        mask = np.empty_like(img_input).astype(bool)
        mask.fill(True)
        for x, y in enumerate(index_list):
            mask[x, y, :] = False
        img_input = img_input[mask].reshape((row, col - 1, channel))

        if mask_ON:
            mask_input = mask_input[mask[:, :, 0]].reshape((row, col - 1))
        img_COST = img_COST[mask[:, :, 0]].reshape((row, col - 1))
    return img_input

def extand(img_input,NUM,STEP,mask_ON=False,mask_input=None):
    if (len(img_input.shape) != 3):
        print("Gray INPUT")
        img_input = np.expand_dims(img_input, axis=2)
    img_COST = ComputerEnergy(img_input)
    if mask_ON:
        img_COST += mask_input
    while NUM > 0:
        if NUM % STEP == 0:
            img_COST = ComputerEnergy(img_input)
            if mask_ON:
                img_COST += mask_input

        row, col, channel = img_input.shape
        index_list = seam_carving(img_COST.copy())
        temp= np.zeros([row, col+1, channel]).astype(np.int16)
        temp_mask= np.zeros([row, col+1]).astype(np.int16)
        temp_COST= np.zeros([row, col+1])
        for x, y in enumerate(index_list):
            temp[x,0:y,:]=img_input[x,0:y,:]
            if y+1>col-1:
                grenz=col-1
            else:
                grenz = y + 1
            temp[x,y,:]=(img_input[x,y,:]+img_input[x,grenz,:])//2
            temp[x,grenz:,:]=img_input[x,y:,:]
            # mask_input=mask_input*2

            temp_mask[x, 0:y] = mask_input[x, 0:y]
            if y + 1 > col-1:
                grenz = col-1
            else:
                grenz = y + 1
            temp_mask[x, y] += 255
            temp_mask[x, grenz+1:] = mask_input[x, grenz:]
            temp_COST[x, 0:y] = img_COST[x, 0:y]

            if (y + 1) > (col - 1):
                grenz = col - 1
                print("warning")
            else:
                grenz = y + 1

            temp_COST[x, y] = 255
            temp_COST[x, grenz:] = img_COST[x,y:]

        img_input=temp
        mask_input=temp_mask
        img_COST=temp_COST
        NUM-=1

    return img_input


img_vincent=cv2.imread('data/vincent-on-cliff.jpg')
img_bird=cv2.imread('data/kingfishers.jpg')
img_vincent=cv2.cvtColor(img_vincent,cv2.COLOR_BGR2RGB).astype(np.uint16)
img_bird=cv2.cvtColor(img_bird,cv2.COLOR_BGR2RGB).astype(np.uint16)

img_bird_mask=cv2.imread('data/kingfishers-mask.png')
img_bird_mask=cv2.cvtColor(img_bird_mask,cv2.COLOR_BGR2GRAY)
imgcopy=img_vincent.copy()
img_mask_vincent=np.zeros_like(img_vincent)
img_mask_vincent[300:860,1100:1400,:]=2550
img_mask_vincent=cv2.cvtColor(img_mask_vincent,cv2.COLOR_BGR2GRAY)

# img_g=cv2.cvtColor(imgc,cv2.COLOR_RGB2GRAY)
#TODO Exercise 4.4: seam carving on images

imgtest_remove=removenew(img_bird,100,10,True,img_bird_mask)
imagetoshow2DMulit(imgtest_remove,img_bird)

#TODO Exercise 5: upscaling images

imgtest=extand(img_vincent,200,10,True,img_mask_vincent)
print(imgtest.shape)
# imgtest=removenew(imgc,450,10,True,img_mask)
imagetoshow2DMulit(img_vincent,imgtest)
