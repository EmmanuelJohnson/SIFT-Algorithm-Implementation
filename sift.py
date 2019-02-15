import cv2
import numpy as np
import math
import os

#Gaussian kernel dimension
GK_DIM = 7

#Read the image using opencv
def get_image(path):
    return cv2.imread(path)

#Read the image in gray scale using opencv
def get_image_gray(path):
    return cv2.imread(path,0)

#Show the resulting image
def show_image(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Save the resulting image
def save_image(name,image):
    cv2.imwrite(name,image) 


def guassianKernel(sigma):
    #7X7 Gaussian kernel x and y
    fb = np.asarray([[(-3,3), (-2,3), (-1,3), (0,3), (1,3), (2,3), (3,3)], 
          [(-3,2), (-2,2), (-1,2), (0,2), (1,2), (2,2), (3,2)],
          [(-3,1), (-2,1), (-1,1), (0,1), (1,1), (2,1), (3,1)],
          [(-3,0), (-2,0), (-1,0), (0,0), (1,0), (2,0), (3,0)],
          [(-3,-1), (-2,-1), (-1,-1), (0,-1), (1,-1), (2,-1), (3,-1)],
          [(-3,-2), (-2,-2), (-1,-2), (0,-2), (1,-2), (2,-2), (3,-2)],
          [(-3,-3), (-2,-3), (-1,-3), (0,-3), (1,-3), (2,-3), (3,-3)]])
    if GK_DIM == 5:
        #5X5 Gaussian kernel x and y
        fb = np.asarray([[(-2,2), (-1,2), (0,2), (1,2), (2,2)],
            [(-2,1), (-1,1), (0,1), (1,1), (2,1)],
            [(-2,0), (-1,0), (0,0), (1,0), (2,0)],
            [(-2,-1), (-1,-1), (0,-1), (1,-1), (2,-1)],
            [(-2,-2), (-1,-2), (0,-2), (1,-2), (2,-2)]])
    gk = createZeroMatrix(GK_DIM,GK_DIM)
    s = 0.
    for i in range(fb.shape[0]):
        for j in range(fb.shape[1]):
            gv = gaussian_value(fb[i][j][0],fb[i][j][1],sigma)
            gk[i,j] = gv
            s = s + gv
    #Normalization of the guassian kernel
    c = 1./s
    gk = c*gk
    return gk

#Gaussian formula
def gaussian_value(x,y, sigma):
    sigma2 = float(sigma*sigma)
    xy = (x*x)+(y*y)
    return (1. / (2*math.pi*sigma2)) * (math.exp(-(xy/(2*sigma2))))

#Create a Matrix with all elements 0
def createZeroMatrix(r, c):
    matrix = []
    for i in range(r):
        row = [0 for j in range(c)]
        matrix.append(row)
    return np.asarray(matrix,dtype="float32")

def smoothImages(octaves):
    #Various sigma values for different octaves
    sigmas = {
        "1":[1/math.sqrt(2),1.,math.sqrt(2),2.,2*math.sqrt(2)],
        "2":[math.sqrt(2),2.,2*math.sqrt(2),4.,4*math.sqrt(2)],
        "3":[2*math.sqrt(2),4.,4*math.sqrt(2),8.,8*math.sqrt(2)],
        "4":[4*math.sqrt(2),8.,8*math.sqrt(2),16.,16*math.sqrt(2)]
    }
    smooths = {"1":[],"2":[],"3":[],"4":[]}
    for i in range(4):
        for j,sigma in enumerate(sigmas[str(i+1)]):
            #oimg = get_image_gray('Octave'+str(i+1)+"/org_"+str(i+1)+".jpg")
            oimg = octaves[str(i+1)][0]
            sigma = float(sigma)
            rimg = applyGaussian(oimg,sigma)
            smooths[str(i+1)].append(rimg)
            save_image('Octave'+str(i+1)+"/"+str(i+1)+"_"+str(j)+".jpg",rimg)
    return smooths

def applyGaussian(img, sigma):
    gk = guassianKernel(sigma)
    result = smooth(img, gk)
    return np.array(result ,dtype=np.float32)

#Add Zero padding around the edge of the image
def add_padding(img_matrix,padding):
    row = [0. for j in range(img_matrix.shape[1]+(2*padding))]
    tempImgMatrix = img_matrix.tolist()
    for i in range(img_matrix.shape[0]):
        for f in range(padding):
            tempImgMatrix[i].append(0.)
            tempImgMatrix[i].insert(0,0.)
    for i in range(padding):
        tempImgMatrix.append(row)
        tempImgMatrix.insert(0,row)
    img_padded_matrix = np.array(tempImgMatrix, dtype='float32')
    return img_padded_matrix

def smooth(img, gk):
    ih,iw = img.shape[0],img.shape[1]
    result = createZeroMatrix(ih,iw)
    #Padding
    pad = GK_DIM//2
    gimg = img.copy()
    gimg = add_padding(gimg, pad)

    for i in range(GK_DIM,ih-1-GK_DIM):
        for j in range(GK_DIM,iw-1-GK_DIM):
            #Convolution between the image and the gaussian kernel
            s = 0.
            subMatrix = gimg[i-pad:i-pad+GK_DIM,j-pad:j-pad+GK_DIM]
            for x in range(GK_DIM):
                for y in range(GK_DIM):
                    s = s + (gk[x][y]*subMatrix[x][y])
            result[i-pad][j-pad] = float(s)
    return result
    
def dog(smooths):
    dogdict = {"1":[],"2":[],"3":[],"4":[]}
    for i in range(4):
        for j in range(4):
            #gimg1 = get_image('Octave'+str(i+1)+"/"+str(i+1)+"_"+str(j)+".jpg")
            #gimg2 = get_image('Octave'+str(i+1)+"/"+str(i+1)+"_"+str(j+1)+".jpg")
            gimg1 = smooths[str(i+1)][j]
            gimg2 = smooths[str(i+1)][j+1]
            result = np.asarray(gimg1,dtype='float32') - np.asarray(gimg2,dtype='float32')#gimg1-gimg2#
            res = result.copy()
            dogdict[str(i+1)].append(res)
            save_image('Octave'+str(i+1)+"/"+str(i+1)+"_"+str(j)+"_dog.jpg",result)
    return dogdict


def keypoints(dogs):
    allKeys = []
    for i in range(4):
        oimg = get_image('test.jpg')
        dog = dogs[str(i+1)]
        upScale = [1,2,4,8]
        for j in range(2):
            dimg1 = dog[j]#get_image_gray('Octave'+str(i+1)+"/"+str(i+1)+"_"+str(j)+"_dog.jpg")#
            dimg2 = dog[j+1]#get_image_gray('Octave'+str(i+1)+"/"+str(i+1)+"_"+str(j+1)+"_dog.jpg")#
            dimg3 = dog[j+2]#get_image_gray('Octave'+str(i+1)+"/"+str(i+1)+"_"+str(j+2)+"_dog.jpg")#
            pixels = find_keys(dimg1,dimg2,dimg3)
            #Plot the keypoints in the image with red color
            for p in pixels:
                allKeys.append((p[0]*(upScale[0]),p[1]*(upScale[0])))
                oimg[p[0]*(upScale[0])][p[1]*(upScale[0])] = 255
            save_image("Octave"+str(i+1)+"/"+str(i+1)+"_"+str(j)+"_kp.jpg",oimg)
        save_image("Octave"+str(i+1)+"/"+str(i+1)+"_kp.jpg",oimg)
    return allKeys


def find_keys(d1,d2,d3):
    keys = []
    for i in range(1,d1.shape[0]-1):
        for j in range(1,d1.shape[1]-1):
            d11 = d1[i-1:i+2,j-1:j+2] #3X3 matrix in dog1
            d22 = d2[i-1:i+2,j-1:j+2] #3X3 matrix in dog2
            d33 = d3[i-1:i+2,j-1:j+2] #3X3 matrix in dog3
            center = d22[1,1] #Center element in dog2 - d22
            d22 = np.delete(d22[1], 1) #Delete center element from the dog2 - d22
            mn1,mx1 = d11.min(),d11.max() #Min and Max element in d11
            mn2,mx2 = d22.min(),d22.max() #Min and Max element in d22
            mn3,mx3 = d33.min(),d33.max() #Min and Max element in d33
            mins = min([mn1,mn2,mn3]) #Mins between the 3 mins above
            maxs = max([mx1,mx2,mx3]) #Maxs between the 3 maxs above
            #Keypoint should be either less than the min or greater than the max
            if center<mins:
                diff = abs(center-mins)
                keys.append((i,j))
            elif center>maxs:
                diff = abs(center-maxs)
                keys.append((i,j))

    return keys

def resizeImage(orgImg,scale):
    #Resize code in python
    resImg = orgImg[::scale] #Keep #scale columns and delete rest
    resImg = resImg[:,1::scale] #Keep #scale rows and delete rest
    npResImg = np.asarray(resImg,dtype='float32')
    return npResImg

def createOctaves(img,t):
    #Reduce size of image from 1 to 1/8 of its size
    resizeScale = [1,2,4,8]
    octaves = {"1":[],"2":[],"3":[],"4":[]}
    resizedImages = []
    for i,v in enumerate(resizeScale):
        if not os.path.exists('Octave'+str(i+1)):
                os.makedirs('Octave'+str(i+1))
        rimg = resizeImage(img,v)
        resizedImages.append(rimg)
        octaves[str(i+1)].append(rimg)
    for i,r in enumerate(resizedImages):
        if t == 'g':
            save_image('Octave'+str(i+1)+'/'+str(i+1)+'.jpg',r)
        else:
            save_image('Octave'+str(i+1)+'/org_'+str(i+1)+'.jpg',r)
    return octaves


if __name__ == '__main__':
    #read the image in gray scale
    img = get_image_gray('test.jpg')
    cimg = get_image('test.jpg')
    #image dimensions
    ih,iw = img.shape[0],img.shape[1]

    print('__Creating Octaves__')
    createOctaves(cimg,'c')
    octaves = createOctaves(img,'g')

    print('__Applying ' +str(GK_DIM)+' X '+str(GK_DIM)+' Gaussian Kernel__')
    print('__THIS MIGHT TAKE A WHILE__')
    smooths = smoothImages(octaves)

    print('__Calculating Difference of gaussian__')
    dogs = dog(smooths)

    print('__Detecting key points__')
    print('__THIS MIGHT TAKE A WHILE__')
    allKeys = keypoints(dogs)

    cimg = get_image('test.jpg')
    print('__Plotting all keys in one image__')
    count = 0
    for k in allKeys:
        if count < 10:
            print(k)
        cimg[k[0]][k[1]] = 255
        count+=1
    save_image("KeypointsResult.jpg",cimg)
