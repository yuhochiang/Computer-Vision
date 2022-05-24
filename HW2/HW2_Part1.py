import cv2
import os
import glob
import imageio
import numpy as np
from matplotlib import pyplot as plt
from math import exp

LOW_COMPONENT = 0.5
D0 = 50

def fourier_transform(img):
    img_shift = np.copy(img)
    rows, cols = img.shape
    for x in range(rows):
        for y in range(cols):
            img_shift[x,y] = img[x,y]*((-1)**(x+y))
    dft_shift = np.fft.fft2(img_shift)
    return dft_shift

def gaussian_filter(dft, filter_type):
    """filter"""
    def H(u,v):
        if filter_type == "lowpass":
            return exp(-((u*u + v*v)/(2*D0*D0)))
        elif filter_type == "highpass":
            return 1-exp(-((u*u + v*v)/(2*D0*D0)))
        else:
            print("filter_type not defined")
    gaussian = np.copy(dft)
    rows, cols = dft.shape
    cen_u = int(rows/2) + 1 if rows % 2 == 1 else int(rows/2)
    cen_v = int(cols/2) + 1 if cols % 2 == 1 else int(cols/2)
    for u in range(rows):
        for v in range(cols):
            gaussian[u,v] = dft[u,v]*H(u-cen_u,v-cen_v)
    return gaussian

def inv_ft(dft):
    """Inverse fourier transform"""
    img_shift = np.fft.ifft2(dft)
    
    img_shift = np.real(img_shift)
    img = np.copy(img_shift)
    
    rows, cols = img_shift.shape
    for x in range(rows):
        for y in range(cols):
            img[x,y] = img_shift[x,y]*((-1)**(x+y))
    return img

def ensure_samesize(a, b):
    if a.shape[0] == b.shape[0] and a.shape[1] == b.shape[1]:
        return a, b
    #print("padding")
    max_row = max(a.shape[0], b.shape[0])
    max_col = max(a.shape[1], b.shape[1])
    padding_a = np.zeros((max_row, max_col), dtype=np.complex128)
    padding_b = np.zeros((max_row, max_col), dtype=np.complex128)
    padrow_a = int((max_row - a.shape[0])/2)
    padcol_a = int((max_col - a.shape[1])/2)
    padrow_b = int((max_row - b.shape[0])/2)
    padcol_b = int((max_col - b.shape[1])/2)
    padding_a[(padrow_a):(padrow_a+a.shape[0]), (padcol_a):(padcol_a+a.shape[1])] = a
    padding_b[(padrow_b):(padrow_b+b.shape[0]), (padcol_b):(padcol_b+b.shape[1])] = b
    return padding_a, padding_b

def hybrid_image( img1 , img2):

    #cv2.imshow('My Image', img)
    img1_dft = fourier_transform(img1)
    img1_gaussian = gaussian_filter(img1_dft, "highpass")

    img2_dft = fourier_transform(img2)
    img2_gaussian = gaussian_filter(img2_dft, "lowpass")

    low, high = ensure_samesize(img1_gaussian, img2_gaussian)
    hybrid_img = low + high 

    hybrid_img = inv_ft(hybrid_img)
    
    
    return hybrid_img

if __name__=="__main__":
    pics = []
    path = './hw2_data/task1,2_hybrid_pyramid/'
    for root, dirs, files in os.walk(path):
        for pic in files:
            if ('.jpg' in pic) | ('.bmp' in pic):
                pics.append(pic)
    for i in range(0,16,2):
        img1 = plt.imread(path + pics[i]).astype(np.float32)/255.0
        #print(type(img1[0][0][0]))
        r, g, b = cv2.split(img1)
        img2 = plt.imread(path + pics[i+1]).astype(np.float32)/255.0
        r2, g2, b2 = cv2.split(img2)

        Image_new = (cv2.merge([hybrid_image(r, r2), hybrid_image(g, g2), hybrid_image(b, b2)]))
        #print(type(Image_new[0][0][0]))
        """
        for x in np.nditer(Image_new):
            if(x > 1):
                x = 1
            if(x < 0):
                x = 0
        """
        plt.subplot(131),plt.imshow(img1)
        plt.subplot(132),plt.imshow(img2)
        #print(type(img1[0][0][0]))
        plt.subplot(133),plt.imshow((np.abs(Image_new)).astype(np.float32))
        plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()