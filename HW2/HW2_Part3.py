import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import time
import os

from HW2_Part2 import pyramid

# generate 2 pyramid lists with the same length
def getPyramidList(img1, img2):
	pyramidList1 = [img1]
	while img1.shape[0] >= 32 or img1.shape[1] >= 32:
		#img1 = cv2.pyrDown(img1)
		img1 = pyramid(img1)
		pyramidList1.append(img1)

	pyramidList2 = [img2]
	for i in range(len(pyramidList1)-1):
		#img2 = cv2.pyrDown(img2)
		img2 = pyramid(img2)
		pyramidList2.append(img2)
	return pyramidList1, pyramidList2

def NCC(img, templ):
	imgMean = img - img.mean()
	templMean = templ - templ.mean()
	ncc = np.sum(imgMean * templMean)
	ncc /= np.linalg.norm(imgMean) * np.linalg.norm(templMean)
	return ncc

def align(image, templImage):
	# only use central part of templImage
	x, y = templImage.shape
	templImage = templImage[int(0.1*x):int(0.9*x), int(0.1*y):int(0.9*y)]
	templOriginalX, templOriginalY = int(0.1*x), int(0.1*y)

	imgPyramidList, templPyramidList = getPyramidList(image, templImage)
	imgPyramidList = imgPyramidList[::-1]
	templPyramidList = templPyramidList[::-1]

	maxNCC = 0
	matchPos = [0, 0]

	# first layer -> exhaustive search
	img = imgPyramidList[0]
	templ = templPyramidList[0]
	h, w = templ.shape

	for i in range(img.shape[0] - h):
		for j in range(img.shape[1] - w):
			ncc = NCC(img[i:i+h, j:j+w], templ)
			if  ncc > maxNCC:
				maxNCC = ncc
				matchPos = [i, j]
			#print('ij', i, j, ssd, matchPos)

	# other layers -> use result of upper layers
	for i in range(1, len(templPyramidList)):
		img = imgPyramidList[i]
		templ = templPyramidList[i]
		h, w = templ.shape
		maxNCC = 0

		# only check 3*3 pixel around matchPos
		currentMatchPos = [2 * matchPos[0], 2 * matchPos[1]]
		x = 0 if currentMatchPos[0] == 0 else currentMatchPos[0]-1
		y = 0 if currentMatchPos[1] == 0 else currentMatchPos[1]-1

		for j in range(x, currentMatchPos[0]+2):
			for k in range(y, currentMatchPos[1]+2):
				ncc = NCC(img[j:j+h, k:k+w], templ)
				if ncc > maxNCC:
					maxNCC = ncc
					matchPos = [j, k]
				#print('jk', j, k, ssd, matchPos)
	return [matchPos[0] - templOriginalX, matchPos[1] - templOriginalY]

if __name__ == '__main__':
	images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in glob.glob('hw2_data/task3_colorizing/*')]

	if not os.path.isdir('task3_output'):
		os.mkdir('task3_output')

	for i, img in enumerate(images):
		start = time.time()
		h, w = img.shape
		img = img[:, int(0.02*w):int(0.98*w)]
		h = int(img.shape[0] * 1/3)
		B = img[:h, :]
		G = img[h:2*h, :]
		R = img[2*h:3*h, :]

		# oringinal image
		mix = np.dstack((R, G, B))
		plt.imsave('./task3_output/'+str(i)+'.jpg', mix)

		# align B and R to G
		shiftB = align(G, B)
		shiftR = align(G, R)

		# shift the image
		B = np.roll(B, shiftB, axis=(0, 1))
		R = np.roll(R, shiftR, axis=(0, 1))
		
		end = time.time()
		print("Blue:", shiftB, ", Red:", shiftR, ", time:", round(end-start, 3), "s")

		# output image after alignment
		mix = np.dstack((R, G, B))
		plt.imsave('./task3_output/'+str(i)+'_align.jpg', mix)

