import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import random
import math
import os

def SIFT(img):
	sift = cv2.SIFT_create()
	# find keypoint and descriptor
	kp, des = sift.detectAndCompute(img, None)
	#keypointImg = cv2.drawKeypoints(img, kp, img)
	return kp, des

def ORB(img):
	orb = cv2.ORB_create()
	kp, des = orb.detectAndCompute(img, None)
	return kp, des

def MSE(x, y):
	re = np.square(x - y)
	return np.sum(re)**0.5

def ratioDistance(des1, des2):
	ratio_distanceList = []
	for i in range(des1.shape[0]):
		L2_distanceList = []
		for j in range(des2.shape[0]):
			L2_distanceList.append((j, MSE(des1[i], des2[j])))
		L2_distanceList.sort(key = lambda x : x[1])
		ratioDist = L2_distanceList[0][1] / L2_distanceList[1][1]
		ratio_distanceList.append((L2_distanceList[0][0], ratioDist))
	return ratio_distanceList

def drawFeatureMatch(img1, kp1, img2, kp2, good):
	img = np.concatenate((img1, img2), axis=1)
	for i, j in good:
		pt1 = (int(kp1[i].pt[0]), int(kp1[i].pt[1]))
		pt2 = (int(kp2[j].pt[0])+img1.shape[1], int(kp2[j].pt[1]))
		color = np.random.randint(0, 255, size=3)
		cv2.line(img, pt1, pt2, color.tolist(), 1)

	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img

def findHomography(points1, points2):
	# P.shape = (2n, 9)
	P = np.zeros((2*points1.shape[0], 9))
	# Pm = 0
	for i in range(points1.shape[0]):
		P[i*2, :] = [points1[i, 0], points1[i, 1], 1, 0, 0, 0, -points1[i, 0]*points2[i, 0], -points1[i, 1]*points2[i, 0], -points2[i, 0]]
		P[i*2+1, :] = [0, 0, 0, points1[i, 0], points1[i, 1], 1, -points1[i, 0]*points2[i, 1], -points1[i, 1]*points2[i, 1], -points2[i, 1]]

	u, s, vT = np.linalg.svd(P, full_matrices=0)
	# last column of v -> last row of vT
	H = vT[-1, :].reshape(3, 3)
	# to normalize H -> H[2, 2] = 1
	H /= H[2, 2]
	return H

def RANSAC_findH(points1, points2):
	# parameter
	iters = 10000
	error_threshold = 5.0

	number_of_point = len(points1)
	best_Homography = np.zeros((3,3))
	best_total_inlier = 0
	p = 0.95
	for i in range(iters):
		sample_src = []
		sample_dst = []
		total_inlier = 0

		sample_index = random.sample(range(number_of_point),4)
		sample_src.append(points1[sample_index])
		sample_dst.append(points2[sample_index])
		sample_src = np.float32(sample_src).reshape(-1,2)
		sample_dst = np.float32(sample_dst).reshape(-1,2)
		H = findHomography(sample_src, sample_dst)
		row_vec_one = np.ones((1,number_of_point))
		homogenous_coordinate = np.r_[points1.transpose(), row_vec_one]
		homo_src_proj = H.dot(homogenous_coordinate)
		last_row = homo_src_proj[2]
		src_proj = ((homo_src_proj *( 1./ last_row.T))[:2]).T
		for i, point in enumerate(src_proj):
			error = MSE(point,points2[i])
			if error < error_threshold:
				total_inlier+=1
		if total_inlier > best_total_inlier:
			if(total_inlier / number_of_point ) > 0 :
				iters = math.log(1 - p) / math.log(1 - pow(total_inlier / number_of_point, 4))
			best_total_inlier = total_inlier
			best_Homography = H
		
	return best_Homography

def findBoundary(H, left, right):
	h, w, _ = left.shape
	corners_l = np.array([[0, 0], [w, 0], [0, h], [w, h]]).astype(np.float32) # x-y coord
	corners_l_homo = np.concatenate((corners_l, np.ones((4, 1), np.float32)), axis=1).transpose()
	corners_r_homo = H.dot(corners_l_homo).transpose()
	scale_vector = corners_r_homo[:,2]
	corners_r = np.around((corners_r_homo / scale_vector[:,None])[:,:2]).astype(int)
	h, w, _ = right.shape
	x_min = (0 - corners_r[:, 0].min()) if corners_r[:, 0].min() < 0 else 0
	x_max = (corners_r[:, 0].max() - w) if corners_r[:, 0].max() > w else 0
	y_min = (0 - corners_r[:, 1].min()) if corners_r[:, 1].min() < 0 else 0
	y_max = (corners_r[:, 1].max() - h) if corners_r[:, 1].max() > h else 0

	return int(x_min), int(x_max), int(y_min), int(y_max)


def drawPano(pano, left, right, Coord_l, more):
	h,w,_ = pano.shape

	xline = np.arange(w)
	yline = np.arange(h)
	X, Y = np.meshgrid(xline, yline)
	Coord_pano = np.stack((X.astype(int), Y.astype(int)), axis=2).reshape((-1, 2))
	PL = np.zeros(pano.shape, np.float32)
	PR = np.zeros(pano.shape, np.float32)

	# draw PR
	more_h, more_w = more
	PR[more_h: more_h+right.shape[0], more_w: more_w+right.shape[1], :] = right

	# draw PL
	rh, rw, _ = left.shape
	for i, (x,y) in enumerate(Coord_pano):
		xr, yr = Coord_l[i]
		if xr < 0 or xr >= rw-1 or yr < 0 or yr >= rh-1:
			continue
		xr_g, yr_g = np.floor(Coord_l[i])
		xr_c, yr_c = np.ceil(Coord_l[i])
		wx1 = xr - xr_g
		wx2 = xr_c - xr
		wy1 = yr - yr_g
		wy2 = yr_c - yr
		gg = left[int(yr_g), int(xr_g), :]
		cg = left[int(yr_c), int(xr_g), :]
		gc = left[int(yr_g), int(xr_c), :]
		cc = left[int(yr_c), int(xr_c), :]

		if (gg==0).all() or (cg==0).all() or (gc==0).all() or (cc==0).all():
			continue
		PL[y, x, :] = (wx1*wy1*gg + wx1*wy2*cg + wx2*wy1*gc + wx2*wy2*cc)
	# Linear blending
	mask_PL = PL != 0
	mask_PR = PR != 0
	mask_PL_and_PR = np.logical_and(mask_PL, mask_PR)
	mask_only_PL = np.logical_xor(mask_PL, mask_PL_and_PR)
	mask_only_PR = np.logical_xor(mask_PR, mask_PL_and_PR)
	pano = (PL*mask_only_PL + PR*mask_only_PR + ((PL+PR)/2)*mask_PL_and_PR).astype(np.uint8)

	return pano

def warp(left, right, H):
	x_min, x_max, y_min, y_max = findBoundary(H, left, right)
	h, w, _ = right.shape
	h_max = max(y_min, y_max) + 15
	w_max = max(x_min, x_max) + 15
	pano_h = 2 * h_max + h
	pano_w = 1 * w_max + w + 15
	pano = np.zeros((pano_h, pano_w, 3), np.uint8)

	xline = np.arange(pano_w)
	yline = np.arange(pano_h)
	X, Y = np.meshgrid(xline - w_max, yline - h_max)
	Coord_pano_in_R = np.stack((X.astype(np.float32), Y.astype(np.float32)), axis=2).reshape((-1, 2))
	Coord_pano_homo = np.transpose(np.concatenate((Coord_pano_in_R, np.ones((pano_h*pano_w, 1), np.float32)), axis=1))
	Coord_pano_in_L_homo = np.transpose(np.linalg.inv(H).dot(Coord_pano_homo))
	scale = Coord_pano_in_L_homo[:, 2]
	Coord_pano_in_L = (Coord_pano_in_L_homo / scale[:, None])[:, :2]
	pano = drawPano(pano, left, right, Coord_pano_in_L, (h_max, w_max))

	return pano

if __name__ == '__main__':
	images = [cv2.imread(img) for img in glob.glob('./data/*')]
	if not os.path.isdir('output'):
		os.mkdir('output')
	for i in range(0, len(images), 2):
		kp1, des1 = SIFT(images[i])
		kp2, des2 = SIFT(images[i+1])
		#kp1, des1 = ORB(images[i])
		#kp2, des2 = ORB(images[i+1])

		# calculate distance between feature point
		ratio_distanceList = ratioDistance(des1, des2)
		# do ratio test
		good = []
		for j, match in enumerate(ratio_distanceList):
			if match[1] < 0.75:
				good.append((j, match[0]))
		# visualize feature matching
		matchImg = drawFeatureMatch(images[i], kp1, images[i+1], kp2, good)
		plt.imsave('./output/'+str(i)+'_match.jpg', matchImg)
		#plt.imsave('./output/'+str(i)+'_match_ORB.jpg', matchImg)
		#plt.imshow(matchImg)
		#plt.show()

		# RANSAC
		if len(good) > 10:
			src = []
			dst = []
			for pair in good:
				src.append(kp1[pair[0]].pt)
				dst.append(kp2[pair[1]].pt)
			src = np.float32(src).reshape(-1,2)
			dst = np.float32(dst).reshape(-1,2)
			H = findHomography(src, dst)
			
			H2 = RANSAC_findH(src, dst)
			print('Homography Mat(no use RANSAC) :\n', H)
			print('Homography Mat(our RANSAC) :\n', H2)
			H3, _ = cv2.findHomography(src,dst,cv2.RANSAC,5.0)
			print('Homography Mat(cv2 RANSAC) :\n', H3)
		# Warping image
		pano = warp(images[i], images[i+1], H2)
		pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
		plt.imsave('./output/'+str(i)+'_pano.jpg', pano)
		#plt.imsave('./output/'+str(i)+'_pano_ORB.jpg', pano)
		#plt.imshow(pano)
		#plt.show()



