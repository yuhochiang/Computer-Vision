import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os

def load_img_K(name):
	if name == 'Mesona':
		# first image set -> intrinsic matrices for both images are the same
		img1 = cv2.imread('./Mesona1.JPG')
		img2 = cv2.imread('./Mesona2.JPG')

		K1 = np.array([[1.4219, 0.0005, 0.5092],
					   [0,      1.4219, 0.3802],
					   [0,      0,      0.0010]])
		K1 = K1 / K1[2, 2]
		K2 = K1
	elif name == 'Statue':
		img1 = cv2.imread('./Statue1.bmp')
		img2 = cv2.imread('./Statue2.bmp')
		K1 = np.array([[5426.566895,    0.678017, 330.096680],
					   [   0.000000, 5423.133301, 648.950012],
					   [   0.000000,    0.000000,   1.000000]])
		K2 = np.array([[5426.566895,    0.678017, 387.430023],
					   [   0.000000, 5423.133301, 620.616699],
					   [   0.000000,    0.000000,   1.000000]])
	elif name == 'our':
		img1 = cv2.imread('./our1.jpg')
		img2 = cv2.imread('./our2.jpg')
		# use camera_calibration.py in HW1 to find K
		K1 = np.array([[3400.33701, -35.2543658, 1475.68441],
					   [   0.00000,  3349.35946, 1408.22869],
					   [   0.00000, 	0.00000,    1.00000]])
		K2 = K1

	return img1, img2, K1, K2

def SIFT(img):
	sift = cv2.SIFT_create()
	# find keypoint and descriptor
	kp, des = sift.detectAndCompute(img, None)
	keypointImg = cv2.drawKeypoints(img, kp, img)
	return kp, des, keypointImg

def featureMatching(kp1, kp2, des1, des2):
	x1 = []
	x2 = []
	for i in range(des1.shape[0]):
		L2_distanceList = []
		for j in range(des2.shape[0]):
			dist = np.sum(np.square(des1[i]-des2[j])) ** 0.5
			L2_distanceList.append((j, dist))
		L2_distanceList.sort(key = lambda x : x[1])
		ratioDist = L2_distanceList[0][1] / L2_distanceList[1][1]
		# do ratio test
		if  ratioDist < 0.75:
			x1.append(kp1[i].pt)
			x2.append(kp2[L2_distanceList[0][0]].pt)
	return np.float32(x1).reshape(-1, 2), np.float32(x2).reshape(-1, 2)

def normalize(points1, points2, shape1, shape2):
	# transform image to [-1, 1] x [-1, 1]
	x1 = points1.transpose()
	x2 = points2.transpose()
	T1 = np.array([[2/shape1[1],           0, -1],
				   [          0, 2/shape1[0], -1],
				   [          0,           0,  1]])
	T2 = np.array([[2/shape2[1],           0, -1],
				   [          0, 2/shape2[0], -1],
				   [          0,           0,  1]])

	# (n, 3)
	x1 = T1.dot(x1).transpose()
	x2 = T2.dot(x2).transpose()
	return x1, x2, T1, T2

def inliers_findF(mask, x1, x2, T1, T2):
	# x1, x2 have already been normalized
	x1 = x1[mask.flatten() == 1]
	x2 = x2[mask.flatten() == 1]
	A = np.zeros((x1.shape[0], 9))
	for i in range(x1.shape[0]):
		A[i, :] = [x1[i, 0]*x2[i, 0], x1[i, 0]*x2[i, 1], x1[i, 0],
				   x1[i, 1]*x2[i, 0], x1[i, 1]*x2[i, 1], x1[i, 1],
				   			x2[i, 0], 		   x2[i, 1],	   1]
	u, s, vT = np.linalg.svd(A)
	F = vT[-1].reshape(3, 3)
	# resolve det(F) = 0 constraint using SVD
	u, s, vT = np.linalg.svd(F)
	# s.shape = (3, )
	s[2] = 0
	F = u.dot(np.diag(s)).dot(vT)
	F = T1.transpose().dot(F).dot(T2)
	return F / F[2, 2]

def RANSAC_findF(points1, points2, shape1, shape2):
	# parameter
	iters = 2000
	error_threshold = 1
	best_Fundamental = np.zeros((3,3)).astype(np.longfloat)
	best_total_inlier = 0
	number_of_point = points1.shape[0]
	points1 = np.concatenate((points1, np.ones((points1.shape[0], 1))), axis=1)
	points2 = np.concatenate((points2, np.ones((points2.shape[0], 1))), axis=1)
	x1, x2, T1, T2 = normalize(points1, points2, shape1, shape2)
	for i in range(iters):
		total_inlier = 0
		sample_index = random.sample(range(number_of_point),8)
		A = np.zeros((8, 9))
		for j, k in enumerate(sample_index):
			A[j, :] = [x1[k, 0]*x2[k, 0], x1[k, 0]*x2[k, 1], x1[k, 0],
					   x1[k, 1]*x2[k, 0], x1[k, 1]*x2[k, 1], x1[k, 1],
					   			x2[k, 0], 		   x2[k, 1],	   1]
		# SVD
		u, s, vT = np.linalg.svd(A)
		F = vT[-1].reshape(3, 3)
		# resolve det(F) = 0 constraint using SVD
		u, s, vT = np.linalg.svd(F)
		# s.shape = (3, )
		s[2] = 0
		F = u.dot(np.diag(s)).dot(vT)
		# denormalize
		F = T1.transpose().dot(F).dot(T2)
		mask = []
		for j in range(x1.shape[0]):
			# sampson distance
			Fx1 = F.transpose().dot(points1[j].transpose())
			Fx2 = F.dot(points2[j].transpose())
			dist = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
			error = (points1[j].dot(F).dot(points2[j].transpose()))**2 / dist

			if error < error_threshold:
				total_inlier += 1
				mask.append(1)
			else:
				mask.append(0)
		if total_inlier > best_total_inlier:
			best_total_inlier = total_inlier
			print(best_total_inlier)
			best_Fundamental = F
			best_mask = np.array(mask)

	best_Fundamental = inliers_findF(best_mask, x1, x2, T1, T2)
	
	return best_Fundamental, best_mask

def find_epipolarLines(pts2, F):
	# l = Fx', l = F^T x
	pts2_homogenous = np.concatenate((pts2, np.ones((pts2.shape[0], 1))), axis=1).transpose()
	lines = F.transpose().dot(pts2_homogenous).transpose()
	# lines: (a, b, c) -> to represent ax+by+c = 0
	# normalized s.t. a^2 + b^2 = 1
	a = lines[:, 0:1]
	b = lines[:, 1:2]
	norm = np.sqrt(a**2 + b**2)
	lines = lines / norm
	return lines

def drawEpipolar(img1, img2, lines, pts1, pts2):
	# draw lines on img1
	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)
	c = img1.shape[1]
	copy1 = img1.copy()
	copy2 = img2.copy()
	for r, pt1, pt2 in zip(lines, pts1, pts2):
		color = tuple(np.random.randint(0,255,3).tolist())
		x0, y0 = map(int, [0, -r[2]/r[1]])
		x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
		copy1 = cv2.line(copy1, (x0,y0), (x1,y1), color, 1)
		copy1 = cv2.circle(copy1, tuple(pt1), 5, color, -1)
		copy2 = cv2.circle(copy2, tuple(pt2), 5, color, -1)
	copy1 = cv2.cvtColor(copy1, cv2.COLOR_BGR2RGB)
	copy2 = cv2.cvtColor(copy2, cv2.COLOR_BGR2RGB)
	return copy1, copy2

def find_P2(E):
	u, s, vT = np.linalg.svd(E)
	m = (s[0] + s[1]) / 2
	E = u.dot(np.diag([m, m, 0])).dot(vT)
	u, s, vT = np.linalg.svd(E)
	if np.linalg.det(u.dot(vT)):
		vT = -vT
	W = np.array([[0, -1, 0],
				  [1,  0, 0],
				  [0,  0, 1]])
	# 4 possible solution
	P2_1 = np.concatenate((u.dot(W).dot(vT), u[:, 2:]), axis=1)
	P2_2 = np.concatenate((u.dot(W).dot(vT), -u[:, 2:]), axis=1)
	P2_3 = np.concatenate((u.dot(W.transpose()).dot(vT), u[:, 2:]), axis=1)
	P2_4 = np.concatenate((u.dot(W.transpose()).dot(vT), -u[:, 2:]), axis=1)
	# do triangulation and find out the most appropiate solution of E
	return [P2_1, P2_2, P2_3, P2_4]

def triangulation(P1, P2, K1, K2, points1, points2):
	x1 = np.concatenate((points1, np.ones((points1.shape[0], 1))), axis=1) # (n, 3)
	x2 = np.concatenate((points2, np.ones((points2.shape[0], 1))), axis=1)
	points_3D = np.zeros((x1.shape[0], 4))
	P1 = K1.dot(P1)
	P2 = K2.dot(P2)
	for i in range(x1.shape[0]):
		A = np.zeros((4, 4))
		A[0, :] = x1[i, 0]*P1[2, :] - P1[0, :]
		A[1, :] = x1[i, 1]*P1[2, :] - P1[1, :]
		A[2, :] = x2[i, 0]*P2[2, :] - P2[0, :]
		A[3, :] = x2[i, 1]*P2[2, :] - P2[1, :]
		u, s, vT = np.linalg.svd(A)
		X = vT[-1, :] / vT[-1, -1]
		points_3D[i, :] = X
	return points_3D

def find_best_P2(points_3D_1, points_3D_2, points_3D_3, points_3D_4, P2_1, P2_3):
	R1 = P2_1[:, 0:3]
	R2 = P2_3[:, 0:3]
	t = P2_1[:, 3:]
	# C = -R^T t  (3, 1)
	C1 = -R1.transpose().dot(t)
	C2 = R1.transpose().dot(t)
	C3 = -R2.transpose().dot(t)
	C4 = R2.transpose().dot(t)
	# test (X-C).R[2, :]^T > 0
	countList = []
	print(points_3D_1.shape)
	x = R1[2, :].dot(points_3D_1.transpose()[:3, :] - C1)
	y = points_3D_1.transpose()[2, :] > 0
	countList.append(np.count_nonzero((x > 0) & y))
	x = R1[2, :].dot(points_3D_2.transpose()[:3, :] - C2)
	y = points_3D_2.transpose()[2, :] > 0
	countList.append(np.count_nonzero((x > 0) & y))
	x = R2[2, :].dot(points_3D_3.transpose()[:3, :] - C3)
	y = points_3D_3.transpose()[2, :] > 0
	countList.append(np.count_nonzero((x > 0) & y))
	x = R2[2, :].dot(points_3D_4.transpose()[:3, :] - C4)
	y = points_3D_4.transpose()[2, :] > 0
	countList.append(np.count_nonzero((x > 0) & y))
	print(countList)
	return np.argmax(countList)

if __name__ == '__main__':
	if not os.path.isdir('output'):
		os.mkdir('output')
	dataName = 'Mesona'
	# dataName = 'Statue'
	# dataName = 'our'
	img1, img2, K1, K2 = load_img_K(dataName)
	kp1, des1, keypointImg1 = SIFT(img1)
	kp2, des2, keypointImg2 = SIFT(img2)

	# calculate distance between feature point and find match points
	points1, points2 = featureMatching(kp1, kp2, des1, des2)
	print(points1.shape)

	F_cv, mask_cv = cv2.findFundamentalMat(points1, points2,cv2.FM_LMEDS)
	print("cv2 Fundamental matrix:\n", F_cv)

	# RANSAC with 8-point algo.
	F, mask = RANSAC_findF(points1, points2, img1.shape, img2.shape)
	F = F.transpose()

	# draw epipolar lines
	points1 = points1[mask.flatten() == 1]
	points2 = points2[mask.flatten() == 1]
	lines1 = find_epipolarLines(points2, F)
	left, right = drawEpipolar(img1, img2, lines1, points1, points2)
	plt.subplot(221), plt.imshow(left), plt.subplot(222), plt.imshow(right)
	lines2 = find_epipolarLines(points1, F.transpose())
	right, left = drawEpipolar(img2, img1, lines2, points2, points1)
	plt.subplot(223), plt.imshow(left), plt.subplot(224), plt.imshow(right)
	plt.savefig('./output/'+dataName+'.jpg')
	#plt.show()

	# draw epipolar lines by cv2
	lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F_cv)
	lines1 = lines1.reshape(-1,3)
	left, right = drawEpipolar(img1, img2, lines1, points1, points2)
	plt.subplot(221), plt.imshow(left), plt.subplot(222), plt.imshow(right)
	lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F_cv)
	lines2 = lines2.reshape(-1,3)
	right, left = drawEpipolar(img2, img1, lines2, points2, points1)
	plt.subplot(223), plt.imshow(left), plt.subplot(224), plt.imshow(right)
	plt.savefig('./output/'+dataName+'_cv.jpg')
	#plt.show()
	
	print("our Fundamental matrix:\n", F)
	# calculate essensial matrix
	E = K1.transpose().dot(F).dot(K2)
	u, s, vT = np.linalg.svd(E)
	I = np.diag([1.0, 1.0, 0.0])
	E = u.dot(I).dot(vT)
	P1 = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
	print("Essential matrix:\n", E)
	# find second camera matrix
	P2List = find_P2(E)
	points_3D_List = []
	points_3D_List.append(triangulation(P1, P2List[0], K1, K2, points1, points2))
	points_3D_List.append(triangulation(P1, P2List[1], K1, K2, points1, points2))
	points_3D_List.append(triangulation(P1, P2List[2], K1, K2, points1, points2))
	points_3D_List.append(triangulation(P1, P2List[3], K1, K2, points1, points2))
	idx = find_best_P2(points_3D_List[0], points_3D_List[1], points_3D_List[2], points_3D_List[3], P2List[0], P2List[2])

	np.savetxt('./output/'+dataName+'_3D_points.csv', points_3D_List[idx], delimiter=",")
	np.savetxt('./output/'+dataName+'_2D_points.csv', points1, delimiter=",")
	np.savetxt('./output/'+dataName+'_camera_matrix.csv', K1.dot(P1), delimiter=",")
	