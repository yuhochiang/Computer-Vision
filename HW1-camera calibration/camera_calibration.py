import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2) # two arrays of corner_x*corner_y

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data/*.jpg')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)


#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = (img.shape[1], img.shape[0])
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
"""
"""
Write your code here
"""
# 1.Use the points in each images to find Hi
H_list = []
for i in range(len(objpoints)):
    # P.shape = (2n, 9)
    P = np.zeros((2*corner_x*corner_y, 9))
    obj = objpoints[i] # (49, 3)
    img = imgpoints[i].reshape(-1, 2) # (49, 2)
    # Pm = 0
    for j in range(corner_x*corner_y):
        P[j*2, :] = [obj[j][0], obj[j][1], 1, 0, 0, 0, -img[j][0]*obj[j][0], -img[j][0]*obj[j][1], -img[j][0]]
        P[j*2+1, :] = [0, 0, 0, obj[j][0], obj[j][1], 1, -img[j][1]*obj[j][0], -img[j][1]*obj[j][1], -img[j][1]]

    u, s, vT = np.linalg.svd(P, full_matrices=0)
    # last column of v -> last row of vT
    H = vT[-1, :].reshape(3, 3)
    # to normalize H -> H[2, 2] = 1
    H /= H[2, 2]
    H_list.append(H)
# 2.Use Hi to find out the intrinsic matrix K
# V.shape = (2n, 6)
V = np.zeros((2*len(objpoints), 6))
# Vb = 0
for i, H in enumerate(H_list):
    V[i*2, :] = [ H[0, 0]*H[0, 1], H[0, 0]*H[1, 1]+H[0, 1]*H[1, 0], H[0, 0]*H[2, 1]+H[0, 1]*H[2, 0],
                  H[1, 0]*H[1, 1], H[1, 0]*H[2, 1]+H[1, 1]*H[2, 0], H[2, 0]*H[2, 1] ]
    V[i*2+1, :] = [ H[0, 0]**2-H[0, 1]**2, 2*(H[0, 0]*H[1, 0]-H[0, 1]*H[1, 1]), 2*(H[0, 0]*H[2, 0]-H[0, 1]*H[2, 1]),
                    H[1, 0]**2-H[1, 1]**2, 2*(H[1, 0]*H[2, 0]-H[1, 1]*H[2, 1]), H[2, 0]**2-H[2, 1]**2 ]

u, s, vT = np.linalg.svd(V, full_matrices=0)
# b = (b11, b12, b13, b22, b23, b33)
b = vT[-1, :]

B = np.array([ [b[0], b[1], b[2]],
               [b[1], b[3], b[4]],
               [b[2], b[4], b[5]] ])
# B = K^(-T) K^(-1), B must be symmetric and postive definite
if B[0, 0] < 0:
    B *= -1
A = np.linalg.cholesky(B) # B = A*(A^T) -> A = K^(-T)
mtx = np.linalg.inv(A.T)
# The bottom right corner of K equals to 1
mtx /= mtx[2, 2]
print("----------our intrinsic----------")
print(mtx)
print("---------------------------------")

# 3.Find out the extrensics matrix of each images
# r1 = lambda * K^(-1) * h1
# r2 = lambda * K^(-1) * h2
# r3 = r1 x r2
# t = lambda * K^(-1) * h3
# lambda = 1 / || K^(-1)*h1 || 

extrinsics = np.zeros((len(objpoints), 3, 4))
K_inv = np.linalg.inv(mtx)
for i, H in enumerate(H_list):
    l = 1 / np.linalg.norm( K_inv.dot(H[:, 0]) )
    r1 = l * K_inv.dot(H[:, 0])
    r2 = l * K_inv.dot(H[:, 1])
    r3 = np.cross(r1, r2).reshape(3, 1)
    t = (l * K_inv.dot(H[:, 2])).reshape(3, 1)
    ex = np.concatenate((r1.reshape(3, 1), r2.reshape(3, 1), r3, t), axis=1)
    extrinsics[i, :, :] = ex

# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
camera_matrix = mtx
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""
