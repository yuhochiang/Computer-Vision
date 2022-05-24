clear all;

dataName = 'Mesona'
%dataName = 'Statue'
%dataName = 'our'

points_3D = csvread(['./output/' dataName '_3D_points.csv']);
points_2D = csvread(['./output/' dataName '_2D_points.csv']);
CameraMatrix = csvread(['./output/' dataName '_camera_matrix.csv']);

CameraMatrix

obj_main(points_3D(:,1:3), points_2D(:,1:2), CameraMatrix, 'Mesona1.JPG', 1)
%obj_main(points_3D(:,1:3), points_2D(:,1:2), CameraMatrix, 'Statue1.bmp', 1)
%obj_main(points_3D(:,1:3), points_2D(:,1:2), CameraMatrix, 'our1.jpg', 1)