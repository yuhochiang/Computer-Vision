import cv2
import numpy as np
import math
import glob
import os
import HW2_Part1 as ref
from matplotlib import pyplot as plt

def pyramid(img):
	dft = ref.fourier_transform(img)
	guassian_dft = ref.gaussian_filter(dft, "lowpass")
	Smooth_img = ref.inv_ft(guassian_dft)
	return Smooth_img[::2, ::2]

def magnitudeSpectrum(img):
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)
	magnitude_spectrum = 20*np.log(np.abs(fshift))
	return magnitude_spectrum

def GaussianPyramid(img):
	Img_G = [img]
	Img_G_mag_spectrum =[magnitudeSpectrum(img)]

	for i in range(5):
		img = pyramid(img)
		Img_G.append(img)
		mag_spectrum = magnitudeSpectrum(img)
		Img_G_mag_spectrum.append(mag_spectrum)

	return Img_G, Img_G_mag_spectrum

def LaplacianPyramid(Img_G):
	Img_L = []
	Img_L_mag_spectrum = []
	for i in range(5):
		Upsample = Img_G[i+1].repeat(2, axis=0)
		Upsample = Upsample.repeat(2, axis=1)
		dft = ref.fourier_transform(Upsample)
		guassian_dft = ref.gaussian_filter(dft, "lowpass")
		Smooth_img = ref.inv_ft(guassian_dft)
		
		# to ensure Img_G.shape == Smooth_img
		if (Img_G[i].shape[0]%2 == 0) and (Img_G[i].shape[1]%2 == 0):
			laplacian = Img_G[i] - Smooth_img
		elif (Img_G[i].shape[0]%2 == 0) and (Img_G[i].shape[1]%2 != 0):
			laplacian = Img_G[i] - Smooth_img[:, :-1]
		elif (Img_G[i].shape[0]%2 != 0) and (Img_G[i].shape[1]%2 == 0):
			laplacian = Img_G[i] - Smooth_img[:-1, :]
		else:
			laplacian = Img_G[i] - Smooth_img[:-1, :-1]

		Img_L.append(laplacian)
		Img_L_mag_spectrum.append(magnitudeSpectrum(laplacian))
	return Img_L, Img_L_mag_spectrum

if __name__ == '__main__':
	images=[cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in glob.glob('./hw2_data/task1,2_hybrid_pyramid/*')]

	if not os.path.isdir('./task2_output'):
		os.mkdir('./task2_output')

	for j, img in enumerate(images):
		Img_G, Img_G_mag_spectrum = GaussianPyramid(img)
		Img_L, Img_L_mag_spectrum = LaplacianPyramid(Img_G)
		for i in range(5):
			plt.subplot(4, 5, i+1)
			plt.imshow(Img_G[i],cmap = 'gray')

			plt.subplot(4, 5, i+6)
			plt.imshow(Img_G_mag_spectrum[i],cmap = 'gray')

			plt.subplot(4, 5, i+11)
			plt.imshow(Img_L[i],cmap = 'gray')

			plt.subplot(4, 5, i+16)
			plt.imshow(Img_L_mag_spectrum[i],cmap = 'gray')

		plt.savefig('./task2_output/'+str(j)+'.jpg')
		#plt.show()
