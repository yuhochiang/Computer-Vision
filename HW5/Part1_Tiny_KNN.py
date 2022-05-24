import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os

def load_data(path, classList, resize=False, normalize=False):
	labelList = []
	imgList = []
	for c in classList:
		for imgPath in glob.glob(path + '/' + c + '/*.jpg'):
			labelList.append(c)
			img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
			if resize:
				img = cv2.resize(img, (16, 16))
			if normalize:
				mean = np.mean(img)
				std = np.std(img)
				img = (img - mean) / std
			imgList.append(img.flatten())

	return imgList, labelList

def MSE(x, y):
	re = np.square(x - y)
	return np.sum(re)**0.5

def accuracy(result, label):
	count = 0
	for i in range(len(result)):
		if result[i] == label[i]:
			count += 1
	return count / len(result) 

def KNN(train, label_train, test, label_test, K, classList):
	test_result = []
	for test_img in test:
		vote = {}
		for c in classList:
			vote[c] = 0
		distList = []
		# calculate distance between test sample and training smaple
		for i in range(len(train)):
			dist = MSE(train[i], test_img)
			distList.append((i, dist))
		# sort distList to find k nearest neighbor
		distList.sort(key = lambda x : x[1])
		for i in range(K):
			# neighbor's label
			label = label_train[distList[i][0]]
			vote[label] += 1
		# find the result with the most votes
		test_result.append(max(vote, key=vote.get))

	return accuracy(test_result, label_test)		

if __name__ == '__main__':
	classList = os.listdir('./hw5_data/train')
	#print(classList)
	# load img for training and testing, and images are resized and not normalize
	imgList_train, labelList_train = load_data('./hw5_data/train', classList, True, False)
	imgList_test, labelList_test = load_data('./hw5_data/test', classList, True, False)

	test_acc = []
	for i in range(1, 21):
		acc = KNN(imgList_train, labelList_train, imgList_test, labelList_test, i, classList)
		test_acc.append(acc)

	# images are normalized
	imgList_train, labelList_train = load_data('./hw5_data/train', classList, True, True)
	imgList_test, labelList_test = load_data('./hw5_data/test', classList, True, True)

	test_acc_norm = []
	for i in range(1, 21):
		acc = KNN(imgList_train, labelList_train, imgList_test, labelList_test, i, classList)
		test_acc_norm.append(acc)

	print('max accuracy without normalization: ' + str(max(test_acc)*100) + '%')
	print('max accuracy with normalization: ' + str(max(test_acc_norm)*100) + '%')

	plt.plot(range(1,21), test_acc, '-bo', label='w/o normalization')
	plt.plot(range(1,21), test_acc_norm, '-ro', label='w normalization')
	plt.xticks(range(1, 21, 1))
	plt.title('Tiny image representation + KNN')
	plt.xlabel('K')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show()
	