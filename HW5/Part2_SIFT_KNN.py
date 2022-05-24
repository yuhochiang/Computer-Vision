import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
import cyvlfeat

from cyvlfeat.kmeans import kmeans,kmeans_quantize
from cyvlfeat.sift import dsift

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
            imgList.append(img)

    return imgList, labelList

def get_clusters(imgs, cluster_size): # 找到clusters的中心
    bag = []
    for img in imgs:
        keypoints, descriptors = dsift(img, fast=False, step=10)
        if bag==[]:
            bag = descriptors # (1, 128)
        else:
            bag = np.vstack((bag, descriptors)).astype('float')
    clusters = kmeans(bag, cluster_size) # (cluster_size, 128)

    return clusters

def bag_of_sift(imgs, clusters):
    histogram = np.zeros((len(imgs), clusters.shape[0])) # clusters.shape[0] = cluster_size
    for i in range(len(imgs)):
        keypoints, descriptors = dsift(imgs[i] ,fast=True, step=10)
        descriptors = descriptors.astype(np.float64)
        assignments = kmeans_quantize(descriptors, clusters) # 把這些descriptors分類到這些clusters
        cluster_idx, counts = np.unique(assignments, return_counts=True) # 計算每張照片的descriptors中each cluster的數量
        counts = counts.astype(np.float64)
        histogram[i, cluster_idx] = counts/np.sum(counts)  

    return histogram

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

    # images w/o resize & w/o normalization
    imgList_train, labelList_train = load_data('./hw5_data/train', classList, False, False)
    imgList_test, labelList_test = load_data('./hw5_data/test', classList, False, False)

    # cluster size = 300
    cluster_size = 300
    clusters = get_clusters(imgList_train, cluster_size)

    feature_train = bag_of_sift(imgList_train, clusters)
    feature_test = bag_of_sift(imgList_test, clusters)

    test_acc_300 = [] # test_acc_xx = []
    for i in range(1, 21):
        acc = KNN(feature_train, labelList_train, feature_test, labelList_test, i, classList) # acc = KNN(feature_train, labelList_train, feature_test, labelList_test, i, classList)
        test_acc_300.append(acc) # test_acc_xx.append(acc)

    plt.plot(range(1,21), test_acc_300, '-ro', label='cluster_size=300 & w/o resize & w/o norm.')
    print("cluster_size=300 & w/o resize & w/o norm.")
    test_acc_300 = [round(i,2) for i in test_acc_300]
    print(test_acc_300)

    ''' For comparison of diff. cluster size
    # cluster size = 150
    cluster_size = 150
    clusters = get_clusters(imgList_train, cluster_size)

    feature_train = bag_of_sift(imgList_train, clusters)
    feature_test = bag_of_sift(imgList_test, clusters)

    test_acc_150 = []
    for i in range(1, 21):
        acc = KNN(feature_train, labelList_train, feature_test, labelList_test, i, classList)
        test_acc_150.append(acc)

    # cluster size = 450
    cluster_size = 450
    clusters = get_clusters(imgList_train, cluster_size)

    feature_train = bag_of_sift(imgList_train, clusters)
    feature_test = bag_of_sift(imgList_test, clusters)

    test_acc_450 = []
    for i in range(1, 21):
        acc = KNN(feature_train, labelList_train, feature_test, labelList_test, i, classList)
        test_acc_450.append(acc)
    '''

    ''' For comparison of diff. resize and normalization combination
    # images with resize & w/o normalization
    imgList_train, labelList_train = load_data('./hw5_data/train', classList, True, False)
    imgList_test, labelList_test = load_data('./hw5_data/test', classList, True, False)

    clusters = get_clusters(imgList_train, cluster_size)

    feature_train = bag_of_sift(imgList_train, clusters)
    feature_test = bag_of_sift(imgList_test, clusters)

    test_acc_ox = []
    for i in range(1, 21):
        acc = KNN(feature_train, labelList_train, feature_test, labelList_test, i, classList)
        test_acc_ox.append(acc)

    # images w/o resize & with normalization
    imgList_train, labelList_train = load_data('./hw5_data/train', classList, False, True)
    imgList_test, labelList_test = load_data('./hw5_data/test', classList, False, True)

    clusters = get_clusters(imgList_train, cluster_size)

    feature_train = bag_of_sift(imgList_train, clusters)
    feature_test = bag_of_sift(imgList_test, clusters)

    test_acc_xo = []
    for i in range(1, 21):
        acc = KNN(feature_train, labelList_train, feature_test, labelList_test, i, classList)
        test_acc_xo.append(acc)

    # images with resize & with normalization
    imgList_train, labelList_train = load_data('./hw5_data/train', classList, True, True)
    imgList_test, labelList_test = load_data('./hw5_data/test', classList, True, True)

    clusters = get_clusters(imgList_train, cluster_size)

    feature_train = bag_of_sift(imgList_train, clusters)
    feature_test = bag_of_sift(imgList_test, clusters)

    test_acc_oo = []
    for i in range(1, 21):
        acc = KNN(feature_train, labelList_train, feature_test, labelList_test, i, classList)
        test_acc_oo.append(acc)'''

    ''' For comparison of diff. cluster size
    plt.plot(range(1,21), test_acc_300, '-ro', label='cluster_size=300')
    plt.plot(range(1,21), test_acc_150, '-go', label='cluster_size=150')
    plt.plot(range(1,21), test_acc_450, '-bo', label='cluster_size=450')
    print("cluster_size=300")
    test_acc_300 = [round(i,2) for i in test_acc_300]
    print(test_acc_300)
    print("cluster_size=150")
    test_acc_150 = [round(i,2) for i in test_acc_150]
    print(test_acc_150)
    print("cluster_size=450")
    test_acc_450 = [round(i,2) for i in test_acc_450]
    print(test_acc_450)
    '''
    
    ''' For comparison of diff. resize and normalization combination
    plt.plot(range(1,21), test_acc_xx, '-ro', label='w/o resize & w/o norm.')
    plt.plot(range(1,21), test_acc_ox, '-go', label='with resize & w/o norm.')
    plt.plot(range(1,21), test_acc_xo, '-bo', label='w/o resize & with norm.')
    plt.plot(range(1,21), test_acc_oo, '-yo', label='with resize & with norm.')
    print("w/o resize & w/o norm.")
    test_acc_xx = [round(i,2) for i in test_acc_xx]
    print(test_acc_xx)
    print("with resize & w/o norm.")
    test_acc_ox = [round(i,2) for i in test_acc_ox]
    print(test_acc_ox)
    print("w/o resize & with norm.")
    test_acc_xo = [round(i,2) for i in test_acc_xo]
    print(test_acc_xo)
    print("with resize & with norm.")
    test_acc_oo = [round(i,2) for i in test_acc_oo]
    print(test_acc_oo)
    
    '''

    plt.xticks(range(1, 21, 1))
    plt.title('Bag of SIFT + KNN')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()   