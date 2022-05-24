import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
import time
import cyvlfeat

from cyvlfeat.kmeans import kmeans,kmeans_quantize
from cyvlfeat.sift import dsift
from libsvm.svmutil import *

def load_data(path, classList, resize=False, normalize=False):
    labelList = []
    imgList = []
    c_idx = 0
    for c in classList:
        for imgPath in glob.glob(path + '/' + c + '/*.jpg'):
            labelList.append(c_idx)
            img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
            if resize:
                img = cv2.resize(img, (16, 16))
            if normalize:
                mean = np.mean(img)
                std = np.std(img)
                img = (img - mean) / std
            imgList.append(img)
        c_idx += 1

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
        histogram[i, cluster_idx] = counts # /np.sum(counts)  

    return histogram

if __name__ == '__main__':
    start = time.time()
    classList = os.listdir('./hw5_data/train')

    # images w/o resize & w/o normalization
    imgList_train, labelList_train = load_data('./hw5_data/train', classList, False, False)
    imgList_test, labelList_test = load_data('./hw5_data/test', classList, False, False)
    # acc_comparison = []

    '''
    # cluster size = 75
    cluster_size = 75
    clusters = get_clusters(imgList_train, cluster_size)

    feature_train = bag_of_sift(imgList_train, clusters)
    feature_test = bag_of_sift(imgList_test, clusters)

    # SVM training model
    model = svm_train(labelList_train, feature_train, '-t 0 -q')

    # testing
    p_label, p_acc, p_val = svm_predict(labelList_test, feature_test, model, '-q')
    acc_comparison.append(p_acc[0])

    # cluster size = 150
    cluster_size = 150
    clusters = get_clusters(imgList_train, cluster_size)

    feature_train = bag_of_sift(imgList_train, clusters)
    feature_test = bag_of_sift(imgList_test, clusters)

    # SVM training model
    model = svm_train(labelList_train, feature_train, '-t 0 -q')

    # testing
    p_label, p_acc, p_val = svm_predict(labelList_test, feature_test, model, '-q')
    acc_comparison.append(p_acc[0])

    # cluster size = 225
    cluster_size = 225
    clusters = get_clusters(imgList_train, cluster_size)

    feature_train = bag_of_sift(imgList_train, clusters)
    feature_test = bag_of_sift(imgList_test, clusters)

    # SVM training model
    model = svm_train(labelList_train, feature_train, '-t 0 -q')

    # testing
    p_label, p_acc, p_val = svm_predict(labelList_test, feature_test, model, '-q')
    acc_comparison.append(p_acc[0])

    # cluster size = 300
    cluster_size = 300
    clusters = get_clusters(imgList_train, cluster_size)

    feature_train = bag_of_sift(imgList_train, clusters)
    feature_test = bag_of_sift(imgList_test, clusters)

    # SVM training model
    model = svm_train(labelList_train, feature_train, '-t 0 -q')

    # testing
    p_label, p_acc, p_val = svm_predict(labelList_test, feature_test, model, '-q')
    acc_comparison.append(p_acc[0])
    '''

    # cluster size = 375
    cluster_size = 375
    clusters = get_clusters(imgList_train, cluster_size)

    feature_train = bag_of_sift(imgList_train, clusters)
    feature_test = bag_of_sift(imgList_test, clusters)

    # SVM training model
    model = svm_train(labelList_train, feature_train, '-t 0 -q')

    # testing
    p_label, p_acc, p_val = svm_predict(labelList_test, feature_test, model, '-q')
    # acc_comparison.append(p_acc[0])

    print('------ACC-------')
    print('Acc: {:.2f}%'.format(p_acc[0]))

    '''
    # cluster size = 450
    cluster_size = 450
    clusters = get_clusters(imgList_train, cluster_size)

    feature_train = bag_of_sift(imgList_train, clusters)
    feature_test = bag_of_sift(imgList_test, clusters)

    # SVM training model
    model = svm_train(labelList_train, feature_train, '-t 0 -q')

    # testing
    p_label, p_acc, p_val = svm_predict(labelList_test, feature_test, model, '-q')
    acc_comparison.append(p_acc[0])
    
    for i in range(1, 7):
        print('------ACC-------')
        print('cluster size = {}'.format(i*75))
        print('Acc: {:.2f}%'.format(acc_comparison[i-1]))

    plt.plot(range(75, 525, 75),acc_comparison , '-ro', label='Accuracy')
    plt.xticks(range(75, 525, 75))
    plt.title('Bag of SIFT + SVM')
    plt.xlabel('Cluster Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    '''

    end = time.time()
    print('執行時間: {}分{:.2f}秒'.format((end-start)/60, (end-start)%60)) 