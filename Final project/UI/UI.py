# -*- coding: utf-8 -*-
import sys
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton
from read_tf import detectFaceTF

model = './frozen_inference_graph.pb'
pbtxt = './face_label.pbtxt'
# for cv
# pbtxt = './rcnn_cv.pbtxt'
score = 0.6

def detectFace(model_path, pbtxt_path, img_path, limit=0.5):
    # read TensorFlow network
    box = []
    cvNet = cv2.dnn.readNetFromTensorflow(model_path, pbtxt_path)
    img = cv2.imread(img_path)

    if pbtxt_path.find('rcnn') != -1:
        img_size = (600, 1024)
    else:
        img_size = (300, 300)
    # (104.0, 177.0, 123.0)
    cvNet.setInput(cv2.dnn.blobFromImage(img, 1.0, img_size, (127.0, 127.0, 127.0), 
                    swapRB=True, crop=False))
    detections = cvNet.forward()

    rows = img.shape[0]
    cols = img.shape[1] 
    for detect in detections[0,0,:,:]:
        score = float(detect[2])
        if score > limit:
            left = detect[3] * cols
            top = detect[4] * rows
            right = detect[5] * cols
            bottom = detect[6] * rows
            box.append((int(left), int(top), int(right), int(bottom)))
            # cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
    print(box)
    return box

class ImageLabel(QLabel):
    scale = 1.0
    def showImage(self, img):
        height, width, channel = img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.setPixmap(QPixmap.fromImage(self.qImg))

    def mousePressEvent(self,event):
        self.x = event.x()
        self.y = event.y()
        print(str(self.x) + ' ' + str(self.y))


    def mouseMoveEvent(self, event):
        self.x = event.x()
        self.y = event.y()
        #print(str(self.x) + ' ' + str(self.y))

    def wheelEvent(self, event):
        numDegrees = event.angleDelta() / 8
        numSteps = numDegrees / 15       
        #print(numSteps.y())
        height, width, _ = self.img.shape
        if numSteps.y() == -1:
            if (self.scale >= 0.1):
                self.scale -= 0.05
        else:
            if (self.scale <= 2.0):
                self.scale += 0.05
        #print(self.scale)
        height2 = int(height * self.scale)
        width2 = int(width * self.scale)
        img2 = cv2.resize(self.img, (width2, height2), interpolation=cv2.INTER_AREA)
        self.showImage(img2)

class MyDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(800, 600)
        self.label = ImageLabel()
        #self.label.setMouseTracking(True)
        self.btnOpen = QPushButton('Open Image', self)
        self.btnProcess = QPushButton('Blur Image', self)
        self.btnSave = QPushButton('Save Image', self)
        self.btnOrigin = QPushButton('Origin Image', self)
        self.btnDetect = QPushButton('Detect Image', self)
        self.btnSave.setEnabled(False)

        layout = QGridLayout(self)
        layout.addWidget(self.label, 0, 0, 5, 5)
        layout.addWidget(self.btnOpen, 5, 0, 1, 1)
        layout.addWidget(self.btnDetect, 5, 1, 1, 1)
        layout.addWidget(self.btnProcess, 5, 2, 1, 1)
        layout.addWidget(self.btnOrigin, 5, 3, 1, 1)
        layout.addWidget(self.btnSave, 5, 4, 1, 1)       

        self.btnOpen.clicked.connect(self.openSlot)
        self.btnProcess.clicked.connect(self.processSlot)
        self.btnSave.clicked.connect(self.saveSlot)
        self.btnOrigin.clicked.connect(self.origin_func)
        self.btnDetect.clicked.connect(self.detect_func)

    def openSlot(self):
        self.filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', 'Image', '*.png *.jpg *.bmp')
        if self.filename == '':
            return
        self.label.img = cv2.imread(self.filename, -1)
        print(self.label.img.shape)
        self.origin_width = self.label.img.shape[1]
        self.origin_height = self.label.img.shape[0]
        self.resize_width = int(800 * self.label.img.shape[1] / self.label.img.shape[0])
        self.resize_height = 800
        self.label.img = cv2.resize(self.label.img, (self.resize_width, self.resize_height), interpolation = cv2.INTER_AREA)
        self.origin_image = self.label.img.copy()
        if self.label.img.size == 1:
            return 
        self.label.showImage(self.label.img)
        height, width, _ = self.label.img.shape
        self.label.setFixedSize(width, height)
        self.btnSave.setEnabled(True)

    def saveSlot(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Image', 'Image', '*.png *.jpg *.bmp')
        if filename == '':
            return
        cv2.imwrite(filename, self.label.img)

    def processSlot(self):
        for box in self.box:
            left, top, right, bottom = box
            left = int(left * self.resize_width / self.origin_width)
            right = int(right * self.resize_width / self.origin_width)
            top = int(top * self.resize_height / self.origin_height)
            bottom = int(bottom * self.resize_height / self.origin_height)
            self.label.img[top:bottom, left:right] = cv2.blur(self.label.img[top:bottom, left:right], (7, 7))
        self.label.showImage(self.label.img)

    def origin_func(self):
        self.label.img = self.origin_image.copy()
        self.label.showImage(self.label.img)

    def detect_func(self):
        self.box = detectFaceTF(model, pbtxt, self.filename, score)
        for box in self.box:
            left, top, right, bottom = box
            left = int(left * self.resize_width / self.origin_width)
            right = int(right * self.resize_width / self.origin_width)
            top = int(top * self.resize_height / self.origin_height)
            bottom = int(bottom * self.resize_height / self.origin_height)
            self.label.img = cv2.rectangle(self.label.img, (left-1,top-1), (right+1,bottom+1), (0,0,255), thickness=2)
        self.label.showImage(self.label.img)

if __name__ == '__main__':
    a = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    sys.exit(a.exec_())