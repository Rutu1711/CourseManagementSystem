import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)

classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offSet = 20
imgSize = 300

folder = "data/0"
counter = 0
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

while True:
    success, img = cap.read()

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    #For producing a cropped image
    if hands:
        hand = hands[0]

        x, y, w, h = hand["bbox"]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offSet:y + h + offSet, x - offSet:x + w + offSet]

        imgCropShape = imgCrop.shape


        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            widthCalculated = math.ceil(k * w)

            imgResize = cv2.resize(imgCrop, (widthCalculated, imgSize))
            imgResizeShape = imgResize.shape

            widthGap = math.ceil((imgSize - widthCalculated) / 2)
            imgWhite[:, widthGap : widthCalculated + widthGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction, index)

        else:
            k = imgSize / w
            heightCalculated = math.ceil(k * h)

            imgResize = cv2.resize(imgCrop, (imgSize, heightCalculated))
            imgResizeShape = imgResize.shape

            heightGap = math.ceil((imgSize - heightCalculated) / 2)
            imgWhite[heightGap : heightCalculated + heightGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        cv2.rectangle(imgOutput, (x-offSet, y-offSet-50), (x-offSet+90, y-offSet-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offSet, y - offSet), (x + w + offSet, y + h + offSet), (255, 0, 255), 4)



        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)


