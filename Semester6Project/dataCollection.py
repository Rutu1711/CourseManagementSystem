import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
offSet = 20
imgSize = 300

folder = "data/0"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    #For producing a cropped image
    if hands:
        hand = hands[0]

        x ,y, w, h = hand["bbox"]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offSet : y + h + offSet, x - offSet: x + w + offSet]

        imgCropShape = imgCrop.shape


        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            widthCalculated = math.ceil(k * w)

            imgResize = cv2.resize(imgCrop, (widthCalculated, imgSize))
            imgResizeShape = imgResize.shape

            widthGap = math.ceil((imgSize - widthCalculated) / 2)
            imgWhite[:, widthGap : widthCalculated + widthGap] = imgResize

        else:
            k = imgSize / w
            heightCalculated = math.ceil(k * h)

            imgResize = cv2.resize(imgCrop, (imgSize, heightCalculated))
            imgResizeShape = imgResize.shape

            heightGap = math.ceil((imgSize - heightCalculated) / 2)
            imgWhite[heightGap : heightCalculated + heightGap, :] = imgResize



        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
