import cv2 
import numpy as np
import time
import utils

##############################################################

webcam = True
path = "stamp_plaster_namecard.jpg"
cap = cv2.VideoCapture(0)
#Brightness
cap.set(10,160)
#Width
cap.set(3,1920)
#Height
cap.set(4,1080)

scaleFactor = 3

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)
    
    img, finalContours = utils.getContours(img, minArea= 50000, filter=4)

    if len(finalContours) != 0:
        # Get the 4 corner points of the biggest contour area.
        biggest = finalContours[0][2]
        img_warp = utils.warp(img, biggest, width = 150*scaleFactor, height = 200*scaleFactor)

        # Problem: Sometimes due to shadow and lighting, number of points of contours is not 4 (ideal) so, filter has to be 0. 
        img_2, finalContours_2 = utils.getContours(img_warp, minArea=20000, filter=0, threshold=[60,60], draw = False)

        if len(finalContours_2) != 0:
            for c in finalContours_2:
                cv2.polylines(img_2, [c[2]], True, (0,255,0), 2)
                utils.findHeightLength(img_2, c[2], scaleFactor, display = True)

        cv2.imshow('A4', img_2)

    # img = cv2.resize(img, (0,0), None, .2, .2)
    cv2.imshow('Original', img)
    cv2.waitKey(delay=1)