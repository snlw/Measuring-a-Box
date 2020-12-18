import cv2 
import numpy as np 

def getContours(img, threshold = [100,100], display = False, minArea = 100, filter = 0, draw = False):
    # Convert to grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Blur 
    blur = cv2.GaussianBlur(gray, ksize = (5,5), sigmaX = 1)

    # Apply Canny Edge Detection 
    canny = cv2.Canny(blur, threshold[0], threshold[1])

    # Apply Image Dilation
    kernel_dilate = np.ones((5,5))
    dilate = cv2.dilate(canny, kernel_dilate, iterations = 5)

    # Apply Image Erosion 
    erode = cv2.erode(dilate, kernel_dilate, iterations = 2)

    if display:
        erode = cv2.resize(canny, (0,0), None, .1, .1)
        cv2.imshow('Image Erosion', erode)

    # Identify contours 
    contours, _ = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []

    # Filter Contours and Create Bounding Box 
    for c in contours:
        area = cv2.contourArea(c)
        if area > minArea:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            boundingbx = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append((len(approx), area, approx, boundingbx,c))
            else:
                finalContours.append((len(approx), area, approx, boundingbx, c))

    # Sort Contours (Descending Order)
    finalContours = sorted(finalContours, key = lambda x: x[1], reverse = True)

    # Draw Contours 
    if draw:
        for c in finalContours:
            cv2.drawContours(img, c[4], -1, color = (255,0,0), thickness = 2)

    return img, finalContours

def reorder(points):
    points = points.reshape((4,2))
    new = np.zeros_like(points)
    add = points.sum(1)
    new[0] = points[np.argmin(add)]
    new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis = 1)
    new[1] = points[np.argmin(diff)]
    new[2] = points[np.argmax(diff)]
    return new

def warp(img, points, width, height, pad = 50):
    points = reorder(points)
    one = np.float32(points)
    two = np.float32([[0,0],[width,0],[0,height],[width, height]])
    matrix = cv2.getPerspectiveTransform(one, two)

    img_warp = cv2.warpPerspective(img, matrix, (width, height))
    img_warp = img_warp[pad:img_warp.shape[0]-pad, pad:img_warp.shape[1]-pad]
    return img_warp

def findHeightLength(img, points, scaleFactor, display = False):
    def pythagoras(a,b,s):    
        return ((b[0]/s-a[0]/s)**2 + (b[1]/s-a[1]/s)**2)**0.5
    l = [] 

    # Open the points in approx
    for p in points:
        l.append(p[0][:])

    highest = sorted(l, key=lambda x:x[1], reverse = True)[0]
    second_highest = sorted(l, key=lambda x:x[1], reverse = True)[1]
    lowest = sorted(l, key=lambda x:x[0], reverse = False)[0]
    second_lowest = sorted(l, key=lambda x:x[0], reverse = False)[1]

    length = round(pythagoras(highest, second_highest, scaleFactor), 1)
    height = round(pythagoras(lowest, second_lowest, scaleFactor),1)
    if display:
        cv2.arrowedLine(img, (highest[0],highest[1]) , (second_highest[0], second_highest[1]), (255,0,255),3,8,0,0.05)
        cv2.arrowedLine(img, (lowest[0], lowest[1]), (second_lowest[0], second_lowest[1]), (255,0,255),3,8,0,0.05)
        cv2.putText(img, "Length: {}mm".format(length), (lowest[0]-50,lowest[1]+50), cv2.FONT_HERSHEY_COMPLEX_SMALL, .5, (255,0,255),1)
        cv2.putText(img, "Height: {}mm".format(height), (lowest[0]-50, lowest[1]+30), cv2.FONT_HERSHEY_COMPLEX_SMALL, .5, (255,0,255),1)
    return




