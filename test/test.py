import cv2
import numpy as np
from math import atan2, degrees, pi, hypot

cap = cv2.VideoCapture(0)


def get_box_angle(box):
    '''
    get angle from a given box points
    '''
    p1 = box[0]
    p2 = box[1]
    p3 = box[2]
    p4 = box[3]

    x1 = p1[0]
    x2 = p2[0]
    x3 = p3[0]
    x4 = p4[0]
    y1 = p1[1]
    y2 = p2[1]
    y3 = p3[1]
    y4 = p4[1]

    #if hypot(x2 - x1, y2 - y1) > hypot(x3 - x4, y3 - y4):


    dx = x2 - x1
    dy = y2 - y1
    rads = atan2(-dy,dx)
    rads %= 2*pi
    degs = int(degrees(rads))
    return degs


while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of color in HSV
    lower_col = np.array([169, 100, 100])
    upper_col = np.array([179, 255, 255])

    lower_col = np.array([74, 50, 50])
    upper_col = np.array([115, 255, 255])

    # Threshold the HSV image to get only the desired colors
    mask = cv2.inRange(hsv, lower_col, upper_col)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    # print position
    ret,thresh = cv2.threshold(mask,237,255,cv2.THRESH_BINARY)
    contours,hierarchy, x = cv2.findContours(thresh, 1, 2)

    img = frame
    for cnt in hierarchy:

        x,y,w,h = cv2.boundingRect(cnt)
        if w > 30 and h > 30:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            im = cv2.drawContours(res,[box],0,(0,0,255),2)
            print str(get_box_angle(box))

    cv2.imshow('original',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('result',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
