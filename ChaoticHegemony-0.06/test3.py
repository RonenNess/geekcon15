import cv2
import numpy as np

cap = cv2.VideoCapture(1)


def printAngle(calculatedRect):
    if calculatedRect.size.width < calculatedRect.size.height:
        print "Angle along longer side: " + str(calculatedRect.angle+180)
    else:
        print "Angle along longer side: " + str(calculatedRect.angle+90)



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
            print dir(box)

    """
    # draw all contours bounding boxes
    for cnt in hierarchy:
        x,y,w,h = cv2.boundingRect(cnt)
        if w > 30 and h > 30:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
    """

    cv2.imshow('original',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('result',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
