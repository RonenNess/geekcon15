import cv2
import numpy as np
import math

cap = cv2.VideoCapture(1)

BODY_SHIP_ORANGE = ([8, 100, 100],[30, 255, 255])
DOT_GREEN_ON_ORANGE = ([8, 100, 100],[30, 255, 255])
DOT_BLACK_ON_ORANGE = ([50, 0, 0],[230, 150, 150])

BODY_SHIP_BLACK = ([0, 0, 0],[50, 50, 50])
DOT_GREEN_TAPE = ([0, 20, 0],[200, 255, 40])
DOT_RED_TAPE = ([0, 0, 40],[100, 100, 255])

BRIGHT_BLUE = ([40, 0, 0],[255, 200, 100])
PENCIL = ([0, 0, 40],[60, 220, 255])

def get_place(hsv, low_bound, high_bound):

    # define range of color in HSV
    lower_col = np.array(low_bound)
    upper_col = np.array(high_bound)

    # Threshold the HSV image to get only the desired colors
    mask = cv2.inRange(hsv, lower_col, upper_col)

    # print position
    ret,thresh = cv2.threshold(mask,237,255,cv2.THRESH_BINARY)
    contours,hierarchy, x = cv2.findContours(thresh, 1, 2)

    return hierarchy

def average(l):
    return int(float(sum(l)) / len(l))

def fix_pos(l):
    return (fix([x[0] for x in l]), fix([x[1] for x in l]))

def fix(l):
    s = sorted(l)
    return average(s[1:-1])




def get_ship(ship_color_bounds, center_color_bounds, head_color_bounds):
    ship = []
    center = []
    head = []
    """
    for i in range(10):
        ship.append(get_rect(ship_color_bounds))
        center.append(get_rect(center_color_bounds,ship[i]))
        head.append(get_rect(head_color_bounds,ship[i]))


    ship_center = map(get_rect_center,ship)
    center_center = map(get_rect_center,center)
    head_center = map(get_rect_center,head)


    avg_ship = (fix([x[0] for x in ship_center]), average([x[1] for x in ship_center]))
    avg_center = (fix([x[0] for x in center_center]), average([x[1] for x in center_center]))
    avg_head = (fix([x[0] for x in head_center]), average([x[1] for x in head_center]))


    print ship, center, head

    print get_rect_center(center)
    print get_rect_center(head)

    print get_angle(get_rect_center(center), get_rect_center(head))
    return avg_ship, 0

    """
    ship = get_rect(ship_color_bounds)
    center = get_rect(center_color_bounds,None) # was ship in none
    head = get_rect(head_color_bounds,None) # was ship in none
    print get_rect_center(ship), get_rect_center(center), get_rect_center(head)
    return get_rect_center(ship), get_rect_center(center), get_rect_center(head)
#    return avg_ship,get_angle(avg_center, avg_head)




def get_rect_center(rect):
    return (rect[0] + (rect[2] / 2),rect[1] + (rect[3] / 2))

def get_angle(point_A, point_B):
    x_delta =  point_B[0] - point_A[0]
    y_delta = point_A[1] - point_B[1]


    rad = math.atan2(y_delta,x_delta)

    return rad*(180/math.pi)

def get_rect(color_bounds, space_bounds = None):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hierarchy = get_place(hsv, color_bounds[0], color_bounds[1])

    rects = [cv2.boundingRect(x) for x in hierarchy]

    if space_bounds:
        rects = filter(lambda x: is_contain_rectangle(space_bounds,x), rects)

    rect = max(rects, key=lambda x: x[2]*x[3])

    return rect

def is_contain_rectangle(big_rect, small_rect, threshold = 18):
    return (big_rect[0] < small_rect[0] + threshold) and\
           (big_rect[1] < small_rect[1] + threshold) and\
           ((big_rect[2] + big_rect[0] + threshold) > (small_rect[2] + small_rect[0])) and\
           ((big_rect[3] + big_rect[1] + threshold ) > (small_rect[3] + small_rect[1]))


def get_distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def get_box_angle(box):
    distance1 = get_distance(box[0],box[1]) + get_distance(box[2],box[3])
    distance2 = get_distance(box[0],box[3]) + get_distance(box[1],box[2])

    if (distance1 < distance2):
        return get_angle(calculate_average_point(box[0],box[1]),calculate_average_point(box[2],box[3]))
    else:
        return get_angle(calculate_average_point(box[0],box[3]),calculate_average_point(box[1],box[2]))

def calculate_average_point(point1, point2):
    return average([point1[0],point2[0]]),average([point1[1],point2[1]])
while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert GBR? BGR?  to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #hierarchy = get_place(hsv, [80,50, 50],[100, 255, 255]) green
    hierarchy = get_place(frame, BRIGHT_BLUE[0], BRIGHT_BLUE[1])

    # draw all contours bounding boxes
    for cnt in hierarchy:
        x,y,w,h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            im = cv2.drawContours(frame,[box],0,(0,0,255),2)



    rects = [cv2.boundingRect(x) for x in hierarchy]
    best = max(range(len(rects)), key=lambda x: rects[x][2]*rects[x][3])





    cv2.imshow('original',frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    if k == 80:
        #print get_ship(BODY_SHIP_BLACK,DOT_RED_TAPE,DOT_GREEN_TAPE)
        print box
        print get_box_angle(box)



cv2.destroyAllWindows()

