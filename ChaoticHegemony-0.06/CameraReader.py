import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

IS_WALL = True

BODY_SHIP_ORANGE = ([8, 100, 100],[30, 255, 255])
DOT_GREEN_ON_ORANGE = ([8, 100, 100],[30, 255, 255])
DOT_BLACK_ON_ORANGE = ([50, 0, 0],[230, 150, 150])

BODY_SHIP_BLACK = ([0, 0, 0],[50, 50, 50])
DOT_GREEN_TAPE = ([0, 20, 0],[200, 255, 40])
DOT_RED_TAPE = ([0, 0, 40],[100, 100, 255])

BRIGHT_BLUE = ([40, 0, 0],[255, 200, 100])
BRIGHT_PINK = ([0, 0, 150],[140, 140, 255])
PENCIL = ([0, 0, 40],[60, 220, 255])

STICK_GREEN = [(74,50,50),(115,255,255)]
STICK_BLACK = [(125, 50, 20), (175,255,255)]
STICK_BLUE = [(75, 0, 0), (110, 255, 255)]

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

def mirror_point(p):
    if IS_WALL:
        return (640-p[0], p[1])
    else:
        return (640-p[0], 480-p[1])

def mirror_box(b):
    return [mirror_point(x) for x in b]

def average(l):
    return int(float(sum(l)) / len(l))

def fix_pos(l):
    return (fix([x[0] for x in l]), fix([x[1] for x in l]))

def fix(l):
    s = sorted(l)
    return average(s[1:-1])

def factor_point(p, factor):
    return [x*factor for x in p]

def factor_box(b, factor):
    return[factor_point(x,factor) for x in b]

def get_ship_box(ship_color_bounds,screen):
    box = get_box(ship_color_bounds,screen)
    if box is None:
        return None,None
    ship = factor_box(mirror_box(box),0.66)

    return get_box_center(ship), get_box_angle(ship)


def get_box_center(box):
    return average([x[0] for x in box]),average([x[1] for x in box])


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

def get_box(color_bounds, screen):
    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hierarchy = get_place(hsv, color_bounds[0], color_bounds[1])

    #rects = [cv2.boundingRect(x) for x in hierarchy]
    #rect = max(rects, key=lambda x: x[2]*x[3])

    boxes = [cv2.boxPoints(cv2.minAreaRect(cnt)) for cnt in hierarchy]

    boxes = filter(lambda b:is_in_half_screen(b,screen),boxes)

    try:
        box = max(boxes,key=lambda x: get_distance(x[0],x[1]) * get_distance(x[1],x[2]))
    except ValueError as e:
        print e
        return None

    return box

 HALF_SCREEN_THRESHOLD = 80
def is_in_half_screen(box, half_screen):
    xs = [p[0] for p in box]
    if half_screen == 1:
        return all([(x+HALF_SCREEN_THRESHOLD)>320 for x in xs])
    if half_screen == 0:
        return all([(x-HALF_SCREEN_THRESHOLD)<320 for x in xs])

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
    hierarchy = get_place(hsv, STICK_BLUE[0], STICK_BLUE[1])

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
    try:
        best = max(range(len(rects)), key=lambda x: rects[x][2]*rects[x][3])
    except ValueError as e:
        print "ValueError: ", e
        continue





    cv2.imshow('original',frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    if k == 80:
        #print get_ship(BODY_SHIP_BLACK,DOT_RED_TAPE,DOT_GREEN_TAPE)
        print box
        print get_box_angle(box)



cv2.destroyAllWindows()

