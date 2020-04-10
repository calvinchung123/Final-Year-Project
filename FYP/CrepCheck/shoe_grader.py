import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui
from operator import itemgetter
import math
from sklearn.cluster import KMeans
from collections import Counter
import os


def RemoveBackGround(img2):
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # convert to grayscale
    blur = cv2.blur(gray, (3, 3)) # blur the image
    #Canny edge detection followed by dilation and erosion
    thresh = cv2.Canny(blur,threshold1=50,threshold2=250)
    thresh = cv2.dilate(thresh, None, iterations=1)
    thresh = cv2.erode(thresh, None, iterations=1)

    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create hull array for convex hull points
    hull = []
    
    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))
    # create an empty black image
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
    
    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0) # green - color for contours
        color = (0, 0, 255) # white - for hull
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8,hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color,1)
    Gdrawing = cv2.cvtColor(drawing,cv2.COLOR_BGR2GRAY)

    for i in range(len(contours)):
        color = (255, 255, 255) # white - for hull
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color,-1)
    Gdrawing = cv2.cvtColor(drawing,cv2.COLOR_BGR2GRAY)
    global foreground
    foreground = cv2.bitwise_and(img2,img2,mask=Gdrawing)
    
    return foreground

#cv2.imshow("Forefround",result)
def sharpen(foreground):
    sharpen = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")
    sharp =cv2.filter2D(foreground,ddepth=-1,kernel=sharpen)
    #cv2.imshow("Sharpen",sharp)
    sharp2=sharp.copy()
    return sharp2

def FrontCrop(sharp2):
    sharp2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # convert to grayscale
    blr = cv2.medianBlur(sharp2,5)
    circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, 1.2, 100)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image
            cv2.circle(sharp2, (x, y), r*2, (0, 0, 0), -1)
    resul= cv2.bitwise_and(foreground,foreground,mask=sharp2)
    #cv2.imshow('ewvsd',resul)
    #cv2.waitKey(0)
    E =cv2.Canny(resul,threshold1=100,threshold2=200)
    E2 =E.copy()
    contours,hierarchy = cv2.findContours(E2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try: hierarchy = hierarchy[0]
    except: hierarchy = []

    height = E2.shape[0]
    width = E2.shape[1]
    min_x, min_y = width, height
    max_x = max_y = 0
    global drawing 
    drawing = np.zeros((E2.shape[0], E2.shape[1], 3), np.uint8)
    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        if w > 80 and h > 80:
            cv2.rectangle(drawing, (x,y), (x+w,y+h), (255, 0, 0), -1)

    drawing = cv2.cvtColor(drawing,cv2.COLOR_BGR2GRAY)
    global front
    front= cv2.bitwise_and(img2,img2,mask=drawing)
    return front

def GetCorners(front):
    G = cv2.cvtColor(front, cv2.COLOR_BGR2GRAY)
    global corners
    corners =cv2.goodFeaturesToTrack(drawing,maxCorners=4, qualityLevel=0.2,minDistance=100)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(drawing,(x,y),3,255,-1)
    #cv2.imshow('corners',drawing)
    return corners
    
def CalcPoints(corners):
    lowx=10000
    lowy=10000
    highy =0
    highx=0
    for i in corners:
        x,y =i.ravel()
        if (x<=lowx):
            lowx = x
        if(y<=lowy):
            lowy = y
        if(x>=highx):
            highx =x
        if(y>=highy):
            highy=y
    #print(lowx)
    #print(lowy)
    #print('------')
    #print(highx)
    #print(highy)

    height= math.sqrt( ((lowx-lowx)**2)+((lowy-highy)**2) )
    width= math.sqrt( ((lowx-highx)**2)+((lowy-lowy)**2) )
    #print(height)
    #print(width)

    #print(lowy+height)
    #print(lowx+width)
    roi = front[int(lowy):int(lowy+height),int(lowx):int(lowx+width)]
    #cv2.imshow('roi',roi)
    return roi

def CreaseCalc(roi):
    E =cv2.Canny(roi,threshold1=100,threshold2=200)
    #cv2.imshow("Canny",E)
    cv2.waitKey(0)
    black=0
    for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                if E[i,j]==0:
                    black=black+1
    white=0
    for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                if E[i,j]==255:
                    white=white+1
    #print(white)
    #print(black)
    total = white+black
    #print(total)
    percentage = white/total*100/1
    #print(percentage)
    creasegrade=0
    if(percentage<10):
        creasegrade=1
    if(percentage>10 and percentage<15):
        creasegrade = 2
    if(percentage>20):
        creasegrade = 3
    return creasegrade

def HoleCrop(foreground,sharp2):
    sharp2 = cv2.cvtColor(sharp2, cv2.COLOR_BGR2GRAY) # convert to grayscale
    blr = cv2.medianBlur(sharp2,5)
    dp = 1
    min_dist = 200
    param_1 = 200
    param_2 =200
    minRadius =50
    MaxRadius = 100
    circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, dp, min_dist, param_1, param_2, minRadius,MaxRadius)
    count = 0
    offset = 20
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            if count ==1:
                break
            # draw the circle in the output image
            cv2.circle(sharp2, (x, y), r+offset, (0, 0, 0), -1)
            count=count+1

    global hole
    hole= cv2.bitwise_and(foreground,foreground,mask=sharp2)
    front = FrontCrop(hole)
    corners = GetCorners(front)
    roi = CalcPoints(corners)
    hole = roi
    #cv2.imshow('geed',hole)
    #print('hi')
    return hole

def get_dominant_color(hole, k=4, image_processing_size = None):
    image=hole
    #resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size, 
                            interpolation = cv2.INTER_AREA)
    
    #reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    #cluster and assign labels to the pixels 
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)

    #count labels to find most popular
    label_counts = Counter(labels)

    #subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return list(dominant_color)

def ShoeColour(dominant_color,hole):
    hsvhole= cv2.cvtColor(hole, cv2.COLOR_BGR2HSV)
    #cv2.imshow('afa',hsvhole)
    #cv2.waitKey(0)
    c1 = dominant_color[0]
    c2= dominant_color[1]
    c3 = dominant_color[2]
    shoe = np.uint8([[[c1,c2,c3 ]]])
    shoeHSV = cv2.cvtColor(shoe, cv2.COLOR_BGR2HSV)
    #print(dominant_color)
    H, S, V = cv2.split(shoeHSV)
    #print(H)
    #print(S)
    #print(V)
    global mask
    if V <=25:
        RangeLower1=(0,0,0)
        RangeUpper1=(180,110,90)
        mask =cv2.inRange(hsvhole,RangeLower1,RangeUpper1)
        mask = 255 - mask
    else:
        RangeLower1=(0,40,90)
        RangeUpper1=(40,255,255)
        mask =cv2.inRange(hsvhole,RangeLower1,RangeUpper1)
        mask = 255 - mask
    #cv2.imshow('mask',mask)
    #mask = cv2.bitwise_not(mask)
    global var
    var= cv2.bitwise_and(hole,hole,mask=mask)
    #cv2.imshow('image',var)
    return var

def FindDirt(var):
    gray = cv2.cvtColor(var, cv2.COLOR_BGR2GRAY) # convert to grayscale
    high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5*high_thresh
    kernel = np.ones((3,3),np.uint8)
    blur = cv2.blur(var, (3, 3)) # blur the image
    var = cv2.morphologyEx(var, cv2.MORPH_OPEN, kernel)
    E =cv2.Canny(var,lowThresh,high_thresh)
    #E = cv2.dilate(E,kernel,iterations = 1)
    #cv2.imshow('ab',E)
    #cv2.imshow('a',var)
    #E = cv2.morphologyEx(E, cv2.MORPH_CLOSE, kernel)
    #area=[]
    #E = cv2.morphologyEx(E, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('canny',E)
    contours,hierarchy = cv2.findContours(E, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #cont= cv2.drawContours(var, contours, -1, (0,0,255), 1)
    for i in range(6):
        cont= cv2.drawContours(var, contours, i, (0,255,0), 1)
        #area[i] = cv2.contourArea(cont[i])
        #print(area)
    #cv2.imshow('awew',cont)
    closed_contours = []
    open_contours = []

    for i in contours:
        if (cv2.contourArea(i) > cv2.arcLength(i, True)) & (cv2.contourArea(i)>100):
            closed_contours.append(i)
        else:
            open_contours.append(i)
    new = cv2.drawContours(hole, closed_contours, -1, (0,255,0), 1)
    #cv2.imshow('new',new)
    #cv2.imshow('wew',new)
    print(len(closed_contours))
    return closed_contours

def CalcDirt(closed_contours):
    global dirtgrade=0
    if(len(closed_contours) <2):
        dirtgrade=0
    if(len(closed_contours) >=2 and len(closed_contours) <4 ):
        dirtgrade=1
    if(len(closed_contours) >=4 and len(closed_contours)<6):
        dirtgrade=2
    if(len(closed_contours) > 5):
        dirtgrade=3
    return dirtgrade

def Grader(creasegrade, dirtgrade):
    grade = 10
    grade=10-creasegrade -dirtgrade
    grade=[grade,creasegrade,dirtgrade]
    return grade



def grade_shoe(filepath):
    #Going into main
    #Reading in image from user upload
    global img
    global img2
    img = cv2.imread(filepath)
    img2=img.copy()
    #--Image Pre-processing--
    #Step 1- Remove Foreground
    #Step 2- Sharpen phto to enhance the defects
    foreground = RemoveBackGround(img2)
    sharp2 = sharpen(foreground)

    #--Defect Detector 1 - Creases--
    #Step 1- Crop front of shoe (return drawing image with best fit bounding rectangle)
    #Step 2- Get corners (using drawing get corners)
    #Step 3- Calculate the points for the best fit bounding rectangle, creates ROI(front of shoe)
    #step 4- Calculate creases -Returns crease grade
    front = FrontCrop(sharp2)
    corners = GetCorners(front)
    roi = CalcPoints(corners)
    creasegrade = CreaseCalc(roi)
    #--Defect Detector 2 -- Dirt --
    hole = HoleCrop(foreground,sharp2)
    dominant_color=get_dominant_color(hole)
    var=ShoeColour(dominant_color,hole)
    closed_contours= FindDirt(var)
    dirtgrade= CalcDirt(closed_contours)

    grade = Grader(creasegrade,dirtgrade)
  return grade
