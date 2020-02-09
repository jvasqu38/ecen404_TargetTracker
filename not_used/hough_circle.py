import cv2
import numpy as np

img = cv2.imread('20191107_201036.jpg',0)
img= cv2.resize(img, (560, 1020), interpolation = cv2.INTER_NEAREST)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=35,maxRadius=50)

circles = np.uint16(np.around(circles))

list_of_centers = []
list_of_centers2 = []
x_coor = []
n = 0

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    x_center, y_center = i[0], i[1]
    x_coor.append(i[0])
    crosshair_center = tuple([x_center, y_center])

    if n == 0:
        list_of_centers.append(crosshair_center)
        pix_per_inch = i[2] * 2 / 4
        # cv2.circle(image, crosshair_center, 35, (0, 255, 255), 2)

    elif n >= 1 and x_coor[n - 1] < x_coor[n] - 50 or x_coor[n - 1] > x_coor[
        n] + 50:  # filters our similar coordinates

        list_of_centers.append(crosshair_center)
        list_of_centers2.append(crosshair_center)
        # cv2.circle(image, crosshair_center, 35, (0, 255, 255), 2)
    n = n + 1
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
print(circles)
cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
\


def find_crosshairs2(image):
    # load image and convert it to grayscale
   # img_rgb = image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # load template
    template = cv2.imread('20191107_201153.jpg', 0)
    #w, h = template.shape[::-1]

    template = cv2.resize(template, (wt,ht))
    #cv2.imshow('template', template)
    #cv2.waitKey(0)
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    # similarity threshold
    threshold = 0.25
    loc = np.where(res >= threshold)

    list_of_centers = []
    list_of_centers2 = []
    x_coor = []
    n = 0

    for pt in zip(*loc[::-1]):  # loop over all the templates found in image
        x_center, y_center = pt[0] + int(wt / 2), pt[1] + int(ht / 2)
        x_coor.append(pt[0] + int(wt / 2))
        crosshair_center = tuple([x_center, y_center])

        if n == 0:
            list_of_centers.append(crosshair_center)

            #cv2.circle(image, crosshair_center, 35, (0, 255, 255), 2)

        elif n >= 1 and x_coor[n - 1] < x_coor[n] - 50 or x_coor[n - 1] > x_coor[
            n] + 50:  # filters our similar coordinates

            list_of_centers.append(crosshair_center)
            list_of_centers2.append(crosshair_center)
            #cv2.circle(image, crosshair_center, 35, (0, 255, 255), 2)
        n = n + 1

    # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
   # print(list_of_centers)

    #img_rgb = cv2.resize(img_rgb, (500, 500))
    cv2.imshow('Template', template)
    #cv2.imshow('Detected Targets', image)

    #use the width (in pixels) of the detected target to find our real world pixel/inch ratio
    pix_per_inch = wt/2
    return list_of_centers, pix_per_inch

