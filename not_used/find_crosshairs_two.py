import cv2
import numpy as np
#load image and convert it to grayscale
img_rgb = cv2.imread('1shot_canvas.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

#load template
template = cv2.imread('template.jpg',0)
w, h = template.shape[::-1]
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

#similarity threshold
threshold = 0.27
loc = np.where( res >= threshold)

list_of_centers = []
list_of_centers2 = []
x_coor = []
n=0

for pt in zip(*loc[::-1]): #loop over all the templates found in image
    x_center, y_center= pt[0] + int(w / 2), pt[1] + int(h / 2)
    x_coor.append(pt[0] +int(w/2))
    crosshair_center = tuple([x_center, y_center])

    if n==0:
        list_of_centers.append(crosshair_center)
        cv2.circle(img_rgb, crosshair_center, 100, (0, 255, 255), 2)

    elif n>=1 and x_coor[n-1] < x_coor[n]-10 or x_coor[n-1] > x_coor[n]+10: #filters our similar coordinates

        list_of_centers.append(crosshair_center)
        list_of_centers2.append(crosshair_center)
        cv2.circle(img_rgb, crosshair_center, 100, (0, 255, 255), 2)
    n=n+1



# cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
print(list_of_centers)


img_rgb = cv2.resize(img_rgb, (1000,1000))
cv2.imshow('Detected',img_rgb)
cv2.waitKey(0)

